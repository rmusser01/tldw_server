"""Generated quizzes must honor the selected question types before persistence."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
    ArtifactVerificationResult,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.exceptions import QuizMalformedOutputError
from tldw_Server_API.app.services import quiz_generator

pytestmark = pytest.mark.integration
QuizContext = tuple[CharactersRAGDB, dict[str, Any], dict[str, Any]]


@pytest.fixture
def quiz_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[QuizContext]:
    """Provide real databases and a source-grounded model response."""
    monkeypatch.setenv("TEST_MODE", "0")
    db = CharactersRAGDB(str(tmp_path / "quizzes.db"), client_id="type-fidelity-test")
    media_db = MediaDatabase(str(tmp_path / "media.db"), client_id="type-fidelity-test")
    note_id = db.add_note(title="Trial", content="The trial lasted 14 days.")
    payload = {
        "questions": [
            {
                "question_type": "multiple_choice",
                "question_text": "How long did the trial last?",
                "options": ["7 days", "14 days", "21 days", "28 days"],
                "correct_answer": 1,
                "explanation": "The trial lasted 14 days.",
                "source_citations": [
                    {
                        "source_type": "note",
                        "source_id": str(note_id),
                        "quote": "The trial lasted 14 days.",
                    }
                ],
            }
        ]
    }
    args = {
        "db": db,
        "media_db": media_db,
        "sources": [{"source_type": "note", "source_id": note_id}],
        "num_questions": 1,
        "api_provider": "llama.cpp",
        "model": "../models/quiz.gguf",
    }
    yield db, args, payload
    db.close_connection()
    media_db.close_connection()


def completion(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap generated JSON in the provider completion format."""
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def grounded_verifier() -> AsyncMock:
    """Stub external claims inference without bypassing quiz persistence."""
    return AsyncMock(
        return_value=ArtifactVerificationResult(verdict="grounded", report={}, unit_results=[], metadata={})
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_type", ["fill_blank", "unrecognized", None])
async def test_unselected_raw_type_fails_before_claims_or_persistence(
    quiz_context: QuizContext, monkeypatch: pytest.MonkeyPatch, bad_type: str | None
) -> None:
    db, args, payload = quiz_context
    args["question_types"] = ["multiple_choice", "true_false"]
    payload["questions"].append({"question_type": bad_type, "question_text": "Extra item"})
    verifier = AsyncMock()
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", AsyncMock(return_value=completion(payload)))
    monkeypatch.setattr(quiz_generator, "_verify_quiz_questions_against_sources", verifier)

    with pytest.raises(ValueError, match="question type"):
        await quiz_generator.generate_quiz_from_sources(**args)

    assert db.list_quizzes(limit=10, offset=0)["count"] == 0
    verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_selection_allows_any_selected_type_when_count_is_smaller(
    quiz_context: QuizContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, args, payload = quiz_context
    args["question_types"] = ["multiple_choice", "true_false"]
    payload["questions"][0].update(
        question_type="true_false",
        question_text="The trial lasted 14 days.",
        correct_answer="true",
    )
    llm = AsyncMock(return_value=completion(payload))
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", llm)
    monkeypatch.setattr(quiz_generator, "_verify_quiz_questions_against_sources", grounded_verifier())

    result = await quiz_generator.generate_quiz_from_sources(**args)

    assert result["questions"][0]["question_type"] == "true_false"
    assert "multiple_choice, true_false" in llm.await_args.kwargs["prompt"]
    assert db.list_quizzes(limit=10, offset=0)["count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("profile", ["standard_recall", "mixed_assessment"])
@pytest.mark.parametrize(
    ("question_type", "answer"),
    [
        ("multi_select", [0, 2]),
        ("matching", {"A": "one", "B": "two", "C": "three", "D": "four"}),
    ],
)
async def test_advanced_type_persists_normalized_tags(
    quiz_context: QuizContext,
    monkeypatch: pytest.MonkeyPatch,
    question_type: str,
    answer: list[int] | dict[str, str],
    planned: bool,
    profile: str,
) -> None:
    db, args, payload = quiz_context
    args["question_types"] = [question_type]
    args["generation_profile"] = profile
    if planned:
        args["question_plan"] = [{"question_type": question_type, "count": 1}]
    payload["questions"][0].update(
        question_type=question_type,
        question_text="Which source terms match?",
        options=["A", "B", "C", "D"],
        correct_answer=answer,
        tags=[" trial ", "TRIAL", "duration", "best-of-five", "assertion reasoning", ""],
    )
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", AsyncMock(return_value=completion(payload)))
    monkeypatch.setattr(quiz_generator, "_verify_quiz_questions_against_sources", grounded_verifier())

    result = await quiz_generator.generate_quiz_from_sources(**args)

    assert result["questions"][0]["question_type"] == question_type
    assert result["questions"][0]["correct_answer"] == answer
    assert result["questions"][0]["tags"] == ["trial", "duration"]
    assert db.list_quizzes(limit=10, offset=0)["count"] == 1


@pytest.mark.asyncio
async def test_qualified_selected_citation_is_stored_with_canonical_source_id(
    quiz_context: QuizContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, args, payload = quiz_context
    source_id = str(args["sources"][0]["source_id"])
    payload["questions"][0]["source_citations"][0]["source_id"] = f"note:{source_id}"
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", AsyncMock(return_value=completion(payload)))
    monkeypatch.setattr(quiz_generator, "_verify_quiz_questions_against_sources", grounded_verifier())

    result = await quiz_generator.generate_quiz_from_sources(**args)

    assert result["questions"][0]["source_citations"][0]["source_id"] == source_id
    assert db.list_quizzes(limit=10, offset=0)["count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", ["note", "media"])
async def test_qualified_unselected_citation_is_rejected_before_persistence(
    quiz_context: QuizContext, monkeypatch: pytest.MonkeyPatch, source_type: str
) -> None:
    db, args, payload = quiz_context
    source_id = str(args["sources"][0]["source_id"])
    payload["questions"][0]["source_citations"][0].update(
        source_type=source_type,
        source_id=f"{source_type}:{source_id}-other",
    )
    verifier = AsyncMock()
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", AsyncMock(return_value=completion(payload)))
    monkeypatch.setattr(quiz_generator, "_verify_quiz_questions_against_sources", verifier)

    with pytest.raises(quiz_generator.QuizProvenanceValidationError, match="do not map"):
        await quiz_generator.generate_quiz_from_sources(**args)

    assert db.list_quizzes(limit=10, offset=0)["count"] == 0
    verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_token_limited_reasoning_response_reports_exhaustion(
    quiz_context: QuizContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, args, _ = quiz_context
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": None, "reasoning_content": "unfinished reasoning"},
            }
        ]
    }
    verifier = AsyncMock()
    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", AsyncMock(return_value=response))
    monkeypatch.setattr(quiz_generator, "_verify_quiz_questions_against_sources", verifier)

    with pytest.raises(QuizMalformedOutputError, match="max_tokens"):
        await quiz_generator.generate_quiz_from_sources(**args)

    assert db.list_quizzes(limit=10, offset=0)["count"] == 0
    verifier.assert_not_awaited()

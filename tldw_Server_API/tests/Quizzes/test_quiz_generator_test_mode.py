import uuid
from unittest.mock import Mock

import pytest

from tldw_Server_API.app.services import quiz_generator
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.services.quiz_generator import (
    _build_test_mode_questions,
    generate_quiz_from_sources,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="function")
def quizzes_db(tmp_path):
    db_path = tmp_path / "quiz-generator-test-mode.db"
    db = CharactersRAGDB(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


@pytest.fixture(scope="function")
def media_db(tmp_path):
    db_path = tmp_path / "quiz-generator-media.db"
    db = MediaDatabase(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_returns_deterministic_payload_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    monkeypatch.setenv("TEST_MODE", "1")
    note_id = quizzes_db.add_note(
        title="Workspace Alpha",
        content="Alpha program requires citations, review boards, and Friday freshness checks.",
    )

    result = await generate_quiz_from_sources(
        db=quizzes_db,
        media_db=media_db,
        sources=[{"source_type": "note", "source_id": note_id}],
        num_questions=2,
        question_types=["multiple_choice", "true_false"],
        workspace_tag="workspace:test",
    )

    assert result["quiz"]["workspace_tag"] == "workspace:test"
    assert len(result["questions"]) == 2
    assert result["questions"][0]["source_citations"][0]["source_type"] == "note"
    assert result["questions"][0]["source_citations"][0]["source_id"] == note_id


def test_build_test_mode_questions_prefers_evidence_source_identity() -> None:
    questions = _build_test_mode_questions(
        evidence=[
            {
                "source_type": "note",
                "source_id": "note-b",
                "text": "Beta evidence should keep its own citation identity.",
            }
        ],
        normalized_sources=[
            {"source_type": "note", "source_id": "note-a"},
            {"source_type": "note", "source_id": "note-b"},
        ],
        num_questions=1,
        question_types=["multiple_choice"],
    )

    citation = questions[0]["source_citations"][0]
    assert citation["source_type"] == "note"
    assert citation["source_id"] == "note-b"
    assert citation["quote"] == "Beta evidence should keep its own citation identity."


def test_build_test_mode_questions_honors_planned_question_shapes() -> None:
    plan = [
        {"question_type": "multiple_choice", "count": 1, "option_count": 5},
        {"question_type": "multi_select", "count": 1, "option_count": 5},
        {"question_type": "matching", "count": 1, "pair_count": 3},
        {"question_type": "true_false", "count": 1},
        {"question_type": "fill_blank", "count": 1},
    ]
    questions = _build_test_mode_questions(
        evidence=[
            {
                "source_type": "note",
                "source_id": "note-plan",
                "text": "Alpha evidence supports planned deterministic questions.",
            }
        ],
        normalized_sources=[{"source_type": "note", "source_id": "note-plan"}],
        num_questions=5,
        question_types=None,
        question_plan=plan,
    )

    assert [question["question_type"] for question in questions] == [
        "multiple_choice",
        "multi_select",
        "matching",
        "true_false",
        "fill_blank",
    ]
    assert len(questions[0]["options"]) == 5
    assert len(questions[1]["options"]) == 5
    assert len(questions[2]["options"]) == 3
    assert len(questions[2]["correct_answer"]) == 3
    assert all(question["source_citations"][0]["source_id"] == "note-plan" for question in questions)


def test_build_test_mode_questions_emits_five_options_for_best_of_five_profile() -> None:
    questions = _build_test_mode_questions(
        evidence=[
            {
                "source_type": "note",
                "source_id": "note-bof",
                "text": "BOF evidence should produce one best answer.",
            }
        ],
        normalized_sources=[{"source_type": "note", "source_id": "note-bof"}],
        num_questions=1,
        question_types=None,
        generation_profile="best_of_five",
    )

    question = questions[0]
    assert question["question_type"] == "multiple_choice"
    assert len(question["options"]) == 5
    assert question["correct_answer"] == 0


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_uses_test_mode_fallback_for_metadata_only_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        lambda *_args, **_kwargs: [{"source_type": "note", "source_id": "note-meta", "text": ""}],
    )
    monkeypatch.setattr(
        quiz_generator,
        "_resolve_generated_quiz_metadata",
        lambda **_kwargs: ("Metadata Fallback Quiz", "Generated from metadata-only evidence."),
    )
    monkeypatch.setattr(
        quiz_generator,
        "_persist_generated_quiz",
        lambda **kwargs: {
            "quiz": {
                "title": kwargs["quiz_title"],
                "description": kwargs["quiz_description"],
                "workspace_tag": kwargs["workspace_tag"],
            },
            "questions": kwargs["questions"],
        },
    )

    result = await generate_quiz_from_sources(
        db=Mock(),
        media_db=Mock(),
        sources=[{"source_type": "note", "source_id": "note-meta"}],
        num_questions=1,
        question_types=["multiple_choice"],
        workspace_tag="workspace:test",
    )

    assert result["quiz"]["workspace_tag"] == "workspace:test"
    assert result["questions"][0]["source_citations"][0]["quote"] == "Study point from note:note-meta."


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_persists_exact_planned_type_counts_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    monkeypatch.setenv("TEST_MODE", "1")
    note_id = quizzes_db.add_note(
        title="Planned Quiz Note",
        content="CPUs execute instructions. RAM stores working data. GPUs accelerate graphics.",
    )

    result = await generate_quiz_from_sources(
        db=quizzes_db,
        media_db=media_db,
        sources=[{"source_type": "note", "source_id": note_id}],
        num_questions=5,
        question_plan=[
            {"question_type": "multiple_choice", "count": 1, "option_count": 5},
            {"question_type": "multi_select", "count": 1, "option_count": 5},
            {"question_type": "matching", "count": 1, "pair_count": 3},
            {"question_type": "true_false", "count": 1},
            {"question_type": "fill_blank", "count": 1},
        ],
    )

    questions = result["questions"]
    assert [question["question_type"] for question in questions] == [
        "multiple_choice",
        "multi_select",
        "matching",
        "true_false",
        "fill_blank",
    ]
    assert len(questions[0]["options"]) == 5
    assert len(questions[1]["options"]) == 5
    assert len(questions[2]["options"]) == 3
    assert len(questions[2]["correct_answer"]) == 3


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_normalizes_patched_llm_with_question_plan(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("TEST_MODE", "1")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        quiz_generator,
        "resolve_quiz_sources",
        lambda *_args, **_kwargs: [
            {
                "source_type": "note",
                "source_id": "note-plan",
                "text": "CPU means processor. RAM means memory.",
            }
        ],
    )
    monkeypatch.setattr(
        quiz_generator,
        "_resolve_generated_quiz_metadata",
        lambda **_kwargs: ("Planned Quiz", "Generated from a plan."),
    )
    monkeypatch.setattr(
        quiz_generator,
        "_persist_generated_quiz",
        lambda **kwargs: {"quiz": {"title": kwargs["quiz_title"]}, "questions": kwargs["questions"]},
    )

    async def fake_generate_llm(**kwargs):
        captured.update(kwargs)
        return {
            "questions": [
                {
                    "question_type": "multiple_choice",
                    "question_text": "What does CPU mean?",
                    "options": ["Processor", "Memory", "Storage", "Display", "Network"],
                    "correct_answer": "A",
                    "source_citations": [{"source_type": "note", "source_id": "note-plan", "quote": "CPU"}],
                },
                {
                    "question_type": "matching",
                    "question_text": "Match each term.",
                    "options": ["CPU", "RAM"],
                    "correct_answer": {"CPU": "Processor", "RAM": "Memory"},
                    "source_citations": [{"source_type": "note", "source_id": "note-plan", "quote": "CPU"}],
                },
            ]
        }

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_generate_llm)

    result = await generate_quiz_from_sources(
        db=Mock(),
        media_db=Mock(),
        sources=[{"source_type": "note", "source_id": "note-plan"}],
        num_questions=2,
        question_plan=[
            {"question_type": "multiple_choice", "count": 1, "option_count": 5},
            {"question_type": "matching", "count": 1, "pair_count": 2},
        ],
    )

    assert "Planned question requirements" in str(captured["prompt"])
    assert captured["max_tokens"] == 2000
    assert [question["question_type"] for question in result["questions"]] == ["multiple_choice", "matching"]
    assert len(result["questions"][0]["options"]) == 5
    assert len(result["questions"][1]["correct_answer"]) == 2


@pytest.mark.asyncio
async def test_generate_quiz_from_sources_planned_shortfall_fails_before_persist(
    monkeypatch: pytest.MonkeyPatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    monkeypatch.setenv("TEST_MODE", "0")
    note_id = quizzes_db.add_note(title="Shortfall Note", content="A note with enough evidence for generation.")

    async def fake_generate_llm(**_kwargs):
        return {
            "questions": [
                {
                    "question_type": "multiple_choice",
                    "question_text": f"Question {index}?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": 0,
                    "source_citations": [{"source_type": "note", "source_id": str(note_id), "quote": "evidence"}],
                }
                for index in range(4)
            ]
        }

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_generate_llm)

    with pytest.raises(ValueError, match="expected 5, got 4"):
        await generate_quiz_from_sources(
            db=quizzes_db,
            media_db=media_db,
            sources=[{"source_type": "note", "source_id": note_id}],
            num_questions=5,
            question_plan=[{"question_type": "multiple_choice", "count": 5, "option_count": 4}],
        )

    assert quizzes_db.list_quizzes(limit=10, offset=0)["count"] == 0

"""SQLite roundtrip coverage for assertion/reasoning quiz questions."""

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.services import quiz_generator

pytestmark = pytest.mark.unit


def test_assertion_reasoning_roundtrip_preserves_public_subtype_contract(
    quiz_db: CharactersRAGDB,
) -> None:
    raw_question = {
        "question_type": "multiple_choice",
        "assertion": "The review board meets every Friday.",
        "reason": "The selected note explicitly requires Friday review boards.",
        "options": ["untrusted LLM option"],
        "correct_answer": "A",
        "explanation": "The citation supports both statements and the explanatory relationship.",
        "tags": ["governance", "assertion reasoning", "assertion_reasoning"],
        "source_citations": [
            {
                "source_type": "note",
                "source_id": "note-ar",
                "quote": "Review boards meet every Friday.",
            }
        ],
        "reasoning_steps": ["hidden"],
        "chain_of_thought": "hidden",
        "unknown_payload": {"hidden": True},
    }
    question = quiz_generator._normalize_questions(
        [raw_question],
        default_source_type="note",
        default_source_id="note-ar",
        generation_profile="assertion_reasoning",
    )[0]

    persisted = quiz_generator._persist_generated_quiz(
        db=quiz_db,
        normalized_sources=[{"source_type": "note", "source_id": "note-ar"}],
        questions=[question],
        quiz_title="Assertion reasoning",
        quiz_description="Source-grounded assertion/reasoning quiz.",
        primary_media_id=None,
        workspace_id=None,
        workspace_tag=None,
    )

    quiz_id = persisted["quiz"]["id"]
    question_id = persisted["questions"][0]["id"]
    fetched_quiz = quiz_db.get_quiz(quiz_id)
    listed_quizzes = quiz_db.list_quizzes(limit=10, offset=0)["items"]
    fetched_question = quiz_db.get_question(question_id)
    listed_questions = quiz_db.list_questions(
        quiz_id,
        include_answers=True,
        limit=10,
        offset=0,
    )["items"]
    attempt = quiz_db.start_attempt(quiz_id)

    assert fetched_quiz is not None
    assert fetched_quiz["id"] == quiz_id
    assert [quiz["id"] for quiz in listed_quizzes] == [quiz_id]
    assert fetched_question is not None
    assert listed_questions == [fetched_question]
    assert fetched_question["tags"] == ["governance", "assertion_reasoning"]
    assert fetched_question["tags"].count("assertion_reasoning") == 1
    assert fetched_question["options"] == list(quiz_generator.ASSERTION_REASONING_OPTIONS)
    assert fetched_question["question_text"] == (
        "**Assertion:** The review board meets every Friday.\n\n"
        "**Reason:** The selected note explicitly requires Friday review boards."
    )
    assert fetched_question["explanation"] == raw_question["explanation"]
    assert fetched_question["source_citations"] == question["source_citations"]
    assert fetched_question["group_id"] is None
    assert fetched_question["group_prompt"] is None
    assert not {"reasoning_steps", "chain_of_thought", "unknown_payload"}.intersection(fetched_question)

    public_question = attempt["questions"][0]
    assert "correct_answer" not in public_question
    assert not {"reasoning_steps", "chain_of_thought", "unknown_payload"}.intersection(public_question)
    assert public_question["tags"] == ["governance", "assertion_reasoning"]

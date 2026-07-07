import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.quizzes import QuizGenerateRequest, SourceCitation


def test_quiz_generate_request_accepts_sources_array():
    payload = QuizGenerateRequest.model_validate(
        {
            "num_questions": 5,
            "sources": [{"source_type": "note", "source_id": "note-1"}],
        }
    )

    assert payload.sources is not None
    assert payload.sources[0].source_type == "note"


def test_quiz_generate_request_accepts_question_plan():
    request = QuizGenerateRequest.model_validate(
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "num_questions": 5,
            "question_plan": [
                {"question_type": "multiple_choice", "count": 3, "option_count": 5},
                {"question_type": "matching", "count": 2, "pair_count": 4},
            ],
        }
    )

    assert request.question_plan is not None
    assert request.question_plan[0].option_count == 5
    assert request.question_plan[1].pair_count == 4


@pytest.mark.parametrize(
    "payload",
    [
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "num_questions": 1,
            "question_types": ["multiple_choice"],
            "question_plan": [{"question_type": "multiple_choice", "count": 1}],
        },
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "question_plan": [{"question_type": "multiple_choice", "count": 1}],
        },
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "num_questions": 3,
            "question_plan": [{"question_type": "multiple_choice", "count": 2}],
        },
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "num_questions": 2,
            "question_plan": [
                {"question_type": "multiple_choice", "count": 1},
                {"question_type": "multiple_choice", "count": 1},
            ],
        },
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "num_questions": 1,
            "question_plan": [{"question_type": "multiple_choice", "count": 1, "unexpected": True}],
        },
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "num_questions": 1,
            "question_plan": [{"question_type": "true_false", "count": 1, "option_count": 2}],
        },
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "num_questions": 1,
            "question_plan": [{"question_type": "matching", "count": 1, "pair_count": 7}],
        },
    ],
)
def test_quiz_generate_request_rejects_invalid_question_plan(payload):
    with pytest.raises(ValidationError):
        QuizGenerateRequest.model_validate(payload)


def test_quiz_generate_request_rejects_unknown_source_type():
    with pytest.raises(ValidationError):
        QuizGenerateRequest.model_validate(
            {
                "sources": [{"source_type": "unknown", "source_id": "1"}],
            }
        )


def test_quiz_generate_request_requires_media_id_or_sources():
    with pytest.raises(ValidationError):
        QuizGenerateRequest.model_validate({"num_questions": 5})


def test_source_citation_accepts_canonical_source_fields():
    citation = SourceCitation.model_validate(
        {
            "source_type": "flashcard_card",
            "source_id": "card-uuid",
            "quote": "sample",
        }
    )

    assert citation.source_type == "flashcard_card"
    assert citation.source_id == "card-uuid"

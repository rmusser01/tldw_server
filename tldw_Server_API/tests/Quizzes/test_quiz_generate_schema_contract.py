import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.quizzes import (
    AvailableQuizGenerationProfile,
    QuestionCreate,
    QuestionPublicResponse,
    QuestionUpdate,
    QuizGenerateRequest,
    QuizGenerateResponse,
    QuizGenerationProfile,
    QuizImportQuestion,
    SourceCitation,
)

pytestmark = pytest.mark.unit


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


def test_quiz_generate_request_does_not_default_mutate_question_plan_items():
    request = QuizGenerateRequest.model_validate(
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "num_questions": 2,
            "question_plan": [
                {"question_type": "multiple_choice", "count": 1},
                {"question_type": "matching", "count": 1},
            ],
        }
    )

    assert request.question_plan is not None
    assert request.question_plan[0].option_count is None
    assert request.question_plan[1].pair_count is None


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
            "num_questions": 1,
            "question_types": None,
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


def test_quiz_generate_request_accepts_generation_profile():
    payload = QuizGenerateRequest.model_validate(
        {
            "num_questions": 5,
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "generation_profile": "best_of_five",
        }
    )

    assert payload.generation_profile == QuizGenerationProfile.BEST_OF_FIVE


def test_quiz_generate_request_parses_planned_osce_profile_for_runtime_guarding():
    request = QuizGenerateRequest.model_validate(
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "generation_profile": "osce_scenario",
        }
    )

    assert request.generation_profile == QuizGenerationProfile.OSCE_SCENARIO
    assert request.num_stations == 1


@pytest.mark.parametrize("num_stations", [1, 10])
def test_osce_generation_accepts_station_count_bounds(num_stations):
    request = QuizGenerateRequest.model_validate(
        {
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "generation_profile": "osce_scenario",
            "num_stations": num_stations,
        }
    )

    assert request.num_stations == num_stations


@pytest.mark.parametrize("num_stations", [0, 11])
def test_osce_generation_rejects_station_count_outside_bounds(num_stations):
    with pytest.raises(ValidationError):
        QuizGenerateRequest.model_validate(
            {
                "sources": [{"source_type": "note", "source_id": "note-1"}],
                "generation_profile": "osce_scenario",
                "num_stations": num_stations,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_questions", 2),
        ("question_types", ["multiple_choice"]),
        (
            "question_plan",
            [{"question_type": "multiple_choice", "count": 1}],
        ),
    ],
)
def test_osce_generation_rejects_question_only_fields(field, value):
    payload = {
        "sources": [{"source_type": "note", "source_id": "note-1"}],
        "generation_profile": "osce_scenario",
        field: value,
    }

    with pytest.raises(ValidationError):
        QuizGenerateRequest.model_validate(payload)


def test_question_generation_defaults_remain_compatible():
    request = QuizGenerateRequest.model_validate(
        {"sources": [{"source_type": "note", "source_id": "note-1"}]}
    )

    assert request.generation_profile == QuizGenerationProfile.STANDARD_RECALL
    assert request.num_questions == 10


def test_question_generation_rejects_explicit_station_count():
    with pytest.raises(ValidationError):
        QuizGenerateRequest.model_validate(
            {
                "sources": [{"source_type": "note", "source_id": "note-1"}],
                "num_stations": 2,
            }
        )


def test_question_generation_response_defaults_remain_compatible():
    response = QuizGenerateResponse.model_validate(
        {
            "quiz": {
                "id": 1,
                "name": "Recall",
                "total_questions": 0,
                "deleted": False,
                "client_id": "test",
                "version": 1,
            }
        }
    )

    assert response.output_kind == "questions"
    assert response.questions == []
    assert response.osce_stations == []


def test_available_generation_profiles_match_non_planned_catalog_profiles() -> None:
    assert {profile.value for profile in AvailableQuizGenerationProfile} == {
        profile.value
        for profile in QuizGenerationProfile
        if profile is not QuizGenerationProfile.OSCE_SCENARIO
    }


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


_QUESTION_MODEL_PAYLOADS = (
    (
        QuestionCreate,
        {"question_type": "multiple_choice", "question_text": "Stem", "correct_answer": 0},
    ),
    (QuestionUpdate, {}),
    (
        QuestionPublicResponse,
        {
            "id": 1,
            "quiz_id": 2,
            "question_type": "multiple_choice",
            "question_text": "Stem",
            "points": 1,
            "order_index": 0,
            "deleted": False,
            "client_id": "test",
            "version": 1,
        },
    ),
    (
        QuizImportQuestion,
        {"question_type": "multiple_choice", "question_text": "Stem", "correct_answer": 0},
    ),
)


@pytest.mark.parametrize(("model_type", "payload"), _QUESTION_MODEL_PAYLOADS)
def test_question_contracts_default_emq_group_metadata_to_none(model_type, payload):
    question = model_type.model_validate(payload)

    assert question.group_id is None
    assert question.group_prompt is None


@pytest.mark.parametrize(("model_type", "payload"), _QUESTION_MODEL_PAYLOADS)
@pytest.mark.parametrize(("field_name", "max_length"), (("group_id", 128), ("group_prompt", 2000)))
def test_question_contracts_enforce_emq_group_metadata_max_lengths(
    model_type,
    payload,
    field_name,
    max_length,
):
    accepted = model_type.model_validate({**payload, field_name: "x" * max_length})
    assert getattr(accepted, field_name) == "x" * max_length

    with pytest.raises(ValidationError):
        model_type.model_validate({**payload, field_name: "x" * (max_length + 1)})

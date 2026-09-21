import json
from copy import deepcopy
from uuid import UUID

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.osce import (
    OsceAttemptCreate,
    OsceAttemptPatch,
    OsceAttemptSummary,
    OsceChecklistItemStored,
    OsceCitation,
    OsceCitationSourceType,
    OsceKeyPointStored,
    OsceRubricDomainStored,
    OsceRubricLevelStored,
    OsceStationCreateContent,
    OsceStationStoredContent,
    OsceStationSummary,
    OsceStationUpdateContent,
)
from tldw_Server_API.app.api.v1.schemas.quizzes import (
    QuizActivityType,
    QuizCreate,
    QuizGenerationProfile,
    QuizResponse,
    QuizUpdate,
)

pytestmark = pytest.mark.unit

CHECKLIST_ID = UUID("00000000-0000-4000-8000-000000000001")
DOMAIN_ID = UUID("00000000-0000-4000-8000-000000000002")
LOW_LEVEL_ID = UUID("00000000-0000-4000-8000-000000000003")
HIGH_LEVEL_ID = UUID("00000000-0000-4000-8000-000000000004")
KEY_POINT_ID = UUID("00000000-0000-4000-8000-000000000005")


@pytest.fixture
def valid_station_payload() -> dict:
    return {
        "schema_version": "osce.station.v1",
        "title": "Discuss safe anticoagulant use",
        "candidate_instructions": "You are speaking with a simulated patient.",
        "candidate_task": "Explain key safety advice and respond to concerns.",
        "patient_context": {
            "text": "A fictional adult has recently started warfarin.",
            "citations": [],
        },
        "recommended_duration_seconds": 480,
        "checklist_items": [
            {
                "label": "Explains the purpose of treatment",
                "rationale": "Understanding the indication supports safe use.",
                "citations": [],
            }
        ],
        "rubric_domains": [
            {
                "label": "Communication",
                "levels": [
                    {
                        "label": "Needs development",
                        "description": "Explanation is incomplete or unclear.",
                    },
                    {
                        "label": "Effective",
                        "description": "Explanation is clear and checks understanding.",
                    },
                ],
            }
        ],
        "expected_key_points": [
            {
                "text": "Discusses monitoring and clinically important warning signs.",
                "citations": [],
            }
        ],
    }


@pytest.fixture
def valid_stored_station_payload(valid_station_payload: dict) -> dict:
    payload = deepcopy(valid_station_payload)
    payload["checklist_items"][0]["id"] = CHECKLIST_ID
    payload["rubric_domains"][0]["id"] = DOMAIN_ID
    payload["rubric_domains"][0]["levels"][0]["id"] = LOW_LEVEL_ID
    payload["rubric_domains"][0]["levels"][1]["id"] = HIGH_LEVEL_ID
    payload["expected_key_points"][0]["id"] = KEY_POINT_ID
    return payload


def test_station_create_accepts_valid_manual_content(valid_station_payload) -> None:
    station = OsceStationCreateContent.model_validate(valid_station_payload)

    assert station.schema_version == "osce.station.v1"
    assert station.recommended_duration_seconds == 480


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("title",), ""),
        (("title",), "x" * 201),
        (("candidate_instructions",), ""),
        (("candidate_instructions",), "x" * 4001),
        (("candidate_task",), ""),
        (("candidate_task",), "x" * 4001),
        (("patient_context", "text"), ""),
        (("patient_context", "text"), "x" * 10001),
        (("recommended_duration_seconds",), 59),
        (("recommended_duration_seconds",), 7201),
        (("checklist_items", 0, "label"), ""),
        (("checklist_items", 0, "label"), "x" * 1001),
        (("checklist_items", 0, "rationale"), "x" * 2001),
        (("rubric_domains", 0, "label"), ""),
        (("rubric_domains", 0, "label"), "x" * 201),
        (("rubric_domains", 0, "levels", 0, "label"), ""),
        (("rubric_domains", 0, "levels", 0, "label"), "x" * 201),
        (("rubric_domains", 0, "levels", 0, "description"), ""),
        (("rubric_domains", 0, "levels", 0, "description"), "x" * 2001),
        (("expected_key_points", 0, "text"), ""),
        (("expected_key_points", 0, "text"), "x" * 2001),
    ],
)
def test_station_create_rejects_text_and_duration_outside_bounds(
    valid_station_payload,
    path,
    value,
) -> None:
    payload = deepcopy(valid_station_payload)
    target = payload
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        OsceStationCreateContent.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "items"),
    [
        ("checklist_items", []),
        ("checklist_items", [{"label": "Item", "citations": []}] * 51),
        ("rubric_domains", []),
        (
            "rubric_domains",
            [
                {
                    "label": f"Domain {index}",
                    "levels": [
                        {"label": "Low", "description": "Low"},
                        {"label": "High", "description": "High"},
                    ],
                }
                for index in range(13)
            ],
        ),
        ("expected_key_points", []),
        ("expected_key_points", [{"text": "Point", "citations": []}] * 51),
    ],
)
def test_station_create_rejects_collection_outside_bounds(
    valid_station_payload,
    field,
    items,
) -> None:
    payload = {**valid_station_payload, field: items}

    with pytest.raises(ValidationError):
        OsceStationCreateContent.model_validate(payload)


@pytest.mark.parametrize("level_count", [1, 7])
def test_station_create_rejects_rubric_level_count_outside_bounds(
    valid_station_payload,
    level_count,
) -> None:
    payload = deepcopy(valid_station_payload)
    payload["rubric_domains"][0]["levels"] = [
        {"label": f"Level {index}", "description": "Description"}
        for index in range(level_count)
    ]

    with pytest.raises(ValidationError):
        OsceStationCreateContent.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("passing_score", 70),
        ("score", 8),
        ("generated_feedback", "Good performance"),
        ("chain_of_thought", "Hidden reasoning"),
        ("unexpected", True),
    ],
)
def test_station_create_rejects_scoring_advisory_and_unknown_fields(
    valid_station_payload,
    field,
    value,
) -> None:
    payload = {**valid_station_payload, field: value}

    with pytest.raises(ValidationError):
        OsceStationCreateContent.model_validate(payload)


def test_station_create_rejects_protected_nested_ids(valid_station_payload) -> None:
    payload = deepcopy(valid_station_payload)
    payload["checklist_items"][0]["id"] = str(CHECKLIST_ID)

    with pytest.raises(ValidationError):
        OsceStationCreateContent.model_validate(payload)


def test_station_update_accepts_missing_and_server_owned_nested_ids(
    valid_stored_station_payload,
) -> None:
    payload = deepcopy(valid_stored_station_payload)
    payload["checklist_items"].append(
        {"id": None, "label": "Checks understanding", "citations": []}
    )

    station = OsceStationUpdateContent.model_validate(payload)

    assert station.checklist_items is not None
    assert station.checklist_items[0].id == CHECKLIST_ID
    assert station.checklist_items[1].id is None


def test_station_update_rejects_explicit_null_for_non_nullable_content() -> None:
    with pytest.raises(ValidationError, match="cannot be null"):
        OsceStationUpdateContent.model_validate({"checklist_items": None})


def test_station_stored_requires_nested_ids(valid_station_payload) -> None:
    with pytest.raises(ValidationError):
        OsceStationStoredContent.model_validate(valid_station_payload)


def test_station_stored_uses_stored_nested_shapes(valid_stored_station_payload) -> None:
    station = OsceStationStoredContent.model_validate(valid_stored_station_payload)

    assert isinstance(station.checklist_items[0], OsceChecklistItemStored)
    assert isinstance(station.rubric_domains[0], OsceRubricDomainStored)
    assert isinstance(station.rubric_domains[0].levels[0], OsceRubricLevelStored)
    assert isinstance(station.expected_key_points[0], OsceKeyPointStored)


def test_station_stored_rejects_duplicate_uuid_across_nested_families(
    valid_stored_station_payload,
) -> None:
    payload = deepcopy(valid_stored_station_payload)
    payload["expected_key_points"][0]["id"] = CHECKLIST_ID

    with pytest.raises(ValidationError, match="duplicate UUID"):
        OsceStationStoredContent.model_validate(payload)


def test_station_stored_rejects_malformed_nested_uuid(
    valid_stored_station_payload,
) -> None:
    payload = deepcopy(valid_stored_station_payload)
    payload["checklist_items"][0]["id"] = "not-a-uuid"

    with pytest.raises(ValidationError):
        OsceStationStoredContent.model_validate_json(json.dumps(payload, default=str))


def test_station_create_rejects_duplicate_rubric_level_labels(
    valid_station_payload,
) -> None:
    payload = deepcopy(valid_station_payload)
    payload["rubric_domains"][0]["levels"][1]["label"] = "needs DEVELOPMENT"

    with pytest.raises(ValidationError, match="duplicate rubric level label"):
        OsceStationCreateContent.model_validate(payload)


def test_station_create_rejects_unsupported_schema_version(
    valid_station_payload,
) -> None:
    payload = {**valid_station_payload, "schema_version": "osce.station.v2"}

    with pytest.raises(ValidationError):
        OsceStationCreateContent.model_validate(payload)


def test_station_create_strictly_rejects_numeric_string_duration(
    valid_station_payload,
) -> None:
    payload = {**valid_station_payload, "recommended_duration_seconds": "480"}

    with pytest.raises(ValidationError):
        OsceStationCreateContent.model_validate(payload)


@pytest.mark.parametrize("field", ["checklist_count", "rubric_domain_count"])
def test_station_summary_counts_accept_one_and_reject_zero(field) -> None:
    payload = {
        "id": 1,
        "quiz_id": 2,
        "title": "Discuss safe anticoagulant use",
        "recommended_duration_seconds": 480,
        "order_index": 0,
        "version": 1,
        "checklist_count": 1,
        "rubric_domain_count": 1,
        "verification_state": "manually_authored",
        "created_at": "2026-09-10T12:00:00Z",
        "updated_at": "2026-09-10T12:00:00Z",
    }

    summary = OsceStationSummary.model_validate(payload)
    assert getattr(summary, field) == 1

    payload[field] = 0
    with pytest.raises(ValidationError):
        OsceStationSummary.model_validate(payload)


def test_attempt_summary_checklist_total_accepts_one_and_rejects_zero() -> None:
    payload = {
        "id": 1,
        "quiz_id": 2,
        "station_id": 3,
        "client_attempt_id": str(CHECKLIST_ID),
        "station_title": "Discuss safe anticoagulant use",
        "state": "completed",
        "version": 2,
        "started_at": "2026-09-10T12:00:00Z",
        "self_assessment_started_at": "2026-09-10T12:05:00Z",
        "completed_at": "2026-09-10T12:10:00Z",
        "last_modified_at": "2026-09-10T12:10:00Z",
        "elapsed_seconds": 300,
        "checklist_met_count": 0,
        "checklist_total": 1,
    }

    summary = OsceAttemptSummary.model_validate(payload)
    assert summary.checklist_total == 1

    payload["checklist_total"] = 0
    with pytest.raises(ValidationError):
        OsceAttemptSummary.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"source_type": "media", "source_id": "12", "timestamp_seconds": 4.5},
        {"source_type": "document", "source_id": "doc-1", "page_number": 3},
        {
            "source_type": "url",
            "source_id": "web-1",
            "source_url": "https://example.com/source",
        },
        {"source_type": "note", "source_id": "note-1"},
        {
            "source_type": "flashcard_deck",
            "source_id": "7",
            "chunk_id": "card-uuid",
        },
        {
            "source_type": "flashcard_card",
            "source_id": "card-uuid",
            "chunk_id": "card-uuid",
        },
        {
            "source_type": "quiz_attempt",
            "source_id": "9",
            "chunk_id": "9:3",
        },
        {
            "source_type": "quiz_attempt_question",
            "source_id": "9:3",
            "chunk_id": "9:3",
        },
    ],
)
def test_osce_citation_accepts_source_specific_locators(payload) -> None:
    citation = OsceCitation.model_validate_json(json.dumps(payload))

    assert isinstance(citation.source_type, OsceCitationSourceType)


def test_osce_wire_enums_and_uuids_parse_from_decoded_json_values() -> None:
    citation = OsceCitation.model_validate(
        {"source_type": "note", "source_id": "note-1"}
    )
    attempt = OsceAttemptCreate.model_validate(
        {"client_attempt_id": str(CHECKLIST_ID)}
    )

    assert citation.source_type is OsceCitationSourceType.NOTE
    assert attempt.client_attempt_id == CHECKLIST_ID


@pytest.mark.parametrize(
    "payload",
    [
        {"source_type": "media", "source_id": "12", "page_number": 1},
        {"source_type": "media", "source_id": "12", "timestamp_seconds": -0.1},
        {"source_type": "document", "source_id": "doc-1", "timestamp_seconds": 1.0},
        {"source_type": "document", "source_id": "doc-1", "page_number": 0},
        {"source_type": "url", "source_id": "web-1"},
        {
            "source_type": "url",
            "source_id": "web-1",
            "source_url": "ftp://example.com/source",
        },
        {
            "source_type": "note",
            "source_id": "note-1",
            "source_url": "https://example.com/source",
        },
        {
            "source_type": "flashcard_card",
            "source_id": "card-uuid",
            "timestamp_seconds": 1.0,
        },
        {"source_type": "unknown", "source_id": "1"},
        {},
    ],
)
def test_osce_citation_rejects_invalid_or_inconsistent_locator(payload) -> None:
    with pytest.raises(ValidationError):
        OsceCitation.model_validate_json(json.dumps(payload))


def test_osce_citation_enforces_string_bounds() -> None:
    payload = {
        "source_type": "note",
        "source_id": "x" * 513,
        "label": "x" * 201,
        "quote": "x" * 1001,
    }

    with pytest.raises(ValidationError):
        OsceCitation.model_validate_json(json.dumps(payload))


def test_attempt_patch_accepts_tri_state_checklist_values() -> None:
    patch = OsceAttemptPatch.model_validate(
        {
            "expected_version": 2,
            "checklist_selections": {
                CHECKLIST_ID: "met",
                KEY_POINT_ID: "not_met",
            },
        }
    )

    assert patch.checklist_selections == {
        CHECKLIST_ID: "met",
        KEY_POINT_ID: "not_met",
    }


def test_attempt_patch_rejects_scoring_and_invalid_checklist_values() -> None:
    with pytest.raises(ValidationError):
        OsceAttemptPatch.model_validate(
            {
                "expected_version": 2,
                "checklist_selections": {CHECKLIST_ID: "partial"},
                "score": 50,
            }
        )


@pytest.mark.parametrize("model_type", [QuizCreate, QuizUpdate])
@pytest.mark.parametrize("field", ["passing_score", "time_limit_seconds"])
def test_osce_quiz_write_rejects_question_only_settings(model_type, field) -> None:
    payload = {"activity_type": "osce", field: None}
    if model_type is QuizCreate:
        payload["name"] = "OSCE practice"

    with pytest.raises(ValidationError):
        model_type.model_validate(payload)


def test_question_quiz_contract_defaults_remain_compatible() -> None:
    created = QuizCreate.model_validate({"name": "Recall"})
    response = QuizResponse.model_validate(
        {
            "id": 1,
            "name": "Recall",
            "total_questions": 4,
            "deleted": False,
            "client_id": "test",
            "version": 1,
        }
    )

    assert created.activity_type == QuizActivityType.QUESTIONS
    assert response.activity_type == QuizActivityType.QUESTIONS
    assert response.generation_profile is None
    assert response.total_stations == 0


def test_quiz_response_accepts_osce_activity_metadata() -> None:
    response = QuizResponse.model_validate(
        {
            "id": 1,
            "name": "OSCE practice",
            "activity_type": "osce",
            "generation_profile": "osce_scenario",
            "total_questions": 0,
            "total_stations": 2,
            "deleted": False,
            "client_id": "test",
            "version": 1,
        }
    )

    assert response.activity_type == QuizActivityType.OSCE
    assert response.generation_profile == QuizGenerationProfile.OSCE_SCENARIO

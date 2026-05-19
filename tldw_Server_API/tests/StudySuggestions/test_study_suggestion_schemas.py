from dataclasses import fields

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.StudySuggestions.types import RankedTopic, TopicCandidate
from tldw_Server_API.app.api.v1.schemas.study_suggestions import (
    SuggestionActionResponse,
    SuggestionRefreshRequest,
    SuggestionStatusResponse,
)


def test_suggestion_status_response_allows_expected_status_values():
    payload = SuggestionStatusResponse(
        anchor_type="quiz_attempt",
        anchor_id=101,
        status="pending",
        job_id=22,
    )

    assert payload.status == "pending"  # nosec B101
    assert payload.snapshot_id is None  # nosec B101

    for allowed in ("none", "pending", "ready", "failed"):
        response = SuggestionStatusResponse(
            anchor_type="quiz_attempt",
            anchor_id=101,
            status=allowed,
        )
        assert response.status == allowed  # nosec B101


def test_suggestion_status_response_rejects_unknown_status():
    with pytest.raises(ValidationError):
        SuggestionStatusResponse(
            anchor_type="quiz_attempt",
            anchor_id=101,
            status="queued",
        )


def test_refresh_request_accepts_optional_reason():
    empty = SuggestionRefreshRequest()
    described = SuggestionRefreshRequest(reason="quiz answers changed")

    assert empty.reason is None  # nosec B101
    assert described.reason == "quiz answers changed"  # nosec B101


def test_action_response_requires_known_disposition_and_target_service():
    payload = SuggestionActionResponse(
        disposition="generated",
        snapshot_id=55,
        selection_fingerprint="sel-abc123",
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-9",
    )

    assert payload.disposition == "generated"  # nosec B101
    assert payload.target_service == "quiz"  # nosec B101

    with pytest.raises(ValidationError):
        SuggestionActionResponse(
            disposition="queued",
            snapshot_id=55,
            selection_fingerprint="sel-abc123",
            target_service="quiz",
            target_type="quiz",
            target_id="quiz-9",
        )

    with pytest.raises(ValidationError):
        SuggestionActionResponse(
            disposition="generated",
            snapshot_id=55,
            selection_fingerprint="sel-abc123",
            target_service="notes",
            target_type="quiz",
            target_id="quiz-9",
        )


@pytest.mark.parametrize("topic_type", [TopicCandidate, RankedTopic])
def test_topic_types_expose_v2_identity_fields(topic_type):
    field_names = {field.name for field in fields(topic_type)}

    assert {"topic_key", "normalization_version", "source_count", "evidence_reasons"} <= field_names  # nosec B101

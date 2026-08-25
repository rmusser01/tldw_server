"""Contract tests for the canonical admin-webhook API schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.admin_webhooks import (
    AdminWebhookRegistrationResponse,
    AdminWebhookStatusResponse,
    WebhookCreateRequest,
    WebhookListResponse,
    WebhookPatchRequest,
)

pytestmark = pytest.mark.unit


def _registration_payload() -> dict[str, object]:
    return {
        "id": 41,
        "description": "Incident receiver",
        "target_display": "https://receiver.example",
        "target_hostname": "receiver.example",
        "event_types": ["incident.created"],
        "active": False,
        "timeout_seconds": 10,
        "revision": 1,
        "delivery_config_version": 1,
        "secret_version": 1,
        "secret_rotation_required": False,
        "created_by": 7,
        "updated_by": 7,
        "created_at": "2026-08-22T12:00:00Z",
        "updated_at": "2026-08-22T12:00:00Z",
    }


@pytest.mark.unit
def test_create_schema_accepts_only_inactive_server_secret_contract() -> None:
    request = WebhookCreateRequest.model_validate(
        {
            "url": "https://receiver.example/hooks/private",
            "event_types": ["incident.created"],
            "description": "Incident receiver",
            "timeout_seconds": 12,
        }
    )

    assert request.model_dump() == {
        "url": "https://receiver.example/hooks/private",
        "event_types": ["incident.created"],
        "description": "Incident receiver",
        "timeout_seconds": 12,
    }


@pytest.mark.parametrize(
    "payload",
    (
        {
            "url": "https://receiver.example/hooks/private",
            "event_types": ["incident.created"],
            "secret": "caller-controlled",
        },
        {
            "url": "https://receiver.example/hooks/private",
            "event_types": ["incident.created"],
            "active": True,
        },
        {
            "url": "https://receiver.example/hooks/private",
            "event_types": ["*"],
        },
    ),
)
@pytest.mark.unit
def test_create_schema_rejects_secret_active_and_wildcard(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        WebhookCreateRequest.model_validate(payload)


@pytest.mark.unit
def test_patch_requires_at_least_one_non_null_recognized_field() -> None:
    for payload in ({}, {"description": None}, {"unknown": "value"}):
        with pytest.raises(ValidationError):
            WebhookPatchRequest.model_validate(payload)

    same_value_candidate = WebhookPatchRequest.model_validate(
        {"description": "Incident receiver"}
    )
    assert same_value_candidate.model_fields_set == {"description"}


@pytest.mark.unit
def test_registration_response_is_redacted_and_uses_numeric_identity() -> None:
    registration = AdminWebhookRegistrationResponse.model_validate(
        _registration_payload()
    )

    assert isinstance(registration.id, int)
    assert isinstance(registration.created_at, datetime)
    assert set(registration.model_dump()) == {
        "id",
        "description",
        "target_display",
        "target_hostname",
        "event_types",
        "active",
        "timeout_seconds",
        "revision",
        "delivery_config_version",
        "secret_version",
        "secret_rotation_required",
        "created_by",
        "updated_by",
        "created_at",
        "updated_at",
    }
    assert "url" not in registration.model_dump()
    assert "secret" not in registration.model_dump()


@pytest.mark.unit
def test_registration_response_rejects_events_outside_the_catalog() -> None:
    payload = _registration_payload()
    payload["event_types"] = ["*"]

    with pytest.raises(ValidationError):
        AdminWebhookRegistrationResponse.model_validate(payload)


@pytest.mark.unit
def test_list_schema_uses_bounded_offset_metadata() -> None:
    response = WebhookListResponse.model_validate(
        {
            "items": [_registration_payload()],
            "total": 1,
            "limit": 50,
            "offset": 0,
        }
    )
    assert response.total == 1
    assert response.limit == 50
    assert response.offset == 0

    with pytest.raises(ValidationError):
        WebhookListResponse.model_validate(
            {"items": [], "total": 0, "limit": 101, "offset": 0}
        )


@pytest.mark.unit
def test_status_schema_exposes_rollback_state_without_artifact_paths() -> None:
    status = AdminWebhookStatusResponse.model_validate(
        {
            "mode": "migrate",
            "route_selection": "canonical",
            "schema_ready": True,
            "key_state": "available",
            "delivery_capability_ready": False,
            "limits": {
                "registrations": 100,
                "active_registrations": 25,
                "current_registrations": 3,
                "current_active_registrations": 0,
                "registrations_over_limit": False,
                "active_registrations_over_limit": False,
            },
            "migration": {
                "phase": "complete",
                "imported_count": 3,
                "unresolved_count": 0,
                "rejected_count": 0,
                "secret_rotation_required_count": 3,
                "legacy_file_restore_permitted": True,
                "rollback_window_expires_at": "2026-08-29T12:00:00Z",
            },
        }
    )

    serialized = status.model_dump(mode="json")
    assert serialized["migration"]["legacy_file_restore_permitted"] is True
    assert "rollback_window_expires_at" in serialized["migration"]
    assert "backup_path" not in str(serialized)
    assert "key_path" not in str(serialized)

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi import Request

from tldw_Server_API.app.api.v1.endpoints import scheduled_tasks_control_plane
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL, TASKS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import ScheduledTasksDatabase
from tldw_Server_API.app.services.scheduled_task_automation_service import (
    ScheduledTaskAutomationError,
    ScheduledTaskAutomationService,
)

RAW_SENTINEL = "RAW_AGENT_SECRET_DO_NOT_LEAK_ENDPOINT_4B"


def _make_principal(
    *,
    permissions: list[str] | None = None,
    user_id: int = 880,
    subject: str = "scheduled-task-automation-api-test",
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        api_key_id=None,
        subject=subject,
        token_type="access",  # nosec B106
        jti=None,
        roles=[],
        permissions=[TASKS_READ, TASKS_CONTROL] if permissions is None else list(permissions),
        is_admin=False,
        org_ids=[],
        team_ids=[],
        active_org_id=None,
        active_team_id=None,
    )


def _override_auth(
    client,
    *,
    permissions: list[str] | None = None,
    user_id: int = 880,
    subject: str = "scheduled-task-automation-api-test",
) -> None:
    principal = _make_principal(permissions=permissions, user_id=user_id, subject=subject)

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
        request.state.request_id = "test-request-id"
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id="test-request-id")
        return principal

    async def _fake_get_request_user() -> User:
        return User(id=user_id, username=f"scheduled-user-{user_id}", email=None, is_active=True)

    client.app.dependency_overrides[get_auth_principal] = _fake_get_auth_principal
    client.app.dependency_overrides[get_request_user] = _fake_get_request_user


@pytest.fixture()
def scheduled_tasks_client(client_user_only, tmp_path):
    repo = ScheduledTasksDatabase(tmp_path / "scheduled_task_automation_api.db")
    repo.ensure_schema()
    service = ScheduledTaskAutomationService(repository=repo)
    _override_auth(client_user_only)
    client_user_only.app.dependency_overrides[
        scheduled_tasks_control_plane.get_scheduled_task_automation_service
    ] = lambda: service
    client_user_only.scheduled_task_automation_service = service
    client_user_only.scheduled_task_automation_repo = repo
    yield client_user_only
    client_user_only.app.dependency_overrides.pop(get_auth_principal, None)
    client_user_only.app.dependency_overrides.pop(get_request_user, None)
    client_user_only.app.dependency_overrides.pop(
        scheduled_tasks_control_plane.get_scheduled_task_automation_service,
        None,
    )


def _payload(
    *,
    family: str = "recurring_question",
    mode: str = "create",
    definition_id: str | None = None,
    definition_version: int | None = None,
    name: str = "Daily research check",
    input_payload: dict[str, Any] | None = None,
    schedule: dict[str, Any] | None = None,
    visibility_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if input_payload is None:
        input_payload = (
            {"question": "What changed in the selected sources?"}
            if family == "recurring_question"
            else {"agent_ref": "agent:triage", "message": "Summarize priority changes."}
        )
    return {
        "mode": mode,
        "family": family,
        "definition_id": definition_id,
        "definition_version": definition_version,
        "name": name,
        "description": "Endpoint lifecycle test",
        "input": input_payload,
        "schedule": schedule or {"kind": "daily", "time": "09:00", "timezone": "UTC"},
        "visibility_policy": visibility_policy or {"mode": "findings_only"},
        "notification_policy": {"channels": ["in_app"]},
        "approval_policy": {"required": False},
    }


def _create_preview(client, auth_headers, **payload_kwargs) -> dict[str, Any]:
    response = client.post(
        "/api/v1/scheduled-tasks/previews",
        headers=auth_headers,
        json=_payload(**payload_kwargs),
    )
    assert response.status_code == 201, response.text  # nosec B101
    return response.json()


def _create_definition(client, auth_headers, **payload_kwargs) -> dict[str, Any]:
    preview = _create_preview(client, auth_headers, **payload_kwargs)
    response = client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=auth_headers,
        json={"preview_id": preview["id"], "initial_lifecycle": "configured"},
    )
    assert response.status_code == 201, response.text  # nosec B101
    return response.json()


def _assert_no_raw_sentinel(value: Any) -> None:
    assert RAW_SENTINEL not in json.dumps(value, sort_keys=True)  # nosec B101


def _assert_error_envelope(response, *, code: str, status_code: int) -> None:
    assert response.status_code == status_code, response.text  # nosec B101
    detail = response.json()["detail"]
    assert detail["code"] == code  # nosec B101
    assert isinstance(detail["message"], str) and detail["message"]  # nosec B101
    assert isinstance(detail["details"], dict)  # nosec B101
    assert isinstance(detail["field_errors"], list)  # nosec B101
    assert detail["retryable"] is False  # nosec B101
    assert "correlation_id" in detail  # nosec B101


def test_scheduled_task_static_child_routes_do_not_resolve_as_task_ids(scheduled_tasks_client, auth_headers):
    for path in (
        "/api/v1/scheduled-tasks/capabilities",
        "/api/v1/scheduled-tasks/previews",
        "/api/v1/scheduled-tasks/definitions",
    ):
        response = scheduled_tasks_client.get(path, headers=auth_headers)
        assert response.status_code != 404, response.text  # nosec B101
        assert response.text != "scheduled_task_not_found"  # nosec B101


def test_capabilities_report_definition_actions_but_no_execution(scheduled_tasks_client, auth_headers):
    response = scheduled_tasks_client.get("/api/v1/scheduled-tasks/capabilities", headers=auth_headers)

    assert response.status_code == 200, response.text  # nosec B101
    body = response.json()
    families = {item["family"]: item for item in body["items"]}
    assert {"recurring_question", "agent_task"} <= set(families)  # nosec B101
    for family in ("recurring_question", "agent_task"):
        actions = families[family]["actions"]
        assert actions["preview"]["status"] == "available"  # nosec B101
        assert actions["create_definition"]["status"] == "available"  # nosec B101
        assert actions["execute"]["status"] == "unavailable"  # nosec B101
        assert actions["execute"]["reason"] == "execution_not_implemented"  # nosec B101


def test_preview_create_list_and_detail_return_valid_and_invalid_statuses(scheduled_tasks_client, auth_headers):
    valid = _create_preview(scheduled_tasks_client, auth_headers)
    invalid = _create_preview(
        scheduled_tasks_client,
        auth_headers,
        name="Invalid preview",
        input_payload={"question": "   "},
        schedule={"kind": "not-a-schedule"},
    )

    detail = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/previews/{valid['id']}",
        headers=auth_headers,
    )
    listed = scheduled_tasks_client.get("/api/v1/scheduled-tasks/previews?status=invalid", headers=auth_headers)

    assert valid["status"] == "valid"  # nosec B101
    assert invalid["status"] == "invalid"  # nosec B101
    assert invalid["validation_errors"]  # nosec B101
    assert detail.status_code == 200, detail.text  # nosec B101
    assert detail.json()["id"] == valid["id"]  # nosec B101
    assert listed.status_code == 200, listed.text  # nosec B101
    assert [item["id"] for item in listed.json()["items"]] == [invalid["id"]]  # nosec B101


def test_create_update_lifecycle_duplicate_and_audit_routes(scheduled_tasks_client, auth_headers):
    definition = _create_definition(scheduled_tasks_client, auth_headers)
    update_preview = _create_preview(
        scheduled_tasks_client,
        auth_headers,
        mode="update",
        definition_id=definition["id"],
        definition_version=definition["version"],
        name="Updated question",
    )
    updated = scheduled_tasks_client.patch(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}",
        headers=auth_headers,
        json={"preview_id": update_preview["id"]},
    )
    paused = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/pause",
        headers=auth_headers,
    )
    resumed = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/resume",
        headers=auth_headers,
    )
    duplicate = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/duplicate",
        headers=auth_headers,
        json={"name": "Duplicate copy"},
    )
    archived = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/archive",
        headers=auth_headers,
    )
    audit = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/audit",
        headers=auth_headers,
    )

    assert updated.status_code == 200, updated.text  # nosec B101
    assert updated.json()["name"] == "Updated question"  # nosec B101
    assert paused.status_code == 200, paused.text  # nosec B101
    assert paused.json()["lifecycle"] == "paused"  # nosec B101
    assert resumed.status_code == 200, resumed.text  # nosec B101
    assert resumed.json()["lifecycle"] == "configured"  # nosec B101
    assert duplicate.status_code == 200, duplicate.text  # nosec B101
    assert duplicate.json()["id"] != definition["id"]  # nosec B101
    assert duplicate.json()["lifecycle"] == "paused"  # nosec B101
    assert archived.status_code == 200, archived.text  # nosec B101
    assert archived.json()["lifecycle"] == "archived"  # nosec B101
    assert audit.status_code == 200, audit.text  # nosec B101
    assert {item["event_type"] for item in audit.json()["items"]} >= {  # nosec B101
        "definition.created",
        "definition.updated",
        "definition.paused",
        "definition.resumed",
        "definition.archived",
        "definition_duplicated",
    }


def test_definition_filters_preview_filters_audit_filters_and_pagination(scheduled_tasks_client, auth_headers):
    alpha = _create_definition(scheduled_tasks_client, auth_headers, name="Alpha daily")
    _create_definition(
        scheduled_tasks_client,
        auth_headers,
        family="agent_task",
        name="Beta agent",
        input_payload={"agent_ref": "agent:triage", "message": "Review alerts"},
        visibility_policy={"mode": "metadata_only"},
    )
    scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{alpha['id']}/pause",
        headers={**auth_headers, "Idempotency-Key": "pause-filter-key"},
    )

    definitions = scheduled_tasks_client.get(
        "/api/v1/scheduled-tasks/definitions?family=recurring_question&lifecycle=paused&q=Alpha&limit=1&offset=0",
        headers=auth_headers,
    )
    previews = scheduled_tasks_client.get(
        "/api/v1/scheduled-tasks/previews?family=agent_task&mode=create&status=consumed&limit=1&offset=0",
        headers=auth_headers,
    )
    audit = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{alpha['id']}/audit"
        "?event_type=definition.paused&actor=scheduled-task-automation-api-test"
        "&idempotency_key=pause-filter-key&limit=1&offset=0",
        headers=auth_headers,
    )

    assert definitions.status_code == 200, definitions.text  # nosec B101
    assert definitions.json()["total"] == 1  # nosec B101
    assert definitions.json()["items"][0]["id"] == alpha["id"]  # nosec B101
    assert definitions.json()["limit"] == 1  # nosec B101
    assert previews.status_code == 200, previews.text  # nosec B101
    assert previews.json()["total"] == 1  # nosec B101
    assert previews.json()["items"][0]["family"] == "agent_task"  # nosec B101
    assert audit.status_code == 200, audit.text  # nosec B101
    assert audit.json()["total"] == 1  # nosec B101
    assert audit.json()["items"][0]["event_type"] == "definition.paused"  # nosec B101


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/scheduled-tasks/previews?family=unknown",
        "/api/v1/scheduled-tasks/previews?mode=delete",
        "/api/v1/scheduled-tasks/previews?status=ready",
        "/api/v1/scheduled-tasks/definitions?family=unknown",
        "/api/v1/scheduled-tasks/definitions?lifecycle=running",
        "/api/v1/scheduled-tasks/definitions?health=healthy",
    ],
)
def test_invalid_automation_filter_values_return_422(scheduled_tasks_client, auth_headers, path):
    response = scheduled_tasks_client.get(path, headers=auth_headers)

    assert response.status_code == 422, response.text  # nosec B101


def test_api_created_audit_events_store_and_filter_by_request_id(scheduled_tasks_client, auth_headers):
    definition = _create_definition(scheduled_tasks_client, auth_headers, name="Request id audit")

    audit = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/audit?request_id=test-request-id",
        headers=auth_headers,
    )

    assert audit.status_code == 200, audit.text  # nosec B101
    body = audit.json()
    assert body["total"] == 1  # nosec B101
    assert body["items"][0]["event_type"] == "definition.created"  # nosec B101
    assert body["items"][0]["request_id"] == "test-request-id"  # nosec B101


def test_datetime_filters_reject_invalid_values_with_error_envelope(scheduled_tasks_client, auth_headers):
    definition = _create_definition(scheduled_tasks_client, auth_headers, name="Invalid datetime filters")

    definitions = scheduled_tasks_client.get(
        "/api/v1/scheduled-tasks/definitions?created_from=not-a-date",
        headers=auth_headers,
    )
    audit = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/audit?created_to=not-a-date",
        headers=auth_headers,
    )

    _assert_error_envelope(definitions, code="scheduled_task_filter_invalid", status_code=422)
    _assert_error_envelope(audit, code="scheduled_task_filter_invalid", status_code=422)


def test_datetime_filters_normalize_offset_timestamps(scheduled_tasks_client, auth_headers):
    definition = _create_definition(scheduled_tasks_client, auth_headers, name="Offset datetime filters")
    created_at = datetime.fromisoformat(definition["created_at"])
    created_from = created_at.astimezone(timezone(timedelta(hours=5, minutes=30))).isoformat()

    definitions = scheduled_tasks_client.get(
        "/api/v1/scheduled-tasks/definitions",
        headers=auth_headers,
        params={"created_from": created_from},
    )
    audit = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/audit",
        headers=auth_headers,
        params={"created_from": created_from},
    )

    assert definitions.status_code == 200, definitions.text  # nosec B101
    assert [item["id"] for item in definitions.json()["items"]] == [definition["id"]]  # nosec B101
    assert audit.status_code == 200, audit.text  # nosec B101
    assert audit.json()["total"] == 1  # nosec B101
    assert audit.json()["items"][0]["definition_id"] == definition["id"]  # nosec B101


def test_routes_require_read_or_control_permissions(client_user_only, auth_headers):
    try:
        _override_auth(client_user_only, permissions=[])
        read_response = client_user_only.get("/api/v1/scheduled-tasks/previews", headers=auth_headers)
        assert read_response.status_code == 403, read_response.text  # nosec B101

        _override_auth(client_user_only, permissions=[TASKS_READ])
        control_response = client_user_only.post(
            "/api/v1/scheduled-tasks/previews",
            headers=auth_headers,
            json=_payload(),
        )
        assert control_response.status_code == 403, control_response.text  # nosec B101
    finally:
        client_user_only.app.dependency_overrides.pop(get_auth_principal, None)
        client_user_only.app.dependency_overrides.pop(get_request_user, None)


def test_cross_user_preview_and_definition_access_is_denied(scheduled_tasks_client, auth_headers):
    preview = _create_preview(scheduled_tasks_client, auth_headers)
    definition = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=auth_headers,
        json={"preview_id": preview["id"]},
    ).json()
    _override_auth(scheduled_tasks_client, user_id=881, subject="other-user")

    preview_detail = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/previews/{preview['id']}",
        headers=auth_headers,
    )
    definition_detail = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}",
        headers=auth_headers,
    )
    audit_list = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/audit",
        headers=auth_headers,
    )
    cross_create = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=auth_headers,
        json={"preview_id": preview["id"]},
    )

    _assert_error_envelope(preview_detail, code="scheduled_task_preview_not_found", status_code=404)
    _assert_error_envelope(definition_detail, code="scheduled_task_definition_not_found", status_code=404)
    _assert_error_envelope(audit_list, code="scheduled_task_definition_not_found", status_code=404)
    _assert_error_envelope(cross_create, code="scheduled_task_preview_required", status_code=400)


def test_agent_task_sentinel_is_redacted_across_preview_definition_list_detail_and_audit(
    scheduled_tasks_client,
    auth_headers,
):
    preview = _create_preview(
        scheduled_tasks_client,
        auth_headers,
        family="agent_task",
        name="Agent redaction",
        input_payload={"agent_ref": "agent:security", "message": RAW_SENTINEL},
        visibility_policy={"mode": "metadata_only"},
    )
    definition = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=auth_headers,
        json={"preview_id": preview["id"]},
    ).json()
    preview_detail = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/previews/{preview['id']}",
        headers=auth_headers,
    ).json()
    definitions = scheduled_tasks_client.get("/api/v1/scheduled-tasks/definitions", headers=auth_headers).json()
    detail = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}",
        headers=auth_headers,
    ).json()
    audit = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/audit",
        headers=auth_headers,
    ).json()

    for response_body in (preview, preview_detail, definition, definitions, detail, audit):
        _assert_no_raw_sentinel(response_body)


def test_idempotent_preview_replay_and_conflict(scheduled_tasks_client, auth_headers):
    headers = {**auth_headers, "Idempotency-Key": "preview-key"}
    first = scheduled_tasks_client.post("/api/v1/scheduled-tasks/previews", headers=headers, json=_payload())
    replay = scheduled_tasks_client.post("/api/v1/scheduled-tasks/previews", headers=headers, json=_payload())
    conflict = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/previews",
        headers=headers,
        json=_payload(name="Different preview"),
    )

    assert first.status_code == 201, first.text  # nosec B101
    assert replay.status_code == 201, replay.text  # nosec B101
    assert replay.json()["id"] == first.json()["id"]  # nosec B101
    _assert_error_envelope(conflict, code="scheduled_task_idempotency_conflict", status_code=409)


def test_idempotent_create_and_update_replay_after_preview_consumed(scheduled_tasks_client, auth_headers):
    preview = _create_preview(scheduled_tasks_client, auth_headers, name="Idempotent create")
    create_headers = {**auth_headers, "Idempotency-Key": "create-key"}
    first_create = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=create_headers,
        json={"preview_id": preview["id"]},
    )
    replay_create = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=create_headers,
        json={"preview_id": preview["id"]},
    )
    update_preview = _create_preview(
        scheduled_tasks_client,
        auth_headers,
        mode="update",
        definition_id=first_create.json()["id"],
        definition_version=first_create.json()["version"],
        name="Idempotent update",
    )
    update_headers = {**auth_headers, "Idempotency-Key": "update-key"}
    first_update = scheduled_tasks_client.patch(
        f"/api/v1/scheduled-tasks/definitions/{first_create.json()['id']}",
        headers=update_headers,
        json={"preview_id": update_preview["id"]},
    )
    replay_update = scheduled_tasks_client.patch(
        f"/api/v1/scheduled-tasks/definitions/{first_create.json()['id']}",
        headers=update_headers,
        json={"preview_id": update_preview["id"]},
    )

    assert first_create.status_code == 201, first_create.text  # nosec B101
    assert replay_create.status_code == 201, replay_create.text  # nosec B101
    assert replay_create.json() == first_create.json()  # nosec B101
    assert first_update.status_code == 200, first_update.text  # nosec B101
    assert replay_update.status_code == 200, replay_update.text  # nosec B101
    assert replay_update.json() == first_update.json()  # nosec B101


def test_idempotent_duplicate_and_lifecycle_replay_without_extra_audit_events(scheduled_tasks_client, auth_headers):
    definition = _create_definition(scheduled_tasks_client, auth_headers, name="Replay source")
    duplicate_headers = {**auth_headers, "Idempotency-Key": "duplicate-key"}

    first_duplicate = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/duplicate",
        headers=duplicate_headers,
        json={"name": "Replay duplicate"},
    )
    replay_duplicate = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/duplicate",
        headers=duplicate_headers,
        json={"name": "Replay duplicate"},
    )
    paused = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/pause",
        headers={**auth_headers, "Idempotency-Key": "pause-key"},
    )
    pause_replay = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/pause",
        headers={**auth_headers, "Idempotency-Key": "pause-key"},
    )
    resumed = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/resume",
        headers={**auth_headers, "Idempotency-Key": "resume-key"},
    )
    resume_replay = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/resume",
        headers={**auth_headers, "Idempotency-Key": "resume-key"},
    )
    archived = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/archive",
        headers={**auth_headers, "Idempotency-Key": "archive-key"},
    )
    archive_replay = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/archive",
        headers={**auth_headers, "Idempotency-Key": "archive-key"},
    )
    definitions = scheduled_tasks_client.get("/api/v1/scheduled-tasks/definitions", headers=auth_headers).json()
    source_audit = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{definition['id']}/audit",
        headers=auth_headers,
    ).json()
    copy_audit = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/definitions/{first_duplicate.json()['id']}/audit",
        headers=auth_headers,
    ).json()

    assert replay_duplicate.json() == first_duplicate.json()  # nosec B101
    assert pause_replay.json() == paused.json()  # nosec B101
    assert resume_replay.json() == resumed.json()  # nosec B101
    assert archive_replay.json() == archived.json()  # nosec B101
    assert definitions["total"] == 2  # nosec B101
    assert len(source_audit["items"]) == 5  # created, duplicated, paused, resumed, archived  # nosec B101
    assert len(copy_audit["items"]) == 1  # duplicate created only  # nosec B101


def test_same_idempotency_key_with_different_payload_conflicts_on_mutating_routes(
    scheduled_tasks_client,
    auth_headers,
):
    first_preview = _create_preview(scheduled_tasks_client, auth_headers, name="Create one")
    headers = {**auth_headers, "Idempotency-Key": "create-conflict-key"}
    scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=headers,
        json={"preview_id": first_preview["id"]},
    )
    second_preview = _create_preview(scheduled_tasks_client, auth_headers, name="Create two")
    response = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=headers,
        json={"preview_id": second_preview["id"]},
    )

    _assert_error_envelope(response, code="scheduled_task_idempotency_conflict", status_code=409)


@pytest.mark.parametrize(
    ("service_error", "public_code", "status_code"),
    [
        ("scheduled_task_family_unavailable", "scheduled_task_family_unavailable", 409),
        ("scheduled_task_preview_required", "scheduled_task_preview_required", 400),
        ("preview_resource_not_found", "scheduled_task_preview_not_found", 404),
        ("scheduled_task_definition_not_found", "scheduled_task_definition_not_found", 404),
        ("scheduled_task_preview_mismatch", "scheduled_task_preview_mismatch", 409),
        ("scheduled_task_preview_expired", "scheduled_task_preview_expired", 409),
        ("scheduled_task_schedule_invalid", "scheduled_task_schedule_invalid", 422),
        ("scheduled_task_scope_invalid", "scheduled_task_scope_invalid", 422),
        ("scheduled_task_agent_ref_invalid", "scheduled_task_agent_ref_invalid", 422),
        ("scheduled_task_permission_policy_invalid", "scheduled_task_permission_policy_invalid", 422),
        ("scheduled_task_execution_unavailable", "scheduled_task_execution_unavailable", 409),
        ("scheduled_task_definition_version_conflict", "scheduled_task_definition_version_conflict", 409),
        ("scheduled_task_definition_archived", "scheduled_task_definition_archived", 409),
        ("scheduled_task_lifecycle_transition_invalid", "scheduled_task_lifecycle_transition_invalid", 409),
        ("scheduled_task_idempotency_conflict", "scheduled_task_idempotency_conflict", 409),
        ("preview_invalid", "scheduled_task_schedule_invalid", 422),
    ],
)
def test_required_public_error_code_aliases_use_public_error_envelopes(
    scheduled_tasks_client,
    auth_headers,
    service_error,
    public_code,
    status_code,
):
    class _FailingService:
        def create_preview(self, **_kwargs):
            raise ScheduledTaskAutomationError(service_error)

    scheduled_tasks_client.app.dependency_overrides[
        scheduled_tasks_control_plane.get_scheduled_task_automation_service
    ] = lambda: _FailingService()

    response = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/previews",
        headers=auth_headers,
        json=_payload(),
    )

    _assert_error_envelope(response, code=public_code, status_code=status_code)


def test_plain_service_value_error_is_not_mapped_as_public_automation_error(
    scheduled_tasks_client,
    auth_headers,
):
    class _FailingService:
        def create_preview(self, **_kwargs):
            raise ValueError("unexpected_bug")

    scheduled_tasks_client.app.dependency_overrides[
        scheduled_tasks_control_plane.get_scheduled_task_automation_service
    ] = lambda: _FailingService()

    with pytest.raises(ValueError, match="unexpected_bug"):
        scheduled_tasks_client.post(
            "/api/v1/scheduled-tasks/previews",
            headers=auth_headers,
            json=_payload(),
        )


def test_sync_automation_handlers_do_not_run_sqlite_work_on_event_loop():
    automation_handlers = [
        scheduled_tasks_control_plane.get_scheduled_task_automation_capabilities,
        scheduled_tasks_control_plane.list_scheduled_task_automation_previews,
        scheduled_tasks_control_plane.create_scheduled_task_automation_preview,
        scheduled_tasks_control_plane.get_scheduled_task_automation_preview,
        scheduled_tasks_control_plane.list_scheduled_task_automation_definitions,
        scheduled_tasks_control_plane.create_scheduled_task_automation_definition,
        scheduled_tasks_control_plane.list_scheduled_task_automation_definition_audit,
        scheduled_tasks_control_plane.get_scheduled_task_automation_definition,
        scheduled_tasks_control_plane.update_scheduled_task_automation_definition,
        scheduled_tasks_control_plane.pause_scheduled_task_automation_definition,
        scheduled_tasks_control_plane.resume_scheduled_task_automation_definition,
        scheduled_tasks_control_plane.archive_scheduled_task_automation_definition,
        scheduled_tasks_control_plane.duplicate_scheduled_task_automation_definition,
    ]

    assert all(not inspect.iscoroutinefunction(handler) for handler in automation_handlers)  # nosec B101
    assert inspect.iscoroutinefunction(scheduled_tasks_control_plane.list_scheduled_tasks)  # nosec B101
    assert inspect.iscoroutinefunction(scheduled_tasks_control_plane.create_scheduled_task_reminder)  # nosec B101

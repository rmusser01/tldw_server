from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import Request

from tldw_Server_API.app.api.v1.endpoints import scheduled_tasks_control_plane
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.permissions import TASKS_CONTROL, TASKS_READ
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import ScheduledTasksDatabase
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.services.scheduled_task_automation_service import ScheduledTaskAutomationService
from tldw_Server_API.app.services.scheduled_tasks_control_plane_service import ScheduledTasksControlPlaneService


def _make_principal(*, permissions: list[str] | None = None) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=880,
        api_key_id=None,
        subject="scheduled-tasks-control-plane-test",
        token_type="access",  # nosec B106 - auth principal test fixture token type
        jti=None,
        roles=[],
        permissions=[TASKS_READ, TASKS_CONTROL] if permissions is None else list(permissions),
        is_admin=False,
        org_ids=[],
        team_ids=[],
        active_org_id=None,
        active_team_id=None,
    )


def _override_auth(client, *, permissions: list[str] | None = None) -> None:
    principal = _make_principal(permissions=permissions)

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    async def _fake_get_request_user() -> User:
        return User(id=880, username="scheduled-user", email=None, is_active=True)

    client.app.dependency_overrides[get_auth_principal] = _fake_get_auth_principal
    client.app.dependency_overrides[get_request_user] = _fake_get_request_user


@pytest.fixture()
def scheduled_tasks_client(client_user_only, tmp_path, monkeypatch):
    automation_repo = ScheduledTasksDatabase(tmp_path / "scheduled_tasks_control_plane_automation.db")
    automation_repo.ensure_schema()
    automation_service = ScheduledTaskAutomationService(repository=automation_repo)
    _override_auth(client_user_only)
    client_user_only.app.dependency_overrides[
        scheduled_tasks_control_plane.get_scheduled_task_automation_service
    ] = lambda: automation_service
    monkeypatch.setattr(
        ScheduledTasksControlPlaneService,
        "_scheduled_tasks_db",
        staticmethod(lambda user_id: automation_repo),
        raising=False,
    )
    yield client_user_only
    client_user_only.app.dependency_overrides.pop(get_auth_principal, None)
    client_user_only.app.dependency_overrides.pop(get_request_user, None)
    client_user_only.app.dependency_overrides.pop(
        scheduled_tasks_control_plane.get_scheduled_task_automation_service,
        None,
    )


def _automation_payload(
    *,
    name: str = "Daily research check",
    family: str = "recurring_question",
) -> dict[str, Any]:
    return {
        "mode": "create",
        "family": family,
        "definition_id": None,
        "definition_version": None,
        "name": name,
        "description": "Control-plane projection test",
        "input": {"question": "Any new answer?"},
        "schedule": {"kind": "daily", "time": "09:00", "timezone": "UTC"},
        "visibility_policy": {"mode": "findings_only"},
        "notification_policy": {},
        "approval_policy": {"required": False},
    }


def _create_automation_definition(
    client,
    auth_headers,
    *,
    name: str = "Daily research check",
    initial_lifecycle: str = "configured",
) -> dict[str, Any]:
    preview_response = client.post(
        "/api/v1/scheduled-tasks/previews",
        headers=auth_headers,
        json=_automation_payload(name=name),
    )
    assert preview_response.status_code == 201, preview_response.text  # nosec B101
    definition_response = client.post(
        "/api/v1/scheduled-tasks/definitions",
        headers=auth_headers,
        json={"preview_id": preview_response.json()["id"], "initial_lifecycle": initial_lifecycle},
    )
    assert definition_response.status_code == 201, definition_response.text  # nosec B101
    return definition_response.json()


def test_scheduled_tasks_endpoint_combines_reminders_and_watchlist_jobs(scheduled_tasks_client, auth_headers):
    collections_db = CollectionsDatabase.for_user(user_id=880)
    reminder_task_id = collections_db.create_reminder_task(
        title="Review notes",
        body="Check the backlog",
        schedule_kind="one_time",
        run_at="2026-03-21T09:00:00+00:00",
        cron=None,
        timezone=None,
        enabled=True,
    )

    watchlists_db = WatchlistsDatabase.for_user(user_id=880)
    watchlists_db.ensure_schema()
    watchlist_job = watchlists_db.create_job(
        name="Morning brief",
        description=None,
        scope_json=json.dumps({"sources": [1]}),
        schedule_expr="0 8 * * *",
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=json.dumps({}),
        output_prefs_json=json.dumps({}),
    )

    response = scheduled_tasks_client.get("/api/v1/scheduled-tasks", headers=auth_headers)

    assert response.status_code == 200, response.text  # nosec B101 - pytest assertion
    body = response.json()
    assert body["partial"] is False  # nosec B101 - pytest assertion
    assert body["errors"] == []  # nosec B101 - pytest assertion
    assert {item["primitive"] for item in body["items"]} == {"reminder_task", "watchlist_job"}  # nosec B101
    reminder_item = next(item for item in body["items"] if item["primitive"] == "reminder_task")
    assert reminder_item["primitive"] == "reminder_task"  # nosec B101 - pytest assertion
    assert reminder_item["source_ref"]["task_id"] == reminder_task_id  # nosec B101 - pytest assertion
    watchlist_item = next(item for item in body["items"] if item["primitive"] == "watchlist_job")
    assert watchlist_item["primitive"] == "watchlist_job"  # nosec B101 - pytest assertion
    assert watchlist_item["source_ref"]["job_id"] == watchlist_job.id  # nosec B101 - pytest assertion
    assert watchlist_item["edit_mode"] == "external"  # nosec B101 - pytest assertion
    assert watchlist_item["manage_url"] == "/watchlists?tab=jobs"  # nosec B101 - pytest assertion


def test_scheduled_tasks_endpoint_projects_automation_definitions(scheduled_tasks_client, auth_headers):
    definition = _create_automation_definition(
        scheduled_tasks_client,
        auth_headers,
        name="Track API question",
    )
    definition_id = definition["id"]

    list_response = scheduled_tasks_client.get("/api/v1/scheduled-tasks", headers=auth_headers)

    assert list_response.status_code == 200, list_response.text  # nosec B101
    body = list_response.json()
    assert body["partial"] is False  # nosec B101
    item = next(item for item in body["items"] if item["primitive"] == "automation_definition")
    assert item["id"] == f"automation_definition:{definition_id}"  # nosec B101
    assert item["status"] == "configured_execution_unavailable"  # nosec B101
    assert item["enabled"] is True  # nosec B101
    assert item["edit_mode"] == "native"  # nosec B101
    assert item["next_run_at"] is None  # nosec B101
    assert item["last_run_at"] is None  # nosec B101
    assert item["source_ref"]["family"] == "recurring_question"  # nosec B101
    assert item["source_ref"]["health"] == "execution_unavailable"  # nosec B101
    assert item["source_ref"]["execution_available"] is False  # nosec B101

    detail_response = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/automation_definition:{definition_id}",
        headers=auth_headers,
    )

    assert detail_response.status_code == 200, detail_response.text  # nosec B101
    assert detail_response.json() == item  # nosec B101


def test_scheduled_tasks_endpoint_projects_paused_and_archived_automation_statuses(
    scheduled_tasks_client,
    auth_headers,
):
    paused_definition = _create_automation_definition(
        scheduled_tasks_client,
        auth_headers,
        name="Paused API question",
        initial_lifecycle="paused",
    )
    configured_definition = _create_automation_definition(
        scheduled_tasks_client,
        auth_headers,
        name="Archived API question",
    )
    archived_response = scheduled_tasks_client.post(
        f"/api/v1/scheduled-tasks/definitions/{configured_definition['id']}/archive",
        headers=auth_headers,
    )
    assert archived_response.status_code == 200, archived_response.text  # nosec B101

    response = scheduled_tasks_client.get("/api/v1/scheduled-tasks", headers=auth_headers)

    assert response.status_code == 200, response.text  # nosec B101
    items = {item["id"]: item for item in response.json()["items"] if item["primitive"] == "automation_definition"}
    paused_item = items[f"automation_definition:{paused_definition['id']}"]
    archived_item = items[f"automation_definition:{configured_definition['id']}"]
    assert paused_item["enabled"] is False  # nosec B101
    assert paused_item["status"] == "paused"  # nosec B101
    assert archived_item["enabled"] is False  # nosec B101
    assert archived_item["status"] == "archived"  # nosec B101


def test_scheduled_tasks_endpoint_returns_partial_payload_when_watchlist_jobs_fail(
    scheduled_tasks_client,
    auth_headers,
    monkeypatch,
):
    collections_db = CollectionsDatabase.for_user(user_id=880)
    reminder_task_id = collections_db.create_reminder_task(
        title="Partial reminder",
        body="Should still render",
        schedule_kind="one_time",
        run_at="2026-03-22T09:00:00+00:00",
        cron=None,
        timezone=None,
        enabled=True,
    )

    class _BrokenWatchlistsDb:
        def list_jobs(self, q, limit, offset):
            raise RuntimeError("watchlists_unavailable")

    monkeypatch.setattr(
        ScheduledTasksControlPlaneService,
        "_watchlists_db",
        staticmethod(lambda user_id: _BrokenWatchlistsDb()),
    )

    response = scheduled_tasks_client.get("/api/v1/scheduled-tasks", headers=auth_headers)

    assert response.status_code == 200, response.text  # nosec B101 - pytest assertion
    body = response.json()
    assert body["partial"] is True  # nosec B101 - pytest assertion
    assert body["errors"] == ["watchlist_jobs_unavailable"]  # nosec B101 - pytest assertion
    assert {item["primitive"] for item in body["items"]} == {"reminder_task"}  # nosec B101 - pytest assertion
    assert any(item["source_ref"]["task_id"] == reminder_task_id for item in body["items"])  # nosec B101


def test_scheduled_tasks_endpoint_returns_partial_payload_when_automation_definitions_fail(
    scheduled_tasks_client,
    auth_headers,
    monkeypatch,
):
    collections_db = CollectionsDatabase.for_user(user_id=880)
    reminder_task_id = collections_db.create_reminder_task(
        title="Partial automation reminder",
        body="Should still render",
        schedule_kind="one_time",
        run_at="2026-03-24T09:00:00+00:00",
        cron=None,
        timezone=None,
        enabled=True,
    )
    watchlists_db = WatchlistsDatabase.for_user(user_id=880)
    watchlists_db.ensure_schema()
    watchlist_job = watchlists_db.create_job(
        name="Partial automation watchlist",
        description=None,
        scope_json=json.dumps({"sources": [2]}),
        schedule_expr="0 9 * * *",
        schedule_timezone="UTC",
        active=True,
        max_concurrency=None,
        per_host_delay_ms=None,
        retry_policy_json=json.dumps({}),
        output_prefs_json=json.dumps({}),
    )

    class _BrokenScheduledTasksDb:
        def list_definitions(self, **kwargs):
            raise RuntimeError("automation_definitions_unavailable")

    monkeypatch.setattr(
        ScheduledTasksControlPlaneService,
        "_scheduled_tasks_db",
        staticmethod(lambda user_id: _BrokenScheduledTasksDb()),
    )

    response = scheduled_tasks_client.get("/api/v1/scheduled-tasks", headers=auth_headers)

    assert response.status_code == 200, response.text  # nosec B101
    body = response.json()
    assert body["partial"] is True  # nosec B101
    assert "automation_definitions_unavailable" in body["errors"]  # nosec B101
    primitives = {item["primitive"] for item in body["items"]}
    assert {"reminder_task", "watchlist_job"} <= primitives  # nosec B101
    assert any(item["source_ref"].get("task_id") == reminder_task_id for item in body["items"])  # nosec B101
    assert any(item["source_ref"].get("job_id") == watchlist_job.id for item in body["items"])  # nosec B101


def test_scheduled_tasks_reminder_mutations_proxy_native_crud(scheduled_tasks_client, auth_headers):
    create_response = scheduled_tasks_client.post(
        "/api/v1/scheduled-tasks/reminders",
        headers=auth_headers,
        json={
            "title": "Follow up",
            "body": "Send the update",
            "schedule_kind": "one_time",
            "run_at": "2026-03-21T10:00:00+00:00",
            "enabled": True,
        },
    )

    assert create_response.status_code == 201, create_response.text  # nosec B101 - pytest assertion
    created = create_response.json()
    assert created["primitive"] == "reminder_task"  # nosec B101 - pytest assertion
    task_id = created["source_ref"]["task_id"]

    detail_response = scheduled_tasks_client.get(
        f"/api/v1/scheduled-tasks/{created['id']}",
        headers=auth_headers,
    )
    assert detail_response.status_code == 200, detail_response.text  # nosec B101 - pytest assertion
    assert detail_response.json()["source_ref"]["task_id"] == task_id  # nosec B101 - pytest assertion

    patch_response = scheduled_tasks_client.patch(
        f"/api/v1/scheduled-tasks/reminders/{task_id}",
        headers=auth_headers,
        json={"enabled": False, "title": "Updated follow up"},
    )
    assert patch_response.status_code == 200, patch_response.text  # nosec B101 - pytest assertion
    assert patch_response.json()["enabled"] is False  # nosec B101 - pytest assertion
    assert patch_response.json()["title"] == "Updated follow up"  # nosec B101 - pytest assertion

    delete_response = scheduled_tasks_client.delete(
        f"/api/v1/scheduled-tasks/reminders/{task_id}",
        headers=auth_headers,
    )
    assert delete_response.status_code == 200, delete_response.text  # nosec B101 - pytest assertion
    assert delete_response.json()["deleted"] is True  # nosec B101 - pytest assertion


def test_scheduled_tasks_routes_require_native_task_permissions(client_user_only, auth_headers):
    try:
        _override_auth(client_user_only, permissions=[])
        list_response = client_user_only.get("/api/v1/scheduled-tasks", headers=auth_headers)
        assert list_response.status_code == 403, list_response.text  # nosec B101 - pytest assertion

        _override_auth(client_user_only, permissions=[TASKS_READ])
        create_response = client_user_only.post(
            "/api/v1/scheduled-tasks/reminders",
            headers=auth_headers,
            json={
                "title": "Permission check",
                "body": "Should be denied",
                "schedule_kind": "one_time",
                "run_at": "2026-03-23T10:00:00+00:00",
                "enabled": True,
            },
        )
        assert create_response.status_code == 403, create_response.text  # nosec B101 - pytest assertion
    finally:
        client_user_only.app.dependency_overrides.pop(get_auth_principal, None)
        client_user_only.app.dependency_overrides.pop(get_request_user, None)

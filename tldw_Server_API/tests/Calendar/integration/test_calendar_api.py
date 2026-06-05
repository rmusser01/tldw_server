from __future__ import annotations

import importlib
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_control_plane_schemas import ScheduledTask
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase

pytestmark = pytest.mark.integration


class _ReminderServiceStub:
    def __init__(self) -> None:
        self.calls: list[tuple[int, Any]] = []

    async def create_reminder(self, *, user_id: int, payload: Any) -> ScheduledTask:
        self.calls.append((user_id, payload))
        return ScheduledTask(
            id="reminder-1",
            primitive="reminder_task",
            title=payload.title,
            description=payload.body,
            status="scheduled",
            enabled=payload.enabled,
            schedule_summary="2026-06-05T18:00:00Z",
            timezone=payload.timezone,
            next_run_at=payload.run_at,
            edit_mode="native",
            manage_url="/scheduled-tasks/reminders/reminder-1",
            source_ref={
                "link_type": payload.link_type,
                "link_id": payload.link_id,
                "link_url": payload.link_url,
            },
        )


@pytest.fixture()
def calendar_api_client(
    tmp_path: Path,
) -> Generator[tuple[TestClient, CalendarDatabase, _ReminderServiceStub], None, None]:
    db = CalendarDatabase(db_path=tmp_path / "calendar_api.db")
    db.ensure_schema()
    reminder_service = _ReminderServiceStub()
    current_user_id = {"value": 1}

    async def override_user() -> User:
        user_id = current_user_id["value"]
        return User(
            id=user_id,
            username=f"user-{user_id}",
            email=f"user-{user_id}@example.test",
            is_active=True,
            is_admin=True,
            roles=["admin"],
            permissions=["*"],
        )

    calendar_endpoint = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.calendar")

    app = FastAPI()
    app.include_router(calendar_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[calendar_endpoint.get_calendar_database] = lambda: db
    app.dependency_overrides[calendar_endpoint.get_scheduled_tasks_service] = lambda: reminder_service

    try:
        with TestClient(app) as client:
            client.current_user_id = current_user_id  # type: ignore[attr-defined]
            yield client, db, reminder_service
    finally:
        app.dependency_overrides.clear()


def _set_user(client: TestClient, user_id: int) -> None:
    client.current_user_id["value"] = user_id  # type: ignore[attr-defined]


def _create_calendar(client: TestClient, **overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "Research",
        "timezone": "UTC",
        "color": "#2563eb",
        **overrides,
    }
    response = client.post("/api/v1/calendar/calendars", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _create_event(client: TestClient, calendar_id: int, **overrides: Any) -> dict[str, Any]:
    payload = {
        "calendar_id": calendar_id,
        "kind": "event",
        "title": "Planning",
        "start_at": "2026-06-05T17:00:00Z",
        "end_at": "2026-06-05T18:00:00Z",
        **overrides,
    }
    response = client.post("/api/v1/calendar/items", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _create_provider_item(db: CalendarDatabase, *, owner_user_id: int = 1, org_id: int | None = None):
    calendar = db.create_calendar(
        tenant_id="default",
        owner_user_id=owner_user_id,
        org_id=org_id,
        name="Imported",
        timezone="UTC",
        color="#64748b",
        visibility="shared" if org_id is not None else "private",
    )
    account = db.create_external_account(
        tenant_id="default",
        user_id=owner_user_id,
        provider="caldav",
        display_name="Fastmail",
        secret_ref=None,
    )
    binding = db.create_external_binding(
        account_id=account.id,
        calendar_id=calendar.id,
        remote_calendar_id="remote-calendar",
    )
    item = db.upsert_provider_item(
        calendar_id=calendar.id,
        external_binding_id=binding.id,
        source_uid="remote-event-1",
        title="Imported meeting",
        start_at="2026-06-05T17:00:00Z",
        end_at="2026-06-05T18:00:00Z",
        provider_payload_json={"uid": "remote-event-1"},
    )
    return calendar, item


def test_create_calendar(calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub]) -> None:
    client, _db, _reminder_service = calendar_api_client

    calendar = _create_calendar(client, name="Deep Work", description="Focus blocks")

    assert calendar["name"] == "Deep Work"
    assert calendar["owner_user_id"] == 1
    assert calendar["timezone"] == "UTC"


def test_list_visible_calendars(calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub]) -> None:
    client, _db, _reminder_service = calendar_api_client
    visible = _create_calendar(client, name="Visible")
    _set_user(client, 2)
    hidden = _create_calendar(client, name="Hidden")

    response = client.get("/api/v1/calendar/calendars")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [calendar["id"] for calendar in payload["items"]] == [hidden["id"]]
    assert visible["id"] not in [calendar["id"] for calendar in payload["items"]]


def test_membership_add_list_remove_and_owner_only_management(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)
    role_by_principal = {
        "2": "viewer",
        "3": "editor",
        "4": "commenter",
    }

    for principal_id, role in role_by_principal.items():
        added = client.post(
            f"/api/v1/calendar/calendars/{calendar['id']}/memberships",
            json={"principal_type": "user", "principal_id": principal_id, "role": role},
        )
        assert added.status_code == 201, added.text
        assert added.json()["principal_id"] == principal_id
        assert added.json()["role"] == role

    listed = client.get(f"/api/v1/calendar/calendars/{calendar['id']}/memberships")
    assert listed.status_code == 200, listed.text
    listed_roles = {
        row["principal_id"]: row["role"]
        for row in listed.json()["items"]
        if row["principal_id"] in role_by_principal
    }
    assert listed_roles == role_by_principal

    _set_user(client, 2)
    denied = client.post(
        f"/api/v1/calendar/calendars/{calendar['id']}/memberships",
        json={"principal_type": "user", "principal_id": "5", "role": "viewer"},
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "calendar_permission_denied"
    denied_list = client.get(f"/api/v1/calendar/calendars/{calendar['id']}/memberships")
    assert denied_list.status_code == 403
    assert denied_list.json()["detail"]["code"] == "calendar_permission_denied"
    denied_remove = client.delete(f"/api/v1/calendar/calendars/{calendar['id']}/memberships/user/3")
    assert denied_remove.status_code == 403
    assert denied_remove.json()["detail"]["code"] == "calendar_permission_denied"

    _set_user(client, 1)
    for principal_id in role_by_principal:
        removed = client.delete(f"/api/v1/calendar/calendars/{calendar['id']}/memberships/user/{principal_id}")
        assert removed.status_code == 200, removed.text
        assert removed.json()["removed"] == 1

    final_list = client.get(f"/api/v1/calendar/calendars/{calendar['id']}/memberships")
    assert final_list.status_code == 200, final_list.text
    final_principals = {row["principal_id"] for row in final_list.json()["items"]}
    assert final_principals.isdisjoint(role_by_principal)


def test_create_event_and_todo_items(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)

    event = _create_event(client, calendar["id"], title="Kickoff")
    todo_response = client.post(
        "/api/v1/calendar/items",
        json={
            "calendar_id": calendar["id"],
            "kind": "todo",
            "title": "Send notes",
            "due_at": "2026-06-06T17:00:00Z",
        },
    )

    assert event["kind"] == "event"
    assert event["source_owner"] == "tldw"
    assert todo_response.status_code == 201, todo_response.text
    assert todo_response.json()["kind"] == "todo"


def test_create_item_validates_required_time_fields(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)

    event = client.post(
        "/api/v1/calendar/items",
        json={"calendar_id": calendar["id"], "kind": "event", "title": "No start"},
    )
    todo = client.post(
        "/api/v1/calendar/items",
        json={"calendar_id": calendar["id"], "kind": "todo", "title": "No date"},
    )

    assert event.status_code == 422
    assert todo.status_code == 422


def test_update_rejects_provider_owned_items(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, db, _reminder_service = calendar_api_client
    _calendar, item = _create_provider_item(db)

    response = client.patch(f"/api/v1/calendar/items/{item.id}", json={"title": "Edited"})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "item_read_only"


def test_agenda_requires_explicit_bounded_range(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client

    missing = client.get("/api/v1/calendar/views/agenda")
    too_large = client.get(
        "/api/v1/calendar/views/agenda",
        params={
            "start_at": "2026-01-01T00:00:00Z",
            "end_at": "2027-12-31T00:00:00Z",
        },
    )

    assert missing.status_code == 422
    assert too_large.status_code == 400
    assert too_large.json()["detail"]["code"] == "calendar_validation_error"


def test_agenda_returns_items_in_bounded_range(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"], title="In range")

    response = client.get(
        "/api/v1/calendar/views/agenda",
        params={
            "start_at": "2026-06-05T00:00:00Z",
            "end_at": "2026-06-06T00:00:00Z",
            "include_scheduled_tasks": "false",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [view_item["calendar_item_id"] for view_item in payload["items"]] == [item["id"]]


def test_create_annotation(calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub]) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"])

    response = client.post(
        f"/api/v1/calendar/items/{item['id']}/annotations",
        json={"body": "Bring notes", "tags": ["meeting"]},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["body"] == "Bring notes"
    assert payload["tags"] == ["meeting"]


def test_create_link(calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub]) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"])

    response = client.post(
        f"/api/v1/calendar/items/{item['id']}/links",
        json={"target_type": "note", "target_id": "note-1", "label": "Briefing"},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["target_type"] == "note"
    assert payload["target_id"] == "note-1"


def test_copy_provider_item_into_local_calendar(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, db, _reminder_service = calendar_api_client
    _provider_calendar, provider_item = _create_provider_item(db)
    target_calendar = _create_calendar(client, name="Local")

    response = client.post(
        f"/api/v1/calendar/items/{provider_item.id}/copy",
        json={"target_calendar_id": target_calendar["id"], "title": "Local copy"},
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["title"] == "Local copy"
    assert payload["source_owner"] == "tldw"
    assert payload["copied_from_item_id"] == provider_item.id


def test_create_calendar_reminder_calls_existing_reminder_primitive(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, reminder_service = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"])

    response = client.post(
        "/api/v1/calendar/reminders",
        json={
            "calendar_item_id": item["id"],
            "title": "Prep for planning",
            "body": "Review agenda",
            "schedule_kind": "one_time",
            "run_at": "2026-06-05T16:30:00Z",
            "timezone": "UTC",
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["scheduled_task"]["id"] == "reminder-1"
    assert payload["calendar_item_id"] == item["id"]
    assert payload["projection"]["source_owner"] == "linked_projection"
    assert reminder_service.calls[0][0] == 1
    assert reminder_service.calls[0][1].link_type == "calendar_item"
    assert reminder_service.calls[0][1].link_id == str(item["id"])


def test_personal_provider_imports_are_hidden_from_shared_org_queries_until_copied(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, db, _reminder_service = calendar_api_client
    _personal_calendar, provider_item = _create_provider_item(db, owner_user_id=1)
    org_calendar = _create_calendar(client, name="Org", org_id=42, visibility="shared")
    db.create_membership(
        calendar_id=org_calendar["id"],
        principal_type="user",
        principal_id="2",
        role="viewer",
    )
    window = {
        "start_at": "2026-06-05T00:00:00Z",
        "end_at": "2026-06-06T00:00:00Z",
        "calendar_ids": str(org_calendar["id"]),
        "include_scheduled_tasks": "false",
    }

    _set_user(client, 2)
    before_copy = client.get("/api/v1/calendar/views/agenda", params=window)
    assert before_copy.status_code == 200, before_copy.text
    assert before_copy.json()["items"] == []

    _set_user(client, 1)
    copy = client.post(
        f"/api/v1/calendar/items/{provider_item.id}/copy",
        json={"target_calendar_id": org_calendar["id"], "title": "Shared copy"},
    )
    assert copy.status_code == 201, copy.text

    _set_user(client, 2)
    after_copy = client.get("/api/v1/calendar/views/agenda", params=window)
    assert after_copy.status_code == 200, after_copy.text
    items = after_copy.json()["items"]
    assert [item["title"] for item in items] == ["Shared copy"]
    assert items[0]["source_owner"] == "tldw"


def test_invalid_raw_rrule_returns_client_error(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)

    response = client.post(
        "/api/v1/calendar/items",
        json={
            "calendar_id": calendar["id"],
            "kind": "event",
            "title": "Bad recurrence",
            "start_at": "2026-06-05T17:00:00Z",
            "recurrence": {"rrule": "FREQ=DAILY;INTERVAL=abc"},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "value_error"


def test_external_binding_placeholders_enforce_account_owner_scope(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client, name="Personal import target")
    account = client.post(
        "/api/v1/calendar/external/accounts",
        json={"provider": "caldav", "display_name": "Fastmail"},
    )
    assert account.status_code == 201, account.text

    _set_user(client, 2)
    response = client.post(
        "/api/v1/calendar/external/bindings",
        json={
            "account_id": account.json()["id"],
            "calendar_id": calendar["id"],
            "remote_calendar_id": "remote-calendar",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "calendar_permission_denied"


def test_external_binding_list_and_sync_placeholders_enforce_owner_scope(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client, name="Personal import target")
    account = client.post(
        "/api/v1/calendar/external/accounts",
        json={"provider": "caldav", "display_name": "Fastmail"},
    )
    assert account.status_code == 201, account.text
    binding = client.post(
        "/api/v1/calendar/external/bindings",
        json={
            "account_id": account.json()["id"],
            "calendar_id": calendar["id"],
            "remote_calendar_id": "remote-calendar",
        },
    )
    assert binding.status_code == 201, binding.text

    _set_user(client, 2)
    listed = client.get(f"/api/v1/calendar/external/accounts/{account.json()['id']}/bindings")
    synced = client.post(f"/api/v1/calendar/external/bindings/{binding.json()['id']}/sync")

    assert listed.status_code == 403
    assert listed.json()["detail"]["code"] == "calendar_permission_denied"
    assert synced.status_code == 403
    assert synced.json()["detail"]["code"] == "calendar_permission_denied"

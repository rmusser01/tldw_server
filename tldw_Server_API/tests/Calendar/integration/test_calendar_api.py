from __future__ import annotations

import asyncio
import base64
import importlib
import json
import re
import threading
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_control_plane_schemas import ScheduledTask
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("db_quota", [None, (1, 1), (1000, 1000)])
@pytest.mark.parametrize(("resource", "method", "path", "payload", "expected", "quota"), [
    ("calendar.read", "GET", "/calendars", None, 200, (120, 240)),
    ("calendar.write", "POST", "/calendars", {"name": "Limited"}, 201, (60, 120)),
    ("calendar.sync", "GET", "/external/accounts", None, 200, (60, 120)),
])
def test_calendar_catalog_limits_enforce_shared_user_budget(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
    resource: str, method: str, path: str, payload: dict[str, str] | None,
    expected: int, quota: tuple[int, int],
    db_quota: tuple[int, int] | None,
) -> None:
    """Real catalog quotas yield HTTP 429 and cannot be reset by switching auth kinds."""
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
    from tldw_Server_API.app.core.AuthNZ.privilege_catalog import load_catalog

    client, db, _reminders = calendar_api_client
    catalog_path = Path(__file__).resolve().parents[3] / "Config_Files" / "privilege_catalog.yaml"
    monkeypatch.setattr(auth_deps, "load_catalog", lambda: load_catalog(catalog_path))
    monkeypatch.setattr(auth_deps, "_AUTH_DEPS_FALLBACK_RATE_WINDOWS", {})
    monkeypatch.setattr(auth_deps, "time", SimpleNamespace(monotonic=lambda: 100.0))
    identity: dict[str, Any] = {"user_id": 1, "kind": "user"}

    async def authenticated_user(request: Request) -> User:
        request.state.user_id = identity["user_id"]
        request.state.auth = AuthContext(principal=AuthPrincipal(
            kind=identity["kind"], user_id=identity["user_id"], api_key_id=99,
        ))
        return User(id=identity["user_id"], username="limited", is_active=True,
                    is_admin=True, roles=["admin"], permissions=["*"])

    override = dict(zip(("limit_per_min", "burst"), db_quota)) if db_quota else None
    pool = SimpleNamespace(pool=True, fetchone=AsyncMock(return_value=override), fetchall=AsyncMock(return_value=[]))
    client.app.dependency_overrides[get_request_user] = authenticated_user
    client.app.dependency_overrides[auth_deps.get_db_pool] = lambda: pool

    assert auth_deps._catalog_rate_limit_for_resource(resource) == quota
    budget = min(quota[1], db_quota[1]) if db_quota else quota[1]
    for index in range(budget):
        if index == budget // 2:
            identity["kind"] = "api_key"
        response = client.request(method, f"/api/v1/calendar{path}", json=payload)
        assert response.status_code == expected, response.text
    before = db.list_calendars(tenant_id="default")
    limited = client.request(method, f"/api/v1/calendar{path}", json=payload)
    assert limited.status_code == 429, limited.text
    assert int(limited.headers["Retry-After"]) > 0
    assert db.list_calendars(tenant_id="default") == before
    for route in client.app.routes:
        dependencies = getattr(getattr(route, "dependant", None), "dependencies", [])
        if any(getattr(dep.call, "_tldw_rate_limit_resource", None) == resource for dep in dependencies):
            route_path = re.sub(r"\{[^}]+\}", "1", route.path)
            for route_method in route.methods:
                blocked = client.request(route_method, route_path, json=payload)
                assert blocked.status_code == 429, (route_method, route_path, blocked.text)
    identity["user_id"] = 2
    assert client.request(method, f"/api/v1/calendar{path}", json=payload).status_code == expected


@pytest.mark.parametrize("zone", ["Unknown/Zone", "../etc/passwd"])
def test_calendar_creation_invalid_zone_is_rejected_before_persistence(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub], zone: str,
) -> None:
    """Unknown and invalid IANA keys never leave unusable calendars in storage."""
    client, db, _reminders = calendar_api_client
    result = client.post("/api/v1/calendar/calendars", json={"name": "Invalid", "timezone": zone})
    assert result.status_code in {400, 422}, result.text
    assert db.list_calendars(tenant_id="default") == []


@pytest.mark.parametrize("window", [
    {"window_start": "not-a-date"},
    {"window_end": "2026-02-30T00:00:00Z"},
    {"window_start": "2026-06-05"},
    {"window_start": "2026-06-05T00:00:00"},
    {"window_start": "2026-06-05T00:00:00Z", "window_end": "2026-06-05T00:00:00Z"},
    {"window_start": "2026-06-06T00:00:00Z", "window_end": "2026-06-05T00:00:00Z"},
    {"window_start": "2026-06-05T00:00:00Z", "window_end": "2026-06-05T01:00:00+02:00"},
    {"window_start": "1900-01-01T00:00:00Z", "window_end": "2100-01-01T00:00:00Z"},
    {"window_start": "x" * 1000},
])
def test_manual_sync_invalid_window_creates_no_job_or_audit(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, window: dict[str, str],
) -> None:
    """Bad overrides are rejected before idempotency keys, Jobs or queued audit writes."""
    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    manager = client.jobs_manager  # type: ignore[attr-defined]
    original = manager.admit_idempotent_operation
    calls: list[Any] = []

    def create_job(command: Any) -> Any:
        calls.append(command)
        return original(command)

    monkeypatch.setattr(manager, "admit_idempotent_operation", create_job)
    result = client.post(f"/api/v1/calendar/external/bindings/{item.external_binding_id}/sync", json=window)
    assert result.status_code in {400, 422}, result.text
    assert calls == []
    assert db.list_sync_events(binding_id=item.external_binding_id) == []


def test_manual_sync_valid_offsets_are_normalized_in_job_payload(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    """UTC normalization makes equivalent offset windows safe and stable for Jobs."""
    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    result = client.post(f"/api/v1/calendar/external/bindings/{item.external_binding_id}/sync", json={
        "window_start": "2026-06-05T02:00:00+02:00", "window_end": "2026-06-06T02:00:00+02:00",
    })
    assert result.status_code == 200, result.text
    job = client.jobs_manager.get_job(result.json()["job_id"])  # type: ignore[attr-defined]
    assert job["payload"]["window_start"] == "2026-06-05T00:00:00+00:00"
    assert job["payload"]["window_end"] == "2026-06-06T00:00:00+00:00"


def test_sync_history_database_io_uses_request_worker(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ownership reads and bounded history queries all leave the request event loop."""
    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    loop_threads: list[int] = []
    db_threads: list[int] = []

    async def user() -> User:
        loop_threads.append(threading.get_ident())
        return User(id=1, username="owner", is_active=True, is_admin=True, permissions=["*"])

    client.app.dependency_overrides[get_request_user] = user
    for name in ["get_external_binding", "get_external_account", "list_sync_events"]:
        original = getattr(db, name)

        def record(*args: Any, _operation: Any = original, **kwargs: Any) -> Any:
            db_threads.append(threading.get_ident())
            return _operation(*args, **kwargs)

        monkeypatch.setattr(db, name, record)
    result = client.get(f"/api/v1/calendar/external/bindings/{item.external_binding_id}/sync-events")
    assert result.status_code == 200, result.text
    assert len(db_threads) == 3 and loop_threads
    assert set(db_threads).isdisjoint(loop_threads)


@pytest.mark.parametrize("outcome", ["false", "error", "success"])
def test_verified_account_setup_has_no_persistence_before_provider_success(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    """Atomic UI setup verifies supplied secrets off-loop before creating an account."""
    client, db, _reminders = calendar_api_client
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", _calendar_secret_key())
    provider = client.caldav_provider  # type: ignore[attr-defined]
    verification_threads: list[int] = []
    loop_threads: list[int] = []
    secret_refs: list[str] = []
    original_create_secret = db.create_secret_ref

    async def user() -> User:
        loop_threads.append(threading.get_ident())
        return User(id=1, username="owner", is_active=True, is_admin=True, permissions=["*"])

    def create_secret(**kwargs: Any) -> str:
        reference = original_create_secret(**kwargs)
        secret_refs.append(reference)
        return reference

    client.app.dependency_overrides[get_request_user] = user
    monkeypatch.setattr(db, "create_secret_ref", create_secret)

    def verify(**credentials: str) -> dict[str, Any]:
        verification_threads.append(threading.get_ident())
        assert db.list_external_accounts_for_user(user_id=1, tenant_id="default") == []
        assert secret_refs == []
        assert credentials == {
            "server_url": "https://calendar.example.test/", "username": "reader", "password": "app-secret",
        }
        if outcome == "error":
            raise CalendarValidationError("sensitive-provider-error")
        return {"verified": outcome == "success", "error": "sensitive-provider-error"}

    monkeypatch.setattr(provider, "verify_account", verify)
    result = client.post("/api/v1/calendar/external/accounts", json={
        "provider": "caldav", "display_name": "Verified", "server_url": "https://calendar.example.test/",
        "username": "reader", "password": "app-secret", "verify_before_create": True,
    })
    assert result.status_code == (201 if outcome == "success" else 400), result.text
    assert verification_threads
    assert loop_threads and set(verification_threads).isdisjoint(loop_threads)
    assert "sensitive-provider-error" not in result.text
    rows = db.list_external_accounts_for_user(user_id=1, tenant_id="default")
    assert len(rows) == (1 if outcome == "success" else 0)
    assert len(secret_refs) == (1 if outcome == "success" else 0)


@pytest.mark.parametrize("missing", ["server_url", "username", "password"])
def test_verified_account_setup_rejects_incomplete_credentials_without_writes(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, missing: str,
) -> None:
    """Opt-in atomic setup cannot create an active account without complete credentials."""
    client, db, _reminders = calendar_api_client
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", _calendar_secret_key())
    payload: dict[str, Any] = {
        "provider": "caldav", "display_name": "Incomplete", "server_url": "https://calendar.example.test/",
        "username": "reader", "password": "app-secret", "verify_before_create": True,
    }
    payload.pop(missing)
    result = client.post("/api/v1/calendar/external/accounts", json=payload)
    assert result.status_code == 400, result.text
    assert db.list_external_accounts_for_user(user_id=1, tenant_id="default") == []
    assert client.caldav_provider.verify_requests == []  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ["event", "todo"])
def test_native_item_delete_soft_deletes_and_hides_from_views(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub], kind: str,
) -> None:
    """Owners can soft-delete events/todos; other users cannot and deleted rows leave views."""
    client, db, _reminders = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"], kind=kind, due_at="2026-06-05T17:00:00Z")
    _set_user(client, 2)
    assert client.delete(f"/api/v1/calendar/items/{item['id']}").status_code == 403
    _set_user(client, 1)
    response = client.delete(f"/api/v1/calendar/items/{item['id']}")
    assert response.status_code == 200, response.text
    assert response.json() == {"deleted": True}
    assert db.get_item(item["id"], include_deleted=True).deleted_at is not None
    agenda = client.get("/api/v1/calendar/views/agenda", params={
        "start_at": "2026-06-05T00:00:00Z", "end_at": "2026-06-06T00:00:00Z",
        "include_scheduled_tasks": False,
    })
    assert agenda.json()["items"] == []
    assert client.delete(f"/api/v1/calendar/items/{item['id']}").status_code == 404


def test_delete_rejects_provider_owned_items(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    """The native delete API preserves provider-owned records."""
    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    assert client.delete(f"/api/v1/calendar/items/{item.id}").status_code == 409
    assert db.get_item(item.id).deleted_at is None


def test_nonrecurring_views_honor_item_timezone_and_return_offsets(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    """Naive item times use their IANA zone and render as explicit-offset instants."""
    client, _db, _reminders = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"], start_at="2026-06-05T09:00:00",
                         end_at="2026-06-05T10:00:00", timezone="America/Los_Angeles")
    response = client.get("/api/v1/calendar/views/agenda", params={
        "start_at": "2026-06-05T15:00:00Z", "end_at": "2026-06-05T18:00:00Z",
        "include_scheduled_tasks": False,
    })
    rows = response.json()["items"]
    assert [row["calendar_item_id"] for row in rows] == [item["id"]]
    assert rows[0]["start_at"] == "2026-06-05T09:00:00-07:00"
    assert rows[0]["metadata"]["timezone"] == "America/Los_Angeles"
    early = client.get("/api/v1/calendar/views/agenda", params={
        "start_at": "2026-06-05T08:00:00Z", "end_at": "2026-06-05T11:00:00Z",
        "include_scheduled_tasks": False,
    })
    assert early.json()["items"] == []


def test_provider_occurrence_limit_marks_agenda_partial(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    """Capped provider recurrence returns retained occurrences and an incompleteness warning."""
    from tldw_Server_API.app.core.Calendar.constants import MAX_EXPANDED_OCCURRENCES

    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    db.upsert_provider_item(calendar_id=item.calendar_id, external_binding_id=item.external_binding_id,
                            source_uid=item.source_uid, title=item.title,
                            start_at="2026-06-05T00:00:00Z", end_at=None, provider_payload_json={})
    db.upsert_recurrence(calendar_item_id=item.id, rrule="FREQ=MINUTELY")
    response = client.get("/api/v1/calendar/views/agenda", params={
        "start_at": "2026-06-05T00:00:00Z", "end_at": "2026-06-12T00:00:00Z",
        "include_scheduled_tasks": False,
    })
    assert response.status_code == 200, response.text
    assert len(response.json()["items"]) == MAX_EXPANDED_OCCURRENCES
    assert response.json()["partial"] is True
    assert response.json()["warnings"]


def test_explicit_provider_offsets_do_not_require_iana_tzid(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    """Resolved VTIMEZONE offsets remain usable even when their custom TZID is not IANA."""
    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    db.upsert_provider_item(calendar_id=item.calendar_id, external_binding_id=item.external_binding_id,
                            source_uid=item.source_uid, title=item.title,
                            start_at="2026-06-05T09:00:00-05:00", end_at="2026-06-05T10:00:00-05:00",
                            timezone="Custom/FixedEastern", provider_payload_json={})
    response = client.get("/api/v1/calendar/views/agenda", params={
        "start_at": "2026-06-05T14:00:00Z", "end_at": "2026-06-05T15:00:00Z",
        "include_scheduled_tasks": False,
    })
    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["start_at"] == "2026-06-05T09:00:00-05:00"


def test_recurrence_null_removes_rule_and_omission_preserves_it(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    """Using the API fixture, omission retains recurrence while explicit null removes its persisted row."""
    client, db, _reminders = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"], recurrence={"rrule": "FREQ=DAILY;COUNT=3"})
    changed = client.patch(f"/api/v1/calendar/items/{item['id']}", json={"title": "Still recurring"})
    assert changed.json()["recurrence"]["rrule"] == "FREQ=DAILY;COUNT=3"
    cleared = client.patch(f"/api/v1/calendar/items/{item['id']}", json={"recurrence": None})
    assert cleared.status_code == 200
    assert cleared.json()["recurrence"] is None
    assert db.list_recurrences_for_items([item["id"]]) == {}


@pytest.mark.parametrize("operation", ["create", "update"])
def test_recurrence_write_failure_rolls_back_item_mutation(calendar_api_client, monkeypatch, operation) -> None:
    client, db, _reminders = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"])

    def fail(**_kwargs: Any) -> None:
        raise RuntimeError("injected recurrence write failure")

    monkeypatch.setattr(db, "upsert_recurrence", fail)
    if operation == "create":
        response = client.post(
            "/api/v1/calendar/items",
            json={
                "calendar_id": calendar["id"],
                "kind": "event",
                "title": "Must roll back",
                "start_at": "2026-06-05T09:00:00Z",
                "recurrence": {"rrule": "FREQ=DAILY;COUNT=2"},
            },
        )
    else:
        response = client.patch(
            f"/api/v1/calendar/items/{item['id']}",
            json={
                "title": "Must roll back",
                "recurrence": {"rrule": "FREQ=DAILY;COUNT=2"},
            },
        )
    assert response.status_code == 500
    rows = db.list_items_for_expansion(
        calendar_ids=[calendar["id"]], window_start="2026-06-01", window_end="2026-06-08"
    )
    assert [(row.id, row.title) for row in rows] == [(item["id"], "Planning")]


@pytest.mark.parametrize("field", ["rdate", "exdate"])
def test_recurrence_rejects_malformed_dates_before_persistence(calendar_api_client, field) -> None:
    client, _db, _reminders = calendar_api_client
    calendar = _create_calendar(client)
    response = client.post(
        "/api/v1/calendar/items",
        json={
            "calendar_id": calendar["id"],
            "kind": "event",
            "title": "Invalid",
            "start_at": "2026-06-05T09:00:00Z",
            "recurrence": {field: ["not-a-date"]},
        },
    )
    assert response.status_code in {400, 422}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["verify", "discover"])
async def test_caldav_api_provider_calls_run_off_event_loop(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import calendar as endpoint
    from tldw_Server_API.app.api.v1.schemas.calendar_schemas import CalDavAccountVerifyRequest

    _client, db, _reminders = calendar_api_client
    account = db.create_external_account(
        tenant_id="default",
        user_id=1,
        provider="caldav",
        display_name="Work",
        secret_ref=None,
    )
    calls: list[int] = []

    class Provider:
        def verify_account(self, **_kwargs: Any) -> dict[str, Any]:
            calls.append(threading.get_ident())
            return {"verified": True, "status": "ok"}

        def discover_calendars(self, **_kwargs: Any) -> list[Any]:
            calls.append(threading.get_ident())
            return []

    user = User(id=1, username="owner", is_active=True)
    handler = (
        endpoint.verify_external_calendar_account if operation == "verify" else endpoint.discover_external_calendars
    )
    await handler(
        payload=CalDavAccountVerifyRequest(
            server_url="https://calendar.example.test/", username="user", password="secret"
        ),
        account_id=account.id,
        current_user=user,
        _principal=None,
        db=db,
        provider=Provider(),
    )
    assert calls and calls[0] != threading.get_ident()


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


class _CalDavProviderStub:
    def __init__(self) -> None:
        self.verify_requests: list[dict[str, Any]] = []
        self.discovery_requests: list[dict[str, Any]] = []

    def verify_account(self, **kwargs: Any) -> dict[str, Any]:
        self.verify_requests.append(kwargs)
        return {"verified": True, "status": "ok"}

    def discover_calendars(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.discovery_requests.append(kwargs)
        return [
            {
                "remote_calendar_id": "https://caldav.example.test/calendars/user/work/",
                "remote_display_name": "Work",
                "provider_capabilities": {"supports_vevent": True, "sync_strategy": "bounded_polling"},
            }
        ]


@pytest.fixture()
def calendar_api_client(
    tmp_path: Path,
) -> Generator[tuple[TestClient, CalendarDatabase, _ReminderServiceStub], None, None]:
    db = CalendarDatabase(db_path=tmp_path / "calendar_api.db")
    db.ensure_schema()
    jobs_db_path = tmp_path / "calendar_jobs.db"
    ensure_jobs_tables(jobs_db_path)
    jobs_manager = JobManager(jobs_db_path)
    reminder_service = _ReminderServiceStub()
    caldav_provider = _CalDavProviderStub()
    current_user_id = {"value": 1}
    current_tenant_id: dict[str, str | None] = {"value": None}
    current_org_ids: dict[str, list[int]] = {"value": []}

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
            tenant_id=current_tenant_id["value"],
            org_ids=current_org_ids["value"],
        )

    calendar_endpoint = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.calendar")

    app = FastAPI()
    app.include_router(calendar_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[calendar_endpoint.get_calendar_database] = lambda: db
    app.dependency_overrides[calendar_endpoint.get_scheduled_tasks_service] = lambda: reminder_service
    if hasattr(calendar_endpoint, "get_caldav_provider"):
        app.dependency_overrides[calendar_endpoint.get_caldav_provider] = lambda: caldav_provider
    if hasattr(calendar_endpoint, "get_calendar_job_manager"):
        app.dependency_overrides[calendar_endpoint.get_calendar_job_manager] = lambda: jobs_manager

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            client.current_user_id = current_user_id  # type: ignore[attr-defined]
            client.current_tenant_id = current_tenant_id  # type: ignore[attr-defined]
            client.current_org_ids = current_org_ids  # type: ignore[attr-defined]
            client.caldav_provider = caldav_provider  # type: ignore[attr-defined]
            client.jobs_manager = jobs_manager  # type: ignore[attr-defined]
            yield client, db, reminder_service
    finally:
        app.dependency_overrides.clear()


def _set_user(client: TestClient, user_id: int) -> None:
    client.current_user_id["value"] = user_id  # type: ignore[attr-defined]


def _set_tenant(client: TestClient, tenant_id: str | None) -> None:
    client.current_tenant_id["value"] = tenant_id  # type: ignore[attr-defined]


@pytest.mark.parametrize("claimed_orgs, membership", [
    ([], {"org_id": 42, "role": "member", "status": "active"}),
    ([42], {"org_id": 42, "role": "member", "status": "inactive"}),
    ([42], {"org_id": 43, "role": "member", "status": "active"}),
])
def test_create_org_calendar_requires_active_authenticated_membership(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, claimed_orgs: list[int], membership: dict[str, Any],
) -> None:
    """Neither an org claim nor an unrelated/inactive DB membership grants creation."""
    from tldw_Server_API.app.api.v1.endpoints import calendar as endpoint

    client, db, _reminders = calendar_api_client
    client.current_org_ids["value"] = claimed_orgs  # type: ignore[attr-defined]

    async def memberships(user_id: int) -> list[dict[str, Any]]:
        return [membership]

    monkeypatch.setattr(endpoint, "list_org_memberships_for_user", memberships)
    result = client.post("/api/v1/calendar/calendars", json={"name": "Forged org", "org_id": 42})
    assert result.status_code == 403, result.text
    assert db.list_calendars(tenant_id="default") == []


def test_create_org_calendar_allows_active_authenticated_member(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creation membership is independent of a specific calendar access role."""
    from tldw_Server_API.app.api.v1.endpoints import calendar as endpoint

    client, _db, _reminders = calendar_api_client
    client.current_org_ids["value"] = [42]  # type: ignore[attr-defined]

    async def memberships(user_id: int) -> list[dict[str, Any]]:
        return [{"org_id": 42, "role": "member", "status": "active"}]

    monkeypatch.setattr(endpoint, "list_org_memberships_for_user", memberships)
    result = client.post("/api/v1/calendar/calendars", json={"name": "Verified org", "org_id": 42})
    assert result.status_code == 201, result.text
    assert result.json()["org_id"] == 42


@pytest.mark.parametrize("port", ["bad", "65536", "0"])
@pytest.mark.parametrize("url_field", ["server_url", "account_metadata"])
def test_create_caldav_account_validates_url_before_persistence(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub], port: str, url_field: str,
) -> None:
    """Bad account metadata cannot be saved and later cause a binding-time 500."""
    client, db, _reminders = calendar_api_client
    url = f"https://example.test:{port}/dav/"
    result = client.post("/api/v1/calendar/external/accounts", json={
        "provider": "caldav", "display_name": "Bad port",
        url_field: {"server_url": url} if url_field == "account_metadata" else url,
    })
    assert result.status_code == 400, result.text
    assert db.list_external_accounts_for_user(user_id=1, tenant_id="default") == []


@pytest.mark.parametrize("remote_url", ["https://[invalid/cal/", "//[invalid/cal/"])
def test_create_binding_rejects_malformed_collection_authority(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub], remote_url: str,
) -> None:
    """URL resolution failures return a client error without persisting a binding."""
    client, db, _reminders = calendar_api_client
    calendar = _create_calendar(client, name="Collection validation")
    account = client.post("/api/v1/calendar/external/accounts", json={
        "provider": "caldav", "display_name": "Valid account", "server_url": "https://example.test/dav/",
    })
    assert account.status_code == 201, account.text
    result = client.post("/api/v1/calendar/external/bindings", json={
        "account_id": account.json()["id"], "calendar_id": calendar["id"], "remote_calendar_id": remote_url,
    })
    assert result.status_code == 400, result.text
    assert result.json()["detail"]["code"] == "calendar_validation_error"
    assert db.list_external_bindings_for_account(account.json()["id"]) == []


@pytest.mark.parametrize(
    "membership, expected_status",
    [
        ({"org_id": 42, "role": "researcher", "status": "active"}, 201),
        ({"org_id": 43, "role": "researcher", "status": "active"}, 403),
        ({"org_id": 42, "role": "researcher", "status": "inactive"}, 403),
        ({"org_id": 42, "role": "member", "status": "active"}, 403),
    ],
)
def test_api_resolves_only_active_scoped_org_roles(
    calendar_api_client, monkeypatch, membership, expected_status
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import calendar as endpoint

    client, _db, _reminders = calendar_api_client
    async def creator_memberships(user_id: int) -> list[dict[str, Any]]:
        return [{"org_id": 42, "role": "member", "status": "active"}] if user_id == 1 else []

    monkeypatch.setattr(endpoint, "list_org_memberships_for_user", creator_memberships)
    client.current_org_ids["value"] = [42]
    calendar = _create_calendar(client, org_id=42, visibility="shared")
    grant = client.post(
        f"/api/v1/calendar/calendars/{calendar['id']}/memberships",
        json={
            "principal_type": "org_role",
            "principal_id": "researcher",
            "role": "editor",
        },
    )
    assert grant.status_code == 201

    async def memberships(user_id: int) -> list[dict[str, Any]]:
        assert user_id == 2
        return [membership]

    monkeypatch.setattr(endpoint, "list_org_memberships_for_user", memberships, raising=False)
    _set_user(client, 2)
    client.current_org_ids["value"] = [42]
    result = client.post(
        "/api/v1/calendar/items",
        json={
            "calendar_id": calendar["id"],
            "kind": "event",
            "title": "Role member",
            "start_at": "2026-06-05T09:00:00Z",
        },
    )
    assert result.status_code == expected_status, result.text


def test_links_survive_agenda_refresh_and_authorized_removal(calendar_api_client) -> None:
    client, _db, _reminders = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"])
    link = client.post(
        f"/api/v1/calendar/items/{item['id']}/links",
        json={
            "target_type": "url",
            "target_id": "https://example.test/notes",
            "label": "Notes",
            "url": "https://example.test/notes",
        },
    ).json()
    listing = client.get(f"/api/v1/calendar/items/{item['id']}/links")
    assert listing.status_code == 200, listing.text
    assert listing.json()["items"] == [link]
    agenda = client.get(
        "/api/v1/calendar/views/agenda",
        params={
            "start_at": "2026-06-01T00:00:00Z",
            "end_at": "2026-06-08T00:00:00Z",
            "include_scheduled_tasks": False,
        },
    )
    assert agenda.json()["items"][0]["links"] == [link]
    _set_user(client, 2)
    assert client.get(f"/api/v1/calendar/items/{item['id']}/links").status_code == 403
    assert client.delete(f"/api/v1/calendar/items/{item['id']}/links/{link['id']}").status_code == 403
    _set_user(client, 1)
    removed = client.delete(f"/api/v1/calendar/items/{item['id']}/links/{link['id']}")
    assert removed.status_code == 200, removed.text
    assert client.get(f"/api/v1/calendar/items/{item['id']}/links").json()["items"] == []


def test_calendar_exceptions_share_central_exports() -> None:
    from tldw_Server_API.app.core import exceptions
    from tldw_Server_API.app.core.Calendar import errors

    for name in [
        "CalendarError",
        "CalendarNotFound",
        "CalendarValidationError",
        "CalendarPermissionDenied",
        "CalendarItemNotFound",
        "CalendarReadOnlyError",
        "CalendarSyncError",
    ]:
        assert getattr(exceptions, name, None) is getattr(errors, name)


def test_calendar_api_module_and_handlers_document_contracts() -> None:
    import inspect

    from tldw_Server_API.app.api.v1.endpoints import calendar as endpoint

    assert endpoint.__doc__
    functions = [
        function
        for _, function in inspect.getmembers(endpoint, inspect.isfunction)
        if function.__module__ == endpoint.__name__
    ]
    assert all(function.__doc__ for function in functions)


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


def _calendar_secret_key() -> str:
    return base64.b64encode(b"c" * 32).decode("ascii")


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


def test_create_named_calendar_assigns_authenticated_owner_and_default_utc_timezone(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
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
        row["principal_id"]: row["role"] for row in listed.json()["items"] if row["principal_id"] in role_by_principal
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


def test_agenda_rejects_invalid_date_params_with_stable_client_error(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client

    response = client.get(
        "/api/v1/calendar/views/agenda",
        params={
            "start_at": "not-a-date",
            "end_at": "2026-06-06T00:00:00Z",
        },
    )

    assert response.status_code in {400, 422}
    assert response.json()["detail"]["code"] == "calendar_validation_error"


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


def test_week_returns_items_through_week_view_route(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)
    item = _create_event(client, calendar["id"], title="Week item")

    response = client.get(
        "/api/v1/calendar/views/week",
        params={"week_start": "2026-06-01", "timezone": "UTC", "include_scheduled_tasks": "false"},
    )

    assert response.status_code == 200, response.text
    assert [view_item["calendar_item_id"] for view_item in response.json()["items"]] == [item["id"]]


def test_agenda_view_items_include_kind_and_local_tags(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)
    todo_response = client.post(
        "/api/v1/calendar/items",
        json={
            "calendar_id": calendar["id"],
            "kind": "todo",
            "title": "Tag figures",
            "due_at": "2026-06-05T17:00:00Z",
            "local_tags": ["draft", "review"],
        },
    )
    assert todo_response.status_code == 201, todo_response.text

    response = client.get(
        "/api/v1/calendar/views/agenda",
        params={
            "start_at": "2026-06-05T00:00:00Z",
            "end_at": "2026-06-06T00:00:00Z",
            "include_scheduled_tasks": "false",
        },
    )

    assert response.status_code == 200, response.text
    [view_item] = response.json()["items"]
    assert view_item["kind"] == "todo"
    assert view_item["start_at"] == "2026-06-05T17:00:00Z"
    assert view_item["due_at"] == "2026-06-05T17:00:00Z"
    assert view_item["local_tags"] == ["draft", "review"]


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


def test_update_provider_owned_item_local_tags(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, db, _reminder_service = calendar_api_client
    _calendar, provider_item = _create_provider_item(db)

    response = client.put(
        f"/api/v1/calendar/items/{provider_item.id}/local-tags",
        json={"tags": ["remote", "follow-up"]},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["body"] == ""
    assert payload["tags"] == ["remote", "follow-up"]
    assert db.get_item(provider_item.id).local_tags_json is None

    view_response = client.get(
        "/api/v1/calendar/views/agenda",
        params={
            "start_at": "2026-06-05T00:00:00Z",
            "end_at": "2026-06-06T00:00:00Z",
            "include_scheduled_tasks": "false",
        },
    )

    assert view_response.status_code == 200, view_response.text
    [view_item] = view_response.json()["items"]
    assert view_item["calendar_item_id"] == provider_item.id
    assert view_item["local_tags"] == ["remote", "follow-up"]


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import calendar as endpoint

    client, db, _reminder_service = calendar_api_client
    async def memberships(user_id: int) -> list[dict[str, Any]]:
        return [{"org_id": 42, "role": "member", "status": "active"}] if user_id == 1 else []

    monkeypatch.setattr(endpoint, "list_org_memberships_for_user", memberships)
    client.current_org_ids["value"] = [42]
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


def test_create_caldav_account_encrypts_credentials_and_redacts_response(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, db, _reminder_service = calendar_api_client
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", _calendar_secret_key())

    response = client.post(
        "/api/v1/calendar/external/accounts",
        json={
            "provider": "caldav",
            "display_name": "Fastmail",
            "server_url": "https://caldav.example.test/dav/",
            "username": "reader@example.test",
            "password": "app-secret",
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert "secret_ref" not in body
    assert "password" not in json.dumps(body)
    assert body["account_metadata"]["server_url"] == "https://caldav.example.test/dav/"
    assert body["account_metadata"]["username"] == "reader@example.test"

    account = db.get_external_account(body["id"])
    assert account.secret_ref is not None
    encrypted_payload = db.resolve_secret_ref(account.secret_ref)
    assert "reader@example.test" not in encrypted_payload
    assert "app-secret" not in encrypted_payload

    from tldw_Server_API.app.core.Calendar.secret_store import CalendarSecretStore

    assert CalendarSecretStore(db=db, tenant_id="default").resolve_secret(
        owner_user_id=1,
        secret_ref=account.secret_ref,
    ) == {
        "server_url": "https://caldav.example.test/dav/",
        "username": "reader@example.test",
        "password": "app-secret",
    }


@pytest.mark.parametrize("failure", ["validation", "after_insert"])
def test_account_creation_failure_rolls_back_secret_and_account(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Failed creation leaves neither an encrypted credential nor a partially created account."""
    client, db, _reminders = calendar_api_client
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", _calendar_secret_key())
    created_refs: list[str] = []
    original_secret = db.create_secret_ref
    original_account = db.create_external_account

    def capture_secret(**kwargs: Any) -> str:
        ref = original_secret(**kwargs)
        created_refs.append(ref)
        return ref

    def fail_account(**kwargs: Any) -> None:
        if failure == "after_insert":
            original_account(**kwargs)
            raise RuntimeError("Injected account persistence failure")
        raise CalendarValidationError("Injected account validation failure")

    monkeypatch.setattr(db, "create_secret_ref", capture_secret)
    monkeypatch.setattr(db, "create_external_account", fail_account)
    response = client.post("/api/v1/calendar/external/accounts", json={
        "provider": "caldav", "display_name": "Atomic account",
        "server_url": "https://caldav.example.test/dav/", "username": "reader@example.test",
        "password": "app-secret",
    })
    assert response.status_code == (400 if failure == "validation" else 500)
    assert len(created_refs) == 1
    assert db.list_external_accounts_for_user(user_id=1, tenant_id="default") == []
    with pytest.raises(CalendarValidationError):
        db.resolve_secret_ref(created_refs[0])


def test_create_caldav_account_requires_encryption_key_for_credentials(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _db, _reminder_service = calendar_api_client
    monkeypatch.delenv("CALENDAR_SECRET_ENCRYPTION_KEY", raising=False)

    response = client.post(
        "/api/v1/calendar/external/accounts",
        json={
            "provider": "caldav",
            "display_name": "Fastmail",
            "server_url": "https://caldav.example.test/dav/",
            "username": "reader@example.test",
            "password": "app-secret",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "calendar_validation_error"
    assert "CALENDAR_SECRET_ENCRYPTION_KEY" in response.json()["detail"]["message"]


@pytest.mark.parametrize(
    ("method", "path_suffix", "response_key"),
    [("post", "/revoke", "revoked"), ("delete", "", "deleted")],
)
def test_caldav_account_revoke_and_delete_clear_secret_material(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path_suffix: str,
    response_key: str,
) -> None:
    client, db, _reminder_service = calendar_api_client
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", _calendar_secret_key())
    account_response = client.post(
        "/api/v1/calendar/external/accounts",
        json={
            "provider": "caldav",
            "display_name": "Fastmail",
            "server_url": "https://caldav.example.test/dav/",
            "username": "reader@example.test",
            "password": "app-secret",
        },
    )
    assert account_response.status_code == 201, account_response.text
    account = db.get_external_account(account_response.json()["id"])
    assert account.secret_ref is not None

    response = getattr(client, method)(
        f"/api/v1/calendar/external/accounts/{account.id}{path_suffix}",
    )

    assert response.status_code == 200, response.text
    assert response.json()[response_key] is True
    with pytest.raises(CalendarValidationError):
        db.resolve_secret_ref(account.secret_ref)


def test_caldav_account_verify_and_discover_use_stored_secret(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _db, _reminder_service = calendar_api_client
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", _calendar_secret_key())
    account_response = client.post(
        "/api/v1/calendar/external/accounts",
        json={
            "provider": "caldav",
            "display_name": "Fastmail",
            "server_url": "https://caldav.example.test/dav/",
            "username": "reader@example.test",
            "password": "app-secret",
        },
    )
    assert account_response.status_code == 201, account_response.text
    account_id = account_response.json()["id"]
    provider = client.caldav_provider  # type: ignore[attr-defined]

    verify_response = client.post(f"/api/v1/calendar/external/accounts/{account_id}/verify")
    discover_response = client.post(f"/api/v1/calendar/external/accounts/{account_id}/discover")

    assert verify_response.status_code == 200, verify_response.text
    assert verify_response.json() == {"account_id": account_id, "verified": True, "status": "ok", "error": None}
    assert discover_response.status_code == 200, discover_response.text
    assert discover_response.json()["items"] == [
        {
            "remote_calendar_id": "https://caldav.example.test/calendars/user/work/",
            "remote_display_name": "Work",
            "provider_capabilities": {"supports_vevent": True, "sync_strategy": "bounded_polling"},
        }
    ]
    assert provider.verify_requests[0]["password"] == "app-secret"
    assert provider.verify_requests[0]["account_id"] == account_id
    assert provider.discovery_requests[0]["password"] == "app-secret"
    assert "app-secret" not in verify_response.text
    assert "app-secret" not in discover_response.text


@pytest.mark.parametrize("operation", ["verify", "discover"])
@pytest.mark.parametrize("destination", [
    "https://other.example.test/dav/", "https://caldav.example.test:444/dav/", "http://caldav.example.test/dav/",
])
@pytest.mark.parametrize("replacement", [{}, {"username": "replacement"}, {"password": "replacement"}])
def test_provider_request_rejects_other_origin_when_reusing_any_stored_credential(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, operation: str, destination: str, replacement: dict[str, str],
) -> None:
    """A request URL cannot redirect either stored Basic credential to another origin."""
    client, _db, _reminders = calendar_api_client
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", _calendar_secret_key())
    account = client.post("/api/v1/calendar/external/accounts", json={
        "provider": "caldav", "display_name": "Pinned account",
        "server_url": "https://caldav.example.test/dav/", "username": "reader@example.test",
        "password": "app-secret",
    }).json()
    response = client.post(f"/api/v1/calendar/external/accounts/{account['id']}/{operation}",
                           json={"server_url": destination, **replacement})
    assert response.status_code == 400, response.text
    provider = client.caldav_provider  # type: ignore[attr-defined]
    assert provider.verify_requests == provider.discovery_requests == []
    assert "app-secret" not in response.text


@pytest.mark.parametrize("override_url", [False, True])
def test_shared_credentials_reject_unpinned_fallback_without_saved_account_origin(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, override_url: bool,
) -> None:
    """A worker fallback is not a trusted origin for an account with unpinned stored credentials."""
    from tldw_Server_API.app.core.Calendar.provider_operations import resolve_caldav_credentials
    from tldw_Server_API.app.core.Calendar.secret_store import CalendarSecretStore

    _client, db, _reminders = calendar_api_client
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", _calendar_secret_key())
    ref = CalendarSecretStore(db=db).create_secret(
        owner_user_id=1, provider="caldav", payload={"username": "reader", "password": "app-secret"},
    )
    account = db.create_external_account(tenant_id="default", user_id=1, provider="caldav",
                                        display_name="Unpinned account", secret_ref=ref)
    target = "https://other.example.test/dav/"
    with pytest.raises(CalendarValidationError, match="account server origin"):
        resolve_caldav_credentials(db, account_id=account.id, actor_user_id=1, tenant_id="default",
                                  overrides={"server_url": target} if override_url else None,
                                  fallback_server_url=target)


@pytest.mark.parametrize("operation", ["verify", "discover"])
@pytest.mark.parametrize("replacement", [
    {"server_url": "https://CALDAV.example.test:443/other/"},
    {"server_url": "https://other.example.test/dav/", "username": "explicit-user", "password": "explicit-secret"},
    {"server_url": "https://other.example.test/dav/", "username": "explicit-user", "token": "explicit-secret"},
])
def test_provider_request_allows_same_origin_or_complete_explicit_credentials(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, operation: str, replacement: dict[str, str],
) -> None:
    """Paths/default ports are origin-equivalent; complete replacements reuse no stored credentials."""
    client, _db, _reminders = calendar_api_client
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", _calendar_secret_key())
    account = client.post("/api/v1/calendar/external/accounts", json={
        "provider": "caldav", "display_name": "Pinned account",
        "server_url": "https://caldav.example.test/dav/", "username": "reader@example.test",
        "password": "app-secret",
    }).json()
    response = client.post(f"/api/v1/calendar/external/accounts/{account['id']}/{operation}", json=replacement)
    assert response.status_code == 200, response.text
    provider = client.caldav_provider  # type: ignore[attr-defined]
    requests = provider.verify_requests if operation == "verify" else provider.discovery_requests
    expected: dict[str, Any] = {"server_url": replacement["server_url"],
                "username": replacement.get("username", "reader@example.test"),
                "password": replacement.get("password", replacement.get("token", "app-secret"))}
    if operation == "verify":
        expected["account_id"] = account["id"]
    assert requests == [expected]


@pytest.mark.parametrize("kind", ["event", "todo"])
def test_created_wall_clock_item_inherits_calendar_zone_in_agenda_and_week(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub], kind: str,
) -> None:
    """Omitted item zones interpret local event/todo times in the selected calendar's zone."""
    client, _db, _reminders = calendar_api_client
    calendar = _create_calendar(client, timezone="America/Los_Angeles")
    fields = {"start_at": "2026-06-05T09:00:00", "end_at": "2026-06-05T10:00:00"} if kind == "event" else {
        "start_at": None, "end_at": None, "due_at": "2026-06-05T09:00:00",
    }
    item = _create_event(client, calendar["id"], kind=kind, **fields)
    assert item["timezone"] == "America/Los_Angeles"
    for view, window in [
        ("agenda", {"start_at": "2026-06-05T15:00:00Z", "end_at": "2026-06-05T18:00:00Z"}),
        ("week", {"week_start": "2026-06-01", "timezone": "America/Los_Angeles"}),
    ]:
        response = client.get(f"/api/v1/calendar/views/{view}", params={**window, "include_scheduled_tasks": False})
        assert response.status_code == 200, response.text
        rows = response.json()["items"]
        assert [row["calendar_item_id"] for row in rows] == [item["id"]]
        assert rows[0]["start_at"] == "2026-06-05T09:00:00-07:00"


def test_binding_rejects_remote_calendar_on_other_origin(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    calendar = _create_calendar(client)
    account = client.post(
        "/api/v1/calendar/external/accounts",
        json={
            "provider": "caldav",
            "display_name": "External",
            "server_url": "https://caldav.example.test/dav/",
            "username": "reader@example.test",
        },
    )
    assert account.status_code == 201, account.text

    response = client.post(
        "/api/v1/calendar/external/bindings",
        json={
            "account_id": account.json()["id"],
            "calendar_id": calendar["id"],
            "remote_calendar_id": "https://other.example.test/calendar/",
            "remote_display_name": "Other",
        },
    )

    assert response.status_code == 400, response.text


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


def test_external_binding_uses_request_user_tenant_for_created_calendar(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, _db, _reminder_service = calendar_api_client
    _set_tenant(client, "tenant-a")

    calendar = _create_calendar(client, name="Tenant import target")
    account = client.post(
        "/api/v1/calendar/external/accounts",
        json={"provider": "caldav", "display_name": "Tenant Fastmail"},
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

    assert calendar["tenant_id"] == "tenant-a"
    assert account.json()["tenant_id"] == "tenant-a"
    assert binding.status_code == 201, binding.text
    assert binding.json()["calendar_id"] == calendar["id"]


def test_trigger_external_calendar_sync_queues_calendar_job(
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
    binding_id = binding.json()["id"]

    response = client.post(
        f"/api/v1/calendar/external/bindings/{binding_id}/sync",
        json={
            "reason": "manual",
            "window_start": "2026-06-01T00:00:00+00:00",
            "window_end": "2026-06-08T00:00:00+00:00",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["binding_id"] == binding_id
    assert body["queued"] is True
    assert body["status"] == "queued"
    assert body["job_id"] is not None
    jobs_manager = client.jobs_manager  # type: ignore[attr-defined]
    job = jobs_manager.get_job(body["job_id"])
    assert job["domain"] == "calendar"
    assert job["job_type"] == "calendar_sync"
    assert job["payload"] == {
        "binding_id": binding_id,
        "window_start": "2026-06-01T00:00:00+00:00",
        "window_end": "2026-06-08T00:00:00+00:00",
        "reason": "manual",
    }
    assert "password" not in json.dumps(job["payload"])
    assert "secret_ref" not in json.dumps(job["payload"])


def test_manual_sync_delegates_validated_input_to_typed_core_use_case(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP must delegate ownership, window policy and admission to Calendar core."""
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import CalendarSyncJobResponse

    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    calls: list[dict[str, Any]] = []

    def core(self: Any, **kwargs: Any) -> CalendarSyncJobResponse:
        calls.append(kwargs)
        return CalendarSyncJobResponse(item.external_binding_id, 42, True, "queued", "admission-test")

    def route_owned_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("HTTP performed a core-owned binding read before delegation")

    monkeypatch.setattr(CalendarService, "trigger_binding_sync", core, raising=False)
    monkeypatch.setattr(db, "get_external_binding", route_owned_read)
    response = client.post(f"/api/v1/calendar/external/bindings/{item.external_binding_id}/sync", json={
        "window_start": "2026-06-01T00:00:00Z", "window_end": "2026-06-08T00:00:00Z", "reason": "manual",
    })
    assert response.status_code == 200, response.text
    assert calls == [{"actor_user_id": 1, "binding_id": item.external_binding_id,
                      "window_start": "2026-06-01T00:00:00Z", "window_end": "2026-06-08T00:00:00Z", "reason": "manual"}]
    assert response.json()["job_id"] == 42


def test_manual_sync_ownership_and_admission_io_run_on_one_request_worker(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async manual trigger must not do ownership, Jobs, or audit I/O on its event loop."""
    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    manager = client.jobs_manager  # type: ignore[attr-defined]
    loop_threads: list[int] = []
    calls: list[tuple[str, int]] = []

    async def owner() -> User:
        loop_threads.append(threading.get_ident())
        return User(id=1, username="owner", is_active=True, is_admin=True, permissions=["*"])

    client.app.dependency_overrides[get_request_user] = owner
    for repository, names in [
        (db, ["get_external_binding", "get_external_account", "record_sync_event"]),
        (manager, ["list_jobs", "admit_idempotent_operation"]),
    ]:
        for name in names:
            original = getattr(repository, name)

            def trace(*args: Any, _operation: Any = original, _name: str = name, **kwargs: Any) -> Any:
                calls.append((_name, threading.get_ident()))
                return _operation(*args, **kwargs)

            monkeypatch.setattr(repository, name, trace)

    response = client.post(f"/api/v1/calendar/external/bindings/{item.external_binding_id}/sync")
    assert response.status_code == 200, response.text
    assert {name for name, _ in calls} == {
        "get_external_binding", "get_external_account", "list_jobs", "admit_idempotent_operation", "record_sync_event",
    }
    assert loop_threads and {thread for _, thread in calls}.isdisjoint(loop_threads)
    assert len({thread for _, thread in calls}) == 1
    assert manager.count_jobs(domain="calendar") == 1
    assert len(db.list_sync_events(binding_id=item.external_binding_id)) == 1


def test_manual_sync_maps_unavailable_job_authority_to_generic_domain_error(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable admission fails closed without exposing internal authority details."""
    from tldw_Server_API.app.core.Calendar.errors import CalendarSyncError

    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)

    def unavailable(self: Any, **kwargs: Any) -> Any:
        raise CalendarSyncError("sensitive internal Job correlation")

    monkeypatch.setattr(CalendarService, "trigger_binding_sync", unavailable)
    response = client.post(f"/api/v1/calendar/external/bindings/{item.external_binding_id}/sync")
    assert response.status_code == 500
    assert response.json() == {"detail": {"code": "calendar_error", "message": "Calendar request failed"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation_kind", ["native", "anyio"])
@pytest.mark.parametrize("fail_audit", [False, True], ids=["audit-success", "audit-failure"])
async def test_manual_sync_cancellation_drains_queue_audit_and_preserves_failure(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, cancellation_kind: str, fail_audit: bool,
) -> None:
    """Cancelling a real HTTP sync request waits for its queue audit and retains an audit failure."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker

    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    manager = client.jobs_manager  # type: ignore[attr-defined]
    loop = asyncio.get_running_loop()
    audit_started = asyncio.Event()
    audit_finished = threading.Event()
    release = threading.Event()
    failure = RuntimeError("manual sync audit failed")
    original_audit = db.record_sync_event
    scopes: list[CancelScope] = []
    cancellations: list[asyncio.CancelledError] = []
    audit_threads: list[int] = []
    drain_attempts = 0
    original_shield = worker.asyncio.shield

    def count_drain(awaitable: Any) -> Any:
        nonlocal drain_attempts
        if audit_started.is_set() and not audit_finished.is_set():
            drain_attempts += 1
        return original_shield(awaitable)

    def blocked_audit(*args: Any, **kwargs: Any) -> Any:
        """Hold the actual audit boundary before any runnable Jobs insert."""
        audit_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(audit_started.set)
        try:
            if not release.wait(3):
                raise TimeoutError("Manual sync audit was not released")
            if fail_audit:
                raise failure
            return original_audit(*args, **kwargs)
        finally:
            audit_finished.set()

    monkeypatch.setattr(db, "record_sync_event", blocked_audit)
    monkeypatch.setattr(worker.asyncio, "shield", count_drain)
    async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as api:
        async def post_sync() -> Any:
            try:
                return await api.post(f"/api/v1/calendar/external/bindings/{item.external_binding_id}/sync")
            except asyncio.CancelledError as exc:
                cancellations.append(exc)
                raise

        async def request() -> Any:
            if cancellation_kind == "anyio":
                with CancelScope() as scope:
                    scopes.append(scope)
                    return await post_sync()
                return None
            return await post_sync()

        # The baseline handler blocks the loop. A watchdog bounds that failure so
        # the test can report it and close every task/thread rather than deadlock.
        watchdog = threading.Timer(1, release.set)
        watchdog.start()
        task = asyncio.create_task(request())
        exited_while_blocked: list[bool] = []
        try:
            await asyncio.wait_for(audit_started.wait(), timeout=3)
            blocked_at_cancel = not audit_finished.is_set()
            for _ in range(3 if cancellation_kind == "native" else 1):
                if cancellation_kind == "native":
                    task.cancel("manual sync shutdown")
                else:
                    scopes[0].cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                exited_while_blocked.append(task.done())
        finally:
            release.set()
            outcomes = await asyncio.gather(task, return_exceptions=True)
            watchdog.cancel()
            watchdog.join()

    assert blocked_at_cancel, "Manual queue/audit I/O blocked cancellation delivery on the request event loop"
    assert not any(exited_while_blocked), "Cancelled request abandoned its queue audit"
    assert audit_finished.is_set()
    assert len(audit_threads) == 1 and audit_threads[0] != threading.get_ident()
    assert drain_attempts <= (4 if cancellation_kind == "native" else 3), "Cancellation caused hot drain retries"
    assert len(cancellations) == 1
    if cancellation_kind == "native":
        assert isinstance(outcomes[0], asyncio.CancelledError)
        assert str(cancellations[0]) == "manual sync shutdown"
    else:
        assert scopes[0].cancelled_caught
    assert cancellations[0].__cause__ is (failure if fail_audit else None)
    assert manager.count_jobs(domain="calendar") == (0 if fail_audit else 1)
    audits = db.list_sync_events(binding_id=item.external_binding_id)
    assert len(audits) == (0 if fail_audit else 1)
    assert db._transaction_connection.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation_kind", ["native", "anyio"])
async def test_manual_sync_waiting_for_scheduler_admission_is_offloop_and_drained(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
    monkeypatch: pytest.MonkeyPatch, cancellation_kind: str,
) -> None:
    """An HTTP trigger waiting on a durable writer remains off-loop and drains on cancellation."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker
    from tldw_Server_API.app.services.calendar_sync_scheduler import queue_due_calendar_sync_jobs

    client, db, _reminders = calendar_api_client
    _calendar, item = _create_provider_item(db)
    manager = client.jobs_manager  # type: ignore[attr-defined]
    scheduler_db = CalendarDatabase(db_path=db.db_path)
    scheduler_jobs = JobManager(manager.db_path)
    loop = asyncio.get_running_loop()
    empty_lookup = asyncio.Event()
    manual_started = asyncio.Event()
    release = threading.Event()
    scopes: list[CancelScope] = []
    original_audit = scheduler_db.record_sync_event
    original_queue = worker.queue_calendar_binding_sync

    def pause_scheduler(**kwargs: Any) -> Any:
        loop.call_soon_threadsafe(empty_lookup.set)
        if not release.wait(3):
            raise TimeoutError("Scheduler admission was not released")
        return original_audit(**kwargs)

    def trace_manual(**kwargs: Any) -> Any:
        if kwargs["job_manager"] is manager:
            loop.call_soon_threadsafe(manual_started.set)
        return original_queue(**kwargs)

    monkeypatch.setattr(scheduler_db, "record_sync_event", pause_scheduler)
    monkeypatch.setattr(worker, "queue_calendar_binding_sync", trace_manual)
    scheduled = asyncio.create_task(queue_due_calendar_sync_jobs(db=scheduler_db, job_manager=scheduler_jobs))
    watchdog = threading.Timer(1, release.set)
    task: asyncio.Task[Any] | None = None
    async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as api:
        async def request() -> Any:
            if cancellation_kind == "anyio":
                with CancelScope() as scope:
                    scopes.append(scope)
                    return await api.post(f"/api/v1/calendar/external/bindings/{item.external_binding_id}/sync")
                return None
            return await api.post(f"/api/v1/calendar/external/bindings/{item.external_binding_id}/sync")

        try:
            await asyncio.wait_for(empty_lookup.wait(), timeout=3)
            watchdog.start()
            task = asyncio.create_task(request())
            await asyncio.wait_for(manual_started.wait(), timeout=3)
            scheduler_blocked = not release.is_set()
            if cancellation_kind == "native":
                task.cancel("manual lock wait shutdown")
            else:
                scopes[0].cancel()
            for _ in range(5):
                await asyncio.sleep(0)
            abandoned = task.done()
        finally:
            release.set()
            outcomes = await asyncio.gather(scheduled, *([task] if task is not None else []), return_exceptions=True)
            watchdog.cancel()
            if watchdog.ident is not None:
                watchdog.join()

    assert scheduler_blocked, "Waiting for admission blocked the HTTP event loop"
    assert not abandoned, "Cancellation abandoned a manual admission waiting on the durable transaction"
    assert len(outcomes[0]) == 1
    if cancellation_kind == "native":
        assert isinstance(outcomes[1], asyncio.CancelledError)
    else:
        assert scopes[0].cancelled_caught
    jobs = manager.list_jobs(domain="calendar")
    audits = db.list_sync_events(binding_id=item.external_binding_id)
    assert len(jobs) == len(audits) == 1
    assert outcomes[0][0].job_id == jobs[0]["id"]
    assert jobs[0]["payload"]["reason"] == "scheduled"
    assert json.loads(audits[0].metadata_json) == {
        "admission_id": db.get_sync_admission(item.external_binding_id).admission_id,
        "job_id": jobs[0]["id"], "reason": "scheduled",
    }
    assert db._transaction_connection.get() is None


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


def test_external_binding_management_and_sync_events_are_owner_scoped(
    calendar_api_client: tuple[TestClient, CalendarDatabase, _ReminderServiceStub],
) -> None:
    client, db, _reminder_service = calendar_api_client
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
    binding_id = binding.json()["id"]
    db.record_sync_event(
        binding_id=binding_id,
        event_type="scan",
        status="success",
        items_seen=2,
        items_upserted=1,
    )

    updated = client.patch(
        f"/api/v1/calendar/external/bindings/{binding_id}",
        json={"sync_interval_minutes": 30, "lookahead_days": 120},
    )
    disabled = client.post(f"/api/v1/calendar/external/bindings/{binding_id}/disable")
    enabled = client.post(f"/api/v1/calendar/external/bindings/{binding_id}/enable")
    status_response = client.get(f"/api/v1/calendar/external/bindings/{binding_id}/sync-status")
    events = client.get(f"/api/v1/calendar/external/bindings/{binding_id}/sync-events")

    assert updated.status_code == 200, updated.text
    assert updated.json()["sync_interval_minutes"] == 30
    assert updated.json()["lookahead_days"] == 120
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["disabled_at"] is not None
    assert enabled.status_code == 200, enabled.text
    assert enabled.json()["sync_enabled"] is True
    assert enabled.json()["disabled_at"] is None
    assert status_response.status_code == 200, status_response.text
    assert status_response.json()["id"] == binding_id
    assert events.status_code == 200, events.text
    assert events.json()["items"][0]["event_type"] == "scan"
    assert events.json()["items"][0]["metadata"] is None

    _set_user(client, 2)
    forbidden = client.get(f"/api/v1/calendar/external/bindings/{binding_id}/sync-status")
    assert forbidden.status_code == 403
    assert forbidden.json()["detail"]["code"] == "calendar_permission_denied"

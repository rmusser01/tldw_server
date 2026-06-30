from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints.notifications import router as notifications_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.permissions import (
    NOTIFICATIONS_CONTROL,
    NOTIFICATIONS_READ,
    TASKS_CONTROL,
    TASKS_READ,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.config import settings


pytestmark = pytest.mark.unit


class _TimeoutSSEStream:
    def __init__(self, **_kwargs):
        self._closed = False

    async def send_event(self, _event: str, _payload: dict, *, event_id: str | None = None) -> None:
        raise asyncio.TimeoutError("notifications timeout leaked /private/notifications-stream.log")

    async def done(self, *, force: bool = False) -> None:
        self._closed = True

    async def iter_sse(self):
        while not self._closed:
            await asyncio.sleep(0)
            yield "data: {}\n\n"


class _LoopErrorSSEStream:
    def __init__(self, **_kwargs):
        self._closed = False

    async def send_event(self, _event: str, _payload: dict, *, event_id: str | None = None) -> None:
        return None

    async def done(self, *, force: bool = False) -> None:
        self._closed = True

    async def iter_sse(self):
        while not self._closed:
            await asyncio.sleep(0)
            yield "data: {}\n\n"


async def _advance_stream(response, *, until=None, steps: int = 10) -> None:
    iterator = response.body_iterator
    try:
        for _ in range(steps):
            with contextlib.suppress(StopAsyncIteration):
                await anext(iterator)
            await asyncio.sleep(0)
            if until is not None and until():
                break
    finally:
        with contextlib.suppress(BaseException):
            await iterator.aclose()


def _collect_sse_events(response, *, max_events: int = 8) -> list[dict]:
    events: list[dict] = []
    current: dict = {}
    for raw_line in response.iter_lines():
        line = raw_line.decode() if isinstance(raw_line, (bytes, bytearray)) else str(raw_line)
        if line == "":
            if not current:
                continue
            data_lines = current.pop("data_lines", [])
            if data_lines:
                data_raw = "\n".join(data_lines)
                try:
                    current["data"] = json.loads(data_raw)
                except json.JSONDecodeError:
                    current["data_raw"] = data_raw
            events.append(current)
            current = {}
            if len(events) >= max_events:
                break
            continue
        if line.startswith("id: "):
            current["id"] = line[4:]
            continue
        if line.startswith("event: "):
            current["event"] = line[7:]
            continue
        if line.startswith("data: "):
            current.setdefault("data_lines", []).append(line[6:])
    return events


@pytest.mark.asyncio
async def test_notifications_stream_sanitizes_send_timeout_log(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import notifications as notifications_endpoint

    fake_logger = MagicMock()
    monkeypatch.setattr(notifications_endpoint, "logger", fake_logger)
    monkeypatch.setattr(notifications_endpoint, "SSEStream", _TimeoutSSEStream)

    response = await notifications_endpoint.stream_notifications(
        after=0,
        last_event_id=None,
        db=SimpleNamespace(user_id=882),
        _principal=object(),
    )

    await _advance_stream(response, until=lambda: fake_logger.warning.called)

    fake_logger.warning.assert_called_once_with("notifications stream send timeout")


@pytest.mark.asyncio
async def test_notifications_stream_sanitizes_loop_error_log(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import notifications as notifications_endpoint

    fake_logger = MagicMock()
    monkeypatch.setattr(notifications_endpoint, "logger", fake_logger)
    monkeypatch.setattr(notifications_endpoint, "SSEStream", _LoopErrorSSEStream)
    monkeypatch.setattr(
        notifications_endpoint.CollectionsDatabase,
        "for_user",
        staticmethod(lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("notifications db leaked /private/notifications.db")
        )),
    )

    response = await notifications_endpoint.stream_notifications(
        after=0,
        last_event_id=None,
        db=SimpleNamespace(user_id=882),
        _principal=object(),
    )

    await _advance_stream(response, until=lambda: fake_logger.warning.called)

    fake_logger.warning.assert_called_once_with("notifications stream loop error")


@pytest.fixture()
def notifications_sse_app(monkeypatch, tmp_path):
    base_dir = tmp_path / "test_notifications_sse"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    monkeypatch.setenv("NOTIFICATIONS_STREAM_HEARTBEAT_SEC", "0.01")
    monkeypatch.setenv("NOTIFICATIONS_STREAM_POLL_SEC", "0.01")
    monkeypatch.setenv("NOTIFICATIONS_STREAM_MAX_DURATION_SEC", "0.3")

    app = FastAPI()
    app.include_router(notifications_router, prefix="/api/v1")

    async def override_user():
        return User(id=882, username="notifications-sse-user", email=None, is_active=True)

    async def override_principal():
        return AuthPrincipal(
            kind="user",
            user_id=882,
            roles=[],
            permissions=[TASKS_READ, TASKS_CONTROL, NOTIFICATIONS_READ, NOTIFICATIONS_CONTROL],
            is_admin=False,
        )

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    try:
        yield app
    finally:
        app.dependency_overrides.clear()
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_notifications_stream_emits_required_notification_fields(notifications_sse_app):
    cdb = CollectionsDatabase.for_user(user_id=882)
    row = cdb.create_user_notification(
        kind="reminder_due",
        title="Reminder",
        message="Review item",
        severity="info",
    )

    with TestClient(notifications_sse_app) as client:
        with client.stream("GET", "/api/v1/notifications/stream?after=0") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "").lower()
            events = _collect_sse_events(response, max_events=8)

    notification_event = next((e for e in events if e.get("event") == "notification"), None)
    assert notification_event is not None
    assert notification_event.get("id") == str(row.id)

    payload = notification_event.get("data")
    assert isinstance(payload, dict)
    assert payload["event_id"] == row.id
    assert payload["notification_id"] == row.id
    assert payload["kind"] == row.kind
    assert payload.get("created_at")


def test_notifications_stream_prefers_last_event_id_header(notifications_sse_app):
    cdb = CollectionsDatabase.for_user(user_id=882)
    first = cdb.create_user_notification(
        kind="reminder_due",
        title="First",
        message="first",
        severity="info",
    )
    second = cdb.create_user_notification(
        kind="job_completed",
        title="Second",
        message="second",
        severity="info",
    )

    with TestClient(notifications_sse_app) as client:
        with client.stream(
            "GET",
            "/api/v1/notifications/stream?after=0",
            headers={"Last-Event-ID": str(first.id)},
        ) as response:
            events = _collect_sse_events(response, max_events=8)

    notification_event = next((e for e in events if e.get("event") == "notification"), None)
    assert notification_event is not None
    assert notification_event.get("id") == str(second.id)


def test_notifications_stream_emits_reset_required_for_stale_cursor(notifications_sse_app, monkeypatch):
    monkeypatch.setenv("NOTIFICATIONS_STREAM_REPLAY_WINDOW", "2")
    cdb = CollectionsDatabase.for_user(user_id=882)
    rows = [
        cdb.create_user_notification(
            kind="reminder_due",
            title=f"Reminder {idx}",
            message=f"message {idx}",
            severity="info",
        )
        for idx in range(5)
    ]

    with TestClient(notifications_sse_app) as client:
        with client.stream(
            "GET",
            "/api/v1/notifications/stream",
            headers={"Last-Event-ID": str(rows[0].id)},
        ) as response:
            events = _collect_sse_events(response, max_events=8)

    reset_event = next((e for e in events if e.get("event") == "reset_required"), None)
    assert reset_event is not None
    payload = reset_event.get("data")
    assert isinstance(payload, dict)
    assert payload["latest_event_id"] >= rows[-1].id
    assert payload["min_event_id"] == rows[-2].id - 1


def test_notifications_stream_rejects_invalid_last_event_id(notifications_sse_app):
    with TestClient(notifications_sse_app) as client:
        response = client.get(
            "/api/v1/notifications/stream",
            headers={"Last-Event-ID": "not-an-integer"},
        )

    assert response.status_code == 400
    assert response.json().get("detail") == "invalid_last_event_id"

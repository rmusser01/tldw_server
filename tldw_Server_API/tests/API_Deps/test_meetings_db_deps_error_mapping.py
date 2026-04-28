from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, URL

from tldw_Server_API.app.api.v1.API_Deps import Meetings_DB_Deps as meetings_deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Meetings_DB import (
    MeetingsDatabaseError,
    SchemaError,
)


_PRIVATE_PATH = "/Users/private-user/.tldw/Databases/user_databases/42/Meetings.db"
_SECRET_DSN = "postgresql://meeting_user:secret-token@db.internal/meetings"
_RAW_DETAIL = f"backend exploded at {_PRIVATE_PATH} using {_SECRET_DSN}"


def _user() -> User:
    return User(id=42, username="meetings-user")


def _capture_meetings_dep_errors() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = meetings_deps.logger.add(
        lambda message: messages.append(str(message.record["message"])),
        level="ERROR",
    )
    return messages, sink_id


def _assert_log_messages_are_sanitized(messages: list[str]) -> None:
    rendered = "\n".join(messages)
    assert _PRIVATE_PATH not in rendered
    assert _SECRET_DSN not in rendered
    assert _RAW_DETAIL not in rendered


def _patch_meetings_factory_failure(monkeypatch, exc: Exception) -> None:
    class FailingMeetingsDatabase:
        @staticmethod
        def for_user(user_id: int):
            raise exc

    monkeypatch.setattr(meetings_deps, "MeetingsDatabase", FailingMeetingsDatabase)


def _websocket() -> SimpleNamespace:
    return SimpleNamespace(
        headers=Headers({}),
        scope={"query_string": b""},
        url=URL("ws://testserver/api/v1/meetings/ws"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected_detail"),
    [
        (MeetingsDatabaseError(_RAW_DETAIL), "Meetings DB unavailable"),
        (SchemaError(_RAW_DETAIL), "Database schema error"),
        (RuntimeError(_RAW_DETAIL), "Meetings DB unavailable"),
    ],
)
async def test_get_meetings_db_for_user_sanitizes_initialization_failure_logs(
    monkeypatch,
    exc: Exception,
    expected_detail: str,
):
    _patch_meetings_factory_failure(monkeypatch, exc)
    messages, sink_id = _capture_meetings_dep_errors()
    try:
        with pytest.raises(HTTPException) as exc_info:
            await meetings_deps.get_meetings_db_for_user(current_user=_user())
    finally:
        meetings_deps.logger.remove(sink_id)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    _assert_log_messages_are_sanitized(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected_detail"),
    [
        (MeetingsDatabaseError(_RAW_DETAIL), "Meetings DB unavailable"),
        (SchemaError(_RAW_DETAIL), "Database schema error"),
        (RuntimeError(_RAW_DETAIL), "Meetings DB unavailable"),
    ],
)
async def test_get_meetings_db_for_websocket_sanitizes_initialization_failure_logs(
    monkeypatch,
    exc: Exception,
    expected_detail: str,
):
    _patch_meetings_factory_failure(monkeypatch, exc)

    async def fake_get_request_user(*args, **kwargs):
        return _user()

    monkeypatch.setattr(meetings_deps, "get_request_user", fake_get_request_user)
    messages, sink_id = _capture_meetings_dep_errors()
    try:
        with pytest.raises(HTTPException) as exc_info:
            await meetings_deps.get_meetings_db_for_websocket(websocket=_websocket())
    finally:
        meetings_deps.logger.remove(sink_id)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    _assert_log_messages_are_sanitized(messages)

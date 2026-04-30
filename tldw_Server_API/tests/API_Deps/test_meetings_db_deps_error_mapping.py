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
_SECRET_TOKEN = "meeting-token-secret-123"
_HEADER_RAW_DETAIL = f"header parser leaked {_PRIVATE_PATH} with token {_SECRET_TOKEN}"


def _user() -> User:
    return User(id=42, username="meetings-user")


def _capture_meetings_dep_errors() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = meetings_deps.logger.add(
        lambda message: messages.append(str(message.record["message"])),
        level="ERROR",
    )
    return messages, sink_id


def _capture_meetings_dep_debug() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = meetings_deps.logger.add(
        lambda message: messages.append(str(message.record["message"])),
        level="DEBUG",
    )
    return messages, sink_id


def _assert_log_messages_are_sanitized(messages: list[str]) -> None:
    rendered = "\n".join(messages)
    assert _PRIVATE_PATH not in rendered
    assert _SECRET_DSN not in rendered
    assert _RAW_DETAIL not in rendered
    assert _SECRET_TOKEN not in rendered
    assert _HEADER_RAW_DETAIL not in rendered


class _RaisingHeaders:
    def __init__(self, header_to_raise: str) -> None:
        self.header_to_raise = header_to_raise

    def get(self, key: str) -> str | None:
        if key == self.header_to_raise:
            raise RuntimeError(_HEADER_RAW_DETAIL)
        return None


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


@pytest.mark.parametrize(
    ("header_to_raise", "input_token", "input_api_key"),
    [
        ("authorization", "query-token", "query-api-key"),
        ("x-api-key", "query-token", "query-api-key"),
        ("sec-websocket-protocol", "query-token", "query-api-key"),
    ],
)
def test_extract_websocket_credentials_sanitizes_header_parse_fallback_logs(
    header_to_raise: str,
    input_token: str,
    input_api_key: str,
):
    websocket = SimpleNamespace(headers=_RaisingHeaders(header_to_raise))
    messages, sink_id = _capture_meetings_dep_debug()
    try:
        resolved_token, resolved_api_key = meetings_deps._extract_websocket_credentials(
            websocket=websocket,
            token=input_token,
            api_key=input_api_key,
        )
    finally:
        meetings_deps.logger.remove(sink_id)

    assert resolved_token == input_token
    assert resolved_api_key == input_api_key
    assert messages
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

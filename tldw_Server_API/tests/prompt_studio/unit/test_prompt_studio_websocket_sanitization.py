from unittest.mock import AsyncMock, MagicMock

import pytest

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_websocket as ws_mod
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.event_broadcaster import (
    EventBroadcaster,
    EventType,
)


pytestmark = pytest.mark.unit


class _FailingWebSocket:
    async def send_text(self, _message: str) -> None:
        raise RuntimeError("websocket send leaked /private/prompt-studio-ws.json")


class _ClosingWebSocket:
    def __init__(self) -> None:
        self.closed_with: int | None = None
        self.headers: dict[str, str] = {}
        self.state = type("State", (), {"auth_principal": None})()

    async def close(self, code: int) -> None:
        self.closed_with = code


def test_sanitize_error_message_logs_safe_error_label(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(ws_mod, "logger", fake_logger)

    message = ws_mod.sanitize_error_message(
        RuntimeError("prompt studio error leaked /private/prompt-studio.json"),
        "SSE streaming",
    )

    assert message == "Operation failed"
    fake_logger.error.assert_called_once_with("Error in {}: {}", "SSE streaming", "RuntimeError")


@pytest.mark.asyncio
async def test_prompt_studio_websocket_rejects_when_no_auth_path_succeeds(monkeypatch):
    websocket = _ClosingWebSocket()

    async def no_cookie(_websocket):
        return None

    async def no_explicit_auth(_websocket, **_kwargs):
        return None

    monkeypatch.setattr(ws_mod, "resolve_single_user_cookie_websocket", no_cookie)
    monkeypatch.setattr(ws_mod, "_authenticate_ws", no_explicit_auth, raising=False)
    monkeypatch.setattr(ws_mod, "cookie_websocket_rejection_code", lambda _websocket: None)

    user_id = await ws_mod._allow_prompt_studio_cookie_websocket(websocket)

    assert user_id is None
    assert websocket.closed_with == 4401


@pytest.mark.asyncio
async def test_prompt_studio_websocket_accepts_explicit_auth_identity(monkeypatch):
    websocket = _ClosingWebSocket()

    async def no_cookie(_websocket):
        return None

    async def explicit_auth(_websocket, **_kwargs):
        return 42

    monkeypatch.setattr(ws_mod, "resolve_single_user_cookie_websocket", no_cookie)
    monkeypatch.setattr(ws_mod, "_authenticate_ws", explicit_auth)
    monkeypatch.setattr(ws_mod, "cookie_websocket_rejection_code", lambda _websocket: None)

    user_id = await ws_mod._allow_prompt_studio_cookie_websocket(
        websocket,
        token="valid-token",
    )

    assert user_id == "42"
    assert websocket.closed_with is None


@pytest.mark.asyncio
async def test_prompt_studio_websocket_accepts_canonical_api_key_header(monkeypatch):
    websocket = _ClosingWebSocket()
    websocket.headers["x-api-key"] = "valid-api-key"

    async def no_cookie(_websocket):
        return None

    async def explicit_auth(_websocket, **kwargs):
        return 42 if kwargs.get("api_key") == "valid-api-key" else None

    monkeypatch.setattr(ws_mod, "resolve_single_user_cookie_websocket", no_cookie)
    monkeypatch.setattr(ws_mod, "_authenticate_ws", explicit_auth)
    monkeypatch.setattr(ws_mod, "cookie_websocket_rejection_code", lambda _websocket: None)

    user_id = await ws_mod._allow_prompt_studio_cookie_websocket(websocket)

    assert user_id == "42"
    assert websocket.closed_with is None


@pytest.mark.asyncio
async def test_project_websocket_resolves_database_after_authentication(monkeypatch):
    websocket = _ClosingWebSocket()
    fake_db = object()
    access_calls: list[tuple[int, dict, object]] = []

    async def allow_auth(_websocket, **_kwargs):
        return "42"

    async def get_db(user_context):
        return fake_db

    async def require_access(project_id, user_context, db):
        access_calls.append((project_id, user_context, db))
        return True

    async def stop_before_accept(_websocket, _operation):
        return False

    monkeypatch.setattr(ws_mod, "_allow_prompt_studio_cookie_websocket", allow_auth)
    monkeypatch.setattr(ws_mod, "get_prompt_studio_db", get_db)
    monkeypatch.setattr(ws_mod, "require_project_access", require_access)
    monkeypatch.setattr(ws_mod, "_guard_prompt_studio_websocket_start", stop_before_accept)

    await ws_mod.websocket_endpoint(websocket, 7, token="valid-token")

    assert access_calls == [
        (
            7,
            {
                "user_id": "42",
                "client_id": "websocket",
                "is_authenticated": True,
                "is_admin": False,
                "permissions": [],
            },
            fake_db,
        )
    ]


@pytest.mark.asyncio
async def test_base_websocket_authorizes_and_joins_requested_project_scope(monkeypatch):
    websocket = _ClosingWebSocket()
    connect = AsyncMock(return_value=False)
    authorize = AsyncMock(return_value=True)

    async def allow_auth(_websocket, **_kwargs):
        return "owner-1"

    monkeypatch.setattr(ws_mod, "_allow_prompt_studio_cookie_websocket", allow_auth)
    monkeypatch.setattr(ws_mod, "_authorize_prompt_studio_project", authorize)
    monkeypatch.setattr(ws_mod.connection_manager, "connect", connect)
    monkeypatch.setattr(ws_mod, "WebSocketStream", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        ws_mod,
        "_guard_prompt_studio_websocket_start",
        AsyncMock(return_value=True),
    )

    await ws_mod.websocket_endpoint_base(websocket, project_id=1)

    user_context = {
        "user_id": "owner-1",
        "client_id": "websocket",
        "is_authenticated": True,
        "is_admin": False,
        "permissions": [],
    }
    authorize.assert_awaited_once_with(websocket, user_context, 1)
    connect.assert_awaited_once_with(
        websocket,
        ws_mod.prompt_studio_connection_scope("owner-1", 1),
        user_context,
    )


@pytest.mark.asyncio
async def test_base_websocket_keeps_projectless_subscribe_in_owner_global_scope(monkeypatch):
    websocket = _ClosingWebSocket()
    websocket.receive_json = AsyncMock(
        side_effect=[{"type": "subscribe"}, ws_mod.WebSocketDisconnect()]
    )
    connect = AsyncMock(return_value=True)
    authorize = AsyncMock(return_value=True)
    rebind = MagicMock()
    disconnect = MagicMock()
    sent: list[dict] = []

    class _Stream:
        ws = websocket

        async def start(self):
            return None

        async def send_json(self, payload):
            sent.append(payload)

        def mark_activity(self):
            return None

    async def allow_auth(_websocket, **_kwargs):
        return "owner-1"

    monkeypatch.setattr(ws_mod, "_allow_prompt_studio_cookie_websocket", allow_auth)
    monkeypatch.setattr(ws_mod, "_authorize_prompt_studio_project", authorize)
    monkeypatch.setattr(ws_mod.connection_manager, "connect", connect)
    monkeypatch.setattr(ws_mod.connection_manager, "rebind", rebind)
    monkeypatch.setattr(ws_mod.connection_manager, "disconnect", disconnect)
    monkeypatch.setattr(ws_mod, "WebSocketStream", lambda *_args, **_kwargs: _Stream())
    monkeypatch.setattr(
        ws_mod,
        "_guard_prompt_studio_websocket_start",
        AsyncMock(return_value=True),
    )

    await ws_mod.websocket_endpoint_base(websocket, project_id=None)

    global_scope = ws_mod.prompt_studio_connection_scope("owner-1")
    authorize.assert_not_awaited()
    connect.assert_awaited_once()
    assert connect.await_args.args[1] == global_scope
    rebind.assert_called_once_with(
        websocket,
        global_scope,
        connect.await_args.args[2],
    )
    assert sent == [{"type": "subscribed", "project_id": None}]
    disconnect.assert_called_once_with(websocket)


@pytest.mark.asyncio
async def test_prompt_studio_events_are_isolated_by_user_and_project():
    manager = ws_mod.ConnectionManager()
    owner_socket = MagicMock()
    owner_socket.send_text = AsyncMock()
    other_socket = MagicMock()
    other_socket.send_text = AsyncMock()
    owner_scope = ws_mod.prompt_studio_connection_scope("owner-1", 1)
    other_scope = ws_mod.prompt_studio_connection_scope("owner-2", 1)
    manager.active_connections = {
        owner_scope: {owner_socket},
        other_scope: {other_socket},
    }

    db = MagicMock()
    db.user_id = "owner-1"
    db.client_id = "test"
    broadcaster = EventBroadcaster(manager, db)
    broadcaster._log_event = AsyncMock()

    await broadcaster.broadcast_event(
        EventType.JOB_PROGRESS,
        {"job_id": 7, "progress": 50},
        project_id=1,
    )

    owner_socket.send_text.assert_awaited_once()
    other_socket.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_personal_message_sanitizes_send_failure_log(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(ws_mod, "logger", fake_logger)
    manager = ws_mod.ConnectionManager()

    await manager.send_personal_message("{}", _FailingWebSocket())

    fake_logger.error.assert_called_once_with("Failed to send message to WebSocket")


@pytest.mark.asyncio
async def test_broadcast_to_client_sanitizes_send_failure_log(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(ws_mod, "logger", fake_logger)
    manager = ws_mod.ConnectionManager()
    websocket = _FailingWebSocket()
    manager.active_connections["client"] = {websocket}
    manager.connection_metadata[websocket] = {"client_id": "client"}

    await manager.broadcast_to_client("client", "{}")

    fake_logger.error.assert_called_once_with("Failed to send to WebSocket")


@pytest.mark.asyncio
async def test_legacy_sse_stream_sanitizes_error_log(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(ws_mod, "logger", fake_logger)
    monkeypatch.setattr(ws_mod, "env_flag_enabled", lambda _name: False)

    async def _failing_sleep(_seconds: int) -> None:
        raise RuntimeError("legacy SSE loop leaked /private/prompt-studio-sse.json")

    monkeypatch.setattr(ws_mod.asyncio, "sleep", _failing_sleep)

    response = await ws_mod.sse_endpoint(client_id="client", project_id=None, db=None, user_context={})
    iterator = response.body_iterator
    initial_frame = await anext(iterator)
    error_frame = await anext(iterator)

    assert "connection" in initial_frame
    assert "Operation failed" in error_frame
    fake_logger.error.assert_any_call("SSE error")

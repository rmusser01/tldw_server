from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_websocket as ws_mod


pytestmark = pytest.mark.unit


class _FailingWebSocket:
    async def send_text(self, _message: str) -> None:
        raise RuntimeError("websocket send leaked /private/prompt-studio-ws.json")


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

"""WebSocket smoke test for MCP Unified (basic initialize/ping flow)."""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.MCP_unified import get_mcp_server
from tldw_Server_API.app.main import app


def _receive_jsonrpc(ws):
    msg = ws.receive_json()
    while isinstance(msg, dict) and msg.get("type") == "ping":
        msg = ws.receive_json()
    return msg


@pytest.fixture
def ws_client(monkeypatch):
    monkeypatch.setenv("MCP_WS_AUTH_REQUIRED", "false")
    monkeypatch.setenv("MCP_ALLOWED_IPS", "")
    client = TestClient(app)
    server = get_mcp_server()
    server.config.ws_auth_required = False
    server.config.allowed_client_ips = []
    server.config.blocked_client_ips = []
    try:
        yield client
    finally:
        client.close()


@pytest.mark.asyncio
async def test_ws_initialize_and_ping(ws_client):
    with ws_client.websocket_connect("/api/v1/mcp/ws?client_id=smoke") as ws:
        # initialize
        ws.send_json(
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {"clientInfo": {"name": "WS Smoke"}},
                "id": 1,
            }
        )
        msg = _receive_jsonrpc(ws)
        assert msg.get("jsonrpc") == "2.0"
        assert msg.get("id") == 1
        assert "error" not in msg
        result = msg.get("result") or {}
        assert result.get("protocolVersion") == "2024-11-05"

        # ping
        ws.send_json(
            {
                "jsonrpc": "2.0",
                "method": "ping",
                "id": 2,
            }
        )
        msg2 = _receive_jsonrpc(ws)
        assert msg2.get("jsonrpc") == "2.0"
        assert msg2.get("id") == 2
        assert "error" not in msg2
        result2 = msg2.get("result") or {}
        assert result2.get("pong") is True


def test_ws_initialized_notification_sends_no_frame_and_ping_still_works(ws_client):
    with ws_client.websocket_connect("/api/v1/mcp/ws?client_id=initialized-notification") as ws:
        ws.send_json({"jsonrpc": "2.0", "method": "notifications/initialized"})
        ws.send_json({"jsonrpc": "2.0", "method": "ping", "id": "after-notification"})

        msg = _receive_jsonrpc(ws)
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == "after-notification"
        assert "error" not in msg
        assert msg["result"]["pong"] is True


def test_ws_explicit_null_id_returns_null_id_response(ws_client):
    with ws_client.websocket_connect("/api/v1/mcp/ws?client_id=null-id") as ws:
        ws.send_json({"jsonrpc": "2.0", "method": "ping", "id": None})

        msg = _receive_jsonrpc(ws)
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] is None
        assert "error" not in msg
        assert msg["result"]["pong"] is True


def test_ws_client_string_id_that_looks_like_null_sentinel_is_preserved(ws_client):
    sentinel_like_id = "__tldw_ws_jsonrpc_explicit_null_id__"
    with ws_client.websocket_connect("/api/v1/mcp/ws?client_id=sentinel-id") as ws:
        ws.send_json({"jsonrpc": "2.0", "method": "ping", "id": sentinel_like_id})

        msg = _receive_jsonrpc(ws)
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == sentinel_like_id
        assert "error" not in msg
        assert msg["result"]["pong"] is True


def test_ws_jsonrpc_keepalive_frames_do_not_satisfy_followup_request(ws_client):
    with ws_client.websocket_connect("/api/v1/mcp/ws?client_id=keepalive") as ws:
        ws.send_json({"type": "ping"})
        ws.send_json({"type": "pong"})
        ws.send_json({"jsonrpc": "2.0", "method": "ping", "id": "real-ping"})

        msg = _receive_jsonrpc(ws)
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] == "real-ping"
        assert "error" not in msg
        assert msg["result"]["pong"] is True


def test_ws_keepalive_with_id_returns_invalid_request_null_id(ws_client):
    with ws_client.websocket_connect("/api/v1/mcp/ws?client_id=bad-keepalive") as ws:
        ws.send_json({"type": "ping", "id": "x"})

        msg = _receive_jsonrpc(ws)
        assert msg["jsonrpc"] == "2.0"
        assert msg["id"] is None
        assert msg["error"]["code"] == -32600
        assert "result" not in msg

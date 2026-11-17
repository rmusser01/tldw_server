import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.app.core.MCP_unified.server import MCPServer
from tldw_Server_API.app.core.Utils.app_services import app_services
from tldw_Server_API.app.main import app

client = TestClient(app)


class _CapturingWsProtocol:
    def __init__(self):
        self.contexts: list[dict] = []

    async def process_request(self, payload, context):
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPResponse

        self.contexts.append(
            {
                "method": payload.get("method") if isinstance(payload, dict) else None,
                "metadata": dict(context.metadata),
                "session_id": context.session_id,
            }
        )
        return MCPResponse(result={"ok": True}, id=getattr(payload, "id", None))


def _build_ws_test_server() -> MCPServer:
    server = MCPServer()
    server.initialized = True
    server.protocol = _CapturingWsProtocol()
    server.config.ws_auth_required = False
    server.config.ws_allow_query_auth = True
    return server


def test_mcp_ws_attaches_workspace_query_params_to_context(monkeypatch):
    server = _build_ws_test_server()
    protocol = server.protocol
    monkeypatch.setattr(app_services(), "mcp_server", server)

    with client.websocket_connect(
        "/api/v1/mcp/ws?client_id=workspace-probe"
        "&mcp_session_id=ws-session-1"
        "&workspace_id=workspace-direct"
        "&cwd=src/app"
    ) as ws:
        ws.send_text(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"clientInfo": {"name": "probe", "version": "0.0.1"}},
                }
            )
        )
        assert ws.receive_json()["result"] == {"ok": True}

    assert protocol.contexts[-1]["metadata"]["workspace_id"] == "workspace-direct"
    assert protocol.contexts[-1]["metadata"]["cwd"] == "src/app"
    assert protocol.contexts[-1]["session_id"] == "ws-session-1"


def test_mcp_ws_initialize_cannot_override_connection_workspace_context(monkeypatch):
    server = _build_ws_test_server()
    protocol = server.protocol
    monkeypatch.setattr(app_services(), "mcp_server", server)

    with client.websocket_connect(
        "/api/v1/mcp/ws?client_id=workspace-probe"
        "&mcp_session_id=ws-session-2"
        "&workspace_id=workspace-direct"
        "&cwd=src/app"
    ) as ws:
        ws.send_text(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "clientInfo": {"name": "probe", "version": "0.0.1"},
                        "workspace_id": "override-workspace",
                        "cwd": "override/cwd",
                    },
                }
            )
        )
        assert ws.receive_json()["result"] == {"ok": True}

    assert protocol.contexts[-1]["metadata"]["workspace_id"] == "workspace-direct"
    assert protocol.contexts[-1]["metadata"]["cwd"] == "src/app"


def test_mcp_ws_reconnect_with_same_context_succeeds(monkeypatch):
    server = _build_ws_test_server()
    protocol = server.protocol
    monkeypatch.setattr(app_services(), "mcp_server", server)

    ws_url = (
        "/api/v1/mcp/ws?client_id=workspace-probe"
        "&mcp_session_id=ws-session-3"
        "&workspace_id=workspace-direct"
        "&cwd=src/app"
    )
    with client.websocket_connect(ws_url) as ws:
        ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
        assert ws.receive_json()["result"] == {"ok": True}

    with client.websocket_connect(ws_url) as ws:
        ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "status", "params": {}}))
        assert ws.receive_json()["result"] == {"ok": True}

    assert protocol.contexts[-1]["session_id"] == "ws-session-3"


def test_mcp_ws_rejects_workspace_id_mismatch_for_same_session(monkeypatch):
    server = _build_ws_test_server()
    monkeypatch.setattr(app_services(), "mcp_server", server)

    with client.websocket_connect(
        "/api/v1/mcp/ws?client_id=workspace-probe"
        "&mcp_session_id=ws-session-4"
        "&workspace_id=workspace-one"
        "&cwd=src/app"
    ) as ws:
        ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
        assert ws.receive_json()["result"] == {"ok": True}

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(
            "/api/v1/mcp/ws?client_id=workspace-probe"
            "&mcp_session_id=ws-session-4"
            "&workspace_id=workspace-two"
            "&cwd=src/app"
        ):
            pass

    assert getattr(exc_info.value, "code", None) == 1008


def test_mcp_ws_rejects_cwd_mismatch_for_same_session(monkeypatch):
    server = _build_ws_test_server()
    monkeypatch.setattr(app_services(), "mcp_server", server)

    with client.websocket_connect(
        "/api/v1/mcp/ws?client_id=workspace-probe"
        "&mcp_session_id=ws-session-5"
        "&workspace_id=workspace-one"
        "&cwd=src/app"
    ) as ws:
        ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}))
        assert ws.receive_json()["result"] == {"ok": True}

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(
            "/api/v1/mcp/ws?client_id=workspace-probe"
            "&mcp_session_id=ws-session-5"
            "&workspace_id=workspace-one"
            "&cwd=src/other"
        ):
            pass

    assert getattr(exc_info.value, "code", None) == 1008

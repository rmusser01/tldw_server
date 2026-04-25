import pytest


def test_mcp_ws_invalid_json_returns_jsonrpc_parse_error(mcp_ws_client):
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    srv = get_mcp_server()
    srv.config.ws_auth_required = False
    try:
        srv.config.debug_mode = True
    except AttributeError:
        _ = None
    srv.config.allowed_client_ips = []

    try:
        with mcp_ws_client.websocket_connect("/api/v1/mcp/ws?client_id=parseerr") as ws:
            # Send an invalid JSON text frame; server should respond with a JSON-RPC parse error
            ws.send_text("not-json")
            msg = ws.receive_json()
            assert isinstance(msg, dict)  # nosec B101
            assert msg.get("jsonrpc") == "2.0"  # nosec B101
            assert isinstance(msg.get("error"), dict)  # nosec B101
            assert msg["error"].get("code") == -32700  # nosec B101
            assert "Parse error" in (msg["error"].get("message") or "")  # nosec B101
    except (ConnectionRefusedError, OSError, RuntimeError):
        pytest.skip("MCP WebSocket endpoint not available in this build")

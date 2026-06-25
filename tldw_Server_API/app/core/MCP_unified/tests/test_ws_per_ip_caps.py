"""
Tests for per-IP WebSocket connection caps in MCP Unified.
"""

import os
import os as _os
from datetime import datetime, timedelta, timezone

import pytest
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.app.core.MCP_unified import get_mcp_server, reset_mcp_server
from tldw_Server_API.app.core.MCP_unified.server import WebSocketConnection

# Minimize startup side-effects for tests
_os.environ.setdefault("TEST_MODE", "true")
_os.environ.setdefault("ENABLE_TRACING", "false")
_os.environ.setdefault("OTEL_METRICS_EXPORTER", "console")
os.environ.setdefault("MCP_WS_AUTH_REQUIRED", "false")
os.environ.setdefault("MCP_ALLOWED_IPS", "")


@pytest.mark.asyncio
async def test_ws_per_ip_cap_enforced(monkeypatch):
    # Configure per-IP cap before app/server initialization
    os.environ["MCP_WS_MAX_CONNECTIONS_PER_IP"] = "2"
    os.environ["MCP_WS_MAX_CONNECTIONS"] = "50"
    os.environ["MCP_WS_ALLOWED_ORIGINS"] = "*"

    # Clear cached config to pick up env vars
    from tldw_Server_API.app.core.MCP_unified.config import get_config
    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    from fastapi.testclient import TestClient

    from tldw_Server_API.app.main import app

    await reset_mcp_server()
    client = TestClient(app)
    server = get_mcp_server()
    server.config.ws_auth_required = False
    server.config.allowed_client_ips = []
    server.config.blocked_client_ips = []
    server.config.ws_max_connections_per_ip = 2
    server.config.ws_max_connections = 50
    server.config.ws_allowed_origins = ["*"]

    # Open two connections (at cap)
    ws1 = client.websocket_connect("/api/v1/mcp/ws?client_id=ipcap1")
    ws1.__enter__()
    ws2 = client.websocket_connect("/api/v1/mcp/ws?client_id=ipcap2")
    ws2.__enter__()

    # Third should be rejected
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/api/v1/mcp/ws?client_id=ipcap3"):
            pass

    assert exc_info.value.code == 1013
    assert exc_info.value.reason == "Too many connections from IP"

    # Cleanup
    ws2.__exit__(None, None, None)
    ws1.__exit__(None, None, None)

    # Assert metrics recorded a rejection
    from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import get_metrics_collector
    collector = get_metrics_collector()
    internal = collector.get_internal_metrics(300)
    if "ws_rejection" not in internal:
        metrics = list(collector._metrics.get("ws_rejection", []))
        assert metrics, "Expected ws_rejection metric to be recorded"
        assert any(m.labels.get("reason") == "per_ip_cap" for m in metrics)
    else:
        assert internal["ws_rejection"]["type"] == "counter"
        assert internal["ws_rejection"]["value"] >= 1


@pytest.mark.asyncio
async def test_stale_connection_cleanup_decrements_per_ip_count():
    await reset_mcp_server()
    server = get_mcp_server()
    server.connections.clear()
    server._ip_connection_counts.clear()

    class _CloseOnlyWebSocket:
        def __init__(self) -> None:
            self.close_calls: list[tuple[int, str]] = []

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.close_calls.append((code, reason))

    websocket = _CloseOnlyWebSocket()
    connection = WebSocketConnection(
        websocket=websocket,  # type: ignore[arg-type]
        connection_id="stale-1",
        client_id="c1",
        metadata={"client_ip": "127.0.0.1"},
    )
    connection.last_activity = datetime.now(timezone.utc) - timedelta(seconds=301)
    server.connections[connection.connection_id] = connection
    server._ip_connection_counts["127.0.0.1"] = 1

    await server._cleanup_stale_connections()

    assert connection.connection_id not in server.connections
    assert websocket.close_calls == [(1001, "Connection timeout")]
    assert server._ip_connection_counts == {}

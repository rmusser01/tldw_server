"""Coverage for the MCP WebSocket connection registry being drained on disconnect.

`handle_websocket` registers the connection and reserves a per-IP slot before accepting,
then releases both in its `finally`. Nothing asserted that the release actually happens,
so a stale entry in `server.connections` or an undecremented `_ip_connection_counts`
bucket would have been invisible -- and once the bucket reaches
`ws_max_connections_per_ip`, that client is refused for the life of the process.

Scope, stated honestly: these tests exercise the ordinary disconnect, which arrives as
`WebSocketDisconnect` and lets the `finally` run to completion. They do NOT reach the
cancelled path (shutdown, `task.cancel()`), where an await inside a finally re-raises
CancelledError and abandons the statements after it. The deregistration in
`server.py` is ordered ahead of `await stream.stop()` to be safe there, but that
ordering is hardening -- these tests pass with either order, and no failing case for the
cancelled path was constructed.

The identical ordering question in the sandbox WebSocket *was* reproducible and is
covered by `tests/sandbox/test_ws_connection_quotas.py`.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as mcp_ep
from tldw_Server_API.app.core.MCP_unified.server import MCPServer
from tldw_Server_API.app.main import app

pytestmark = pytest.mark.integration


class _OkProtocol:
    async def process_request(self, payload, context):
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPResponse

        return MCPResponse(result={"ok": True}, id=None)


def _build_server() -> MCPServer:
    server = MCPServer()
    server.initialized = True
    server.protocol = _OkProtocol()
    server.config.ws_auth_required = False
    server.config.ws_allow_query_auth = True
    return server


def _connect_once(client: TestClient, client_id: str) -> None:
    with client.websocket_connect(f"/api/v1/mcp/ws?client_id={client_id}") as ws:
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


def test_disconnect_drains_the_connection_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _build_server()
    monkeypatch.setattr(mcp_ep, "get_mcp_server", lambda: server)
    client = TestClient(app)

    _connect_once(client, "release-probe")

    assert server.get_active_connection_count() == 0, (
        "the connection was left in the registry after disconnect, so the finally's "
        "deregistration did not run to completion"
    )
    # No public accessor exists for the per-IP reservation, and it is the half that
    # actually locks a client out, so it is asserted directly.
    assert server._ip_connection_counts == {}, (
        f"per-IP connection slot not released: {dict(server._ip_connection_counts)}. "
        "Once a bucket reaches ws_max_connections_per_ip that client is refused for "
        "the life of the process."
    )


def test_repeated_connects_do_not_exhaust_the_per_ip_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consequence, stated as behaviour rather than as internal state."""
    server = _build_server()
    server.config.ws_max_connections_per_ip = 2
    monkeypatch.setattr(mcp_ep, "get_mcp_server", lambda: server)
    client = TestClient(app)

    for attempt in range(4):
        _connect_once(client, f"cap-probe-{attempt}")

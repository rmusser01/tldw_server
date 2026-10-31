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
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as mcp_ep
from tldw_Server_API.app.core.MCP_unified.protocol import MCPResponse
from tldw_Server_API.app.core.MCP_unified.server import MCPServer
from tldw_Server_API.app.main import app

pytestmark = pytest.mark.integration


class _OkProtocol:
    """Minimal protocol stand-in that accepts any request and answers success.

    Only `process_request` is exercised: these tests are about what the connection
    registry does around the handler, not about protocol dispatch.
    """

    async def process_request(
        self, payload: dict[str, Any], context: Any
    ) -> MCPResponse:
        return MCPResponse(result={"ok": True}, id=None)


def _build_server() -> MCPServer:
    """Return a server that accepts unauthenticated query-param WebSocket clients."""
    server = MCPServer()
    server.initialized = True
    server.protocol = _OkProtocol()
    server.config.ws_auth_required = False
    server.config.ws_allow_query_auth = True
    return server


def _connect_once(client: TestClient, client_id: str) -> None:
    """Open one WebSocket, complete an initialize round-trip, and close it."""
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
    # The per-IP reservation is deliberately NOT asserted here. It has no public
    # accessor, and asserting the private counter couples this test to a field name
    # while adding nothing: the test below already proves the slot is released, by
    # exhausting the cap if it is not.


def test_repeated_connects_do_not_exhaust_the_per_ip_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consequence, stated as behaviour rather than as internal state.

    With the cap at 2, a fourth sequential connect can only succeed if each of the first
    three released its reservation on close. This is what makes asserting the private
    counter unnecessary.
    """
    server = _build_server()
    server.config.ws_max_connections_per_ip = 2
    monkeypatch.setattr(mcp_ep, "get_mcp_server", lambda: server)
    client = TestClient(app)

    for attempt in range(4):
        _connect_once(client, f"cap-probe-{attempt}")

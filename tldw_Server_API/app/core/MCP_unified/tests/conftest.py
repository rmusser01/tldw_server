"""Local test config for MCP WS tests.

- Adds a reusable auth-disabled WebSocket TestClient fixture for MCP.
"""
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.MCP_unified import get_mcp_server
from tldw_Server_API.app.core.MCP_unified.tests.support import build_mcp_test_client


@pytest.fixture
def mcp_ws_client(monkeypatch):
    """Reusable MCP WS client with auth disabled and relaxed IP checks.

    - Forces TEST_MODE to simplify route gating and startup
    - Disables MCP WS auth and IP allowlist for local tests
    """
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("MCP_WS_AUTH_REQUIRED", "false")
    # Accept both empty and JSON list for env-based list parsing
    monkeypatch.setenv("MCP_ALLOWED_IPS", "")
    with build_mcp_test_client() as client:
        server = get_mcp_server()
        server.config.ws_auth_required = False
        server.config.allowed_client_ips = []
        server.config.blocked_client_ips = []
        try:
            server.config.debug_mode = True
        except AttributeError:
            _ = None
        yield client


@pytest.fixture
def ws_client(mcp_ws_client):
    """Alias for mcp_ws_client to match common fixture name across tests."""
    yield mcp_ws_client

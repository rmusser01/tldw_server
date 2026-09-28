"""Local test config for MCP WS tests.

- Adds a reusable auth-disabled WebSocket TestClient fixture for MCP.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from tldw_Server_API.app.core.MCP_unified import get_mcp_server
from tldw_Server_API.app.core.MCP_unified.tests.support import build_mcp_test_client


@pytest.fixture(scope="session", autouse=True)
def _seed_single_user_admin_rbac():
    """Grant the single-user principal (id 1) its admin role once per session.

    Production seeds this at startup (services/startup_auth.py). These tests
    build bare apps or call modules with user_id="1" directly, so without the
    seed the real AuthNZ RBAC adapter denies them on a fresh users.db (they
    previously passed only when an earlier test had already seeded it).
    """
    from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    saved = os.environ.get("AUTH_MODE")
    os.environ["AUTH_MODE"] = "single_user"
    try:
        asyncio.run(ensure_single_user_rbac_seed_if_needed())
    finally:
        if saved is None:
            os.environ.pop("AUTH_MODE", None)
        else:
            os.environ["AUTH_MODE"] = saved
        reset_settings()
    yield


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

"""Local test config for MCP WS tests.

- Adds a reusable auth-disabled WebSocket TestClient fixture for MCP.
"""
from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.MCP_unified import get_mcp_server
from tldw_Server_API.app.core.MCP_unified import server as mcp_server_module
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


@pytest.fixture(autouse=True)
def _healthy_empty_llm_provider_override_store(request):
    """Every test starts with the healthy, empty provider-override store app startup gives production.

    The tldw_Server_API/tests conftest does the same for its tests; this directory is outside it. The MCP
    authoritative path reads the store on every resolution (TASK-15581), and fails closed when it cannot. The
    store's own setter also resets its recovery state, and the store is restored after (TASK-15577 M2). Mark a test
    ``ambient_llm_provider_override_store`` to see the import-time (unloaded, unhealthy) store instead.
    """
    if request.node.get_closest_marker("ambient_llm_provider_override_store") is not None:
        yield
        return
    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides as store

    with store._OVERRIDE_LOCK:
        overrides = dict(store._OVERRIDE_CACHE)
        healthy = store._OVERRIDE_CACHE_HEALTHY
        ttl_disabled = store._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
    set_overrides = store.set_llm_provider_overrides_cache_for_tests  # bound now: a test may patch the setter
    set_overrides({})
    try:
        yield
    finally:
        set_overrides(overrides, healthy=healthy, ttl_enabled=not ttl_disabled)


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


@pytest.fixture
def allow_knowledge_source_tools(monkeypatch):
    """Give aggregation-only stub tests explicit source-tool permission."""
    async def allow(*_args, **_kwargs):
        return True

    protocol = SimpleNamespace(_has_tool_permission=allow)
    monkeypatch.setattr(mcp_server_module, "get_mcp_server", lambda: SimpleNamespace(protocol=protocol))

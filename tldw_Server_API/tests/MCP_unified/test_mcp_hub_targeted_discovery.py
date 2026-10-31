"""Targeted managed discovery through real storage, module, registry and router."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import mcp_hub_management as routes
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.MCP_unified.external_servers import manager as manager_mod
from tldw_Server_API.app.core.MCP_unified.external_servers.transports.base import (
    ExternalMCPTransportAdapter,
    ExternalToolCallResult,
    ExternalToolDefinition,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleCircuitBreakerOpenError, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.external_federation_module import (
    ExternalFederationModule,
)
from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry
from tldw_Server_API.app.services.mcp_hub_external_registry_service import McpHubExternalRegistryService
from tldw_Server_API.app.services.mcp_hub_service import McpHubService
from tldw_Server_API.app.services.mcp_hub_tool_registry import McpHubToolRegistryService

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
BASE = "/api/v1/mcp/hub/external-servers"
SMOKE = Path(__file__).parents[2] / "app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py"


def _principal(*, admin: bool = True, configure: bool = True) -> AuthPrincipal:
    return AuthPrincipal(kind="user", user_id=1, subject="1", roles=["admin"] if admin else [],
                         permissions=["system.configure"] if configure else [], org_ids=[], team_ids=[])


class _DiscoveryAdapter(ExternalMCPTransportAdapter):
    """External discovery seam for bounded failure and concurrency controls."""

    def __init__(self, server_id: str, *, fail: bool = False, block: bool = False):
        super().__init__(server_id)
        self.fail = fail
        self.block = block
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.active = 0
        self.max_active = 0

    @property
    def transport_name(self) -> str:
        return "stdio"

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        self.release.set()

    async def health_check(self) -> dict[str, bool]:
        return {"configured": True, "connected": True}

    async def list_tools(self) -> list[ExternalToolDefinition]:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            if self.block:
                await self.release.wait()
            if self.fail:
                raise RuntimeError("private upstream credential detail")
            return [ExternalToolDefinition(name="echo.search", input_schema={"type": "object"})]
        finally:
            self.active -= 1

    async def call_tool(self, tool_name: str, arguments: dict[str, Any], context=None,
                        runtime_auth=None) -> ExternalToolCallResult:
        raise AssertionError("Discovery must not execute an upstream tool")


@pytest.fixture
async def runtime(tmp_path, monkeypatch, fresh_services):
    """Existing canonical temporary AuthNZ storage and local module lifecycle."""
    with monkeypatch.context() as patch:
        path = tmp_path / "users.db"
        patch.setenv("AUTH_MODE", "multi_user")
        patch.setenv("DATABASE_URL", f"sqlite:///{path}")
        reset_settings()
        pool = await get_db_pool()
        ensure_authnz_tables(path)
        repo = McpHubRepo(pool)
        await repo.ensure_tables()
        service = McpHubService(repo=repo)
        registry = ModuleRegistry()
        server = SimpleNamespace(initialized=True, module_registry=registry)
        patch.setattr(routes, "get_mcp_server", lambda: server)
        config = ModuleConfig(name="external_federation", timeout_seconds=2, max_concurrent=1,
                              settings={"external_server_loader": McpHubExternalRegistryService(repo).list_runtime_servers,
                                        "external_credential_broker": lambda **_kwargs: None})
        app = FastAPI()
        app.include_router(routes.router, prefix="/api/v1")
        app.dependency_overrides[get_auth_principal] = lambda: _principal()
        app.dependency_overrides[routes.get_mcp_hub_service] = lambda: service
        app.dependency_overrides[routes.get_mcp_hub_tool_registry_dep] = lambda: McpHubToolRegistryService(registry)
        try:
            await registry.register_module("external_federation", ExternalFederationModule, config)
            async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
                yield SimpleNamespace(repo=repo, service=service, registry=registry, app=app, client=client)
        finally:
            await registry.shutdown_all()
    reset_settings()


async def _save(runtime, *, server_id="uat-smoke", enabled=True, scope="user", owner=1,
                source="managed", superseded=None, config=None):
    return await runtime.repo.upsert_external_server(
        server_id=server_id, name="UAT Smoke", transport="stdio",
        config_json=json.dumps(config if config is not None else {"stdio": {"command": sys.executable,
                                                                         "args": ["-u", str(SMOKE)]}}),
        owner_scope_type=scope, owner_scope_id=owner, enabled=enabled,
        server_source=source, superseded_by_server_id=superseded, actor_id=1,
    )


async def test_saved_after_runtime_startup_discovers_existing_stdio_fixture_and_registers_tools(runtime):
    assert await runtime.registry.find_module_for_tool("ext.uat-smoke.echo.search") is None  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    await _save(runtime)
    response = await runtime.client.post(f"{BASE}/uat-smoke/refresh-discovery")
    assert response.status_code == 200  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert response.json()["tool_count"] == 1  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert response.json()["display_state"] == "ready"  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert response.json()["refresh_result"]["errors"] == {}  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert await runtime.registry.find_module_for_tool("ext.uat-smoke.echo.search") is not None  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert (await runtime.repo.get_external_server("uat-smoke"))["name"] == "UAT Smoke"  # nosec B101 - pytest behavioral oracle, not a runtime guard.


async def test_new_server_discovery_failure_is_attempted_and_sanitized(runtime, monkeypatch):
    adapter = _DiscoveryAdapter("uat-smoke", fail=True)
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _config: adapter)
    await _save(runtime)
    response = await runtime.client.post(f"{BASE}/uat-smoke/refresh-discovery")
    assert response.status_code == 200  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert adapter.entered.is_set()  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert response.json()["primary_reason_code"] == "discovery_failed"  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert response.json()["refresh_result"]["errors"] == {"uat-smoke": "Discovery refresh failed."}  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert "private upstream credential" not in response.text  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert await runtime.registry.find_module_for_tool("ext.uat-smoke.echo.search") is None  # nosec B101 - pytest behavioral oracle, not a runtime guard.


async def test_unavailable_module_stays_unavailable_without_auto_enabling(runtime):
    await _save(runtime)
    await runtime.registry.unregister_module("external_federation")
    response = await runtime.client.post(f"{BASE}/uat-smoke/refresh-discovery")
    assert response.status_code == 503  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert await runtime.registry.get_all_modules() == {}  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert await runtime.repo.get_external_server("uat-smoke") is not None  # nosec B101 - pytest behavioral oracle, not a runtime guard.


@pytest.mark.parametrize(("admin", "configure", "owner", "status"), [
    (False, False, 1, 403), (False, False, 2, 404), (True, True, 1, 404),
])
async def test_mutation_and_visibility_denials_do_not_start_discovery(runtime, monkeypatch, admin, configure, owner, status):
    def unexpected_adapter(_config):
        raise AssertionError("Denied request must not start a transport")
    monkeypatch.setattr(manager_mod, "build_transport_adapter", unexpected_adapter)
    await _save(runtime, owner=owner)
    runtime.app.dependency_overrides[get_auth_principal] = lambda: _principal(admin=admin, configure=configure)
    server_id = "missing" if admin else "uat-smoke"
    action = "validate" if owner == 2 else "refresh-discovery"
    before = await runtime.repo.get_external_server("uat-smoke")
    response = await runtime.client.post(f"{BASE}/{server_id}/{action}")
    assert response.status_code == status  # nosec B101 - pytest behavioral oracle, not a runtime guard.

    assert await runtime.repo.get_external_server("uat-smoke") == before  # nosec B101 - Rejected discovery must preserve the managed server row.


@pytest.mark.parametrize(("enabled", "source", "superseded"), [
    (False, "managed", None), (True, "legacy", None), (True, "managed", "replacement"),
])
async def test_unavailable_rows_refuse_before_transport(runtime, monkeypatch, enabled, source, superseded):
    def unexpected_adapter(_config):
        raise AssertionError("Unavailable row must not start a transport")
    monkeypatch.setattr(manager_mod, "build_transport_adapter", unexpected_adapter)
    await _save(runtime, enabled=enabled, source=source, superseded=superseded)
    before = await runtime.repo.get_external_server("uat-smoke")
    response = await runtime.client.post(f"{BASE}/uat-smoke/refresh-discovery")
    assert response.status_code == 409  # nosec B101 - pytest behavioral oracle, not a runtime guard.

    assert await runtime.repo.get_external_server("uat-smoke") == before  # nosec B101 - Rejected discovery must preserve the managed server row.


async def test_same_server_requests_reconcile_serially(runtime, monkeypatch):
    adapter = _DiscoveryAdapter("uat-smoke", block=True)
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _config: adapter)
    await _save(runtime)
    first = asyncio.create_task(runtime.client.post(f"{BASE}/uat-smoke/refresh-discovery"))
    second = None
    try:
        await asyncio.wait_for(adapter.entered.wait(), timeout=1)
        second = asyncio.create_task(runtime.client.post(f"{BASE}/uat-smoke/refresh-discovery"))
        await asyncio.sleep(0)
        assert not second.done()  # nosec B101 - pytest behavioral oracle, not a runtime guard.
        adapter.release.set()
        results = await asyncio.gather(first, second)
        assert [result.status_code for result in results] == [200, 200]  # nosec B101 - pytest behavioral oracle, not a runtime guard.
        assert [result.json()["tool_count"] for result in results] == [1, 1]  # nosec B101 - pytest behavioral oracle, not a runtime guard.
        assert adapter.max_active == 1  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    finally:
        adapter.release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


async def test_module_semaphore_blocks_reconciliation_until_admitted(runtime, monkeypatch):
    adapter = _DiscoveryAdapter("uat-smoke")
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _config: adapter)
    await _save(runtime)
    module = await runtime.registry.get_module("external_federation")
    await module._semaphore.acquire()
    task = asyncio.create_task(runtime.client.post(f"{BASE}/uat-smoke/refresh-discovery"))
    try:
        await asyncio.sleep(0)
        assert not adapter.entered.is_set()  # nosec B101 - pytest behavioral oracle, not a runtime guard.
        assert not task.done()  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    finally:
        module._semaphore.release()
    response = await task
    assert response.status_code == 200  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert response.json()["tool_count"] == 1  # nosec B101 - pytest behavioral oracle, not a runtime guard.


async def test_open_module_breaker_refuses_before_new_transport_dispatch(runtime, monkeypatch):
    adapter = _DiscoveryAdapter("uat-smoke")
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _config: adapter)
    await _save(runtime)
    module = await runtime.registry.get_module("external_federation")
    for _ in range(module.config.circuit_breaker_threshold):
        module.record_circuit_breaker_failure()
    with pytest.raises(ModuleCircuitBreakerOpenError):
        await routes._execute_external_refresh_tool("uat-smoke")
    assert not adapter.entered.is_set()  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert await runtime.registry.find_module_for_tool("ext.uat-smoke.echo.search") is None  # nosec B101 - pytest behavioral oracle, not a runtime guard.


async def test_module_timeout_preserves_bound_on_semaphore_admission(runtime, monkeypatch):
    adapter = _DiscoveryAdapter("uat-smoke")
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _config: adapter)
    await _save(runtime)
    module = await runtime.registry.get_module("external_federation")
    module.config.timeout_seconds = 0.02
    await module._semaphore.acquire()
    try:
        with pytest.raises(Exception, match="Operation timeout after 0.02s"):
            await routes._execute_external_refresh_tool("uat-smoke")
    finally:
        module._semaphore.release()
    assert not adapter.entered.is_set()  # nosec B101 - pytest behavioral oracle, not a runtime guard.


async def test_malformed_owned_config_is_refused_without_starting_transport(runtime, monkeypatch):
    def unexpected_adapter(_config):
        raise AssertionError("Invalid runtime configuration must not start a transport")
    monkeypatch.setattr(manager_mod, "build_transport_adapter", unexpected_adapter)
    await _save(runtime, config={})
    response = await runtime.client.post(f"{BASE}/uat-smoke/refresh-discovery")
    assert response.status_code == 200  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert response.json()["primary_reason_code"] == "discovery_failed"  # nosec B101 - pytest behavioral oracle, not a runtime guard.
    assert response.json()["refresh_result"]["errors"] == {"uat-smoke": "Discovery refresh failed."}  # nosec B101 - pytest behavioral oracle, not a runtime guard.

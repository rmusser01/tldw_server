"""Current managed authority before actual external-manager dispatch."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import mcp_hub_management as routes
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.MCP_unified.external_servers import manager as manager_mod
from tldw_Server_API.app.core.MCP_unified.external_servers.manager import ExternalServerManager
from tldw_Server_API.app.core.MCP_unified.external_servers.transports.base import (
    ExternalMCPTransportAdapter,
    ExternalToolCallResult,
    ExternalToolDefinition,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.external_federation_module import (
    ExternalFederationModule,
)
from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry
from tldw_Server_API.app.services.mcp_hub_external_registry_service import McpHubExternalRegistryService
from tldw_Server_API.app.services.mcp_hub_service import McpHubService

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
SMOKE = Path(__file__).parents[2] / "app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py"
SERVER = "uat-policy"
READ = f"ext.{SERVER}.artifact.read"
WRITE = f"ext.{SERVER}.artifact.write"


def config(policy=None, *, args=None):
    return {"stdio": {"command": sys.executable, "args": args or ["-u", str(SMOKE)]},
            "policy": policy if policy is not None else {"allow_tool_patterns": ["*"], "allow_writes": True,
                                                         "require_write_confirmation": False}}


class CallAdapter(ExternalMCPTransportAdapter):
    """Bounded external wire seam; manager/storage/policy remain real."""

    def __init__(self, server_id):
        super().__init__(server_id)
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False
        self.fail_discovery = False

    @property
    def transport_name(self):
        return "stdio"

    async def connect(self):
        return None

    async def close(self):
        self.release.set()

    async def health_check(self):
        return {"configured": True, "connected": True}

    async def list_tools(self):
        if self.fail_discovery:
            raise RuntimeError("private transport detail")
        return [ExternalToolDefinition(name="artifact.read", metadata={"read_only": True}),
                ExternalToolDefinition(name="artifact.write", metadata={"read_only": False})]

    async def call_tool(self, tool_name, arguments, context=None, runtime_auth=None):
        self.calls.append((tool_name, dict(arguments)))
        self.entered.set()
        if self.block:
            await self.release.wait()
        return ExternalToolCallResult(content={"accepted": tool_name}, is_error=False)


@pytest.fixture
async def runtime(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        path = tmp_path / "users.db"
        patch.setenv("AUTH_MODE", "multi_user")
        patch.setenv("DATABASE_URL", f"sqlite:///{path}")
        reset_settings()
        await reset_db_pool()
        pool = await get_db_pool()
        ensure_authnz_tables(path)
        repo = McpHubRepo(pool)
        await repo.ensure_tables()
        service = McpHubService(repo)
        await service.create_external_server(server_id=SERVER, name="Owned policy fixture", transport="stdio",
                                            config=config(), owner_scope_type="user", owner_scope_id=1,
                                            enabled=True, actor_id=1)
        adapters = []
        original_factory = manager_mod.build_transport_adapter

        def factory(server):
            adapter = CallAdapter(server.id)
            adapters.append(adapter)
            return adapter

        patch.setattr(manager_mod, "build_transport_adapter", factory)
        registry = ModuleRegistry()
        module_config = ModuleConfig(name="external_federation", timeout_seconds=3,
                                     settings={"external_server_loader": McpHubExternalRegistryService(repo).list_runtime_servers,
                                               "external_credential_broker": lambda **_kwargs: None})
        app = FastAPI()
        app.include_router(routes.router, prefix="/api/v1")
        principal = AuthPrincipal(kind="user", user_id=1, subject="1", roles=["admin"],
                                  permissions=["system.configure"], org_ids=[], team_ids=[])
        app.dependency_overrides[get_auth_principal] = lambda: principal
        app.dependency_overrides[routes.get_mcp_hub_service] = lambda: service
        patch.setattr(routes, "get_mcp_server", lambda: SimpleNamespace(initialized=True, module_registry=registry))
        try:
            await registry.register_module("external_federation", ExternalFederationModule, module_config)
            module = await registry.get_module("external_federation")
            async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
                yield SimpleNamespace(repo=repo, service=service, registry=registry, module=module,
                                      manager=module._manager, adapters=adapters, client=client, app=app,
                                      original_factory=original_factory)
        finally:
            for adapter in adapters:
                adapter.release.set()
            await registry.shutdown_all()
            await reset_db_pool()
    reset_settings()


async def test_saved_deny_all_immediately_blocks_cached_read(runtime):
    await runtime.module.execute_tool(READ, {})
    await runtime.service.update_external_server(SERVER, config=config({"allow_tool_patterns": [], "deny_tool_patterns": ["*"]}), actor_id=1)
    with pytest.raises(PermissionError, match="blocked by local policy"):
        await runtime.module.execute_tool(READ, {})
    assert len(runtime.adapters[0].calls) == 1  # nosec B101 - pytest effect oracle.
    assert (await runtime.repo.get_external_server(SERVER))["config"]["policy"]["deny_tool_patterns"] == ["*"]  # nosec B101 - pytest authority/effect oracle.


async def test_real_update_route_restricts_existing_registry_execution(runtime):
    module = await runtime.registry.find_module_for_tool(READ)
    assert module is runtime.module  # nosec B101 - pytest authority/effect oracle.
    response = await runtime.client.put(f"/api/v1/mcp/hub/external-servers/{SERVER}", json={"config": config({"allow_tool_patterns": ["echo.search"]})})
    assert response.status_code == 200  # nosec B101 - pytest authority/effect oracle.
    with pytest.raises(PermissionError, match="blocked by local policy"):
        await module.execute_tool(READ, {})
    assert runtime.adapters[0].calls == []  # nosec B101 - pytest authority/effect oracle.


@pytest.mark.parametrize("change", ["disabled", "removed"])
async def test_disabled_or_removed_managed_server_cannot_use_cached_adapter(runtime, change):
    if change == "disabled":
        await runtime.service.update_external_server(SERVER, enabled=False, actor_id=1)
    else:
        assert await runtime.service.delete_external_server(SERVER, actor_id=1)  # nosec B101 - pytest authority/effect oracle.
    with pytest.raises(PermissionError):
        await runtime.module.execute_tool(READ, {})
    assert runtime.adapters[0].calls == []  # nosec B101 - pytest authority/effect oracle.


@pytest.mark.parametrize("policy,arguments,reason", [
    ({"allow_writes": False}, {"__confirm_write": True}, "disabled by local policy"),
    ({"allow_writes": True, "require_write_confirmation": True}, {}, "Write confirmation required"),
])
async def test_current_write_and_confirmation_controls_apply_without_reconcile(runtime, policy, arguments, reason):
    await runtime.service.update_external_server(SERVER, config=config(policy), actor_id=1)
    with pytest.raises(PermissionError, match=reason):
        await runtime.module.execute_tool(WRITE, arguments)
    assert runtime.adapters[0].calls == []  # nosec B101 - pytest authority/effect oracle.


async def test_unchanged_allowed_read_and_current_confirmed_write_preserve_arguments(runtime):
    result = await runtime.module.execute_tool(READ, {"path": "owned.txt"})
    assert result["is_error"] is False  # nosec B101 - pytest authority/effect oracle.
    await runtime.service.update_external_server(SERVER, name="New display name", config=config({"allow_writes": True}), actor_id=1)
    await runtime.module.execute_tool(WRITE, {"content": "confirmed", "__confirm_write": True})
    assert runtime.adapters[0].calls == [("artifact.read", {"path": "owned.txt"}), ("artifact.write", {"content": "confirmed"})]  # nosec B101 - pytest authority/effect oracle.
    assert len(runtime.adapters) == 1  # nosec B101 - no policy/display transport restart.


async def test_malformed_current_row_refuses_but_unrelated_partial_error_is_isolated(runtime):
    await runtime.service.create_external_server(server_id="broken-other", name="Broken other", transport="stdio",
                                                config={"stdio": {}}, owner_scope_type="user", owner_scope_id=1,
                                                enabled=True, actor_id=1)
    assert (await runtime.module.execute_tool(READ, {}))["is_error"] is False  # nosec B101 - pytest authority/effect oracle.
    await runtime.service.update_external_server(SERVER, config={"stdio": {}}, actor_id=1)
    with pytest.raises(PermissionError):
        await runtime.module.execute_tool(READ, {})
    assert len(runtime.adapters[0].calls) == 1  # nosec B101 - pytest authority/effect oracle.


async def test_unavailable_loader_refuses_with_safe_error(runtime, monkeypatch):
    async def failed_storage(*_args, **_kwargs):
        raise RuntimeError("private credential/storage detail")
    monkeypatch.setattr(runtime.repo, "list_external_servers", failed_storage)
    with pytest.raises(PermissionError) as failure:
        await runtime.module.execute_tool(READ, {})
    assert "private credential" not in str(failure.value)  # nosec B101 - pytest authority/effect oracle.
    assert runtime.adapters[0].calls == []  # nosec B101 - pytest authority/effect oracle.


@pytest.mark.parametrize("failure", ["construction", "discovery"])
async def test_failed_changed_transport_reconcile_does_not_authorize_old_adapter(runtime, monkeypatch, failure):
    await runtime.service.update_external_server(SERVER, config=config(args=["-u", str(SMOKE), "changed"]), actor_id=1)
    if failure == "construction":
        def broken_factory(_server):
            raise RuntimeError("private build detail")
        monkeypatch.setattr(manager_mod, "build_transport_adapter", broken_factory)
    else:
        def broken_factory(server):
            adapter = CallAdapter(server.id)
            adapter.fail_discovery = True
            runtime.adapters.append(adapter)
            return adapter
        monkeypatch.setattr(manager_mod, "build_transport_adapter", broken_factory)
    refresh = await runtime.manager.reconcile_servers(SERVER)
    assert refresh["errors"] == {SERVER: "external_server_discovery_failed"}  # nosec B101 - pytest authority/effect oracle.
    assert any(tool.virtual_name == READ for tool in runtime.manager.list_virtual_tools())  # nosec B101 - last-good inventory retained.
    with pytest.raises(PermissionError):
        await runtime.module.execute_tool(READ, {})
    assert all(adapter.calls == [] for adapter in runtime.adapters)  # nosec B101 - pytest authority/effect oracle.


async def test_successful_explicit_reconcile_admits_changed_transport(runtime):
    await runtime.service.update_external_server(SERVER, config=config(args=["-u", str(SMOKE), "changed"]), actor_id=1)
    refresh = await routes._refresh_external_server_discovery_runtime(server_id=SERVER)
    assert refresh.ok is True  # nosec B101 - pytest authority/effect oracle.
    module = await runtime.registry.find_module_for_tool(READ)
    assert (await module.execute_tool(READ, {}))["is_error"] is False  # nosec B101 - pytest authority/effect oracle.
    assert len(runtime.adapters) == 2  # nosec B101 - pytest authority/effect oracle.
    assert runtime.adapters[0].calls == []  # nosec B101 - pytest authority/effect oracle.


async def test_call_waiting_for_existing_lock_rechecks_policy_after_inflight_call(runtime):
    first_adapter = runtime.adapters[0]
    first_adapter.block = True
    admitted = asyncio.create_task(runtime.module.execute_tool(READ, {"first": True}))
    queued = None
    try:
        await asyncio.wait_for(first_adapter.entered.wait(), timeout=2)
        queued = asyncio.create_task(runtime.module.execute_tool(READ, {"later": True}))
        await asyncio.sleep(0)
        assert not queued.done()  # nosec B101 - pytest authority/effect oracle.
        await runtime.service.update_external_server(SERVER, config=config({"deny_tool_patterns": ["*"]}), actor_id=1)
        first_adapter.release.set()
        assert (await admitted)["is_error"] is False  # nosec B101 - already admitted call may finish.
        with pytest.raises(PermissionError, match="blocked by local policy"):
            await queued
        assert first_adapter.calls == [("artifact.read", {"first": True})]  # nosec B101 - pytest authority/effect oracle.
    finally:
        first_adapter.release.set()
        await asyncio.gather(admitted, *([queued] if queued else []), return_exceptions=True)


async def test_supported_standalone_file_loader_uses_current_policy(tmp_path, monkeypatch):
    path = tmp_path / "external.json"
    payload = {"servers": [{"id": SERVER, "name": "File fixture", "transport": "stdio", **config()}]}
    path.write_text(json.dumps(payload))
    adapter = CallAdapter(SERVER)
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _server: adapter)
    manager = ExternalServerManager(str(path))
    try:
        await manager.initialize()
        assert (await manager.execute_virtual_tool(READ, {}))["is_error"] is False  # nosec B101 - pytest authority/effect oracle.
        payload["servers"][0]["policy"] = {"deny_tool_patterns": ["*"]}
        path.write_text(json.dumps(payload))
        with pytest.raises(PermissionError, match="blocked by local policy"):
            await manager.execute_virtual_tool(READ, {})
        assert len(adapter.calls) == 1  # nosec B101 - pytest authority/effect oracle.
    finally:
        await manager.shutdown()


async def test_real_installed_stdio_fixture_still_executes_after_reconcile(runtime, monkeypatch):
    monkeypatch.setattr(manager_mod, "build_transport_adapter", runtime.original_factory)
    await runtime.service.update_external_server(SERVER, config=config({"allow_tool_patterns": ["echo.search"]}), actor_id=1)
    assert (await runtime.manager.reconcile_servers(SERVER))["errors"] == {}  # nosec B101 - pytest authority/effect oracle.
    result = await runtime.module.execute_tool(f"ext.{SERVER}.echo.search", {"query": "current authority"})
    assert result["content"] == [{"type": "text", "text": "echo.search:current authority"}]  # nosec B101 - pytest authority/effect oracle.

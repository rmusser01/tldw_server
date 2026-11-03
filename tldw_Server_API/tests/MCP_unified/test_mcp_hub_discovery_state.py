"""Readiness follows real stored discovery outcomes, including an empty catalog."""

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
from tldw_Server_API.app.services.mcp_hub_service import McpHubService

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)
SERVER = "c11-empty"
SMOKE = Path(__file__).parents[2] / "app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py"


class EmptyRegistry:
    def __init__(self):
        self.entries = []

    async def list_entries(self):
        return self.entries


@pytest.fixture
async def discovery_api(tmp_path, monkeypatch):
    path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    reset_settings()
    await reset_db_pool()
    try:
        pool = await get_db_pool()
        ensure_authnz_tables(path)
        repo = McpHubRepo(pool)
        await repo.ensure_tables()
        service = McpHubService(repo)
        config = {"stdio": {"command": sys.executable, "args": ["-u", str(SMOKE)]}}
        await service.create_external_server(
            server_id=SERVER,
            name="Empty discovery",
            transport="stdio",
            config=config,
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=True,
            actor_id=1,
        )
        state = {"fail": False, "change_config": False, "global_fail": False, "incomplete": False, "calls": []}
        registry = EmptyRegistry()

        async def refresh(server_id):
            state["calls"].append(server_id)
            if state["change_config"]:
                await service.update_external_server(
                    SERVER,
                    config={"stdio": {"command": sys.executable, "args": ["-I", "-u", str(SMOKE)]}},
                    actor_id=1,
                )
            return {
                "ok": not (state["fail"] or state["global_fail"]),
                "server_id": server_id,
                "refreshed_servers": 0 if state["fail"] or state["incomplete"] else 1,
                "total_servers": 1,
                "virtual_tools": 0,
                "errors": (
                    {SERVER: "discarded failure detail"}
                    if state["fail"]
                    else {"module_registry": "discarded failure detail"}
                    if state["global_fail"]
                    else {}
                ),
            }

        async def global_refresh(*, server_id):
            return routes.ExternalServerDiscoveryRefreshResponse.model_validate(await refresh(server_id))

        monkeypatch.setattr(routes, "_refresh_external_server_discovery_runtime", global_refresh)
        app = FastAPI()
        app.include_router(routes.router, prefix="/api/v1")
        principal = AuthPrincipal(
            kind="user", user_id=1, username="admin", roles=["admin"], is_admin=True, permissions=["system.configure"]
        )
        app.dependency_overrides[get_auth_principal] = lambda: principal
        app.dependency_overrides[routes.get_mcp_hub_service] = lambda: service
        app.dependency_overrides[routes.get_mcp_hub_tool_registry_dep] = lambda: registry
        app.dependency_overrides[routes.get_mcp_hub_external_refresh_executor] = lambda: refresh
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            yield SimpleNamespace(
                client=client, state=state, app=app, pool=pool, service=service, registry=registry, refresh=refresh
            )
    finally:
        await reset_db_pool()
        reset_settings()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["scoped", "global-target", "global-all"])
@pytest.mark.parametrize("failed,expected", [(False, "no_tools_returned"), (True, "discovery_failed")])
async def test_discovery_outcome_survives_fresh_service_read(discovery_api, failed, expected, route):
    runtime = discovery_api
    runtime.state["fail"] = failed
    response = await refresh_route(runtime.client, route)
    assert response.status_code == 200  # nosec B101 -- normal refresh finishes before the fresh read.
    runtime.app.dependency_overrides[routes.get_mcp_hub_service] = lambda: McpHubService(McpHubRepo(runtime.pool))
    readiness = await runtime.client.get("/api/v1/mcp/hub/readiness")
    row = readiness.json()["servers"][0]
    assert row["primary_reason_code"] == expected  # nosec B101 -- an actual stored outcome must survive a new service instance.
    assert row["last_discovery_at"] is not None  # nosec B101 -- no synthetic timestamp is seeded by the test.
    assert "discarded failure detail" not in str(readiness.json())  # nosec B101 -- raw runtime failure text is never persisted/exposed.


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["scoped", "global-target", "global-all"])
async def test_discovery_does_not_publish_old_configuration_outcome(discovery_api, route):
    discovery_api.state["change_config"] = True
    response = await refresh_route(discovery_api.client, route)
    assert response.status_code == 409  # nosec B101 -- concurrent configuration changes require a new discovery.

    assert response.json() == {
        "detail": "External server configuration changed during discovery; refresh again"
    }  # nosec B101 - callers receive the actual recoverable configuration conflict.
    stored = await discovery_api.service.repo.get_external_server(SERVER)
    assert stored["last_successful_discovery_at"] is None  # nosec B101 - the retired discovery never stamps current success.


async def refresh_route(client, route):
    """Exercise both current refresh contracts, including UI's optional global target."""
    if route == "scoped":
        return await client.post(f"/api/v1/mcp/hub/external-servers/{SERVER}/refresh-discovery")
    return await client.post(
        "/api/v1/mcp/hub/external-servers/refresh-discovery",
        json={"server_id": SERVER} if route == "global-target" else {},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("upsert", [False, True])
async def test_changed_config_never_reuses_stale_nonempty_readiness(discovery_api, upsert):
    runtime = discovery_api
    await refresh_route(runtime.client, "scoped")
    original = await runtime.service.repo.get_external_server(SERVER)
    await runtime.service.update_external_server(SERVER, name="Renamed only", actor_id=1)
    renamed = await runtime.service.repo.get_external_server(SERVER)
    assert renamed["last_successful_discovery_at"] == original["last_successful_discovery_at"]  # nosec B101 -- names do not invalidate tool discovery.
    assert renamed["last_error_category"] is None  # nosec B101 -- display-only updates preserve success.
    config = {"stdio": {"command": sys.executable, "args": ["-I", "-u", str(SMOKE)]}}
    if upsert:
        await runtime.service.create_external_server(
            server_id=SERVER,
            name="Renamed only",
            transport="stdio",
            config=config,
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=True,
            actor_id=1,
            allow_existing=True,
        )
    else:
        await runtime.service.update_external_server(SERVER, config=config, actor_id=1)
    runtime.registry.entries = [
        {"tool_name": f"ext.{SERVER}.old", "metadata": {"server_id": SERVER, "federated": True}}
    ]
    readiness = await runtime.client.get("/api/v1/mcp/hub/readiness")
    row = readiness.json()["servers"][0]
    assert row["tool_count"] == 1  # nosec B101 -- the stale nonempty catalog is real readiness input.
    assert (row["display_state"], row["primary_reason_code"]) == ("stale", "config_changed")  # nosec B101 -- old nonempty names cannot imply current-config readiness.


@pytest.mark.asyncio
async def test_discovery_failure_keeps_prior_success_without_advancing_config_revision(discovery_api):
    runtime = discovery_api
    before = await runtime.service.repo.get_external_server(SERVER)
    await refresh_route(runtime.client, "global-all")
    success = await runtime.service.repo.get_external_server(SERVER)
    runtime.state["fail"] = True
    await refresh_route(runtime.client, "global-all")
    failed = await runtime.service.repo.get_external_server(SERVER)
    assert failed["last_successful_discovery_at"] == success["last_successful_discovery_at"]  # nosec B101 -- last-good history survives a failure.
    assert failed["last_error_category"] == "discovery_failed"  # nosec B101 -- failure dominates the old success.
    assert failed["updated_at"] == success["updated_at"] == before["updated_at"]  # nosec B101 -- status publication cannot rewrite the CAS config revision.


@pytest.mark.asyncio
async def test_global_catalog_publication_failure_cannot_stamp_success(discovery_api):
    runtime = discovery_api
    runtime.state["global_fail"] = True
    await refresh_route(runtime.client, "global-all")
    row = await runtime.service.repo.get_external_server(SERVER)
    assert row["last_error_category"] == "discovery_failed"  # nosec B101 -- registry-wide failures affect every refreshed server.
    assert row["last_successful_discovery_at"] is None  # nosec B101 -- transport discovery alone is not successful publication.


@pytest.mark.asyncio
async def test_global_refresh_still_retires_disabled_target_without_success_stamp(discovery_api):
    runtime = discovery_api
    await runtime.service.update_external_server(SERVER, enabled=False, actor_id=1)
    response = await refresh_route(runtime.client, "global-target")
    row = await runtime.service.repo.get_external_server(SERVER)
    assert response.status_code == 200 and runtime.state["calls"] == [SERVER]  # nosec B101 -- runtime reconciliation still removes a disabled adapter.
    assert row["last_discovery_at"] is None  # nosec B101 -- disabled rows cannot gain a discovery success stamp.


@pytest.mark.asyncio
async def test_managed_tool_refresh_uses_shared_persistence(discovery_api):
    from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.external_federation_module import (
        ExternalFederationModule,
    )

    runtime = discovery_api
    module = ExternalFederationModule(ModuleConfig(name="external_federation"))
    module._hub_service = runtime.service
    module._manager = SimpleNamespace(reconcile_servers=runtime.refresh)
    await module.execute_tool("external.tools.refresh", {"server_id": SERVER})
    row = await runtime.service.repo.get_external_server(SERVER)
    assert row["last_successful_discovery_at"] is not None  # nosec B101 -- non-HTTP managed refresh publishes the same outcome.


def test_postgres_discovery_state_uses_official_isolation(isolated_test_environment):
    """Exercise native PG timestamps/CAS and idempotent schema ensure on its official DB."""
    client, _ = isolated_test_environment

    async def exercise():
        from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_mcp_hub_tables_pg

        pool = await get_db_pool()
        await ensure_mcp_hub_tables_pg(pool)
        await ensure_mcp_hub_tables_pg(pool)
        repo = McpHubRepo(pool)
        config = {"stdio": {"command": sys.executable, "args": ["-u", str(SMOKE)]}}
        row = await repo.upsert_external_server(
            server_id=SERVER,
            name="PG discovery",
            transport="stdio",
            config_json=json.dumps(config),
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=True,
            actor_id=None,
        )
        revision = row["updated_at"]
        assert await repo.record_external_server_discovery(SERVER, expected_updated_at=revision, failed=False)  # nosec B101 -- PG native timestamp comparison must admit the actual revision.
        success = await repo.get_external_server(SERVER)
        assert await repo.record_external_server_discovery(SERVER, expected_updated_at=revision, failed=True)  # nosec B101 -- status updates retain the configuration revision.
        failed = await repo.get_external_server(SERVER)
        assert (
            failed["last_error_category"] == "discovery_failed"
            and failed["last_successful_discovery_at"] == success["last_successful_discovery_at"]
        )  # nosec B101 -- failure is durable without destroying success history.
        await repo.upsert_external_server(
            server_id=SERVER,
            name="PG discovery",
            transport="stdio",
            config_json=json.dumps({**config, "tags": ["changed"]}),
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=True,
            actor_id=None,
        )
        assert not await repo.record_external_server_discovery(SERVER, expected_updated_at=revision, failed=False)  # nosec B101 -- stale UPDATE 0 must never be mistaken for success.
        fresh = await repo.get_external_server(SERVER)
        assert fresh["last_error_category"] == "config_changed"  # nosec B101 -- upsert invalidates old current readiness in PG too.

    client.portal.call(exercise)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["global_fail", "incomplete"])
async def test_scoped_refresh_cannot_clear_its_just_persisted_publication_failure(discovery_api, failure):
    runtime = discovery_api
    runtime.state[failure] = True
    response = await refresh_route(runtime.client, "scoped")
    assert response.status_code == 200  # nosec B101 -- discovery completed with a reported outcome.
    row = response.json()
    assert (row["display_state"], row["primary_reason_code"]) == ("needs_attention", "discovery_failed")  # nosec B101 -- immediate response must agree with the stored global/incomplete failure.


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_failed", [False, True])
async def test_overlapping_services_cannot_publish_discovery_outcomes_out_of_order(
    discovery_api, monkeypatch, newer_failed
):
    """Hold the older publication while another service refreshes the same config."""
    runtime = discovery_api
    before = await runtime.service.repo.get_external_server(SERVER)
    older_at_publication = asyncio.Event()
    release_older = asyncio.Event()
    record = runtime.service.repo.record_external_server_discovery

    async def held_record(*args, **kwargs):
        older_at_publication.set()
        await release_older.wait()
        return await record(*args, **kwargs)

    def outcome(failed):
        return {
            "ok": not failed,
            "refreshed_servers": 0 if failed else 1,
            "errors": {SERVER: "discarded failure detail"} if failed else {},
        }

    async def older_executor(_server_id):
        return outcome(not newer_failed)

    async def newer_executor(_server_id):
        return outcome(newer_failed)

    monkeypatch.setattr(runtime.service.repo, "record_external_server_discovery", held_record)
    older = asyncio.create_task(runtime.service.refresh_external_server_discovery(SERVER, older_executor))
    newer = None
    try:
        await asyncio.wait_for(older_at_publication.wait(), timeout=5)
        other_service = McpHubService(McpHubRepo(runtime.pool))
        newer = asyncio.create_task(other_service.refresh_external_server_discovery(None, newer_executor))
        # Permit an unprotected newer operation to finish; serialization may keep it pending.
        await asyncio.wait({newer}, timeout=0.2)
        release_older.set()
        await asyncio.wait_for(asyncio.gather(older, newer), timeout=5)
        row = await runtime.service.repo.get_external_server(SERVER)
        assert row["last_error_category"] == (
            "discovery_failed" if newer_failed else None
        )  # nosec B101 -- the newer refresh must determine current readiness.
        assert (
            row["updated_at"] == before["updated_at"]
        )  # nosec B101 -- ordering status cannot alter the config revision.
    finally:
        release_older.set()
        await asyncio.gather(*(task for task in (older, newer) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_discovery_releases_shared_publication_lock(discovery_api):
    """A cancelled executor must not block another service's legitimate refresh."""
    runtime = discovery_api
    entered = asyncio.Event()

    async def held_executor(_server_id):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(runtime.service.refresh_external_server_discovery(SERVER, held_executor))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        other = McpHubService(McpHubRepo(runtime.pool))
        await asyncio.wait_for(other.refresh_external_server_discovery(SERVER, runtime.refresh), timeout=5)
        row = await other.repo.get_external_server(SERVER)
        assert (
            row["last_successful_discovery_at"] is not None
        )  # nosec B101 -- cancellation releases another service's publication path.
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

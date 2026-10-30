"""The app-service container holds the AuthNZ DB pool, and tests can swap it out."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from tldw_Server_API.app import main
from tldw_Server_API.app.core.AuthNZ import session_manager as sm
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.Utils.app_services import app_services
from tldw_Server_API.app.services import lifespan_shutdown_sequence, lifespan_startup_sequence
from tldw_Server_API.tests._plugins.app_services_fixtures import close_app_service, fresh_app_services

pytestmark = pytest.mark.unit


@pytest.fixture
def authnz_sqlite(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    reset_settings()
    yield
    reset_settings()


async def test_get_db_pool_fills_and_reuses_the_container_slot(fresh_services, authnz_sqlite) -> None:
    pool = await get_db_pool()
    assert fresh_services.db_pool is pool
    assert await get_db_pool() is pool


async def test_reset_db_pool_closes_and_empties_the_slot(fresh_services, authnz_sqlite) -> None:
    pool = await get_db_pool()
    await reset_db_pool()
    assert app_services().db_pool is None
    assert pool._initialized is False


async def test_fresh_app_services_empties_slots_then_closes_and_restores(monkeypatch, authnz_sqlite) -> None:
    previous = object()
    monkeypatch.setattr(app_services(), "db_pool", previous)
    async with fresh_app_services() as services:
        assert services is app_services()
        assert services.db_pool is None
        pool = await get_db_pool()
    assert app_services().db_pool is previous
    assert pool._initialized is False


class _StubManager:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.shutdowns = 0

    async def shutdown(self) -> None:
        self.shutdowns += 1
        if self.fail:
            raise RuntimeError("boom")


class _CloseOnly:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


async def test_close_app_service_closes_and_empties_the_slot(fresh_services) -> None:
    stub = _StubManager()
    fresh_services.db_pool = stub
    await close_app_service("db_pool")
    assert stub.shutdowns == 1
    assert fresh_services.db_pool is None


async def test_close_app_service_swallows_a_failing_close(fresh_services, monkeypatch) -> None:
    monkeypatch.setattr(fresh_services, "db_pool", _StubManager(fail=True))
    await close_app_service("db_pool")
    assert fresh_services.db_pool is None


async def test_close_app_service_is_a_no_op_when_the_slot_holds_restore(fresh_services) -> None:
    keep = _StubManager()
    fresh_services.db_pool = keep
    await close_app_service("db_pool", restore=keep)
    assert keep.shutdowns == 0
    assert fresh_services.db_pool is keep


async def test_close_app_service_closes_a_close_only_object(fresh_services) -> None:
    obj = _CloseOnly()
    fresh_services.db_pool = obj
    await close_app_service("db_pool")
    assert obj.closed == 1
    assert fresh_services.db_pool is None


async def test_lifespan_publishes_the_container_and_shuts_down_what_it_holds(monkeypatch, fresh_services) -> None:
    pool = object()
    manager = object()

    async def _fill_slot(**_kwargs):
        app_services().db_pool = pool  # startup fills the slots; shutdown must see them
        app_services().session_manager = manager
        return SimpleNamespace(heavy_startup_handles=None)

    startup = AsyncMock(side_effect=_fill_slot)
    shutdown_spy = AsyncMock()
    monkeypatch.setattr(lifespan_startup_sequence, "run_lifespan_startup_sequence", startup)
    monkeypatch.setattr(lifespan_shutdown_sequence, "run_lifespan_shutdown_sequence", shutdown_spy)
    monkeypatch.setattr(main, "_run_startup_config_validation", lambda: None)
    assert fresh_services.db_pool is None
    assert fresh_services.session_manager is None
    app = FastAPI()
    async with main.lifespan(app):
        assert app.state.services is app_services()
    assert startup.call_args.kwargs["app"] is app
    assert shutdown_spy.call_args.kwargs["db_pool"] is pool
    assert shutdown_spy.call_args.kwargs["session_manager"] is manager


async def test_get_session_manager_fills_the_slot_once_under_concurrency(fresh_services, monkeypatch) -> None:
    calls = 0

    async def slow_init(self):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        self._initialized = True

    monkeypatch.setattr(sm.SessionManager, "initialize", slow_init)
    managers = await asyncio.gather(*(get_session_manager() for _ in range(20)))
    assert len({id(m) for m in managers}) == 1
    assert fresh_services.session_manager is managers[0]
    assert calls == 1


def test_root_autouse_empties_the_session_slot(fresh_services) -> None:
    """Drive the root autouse fixture's generator: setup, a test publishes a manager, teardown."""
    from tldw_Server_API.tests import conftest as root_conftest

    fixture = root_conftest._restore_auth_env_and_singletons
    gen = (getattr(fixture, "__wrapped__", None) or fixture._get_wrapped_function())()
    next(gen)  # setup: the slot is empty here, so teardown restores it to None
    stub = _StubManager()
    fresh_services.session_manager = stub  # published by "the test", without monkeypatch
    fresh_services.jwt_service = object()
    with pytest.raises(StopIteration):
        next(gen)  # teardown
    assert stub.shutdowns == 1
    assert fresh_services.session_manager is None
    assert fresh_services.jwt_service is None


def test_get_jwt_service_fills_the_slot_once_across_threads(fresh_services, monkeypatch) -> None:
    import threading
    import time

    from tldw_Server_API.app.core.AuthNZ import jwt_service as jwt_mod

    built: list[object] = []

    class _SlowJWT:
        def __init__(self) -> None:
            time.sleep(0.05)
            built.append(self)

    monkeypatch.setattr(jwt_mod, "JWTService", _SlowJWT)
    barrier = threading.Barrier(8)
    results: list[object] = []

    def _worker() -> None:
        barrier.wait()
        results.append(jwt_mod.get_jwt_service())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1
    assert len(results) == 8
    assert all(r is fresh_services.jwt_service for r in results)


async def test_heavy_init_returns_the_container_mcp_server(fresh_services, monkeypatch) -> None:
    from tldw_Server_API.app.core.MCP_unified import server as mcp_server_mod
    from tldw_Server_API.app.services import startup_heavy_init

    class _FakeMCPServer:
        initialized = False

        async def initialize(self) -> None:
            self.initialized = True

    monkeypatch.setattr(mcp_server_mod, "MCPServer", _FakeMCPServer)
    handle = await startup_heavy_init._init_mcp_server(SimpleNamespace(state=SimpleNamespace()), deferred=False)
    assert handle is fresh_services.mcp_server
    assert handle.initialized is True


def test_mcp_routes_take_the_server_from_the_container(fresh_services, monkeypatch) -> None:
    import inspect

    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.API_Deps.mcp_deps import get_mcp_server_dep
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as mcp_ep

    class _FakeServer:
        initialized = True

        async def get_status(self) -> dict:
            return {"status": "healthy"}

    monkeypatch.setattr(app_services(), "mcp_server", _FakeServer())
    app = FastAPI()
    app.include_router(mcp_ep.router)
    app.dependency_overrides[mcp_ep.enforce_http_security] = lambda: None
    resp = TestClient(app).get("/mcp/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}
    assert inspect.signature(mcp_ep.health_check).parameters["server"].default.dependency is get_mcp_server_dep

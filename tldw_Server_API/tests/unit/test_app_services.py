"""The app-service container holds the AuthNZ DB pool, and tests can swap it out."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI

from tldw_Server_API.app import main
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.Utils.app_services import app_services
from tldw_Server_API.app.services import lifespan_shutdown_sequence, lifespan_startup_sequence
from tldw_Server_API.tests._plugins.app_services_fixtures import fresh_app_services

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


async def test_lifespan_publishes_the_container_and_shuts_down_what_it_holds(monkeypatch) -> None:
    from types import SimpleNamespace

    # main.py still reads .session_manager from the startup handles until Task 4.
    startup = AsyncMock(return_value=SimpleNamespace(heavy_startup_handles=None, session_manager=None))
    shutdown_spy = AsyncMock()
    monkeypatch.setattr(lifespan_startup_sequence, "run_lifespan_startup_sequence", startup)
    monkeypatch.setattr(lifespan_shutdown_sequence, "run_lifespan_shutdown_sequence", shutdown_spy)
    monkeypatch.setattr(main, "_run_startup_config_validation", lambda: None)
    pool = object()
    monkeypatch.setattr(app_services(), "db_pool", pool)
    app = FastAPI()
    async with main.lifespan(app):
        assert app.state.services is app_services()
    assert startup.call_args.kwargs["app"] is app
    assert shutdown_spy.call_args.kwargs["db_pool"] is pool

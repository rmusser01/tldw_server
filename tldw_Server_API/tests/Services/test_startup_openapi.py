"""Cold OpenAPI work must finish before serving without blocking the startup loop."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI


@pytest.fixture
def lifespan_context(monkeypatch: pytest.MonkeyPatch, fresh_services):
    from tldw_Server_API.app import main
    from tldw_Server_API.app.services import lifespan_shutdown_sequence, lifespan_startup_sequence

    handles = SimpleNamespace(db_pool=object(), session_manager=object(), heavy_startup_handles=object())
    monkeypatch.setattr(fresh_services, "db_pool", handles.db_pool)
    monkeypatch.setattr(fresh_services, "session_manager", handles.session_manager)
    shutdown = AsyncMock()
    monkeypatch.setattr(lifespan_startup_sequence, "run_lifespan_startup_sequence", AsyncMock(return_value=handles))
    monkeypatch.setattr(lifespan_shutdown_sequence, "run_lifespan_shutdown_sequence", shutdown)
    monkeypatch.setattr(main, "_run_startup_config_validation", lambda: None)
    return main, shutdown, handles


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("cached", [False, True])
async def test_startup_openapi_cache_precedes_http_and_keeps_the_loop_responsive(lifespan_context, monkeypatch, cached):
    main, shutdown, _handles = lifespan_context
    app = FastAPI()
    original = app.openapi
    loop = asyncio.get_running_loop()
    owner_thread = threading.get_ident()
    release = threading.Event()
    calls = []
    if cached:
        app.openapi_schema = {"openapi": "3.1.0", "info": {"title": "Retained", "version": "1"}, "paths": {}}

    def build():
        if app.openapi_schema:
            return app.openapi_schema
        loop.call_soon_threadsafe(release.set)
        calls.append((threading.get_ident(), release.wait(timeout=2)))
        return original()

    monkeypatch.setattr(app, "openapi", build)
    try:
        async with main.lifespan(app):
            assert app.openapi_schema is not None  # nosec B101 -- Startup must finish its required cache before serving.
            schema = app.openapi_schema
            assert len(calls) == (0 if cached else 1)  # nosec B101 -- Only a cold cache admits a builder.
            if not cached:
                assert calls[0][1] is True  # nosec B101 -- A loop callback must release the off-thread builder.
                assert calls[0][0] != owner_thread  # nosec B101 -- CPU schema generation must leave the event loop free.
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                for _ in range(2):
                    response = await client.get("/openapi.json")
                    assert response.status_code == 200  # nosec B101 -- Real FastAPI OpenAPI HTTP route remains usable.
                    assert response.json() == schema  # nosec B101 -- Warming preserves the schema response values.
            assert app.openapi_schema is schema  # nosec B101 -- Existing cache object identity is preserved.
            assert len(calls) == (0 if cached else 1)  # nosec B101 -- Warm HTTP reads must not rebuild.
    finally:
        release.set()
    shutdown.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_startup_openapi_failure_aborts_startup_and_tears_down_resources(lifespan_context, monkeypatch):
    main, shutdown, handles = lifespan_context
    app = FastAPI()
    error = RuntimeError("synthetic schema failure")

    def build():
        raise error

    monkeypatch.setattr(app, "openapi", build)
    with pytest.raises(RuntimeError, match="synthetic schema failure") as caught:
        async with main.lifespan(app):
            pytest.fail("Startup admitted requests without the required schema cache")
    assert caught.value is error  # nosec B101 -- Preserve the original generation error.
    shutdown.assert_awaited_once()
    assert shutdown.call_args.kwargs["db_pool"] is handles.db_pool  # nosec B101 -- The same admitted resources reach teardown.


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("cancellations,builder_error", [(1, False), (2, False), (2, True)])
async def test_startup_openapi_cancellation_drains_build_before_teardown(
    lifespan_context,
    monkeypatch,
    cancellations,
    builder_error,
):
    main, shutdown, _handles = lifespan_context
    app = FastAPI()
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    cleanup_finished_states = []

    def build():
        loop.call_soon_threadsafe(started.set)
        try:
            if not release.wait(timeout=2):
                raise RuntimeError("test builder release expired")
            if builder_error:
                raise RuntimeError("synthetic schema failure after cancellation")
            app.openapi_schema = {"openapi": "3.1.0", "paths": {}}
            return app.openapi_schema
        finally:
            finished.set()

    async def cleanup(**_kwargs):
        cleanup_finished_states.append(finished.is_set())

    shutdown.side_effect = cleanup
    monkeypatch.setattr(app, "openapi", build)

    async def serve():
        async with main.lifespan(app):
            pytest.fail("Cancelled startup must never admit requests")

    task = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()  # nosec B101 -- Cancellation must not abandon admitted CPU work.
        assert not cleanup_finished_states  # nosec B101 -- Resource teardown waits for the active builder.
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()  # nosec B101 -- No builder remains after lifespan returns.
        assert cleanup_finished_states == [True]  # nosec B101 -- Exactly one teardown follows builder completion.
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("store_failure", [False, True], ids=["rotated-policy", "store-unavailable"])
async def test_slow_startup_reverifies_provider_policy_before_admitting_requests(
    monkeypatch: pytest.MonkeyPatch, fresh_services: Any, store_failure: bool,
) -> None:
    import time

    from tldw_Server_API.app import main
    from tldw_Server_API.app.core.AuthNZ import database, initialize
    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides as store
    from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.services import (
        lifespan_shutdown_sequence,
        lifespan_startup_sequence,
        startup_auth,
        startup_openapi,
    )

    pool = object()
    monkeypatch.setattr(fresh_services, "db_pool", pool)
    get_pool = AsyncMock(return_value=pool)
    monkeypatch.setattr(database, "get_db_pool", get_pool)
    monkeypatch.setattr(store, "get_db_pool", get_pool)
    monkeypatch.setattr(initialize, "ensure_authnz_schema_ready_once", AsyncMock())
    monkeypatch.setattr(initialize, "ensure_single_user_rbac_seed_if_needed", AsyncMock())
    elapsed = 0.0
    prepared = False
    monotonic = time.monotonic
    monkeypatch.setattr(store, "time", SimpleNamespace(monotonic=lambda: monotonic() + elapsed))

    class Repository:
        def __init__(self, _pool: object) -> None:
            pass

        async def list_overrides(self) -> list[dict[str, object]]:
            if prepared and store_failure:
                raise OSError("synthetic credential store outage")
            return [{"provider": "openai", "is_enabled": not prepared}]

    monkeypatch.setattr(store, "AuthnzLLMProviderOverridesRepo", Repository)

    async def startup(**_kwargs: object) -> SimpleNamespace:
        await startup_auth.init_auth_services()
        return SimpleNamespace(heavy_startup_handles=None)

    warm_openapi = startup_openapi.warm_openapi_schema

    async def prepare(app: FastAPI) -> None:
        nonlocal elapsed, prepared
        await warm_openapi(app)
        # A startup longer than the cache's unchanged hard age must not admit stale policy.
        elapsed += store._OVERRIDE_MAX_STALE_SECONDS + 0.1
        prepared = True

    monkeypatch.setattr(lifespan_startup_sequence, "run_lifespan_startup_sequence", startup)
    monkeypatch.setattr(lifespan_shutdown_sequence, "run_lifespan_shutdown_sequence", AsyncMock())
    monkeypatch.setattr(startup_openapi, "warm_openapi_schema", prepare)
    monkeypatch.setattr(main, "_run_startup_config_validation", lambda: None)
    try:
        async with main.lifespan(FastAPI()):
            if store_failure:
                with pytest.raises(ByokResolutionError) as error:
                    store.get_llm_provider_override("openai")
                assert error.value.code == "credential_store_unavailable"  # nosec B101 -- An actual store outage must still deny credentials.
            else:
                policy = store.get_llm_provider_override("openai")
                assert policy is not None and policy.is_enabled is False  # nosec B101 -- Admission observes the newly verified disabled policy.
    finally:
        await store.shutdown_llm_provider_override_recovery()

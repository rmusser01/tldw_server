from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_heavy_init():
    sys.modules.pop("tldw_Server_API.app.services.startup_heavy_init", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_heavy_init")


@pytest.mark.asyncio
async def test_run_heavy_initializations_runs_steps_in_order_and_updates_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_heavy = _import_startup_heavy_init()
    calls: list[tuple[str, bool]] = []
    info_messages: list[str] = []
    app = SimpleNamespace(state=SimpleNamespace())
    handles = startup_heavy.HeavyStartupHandles()

    async def _record_void(name: str, *, deferred: bool, **_kwargs):
        calls.append((name, deferred))

    async def _record_mcp(_app, *, deferred: bool):
        calls.append(("mcp", deferred))
        return "mcp-server"

    async def _record_provider(*, deferred: bool):
        calls.append(("provider", deferred))
        return "provider-manager"

    async def _record_queue(*, deferred: bool):
        calls.append(("queue", deferred))
        return "request-queue"

    monkeypatch.setattr(
        startup_heavy,
        "_init_local_llm_manager",
        lambda app, route_enabled, *, deferred: _record_void("llm", deferred=deferred, app=app, route_enabled=route_enabled),
    )
    monkeypatch.setattr(startup_heavy, "_init_mcp_server", _record_mcp)
    monkeypatch.setattr(startup_heavy, "_init_provider_manager", _record_provider)
    monkeypatch.setattr(startup_heavy, "_init_request_queue", _record_queue)
    monkeypatch.setattr(
        startup_heavy,
        "_init_rate_limiter",
        lambda *, deferred: _record_void("rate_limiter", deferred=deferred),
    )
    monkeypatch.setattr(
        startup_heavy,
        "_init_tts_service",
        lambda *, deferred: _record_void("tts", deferred=deferred),
    )
    monkeypatch.setattr(
        startup_heavy,
        "_init_chunking_templates",
        lambda *, deferred: _record_void("chunking", deferred=deferred),
    )
    monkeypatch.setattr(
        startup_heavy,
        "_init_embeddings_dim_check",
        lambda *, deferred: _record_void("embeddings_dim", deferred=deferred),
    )
    monkeypatch.setattr(
        startup_heavy.logger,
        "info",
        lambda message, *args, **kwargs: info_messages.append(str(message)),
    )

    await startup_heavy.run_heavy_initializations(
        app,
        handles=handles,
        route_enabled=lambda _key: True,
        deferred=True,
    )

    assert calls == [
        ("llm", True),
        ("mcp", True),
        ("provider", True),
        ("queue", True),
        ("rate_limiter", True),
        ("tts", True),
        ("chunking", True),
        ("embeddings_dim", True),
    ]
    assert handles.mcp_server == "mcp-server"
    assert handles.provider_manager == "provider-manager"
    assert handles.request_queue == "request-queue"
    assert info_messages == [
        "Deferred startup: beginning non-critical initializations in background",
        "Deferred startup: completed non-critical initializations",
    ]


@pytest.mark.asyncio
async def test_start_heavy_initializations_runs_inline_when_not_deferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_heavy = _import_startup_heavy_init()
    app = SimpleNamespace(state=SimpleNamespace(bg_tasks={}))
    observed: list[tuple[bool, object]] = []

    async def _fake_run(_app, *, handles, route_enabled, deferred: bool):
        observed.append((deferred, route_enabled("llm")))
        handles.mcp_server = "inline-mcp"

    monkeypatch.setattr(startup_heavy, "run_heavy_initializations", _fake_run)

    handles = await startup_heavy.start_heavy_initializations(
        app,
        route_enabled=lambda key: key == "llm",
        defer_heavy=False,
    )

    assert observed == [(False, True)]
    assert handles.mcp_server == "inline-mcp"
    assert app.state.bg_tasks == {}


@pytest.mark.asyncio
async def test_start_heavy_initializations_schedules_background_task_when_deferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_heavy = _import_startup_heavy_init()
    app = SimpleNamespace(state=SimpleNamespace(bg_tasks={}))
    observed: list[tuple[bool, object]] = []
    original_create_task = startup_heavy.asyncio.create_task
    created_tasks = []

    async def _fake_run(_app, *, handles, route_enabled, deferred: bool):
        observed.append((deferred, route_enabled("llmacpp")))
        handles.provider_manager = "deferred-provider"

    def _record_create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(startup_heavy, "run_heavy_initializations", _fake_run)
    monkeypatch.setattr(startup_heavy.asyncio, "create_task", _record_create_task)

    handles = await startup_heavy.start_heavy_initializations(
        app,
        route_enabled=lambda key: key == "llmacpp",
        defer_heavy=True,
    )

    assert "deferred_startup" in app.state.bg_tasks
    assert app.state.bg_tasks["deferred_startup"] is created_tasks[0]
    await created_tasks[0]
    assert observed == [(True, True)]
    assert handles.provider_manager == "deferred-provider"


@pytest.mark.asyncio
async def test_init_embeddings_dim_check_strict_mode_reraises_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_heavy = _import_startup_heavy_init()

    class _FakeCollection:
        name = "docs"
        metadata = {"embedding_dimension": 3}

        def get(self, *, limit, include):
            assert limit == 1
            assert include == ["embeddings"]
            return {"embeddings": [[1.0, 2.0]]}

    class _FakeClient:
        def list_collections(self):
            return [_FakeCollection()]

        def get_collection(self, *, name):
            assert name == "docs"
            return _FakeCollection()

    class _FakeChromaDBManager:
        def __init__(self, *, user_id, user_embedding_config):
            assert user_id == "1"
            assert user_embedding_config["AUTH_MODE"] == "single_user"
            self.client = _FakeClient()

        def close(self) -> None:
            return None

    config_module = ModuleType("tldw_Server_API.app.core.config")
    config_module.settings = {"AUTH_MODE": "single_user", "SINGLE_USER_FIXED_ID": "1"}
    chroma_module = ModuleType("tldw_Server_API.app.core.Embeddings.ChromaDB_Library")
    chroma_module.ChromaDBManager = _FakeChromaDBManager
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.config", config_module)
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Embeddings.ChromaDB_Library",
        chroma_module,
    )
    monkeypatch.setenv("EMBEDDINGS_STARTUP_DIM_CHECK_ENABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_DIM_CHECK_STRICT", "true")

    with pytest.raises(RuntimeError, match="EMBEDDINGS_STARTUP_DIM_CHECK_FAILED"):
        await startup_heavy._init_embeddings_dim_check(deferred=False)

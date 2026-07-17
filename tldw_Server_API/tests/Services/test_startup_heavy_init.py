from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit


def _import_startup_heavy_init():
    sys.modules.pop("tldw_Server_API.app.services.startup_heavy_init", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_heavy_init")


@pytest.mark.asyncio
async def test_init_tts_service_uses_canonical_gateway_config_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_heavy = _import_startup_heavy_init()
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.TTS import adapter_registry, tts_config, tts_service_v2, voice_manager
    from tldw_Server_API.app.core.TTS.adapters.openai_compatible_speech_adapter import (
        OpenAICompatibleSpeechAdapter,
    )
    from tldw_Server_API.app.core.TTS.gateway_config import normalize_gateway_specs
    from tldw_Server_API.app.core.TTS.tts_config import TTSConfig, TTSConfigManager

    definition = {
        "enabled": True,
        "base_url": "https://speech.example/v1/",
        "speech_path": "audio/speech",
        "api_key": "admin-key",
        "default_model": "Vendor/Exact",
        "default_voice": "Narrator",
        "allowed_models": ["Vendor/Exact"],
    }
    manager = TTSConfigManager.__new__(TTSConfigManager)
    manager._config = TTSConfig(gateways={"startup": definition})
    manager._gateway_specs = normalize_gateway_specs({}, {"startup": definition})
    manager._sources = {}
    legacy_config = {"adapter_failure_retry_seconds": 9.0}
    config_loader = MagicMock(
        return_value=SimpleNamespace(get_tts_config=lambda: legacy_config)
    )
    circuit_factory = AsyncMock(return_value=MagicMock())
    voice_init = AsyncMock()
    original_get_service = tts_service_v2.get_tts_service_v2
    service_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    services = []

    async def _record_get_service(*args: object, **kwargs: object):
        service_calls.append((args, kwargs))
        service = await original_get_service(*args, **kwargs)
        services.append(service)
        return service

    monkeypatch.setattr(adapter_registry, "_factory_instance", None)
    monkeypatch.setattr(tts_service_v2, "_service_instance", None)
    monkeypatch.setattr(adapter_registry, "get_tts_config_manager", lambda: manager)
    monkeypatch.setattr(tts_config, "get_tts_config_manager", lambda: manager)
    monkeypatch.setattr(core_config, "load_comprehensive_config_with_tts", config_loader)
    monkeypatch.setattr(tts_service_v2, "get_circuit_manager", circuit_factory)
    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", _record_get_service)
    monkeypatch.setattr(voice_manager, "init_voice_manager", voice_init)

    await startup_heavy._init_tts_service(deferred=True)

    assert service_calls == [((), {})]
    assert len(services) == 1
    registry = services[0].factory.registry
    assert registry.config_manager is manager
    assert registry.resolve_provider_key("gateway:startup") == "gateway:startup"
    assert registry._adapter_specs["gateway:startup"] is OpenAICompatibleSpeechAdapter
    config_loader.assert_called_once_with()
    circuit_factory.assert_awaited_once_with(legacy_config)
    voice_init.assert_awaited_once_with()


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
async def test_init_local_llm_manager_runs_llamacpp_runtime_reconciler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_heavy = _import_startup_heavy_init()
    calls: list[str] = []
    app = SimpleNamespace(state=SimpleNamespace())
    supervisor = object()

    class _Manager:
        def __init__(self, _config: object) -> None:
            self.llamacpp_supervisor = supervisor

    class _Reconciler:
        def __init__(self, observed_supervisor: object) -> None:
            assert observed_supervisor is supervisor
            calls.append("created")

        async def reconcile_startup(self) -> list[object]:
            calls.append("reconcile")
            return []

    async def _fake_to_thread(fn: Callable[..., object], *args: object, **kwargs: object) -> object:
        return fn(*args, **kwargs)

    from tldw_Server_API.app.api.v1 import endpoints as endpoints_package
    from tldw_Server_API.app.core import Local_LLM as local_llm_package
    from tldw_Server_API.app.core import config as config_module
    from tldw_Server_API.app.core.Local_LLM import llamacpp_runtime_reconciler

    llamacpp_endpoint_module = ModuleType("tldw_Server_API.app.api.v1.endpoints.llamacpp")
    monkeypatch.setattr(config_module, "get_llamacpp_handler_config", lambda: object())
    monkeypatch.setattr(local_llm_package, "LLMInferenceManager", _Manager, raising=False)
    monkeypatch.setattr(local_llm_package, "LLMManagerConfig", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(llamacpp_runtime_reconciler, "LlamaCppRuntimeReconciler", _Reconciler)
    monkeypatch.setattr(endpoints_package, "llamacpp", llamacpp_endpoint_module, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.api.v1.endpoints.llamacpp",
        llamacpp_endpoint_module,
    )
    monkeypatch.setattr(startup_heavy.asyncio, "to_thread", _fake_to_thread)

    await startup_heavy._init_local_llm_manager(
        app,
        route_enabled=lambda key: key in {"llm", "llamacpp"},
        deferred=False,
    )

    assert calls == ["created", "reconcile"]
    assert app.state.llm_manager.llamacpp_supervisor is supervisor
    assert app.state.llamacpp_runtime_reconciler.__class__ is _Reconciler
    assert llamacpp_endpoint_module.llm_manager is app.state.llm_manager


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

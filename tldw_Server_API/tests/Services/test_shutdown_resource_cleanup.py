from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_resource_cleanup():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_resource_cleanup", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_resource_cleanup")


@pytest.mark.asyncio
async def test_shutdown_resource_cleanup_runs_steps_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_resources = _import_shutdown_resource_cleanup()
    calls: list[str] = []
    app = SimpleNamespace(state=SimpleNamespace(llm_manager=None))

    async def _record_session_manager(*, session_manager, guard_exceptions):
        assert session_manager == "session-manager"
        assert guard_exceptions == (RuntimeError,)
        calls.append("session")

    async def _record_mcp_server(*, heavy_startup_handles, guard_exceptions):
        assert heavy_startup_handles == "heavy-handles"
        assert guard_exceptions == (RuntimeError,)
        calls.append("mcp-server")

    async def _record_mcp_rate_limiter(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("mcp-rate-limiter")

    async def _record_tts_service(*, in_pytest_for_tts_shutdown, guard_exceptions):
        assert in_pytest_for_tts_shutdown is False
        assert guard_exceptions == (RuntimeError,)
        calls.append("tts-service")

    async def _record_tts_resource_manager(*, in_pytest_for_tts_shutdown, guard_exceptions):
        assert in_pytest_for_tts_shutdown is False
        assert guard_exceptions == (RuntimeError,)
        calls.append("tts-resource-manager")

    async def _record_http_client(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("http-client")

    async def _record_chacha(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("chacha")

    async def _record_prompts(*, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("prompts")

    async def _record_chat_workflows(*, app, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("chat-workflows")

    async def _record_provider_manager(*, heavy_startup_handles, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("provider-manager")

    async def _record_request_queue(*, heavy_startup_handles, guard_exceptions):
        assert guard_exceptions == (RuntimeError,)
        calls.append("request-queue")

    async def _record_local_llm_manager(*, app, guard_exceptions, run_in_thread):
        assert guard_exceptions == (RuntimeError,)
        assert run_in_thread == "thread-helper"
        calls.append("local-llm")

    monkeypatch.setattr(shutdown_resources, "_shutdown_session_manager", _record_session_manager)
    monkeypatch.setattr(shutdown_resources, "_shutdown_mcp_server", _record_mcp_server)
    monkeypatch.setattr(shutdown_resources, "_shutdown_mcp_rate_limiter", _record_mcp_rate_limiter)
    monkeypatch.setattr(shutdown_resources, "_shutdown_tts_service", _record_tts_service)
    monkeypatch.setattr(shutdown_resources, "_shutdown_tts_resource_manager", _record_tts_resource_manager)
    monkeypatch.setattr(shutdown_resources, "_shutdown_http_client", _record_http_client)
    monkeypatch.setattr(shutdown_resources, "_shutdown_chacha_resources", _record_chacha)
    monkeypatch.setattr(shutdown_resources, "_shutdown_prompts_resources", _record_prompts)
    monkeypatch.setattr(shutdown_resources, "_shutdown_chat_workflows_resources", _record_chat_workflows)
    monkeypatch.setattr(shutdown_resources, "_shutdown_provider_manager", _record_provider_manager)
    monkeypatch.setattr(shutdown_resources, "_shutdown_request_queue", _record_request_queue)
    monkeypatch.setattr(shutdown_resources, "_shutdown_local_llm_manager", _record_local_llm_manager)

    await shutdown_resources.shutdown_resource_cleanup(
        app=app,
        session_manager="session-manager",
        heavy_startup_handles="heavy-handles",
        in_pytest_for_tts_shutdown=False,
        import_exceptions=(LookupError,),
        startup_guard_exceptions=(RuntimeError,),
        run_in_thread="thread-helper",
    )

    assert calls == [
        "session",
        "mcp-server",
        "mcp-rate-limiter",
        "tts-service",
        "tts-resource-manager",
        "http-client",
        "chacha",
        "prompts",
        "chat-workflows",
        "provider-manager",
        "request-queue",
        "local-llm",
    ]


@pytest.mark.asyncio
async def test_shutdown_tts_service_skips_in_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_resources = _import_shutdown_resource_cleanup()
    called = False

    async def _fake_shutdown_tts_components():
        nonlocal called
        called = True

    monkeypatch.setattr(
        shutdown_resources,
        "_shutdown_tts_service_components",
        _fake_shutdown_tts_components,
    )

    await shutdown_resources._shutdown_tts_service(
        in_pytest_for_tts_shutdown=True,
        guard_exceptions=(RuntimeError,),
    )

    assert called is False


@pytest.mark.asyncio
async def test_shutdown_tts_resource_manager_skips_in_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_resources = _import_shutdown_resource_cleanup()
    called = False

    async def _fake_shutdown_tts_resource_manager_components():
        nonlocal called
        called = True

    monkeypatch.setattr(
        shutdown_resources,
        "_shutdown_tts_resource_manager_components",
        _fake_shutdown_tts_resource_manager_components,
    )

    await shutdown_resources._shutdown_tts_resource_manager(
        in_pytest_for_tts_shutdown=True,
        guard_exceptions=(RuntimeError,),
    )

    assert called is False


@pytest.mark.asyncio
async def test_shutdown_provider_manager_stops_health_checks_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_resources = _import_shutdown_resource_cleanup()
    calls: list[str] = []

    class _ProviderManager:
        async def stop_health_checks(self) -> None:
            calls.append("stop-health-checks")

    heavy_handles = SimpleNamespace(provider_manager=_ProviderManager())

    await shutdown_resources._shutdown_provider_manager(
        heavy_startup_handles=heavy_handles,
        guard_exceptions=(RuntimeError,),
    )

    assert calls == ["stop-health-checks"]


@pytest.mark.asyncio
async def test_shutdown_mcp_server_handles_runtime_guard_exception() -> None:
    shutdown_resources = _import_shutdown_resource_cleanup()

    class _MCPServer:
        async def shutdown(self) -> None:
            raise RuntimeError("mcp boom")

    await shutdown_resources._shutdown_mcp_server(
        heavy_startup_handles=SimpleNamespace(mcp_server=_MCPServer()),
        guard_exceptions=(RuntimeError,),
    )


@pytest.mark.asyncio
async def test_shutdown_request_queue_handles_runtime_guard_exception() -> None:
    shutdown_resources = _import_shutdown_resource_cleanup()

    class _RequestQueue:
        async def stop(self) -> None:
            raise RuntimeError("queue boom")

    await shutdown_resources._shutdown_request_queue(
        heavy_startup_handles=SimpleNamespace(request_queue=_RequestQueue()),
        guard_exceptions=(RuntimeError,),
    )


@pytest.mark.asyncio
async def test_shutdown_local_llm_manager_uses_thread_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_resources = _import_shutdown_resource_cleanup()
    calls: list[str] = []

    class _LocalLLMManager:
        def cleanup_on_exit(self) -> None:
            calls.append("cleanup")

    async def _fake_run_in_thread(fn):
        calls.append("to-thread")
        fn()

    app = SimpleNamespace(state=SimpleNamespace(llm_manager=_LocalLLMManager()))

    await shutdown_resources._shutdown_local_llm_manager(
        app=app,
        guard_exceptions=(RuntimeError,),
        run_in_thread=_fake_run_in_thread,
    )

    assert calls == ["to-thread", "cleanup"]


@pytest.mark.asyncio
async def test_shutdown_local_llm_manager_stops_llamacpp_reconciler_before_sync_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_resources = _import_shutdown_resource_cleanup()
    calls: list[str] = []

    class _Reconciler:
        async def shutdown(self) -> None:
            calls.append("reconciler")

    class _LocalLLMManager:
        def cleanup_on_exit(self) -> None:
            calls.append("cleanup")

    async def _fake_run_in_thread(fn):
        calls.append("to-thread")
        fn()

    app = SimpleNamespace(
        state=SimpleNamespace(
            llamacpp_runtime_reconciler=_Reconciler(),
            llm_manager=_LocalLLMManager(),
        )
    )

    await shutdown_resources._shutdown_local_llm_manager(
        app=app,
        guard_exceptions=(RuntimeError,),
        run_in_thread=_fake_run_in_thread,
    )

    assert calls == ["reconciler", "to-thread", "cleanup"]
    assert app.state.llamacpp_runtime_reconciler is None

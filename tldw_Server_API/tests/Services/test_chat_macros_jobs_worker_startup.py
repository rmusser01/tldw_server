from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from typing import Any

import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)

pytestmark = pytest.mark.unit


def _import_worker_module():
    try:
        return importlib.import_module("tldw_Server_API.app.services.chat_macros_jobs_worker")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Chat macro jobs worker module is missing: {exc}")
    except ImportError as exc:
        pytest.fail(f"Chat macro jobs worker imports are not usable: {exc}")


def _import_startup_content_jobs_pollers():
    sys.modules.pop("tldw_Server_API.app.services.startup_content_jobs_pollers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_content_jobs_pollers")


def _context(route_enabled: Callable[..., bool] | None = None) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app="app",
        settings={},
        test_mode=True,
        route_enabled=route_enabled or (lambda *_args, **_kwargs: True),
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


@pytest.mark.asyncio
async def test_run_chat_macros_jobs_worker_uses_domain_queue_and_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_mod = _import_worker_module()
    captured: dict[str, Any] = {}

    class _FakeSDK:
        def __init__(self, job_manager: Any, config: Any) -> None:
            captured["job_manager"] = job_manager
            captured["config"] = config

        async def run(self, **kwargs: Any) -> None:
            captured["run_kwargs"] = kwargs

        def stop(self) -> None:
            captured["stopped"] = True

    monkeypatch.setenv("CHAT_MACROS_JOBS_QUEUE", "high")
    monkeypatch.setenv("CHAT_MACROS_JOBS_WORKER_ID", "macro-worker-test")
    monkeypatch.setattr(worker_mod, "JobManager", lambda: "job-manager")
    monkeypatch.setattr(worker_mod, "WorkerSDK", _FakeSDK)

    await worker_mod.run_chat_macros_jobs_worker()

    config = captured["config"]
    assert captured["job_manager"] == "job-manager"
    assert config.domain == "chat_macros"
    assert config.queue == "high"
    assert config.worker_id == "macro-worker-test"
    assert captured["run_kwargs"]["handler"] is worker_mod.handle_chat_macro_job
    assert callable(captured["run_kwargs"]["cancel_check"])


@pytest.mark.asyncio
async def test_run_chat_macros_jobs_worker_defaults_invalid_numeric_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_mod = _import_worker_module()
    captured: dict[str, Any] = {}

    class _FakeSDK:
        def __init__(self, _job_manager: Any, config: Any) -> None:
            captured["config"] = config

        async def run(self, **_kwargs: Any) -> None:
            return None

        def stop(self) -> None:
            return None

    monkeypatch.setenv("CHAT_MACROS_JOBS_LEASE_SECONDS", "invalid")
    monkeypatch.setenv("CHAT_MACROS_JOBS_RENEW_THRESHOLD_SECONDS", "-1")
    monkeypatch.setenv("CHAT_MACROS_JOBS_RENEW_JITTER_SECONDS", "invalid")
    monkeypatch.setattr(worker_mod, "JobManager", lambda: "job-manager")
    monkeypatch.setattr(worker_mod, "WorkerSDK", _FakeSDK)

    await worker_mod.run_chat_macros_jobs_worker()

    config = captured["config"]
    assert config.lease_seconds == 120
    assert config.renew_threshold_seconds == 10
    assert config.renew_jitter_seconds == 0


def test_chat_macros_content_worker_spec_is_registered_and_delegates_to_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    calls: list[object] = []
    monkeypatch.setattr(
        startup_pollers,
        "_run_chat_macros_jobs_worker_service",
        lambda stop_event: calls.append(stop_event) or "chat-macros-awaitable",
    )
    specs = {spec.name: spec for spec in startup_pollers.provide_content_jobs_worker_specs()}

    spec = specs["chat_macros_jobs_task"]

    assert spec.task_name == "chat_macros_jobs_task"
    assert spec.category == "jobs"
    assert spec.phase is ShutdownPhase.JOB_POLLER_QUIESCE
    assert spec.timeout_sec == 5.0
    assert spec.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert spec.factory is not None
    assert spec.factory(_context(), "chat-stop") == "chat-macros-awaitable"
    assert calls == ["chat-stop"]


def test_chat_macros_content_worker_spec_uses_expected_flag_and_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setenv("CHAT_MACROS_JOBS_WORKER_ENABLED", "true")

    def _route_enabled(*args: object, **kwargs: object) -> bool:
        calls.append((args, kwargs))
        return False

    specs = {spec.name: spec for spec in startup_pollers.provide_content_jobs_worker_specs()}

    assert specs["chat_macros_jobs_task"].enabled(_context(_route_enabled)) is False
    assert calls == [(("chat-macros",), {})]


@pytest.mark.asyncio
async def test_start_chat_macros_jobs_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "chat-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "chat-task",
    )
    monkeypatch.setattr(
        startup_pollers,
        "_run_chat_macros_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "chat-coro",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        registrations.append(
            {
                "app": app,
                "owned_job_pollers": owned_job_pollers,
                "name": name,
                "task": task,
                "stop_event": stop_event,
            }
        )

    owned_job_pollers: list[object] = []
    stop_event, task = await startup_pollers._start_chat_macros_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "CHAT_MACROS_JOBS_WORKER_ENABLED",
            "chat-macros",
            {},
        ),
    )

    assert stop_event == "chat-stop"
    assert task == "chat-task"
    assert captured_stop_events == ["chat-stop"]
    assert created_coroutines == ["chat-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "chat_macros_jobs_task",
            "task": "chat-task",
            "stop_event": "chat-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_chat_macros_jobs_worker_cancels_task_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_content_jobs_pollers()

    class _FakeTask:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()
    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "chat-stop")
    monkeypatch.setattr(startup_pollers, "_create_task", lambda _coro: task)
    monkeypatch.setattr(
        startup_pollers,
        "_run_chat_macros_jobs_worker_service",
        lambda _stop_event: "chat-coro",
    )

    def fail_registration(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("ownership registration failed")

    result = await startup_pollers._start_chat_macros_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=fail_registration,
        should_start_worker=lambda *_args, **_kwargs: True,
    )

    assert result == (None, None)
    assert task.cancelled is True

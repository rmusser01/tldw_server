from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)

pytestmark = pytest.mark.unit


def _import_startup_notifications_abtest_workers():
    sys.modules.pop("tldw_Server_API.app.services.startup_notifications_abtest_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_notifications_abtest_workers")


def _context(
    *,
    settings: dict[str, object] | None = None,
    sidecar_mode: bool = False,
) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app="app",
        settings=settings or {},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
        sidecar_mode=sidecar_mode,
    )


def _specs_by_name(startup_workers: Any) -> dict[str, Any]:
    return {
        spec.name: spec
        for spec in startup_workers.provide_notifications_abtest_worker_specs()
    }


def test_notifications_abtest_worker_specs_match_legacy_worker_contract() -> None:
    startup_workers = _import_startup_notifications_abtest_workers()

    specs = _specs_by_name(startup_workers)

    bridge = specs["jobs_notifications_bridge_task"]
    assert bridge.task_name == "jobs_notifications_bridge_task"
    assert bridge.category == "jobs"
    assert bridge.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert bridge.timeout_sec == 5.0
    assert bridge.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert bridge.factory is not None
    assert bridge.shutdown_callback_factory is None

    abtest = specs["evals_abtest_jobs_task"]
    assert abtest.task_name == "evals_abtest_jobs_task"
    assert abtest.category == "jobs"
    assert abtest.phase is ShutdownPhase.JOB_POLLER_QUIESCE
    assert abtest.timeout_sec == 5.0
    assert abtest.strategy is WorkerStrategy.STOP_EVENT_TASK
    assert abtest.factory is not None


def test_notifications_abtest_worker_specs_use_expected_names() -> None:
    startup_workers = _import_startup_notifications_abtest_workers()

    assert [spec.name for spec in startup_workers.provide_notifications_abtest_worker_specs()] == [
        "jobs_notifications_bridge_task",
        "evals_abtest_jobs_task",
    ]


def test_evals_abtest_worker_spec_factory_delegates_to_existing_worker_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    calls: list[object] = []
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_abtest_jobs_worker_service",
        lambda stop_event: calls.append(stop_event) or "abtest-awaitable",
    )

    spec = _specs_by_name(startup_workers)["evals_abtest_jobs_task"]

    assert spec.factory is not None
    assert spec.factory(_context(), "abtest-stop") == "abtest-awaitable"
    assert calls == ["abtest-stop"]


@pytest.mark.asyncio
async def test_jobs_notifications_bridge_worker_spec_factory_starts_and_cancels_legacy_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    bridge_started = asyncio.Event()
    cancelled: list[str] = []

    async def _bridge_loop() -> None:
        bridge_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append("bridge")
            raise

    monkeypatch.setattr(
        startup_workers,
        "_start_jobs_notifications_service",
        lambda: asyncio.create_task(_bridge_loop(), name="jobs_notifications_bridge"),
    )

    spec = _specs_by_name(startup_workers)["jobs_notifications_bridge_task"]
    stop_event = asyncio.Event()
    assert spec.factory is not None
    lifecycle_task = asyncio.create_task(spec.factory(_context(), stop_event))

    await asyncio.wait_for(bridge_started.wait(), timeout=1)
    assert lifecycle_task.done() is False

    stop_event.set()
    await asyncio.wait_for(lifecycle_task, timeout=1)

    assert cancelled == ["bridge"]


def test_notifications_abtest_worker_specs_disable_in_sidecar_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    monkeypatch.setattr(startup_workers, "_jobs_notifications_bridge_enabled", lambda: True)
    monkeypatch.setattr(startup_workers, "_evals_abtest_jobs_worker_enabled", lambda: True)

    specs = _specs_by_name(startup_workers)

    assert specs["jobs_notifications_bridge_task"].enabled(_context(sidecar_mode=True)) is False
    assert specs["evals_abtest_jobs_task"].enabled(_context(sidecar_mode=True)) is False


@pytest.mark.asyncio
async def test_start_notifications_abtest_workers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    calls: list[str] = []
    worker_inventory = object()

    async def _record_notifications(**kwargs: object) -> str:
        assert kwargs["worker_inventory"] is worker_inventory
        calls.append("notifications")
        return "bridge-task"

    async def _record_abtest(**kwargs: object) -> tuple[str, str]:
        assert kwargs["worker_inventory"] is worker_inventory
        calls.append("abtest")
        return ("abtest-stop", "abtest-task")

    monkeypatch.setattr(startup_workers, "_start_jobs_notifications_bridge_worker", _record_notifications)
    monkeypatch.setattr(startup_workers, "_start_evals_abtest_jobs_worker", _record_abtest)

    handles = await startup_workers.start_notifications_abtest_workers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=False,
        worker_inventory=worker_inventory,
    )

    assert calls == ["notifications", "abtest"]
    assert handles.jobs_notifications_bridge_task == "bridge-task"
    assert handles.evals_abtest_jobs_stop_event == "abtest-stop"
    assert handles.evals_abtest_jobs_task == "abtest-task"


@pytest.mark.asyncio
async def test_start_jobs_notifications_bridge_worker_skips_in_sidecar_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    monkeypatch.setattr(
        startup_workers,
        "_start_jobs_notifications_service",
        lambda: (_ for _ in ()).throw(AssertionError("no start")),
    )

    task = await startup_workers._start_jobs_notifications_bridge_worker(sidecar_mode=True)

    assert task is None


@pytest.mark.asyncio
async def test_start_jobs_notifications_bridge_worker_returns_task_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    monkeypatch.setattr(
        startup_workers,
        "_start_jobs_notifications_service",
        lambda: "bridge-task",
    )

    task = await startup_workers._start_jobs_notifications_bridge_worker(sidecar_mode=False)

    assert task == "bridge-task"


@pytest.mark.asyncio
async def test_start_jobs_notifications_bridge_worker_registers_with_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    registered: list[object] = []
    worker_inventory = SimpleNamespace(register=lambda worker: registered.append(worker))

    monkeypatch.setattr(
        startup_workers,
        "_start_jobs_notifications_service",
        lambda: "bridge-task",
    )

    task = await startup_workers._start_jobs_notifications_bridge_worker(
        sidecar_mode=False,
        worker_inventory=worker_inventory,
    )

    assert task == "bridge-task"
    [worker] = registered
    assert worker.name == "jobs_notifications_bridge_task"
    assert worker.task == "bridge-task"
    assert worker.stop_event is None
    assert worker.category == "jobs"
    assert worker.shutdown_phase == startup_workers.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN


@pytest.mark.asyncio
async def test_start_jobs_notifications_bridge_worker_cancels_task_when_inventory_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()

    def _failing_register(worker: object) -> None:
        del worker
        raise RuntimeError("registration boom")

    monkeypatch.setattr(
        startup_workers,
        "_start_jobs_notifications_service",
        lambda: task,
    )

    returned_task = await startup_workers._start_jobs_notifications_bridge_worker(
        sidecar_mode=False,
        worker_inventory=SimpleNamespace(register=_failing_register),
    )

    assert returned_task is None
    assert task.cancelled is True


def test_safe_cancel_task_tolerates_cancelled_error_without_current_task_cancelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()

    class _FakeTask:
        def cancel(self) -> None:
            raise asyncio.CancelledError()

    monkeypatch.setattr(startup_workers.asyncio, "current_task", lambda: object())

    startup_workers._safe_cancel_task(_FakeTask())


def test_safe_cancel_task_preserves_current_task_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()

    class _FakeTask:
        def cancel(self) -> None:
            raise asyncio.CancelledError()

    class _CancellingCurrentTask:
        def cancelling(self) -> int:
            return 1

    monkeypatch.setattr(startup_workers.asyncio, "current_task", lambda: _CancellingCurrentTask())

    with pytest.raises(asyncio.CancelledError):
        startup_workers._safe_cancel_task(_FakeTask())


@pytest.mark.asyncio
async def test_start_evals_abtest_jobs_worker_registers_owned_poller_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registrations: list[dict[str, object]] = []
    env_reads = iter(["true"])

    monkeypatch.setattr(startup_workers, "_make_event", lambda: "abtest-stop")
    monkeypatch.setattr(
        startup_workers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "abtest-task",
    )
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_abtest_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "abtest-coro",
    )
    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: next(env_reads) if key == "EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED" else default,
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
    stop_event, task = await startup_workers._start_evals_abtest_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        sidecar_mode=False,
    )

    assert stop_event == "abtest-stop"
    assert task == "abtest-task"
    assert captured_stop_events == ["abtest-stop"]
    assert created_coroutines == ["abtest-coro"]
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "evals_abtest_jobs_task",
            "task": "abtest-task",
            "stop_event": "abtest-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_evals_abtest_jobs_worker_registers_with_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []
    registered: list[object] = []

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED" else default,
    )
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "abtest-stop")
    monkeypatch.setattr(
        startup_workers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "abtest-task",
    )
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_abtest_jobs_worker_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "abtest-coro",
    )

    def _legacy_register(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("legacy poller registration should not run with worker_inventory")

    stop_event, task = await startup_workers._start_evals_abtest_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_legacy_register,
        sidecar_mode=False,
        worker_inventory=SimpleNamespace(register=lambda worker: registered.append(worker)),
    )

    assert stop_event == "abtest-stop"
    assert task == "abtest-task"
    assert captured_stop_events == ["abtest-stop"]
    assert created_coroutines == ["abtest-coro"]
    [worker] = registered
    assert worker.name == "evals_abtest_jobs_task"
    assert worker.task == "abtest-task"
    assert worker.stop_event == "abtest-stop"
    assert worker.timeout_sec == 5.0
    assert worker.category == "jobs"
    assert worker.shutdown_phase == startup_workers.ShutdownPhase.JOB_POLLER_QUIESCE


@pytest.mark.asyncio
async def test_start_evals_abtest_jobs_worker_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED" else default,
    )
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "abtest-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_workers, "_create_task", _failing_create_task)
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_abtest_jobs_worker_service",
        lambda stop_event: stop_event,
    )

    stop_event, task = await startup_workers._start_evals_abtest_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=False,
    )

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_evals_abtest_jobs_worker_cancels_task_when_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_notifications_abtest_workers()

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED" else default,
    )
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "abtest-stop")
    monkeypatch.setattr(startup_workers, "_create_task", lambda coro: task)
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_abtest_jobs_worker_service",
        lambda stop_event: stop_event,
    )

    def _failing_register(*args, **kwargs):
        raise RuntimeError("registration boom")

    stop_event, returned_task = await startup_workers._start_evals_abtest_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_failing_register,
        sidecar_mode=False,
    )

    assert stop_event is None
    assert returned_task is None
    assert task.cancelled is True

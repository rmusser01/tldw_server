from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


def _import_startup_runtime_monitors():
    sys.modules.pop("tldw_Server_API.app.services.startup_runtime_monitors", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_runtime_monitors")


@pytest.mark.asyncio
async def test_start_runtime_monitors_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_monitors = _import_startup_runtime_monitors()
    calls: list[str] = []

    async def _record_jobs_metrics():
        calls.append("jobs-metrics")
        return ("jobs-metrics-stop", "jobs-metrics-task")

    async def _record_loop_lag(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append("loop-lag")
        return ("loop-lag-stop", "loop-lag-task")

    monkeypatch.setattr(startup_monitors, "_start_jobs_metrics_gauge_worker", _record_jobs_metrics)
    monkeypatch.setattr(startup_monitors, "_start_loop_lag_watchdog", _record_loop_lag)

    handles = await startup_monitors.start_runtime_monitors()

    assert calls == ["jobs-metrics", "loop-lag"]
    assert handles.jobs_metrics_stop_event == "jobs-metrics-stop"
    assert handles.jobs_metrics_task == "jobs-metrics-task"
    assert handles.loop_lag_stop_event == "loop-lag-stop"
    assert handles.loop_lag_task == "loop-lag-task"


@pytest.mark.asyncio
async def test_start_runtime_monitors_passes_inventory_to_registered_monitors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_monitors = _import_startup_runtime_monitors()
    worker_inventory = object()
    calls: list[str] = []

    async def _record_jobs_metrics(*, worker_inventory: object) -> tuple[object, str]:
        calls.append("jobs-metrics")
        return (worker_inventory, "jobs-metrics-task")

    async def _record_loop_lag(*, worker_inventory: object) -> tuple[object, str]:
        calls.append("loop-lag")
        return (worker_inventory, "loop-lag-task")

    monkeypatch.setattr(startup_monitors, "_start_jobs_metrics_gauge_worker", _record_jobs_metrics)
    monkeypatch.setattr(startup_monitors, "_start_loop_lag_watchdog", _record_loop_lag)

    handles = await startup_monitors.start_runtime_monitors(worker_inventory=worker_inventory)

    assert calls == ["jobs-metrics", "loop-lag"]
    assert handles.jobs_metrics_stop_event is worker_inventory
    assert handles.loop_lag_stop_event is worker_inventory


@pytest.mark.asyncio
async def test_start_jobs_metrics_gauge_worker_starts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_monitors = _import_startup_runtime_monitors()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []

    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "true")
    monkeypatch.setattr(startup_monitors, "_make_event", lambda: "jobs-metrics-stop")
    monkeypatch.setattr(
        startup_monitors,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "jobs-metrics-task",
    )
    monkeypatch.setattr(
        startup_monitors,
        "_run_jobs_metrics_gauges_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "jobs-metrics-coro",
    )

    stop_event, task = await startup_monitors._start_jobs_metrics_gauge_worker()

    assert stop_event == "jobs-metrics-stop"
    assert task == "jobs-metrics-task"
    assert captured_stop_events == ["jobs-metrics-stop"]
    assert created_coroutines == ["jobs-metrics-coro"]


@pytest.mark.asyncio
async def test_start_loop_lag_watchdog_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_monitors = _import_startup_runtime_monitors()

    monkeypatch.delenv("EVENT_LOOP_LAG_WATCHDOG_ENABLED", raising=False)
    monkeypatch.setattr(startup_monitors, "_make_event", lambda: (_ for _ in ()).throw(AssertionError("no event")))
    monkeypatch.setattr(startup_monitors, "_create_task", lambda coro: (_ for _ in ()).throw(AssertionError("no task")))

    stop_event, task = await startup_monitors._start_loop_lag_watchdog()

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_loop_lag_watchdog_registers_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_monitors = _import_startup_runtime_monitors()
    registrations: list[dict[str, object]] = []

    class _FakeWorkerInventory:
        async def register_custom(self, **kwargs: object) -> tuple[str, str]:
            registrations.append(kwargs)
            return "loop-lag-task", "loop-lag-stop"

    monkeypatch.setenv("EVENT_LOOP_LAG_WATCHDOG_ENABLED", "true")
    monkeypatch.setattr(
        startup_monitors,
        "_make_event",
        lambda: (_ for _ in ()).throw(AssertionError("legacy event path should not run")),
    )
    monkeypatch.setattr(
        startup_monitors,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("legacy task path should not run")),
    )

    stop_event, task = await startup_monitors._start_loop_lag_watchdog(
        worker_inventory=_FakeWorkerInventory(),
    )

    assert stop_event == "loop-lag-stop"
    assert task == "loop-lag-task"
    assert registrations == [
        {
            "name": "loop_lag_task",
            "task_name": "loop_lag_watchdog",
            "coroutine_factory": startup_monitors._run_loop_lag_watchdog_service,
            "timeout_sec": 2.0,
            "category": "monitoring",
            "shutdown_phase": startup_monitors.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        }
    ]


@pytest.mark.asyncio
async def test_start_loop_lag_watchdog_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_monitors = _import_startup_runtime_monitors()

    monkeypatch.setenv("EVENT_LOOP_LAG_WATCHDOG_ENABLED", "true")
    monkeypatch.setattr(startup_monitors, "_make_event", lambda: "loop-lag-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_monitors, "_create_task", _failing_create_task)
    monkeypatch.setattr(startup_monitors, "_run_loop_lag_watchdog_service", lambda stop_event: stop_event)

    stop_event, task = await startup_monitors._start_loop_lag_watchdog()

    assert stop_event is None
    assert task is None

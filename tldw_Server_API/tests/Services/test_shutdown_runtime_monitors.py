from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_runtime_monitors():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_runtime_monitors", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_runtime_monitors")


class _FakeStopEvent:
    def __init__(self) -> None:
        self.is_set = False

    def set(self) -> None:
        self.is_set = True


@pytest.mark.asyncio
async def test_shutdown_runtime_monitors_stops_jobs_metrics_then_loop_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_runtime = _import_shutdown_runtime_monitors()
    waits: list[tuple[object, float]] = []
    jobs_metrics_stop_event = _FakeStopEvent()
    loop_lag_stop_event = _FakeStopEvent()

    async def _fake_wait(task, *, timeout):
        waits.append((task, timeout))

    monkeypatch.setattr(shutdown_runtime, "_wait_for_task", _fake_wait)

    handles = await shutdown_runtime.shutdown_runtime_monitors(
        jobs_metrics_task="jobs-metrics-task",
        jobs_metrics_stop_event=jobs_metrics_stop_event,
        loop_lag_task="loop-lag-task",
        loop_lag_stop_event=loop_lag_stop_event,
        guard_exceptions=(RuntimeError,),
    )

    assert waits == [("jobs-metrics-task", 5.0), ("loop-lag-task", 2.0)]
    assert jobs_metrics_stop_event.is_set is True
    assert loop_lag_stop_event.is_set is True
    assert handles.jobs_metrics_task == "jobs-metrics-task"
    assert handles.loop_lag_task == "loop-lag-task"


@pytest.mark.asyncio
async def test_shutdown_runtime_monitors_cancels_jobs_metrics_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_runtime = _import_shutdown_runtime_monitors()

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    async def _failing_wait(_task, *, timeout):
        del timeout
        raise RuntimeError("boom")

    task = _FakeTask()
    stop_event = _FakeStopEvent()
    monkeypatch.setattr(shutdown_runtime, "_wait_for_task", _failing_wait)

    await shutdown_runtime.shutdown_runtime_monitors(
        jobs_metrics_task=task,
        jobs_metrics_stop_event=stop_event,
        loop_lag_task=None,
        loop_lag_stop_event=None,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_runtime_monitors_logs_debug_when_loop_lag_cancel_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_runtime = _import_shutdown_runtime_monitors()
    debug_messages: list[str] = []

    class _FakeTask:
        def cancel(self) -> None:
            raise RuntimeError("cancel boom")

    async def _failing_wait(_task, *, timeout):
        del timeout
        raise RuntimeError("wait boom")

    class _FakeLogger:
        def debug(self, message: str) -> None:
            debug_messages.append(message)

        def info(self, message: str) -> None:
            del message

    monkeypatch.setattr(shutdown_runtime, "_wait_for_task", _failing_wait)
    monkeypatch.setattr(shutdown_runtime, "logger", _FakeLogger())

    await shutdown_runtime.shutdown_runtime_monitors(
        jobs_metrics_task=None,
        jobs_metrics_stop_event=None,
        loop_lag_task=_FakeTask(),
        loop_lag_stop_event=_FakeStopEvent(),
        guard_exceptions=(RuntimeError,),
    )

    assert debug_messages == ["Event loop lag watchdog cancel failed: cancel boom"]

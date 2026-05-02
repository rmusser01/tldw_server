from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_jobs_metrics_reconcile():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_jobs_metrics_reconcile", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_jobs_metrics_reconcile")


class _FakeStopEvent:
    def __init__(self) -> None:
        self.is_set = False

    def set(self) -> None:
        self.is_set = True


@pytest.mark.asyncio
async def test_shutdown_jobs_metrics_reconcile_stops_via_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_reconcile = _import_shutdown_jobs_metrics_reconcile()
    waits: list[tuple[object, float]] = []
    stop_event = _FakeStopEvent()

    async def _fake_wait(task, *, timeout):
        waits.append((task, timeout))

    monkeypatch.setattr(shutdown_reconcile, "_wait_for_task", _fake_wait)

    handles = await shutdown_reconcile.shutdown_jobs_metrics_reconcile(
        jobs_metrics_reconcile_task="jobs-metrics-reconcile-task",
        jobs_metrics_reconcile_stop=stop_event,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert waits == [("jobs-metrics-reconcile-task", 5.0)]
    assert handles.jobs_metrics_reconcile_task == "jobs-metrics-reconcile-task"
    assert handles.jobs_metrics_reconcile_stop is stop_event


@pytest.mark.asyncio
async def test_shutdown_jobs_metrics_reconcile_cancels_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_reconcile = _import_shutdown_jobs_metrics_reconcile()

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
    monkeypatch.setattr(shutdown_reconcile, "_wait_for_task", _failing_wait)

    await shutdown_reconcile.shutdown_jobs_metrics_reconcile(
        jobs_metrics_reconcile_task=task,
        jobs_metrics_reconcile_stop=stop_event,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_jobs_metrics_reconcile_skips_missing_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_reconcile = _import_shutdown_jobs_metrics_reconcile()
    called = False

    async def _fake_wait(_task, *, timeout):
        nonlocal called
        del timeout
        called = True

    monkeypatch.setattr(shutdown_reconcile, "_wait_for_task", _fake_wait)

    handles = await shutdown_reconcile.shutdown_jobs_metrics_reconcile(
        jobs_metrics_reconcile_task=None,
        jobs_metrics_reconcile_stop=None,
        guard_exceptions=(RuntimeError,),
    )

    assert called is False
    assert handles.jobs_metrics_reconcile_task is None
    assert handles.jobs_metrics_reconcile_stop is None

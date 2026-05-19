from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_privilege_snapshot_worker():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_privilege_snapshot_worker", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_privilege_snapshot_worker")


class _FakeStopEvent:
    def __init__(self) -> None:
        self.is_set = False

    def set(self) -> None:
        self.is_set = True


class _FakeTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


@pytest.mark.asyncio
async def test_shutdown_privilege_snapshot_worker_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_worker = _import_shutdown_privilege_snapshot_worker()
    calls: list[dict[str, object]] = []
    should_run_late_stop = lambda *args, **kwargs: True

    async def _record_shutdown(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(shutdown_worker, "_shutdown_privilege_snapshot_worker", _record_shutdown)

    handles = await shutdown_worker.shutdown_privilege_snapshot_worker(
        privilege_snapshot_task="privilege-task",
        privilege_snapshot_stop_event="privilege-stop",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=(RuntimeError,),
    )

    assert len(calls) == 1
    assert calls[0]["task"] == "privilege-task"
    assert calls[0]["stop_event"] == "privilege-stop"
    assert calls[0]["should_run_late_stop"] is should_run_late_stop
    assert calls[0]["guard_exceptions"] == (RuntimeError,)
    assert handles.privilege_snapshot_task == "privilege-task"
    assert handles.privilege_snapshot_stop_event == "privilege-stop"


@pytest.mark.asyncio
async def test_shutdown_privilege_snapshot_worker_stops_via_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_worker = _import_shutdown_privilege_snapshot_worker()
    waits: list[tuple[object, float]] = []
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_worker, "_wait_for_task", _fake_wait)

    await shutdown_worker._shutdown_privilege_snapshot_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("privilege_snapshot_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert waits == [(task, 5.0)]
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_privilege_snapshot_worker_skips_when_late_stop_says_false() -> None:
    shutdown_worker = _import_shutdown_privilege_snapshot_worker()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    await shutdown_worker._shutdown_privilege_snapshot_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: False,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is False
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_privilege_snapshot_worker_cancels_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_worker = _import_shutdown_privilege_snapshot_worker()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _failing_wait(_task, *, timeout):
        del timeout
        raise RuntimeError("boom")

    monkeypatch.setattr(shutdown_worker, "_wait_for_task", _failing_wait)

    await shutdown_worker._shutdown_privilege_snapshot_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("privilege_snapshot_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_privilege_snapshot_worker_cancels_without_stop_event() -> None:
    shutdown_worker = _import_shutdown_privilege_snapshot_worker()
    task = _FakeTask()

    await shutdown_worker._shutdown_privilege_snapshot_worker(
        task=task,
        stop_event=None,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("privilege_snapshot_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is True

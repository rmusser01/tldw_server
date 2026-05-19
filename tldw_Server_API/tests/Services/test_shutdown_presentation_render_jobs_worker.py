from __future__ import annotations

import asyncio
import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_presentation_render_jobs_worker():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_presentation_render_jobs_worker", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_presentation_render_jobs_worker")


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
async def test_shutdown_presentation_render_jobs_worker_returns_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_worker = _import_shutdown_presentation_render_jobs_worker()
    calls: list[dict[str, object]] = []
    should_run_late_stop = lambda *args, **kwargs: True

    async def _record_shutdown(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(shutdown_worker, "_shutdown_presentation_render_jobs_worker", _record_shutdown)

    handles = await shutdown_worker.shutdown_presentation_render_jobs_worker(
        presentation_render_jobs_task="presentation-task",
        presentation_render_jobs_stop_event="presentation-stop",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=(RuntimeError,),
    )

    assert len(calls) == 1
    assert calls[0]["task"] == "presentation-task"
    assert calls[0]["stop_event"] == "presentation-stop"
    assert calls[0]["should_run_late_stop"] is should_run_late_stop
    assert calls[0]["guard_exceptions"] == (RuntimeError,)
    assert handles.presentation_render_jobs_task == "presentation-task"
    assert handles.presentation_render_jobs_stop_event == "presentation-stop"


@pytest.mark.asyncio
async def test_shutdown_presentation_render_jobs_worker_stops_via_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_worker = _import_shutdown_presentation_render_jobs_worker()
    waits: list[tuple[object, float]] = []
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_worker, "_wait_for_task", _fake_wait)

    await shutdown_worker._shutdown_presentation_render_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("presentation_render_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert waits == [(task, 5.0)]
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_presentation_render_jobs_worker_reraises_cancelled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_worker = _import_shutdown_presentation_render_jobs_worker()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _cancelled_wait(_task, *, timeout):
        del timeout
        raise asyncio.CancelledError()

    monkeypatch.setattr(shutdown_worker, "_wait_for_task", _cancelled_wait)

    with pytest.raises(asyncio.CancelledError):
        await shutdown_worker._shutdown_presentation_render_jobs_worker(
            task=task,
            stop_event=stop_event,
            should_run_late_stop=lambda name, current_task: (
                name,
                current_task,
            ) == ("presentation_render_jobs_task", task),
            guard_exceptions=(RuntimeError,),
        )

    assert stop_event.is_set is True
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_presentation_render_jobs_worker_cancels_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_worker = _import_shutdown_presentation_render_jobs_worker()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _failing_wait(_task, *, timeout):
        del timeout
        raise RuntimeError("boom")

    monkeypatch.setattr(shutdown_worker, "_wait_for_task", _failing_wait)

    await shutdown_worker._shutdown_presentation_render_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("presentation_render_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_presentation_render_jobs_worker_logs_and_cancels_on_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_worker = _import_shutdown_presentation_render_jobs_worker()
    task = _FakeTask()
    stop_event = _FakeStopEvent()
    warnings: list[str] = []

    async def _unexpected_wait(_task, *, timeout):
        del timeout
        raise Exception("boom")

    monkeypatch.setattr(shutdown_worker, "_wait_for_task", _unexpected_wait)
    monkeypatch.setattr(shutdown_worker.logger, "warning", warnings.append)

    await shutdown_worker._shutdown_presentation_render_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("presentation_render_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True
    assert warnings == ["Presentation Render Jobs worker exited with exception before shutdown completion: boom"]


@pytest.mark.asyncio
async def test_shutdown_presentation_render_jobs_worker_cancels_without_stop_event() -> None:
    shutdown_worker = _import_shutdown_presentation_render_jobs_worker()
    task = _FakeTask()

    await shutdown_worker._shutdown_presentation_render_jobs_worker(
        task=task,
        stop_event=None,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("presentation_render_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is True

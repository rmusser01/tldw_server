from __future__ import annotations

import asyncio
import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_media_ingest_jobs_workers():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_media_ingest_jobs_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_media_ingest_jobs_workers")


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
async def test_shutdown_media_ingest_jobs_workers_runs_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_media_ingest_jobs_workers()
    calls: list[str] = []

    async def _record_media(**kwargs):
        del kwargs
        calls.append("media")

    async def _record_media_heavy(**kwargs):
        del kwargs
        calls.append("media-heavy")

    monkeypatch.setattr(shutdown_workers, "_shutdown_media_ingest_jobs_worker", _record_media)
    monkeypatch.setattr(shutdown_workers, "_shutdown_media_ingest_heavy_jobs_worker", _record_media_heavy)

    handles = await shutdown_workers.shutdown_media_ingest_jobs_workers(
        media_ingest_jobs_task="media-task",
        media_ingest_jobs_stop_event="media-stop",
        media_ingest_heavy_jobs_task="media-heavy-task",
        media_ingest_heavy_jobs_stop_event="media-heavy-stop",
        should_run_late_stop=lambda *args, **kwargs: True,
        guard_exceptions=(RuntimeError,),
    )

    assert calls == ["media", "media-heavy"]
    assert handles.media_ingest_jobs_task == "media-task"
    assert handles.media_ingest_jobs_stop_event == "media-stop"
    assert handles.media_ingest_heavy_jobs_task == "media-heavy-task"
    assert handles.media_ingest_heavy_jobs_stop_event == "media-heavy-stop"


@pytest.mark.asyncio
async def test_shutdown_media_ingest_jobs_worker_stops_via_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_media_ingest_jobs_workers()
    waits: list[tuple[object, float]] = []
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _fake_wait)

    await shutdown_workers._shutdown_media_ingest_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (name, current_task) == ("media_ingest_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert waits == [(task, 5.0)]
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_media_ingest_heavy_jobs_worker_skips_when_late_stop_says_false() -> None:
    shutdown_workers = _import_shutdown_media_ingest_jobs_workers()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    await shutdown_workers._shutdown_media_ingest_heavy_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: False,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is False
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_media_ingest_heavy_jobs_worker_cancels_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_media_ingest_jobs_workers()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _failing_wait(_task, *, timeout):
        del timeout
        raise RuntimeError("boom")

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _failing_wait)

    await shutdown_workers._shutdown_media_ingest_heavy_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("media_ingest_heavy_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_media_ingest_jobs_worker_cancels_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_media_ingest_jobs_workers()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _timeout_wait(_task, *, timeout):
        del timeout
        raise asyncio.TimeoutError()

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _timeout_wait)

    await shutdown_workers._shutdown_media_ingest_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("media_ingest_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True

from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_notifications_compactor_websub_workers():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_notifications_compactor_websub_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_notifications_compactor_websub_workers")


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
async def test_shutdown_notifications_compactor_websub_workers_runs_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_notifications_compactor_websub_workers()
    calls: list[str] = []

    async def _record_notifications(**kwargs):
        del kwargs
        calls.append("notifications")

    async def _record_compactor(**kwargs):
        del kwargs
        calls.append("compactor")

    async def _record_websub(**kwargs):
        del kwargs
        calls.append("websub")

    monkeypatch.setattr(shutdown_workers, "_shutdown_jobs_notifications_bridge_worker", _record_notifications)
    monkeypatch.setattr(shutdown_workers, "_shutdown_embeddings_vector_compactor_worker", _record_compactor)
    monkeypatch.setattr(shutdown_workers, "_shutdown_websub_renewal_worker", _record_websub)

    handles = await shutdown_workers.shutdown_notifications_compactor_websub_workers(
        jobs_notifications_bridge_task="bridge-task",
        embeddings_compactor_task="compactor-task",
        embeddings_compactor_stop_event="compactor-stop",
        websub_renewal_task="websub-task",
        guard_exceptions=(RuntimeError,),
    )

    assert calls == ["notifications", "compactor", "websub"]
    assert handles.jobs_notifications_bridge_task == "bridge-task"
    assert handles.embeddings_compactor_task == "compactor-task"
    assert handles.embeddings_compactor_stop_event == "compactor-stop"
    assert handles.websub_renewal_task == "websub-task"


@pytest.mark.asyncio
async def test_shutdown_jobs_notifications_bridge_worker_cancels_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_notifications_compactor_websub_workers()
    waits: list[tuple[object, float]] = []
    task = _FakeTask()

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _fake_wait)

    await shutdown_workers._shutdown_jobs_notifications_bridge_worker(
        task=task,
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is True
    assert waits == [(task, 5.0)]


@pytest.mark.asyncio
async def test_shutdown_embeddings_vector_compactor_worker_stops_via_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_notifications_compactor_websub_workers()
    waits: list[tuple[object, float]] = []
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _fake_wait)

    await shutdown_workers._shutdown_embeddings_vector_compactor_worker(
        task=task,
        stop_event=stop_event,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert waits == [(task, 5.0)]
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_embeddings_vector_compactor_worker_cancels_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_notifications_compactor_websub_workers()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _failing_wait(_task, *, timeout):
        del timeout
        raise RuntimeError("boom")

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _failing_wait)

    await shutdown_workers._shutdown_embeddings_vector_compactor_worker(
        task=task,
        stop_event=stop_event,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_websub_renewal_worker_cancels_task() -> None:
    shutdown_workers = _import_shutdown_notifications_compactor_websub_workers()
    task = _FakeTask()

    await shutdown_workers._shutdown_websub_renewal_worker(task=task)

    assert task.cancelled is True

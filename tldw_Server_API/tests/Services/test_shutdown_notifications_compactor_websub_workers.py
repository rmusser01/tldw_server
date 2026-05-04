from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import fields

import pytest

pytestmark = pytest.mark.unit


def _import_shutdown_notifications_compactor_websub_workers():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_notifications_compactor_websub_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_notifications_compactor_websub_workers")


class _FakeTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def test_shutdown_notifications_compactor_websub_workers_no_longer_owns_registry_worker_stops() -> None:
    shutdown_workers = _import_shutdown_notifications_compactor_websub_workers()

    obsolete_fields = {
        "embeddings_compactor_task",
        "embeddings_compactor_stop_event",
        "websub_renewal_task",
    }
    assert obsolete_fields.isdisjoint(
        inspect.signature(
            shutdown_workers.shutdown_notifications_compactor_websub_workers
        ).parameters
    )
    assert obsolete_fields.isdisjoint(
        {field.name for field in fields(shutdown_workers.NotificationsCompactorWebsubShutdownHandles)}
    )
    assert not hasattr(shutdown_workers, "_shutdown_embeddings_vector_compactor_worker")
    assert not hasattr(shutdown_workers, "_shutdown_websub_renewal_worker")


@pytest.mark.asyncio
async def test_shutdown_notifications_compactor_websub_workers_runs_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_notifications_compactor_websub_workers()
    calls: list[str] = []

    async def _record_notifications(**kwargs):
        del kwargs
        calls.append("notifications")

    monkeypatch.setattr(shutdown_workers, "_shutdown_jobs_notifications_bridge_worker", _record_notifications)

    handles = await shutdown_workers.shutdown_notifications_compactor_websub_workers(
        jobs_notifications_bridge_task="bridge-task",
        guard_exceptions=(RuntimeError,),
    )

    assert calls == ["notifications"]
    assert handles.jobs_notifications_bridge_task == "bridge-task"


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

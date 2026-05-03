from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import fields

import pytest

pytestmark = pytest.mark.unit


def _import_shutdown_claims_maintenance_tasks():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_claims_maintenance_tasks", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_claims_maintenance_tasks")


class _FakeTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def test_shutdown_claims_maintenance_tasks_no_longer_owns_claims_rebuild_stop() -> None:
    shutdown_tasks = _import_shutdown_claims_maintenance_tasks()

    assert "claims_task" not in inspect.signature(
        shutdown_tasks.shutdown_claims_maintenance_tasks
    ).parameters
    assert "claims_task" not in {
        field.name for field in fields(shutdown_tasks.ClaimsMaintenanceShutdownHandles)
    }
    assert not hasattr(shutdown_tasks, "_shutdown_claims_task")


@pytest.mark.asyncio
async def test_shutdown_claims_maintenance_tasks_runs_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_tasks = _import_shutdown_claims_maintenance_tasks()
    calls: list[str] = []

    async def _record_jobs_prune(**kwargs):
        del kwargs
        calls.append("jobs-prune")

    async def _record_files_gc(**kwargs):
        del kwargs
        calls.append("files-gc")

    async def _record_notifications_prune(**kwargs):
        del kwargs
        calls.append("notifications-prune")

    monkeypatch.setattr(shutdown_tasks, "_shutdown_jobs_prune_task", _record_jobs_prune)
    monkeypatch.setattr(shutdown_tasks, "_shutdown_files_export_gc_task", _record_files_gc)
    monkeypatch.setattr(shutdown_tasks, "_shutdown_notifications_prune_task", _record_notifications_prune)

    handles = await shutdown_tasks.shutdown_claims_maintenance_tasks(
        jobs_prune_task="jobs-prune-task",
        files_export_gc_task="files-gc-task",
        notifications_prune_task="notifications-prune-task",
    )

    assert calls == ["jobs-prune", "files-gc", "notifications-prune"]
    assert handles.jobs_prune_task == "jobs-prune-task"
    assert handles.files_export_gc_task == "files-gc-task"
    assert handles.notifications_prune_task == "notifications-prune-task"


@pytest.mark.asyncio
async def test_shutdown_notifications_prune_task_skips_none() -> None:
    shutdown_tasks = _import_shutdown_claims_maintenance_tasks()

    await shutdown_tasks._shutdown_notifications_prune_task(task=None)


@pytest.mark.asyncio
async def test_shutdown_files_export_gc_task_cancels_task() -> None:
    shutdown_tasks = _import_shutdown_claims_maintenance_tasks()
    task = _FakeTask()

    await shutdown_tasks._shutdown_files_export_gc_task(task=task)

    assert task.cancelled is True

from __future__ import annotations

import importlib
import sys

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


class _FakeStopEvent:
    def __init__(self) -> None:
        self.is_set = False

    def set(self) -> None:
        self.is_set = True


@pytest.mark.asyncio
async def test_shutdown_claims_maintenance_tasks_runs_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_tasks = _import_shutdown_claims_maintenance_tasks()
    calls: list[str] = []

    async def _record_claims(**kwargs):
        del kwargs
        calls.append("claims")

    async def _record_jobs_prune(**kwargs):
        del kwargs
        calls.append("jobs-prune")

    async def _record_files_gc(**kwargs):
        del kwargs
        calls.append("files-gc")

    async def _record_notifications_prune(**kwargs):
        del kwargs
        calls.append("notifications-prune")

    monkeypatch.setattr(shutdown_tasks, "_shutdown_claims_task", _record_claims)
    monkeypatch.setattr(shutdown_tasks, "_shutdown_jobs_prune_task", _record_jobs_prune)
    monkeypatch.setattr(shutdown_tasks, "_shutdown_files_export_gc_task", _record_files_gc)
    monkeypatch.setattr(shutdown_tasks, "_shutdown_notifications_prune_task", _record_notifications_prune)

    handles = await shutdown_tasks.shutdown_claims_maintenance_tasks(
        claims_task="claims-task",
        jobs_prune_task="jobs-prune-task",
        files_export_gc_task="files-gc-task",
        notifications_prune_task="notifications-prune-task",
    )

    assert calls == ["claims", "jobs-prune", "files-gc", "notifications-prune"]
    assert handles.claims_task == "claims-task"
    assert handles.jobs_prune_task == "jobs-prune-task"
    assert handles.files_export_gc_task == "files-gc-task"
    assert handles.notifications_prune_task == "notifications-prune-task"


@pytest.mark.asyncio
async def test_shutdown_claims_task_cancels_task() -> None:
    shutdown_tasks = _import_shutdown_claims_maintenance_tasks()
    task = _FakeTask()

    await shutdown_tasks._shutdown_claims_task(task=task)

    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_claims_task_sets_attached_stop_event_before_cancel() -> None:
    shutdown_tasks = _import_shutdown_claims_maintenance_tasks()
    task = _FakeTask()
    stop_event = _FakeStopEvent()
    task._tldw_claims_rebuild_stop_event = stop_event

    await shutdown_tasks._shutdown_claims_task(task=task)

    assert stop_event.is_set is True
    assert task.cancelled is True


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

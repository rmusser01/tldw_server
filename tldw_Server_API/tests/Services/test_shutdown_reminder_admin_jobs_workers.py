from __future__ import annotations

import asyncio
import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_reminder_admin_jobs_workers():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_reminder_admin_jobs_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_reminder_admin_jobs_workers")


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
async def test_shutdown_reminder_admin_jobs_workers_runs_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_reminder_admin_jobs_workers()
    calls: list[str] = []

    async def _record_reminder(**kwargs):
        del kwargs
        calls.append("reminder")

    async def _record_admin_backup(**kwargs):
        del kwargs
        calls.append("admin-backup")

    async def _record_admin_maintenance(**kwargs):
        del kwargs
        calls.append("admin-maintenance")

    monkeypatch.setattr(shutdown_workers, "_shutdown_reminder_jobs_worker", _record_reminder)
    monkeypatch.setattr(shutdown_workers, "_shutdown_admin_backup_jobs_worker", _record_admin_backup)
    monkeypatch.setattr(shutdown_workers, "_shutdown_admin_maintenance_rotation_jobs_worker", _record_admin_maintenance)

    handles = await shutdown_workers.shutdown_reminder_admin_jobs_workers(
        reminder_jobs_task="reminder-task",
        admin_backup_jobs_task="admin-backup-task",
        admin_maintenance_rotation_jobs_task="admin-maintenance-task",
        admin_maintenance_rotation_jobs_stop_event="admin-maintenance-stop",
        should_run_late_stop=lambda *args, **kwargs: True,
        guard_exceptions=(RuntimeError,),
    )

    assert calls == ["reminder", "admin-backup", "admin-maintenance"]
    assert handles.reminder_jobs_task == "reminder-task"
    assert handles.admin_backup_jobs_task == "admin-backup-task"
    assert handles.admin_maintenance_rotation_jobs_task == "admin-maintenance-task"
    assert handles.admin_maintenance_rotation_jobs_stop_event == "admin-maintenance-stop"


@pytest.mark.asyncio
async def test_shutdown_reminder_jobs_worker_cancels_and_waits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_reminder_admin_jobs_workers()
    waits: list[tuple[object, float]] = []
    task = _FakeTask()

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _fake_wait)

    await shutdown_workers._shutdown_reminder_jobs_worker(
        task=task,
        should_run_late_stop=lambda name, current_task: (name, current_task) == ("reminder_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is True
    assert waits == [(task, 5.0)]


@pytest.mark.asyncio
async def test_shutdown_admin_backup_jobs_worker_skips_when_late_stop_says_false() -> None:
    shutdown_workers = _import_shutdown_reminder_admin_jobs_workers()
    task = _FakeTask()

    await shutdown_workers._shutdown_admin_backup_jobs_worker(
        task=task,
        should_run_late_stop=lambda name, current_task: False,
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_admin_maintenance_rotation_jobs_worker_stops_via_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_reminder_admin_jobs_workers()
    waits: list[tuple[object, float]] = []
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _fake_wait)

    await shutdown_workers._shutdown_admin_maintenance_rotation_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("admin_maintenance_rotation_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert waits == [(task, 5.0)]
    assert task.cancelled is False


@pytest.mark.asyncio
async def test_shutdown_admin_maintenance_rotation_jobs_worker_cancels_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_reminder_admin_jobs_workers()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _timeout_wait(_task, *, timeout):
        del timeout
        raise asyncio.TimeoutError()

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _timeout_wait)

    await shutdown_workers._shutdown_admin_maintenance_rotation_jobs_worker(
        task=task,
        stop_event=stop_event,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("admin_maintenance_rotation_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_admin_maintenance_rotation_jobs_worker_waits_after_cancel_without_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_workers = _import_shutdown_reminder_admin_jobs_workers()
    task = _FakeTask()
    waits: list[tuple[object, float]] = []

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_workers, "_wait_for_task", _fake_wait)

    await shutdown_workers._shutdown_admin_maintenance_rotation_jobs_worker(
        task=task,
        stop_event=None,
        should_run_late_stop=lambda name, current_task: (
            name,
            current_task,
        ) == ("admin_maintenance_rotation_jobs_task", task),
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is True
    assert waits == [(task, 5.0)]

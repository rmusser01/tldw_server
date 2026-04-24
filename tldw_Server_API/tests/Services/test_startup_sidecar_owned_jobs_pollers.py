from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_sidecar_owned_jobs_pollers():
    sys.modules.pop("tldw_Server_API.app.services.startup_sidecar_owned_jobs_pollers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_sidecar_owned_jobs_pollers")


@pytest.mark.asyncio
async def test_start_sidecar_owned_jobs_pollers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_sidecar_owned_jobs_pollers()
    calls: list[str] = []

    async def _record_reminder(**kwargs):
        del kwargs
        calls.append("reminder")
        return ("reminder-stop", "reminder-task")

    async def _record_admin_backup(**kwargs):
        del kwargs
        calls.append("admin-backup")
        return ("admin-backup-stop", "admin-backup-task")

    async def _record_admin_byok(**kwargs):
        del kwargs
        calls.append("admin-byok")
        return ("admin-byok-stop", "admin-byok-task")

    async def _record_admin_maintenance(**kwargs):
        del kwargs
        calls.append("admin-maintenance")
        return ("admin-maintenance-stop", "admin-maintenance-task")

    async def _record_recipe(**kwargs):
        del kwargs
        calls.append("recipe")
        return ("recipe-stop", "recipe-task")

    monkeypatch.setattr(startup_pollers, "_start_reminder_jobs_worker", _record_reminder)
    monkeypatch.setattr(startup_pollers, "_start_admin_backup_jobs_worker", _record_admin_backup)
    monkeypatch.setattr(startup_pollers, "_start_admin_byok_validation_jobs_worker", _record_admin_byok)
    monkeypatch.setattr(
        startup_pollers,
        "_start_admin_maintenance_rotation_jobs_worker",
        _record_admin_maintenance,
    )
    monkeypatch.setattr(startup_pollers, "_start_recipe_run_jobs_worker", _record_recipe)

    handles = await startup_pollers.start_sidecar_owned_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=False,
    )

    assert calls == ["reminder", "admin-backup", "admin-byok", "admin-maintenance", "recipe"]
    assert handles.reminder_jobs_stop_event == "reminder-stop"
    assert handles.reminder_jobs_task == "reminder-task"
    assert handles.admin_backup_jobs_stop_event == "admin-backup-stop"
    assert handles.admin_backup_jobs_task == "admin-backup-task"
    assert handles.admin_byok_validation_jobs_stop_event == "admin-byok-stop"
    assert handles.admin_byok_validation_jobs_task == "admin-byok-task"
    assert handles.admin_maintenance_rotation_jobs_stop_event == "admin-maintenance-stop"
    assert handles.admin_maintenance_rotation_jobs_task == "admin-maintenance-task"
    assert handles.recipe_run_jobs_stop_event == "recipe-stop"
    assert handles.recipe_run_jobs_task == "recipe-task"


@pytest.mark.asyncio
async def test_start_reminder_jobs_worker_skips_in_sidecar_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_sidecar_owned_jobs_pollers()

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: (_ for _ in ()).throw(AssertionError("no event")))

    stop_event, task = await startup_pollers._start_reminder_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=True,
    )

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_admin_backup_jobs_worker_registers_owned_poller_when_task_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_sidecar_owned_jobs_pollers()
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "admin-backup-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_start_admin_backup_jobs_worker_service",
        lambda *, stop_event: "admin-backup-task",
    )

    def _register_owned_job_poller(app, owned_job_pollers, *, name, task, stop_event):
        registrations.append(
            {
                "app": app,
                "owned_job_pollers": owned_job_pollers,
                "name": name,
                "task": task,
                "stop_event": stop_event,
            }
        )

    owned_job_pollers: list[object] = []
    stop_event, task = await startup_pollers._start_admin_backup_jobs_worker(
        app="app",
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=_register_owned_job_poller,
        sidecar_mode=False,
    )

    assert stop_event == "admin-backup-stop"
    assert task == "admin-backup-task"
    assert registrations == [
        {
            "app": "app",
            "owned_job_pollers": owned_job_pollers,
            "name": "admin_backup_jobs_task",
            "task": "admin-backup-task",
            "stop_event": "admin-backup-stop",
        }
    ]


@pytest.mark.asyncio
async def test_start_admin_maintenance_rotation_jobs_worker_skips_registration_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_sidecar_owned_jobs_pollers()
    registrations: list[dict[str, object]] = []

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "admin-maintenance-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_start_admin_maintenance_rotation_jobs_worker_service",
        lambda *, stop_event: None,
    )

    stop_event, task = await startup_pollers._start_admin_maintenance_rotation_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: registrations.append(kwargs),
        sidecar_mode=False,
    )

    assert stop_event == "admin-maintenance-stop"
    assert task is None
    assert registrations == []


@pytest.mark.asyncio
async def test_start_recipe_run_jobs_worker_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_sidecar_owned_jobs_pollers()

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "recipe-stop")

    def _failing_start(*, stop_event):
        del stop_event
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_pollers, "_start_recipe_run_jobs_worker_service", _failing_start)

    stop_event, task = await startup_pollers._start_recipe_run_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=False,
    )

    assert stop_event is None
    assert task is None

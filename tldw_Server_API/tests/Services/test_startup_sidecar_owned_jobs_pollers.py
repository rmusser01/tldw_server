from __future__ import annotations

import importlib
import sys
from collections.abc import Callable

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

    async def _record_reminder(**kwargs: object) -> tuple[str, str]:
        """Record that the reminder worker starter ran."""

        del kwargs
        calls.append("reminder")
        return ("reminder-stop", "reminder-task")

    async def _record_admin_backup(**kwargs: object) -> tuple[str, str]:
        """Record that the admin backup worker starter ran."""

        del kwargs
        calls.append("admin-backup")
        return ("admin-backup-stop", "admin-backup-task")

    async def _record_admin_byok(**kwargs: object) -> tuple[str, str]:
        """Record that the admin BYOK worker starter ran."""

        del kwargs
        calls.append("admin-byok")
        return ("admin-byok-stop", "admin-byok-task")

    async def _record_admin_maintenance(**kwargs: object) -> tuple[str, str]:
        """Record that the admin maintenance worker starter ran."""

        del kwargs
        calls.append("admin-maintenance")
        return ("admin-maintenance-stop", "admin-maintenance-task")

    async def _record_recipe(**kwargs: object) -> tuple[str, str]:
        """Record that the recipe-run worker starter ran."""

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
async def test_start_sidecar_owned_jobs_pollers_passes_inventory_to_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_sidecar_owned_jobs_pollers()
    worker_inventory = object()
    captured_kwargs_by_worker: dict[str, dict[str, object]] = {}

    def _record_worker(label: str) -> Callable[..., object]:
        """Build a starter stub that captures kwargs for one worker label."""

        async def _record(**kwargs: object) -> tuple[str, str]:
            """Capture worker startup kwargs and return deterministic handles."""

            captured_kwargs_by_worker[label] = kwargs
            return (f"{label}-stop", f"{label}-task")

        return _record

    monkeypatch.setattr(startup_pollers, "_start_reminder_jobs_worker", _record_worker("reminder"))
    monkeypatch.setattr(startup_pollers, "_start_admin_backup_jobs_worker", _record_worker("admin-backup"))
    monkeypatch.setattr(startup_pollers, "_start_admin_byok_validation_jobs_worker", _record_worker("admin-byok"))
    monkeypatch.setattr(
        startup_pollers,
        "_start_admin_maintenance_rotation_jobs_worker",
        _record_worker("admin-maintenance"),
    )
    monkeypatch.setattr(startup_pollers, "_start_recipe_run_jobs_worker", _record_worker("recipe"))

    await startup_pollers.start_sidecar_owned_jobs_pollers(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=False,
        worker_inventory=worker_inventory,
    )

    assert {
        worker: kwargs["worker_inventory"]
        for worker, kwargs in captured_kwargs_by_worker.items()
    } == {
        "reminder": worker_inventory,
        "admin-backup": worker_inventory,
        "admin-byok": worker_inventory,
        "admin-maintenance": worker_inventory,
        "recipe": worker_inventory,
    }


@pytest.mark.asyncio
async def test_sidecar_owned_jobs_worker_cancels_task_when_inventory_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_sidecar_owned_jobs_pollers()
    cancelled: list[bool] = []

    class _FakeTask:
        """Started task double that records cancellation requests."""

        def cancel(self) -> None:
            """Record that rollback requested task cancellation."""

            cancelled.append(True)

    class _FailingWorkerInventory:
        """Lifecycle inventory double that fails worker registration."""

        def register(self, worker: object) -> None:
            """Simulate a guarded inventory registration failure."""

            del worker
            raise RuntimeError("boom")

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "reminder-stop")
    monkeypatch.setattr(
        startup_pollers,
        "_start_reminder_jobs_worker_service",
        lambda *, stop_event: _FakeTask(),
    )

    def _register_owned_job_poller(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy poller registration should not run")

    stop_event, task = await startup_pollers._start_reminder_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=_register_owned_job_poller,
        sidecar_mode=False,
        worker_inventory=_FailingWorkerInventory(),
    )

    assert cancelled == [True]
    assert stop_event is None
    assert task is None


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


@pytest.mark.asyncio
async def test_start_recipe_run_jobs_worker_handles_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_pollers = _import_startup_sidecar_owned_jobs_pollers()

    monkeypatch.setattr(startup_pollers, "_make_event", lambda: "recipe-stop")

    def _failing_start(*, stop_event):
        del stop_event
        raise ImportError("missing recipe worker")

    monkeypatch.setattr(startup_pollers, "_start_recipe_run_jobs_worker_service", _failing_start)

    stop_event, task = await startup_pollers._start_recipe_run_jobs_worker(
        app="app",
        owned_job_pollers=[],
        register_owned_job_poller=lambda *args, **kwargs: None,
        sidecar_mode=False,
    )

    assert stop_event is None
    assert task is None

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


def _import_startup_maintenance_schedulers():
    sys.modules.pop("tldw_Server_API.app.services.startup_maintenance_schedulers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_maintenance_schedulers")


@pytest.mark.asyncio
async def test_start_maintenance_schedulers_combines_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    calls: list[str] = []

    async def _fake_quality():
        calls.append("quality")
        return "quality-task"

    async def _fake_outputs():
        calls.append("outputs")
        return "outputs-task"

    async def _fake_kanban_activity(*, worker_inventory=None):
        assert worker_inventory is None
        calls.append("kanban-activity")
        return "kanban-activity-task"

    async def _fake_ingestion_sources(*, worker_inventory=None):
        assert worker_inventory is None
        calls.append("ingestion-sources")
        return "ingestion-sources-task"

    async def _fake_kanban_purge(*, worker_inventory=None):
        assert worker_inventory is None
        calls.append("kanban-purge")
        return "kanban-purge-task"

    async def _fake_files_gc():
        calls.append("files-gc")
        return "files-gc-task"

    async def _fake_notifications():
        calls.append("notifications")
        return "notifications-task"

    async def _fake_jobs_prune():
        calls.append("jobs-prune")
        return "jobs-prune-task"

    monkeypatch.setattr(startup_maintenance, "_start_quality_eval_scheduler", _fake_quality)
    monkeypatch.setattr(startup_maintenance, "_start_outputs_purge_scheduler", _fake_outputs)
    monkeypatch.setattr(startup_maintenance, "_start_kanban_activity_cleanup_scheduler", _fake_kanban_activity)
    monkeypatch.setattr(startup_maintenance, "_start_ingestion_sources_cleanup_scheduler", _fake_ingestion_sources)
    monkeypatch.setattr(startup_maintenance, "_start_kanban_purge_scheduler", _fake_kanban_purge)
    monkeypatch.setattr(startup_maintenance, "_start_file_artifacts_export_gc_scheduler", _fake_files_gc)
    monkeypatch.setattr(startup_maintenance, "_start_notifications_prune_scheduler", _fake_notifications)
    monkeypatch.setattr(startup_maintenance, "_start_jobs_prune_scheduler", _fake_jobs_prune)

    handles = await startup_maintenance.start_maintenance_schedulers()

    assert calls == [
        "quality",
        "outputs",
        "kanban-activity",
        "ingestion-sources",
        "kanban-purge",
        "files-gc",
        "notifications",
        "jobs-prune",
    ]
    assert handles.quality_eval_task == "quality-task"
    assert handles.outputs_purge_task == "outputs-task"
    assert handles.kanban_activity_cleanup_task == "kanban-activity-task"
    assert handles.ingestion_sources_cleanup_task == "ingestion-sources-task"
    assert handles.kanban_purge_task == "kanban-purge-task"
    assert handles.files_export_gc_task == "files-gc-task"
    assert handles.notifications_prune_task == "notifications-task"
    assert handles.jobs_prune_task == "jobs-prune-task"


@pytest.mark.asyncio
async def test_start_maintenance_schedulers_passes_worker_inventory_to_registered_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    worker_inventory = object()
    calls: list[tuple[str, object | None]] = []

    async def _fake_quality() -> None:
        return None

    async def _fake_outputs() -> None:
        return None

    async def _fake_kanban_activity(
        *,
        worker_inventory: object | None = None,
    ) -> str:
        calls.append(("kanban-activity", worker_inventory))
        return "kanban-activity-task"

    async def _fake_ingestion_sources(
        *,
        worker_inventory: object | None = None,
    ) -> str:
        calls.append(("ingestion-sources", worker_inventory))
        return "ingestion-sources-task"

    async def _fake_kanban_purge(
        *,
        worker_inventory: object | None = None,
    ) -> str:
        calls.append(("kanban-purge", worker_inventory))
        return "kanban-purge-task"

    async def _fake_files_gc() -> None:
        return None

    async def _fake_notifications() -> None:
        return None

    async def _fake_jobs_prune() -> None:
        return None

    monkeypatch.setattr(startup_maintenance, "_start_quality_eval_scheduler", _fake_quality)
    monkeypatch.setattr(startup_maintenance, "_start_outputs_purge_scheduler", _fake_outputs)
    monkeypatch.setattr(startup_maintenance, "_start_kanban_activity_cleanup_scheduler", _fake_kanban_activity)
    monkeypatch.setattr(startup_maintenance, "_start_ingestion_sources_cleanup_scheduler", _fake_ingestion_sources)
    monkeypatch.setattr(startup_maintenance, "_start_kanban_purge_scheduler", _fake_kanban_purge)
    monkeypatch.setattr(startup_maintenance, "_start_file_artifacts_export_gc_scheduler", _fake_files_gc)
    monkeypatch.setattr(startup_maintenance, "_start_notifications_prune_scheduler", _fake_notifications)
    monkeypatch.setattr(startup_maintenance, "_start_jobs_prune_scheduler", _fake_jobs_prune)

    handles = await startup_maintenance.start_maintenance_schedulers(
        worker_inventory=worker_inventory,
    )

    assert calls == [
        ("kanban-activity", worker_inventory),
        ("ingestion-sources", worker_inventory),
        ("kanban-purge", worker_inventory),
    ]
    assert handles.kanban_activity_cleanup_task == "kanban-activity-task"
    assert handles.ingestion_sources_cleanup_task == "ingestion-sources-task"
    assert handles.kanban_purge_task == "kanban-purge-task"


@pytest.mark.asyncio
async def test_ingestion_sources_cleanup_scheduler_registers_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    worker_inventory = object()
    task = object()
    stop_event = object()
    calls: list[dict[str, object]] = []

    async def _fake_start_stop_event_worker(
        inventory: object,
        **kwargs: object,
    ) -> tuple[object, object]:
        calls.append({"inventory": inventory, **kwargs})
        return task, stop_event

    monkeypatch.setattr(startup_maintenance, "_env_enabled", lambda key: True)
    monkeypatch.setattr(
        startup_maintenance,
        "start_stop_event_worker",
        _fake_start_stop_event_worker,
    )

    returned_task = await startup_maintenance._start_ingestion_sources_cleanup_scheduler(
        worker_inventory=worker_inventory,
    )

    assert returned_task is task
    assert calls == [
        {
            "inventory": worker_inventory,
            "name": "ingestion_sources_cleanup",
            "task_name": "ingestion_sources_cleanup_task",
            "coroutine_factory": startup_maintenance._run_ingestion_sources_cleanup_loop,
            "category": "maintenance",
            "shutdown_phase": startup_maintenance.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        }
    ]


@pytest.mark.asyncio
async def test_start_quality_eval_scheduler_skips_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()

    monkeypatch.setattr(startup_maintenance, "_env_enabled", lambda key: False)

    task = await startup_maintenance._start_quality_eval_scheduler()

    assert task is None


@pytest.mark.asyncio
async def test_start_jobs_prune_scheduler_starts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()

    async def _fake_start():
        return "jobs-prune-task"

    monkeypatch.setattr(startup_maintenance, "_env_enabled", lambda key: True)
    monkeypatch.setattr(startup_maintenance, "_start_jobs_prune_scheduler_service", _fake_start)

    task = await startup_maintenance._start_jobs_prune_scheduler()

    assert task == "jobs-prune-task"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("starter_name", "service_name", "expected_name"),
    [
        (
            "_start_kanban_activity_cleanup_scheduler",
            "_start_kanban_activity_cleanup_scheduler_service",
            "kanban_activity_cleanup_scheduler",
        ),
        (
            "_start_kanban_purge_scheduler",
            "_start_kanban_purge_scheduler_service",
            "kanban_purge_scheduler",
        ),
    ],
)
async def test_kanban_maintenance_schedulers_register_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
    starter_name: str,
    service_name: str,
    expected_name: str,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    registrations = []
    task = object()

    class _FakeInventory:
        def register(self, worker):
            registrations.append(worker)
            return worker

    async def _fake_start():
        return task

    monkeypatch.setattr(startup_maintenance, "_env_enabled", lambda key: True)
    monkeypatch.setattr(startup_maintenance, service_name, _fake_start)

    started_task = await getattr(startup_maintenance, starter_name)(
        worker_inventory=_FakeInventory(),
    )

    assert started_task is task
    assert len(registrations) == 1
    worker = registrations[0]
    assert worker.name == expected_name
    assert worker.task is task
    assert worker.stop_event is None
    assert worker.timeout_sec == 5.0
    assert worker.category == "maintenance"
    assert worker.shutdown_phase == startup_maintenance.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN


@pytest.mark.asyncio
async def test_start_env_gated_task_returns_task_when_inventory_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    task = object()

    class _FailingInventory:
        def register(self, worker):
            raise AttributeError("registration unavailable")

    async def _fake_start():
        return task

    monkeypatch.setattr(startup_maintenance, "_env_enabled", lambda key: True)

    started_task = await startup_maintenance._start_env_gated_task(
        env_key="KANBAN_PURGE_ENABLED",
        disabled_message="disabled",
        started_message="started",
        failure_message="failed: {exc}",
        starter=_fake_start,
        worker_inventory=_FailingInventory(),
        worker_name="kanban_purge_scheduler",
    )

    assert started_task is task

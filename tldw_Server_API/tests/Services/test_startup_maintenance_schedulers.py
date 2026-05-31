from __future__ import annotations

import asyncio
import importlib
import sys

import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.unit


def _import_startup_maintenance_schedulers():
    sys.modules.pop("tldw_Server_API.app.services.startup_maintenance_schedulers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_maintenance_schedulers")


async def _wait_forever() -> None:
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_start_maintenance_schedulers_combines_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    calls: list[str] = []

    async def _fake_quality(*, worker_inventory=None):
        assert worker_inventory is None
        calls.append("quality")
        return "quality-task"

    async def _fake_outputs(*, worker_inventory=None):
        assert worker_inventory is None
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

    async def _fake_files_gc(*, worker_inventory=None):
        assert worker_inventory is None
        calls.append("files-gc")
        return "files-gc-task"

    async def _fake_notifications(*, worker_inventory=None):
        assert worker_inventory is None
        calls.append("notifications")
        return "notifications-task"

    async def _fake_jobs_prune(*, worker_inventory=None):
        assert worker_inventory is None
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

    async def _fake_quality(
        *,
        worker_inventory: object | None = None,
    ) -> str:
        calls.append(("quality", worker_inventory))
        return "quality-task"

    async def _fake_outputs(
        *,
        worker_inventory: object | None = None,
    ) -> str:
        calls.append(("outputs", worker_inventory))
        return "outputs-task"

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

    async def _fake_files_gc(
        *,
        worker_inventory: object | None = None,
    ) -> str:
        calls.append(("files-gc", worker_inventory))
        return "files-gc-task"

    async def _fake_notifications(
        *,
        worker_inventory: object | None = None,
    ) -> str:
        calls.append(("notifications", worker_inventory))
        return "notifications-task"

    async def _fake_jobs_prune(
        *,
        worker_inventory: object | None = None,
    ) -> str:
        calls.append(("jobs-prune", worker_inventory))
        return "jobs-prune-task"

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
        ("quality", worker_inventory),
        ("outputs", worker_inventory),
        ("kanban-activity", worker_inventory),
        ("ingestion-sources", worker_inventory),
        ("kanban-purge", worker_inventory),
        ("files-gc", worker_inventory),
        ("notifications", worker_inventory),
        ("jobs-prune", worker_inventory),
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
async def test_start_env_gated_task_rolls_back_when_inventory_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    task = object()
    cancelled_tasks: list[object] = []

    class _FailingInventory:
        def register(self, worker):
            raise AttributeError("registration unavailable")

    async def _fake_start():
        return task

    async def _fake_cancel(seen_task):
        cancelled_tasks.append(seen_task)

    monkeypatch.setattr(startup_maintenance, "_env_enabled", lambda key: True)
    monkeypatch.setattr(startup_maintenance, "_cancel_unregistered_task", _fake_cancel)

    started_task = await startup_maintenance._start_env_gated_task(
        env_key="KANBAN_PURGE_ENABLED",
        disabled_message="disabled",
        started_message="started",
        failure_message="failed: {exc}",
        starter=_fake_start,
        worker_inventory=_FailingInventory(),
        worker_name="kanban_purge_scheduler",
    )

    assert started_task is None
    assert cancelled_tasks == [task]


@pytest.mark.asyncio
async def test_cancel_unregistered_task_bounds_rollback_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    waits: list[tuple[object, float]] = []
    warnings: list[tuple[str, tuple[object, ...]]] = []

    class _Task:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    async def _timeout_wait(task: object, *, timeout: float) -> None:
        waits.append((task, timeout))
        raise asyncio.TimeoutError

    monkeypatch.setattr(startup_maintenance.asyncio, "wait_for", _timeout_wait)
    monkeypatch.setattr(
        startup_maintenance.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    task = _Task()
    await startup_maintenance._cancel_unregistered_task(task, timeout=0.25)

    assert task.cancelled is True
    assert waits == [(task, 0.25)]
    assert warnings == [
        (
            "Maintenance scheduler did not cancel within {}s during startup rollback",
            (0.25,),
        )
    ]


@pytest.mark.asyncio
async def test_start_remaining_maintenance_schedulers_registers_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_maintenance = _import_startup_maintenance_schedulers()
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase, WorkerRegistry

    selected_envs = {
        "RAG_QUALITY_EVAL_ENABLED",
        "OUTPUTS_PURGE_ENABLED",
        "FILES_EXPORT_GC_ENABLED",
        "NOTIFICATIONS_PRUNE_ENABLED",
        "JOBS_PRUNE_ENFORCE",
    }
    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    created_tasks: list[asyncio.Task[None]] = []

    def _make_task(name: str) -> asyncio.Task[None]:
        task = asyncio.create_task(_wait_forever(), name=name)
        created_tasks.append(task)
        return task

    async def _fake_quality():
        return _make_task("rag_quality_eval_scheduler")

    async def _fake_outputs():
        return _make_task("outputs_purge_scheduler")

    async def _fake_files_gc():
        return _make_task("file_artifacts_export_gc")

    async def _fake_notifications():
        return _make_task("notifications_prune_scheduler")

    async def _fake_jobs_prune():
        return _make_task("jobs_prune_scheduler")

    monkeypatch.setattr(startup_maintenance, "_env_enabled", lambda key: key in selected_envs)
    monkeypatch.setattr(startup_maintenance, "_start_quality_eval_scheduler_service", _fake_quality)
    monkeypatch.setattr(startup_maintenance, "_start_outputs_purge_scheduler_service", _fake_outputs)
    monkeypatch.setattr(startup_maintenance, "_start_file_artifacts_export_gc_scheduler_service", _fake_files_gc)
    monkeypatch.setattr(startup_maintenance, "_start_notifications_prune_scheduler_service", _fake_notifications)
    monkeypatch.setattr(startup_maintenance, "_start_jobs_prune_scheduler_service", _fake_jobs_prune)

    try:
        handles = await startup_maintenance.start_maintenance_schedulers(
            worker_inventory=worker_inventory,
        )

        assert handles.quality_eval_task in created_tasks
        assert handles.outputs_purge_task in created_tasks
        assert handles.kanban_activity_cleanup_task is None
        assert handles.ingestion_sources_cleanup_task is None
        assert handles.kanban_purge_task is None
        assert handles.files_export_gc_task in created_tasks
        assert handles.notifications_prune_task in created_tasks
        assert handles.jobs_prune_task in created_tasks

        assert {
            handle.name: handle.shutdown_phase
            for handle in worker_inventory.handles
        } == {
            "quality_eval_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "outputs_purge_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "files_export_gc_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "notifications_prune_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            "jobs_prune_task": ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        }
        assert {handle.stop_event for handle in worker_inventory.handles} == {None}
        assert {handle.category for handle in worker_inventory.handles} == {"maintenance"}
        assert app.state._tldw_shutdown_job_poller_inventory == []
        assert {
            entry["name"]: entry["shutdown_phase"]
            for entry in app.state._tldw_shutdown_worker_inventory
        } == {
            "quality_eval_task": "background_worker_shutdown",
            "outputs_purge_task": "background_worker_shutdown",
            "files_export_gc_task": "background_worker_shutdown",
            "notifications_prune_task": "background_worker_shutdown",
            "jobs_prune_task": "background_worker_shutdown",
        }
    finally:
        for task in created_tasks:
            task.cancel()
        await asyncio.gather(*created_tasks, return_exceptions=True)

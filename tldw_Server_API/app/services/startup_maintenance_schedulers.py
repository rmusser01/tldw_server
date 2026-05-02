"""
Maintenance-scheduler startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy as _is_truthy
from tldw_Server_API.app.services.lifecycle_workers import (
    ManagedWorker,
    ShutdownPhase,
    WorkerInventory,
    start_stop_event_worker,
)

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class MaintenanceSchedulerHandles:
    """Startup-owned maintenance scheduler handles that should stay referenced in lifespan."""

    quality_eval_task: Any | None = None
    outputs_purge_task: Any | None = None
    kanban_activity_cleanup_task: Any | None = None
    ingestion_sources_cleanup_task: Any | None = None
    kanban_purge_task: Any | None = None
    files_export_gc_task: Any | None = None
    notifications_prune_task: Any | None = None
    jobs_prune_task: Any | None = None


async def start_maintenance_schedulers(
    *,
    worker_inventory: WorkerInventory | None = None,
) -> MaintenanceSchedulerHandles:
    """Start the env-gated maintenance scheduler batch and return explicit task handles."""
    return MaintenanceSchedulerHandles(
        quality_eval_task=await _start_quality_eval_scheduler(
            worker_inventory=worker_inventory,
        ),
        outputs_purge_task=await _start_outputs_purge_scheduler(
            worker_inventory=worker_inventory,
        ),
        kanban_activity_cleanup_task=await _start_kanban_activity_cleanup_scheduler(
            worker_inventory=worker_inventory,
        ),
        ingestion_sources_cleanup_task=await _start_ingestion_sources_cleanup_scheduler(
            worker_inventory=worker_inventory,
        ),
        kanban_purge_task=await _start_kanban_purge_scheduler(
            worker_inventory=worker_inventory,
        ),
        files_export_gc_task=await _start_file_artifacts_export_gc_scheduler(
            worker_inventory=worker_inventory,
        ),
        notifications_prune_task=await _start_notifications_prune_scheduler(
            worker_inventory=worker_inventory,
        ),
        jobs_prune_task=await _start_jobs_prune_scheduler(
            worker_inventory=worker_inventory,
        ),
    )


def _env_enabled(key: str) -> bool:
    return _is_truthy(os.getenv(key, "false"))


async def _start_quality_eval_scheduler(
    *,
    worker_inventory: WorkerInventory | None = None,
) -> Any | None:
    return await _start_env_gated_task(
        env_key="RAG_QUALITY_EVAL_ENABLED",
        disabled_message="RAG quality eval scheduler disabled (RAG_QUALITY_EVAL_ENABLED != true)",
        started_message="RAG quality eval scheduler started",
        failure_message="Failed to start RAG quality eval scheduler: {exc}",
        starter=_start_quality_eval_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="quality_eval_task",
    )


async def _start_outputs_purge_scheduler(
    *,
    worker_inventory: WorkerInventory | None = None,
) -> Any | None:
    return await _start_env_gated_task(
        env_key="OUTPUTS_PURGE_ENABLED",
        disabled_message="Outputs purge scheduler disabled (OUTPUTS_PURGE_ENABLED != true)",
        started_message="Outputs purge scheduler started",
        failure_message="Failed to start Outputs purge scheduler: {exc}",
        starter=_start_outputs_purge_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="outputs_purge_task",
    )


async def _start_kanban_activity_cleanup_scheduler(
    *,
    worker_inventory: WorkerInventory | None = None,
) -> Any | None:
    return await _start_env_gated_task(
        env_key="KANBAN_ACTIVITY_CLEANUP_ENABLED",
        disabled_message="Kanban activity cleanup scheduler disabled (KANBAN_ACTIVITY_CLEANUP_ENABLED != true)",
        started_message="Kanban activity cleanup scheduler started",
        failure_message="Failed to start Kanban activity cleanup scheduler: {exc}",
        starter=_start_kanban_activity_cleanup_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="kanban_activity_cleanup_scheduler",
    )


async def _start_ingestion_sources_cleanup_scheduler(
    *,
    worker_inventory: WorkerInventory | None = None,
) -> Any | None:
    async def _start_service() -> Any | None:
        if worker_inventory is None:
            return await _start_ingestion_sources_cleanup_scheduler_service()
        return await _start_ingestion_sources_cleanup_registered(worker_inventory)

    return await _start_env_gated_task(
        env_key="INGESTION_SOURCES_CLEANUP_ENABLED",
        disabled_message=(
            "Ingestion source archive cleanup scheduler disabled "
            "(INGESTION_SOURCES_CLEANUP_ENABLED != true)"
        ),
        started_message="Ingestion source archive cleanup scheduler started",
        failure_message="Failed to start ingestion source archive cleanup scheduler: {exc}",
        starter=_start_service,
    )


async def _start_kanban_purge_scheduler(
    *,
    worker_inventory: WorkerInventory | None = None,
) -> Any | None:
    return await _start_env_gated_task(
        env_key="KANBAN_PURGE_ENABLED",
        disabled_message="Kanban purge scheduler disabled (KANBAN_PURGE_ENABLED != true)",
        started_message="Kanban purge scheduler started",
        failure_message="Failed to start Kanban purge scheduler: {exc}",
        starter=_start_kanban_purge_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="kanban_purge_scheduler",
    )


async def _start_file_artifacts_export_gc_scheduler(
    *,
    worker_inventory: WorkerInventory | None = None,
) -> Any | None:
    return await _start_env_gated_task(
        env_key="FILES_EXPORT_GC_ENABLED",
        disabled_message="File artifacts export GC scheduler disabled (FILES_EXPORT_GC_ENABLED != true)",
        started_message="File artifacts export GC scheduler started",
        failure_message="Failed to start File artifacts export GC scheduler: {exc}",
        starter=_start_file_artifacts_export_gc_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="files_export_gc_task",
    )


async def _start_notifications_prune_scheduler(
    *,
    worker_inventory: WorkerInventory | None = None,
) -> Any | None:
    return await _start_env_gated_task(
        env_key="NOTIFICATIONS_PRUNE_ENABLED",
        disabled_message="Notifications prune scheduler disabled (NOTIFICATIONS_PRUNE_ENABLED != true)",
        started_message="Notifications prune scheduler started",
        failure_message="Failed to start Notifications prune scheduler: {exc}",
        starter=_start_notifications_prune_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="notifications_prune_task",
    )


async def _start_jobs_prune_scheduler(
    *,
    worker_inventory: WorkerInventory | None = None,
) -> Any | None:
    return await _start_env_gated_task(
        env_key="JOBS_PRUNE_ENFORCE",
        disabled_message="Jobs prune scheduler disabled (JOBS_PRUNE_ENFORCE != true)",
        started_message="Jobs prune scheduler started",
        failure_message="Failed to start Jobs prune scheduler: {exc}",
        starter=_start_jobs_prune_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="jobs_prune_task",
    )


async def _start_env_gated_task(
    *,
    env_key: str,
    disabled_message: str,
    started_message: str,
    failure_message: str,
    starter: Callable[[], Awaitable[Any | None]],
    worker_inventory: WorkerInventory | None = None,
    worker_name: str | None = None,
) -> Any | None:
    """Start one env-gated scheduler and optionally register it for managed shutdown.

    Scheduler startup failures return ``None``. Inventory registration failures
    roll back the started task so unregistered maintenance work does not leak.
    """

    try:
        if not _env_enabled(env_key):
            logger.info(disabled_message)
            return None
        task = await starter()
        if task:
            await _register_maintenance_scheduler_task(
                worker_inventory=worker_inventory,
                task=task,
                worker_name=worker_name,
            )
            logger.info(started_message)
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(failure_message.format(exc=exc))
        return None


async def _register_maintenance_scheduler_task(
    *,
    worker_inventory: WorkerInventory | None,
    task: Any,
    worker_name: str | None,
) -> None:
    if worker_inventory is None or worker_name is None:
        return

    try:
        worker_inventory.register(
            ManagedWorker(
                name=worker_name,
                task=task,
                stop_event=None,
                category="maintenance",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
        )
    except _STARTUP_GUARD_EXCEPTIONS:
        await _cancel_unregistered_task(task)
        raise


async def _cancel_unregistered_task(task: Any, *, timeout: float = 1.0) -> None:
    try:
        task.cancel()
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"Maintenance scheduler startup rollback cancel failed: {exc}")
        return
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        logger.warning(
            "Maintenance scheduler did not cancel within {}s during startup rollback",
            timeout,
        )
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"Maintenance scheduler raised during startup rollback: {exc}")


async def _start_quality_eval_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.quality_eval_scheduler import start_quality_eval_scheduler

    return await start_quality_eval_scheduler()


async def _start_outputs_purge_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.outputs_purge_scheduler import start_outputs_purge_scheduler

    return await start_outputs_purge_scheduler()


async def _start_kanban_activity_cleanup_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.kanban_activity_cleanup_service import (
        start_kanban_activity_cleanup_scheduler,
    )

    return await start_kanban_activity_cleanup_scheduler()


async def _start_ingestion_sources_cleanup_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.ingestion_sources_cleanup_service import (
        start_ingestion_sources_cleanup_scheduler,
    )

    return await start_ingestion_sources_cleanup_scheduler()


async def _start_ingestion_sources_cleanup_registered(
    worker_inventory: WorkerInventory,
) -> Any | None:
    task, _stop_event = await start_stop_event_worker(
        worker_inventory,
        name="ingestion_sources_cleanup",
        task_name="ingestion_sources_cleanup_task",
        coroutine_factory=_run_ingestion_sources_cleanup_loop,
        category="maintenance",
        shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
    )
    return task


async def _run_ingestion_sources_cleanup_loop(stop_event: asyncio.Event) -> None:
    from tldw_Server_API.app.services.ingestion_sources_cleanup_service import (
        run_ingestion_sources_cleanup_loop,
    )

    await run_ingestion_sources_cleanup_loop(stop_event)


async def _start_kanban_purge_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.kanban_purge_service import start_kanban_purge_scheduler

    return await start_kanban_purge_scheduler()


async def _start_file_artifacts_export_gc_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.file_artifacts_export_gc_service import (
        start_file_artifacts_export_gc_scheduler,
    )

    return await start_file_artifacts_export_gc_scheduler()


async def _start_notifications_prune_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.notifications_prune_service import start_notifications_prune_scheduler

    return await start_notifications_prune_scheduler()


async def _start_jobs_prune_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.jobs_prune_scheduler import start_jobs_prune_scheduler

    return await start_jobs_prune_scheduler()

"""
Maintenance-scheduler startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy as _is_truthy

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


async def start_maintenance_schedulers() -> MaintenanceSchedulerHandles:
    """Start the env-gated maintenance scheduler batch and return explicit task handles."""
    return MaintenanceSchedulerHandles(
        quality_eval_task=await _start_quality_eval_scheduler(),
        outputs_purge_task=await _start_outputs_purge_scheduler(),
        kanban_activity_cleanup_task=await _start_kanban_activity_cleanup_scheduler(),
        ingestion_sources_cleanup_task=await _start_ingestion_sources_cleanup_scheduler(),
        kanban_purge_task=await _start_kanban_purge_scheduler(),
        files_export_gc_task=await _start_file_artifacts_export_gc_scheduler(),
        notifications_prune_task=await _start_notifications_prune_scheduler(),
        jobs_prune_task=await _start_jobs_prune_scheduler(),
    )


def _env_enabled(key: str) -> bool:
    return _is_truthy(os.getenv(key, "false"))


async def _start_quality_eval_scheduler() -> Any | None:
    return await _start_env_gated_task(
        env_key="RAG_QUALITY_EVAL_ENABLED",
        disabled_message="RAG quality eval scheduler disabled (RAG_QUALITY_EVAL_ENABLED != true)",
        started_message="RAG quality eval scheduler started",
        failure_message="Failed to start RAG quality eval scheduler: {exc}",
        starter=_start_quality_eval_scheduler_service,
    )


async def _start_outputs_purge_scheduler() -> Any | None:
    return await _start_env_gated_task(
        env_key="OUTPUTS_PURGE_ENABLED",
        disabled_message="Outputs purge scheduler disabled (OUTPUTS_PURGE_ENABLED != true)",
        started_message="Outputs purge scheduler started",
        failure_message="Failed to start Outputs purge scheduler: {exc}",
        starter=_start_outputs_purge_scheduler_service,
    )


async def _start_kanban_activity_cleanup_scheduler() -> Any | None:
    return await _start_env_gated_task(
        env_key="KANBAN_ACTIVITY_CLEANUP_ENABLED",
        disabled_message="Kanban activity cleanup scheduler disabled (KANBAN_ACTIVITY_CLEANUP_ENABLED != true)",
        started_message="Kanban activity cleanup scheduler started",
        failure_message="Failed to start Kanban activity cleanup scheduler: {exc}",
        starter=_start_kanban_activity_cleanup_scheduler_service,
    )


async def _start_ingestion_sources_cleanup_scheduler() -> Any | None:
    return await _start_env_gated_task(
        env_key="INGESTION_SOURCES_CLEANUP_ENABLED",
        disabled_message=(
            "Ingestion source archive cleanup scheduler disabled "
            "(INGESTION_SOURCES_CLEANUP_ENABLED != true)"
        ),
        started_message="Ingestion source archive cleanup scheduler started",
        failure_message="Failed to start ingestion source archive cleanup scheduler: {exc}",
        starter=_start_ingestion_sources_cleanup_scheduler_service,
    )


async def _start_kanban_purge_scheduler() -> Any | None:
    return await _start_env_gated_task(
        env_key="KANBAN_PURGE_ENABLED",
        disabled_message="Kanban purge scheduler disabled (KANBAN_PURGE_ENABLED != true)",
        started_message="Kanban purge scheduler started",
        failure_message="Failed to start Kanban purge scheduler: {exc}",
        starter=_start_kanban_purge_scheduler_service,
    )


async def _start_file_artifacts_export_gc_scheduler() -> Any | None:
    return await _start_env_gated_task(
        env_key="FILES_EXPORT_GC_ENABLED",
        disabled_message="File artifacts export GC scheduler disabled (FILES_EXPORT_GC_ENABLED != true)",
        started_message="File artifacts export GC scheduler started",
        failure_message="Failed to start File artifacts export GC scheduler: {exc}",
        starter=_start_file_artifacts_export_gc_scheduler_service,
    )


async def _start_notifications_prune_scheduler() -> Any | None:
    return await _start_env_gated_task(
        env_key="NOTIFICATIONS_PRUNE_ENABLED",
        disabled_message="Notifications prune scheduler disabled (NOTIFICATIONS_PRUNE_ENABLED != true)",
        started_message="Notifications prune scheduler started",
        failure_message="Failed to start Notifications prune scheduler: {exc}",
        starter=_start_notifications_prune_scheduler_service,
    )


async def _start_jobs_prune_scheduler() -> Any | None:
    return await _start_env_gated_task(
        env_key="JOBS_PRUNE_ENFORCE",
        disabled_message="Jobs prune scheduler disabled (JOBS_PRUNE_ENFORCE != true)",
        started_message="Jobs prune scheduler started",
        failure_message="Failed to start Jobs prune scheduler: {exc}",
        starter=_start_jobs_prune_scheduler_service,
    )


async def _start_env_gated_task(
    *,
    env_key: str,
    disabled_message: str,
    started_message: str,
    failure_message: str,
    starter,
) -> Any | None:
    try:
        if not _env_enabled(env_key):
            logger.info(disabled_message)
            return None
        task = await starter()
        if task:
            logger.info(started_message)
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(failure_message.format(exc=exc))
        return None


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

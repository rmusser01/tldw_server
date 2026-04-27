"""
Grouped late-stop worker shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class GroupedLateStopWorkerHandles:
    """Updated task and stop-event handles for grouped late-stop workers."""

    media_ingest_jobs_task: Any | None = None
    media_ingest_jobs_stop_event: Any | None = None
    media_ingest_heavy_jobs_task: Any | None = None
    media_ingest_heavy_jobs_stop_event: Any | None = None
    reading_digest_jobs_task: Any | None = None
    reading_digest_jobs_stop_event: Any | None = None
    study_pack_jobs_task: Any | None = None
    study_pack_jobs_stop_event: Any | None = None
    study_suggestions_jobs_task: Any | None = None
    study_suggestions_jobs_stop_event: Any | None = None
    companion_reflection_jobs_task: Any | None = None
    companion_reflection_jobs_stop_event: Any | None = None
    reminder_jobs_task: Any | None = None
    admin_backup_jobs_task: Any | None = None
    admin_maintenance_rotation_jobs_task: Any | None = None
    admin_maintenance_rotation_jobs_stop_event: Any | None = None
    recipe_run_jobs_task: Any | None = None
    recipe_run_jobs_stop_event: Any | None = None
    evals_abtest_jobs_task: Any | None = None
    evals_abtest_jobs_stop_event: Any | None = None


async def shutdown_grouped_late_stop_workers(
    *,
    media_ingest_jobs_task: Any | None,
    media_ingest_jobs_stop_event: Any | None,
    media_ingest_heavy_jobs_task: Any | None,
    media_ingest_heavy_jobs_stop_event: Any | None,
    reading_digest_jobs_task: Any | None,
    reading_digest_jobs_stop_event: Any | None,
    study_pack_jobs_task: Any | None,
    study_pack_jobs_stop_event: Any | None,
    study_suggestions_jobs_task: Any | None,
    study_suggestions_jobs_stop_event: Any | None,
    companion_reflection_jobs_task: Any | None,
    companion_reflection_jobs_stop_event: Any | None,
    reminder_jobs_task: Any | None,
    admin_backup_jobs_task: Any | None,
    admin_maintenance_rotation_jobs_task: Any | None,
    admin_maintenance_rotation_jobs_stop_event: Any | None,
    recipe_run_jobs_task: Any | None,
    recipe_run_jobs_stop_event: Any | None,
    evals_abtest_jobs_task: Any | None,
    evals_abtest_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> GroupedLateStopWorkerHandles:
    """Stop grouped late-stop workers in the legacy shutdown order."""
    media_ingest_shutdown_handles = await _shutdown_media_ingest_jobs_workers(
        media_ingest_jobs_task=media_ingest_jobs_task,
        media_ingest_jobs_stop_event=media_ingest_jobs_stop_event,
        media_ingest_heavy_jobs_task=media_ingest_heavy_jobs_task,
        media_ingest_heavy_jobs_stop_event=media_ingest_heavy_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    reading_study_companion_shutdown_handles = await _shutdown_reading_study_companion_jobs_workers(
        reading_digest_jobs_task=reading_digest_jobs_task,
        reading_digest_jobs_stop_event=reading_digest_jobs_stop_event,
        study_pack_jobs_task=study_pack_jobs_task,
        study_pack_jobs_stop_event=study_pack_jobs_stop_event,
        study_suggestions_jobs_task=study_suggestions_jobs_task,
        study_suggestions_jobs_stop_event=study_suggestions_jobs_stop_event,
        companion_reflection_jobs_task=companion_reflection_jobs_task,
        companion_reflection_jobs_stop_event=companion_reflection_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    reminder_admin_shutdown_handles = await _shutdown_reminder_admin_jobs_workers(
        reminder_jobs_task=reminder_jobs_task,
        admin_backup_jobs_task=admin_backup_jobs_task,
        admin_maintenance_rotation_jobs_task=admin_maintenance_rotation_jobs_task,
        admin_maintenance_rotation_jobs_stop_event=admin_maintenance_rotation_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    recipe_abtest_shutdown_handles = await _shutdown_recipe_abtest_jobs_workers(
        recipe_run_jobs_task=recipe_run_jobs_task,
        recipe_run_jobs_stop_event=recipe_run_jobs_stop_event,
        evals_abtest_jobs_task=evals_abtest_jobs_task,
        evals_abtest_jobs_stop_event=evals_abtest_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return GroupedLateStopWorkerHandles(
        media_ingest_jobs_task=media_ingest_shutdown_handles.media_ingest_jobs_task,
        media_ingest_jobs_stop_event=media_ingest_shutdown_handles.media_ingest_jobs_stop_event,
        media_ingest_heavy_jobs_task=media_ingest_shutdown_handles.media_ingest_heavy_jobs_task,
        media_ingest_heavy_jobs_stop_event=media_ingest_shutdown_handles.media_ingest_heavy_jobs_stop_event,
        reading_digest_jobs_task=reading_study_companion_shutdown_handles.reading_digest_jobs_task,
        reading_digest_jobs_stop_event=reading_study_companion_shutdown_handles.reading_digest_jobs_stop_event,
        study_pack_jobs_task=reading_study_companion_shutdown_handles.study_pack_jobs_task,
        study_pack_jobs_stop_event=reading_study_companion_shutdown_handles.study_pack_jobs_stop_event,
        study_suggestions_jobs_task=reading_study_companion_shutdown_handles.study_suggestions_jobs_task,
        study_suggestions_jobs_stop_event=(
            reading_study_companion_shutdown_handles.study_suggestions_jobs_stop_event
        ),
        companion_reflection_jobs_task=(
            reading_study_companion_shutdown_handles.companion_reflection_jobs_task
        ),
        companion_reflection_jobs_stop_event=(
            reading_study_companion_shutdown_handles.companion_reflection_jobs_stop_event
        ),
        reminder_jobs_task=reminder_admin_shutdown_handles.reminder_jobs_task,
        admin_backup_jobs_task=reminder_admin_shutdown_handles.admin_backup_jobs_task,
        admin_maintenance_rotation_jobs_task=(
            reminder_admin_shutdown_handles.admin_maintenance_rotation_jobs_task
        ),
        admin_maintenance_rotation_jobs_stop_event=(
            reminder_admin_shutdown_handles.admin_maintenance_rotation_jobs_stop_event
        ),
        recipe_run_jobs_task=recipe_abtest_shutdown_handles.recipe_run_jobs_task,
        recipe_run_jobs_stop_event=recipe_abtest_shutdown_handles.recipe_run_jobs_stop_event,
        evals_abtest_jobs_task=recipe_abtest_shutdown_handles.evals_abtest_jobs_task,
        evals_abtest_jobs_stop_event=recipe_abtest_shutdown_handles.evals_abtest_jobs_stop_event,
    )


async def run_shutdown_grouped_late_stop_workers(
    *,
    media_ingest_jobs_task: Any | None,
    media_ingest_jobs_stop_event: Any | None,
    media_ingest_heavy_jobs_task: Any | None,
    media_ingest_heavy_jobs_stop_event: Any | None,
    reading_digest_jobs_task: Any | None,
    reading_digest_jobs_stop_event: Any | None,
    study_pack_jobs_task: Any | None,
    study_pack_jobs_stop_event: Any | None,
    study_suggestions_jobs_task: Any | None,
    study_suggestions_jobs_stop_event: Any | None,
    companion_reflection_jobs_task: Any | None,
    companion_reflection_jobs_stop_event: Any | None,
    reminder_jobs_task: Any | None,
    admin_backup_jobs_task: Any | None,
    admin_maintenance_rotation_jobs_task: Any | None,
    admin_maintenance_rotation_jobs_stop_event: Any | None,
    recipe_run_jobs_task: Any | None,
    recipe_run_jobs_stop_event: Any | None,
    evals_abtest_jobs_task: Any | None,
    evals_abtest_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> GroupedLateStopWorkerHandles:
    """Run grouped late-stop worker shutdown with main-lifespan fallback behavior."""
    try:
        return await shutdown_grouped_late_stop_workers(
            media_ingest_jobs_task=media_ingest_jobs_task,
            media_ingest_jobs_stop_event=media_ingest_jobs_stop_event,
            media_ingest_heavy_jobs_task=media_ingest_heavy_jobs_task,
            media_ingest_heavy_jobs_stop_event=media_ingest_heavy_jobs_stop_event,
            reading_digest_jobs_task=reading_digest_jobs_task,
            reading_digest_jobs_stop_event=reading_digest_jobs_stop_event,
            study_pack_jobs_task=study_pack_jobs_task,
            study_pack_jobs_stop_event=study_pack_jobs_stop_event,
            study_suggestions_jobs_task=study_suggestions_jobs_task,
            study_suggestions_jobs_stop_event=study_suggestions_jobs_stop_event,
            companion_reflection_jobs_task=companion_reflection_jobs_task,
            companion_reflection_jobs_stop_event=companion_reflection_jobs_stop_event,
            reminder_jobs_task=reminder_jobs_task,
            admin_backup_jobs_task=admin_backup_jobs_task,
            admin_maintenance_rotation_jobs_task=admin_maintenance_rotation_jobs_task,
            admin_maintenance_rotation_jobs_stop_event=admin_maintenance_rotation_jobs_stop_event,
            recipe_run_jobs_task=recipe_run_jobs_task,
            recipe_run_jobs_stop_event=recipe_run_jobs_stop_event,
            evals_abtest_jobs_task=evals_abtest_jobs_task,
            evals_abtest_jobs_stop_event=evals_abtest_jobs_stop_event,
            should_run_late_stop=should_run_late_stop,
            guard_exceptions=guard_exceptions,
        )
    except guard_exceptions as exc:
        logger.debug(f"Grouped late-stop workers skipped: {exc}")
        return GroupedLateStopWorkerHandles(
            media_ingest_jobs_task=media_ingest_jobs_task,
            media_ingest_jobs_stop_event=media_ingest_jobs_stop_event,
            media_ingest_heavy_jobs_task=media_ingest_heavy_jobs_task,
            media_ingest_heavy_jobs_stop_event=media_ingest_heavy_jobs_stop_event,
            reading_digest_jobs_task=reading_digest_jobs_task,
            reading_digest_jobs_stop_event=reading_digest_jobs_stop_event,
            study_pack_jobs_task=study_pack_jobs_task,
            study_pack_jobs_stop_event=study_pack_jobs_stop_event,
            study_suggestions_jobs_task=study_suggestions_jobs_task,
            study_suggestions_jobs_stop_event=study_suggestions_jobs_stop_event,
            companion_reflection_jobs_task=companion_reflection_jobs_task,
            companion_reflection_jobs_stop_event=companion_reflection_jobs_stop_event,
            reminder_jobs_task=reminder_jobs_task,
            admin_backup_jobs_task=admin_backup_jobs_task,
            admin_maintenance_rotation_jobs_task=admin_maintenance_rotation_jobs_task,
            admin_maintenance_rotation_jobs_stop_event=admin_maintenance_rotation_jobs_stop_event,
            recipe_run_jobs_task=recipe_run_jobs_task,
            recipe_run_jobs_stop_event=recipe_run_jobs_stop_event,
            evals_abtest_jobs_task=evals_abtest_jobs_task,
            evals_abtest_jobs_stop_event=evals_abtest_jobs_stop_event,
        )


async def _shutdown_media_ingest_jobs_workers(**kwargs):
    from tldw_Server_API.app.services.shutdown_media_ingest_jobs_workers import (
        shutdown_media_ingest_jobs_workers,
    )

    return await shutdown_media_ingest_jobs_workers(**kwargs)


async def _shutdown_reading_study_companion_jobs_workers(**kwargs):
    from tldw_Server_API.app.services.shutdown_reading_study_companion_jobs_workers import (
        shutdown_reading_study_companion_jobs_workers,
    )

    return await shutdown_reading_study_companion_jobs_workers(**kwargs)


async def _shutdown_reminder_admin_jobs_workers(**kwargs):
    from tldw_Server_API.app.services.shutdown_reminder_admin_jobs_workers import (
        shutdown_reminder_admin_jobs_workers,
    )

    return await shutdown_reminder_admin_jobs_workers(**kwargs)


async def _shutdown_recipe_abtest_jobs_workers(**kwargs):
    from tldw_Server_API.app.services.shutdown_recipe_abtest_jobs_workers import (
        shutdown_recipe_abtest_jobs_workers,
    )

    return await shutdown_recipe_abtest_jobs_workers(**kwargs)

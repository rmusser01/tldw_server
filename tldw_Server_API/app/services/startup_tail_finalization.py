"""
Startup tail finalization extracted from the application lifespan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tldw_Server_API.app.services.startup_recurring_schedulers import (
        RecurringSchedulerHandles,
    )


def _build_owned_job_poller_registrations(
    *,
    startup_worker_group_handles: Any,
    startup_service_group_handles: Any,
) -> list[tuple[str, Any, Any, float]]:
    return [
        (
            "core_jobs_task",
            startup_worker_group_handles.core_jobs_task,
            startup_worker_group_handles.core_jobs_stop_event,
            5.0,
        ),
        (
            "files_jobs_task",
            startup_worker_group_handles.files_jobs_task,
            startup_worker_group_handles.files_jobs_stop_event,
            5.0,
        ),
        (
            "data_tables_jobs_task",
            startup_worker_group_handles.data_tables_jobs_task,
            startup_worker_group_handles.data_tables_jobs_stop_event,
            5.0,
        ),
        (
            "prompt_studio_jobs_task",
            startup_worker_group_handles.prompt_studio_jobs_task,
            startup_worker_group_handles.prompt_studio_jobs_stop_event,
            5.0,
        ),
        (
            "study_pack_jobs_task",
            startup_worker_group_handles.study_pack_jobs_task,
            startup_worker_group_handles.study_pack_jobs_stop_event,
            5.0,
        ),
        (
            "study_suggestions_jobs_task",
            startup_worker_group_handles.study_suggestions_jobs_task,
            startup_worker_group_handles.study_suggestions_jobs_stop_event,
            5.0,
        ),
        (
            "privilege_snapshot_task",
            startup_worker_group_handles.privilege_snapshot_task,
            startup_worker_group_handles.privilege_snapshot_stop_event,
            5.0,
        ),
        (
            "audio_jobs_task",
            startup_worker_group_handles.audio_jobs_task,
            startup_worker_group_handles.audio_jobs_stop_event,
            5.0,
        ),
        (
            "audiobook_jobs_task",
            startup_worker_group_handles.audiobook_jobs_task,
            startup_worker_group_handles.audiobook_jobs_stop_event,
            5.0,
        ),
        (
            "presentation_render_jobs_task",
            startup_worker_group_handles.presentation_render_jobs_task,
            startup_worker_group_handles.presentation_render_jobs_stop_event,
            5.0,
        ),
        (
            "media_ingest_jobs_task",
            startup_worker_group_handles.media_ingest_jobs_task,
            startup_worker_group_handles.media_ingest_jobs_stop_event,
            5.0,
        ),
        (
            "media_ingest_heavy_jobs_task",
            startup_worker_group_handles.media_ingest_heavy_jobs_task,
            startup_worker_group_handles.media_ingest_heavy_jobs_stop_event,
            5.0,
        ),
        (
            "reading_digest_jobs_task",
            startup_worker_group_handles.reading_digest_jobs_task,
            startup_worker_group_handles.reading_digest_jobs_stop_event,
            5.0,
        ),
        (
            "vn_asset_jobs_task",
            startup_worker_group_handles.vn_asset_jobs_task,
            startup_worker_group_handles.vn_asset_jobs_stop_event,
            5.0,
        ),
        (
            "vn_asset_generation_jobs_task",
            startup_worker_group_handles.vn_asset_generation_jobs_task,
            startup_worker_group_handles.vn_asset_generation_jobs_stop_event,
            5.0,
        ),
        (
            "companion_reflection_jobs_task",
            startup_worker_group_handles.companion_reflection_jobs_task,
            startup_worker_group_handles.companion_reflection_jobs_stop_event,
            5.0,
        ),
        (
            "reminder_jobs_task",
            startup_worker_group_handles.reminder_jobs_task,
            startup_worker_group_handles.reminder_jobs_stop_event,
            5.0,
        ),
        (
            "admin_backup_jobs_task",
            startup_worker_group_handles.admin_backup_jobs_task,
            startup_worker_group_handles.admin_backup_jobs_stop_event,
            5.0,
        ),
        (
            "admin_byok_validation_jobs_task",
            startup_worker_group_handles.admin_byok_validation_jobs_task,
            startup_worker_group_handles.admin_byok_validation_jobs_stop_event,
            5.0,
        ),
        (
            "admin_maintenance_rotation_jobs_task",
            startup_worker_group_handles.admin_maintenance_rotation_jobs_task,
            startup_worker_group_handles.admin_maintenance_rotation_jobs_stop_event,
            5.0,
        ),
        (
            "recipe_run_jobs_task",
            startup_worker_group_handles.recipe_run_jobs_task,
            startup_worker_group_handles.recipe_run_jobs_stop_event,
            5.0,
        ),
        (
            "evals_abtest_jobs_task",
            startup_worker_group_handles.evals_abtest_jobs_task,
            startup_worker_group_handles.evals_abtest_jobs_stop_event,
            5.0,
        ),
        (
            "connectors_jobs_task",
            startup_service_group_handles.connectors_jobs_task,
            startup_service_group_handles.connectors_jobs_stop_event,
            5.0,
        ),
    ]


async def finalize_startup_tail(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    startup_worker_group_handles: Any,
    startup_service_group_handles: Any,
    replace_owned_job_poller_inventory: Callable[..., None],
    test_mode: bool,
    worker_inventory: Any | None = None,
) -> RecurringSchedulerHandles:
    replace_owned_job_poller_inventory(
        app,
        owned_job_pollers,
        registrations=_build_owned_job_poller_registrations(
            startup_worker_group_handles=startup_worker_group_handles,
            startup_service_group_handles=startup_service_group_handles,
        ),
    )
    if worker_inventory is None:
        return await _start_recurring_schedulers(test_mode=test_mode)
    return await _start_recurring_schedulers(
        test_mode=test_mode,
        worker_inventory=worker_inventory,
    )


async def _start_recurring_schedulers(**kwargs: Any) -> RecurringSchedulerHandles:
    from tldw_Server_API.app.services.startup_recurring_schedulers import (
        start_recurring_schedulers,
    )

    return await start_recurring_schedulers(**kwargs)

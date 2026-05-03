"""
Startup worker-group orchestration extracted from the application lifespan.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

_TRUTHY_ENV_VALUES = {"true", "1", "yes", "y", "on"}


@dataclass
class StartupWorkerGroupHandles:
    """Combined startup handles produced by the worker-group startup burst."""

    core_jobs_stop_event: Any | None = None
    core_jobs_task: Any | None = None
    files_jobs_stop_event: Any | None = None
    files_jobs_task: Any | None = None
    data_tables_jobs_stop_event: Any | None = None
    data_tables_jobs_task: Any | None = None
    prompt_studio_jobs_stop_event: Any | None = None
    prompt_studio_jobs_task: Any | None = None
    study_pack_jobs_stop_event: Any | None = None
    study_pack_jobs_task: Any | None = None
    study_suggestions_jobs_stop_event: Any | None = None
    study_suggestions_jobs_task: Any | None = None
    privilege_snapshot_stop_event: Any | None = None
    privilege_snapshot_task: Any | None = None
    audio_jobs_stop_event: Any | None = None
    audio_jobs_task: Any | None = None
    audiobook_jobs_stop_event: Any | None = None
    audiobook_jobs_task: Any | None = None
    presentation_render_jobs_stop_event: Any | None = None
    presentation_render_jobs_task: Any | None = None
    media_ingest_jobs_stop_event: Any | None = None
    media_ingest_jobs_task: Any | None = None
    media_ingest_heavy_jobs_stop_event: Any | None = None
    media_ingest_heavy_jobs_task: Any | None = None
    reading_digest_jobs_stop_event: Any | None = None
    reading_digest_jobs_task: Any | None = None
    vn_asset_jobs_stop_event: Any | None = None
    vn_asset_jobs_task: Any | None = None
    vn_asset_generation_jobs_stop_event: Any | None = None
    vn_asset_generation_jobs_task: Any | None = None
    companion_reflection_jobs_stop_event: Any | None = None
    companion_reflection_jobs_task: Any | None = None
    reminder_jobs_stop_event: Any | None = None
    reminder_jobs_task: Any | None = None
    admin_backup_jobs_stop_event: Any | None = None
    admin_backup_jobs_task: Any | None = None
    admin_byok_validation_jobs_stop_event: Any | None = None
    admin_byok_validation_jobs_task: Any | None = None
    admin_maintenance_rotation_jobs_stop_event: Any | None = None
    admin_maintenance_rotation_jobs_task: Any | None = None
    recipe_run_jobs_stop_event: Any | None = None
    recipe_run_jobs_task: Any | None = None
    jobs_notifications_bridge_task: Any | None = None
    evals_abtest_jobs_stop_event: Any | None = None
    evals_abtest_jobs_task: Any | None = None


async def start_worker_groups(
    *,
    app: Any,
    app_settings: Mapping[str, Any],
    test_mode: bool,
    route_enabled: Callable[..., bool],
    startup_guard_exceptions: tuple[type[BaseException], ...],
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    worker_inventory: Any | None = None,
) -> StartupWorkerGroupHandles:
    """Start the startup worker/poller groups in the legacy order."""
    if worker_inventory is None:
        raise RuntimeError("worker_inventory is required to start worker groups")

    await _start_cleanup_workers(
        app_settings=app_settings,
        test_mode=test_mode,
        worker_inventory=worker_inventory,
    )

    def _env_flag(key: str, default: bool) -> bool:
        raw = os.getenv(key)
        if raw is None or str(raw).strip() == "":
            return bool(default)
        return str(raw).strip().lower() in _TRUTHY_ENV_VALUES

    def _route_default(route_key: str, *, default_stable: bool = True) -> bool:
        if test_mode:
            return False
        try:
            return bool(route_enabled(route_key, default_stable=default_stable))
        except startup_guard_exceptions:
            return bool(default_stable)

    sidecar_mode = _env_flag("TLDW_WORKERS_SIDECAR_MODE", False)

    def _should_start_worker(flag_key: str, route_key: str, *, default_stable: bool = True) -> bool:
        if sidecar_mode:
            return False
        return _env_flag(flag_key, _route_default(route_key, default_stable=default_stable))

    if sidecar_mode:
        logger.info("Sidecar worker mode enabled; in-process Jobs workers are disabled")

    primary_jobs_poller_handles = await _start_primary_jobs_pollers(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=_should_start_worker,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    study_privilege_jobs_poller_handles = await _start_study_privilege_jobs_pollers(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=_should_start_worker,
        worker_inventory=worker_inventory,
    )
    await _start_compactor_websub_workers(
        should_start_worker=_should_start_worker,
        worker_inventory=worker_inventory,
    )
    content_jobs_poller_handles = await _start_content_jobs_pollers(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=_should_start_worker,
        worker_inventory=worker_inventory,
    )
    sidecar_owned_jobs_poller_handles = await _start_sidecar_owned_jobs_pollers(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    notifications_abtest_startup_handles = await _start_notifications_abtest_workers(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    return StartupWorkerGroupHandles(
        core_jobs_stop_event=primary_jobs_poller_handles.core_jobs_stop_event,
        core_jobs_task=primary_jobs_poller_handles.core_jobs_task,
        files_jobs_stop_event=primary_jobs_poller_handles.files_jobs_stop_event,
        files_jobs_task=primary_jobs_poller_handles.files_jobs_task,
        data_tables_jobs_stop_event=primary_jobs_poller_handles.data_tables_jobs_stop_event,
        data_tables_jobs_task=primary_jobs_poller_handles.data_tables_jobs_task,
        prompt_studio_jobs_stop_event=primary_jobs_poller_handles.prompt_studio_jobs_stop_event,
        prompt_studio_jobs_task=primary_jobs_poller_handles.prompt_studio_jobs_task,
        study_pack_jobs_stop_event=study_privilege_jobs_poller_handles.study_pack_jobs_stop_event,
        study_pack_jobs_task=study_privilege_jobs_poller_handles.study_pack_jobs_task,
        study_suggestions_jobs_stop_event=(study_privilege_jobs_poller_handles.study_suggestions_jobs_stop_event),
        study_suggestions_jobs_task=study_privilege_jobs_poller_handles.study_suggestions_jobs_task,
        privilege_snapshot_stop_event=study_privilege_jobs_poller_handles.privilege_snapshot_stop_event,
        privilege_snapshot_task=study_privilege_jobs_poller_handles.privilege_snapshot_task,
        audio_jobs_stop_event=content_jobs_poller_handles.audio_jobs_stop_event,
        audio_jobs_task=content_jobs_poller_handles.audio_jobs_task,
        audiobook_jobs_stop_event=content_jobs_poller_handles.audiobook_jobs_stop_event,
        audiobook_jobs_task=content_jobs_poller_handles.audiobook_jobs_task,
        presentation_render_jobs_stop_event=(content_jobs_poller_handles.presentation_render_jobs_stop_event),
        presentation_render_jobs_task=content_jobs_poller_handles.presentation_render_jobs_task,
        media_ingest_jobs_stop_event=content_jobs_poller_handles.media_ingest_jobs_stop_event,
        media_ingest_jobs_task=content_jobs_poller_handles.media_ingest_jobs_task,
        media_ingest_heavy_jobs_stop_event=(content_jobs_poller_handles.media_ingest_heavy_jobs_stop_event),
        media_ingest_heavy_jobs_task=content_jobs_poller_handles.media_ingest_heavy_jobs_task,
        reading_digest_jobs_stop_event=content_jobs_poller_handles.reading_digest_jobs_stop_event,
        reading_digest_jobs_task=content_jobs_poller_handles.reading_digest_jobs_task,
        vn_asset_jobs_stop_event=content_jobs_poller_handles.vn_asset_jobs_stop_event,
        vn_asset_jobs_task=content_jobs_poller_handles.vn_asset_jobs_task,
        vn_asset_generation_jobs_stop_event=(content_jobs_poller_handles.vn_asset_generation_jobs_stop_event),
        vn_asset_generation_jobs_task=(content_jobs_poller_handles.vn_asset_generation_jobs_task),
        companion_reflection_jobs_stop_event=(content_jobs_poller_handles.companion_reflection_jobs_stop_event),
        companion_reflection_jobs_task=content_jobs_poller_handles.companion_reflection_jobs_task,
        reminder_jobs_stop_event=sidecar_owned_jobs_poller_handles.reminder_jobs_stop_event,
        reminder_jobs_task=sidecar_owned_jobs_poller_handles.reminder_jobs_task,
        admin_backup_jobs_stop_event=sidecar_owned_jobs_poller_handles.admin_backup_jobs_stop_event,
        admin_backup_jobs_task=sidecar_owned_jobs_poller_handles.admin_backup_jobs_task,
        admin_byok_validation_jobs_stop_event=(sidecar_owned_jobs_poller_handles.admin_byok_validation_jobs_stop_event),
        admin_byok_validation_jobs_task=sidecar_owned_jobs_poller_handles.admin_byok_validation_jobs_task,
        admin_maintenance_rotation_jobs_stop_event=(
            sidecar_owned_jobs_poller_handles.admin_maintenance_rotation_jobs_stop_event
        ),
        admin_maintenance_rotation_jobs_task=(sidecar_owned_jobs_poller_handles.admin_maintenance_rotation_jobs_task),
        recipe_run_jobs_stop_event=sidecar_owned_jobs_poller_handles.recipe_run_jobs_stop_event,
        recipe_run_jobs_task=sidecar_owned_jobs_poller_handles.recipe_run_jobs_task,
        jobs_notifications_bridge_task=notifications_abtest_startup_handles.jobs_notifications_bridge_task,
        evals_abtest_jobs_stop_event=notifications_abtest_startup_handles.evals_abtest_jobs_stop_event,
        evals_abtest_jobs_task=notifications_abtest_startup_handles.evals_abtest_jobs_task,
    )


async def _start_cleanup_workers(
    *,
    app_settings: Mapping[str, Any],
    test_mode: bool,
    worker_inventory: Any | None = None,
) -> Any:
    from tldw_Server_API.app.services.startup_cleanup_workers import start_cleanup_workers

    return await start_cleanup_workers(
        app_settings,
        test_mode=test_mode,
        worker_inventory=worker_inventory,
    )


async def _start_primary_jobs_pollers(**kwargs):
    from tldw_Server_API.app.services.startup_primary_jobs_pollers import (
        start_primary_jobs_pollers,
    )

    return await start_primary_jobs_pollers(**kwargs)


async def _start_study_privilege_jobs_pollers(**kwargs):
    from tldw_Server_API.app.services.startup_study_privilege_jobs_pollers import (
        start_study_privilege_jobs_pollers,
    )

    return await start_study_privilege_jobs_pollers(**kwargs)


async def _start_compactor_websub_workers(**kwargs):
    from tldw_Server_API.app.services.startup_compactor_websub_workers import (
        start_compactor_websub_workers,
    )

    return await start_compactor_websub_workers(**kwargs)


async def _start_content_jobs_pollers(**kwargs):
    from tldw_Server_API.app.services.startup_content_jobs_pollers import (
        start_content_jobs_pollers,
    )

    return await start_content_jobs_pollers(**kwargs)


async def _start_sidecar_owned_jobs_pollers(**kwargs):
    from tldw_Server_API.app.services.startup_sidecar_owned_jobs_pollers import (
        start_sidecar_owned_jobs_pollers,
    )

    return await start_sidecar_owned_jobs_pollers(**kwargs)


async def _start_notifications_abtest_workers(**kwargs):
    from tldw_Server_API.app.services.startup_notifications_abtest_workers import (
        start_notifications_abtest_workers,
    )

    return await start_notifications_abtest_workers(**kwargs)

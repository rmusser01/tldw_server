"""
Primary late-stop worker shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class PrimaryLateStopWorkerHandles:
    """Updated task and stop-event handles for the primary late-stop workers."""

    core_jobs_task: Any | None = None
    core_jobs_stop_event: Any | None = None
    files_jobs_task: Any | None = None
    files_jobs_stop_event: Any | None = None
    data_tables_jobs_task: Any | None = None
    data_tables_jobs_stop_event: Any | None = None
    prompt_studio_jobs_task: Any | None = None
    prompt_studio_jobs_stop_event: Any | None = None
    vllm_management_task: Any | None = None
    vllm_management_stop_event: Any | None = None
    privilege_snapshot_task: Any | None = None
    privilege_snapshot_stop_event: Any | None = None
    audio_jobs_task: Any | None = None
    audio_jobs_stop_event: Any | None = None
    presentation_render_jobs_task: Any | None = None
    presentation_render_jobs_stop_event: Any | None = None


async def shutdown_primary_late_stop_workers(
    *,
    core_jobs_task: Any | None,
    core_jobs_stop_event: Any | None,
    files_jobs_task: Any | None,
    files_jobs_stop_event: Any | None,
    data_tables_jobs_task: Any | None,
    data_tables_jobs_stop_event: Any | None,
    prompt_studio_jobs_task: Any | None,
    prompt_studio_jobs_stop_event: Any | None,
    vllm_management_task: Any | None,
    vllm_management_stop_event: Any | None,
    privilege_snapshot_task: Any | None,
    privilege_snapshot_stop_event: Any | None,
    audio_jobs_task: Any | None,
    audio_jobs_stop_event: Any | None,
    presentation_render_jobs_task: Any | None,
    presentation_render_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> PrimaryLateStopWorkerHandles:
    """Stop the primary single-worker late-stop services in the legacy shutdown order."""
    core_jobs_shutdown_handles = await _shutdown_core_jobs_worker(
        core_jobs_task=core_jobs_task,
        core_jobs_stop_event=core_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    files_jobs_shutdown_handles = await _shutdown_files_jobs_worker(
        files_jobs_task=files_jobs_task,
        files_jobs_stop_event=files_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    data_tables_jobs_shutdown_handles = await _shutdown_data_tables_jobs_worker(
        data_tables_jobs_task=data_tables_jobs_task,
        data_tables_jobs_stop_event=data_tables_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    prompt_studio_jobs_shutdown_handles = await _shutdown_prompt_studio_jobs_worker(
        prompt_studio_jobs_task=prompt_studio_jobs_task,
        prompt_studio_jobs_stop_event=prompt_studio_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    vllm_management_shutdown_handles = await _shutdown_vllm_management_worker(
        vllm_management_task=vllm_management_task,
        vllm_management_stop_event=vllm_management_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    privilege_snapshot_shutdown_handles = await _shutdown_privilege_snapshot_worker(
        privilege_snapshot_task=privilege_snapshot_task,
        privilege_snapshot_stop_event=privilege_snapshot_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    audio_jobs_shutdown_handles = await _shutdown_audio_jobs_worker(
        audio_jobs_task=audio_jobs_task,
        audio_jobs_stop_event=audio_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    presentation_render_shutdown_handles = await _shutdown_presentation_render_jobs_worker(
        presentation_render_jobs_task=presentation_render_jobs_task,
        presentation_render_jobs_stop_event=presentation_render_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return PrimaryLateStopWorkerHandles(
        core_jobs_task=core_jobs_shutdown_handles.core_jobs_task,
        core_jobs_stop_event=core_jobs_shutdown_handles.core_jobs_stop_event,
        files_jobs_task=files_jobs_shutdown_handles.files_jobs_task,
        files_jobs_stop_event=files_jobs_shutdown_handles.files_jobs_stop_event,
        data_tables_jobs_task=data_tables_jobs_shutdown_handles.data_tables_jobs_task,
        data_tables_jobs_stop_event=data_tables_jobs_shutdown_handles.data_tables_jobs_stop_event,
        prompt_studio_jobs_task=prompt_studio_jobs_shutdown_handles.prompt_studio_jobs_task,
        prompt_studio_jobs_stop_event=prompt_studio_jobs_shutdown_handles.prompt_studio_jobs_stop_event,
        vllm_management_task=vllm_management_shutdown_handles.vllm_management_task,
        vllm_management_stop_event=vllm_management_shutdown_handles.vllm_management_stop_event,
        privilege_snapshot_task=privilege_snapshot_shutdown_handles.privilege_snapshot_task,
        privilege_snapshot_stop_event=privilege_snapshot_shutdown_handles.privilege_snapshot_stop_event,
        audio_jobs_task=audio_jobs_shutdown_handles.audio_jobs_task,
        audio_jobs_stop_event=audio_jobs_shutdown_handles.audio_jobs_stop_event,
        presentation_render_jobs_task=presentation_render_shutdown_handles.presentation_render_jobs_task,
        presentation_render_jobs_stop_event=(
            presentation_render_shutdown_handles.presentation_render_jobs_stop_event
        ),
    )


async def run_shutdown_primary_late_stop_workers(
    *,
    core_jobs_task: Any | None,
    core_jobs_stop_event: Any | None,
    files_jobs_task: Any | None,
    files_jobs_stop_event: Any | None,
    data_tables_jobs_task: Any | None,
    data_tables_jobs_stop_event: Any | None,
    prompt_studio_jobs_task: Any | None,
    prompt_studio_jobs_stop_event: Any | None,
    vllm_management_task: Any | None,
    vllm_management_stop_event: Any | None,
    privilege_snapshot_task: Any | None,
    privilege_snapshot_stop_event: Any | None,
    audio_jobs_task: Any | None,
    audio_jobs_stop_event: Any | None,
    presentation_render_jobs_task: Any | None,
    presentation_render_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> PrimaryLateStopWorkerHandles:
    """Run primary late-stop worker shutdown with main-lifespan fallback behavior."""
    try:
        return await shutdown_primary_late_stop_workers(
            core_jobs_task=core_jobs_task,
            core_jobs_stop_event=core_jobs_stop_event,
            files_jobs_task=files_jobs_task,
            files_jobs_stop_event=files_jobs_stop_event,
            data_tables_jobs_task=data_tables_jobs_task,
            data_tables_jobs_stop_event=data_tables_jobs_stop_event,
            prompt_studio_jobs_task=prompt_studio_jobs_task,
            prompt_studio_jobs_stop_event=prompt_studio_jobs_stop_event,
            vllm_management_task=vllm_management_task,
            vllm_management_stop_event=vllm_management_stop_event,
            privilege_snapshot_task=privilege_snapshot_task,
            privilege_snapshot_stop_event=privilege_snapshot_stop_event,
            audio_jobs_task=audio_jobs_task,
            audio_jobs_stop_event=audio_jobs_stop_event,
            presentation_render_jobs_task=presentation_render_jobs_task,
            presentation_render_jobs_stop_event=presentation_render_jobs_stop_event,
            should_run_late_stop=should_run_late_stop,
            guard_exceptions=guard_exceptions,
        )
    except guard_exceptions as exc:
        logger.debug(f"Primary late-stop workers skipped: {exc}")
        return PrimaryLateStopWorkerHandles(
            core_jobs_task=core_jobs_task,
            core_jobs_stop_event=core_jobs_stop_event,
            files_jobs_task=files_jobs_task,
            files_jobs_stop_event=files_jobs_stop_event,
            data_tables_jobs_task=data_tables_jobs_task,
            data_tables_jobs_stop_event=data_tables_jobs_stop_event,
            prompt_studio_jobs_task=prompt_studio_jobs_task,
            prompt_studio_jobs_stop_event=prompt_studio_jobs_stop_event,
            vllm_management_task=vllm_management_task,
            vllm_management_stop_event=vllm_management_stop_event,
            privilege_snapshot_task=privilege_snapshot_task,
            privilege_snapshot_stop_event=privilege_snapshot_stop_event,
            audio_jobs_task=audio_jobs_task,
            audio_jobs_stop_event=audio_jobs_stop_event,
            presentation_render_jobs_task=presentation_render_jobs_task,
            presentation_render_jobs_stop_event=presentation_render_jobs_stop_event,
        )


async def _shutdown_core_jobs_worker(**kwargs):
    from tldw_Server_API.app.services.shutdown_core_jobs_worker import shutdown_core_jobs_worker

    return await shutdown_core_jobs_worker(**kwargs)


async def _shutdown_files_jobs_worker(**kwargs):
    from tldw_Server_API.app.services.shutdown_files_jobs_worker import shutdown_files_jobs_worker

    return await shutdown_files_jobs_worker(**kwargs)


async def _shutdown_data_tables_jobs_worker(**kwargs):
    from tldw_Server_API.app.services.shutdown_data_tables_jobs_worker import (
        shutdown_data_tables_jobs_worker,
    )

    return await shutdown_data_tables_jobs_worker(**kwargs)


async def _shutdown_prompt_studio_jobs_worker(**kwargs):
    from tldw_Server_API.app.services.shutdown_prompt_studio_jobs_worker import (
        shutdown_prompt_studio_jobs_worker,
    )

    return await shutdown_prompt_studio_jobs_worker(**kwargs)


async def _shutdown_vllm_management_worker(**kwargs):
    from tldw_Server_API.app.services.shutdown_vllm_management_worker import (
        shutdown_vllm_management_worker,
    )

    return await shutdown_vllm_management_worker(**kwargs)


async def _shutdown_privilege_snapshot_worker(**kwargs):
    from tldw_Server_API.app.services.shutdown_privilege_snapshot_worker import (
        shutdown_privilege_snapshot_worker,
    )

    return await shutdown_privilege_snapshot_worker(**kwargs)


async def _shutdown_audio_jobs_worker(**kwargs):
    from tldw_Server_API.app.services.shutdown_audio_jobs_worker import shutdown_audio_jobs_worker

    return await shutdown_audio_jobs_worker(**kwargs)


async def _shutdown_presentation_render_jobs_worker(**kwargs):
    from tldw_Server_API.app.services.shutdown_presentation_render_jobs_worker import (
        shutdown_presentation_render_jobs_worker,
    )

    return await shutdown_presentation_render_jobs_worker(**kwargs)

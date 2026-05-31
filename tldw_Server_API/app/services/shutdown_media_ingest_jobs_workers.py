"""
Media ingest job shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class MediaIngestJobsShutdownHandles:
    """Updated media ingest job handles after shutdown processing."""

    media_ingest_jobs_task: Any | None = None
    media_ingest_jobs_stop_event: Any | None = None
    media_ingest_heavy_jobs_task: Any | None = None
    media_ingest_heavy_jobs_stop_event: Any | None = None


async def shutdown_media_ingest_jobs_workers(
    *,
    media_ingest_jobs_task: Any | None,
    media_ingest_jobs_stop_event: Any | None,
    media_ingest_heavy_jobs_task: Any | None,
    media_ingest_heavy_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> MediaIngestJobsShutdownHandles:
    """Stop media ingest workers while preserving legacy late-stop semantics."""
    await _shutdown_media_ingest_jobs_worker(
        task=media_ingest_jobs_task,
        stop_event=media_ingest_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_media_ingest_heavy_jobs_worker(
        task=media_ingest_heavy_jobs_task,
        stop_event=media_ingest_heavy_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return MediaIngestJobsShutdownHandles(
        media_ingest_jobs_task=media_ingest_jobs_task,
        media_ingest_jobs_stop_event=media_ingest_jobs_stop_event,
        media_ingest_heavy_jobs_task=media_ingest_heavy_jobs_task,
        media_ingest_heavy_jobs_stop_event=media_ingest_heavy_jobs_stop_event,
    )


async def _shutdown_media_ingest_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_late_stop_event_worker(
        task_name="media_ingest_jobs_task",
        task=task,
        stop_event=stop_event,
        stop_message="Media Ingest Jobs worker stopped via stop_event",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_media_ingest_heavy_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_late_stop_event_worker(
        task_name="media_ingest_heavy_jobs_task",
        task=task,
        stop_event=stop_event,
        stop_message="Media Ingest Heavy Jobs worker stopped via stop_event",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_late_stop_event_worker(
    *,
    task_name: str,
    task: Any | None,
    stop_event: Any | None,
    stop_message: str,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if task is None:
        return
    if not should_run_late_stop(task_name, task):
        return
    fallback_exceptions = (asyncio.TimeoutError,) + guard_exceptions
    if stop_event is not None:
        try:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
            logger.info(stop_message)
        except fallback_exceptions:
            _safe_cancel_task(task, guard_exceptions=guard_exceptions)
    else:
        _safe_cancel_task(task, guard_exceptions=guard_exceptions)


def _safe_cancel_task(
    task: Any,
    *,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    try:
        task.cancel()
    except guard_exceptions:
        pass


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)

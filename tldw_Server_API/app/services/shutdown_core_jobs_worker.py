"""
Core jobs worker shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class CoreJobsShutdownHandles:
    """Updated core jobs worker handles after shutdown processing."""

    core_jobs_task: Any | None = None
    core_jobs_stop_event: Any | None = None


async def shutdown_core_jobs_worker(
    *,
    core_jobs_task: Any | None,
    core_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> CoreJobsShutdownHandles:
    """Stop the core jobs worker while preserving legacy late-stop semantics."""
    await _shutdown_core_jobs_worker(
        task=core_jobs_task,
        stop_event=core_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return CoreJobsShutdownHandles(
        core_jobs_task=core_jobs_task,
        core_jobs_stop_event=core_jobs_stop_event,
    )


async def _shutdown_core_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if task is None:
        return
    if not should_run_late_stop("core_jobs_task", task):
        return
    fallback_exceptions = (asyncio.TimeoutError,) + guard_exceptions
    if stop_event is not None:
        try:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
            logger.info("Core Jobs worker (Chatbooks) stopped via stop_event")
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

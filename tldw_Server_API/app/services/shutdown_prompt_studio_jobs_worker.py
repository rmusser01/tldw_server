"""
Prompt studio jobs worker shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class PromptStudioJobsShutdownHandles:
    """Updated prompt studio jobs worker handles after shutdown processing."""

    prompt_studio_jobs_task: Any | None = None
    prompt_studio_jobs_stop_event: Any | None = None


async def shutdown_prompt_studio_jobs_worker(
    *,
    prompt_studio_jobs_task: Any | None,
    prompt_studio_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> PromptStudioJobsShutdownHandles:
    """Stop the prompt studio jobs worker while preserving legacy late-stop semantics."""
    await _shutdown_prompt_studio_jobs_worker(
        task=prompt_studio_jobs_task,
        stop_event=prompt_studio_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return PromptStudioJobsShutdownHandles(
        prompt_studio_jobs_task=prompt_studio_jobs_task,
        prompt_studio_jobs_stop_event=prompt_studio_jobs_stop_event,
    )


async def _shutdown_prompt_studio_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if task is None:
        return
    if not should_run_late_stop("prompt_studio_jobs_task", task):
        return
    fallback_exceptions = (asyncio.TimeoutError,) + guard_exceptions
    if stop_event is not None:
        try:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
            logger.info("Prompt Studio Jobs worker stopped via stop_event")
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

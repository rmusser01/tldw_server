"""
Evaluation recipe-run and embeddings A/B shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class RecipeAbtestJobsShutdownHandles:
    """Updated recipe-run and embeddings A/B job handles after shutdown processing."""

    recipe_run_jobs_task: Any | None = None
    recipe_run_jobs_stop_event: Any | None = None
    evals_abtest_jobs_task: Any | None = None
    evals_abtest_jobs_stop_event: Any | None = None


async def shutdown_recipe_abtest_jobs_workers(
    *,
    recipe_run_jobs_task: Any | None,
    recipe_run_jobs_stop_event: Any | None,
    evals_abtest_jobs_task: Any | None,
    evals_abtest_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> RecipeAbtestJobsShutdownHandles:
    """Stop recipe-run and embeddings A/B workers while preserving legacy late-stop semantics."""
    await _shutdown_recipe_run_jobs_worker(
        task=recipe_run_jobs_task,
        stop_event=recipe_run_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_evals_abtest_jobs_worker(
        task=evals_abtest_jobs_task,
        stop_event=evals_abtest_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return RecipeAbtestJobsShutdownHandles(
        recipe_run_jobs_task=recipe_run_jobs_task,
        recipe_run_jobs_stop_event=recipe_run_jobs_stop_event,
        evals_abtest_jobs_task=evals_abtest_jobs_task,
        evals_abtest_jobs_stop_event=evals_abtest_jobs_stop_event,
    )


async def _shutdown_recipe_run_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_late_stop_event_worker(
        task_name="recipe_run_jobs_task",
        task=task,
        stop_event=stop_event,
        stop_message="Evaluation recipe-run Jobs worker stopped via stop_event",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_evals_abtest_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_late_stop_event_worker(
        task_name="evals_abtest_jobs_task",
        task=task,
        stop_event=stop_event,
        stop_message="Embeddings A/B Jobs worker stopped via stop_event",
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
    try:
        if stop_event is not None:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
            logger.info(stop_message)
        else:
            await _cancel_and_wait_for_task(task, guard_exceptions=guard_exceptions)
    except fallback_exceptions:
        await _cancel_and_wait_for_task(task, guard_exceptions=guard_exceptions)


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)


async def _cancel_and_wait_for_task(
    task: Any,
    *,
    guard_exceptions: tuple[type[BaseException], ...],
    timeout: float = 5.0,
) -> None:
    try:
        task.cancel()
    except guard_exceptions:
        return
    try:
        await _wait_for_task(task, timeout=timeout)
    except asyncio.CancelledError:
        pass
    except (asyncio.TimeoutError,) + guard_exceptions:
        pass

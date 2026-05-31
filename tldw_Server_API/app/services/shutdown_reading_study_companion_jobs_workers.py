"""
Reading, study, and companion job shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class ReadingStudyCompanionJobsShutdownHandles:
    """Updated reading, study, and companion job handles after shutdown processing."""

    reading_digest_jobs_task: Any | None = None
    reading_digest_jobs_stop_event: Any | None = None
    study_pack_jobs_task: Any | None = None
    study_pack_jobs_stop_event: Any | None = None
    study_suggestions_jobs_task: Any | None = None
    study_suggestions_jobs_stop_event: Any | None = None
    companion_reflection_jobs_task: Any | None = None
    companion_reflection_jobs_stop_event: Any | None = None


async def shutdown_reading_study_companion_jobs_workers(
    *,
    reading_digest_jobs_task: Any | None,
    reading_digest_jobs_stop_event: Any | None,
    study_pack_jobs_task: Any | None,
    study_pack_jobs_stop_event: Any | None,
    study_suggestions_jobs_task: Any | None,
    study_suggestions_jobs_stop_event: Any | None,
    companion_reflection_jobs_task: Any | None,
    companion_reflection_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> ReadingStudyCompanionJobsShutdownHandles:
    """Stop reading, study, and companion workers while preserving legacy late-stop semantics."""
    await _shutdown_reading_digest_jobs_worker(
        task=reading_digest_jobs_task,
        stop_event=reading_digest_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_study_pack_jobs_worker(
        task=study_pack_jobs_task,
        stop_event=study_pack_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_study_suggestions_jobs_worker(
        task=study_suggestions_jobs_task,
        stop_event=study_suggestions_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_companion_reflection_jobs_worker(
        task=companion_reflection_jobs_task,
        stop_event=companion_reflection_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return ReadingStudyCompanionJobsShutdownHandles(
        reading_digest_jobs_task=reading_digest_jobs_task,
        reading_digest_jobs_stop_event=reading_digest_jobs_stop_event,
        study_pack_jobs_task=study_pack_jobs_task,
        study_pack_jobs_stop_event=study_pack_jobs_stop_event,
        study_suggestions_jobs_task=study_suggestions_jobs_task,
        study_suggestions_jobs_stop_event=study_suggestions_jobs_stop_event,
        companion_reflection_jobs_task=companion_reflection_jobs_task,
        companion_reflection_jobs_stop_event=companion_reflection_jobs_stop_event,
    )


async def _shutdown_reading_digest_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_late_stop_event_worker(
        task_name="reading_digest_jobs_task",
        task=task,
        stop_event=stop_event,
        stop_message="Reading digest Jobs worker stopped via stop_event",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_study_pack_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_late_stop_event_worker(
        task_name="study_pack_jobs_task",
        task=task,
        stop_event=stop_event,
        stop_message="Study-pack Jobs worker stopped via stop_event",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_study_suggestions_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_late_stop_event_worker(
        task_name="study_suggestions_jobs_task",
        task=task,
        stop_event=stop_event,
        stop_message="Study-suggestions Jobs worker stopped via stop_event",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_companion_reflection_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_late_stop_event_worker(
        task_name="companion_reflection_jobs_task",
        task=task,
        stop_event=stop_event,
        stop_message="Companion reflection Jobs worker stopped via stop_event",
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
    if not should_run_late_stop(task_name, task):
        return
    if stop_event:
        try:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
            logger.info(stop_message)
        except guard_exceptions:
            task.cancel()
    else:
        task.cancel()


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)

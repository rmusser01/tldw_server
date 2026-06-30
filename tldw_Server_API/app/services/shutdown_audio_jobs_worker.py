"""
Audio jobs worker shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class AudioJobsShutdownHandles:
    """Updated audio jobs worker handles after shutdown processing."""

    audio_jobs_task: Any | None = None
    audio_jobs_stop_event: Any | None = None


async def shutdown_audio_jobs_worker(
    *,
    audio_jobs_task: Any | None,
    audio_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> AudioJobsShutdownHandles:
    """Stop the audio jobs worker while preserving legacy late-stop semantics."""
    await _shutdown_audio_jobs_worker(
        task=audio_jobs_task,
        stop_event=audio_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return AudioJobsShutdownHandles(
        audio_jobs_task=audio_jobs_task,
        audio_jobs_stop_event=audio_jobs_stop_event,
    )


async def _shutdown_audio_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if not should_run_late_stop("audio_jobs_task", task):
        return
    if stop_event:
        try:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
            logger.info("Audio Jobs worker stopped via stop_event")
        except asyncio.CancelledError:
            raise
        except guard_exceptions:
            task.cancel()
        except Exception as exc:
            logger.warning(
                f"Audio Jobs worker exited with exception before shutdown completion: {exc}"
            )
            with suppress(*guard_exceptions):
                task.cancel()
    else:
        task.cancel()


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)

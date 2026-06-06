"""Calendar sync worker shutdown helper."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class CalendarSyncShutdownHandles:
    """Updated Calendar sync handles after shutdown processing."""

    calendar_sync_jobs_task: Any | None = None
    calendar_sync_jobs_stop_event: Any | None = None
    calendar_sync_scheduler_task: Any | None = None
    calendar_sync_scheduler_stop_event: Any | None = None


async def shutdown_calendar_sync_worker(
    *,
    calendar_sync_jobs_task: Any | None,
    calendar_sync_jobs_stop_event: Any | None,
    calendar_sync_scheduler_task: Any | None = None,
    calendar_sync_scheduler_stop_event: Any | None = None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> CalendarSyncShutdownHandles:
    """Stop Calendar sync worker and scheduler tasks when managed outside worker inventory."""
    await _shutdown_stop_event_task(
        task_name="calendar_sync_jobs_task",
        task=calendar_sync_jobs_task,
        stop_event=calendar_sync_jobs_stop_event,
        stop_message="Calendar sync Jobs worker stopped via stop_event",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_stop_event_task(
        task_name="calendar_sync_scheduler_task",
        task=calendar_sync_scheduler_task,
        stop_event=calendar_sync_scheduler_stop_event,
        stop_message="Calendar sync scheduler stopped via stop_event",
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return CalendarSyncShutdownHandles(
        calendar_sync_jobs_task=calendar_sync_jobs_task,
        calendar_sync_jobs_stop_event=calendar_sync_jobs_stop_event,
        calendar_sync_scheduler_task=calendar_sync_scheduler_task,
        calendar_sync_scheduler_stop_event=calendar_sync_scheduler_stop_event,
    )


async def _shutdown_stop_event_task(
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
            await asyncio.wait_for(task, timeout=5.0)
            logger.info(stop_message)
        except asyncio.CancelledError:
            raise
        except guard_exceptions:
            task.cancel()
        except Exception as exc:
            logger.warning("{} exited with exception before shutdown completion: {}", task_name, exc)
            with suppress(*guard_exceptions):
                task.cancel()
    else:
        task.cancel()


__all__ = [
    "CalendarSyncShutdownHandles",
    "shutdown_calendar_sync_worker",
]

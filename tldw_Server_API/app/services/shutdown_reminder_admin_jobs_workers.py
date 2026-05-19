"""
Reminder and admin job shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class ReminderAdminJobsShutdownHandles:
    """Updated reminder and admin job handles after shutdown processing."""

    reminder_jobs_task: Any | None = None
    admin_backup_jobs_task: Any | None = None
    admin_maintenance_rotation_jobs_task: Any | None = None
    admin_maintenance_rotation_jobs_stop_event: Any | None = None


async def shutdown_reminder_admin_jobs_workers(
    *,
    reminder_jobs_task: Any | None,
    admin_backup_jobs_task: Any | None,
    admin_maintenance_rotation_jobs_task: Any | None,
    admin_maintenance_rotation_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> ReminderAdminJobsShutdownHandles:
    """Stop reminder and admin workers while preserving legacy late-stop semantics."""
    await _shutdown_reminder_jobs_worker(
        task=reminder_jobs_task,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_admin_backup_jobs_worker(
        task=admin_backup_jobs_task,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_admin_maintenance_rotation_jobs_worker(
        task=admin_maintenance_rotation_jobs_task,
        stop_event=admin_maintenance_rotation_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return ReminderAdminJobsShutdownHandles(
        reminder_jobs_task=reminder_jobs_task,
        admin_backup_jobs_task=admin_backup_jobs_task,
        admin_maintenance_rotation_jobs_task=admin_maintenance_rotation_jobs_task,
        admin_maintenance_rotation_jobs_stop_event=admin_maintenance_rotation_jobs_stop_event,
    )


async def _shutdown_reminder_jobs_worker(
    *,
    task: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if task is None:
        return
    if not should_run_late_stop("reminder_jobs_task", task):
        return
    try:
        task.cancel()
        await _wait_for_task(task, timeout=5.0)
        logger.info("Reminder Jobs worker cancelled")
    except asyncio.CancelledError:
        pass
    except (asyncio.TimeoutError,) + guard_exceptions:
        await _cancel_and_wait_for_task(task, guard_exceptions=guard_exceptions)


async def _shutdown_admin_backup_jobs_worker(
    *,
    task: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if task is None:
        return
    if not should_run_late_stop("admin_backup_jobs_task", task):
        return
    try:
        task.cancel()
        await _wait_for_task(task, timeout=5.0)
        logger.info("Admin backup Jobs worker cancelled")
    except asyncio.CancelledError:
        pass
    except (asyncio.TimeoutError,) + guard_exceptions:
        await _cancel_and_wait_for_task(task, guard_exceptions=guard_exceptions)


async def _shutdown_admin_maintenance_rotation_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if task is None:
        return
    if not should_run_late_stop("admin_maintenance_rotation_jobs_task", task):
        return
    fallback_exceptions = (asyncio.TimeoutError,) + guard_exceptions
    if stop_event is not None:
        try:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
            logger.info("Admin maintenance rotation Jobs worker stopped via stop_event")
        except fallback_exceptions:
            await _cancel_and_wait_for_task(task, guard_exceptions=guard_exceptions)
    else:
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

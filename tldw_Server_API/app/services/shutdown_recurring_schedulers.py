"""
Recurring-scheduler shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from typing import Any


_SHUTDOWN_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


async def stop_recurring_schedulers(
    *,
    workflows_sched_task: Any | None,
    reading_digest_sched_task: Any | None,
    admin_backup_sched_task: Any | None,
    companion_reflection_sched_task: Any | None,
    reminders_sched_task: Any | None,
    connectors_sync_sched_task: Any | None,
) -> None:
    """Stop the recurring scheduler batch in the same order used by lifespan shutdown."""
    await _stop_workflows_scheduler(workflows_sched_task)
    await _stop_reading_digest_scheduler(reading_digest_sched_task)
    await _stop_admin_backup_scheduler(admin_backup_sched_task)
    await _stop_companion_reflection_scheduler(companion_reflection_sched_task)
    await _stop_reminders_scheduler(reminders_sched_task)
    await _stop_connectors_sync_scheduler(connectors_sync_sched_task)


async def _stop_workflows_scheduler(task: Any | None) -> None:
    if not task:
        return
    try:
        await _stop_workflows_scheduler_service(task)
    except _SHUTDOWN_GUARD_EXCEPTIONS:
        try:
            task.cancel()
        except _SHUTDOWN_GUARD_EXCEPTIONS:
            pass


async def _stop_reading_digest_scheduler(task: Any | None) -> None:
    if not task:
        return
    try:
        await _stop_reading_digest_scheduler_service(task)
    except _SHUTDOWN_GUARD_EXCEPTIONS:
        try:
            task.cancel()
        except _SHUTDOWN_GUARD_EXCEPTIONS:
            pass


async def _stop_admin_backup_scheduler(task: Any | None) -> None:
    if not task:
        return
    try:
        await _stop_admin_backup_scheduler_service(task)
    except _SHUTDOWN_GUARD_EXCEPTIONS:
        try:
            task.cancel()
        except _SHUTDOWN_GUARD_EXCEPTIONS:
            pass


async def _stop_companion_reflection_scheduler(task: Any | None) -> None:
    if not task:
        return
    try:
        await _stop_companion_reflection_scheduler_service(task)
    except _SHUTDOWN_GUARD_EXCEPTIONS:
        try:
            task.cancel()
        except _SHUTDOWN_GUARD_EXCEPTIONS:
            pass


async def _stop_reminders_scheduler(task: Any | None) -> None:
    try:
        await _stop_reminders_scheduler_service(task)
    except _SHUTDOWN_GUARD_EXCEPTIONS:
        try:
            if task:
                task.cancel()
        except _SHUTDOWN_GUARD_EXCEPTIONS:
            pass


async def _stop_connectors_sync_scheduler(task: Any | None) -> None:
    try:
        await _stop_connectors_sync_scheduler_service(task)
    except _SHUTDOWN_GUARD_EXCEPTIONS:
        try:
            if task:
                task.cancel()
        except _SHUTDOWN_GUARD_EXCEPTIONS:
            pass


async def _stop_workflows_scheduler_service(task: Any) -> None:
    from tldw_Server_API.app.services.workflows_scheduler import stop_workflows_scheduler

    await stop_workflows_scheduler(task)


async def _stop_reading_digest_scheduler_service(task: Any) -> None:
    from tldw_Server_API.app.services.reading_digest_scheduler import stop_reading_digest_scheduler

    await stop_reading_digest_scheduler(task)


async def _stop_admin_backup_scheduler_service(task: Any) -> None:
    from tldw_Server_API.app.services.admin_backup_scheduler import stop_admin_backup_scheduler

    await stop_admin_backup_scheduler(task)


async def _stop_companion_reflection_scheduler_service(task: Any) -> None:
    from tldw_Server_API.app.services.companion_reflection_scheduler import (
        stop_companion_reflection_scheduler,
    )

    await stop_companion_reflection_scheduler(task)


async def _stop_reminders_scheduler_service(task: Any | None) -> None:
    from tldw_Server_API.app.services.reminders_scheduler import stop_reminders_scheduler

    await stop_reminders_scheduler(task)


async def _stop_connectors_sync_scheduler_service(task: Any | None) -> None:
    from tldw_Server_API.app.services.connectors_sync_scheduler import (
        stop_connectors_sync_scheduler,
    )

    await stop_connectors_sync_scheduler(task)

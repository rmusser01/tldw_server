"""
Recurring-scheduler startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import env_flag_enabled as _env_flag_enabled
from tldw_Server_API.app.core.testing import is_truthy as _is_truthy

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class RecurringSchedulerHandles:
    """Scheduler startup handles that must remain available to the lifespan."""

    authnz_scheduler_started: bool = False
    workflows_sched_task: Any | None = None
    reading_digest_sched_task: Any | None = None
    admin_backup_sched_task: Any | None = None
    companion_reflection_sched_task: Any | None = None
    reminders_sched_task: Any | None = None
    connectors_sync_sched_task: Any | None = None


async def start_recurring_schedulers(*, test_mode: bool) -> RecurringSchedulerHandles:
    """Start the recurring scheduler batch and return explicit scheduler handles."""
    return RecurringSchedulerHandles(
        authnz_scheduler_started=await _start_authnz_scheduler(),
        workflows_sched_task=await _start_workflows_scheduler(),
        reading_digest_sched_task=await _start_reading_digest_scheduler(test_mode=test_mode),
        admin_backup_sched_task=await _start_admin_backup_scheduler(),
        companion_reflection_sched_task=await _start_companion_reflection_scheduler(),
        reminders_sched_task=await _start_reminders_scheduler(),
        connectors_sync_sched_task=await _start_connectors_sync_scheduler(),
    )


def _env_flag(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"true", "1", "yes", "y", "on"}


async def _start_authnz_scheduler() -> bool:
    try:
        if _env_flag_enabled("DISABLE_AUTHNZ_SCHEDULER"):
            logger.info("AuthNZ scheduler disabled via DISABLE_AUTHNZ_SCHEDULER env var")
            return False
        await _start_authnz_scheduler_service()
        logger.info("AuthNZ scheduler started")
        return True
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start AuthNZ scheduler: {exc}")
        return False


async def _start_workflows_scheduler() -> Any | None:
    return await _start_optional_scheduler(
        started_message="Workflows recurring scheduler started",
        disabled_message="Workflows recurring scheduler disabled (WORKFLOWS_SCHEDULER_ENABLED != true)",
        failure_message="Failed to start Workflows recurring scheduler: {exc}",
        starter=_start_workflows_scheduler_service,
    )


async def _start_reading_digest_scheduler(*, test_mode: bool) -> Any | None:
    try:
        try:
            enabled = _env_flag("READING_DIGEST_SCHEDULER_ENABLED", True)
            if test_mode and os.getenv("READING_DIGEST_SCHEDULER_ENABLED") is None:
                enabled = False
        except _STARTUP_GUARD_EXCEPTIONS:
            enabled = _is_truthy(os.getenv("READING_DIGEST_SCHEDULER_ENABLED", "true"))
        if not enabled:
            logger.info("Reading digest scheduler disabled (READING_DIGEST_SCHEDULER_ENABLED != true)")
            return None
        task = await _start_reading_digest_scheduler_service(enabled=True)
        if task:
            logger.info("Reading digest scheduler started")
        else:
            logger.info("Reading digest scheduler disabled (READING_DIGEST_SCHEDULER_ENABLED != true)")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Reading digest scheduler: {exc}")
        return None


async def _start_admin_backup_scheduler() -> Any | None:
    return await _start_optional_scheduler(
        started_message="Admin backup scheduler started",
        disabled_message="Admin backup scheduler disabled (ADMIN_BACKUP_SCHEDULER_ENABLED != true)",
        failure_message="Failed to start Admin backup scheduler: {exc}",
        starter=_start_admin_backup_scheduler_service,
    )


async def _start_companion_reflection_scheduler() -> Any | None:
    try:
        try:
            enabled = _env_flag("COMPANION_REFLECTION_SCHEDULER_ENABLED", False)
        except _STARTUP_GUARD_EXCEPTIONS:
            enabled = _is_truthy(os.getenv("COMPANION_REFLECTION_SCHEDULER_ENABLED", "false"))
        if not enabled:
            logger.info(
                "Companion reflection scheduler disabled (COMPANION_REFLECTION_SCHEDULER_ENABLED != true)"
            )
            return None
        task = await _start_companion_reflection_scheduler_service(enabled=True)
        if task:
            logger.info("Companion reflection scheduler started")
        else:
            logger.info(
                "Companion reflection scheduler disabled (COMPANION_REFLECTION_SCHEDULER_ENABLED != true)"
            )
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Companion reflection scheduler: {exc}")
        return None


async def _start_reminders_scheduler() -> Any | None:
    return await _start_optional_scheduler(
        started_message="Reminders scheduler started",
        disabled_message="Reminders scheduler disabled (REMINDERS_SCHEDULER_ENABLED != true)",
        failure_message="Failed to start Reminders scheduler: {exc}",
        starter=_start_reminders_scheduler_service,
    )


async def _start_connectors_sync_scheduler() -> Any | None:
    return await _start_optional_scheduler(
        started_message="Connectors sync scheduler started",
        disabled_message="Connectors sync scheduler disabled (CONNECTORS_SYNC_SCHEDULER_ENABLED != true)",
        failure_message="Failed to start Connectors sync scheduler: {exc}",
        starter=_start_connectors_sync_scheduler_service,
    )


async def _start_optional_scheduler(
    *,
    started_message: str,
    disabled_message: str,
    failure_message: str,
    starter,
) -> Any | None:
    try:
        task = await starter()
        if task:
            logger.info(started_message)
        else:
            logger.info(disabled_message)
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(failure_message.format(exc=exc))
        return None


async def _start_authnz_scheduler_service() -> None:
    from tldw_Server_API.app.core.AuthNZ.scheduler import start_authnz_scheduler

    await start_authnz_scheduler()


async def _start_workflows_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.workflows_scheduler import start_workflows_scheduler

    return await start_workflows_scheduler()


async def _start_reading_digest_scheduler_service(*, enabled: bool) -> Any | None:
    from tldw_Server_API.app.services.reading_digest_scheduler import start_reading_digest_scheduler

    return await start_reading_digest_scheduler(enabled=enabled)


async def _start_admin_backup_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.admin_backup_scheduler import start_admin_backup_scheduler

    return await start_admin_backup_scheduler()


async def _start_companion_reflection_scheduler_service(*, enabled: bool) -> Any | None:
    from tldw_Server_API.app.services.companion_reflection_scheduler import (
        start_companion_reflection_scheduler,
    )

    return await start_companion_reflection_scheduler(enabled=enabled)


async def _start_reminders_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.reminders_scheduler import start_reminders_scheduler

    return await start_reminders_scheduler()


async def _start_connectors_sync_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.connectors_sync_scheduler import (
        start_connectors_sync_scheduler,
    )

    return await start_connectors_sync_scheduler()

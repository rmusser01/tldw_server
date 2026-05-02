"""
Recurring-scheduler startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
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
_SCHEDULER_ROLLBACK_TIMEOUT_SEC = 1.0

SchedulerHandle = Any
SchedulerStarter = Callable[[], Awaitable[SchedulerHandle | None]]
InventorySchedulerStarter = Callable[..., Awaitable[SchedulerHandle | None]]
SchedulerStopper = Callable[[SchedulerHandle], Awaitable[None]]
SchedulerShutdownCallback = Callable[[], Awaitable[None]]


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


async def start_recurring_schedulers(
    *,
    test_mode: bool,
    worker_inventory: Any | None = None,
) -> RecurringSchedulerHandles:
    """Start the recurring scheduler batch and return explicit scheduler handles."""
    return RecurringSchedulerHandles(
        authnz_scheduler_started=await _start_authnz_scheduler(
            worker_inventory=worker_inventory,
        ),
        workflows_sched_task=await _start_with_optional_inventory(
            _start_workflows_scheduler,
            worker_inventory,
        ),
        reading_digest_sched_task=await _start_reading_digest_with_optional_inventory(
            test_mode=test_mode,
            worker_inventory=worker_inventory,
        ),
        admin_backup_sched_task=await _start_with_optional_inventory(
            _start_admin_backup_scheduler,
            worker_inventory,
        ),
        companion_reflection_sched_task=await _start_with_optional_inventory(
            _start_companion_reflection_scheduler,
            worker_inventory,
        ),
        reminders_sched_task=await _start_with_optional_inventory(
            _start_reminders_scheduler,
            worker_inventory,
        ),
        connectors_sync_sched_task=await _start_with_optional_inventory(
            _start_connectors_sync_scheduler,
            worker_inventory,
        ),
    )


def _env_flag(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in {"true", "1", "yes", "y", "on"}


async def _start_with_optional_inventory(
    starter: InventorySchedulerStarter,
    worker_inventory: Any | None,
) -> SchedulerHandle | None:
    """Start a scheduler with inventory when the caller provided one."""

    if worker_inventory is None:
        return await starter()
    return await starter(worker_inventory=worker_inventory)


async def _start_reading_digest_with_optional_inventory(
    *,
    test_mode: bool,
    worker_inventory: Any | None,
) -> SchedulerHandle | None:
    """Start the reading digest scheduler while preserving its test-mode flag."""

    if worker_inventory is None:
        return await _start_reading_digest_scheduler(test_mode=test_mode)
    return await _start_reading_digest_scheduler(
        test_mode=test_mode,
        worker_inventory=worker_inventory,
    )


async def _start_authnz_scheduler(
    *,
    worker_inventory: Any | None = None,
) -> bool:
    try:
        if _env_flag_enabled("DISABLE_AUTHNZ_SCHEDULER"):
            logger.info("AuthNZ scheduler disabled via DISABLE_AUTHNZ_SCHEDULER env var")
            return False
        await _start_authnz_scheduler_service()
        try:
            await _register_authnz_scheduler(worker_inventory=worker_inventory)
        except _STARTUP_GUARD_EXCEPTIONS:
            await _stop_authnz_scheduler_after_registration_failure()
            raise
        logger.info("AuthNZ scheduler started")
        return True
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start AuthNZ scheduler: {exc}")
        return False


async def _register_authnz_scheduler(
    *,
    worker_inventory: Any | None,
) -> None:
    """Register AuthNZ scheduler as callback-only lifecycle work."""

    if worker_inventory is None:
        return

    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        ShutdownPhase,
    )

    worker_inventory.register(
        ManagedWorker(
            name="authnz_scheduler",
            task=None,
            stop_event=None,
            shutdown_callback=_stop_authnz_scheduler_service,
            category="recurring-scheduler",
            shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        )
    )


async def _stop_authnz_scheduler_after_registration_failure() -> None:
    try:
        await _stop_authnz_scheduler_service()
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"AuthNZ scheduler startup rollback stop failed: {exc}")


async def _start_workflows_scheduler(
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    return await _start_optional_scheduler(
        started_message="Workflows recurring scheduler started",
        disabled_message="Workflows recurring scheduler disabled (WORKFLOWS_SCHEDULER_ENABLED != true)",
        failure_message="Failed to start Workflows recurring scheduler: {exc}",
        starter=_start_workflows_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="workflows_sched_task",
        stopper=_stop_workflows_scheduler_service,
    )


async def _start_reading_digest_scheduler(
    *,
    test_mode: bool,
    worker_inventory: Any | None = None,
) -> Any | None:
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
            await _register_recurring_scheduler_task(
                worker_inventory=worker_inventory,
                task=task,
                worker_name="reading_digest_sched_task",
                stopper=_stop_reading_digest_scheduler_service,
            )
            logger.info("Reading digest scheduler started")
        else:
            logger.info("Reading digest scheduler disabled (READING_DIGEST_SCHEDULER_ENABLED != true)")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Reading digest scheduler: {exc}")
        return None


async def _start_admin_backup_scheduler(
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    return await _start_optional_scheduler(
        started_message="Admin backup scheduler started",
        disabled_message="Admin backup scheduler disabled (ADMIN_BACKUP_SCHEDULER_ENABLED != true)",
        failure_message="Failed to start Admin backup scheduler: {exc}",
        starter=_start_admin_backup_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="admin_backup_sched_task",
        stopper=_stop_admin_backup_scheduler_service,
    )


async def _start_companion_reflection_scheduler(
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
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
            await _register_recurring_scheduler_task(
                worker_inventory=worker_inventory,
                task=task,
                worker_name="companion_reflection_sched_task",
                stopper=_stop_companion_reflection_scheduler_service,
            )
            logger.info("Companion reflection scheduler started")
        else:
            logger.info(
                "Companion reflection scheduler disabled (COMPANION_REFLECTION_SCHEDULER_ENABLED != true)"
            )
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Companion reflection scheduler: {exc}")
        return None


async def _start_reminders_scheduler(
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    return await _start_optional_scheduler(
        started_message="Reminders scheduler started",
        disabled_message="Reminders scheduler disabled (REMINDERS_SCHEDULER_ENABLED != true)",
        failure_message="Failed to start Reminders scheduler: {exc}",
        starter=_start_reminders_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="reminders_sched_task",
        stopper=_stop_reminders_scheduler_service,
    )


async def _start_connectors_sync_scheduler(
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    return await _start_optional_scheduler(
        started_message="Connectors sync scheduler started",
        disabled_message="Connectors sync scheduler disabled (CONNECTORS_SYNC_SCHEDULER_ENABLED != true)",
        failure_message="Failed to start Connectors sync scheduler: {exc}",
        starter=_start_connectors_sync_scheduler_service,
        worker_inventory=worker_inventory,
        worker_name="connectors_sync_sched_task",
        stopper=_stop_connectors_sync_scheduler_service,
    )


async def _start_optional_scheduler(
    *,
    started_message: str,
    disabled_message: str,
    failure_message: str,
    starter: SchedulerStarter,
    worker_inventory: Any | None = None,
    worker_name: str | None = None,
    stopper: SchedulerStopper | None = None,
) -> SchedulerHandle | None:
    """Start an optional scheduler and register it for managed shutdown."""

    try:
        task = await starter()
        if task:
            await _register_recurring_scheduler_task(
                worker_inventory=worker_inventory,
                task=task,
                worker_name=worker_name,
                stopper=stopper,
            )
            logger.info(started_message)
        else:
            logger.info(disabled_message)
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(failure_message.format(exc=exc))
        return None


async def _register_recurring_scheduler_task(
    *,
    worker_inventory: Any | None,
    task: SchedulerHandle,
    worker_name: str | None,
    stopper: SchedulerStopper | None,
) -> None:
    """Register a started recurring scheduler with the lifecycle worker registry."""

    if worker_inventory is None:
        return
    if worker_name is None or stopper is None:
        logger.warning(
            "Recurring scheduler registration skipped for task {}: missing worker name or stopper",
            task,
        )
        return

    from tldw_Server_API.app.services.lifecycle_workers import (
        ManagedWorker,
        ShutdownPhase,
    )

    try:
        worker_inventory.register(
            ManagedWorker(
                name=worker_name,
                task=task,
                stop_event=None,
                shutdown_callback=_build_scheduler_shutdown_callback(
                    task=task,
                    stopper=stopper,
                ),
                category="recurring-scheduler",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
        )
    except _STARTUP_GUARD_EXCEPTIONS:
        await _cancel_unregistered_task(task)
        raise


def _build_scheduler_shutdown_callback(
    *,
    task: SchedulerHandle,
    stopper: SchedulerStopper,
) -> SchedulerShutdownCallback:
    """Build a no-argument lifecycle callback for a scheduler stop function."""

    async def _shutdown() -> None:
        await stopper(task)

    return _shutdown


async def _cancel_unregistered_task(task: SchedulerHandle) -> None:
    """Cancel a scheduler that could not be registered without blocking startup."""

    try:
        task.cancel()
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"Recurring scheduler startup rollback cancel failed: {exc}")
        return

    done, pending = await asyncio.wait(
        {task},
        timeout=_SCHEDULER_ROLLBACK_TIMEOUT_SEC,
    )
    if pending:
        logger.warning(
            "Recurring scheduler startup rollback timed out after {}s",
            _SCHEDULER_ROLLBACK_TIMEOUT_SEC,
        )
        return

    try:
        done.pop().result()
    except asyncio.CancelledError:
        pass
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"Recurring scheduler raised during startup rollback: {exc}")


async def _start_authnz_scheduler_service() -> None:
    from tldw_Server_API.app.core.AuthNZ.scheduler import start_authnz_scheduler

    await start_authnz_scheduler()


async def _stop_authnz_scheduler_service() -> None:
    from tldw_Server_API.app.core.AuthNZ.scheduler import stop_authnz_scheduler

    await stop_authnz_scheduler()


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

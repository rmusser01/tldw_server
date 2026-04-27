"""
Sidecar-gated owned jobs poller startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class SidecarOwnedJobsPollerHandles:
    """Startup-owned sidecar-gated jobs poller handles used later in shutdown flow."""

    reminder_jobs_stop_event: Any | None = None
    reminder_jobs_task: Any | None = None
    admin_backup_jobs_stop_event: Any | None = None
    admin_backup_jobs_task: Any | None = None
    admin_byok_validation_jobs_stop_event: Any | None = None
    admin_byok_validation_jobs_task: Any | None = None
    admin_maintenance_rotation_jobs_stop_event: Any | None = None
    admin_maintenance_rotation_jobs_task: Any | None = None
    recipe_run_jobs_stop_event: Any | None = None
    recipe_run_jobs_task: Any | None = None


async def start_sidecar_owned_jobs_pollers(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
) -> SidecarOwnedJobsPollerHandles:
    """Start sidecar-gated owned jobs pollers and return their handles."""

    reminder_jobs_stop_event, reminder_jobs_task = await _start_reminder_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
    )
    admin_backup_jobs_stop_event, admin_backup_jobs_task = await _start_admin_backup_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
    )
    (
        admin_byok_validation_jobs_stop_event,
        admin_byok_validation_jobs_task,
    ) = await _start_admin_byok_validation_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
    )
    (
        admin_maintenance_rotation_jobs_stop_event,
        admin_maintenance_rotation_jobs_task,
    ) = await _start_admin_maintenance_rotation_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
    )
    recipe_run_jobs_stop_event, recipe_run_jobs_task = await _start_recipe_run_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
    )
    return SidecarOwnedJobsPollerHandles(
        reminder_jobs_stop_event=reminder_jobs_stop_event,
        reminder_jobs_task=reminder_jobs_task,
        admin_backup_jobs_stop_event=admin_backup_jobs_stop_event,
        admin_backup_jobs_task=admin_backup_jobs_task,
        admin_byok_validation_jobs_stop_event=admin_byok_validation_jobs_stop_event,
        admin_byok_validation_jobs_task=admin_byok_validation_jobs_task,
        admin_maintenance_rotation_jobs_stop_event=admin_maintenance_rotation_jobs_stop_event,
        admin_maintenance_rotation_jobs_task=admin_maintenance_rotation_jobs_task,
        recipe_run_jobs_stop_event=recipe_run_jobs_stop_event,
        recipe_run_jobs_task=recipe_run_jobs_task,
    )


def _make_event() -> Any:
    return asyncio.Event()


async def _resolve_result(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


async def _start_reminder_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Reminder Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_reminder_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Reminder Jobs worker started")
            register_owned_job_poller(
                app,
                owned_job_pollers,
                name="reminder_jobs_task",
                task=task,
                stop_event=stop_event,
            )
        else:
            logger.info("Reminder Jobs worker disabled (REMINDER_JOBS_WORKER_ENABLED != true)")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Reminder Jobs worker: {exc}")
        return None, None


async def _start_admin_backup_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Admin backup Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_admin_backup_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Admin backup Jobs worker started")
            register_owned_job_poller(
                app,
                owned_job_pollers,
                name="admin_backup_jobs_task",
                task=task,
                stop_event=stop_event,
            )
        else:
            logger.info("Admin backup Jobs worker disabled (ADMIN_BACKUP_JOBS_WORKER_ENABLED != true)")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Admin backup Jobs worker: {exc}")
        return None, None


async def _start_admin_byok_validation_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Admin BYOK validation Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_admin_byok_validation_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Admin BYOK validation Jobs worker started")
            register_owned_job_poller(
                app,
                owned_job_pollers,
                name="admin_byok_validation_jobs_task",
                task=task,
                stop_event=stop_event,
            )
        else:
            logger.info(
                "Admin BYOK validation Jobs worker disabled "
                "(ADMIN_BYOK_VALIDATION_JOBS_WORKER_ENABLED != true)"
            )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Admin BYOK validation Jobs worker: {exc}")
        return None, None


async def _start_admin_maintenance_rotation_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Admin maintenance rotation Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_admin_maintenance_rotation_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Admin maintenance rotation Jobs worker started")
            register_owned_job_poller(
                app,
                owned_job_pollers,
                name="admin_maintenance_rotation_jobs_task",
                task=task,
                stop_event=stop_event,
            )
        else:
            logger.info(
                "Admin maintenance rotation Jobs worker disabled "
                "(ADMIN_MAINTENANCE_ROTATION_JOBS_WORKER_ENABLED != true)"
            )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Admin maintenance rotation Jobs worker: {exc}")
        return None, None


async def _start_recipe_run_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Evaluation recipe-run Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_recipe_run_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Evaluation recipe-run Jobs worker started")
            register_owned_job_poller(
                app,
                owned_job_pollers,
                name="recipe_run_jobs_task",
                task=task,
                stop_event=stop_event,
            )
        else:
            logger.info(
                "Evaluation recipe-run Jobs worker disabled "
                "(EVALUATIONS_RECIPE_RUN_JOBS_WORKER_ENABLED != true)"
            )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start evaluation recipe-run Jobs worker: {exc}")
        return None, None


def _start_reminder_jobs_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.services.reminder_jobs_worker import (
        start_reminder_jobs_worker as _start_reminder_jobs_worker_impl,
    )

    return _start_reminder_jobs_worker_impl(stop_event=stop_event)


def _start_admin_backup_jobs_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.services.admin_backup_jobs_worker import (
        start_admin_backup_jobs_worker as _start_admin_backup_jobs_worker_impl,
    )

    return _start_admin_backup_jobs_worker_impl(stop_event=stop_event)


def _start_admin_byok_validation_jobs_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        start_admin_byok_validation_jobs_worker as _start_admin_byok_validation_jobs_worker_impl,
    )

    return _start_admin_byok_validation_jobs_worker_impl(stop_event=stop_event)


def _start_admin_maintenance_rotation_jobs_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
        start_admin_maintenance_rotation_jobs_worker as _start_admin_maintenance_rotation_jobs_worker_impl,
    )

    return _start_admin_maintenance_rotation_jobs_worker_impl(stop_event=stop_event)


def _start_recipe_run_jobs_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker import (
        start_recipe_run_jobs_worker as _start_recipe_run_jobs_worker_impl,
    )

    return _start_recipe_run_jobs_worker_impl(stop_event=stop_event)

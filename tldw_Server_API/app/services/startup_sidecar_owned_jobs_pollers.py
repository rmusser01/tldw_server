"""
Sidecar-gated owned jobs poller startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerSpec,
    stop_event_worker_spec,
)
from tldw_Server_API.app.services.lifecycle_workers import (
    ManagedWorker,
    WorkerRegistry,
)

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


def provide_sidecar_owned_jobs_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return declarative specs for sidecar-gated owned Jobs pollers."""

    return (
        stop_event_worker_spec(
            name="reminder_jobs_task",
            worker_service=_run_reminder_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=lambda context: _sidecar_owned_worker_enabled(
                context,
                _reminder_jobs_worker_enabled,
            ),
        ),
        stop_event_worker_spec(
            name="admin_backup_jobs_task",
            worker_service=_run_admin_backup_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=lambda context: _sidecar_owned_worker_enabled(
                context,
                _admin_backup_jobs_worker_enabled,
            ),
        ),
        stop_event_worker_spec(
            name="admin_byok_validation_jobs_task",
            worker_service=_run_admin_byok_validation_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=lambda context: _sidecar_owned_worker_enabled(
                context,
                _admin_byok_validation_jobs_worker_enabled,
            ),
        ),
        stop_event_worker_spec(
            name="admin_maintenance_rotation_jobs_task",
            worker_service=_run_admin_maintenance_rotation_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=lambda context: _sidecar_owned_worker_enabled(
                context,
                _admin_maintenance_rotation_jobs_worker_enabled,
            ),
        ),
        stop_event_worker_spec(
            name="recipe_run_jobs_task",
            worker_service=_run_recipe_run_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=lambda context: _sidecar_owned_worker_enabled(
                context,
                _recipe_run_jobs_worker_enabled,
            ),
        ),
    )


def _sidecar_owned_worker_enabled(
    context: WorkerLifecycleContext,
    worker_enabled: Callable[[], bool],
) -> bool:
    if context.sidecar_mode:
        return False
    try:
        return worker_enabled()
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(
            "Sidecar-owned Jobs worker predicate disabled after {}",
            type(exc).__name__,
        )
        return False


def _reminder_jobs_worker_enabled() -> bool:
    from tldw_Server_API.app.core.testing import env_flag_enabled

    return env_flag_enabled("REMINDER_JOBS_WORKER_ENABLED")


def _admin_backup_jobs_worker_enabled() -> bool:
    from tldw_Server_API.app.core.testing import env_flag_enabled

    return env_flag_enabled("ADMIN_BACKUP_JOBS_WORKER_ENABLED")


def _admin_byok_validation_jobs_worker_enabled() -> bool:
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        byok_validation_worker_enabled,
    )

    return byok_validation_worker_enabled()


def _admin_maintenance_rotation_jobs_worker_enabled() -> bool:
    from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
        maintenance_rotation_worker_enabled,
    )

    return maintenance_rotation_worker_enabled()


def _recipe_run_jobs_worker_enabled() -> bool:
    from tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker import (
        recipe_run_jobs_worker_enabled,
    )

    return recipe_run_jobs_worker_enabled()


async def start_sidecar_owned_jobs_pollers(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
    worker_inventory: WorkerRegistry | None = None,
) -> SidecarOwnedJobsPollerHandles:
    """Start sidecar-gated owned jobs pollers and return their handles."""

    reminder_jobs_stop_event, reminder_jobs_task = await _start_reminder_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    admin_backup_jobs_stop_event, admin_backup_jobs_task = await _start_admin_backup_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    (
        admin_byok_validation_jobs_stop_event,
        admin_byok_validation_jobs_task,
    ) = await _start_admin_byok_validation_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    (
        admin_maintenance_rotation_jobs_stop_event,
        admin_maintenance_rotation_jobs_task,
    ) = await _start_admin_maintenance_rotation_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    recipe_run_jobs_stop_event, recipe_run_jobs_task = await _start_recipe_run_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
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


def _safe_cancel_task(task: Any) -> None:
    """Best-effort cancel a started worker after startup registration rollback."""

    cancel = getattr(task, "cancel", None)
    if cancel is None:
        return
    try:
        cancel()
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"Failed to cancel sidecar-owned Jobs worker during startup rollback: {exc}")


def _register_started_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    worker_inventory: WorkerRegistry | None,
    name: str,
    task: Any,
    stop_event: Any,
) -> None:
    """Register an already-started sidecar-owned jobs poller."""

    if worker_inventory is not None:
        try:
            worker_inventory.register(
                ManagedWorker(
                    name=name,
                    task=task,
                    stop_event=stop_event,
                    timeout_sec=5.0,
                    category="jobs",
                    shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
                )
            )
        except _STARTUP_GUARD_EXCEPTIONS:
            _safe_cancel_task(task)
            raise
        return

    register_owned_job_poller(
        app,
        owned_job_pollers,
        name=name,
        task=task,
        stop_event=stop_event,
    )


async def _start_reminder_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Reminder Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_reminder_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Reminder Jobs worker started")
            _register_started_jobs_worker(
                app=app,
                owned_job_pollers=owned_job_pollers,
                register_owned_job_poller=register_owned_job_poller,
                worker_inventory=worker_inventory,
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
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Admin backup Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_admin_backup_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Admin backup Jobs worker started")
            _register_started_jobs_worker(
                app=app,
                owned_job_pollers=owned_job_pollers,
                register_owned_job_poller=register_owned_job_poller,
                worker_inventory=worker_inventory,
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
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Admin BYOK validation Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_admin_byok_validation_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Admin BYOK validation Jobs worker started")
            _register_started_jobs_worker(
                app=app,
                owned_job_pollers=owned_job_pollers,
                register_owned_job_poller=register_owned_job_poller,
                worker_inventory=worker_inventory,
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
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Admin maintenance rotation Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_admin_maintenance_rotation_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Admin maintenance rotation Jobs worker started")
            _register_started_jobs_worker(
                app=app,
                owned_job_pollers=owned_job_pollers,
                register_owned_job_poller=register_owned_job_poller,
                worker_inventory=worker_inventory,
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
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    try:
        if sidecar_mode:
            logger.info("Evaluation recipe-run Jobs worker disabled in sidecar mode")
            return None, None

        stop_event = _make_event()
        task = await _resolve_result(_start_recipe_run_jobs_worker_service(stop_event=stop_event))
        if task:
            logger.info("Evaluation recipe-run Jobs worker started")
            _register_started_jobs_worker(
                app=app,
                owned_job_pollers=owned_job_pollers,
                register_owned_job_poller=register_owned_job_poller,
                worker_inventory=worker_inventory,
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


def _run_reminder_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.reminder_jobs_worker import (
        run_reminder_jobs_worker as _run_reminder_jobs_worker_impl,
    )

    return _run_reminder_jobs_worker_impl(stop_event=stop_event)


def _start_admin_backup_jobs_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.services.admin_backup_jobs_worker import (
        start_admin_backup_jobs_worker as _start_admin_backup_jobs_worker_impl,
    )

    return _start_admin_backup_jobs_worker_impl(stop_event=stop_event)


def _run_admin_backup_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.admin_backup_jobs_worker import (
        run_admin_backup_jobs_worker as _run_admin_backup_jobs_worker_impl,
    )

    return _run_admin_backup_jobs_worker_impl(stop_event=stop_event)


def _start_admin_byok_validation_jobs_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        start_admin_byok_validation_jobs_worker as _start_admin_byok_validation_jobs_worker_impl,
    )

    return _start_admin_byok_validation_jobs_worker_impl(stop_event=stop_event)


def _run_admin_byok_validation_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.admin_byok_validation_jobs_worker import (
        run_admin_byok_validation_jobs_worker as _run_admin_byok_validation_jobs_worker_impl,
    )

    return _run_admin_byok_validation_jobs_worker_impl(stop_event=stop_event)


def _start_admin_maintenance_rotation_jobs_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
        start_admin_maintenance_rotation_jobs_worker as _start_admin_maintenance_rotation_jobs_worker_impl,
    )

    return _start_admin_maintenance_rotation_jobs_worker_impl(stop_event=stop_event)


def _run_admin_maintenance_rotation_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
        run_admin_maintenance_rotation_jobs_worker as _run_admin_maintenance_rotation_jobs_worker_impl,
    )

    return _run_admin_maintenance_rotation_jobs_worker_impl(stop_event=stop_event)


def _start_recipe_run_jobs_worker_service(*, stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker import (
        start_recipe_run_jobs_worker as _start_recipe_run_jobs_worker_impl,
    )

    return _start_recipe_run_jobs_worker_impl(stop_event=stop_event)


def _run_recipe_run_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Evaluations.recipe_runs_jobs_worker import (
        run_recipe_run_jobs_worker as _run_recipe_run_jobs_worker_impl,
    )

    return _run_recipe_run_jobs_worker_impl(stop_event=stop_event)

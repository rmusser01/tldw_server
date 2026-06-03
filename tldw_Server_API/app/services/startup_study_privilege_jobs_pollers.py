"""
Study and privilege jobs poller startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerSpec,
    route_enabled_predicate,
    stop_event_worker_spec,
)
from tldw_Server_API.app.services.lifecycle_workers import WorkerRegistry

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class StudyPrivilegeJobsPollerHandles:
    """Startup-owned study and privilege jobs poller handles used later in shutdown flow."""

    study_pack_jobs_stop_event: Any | None = None
    study_pack_jobs_task: Any | None = None
    study_suggestions_jobs_stop_event: Any | None = None
    study_suggestions_jobs_task: Any | None = None
    privilege_snapshot_stop_event: Any | None = None
    privilege_snapshot_task: Any | None = None


def provide_study_privilege_jobs_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return declarative specs for study and privilege jobs pollers."""

    return (
        stop_event_worker_spec(
            name="study_pack_jobs_task",
            worker_service=_run_study_pack_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "STUDY_PACK_JOBS_WORKER_ENABLED",
                "flashcards",
            ),
        ),
        stop_event_worker_spec(
            name="study_suggestions_jobs_task",
            worker_service=_run_study_suggestions_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED",
                "study-suggestions",
            ),
        ),
        stop_event_worker_spec(
            name="privilege_snapshot_task",
            worker_service=_run_privilege_snapshot_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "PRIVILEGE_SNAPSHOT_WORKER_ENABLED",
                "privileges",
            ),
        ),
    )


async def start_study_privilege_jobs_pollers(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> StudyPrivilegeJobsPollerHandles:
    """Start study and privilege jobs pollers and return their handles."""

    study_pack_jobs_stop_event, study_pack_jobs_task = await _start_study_pack_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    (
        study_suggestions_jobs_stop_event,
        study_suggestions_jobs_task,
    ) = await _start_study_suggestions_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    privilege_snapshot_stop_event, privilege_snapshot_task = await _start_privilege_snapshot_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    return StudyPrivilegeJobsPollerHandles(
        study_pack_jobs_stop_event=study_pack_jobs_stop_event,
        study_pack_jobs_task=study_pack_jobs_task,
        study_suggestions_jobs_stop_event=study_suggestions_jobs_stop_event,
        study_suggestions_jobs_task=study_suggestions_jobs_task,
        privilege_snapshot_stop_event=privilege_snapshot_stop_event,
        privilege_snapshot_task=privilege_snapshot_task,
    )


def _make_event() -> Any:
    return asyncio.Event()


def _create_task(awaitable: Any) -> Any:
    return asyncio.create_task(awaitable)


async def _start_study_pack_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Study Pack jobs worker and return its shutdown handles."""

    try:
        enabled = should_start_worker("STUDY_PACK_JOBS_WORKER_ENABLED", "flashcards")
        if not enabled:
            logger.info("Study-pack Jobs worker disabled by flag (STUDY_PACK_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="study_pack_jobs_task",
                task_name="study_pack_jobs_task",
                coroutine_factory=_run_study_pack_jobs_worker_service,
                timeout_sec=5.0,
                category="jobs",
                shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            )
            logger.info("Study-pack Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_study_pack_jobs_worker_service(stop_event))
        logger.info("Study-pack Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="study_pack_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Study-pack Jobs worker: {exc}")
        return None, None


async def _start_study_suggestions_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Study Suggestions jobs worker and return its shutdown handles."""

    try:
        enabled = should_start_worker("STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED", "study-suggestions")
        if not enabled:
            logger.info("Study-suggestions Jobs worker disabled by flag (STUDY_SUGGESTIONS_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="study_suggestions_jobs_task",
                task_name="study_suggestions_jobs_task",
                coroutine_factory=_run_study_suggestions_jobs_worker_service,
                timeout_sec=5.0,
                category="jobs",
                shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            )
            logger.info("Study-suggestions Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_study_suggestions_jobs_worker_service(stop_event))
        logger.info("Study-suggestions Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="study_suggestions_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Study-suggestions Jobs worker: {exc}")
        return None, None


async def _start_privilege_snapshot_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the privilege snapshot worker and return its shutdown handles."""

    try:
        enabled = should_start_worker("PRIVILEGE_SNAPSHOT_WORKER_ENABLED", "privileges")
        if not enabled:
            logger.info("Privilege snapshot worker disabled by flag (PRIVILEGE_SNAPSHOT_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="privilege_snapshot_task",
                task_name="privilege_snapshot_task",
                coroutine_factory=_run_privilege_snapshot_worker_service,
                timeout_sec=5.0,
                category="jobs",
                shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            )
            logger.info("Privilege snapshot worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_privilege_snapshot_worker_service(stop_event))
        logger.info("Privilege snapshot worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="privilege_snapshot_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start privilege snapshot worker: {exc}")
        return None, None


def _run_study_pack_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.study_pack_jobs_worker import (
        run_study_pack_jobs_worker as _run_study_pack_jobs_worker,
    )

    return _run_study_pack_jobs_worker(stop_event)


def _run_study_suggestions_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.study_suggestions_jobs_worker import (
        run_study_suggestions_jobs_worker as _run_study_suggestions_jobs_worker,
    )

    return _run_study_suggestions_jobs_worker(stop_event)


def _run_privilege_snapshot_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.privilege_snapshot_worker import (
        run_privilege_snapshot_worker as _run_privilege_snapshot_worker,
    )

    return _run_privilege_snapshot_worker(stop_event)

"""
Jobs notifications bridge and embeddings A/B startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.services.lifecycle_exceptions import LIFECYCLE_GUARD_EXCEPTIONS
from tldw_Server_API.app.services.lifecycle_workers import (
    ManagedWorker,
    ShutdownPhase,
    WorkerRegistry,
)

_STARTUP_GUARD_EXCEPTIONS = LIFECYCLE_GUARD_EXCEPTIONS
_TRUTHY_ENV_VALUES = {"true", "1", "yes", "y", "on"}


@dataclass
class NotificationsAbtestStartupHandles:
    """Startup handles for the notifications bridge and embeddings A/B worker."""

    jobs_notifications_bridge_task: Any | None = None
    evals_abtest_jobs_stop_event: Any | None = None
    evals_abtest_jobs_task: Any | None = None


async def start_notifications_abtest_workers(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
    worker_inventory: WorkerRegistry | None = None,
) -> NotificationsAbtestStartupHandles:
    """Start the notifications bridge and embeddings A/B worker."""

    jobs_notifications_bridge_task = await _start_jobs_notifications_bridge_worker(
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    evals_abtest_jobs_stop_event, evals_abtest_jobs_task = await _start_evals_abtest_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
        worker_inventory=worker_inventory,
    )
    return NotificationsAbtestStartupHandles(
        jobs_notifications_bridge_task=jobs_notifications_bridge_task,
        evals_abtest_jobs_stop_event=evals_abtest_jobs_stop_event,
        evals_abtest_jobs_task=evals_abtest_jobs_task,
    )


def _make_event() -> Any:
    return asyncio.Event()


def _create_task(awaitable: Any) -> Any:
    return asyncio.create_task(awaitable)


async def _resolve_result(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _safe_cancel_task(task: Any | None) -> None:
    """Best-effort cancel a started worker after startup registration rollback."""

    if task is None:
        return
    cancel = getattr(task, "cancel", None)
    if cancel is None:
        return
    try:
        cancel()
    except asyncio.CancelledError:
        current_task = asyncio.current_task()
        current_task_cancelling = getattr(current_task, "cancelling", None)
        if callable(current_task_cancelling) and current_task_cancelling():
            raise
        logger.debug("Lifecycle worker task cancel raised CancelledError during startup rollback")
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.debug(f"Failed to cancel lifecycle worker during startup rollback: {exc}")


def _register_notifications_bridge_worker(
    *,
    worker_inventory: WorkerRegistry,
    task: Any,
) -> None:
    """Register the notifications bridge for background shutdown."""

    try:
        worker_inventory.register(
            ManagedWorker(
                name="jobs_notifications_bridge_task",
                task=task,
                stop_event=None,
                category="jobs",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
        )
    except _STARTUP_GUARD_EXCEPTIONS:
        _safe_cancel_task(task)
        raise


def _register_evals_abtest_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    worker_inventory: WorkerRegistry | None,
    task: Any,
    stop_event: Any,
) -> None:
    """Register the embeddings A/B job poller with inventory or the legacy hook."""

    if worker_inventory is not None:
        try:
            worker_inventory.register(
                ManagedWorker(
                    name="evals_abtest_jobs_task",
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
        name="evals_abtest_jobs_task",
        task=task,
        stop_event=stop_event,
    )


async def _start_jobs_notifications_bridge_worker(
    *,
    sidecar_mode: bool,
    worker_inventory: WorkerRegistry | None = None,
) -> Any | None:
    try:
        if sidecar_mode:
            logger.info("Jobs notifications bridge worker disabled in sidecar mode")
            return None

        task = await _resolve_result(_start_jobs_notifications_service())
        if task:
            if worker_inventory is not None:
                _register_notifications_bridge_worker(
                    worker_inventory=worker_inventory,
                    task=task,
                )
            logger.info("Jobs notifications bridge worker started")
        else:
            logger.info("Jobs notifications bridge worker disabled (JOBS_NOTIFICATIONS_BRIDGE_ENABLED != true)")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Jobs notifications bridge worker: {exc}")
        return None


async def _start_evals_abtest_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    sidecar_mode: bool,
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    task = None
    try:
        enabled = os.getenv("EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED", "false").lower() in _TRUTHY_ENV_VALUES
        if not enabled:
            enabled = os.getenv("EVALS_ABTEST_JOBS_WORKER_ENABLED", "false").lower() in _TRUTHY_ENV_VALUES
        if sidecar_mode:
            enabled = False

        if not enabled:
            logger.info("Embeddings A/B Jobs worker disabled by flag")
            return None, None

        stop_event = _make_event()
        task = _create_task(_run_embeddings_abtest_jobs_worker_service(stop_event))
        try:
            _register_evals_abtest_jobs_worker(
                app=app,
                owned_job_pollers=owned_job_pollers,
                register_owned_job_poller=register_owned_job_poller,
                worker_inventory=worker_inventory,
                task=task,
                stop_event=stop_event,
            )
        except _STARTUP_GUARD_EXCEPTIONS:
            _safe_cancel_task(task)
            raise
        logger.info("Embeddings A/B Jobs worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Embeddings A/B Jobs worker: {exc}")
        return None, None


def _start_jobs_notifications_service() -> Any:
    from tldw_Server_API.app.services.jobs_notifications_service import (
        start_jobs_notifications_service as _start_jobs_notifications_service_impl,
    )

    return _start_jobs_notifications_service_impl()


def _run_embeddings_abtest_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Evaluations.embeddings_abtest_jobs_worker import (
        run_embeddings_abtest_jobs_worker as _run_embeddings_abtest_jobs_worker,
    )

    return _run_embeddings_abtest_jobs_worker(stop_event)

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

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

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
) -> NotificationsAbtestStartupHandles:
    """Start the notifications bridge and embeddings A/B worker."""

    jobs_notifications_bridge_task = await _start_jobs_notifications_bridge_worker(
        sidecar_mode=sidecar_mode,
    )
    evals_abtest_jobs_stop_event, evals_abtest_jobs_task = await _start_evals_abtest_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        sidecar_mode=sidecar_mode,
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


async def _start_jobs_notifications_bridge_worker(*, sidecar_mode: bool) -> Any | None:
    try:
        if sidecar_mode:
            logger.info("Jobs notifications bridge worker disabled in sidecar mode")
            return None

        task = await _resolve_result(_start_jobs_notifications_service())
        if task:
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
            register_owned_job_poller(
                app,
                owned_job_pollers,
                name="evals_abtest_jobs_task",
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


def _safe_cancel_task(task: Any | None) -> None:
    if task is None:
        return
    try:
        task.cancel()
    except _STARTUP_GUARD_EXCEPTIONS:
        pass


def _run_embeddings_abtest_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Evaluations.embeddings_abtest_jobs_worker import (
        run_embeddings_abtest_jobs_worker as _run_embeddings_abtest_jobs_worker,
    )

    return _run_embeddings_abtest_jobs_worker(stop_event)

"""
Runtime monitor startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import env_flag_enabled as _env_flag_enabled
from tldw_Server_API.app.core.testing import is_truthy as _is_truthy
from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class RuntimeMonitorHandles:
    """Startup-owned runtime monitor handles preserved for later shutdown."""

    jobs_metrics_stop_event: Any | None = None
    jobs_metrics_task: Any | None = None
    loop_lag_stop_event: Any | None = None
    loop_lag_task: Any | None = None


async def start_runtime_monitors(
    *,
    worker_inventory: Any | None = None,
) -> RuntimeMonitorHandles:
    """Start small runtime monitor tasks and return explicit handles."""
    if worker_inventory is None:
        jobs_metrics_stop_event, jobs_metrics_task = await _start_jobs_metrics_gauge_worker()
    else:
        jobs_metrics_stop_event, jobs_metrics_task = await _start_jobs_metrics_gauge_worker(
            worker_inventory=worker_inventory,
        )
    loop_lag_stop_event, loop_lag_task = await _start_loop_lag_watchdog(
        worker_inventory=worker_inventory,
    )
    return RuntimeMonitorHandles(
        jobs_metrics_stop_event=jobs_metrics_stop_event,
        jobs_metrics_task=jobs_metrics_task,
        loop_lag_stop_event=loop_lag_stop_event,
        loop_lag_task=loop_lag_task,
    )


def _make_event() -> Any:
    return asyncio.Event()


def _create_task(awaitable: Any) -> Any:
    return asyncio.create_task(awaitable)


async def _start_jobs_metrics_gauge_worker(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    try:
        enabled = _is_truthy(os.getenv("JOBS_METRICS_GAUGES_ENABLED", "true"))
        if not enabled:
            logger.info("Jobs metrics gauge worker disabled by flag")
            return None, None
        if worker_inventory is not None:
            from tldw_Server_API.app.services.lifecycle_workers import (
                ShutdownPhase,
                start_stop_event_worker,
            )

            task, stop_event = await start_stop_event_worker(
                worker_inventory,
                name="jobs_metrics_task",
                task_name="jobs_metrics_task",
                coroutine_factory=_run_jobs_metrics_gauges_service,
                category="jobs",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
            logger.info("Jobs metrics gauge worker started with explicit stop_event signal")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_jobs_metrics_gauges_service(stop_event))
        logger.info("Jobs metrics gauge worker started with explicit stop_event signal")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Jobs metrics gauge worker: {exc}")
        return None, None


async def _start_loop_lag_watchdog(
    *,
    worker_inventory: Any | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the event-loop lag watchdog directly or through worker inventory.

    The monitor is skipped when EVENT_LOOP_LAG_WATCHDOG_ENABLED is disabled.
    When a worker inventory is supplied, the watchdog is registered as a
    background-phase custom worker so WorkerRegistry owns shutdown; otherwise
    this helper preserves the legacy explicit stop-event/task handles.
    """
    try:
        if not _env_flag_enabled("EVENT_LOOP_LAG_WATCHDOG_ENABLED"):
            logger.info("Event loop lag watchdog disabled by flag")
            return None, None
        if worker_inventory is not None:
            task, stop_event = await worker_inventory.register_custom(
                name="loop_lag_task",
                task_name="loop_lag_watchdog",
                coroutine_factory=_run_loop_lag_watchdog_service,
                timeout_sec=2.0,
                category="monitoring",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
            logger.info("Event loop lag watchdog started")
            return stop_event, task
        stop_event = _make_event()
        task = _create_task(_run_loop_lag_watchdog_service(stop_event))
        logger.info("Event loop lag watchdog started")
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start event loop lag watchdog: {exc}")
        return None, None


def _run_jobs_metrics_gauges_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.jobs_metrics_service import (
        run_jobs_metrics_gauges as _run_jobs_metrics,
    )

    return _run_jobs_metrics(stop_event)


def _run_loop_lag_watchdog_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.loop_lag_watchdog import (
        run_loop_lag_watchdog as _run_loop_lag_watchdog,
    )

    return _run_loop_lag_watchdog(stop_event)

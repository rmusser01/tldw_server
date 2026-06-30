"""
Runtime monitor shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class RuntimeMonitorShutdownHandles:
    """Updated runtime-monitor task handles after shutdown processing."""

    jobs_metrics_task: Any | None = None
    loop_lag_task: Any | None = None


async def shutdown_runtime_monitors(
    *,
    jobs_metrics_task: Any | None,
    jobs_metrics_stop_event: Any | None,
    loop_lag_task: Any | None,
    loop_lag_stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> RuntimeMonitorShutdownHandles:
    """Stop runtime monitor tasks while preserving legacy timeout/cancel semantics."""
    if jobs_metrics_task:
        try:
            if jobs_metrics_stop_event:
                jobs_metrics_stop_event.set()
                await _wait_for_task(jobs_metrics_task, timeout=5.0)
                logger.info("Jobs metrics gauge worker stopped via stop_event")
            else:
                jobs_metrics_task.cancel()
        except guard_exceptions:
            try:
                jobs_metrics_task.cancel()
            except guard_exceptions:
                pass

    if loop_lag_task:
        try:
            if loop_lag_stop_event:
                loop_lag_stop_event.set()
                await _wait_for_task(loop_lag_task, timeout=2.0)
                logger.info("Event loop lag watchdog stopped via stop_event")
            else:
                loop_lag_task.cancel()
        except guard_exceptions:
            try:
                loop_lag_task.cancel()
            except guard_exceptions as exc:
                logger.debug(f"Event loop lag watchdog cancel failed: {exc}")

    return RuntimeMonitorShutdownHandles(
        jobs_metrics_task=jobs_metrics_task,
        loop_lag_task=loop_lag_task,
    )


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)

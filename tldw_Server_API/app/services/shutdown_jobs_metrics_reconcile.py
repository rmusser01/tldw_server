"""
Jobs metrics reconcile shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class JobsMetricsReconcileShutdownHandles:
    """Updated jobs-metrics-reconcile handles after shutdown processing."""

    jobs_metrics_reconcile_task: Any | None = None
    jobs_metrics_reconcile_stop: Any | None = None


async def shutdown_jobs_metrics_reconcile(
    *,
    jobs_metrics_reconcile_task: Any | None,
    jobs_metrics_reconcile_stop: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> JobsMetricsReconcileShutdownHandles:
    """Stop the jobs metrics reconcile worker while preserving legacy semantics."""
    if jobs_metrics_reconcile_task:
        try:
            if jobs_metrics_reconcile_stop:
                jobs_metrics_reconcile_stop.set()
                await _wait_for_task(jobs_metrics_reconcile_task, timeout=5.0)
                logger.info("Jobs metrics reconcile worker stopped via stop_event")
            else:
                jobs_metrics_reconcile_task.cancel()
        except guard_exceptions:
            try:
                jobs_metrics_reconcile_task.cancel()
            except guard_exceptions:
                pass

    return JobsMetricsReconcileShutdownHandles(
        jobs_metrics_reconcile_task=jobs_metrics_reconcile_task,
        jobs_metrics_reconcile_stop=jobs_metrics_reconcile_stop,
    )


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)

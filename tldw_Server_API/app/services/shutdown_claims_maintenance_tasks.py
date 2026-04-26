"""
Claims rebuild and maintenance cancel-only shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ClaimsMaintenanceShutdownHandles:
    """Updated claims and maintenance task handles after shutdown cancellation."""

    claims_task: Any | None = None
    jobs_prune_task: Any | None = None
    files_export_gc_task: Any | None = None
    notifications_prune_task: Any | None = None


async def shutdown_claims_maintenance_tasks(
    *,
    claims_task: Any | None,
    jobs_prune_task: Any | None,
    files_export_gc_task: Any | None,
    notifications_prune_task: Any | None,
) -> ClaimsMaintenanceShutdownHandles:
    """Cancel claims and maintenance tasks while preserving legacy ordering semantics."""
    await _shutdown_claims_task(task=claims_task)
    await _shutdown_jobs_prune_task(task=jobs_prune_task)
    await _shutdown_files_export_gc_task(task=files_export_gc_task)
    await _shutdown_notifications_prune_task(task=notifications_prune_task)
    return ClaimsMaintenanceShutdownHandles(
        claims_task=claims_task,
        jobs_prune_task=jobs_prune_task,
        files_export_gc_task=files_export_gc_task,
        notifications_prune_task=notifications_prune_task,
    )


async def _shutdown_claims_task(*, task: Any | None) -> None:
    stop_event = getattr(task, "_tldw_claims_rebuild_stop_event", None)
    if stop_event is not None:
        stop_event.set()
    _cancel_task(task)


async def _shutdown_jobs_prune_task(*, task: Any | None) -> None:
    _cancel_task(task)


async def _shutdown_files_export_gc_task(*, task: Any | None) -> None:
    _cancel_task(task)


async def _shutdown_notifications_prune_task(*, task: Any | None) -> None:
    _cancel_task(task)


def _cancel_task(task: Any | None) -> None:
    if task:
        task.cancel()

"""
Privilege snapshot worker shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class PrivilegeSnapshotShutdownHandles:
    """Updated privilege snapshot worker handles after shutdown processing."""

    privilege_snapshot_task: Any | None = None
    privilege_snapshot_stop_event: Any | None = None


async def shutdown_privilege_snapshot_worker(
    *,
    privilege_snapshot_task: Any | None,
    privilege_snapshot_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> PrivilegeSnapshotShutdownHandles:
    """Stop the privilege snapshot worker while preserving legacy late-stop semantics."""
    await _shutdown_privilege_snapshot_worker(
        task=privilege_snapshot_task,
        stop_event=privilege_snapshot_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return PrivilegeSnapshotShutdownHandles(
        privilege_snapshot_task=privilege_snapshot_task,
        privilege_snapshot_stop_event=privilege_snapshot_stop_event,
    )


async def _shutdown_privilege_snapshot_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if not should_run_late_stop("privilege_snapshot_task", task):
        return
    if stop_event:
        try:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
        except guard_exceptions:
            task.cancel()
    else:
        task.cancel()


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)

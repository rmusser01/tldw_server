"""
Jobs notifications bridge, embeddings compactor, and WebSub shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class NotificationsCompactorWebsubShutdownHandles:
    """Updated late-worker handles after shutdown processing."""

    jobs_notifications_bridge_task: Any | None = None


async def shutdown_notifications_compactor_websub_workers(
    *,
    jobs_notifications_bridge_task: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> NotificationsCompactorWebsubShutdownHandles:
    """Stop late notifications/compactor/WebSub workers while preserving legacy semantics."""
    await _shutdown_jobs_notifications_bridge_worker(
        task=jobs_notifications_bridge_task,
        guard_exceptions=guard_exceptions,
    )
    return NotificationsCompactorWebsubShutdownHandles(
        jobs_notifications_bridge_task=jobs_notifications_bridge_task,
    )


async def _shutdown_jobs_notifications_bridge_worker(
    *,
    task: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if not task:
        return
    try:
        task.cancel()
        await _wait_for_task(task, timeout=5.0)
        logger.info("Jobs notifications bridge worker cancelled")
    except asyncio.CancelledError:
        pass
    except (asyncio.TimeoutError,) + guard_exceptions:
        await _cancel_and_wait_for_task(task, guard_exceptions=guard_exceptions)


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)


async def _cancel_and_wait_for_task(
    task: Any,
    *,
    guard_exceptions: tuple[type[BaseException], ...],
    timeout: float = 5.0,
) -> None:
    try:
        task.cancel()
    except guard_exceptions:
        return
    try:
        await _wait_for_task(task, timeout=timeout)
    except asyncio.CancelledError:
        pass
    except (asyncio.TimeoutError,) + guard_exceptions:
        pass

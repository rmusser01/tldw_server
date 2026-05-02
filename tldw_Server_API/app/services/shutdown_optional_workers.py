"""
Optional worker shutdown helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class OptionalWorkerShutdownHandles:
    """Updated optional-worker task handles after shutdown processing."""

    jobs_crypto_rotate_task: Any | None = None
    jobs_integrity_task: Any | None = None
    jobs_webhooks_task: Any | None = None
    meetings_webhook_dlq_task: Any | None = None
    workflows_dlq_task: Any | None = None
    workflows_gc_task: Any | None = None
    workflows_maint_task: Any | None = None


async def shutdown_optional_workers(
    *,
    jobs_crypto_rotate_task: Any | None,
    jobs_crypto_rotate_stop_event: Any | None,
    jobs_integrity_task: Any | None,
    jobs_integrity_stop_event: Any | None,
    jobs_webhooks_task: Any | None,
    jobs_webhooks_stop_event: Any | None,
    meetings_webhook_dlq_task: Any | None,
    meetings_webhook_dlq_stop_event: Any | None,
    workflows_dlq_task: Any | None,
    workflows_dlq_stop_event: Any | None,
    workflows_gc_task: Any | None,
    workflows_gc_stop_event: Any | None,
    workflows_maint_task: Any | None,
    workflows_maint_stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> OptionalWorkerShutdownHandles:
    """Stop optional workers while preserving legacy ordering and timeout semantics."""
    await _shutdown_jobs_crypto_rotate_worker(
        task=jobs_crypto_rotate_task,
        stop_event=jobs_crypto_rotate_stop_event,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_jobs_integrity_worker(
        task=jobs_integrity_task,
        stop_event=jobs_integrity_stop_event,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_jobs_webhooks_worker(
        task=jobs_webhooks_task,
        stop_event=jobs_webhooks_stop_event,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_meetings_webhook_dlq_worker(
        task=meetings_webhook_dlq_task,
        stop_event=meetings_webhook_dlq_stop_event,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_workflows_webhook_dlq_worker(
        task=workflows_dlq_task,
        stop_event=workflows_dlq_stop_event,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_workflows_artifact_gc_worker(
        task=workflows_gc_task,
        stop_event=workflows_gc_stop_event,
        guard_exceptions=guard_exceptions,
    )
    await _shutdown_workflows_db_maintenance_worker(
        task=workflows_maint_task,
        stop_event=workflows_maint_stop_event,
        guard_exceptions=guard_exceptions,
    )
    return OptionalWorkerShutdownHandles(
        jobs_crypto_rotate_task=jobs_crypto_rotate_task,
        jobs_integrity_task=jobs_integrity_task,
        jobs_webhooks_task=jobs_webhooks_task,
        meetings_webhook_dlq_task=meetings_webhook_dlq_task,
        workflows_dlq_task=workflows_dlq_task,
        workflows_gc_task=workflows_gc_task,
        workflows_maint_task=workflows_maint_task,
    )


async def _shutdown_jobs_crypto_rotate_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_stop_event_worker(
        task=task,
        stop_event=stop_event,
        timeout=5.0,
        stop_message="Jobs crypto rotate worker stopped via stop_event",
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_jobs_integrity_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_stop_event_worker(
        task=task,
        stop_event=stop_event,
        timeout=5.0,
        stop_message="Jobs integrity sweeper stopped via stop_event",
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_jobs_webhooks_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_stop_event_worker(
        task=task,
        stop_event=stop_event,
        timeout=5.0,
        stop_message="Jobs webhooks worker stopped via stop_event",
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_meetings_webhook_dlq_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_stop_event_worker(
        task=task,
        stop_event=stop_event,
        timeout=5.0,
        stop_message="Meetings webhook DLQ worker stopped via stop_event",
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_workflows_webhook_dlq_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_stop_event_worker(
        task=task,
        stop_event=stop_event,
        timeout=5.0,
        stop_message="Workflows webhook DLQ worker stopped via stop_event",
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_workflows_artifact_gc_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_stop_event_worker(
        task=task,
        stop_event=stop_event,
        timeout=5.0,
        stop_message="Workflows artifact GC worker stopped via stop_event",
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_workflows_db_maintenance_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    await _shutdown_stop_event_worker(
        task=task,
        stop_event=stop_event,
        timeout=5.0,
        stop_message="Workflows DB maintenance worker stopped via stop_event",
        guard_exceptions=guard_exceptions,
    )


async def _shutdown_stop_event_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    timeout: float,
    stop_message: str,
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if not task:
        return
    fallback_exceptions = (asyncio.TimeoutError,) + guard_exceptions
    try:
        if stop_event is not None:
            stop_event.set()
            await _wait_for_task(task, timeout=timeout)
            logger.info(stop_message)
        else:
            await _cancel_and_wait_for_task(
                task,
                timeout=timeout,
                guard_exceptions=guard_exceptions,
            )
    except fallback_exceptions:
        await _cancel_and_wait_for_task(
            task,
            timeout=timeout,
            guard_exceptions=guard_exceptions,
        )


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)


async def _cancel_and_wait_for_task(
    task: Any,
    *,
    timeout: float,
    guard_exceptions: tuple[type[BaseException], ...],
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

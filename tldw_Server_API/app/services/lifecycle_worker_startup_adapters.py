"""Adapters for legacy startup helpers used by declarative worker specs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from tldw_Server_API.app.services.lifecycle_exceptions import LIFECYCLE_GUARD_EXCEPTIONS

StartedTaskStarter = Callable[[], Awaitable[Any | None]]
StartedTaskStopper = Callable[[Any], Awaitable[None]]
ServiceStarter = Callable[[], Awaitable[Any]]
ServiceStopper = Callable[[], Awaitable[None]]


async def run_started_task_until_stop(
    stop_event: Any,
    *,
    starter: StartedTaskStarter,
    stopper: StartedTaskStopper | None = None,
    timeout_sec: float = 5.0,
) -> None:
    """Start a legacy task-returning service and own shutdown via stop_event."""

    task = await starter()
    if task is None:
        return
    try:
        await stop_event.wait()
    finally:
        if stopper is not None:
            try:
                await asyncio.wait_for(stopper(task), timeout=timeout_sec)
            except (asyncio.TimeoutError,) + LIFECYCLE_GUARD_EXCEPTIONS as exc:
                logger.debug(
                    "Lifecycle startup adapter stopper cleanup fell back to task "
                    "cancellation after {}",
                    type(exc).__name__,
                )
                await cancel_started_task(task, timeout_sec=timeout_sec)
        else:
            await cancel_started_task(task, timeout_sec=timeout_sec)


async def run_start_stop_service_until_stop(
    stop_event: Any,
    *,
    starter: ServiceStarter,
    stopper: ServiceStopper,
    timeout_sec: float = 5.0,
) -> None:
    """Start a service with separate start/stop functions and wait for shutdown."""

    await starter()
    try:
        await stop_event.wait()
    finally:
        try:
            await asyncio.wait_for(stopper(), timeout=timeout_sec)
        except (asyncio.TimeoutError,) + LIFECYCLE_GUARD_EXCEPTIONS as exc:
            logger.debug(
                "Lifecycle startup adapter service cleanup skipped after {}",
                type(exc).__name__,
            )


async def cancel_started_task(
    task: Any,
    *,
    timeout_sec: float = 5.0,
) -> None:
    """Cancel and await a legacy task without letting cleanup block teardown."""

    cancel = getattr(task, "cancel", None)
    if callable(cancel):
        try:
            cancel()
        except LIFECYCLE_GUARD_EXCEPTIONS as exc:
            logger.debug("Lifecycle startup adapter cancel failed: {}", exc)
            return
    if not inspect.isawaitable(task):
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout_sec)
    except asyncio.CancelledError:
        pass
    except (asyncio.TimeoutError,) + LIFECYCLE_GUARD_EXCEPTIONS as exc:
        logger.debug("Lifecycle startup adapter task cleanup skipped: {}", exc)

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.services.lifecycle_workers import (
    ShutdownPhase,
    start_stop_event_worker,
)

_USAGE_AGGREGATOR_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


async def aggregate_usage_daily(db_pool: DatabasePool | None = None, day: str | None = None) -> None:
    """
    Aggregate per-request usage from usage_log into usage_daily.

    Args:
        db_pool: Optional DatabasePool; if None, fetch singleton
        day: Optional ISO date (YYYY-MM-DD). If None, uses current UTC date.
    """
    try:
        pool = db_pool or await get_db_pool()
        # Resolve day (UTC)
        day_val = datetime.now(timezone.utc).date()
        if day:
            try:
                day_val = datetime.fromisoformat(day).date()
            except (TypeError, ValueError):
                day_val = datetime.now(timezone.utc).date()

        repo = AuthnzUsageRepo(pool)
        await repo.aggregate_usage_daily_for_day(day=day_val)
        logger.debug(f"usage_daily aggregated for {day_val.isoformat()}")
    except _USAGE_AGGREGATOR_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(error_type=type(e).__name__).debug("usage_daily aggregation skipped/failed")


async def _aggregator_loop(stop_event: asyncio.Event):
    settings = get_settings()
    if not getattr(settings, "USAGE_LOG_ENABLED", False):
        logger.info("Usage aggregator disabled (USAGE_LOG_ENABLED is false)")
        return
    interval_minutes = int(getattr(settings, "USAGE_AGGREGATOR_INTERVAL_MINUTES", 60) or 60)
    logger.info(f"Starting usage aggregator task (interval: {interval_minutes} min)")
    try:
        while not stop_event.is_set():
            await aggregate_usage_daily()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_minutes * 60)
            except asyncio.TimeoutError:
                continue
    except _USAGE_AGGREGATOR_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(error_type=type(e).__name__).warning("Usage aggregator loop exited")


async def start_usage_aggregator(
    *,
    worker_inventory: Any | None = None,
) -> asyncio.Task | None:
    """Start background usage aggregation if enabled.

    With ``worker_inventory``, register the loop with lifecycle worker
    management while retaining the legacy stop-event task attribute used by
    ``stop_usage_aggregator``. Without inventory, create the legacy local task.
    """
    settings = get_settings()
    if not getattr(settings, "USAGE_LOG_ENABLED", False):
        return None
    if worker_inventory is not None:
        task, stop_event = await start_stop_event_worker(
            worker_inventory,
            name="usage_aggregator",
            task_name="usage_aggregator",
            coroutine_factory=_aggregator_loop,
            category="usage",
            shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        )
        task._tldw_stop_event = stop_event  # type: ignore[attr-defined]
        return task
    stop_event = asyncio.Event()
    task = asyncio.create_task(_aggregator_loop(stop_event))
    # Attach a helper to task for stopping
    task._tldw_stop_event = stop_event  # type: ignore[attr-defined]
    return task


async def stop_usage_aggregator(task: asyncio.Task | None) -> None:
    if not task:
        return
    try:
        stop_event = getattr(task, "_tldw_stop_event", None)
        if isinstance(stop_event, asyncio.Event):
            stop_event.set()
        task.cancel()
    except _USAGE_AGGREGATOR_NONCRITICAL_EXCEPTIONS:
        pass

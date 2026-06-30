from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
    stop_event_worker_spec,
)
from tldw_Server_API.app.services.lifecycle_workers import (
    ShutdownPhase,
    start_stop_event_worker,
)

_LLM_USAGE_AGGREGATOR_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def provide_llm_usage_aggregator_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return the declarative spec for LLM usage aggregation."""

    return (
        stop_event_worker_spec(
            name="llm_usage_aggregator",
            worker_service=_aggregator_loop,
            category="usage",
            phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            enabled=_llm_usage_aggregator_worker_enabled,
        ),
    )


def _llm_usage_aggregator_worker_enabled(_context: WorkerLifecycleContext) -> bool:
    settings = _context.settings
    if settings is None:
        settings = get_settings()
    if isinstance(settings, Mapping):
        return bool(settings.get("LLM_USAGE_AGGREGATOR_ENABLED", True))
    return bool(getattr(settings, "LLM_USAGE_AGGREGATOR_ENABLED", True))


async def aggregate_llm_usage_daily(db_pool: DatabasePool | None = None, day: str | None = None) -> None:
    """
    Aggregate llm_usage_log into llm_usage_daily for a given UTC day.

    Args:
        db_pool: Optional database pool; if None, fetch singleton
        day: Optional ISO date string (YYYY-MM-DD). Defaults to current UTC date.
    """
    try:
        pool = db_pool or await get_db_pool()
        day_val = datetime.now(timezone.utc).date()
        if day:
            try:
                day_val = datetime.fromisoformat(day).date()
            except (TypeError, ValueError):
                day_val = datetime.now(timezone.utc).date()

        repo = AuthnzUsageRepo(pool)
        await repo.aggregate_llm_usage_daily_for_day(day=day_val)

        logger.debug(f"llm_usage_daily aggregated for {day_val.isoformat()}")
    except _LLM_USAGE_AGGREGATOR_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(error_type=type(e).__name__).debug("llm_usage_daily aggregation skipped/failed")


async def _aggregator_loop(stop_event: asyncio.Event):
    settings = get_settings()
    if not getattr(settings, "LLM_USAGE_AGGREGATOR_ENABLED", True):
        logger.info("LLM usage aggregator disabled (LLM_USAGE_AGGREGATOR_ENABLED is false)")
        return
    interval_minutes = int(getattr(settings, "LLM_USAGE_AGGREGATOR_INTERVAL_MINUTES", 60) or 60)
    logger.info(f"Starting LLM usage aggregator task (interval: {interval_minutes} min)")
    try:
        while not stop_event.is_set():
            await aggregate_llm_usage_daily()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_minutes * 60)
            except asyncio.TimeoutError:
                continue
    except _LLM_USAGE_AGGREGATOR_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(error_type=type(e).__name__).warning("LLM usage aggregator loop exited")


async def start_llm_usage_aggregator(
    *,
    worker_inventory: Any | None = None,
) -> asyncio.Task | None:
    """Start background LLM usage aggregation if enabled.

    With ``worker_inventory``, register the loop with lifecycle worker
    management while retaining the legacy stop-event task attribute used by
    ``stop_llm_usage_aggregator``. Without inventory, create the legacy local
    task.
    """
    settings = get_settings()
    if not getattr(settings, "LLM_USAGE_AGGREGATOR_ENABLED", True):
        return None
    if worker_inventory is not None:
        task, stop_event = await start_stop_event_worker(
            worker_inventory,
            name="llm_usage_aggregator",
            task_name="llm_usage_aggregator",
            coroutine_factory=_aggregator_loop,
            category="usage",
            shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        )
        task._tldw_stop_event = stop_event  # type: ignore[attr-defined]
        return task
    stop_event = asyncio.Event()
    task = asyncio.create_task(_aggregator_loop(stop_event))
    task._tldw_stop_event = stop_event  # type: ignore[attr-defined]
    return task


async def stop_llm_usage_aggregator(task: asyncio.Task | None) -> None:
    if not task:
        return
    try:
        stop_event = getattr(task, "_tldw_stop_event", None)
        if isinstance(stop_event, asyncio.Event):
            stop_event.set()
        task.cancel()
    except _LLM_USAGE_AGGREGATOR_NONCRITICAL_EXCEPTIONS:
        pass

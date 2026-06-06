"""Calendar external-sync scheduler that enqueues due Jobs work."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

from loguru import logger

from tldw_Server_API.app.core.Calendar.calendar_sync_worker import (
    CalendarSyncJobResponse,
    queue_calendar_binding_sync,
)
from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase
from tldw_Server_API.app.core.Jobs.manager import JobManager

_SCHEDULER_GUARD_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def queue_due_calendar_sync_jobs(
    *,
    db: CalendarDatabase | None = None,
    job_manager: JobManager | None = None,
    now: datetime | None = None,
    limit: int = 100,
) -> list[CalendarSyncJobResponse]:
    calendar_db = db or CalendarDatabase()
    jobs = job_manager or JobManager()
    scan_at = now or datetime.now(timezone.utc)
    queued: list[CalendarSyncJobResponse] = []
    for binding in calendar_db.list_sync_enabled_bindings_due_for_scan(
        now_iso=scan_at.isoformat(),
        limit=limit,
    ):
        try:
            account = calendar_db.get_external_account(binding.account_id)
            queued.append(
                queue_calendar_binding_sync(
                    db=calendar_db,
                    job_manager=jobs,
                    actor_user_id=account.user_id,
                    tenant_id=account.tenant_id,
                    binding_id=binding.id,
                    reason="scheduled",
                    window_start=(scan_at - timedelta(days=int(binding.lookback_days))).isoformat(),
                    window_end=(scan_at + timedelta(days=int(binding.lookahead_days))).isoformat(),
                )
            )
        except CalendarValidationError as exc:
            logger.warning("Calendar sync scheduler skipped binding {}: {}", binding.id, exc)
        except _SCHEDULER_GUARD_EXCEPTIONS as exc:
            logger.warning("Calendar sync scheduler failed to queue binding {}: {}", binding.id, exc)
    return queued


async def run_calendar_sync_scheduler(
    stop_event: asyncio.Event | None = None,
    *,
    interval_seconds: float | None = None,
    db: CalendarDatabase | None = None,
    job_manager: JobManager | None = None,
) -> None:
    interval = interval_seconds if interval_seconds is not None else _scheduler_interval_seconds()
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        await queue_due_calendar_sync_jobs(db=db, job_manager=job_manager)
        if stop_event is None:
            await asyncio.sleep(interval)
            continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            continue


def _scheduler_interval_seconds() -> float:
    try:
        return max(5.0, float(os.getenv("CALENDAR_SYNC_SCHEDULER_INTERVAL_SECONDS", "60") or "60"))
    except _SCHEDULER_GUARD_EXCEPTIONS:
        return 60.0


__all__ = [
    "queue_due_calendar_sync_jobs",
    "run_calendar_sync_scheduler",
]

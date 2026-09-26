"""Calendar external-sync scheduler that enqueues due Jobs work."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from tldw_Server_API.app.core.Calendar.calendar_sync_worker import (
    CalendarSyncJobResponse,
    _run_db_phase,
    queue_calendar_binding_sync,
)
from tldw_Server_API.app.core.Calendar.errors import CalendarError
from tldw_Server_API.app.core.Calendar.provider_operations import log_calendar_failure
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
    sqlite3.Error,
)


async def queue_due_calendar_sync_jobs(
    *,
    db: CalendarDatabase | None = None,
    job_manager: JobManager | None = None,
    now: datetime | None = None,
    limit: int = 100,
) -> list[CalendarSyncJobResponse]:
    """Queue due bindings off-loop and drain the complete synchronous phase.

    Args:
        db: Repository to use; None initializes the default repository off-loop.
        job_manager: Jobs manager to use; None initializes the default off-loop.
        now: Scan timestamp; None samples UTC time after repository initialization.
        limit: Due-scan limit forwarded unchanged to the repository.

    Returns:
        Responses for successfully processed bindings, including reused active jobs.

    Raises:
        asyncio.CancelledError: Re-raises caller cancellation only after started
            DB work and transactions finish. An unhandled phase failure is retained
            as the cancellation's cause.
        Exception: Propagates unhandled initialization, due-scan, or per-binding
            failures when not cancelled. Per-binding CalendarError and scheduler
            guard exceptions are safely logged and skipped.
    """
    return await _run_db_phase(_queue_due_calendar_sync_jobs, db, job_manager, now, limit)


def _queue_due_calendar_sync_jobs(
    db: CalendarDatabase | None,
    job_manager: JobManager | None,
    now: datetime | None,
    limit: int,
) -> list[CalendarSyncJobResponse]:
    """Initialize, scan, authorize, and queue entirely on the owning DB thread.

    Args:
        db: Repository to use, or None to initialize the default repository.
        job_manager: Jobs manager to use, or None to initialize the default manager.
        now: Scan timestamp, or None to sample the current UTC time.
        limit: Due-scan limit forwarded unchanged to the repository.

    Returns:
        Responses for successfully queued or already-active binding jobs.

    Raises:
        Exception: Propagates initialization, due-scan, and unguarded per-binding
            failures. Per-binding CalendarError and scheduler guard exceptions are
            safely logged and skipped so later bindings can still be queued.
    """
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
        except CalendarError as exc:
            log_calendar_failure("queue_binding", exc, binding_id=binding.id, account_id=binding.account_id)
        except _SCHEDULER_GUARD_EXCEPTIONS as exc:
            log_calendar_failure("queue_binding", exc, binding_id=binding.id, account_id=binding.account_id)
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
        try:
            await queue_due_calendar_sync_jobs(db=db, job_manager=job_manager)
        except (CalendarError, *_SCHEDULER_GUARD_EXCEPTIONS) as exc:
            log_calendar_failure("scan_due_bindings", exc)
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

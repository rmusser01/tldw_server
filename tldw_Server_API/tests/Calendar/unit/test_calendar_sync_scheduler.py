"""Scheduler responsiveness, cancellation ownership, and bounded queue regressions."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import pytest
from anyio import CancelScope
from loguru import logger

from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase, ExternalCalendarBindingRow
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.services import calendar_sync_scheduler as scheduler

pytestmark = pytest.mark.unit
_SCAN_AT = datetime(2026, 6, 10, 12, tzinfo=timezone.utc)


@pytest.fixture
def calendar_db(tmp_path: Path) -> CalendarDatabase:
    """Use a real isolated repository for queue and transaction assertions."""
    return CalendarDatabase(db_path=tmp_path / "calendar.db")


@pytest.fixture
def jobs_manager(tmp_path: Path) -> JobManager:
    """Keep Jobs persistence real without touching the default database."""
    path = tmp_path / "jobs.db"
    ensure_jobs_tables(path)
    return JobManager(path)


def _binding(db: CalendarDatabase) -> ExternalCalendarBindingRow:
    """Create an owner-scoped hourly binding without needing provider credentials."""
    calendar = db.create_calendar(tenant_id="default", owner_user_id=1, org_id=None, name="Imported", timezone="UTC")
    account = db.create_external_account(
        tenant_id="default", user_id=1, provider="caldav", display_name="Read-only provider", secret_ref=None,
    )
    return db.create_external_binding(
        account_id=account.id, calendar_id=calendar.id,
        remote_calendar_id="https://caldav.example.test/calendar/", lookback_days=14, lookahead_days=30,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("delayed_operation", [
    "calendar_init", "jobs_init", "scan", "account", "create_job", "audit",
])
async def test_scheduler_heartbeat_runs_during_complete_synchronous_phase(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
    delayed_operation: str,
) -> None:
    """Initialization, reads, Jobs, and audit transactions cannot block another coroutine."""
    binding = _binding(calendar_db)
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    calls: list[tuple[str, int]] = []
    ticks: list[bool] = []

    def trace(operation: str, original: Any) -> Any:
        """Delay one real operation while recording all phase thread ownership."""
        @wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            calls.append((operation, threading.get_ident()))
            if operation == delayed_operation and not finished.is_set():
                loop.call_soon_threadsafe(started.set)
                try:
                    if not release.wait(timeout=5):
                        raise TimeoutError("Scheduler test did not release the DB operation")
                    return original(*args, **kwargs)
                finally:
                    finished.set()
            return original(*args, **kwargs)
        return wrapped

    @contextmanager
    def transaction(*args: Any, **kwargs: Any) -> Iterator[Any]:
        """Observe real audit transaction entry and exit on its owning thread."""
        calls.append(("transaction_enter", threading.get_ident()))
        try:
            with original_transaction(*args, **kwargs) as conn:
                yield conn
        finally:
            calls.append(("transaction_exit", threading.get_ident()))

    async def heartbeat() -> None:
        """Release the DB gate only after the event loop makes independent progress."""
        await started.wait()
        for _ in range(3):
            ticks.append(not finished.is_set())
            await asyncio.sleep(0)
        release.set()

    original_transaction = CalendarDatabase.transaction
    monkeypatch.setattr(scheduler, "CalendarDatabase", trace(
        "calendar_init", lambda: CalendarDatabase(calendar_db.db_path),
    ))
    monkeypatch.setattr(scheduler, "JobManager", trace("jobs_init", lambda: JobManager(jobs_manager.db_path)))
    for target, method, label in [
        (CalendarDatabase, "list_sync_enabled_bindings_due_for_scan", "scan"),
        (CalendarDatabase, "get_external_account", "account"),
        (CalendarDatabase, "record_sync_event", "audit"),
        (JobManager, "create_job", "create_job"),
    ]:
        monkeypatch.setattr(target, method, trace(label, getattr(target, method)))
    monkeypatch.setattr(CalendarDatabase, "transaction", transaction)
    beat = asyncio.create_task(heartbeat())
    # The watchdog releases the baseline's blocked event loop; all threads are joined below.
    watchdog = threading.Timer(0.2, release.set)
    watchdog.start()
    try:
        queued = await scheduler.queue_due_calendar_sync_jobs(now=_SCAN_AT)
    finally:
        release.set()
        watchdog.cancel()
        watchdog.join()
        await asyncio.wait_for(beat, timeout=5)

    assert ticks == [True, True, True], f"{delayed_operation} blocked the heartbeat"
    assert {thread_id for _, thread_id in calls} != {loop_thread}
    assert len({thread_id for _, thread_id in calls}) == 1
    assert {label for label, _ in calls} >= {
        "calendar_init", "jobs_init", "scan", "account", "create_job", "audit",
        "transaction_enter", "transaction_exit",
    }
    assert queued[0].binding_id == binding.id
    assert jobs_manager.count_jobs(domain="calendar") == 1
    audits = calendar_db.list_sync_events(binding_id=binding.id)
    assert len(audits) == 1 and audits[0].event_type == "sync_queued"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation_kind", ["native", "anyio"])
@pytest.mark.parametrize("phase", ["scan", "audit", "failed_scan"])
async def test_scheduler_cancellation_drains_owner_thread_without_hot_retry_or_orphan_job(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
    cancellation_kind: str, phase: str,
) -> None:
    """Cancellation waits for the entire scan/queue/audit phase, including commit or rollback."""
    binding = _binding(calendar_db)
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    started = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    scopes: list[CancelScope] = []
    retries: list[None] = []
    exited_early: list[bool] = []
    transaction_calls: list[tuple[str, int]] = []
    scans: list[None] = []
    cancellations: list[asyncio.CancelledError] = []
    failure = RuntimeError("scan failed password=do-not-log-this")
    original_scan = calendar_db.list_sync_enabled_bindings_due_for_scan
    original_audit = calendar_db.record_sync_event
    original_transaction = calendar_db.transaction
    original_shield = asyncio.shield
    task: asyncio.Task[None] | None = None

    def pause() -> None:
        """Hold the owning thread until the cancellation observer explicitly releases it."""
        loop.call_soon_threadsafe(started.set)
        if not release.wait(timeout=5):
            raise TimeoutError("Scheduler test did not release cancelled DB work")

    def scan(**kwargs: Any) -> list[ExternalCalendarBindingRow]:
        """Leave real due queries and their queue continuation intact."""
        scans.append(None)
        if phase in {"scan", "failed_scan"}:
            try:
                pause()
                if phase == "failed_scan":
                    raise failure
                return original_scan(**kwargs)
            finally:
                finished.set()
        return original_scan(**kwargs)

    @contextmanager
    def transaction() -> Iterator[Any]:
        """Record complete transaction lifetimes rather than only method entry."""
        transaction_calls.append(("enter", threading.get_ident()))
        try:
            with original_transaction() as conn:
                yield conn
        finally:
            transaction_calls.append(("exit", threading.get_ident()))

    def audit(**kwargs: Any) -> Any:
        """Pause with a real uncommitted audit and a job already persisted."""
        try:
            with calendar_db.transaction():
                result = original_audit(**kwargs)
                pause()
            return result
        finally:
            finished.set()

    def counted_shield(awaitable: Any) -> asyncio.Future[Any]:
        """Detect a drain that spins under level-triggered scope cancellation."""
        if asyncio.current_task() is task and started.is_set():
            retries.append(None)
        return original_shield(awaitable)

    async def run() -> None:
        """Exercise cancellation through the periodic runner, not just the queue helper."""
        try:
            if cancellation_kind == "anyio":
                with CancelScope() as scope:
                    scopes.append(scope)
                    await scheduler.run_calendar_sync_scheduler(
                        interval_seconds=60, db=calendar_db, job_manager=jobs_manager,
                    )
            else:
                await scheduler.run_calendar_sync_scheduler(
                    interval_seconds=60, db=calendar_db, job_manager=jobs_manager,
                )
        except asyncio.CancelledError as exc:
            cancellations.append(exc)
            assert finished.is_set(), "Cancellation escaped while DB work was still running"
            raise

    monkeypatch.setattr(calendar_db, "list_sync_enabled_bindings_due_for_scan", scan)
    monkeypatch.setattr(calendar_db, "transaction", transaction)
    if phase == "audit":
        monkeypatch.setattr(calendar_db, "record_sync_event", audit)
    monkeypatch.setattr(asyncio, "shield", counted_shield)
    watchdog = threading.Timer(1, release.set)
    watchdog.start()
    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        blocked_when_cancelled = not finished.is_set()
        if cancellation_kind == "native":
            for _ in range(3):
                task.cancel("scheduler shutdown")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                exited_early.append(task.done())
        else:
            scopes[0].cancel()
            for _ in range(20):
                await asyncio.sleep(0)
            exited_early.append(task.done())
        retry_count = len(retries)
    finally:
        release.set()
        watchdog.cancel()
        watchdog.join()
        if cancellation_kind == "native":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)
        else:
            await asyncio.wait_for(task, timeout=5)

    assert blocked_when_cancelled, "Scheduler blocked cancellation delivery until its DB work ended"
    assert not any(exited_early), "Scheduler abandoned its owning DB thread"
    assert retry_count <= 3, f"Cancellation caused {retry_count} hot drain retries"
    assert len(scans) == 1, "Cancelled scheduler restarted its scan"
    assert finished.is_set()
    assert calendar_db._transaction_connection.get() is None
    assert all(thread_id != loop_thread for _, thread_id in transaction_calls)
    assert len({thread_id for _, thread_id in transaction_calls}) <= 1
    assert [label for label, _ in transaction_calls].count("enter") == [
        label for label, _ in transaction_calls
    ].count("exit")
    if cancellation_kind == "native":
        assert str(cancellations[0]) == "scheduler shutdown"
        if phase == "failed_scan":
            assert cancellations[0].__cause__ is failure
    else:
        assert scopes[0].cancelled_caught
    audits = calendar_db.list_sync_events(binding_id=binding.id)
    assert jobs_manager.count_jobs(domain="calendar") == (0 if phase == "failed_scan" else 1)
    assert len(audits) == (0 if phase == "failed_scan" else 1)
    if audits:
        assert audits[0].event_type == "sync_queued"
        assert jobs_manager.get_job(json.loads(audits[0].metadata_json)["job_id"]) is not None


@pytest.mark.asyncio
async def test_scheduler_preserves_scan_bound_windows_hourly_and_manual_cadence(
    calendar_db: CalendarDatabase, jobs_manager: JobManager,
) -> None:
    """A bounded scan queues only due periodic bindings and reuses active Jobs."""
    manual = _binding(calendar_db)
    calendar_db.update_external_binding(manual.id, sync_interval_minutes=None)
    future = _binding(calendar_db)
    calendar_db.update_binding_sync_state(future.id, next_scan_at="2026-06-10T13:00:00+00:00")
    first = _binding(calendar_db)
    second = _binding(calendar_db)
    queued = await scheduler.queue_due_calendar_sync_jobs(
        db=calendar_db, job_manager=jobs_manager, now=_SCAN_AT, limit=1,
    )
    repeated = await scheduler.queue_due_calendar_sync_jobs(
        db=calendar_db, job_manager=jobs_manager, now=_SCAN_AT, limit=1,
    )
    assert [row.binding_id for row in queued] == [first.id]
    assert repeated[0].job_id == queued[0].job_id and repeated[0].queued is False
    assert first.sync_interval_minutes == 60
    assert jobs_manager.get_job(queued[0].job_id)["payload"] == {
        "binding_id": first.id, "reason": "scheduled",
        "window_start": "2026-05-27T12:00:00+00:00", "window_end": "2026-07-10T12:00:00+00:00",
    }
    assert jobs_manager.count_jobs(domain="calendar") == 1
    assert len(calendar_db.list_sync_events(binding_id=first.id)) == 1
    assert all(calendar_db.list_sync_events(binding_id=row.id) == [] for row in [manual, future, second])


@pytest.mark.asyncio
async def test_scheduler_guard_failure_is_redacted_and_later_binding_is_queued(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving queue work off-loop retains per-binding continuation and secret-free diagnostics."""
    broken = _binding(calendar_db)
    healthy = _binding(calendar_db)
    original = calendar_db.get_external_account
    messages: list[Any] = []

    def account(account_id: int) -> Any:
        """Fail a real account lookup with sensitive text only for the first binding."""
        if account_id == broken.account_id:
            raise RuntimeError("password=do-not-log-this")
        return original(account_id)

    monkeypatch.setattr(calendar_db, "get_external_account", account)
    sink = logger.add(messages.append)
    try:
        queued = await scheduler.queue_due_calendar_sync_jobs(db=calendar_db, job_manager=jobs_manager, now=_SCAN_AT)
    finally:
        logger.remove(sink)
    assert [row.binding_id for row in queued] == [healthy.id]
    assert jobs_manager.count_jobs(domain="calendar") == 1
    assert calendar_db.list_sync_events(binding_id=broken.id) == []
    assert len(messages) == 1
    assert "do-not-log-this" not in str(messages[0]) + str(messages[0].record)
    assert messages[0].record["extra"]["binding_id"] == broken.id
    assert messages[0].record["extra"]["error_type"] == "RuntimeError"
    assert messages[0].record["extra"]["frames"]

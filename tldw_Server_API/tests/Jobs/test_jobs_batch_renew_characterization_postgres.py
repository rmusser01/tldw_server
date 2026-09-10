"""Characterize the public PostgreSQL batch lease renewal contract."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
import pytest
from psycopg import sql as psycopg_sql

from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = [pytest.mark.integration, pytest.mark.pg_jobs]

NOW = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
WORKER_ID = "worker-1"
LEASE_ID = "lease-1"


class RecordingClock:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.calls = 0

    def now_utc(self) -> datetime:
        self.calls += 1
        return self.now


def _manager(jobs_pg_dsn: str, clock: RecordingClock, monkeypatch: pytest.MonkeyPatch) -> JobManager:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "false")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn, clock=clock)


def _execute(manager: JobManager, sql: str, params: tuple[Any, ...] = ()) -> None:
    connection = manager._connect()
    try:
        with connection, manager._pg_cursor(connection) as cursor:
            cursor.execute(sql, params)
    finally:
        connection.close()


def _insert_job(
    manager: JobManager,
    *,
    status: str = "processing",
    worker_id: str | None = WORKER_ID,
    lease_id: str | None = LEASE_ID,
    leased_until: datetime | None = NOW - timedelta(minutes=1),
) -> int:
    job = manager.create_job(
        domain="characterization",
        queue="default",
        job_type="renew",
        payload={},
        owner_user_id="owner-1",
    )
    job_id = int(job["id"])
    _execute(
        manager,
        "UPDATE jobs SET status = %s, worker_id = %s, lease_id = %s, leased_until = %s WHERE id = %s",
        (status, worker_id, lease_id, leased_until, job_id),
    )
    return job_id


def _fetch_lease(manager: JobManager, job_id: int) -> datetime | None:
    connection = manager._connect()
    try:
        with connection, manager._pg_cursor(connection) as cursor:
            cursor.execute("SELECT leased_until FROM jobs WHERE id = %s", (job_id,))
            row = cursor.fetchone()
    finally:
        connection.close()
    assert row is not None
    return row["leased_until"]


def test_batch_renew_counts_ordered_attempts_and_preserves_longer_lease_postgres(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = RecordingClock(NOW)
    manager = _manager(jobs_pg_dsn, clock, monkeypatch)
    valid_id = _insert_job(manager, status="processing", worker_id="worker-1", lease_id="lease-1")
    queued_id = _insert_job(manager, status="queued", worker_id=None, lease_id=None, leased_until=None)
    stale_id = _insert_job(manager, status="processing", worker_id="worker-1", lease_id="lease-1")
    long_id = _insert_job(
        manager,
        status="processing",
        worker_id="worker-1",
        lease_id="lease-1",
        leased_until=NOW + timedelta(minutes=10),
    )

    items = [
        {"job_id": valid_id, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
        {"job_id": 999_999, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
        {"job_id": queued_id, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
        {"job_id": stale_id, "seconds": 30, "worker_id": "worker-2", "lease_id": "lease-1"},
        {"job_id": valid_id, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
        {"job_id": long_id, "seconds": 30, "worker_id": "worker-1", "lease_id": "lease-1"},
    ]

    assert manager.batch_renew_leases(items, enforce=True) == 3
    assert _fetch_lease(manager, valid_id) == NOW + timedelta(seconds=30)
    assert _fetch_lease(manager, queued_id) is None
    assert _fetch_lease(manager, stale_id) == NOW - timedelta(minutes=1)
    assert _fetch_lease(manager, long_id) == NOW + timedelta(minutes=10)


def test_batch_renew_clamps_each_item_and_reads_the_clock_once_postgres(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "60")
    clock = RecordingClock(NOW)
    manager = _manager(jobs_pg_dsn, clock, monkeypatch)
    one_second_id = _insert_job(manager)
    thirty_second_id = _insert_job(manager)
    sixty_second_id = _insert_job(manager)
    clock.calls = 0

    assert manager.batch_renew_leases(
        [
            {"job_id": one_second_id, "seconds": 0, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
            {"job_id": thirty_second_id, "seconds": 30, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
            {"job_id": sixty_second_id, "seconds": 120, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
        ],
        enforce=True,
    ) == 3
    assert _fetch_lease(manager, one_second_id) == NOW + timedelta(seconds=1)
    assert _fetch_lease(manager, thirty_second_id) == NOW + timedelta(seconds=30)
    assert _fetch_lease(manager, sixty_second_id) == NOW + timedelta(seconds=60)
    assert clock.calls == 1


def test_batch_renew_empty_batch_reads_the_clock_without_lease_cap_lookup_postgres(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_getenv = os.getenv

    def guarded_getenv(key: str, default: Any = None) -> Any:
        if key == "JOBS_LEASE_MAX_SECONDS":
            raise AssertionError("empty batch must not read the lease cap")
        return original_getenv(key, default)

    monkeypatch.setattr(os, "getenv", guarded_getenv)
    clock = RecordingClock(NOW)
    manager = _manager(jobs_pg_dsn, clock, monkeypatch)

    assert manager.batch_renew_leases([], enforce=True) == 0
    assert clock.calls == 1


def test_batch_renew_rolls_back_prior_update_when_a_later_item_is_malformed_postgres(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = RecordingClock(NOW)
    manager = _manager(jobs_pg_dsn, clock, monkeypatch)
    first_job_id = _insert_job(manager)
    second_job_id = _insert_job(manager)
    original_first_lease = _fetch_lease(manager, first_job_id)
    original_second_lease = _fetch_lease(manager, second_job_id)

    with pytest.raises(ValueError):
        manager.batch_renew_leases(
            [
                {"job_id": first_job_id, "seconds": 30, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
                {"job_id": "not-an-int", "seconds": 30},
            ],
            enforce=True,
        )

    assert _fetch_lease(manager, first_job_id) == original_first_lease
    assert _fetch_lease(manager, second_job_id) == original_second_lease


def test_batch_renew_rolls_back_every_update_on_database_failure_postgres(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = RecordingClock(NOW)
    manager = _manager(jobs_pg_dsn, clock, monkeypatch)
    first_job_id = _insert_job(manager)
    second_job_id = _insert_job(manager)
    original_first_lease = _fetch_lease(manager, first_job_id)
    original_second_lease = _fetch_lease(manager, second_job_id)
    function_name = f"force_batch_renew_function_{uuid.uuid4().hex}"
    trigger_name = f"force_batch_renew_trigger_{uuid.uuid4().hex}"
    connection = manager._connect()
    try:
        with connection, manager._pg_cursor(connection) as cursor:
            cursor.execute(
                psycopg_sql.SQL(
                    "CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS $$ "
                    "BEGIN RAISE EXCEPTION 'forced batch renewal failure'; END; $$"
                ).format(psycopg_sql.Identifier(function_name))
            )
            cursor.execute(
                psycopg_sql.SQL(
                    "CREATE TRIGGER {} BEFORE UPDATE ON jobs "
                    "FOR EACH ROW WHEN (OLD.id = {}) EXECUTE FUNCTION {}()"
                ).format(
                    psycopg_sql.Identifier(trigger_name),
                    psycopg_sql.Literal(second_job_id),
                    psycopg_sql.Identifier(function_name),
                )
            )
    finally:
        connection.close()

    try:
        with pytest.raises(psycopg.Error, match="forced batch renewal failure"):
            manager.batch_renew_leases(
                [
                    {"job_id": first_job_id, "seconds": 30, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
                    {"job_id": second_job_id, "seconds": 30, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
                ],
                enforce=True,
            )
        assert _fetch_lease(manager, first_job_id) == original_first_lease
        assert _fetch_lease(manager, second_job_id) == original_second_lease
    finally:
        cleanup_connection = manager._connect()
        try:
            with cleanup_connection, manager._pg_cursor(cleanup_connection) as cursor:
                cursor.execute(
                    psycopg_sql.SQL("DROP TRIGGER IF EXISTS {} ON jobs").format(
                        psycopg_sql.Identifier(trigger_name)
                    )
                )
                cursor.execute(
                    psycopg_sql.SQL("DROP FUNCTION IF EXISTS {}()").format(
                        psycopg_sql.Identifier(function_name)
                    )
                )
        finally:
            cleanup_connection.close()

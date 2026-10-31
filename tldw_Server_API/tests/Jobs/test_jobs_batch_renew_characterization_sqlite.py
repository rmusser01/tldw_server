"""Characterize the public SQLite batch lease renewal contract."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables

pytestmark = pytest.mark.integration

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


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "jobs.db"
    ensure_jobs_tables(path)
    return path


def _format_sqlite_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _insert_job(
    db_path: Path,
    *,
    status: str = "processing",
    worker_id: str | None = WORKER_ID,
    lease_id: str | None = LEASE_ID,
    leased_until: datetime | None = NOW - timedelta(minutes=1),
) -> int:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            row_count = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            cursor = connection.execute(
                "INSERT INTO jobs (uuid, domain, queue, job_type, payload, status, "
                "worker_id, lease_id, leased_until) "
                "VALUES (?, 'characterization', 'default', 'renew', '{}', ?, ?, ?, ?)",
                (
                    f"characterization-{row_count + 1}",
                    status,
                    worker_id,
                    lease_id,
                    _format_sqlite_datetime(leased_until) if leased_until else None,
                ),
            )
            return int(cursor.lastrowid)
    finally:
        connection.close()


def _fetch_lease(db_path: Path, job_id: int) -> str | None:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            "SELECT leased_until FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    finally:
        connection.close()
    assert row is not None
    return row["leased_until"]


def _manager(db_path: Path, clock: RecordingClock) -> JobManager:
    return JobManager(db_path, clock=clock)


def test_batch_renew_counts_ordered_attempts_and_preserves_longer_lease_sqlite(db_path: Path) -> None:
    clock = RecordingClock(NOW)
    manager = _manager(db_path, clock)
    valid_id = _insert_job(db_path, status="processing", worker_id="worker-1", lease_id="lease-1")
    queued_id = _insert_job(db_path, status="queued", worker_id=None, lease_id=None, leased_until=None)
    stale_id = _insert_job(db_path, status="processing", worker_id="worker-1", lease_id="lease-1")
    long_id = _insert_job(
        db_path,
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
    assert _fetch_lease(db_path, valid_id) == _format_sqlite_datetime(NOW + timedelta(seconds=30))
    assert _fetch_lease(db_path, queued_id) is None
    assert _fetch_lease(db_path, stale_id) == _format_sqlite_datetime(NOW - timedelta(minutes=1))
    assert _fetch_lease(db_path, long_id) == _format_sqlite_datetime(NOW + timedelta(minutes=10))


def test_batch_renew_clamps_each_item_and_reads_the_clock_per_item_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    db_path: Path,
) -> None:
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "60")
    clock = RecordingClock(NOW)
    manager = _manager(db_path, clock)
    one_second_id = _insert_job(db_path)
    thirty_second_id = _insert_job(db_path)
    sixty_second_id = _insert_job(db_path)

    assert manager.batch_renew_leases(
        [
            {"job_id": one_second_id, "seconds": 0, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
            {"job_id": thirty_second_id, "seconds": 30, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
            {"job_id": sixty_second_id, "seconds": 120, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
        ],
        enforce=True,
    ) == 3
    assert _fetch_lease(db_path, one_second_id) == _format_sqlite_datetime(NOW + timedelta(seconds=1))
    assert _fetch_lease(db_path, thirty_second_id) == _format_sqlite_datetime(NOW + timedelta(seconds=30))
    assert _fetch_lease(db_path, sixty_second_id) == _format_sqlite_datetime(NOW + timedelta(seconds=60))
    assert clock.calls == 3


def test_batch_renew_empty_batch_skips_clock_and_lease_cap_lookup_sqlite(
    monkeypatch: pytest.MonkeyPatch,
    db_path: Path,
) -> None:
    original_getenv = os.getenv

    def guarded_getenv(key: str, default: Any = None) -> Any:
        if key == "JOBS_LEASE_MAX_SECONDS":
            raise AssertionError("empty batch must not read the lease cap")
        return original_getenv(key, default)

    monkeypatch.setattr(os, "getenv", guarded_getenv)
    clock = RecordingClock(NOW)

    assert _manager(db_path, clock).batch_renew_leases([], enforce=True) == 0
    assert clock.calls == 0


def test_batch_renew_rolls_back_prior_update_when_a_later_item_is_malformed_sqlite(db_path: Path) -> None:
    clock = RecordingClock(NOW)
    manager = _manager(db_path, clock)
    first_job_id = _insert_job(db_path)
    second_job_id = _insert_job(db_path)
    original_first_lease = _fetch_lease(db_path, first_job_id)
    original_second_lease = _fetch_lease(db_path, second_job_id)

    with pytest.raises(ValueError):
        manager.batch_renew_leases(
            [
                {"job_id": first_job_id, "seconds": 30, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
                {"job_id": "not-an-int", "seconds": 30},
            ],
            enforce=True,
        )

    assert _fetch_lease(db_path, first_job_id) == original_first_lease
    assert _fetch_lease(db_path, second_job_id) == original_second_lease


def test_batch_renew_rolls_back_every_update_on_database_failure_sqlite(db_path: Path) -> None:
    clock = RecordingClock(NOW)
    manager = _manager(db_path, clock)
    first_job_id = _insert_job(db_path)
    second_job_id = _insert_job(db_path, worker_id="worker-2")
    original_first_lease = _fetch_lease(db_path, first_job_id)
    original_second_lease = _fetch_lease(db_path, second_job_id)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TRIGGER force_batch_renew_failure BEFORE UPDATE ON jobs "
            "WHEN OLD.worker_id = 'worker-2' "
            "BEGIN SELECT RAISE(ABORT, 'forced batch renewal failure'); END"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(sqlite3.IntegrityError, match="forced batch renewal failure"):
        manager.batch_renew_leases(
            [
                {"job_id": first_job_id, "seconds": 30, "worker_id": WORKER_ID, "lease_id": LEASE_ID},
                {"job_id": second_job_id, "seconds": 30, "worker_id": "worker-2", "lease_id": LEASE_ID},
            ],
            enforce=True,
        )

    assert _fetch_lease(db_path, first_job_id) == original_first_lease
    assert _fetch_lease(db_path, second_job_id) == original_second_lease

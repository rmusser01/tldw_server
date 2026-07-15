import sqlite3
import threading
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.operations.contracts import CreateJobCommand
from tldw_Server_API.app.core.Jobs.operations.sqlite.admission import create_job_admission


class _DelayedInsertJobManager(JobManager):
    def _connect(self):
        conn = super()._connect()
        conn.create_function("jobs_test_sleep", 1, time.sleep)
        return conn


class _RecordingConnection:
    def __init__(self, inner: sqlite3.Connection) -> None:
        self._inner = inner
        self.statements: list[str] = []

    def __enter__(self) -> "_RecordingConnection":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        self.statements.append(str(sql).strip())
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _admission_command(*, owner_user_id: str, job_type: str) -> CreateJobCommand:
    return CreateJobCommand(
        domain="quota-fast-path",
        queue="default",
        job_type=job_type,
        payload={},
        owner_user_id=owner_user_id,
        priority=5,
    )


def test_max_queued_quota_serializes_concurrent_admission(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_quota_race.db"
    ensure_jobs_tables(db_path)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")

    managers = [_DelayedInsertJobManager(db_path), _DelayedInsertJobManager(db_path)]
    conn = managers[0]._connect()
    try:
        conn.execute(
            """
            CREATE TRIGGER jobs_test_delay_admission
            BEFORE INSERT ON jobs
            BEGIN
                SELECT jobs_test_sleep(0.20);
            END
            """
        )
        conn.commit()
    finally:
        conn.close()

    barrier = threading.Barrier(2)
    results = []
    results_lock = threading.Lock()

    def submit(manager, job_type):
        try:
            barrier.wait(timeout=5)
            row = manager.create_job(
                domain="quota-race",
                queue="default",
                job_type=job_type,
                payload={},
                owner_user_id="owner-1",
            )
        except ValueError as exc:
            outcome = ("rejected", str(exc))
        except sqlite3.OperationalError as exc:
            outcome = ("operational-error", str(exc))
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - reports worker failures
            outcome = ("error", repr(exc))
        else:
            outcome = ("created", row)
        with results_lock:
            results.append(outcome)

    threads = [
        threading.Thread(target=submit, args=(managers[0], "race-a")),
        threading.Thread(target=submit, args=(managers[1], "race-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcome for outcome, _ in results) == ["created", "rejected"]
    assert [detail for outcome, detail in results if outcome == "rejected"] == [
        "Quota exceeded: max queued per user/domain"
    ]
    assert managers[0].count_jobs(domain="quota-race", owner_user_id="owner-1", status="queued") == 1


def test_sqlite_immediate_transaction_is_limited_to_enabled_owner_quota(tmp_path):
    db_path = ensure_jobs_tables(tmp_path / "jobs_quota_fast_path.db")
    inner = sqlite3.connect(db_path)
    inner.row_factory = sqlite3.Row
    conn = _RecordingConnection(inner)
    try:
        create_job_admission(
            conn,
            command=_admission_command(owner_user_id="owner-fast", job_type="fast"),
            uuid_value="uuid-fast",
            now=datetime(2026, 1, 1, tzinfo=UTC),
            max_queued_quota=0,
            submits_per_minute_quota=0,
            counters_enabled=False,
        )
        assert all(statement.upper() != "BEGIN IMMEDIATE" for statement in conn.statements)

        conn.statements.clear()
        create_job_admission(
            conn,
            command=_admission_command(owner_user_id="owner-locked", job_type="locked"),
            uuid_value="uuid-locked",
            now=datetime(2026, 1, 1, tzinfo=UTC),
            max_queued_quota=1,
            submits_per_minute_quota=0,
            counters_enabled=False,
        )
        assert conn.statements[0].upper() == "BEGIN IMMEDIATE"
    finally:
        inner.close()


def test_max_queued_quota_sqlite(monkeypatch, tmp_path):


    db_path = tmp_path / "jobs_quota.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    # Global max queued per user/domain
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")

    # First job for user 1 in domain chatbooks succeeds
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    # Second job for same user/domain should hit quota
    with pytest.raises(ValueError):
        jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    # Different user should not be blocked by user-specific quota
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="2")

    # Different domain should not be blocked by domain scoping
    jm.create_job(domain="other", queue="default", job_type="t", payload={}, owner_user_id="1")


def test_submits_per_minute_quota_precedence_sqlite(monkeypatch, tmp_path):


    db_path = tmp_path / "jobs_quota_spm.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    # Global limit 1/min; domain+user override to 2/min should take precedence
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN", "1")
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN_CHATBOOKS_USER_1", "2")

    # Two submits within a minute for domain chatbooks, user 1 should be allowed
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t2", payload={}, owner_user_id="1")

    # Third submit should be blocked by the 2/min override
    with pytest.raises(ValueError):
        jm.create_job(domain="chatbooks", queue="default", job_type="t3", payload={}, owner_user_id="1")

    # For another domain, the global 1/min applies; second submit should fail
    jm.create_job(domain="other", queue="default", job_type="x", payload={}, owner_user_id="1")
    with pytest.raises(ValueError):
        jm.create_job(domain="other", queue="default", job_type="y", payload={}, owner_user_id="1")


def test_max_inflight_quota_sqlite(monkeypatch, tmp_path):


    db_path = tmp_path / "jobs_quota_inflight.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    # Enforce max inflight of 1 per user/domain
    monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")

    # Seed two queued jobs for user 1
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    # First acquire succeeds when passing owner_user_id for quota scope
    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1", owner_user_id="1")
    assert acq1 is not None

    # Second acquire for same owner should be blocked by inflight quota
    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2", owner_user_id="1")
    assert acq2 is None

    # Different user is not blocked by user-specific inflight quota
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="2")
    acq_other = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w3", owner_user_id="2")
    assert acq_other is not None


def test_max_inflight_ignores_expired_leases_sqlite(monkeypatch, tmp_path):


    db_path = tmp_path / "jobs_quota_inflight_expired.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")

    # Two queued jobs for user 1
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1", owner_user_id="1")
    assert acq1 is not None

    # Expire the lease so the processing job should not count toward inflight
    conn = jm._connect()
    try:
        conn.execute(
            "UPDATE jobs SET leased_until = DATETIME('now', '-10 seconds') WHERE id = ?",
            (int(acq1["id"]),),
        )
        conn.commit()
    finally:
        conn.close()

    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2", owner_user_id="1")
    assert acq2 is not None

import json
import sqlite3
import threading
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.operations.contracts import CreateJobCommand
from tldw_Server_API.app.core.Jobs.operations.sqlite.admission import create_job_admission


def _run_concurrent_calls(
    calls: list[Callable[[], dict[str, Any]]],
    *,
    release_events: Iterable[threading.Event] = (),
) -> list[tuple[str, Any]]:
    barrier = threading.Barrier(len(calls))
    results: list[tuple[str, Any]] = []
    results_lock = threading.Lock()

    def run(call: Callable[[], dict[str, Any]]) -> None:
        try:
            barrier.wait(timeout=5)
            outcome = ("returned", call())
        except ValueError as exc:
            outcome = ("rejected", str(exc))
        except sqlite3.OperationalError as exc:
            outcome = ("operational-error", str(exc))
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - reports worker failures
            outcome = ("error", repr(exc))
        with results_lock:
            results.append(outcome)

    threads = [threading.Thread(target=run, args=(call,), daemon=True) for call in calls]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    finally:
        barrier.abort()
        for event in release_events:
            event.set()
        for thread in threads:
            thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == len(calls)
    return results


class _QuotaReadCoordinationConnection:
    def __init__(
        self,
        inner: sqlite3.Connection,
        admission_progress: threading.Event,
        other_admission_progress: threading.Event,
    ) -> None:
        self._inner = inner
        self._admission_progress = admission_progress
        self._other_admission_progress = other_admission_progress

    def __enter__(self) -> "_QuotaReadCoordinationConnection":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        if str(sql).strip().upper() == "BEGIN IMMEDIATE":
            self._admission_progress.set()
            return self._inner.execute(sql, params)

        result = self._inner.execute(sql, params)
        statement = str(sql)
        if "SELECT COUNT(*) FROM jobs" in statement and "status='queued'" in statement:
            self._admission_progress.set()
            if not self._other_admission_progress.wait(timeout=2):
                raise AssertionError("other admission made no quota or lock progress")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _QuotaReadCoordinationJobManager(JobManager):
    def __init__(
        self,
        db_path,
        *,
        admission_progress: threading.Event,
        other_admission_progress: threading.Event,
    ) -> None:
        super().__init__(db_path)
        self._admission_progress = admission_progress
        self._other_admission_progress = other_admission_progress

    def _connect(self):
        return _QuotaReadCoordinationConnection(
            super()._connect(),
            self._admission_progress,
            self._other_admission_progress,
        )


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


def test_max_queued_quota_allows_sequential_idempotent_replay(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_quota_replay.db"
    ensure_jobs_tables(db_path)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    manager = JobManager(db_path)

    first = manager.create_job(
        domain="quota-replay",
        queue="default",
        job_type="same",
        payload={"attempt": 1},
        owner_user_id="owner-1",
        idempotency_key="replay-key",
        request_id="request-first",
    )
    replay = manager.create_job(
        domain="quota-replay",
        queue="default",
        job_type="same",
        payload={"attempt": 2},
        owner_user_id="owner-1",
        idempotency_key="replay-key",
        request_id="request-replay",
    )

    assert replay["id"] == first["id"]
    assert manager.count_jobs(domain="quota-replay", owner_user_id="owner-1", status="queued") == 1
    with sqlite3.connect(db_path) as conn:
        events = conn.execute(
            "SELECT attrs_json, request_id FROM job_events WHERE job_id = ? AND event_type = 'job.created' ORDER BY id",
            (int(first["id"]),),
        ).fetchall()
    assert [(json.loads(event[0])["idempotent"], event[1]) for event in events] == [
        (False, "request-first"),
        (True, "request-replay"),
    ]


def test_max_queued_quota_allows_concurrent_idempotent_replay(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_quota_replay_race.db"
    ensure_jobs_tables(db_path)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    managers = [JobManager(db_path), JobManager(db_path)]

    def submit(manager: JobManager, request_id: str) -> dict[str, Any]:
        return manager.create_job(
            domain="quota-replay-race",
            queue="default",
            job_type="same",
            payload={"request_id": request_id},
            owner_user_id="owner-1",
            idempotency_key="shared-key",
            request_id=request_id,
        )

    results = _run_concurrent_calls(
        [
            lambda: submit(managers[0], "request-a"),
            lambda: submit(managers[1], "request-b"),
        ]
    )

    assert [outcome for outcome, _ in results] == ["returned", "returned"]
    assert len({int(row["id"]) for _, row in results}) == 1
    assert managers[0].count_jobs(domain="quota-replay-race", owner_user_id="owner-1", status="queued") == 1


def test_max_queued_quota_serializes_concurrent_admission(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_quota_race.db"
    ensure_jobs_tables(db_path)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    first_progress = threading.Event()
    second_progress = threading.Event()
    managers = [
        _QuotaReadCoordinationJobManager(
            db_path,
            admission_progress=first_progress,
            other_admission_progress=second_progress,
        ),
        _QuotaReadCoordinationJobManager(
            db_path,
            admission_progress=second_progress,
            other_admission_progress=first_progress,
        ),
    ]
    results = _run_concurrent_calls(
        [
            lambda: managers[0].create_job(
                domain="quota-race",
                queue="default",
                job_type="race-a",
                payload={},
                owner_user_id="owner-1",
            ),
            lambda: managers[1].create_job(
                domain="quota-race",
                queue="default",
                job_type="race-b",
                payload={},
                owner_user_id="owner-1",
            ),
        ],
        release_events=(first_progress, second_progress),
    )

    assert sorted(outcome for outcome, _ in results) == ["rejected", "returned"]
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

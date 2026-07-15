import threading
from typing import Any

import pytest

psycopg = pytest.importorskip("psycopg")

from tldw_Server_API.app.core.Jobs.manager import JobManager  # noqa: E402, I001


pytestmark = [
    pytest.mark.pg_jobs,
]


def _install_delay_trigger(dsn: str) -> None:
    with psycopg.connect(dsn) as conn:
        conn.execute(
            """
            CREATE FUNCTION jobs_test_delay_insert() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                PERFORM pg_sleep(0.20);
                RETURN NEW;
            END;
            $$
            """
        )
        conn.execute(
            """
            CREATE TRIGGER jobs_test_delay_admission
            BEFORE INSERT ON jobs
            FOR EACH ROW EXECUTE FUNCTION jobs_test_delay_insert()
            """
        )


class _InsertOverlapCursor:
    def __init__(
        self,
        inner: Any,
        entered_insert: threading.Event,
        other_entered_insert: threading.Event,
        overlap_observed: threading.Event,
    ) -> None:
        self._inner = inner
        self._entered_insert = entered_insert
        self._other_entered_insert = other_entered_insert
        self._overlap_observed = overlap_observed

    def __enter__(self) -> "_InsertOverlapCursor":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        if str(sql).lstrip().startswith("INSERT INTO jobs"):
            self._entered_insert.set()
            if self._other_entered_insert.wait(timeout=2):
                self._overlap_observed.set()
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _InsertOverlapJobManager(JobManager):
    def __init__(
        self,
        *,
        db_url: str,
        entered_insert: threading.Event,
        other_entered_insert: threading.Event,
    ) -> None:
        super().__init__(backend="postgres", db_url=db_url)
        self._entered_insert = entered_insert
        self._other_entered_insert = other_entered_insert
        self.overlap_observed = threading.Event()

    def _pg_cursor(self, conn):
        return _InsertOverlapCursor(
            super()._pg_cursor(conn),
            self._entered_insert,
            self._other_entered_insert,
            self.overlap_observed,
        )


def test_max_queued_quota_is_atomic_under_concurrent_admission(monkeypatch, jobs_pg_dsn):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    _install_delay_trigger(jobs_pg_dsn)

    managers = [
        JobManager(backend="postgres", db_url=jobs_pg_dsn),
        JobManager(backend="postgres", db_url=jobs_pg_dsn),
    ]
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


def test_quota_lock_is_scoped_by_owner_within_domain(monkeypatch, jobs_pg_dsn):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    _install_delay_trigger(jobs_pg_dsn)

    first_entered = threading.Event()
    second_entered = threading.Event()
    managers = [
        _InsertOverlapJobManager(
            db_url=jobs_pg_dsn,
            entered_insert=first_entered,
            other_entered_insert=second_entered,
        ),
        _InsertOverlapJobManager(
            db_url=jobs_pg_dsn,
            entered_insert=second_entered,
            other_entered_insert=first_entered,
        ),
    ]
    barrier = threading.Barrier(2)
    results = []
    results_lock = threading.Lock()

    def submit(manager, owner_user_id, job_type):
        try:
            barrier.wait(timeout=5)
            row = manager.create_job(
                domain="quota-scope",
                queue="default",
                job_type=job_type,
                payload={},
                owner_user_id=owner_user_id,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - reports worker failures
            outcome = ("error", repr(exc))
        else:
            outcome = ("created", row)
        with results_lock:
            results.append(outcome)

    threads = [
        threading.Thread(target=submit, args=(managers[0], "owner-1", "scope-a")),
        threading.Thread(target=submit, args=(managers[1], "owner-2", "scope-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(outcome for outcome, _ in results) == ["created", "created"]
    assert all(manager.overlap_observed.is_set() for manager in managers)


def test_pg_max_queued_quota(monkeypatch, jobs_pg_dsn):


    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)

    # Global max queued per user/domain
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")

    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    with pytest.raises(ValueError):
        jm.create_job(domain="chatbooks", queue="default", job_type="t2", payload={}, owner_user_id="1")
    # Different user not blocked
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="2")


def test_pg_submits_per_minute_quota_precedence(monkeypatch, jobs_pg_dsn):


    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)

    # Global limit 1/min; domain+user override to 2/min
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN", "1")
    monkeypatch.setenv("JOBS_QUOTA_SUBMITS_PER_MIN_CHATBOOKS_USER_1", "2")

    jm.create_job(domain="chatbooks", queue="default", job_type="a", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="b", payload={}, owner_user_id="1")
    with pytest.raises(ValueError):
        jm.create_job(domain="chatbooks", queue="default", job_type="c", payload={}, owner_user_id="1")

    # Other domain -> global 1/min applies
    jm.create_job(domain="other", queue="default", job_type="x", payload={}, owner_user_id="1")
    with pytest.raises(ValueError):
        jm.create_job(domain="other", queue="default", job_type="y", payload={}, owner_user_id="1")


def test_pg_max_inflight_quota(monkeypatch, jobs_pg_dsn):


    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)

    monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")

    # Seed two queued for user 1
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w", owner_user_id="1")
    assert acq1 is not None
    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2", owner_user_id="1")
    assert acq2 is None

    # Different user can still acquire
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="2")
    acq_other = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w3", owner_user_id="2")
    assert acq_other is not None


def test_pg_max_inflight_ignores_expired_leases(monkeypatch, jobs_pg_dsn):


    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)

    monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")

    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="1")

    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w", owner_user_id="1")
    assert acq1 is not None

    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "UPDATE jobs SET leased_until = NOW() - interval '10 seconds' WHERE id = %s",
                (int(acq1["id"]),),
            )
        conn.commit()
    finally:
        conn.close()

    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2", owner_user_id="1")
    assert acq2 is not None

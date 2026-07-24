import threading
from collections.abc import Callable, Iterable
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager

psycopg = pytest.importorskip("psycopg")

pytestmark = [
    pytest.mark.pg_jobs,
]


def _run_concurrent_calls(
    calls: list[Callable[[], Any]],
    *,
    release_events: Iterable[threading.Event] = (),
) -> list[tuple[str, Any]]:
    barrier = threading.Barrier(len(calls))
    results: list[tuple[str, Any]] = []
    results_lock = threading.Lock()

    def run(call: Callable[[], Any]) -> None:
        try:
            barrier.wait(timeout=5)
            outcome = ("returned", call())
        except ValueError as exc:
            outcome = ("rejected", str(exc))
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


class _QuotaReadCoordinationCursor:
    def __init__(
        self,
        inner: Any,
        admission_progress: threading.Event,
        other_admission_progress: threading.Event,
    ) -> None:
        self._inner = inner
        self._admission_progress = admission_progress
        self._other_admission_progress = other_admission_progress

    def __enter__(self) -> "_QuotaReadCoordinationCursor":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        if "pg_advisory_xact_lock" in str(sql):
            self._admission_progress.set()
            return self._inner.execute(sql, params)

        result = self._inner.execute(sql, params)
        statement = str(sql)
        if "SELECT COUNT(*) AS c FROM jobs" in statement and "status='queued'" in statement:
            self._admission_progress.set()
            if not self._other_admission_progress.wait(timeout=2):
                raise AssertionError("other admission made no quota or lock progress")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _QuotaReadCoordinationJobManager(JobManager):
    def __init__(
        self,
        *,
        db_url: str,
        admission_progress: threading.Event,
        other_admission_progress: threading.Event,
    ) -> None:
        super().__init__(backend="postgres", db_url=db_url)
        self._admission_progress = admission_progress
        self._other_admission_progress = other_admission_progress

    def _pg_cursor(self, conn):
        return _QuotaReadCoordinationCursor(
            super()._pg_cursor(conn),
            self._admission_progress,
            self._other_admission_progress,
        )


class _AdvisoryLockCoordinationCursor:
    def __init__(
        self,
        inner: Any,
        lock_attempted: threading.Event,
        other_lock_attempted: threading.Event,
    ) -> None:
        self._inner = inner
        self._lock_attempted = lock_attempted
        self._other_lock_attempted = other_lock_attempted

    def __enter__(self) -> "_AdvisoryLockCoordinationCursor":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        if "pg_advisory_xact_lock" not in str(sql):
            return self._inner.execute(sql, params)

        self._lock_attempted.set()
        result = self._inner.execute(sql, params)
        if not self._other_lock_attempted.wait(timeout=5):
            raise AssertionError("other admission did not attempt the quota advisory lock")
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _RepeatableReadJobManager(JobManager):
    def __init__(
        self,
        *,
        db_url: str,
        lock_attempted: threading.Event,
        other_lock_attempted: threading.Event,
    ) -> None:
        super().__init__(backend="postgres", db_url=db_url)
        self._lock_attempted = lock_attempted
        self._other_lock_attempted = other_lock_attempted

    def _connect(self):
        conn = super()._connect()
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        return conn

    def _pg_cursor(self, conn):
        return _AdvisoryLockCoordinationCursor(
            super()._pg_cursor(conn),
            self._lock_attempted,
            self._other_lock_attempted,
        )


class _ReplayPruneCoordinationCursor:
    def __init__(
        self,
        inner: Any,
        replay_detected: threading.Event,
        delete_attempted: threading.Event,
        delete_committed: threading.Event,
    ) -> None:
        self._inner = inner
        self._replay_detected = replay_detected
        self._delete_attempted = delete_attempted
        self._delete_committed = delete_committed
        self._coordinate_replay_fetch = False
        self._probe_locks_row = False

    def __enter__(self) -> "_ReplayPruneCoordinationCursor":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        statement = str(sql)
        self._coordinate_replay_fetch = (
            (
                "SELECT 1 FROM jobs WHERE domain = %s AND queue = %s AND job_type = %s"
                in statement
                or "SELECT * FROM jobs WHERE domain = %s AND queue = %s AND job_type = %s"
                in statement
            )
            and "idempotency_key = %s" in statement
        )
        self._probe_locks_row = "FOR KEY SHARE" in statement
        return self._inner.execute(sql, params)

    def fetchone(self) -> Any:
        row = self._inner.fetchone()
        if self._coordinate_replay_fetch and row is not None:
            self._coordinate_replay_fetch = False
            self._replay_detected.set()
            progress = self._delete_attempted if self._probe_locks_row else self._delete_committed
            if not progress.wait(timeout=5):
                raise AssertionError("prune did not attempt the coordinated destructive row lock")
        return row

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ReplayPruneCoordinationJobManager(JobManager):
    def __init__(
        self,
        *,
        db_url: str,
        replay_detected: threading.Event,
        delete_attempted: threading.Event,
        delete_committed: threading.Event,
    ) -> None:
        super().__init__(backend="postgres", db_url=db_url)
        self._replay_detected = replay_detected
        self._delete_attempted = delete_attempted
        self._delete_committed = delete_committed

    def _pg_cursor(self, conn):
        return _ReplayPruneCoordinationCursor(
            super()._pg_cursor(conn),
            self._replay_detected,
            self._delete_attempted,
            self._delete_committed,
        )


class _PruneDeleteCoordinationCursor:
    def __init__(
        self,
        inner: Any,
        replay_detected: threading.Event,
        delete_attempted: threading.Event,
    ) -> None:
        self._inner = inner
        self._replay_detected = replay_detected
        self._delete_attempted = delete_attempted

    def __enter__(self) -> "_PruneDeleteCoordinationCursor":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        statement = str(sql).lstrip()
        candidate_row_lock = statement.startswith("SELECT id FROM jobs") and "FOR UPDATE" in statement
        if candidate_row_lock or statement.startswith("DELETE FROM jobs"):
            if not self._replay_detected.wait(timeout=5):
                raise AssertionError("replay probe did not detect the existing row")
            self._delete_attempted.set()
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _PruneDeleteCoordinationConnection:
    def __init__(
        self,
        inner: Any,
        replay_detected: threading.Event,
        delete_attempted: threading.Event,
    ) -> None:
        self._inner = inner
        self._replay_detected = replay_detected
        self._delete_attempted = delete_attempted

    def __enter__(self) -> "_PruneDeleteCoordinationConnection":
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._inner.__exit__(exc_type, exc, tb)

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        cursor = self._inner.cursor(*args, **kwargs)
        if kwargs.get("name") == "jobs_prune_candidates":
            return _PruneDeleteCoordinationCursor(
                cursor,
                self._replay_detected,
                self._delete_attempted,
            )
        return cursor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _PruneDeleteCoordinationJobManager(JobManager):
    def __init__(
        self,
        *,
        db_url: str,
        replay_detected: threading.Event,
        delete_attempted: threading.Event,
    ) -> None:
        super().__init__(backend="postgres", db_url=db_url)
        self._replay_detected = replay_detected
        self._delete_attempted = delete_attempted

    def _connect(self):
        return _PruneDeleteCoordinationConnection(
            super()._connect(),
            self._replay_detected,
            self._delete_attempted,
        )

    def _pg_cursor(self, conn):
        return _PruneDeleteCoordinationCursor(
            super()._pg_cursor(conn),
            self._replay_detected,
            self._delete_attempted,
        )


def test_max_queued_quota_allows_sequential_idempotent_replay(monkeypatch, jobs_pg_dsn):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    manager = JobManager(backend="postgres", db_url=jobs_pg_dsn)

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
    with psycopg.connect(jobs_pg_dsn) as conn:
        events = conn.execute(
            "SELECT attrs_json, request_id FROM job_events WHERE job_id = %s AND event_type = 'job.created' ORDER BY id",
            (int(first["id"]),),
        ).fetchall()
    assert [(event[0]["idempotent"], event[1]) for event in events] == [
        (False, "request-first"),
        (True, "request-replay"),
    ]


def test_max_queued_quota_allows_concurrent_idempotent_replay(monkeypatch, jobs_pg_dsn):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    managers = [
        JobManager(backend="postgres", db_url=jobs_pg_dsn),
        JobManager(backend="postgres", db_url=jobs_pg_dsn),
    ]

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


def test_max_queued_replay_holds_row_against_concurrent_prune(monkeypatch, jobs_pg_dsn):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    seed_manager = JobManager(backend="postgres", db_url=jobs_pg_dsn)
    original = seed_manager.create_job(
        domain="quota-prune-replay",
        queue="default",
        job_type="replay-target",
        payload={"attempt": "original"},
        owner_user_id="owner-1",
        idempotency_key="replay-key",
    )
    seed_manager.create_job(
        domain="quota-prune-replay",
        queue="default",
        job_type="quota-blocker",
        payload={},
        owner_user_id="owner-1",
    )
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")

    replay_detected = threading.Event()
    delete_attempted = threading.Event()
    delete_committed = threading.Event()
    replay_manager = _ReplayPruneCoordinationJobManager(
        db_url=jobs_pg_dsn,
        replay_detected=replay_detected,
        delete_attempted=delete_attempted,
        delete_committed=delete_committed,
    )
    prune_manager = _PruneDeleteCoordinationJobManager(
        db_url=jobs_pg_dsn,
        replay_detected=replay_detected,
        delete_attempted=delete_attempted,
    )

    def replay() -> dict[str, Any]:
        return replay_manager.create_job(
            domain="quota-prune-replay",
            queue="default",
            job_type="replay-target",
            payload={"attempt": "replay"},
            owner_user_id="owner-1",
            idempotency_key="replay-key",
            request_id="request-replay",
        )

    def prune() -> int:
        try:
            return prune_manager.prune_jobs(
                statuses=["queued"],
                older_than_days=-1,
                domain="quota-prune-replay",
                queue="default",
                job_type="replay-target",
            )
        finally:
            delete_committed.set()

    results = _run_concurrent_calls(
        [replay, prune],
        release_events=(replay_detected, delete_attempted, delete_committed),
    )

    replay_rows = [value for outcome, value in results if outcome == "returned" and isinstance(value, dict)]
    prune_counts = [value for outcome, value in results if outcome == "returned" and isinstance(value, int)]
    assert [int(row["id"]) for row in replay_rows] == [int(original["id"])], results
    assert prune_counts == [1], results
    assert seed_manager.count_jobs(
        domain="quota-prune-replay",
        owner_user_id="owner-1",
        status="queued",
    ) == 1


def test_idempotent_replay_without_quota_commits_before_concurrent_prune(
    monkeypatch,
    jobs_pg_dsn,
):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.delenv("JOBS_QUOTA_MAX_QUEUED", raising=False)
    monkeypatch.delenv("JOBS_QUOTA_SUBMITS_PER_MINUTE", raising=False)
    seed_manager = JobManager(backend="postgres", db_url=jobs_pg_dsn)
    original = seed_manager.create_job(
        domain="no-quota-prune-replay",
        queue="default",
        job_type="replay-target",
        payload={"attempt": "original"},
        owner_user_id="owner-1",
        idempotency_key="replay-key",
    )

    replay_detected = threading.Event()
    delete_attempted = threading.Event()
    delete_committed = threading.Event()
    replay_manager = _ReplayPruneCoordinationJobManager(
        db_url=jobs_pg_dsn,
        replay_detected=replay_detected,
        delete_attempted=delete_attempted,
        delete_committed=delete_committed,
    )
    prune_manager = _PruneDeleteCoordinationJobManager(
        db_url=jobs_pg_dsn,
        replay_detected=replay_detected,
        delete_attempted=delete_attempted,
    )

    def replay() -> dict[str, Any]:
        return replay_manager.create_job(
            domain="no-quota-prune-replay",
            queue="default",
            job_type="replay-target",
            payload={"attempt": "replay"},
            owner_user_id="owner-1",
            idempotency_key="replay-key",
            request_id="request-replay",
        )

    def prune() -> int:
        try:
            deleted = prune_manager.prune_jobs(
                statuses=["queued"],
                older_than_days=-1,
                domain="no-quota-prune-replay",
                queue="default",
                job_type="replay-target",
            )
            with psycopg.connect(jobs_pg_dsn) as conn:
                replay_event = conn.execute(
                    "SELECT 1 FROM job_events WHERE job_id = %s AND request_id = %s",
                    (int(original["id"]), "request-replay"),
                ).fetchone()
            if replay_event is None:
                raise AssertionError("prune committed before the idempotent replay event")
            return deleted
        finally:
            delete_committed.set()

    results = _run_concurrent_calls(
        [replay, prune],
        release_events=(replay_detected, delete_attempted, delete_committed),
    )

    replay_rows = [value for outcome, value in results if outcome == "returned" and isinstance(value, dict)]
    prune_counts = [value for outcome, value in results if outcome == "returned" and isinstance(value, int)]
    assert [int(row["id"]) for row in replay_rows] == [int(original["id"])], results
    assert prune_counts == [1], results


def test_repeatable_read_max_queued_quota_is_atomic_under_concurrent_admission(monkeypatch, jobs_pg_dsn):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    monkeypatch.setenv("JOBS_PG_RLS_DEBUG", "true")
    first_attempted = threading.Event()
    second_attempted = threading.Event()
    managers = [
        _RepeatableReadJobManager(
            db_url=jobs_pg_dsn,
            lock_attempted=first_attempted,
            other_lock_attempted=second_attempted,
        ),
        _RepeatableReadJobManager(
            db_url=jobs_pg_dsn,
            lock_attempted=second_attempted,
            other_lock_attempted=first_attempted,
        ),
    ]

    def submit(manager: JobManager, job_type: str) -> dict[str, Any]:
        return manager.create_job(
            domain="quota-repeatable-read",
            queue="default",
            job_type=job_type,
            payload={},
            owner_user_id="owner-1",
        )

    results = _run_concurrent_calls(
        [
            lambda: submit(managers[0], "race-a"),
            lambda: submit(managers[1], "race-b"),
        ],
        release_events=(first_attempted, second_attempted),
    )

    assert sorted(outcome for outcome, _ in results) == ["rejected", "returned"], results
    assert [detail for outcome, detail in results if outcome == "rejected"] == [
        "Quota exceeded: max queued per user/domain"
    ]
    assert managers[0].count_jobs(
        domain="quota-repeatable-read",
        owner_user_id="owner-1",
        status="queued",
    ) == 1


def test_max_queued_quota_is_atomic_under_concurrent_admission(monkeypatch, jobs_pg_dsn):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_QUOTA_MAX_QUEUED", "1")
    first_progress = threading.Event()
    second_progress = threading.Event()
    managers = [
        _QuotaReadCoordinationJobManager(
            db_url=jobs_pg_dsn,
            admission_progress=first_progress,
            other_admission_progress=second_progress,
        ),
        _QuotaReadCoordinationJobManager(
            db_url=jobs_pg_dsn,
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
    results = _run_concurrent_calls(
        [
            lambda: managers[0].create_job(
                domain="quota-scope",
                queue="default",
                job_type="scope-a",
                payload={},
                owner_user_id="owner-1",
            ),
            lambda: managers[1].create_job(
                domain="quota-scope",
                queue="default",
                job_type="scope-b",
                payload={},
                owner_user_id="owner-2",
            ),
        ],
        release_events=(first_entered, second_entered),
    )

    assert [outcome for outcome, _ in results] == ["returned", "returned"]
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

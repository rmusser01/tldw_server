from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager

_DOMAIN = "finalize-cancelled-boundary"
_QUEUE = "default"
_JOB_TYPE = "worker-cancel"
_OWNER = "owner-42"
_REQUEST_ID = "request-42"
_TRACE_ID = "trace-42"
_REASON = "cancel requested during processing"


class _RollbackInsteadOfCommit:
    """Verify transactional writes, then replace the commit with a rollback."""

    def __init__(self, inner: Any, verify_uncommitted: Callable[[Any], None]) -> None:
        self._inner = inner
        self._verify_uncommitted = verify_uncommitted

    def __enter__(self) -> _RollbackInsteadOfCommit:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if exc_type is not None:
            return self._inner.__exit__(exc_type, exc, tb)
        try:
            self._verify_uncommitted(self._inner)
        finally:
            self._inner.rollback()
        raise RuntimeError("forced commit failure")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CloseThenRaise:
    """Delegate transaction handling, then fail only after closing the connection."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _CloseThenRaise:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def close(self) -> None:
        self._inner.close()
        raise RuntimeError("forced close failure after commit")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FetchOneHookCursor:
    """Return one row that was captured before a concurrency callback ran."""

    def __init__(self, prefetched: Any) -> None:
        self._prefetched = prefetched

    def fetchone(self) -> Any:
        row = self._prefetched
        self._prefetched = None
        return row


class _SQLiteQueuedSelectHookConnection:
    """Hook only finalize_cancelled's initial state read."""

    def __init__(self, inner: Any, callback: Callable[[bool], None]) -> None:
        self._inner = inner
        self._callback = callback
        self._armed = True
        self.fired = False
        self.write_transaction_started = False

    def __enter__(self) -> _SQLiteQueuedSelectHookConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, parameters: Any = ()) -> Any:
        cursor = self._inner.execute(sql, parameters)
        if self._armed and sql.lstrip().startswith(
            "SELECT status, domain, queue, job_type, available_at, uuid"
        ):
            self._armed = False
            self.fired = True
            prefetched = cursor.fetchone()
            cursor.close()
            self.write_transaction_started = bool(self._inner.in_transaction)
            self._callback(self.write_transaction_started)
            return _FetchOneHookCursor(prefetched)
        return cursor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _PostgresQueuedSelectHookCursor:
    """Run a callback after Postgres returns finalize_cancelled's queued snapshot."""

    def __init__(self, inner: Any, callback: Callable[[bool], None]) -> None:
        self._inner = inner
        self._callback = callback
        self._armed = False
        self.fired = False
        self.row_locked = False

    def execute(self, sql: Any, parameters: Any = None) -> Any:
        result = self._inner.execute(sql, parameters)
        normalized = " ".join(str(sql).split())
        if normalized.startswith(
            "SELECT status, domain, queue, job_type, available_at, uuid"
        ):
            self._armed = True
            self.fired = True
            self.row_locked = "FOR UPDATE" in normalized.upper()
        return result

    def fetchone(self) -> Any:
        row = self._inner.fetchone()
        if self._armed:
            self._armed = False
            self._callback(self.row_locked)
        return row

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _create_queued_job(jm: JobManager) -> dict[str, Any]:
    return jm.create_job(
        domain=_DOMAIN,
        queue=_QUEUE,
        job_type=_JOB_TYPE,
        payload={},
        owner_user_id=_OWNER,
        request_id=_REQUEST_ID,
        trace_id=_TRACE_ID,
    )


def _create_processing_job(jm: JobManager) -> dict[str, Any]:
    created = _create_queued_job(jm)
    acquired = jm.acquire_next_job(
        domain=_DOMAIN,
        queue=_QUEUE,
        lease_seconds=30,
        worker_id="worker-42",
    )
    assert acquired is not None
    assert int(acquired["id"]) == int(created["id"])
    assert acquired["status"] == "processing"
    assert acquired["leased_until"] is not None
    assert acquired["worker_id"] == "worker-42"
    assert acquired["lease_id"] is not None
    return acquired


def _finalize_owned(jm: JobManager, job: dict[str, Any], *, reason: str = _REASON) -> bool:
    return jm.finalize_cancelled(
        int(job["id"]),
        reason=reason,
        expected_uuid=str(job["uuid"]),
        worker_id=str(job["worker_id"]),
        lease_id=str(job["lease_id"]),
    )


def _job_snapshot(jm: JobManager, job_id: int) -> tuple[Any, ...]:
    job = jm.get_job(job_id)
    assert job is not None
    return (
        job["status"],
        job["leased_until"],
        job["worker_id"],
        job["lease_id"],
        job["cancellation_reason"],
    )


def _counter_snapshot(jm: JobManager) -> tuple[int, int, int]:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT ready_count, scheduled_count, processing_count "
                    "FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
                    (_DOMAIN, _QUEUE, _JOB_TYPE),
                )
                row = cur.fetchone()
            assert row is not None
            return (
                int(row["ready_count"]),
                int(row["scheduled_count"]),
                int(row["processing_count"]),
            )
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count "
            "FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
            (_DOMAIN, _QUEUE, _JOB_TYPE),
        ).fetchone()
        assert row is not None
        return int(row[0]), int(row[1]), int(row[2])
    finally:
        conn.close()


def _cancelled_events(jm: JobManager, job_id: int) -> list[dict[str, Any]]:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT event_type, attrs_json, owner_user_id, request_id, trace_id "
                    "FROM job_events WHERE job_id=%s AND event_type='job.cancelled' ORDER BY id",
                    (job_id,),
                )
                rows = [dict(row) for row in (cur.fetchall() or [])]
        else:
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT event_type, attrs_json, owner_user_id, request_id, trace_id "
                    "FROM job_events WHERE job_id=? AND event_type='job.cancelled' ORDER BY id",
                    (job_id,),
                ).fetchall()
            ]
    finally:
        conn.close()
    for row in rows:
        if isinstance(row["attrs_json"], str):
            row["attrs_json"] = json.loads(row["attrs_json"])
    return rows


def _uncommitted_terminal_state(conn: Any, backend: str, job_id: int) -> tuple[Any, ...]:
    if backend == "postgres":
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT status, leased_until, worker_id, lease_id FROM jobs WHERE id=%s",
                (job_id,),
            )
            job = cur.fetchone()
            cur.execute(
                "SELECT processing_count FROM job_counters "
                "WHERE domain=%s AND queue=%s AND job_type=%s",
                (_DOMAIN, _QUEUE, _JOB_TYPE),
            )
            counter = cur.fetchone()
            cur.execute(
                "SELECT COUNT(*) AS count FROM job_events "
                "WHERE job_id=%s AND event_type='job.cancelled'",
                (job_id,),
            )
            event_count = cur.fetchone()
        assert job is not None and counter is not None and event_count is not None
        return (
            job["status"],
            job["leased_until"],
            job["worker_id"],
            job["lease_id"],
            int(counter["processing_count"]),
            int(event_count["count"]),
        )
    job = conn.execute(
        "SELECT status, leased_until, worker_id, lease_id FROM jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    counter = conn.execute(
        "SELECT processing_count FROM job_counters "
        "WHERE domain=? AND queue=? AND job_type=?",
        (_DOMAIN, _QUEUE, _JOB_TYPE),
    ).fetchone()
    event_count = conn.execute(
        "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type='job.cancelled'",
        (job_id,),
    ).fetchone()
    assert job is not None and counter is not None and event_count is not None
    return job[0], job[1], job[2], job[3], int(counter[0]), int(event_count[0])


def _assert_commit_failure_is_observer_free(jm: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    import tldw_Server_API.app.core.Jobs.manager as jobs_manager_module

    acquired = _create_processing_job(jm)
    job_id = int(acquired["id"])
    before_job = _job_snapshot(jm, job_id)
    before_counter = _counter_snapshot(jm)
    assert before_counter == (0, 0, 1)

    calls = {"gauge": [], "metric": [], "event": [], "audit": []}
    monkeypatch.setattr(jm, "_update_gauges", lambda **kwargs: calls["gauge"].append(kwargs))
    monkeypatch.setattr(
        jobs_manager_module,
        "increment_cancelled",
        lambda labels: calls["metric"].append(dict(labels)),
    )
    monkeypatch.setattr(
        jobs_manager_module,
        "emit_job_event",
        lambda event_type, **kwargs: calls["event"].append((event_type, kwargs)),
    )
    monkeypatch.setattr(
        jobs_manager_module,
        "submit_job_audit_event",
        lambda event_type, **kwargs: calls["audit"].append((event_type, kwargs)),
    )

    original_connect = jm._connect
    connect_count = 0

    def fail_finalize_commit() -> Any:
        nonlocal connect_count
        conn = original_connect()
        connect_count += 1
        if connect_count != 1:
            return conn

        def verify_uncommitted(inner: Any) -> None:
            assert _uncommitted_terminal_state(inner, jm.backend, job_id) == (
                "cancelled",
                None,
                None,
                None,
                0,
                1,
            )

        return _RollbackInsteadOfCommit(conn, verify_uncommitted)

    monkeypatch.setattr(jm, "_connect", fail_finalize_commit)

    with pytest.raises(RuntimeError, match="forced commit failure"):
        _finalize_owned(jm, acquired)

    assert _job_snapshot(jm, job_id) == before_job
    assert _counter_snapshot(jm) == before_counter
    assert _cancelled_events(jm, job_id) == []
    assert calls == {"gauge": [], "metric": [], "event": [], "audit": []}


def _assert_success_observers_see_commit(
    jm: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    outbox_enabled: bool,
    close_raises: bool = False,
) -> None:
    import tldw_Server_API.app.core.Jobs.manager as jobs_manager_module

    acquired = _create_processing_job(jm)
    job_id = int(acquired["id"])
    expected_snapshot = ("cancelled", None, None, None, _REASON)
    observed: dict[str, list[Any]] = {"gauge": [], "metric": [], "event": [], "audit": []}

    def record_gauge(**kwargs: Any) -> None:
        assert kwargs == {"domain": _DOMAIN, "queue": _QUEUE, "job_type": _JOB_TYPE}
        observed["gauge"].append(_job_snapshot(jm, job_id))

    def record_metric(labels: dict[str, Any]) -> None:
        assert {key: labels[key] for key in ("domain", "queue", "job_type")} == {
            "domain": _DOMAIN,
            "queue": _QUEUE,
            "job_type": _JOB_TYPE,
        }
        observed["metric"].append(_job_snapshot(jm, job_id))

    def record_route(route: str, event_type: str, *, job: dict[str, Any], attrs: dict[str, Any]) -> None:
        assert event_type == "job.cancelled"
        assert {key: job[key] for key in ("id", "domain", "queue", "job_type")} == {
            "id": job_id,
            "domain": _DOMAIN,
            "queue": _QUEUE,
            "job_type": _JOB_TYPE,
        }
        assert job["owner_user_id"] == _OWNER
        assert job["request_id"] == _REQUEST_ID
        assert job["trace_id"] == _TRACE_ID
        assert attrs == {"reason": _REASON, "terminal": True}
        observed[route].append(_job_snapshot(jm, job_id))

    monkeypatch.setattr(jm, "_update_gauges", record_gauge)
    monkeypatch.setattr(jobs_manager_module, "increment_cancelled", record_metric)
    monkeypatch.setattr(
        jobs_manager_module,
        "emit_job_event",
        lambda event_type, **kwargs: record_route("event", event_type, **kwargs),
    )
    monkeypatch.setattr(
        jobs_manager_module,
        "submit_job_audit_event",
        lambda event_type, **kwargs: record_route("audit", event_type, **kwargs),
    )
    if close_raises:
        original_connect = jm._connect
        connect_count = 0

        def fail_first_close() -> Any:
            nonlocal connect_count
            conn = original_connect()
            connect_count += 1
            if connect_count == 1:
                return _CloseThenRaise(conn)
            return conn

        monkeypatch.setattr(jm, "_connect", fail_first_close)

    assert _finalize_owned(jm, acquired) is True

    assert _job_snapshot(jm, job_id) == expected_snapshot
    assert _counter_snapshot(jm) == (0, 0, 0)
    assert observed["gauge"] == [expected_snapshot]
    assert observed["metric"] == [expected_snapshot]
    if outbox_enabled:
        assert observed["event"] == []
        assert observed["audit"] == [expected_snapshot]
        assert _cancelled_events(jm, job_id) == [
            {
                "event_type": "job.cancelled",
                "attrs_json": {"reason": _REASON, "terminal": True},
                "owner_user_id": _OWNER,
                "request_id": _REQUEST_ID,
                "trace_id": _TRACE_ID,
            }
        ]
    else:
        assert observed["event"] == [expected_snapshot]
        assert observed["audit"] == []
        assert _cancelled_events(jm, job_id) == []


@pytest.mark.unit
def test_finalize_cancelled_sqlite_commit_failure_rolls_back_without_observers(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    _assert_commit_failure_is_observer_free(JobManager(tmp_path / "finalize-cancelled.db"), monkeypatch)


@pytest.mark.unit
@pytest.mark.parametrize("outbox_enabled", [False, True], ids=["direct-event", "durable-outbox"])
def test_finalize_cancelled_sqlite_observers_run_once_after_commit(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    outbox_enabled: bool,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true" if outbox_enabled else "false")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    _assert_success_observers_see_commit(
        JobManager(tmp_path / "finalize-cancelled.db"),
        monkeypatch,
        outbox_enabled=outbox_enabled,
    )


@pytest.mark.pg_jobs
def test_finalize_cancelled_postgres_commit_failure_rolls_back_without_observers(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("psycopg")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    _assert_commit_failure_is_observer_free(
        JobManager(None, backend="postgres", db_url=jobs_pg_dsn),
        monkeypatch,
    )


@pytest.mark.pg_jobs
@pytest.mark.parametrize("outbox_enabled", [False, True], ids=["direct-event", "durable-outbox"])
def test_finalize_cancelled_postgres_observers_run_once_after_commit(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    outbox_enabled: bool,
) -> None:
    pytest.importorskip("psycopg")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true" if outbox_enabled else "false")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    _assert_success_observers_see_commit(
        JobManager(None, backend="postgres", db_url=jobs_pg_dsn),
        monkeypatch,
        outbox_enabled=outbox_enabled,
    )


@pytest.mark.unit
def test_finalize_cancelled_sqlite_uses_acquired_state_for_counter_transition(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    db_path = tmp_path / "finalize-cancelled-race.db"
    jm = JobManager(db_path)
    acquirer = JobManager(db_path)
    monkeypatch.setattr(acquirer, "_reconcile_terminal_dependents", lambda **_scope: 0)
    queued = _create_queued_job(jm)
    job_id = int(queued["id"])
    assert _counter_snapshot(jm) == (1, 0, 0)
    original_connect = jm._connect
    connect_count = 0
    hooked_connection: _SQLiteQueuedSelectHookConnection | None = None
    competing_acquire = None

    with ThreadPoolExecutor(max_workers=1) as pool:

        def acquire_after_queued_read(write_transaction_started: bool) -> None:
            nonlocal competing_acquire
            competing_acquire = pool.submit(
                acquirer.acquire_next_job,
                domain=_DOMAIN,
                queue=_QUEUE,
                lease_seconds=30,
                worker_id="racing-worker",
            )
            if not write_transaction_started:
                competing_acquire.result(timeout=10)

        def hook_finalize_connection() -> Any:
            nonlocal connect_count, hooked_connection
            conn = original_connect()
            connect_count += 1
            if connect_count == 1:
                hooked_connection = _SQLiteQueuedSelectHookConnection(
                    conn,
                    acquire_after_queued_read,
                )
                return hooked_connection
            return conn

        monkeypatch.setattr(jm, "_connect", hook_finalize_connection)

        assert jm.finalize_cancelled(
            job_id,
            reason=_REASON,
            expected_uuid=str(queued["uuid"]),
            allow_queued=True,
        ) is True
        assert competing_acquire is not None
        raced = competing_acquire.result(timeout=10)

    assert hooked_connection is not None
    assert hooked_connection.fired is True
    if hooked_connection.write_transaction_started:
        assert raced is None
    else:
        assert raced is not None
        assert int(raced["id"]) == job_id
        assert raced["status"] == "processing"
        assert raced["lease_id"] is not None
    assert _job_snapshot(jm, job_id) == ("cancelled", None, None, None, _REASON)
    assert _counter_snapshot(jm) == (0, 0, 0)


@pytest.mark.pg_jobs
def test_finalize_cancelled_postgres_uses_acquired_state_for_counter_transition(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("psycopg")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    acquirer = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    queued = _create_queued_job(jm)
    job_id = int(queued["id"])
    assert _counter_snapshot(jm) == (1, 0, 0)
    raced_acquisitions: list[dict[str, Any] | None] = []

    def acquire_after_queued_read(_row_locked: bool) -> None:
        acquired = acquirer.acquire_next_job(
            domain=_DOMAIN,
            queue=_QUEUE,
            lease_seconds=30,
            worker_id="racing-worker",
        )
        raced_acquisitions.append(acquired)

    original_pg_cursor = jm._pg_cursor
    target_hook: _PostgresQueuedSelectHookCursor | None = None

    @contextmanager
    def hook_finalize_cursor(conn: Any) -> Any:
        nonlocal target_hook
        with original_pg_cursor(conn) as cursor:
            hook = _PostgresQueuedSelectHookCursor(cursor, acquire_after_queued_read)
            yield hook
            if hook.fired:
                target_hook = hook

    monkeypatch.setattr(jm, "_pg_cursor", hook_finalize_cursor)

    assert jm.finalize_cancelled(
        job_id,
        reason=_REASON,
        expected_uuid=str(queued["uuid"]),
        allow_queued=True,
    ) is True
    assert target_hook is not None
    assert len(raced_acquisitions) == 1
    raced = raced_acquisitions[0]
    if target_hook.row_locked:
        assert raced is None
    else:
        assert raced is not None
        assert int(raced["id"]) == job_id
        assert raced["status"] == "processing"
        assert raced["lease_id"] is not None
    assert _job_snapshot(jm, job_id) == ("cancelled", None, None, None, _REASON)
    assert _counter_snapshot(jm) == (0, 0, 0)


@pytest.mark.unit
@pytest.mark.parametrize("outbox_enabled", [False, True], ids=["direct-event", "durable-outbox"])
def test_finalize_cancelled_sqlite_close_error_after_commit_is_nonfatal(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    outbox_enabled: bool,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true" if outbox_enabled else "false")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    _assert_success_observers_see_commit(
        JobManager(tmp_path / "finalize-cancelled-close.db"),
        monkeypatch,
        outbox_enabled=outbox_enabled,
        close_raises=True,
    )


@pytest.mark.pg_jobs
@pytest.mark.parametrize("outbox_enabled", [False, True], ids=["direct-event", "durable-outbox"])
def test_finalize_cancelled_postgres_close_error_after_commit_is_nonfatal(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    outbox_enabled: bool,
) -> None:
    pytest.importorskip("psycopg")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true" if outbox_enabled else "false")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    _assert_success_observers_see_commit(
        JobManager(None, backend="postgres", db_url=jobs_pg_dsn),
        monkeypatch,
        outbox_enabled=outbox_enabled,
        close_raises=True,
    )


def _assert_stale_lease_cannot_finalize_reassigned_job(jm: JobManager) -> None:
    stale = _create_processing_job(jm)
    assert jm.release_job(
        int(stale["id"]),
        worker_id=str(stale["worker_id"]),
        lease_id=str(stale["lease_id"]),
        enforce=True,
    ) is True
    replacement_lease = jm.acquire_next_job(
        domain=_DOMAIN,
        queue=_QUEUE,
        lease_seconds=30,
        worker_id="replacement-worker",
    )
    assert replacement_lease is not None

    assert _finalize_owned(jm, stale) is False
    current = jm.get_job(int(stale["id"]))
    assert current is not None
    assert current["status"] == "processing"
    assert current["worker_id"] == "replacement-worker"
    assert current["lease_id"] == replacement_lease["lease_id"]


@pytest.mark.unit
@pytest.mark.concurrent
def test_finalize_cancelled_sqlite_rejects_stale_lease(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _assert_stale_lease_cannot_finalize_reassigned_job(
        JobManager(tmp_path / "finalize-cancelled-stale-lease.db")
    )


@pytest.mark.pg_jobs
@pytest.mark.concurrent
def test_finalize_cancelled_postgres_rejects_stale_lease(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _assert_stale_lease_cannot_finalize_reassigned_job(
        JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    )


@pytest.mark.unit
@pytest.mark.concurrent
def test_finalize_cancelled_sqlite_rejects_reused_numeric_id(
    tmp_path: Any,
) -> None:
    jm = JobManager(tmp_path / "finalize-cancelled-reused-id.db")
    stale = _create_processing_job(jm)
    assert jm.complete_job(
        int(stale["id"]),
        result={"ok": True},
        worker_id=str(stale["worker_id"]),
        lease_id=str(stale["lease_id"]),
    ) is True
    conn = jm._connect()
    try:
        with conn:
            conn.execute("DELETE FROM job_events WHERE job_id = ?", (int(stale["id"]),))
            conn.execute("DELETE FROM jobs WHERE id = ?", (int(stale["id"]),))
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'jobs'")
    finally:
        conn.close()

    replacement = _create_processing_job(jm)
    assert int(replacement["id"]) == int(stale["id"])
    assert replacement["uuid"] != stale["uuid"]

    assert _finalize_owned(jm, stale) is False
    current = jm.get_job(int(replacement["id"]))
    assert current is not None
    assert current["uuid"] == replacement["uuid"]
    assert current["status"] == "processing"


@pytest.mark.unit
def test_finalize_cancelled_sqlite_processing_requires_complete_ownership(
    tmp_path: Any,
) -> None:
    jm = JobManager(tmp_path / "finalize-cancelled-ownership.db")
    acquired = _create_processing_job(jm)
    job_id = int(acquired["id"])
    expected_uuid = str(acquired["uuid"])

    assert jm.finalize_cancelled(job_id, expected_uuid=expected_uuid) is False
    assert (
        jm.finalize_cancelled(
            job_id,
            expected_uuid=expected_uuid,
            worker_id=str(acquired["worker_id"]),
        )
        is False
    )
    assert (
        jm.finalize_cancelled(
            job_id,
            expected_uuid=expected_uuid,
            worker_id="wrong-worker",
            lease_id=str(acquired["lease_id"]),
        )
        is False
    )
    assert jm.get_job(job_id)["status"] == "processing"


@pytest.mark.unit
def test_finalize_cancelled_sqlite_queued_requires_explicit_permission(
    tmp_path: Any,
) -> None:
    jm = JobManager(tmp_path / "finalize-cancelled-queued-ownership.db")
    queued = _create_queued_job(jm)
    job_id = int(queued["id"])

    assert (
        jm.finalize_cancelled(
            job_id,
            expected_uuid=str(queued["uuid"]),
        )
        is False
    )
    assert (
        jm.finalize_cancelled(
            job_id,
            expected_uuid="different-incarnation",
            allow_queued=True,
        )
        is False
    )
    assert jm.get_job(job_id)["status"] == "queued"

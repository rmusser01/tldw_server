from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager

_OPERATIONS = ("complete", "terminal_fail", "retry", "quarantine")
_EXPECTED_EVENT = {
    "complete": "job.completed",
    "terminal_fail": "job.failed",
    "retry": "job.retry_scheduled",
    "quarantine": "job.quarantined",
}


class _FailOutboxInsertSQLite:
    """Connection adapter that fails only durable lifecycle event inserts."""

    def __init__(self, inner: sqlite3.Connection) -> None:
        self._inner = inner

    def __enter__(self) -> _FailOutboxInsertSQLite:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, params: Any = ()) -> Any:
        if "INSERT INTO job_events" in str(sql):
            raise sqlite3.OperationalError("forced lifecycle outbox insert failure")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailCommitSQLite:
    """Connection adapter that rolls back at either SQLite commit boundary."""

    def __init__(self, inner: sqlite3.Connection) -> None:
        self._inner = inner

    def __enter__(self) -> _FailCommitSQLite:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is not None:
            return self._inner.__exit__(exc_type, exc, tb)
        self._inner.rollback()
        raise RuntimeError("forced lifecycle commit failure")

    def commit(self) -> None:
        self._inner.rollback()
        raise RuntimeError("forced lifecycle commit failure")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailOutboxCursorPostgres:
    """PostgreSQL cursor adapter that fails durable lifecycle event inserts."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _FailOutboxCursorPostgres:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        if "INSERT INTO job_events" in str(sql):
            import psycopg

            raise psycopg.OperationalError("forced lifecycle outbox insert failure")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailOutboxInsertPostgres:
    """PostgreSQL connection adapter for lifecycle outbox fault injection."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _FailOutboxInsertPostgres:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def cursor(self, *args: Any, **kwargs: Any) -> _FailOutboxCursorPostgres:
        return _FailOutboxCursorPostgres(self._inner.cursor(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailCommitPostgres:
    """PostgreSQL connection adapter that rolls back explicit commits."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _FailCommitPostgres:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is not None:
            return self._inner.__exit__(exc_type, exc, tb)
        self._inner.rollback()
        raise RuntimeError("forced lifecycle commit failure")

    def commit(self) -> None:
        self._inner.rollback()
        raise RuntimeError("forced lifecycle commit failure")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _manager(tmp_path: Any, jobs_pg_dsn: str | None) -> JobManager:
    if jobs_pg_dsn is not None:
        return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    return JobManager(tmp_path / "lifecycle-event-atomicity.db")


def _processing_job(jm: JobManager, operation: str) -> dict[str, Any]:
    created = jm.create_job(
        domain=f"lifecycle-event-{operation}",
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner-1",
        request_id="request-1",
        trace_id="trace-1",
        max_retries=3,
    )
    acquired = jm.acquire_next_job(
        domain=str(created["domain"]),
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert acquired is not None
    assert int(acquired["id"]) == int(created["id"])
    return acquired


def _invoke_lifecycle(jm: JobManager, acquired: dict[str, Any], operation: str) -> bool:
    common = {
        "worker_id": str(acquired["worker_id"]),
        "lease_id": str(acquired["lease_id"]),
    }
    if operation == "complete":
        return jm.complete_job(int(acquired["id"]), result={"ok": True}, **common)
    return jm.fail_job(
        int(acquired["id"]),
        error="boom",
        error_code="E_LIFECYCLE",
        retryable=operation != "terminal_fail",
        backoff_seconds=1,
        **common,
    )


def _event_count(jm: JobManager, job_id: int, event_type: str) -> int:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS count FROM job_events WHERE job_id=%s AND event_type=%s",
                    (job_id, event_type),
                )
                row = cur.fetchone()
                return int((row or {}).get("count") or 0)
        row = conn.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type=?",
            (job_id, event_type),
        ).fetchone()
        return int(row[0] if row else 0)
    finally:
        conn.close()


def _patch_lifecycle_observers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    import tldw_Server_API.app.core.Jobs.event_stream as event_stream_module
    import tldw_Server_API.app.core.Jobs.manager as manager_module

    observed: list[str] = []
    audited: list[str] = []
    retries: list[dict[str, Any]] = []
    logged: list[str] = []
    real_observe = manager_module.observe_job_event

    class _RecordingLogger:
        def bind(self, **_kwargs: Any) -> _RecordingLogger:
            return self

        def info(self, message: str) -> None:
            logged.append(message)

    def record_observe(event_type: str, **kwargs: Any) -> None:
        observed.append(event_type)
        real_observe(event_type, **kwargs)

    monkeypatch.setattr(
        event_stream_module,
        "submit_job_audit_event",
        lambda event_type, **_kwargs: audited.append(event_type),
    )
    monkeypatch.setattr(event_stream_module, "logger", _RecordingLogger())
    monkeypatch.setattr(manager_module, "observe_job_event", record_observe)
    monkeypatch.setattr(manager_module, "increment_retries", lambda job: retries.append(job))
    return observed, audited, retries, logged


def _counter_snapshot(jm: JobManager, job: dict[str, Any]) -> tuple[int, int, int, int]:
    conn = jm._connect()
    try:
        params = (job["domain"], job["queue"], job["job_type"])
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT ready_count, scheduled_count, processing_count, quarantined_count "
                    "FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
                    params,
                )
                row = cur.fetchone()
                assert row is not None
                return tuple(int(row[key]) for key in (
                    "ready_count",
                    "scheduled_count",
                    "processing_count",
                    "quarantined_count",
                ))
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count, quarantined_count "
            "FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
            params,
        ).fetchone()
        assert row is not None
        return tuple(int(value) for value in row)
    finally:
        conn.close()


def _assert_flag_contract(
    jm: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    operation: str,
    events_enabled: bool,
    outbox_enabled: bool,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "true" if events_enabled else "false")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true" if outbox_enabled else "false")
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "1" if operation == "quarantine" else "99")
    acquired = _processing_job(jm, operation)
    observed, audited, retries, logged = _patch_lifecycle_observers(monkeypatch)

    assert _invoke_lifecycle(jm, acquired, operation)

    expected_status = {
        "complete": "completed",
        "terminal_fail": "failed",
        "retry": "queued",
        "quarantine": "quarantined",
    }[operation]
    persisted = jm.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] == expected_status
    expected_event = _EXPECTED_EVENT[operation]
    assert observed == [expected_event]
    assert audited == [expected_event]
    assert len(logged) == int(events_enabled)
    assert _event_count(jm, int(acquired["id"]), expected_event) == int(outbox_enabled)
    assert len(retries) == int(operation == "retry")


def _assert_transaction_failure(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    operation: str,
    wrapper: type[Any],
    error_type: type[BaseException],
    error_match: str,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "1" if operation == "quarantine" else "99")
    acquired = _processing_job(jm, operation)
    counter_before = _counter_snapshot(reader, acquired)
    observed, audited, retries, logged = _patch_lifecycle_observers(monkeypatch)
    original_connect = jm._connect
    monkeypatch.setattr(jm, "_connect", lambda: wrapper(original_connect()))

    with pytest.raises(error_type, match=error_match):
        _invoke_lifecycle(jm, acquired, operation)

    persisted = reader.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] == "processing"
    assert persisted["worker_id"] == "worker-1"
    assert persisted["lease_id"] == acquired["lease_id"]
    assert _event_count(reader, int(acquired["id"]), _EXPECTED_EVENT[operation]) == 0
    assert _counter_snapshot(reader, acquired) == counter_before
    assert observed == []
    assert audited == []
    assert retries == []
    assert logged == []


@pytest.mark.unit
@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("events_enabled", [False, True], ids=["events-off", "events-on"])
@pytest.mark.parametrize("outbox_enabled", [False, True], ids=["outbox-off", "outbox-on"])
def test_sqlite_lifecycle_event_flag_contract(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    events_enabled: bool,
    outbox_enabled: bool,
) -> None:
    _assert_flag_contract(
        _manager(tmp_path, None),
        monkeypatch,
        operation=operation,
        events_enabled=events_enabled,
        outbox_enabled=outbox_enabled,
    )


@pytest.mark.pg_jobs
@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("events_enabled", [False, True], ids=["events-off", "events-on"])
@pytest.mark.parametrize("outbox_enabled", [False, True], ids=["outbox-off", "outbox-on"])
def test_postgres_lifecycle_event_flag_contract(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    events_enabled: bool,
    outbox_enabled: bool,
) -> None:
    _assert_flag_contract(
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
        operation=operation,
        events_enabled=events_enabled,
        outbox_enabled=outbox_enabled,
    )


@pytest.mark.unit
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_sqlite_lifecycle_outbox_insert_failure_rolls_back_state(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_transaction_failure(
        jm,
        _manager(tmp_path, None),
        monkeypatch,
        operation=operation,
        wrapper=_FailOutboxInsertSQLite,
        error_type=sqlite3.OperationalError,
        error_match="lifecycle outbox insert failure",
    )


@pytest.mark.pg_jobs
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_postgres_lifecycle_outbox_insert_failure_rolls_back_state(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import psycopg

    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_transaction_failure(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
        operation=operation,
        wrapper=_FailOutboxInsertPostgres,
        error_type=psycopg.OperationalError,
        error_match="lifecycle outbox insert failure",
    )


@pytest.mark.unit
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_sqlite_lifecycle_commit_failure_rolls_back_state_and_outbox(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_transaction_failure(
        jm,
        _manager(tmp_path, None),
        monkeypatch,
        operation=operation,
        wrapper=_FailCommitSQLite,
        error_type=RuntimeError,
        error_match="lifecycle commit failure",
    )


@pytest.mark.pg_jobs
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_postgres_lifecycle_commit_failure_rolls_back_state_and_outbox(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_transaction_failure(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
        operation=operation,
        wrapper=_FailCommitPostgres,
        error_type=RuntimeError,
        error_match="lifecycle commit failure",
    )

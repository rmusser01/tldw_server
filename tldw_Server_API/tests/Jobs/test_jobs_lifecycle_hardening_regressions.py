from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.unit

_BACKENDS = (
    pytest.param("sqlite", id="sqlite"),
    pytest.param("postgres", marks=pytest.mark.pg_jobs, id="postgres"),
)


def _manager(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    name: str,
) -> JobManager:
    if backend == "postgres":
        return JobManager(
            None,
            backend="postgres",
            db_url=str(request.getfixturevalue("jobs_pg_dsn")),
        )
    return JobManager(tmp_path / f"{name}.db")


def _clone_manager(manager: JobManager) -> JobManager:
    if manager.backend == "postgres":
        return JobManager(None, backend="postgres", db_url=manager.db_url)
    return JobManager(manager.db_path)


def _create_job(
    manager: JobManager,
    *,
    domain: str,
    available_at: datetime | None = None,
) -> dict[str, Any]:
    return manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner-1",
        request_id="request-1",
        trace_id="trace-1",
        max_retries=3,
        available_at=available_at,
    )


def _acquire(manager: JobManager, job: dict[str, Any]) -> dict[str, Any]:
    acquired = manager.acquire_next_job(
        domain=str(job["domain"]),
        queue=str(job["queue"]),
        job_type=str(job["job_type"]),
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])
    return acquired


def _event_count(manager: JobManager, job_id: int, event_type: str) -> int:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS count FROM job_events "
                    "WHERE job_id=%s AND event_type=%s",
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


def _attachment_count(manager: JobManager, job_id: int) -> int:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS count FROM job_attachments WHERE job_id=%s",
                    (job_id,),
                )
                row = cur.fetchone()
                return int((row or {}).get("count") or 0)
        row = conn.execute(
            "SELECT COUNT(*) FROM job_attachments WHERE job_id=?",
            (job_id,),
        ).fetchone()
        return int(row[0] if row else 0)
    finally:
        conn.close()


def _counter_snapshot(
    manager: JobManager,
    job: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    params = (job["domain"], job["queue"], job["job_type"])
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT ready_count, scheduled_count, processing_count, "
                    "quarantined_count FROM job_counters "
                    "WHERE domain=%s AND queue=%s AND job_type=%s",
                    params,
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return (
                    int(row["ready_count"]),
                    int(row["scheduled_count"]),
                    int(row["processing_count"]),
                    int(row["quarantined_count"]),
                )
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count, "
            "quarantined_count FROM job_counters "
            "WHERE domain=? AND queue=? AND job_type=?",
            params,
        ).fetchone()
        if row is None:
            return None
        return tuple(int(value) for value in row)
    finally:
        conn.close()


def _set_completion_token(manager: JobManager, job_id: int, token: str) -> None:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with conn, manager._pg_cursor(conn) as cur:
                cur.execute(
                    "UPDATE jobs SET completion_token=%s WHERE id=%s",
                    (token, job_id),
                )
            return
        with conn:
            conn.execute(
                "UPDATE jobs SET completion_token=? WHERE id=?",
                (token, job_id),
            )
    finally:
        conn.close()


def _delete_counter(manager: JobManager, job: dict[str, Any]) -> None:
    params = (job["domain"], job["queue"], job["job_type"])
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with conn, manager._pg_cursor(conn) as cur:
                cur.execute(
                    "DELETE FROM job_counters "
                    "WHERE domain=%s AND queue=%s AND job_type=%s",
                    params,
                )
            return
        with conn:
            conn.execute(
                "DELETE FROM job_counters "
                "WHERE domain=? AND queue=? AND job_type=?",
                params,
            )
    finally:
        conn.close()


def _backdate_started_at(manager: JobManager, job_id: int) -> None:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with conn, manager._pg_cursor(conn) as cur:
                cur.execute(
                    "UPDATE jobs SET started_at=NOW() - INTERVAL '1 hour' WHERE id=%s",
                    (job_id,),
                )
            return
        with conn:
            conn.execute(
                "UPDATE jobs SET started_at=DATETIME('now', '-1 hour') WHERE id=?",
                (job_id,),
            )
    finally:
        conn.close()


def _terminalize_with_token(
    manager: JobManager,
    acquired: dict[str, Any],
    status: str,
    token: str,
) -> None:
    common = {
        "worker_id": str(acquired["worker_id"]),
        "lease_id": str(acquired["lease_id"]),
        "completion_token": token,
        "enforce": True,
    }
    if status == "completed":
        assert manager.complete_job(int(acquired["id"]), **common)
        return
    if status == "failed":
        assert manager.fail_job(
            int(acquired["id"]),
            error="terminal failure",
            retryable=False,
            **common,
        )
        return
    if status == "quarantined":
        assert manager.fail_job(
            int(acquired["id"]),
            error="poison message",
            retryable=True,
            backoff_seconds=1,
            **common,
        )
        return
    if status == "cancelled":
        _set_completion_token(manager, int(acquired["id"]), token)
        assert manager.finalize_cancelled(
            int(acquired["id"]),
            expected_uuid=str(acquired["uuid"]),
            worker_id=str(acquired["worker_id"]),
            lease_id=str(acquired["lease_id"]),
        )
        return
    raise AssertionError(f"unsupported terminal status: {status}")


def _invoke_terminal_operation(
    manager: JobManager,
    acquired: dict[str, Any],
    operation: str,
    token: str,
) -> bool:
    common = {
        "worker_id": str(acquired["worker_id"]),
        "lease_id": str(acquired["lease_id"]),
        "completion_token": token,
        "enforce": True,
    }
    if operation == "complete":
        return manager.complete_job(int(acquired["id"]), **common)
    if operation == "terminal_fail":
        return manager.fail_job(
            int(acquired["id"]),
            error="terminal failure",
            retryable=False,
            **common,
        )
    if operation == "retry":
        return manager.fail_job(
            int(acquired["id"]),
            error="retryable failure",
            retryable=True,
            backoff_seconds=2,
            **common,
        )
    if operation == "quarantine":
        return manager.fail_job(
            int(acquired["id"]),
            error="poison message",
            retryable=True,
            backoff_seconds=1,
            **common,
        )
    raise AssertionError(f"unsupported lifecycle operation: {operation}")


def _is_counter_mutation(sql: Any) -> bool:
    normalized = " ".join(str(sql).upper().split())
    return "JOB_COUNTERS" in normalized and normalized.startswith(
        ("INSERT", "UPDATE", "DELETE")
    )


class _FailCounterCursor:
    """PostgreSQL cursor adapter that fails before a counter write is sent."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _FailCounterCursor:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        if _is_counter_mutation(sql):
            import psycopg

            raise psycopg.OperationalError("forced lifecycle counter write failure")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailCounterConnection:
    """Connection adapter that fails only lifecycle counter mutations."""

    def __init__(self, inner: Any, backend: str) -> None:
        self._inner = inner
        self._backend = backend

    def __enter__(self) -> _FailCounterConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        cursor = self._inner.cursor(*args, **kwargs)
        if self._backend == "postgres":
            return _FailCounterCursor(cursor)
        return cursor

    def execute(self, sql: Any, params: Any = ()) -> Any:
        if _is_counter_mutation(sql):
            raise sqlite3.OperationalError("forced lifecycle counter write failure")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _configure_sla(manager: JobManager, domain: str) -> None:
    manager.upsert_sla_policy(
        domain=domain,
        queue="default",
        job_type="work",
        max_duration_seconds=0,
        enabled=True,
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_nonallowlisted_queued_terminal_failure_is_not_reported_as_success(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "allowed-only")
    manager = _manager(backend, request, tmp_path, "queued-not-allowed")
    job = _create_job(manager, domain="not-allowed")
    counters_before = _counter_snapshot(manager, job)

    assert not manager.fail_job(
        int(job["id"]),
        error="must remain queued",
        retryable=False,
        completion_token="token-a",
        enforce=False,
    )

    persisted = manager.get_job(int(job["id"]))
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert _event_count(manager, int(job["id"]), "job.failed") == 0
    assert _counter_snapshot(manager, job) == counters_before


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("bucket", ("ready", "scheduled"))
def test_queued_terminal_failure_decrements_its_original_counter_bucket(
    backend: str,
    bucket: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "allowed")
    manager = _manager(backend, request, tmp_path, f"queued-{bucket}")
    available_at = (
        datetime.now(timezone.utc) + timedelta(hours=1)
        if bucket == "scheduled"
        else None
    )
    job = _create_job(manager, domain="allowed", available_at=available_at)
    expected_before = (1, 0, 0, 0) if bucket == "ready" else (0, 1, 0, 0)
    assert _counter_snapshot(manager, job) == expected_before

    assert manager.fail_job(
        int(job["id"]),
        error="administrative failure",
        retryable=False,
        completion_token="token-a",
        enforce=False,
    )

    persisted = manager.get_job(int(job["id"]))
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert _event_count(manager, int(job["id"]), "job.failed") == 1
    assert _counter_snapshot(manager, job) == (0, 0, 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_queued_terminal_failure_rejects_a_conflicting_completion_token(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
    monkeypatch.setenv("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "allowed")
    manager = _manager(backend, request, tmp_path, "queued-token-conflict")
    job = _create_job(manager, domain="allowed")
    _set_completion_token(manager, int(job["id"]), "token-a")
    counters_before = _counter_snapshot(manager, job)

    assert not manager.fail_job(
        int(job["id"]),
        error="conflicting finalizer",
        retryable=False,
        completion_token="token-b",
        enforce=False,
    )

    persisted = manager.get_job(int(job["id"]))
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert persisted["completion_token"] == "token-a"
    assert _event_count(manager, int(job["id"]), "job.failed") == 0
    assert _counter_snapshot(manager, job) == counters_before


@pytest.mark.parametrize("backend", _BACKENDS)
def test_quarantine_persists_token_and_replays_only_the_matching_token(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "1")
    manager = _manager(backend, request, tmp_path, "quarantine-token")
    acquired = _acquire(manager, _create_job(manager, domain="quarantine-token"))
    common = {
        "error": "poison message",
        "error_code": "E_POISON",
        "retryable": True,
        "backoff_seconds": 1,
        "worker_id": str(acquired["worker_id"]),
        "lease_id": str(acquired["lease_id"]),
        "enforce": True,
    }

    assert manager.fail_job(
        int(acquired["id"]),
        completion_token="token-a",
        **common,
    )
    persisted = manager.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] == "quarantined"
    assert persisted["completion_token"] == "token-a"
    assert manager.fail_job(
        int(acquired["id"]),
        completion_token="token-a",
        **common,
    )
    assert not manager.fail_job(
        int(acquired["id"]),
        completion_token="token-b",
        **common,
    )
    assert _event_count(manager, int(acquired["id"]), "job.quarantined") == 1
    assert _counter_snapshot(manager, acquired) == (0, 0, 0, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    ("persisted_status", "replayed_operation", "replayed_event"),
    (
        ("failed", "complete", "job.completed"),
        ("quarantined", "complete", "job.completed"),
        ("cancelled", "complete", "job.completed"),
        ("completed", "terminal_fail", "job.failed"),
        ("cancelled", "terminal_fail", "job.failed"),
    ),
)
def test_completion_token_replay_is_scoped_to_the_requested_terminal_operation(
    backend: str,
    persisted_status: str,
    replayed_operation: str,
    replayed_event: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
    monkeypatch.setenv(
        "JOBS_QUARANTINE_THRESHOLD",
        "1" if persisted_status == "quarantined" else "99",
    )
    manager = _manager(
        backend,
        request,
        tmp_path,
        f"cross-replay-{persisted_status}-{replayed_operation}",
    )
    acquired = _acquire(
        manager,
        _create_job(
            manager,
            domain=f"cross-replay-{persisted_status}-{replayed_operation}",
        ),
    )
    token = "shared-finalization-token"
    _terminalize_with_token(manager, acquired, persisted_status, token)
    counters_before = _counter_snapshot(manager, acquired)
    event_count_before = _event_count(
        manager,
        int(acquired["id"]),
        replayed_event,
    )

    assert not _invoke_terminal_operation(
        manager,
        acquired,
        replayed_operation,
        token,
    )

    persisted = manager.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] == persisted_status
    assert persisted["completion_token"] == token
    assert _counter_snapshot(manager, acquired) == counters_before
    assert (
        _event_count(manager, int(acquired["id"]), replayed_event)
        == event_count_before
    )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    ("operation", "event_type"),
    (
        ("complete", "job.completed"),
        ("retry", "job.retry_scheduled"),
        ("quarantine", "job.quarantined"),
        ("terminal_fail", "job.failed"),
    ),
)
def test_lifecycle_counter_write_failure_rolls_back_state_event_and_observers(
    backend: str,
    operation: str,
    event_type: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
    monkeypatch.setenv(
        "JOBS_QUARANTINE_THRESHOLD",
        "1" if operation == "quarantine" else "99",
    )
    manager = _manager(backend, request, tmp_path, f"counter-failure-{operation}")
    acquired = _acquire(
        manager,
        _create_job(manager, domain=f"counter-failure-{operation}"),
    )
    reader = _clone_manager(manager)
    counters_before = _counter_snapshot(reader, acquired)
    observed: list[str] = []
    import tldw_Server_API.app.core.Jobs.manager as manager_module

    monkeypatch.setattr(
        manager_module,
        "observe_job_event",
        lambda event, **_kwargs: observed.append(event),
    )
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _FailCounterConnection(original_connect(), backend),
    )
    if backend == "postgres":
        import psycopg

        error_type = psycopg.OperationalError
    else:
        error_type = sqlite3.OperationalError

    with pytest.raises(error_type, match="lifecycle counter write failure"):
        _invoke_terminal_operation(
            manager,
            acquired,
            operation,
            "counter-failure-token",
        )

    persisted = reader.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] == "processing"
    assert persisted["worker_id"] == acquired["worker_id"]
    assert persisted["lease_id"] == acquired["lease_id"]
    assert persisted["completion_token"] is None
    assert _counter_snapshot(reader, acquired) == counters_before
    assert _event_count(reader, int(acquired["id"]), event_type) == 0
    assert observed == []


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    ("operation", "event_type", "expected_status"),
    (
        ("complete", "job.completed", "completed"),
        ("terminal_fail", "job.failed", "failed"),
    ),
)
def test_terminal_lifecycle_reconciles_a_missing_counter_from_current_jobs(
    backend: str,
    operation: str,
    event_type: str,
    expected_status: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
    domain = f"missing-terminal-counter-{operation}"
    manager = _manager(backend, request, tmp_path, domain)
    target = _create_job(manager, domain=domain)
    sibling = _create_job(manager, domain=domain)
    acquired = _acquire(manager, target)
    assert manager.get_job(int(sibling["id"]))["status"] == "queued"
    _delete_counter(manager, acquired)
    assert _counter_snapshot(manager, acquired) is None

    assert _invoke_terminal_operation(
        manager,
        acquired,
        operation,
        "missing-counter-token",
    )

    persisted = manager.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] == expected_status
    assert manager.get_job(int(sibling["id"]))["status"] == "queued"
    assert _counter_snapshot(manager, acquired) == (1, 0, 0, 0)
    assert _event_count(manager, int(acquired["id"]), event_type) == 1


@pytest.mark.parametrize("backend", _BACKENDS)
def test_zero_delay_retry_is_ready_and_uses_null_available_at(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "99")
    manager = _manager(backend, request, tmp_path, "zero-delay")
    acquired = _acquire(manager, _create_job(manager, domain="zero-delay"))

    assert manager.fail_job(
        int(acquired["id"]),
        error="retry immediately",
        retryable=True,
        backoff_seconds=0,
        worker_id=str(acquired["worker_id"]),
        lease_id=str(acquired["lease_id"]),
        completion_token="token-a",
        enforce=True,
    )

    persisted = manager.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert persisted["available_at"] is None
    assert _counter_snapshot(manager, acquired) == (1, 0, 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("transition", ("retry", "quarantine"))
def test_retry_transition_recreates_a_missing_counter_with_transition_delta(
    backend: str,
    transition: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
    monkeypatch.setenv(
        "JOBS_QUARANTINE_THRESHOLD",
        "1" if transition == "quarantine" else "99",
    )
    manager = _manager(backend, request, tmp_path, f"missing-counter-{transition}")
    acquired = _acquire(
        manager,
        _create_job(manager, domain=f"missing-counter-{transition}"),
    )
    _delete_counter(manager, acquired)
    assert _counter_snapshot(manager, acquired) is None

    assert manager.fail_job(
        int(acquired["id"]),
        error="transition",
        retryable=True,
        backoff_seconds=2,
        worker_id=str(acquired["worker_id"]),
        lease_id=str(acquired["lease_id"]),
        completion_token="token-a",
        enforce=True,
    )

    expected = (0, 0, 0, 1) if transition == "quarantine" else (0, 1, 0, 0)
    assert _counter_snapshot(manager, acquired) == expected


class _CommitSnapshotConnection:
    """Record lifecycle rows visible immediately before each explicit commit."""

    def __init__(
        self,
        inner: Any,
        backend: str,
        snapshots: list[tuple[int, int, int]],
    ) -> None:
        self._inner = inner
        self._backend = backend
        self._snapshots = snapshots

    def __enter__(self) -> _CommitSnapshotConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def _count(self, table: str, predicate: str, params: tuple[Any, ...]) -> int:
        if self._backend == "postgres":
            with self._inner.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {predicate}", params)
                row = cur.fetchone()
                return int(row[0] if row else 0)
        row = self._inner.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {predicate}",
            params,
        ).fetchone()
        return int(row[0] if row else 0)

    def commit(self) -> None:
        placeholder = "%s" if self._backend == "postgres" else "?"
        completed = self._count(
            "job_events",
            f"event_type={placeholder}",
            ("job.completed",),
        )
        breached = self._count(
            "job_events",
            f"event_type={placeholder}",
            ("job.sla_breached",),
        )
        attachments = self._count(
            "job_attachments",
            f"kind={placeholder}",
            ("tag",),
        )
        self._snapshots.append((completed, breached, attachments))
        self._inner.commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_completion_commits_sla_attachment_and_outbox_event_atomically(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    manager = _manager(backend, request, tmp_path, "sla-atomic")
    _configure_sla(manager, "sla-atomic")
    acquired = _acquire(manager, _create_job(manager, domain="sla-atomic"))
    _backdate_started_at(manager, int(acquired["id"]))
    if backend == "postgres":
        persisted = manager.get_job(int(acquired["id"]))
        assert persisted is not None
        assert persisted["started_at"].tzinfo is not None

    snapshots: list[tuple[int, int, int]] = []
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _CommitSnapshotConnection(original_connect(), backend, snapshots),
    )

    assert manager.complete_job(
        int(acquired["id"]),
        worker_id=str(acquired["worker_id"]),
        lease_id=str(acquired["lease_id"]),
        completion_token="token-a",
        enforce=True,
    )

    assert snapshots
    assert snapshots[0] == (1, 1, 1)
    reader = _clone_manager(manager)
    assert _event_count(reader, int(acquired["id"]), "job.completed") == 1
    assert _event_count(reader, int(acquired["id"]), "job.sla_breached") == 1
    assert _attachment_count(reader, int(acquired["id"])) == 1


class _FailSlaCursor:
    """PostgreSQL cursor adapter that injects a real statement failure."""

    def __init__(self, inner: Any, flags: dict[str, bool]) -> None:
        self._inner = inner
        self._flags = flags

    def __enter__(self) -> _FailSlaCursor:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        normalized = str(sql).strip().upper()
        _record_savepoint_statement(normalized, self._flags)
        if "INSERT INTO JOB_ATTACHMENTS" in normalized:
            return self._inner.execute("SELECT * FROM missing_jobs_sla_table")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _record_savepoint_statement(sql: str, flags: dict[str, bool]) -> None:
    if sql.startswith("SAVEPOINT JOB_COMPLETION_SLA"):
        flags["savepoint"] = True
    elif sql.startswith("ROLLBACK TO SAVEPOINT JOB_COMPLETION_SLA"):
        flags["rollback"] = True
    elif sql.startswith("RELEASE SAVEPOINT JOB_COMPLETION_SLA"):
        flags["release"] = True


class _FailSlaConnection:
    """Connection adapter that fails optional SLA attachment persistence."""

    def __init__(self, inner: Any, backend: str, flags: dict[str, bool]) -> None:
        self._inner = inner
        self._backend = backend
        self._flags = flags

    def __enter__(self) -> _FailSlaConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self._inner.__exit__(exc_type, exc, tb)

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        cursor = self._inner.cursor(*args, **kwargs)
        if self._backend == "postgres":
            return _FailSlaCursor(cursor, self._flags)
        return cursor

    def execute(self, sql: str, params: Any = ()) -> Any:
        normalized = str(sql).strip().upper()
        _record_savepoint_statement(normalized, self._flags)
        if "INSERT INTO JOB_ATTACHMENTS" in normalized:
            return self._inner.execute("SELECT * FROM missing_jobs_sla_table")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_optional_sla_write_failure_rolls_back_to_savepoint_without_losing_completion(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    manager = _manager(backend, request, tmp_path, "sla-savepoint")
    _configure_sla(manager, "sla-savepoint")
    acquired = _acquire(manager, _create_job(manager, domain="sla-savepoint"))
    _backdate_started_at(manager, int(acquired["id"]))

    flags = {"savepoint": False, "rollback": False, "release": False}
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _FailSlaConnection(original_connect(), backend, flags),
    )

    assert manager.complete_job(
        int(acquired["id"]),
        worker_id=str(acquired["worker_id"]),
        lease_id=str(acquired["lease_id"]),
        completion_token="token-a",
        enforce=True,
    )

    reader = _clone_manager(manager)
    persisted = reader.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] == "completed"
    assert _event_count(reader, int(acquired["id"]), "job.completed") == 1
    assert _event_count(reader, int(acquired["id"]), "job.sla_breached") == 0
    assert _attachment_count(reader, int(acquired["id"])) == 0
    assert flags == {"savepoint": True, "rollback": True, "release": True}


def _concurrent_lifecycle_call(
    manager: JobManager,
    acquired: dict[str, Any],
    operation: str,
    barrier: threading.Barrier,
) -> bool:
    barrier.wait(timeout=10)
    common = {
        "worker_id": str(acquired["worker_id"]),
        "lease_id": str(acquired["lease_id"]),
        "completion_token": "shared-token",
        "enforce": True,
    }
    if operation == "complete":
        return manager.complete_job(
            int(acquired["id"]),
            result={"ok": True},
            **common,
        )
    return manager.fail_job(
        int(acquired["id"]),
        error="concurrent finalizer",
        error_code="E_CONCURRENT",
        retryable=operation != "terminal_fail",
        backoff_seconds=1,
        **common,
    )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    ("operation", "expected_status", "event_type", "expected_results"),
    (
        ("complete", "completed", "job.completed", [True, True]),
        ("terminal_fail", "failed", "job.failed", [True, True]),
        ("retry", "queued", "job.retry_scheduled", [False, True]),
        ("quarantine", "quarantined", "job.quarantined", [True, True]),
    ),
)
def test_simultaneous_same_operation_finalizers_emit_one_durable_event(
    backend: str,
    operation: str,
    expected_status: str,
    event_type: str,
    expected_results: list[bool],
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
    monkeypatch.setenv(
        "JOBS_QUARANTINE_THRESHOLD",
        "1" if operation == "quarantine" else "99",
    )
    manager = _manager(backend, request, tmp_path, f"concurrent-{operation}")
    acquired = _acquire(
        manager,
        _create_job(manager, domain=f"concurrent-{operation}"),
    )
    managers = (manager, _clone_manager(manager))
    barrier = threading.Barrier(2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _concurrent_lifecycle_call,
                current,
                acquired,
                operation,
                barrier,
            )
            for current in managers
        ]
        results = sorted(future.result(timeout=20) for future in futures)

    persisted = manager.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] == expected_status
    assert results == sorted(expected_results)
    assert _event_count(manager, int(acquired["id"]), event_type) == 1


@pytest.mark.parametrize("backend", _BACKENDS)
def test_simultaneous_complete_and_fail_with_one_token_apply_exactly_one_outcome(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
    manager = _manager(backend, request, tmp_path, "concurrent-cross-operation")
    acquired = _acquire(
        manager,
        _create_job(manager, domain="concurrent-cross-operation"),
    )
    managers = (manager, _clone_manager(manager))
    barrier = threading.Barrier(2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _concurrent_lifecycle_call,
                current,
                acquired,
                operation,
                barrier,
            )
            for current, operation in zip(
                managers,
                ("complete", "terminal_fail"),
                strict=True,
            )
        ]
        results = sorted(future.result(timeout=20) for future in futures)

    persisted = manager.get_job(int(acquired["id"]))
    assert persisted is not None
    assert persisted["status"] in {"completed", "failed"}
    assert results == [False, True]
    completed_events = _event_count(
        manager,
        int(acquired["id"]),
        "job.completed",
    )
    failed_events = _event_count(manager, int(acquired["id"]), "job.failed")
    assert completed_events + failed_events == 1
    assert completed_events == int(persisted["status"] == "completed")
    assert failed_events == int(persisted["status"] == "failed")
    assert _counter_snapshot(manager, acquired) == (0, 0, 0, 0)

"""Regressions for completing exactly the row observed under the transaction lock."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager

_DOMAIN = "chatbooks"
_QUEUE = "default"
_JOB_TYPE = "completion-identity"
_SCOPE = (_DOMAIN, _QUEUE, _JOB_TYPE)
_COMPLETION_A = "token-a"
_COMPLETION_B = "token-b"


def _is_completion_read(sql: Any) -> bool:
    """Identify the authoritative completion lookup for the race hooks."""
    normalized = " ".join(str(sql).upper().split())
    return (
        normalized.startswith("SELECT ")
        and "COMPLETION_TOKEN" in normalized
        and "DOMAIN" in normalized
        and "FROM JOBS WHERE ID" in normalized
    )


class _FetchedRow:
    """Return the snapshot consumed before the competing write was attempted."""

    def __init__(self, row: Any) -> None:
        """Retain the row fetched before invoking the competing writer."""
        self._row = row

    def fetchone(self) -> Any:
        """Return the retained row once, matching cursor consumption."""
        row, self._row = self._row, None
        return row


class _SQLiteCompletionReadHook:
    """Delegate the connection and hook only its first completion state read."""

    def __init__(self, inner: Any, callback: Callable[[], None]) -> None:
        """Wrap a SQLite connection with a one-shot completion callback."""
        self._inner = inner
        self._callback = callback
        self.fired = False

    def __enter__(self) -> _SQLiteCompletionReadHook:
        """Enter the wrapped transaction while retaining the read hook."""
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        """Delegate transaction commit or rollback to the connection."""
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: str, parameters: Any = ()) -> Any:
        """Attempt the competing write after fetching completion facts."""
        cursor = self._inner.execute(sql, parameters)
        if not self.fired and _is_completion_read(sql):
            self.fired = True
            row = cursor.fetchone()
            cursor.close()
            self._callback()
            return _FetchedRow(row)
        return cursor

    def __getattr__(self, name: str) -> Any:
        """Delegate connection operations unrelated to the hooked read."""
        return getattr(self._inner, name)


class _PostgresCompletionReadHook:
    """Hook fetchone after the real, RLS-aware cursor executes the state read."""

    def __init__(self, inner: Any, callback: Callable[[], None]) -> None:
        """Wrap the real PostgreSQL cursor without replacing its RLS setup."""
        self._inner = inner
        self._callback = callback
        self._armed = False
        self.fired = False
        self.row_locked = False

    def execute(self, sql: Any, parameters: Any = None) -> Any:
        """Arm the callback and record whether completion requested a lock."""
        result = self._inner.execute(sql, parameters)
        if not self.fired and _is_completion_read(sql):
            self._armed = True
            self.row_locked = "FOR UPDATE" in " ".join(str(sql).upper().split())
        return result

    def fetchone(self) -> Any:
        """Invoke the competing transaction after consuming completion facts."""
        row = self._inner.fetchone()
        if self._armed:
            self._armed = False
            self.fired = True
            self._callback()
        return row

    def __getattr__(self, name: str) -> Any:
        """Delegate cursor operations unrelated to the hooked read."""
        return getattr(self._inner, name)


def _hook_completion_read(
    jm: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    callback: Callable[[], None],
) -> list[Any]:
    """Install a one-shot read callback while keeping real backend execution."""
    hooks: list[Any] = []
    if jm.backend == "postgres":
        original_cursor = jm._pg_cursor

        @contextmanager
        def pg_cursor(conn: Any) -> Iterator[Any]:
            """Yield the hooked first cursor using the manager's RLS context."""
            with original_cursor(conn) as cursor:
                if not hooks:
                    hook = _PostgresCompletionReadHook(cursor, callback)
                    hooks.append(hook)
                    yield hook
                else:
                    yield cursor

        monkeypatch.setattr(jm, "_pg_cursor", pg_cursor)
    else:
        original_connect = jm._connect

        def connect() -> Any:
            """Wrap only the first connection used by the completion attempt."""
            conn = original_connect()
            if not hooks:
                hook = _SQLiteCompletionReadHook(conn, callback)
                hooks.append(hook)
                return hook
            return conn

        monkeypatch.setattr(jm, "_connect", connect)
    return hooks


@pytest.fixture(autouse=True)
def _completion_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable atomic bookkeeping and isolate completion policy settings."""
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EVENTS_ENABLED", "false")
    monkeypatch.setenv("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", _DOMAIN)
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "false")
    monkeypatch.setenv("JOBS_JSON_TRUNCATE", "false")
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "1048576")


@pytest.fixture(
    params=[
        pytest.param("sqlite", marks=pytest.mark.unit),
        pytest.param("postgres", marks=pytest.mark.pg_jobs),
    ]
)
def jm(request: pytest.FixtureRequest, tmp_path: Path) -> JobManager:
    """Provide SQLite or the existing isolated Jobs PostgreSQL fixture."""
    if request.param == "postgres":
        pytest.importorskip("psycopg")
        dsn = request.getfixturevalue("jobs_pg_dsn")
        return JobManager(None, backend="postgres", db_url=dsn)
    return JobManager(tmp_path / "completion-identity.db")


def _create_job(jm: JobManager, *, processing: bool = False) -> dict[str, Any]:
    """Create a scoped job and optionally acquire its real processing lease."""
    created = jm.create_job(
        domain=_DOMAIN,
        queue=_QUEUE,
        job_type=_JOB_TYPE,
        payload={},
        owner_user_id="owner-1",
        request_id="request-1",
        trace_id="trace-1",
    )
    if not processing:
        return created
    acquired = jm.acquire_next_job(
        domain=_DOMAIN,
        queue=_QUEUE,
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert acquired is not None
    assert acquired["id"] == created["id"]
    assert acquired["status"] == "processing"
    assert acquired["lease_id"]
    return acquired


def _rows(
    jm: JobManager,
    sqlite_sql: str,
    postgres_sql: str,
    parameters: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Read durable backend rows through the manager's connection policy."""
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cursor:
                cursor.execute(postgres_sql, parameters)
                return [dict(row) for row in cursor.fetchall()]
        return [dict(row) for row in conn.execute(sqlite_sql, parameters).fetchall()]
    finally:
        conn.close()


def _raw_job(jm: JobManager, job_id: int) -> dict[str, Any] | None:
    """Read stored job fields without facade normalization."""
    rows = _rows(
        jm,
        "SELECT * FROM jobs WHERE id=?",
        "SELECT * FROM jobs WHERE id=%s",
        (job_id,),
    )
    return rows[0] if rows else None


def _counters(jm: JobManager) -> tuple[int, ...]:
    """Read all lifecycle counters for the regression job scope."""
    rows = _rows(
        jm,
        "SELECT ready_count, scheduled_count, processing_count, quarantined_count "
        "FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
        "SELECT ready_count, scheduled_count, processing_count, quarantined_count "
        "FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
        _SCOPE,
    )
    assert len(rows) == 1
    return tuple(
        int(rows[0][key])
        for key in (
            "ready_count",
            "scheduled_count",
            "processing_count",
            "quarantined_count",
        )
    )


def _events(jm: JobManager, job_id: int) -> list[dict[str, Any]]:
    """Read the job's durable outbox events in insertion order."""
    return _rows(
        jm,
        "SELECT * FROM job_events WHERE job_id=? ORDER BY id",
        "SELECT * FROM job_events WHERE job_id=%s ORDER BY id",
        (job_id,),
    )


def _snapshot(jm: JobManager, job_id: int) -> tuple[Any, ...]:
    """Capture job state, counters and events for no-mutation assertions."""
    return _raw_job(jm, job_id), _counters(jm), _events(jm, job_id)


def _set_identity_and_token(
    jm: JobManager,
    job_id: int,
    raw_uuid: str | None,
    token: str | None = None,
) -> None:
    """Seed raw identity and token values without normalizing legacy UUIDs."""
    conn = jm._connect()
    try:
        with conn:
            parameters = (raw_uuid, token, job_id)
            if jm.backend == "postgres":
                with jm._pg_cursor(conn) as cursor:
                    cursor.execute(
                        "UPDATE jobs SET uuid=%s, completion_token=%s WHERE id=%s",
                        parameters,
                    )
            else:
                conn.execute("UPDATE jobs SET uuid=?, completion_token=? WHERE id=?", parameters)
    finally:
        conn.close()


def _completion_observers(jm: JobManager, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Capture post-commit metrics and observer calls without external effects."""
    import tldw_Server_API.app.core.Jobs.manager as manager_module

    calls: dict[str, list[Any]] = {
        "completed": [],
        "duration": [],
        "gauge": [],
        "event": [],
    }
    monkeypatch.setattr(
        manager_module,
        "increment_completed",
        lambda labels: calls["completed"].append(labels),
    )
    monkeypatch.setattr(
        manager_module,
        "observe_duration",
        lambda *args: calls["duration"].append(args),
    )
    monkeypatch.setattr(jm, "_update_gauges", lambda **kwargs: calls["gauge"].append(kwargs))
    monkeypatch.setattr(
        manager_module,
        "observe_job_event",
        lambda event_type, **kwargs: calls["event"].append((event_type, kwargs)),
    )
    return calls


def _assert_one_completion(jm: JobManager, job_id: int, *, processing: bool = False) -> None:
    """Assert one completion with balanced counters and original event context."""
    assert _counters(jm) == (0, 0, 0, 0)
    expected_events = ["job.created"]
    if processing:
        expected_events.append("job.acquired")
    expected_events.append("job.completed")
    events = _events(jm, job_id)
    assert [row["event_type"] for row in events] == expected_events
    assert events[-1]["owner_user_id"] == "owner-1"
    assert events[-1]["request_id"] == "request-1"
    assert events[-1]["trace_id"] == "trace-1"


def _zero_timeout_connection(db_path: Path) -> sqlite3.Connection:
    """Open a competing SQLite writer that reports contention immediately."""
    conn = sqlite3.connect(str(db_path), timeout=0)
    try:
        configure_sqlite_connection(conn, busy_timeout_ms=0)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        conn.close()
        raise


def _assert_sqlite_busy(codes: list[int]) -> None:
    """Check contention using SQLite error codes rather than message text."""
    assert len(codes) == 1
    assert codes[0] & 0xFF in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


@pytest.mark.unit
def test_sqlite_missing_row_cannot_complete_concurrent_insert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A locked miss cannot complete a row inserted after the transaction."""
    db_path = tmp_path / "completion-missing.db"
    manager = JobManager(db_path)
    creator = JobManager(db_path)
    monkeypatch.setattr(creator, "_connect", lambda: _zero_timeout_connection(db_path))
    busy_codes: list[int] = []
    inserted: list[dict[str, Any]] = []

    def insert_after_missing_read() -> None:
        """Attempt insertion while completion holds its immediate transaction."""
        try:
            inserted.append(_create_job(creator))
        except sqlite3.OperationalError as exc:
            busy_codes.append(exc.sqlite_errorcode)

    hooks = _hook_completion_read(manager, monkeypatch, insert_after_missing_read)
    completed = manager.complete_job(1, result={"ok": True}, completion_token=_COMPLETION_A)
    assert hooks[0].fired
    persisted = _raw_job(manager, 1)
    assert completed is False, {
        "status": persisted["status"] if persisted else None,
        "counters": _counters(manager) if persisted else None,
        "events": [row["event_type"] for row in _events(manager, 1)],
    }
    _assert_sqlite_busy(busy_codes)
    assert inserted == []
    assert _raw_job(manager, 1) is None

    created = _create_job(creator)
    assert created["id"] == 1
    assert _raw_job(manager, 1)["status"] == "queued"
    assert _counters(manager) == (1, 0, 0, 0)
    assert [row["event_type"] for row in _events(manager, 1)] == ["job.created"]


@pytest.mark.unit
def test_sqlite_existing_row_cannot_be_replaced_after_completion_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLite holds the original row identity through atomic completion."""
    db_path = tmp_path / "completion-replacement.db"
    manager = JobManager(db_path)
    original = _create_job(manager)
    job_id = int(original["id"])
    busy_codes: list[int] = []

    def replace_after_read() -> None:
        """Attempt same-ID replacement through an independent SQLite writer."""
        conn = _zero_timeout_connection(db_path)
        try:
            with conn:
                conn.execute("DELETE FROM job_events WHERE job_id=?", (job_id,))
                conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
                conn.execute(
                    "INSERT INTO jobs "
                    "(id, uuid, domain, queue, job_type, owner_user_id, payload, status, "
                    "priority, max_retries, retry_count, expired_lease_policy, "
                    "request_id, trace_id, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, '{}', 'queued', 5, 3, 0, "
                    "'consume_retry', ?, ?, DATETIME('now'), DATETIME('now'))",
                    (
                        job_id,
                        "replacement-uuid",
                        *_SCOPE,
                        "replacement-owner",
                        "replacement-request",
                        "replacement-trace",
                    ),
                )
        except sqlite3.OperationalError as exc:
            busy_codes.append(exc.sqlite_errorcode)
        finally:
            conn.close()

    hooks = _hook_completion_read(manager, monkeypatch, replace_after_read)
    assert manager.complete_job(job_id, result={"ok": True}, completion_token=_COMPLETION_A)
    assert hooks[0].fired
    persisted = _raw_job(manager, job_id)
    assert persisted["uuid"] == original["uuid"], persisted
    assert persisted["status"] == "completed"
    _assert_sqlite_busy(busy_codes)
    _assert_one_completion(manager, job_id)


@pytest.mark.pg_jobs
def test_postgres_missing_row_cannot_complete_concurrent_insert(
    jobs_pg_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PostgreSQL miss remains false even when a new row commits afterward."""
    pytest.importorskip("psycopg")
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    creator = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    inserted: list[dict[str, Any]] = []
    hooks = _hook_completion_read(manager, monkeypatch, lambda: inserted.append(_create_job(creator)))

    completed = manager.complete_job(1, result={"ok": True}, completion_token=_COMPLETION_A)

    assert hooks[0].fired
    assert len(inserted) == 1
    assert inserted[0]["id"] == 1
    assert completed is False, _raw_job(manager, 1)
    assert _raw_job(manager, 1)["status"] == "queued"
    assert _counters(manager) == (1, 0, 0, 0)
    assert [row["event_type"] for row in _events(manager, 1)] == ["job.created"]


@pytest.mark.pg_jobs
def test_postgres_completion_locks_processing_row_before_fetchone_callback(
    jobs_pg_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The authoritative PostgreSQL read locks the row before completion."""
    psycopg = pytest.importorskip("psycopg")
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    acquired = _create_job(manager, processing=True)
    job_id = int(acquired["id"])
    calls = _completion_observers(manager, monkeypatch)
    lock_errors: list[Any] = []

    def attempt_competing_lock() -> None:
        """Prove a second transaction cannot acquire the observed row lock."""
        conn = psycopg.connect(jobs_pg_dsn)
        try:
            with conn.cursor() as cursor:
                try:
                    cursor.execute("SELECT id FROM jobs WHERE id=%s FOR UPDATE NOWAIT", (job_id,))
                    cursor.fetchone()
                except psycopg.errors.LockNotAvailable as exc:
                    lock_errors.append(exc)
        finally:
            try:
                conn.rollback()
            finally:
                conn.close()

    hooks = _hook_completion_read(manager, monkeypatch, attempt_competing_lock)
    arguments = {
        "worker_id": str(acquired["worker_id"]),
        "lease_id": str(acquired["lease_id"]),
        "completion_token": _COMPLETION_A,
        "enforce": True,
    }
    assert manager.complete_job(job_id, result={"ok": True}, **arguments)
    assert hooks[0].fired
    assert hooks[0].row_locked
    assert len(lock_errors) == 1
    persisted = manager.get_job(job_id)
    assert persisted["uuid"] == acquired["uuid"]
    assert persisted["owner_user_id"] == acquired["owner_user_id"]
    assert persisted["status"] == "completed"
    assert persisted["completion_token"] == "token-a"
    assert persisted["result"] == {"ok": True}
    assert (persisted["worker_id"], persisted["lease_id"], persisted["leased_until"]) == (
        None,
        None,
        None,
    )
    after = _snapshot(manager, job_id)
    assert manager.complete_job(job_id, result={"ignored": True}, **arguments)
    assert _snapshot(manager, job_id) == after
    _assert_one_completion(manager, job_id, processing=True)
    assert all(len(observed) == 1 for observed in calls.values()), calls
    assert calls["event"][0][0] == "job.completed"


@pytest.mark.parametrize("processing", [False, True], ids=["queued", "processing"])
def test_expected_uuid_mismatch_has_no_mutation_or_side_effects(
    jm: JobManager, monkeypatch: pytest.MonkeyPatch, processing: bool
) -> None:
    """Reject stale UUIDs without changing queued or processing jobs."""
    job = _create_job(jm, processing=processing)
    job_id = int(job["id"])
    before = _snapshot(jm, job_id)
    calls = _completion_observers(jm, monkeypatch)

    assert (
        jm.complete_job(
            job_id,
            result={"ok": True},
            expected_uuid="stale-uuid",
            completion_token=_COMPLETION_A,
            worker_id=job.get("worker_id"),
            lease_id=job.get("lease_id"),
            enforce=processing,
        )
        is False
    )

    assert _snapshot(jm, job_id) == before
    assert not any(calls.values()), calls


def test_stale_uuid_cannot_replay_replacement_completion_token(jm: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    """A matching token cannot replay completion for a replacement UUID."""
    original = _create_job(jm)
    job_id = int(original["id"])
    _set_identity_and_token(jm, job_id, "replacement-uuid")
    assert jm.complete_job(job_id, result={"replacement": True}, completion_token=_COMPLETION_A)
    before = _snapshot(jm, job_id)
    calls = _completion_observers(jm, monkeypatch)

    assert (
        jm.complete_job(
            job_id,
            result={"stale": True},
            expected_uuid=original["uuid"],
            completion_token=_COMPLETION_A,
        )
        is False
    )

    assert _snapshot(jm, job_id) == before
    assert not any(calls.values()), calls
    _assert_one_completion(jm, job_id)


@pytest.mark.parametrize("raw_uuid", [None, "", "  NONcanonical-UUID  "], ids=["null", "empty", "noncanonical"])
def test_legacy_raw_uuid_is_preserved_and_same_token_replays_once(
    jm: JobManager, monkeypatch: pytest.MonkeyPatch, raw_uuid: str | None
) -> None:
    """Preserve raw legacy UUIDs and keep matching-token replay idempotent."""
    created = _create_job(jm)
    job_id = int(created["id"])
    _set_identity_and_token(jm, job_id, raw_uuid)
    calls = _completion_observers(jm, monkeypatch)
    external_uuid = "" if raw_uuid is None else raw_uuid

    assert jm.complete_job(
        job_id,
        result={"ok": True},
        expected_uuid=external_uuid,
        completion_token=_COMPLETION_A,
    )
    persisted = jm.get_job(job_id)
    assert _raw_job(jm, job_id)["uuid"] == raw_uuid
    assert persisted["status"] == "completed"
    assert persisted["result"] == {"ok": True}
    assert persisted["completion_token"] == "token-a"
    after = _snapshot(jm, job_id)

    assert jm.complete_job(
        job_id,
        result={"ignored": True},
        expected_uuid=external_uuid,
        completion_token=_COMPLETION_A,
    )
    assert _snapshot(jm, job_id) == after
    _assert_one_completion(jm, job_id)
    assert all(len(observed) == 1 for observed in calls.values()), calls


def test_queued_stored_token_rejects_different_token_then_transitions_once(
    jm: JobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a conflicting queued token before accepting its matching token."""
    created = _create_job(jm)
    job_id = int(created["id"])
    _set_identity_and_token(jm, job_id, created["uuid"], _COMPLETION_A)
    before = _snapshot(jm, job_id)
    calls = _completion_observers(jm, monkeypatch)

    assert (
        jm.complete_job(
            job_id,
            result={"wrong": True},
            expected_uuid=created["uuid"],
            completion_token=_COMPLETION_B,
        )
        is False
    )
    assert _snapshot(jm, job_id) == before
    assert not any(calls.values()), calls

    assert jm.complete_job(
        job_id,
        result={"ok": True},
        expected_uuid=created["uuid"],
        completion_token=_COMPLETION_A,
    )
    persisted = jm.get_job(job_id)
    assert persisted["status"] == "completed"
    assert persisted["result"] == {"ok": True}
    assert persisted["completion_token"] == "token-a"
    _assert_one_completion(jm, job_id)
    assert all(len(observed) == 1 for observed in calls.values()), calls


@pytest.mark.unit
@pytest.mark.parametrize("identity", ["missing", "stale"])
@pytest.mark.parametrize("preflight", ["token-required", "result-too-large", "invalid-limit"])
def test_sqlite_completion_preflight_precedes_identity_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: str,
    preflight: str,
) -> None:
    """Preflight errors retain precedence over missing or stale identity."""
    manager = JobManager(tmp_path / "completion-preflight.db")
    job_id = 1
    before = None
    if identity == "stale":
        created = _create_job(manager)
        job_id = int(created["id"])
        before = _snapshot(manager, job_id)
    arguments: dict[str, Any] = {
        "result": {"data": "x" * 300},
        "expected_uuid": "stale-uuid",
        "completion_token": _COMPLETION_A,
    }
    if preflight == "token-required":
        monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")
        monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "invalid")
        arguments["completion_token"] = None
        message = "completion_token required"
    elif preflight == "result-too-large":
        monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "128")
        message = "Result too large"
    else:
        monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "invalid")
        message = "invalid literal for int"
    calls = _completion_observers(manager, monkeypatch)

    with monkeypatch.context() as preflight_patch:
        preflight_patch.setattr(
            manager,
            "_connect",
            lambda: pytest.fail("preflight must run before connecting for identity lookup"),
        )
        with pytest.raises(ValueError, match=message):
            manager.complete_job(job_id, **arguments)

    assert not any(calls.values()), calls
    if before is not None:
        assert _snapshot(manager, job_id) == before
    else:
        assert _raw_job(manager, job_id) is None
        assert _events(manager, job_id) == []

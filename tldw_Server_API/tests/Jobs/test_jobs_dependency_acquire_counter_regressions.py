from __future__ import annotations

import ast
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.unit

_BACKENDS = [
    pytest.param("sqlite", id="sqlite"),
    pytest.param("postgres", marks=pytest.mark.pg_jobs, id="postgres"),
]

_ADMISSION_ADAPTERS = (
    "sqlite/admission.py",
    "postgres/admission.py",
)


@pytest.mark.parametrize("adapter", _ADMISSION_ADAPTERS)
def test_admission_adapter_remains_importable_on_python_3_10(adapter: str) -> None:
    """Guard against importing datetime.UTC, which was added in Python 3.11."""

    operations = Path(__file__).parents[2] / "app" / "core" / "Jobs" / "operations"
    module = ast.parse((operations / adapter).read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module == "datetime"
        for alias in node.names
    }

    assert "UTC" not in imported_names


class _RollbackInsteadOfCommit:
    """Rollback a real adapter transaction at its commit boundary."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _RollbackInsteadOfCommit:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if exc_type is not None:
            return self._inner.__exit__(exc_type, exc, tb)
        self._inner.rollback()
        raise RuntimeError("forced acquire commit failure")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailAcquireCounterSQLite:
    """Fail the queued-to-processing counter mutation at the adapter boundary."""

    def __init__(self, inner: sqlite3.Connection) -> None:
        self._inner = inner

    def __enter__(self) -> _FailAcquireCounterSQLite:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = ()) -> Any:
        normalized = " ".join(str(sql).split()).lower()
        if normalized.startswith("insert into job_counters"):
            raise sqlite3.OperationalError("forced acquire counter failure")
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _PauseSelectedCursorSQLite:
    """Pause a two-step acquisition after its candidate row is selected."""

    def __init__(
        self,
        inner: Any,
        *,
        selected: threading.Event,
        resume: threading.Event,
    ) -> None:
        self._inner = inner
        self._selected = selected
        self._resume = resume

    def fetchone(self) -> Any:
        row = self._inner.fetchone()
        if row is not None:
            self._selected.set()
            assert self._resume.wait(timeout=10)
        return row

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _PauseSelectedConnectionSQLite:
    """Wrap the eligible-job SELECT cursor used by two-step acquisition."""

    def __init__(
        self,
        inner: sqlite3.Connection,
        *,
        selected: threading.Event,
        resume: threading.Event,
    ) -> None:
        self._inner = inner
        self._selected = selected
        self._resume = resume

    def __enter__(self) -> _PauseSelectedConnectionSQLite:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = ()) -> Any:
        cursor = self._inner.execute(sql, params)
        normalized = " ".join(str(sql).split()).lower()
        if normalized.startswith("select id from jobs where domain = ?") and "status = 'queued'" in normalized:
            return _PauseSelectedCursorSQLite(
                cursor,
                selected=self._selected,
                resume=self._resume,
            )
        return cursor

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _NotifyBeginImmediateSQLite:
    """Signal immediately before a dependency writer requests its lock."""

    def __init__(self, inner: sqlite3.Connection, *, attempted: threading.Event) -> None:
        self._inner = inner
        self._attempted = attempted

    def __enter__(self) -> _NotifyBeginImmediateSQLite:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = ()) -> Any:
        if " ".join(str(sql).split()).lower() == "begin immediate":
            self._attempted.set()
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CursorProxy:
    """Delegate cursor operations while allowing narrow boundary overrides."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _CursorProxy:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ConnectionProxy:
    """Delegate connection operations while allowing narrow boundary overrides."""

    cursor_proxy = _CursorProxy

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _ConnectionProxy:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return self.cursor_proxy(self._inner.cursor(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _InflightCountBarrierCursor(_CursorProxy):
    """Hold both workers after their optimistic inflight preflight read."""

    def __init__(self, inner: Any, *, barrier: threading.Barrier) -> None:
        super().__init__(inner)
        self._barrier = barrier
        self._pause_next_fetch = False

    def execute(self, sql: Any, params: Any = ()) -> Any:
        normalized = " ".join(str(sql).split()).lower()
        self._pause_next_fetch = (
            normalized.startswith("select count(*)")
            and "status='processing'" in normalized
        )
        self._inner.execute(sql, params)
        return self

    def fetchone(self) -> Any:
        row = self._inner.fetchone()
        if self._pause_next_fetch:
            self._pause_next_fetch = False
            self._barrier.wait(timeout=10)
        return row


class _InflightCountBarrierConnection(_ConnectionProxy):
    """Pause an inflight-count read without changing adapter behavior."""

    def __init__(self, inner: Any, *, barrier: threading.Barrier) -> None:
        super().__init__(inner)
        self._barrier = barrier

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return _InflightCountBarrierCursor(
            self._inner.cursor(*args, **kwargs),
            barrier=self._barrier,
        )

    def execute(self, sql: Any, params: Any = ()) -> Any:
        cursor = self._inner.execute(sql, params)
        normalized = " ".join(str(sql).split()).lower()
        if normalized.startswith("select count(*)") and "status='processing'" in normalized:
            return _BarrierFetchCursor(cursor, barrier=self._barrier)
        return cursor


class _BarrierFetchCursor(_CursorProxy):
    """Synchronize once after returning a database row."""

    def __init__(self, inner: Any, *, barrier: threading.Barrier) -> None:
        super().__init__(inner)
        self._barrier = barrier

    def fetchone(self) -> Any:
        row = self._inner.fetchone()
        self._barrier.wait(timeout=10)
        return row


class _FailCounterCursor(_CursorProxy):
    """Raise a non-database adapter error for any counter mutation."""

    def execute(self, sql: Any, params: Any = ()) -> Any:
        normalized = " ".join(str(sql).split()).lower()
        if (
            normalized.startswith("insert into job_counters")
            or normalized.startswith("update job_counters")
        ):
            raise RuntimeError("forced jobs counter adapter failure")
        self._inner.execute(sql, params)
        return self


class _FailCounterConnection(_ConnectionProxy):
    """Fail counter mutations at either database adapter boundary."""

    cursor_proxy = _FailCounterCursor

    def execute(self, sql: Any, params: Any = ()) -> Any:
        normalized = " ".join(str(sql).split()).lower()
        if (
            normalized.startswith("insert into job_counters")
            or normalized.startswith("update job_counters")
        ):
            raise RuntimeError("forced jobs counter adapter failure")
        return self._inner.execute(sql, params)


class _PauseRescheduleCursor(_CursorProxy):
    """Pause immediately before a delayed reschedule mutation."""

    def __init__(
        self,
        inner: Any,
        *,
        attempted: threading.Event,
        resume: threading.Event,
    ) -> None:
        super().__init__(inner)
        self._attempted = attempted
        self._resume = resume

    def execute(self, sql: Any, params: Any = ()) -> Any:
        normalized = " ".join(str(sql).split()).lower()
        if "update jobs" in normalized and "set available_at" in normalized:
            self._attempted.set()
            assert self._resume.wait(timeout=10)
        self._inner.execute(sql, params)
        return self


class _PauseRescheduleConnection(_ConnectionProxy):
    """Expose the delayed-reschedule/acquire race deterministically."""

    def __init__(
        self,
        inner: Any,
        *,
        attempted: threading.Event,
        resume: threading.Event,
    ) -> None:
        super().__init__(inner)
        self._attempted = attempted
        self._resume = resume

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return _PauseRescheduleCursor(
            self._inner.cursor(*args, **kwargs),
            attempted=self._attempted,
            resume=self._resume,
        )

    def execute(self, sql: Any, params: Any = ()) -> Any:
        normalized = " ".join(str(sql).split()).lower()
        if "update jobs" in normalized and "set available_at" in normalized:
            self._attempted.set()
            assert self._resume.wait(timeout=10)
        return self._inner.execute(sql, params)


class _RejectPerRowRescheduleCursor(_CursorProxy):
    """Reject adapter queries that return one Python row per candidate job."""

    def execute(self, sql: Any, params: Any = ()) -> Any:
        normalized = " ".join(str(sql).split()).lower()
        if normalized.startswith("select id,domain,queue,job_type,status from jobs"):
            raise AssertionError("reschedule candidate rows must remain inside the database")
        self._inner.execute(sql, params)
        return self


class _RejectPerRowRescheduleConnection(_ConnectionProxy):
    """Enforce grouped reschedule results at both adapter boundaries."""

    cursor_proxy = _RejectPerRowRescheduleCursor

    def execute(self, sql: Any, params: Any = ()) -> Any:
        normalized = " ".join(str(sql).split()).lower()
        if normalized.startswith("select id,domain,queue,job_type,status from jobs"):
            raise AssertionError("reschedule candidate rows must remain inside the database")
        return self._inner.execute(sql, params)


def _manager(
    backend: str,
    *,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    name: str,
) -> JobManager:
    if backend == "postgres":
        dsn = str(request.getfixturevalue("jobs_pg_dsn"))
        return JobManager(None, backend="postgres", db_url=dsn)
    return JobManager(tmp_path / f"{name}.db")


def _configure_acquire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: str,
    single_update: bool,
) -> None:
    flag = "JOBS_PG_SINGLE_UPDATE_ACQUIRE" if backend == "postgres" else "JOBS_SQLITE_SINGLE_UPDATE_ACQUIRE"
    if single_update:
        monkeypatch.setenv(flag, "true")
    else:
        monkeypatch.delenv(flag, raising=False)


def _counter(manager: JobManager, *, domain: str, job_type: str) -> tuple[int, int, int]:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
                    "WHERE domain=%s AND queue='default' AND job_type=%s",
                    (domain, job_type),
                )
                row = cur.fetchone()
                assert row is not None
                return (
                    int(row["ready_count"]),
                    int(row["scheduled_count"]),
                    int(row["processing_count"]),
                )
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
            "WHERE domain=? AND queue='default' AND job_type=?",
            (domain, job_type),
        ).fetchone()
        assert row is not None
        return int(row[0]), int(row[1]), int(row[2])
    finally:
        conn.close()


def _counter_row_count(manager: JobManager, *, domain: str, job_type: str) -> int:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS count FROM job_counters "
                    "WHERE domain=%s AND queue='default' AND job_type=%s",
                    (domain, job_type),
                )
                return int(cur.fetchone()["count"])
        row = conn.execute(
            "SELECT COUNT(*) FROM job_counters "
            "WHERE domain=? AND queue='default' AND job_type=?",
            (domain, job_type),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _delete_counter(manager: JobManager, *, domain: str, job_type: str) -> None:
    """Remove one aggregate row to exercise counter reconstruction paths."""

    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with conn, manager._pg_cursor(conn) as cur:
                cur.execute(
                    "DELETE FROM job_counters WHERE domain=%s AND queue='default' AND job_type=%s",
                    (domain, job_type),
                )
        else:
            with conn:
                conn.execute(
                    "DELETE FROM job_counters WHERE domain=? AND queue='default' AND job_type=?",
                    (domain, job_type),
                )
    finally:
        conn.close()


def _set_available_at(manager: JobManager, job_id: int, *, future: bool) -> None:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with conn, manager._pg_cursor(conn) as cur:
                direction = "+" if future else "-"
                cur.execute(
                    f"UPDATE jobs SET available_at=NOW() {direction} INTERVAL '1 day' WHERE id=%s",  # nosec B608
                    (job_id,),
                )
        else:
            with conn:
                modifier = "+1 day" if future else "-1 day"
                conn.execute(
                    "UPDATE jobs SET available_at=DATETIME('now', ?) WHERE id=?",
                    (modifier, job_id),
                )
    finally:
        conn.close()


def _expire_lease(manager: JobManager, job_id: int) -> None:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with conn, manager._pg_cursor(conn) as cur:
                cur.execute(
                    "UPDATE jobs SET leased_until=NOW() - INTERVAL '1 day' WHERE id=%s",
                    (job_id,),
                )
        else:
            with conn:
                conn.execute(
                    "UPDATE jobs SET leased_until=DATETIME('now', '-1 day') WHERE id=?",
                    (job_id,),
                )
    finally:
        conn.close()


def _dependency_count(manager: JobManager, child_uuid: str) -> int:
    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS count FROM job_dependencies WHERE job_uuid=%s",
                    (child_uuid,),
                )
                row = cur.fetchone()
                return int(row["count"])
        row = conn.execute(
            "SELECT COUNT(*) FROM job_dependencies WHERE job_uuid=?",
            (child_uuid,),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


def _job_and_event_counts(manager: JobManager, *, domain: str) -> tuple[int, int]:
    """Return persisted job and lifecycle-event counts for one isolated domain."""

    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS count FROM jobs WHERE domain=%s",
                    (domain,),
                )
                jobs = int(cur.fetchone()["count"])
                cur.execute(
                    "SELECT COUNT(*) AS count FROM job_events WHERE domain=%s",
                    (domain,),
                )
                events = int(cur.fetchone()["count"])
                return jobs, events
        jobs_row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE domain=?",
            (domain,),
        ).fetchone()
        events_row = conn.execute(
            "SELECT COUNT(*) FROM job_events WHERE domain=?",
            (domain,),
        ).fetchone()
        return int(jobs_row[0]), int(events_row[0])
    finally:
        conn.close()


def _set_completion_token(manager: JobManager, job_id: int, token: str) -> None:
    """Seed an earlier attempt token at the adapter boundary."""

    conn = manager._connect()
    try:
        if manager.backend == "postgres":
            with conn, manager._pg_cursor(conn) as cur:
                cur.execute(
                    "UPDATE jobs SET completion_token=%s WHERE id=%s",
                    (token, job_id),
                )
        else:
            with conn:
                conn.execute(
                    "UPDATE jobs SET completion_token=? WHERE id=?",
                    (token, job_id),
                )
    finally:
        conn.close()


def _disable_acquire_preflight(manager: JobManager, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager, "_get_queue_flags", lambda *_args: {"paused": False, "drain": False})
    monkeypatch.setattr(manager, "_reconcile_terminal_dependents", lambda **_kwargs: 0)
    monkeypatch.setattr(manager, "_recover_expired_processing_jobs", lambda **_kwargs: 0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_dependency_cannot_be_added_after_child_is_acquired(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dependency insert must re-check child state in its write transaction."""

    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    add_manager = _manager(backend, request=request, tmp_path=tmp_path, name="dependency-acquire")
    acquire_manager = _manager(backend, request=request, tmp_path=tmp_path, name="dependency-acquire")
    domain = f"dependency-acquire-{backend}-{single_update}"
    parent = add_manager.create_job(
        domain=domain,
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="owner",
    )
    child = add_manager.create_job(
        domain=domain,
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="owner",
    )

    prechecks_complete = threading.Event()
    allow_transaction = threading.Event()
    original_connect = add_manager._connect
    connection_calls = 0

    def pause_before_dependency_transaction() -> Any:
        nonlocal connection_calls
        connection_calls += 1
        if connection_calls == 3:
            prechecks_complete.set()
            assert allow_transaction.wait(timeout=10)
        return original_connect()

    monkeypatch.setattr(add_manager, "_connect", pause_before_dependency_transaction)
    with ThreadPoolExecutor(max_workers=1) as pool:
        added = pool.submit(
            add_manager.add_job_dependency,
            str(child["uuid"]),
            str(parent["uuid"]),
        )
        assert prechecks_complete.wait(timeout=10)
        acquired = acquire_manager.acquire_next_job(
            domain=domain,
            queue="default",
            job_type="child",
            lease_seconds=30,
            worker_id="dependency-race-worker",
        )
        assert acquired is not None
        assert int(acquired["id"]) == int(child["id"])
        allow_transaction.set()
        assert added.result(timeout=10) is False

    monkeypatch.setattr(add_manager, "_connect", original_connect)
    assert _dependency_count(add_manager, str(child["uuid"])) == 0


def test_sqlite_two_step_acquire_serializes_selection_with_dependency_insert(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dependency writer cannot commit between candidate selection and transition."""

    _configure_acquire(monkeypatch, backend="sqlite", single_update=False)
    add_manager = _manager("sqlite", request=request, tmp_path=tmp_path, name="dependency-overlap")
    acquire_manager = _manager("sqlite", request=request, tmp_path=tmp_path, name="dependency-overlap")
    _disable_acquire_preflight(acquire_manager, monkeypatch)
    domain = "dependency-overlap-sqlite"
    parent = add_manager.create_job(
        domain=domain,
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="owner",
    )
    child = add_manager.create_job(
        domain=domain,
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="owner",
    )

    selected = threading.Event()
    resume_acquire = threading.Event()
    dependency_lock_attempted = threading.Event()
    original_acquire_connect = acquire_manager._connect
    original_add_connect = add_manager._connect
    add_connection_calls = 0

    def pause_acquire_after_selection() -> Any:
        return _PauseSelectedConnectionSQLite(
            original_acquire_connect(),
            selected=selected,
            resume=resume_acquire,
        )

    def notify_dependency_lock_attempt() -> Any:
        nonlocal add_connection_calls
        add_connection_calls += 1
        conn = original_add_connect()
        if add_connection_calls == 3:
            return _NotifyBeginImmediateSQLite(conn, attempted=dependency_lock_attempted)
        return conn

    monkeypatch.setattr(acquire_manager, "_connect", pause_acquire_after_selection)
    monkeypatch.setattr(add_manager, "_connect", notify_dependency_lock_attempt)
    with ThreadPoolExecutor(max_workers=2) as pool:
        acquired_future = pool.submit(
            acquire_manager.acquire_next_job,
            domain=domain,
            queue="default",
            job_type="child",
            lease_seconds=30,
            worker_id="dependency-overlap-worker",
        )
        assert selected.wait(timeout=10)
        added_future = pool.submit(
            add_manager.add_job_dependency,
            str(child["uuid"]),
            str(parent["uuid"]),
        )
        assert dependency_lock_attempted.wait(timeout=10)
        resume_acquire.set()
        acquired = acquired_future.result(timeout=10)
        added = added_future.result(timeout=10)

    assert acquired is not None
    assert int(acquired["id"]) == int(child["id"])
    assert added is False
    assert _dependency_count(add_manager, str(child["uuid"])) == 0


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_acquire_commit_failure_rolls_back_state_counters_and_observers(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queued transition, counter mutation, and observers share one commit boundary."""

    import tldw_Server_API.app.core.Jobs.manager as manager_module

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="acquire-rollback")
    domain = f"acquire-rollback-{backend}-{single_update}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    observed: list[str] = []
    monkeypatch.setattr(manager_module, "emit_job_event", lambda *_args, **_kwargs: observed.append("event"))
    monkeypatch.setattr(
        manager_module,
        "observe_queue_latency",
        lambda *_args, **_kwargs: observed.append("latency"),
    )
    monkeypatch.setattr(manager, "_update_gauges", lambda **_kwargs: observed.append("gauge"))
    _disable_acquire_preflight(manager, monkeypatch)
    original_connect = manager._connect
    monkeypatch.setattr(manager, "_connect", lambda: _RollbackInsteadOfCommit(original_connect()))

    with pytest.raises(RuntimeError, match="forced acquire commit failure"):
        manager.acquire_next_job(
            domain=domain,
            queue="default",
            job_type="work",
            lease_seconds=30,
            worker_id="rollback-worker",
        )

    monkeypatch.setattr(manager, "_connect", original_connect)
    assert manager.get_job(int(job["id"]))["status"] == "queued"
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 0)
    assert observed == []


@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_sqlite_acquire_counter_failure_rolls_back_transition(
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed counter write cannot leave a processing row with queued counters."""

    import tldw_Server_API.app.core.Jobs.manager as manager_module

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _configure_acquire(monkeypatch, backend="sqlite", single_update=single_update)
    manager = _manager("sqlite", request=request, tmp_path=tmp_path, name="acquire-counter-failure")
    domain = f"acquire-counter-failure-{single_update}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    observed: list[str] = []
    monkeypatch.setattr(manager_module, "emit_job_event", lambda *_args, **_kwargs: observed.append("event"))
    monkeypatch.setattr(manager, "_update_gauges", lambda **_kwargs: observed.append("gauge"))
    _disable_acquire_preflight(manager, monkeypatch)
    original_connect = manager._connect
    monkeypatch.setattr(manager, "_connect", lambda: _FailAcquireCounterSQLite(original_connect()))

    with pytest.raises(sqlite3.OperationalError, match="forced acquire counter failure"):
        manager.acquire_next_job(
            domain=domain,
            queue="default",
            job_type="work",
            lease_seconds=30,
            worker_id="counter-failure-worker",
        )

    monkeypatch.setattr(manager, "_connect", original_connect)
    assert manager.get_job(int(job["id"]))["status"] == "queued"
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 0)
    assert observed == []


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_naturally_due_scheduled_job_decrements_stored_scheduled_bucket(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="naturally-due")
    domain = f"naturally-due-{backend}-{single_update}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
    )
    _set_available_at(manager, int(job["id"]), future=False)

    assert _counter(manager, domain=domain, job_type="work") == (0, 1, 0)
    stats = manager.get_queue_stats(domain=domain, queue="default", job_type="work")
    assert stats[0]["queued"] == 0
    assert stats[0]["scheduled"] == 1
    assert manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="due-worker",
    ) is not None
    assert _counter(manager, domain=domain, job_type="work") == (0, 0, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_past_admission_is_normalized_to_ready_null_timestamp(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="past-admission")
    domain = f"past-admission-{backend}-{single_update}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        available_at=datetime.now(tz=timezone.utc) - timedelta(days=1),
    )

    assert manager.get_job(int(job["id"]))["available_at"] is None
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 0)
    assert manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="past-worker",
    ) is not None
    assert _counter(manager, domain=domain, job_type="work") == (0, 0, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_set_now_normalizes_scheduled_job_to_ready_null_timestamp(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="set-now")
    domain = f"set-now-{backend}-{single_update}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
    )

    assert manager.reschedule_jobs(
        domain=domain,
        queue="default",
        job_type="work",
        status="queued",
        set_now=True,
    ) == 1
    assert manager.get_job(int(job["id"]))["available_at"] is None
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 0)
    assert manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="set-now-worker",
    ) is not None
    assert _counter(manager, domain=domain, job_type="work") == (0, 0, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_retry_now_normalizes_scheduled_job_to_ready_null_timestamp(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="retry-now")
    domain = f"retry-now-{backend}-{single_update}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
    )

    assert manager.retry_now_jobs(
        domain=domain,
        queue="default",
        job_type="work",
        only_failed=False,
    ) == 1
    assert manager.get_job(int(job["id"]))["available_at"] is None
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 0)
    assert manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="retry-now-worker",
    ) is not None
    assert _counter(manager, domain=domain, job_type="work") == (0, 0, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_release_normalizes_due_scheduled_job_to_ready_null_timestamp(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="release")
    domain = f"release-{backend}-{single_update}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
    )
    _set_available_at(manager, int(job["id"]), future=False)
    acquired = manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="release-worker",
    )
    assert acquired is not None

    assert manager.release_job(int(job["id"]), enforce=False)
    assert manager.get_job(int(job["id"]))["available_at"] is None
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_expired_recovery_normalizes_immediate_retry_to_ready_null_timestamp(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="recovery")
    domain = f"recovery-{backend}-{single_update}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        max_retries=1,
    )
    assert manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="recovery-worker",
    ) is not None
    _expire_lease(manager, int(job["id"]))

    assert manager._recover_expired_processing_jobs(
        domain=domain,
        queue="default",
        job_type="work",
    ) == 1
    persisted = manager.get_job(int(job["id"]))
    assert persisted["status"] == "queued"
    assert persisted["available_at"] is None
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_concurrent_acquire_rechecks_max_inflight_inside_write_transaction(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two optimistic preflight reads cannot both consume a one-job quota."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    first = _manager(backend, request=request, tmp_path=tmp_path, name="max-inflight")
    second = _manager(backend, request=request, tmp_path=tmp_path, name="max-inflight")
    _disable_acquire_preflight(first, monkeypatch)
    _disable_acquire_preflight(second, monkeypatch)
    domain = f"max-inflight-{backend}-{single_update}"
    for _ in range(2):
        first.create_job(
            domain=domain,
            queue="default",
            job_type="work",
            payload={},
            owner_user_id="owner",
        )

    def quota(key: str, *_args: Any) -> int:
        return 1 if key == "JOBS_QUOTA_MAX_INFLIGHT" else 0

    monkeypatch.setattr(first, "_quota_get", quota)
    monkeypatch.setattr(second, "_quota_get", quota)
    barrier = threading.Barrier(2)

    def barrier_connect(original_connect: Any) -> Any:
        connection_calls = 0

        def connect_with_preflight_barrier() -> Any:
            nonlocal connection_calls
            connection_calls += 1
            conn = original_connect()
            if connection_calls == 1:
                return _InflightCountBarrierConnection(conn, barrier=barrier)
            return conn

        return connect_with_preflight_barrier

    for manager in (first, second):
        monkeypatch.setattr(manager, "_connect", barrier_connect(manager._connect))

    def acquire(manager: JobManager, worker_id: str) -> dict[str, Any] | None:
        return manager.acquire_next_job(
            domain=domain,
            queue="default",
            job_type="work",
            lease_seconds=30,
            worker_id=worker_id,
            owner_user_id="owner",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            pool.submit(acquire, first, "quota-worker-1"),
            pool.submit(acquire, second, "quota-worker-2"),
        ]
        acquired = [future.result(timeout=15) for future in results]

    assert sum(job is not None for job in acquired) == 1
    assert _counter(first, domain=domain, job_type="work") == (1, 0, 1)


@pytest.mark.parametrize("idempotency_key", [None, "same"], ids=["plain", "idempotent"])
def test_sqlite_admission_counter_failure_rolls_back_job_and_created_event(
    idempotency_key: str | None,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite admission rolls back when its transaction-critical counter fails."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager("sqlite", request=request, tmp_path=tmp_path, name="admission-counter")
    domain = "admission-counter-failure-sqlite"
    import tldw_Server_API.app.core.Jobs.operations.sqlite.admission as adapter

    def fail_counter(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("forced admission counter failure")

    monkeypatch.setattr(adapter, "_bump_counters", fail_counter)
    with pytest.raises(RuntimeError, match="forced admission counter failure"):
        manager.create_job(
            domain=domain,
            queue="default",
            job_type="work",
            payload={},
            owner_user_id="owner",
            idempotency_key=idempotency_key,
        )

    assert _job_and_event_counts(manager, domain=domain) == (0, 0)


@pytest.mark.pg_jobs
@pytest.mark.parametrize("idempotency_key", [None, "same"], ids=["plain", "idempotent"])
def test_postgres_admission_counter_failure_keeps_job_and_created_event(
    idempotency_key: str | None,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PostgreSQL isolates its optional counter failure from durable admission."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager("postgres", request=request, tmp_path=tmp_path, name="admission-counter")
    domain = "admission-counter-failure-postgres"
    import tldw_Server_API.app.core.Jobs.operations.postgres.admission as adapter

    def fail_counter(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("forced admission counter failure")

    monkeypatch.setattr(adapter, "_bump_counters", fail_counter)
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        idempotency_key=idempotency_key,
    )

    assert job["domain"] == domain
    assert _job_and_event_counts(manager, domain=domain) == (1, 1)
    assert _counter_row_count(manager, domain=domain, job_type="work") == 0


@pytest.mark.parametrize("backend", _BACKENDS)
def test_release_counter_failure_rolls_back_transition_and_observers(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release cannot commit when its processing-to-ready counter write fails."""

    import tldw_Server_API.app.core.Jobs.manager as manager_module

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="release-counter")
    domain = f"release-counter-failure-{backend}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    acquired = manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="release-worker",
    )
    assert acquired is not None
    observed: list[str] = []
    monkeypatch.setattr(manager, "_update_gauges", lambda **_kwargs: observed.append("gauge"))
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: observed.append("event"),
    )
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _FailCounterConnection(original_connect()),
    )

    with pytest.raises(RuntimeError, match="forced jobs counter adapter failure"):
        manager.release_job(
            int(job["id"]),
            worker_id="release-worker",
            lease_id=str(acquired["lease_id"]),
            reason="worker shutdown",
            enforce=True,
        )

    monkeypatch.setattr(manager, "_connect", original_connect)
    assert manager.get_job(int(job["id"]))["status"] == "processing"
    assert _counter(manager, domain=domain, job_type="work") == (0, 0, 1)
    assert observed == []


@pytest.mark.parametrize("backend", _BACKENDS)
def test_release_commit_failure_suppresses_event_and_gauge(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release observers run only after the owning transaction commits."""

    import tldw_Server_API.app.core.Jobs.manager as manager_module

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="release-commit")
    domain = f"release-commit-failure-{backend}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    acquired = manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="release-worker",
    )
    assert acquired is not None
    observed: list[str] = []
    monkeypatch.setattr(manager, "_update_gauges", lambda **_kwargs: observed.append("gauge"))
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: observed.append("event"),
    )
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _RollbackInsteadOfCommit(original_connect()),
    )

    with pytest.raises(RuntimeError, match="forced acquire commit failure"):
        manager.release_job(
            int(job["id"]),
            worker_id="release-worker",
            lease_id=str(acquired["lease_id"]),
            reason="worker shutdown",
            enforce=True,
        )

    monkeypatch.setattr(manager, "_connect", original_connect)
    assert manager.get_job(int(job["id"]))["status"] == "processing"
    assert _counter(manager, domain=domain, job_type="work") == (0, 0, 1)
    assert observed == []


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("single_update", [False, True], ids=["two-step", "single-update"])
def test_acquire_clears_stale_completion_token_for_new_attempt(
    backend: str,
    single_update: bool,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every queued-to-processing transition starts with a fresh attempt token."""

    _configure_acquire(monkeypatch, backend=backend, single_update=single_update)
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="acquire-token")
    domain = f"acquire-token-{backend}-{single_update}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    _set_completion_token(manager, int(job["id"]), "previous-attempt-token")

    acquired = manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="token-worker",
    )

    assert acquired is not None
    assert acquired["completion_token"] is None
    assert manager.get_job(int(job["id"]))["completion_token"] is None


@pytest.mark.parametrize("backend", _BACKENDS)
def test_release_clears_stale_completion_token_before_requeue(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A released job cannot carry an earlier attempt's finalize token."""

    manager = _manager(backend, request=request, tmp_path=tmp_path, name="release-token")
    domain = f"release-token-{backend}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    acquired = manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="release-token-worker",
    )
    assert acquired is not None
    _set_completion_token(manager, int(job["id"]), "previous-attempt-token")

    assert manager.release_job(
        int(job["id"]),
        worker_id="release-token-worker",
        lease_id=str(acquired["lease_id"]),
        enforce=True,
    )
    persisted = manager.get_job(int(job["id"]))
    assert persisted["status"] == "queued"
    assert persisted["completion_token"] is None


@pytest.mark.parametrize("backend", _BACKENDS)
def test_expired_recovery_clears_stale_completion_token_before_requeue(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lease recovery creates a fresh queued attempt with no finalize token."""

    manager = _manager(backend, request=request, tmp_path=tmp_path, name="recovery-token")
    domain = f"recovery-token-{backend}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        max_retries=1,
    )
    acquired = manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="recovery-token-worker",
    )
    assert acquired is not None
    _set_completion_token(manager, int(job["id"]), "previous-attempt-token")
    _expire_lease(manager, int(job["id"]))

    assert manager._recover_expired_processing_jobs(
        domain=domain,
        queue="default",
        job_type="work",
    ) == 1
    persisted = manager.get_job(int(job["id"]))
    assert persisted["status"] == "queued"
    assert persisted["completion_token"] is None


@pytest.mark.parametrize("backend", _BACKENDS)
def test_delayed_reschedule_moves_ready_counter_to_scheduled(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NULL-to-non-NULL scheduling moves the durable bucket exactly once."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="delay-counter")
    domain = f"delay-counter-{backend}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )

    assert manager.reschedule_jobs(
        domain=domain,
        queue="default",
        job_type="work",
        status="queued",
        set_now=False,
        delta_seconds=60,
    ) == 1
    persisted = manager.get_job(int(job["id"]))
    assert persisted["available_at"] is not None
    assert _counter(manager, domain=domain, job_type="work") == (0, 1, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_delayed_reschedule_counter_failure_rolls_back_timestamp(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The delayed timestamp and its counter movement share one transaction."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="delay-rollback")
    domain = f"delay-counter-rollback-{backend}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _FailCounterConnection(original_connect()),
    )

    with pytest.raises(RuntimeError, match="forced jobs counter adapter failure"):
        manager.reschedule_jobs(
            domain=domain,
            queue="default",
            job_type="work",
            status="queued",
            set_now=False,
            delta_seconds=60,
        )

    monkeypatch.setattr(manager, "_connect", original_connect)
    assert manager.get_job(int(job["id"]))["available_at"] is None
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_delayed_reschedule_and_acquire_count_only_the_winning_transition(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent acquire cannot leave a phantom scheduled counter delta."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    rescheduler = _manager(backend, request=request, tmp_path=tmp_path, name="delay-race")
    acquirer = _manager(backend, request=request, tmp_path=tmp_path, name="delay-race")
    _disable_acquire_preflight(acquirer, monkeypatch)
    domain = f"delay-acquire-race-{backend}"
    job = rescheduler.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    update_attempted = threading.Event()
    resume_update = threading.Event()
    original_connect = rescheduler._connect
    monkeypatch.setattr(
        rescheduler,
        "_connect",
        lambda: _PauseRescheduleConnection(
            original_connect(),
            attempted=update_attempted,
            resume=resume_update,
        ),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        moved_future = pool.submit(
            rescheduler.reschedule_jobs,
            domain=domain,
            queue="default",
            job_type="work",
            status="queued",
            set_now=False,
            delta_seconds=60,
        )
        assert update_attempted.wait(timeout=10)
        acquired_future = pool.submit(
            acquirer.acquire_next_job,
            domain=domain,
            queue="default",
            job_type="work",
            lease_seconds=30,
            worker_id="delay-race-worker",
        )
        resume_update.set()
        moved = moved_future.result(timeout=15)
        acquired = acquired_future.result(timeout=15)

    monkeypatch.setattr(rescheduler, "_connect", original_connect)
    persisted = rescheduler.get_job(int(job["id"]))
    if moved == 1:
        assert acquired is None
        assert persisted["status"] == "queued"
        assert persisted["available_at"] is not None
        assert _counter(rescheduler, domain=domain, job_type="work") == (0, 1, 0)
    else:
        assert moved == 0
        assert acquired is not None
        assert persisted["status"] == "processing"
        assert _counter(rescheduler, domain=domain, job_type="work") == (0, 0, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_concurrent_set_now_moves_scheduled_counter_once_and_returns_actual_winner(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate set-now calls cannot both claim the same scheduled transition."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    first = _manager(backend, request=request, tmp_path=tmp_path, name="set-now-race")
    second = _manager(backend, request=request, tmp_path=tmp_path, name="set-now-race")
    domain = f"set-now-race-{backend}"
    job = first.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
    )
    update_attempted = threading.Event()
    resume_update = threading.Event()
    second_started = threading.Event()
    original_connect = first._connect
    monkeypatch.setattr(
        first,
        "_connect",
        lambda: _PauseRescheduleConnection(
            original_connect(),
            attempted=update_attempted,
            resume=resume_update,
        ),
    )

    def set_now(manager: JobManager, *, started: threading.Event | None = None) -> int:
        if started is not None:
            started.set()
        return manager.reschedule_jobs(
            domain=domain,
            queue="default",
            job_type="work",
            status="queued",
            set_now=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(set_now, first)
        assert update_attempted.wait(timeout=10)
        second_future = pool.submit(set_now, second, started=second_started)
        assert second_started.wait(timeout=10)
        resume_update.set()
        results = [
            first_future.result(timeout=15),
            second_future.result(timeout=15),
        ]

    monkeypatch.setattr(first, "_connect", original_connect)
    assert sorted(results) == [0, 1]
    assert first.get_job(int(job["id"]))["available_at"] is None
    assert _counter(first, domain=domain, job_type="work") == (1, 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_set_now_counter_failure_rolls_back_timestamp_and_suppresses_observers(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduled-to-ready counter write is part of the set-now transaction."""

    import tldw_Server_API.app.core.Jobs.manager as manager_module

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="set-now-rollback")
    domain = f"set-now-counter-rollback-{backend}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
    )
    before = manager.get_job(int(job["id"]))["available_at"]
    observed: list[str] = []
    monkeypatch.setattr(manager, "_update_gauges", lambda **_kwargs: observed.append("gauge"))
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: observed.append("event"),
    )
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _FailCounterConnection(original_connect()),
    )

    with pytest.raises(RuntimeError, match="forced jobs counter adapter failure"):
        manager.reschedule_jobs(
            domain=domain,
            queue="default",
            job_type="work",
            status="queued",
            set_now=True,
        )

    monkeypatch.setattr(manager, "_connect", original_connect)
    assert manager.get_job(int(job["id"]))["available_at"] == before
    assert _counter(manager, domain=domain, job_type="work") == (0, 1, 0)
    assert observed == []


@pytest.mark.parametrize("backend", _BACKENDS)
def test_set_now_recreates_missing_counter_row(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing aggregate row is rebuilt during scheduled-to-ready movement."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="set-now-upsert")
    domain = f"set-now-counter-upsert-{backend}"
    job = manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
        available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
    )
    manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    sibling = manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="set-now-sibling-worker",
    )
    assert sibling is not None
    _delete_counter(manager, domain=domain, job_type="work")

    assert manager.reschedule_jobs(
        domain=domain,
        queue="default",
        job_type="work",
        status="queued",
        set_now=True,
    ) == 1
    assert manager.get_job(int(job["id"]))["available_at"] is None
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_set_now_returns_zero_when_no_timestamp_changes(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Already-ready rows are not reported as set-now transitions."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="set-now-rowcount")
    domain = f"set-now-rowcount-{backend}"
    manager.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )

    assert manager.reschedule_jobs(
        domain=domain,
        queue="default",
        job_type="work",
        status="queued",
        set_now=True,
    ) == 0
    assert _counter(manager, domain=domain, job_type="work") == (1, 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_delayed_reschedule_recreates_exact_missing_counter_row(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delayed scheduling reconciles all buckets when its counter row is absent."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="delay-upsert")
    domain = f"delay-counter-upsert-{backend}"
    for _ in range(2):
        manager.create_job(
            domain=domain,
            queue="default",
            job_type="work",
            payload={},
            owner_user_id="owner",
        )
    sibling = manager.acquire_next_job(
        domain=domain,
        queue="default",
        job_type="work",
        lease_seconds=30,
        worker_id="delay-sibling-worker",
    )
    assert sibling is not None
    _delete_counter(manager, domain=domain, job_type="work")

    assert manager.reschedule_jobs(
        domain=domain,
        queue="default",
        job_type="work",
        status="queued",
        set_now=False,
        delta_seconds=60,
    ) == 1
    assert _counter(manager, domain=domain, job_type="work") == (0, 1, 1)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_reschedule_candidates_are_grouped_inside_each_adapter(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither reschedule mode may materialize one Python object per matched job."""

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    manager = _manager(backend, request=request, tmp_path=tmp_path, name="grouped-reschedule")
    domain = f"grouped-reschedule-{backend}"
    manager.create_job(
        domain=domain,
        queue="default",
        job_type="set-now",
        payload={},
        owner_user_id="owner",
        available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
    )
    manager.create_job(
        domain=domain,
        queue="default",
        job_type="delay",
        payload={},
        owner_user_id="owner",
    )
    original_connect = manager._connect
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: _RejectPerRowRescheduleConnection(original_connect()),
    )

    assert manager.reschedule_jobs(
        domain=domain,
        queue="default",
        job_type="set-now",
        status="queued",
        set_now=True,
    ) == 1
    assert manager.reschedule_jobs(
        domain=domain,
        queue="default",
        job_type="delay",
        status="queued",
        set_now=False,
        delta_seconds=60,
    ) == 1

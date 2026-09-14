from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


class _RollbackInsteadOfCommit:
    """Replace a successful transaction commit with a rollback and error."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __enter__(self) -> _RollbackInsteadOfCommit:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if exc_type is not None:
            return self._inner.__exit__(exc_type, exc, tb)
        self._inner.rollback()
        raise RuntimeError("forced commit failure")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _MetricRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def increment(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((*args, kwargs))


class _SQLiteSweepBarrier:
    """Synchronize either the new write claim or the legacy candidate read."""

    def __init__(self, inner: Any, barrier: threading.Barrier) -> None:
        self._inner = inner
        self._barrier = barrier
        self._waited = False

    def execute(self, sql: str, params: Any = ()) -> Any:
        normalized = " ".join(str(sql).split())
        is_claim = normalized == "BEGIN IMMEDIATE"
        is_legacy_read = normalized.startswith("SELECT domain, queue, job_type, COUNT(*)") and (
            "FROM jobs WHERE status='queued'" in normalized or "FROM jobs WHERE status='processing'" in normalized
        )
        if not self._waited and (is_claim or is_legacy_read):
            self._waited = True
            self._barrier.wait(timeout=10)
        return self._inner.execute(sql, params)

    def __enter__(self) -> _SQLiteSweepBarrier:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _AdvancingSQLitePruneClock:
    """Expose SQLite's statement-scoped ``'now'`` behavior deterministically."""

    def __init__(
        self,
        inner: Any,
        *,
        count_time: str,
        mutation_time: str,
        status_param_count: int,
    ) -> None:
        self._inner = inner
        self._count_time = count_time
        self._mutation_time = mutation_time
        self._status_param_count = status_param_count

    def execute(self, sql: str, params: Any = ()) -> Any:
        query = str(sql)
        if "julianday('now', ?)" not in query:
            return self._inner.execute(sql, params)

        normalized = " ".join(query.split())
        reference_time = (
            self._count_time
            if normalized.startswith("SELECT COUNT(*) FROM jobs")
            else self._mutation_time
        )
        rewritten = query.replace("julianday('now', ?)", "julianday(?, ?)")
        rewritten_params = list(params)
        rewritten_params.insert(self._status_param_count, reference_time)
        return self._inner.execute(rewritten, tuple(rewritten_params))

    def __enter__(self) -> _AdvancingSQLitePruneClock:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _PostgresSweepBarrierCursor:
    """Synchronize either the new update claim or legacy candidate read."""

    def __init__(self, inner: Any, barrier: threading.Barrier) -> None:
        self._inner = inner
        self._barrier = barrier
        self._waited = False

    def execute(self, sql: Any, params: Any = None) -> Any:
        normalized = " ".join(str(sql).split())
        is_claim = normalized.startswith(
            ("UPDATE jobs SET", "WITH changed AS (UPDATE jobs SET")
        ) and (
            "WHERE status='queued'" in normalized or "WHERE status='processing'" in normalized
        )
        is_legacy_read = normalized.startswith("SELECT domain, queue, job_type, COUNT(*)") and (
            "FROM jobs WHERE status='queued'" in normalized or "FROM jobs WHERE status='processing'" in normalized
        )
        if not self._waited and (is_claim or is_legacy_read):
            self._waited = True
            self._barrier.wait(timeout=10)
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailCounterWriteCursor:
    """Delegate a cursor while injecting one durable-counter write failure."""

    def __init__(self, inner: Any, before_execute: Any) -> None:
        self._inner = inner
        self._before_execute = before_execute

    def __enter__(self) -> _FailCounterWriteCursor:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        self._before_execute(sql)
        if params is None:
            return self._inner.execute(sql)
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _FailCounterWriteConnection:
    """Delegate a connection while injecting one counter adapter failure."""

    def __init__(self, inner: Any, error: Exception) -> None:
        self._inner = inner
        self._error = error
        self._failed = False

    def _before_execute(self, sql: Any) -> None:
        normalized = " ".join(str(sql).lower().split())
        is_counter_write = "job_counters" in normalized and normalized.startswith(("insert", "update"))
        if self._failed or not is_counter_write:
            return
        self._failed = True
        raise self._error

    def execute(self, sql: Any, params: Any = ()) -> Any:
        self._before_execute(sql)
        return self._inner.execute(sql, params)

    def cursor(self, *args: Any, **kwargs: Any) -> _FailCounterWriteCursor:
        return _FailCounterWriteCursor(
            self._inner.cursor(*args, **kwargs),
            self._before_execute,
        )

    def __enter__(self) -> _FailCounterWriteConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _SQLTraceCursor:
    """Record adapter SQL and bounded prune batch fetch sizes."""

    def __init__(
        self,
        inner: Any,
        traces: list[str],
        batch_lengths: list[int] | None,
    ) -> None:
        self._inner = inner
        self._traces = traces
        self._batch_lengths = batch_lengths
        self._last_sql = ""

    def __enter__(self) -> _SQLTraceCursor:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        self._last_sql = " ".join(str(sql).split())
        self._traces.append(self._last_sql)
        if params is None:
            return self._inner.execute(sql)
        return self._inner.execute(sql, params)

    def fetchall(self) -> Any:
        return self._inner.fetchall()

    def fetchmany(self, size: int | None = None) -> Any:
        rows = (
            self._inner.fetchmany(size)
            if size is not None
            else self._inner.fetchmany()
        )
        if (
            self._batch_lengths is not None
            and self._last_sql.startswith("SELECT id FROM jobs")
            and "ORDER BY id FOR UPDATE" in self._last_sql
        ):
            self._batch_lengths.append(len(rows or []))
        return rows

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _SQLTraceConnection:
    """Delegate a connection while recording executed SQL."""

    def __init__(
        self,
        inner: Any,
        traces: list[str],
        batch_lengths: list[int] | None = None,
    ) -> None:
        self._inner = inner
        self._traces = traces
        self._batch_lengths = batch_lengths

    def execute(self, sql: Any, params: Any = ()) -> Any:
        self._traces.append(" ".join(str(sql).split()))
        return self._inner.execute(sql, params)

    def cursor(self, *args: Any, **kwargs: Any) -> _SQLTraceCursor:
        return _SQLTraceCursor(
            self._inner.cursor(*args, **kwargs),
            self._traces,
            self._batch_lengths,
        )

    def __enter__(self) -> _SQLTraceConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _manager(tmp_path: Any, jobs_pg_dsn: str | None) -> JobManager:
    if jobs_pg_dsn is not None:
        return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    return JobManager(tmp_path / "ttl-prune-boundaries.db")


def _execute(jm: JobManager, sql: str, params: tuple[Any, ...]) -> None:
    conn = jm._connect()
    try:
        with conn:
            if jm.backend == "postgres":
                with jm._pg_cursor(conn) as cur:
                    cur.execute(sql, params)
            else:
                conn.execute(sql, params)
    finally:
        conn.close()


def _seed_old_queued(jm: JobManager, *, domain: str) -> dict[str, Any]:
    job = jm.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    if jm.backend == "postgres":
        _execute(
            jm,
            "UPDATE jobs SET created_at = TIMESTAMPTZ '2019-01-01 00:00:00+00' WHERE id=%s",
            (int(job["id"]),),
        )
    else:
        _execute(
            jm,
            "UPDATE jobs SET created_at = '2019-01-01 00:00:00' WHERE id=?",
            (int(job["id"]),),
        )
    return job


def _seed_old_completed(jm: JobManager, *, domain: str) -> dict[str, Any]:
    job = jm.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    acquired = jm.acquire_next_job(
        domain=domain,
        queue="default",
        lease_seconds=30,
        worker_id="worker",
    )
    assert acquired is not None
    assert jm.complete_job(
        int(job["id"]),
        result={},
        worker_id="worker",
        lease_id=str(acquired["lease_id"]),
    )
    if jm.backend == "postgres":
        _execute(
            jm,
            "UPDATE jobs SET completed_at = NOW() - INTERVAL '90 days' WHERE id=%s",
            (int(job["id"]),),
        )
    else:
        _execute(
            jm,
            "UPDATE jobs SET completed_at = DATETIME('now', '-90 days') WHERE id=?",
            (int(job["id"]),),
        )
    return job


def _patch_observers(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[Any, ...]], _MetricRegistry]:
    import tldw_Server_API.app.core.Jobs.manager as manager_module
    import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_module

    events: list[tuple[Any, ...]] = []
    registry = _MetricRegistry()
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *args, **kwargs: events.append((*args, kwargs)),
    )
    monkeypatch.setattr(metrics_module, "get_metrics_registry", lambda: registry)
    return events, registry


def _assert_prune_counter_failure_rolls_back(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: Exception,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    job = _seed_old_queued(jm, domain="prune-counter-rollback")
    events, _ = _patch_observers(monkeypatch)
    original_connect = jm._connect
    monkeypatch.setattr(
        jm,
        "_connect",
        lambda: _FailCounterWriteConnection(original_connect(), error),
    )

    with pytest.raises(type(error), match="injected counter write failure"):
        jm.prune_jobs(
            statuses=["queued"],
            older_than_days=30,
            domain="prune-counter-rollback",
        )

    persisted = reader.get_job(int(job["id"]))
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert _counter(reader, domain="prune-counter-rollback") == (1, 0, 0)
    assert events == []


def _assert_prune_dry_run_observer_contract(
    jm: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _seed_old_completed(jm, domain="prune-dry-run-observer")
    events, _ = _patch_observers(monkeypatch)

    assert (
        jm.prune_jobs(
            statuses=["completed"],
            older_than_days=30,
            domain="prune-dry-run-observer",
            dry_run=True,
        )
        == 1
    )
    assert jm.get_job(int(job["id"])) is not None
    assert len(events) == 1
    assert events[0][0] == "jobs.pruned"
    attrs = events[0][-1]["attrs"]
    assert attrs["dry_run"] is True
    assert attrs["deleted"] == 1


def _assert_ttl_success_observer_sees_commit(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Jobs.manager as manager_module

    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    job = _seed_old_queued(jm, domain="ttl-success-observer")
    observed: list[tuple[str, str, int]] = []

    def observe(event_type: str, *, job: Any = None, attrs: Any = None) -> None:
        assert job is None
        persisted = reader.get_job(int(job_id))
        assert persisted is not None
        observed.append(
            (
                event_type,
                str(persisted["status"]),
                int((attrs or {}).get("affected") or 0),
            )
        )

    job_id = int(job["id"])
    monkeypatch.setattr(manager_module, "emit_job_event", observe)

    assert (
        jm.apply_ttl_policies(
            age_seconds=1,
            action="cancel",
            domain="ttl-success-observer",
            reference_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        == 1
    )
    assert observed == [("jobs.ttl_sweep", "cancelled", 1)]


def _assert_ttl_commit_failure(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    job = _seed_old_queued(jm, domain="ttl-rollback")
    events, registry = _patch_observers(monkeypatch)
    original_connect = jm._connect
    monkeypatch.setattr(jm, "_connect", lambda: _RollbackInsteadOfCommit(original_connect()))

    with pytest.raises(RuntimeError, match="forced commit failure"):
        jm.apply_ttl_policies(
            age_seconds=1,
            action="cancel",
            domain="ttl-rollback",
            reference_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )

    persisted = reader.get_job(int(job["id"]))
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert events == []
    assert registry.calls == []


def _assert_ttl_counter_failure_rolls_back(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: Exception,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    job = _seed_old_queued(jm, domain="ttl-counter-rollback")
    events, registry = _patch_observers(monkeypatch)
    original_connect = jm._connect
    monkeypatch.setattr(
        jm,
        "_connect",
        lambda: _FailCounterWriteConnection(original_connect(), error),
    )

    with pytest.raises(type(error), match="injected counter write failure"):
        jm.apply_ttl_policies(
            age_seconds=1,
            action="cancel",
            domain="ttl-counter-rollback",
            reference_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )

    persisted = reader.get_job(int(job["id"]))
    assert persisted is not None
    assert persisted["status"] == "queued"
    assert _counter(reader, domain="ttl-counter-rollback") == (1, 0, 0)
    assert events == []
    assert registry.calls == []


@pytest.mark.unit
def test_ttl_sqlite_commit_failure_rolls_back_without_observers(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_ttl_commit_failure(jm, _manager(tmp_path, None), monkeypatch)


@pytest.mark.pg_jobs
def test_ttl_postgres_commit_failure_rolls_back_without_observers(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_ttl_commit_failure(jm, _manager(tmp_path, jobs_pg_dsn), monkeypatch)


@pytest.mark.unit
def test_ttl_sqlite_counter_failure_rolls_back_state_and_observers(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_ttl_counter_failure_rolls_back(
        jm,
        _manager(tmp_path, None),
        monkeypatch,
        error=sqlite3.OperationalError("injected counter write failure"),
    )


@pytest.mark.pg_jobs
def test_ttl_postgres_counter_failure_rolls_back_state_and_observers(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psycopg = pytest.importorskip("psycopg")
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_ttl_counter_failure_rolls_back(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
        error=psycopg.OperationalError("injected counter write failure"),
    )


def _assert_prune_commit_failure(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _seed_old_completed(jm, domain="prune-rollback")
    events, _ = _patch_observers(monkeypatch)
    original_connect = jm._connect
    monkeypatch.setattr(jm, "_connect", lambda: _RollbackInsteadOfCommit(original_connect()))

    with pytest.raises(RuntimeError, match="forced commit failure"):
        jm.prune_jobs(
            statuses=["completed"],
            older_than_days=30,
            domain="prune-rollback",
        )

    assert reader.get_job(int(job["id"])) is not None
    assert events == []


@pytest.mark.unit
def test_prune_sqlite_commit_failure_is_observer_free(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_prune_commit_failure(jm, _manager(tmp_path, None), monkeypatch)


@pytest.mark.pg_jobs
def test_prune_postgres_commit_failure_is_observer_free(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_prune_commit_failure(jm, _manager(tmp_path, jobs_pg_dsn), monkeypatch)


def _assert_prune_success_observer_sees_commit(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Jobs.manager as manager_module

    job = _seed_old_completed(jm, domain="prune-observer")
    observed: list[dict[str, Any] | None] = []
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: observed.append(reader.get_job(int(job["id"]))),
    )

    assert (
        jm.prune_jobs(
            statuses=["completed"],
            older_than_days=30,
            domain="prune-observer",
        )
        == 1
    )
    assert observed == [None]


@pytest.mark.unit
def test_prune_sqlite_success_observer_sees_committed_delete(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_prune_success_observer_sees_commit(jm, _manager(tmp_path, None), monkeypatch)


@pytest.mark.pg_jobs
def test_prune_postgres_success_observer_sees_committed_delete(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_prune_success_observer_sees_commit(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
    )


@pytest.mark.unit
def test_prune_sqlite_counter_failure_rolls_back_delete_and_observers(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_prune_counter_failure_rolls_back(
        jm,
        _manager(tmp_path, None),
        monkeypatch,
        error=sqlite3.OperationalError("injected counter write failure"),
    )


@pytest.mark.pg_jobs
def test_prune_postgres_counter_failure_rolls_back_delete_and_observers(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psycopg = pytest.importorskip("psycopg")
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_prune_counter_failure_rolls_back(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
        error=psycopg.OperationalError("injected counter write failure"),
    )


@pytest.mark.unit
def test_prune_sqlite_dry_run_emits_observer_contract(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_prune_dry_run_observer_contract(
        _manager(tmp_path, None),
        monkeypatch,
    )


@pytest.mark.pg_jobs
def test_prune_postgres_dry_run_emits_observer_contract(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_prune_dry_run_observer_contract(
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
    )


@pytest.mark.unit
def test_ttl_sqlite_success_observer_sees_committed_terminal_state(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_ttl_success_observer_sees_commit(
        jm,
        _manager(tmp_path, None),
        monkeypatch,
    )


@pytest.mark.pg_jobs
def test_ttl_postgres_success_observer_sees_committed_terminal_state(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_ttl_success_observer_sees_commit(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
    )


def _counter(jm: JobManager, *, domain: str) -> tuple[int, int, int]:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
                    "WHERE domain=%s AND queue='default' AND job_type='work'",
                    (domain,),
                )
                row = cur.fetchone()
                assert row is not None
                return int(row["ready_count"]), int(row["scheduled_count"]), int(row["processing_count"])
        row = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
            "WHERE domain=? AND queue='default' AND job_type='work'",
            (domain,),
        ).fetchone()
        assert row is not None
        return int(row[0]), int(row[1]), int(row[2])
    finally:
        conn.close()


def _assert_ttl_uses_grouped_adapter_boundary(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    domain = f"ttl-grouped-{jm.backend}"
    jobs = [_seed_old_queued(jm, domain=domain) for _ in range(4)]
    traces: list[str] = []
    original_connect = jm._connect
    monkeypatch.setattr(
        jm,
        "_connect",
        lambda: _SQLTraceConnection(original_connect(), traces),
    )

    assert (
        jm.apply_ttl_policies(
            age_seconds=1,
            action="cancel",
            domain=domain,
            reference_time=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        == len(jobs)
    )
    assert _counter(reader, domain=domain) == (0, 0, 0)

    ttl_updates = [
        sql
        for sql in traces
        if "ttl_age" in sql.lower() and "update jobs set" in sql.lower()
    ]
    assert len(ttl_updates) == 1
    update_sql = ttl_updates[0].lower()
    if jm.backend == "postgres":
        assert update_sql.startswith("with changed as (update jobs set")
        assert "returning domain, queue, job_type, available_at" in update_sql
        assert "group by domain, queue, job_type" in update_sql
    else:
        assert update_sql.startswith("update jobs set")
        assert "returning" not in update_sql
        assert any(
            "select domain, queue, job_type" in sql.lower()
            and "group by domain, queue, job_type" in sql.lower()
            and "status='queued'" in sql.lower()
            for sql in traces
        )


@pytest.mark.unit
def test_ttl_sqlite_preaggregates_under_lock_without_returning(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_ttl_uses_grouped_adapter_boundary(
        jm,
        _manager(tmp_path, None),
        monkeypatch,
    )


@pytest.mark.pg_jobs
def test_ttl_postgres_aggregates_update_returning_server_side(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_ttl_uses_grouped_adapter_boundary(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
    )


@pytest.mark.pg_jobs
def test_prune_postgres_processes_fixed_candidates_in_bounded_batches(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Jobs.manager as manager_module

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setattr(manager_module, "_PRUNE_BATCH_SIZE", 2, raising=False)
    jm = _manager(tmp_path, jobs_pg_dsn)
    reader = _manager(tmp_path, jobs_pg_dsn)
    domain = "prune-bounded-batches"
    jobs = [_seed_old_queued(jm, domain=domain) for _ in range(5)]
    traces: list[str] = []
    batch_lengths: list[int] = []
    original_connect = jm._connect
    monkeypatch.setattr(
        jm,
        "_connect",
        lambda: _SQLTraceConnection(
            original_connect(),
            traces,
            batch_lengths,
        ),
    )

    assert (
        jm.prune_jobs(
            statuses=["queued"],
            older_than_days=30,
            domain=domain,
        )
        == len(jobs)
    )
    assert batch_lengths == [2, 2, 1, 0]
    assert _counter(reader, domain=domain) == (0, 0, 0)
    assert all(reader.get_job(int(job["id"])) is None for job in jobs)

    conn = reader._connect()
    try:
        with reader._pg_cursor(conn) as cur:
            cur.execute(
                "SELECT COUNT(*) AS c FROM jobs_archive WHERE domain=%s",
                (domain,),
            )
            archived = cur.fetchone()
    finally:
        conn.close()
    assert archived is not None
    assert int(archived["c"]) == len(jobs)
    assert any(
        sql.lower().startswith("select id from jobs")
        and "order by id for update" in sql.lower()
        for sql in traces
    )
    assert not any("temp table" in sql.lower() for sql in traces)


@pytest.mark.unit
def test_prune_sqlite_uses_one_cutoff_for_snapshot_archive_counters_and_delete(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv(
        "JOBS_TEST_NOW_EPOCH",
        str(datetime(2020, 1, 2, tzinfo=timezone.utc).timestamp()),
    )
    jm = _manager(tmp_path, None)
    reader = _manager(tmp_path, None)
    domain = "prune-fixed-cutoff"
    parent = jm.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    child = jm.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    _execute(
        jm,
        "UPDATE jobs SET created_at='2020-01-01 00:00:01' WHERE id=?",
        (int(parent["id"]),),
    )

    original_connect = jm._connect
    monkeypatch.setattr(
        jm,
        "_connect",
        lambda: _AdvancingSQLitePruneClock(
            original_connect(),
            count_time="2020-01-02 00:00:00",
            mutation_time="2020-01-02 00:00:02",
            status_param_count=1,
        ),
    )

    assert (
        jm.prune_jobs(
            statuses=["queued"],
            older_than_days=1,
            domain=domain,
        )
        == 0
    )
    persisted_parent = reader.get_job(int(parent["id"]))
    assert persisted_parent is not None
    assert persisted_parent["status"] == "queued"
    assert _counter(reader, domain=domain) == (2, 0, 0)

    conn = reader._connect()
    try:
        dependency = conn.execute(
            "SELECT depends_on_terminal_status, depends_on_cancellation_reason "
            "FROM job_dependencies WHERE job_uuid=? AND depends_on_job_uuid=?",
            (str(child["uuid"]), str(parent["uuid"])),
        ).fetchone()
        archived = conn.execute(
            "SELECT COUNT(*) FROM jobs_archive WHERE uuid=?",
            (str(parent["uuid"]),),
        ).fetchone()
    finally:
        conn.close()
    assert dependency is not None
    assert tuple(dependency) == (None, None)
    assert archived is not None
    assert int(archived[0]) == 0


@pytest.mark.unit
@pytest.mark.parametrize("action", ["cancel", "fail"])
def test_ttl_sqlite_uses_reference_time_for_bucket_and_terminal_timestamp(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    jm = _manager(tmp_path, None)
    domain = f"ttl-reference-{action}"
    job = _seed_old_queued(jm, domain=domain)
    reference = datetime(2020, 1, 1, tzinfo=timezone.utc)
    _execute(
        jm,
        "UPDATE jobs SET available_at='2020-01-01 00:00:30' WHERE id=?",
        (int(job["id"]),),
    )
    _execute(
        jm,
        "UPDATE job_counters SET ready_count=0, scheduled_count=1 WHERE domain=? AND queue='default' AND job_type='work'",
        (domain,),
    )

    assert (
        jm.apply_ttl_policies(
            age_seconds=1,
            action=action,
            domain=domain,
            reference_time=reference,
        )
        == 1
    )

    persisted = jm.get_job(int(job["id"]))
    assert persisted is not None
    assert persisted["status"] == ("cancelled" if action == "cancel" else "failed")
    timestamp = persisted["cancelled_at"] if action == "cancel" else persisted["completed_at"]
    parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    assert parsed == reference
    assert _counter(jm, domain=domain) == (0, 0, 0)


@pytest.mark.pg_jobs
@pytest.mark.parametrize("action", ["cancel", "fail"])
def test_ttl_postgres_uses_reference_time_for_bucket_and_terminal_timestamp(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    jm = _manager(tmp_path, jobs_pg_dsn)
    domain = f"ttl-reference-{action}"
    job = _seed_old_queued(jm, domain=domain)
    reference = datetime(2020, 1, 1, tzinfo=timezone.utc)
    _execute(
        jm,
        "UPDATE jobs SET available_at=TIMESTAMPTZ '2020-01-01 00:00:30+00' WHERE id=%s",
        (int(job["id"]),),
    )
    _execute(
        jm,
        "UPDATE job_counters SET ready_count=0, scheduled_count=1 "
        "WHERE domain=%s AND queue='default' AND job_type='work'",
        (domain,),
    )

    assert (
        jm.apply_ttl_policies(
            age_seconds=1,
            action=action,
            domain=domain,
            reference_time=reference,
        )
        == 1
    )

    persisted = jm.get_job(int(job["id"]))
    assert persisted is not None
    assert persisted["status"] == ("cancelled" if action == "cancel" else "failed")
    timestamp = persisted["cancelled_at"] if action == "cancel" else persisted["completed_at"]
    if isinstance(timestamp, datetime):
        parsed = timestamp
    else:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    assert parsed.astimezone(timezone.utc) == reference
    assert _counter(jm, domain=domain) == (0, 0, 0)


def _install_sweep_barrier(
    managers: tuple[JobManager, JobManager],
    monkeypatch: pytest.MonkeyPatch,
    barrier: threading.Barrier,
) -> None:
    if managers[0].backend == "postgres":
        for manager in managers:
            original_cursor = manager._pg_cursor

            @contextmanager
            def barrier_cursor(
                conn: Any,
                _original: Any = original_cursor,
            ) -> Any:
                with _original(conn) as cursor:
                    yield _PostgresSweepBarrierCursor(cursor, barrier)

            monkeypatch.setattr(manager, "_pg_cursor", barrier_cursor)
    else:
        for manager in managers:
            original_connect = manager._connect
            monkeypatch.setattr(
                manager,
                "_connect",
                lambda _original=original_connect: _SQLiteSweepBarrier(
                    _original(),
                    barrier,
                ),
            )


def _assert_concurrent_ttl_sweeps_claim_once(
    managers: tuple[JobManager, JobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    primary, _ = managers
    domain = "ttl-concurrent-claim"
    target = _seed_old_queued(primary, domain=domain)
    control = primary.create_job(
        domain=domain,
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="control",
    )
    _install_sweep_barrier(managers, monkeypatch, threading.Barrier(2))

    reference = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                manager.apply_ttl_policies,
                age_seconds=1,
                action="cancel",
                domain=domain,
                reference_time=reference,
            )
            for manager in managers
        ]
        results = sorted(future.result(timeout=15) for future in futures)

    assert results == [0, 1]
    assert primary.get_job(int(target["id"]))["status"] == "cancelled"
    assert primary.get_job(int(control["id"]))["status"] == "queued"
    assert _counter(primary, domain=domain) == (1, 0, 0)


def _assert_concurrent_runtime_sweeps_claim_once(
    managers: tuple[JobManager, JobManager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    primary, _ = managers
    domain = "ttl-concurrent-runtime"
    for index in range(2):
        primary.create_job(
            domain=domain,
            queue="default",
            job_type="work",
            payload={},
            owner_user_id=f"owner-{index}",
        )
    acquired = [
        primary.acquire_next_job(
            domain=domain,
            queue="default",
            lease_seconds=30,
            worker_id=f"worker-{index}",
        )
        for index in range(2)
    ]
    assert all(job is not None for job in acquired)
    target = acquired[0]
    control = acquired[1]
    assert target is not None and control is not None
    if primary.backend == "postgres":
        _execute(
            primary,
            "UPDATE jobs SET started_at=TIMESTAMPTZ '2019-01-01 00:00:00+00', "
            "acquired_at=TIMESTAMPTZ '2019-01-01 00:00:00+00' WHERE id=%s",
            (int(target["id"]),),
        )
    else:
        _execute(
            primary,
            "UPDATE jobs SET started_at='2019-01-01 00:00:00', acquired_at='2019-01-01 00:00:00' WHERE id=?",
            (int(target["id"]),),
        )
    _install_sweep_barrier(managers, monkeypatch, threading.Barrier(2))

    reference = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                manager.apply_ttl_policies,
                runtime_seconds=1,
                action="cancel",
                domain=domain,
                reference_time=reference,
            )
            for manager in managers
        ]
        results = sorted(future.result(timeout=15) for future in futures)

    assert results == [0, 1]
    assert primary.get_job(int(target["id"]))["status"] == "cancelled"
    assert primary.get_job(int(control["id"]))["status"] == "processing"
    assert _counter(primary, domain=domain) == (0, 0, 1)


@pytest.mark.unit
@pytest.mark.concurrent
def test_ttl_sqlite_concurrent_sweeps_claim_candidate_once(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_concurrent_ttl_sweeps_claim_once(
        (_manager(tmp_path, None), _manager(tmp_path, None)),
        monkeypatch,
    )


@pytest.mark.pg_jobs
@pytest.mark.concurrent
def test_ttl_postgres_concurrent_sweeps_claim_candidate_once(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_concurrent_ttl_sweeps_claim_once(
        (
            _manager(tmp_path, jobs_pg_dsn),
            _manager(tmp_path, jobs_pg_dsn),
        ),
        monkeypatch,
    )


@pytest.mark.unit
@pytest.mark.concurrent
def test_ttl_sqlite_concurrent_runtime_sweeps_claim_candidate_once(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_concurrent_runtime_sweeps_claim_once(
        (_manager(tmp_path, None), _manager(tmp_path, None)),
        monkeypatch,
    )


@pytest.mark.pg_jobs
@pytest.mark.concurrent
def test_ttl_postgres_concurrent_runtime_sweeps_claim_candidate_once(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_concurrent_runtime_sweeps_claim_once(
        (
            _manager(tmp_path, jobs_pg_dsn),
            _manager(tmp_path, jobs_pg_dsn),
        ),
        monkeypatch,
    )

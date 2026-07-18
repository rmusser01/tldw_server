from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs import migrations as jobs_migrations
from tldw_Server_API.app.core.Jobs import pg_migrations as jobs_pg_migrations
from tldw_Server_API.app.core.Jobs.manager import JobManager


class _SignalAfterCommit:
    """Expose the instant a prune transaction has committed."""

    def __init__(self, inner: Any, committed: threading.Event, release: threading.Event) -> None:
        self._inner = inner
        self._committed = committed
        self._release = release

    def __enter__(self) -> _SignalAfterCommit:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        result = self._inner.__exit__(exc_type, exc, tb)
        if exc_type is None:
            self._committed.set()
            assert self._release.wait(timeout=10)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ExecuteHookCursor:
    """Delegate a DB cursor while exposing SQL adapter boundaries to a test."""

    def __init__(self, inner: Any, before_execute: Any) -> None:
        self._inner = inner
        self._before_execute = before_execute

    def __enter__(self) -> _ExecuteHookCursor:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, query: Any, params: Any = None) -> Any:
        self._before_execute(str(query))
        if params is None:
            return self._inner.execute(query)
        return self._inner.execute(query, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ExecuteHookConnection:
    """Delegate a DB connection while exposing SQL adapter boundaries to a test."""

    def __init__(self, inner: Any, before_execute: Any) -> None:
        self._inner = inner
        self._before_execute = before_execute

    def __enter__(self) -> _ExecuteHookConnection:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, query: Any, params: Any = None) -> Any:
        self._before_execute(str(query))
        if params is None:
            return self._inner.execute(query)
        return self._inner.execute(query, params)

    def cursor(self, *args: Any, **kwargs: Any) -> _ExecuteHookCursor:
        return _ExecuteHookCursor(
            self._inner.cursor(*args, **kwargs),
            self._before_execute,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _manager(tmp_path: Any, jobs_pg_dsn: str | None) -> JobManager:
    if jobs_pg_dsn is not None:
        return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    return JobManager(tmp_path / "dependency-prune.db")


def _set_terminal_and_old(jm: JobManager, job_id: int, status: str) -> None:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with conn, jm._pg_cursor(conn) as cur:
                cur.execute(
                    "UPDATE jobs SET status=%s, completed_at=NOW() - INTERVAL '2 days', "
                    "cancellation_reason=%s WHERE id=%s",
                    (status, "user_cancelled" if status == "cancelled" else None, job_id),
                )
        else:
            with conn:
                conn.execute(
                    "UPDATE jobs SET status=?, completed_at=DATETIME('now', '-2 days'), "
                    "cancellation_reason=? WHERE id=?",
                    (status, "user_cancelled" if status == "cancelled" else None, job_id),
                )
    finally:
        conn.close()


def _backdate_created_at(jm: JobManager, job_id: int) -> None:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with conn, jm._pg_cursor(conn) as cur:
                cur.execute(
                    "UPDATE jobs SET created_at=NOW() - INTERVAL '2 days' WHERE id=%s",
                    (job_id,),
                )
        else:
            with conn:
                conn.execute(
                    "UPDATE jobs SET created_at=DATETIME('now', '-2 days') WHERE id=?",
                    (job_id,),
                )
    finally:
        conn.close()


def _cancel_without_completion_timestamp(jm: JobManager, job_id: int) -> None:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with conn, jm._pg_cursor(conn) as cur:
                cur.execute(
                    "UPDATE jobs SET status='cancelled', cancelled_at=NOW(), "
                    "cancellation_reason='user_cancelled', completed_at=NULL WHERE id=%s",
                    (job_id,),
                )
        else:
            with conn:
                conn.execute(
                    "UPDATE jobs SET status='cancelled', cancelled_at=DATETIME('now'), "
                    "cancellation_reason='user_cancelled', completed_at=NULL WHERE id=?",
                    (job_id,),
                )
    finally:
        conn.close()


def _dependency_snapshot(jm: JobManager, child_uuid: str) -> tuple[str | None, str | None]:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT depends_on_terminal_status, depends_on_cancellation_reason "
                    "FROM job_dependencies WHERE job_uuid=%s",
                    (child_uuid,),
                )
                row = cur.fetchone()
                assert row is not None
                return row["depends_on_terminal_status"], row["depends_on_cancellation_reason"]
        row = conn.execute(
            "SELECT depends_on_terminal_status, depends_on_cancellation_reason FROM job_dependencies WHERE job_uuid=?",
            (child_uuid,),
        ).fetchone()
        assert row is not None
        return row[0], row[1]
    finally:
        conn.close()


def _dependency_count(jm: JobManager, child_uuid: str) -> int:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
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


def _assert_reciprocal_dependency_race_preserves_acyclic_graph(
    first_manager: JobManager,
    second_manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Race reciprocal edge inserts after both legacy prechecks have completed."""

    domain = f"dependency-cycle-race-{first_manager.backend}"
    first_job = first_manager.create_job(
        domain=domain,
        queue="default",
        job_type="first",
        payload={},
        owner_user_id="owner-1",
    )
    second_job = first_manager.create_job(
        domain=domain,
        queue="default",
        job_type="second",
        payload={},
        owner_user_id="owner-1",
    )
    first_uuid = str(first_job["uuid"])
    second_uuid = str(second_job["uuid"])

    adapter_boundary = threading.Barrier(2)
    trace_lock = threading.Lock()
    query_traces: dict[object, list[str]] = {}
    synchronized_connections: set[object] = set()

    for manager in (first_manager, second_manager):
        original_connect = manager._connect

        def hooked_connect(
            *,
            _original: Any = original_connect,
        ) -> _ExecuteHookConnection:
            connection_token = object()
            with trace_lock:
                query_traces[connection_token] = []

            def before_execute(query: str) -> None:
                normalized = " ".join(query.lower().split())
                dependency_insert = "insert" in normalized and "job_dependencies" in normalized
                if first_manager.backend == "postgres":
                    lock_boundary = "pg_advisory_xact_lock" in normalized
                else:
                    lock_boundary = normalized.startswith("begin")

                with trace_lock:
                    query_traces[connection_token].append(normalized)
                    should_synchronize = (
                        lock_boundary or dependency_insert
                    ) and connection_token not in synchronized_connections
                    if should_synchronize:
                        synchronized_connections.add(connection_token)

                if should_synchronize:
                    adapter_boundary.wait(timeout=10)

            return _ExecuteHookConnection(_original(), before_execute)

        monkeypatch.setattr(
            manager,
            "_connect",
            hooked_connect,
        )

    def add_edge(manager: JobManager, child_uuid: str, parent_uuid: str) -> str:
        try:
            assert manager.add_job_dependency(child_uuid, parent_uuid)
        except ValueError as exc:
            assert str(exc) == "Dependency would create a cycle"
            return "cycle_rejected"
        return "inserted"

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            add_edge,
            first_manager,
            first_uuid,
            second_uuid,
        )
        second = pool.submit(
            add_edge,
            second_manager,
            second_uuid,
            first_uuid,
        )
        outcomes = [first.result(timeout=15), second.result(timeout=15)]

    assert sorted(outcomes) == ["cycle_rejected", "inserted"]
    with trace_lock:
        traces = list(query_traces.values())
        synchronized_count = len(synchronized_connections)

    mutation_traces = [
        queries
        for queries in traces
        if any(
            "pg_advisory_xact_lock" in query
            or query.startswith("begin")
            or ("insert" in query and "job_dependencies" in query)
            for query in queries
        )
    ]
    assert synchronized_count == 2
    assert len(mutation_traces) == 2

    dependency_insert_count = 0
    for queries in mutation_traces:
        path_index = next(
            index for index, query in enumerate(queries) if query.startswith("with recursive dependency_path")
        )
        if first_manager.backend == "postgres":
            isolation_index = queries.index("set transaction isolation level read committed")
            lock_index = next(index for index, query in enumerate(queries) if "pg_advisory_xact_lock" in query)
            assert isolation_index < lock_index < path_index
        else:
            lock_index = queries.index("begin immediate")
            assert lock_index < path_index

        insert_indexes = [
            index for index, query in enumerate(queries) if "insert" in query and "job_dependencies" in query
        ]
        dependency_insert_count += len(insert_indexes)
        assert not insert_indexes or path_index < insert_indexes[0]

    assert dependency_insert_count == 1
    assert _dependency_count(first_manager, first_uuid) + _dependency_count(first_manager, second_uuid) == 1


def _assert_dependency_add_prune_race_is_atomic(
    add_manager: JobManager,
    prune_manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = f"dependency-add-prune-{add_manager.backend}"
    parent = add_manager.create_job(
        domain=domain,
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="owner-1",
    )
    child = add_manager.create_job(
        domain=domain,
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="owner-1",
    )
    _set_terminal_and_old(add_manager, int(parent["id"]), "completed")

    mutation_seen = threading.Event()
    original_connect = add_manager._connect

    def before_execute(query: str) -> None:
        normalized = " ".join(query.lower().split())
        is_dependency_insert = "insert" in normalized and "job_dependencies" in normalized
        if mutation_seen.is_set() or not (normalized.startswith("begin immediate") or is_dependency_insert):
            return
        mutation_seen.set()
        with ThreadPoolExecutor(max_workers=1) as pool:
            prune = pool.submit(
                prune_manager.prune_jobs,
                statuses=["completed"],
                older_than_days=1,
                domain=domain,
            )
            assert prune.result(timeout=10) == 1

    def hooked_connect() -> Any:
        return _ExecuteHookConnection(original_connect(), before_execute)

    monkeypatch.setattr(add_manager, "_connect", hooked_connect)
    try:
        added = add_manager.add_job_dependency(
            str(child["uuid"]),
            str(parent["uuid"]),
        )
    except ValueError:
        added = False

    assert mutation_seen.is_set()
    assert added is False
    assert _dependency_count(prune_manager, str(child["uuid"])) == 0


def _assert_existing_terminal_dependency_is_snapshotted(jm: JobManager) -> None:
    domain = f"dependency-add-snapshot-{jm.backend}"
    parent = jm.create_job(
        domain=domain,
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="owner-1",
    )
    child = jm.create_job(
        domain=domain,
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="owner-1",
    )
    _set_terminal_and_old(jm, int(parent["id"]), "completed")

    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    assert _dependency_snapshot(jm, str(child["uuid"])) == ("completed", None)


def _assert_pruned_terminal_dependency_stays_blocking(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str,
) -> None:
    domain = f"dependency-prune-{status}-{jm.backend}"
    parent = jm.create_job(
        domain=domain,
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="owner-1",
    )
    child = jm.create_job(
        domain=domain,
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="owner-1",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    _set_terminal_and_old(jm, int(parent["id"]), status)

    committed = threading.Event()
    acquisition_done = threading.Event()
    original_connect = jm._connect
    first_connection = True

    def hooked_connect() -> Any:
        nonlocal first_connection
        conn = original_connect()
        if first_connection:
            first_connection = False
            return _SignalAfterCommit(conn, committed, acquisition_done)
        return conn

    monkeypatch.setattr(jm, "_connect", hooked_connect)

    with ThreadPoolExecutor(max_workers=2) as pool:
        prune_future = pool.submit(
            jm.prune_jobs,
            statuses=[status],
            older_than_days=1,
            domain=domain,
        )
        assert committed.wait(timeout=10)
        try:
            acquire_future = pool.submit(
                reader.acquire_next_job,
                domain=domain,
                queue="default",
                lease_seconds=30,
                worker_id="worker-after-prune",
            )
            assert acquire_future.result(timeout=10) is None
        finally:
            acquisition_done.set()
        assert prune_future.result(timeout=10) == 1

    assert reader.get_job(int(parent["id"])) is None
    persisted_child = reader.get_job(int(child["id"]))
    assert persisted_child is not None
    assert persisted_child["status"] == "cancelled"
    expected_reason = "dependency_failed" if status == "failed" else "dependency_cancelled"
    assert persisted_child["cancellation_reason"] == expected_reason
    assert _dependency_snapshot(reader, str(child["uuid"])) == (
        status,
        "user_cancelled" if status == "cancelled" else None,
    )


def _assert_pruned_active_parent_never_strands_child(
    jm: JobManager,
    reader: JobManager,
    *,
    status: str,
) -> None:
    domain = f"dependency-prune-active-{status}-{jm.backend}"
    parent = jm.create_job(
        domain=domain,
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="owner-1",
    )
    child = jm.create_job(
        domain=domain,
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="owner-1",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    if status == "processing":
        acquired = jm.acquire_next_job(
            domain=domain,
            queue="default",
            job_type="parent",
            lease_seconds=30,
            worker_id="active-parent-worker",
        )
        assert acquired is not None
        assert int(acquired["id"]) == int(parent["id"])
    _backdate_created_at(jm, int(parent["id"]))

    assert (
        jm.prune_jobs(
            statuses=[status],
            older_than_days=1,
            domain=domain,
        )
        == 1
    )
    assert reader.get_job(int(parent["id"])) is None
    assert _dependency_snapshot(reader, str(child["uuid"])) == (
        "cancelled",
        "pruned",
    )

    assert (
        reader.acquire_next_job(
            domain=domain,
            queue="default",
            lease_seconds=30,
            worker_id="worker-after-active-prune",
        )
        is None
    )
    persisted_child = reader.get_job(int(child["id"]))
    assert persisted_child is not None
    assert persisted_child["status"] == "cancelled"
    assert persisted_child["cancellation_reason"] == "dependency_cancelled"


def _assert_postgres_prune_uses_fixed_candidate_membership(
    jm: JobManager,
    transition_manager: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = "dependency-prune-fixed-membership-postgres"
    initial_candidate = jm.create_job(
        domain=domain,
        queue="default",
        job_type="initial",
        payload={},
        owner_user_id="owner-1",
    )
    _set_terminal_and_old(jm, int(initial_candidate["id"]), "cancelled")
    entering_parent = jm.create_job(
        domain=domain,
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="owner-1",
    )
    child = jm.create_job(
        domain=domain,
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="owner-1",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(entering_parent["uuid"]))
    _backdate_created_at(jm, int(entering_parent["id"]))

    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    transition_seen = threading.Event()
    original_connect = jm._connect

    def before_execute(query: str) -> None:
        normalized = " ".join(query.lower().split())
        if transition_seen.is_set() or "insert into jobs_archive" not in normalized:
            return
        _cancel_without_completion_timestamp(
            transition_manager,
            int(entering_parent["id"]),
        )
        transition_seen.set()

    monkeypatch.setattr(
        jm,
        "_connect",
        lambda: _ExecuteHookConnection(original_connect(), before_execute),
    )

    assert (
        jm.prune_jobs(
            statuses=["cancelled"],
            older_than_days=1,
            domain=domain,
        )
        == 1
    )
    assert transition_seen.is_set()
    assert jm.get_job(int(initial_candidate["id"])) is None
    archived = jm.get_job_or_archived(
        int(initial_candidate["id"]),
        domain=domain,
    )
    assert archived is not None
    assert archived["archived"] is True

    persisted_parent = jm.get_job(int(entering_parent["id"]))
    assert persisted_parent is not None
    assert persisted_parent["status"] == "cancelled"
    assert _dependency_snapshot(jm, str(child["uuid"])) == (None, None)


@pytest.mark.unit
@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_sqlite_pruned_terminal_dependency_never_unblocks_child(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_pruned_terminal_dependency_stays_blocking(
        jm,
        _manager(tmp_path, None),
        monkeypatch,
        status=status,
    )


@pytest.mark.pg_jobs
@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_postgres_pruned_terminal_dependency_never_unblocks_child(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_pruned_terminal_dependency_stays_blocking(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
        status=status,
    )


@pytest.mark.unit
@pytest.mark.parametrize("status", ["queued", "processing"])
def test_sqlite_pruned_active_parent_never_strands_child(
    tmp_path: Any,
    status: str,
) -> None:
    jm = _manager(tmp_path, None)
    _assert_pruned_active_parent_never_strands_child(
        jm,
        _manager(tmp_path, None),
        status=status,
    )


@pytest.mark.pg_jobs
@pytest.mark.parametrize("status", ["queued", "processing"])
def test_postgres_pruned_active_parent_never_strands_child(
    tmp_path: Any,
    jobs_pg_dsn: str,
    status: str,
) -> None:
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_pruned_active_parent_never_strands_child(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        status=status,
    )


@pytest.mark.pg_jobs
@pytest.mark.concurrent
def test_postgres_prune_keeps_fixed_candidate_membership_when_job_enters_filter(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_postgres_prune_uses_fixed_candidate_membership(
        _manager(tmp_path, jobs_pg_dsn),
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
    )


@pytest.mark.unit
def test_sqlite_add_dependency_racing_prune_never_persists_missing_parent_edge(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_dependency_add_prune_race_is_atomic(
        _manager(tmp_path, None),
        _manager(tmp_path, None),
        monkeypatch,
    )


@pytest.mark.pg_jobs
def test_postgres_add_dependency_racing_prune_never_persists_missing_parent_edge(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_dependency_add_prune_race_is_atomic(
        _manager(tmp_path, jobs_pg_dsn),
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
    )


@pytest.mark.unit
def test_sqlite_concurrent_reciprocal_dependencies_preserve_acyclic_graph(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_reciprocal_dependency_race_preserves_acyclic_graph(
        _manager(tmp_path, None),
        _manager(tmp_path, None),
        monkeypatch,
    )


@pytest.mark.pg_jobs
def test_postgres_concurrent_reciprocal_dependencies_preserve_acyclic_graph(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_reciprocal_dependency_race_preserves_acyclic_graph(
        _manager(tmp_path, jobs_pg_dsn),
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
    )


@pytest.mark.unit
def test_sqlite_add_dependency_captures_existing_terminal_parent_snapshot(
    tmp_path: Any,
) -> None:
    _assert_existing_terminal_dependency_is_snapshotted(_manager(tmp_path, None))


@pytest.mark.pg_jobs
def test_postgres_add_dependency_captures_existing_terminal_parent_snapshot(
    tmp_path: Any,
    jobs_pg_dsn: str,
) -> None:
    _assert_existing_terminal_dependency_is_snapshotted(_manager(tmp_path, jobs_pg_dsn))


@pytest.mark.unit
def test_sqlite_dependency_snapshot_columns_forward_migrate(tmp_path: Any) -> None:
    db_path = tmp_path / "old-dependencies.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE job_dependencies ("
            "job_uuid TEXT NOT NULL, depends_on_job_uuid TEXT NOT NULL, "
            "created_at TEXT DEFAULT (DATETIME('now')), "
            "PRIMARY KEY (job_uuid, depends_on_job_uuid))"
        )

    JobManager(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(job_dependencies)")}
    assert {"depends_on_terminal_status", "depends_on_cancellation_reason"} <= columns


@pytest.mark.unit
def test_sqlite_dependency_snapshot_migration_failure_fails_closed(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "broken-dependency-migration.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE job_dependencies ("
            "job_uuid TEXT NOT NULL, depends_on_job_uuid TEXT NOT NULL, "
            "created_at TEXT DEFAULT (DATETIME('now')), "
            "PRIMARY KEY (job_uuid, depends_on_job_uuid))"
        )

    real_connect = jobs_migrations.sqlite3.connect
    injected = threading.Event()

    def fail_required_migration(query: str) -> None:
        normalized = " ".join(query.lower().split())
        if (
            not injected.is_set()
            and "alter table job_dependencies" in normalized
            and "depends_on_terminal_status" in normalized
        ):
            injected.set()
            raise sqlite3.OperationalError("injected dependency snapshot migration failure")

    def failing_connect(*args: Any, **kwargs: Any) -> _ExecuteHookConnection:
        return _ExecuteHookConnection(
            real_connect(*args, **kwargs),
            fail_required_migration,
        )

    monkeypatch.setattr(jobs_migrations.sqlite3, "connect", failing_connect)

    with pytest.raises((RuntimeError, sqlite3.Error)):
        jobs_migrations.ensure_jobs_tables(db_path)
    assert injected.is_set()


@pytest.mark.pg_jobs
def test_postgres_dependency_snapshot_columns_are_migrated(
    tmp_path: Any,
    jobs_pg_dsn: str,
) -> None:
    jm = _manager(tmp_path, jobs_pg_dsn)
    conn = jm._connect()
    try:
        with jm._pg_cursor(conn) as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name='job_dependencies'"
            )
            columns = {row["column_name"] for row in (cur.fetchall() or [])}
    finally:
        conn.close()
    assert {"depends_on_terminal_status", "depends_on_cancellation_reason"} <= columns


@pytest.mark.pg_jobs
def test_postgres_dependency_snapshot_migration_failure_fails_closed(
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE job_dependencies DROP COLUMN IF EXISTS depends_on_terminal_status")
        cur.execute("ALTER TABLE job_dependencies DROP COLUMN IF EXISTS depends_on_cancellation_reason")

    real_connect = psycopg.connect
    injected = threading.Event()

    def fail_required_migration(query: str) -> None:
        normalized = " ".join(query.lower().split())
        if (
            not injected.is_set()
            and "alter table job_dependencies" in normalized
            and "depends_on_terminal_status" in normalized
        ):
            injected.set()
            raise RuntimeError("injected dependency snapshot migration failure")

    def failing_connect(*args: Any, **kwargs: Any) -> _ExecuteHookConnection:
        return _ExecuteHookConnection(
            real_connect(*args, **kwargs),
            fail_required_migration,
        )

    try:
        with monkeypatch.context() as context:
            context.setattr(psycopg, "connect", failing_connect)
            with pytest.raises(RuntimeError, match="dependency snapshot"):
                jobs_pg_migrations.ensure_jobs_tables_pg(jobs_pg_dsn)
    finally:
        jobs_pg_migrations.ensure_jobs_tables_pg(jobs_pg_dsn)
    assert injected.is_set()

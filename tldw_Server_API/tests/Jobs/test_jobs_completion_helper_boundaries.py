"""Direct ownership and cold-import checks for the mechanically moved helpers."""

from __future__ import annotations

import builtins
import importlib
import json
import multiprocessing
import sqlite3
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]
USE_SHARED_JOBS_POSTGRES = True
DB_PACKAGE = "tldw_Server_API.app.core.DB_Management"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "names"),
    [
        ("jobs_lifecycle_persistence", ("_insert_lifecycle_event", "_reconcile_lifecycle_counter_row")),
        ("jobs_transaction_helpers", ("_commit_postgres_transaction",)),
        ("jobs_exception_policy", ("_PG_ERRORS", "_JOB_NONCRITICAL_EXCEPTIONS")),
    ],
)
def test_manager_exports_exact_helper_objects(module_name, names):
    """A facade wrapper or independently constructed tuple is not an alias."""
    module = importlib.import_module(f"{DB_PACKAGE}.{module_name}")
    manager = importlib.import_module("tldw_Server_API.app.core.Jobs.manager")
    for name in names:
        assert getattr(manager, name) is getattr(module, name)


class ExecutorProbe:
    """Delegate real SQL while rejecting transaction ownership by a helper."""

    def __init__(self, cursor: Any) -> None:
        self.cursor = cursor
        self.boundaries: list[str] = []

    def execute(self, sql: str, params: Any = ()) -> ExecutorProbe:
        self.cursor.execute(sql, params)
        return self

    def fetchone(self) -> Any:
        return self.cursor.fetchone()

    def commit(self) -> None:
        self.boundaries.append("commit")
        raise AssertionError("helper committed")

    def rollback(self) -> None:
        self.boundaries.append("rollback")
        raise AssertionError("helper rolled back")

    def close(self) -> None:
        self.boundaries.append("close")
        raise AssertionError("helper closed executor")


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.pg_jobs)])
def helper_owner(request, tmp_path, monkeypatch):
    """Use native transactions and the existing shared PostgreSQL provisioner."""
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    if request.param == "postgres":
        manager = JobManager(backend="postgres", db_url=request.getfixturevalue("jobs_pg_dsn"))
    else:
        monkeypatch.delenv("JOBS_DB_URL", raising=False)
        manager = JobManager(tmp_path / "helper-boundaries.db")
    with ExitStack() as stack:
        conn = manager._connect()
        stack.callback(conn.close)
        stack.callback(conn.rollback)
        cursor = stack.enter_context(manager._pg_cursor(conn)) if manager.backend == "postgres" else conn.cursor()
        if manager.backend == "sqlite":
            conn.execute("BEGIN IMMEDIATE")
        else:
            cursor.execute("SELECT current_database() AS name")
            _, database_name = request.getfixturevalue("isolated_test_environment")
            assert cursor.fetchone()["name"] == database_name
            assert "pg_temp_db" not in request.fixturenames
        yield manager, conn, ExecutorProbe(cursor)


@pytest.mark.integration
@pytest.mark.parametrize("attrs", [None, {}, {"nested": [1, False, None], "label": "raw"}])
@pytest.mark.parametrize("finish", ["rollback", "commit"])
def test_event_attribution_and_owning_transaction(helper_owner, attrs, finish):
    """Event values stay raw; only the owner decides their durability."""
    from tldw_Server_API.tests.Jobs._completion_characterization_support import rows

    helper = importlib.import_module(f"{DB_PACKAGE}.jobs_lifecycle_persistence")
    manager, conn, executor = helper_owner
    job = {
        "id": "123",
        "domain": "chatbooks",
        "queue": " raw queue ",
        "job_type": None,
        "owner_user_id": " owner ",
        "request_id": None,
        "trace_id": " trace ",
    }
    helper._insert_lifecycle_event(executor, backend=manager.backend, event_type="helper.probe", job=job, attrs=attrs)
    executor.execute("SELECT * FROM job_events WHERE event_type='helper.probe'")
    event = dict(executor.fetchone())
    assert event["job_id"] == 123
    for field in ("domain", "queue", "job_type", "owner_user_id", "request_id", "trace_id"):
        assert event[field] == job[field]
    actual_attrs = event["attrs_json"]
    assert (json.loads(actual_attrs) if isinstance(actual_attrs, str) else actual_attrs) == (attrs or {})
    assert event["created_at"] is not None
    assert executor.boundaries == []
    getattr(conn, finish)()
    persisted = rows(manager, "SELECT * FROM job_events WHERE event_type='helper.probe'")
    assert len(persisted) == (1 if finish == "commit" else 0)


@pytest.mark.integration
def test_counter_reconstruction_is_transaction_local(helper_owner):
    """Rebuild all four buckets, excluding terminal and other-key jobs."""
    from tldw_Server_API.tests.Jobs._completion_characterization_support import rows

    helper = importlib.import_module(f"{DB_PACKAGE}.jobs_lifecycle_persistence")
    manager, conn, executor = helper_owner
    insert = (
        "INSERT INTO jobs(domain,queue,job_type,status,available_at) VALUES(%s,%s,%s,%s,%s)"
        if manager.backend == "postgres"
        else "INSERT INTO jobs(domain,queue,job_type,status,available_at) VALUES(?,?,?,?,?)"
    )
    for status, available in [
        ("queued", None),
        ("queued", None),
        ("queued", "2099-01-01"),
        ("processing", None),
        ("quarantined", None),
        ("completed", None),
    ]:
        executor.execute(insert, ("chatbooks", "default", "helper-counter", status, available))
    executor.execute(insert, ("chatbooks", "other", "helper-counter", "processing", None))
    executor.execute("DELETE FROM job_counters")
    helper._reconcile_lifecycle_counter_row(
        executor, backend=manager.backend, domain="chatbooks", queue="default", job_type="helper-counter"
    )
    executor.execute("SELECT * FROM job_counters")
    row = dict(executor.fetchone())
    assert tuple(row[name] for name in ("ready_count", "scheduled_count", "processing_count", "quarantined_count")) == (
        2,
        1,
        1,
        1,
    )
    assert executor.boundaries == []
    conn.rollback()
    assert rows(manager, "SELECT * FROM job_counters") == []
    assert rows(manager, "SELECT * FROM jobs") == []


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("failure", ["execute", "malformed"])
def test_counter_errors_propagate_without_owning_transaction(backend, failure):
    """Keep the original conversion and SQL exception precedence."""
    helper = importlib.import_module(f"{DB_PACKAGE}.jobs_lifecycle_persistence")
    error = RuntimeError("counter SQL probe")

    class BrokenCounts(ExecutorProbe):
        def execute(self, sql, params=()):
            if failure == "execute":
                raise error
            return self

        def fetchone(self):
            return {"ready_count": "bad"} if backend == "postgres" else ("bad", 0, 0, 0)

    executor = BrokenCounts(None)
    with pytest.raises(RuntimeError if failure == "execute" else ValueError) as caught:
        helper._reconcile_lifecycle_counter_row(executor, backend=backend, domain="d", queue="q", job_type="t")
    if failure == "execute":
        assert caught.value is error
    assert executor.boundaries == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "state,fail_at", [("INTRANS", None), ("INTRANS", "commit"), ("INERROR", None), ("INERROR", "rollback")]
)
def test_commit_guard_order_and_error_precedence(state, fail_at, monkeypatch):
    """Only an aborted state rolls back; rollback errors retain precedence."""
    from psycopg.pq import TransactionStatus

    import tldw_Server_API.app.core.Jobs.manager as manager

    helper = importlib.import_module(f"{DB_PACKAGE}.jobs_transaction_helpers")
    calls = []
    error = OSError("boundary probe")

    def boundary(name):
        calls.append(name)
        if fail_at == name:
            raise error

    monkeypatch.setattr(manager, "_run_post_commit_side_effects", lambda *args: calls.append("observer"))
    conn = SimpleNamespace(
        info=SimpleNamespace(transaction_status=getattr(TransactionStatus, state)),
        commit=lambda: boundary("commit"),
        rollback=lambda: boundary("rollback"),
    )
    if fail_at:
        with pytest.raises(OSError) as caught:
            helper._commit_postgres_transaction(conn, operation="helper probe")
        assert caught.value is error
    elif state == "INERROR":
        with pytest.raises(RuntimeError, match="Cannot commit PostgreSQL helper probe: transaction is aborted"):
            helper._commit_postgres_transaction(conn, operation="helper probe")
    else:
        helper._commit_postgres_transaction(conn, operation="helper probe")
    assert calls == (["rollback"] if state == "INERROR" else ["commit"])


@pytest.mark.integration
@pytest.mark.pg_jobs
def test_real_aborted_transaction_guard(jobs_pg_dsn, monkeypatch):
    """A native PostgreSQL diagnostic abort is rolled back, never committed."""
    import psycopg
    from psycopg.pq import TransactionStatus

    import tldw_Server_API.app.core.Jobs.manager as manager

    helper = importlib.import_module(f"{DB_PACKAGE}.jobs_transaction_helpers")
    observers = []
    monkeypatch.setattr(manager, "_run_post_commit_side_effects", lambda *args: observers.append(args))
    with psycopg.connect(jobs_pg_dsn) as conn:
        conn.execute("CREATE TEMP TABLE helper_guard(value INTEGER)")
        conn.commit()
        conn.execute("INSERT INTO helper_guard VALUES (1)")
        with pytest.raises(psycopg.errors.DivisionByZero):
            conn.execute("SELECT 1/0")
        assert conn.info.transaction_status == TransactionStatus.INERROR
        with pytest.raises(RuntimeError, match="transaction is aborted"):
            helper._commit_postgres_transaction(conn, operation="native probe")
        assert conn.info.transaction_status == TransactionStatus.IDLE
        assert conn.execute("SELECT * FROM helper_guard").fetchall() == []
    assert observers == []


def driver_import_probe(present: bool, manager_first: bool) -> None:
    """Run in a fresh interpreter so import order and optional binding are real."""
    original_import = builtins.__import__
    guard_imports = []
    manager_imports = []

    def import_driver(name, globals=None, locals=None, fromlist=(), level=0):
        if globals and globals.get("__name__") == "tldw_Server_API.app.core.Jobs.manager":
            if name == f"{DB_PACKAGE}.jobs_exception_policy":
                assert "tracing" in manager_imports
            manager_imports.append(name)
        if name == "psycopg.pq" and globals and globals.get("__name__") == f"{DB_PACKAGE}.jobs_transaction_helpers":
            guard_imports.append(tuple(fromlist))
        if not present and (name == "psycopg" or name.startswith("psycopg.")):
            raise ImportError("driver deliberately absent")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = import_driver
    try:
        assert not any(name.startswith("tldw_Server_API.app.core.Jobs") for name in sys.modules)
        if manager_first:
            importlib.import_module("tldw_Server_API.app.core.Jobs.manager")
        policy = importlib.import_module(f"{DB_PACKAGE}.jobs_exception_policy")
        persistence = importlib.import_module(f"{DB_PACKAGE}.jobs_lifecycle_persistence")
        transaction = importlib.import_module(f"{DB_PACKAGE}.jobs_transaction_helpers")
        if not manager_first:
            assert not any(name.startswith("tldw_Server_API.app.core.Jobs") for name in sys.modules)
        from tldw_Server_API.app.core.exceptions import BadRequestError

        expected = (
            AttributeError,
            BadRequestError,
            ConnectionError,
            ImportError,
            IndexError,
            KeyError,
            LookupError,
            OSError,
            RuntimeError,
            TimeoutError,
            TypeError,
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
            sqlite3.Error,
        )
        if present:
            import psycopg

            assert (psycopg.Error,) == policy._PG_ERRORS
            expected += (psycopg.Error,)
        else:
            assert policy._PG_ERRORS == ()
            assert not hasattr(policy, "psycopg")
        assert expected == policy._JOB_NONCRITICAL_EXCEPTIONS
        manager = importlib.import_module("tldw_Server_API.app.core.Jobs.manager")
        assert manager._PG_ERRORS is policy._PG_ERRORS
        assert manager._JOB_NONCRITICAL_EXCEPTIONS is policy._JOB_NONCRITICAL_EXCEPTIONS
        assert f"{DB_PACKAGE}.jobs_exception_policy" in manager_imports
        assert hasattr(manager, "psycopg") is present
        if present:
            assert manager.psycopg is policy.psycopg
        assert manager._insert_lifecycle_event is persistence._insert_lifecycle_event
        assert manager._commit_postgres_transaction is transaction._commit_postgres_transaction
        with tempfile.TemporaryDirectory() as directory:
            instance = manager.JobManager(Path(directory) / "cold-import.db", backend="sqlite")
            assert instance.backend == "sqlite"
            created = instance.create_job(
                domain="chatbooks", queue="default", job_type="cold-import", payload={}, owner_user_id=None
            )
            assert instance.get_job(created["id"])["status"] == "queued"
        assert guard_imports == []
        if present:
            conn = SimpleNamespace(info=SimpleNamespace(transaction_status=0), commit=lambda: None)
            transaction._commit_postgres_transaction(conn, operation="cold probe")
        else:
            try:
                transaction._commit_postgres_transaction(None, operation="cold probe")
            except ImportError as error:
                assert str(error) == "driver deliberately absent"
            else:
                raise AssertionError("guard unexpectedly imported absent driver")
        assert guard_imports == [("TransactionStatus",)]
    finally:
        builtins.__import__ = original_import


@pytest.mark.unit
@pytest.mark.parametrize("present", [False, True], ids=["absent-driver", "present-driver"])
@pytest.mark.parametrize("manager_first", [False, True], ids=["helpers-first", "manager-first"])
def test_isolated_driver_imports(present, manager_first):
    """Spawned subprocesses must not inherit pytest's imported driver/modules."""
    process = multiprocessing.get_context("spawn").Process(target=driver_import_probe, args=(present, manager_first))
    process.start()
    process.join(60)
    if process.is_alive():
        process.terminate()
        process.join(10)
        pytest.fail("cold-import subprocess timed out")
    assert process.exitcode == 0

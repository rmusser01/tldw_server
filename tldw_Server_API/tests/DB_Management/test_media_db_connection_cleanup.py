from __future__ import annotations

import builtins
import sqlite3
import sys
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from types import MethodType
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseBackend
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime import execution_ops as media_db_execution_ops


class _FailingCursor:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.rowcount = -1
        self.lastrowid = None
        self.description = None
        self.closed = False

    def execute(self, *args, **kwargs):
        raise self._exc

    def executemany(self, *args, **kwargs):
        raise self._exc

    def close(self) -> None:
        self.closed = True


class _FailingConnection:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.closed = False
        self.row_factory = None
        self._cursor: _FailingCursor | None = None

    def execute(self, *args, **kwargs):
        return None

    def cursor(self) -> _FailingCursor:
        self._cursor = _FailingCursor(self._exc)
        return self._cursor

    def commit(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_postgres_get_connection_reapplies_scope() -> None:
    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    backend = MagicMock()
    pool = MagicMock()
    conn = object()
    pool.get_connection.return_value = conn
    backend.get_pool.return_value = pool
    backend.apply_scope = MagicMock()
    db.backend = backend
    db._txn_conn_var = ContextVar("test_txn_conn", default=None)
    db._persistent_conn_var = ContextVar("test_persistent_conn", default=None)

    first = MediaDatabase.get_connection(db)
    second = MediaDatabase.get_connection(db)

    assert first is conn
    assert second is conn
    pool.get_connection.assert_called_once()
    assert backend.apply_scope.call_count == 2
    backend.apply_scope.assert_called_with(conn)


def test_execute_query_closes_ephemeral_connection_on_error(tmp_path: Path, monkeypatch) -> None:
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="cleanup-test")
    failing_conn = _FailingConnection(sqlite3.OperationalError("boom"))
    monkeypatch.setattr(media_db_execution_ops.sqlite3, "connect", lambda *args, **kwargs: failing_conn)

    with pytest.raises(DatabaseError):
        db.execute_query("SELECT 1")

    assert failing_conn.closed is True


def test_execute_query_passthroughs_sync_integrity_error(tmp_path: Path, monkeypatch) -> None:
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="cleanup-test")
    failing_conn = _FailingConnection(sqlite3.IntegrityError("sync error: boom"))
    monkeypatch.setattr(media_db_execution_ops.sqlite3, "connect", lambda *args, **kwargs: failing_conn)

    with pytest.raises(sqlite3.IntegrityError, match="sync error"):
        db.execute_query("SELECT 1")

    assert failing_conn.closed is True


def test_execute_many_closes_ephemeral_connection_on_error(tmp_path: Path, monkeypatch) -> None:
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="cleanup-test")
    failing_conn = _FailingConnection(sqlite3.OperationalError("boom"))
    monkeypatch.setattr(media_db_execution_ops.sqlite3, "connect", lambda *args, **kwargs: failing_conn)

    with pytest.raises(DatabaseError):
        db.execute_many(
            "INSERT INTO Media (uuid, title, type, content_hash, last_modified, version, client_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [("uuid", "title", "text", "hash", "now", 1, "client")],
        )

    assert failing_conn.closed is True


def test_execute_many_returns_none_for_empty_params_list(tmp_path: Path) -> None:
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="cleanup-test")

    result = db.execute_many(
        "INSERT INTO Media (uuid, title, type, content_hash, last_modified, version, client_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [],
    )

    assert result is None


def test_execute_many_rejects_non_list_params_list(tmp_path: Path) -> None:
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="cleanup-test")

    with pytest.raises(TypeError, match="params_list must be a list"):
        db.execute_many(
            "INSERT INTO Media (uuid, title, type, content_hash, last_modified, version, client_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("uuid", "title"),
        )


@pytest.mark.parametrize("grouped", [False, True])
def test_sqlite_transaction_tolerates_row_factory_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grouped: bool,
) -> None:
    if grouped and sys.version_info < (3, 12):
        pytest.skip("contextlib.suppress supports exception groups on Python 3.12+")

    class RowFactoryFailure(sqlite3.Connection):
        def __setattr__(self, name: str, value: object) -> None:
            if name == "row_factory":
                error = RuntimeError("row factory fault")
                raise builtins.ExceptionGroup("cleanup", [error]) if grouped else error
            super().__setattr__(name, value)

    db = MediaDatabase(tmp_path / "row-factory.db", client_id="cleanup-test")
    conn = sqlite3.connect(db.db_path_str, isolation_level=None, factory=RowFactoryFailure)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(db.backend, "connect", lambda: conn)
            with db.transaction() as active:
                assert active is conn
                assert active.execute("SELECT 1").fetchone() == (1,)
        assert db._get_tx_depth() == 0
        assert db._get_txn_conn() is None
    finally:
        sqlite3.Connection.close(conn)
        db.close_connection()
        db.backend.get_pool().close_all()


@pytest.mark.parametrize("body_fails", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_sqlite_transaction_tolerates_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body_fails: bool,
    grouped: bool,
) -> None:
    if grouped and sys.version_info < (3, 12):
        pytest.skip("contextlib.suppress supports exception groups on Python 3.12+")
    close_calls: list[sqlite3.Connection] = []

    class CloseFailure(sqlite3.Connection):
        def close(self) -> None:
            close_calls.append(self)
            super().close()
            error = RuntimeError("close fault")
            raise builtins.ExceptionGroup("cleanup", [error]) if grouped else error

    db = MediaDatabase(tmp_path / "close.db", client_id="cleanup-test")
    conn = sqlite3.connect(db.db_path_str, isolation_level=None, factory=CloseFailure)
    body_error = ValueError("body fault")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(db.backend, "connect", lambda: conn)
            if body_fails:
                with pytest.raises(ValueError, match="body fault") as raised:
                    with db.transaction() as active:
                        assert active is conn
                        raise body_error
                assert raised.value is body_error
            else:
                with db.transaction() as active:
                    assert active.execute("SELECT 1").fetchone()[0] == 1
        assert close_calls == [conn]
        assert db._get_tx_depth() == 0
        assert db._get_txn_conn() is None
    finally:
        sqlite3.Connection.close(conn)
        db.close_connection()
        db.backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.parametrize("grouped", [False, True])
def test_postgres_transaction_tolerates_pool_release_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pg_restricted_backend: DatabaseBackend,
    grouped: bool,
) -> None:
    if grouped and sys.version_info < (3, 12):
        pytest.skip("contextlib.suppress supports exception groups on Python 3.12+")
    db = MediaDatabase(tmp_path / "postgres-release.db", client_id="1", backend=pg_restricted_backend)
    pool = db.backend.get_pool()
    original_return = pool.return_connection
    returned: list[object] = []

    def fail_after_return(connection: object) -> None:
        original_return(connection)
        returned.append(connection)
        error = RuntimeError("pool release fault")
        raise builtins.ExceptionGroup("cleanup", [error]) if grouped else error

    try:
        with monkeypatch.context() as patch:
            patch.setattr(pool, "return_connection", fail_after_return)
            with db.transaction() as active:
                assert active.execute("SELECT 1 AS value").fetchone()["value"] == 1
        assert returned == [active]
        assert db._get_tx_depth() == 0
        assert db._get_txn_conn() is None
        with db.transaction() as reused:
            assert reused.execute("SELECT 1 AS value").fetchone()["value"] == 1
    finally:
        db.close_connection()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.parametrize("fault", ["known", "known_group", "unexpected", "mixed_group"])
def test_postgres_get_connection_preserves_scope_exception_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pg_restricted_backend: DatabaseBackend,
    fault: str,
) -> None:
    if fault.endswith("group") and sys.version_info < (3, 12):
        pytest.skip("contextlib.suppress supports exception groups on Python 3.12+")

    class UnexpectedScopeError(Exception):
        pass

    db = MediaDatabase(tmp_path / "postgres-scope.db", client_id="1", backend=pg_restricted_backend)
    original_apply = db.backend.apply_scope
    scoped: list[object] = []
    known_error = RuntimeError("scope callback fault")
    unexpected_error = UnexpectedScopeError("unexpected scope fault")
    scope_error: Exception = known_error
    if fault == "known_group":
        scope_error = builtins.ExceptionGroup("scope", [known_error])
    elif fault == "unexpected":
        scope_error = unexpected_error
    elif fault == "mixed_group":
        scope_error = builtins.ExceptionGroup("scope", [known_error, unexpected_error])

    def fail_after_scope(connection: object = None) -> None:
        original_apply(connection)
        scoped.append(connection)
        raise scope_error

    try:
        with monkeypatch.context() as patch:
            patch.setattr(db.backend, "apply_scope", fail_after_scope)
            for _ in range(2):
                if fault == "unexpected":
                    with pytest.raises(UnexpectedScopeError) as raised:
                        db.get_connection()
                    assert raised.value is unexpected_error
                elif fault.endswith("group"):
                    with pytest.raises(builtins.ExceptionGroup) as grouped_raised:
                        db.get_connection()
                    assert grouped_raised.value is scope_error
                    expected_leaves = (known_error, unexpected_error) if fault == "mixed_group" else (known_error,)
                    assert grouped_raised.value.exceptions == expected_leaves
                    assert all(
                        actual is expected for actual, expected in zip(grouped_raised.value.exceptions, expected_leaves)
                    )
                else:
                    assert db.get_connection() is db._get_persistent_conn()
            cached = db._get_persistent_conn()
            assert cached is not None
            assert scoped == [cached, cached]
        db.close_connection()
        assert db._get_persistent_conn() is None
        assert db._get_tx_depth() == 0
        assert db._get_txn_conn() is None
        with db.transaction() as reused:
            flags = reused.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).fetchone()
            assert not flags["rolsuper"]
            assert not flags["rolbypassrls"]
    finally:
        db.close_connection()


def test_native_media_subclass_preserves_static_and_cleanup_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DerivedMediaDatabase(MediaDatabase):
        pass

    original_close: Callable[[MediaDatabase], None] = MediaDatabase.close_connection
    receivers: list[MediaDatabase] = []

    def record_close(self: MediaDatabase) -> None:
        receivers.append(self)
        original_close(self)

    db = DerivedMediaDatabase(tmp_path / "subclass.db", client_id="cleanup-test")
    try:
        bound_close = db.close_connection
        assert isinstance(bound_close, MethodType)
        assert bound_close.__self__ is db
        assert isinstance(db, DerivedMediaDatabase)
        labels = " Work,work, Research , ,RESEARCH"
        expected = {"work": "Work", "research": "Research"}
        assert DerivedMediaDatabase._normalize_email_label_values(labels) == expected
        assert db._normalize_email_label_values(labels) == expected
        with monkeypatch.context() as patch:
            patch.setattr(DerivedMediaDatabase, "close_connection", record_close)
            db.close_connection()
        assert receivers == [db]
        assert DerivedMediaDatabase.close_connection is original_close
        db.close_connection()
        assert receivers == [db]
    finally:
        original_close(db)
        db.backend.get_pool().close_all()

"""Backends must signal WHICH failure occurred, without revealing anything about it.

Both SQL backends deliberately redact driver errors: the raise sits outside the except
block behind a `redacted_failure` flag, so the original exception is never chained and
its message -- which can carry query text and bound parameters -- cannot reach a
traceback. Several tests pin the resulting strings with anchored patterns
(test_media_postgres_support.py, test_postgres_unique_conflict.py) precisely to assert
that nothing is appended.

That left callers unable to tell a constraint violation from a disk error or a locked
database without re-running the query by hand. The TYPE now carries that distinction,
which is the mechanism this module already used for uniqueness.

These tests hold both halves at once: the classification is available, and the
redaction is intact.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    ConstraintViolationError,
    DatabaseError,
    TransientContentionError,
    UniqueConstraintError,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase

_SENSITIVE = "PARAMETER-THAT-MUST-NOT-LEAK"


@pytest.fixture
def sync_db() -> SyncDatabase:
    return SyncDatabase(sqlite_path=Path(tempfile.mkdtemp()) / "constraints.sqlite")


def test_uniqueness_is_a_constraint_violation() -> None:
    """Widening UniqueConstraintError's base must not break `except` clauses."""
    assert issubclass(UniqueConstraintError, ConstraintViolationError)
    assert issubclass(UniqueConstraintError, DatabaseError)
    assert issubclass(ConstraintViolationError, DatabaseError)


def test_a_constraint_violation_is_typed_as_one(sync_db: SyncDatabase) -> None:
    with pytest.raises(ConstraintViolationError):
        sync_db.execute(
            "INSERT INTO sync_datasets (dataset_id) VALUES (?)", ("missing-columns",)
        )


def test_a_constraint_violation_is_still_a_database_error(sync_db: SyncDatabase) -> None:
    """Existing callers catch DatabaseError and must keep working."""
    with pytest.raises(DatabaseError):
        sync_db.execute(
            "INSERT INTO sync_datasets (dataset_id) VALUES (?)", ("missing-columns",)
        )


def test_a_non_constraint_failure_stays_a_plain_database_error(
    sync_db: SyncDatabase,
) -> None:
    """A syntax error is not a constraint violation, so it must not be typed as one."""
    with pytest.raises(DatabaseError) as exc_info:
        sync_db.execute("SELECT * FROM a_table_that_does_not_exist")
    assert not isinstance(exc_info.value, ConstraintViolationError)


def test_the_message_is_unchanged_and_reveals_nothing(sync_db: SyncDatabase) -> None:
    """The redaction is the reason the type exists; it must survive the change."""
    with pytest.raises(DatabaseError) as exc_info:
        sync_db.execute(
            "INSERT INTO sync_datasets (dataset_id, owner_user_id) VALUES (?, ?)",
            ("dataset-1", _SENSITIVE),
        )
    error = exc_info.value
    assert str(error) == "SQLite query execution failed"
    assert _SENSITIVE not in str(error)
    assert "INSERT INTO" not in str(error)
    # The driver exception is never chained, so nothing reaches a traceback either.
    assert error.__cause__ is None
    assert error.__suppress_context__ or error.__context__ is None


def test_sqlite_integrity_error_is_the_classification_source() -> None:
    """Pin the driver class this maps from, so the mapping is not silently widened."""
    assert issubclass(sqlite3.IntegrityError, sqlite3.Error)
    assert not issubclass(sqlite3.OperationalError, sqlite3.IntegrityError)


# --- Contention (TASK-13319) ---------------------------------------------------------
# Same mechanism for a different class: lock or serialization contention that retrying
# the whole transaction can clear. Without the type, PostgreSQL's 40001/40P01 reached
# callers as a plain DatabaseError and no retry policy could recognise it.


def test_contention_is_a_database_error_but_not_a_constraint_violation() -> None:
    assert issubclass(TransientContentionError, DatabaseError)
    assert not issubclass(TransientContentionError, ConstraintViolationError)


def test_a_locked_sqlite_database_is_typed_as_contention(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
    from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend

    path = tmp_path / "locked.sqlite"
    holder = sqlite3.connect(path)
    holder.execute("CREATE TABLE t (x INTEGER)")
    holder.commit()
    holder.execute("BEGIN EXCLUSIVE")
    try:
        backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(path)))
        conn = sqlite3.connect(path, timeout=0)
        with pytest.raises(TransientContentionError) as exc_info:
            backend.execute("INSERT INTO t (x) VALUES (?)", (_SENSITIVE,), connection=conn)
        assert str(exc_info.value) == "SQLite query execution failed"
        assert exc_info.value.__cause__ is None
    finally:
        holder.rollback()
        holder.close()


@pytest.mark.parametrize(
    ("sqlstate", "expected"),
    [
        ("40001", TransientContentionError),  # serialization_failure
        ("40P01", TransientContentionError),  # deadlock_detected
        ("55P03", TransientContentionError),  # lock_not_available
        ("23505", UniqueConstraintError),
        ("42P01", DatabaseError),  # undefined_table: not retryable
    ],
)
def test_postgres_sqlstate_classification(monkeypatch, sqlstate, expected) -> None:
    from tldw_Server_API.app.core.DB_Management.backends import postgresql_backend as pg

    class _DriverError(Exception):
        pass

    driver_error = _DriverError(f"driver says {_SENSITIVE}")
    driver_error.sqlstate = sqlstate  # type: ignore[attr-defined]
    monkeypatch.setattr(pg, "_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS", (_DriverError,))
    monkeypatch.setattr(pg, "_PSYCOPG_DRIVER_EXCEPTIONS", (_DriverError,))

    class _Cursor:
        def execute(self, *_a, **_k):
            raise driver_error

    class _Conn:
        def cursor(self, *_a, **_k):
            return _Cursor()

        def rollback(self):
            pass

    backend = object.__new__(pg.PostgreSQLBackend)
    with pytest.raises(DatabaseError) as exc_info:
        backend.execute("SELECT 1", connection=_Conn())
    assert type(exc_info.value) is expected
    assert str(exc_info.value) == "PostgreSQL query execution failed"
    assert exc_info.value.__cause__ is None

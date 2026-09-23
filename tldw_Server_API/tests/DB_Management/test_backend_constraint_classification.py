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

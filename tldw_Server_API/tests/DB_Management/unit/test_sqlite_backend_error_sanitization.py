from __future__ import annotations

import io
import sqlite3

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend


class _CursorRaises:
    def execute(self, query: str, params: object = None) -> None:
        del query, params
        raise sqlite3.IntegrityError(
            "private@example.com constraint=users_email_key"
        )

    def executemany(self, query: str, params: object) -> None:
        del query, params
        raise sqlite3.IntegrityError(
            "private@example.com constraint=users_email_key"
        )


class _Connection:
    def cursor(self) -> _CursorRaises:
        return _CursorRaises()


def test_execute_sanitizes_sqlite_driver_errors_and_logs() -> None:
    backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE))
    sentinel = "private@example.com constraint=users_email_key"
    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(DatabaseError) as raised:
            backend.execute("UPDATE users SET email = ?", (sentinel,), _Connection())
    finally:
        logger.remove(sink)

    assert str(raised.value) == "SQLite query execution failed"
    assert raised.value.__cause__ is None
    assert sentinel not in output.getvalue()


def test_execute_many_sanitizes_sqlite_driver_errors_and_logs() -> None:
    backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE))
    sentinel = "private@example.com constraint=users_email_key"
    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(DatabaseError) as raised:
            backend.execute_many("UPDATE users SET email = ?", [(sentinel,)], _Connection())
    finally:
        logger.remove(sink)

    assert str(raised.value) == "SQLite batch execution failed"
    assert raised.value.__cause__ is None
    assert sentinel not in output.getvalue()


class _TransactionConnection:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.in_transaction = False
        self.fail_commit = fail_commit
        self.statements: list[str] = []

    def execute(self, statement: str) -> None:
        self.statements.append(statement)
        if statement == "BEGIN IMMEDIATE":
            self.in_transaction = True
        elif statement == "COMMIT":
            if self.fail_commit:
                raise sqlite3.IntegrityError("private transaction detail")
            self.in_transaction = False
        elif statement == "ROLLBACK":
            self.in_transaction = False


def test_transaction_preserves_base_exception_and_rolls_back() -> None:
    class ControlSignal(BaseException):
        pass

    backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE))
    conn = _TransactionConnection()
    signal = ControlSignal()

    with pytest.raises(ControlSignal) as raised:
        with backend.transaction(connection=conn):  # type: ignore[arg-type]
            raise signal

    assert raised.value is signal
    assert conn.statements == ["BEGIN IMMEDIATE", "ROLLBACK"]


def test_transaction_sanitizes_commit_failure_and_rolls_back() -> None:
    backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE))
    conn = _TransactionConnection(fail_commit=True)
    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(DatabaseError) as raised:
            with backend.transaction(connection=conn):  # type: ignore[arg-type]
                pass
    finally:
        logger.remove(sink)

    assert str(raised.value) == "SQLite transaction commit failed"
    assert raised.value.__cause__ is None
    assert conn.statements == ["BEGIN IMMEDIATE", "COMMIT", "ROLLBACK"]
    assert "private transaction detail" not in output.getvalue()

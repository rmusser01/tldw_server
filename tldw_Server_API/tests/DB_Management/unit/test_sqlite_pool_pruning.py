"""Tests for SQLite pool cleanup and connection ownership."""

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteConnectionPool


@pytest.fixture
def sqlite_pool(tmp_path: Path) -> Iterator[SQLiteConnectionPool]:
    """Provide an isolated SQLite pool and close all its connections after each test."""
    db_path = tmp_path / "invalidation.db"
    config = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    pool = SQLiteConnectionPool(str(db_path), config)
    try:
        yield pool
    finally:
        pool.close_all()


@pytest.mark.unit
@pytest.mark.parametrize("connection_kind", ["current", "stale", "external"], ids=[
    "invalidate-current-handle", "preserve-replacement-handle", "preserve-unrelated-handle",
])
def test_sqlite_pool_invalidation_closes_only_the_supplied_connection(
    sqlite_pool: SQLiteConnectionPool, connection_kind: str,
) -> None:
    """Invalidation clears the matching cache entry without closing a different live handle."""
    current = sqlite_pool.get_connection()
    supplied = current
    if connection_kind == "stale":
        sqlite_pool.clear_thread_local_connection()
        current = sqlite_pool.get_connection()
    elif connection_kind == "external":
        supplied = sqlite3.connect(sqlite_pool.db_path)
    try:
        sqlite_pool.invalidate_connection(supplied)
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            supplied.execute("SELECT 1")
        if connection_kind == "current":
            assert sqlite_pool.get_stats()["active_connections"] == 0
            replacement = sqlite_pool.get_connection()
            assert replacement is not supplied
            assert replacement.execute("SELECT 1").fetchone()[0] == 1
        else:
            assert sqlite_pool.get_connection() is current
            assert current.execute("SELECT 1").fetchone()[0] == 1
    finally:
        supplied.close()


@pytest.mark.unit
@pytest.mark.parametrize("close_error", [
    sqlite3.OperationalError, OSError, RuntimeError, TypeError, ValueError, KeyboardInterrupt, SystemExit,
], ids=[
    "sqlite-close-error", "os-close-error", "runtime-close-error", "type-close-error", "value-close-error",
    "keyboard-interrupt", "system-exit",
])
@pytest.mark.parametrize("operation", ["invalidate_connection", "clear_thread_local_connection"], ids=[
    "invalidate-checkout", "clear-current-checkout",
])
def test_sqlite_pool_logs_close_failure_and_detaches_rejected_handle(
    sqlite_pool: SQLiteConnectionPool,
    monkeypatch: pytest.MonkeyPatch,
    close_error: type[BaseException],
    operation: str,
) -> None:
    """Close failures never permit reuse; ordinary errors log and interruptions propagate."""
    close_attempted = False

    class CloseFailureConnection(sqlite3.Connection):
        """Simulate a driver close failure while keeping real SQLite queries."""

        def close(self) -> None:
            """Fail the first close and allow later test teardown to release resources."""
            nonlocal close_attempted
            if not close_attempted:
                close_attempted = True
                raise close_error("close unavailable")
            super().close()

    failed = sqlite3.connect(sqlite_pool.db_path, factory=CloseFailureConnection)
    messages = []
    sink = logger.add(messages.append, level="WARNING")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sqlite_pool, "_create_connection", lambda: failed)
            checkout = sqlite_pool.get_connection()
        ordinary_error = issubclass(close_error, Exception)
        expected = nullcontext() if ordinary_error else pytest.raises(close_error, match="close unavailable")
        with expected:
            if operation == "invalidate_connection":
                sqlite_pool.invalidate_connection(checkout)
            else:
                sqlite_pool.clear_thread_local_connection()
        assert sqlite_pool.get_stats()["active_connections"] == 0
        replacement = sqlite_pool.get_connection()
        assert replacement is not failed
        assert replacement.execute("SELECT 1").fetchone()[0] == 1
        records = [message.record for message in messages if "Failed to close" in message.record["message"]]
        if ordinary_error:
            assert len(records) == 1
            assert records[0]["extra"]["connection_id"] == id(failed)
            assert records[0]["extra"]["thread_id"] == threading.get_ident()
            assert records[0]["exception"].type is close_error
            assert records[0]["exception"].traceback is not None
        else:
            assert records == []
    finally:
        logger.remove(sink)
        sqlite3.Connection.close(failed)


def test_sqlite_pool_prunes_dead_threads(tmp_path):
    db_path = tmp_path / "pool_test.db"
    config = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    pool = SQLiteConnectionPool(str(db_path), config)

    def worker():
        conn = pool.get_connection()
        conn.execute("SELECT 1")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    pool.get_connection()
    stats = pool.get_stats()
    assert stats["total_connections"] == 1
    pool.close_all()

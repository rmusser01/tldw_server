from __future__ import annotations

from collections.abc import Iterable
import sqlite3

import pytest


class _Row(tuple):
    pass


class _RecordingConnection:
    def __init__(self, *, database_name: str = "/tmp/test.db") -> None:
        self.database_name = database_name
        self.in_transaction = False
        self.statements: list[str] = []

    def execute(self, sql: str):
        self.statements.append(sql)
        normalized = sql.strip().upper()
        if normalized == "PRAGMA DATABASE_LIST":
            return [_Row((0, "main", self.database_name))]
        if normalized.startswith("BEGIN"):
            self.in_transaction = True
        return []


def _non_probe_statements(conn: _RecordingConnection) -> list[str]:
    return [sql for sql in conn.statements if sql != "PRAGMA database_list"]


class _AsyncCursor:
    def __init__(self, rows: list[_Row]) -> None:
        self._rows = rows

    async def fetchall(self):
        return self._rows


class _RecordingAsyncConnection:
    def __init__(self, *, database_name: str = "/tmp/test.db") -> None:
        self.database_name = database_name
        self.statements: list[str] = []

    async def execute(self, sql: str):
        self.statements.append(sql)
        normalized = sql.strip().upper()
        if normalized == "PRAGMA DATABASE_LIST":
            return _AsyncCursor([_Row((0, "main", self.database_name))])
        return _AsyncCursor([])


def _non_probe_async_statements(conn: _RecordingAsyncConnection) -> list[str]:
    return [sql for sql in conn.statements if sql != "PRAGMA database_list"]


def test_configure_sqlite_connection_applies_standard_pragmas():
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection

    conn = _RecordingConnection()

    configure_sqlite_connection(
        conn,
        cache_size=-2048,
        busy_timeout_ms=7000,
    )

    assert _non_probe_statements(conn) == [
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA foreign_keys=ON",
        "PRAGMA busy_timeout=7000",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA cache_size=-2048",
    ]


def test_configure_sqlite_connection_skips_wal_for_in_memory_by_default():
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection

    conn = _RecordingConnection(database_name="")

    configure_sqlite_connection(conn)

    assert "PRAGMA journal_mode=WAL" not in conn.statements
    assert _non_probe_statements(conn) == [
        "PRAGMA synchronous=NORMAL",
        "PRAGMA foreign_keys=ON",
        "PRAGMA busy_timeout=5000",
        "PRAGMA temp_store=MEMORY",
    ]


def test_run_sqlite_quick_check_reads_existing_database(tmp_path):
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import run_sqlite_quick_check

    db_path = tmp_path / "healthy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY)")

    assert run_sqlite_quick_check(db_path) == ["ok"]


def test_run_sqlite_quick_check_skips_missing_database(tmp_path):
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import run_sqlite_quick_check

    assert run_sqlite_quick_check(tmp_path / "missing.db") == []


def test_begin_immediate_if_needed_only_starts_outermost_transaction():
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import begin_immediate_if_needed

    conn = _RecordingConnection()

    assert begin_immediate_if_needed(conn) is True
    assert begin_immediate_if_needed(conn) is False
    assert conn.statements == ["BEGIN IMMEDIATE"]


@pytest.mark.asyncio
async def test_configure_sqlite_connection_async_applies_standard_pragmas():
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection_async

    conn = _RecordingAsyncConnection()

    await configure_sqlite_connection_async(
        conn,
        cache_size=-1024,
        busy_timeout_ms=8000,
    )

    assert _non_probe_async_statements(conn) == [
        "PRAGMA journal_mode=WAL",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA foreign_keys=ON",
        "PRAGMA busy_timeout=8000",
        "PRAGMA temp_store=MEMORY",
        "PRAGMA cache_size=-1024",
    ]

from __future__ import annotations

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


class _LockedOnceWalConnection(_RecordingConnection):
    def execute(self, sql: str):
        if sql == "PRAGMA journal_mode=WAL" and sql not in self.statements:
            self.statements.append(sql)
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        return super().execute(sql)


def test_configure_sqlite_connection_retries_wal_switch_when_locked():
    # SQLite returns SQLITE_BUSY for the rollback->WAL switch without calling the
    # busy handler when another connection is writing; concurrent first opens of a
    # legacy DB must retry instead of failing with "database is locked".
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection

    conn = _LockedOnceWalConnection()
    configure_sqlite_connection(conn)

    assert _non_probe_statements(conn).count("PRAGMA journal_mode=WAL") == 2


def test_configure_sqlite_connection_gives_up_wal_switch_after_busy_timeout():
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection

    class _AlwaysLocked(_RecordingConnection):
        def execute(self, sql: str):
            if sql == "PRAGMA journal_mode=WAL":
                error = sqlite3.OperationalError("database is locked")
                error.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise error
            return super().execute(sql)

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        configure_sqlite_connection(_AlwaysLocked(), busy_timeout_ms=30)


@pytest.mark.asyncio
async def test_async_wal_setup_retries_sqlite_busy_once():
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection_async

    class BusyOnce(_RecordingAsyncConnection):
        busy = True

        async def execute(self, sql: str):
            if sql == "PRAGMA journal_mode=WAL" and self.busy:
                self.busy = False
                error = sqlite3.OperationalError("database is locked")
                error.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise error
            return await super().execute(sql)

    conn = BusyOnce()
    await configure_sqlite_connection_async(conn, busy_timeout_ms=100)

    assert not conn.busy
    assert "PRAGMA journal_mode=WAL" in conn.statements


def test_wal_setup_preserves_non_lock_sqlite_errors():
    from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection

    class Broken(_RecordingConnection):
        wal_attempts = 0

        def execute(self, sql: str):
            if sql == "PRAGMA journal_mode=WAL":
                self.wal_attempts += 1
                error = sqlite3.OperationalError("disk I/O error")
                error.sqlite_errorcode = sqlite3.SQLITE_IOERR
                raise error
            return super().execute(sql)

    conn = Broken()
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        configure_sqlite_connection(conn)
    assert conn.wal_attempts == 1

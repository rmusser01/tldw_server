"""Schema scripts retain SQLite transaction ownership and script syntax."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig, DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend


@pytest.fixture()
def sqlite_backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "schema.db")))
    try:
        yield backend
    finally:
        backend.get_pool().close_all()


@pytest.mark.parametrize("transactional", [False, True])
@pytest.mark.parametrize("supply_connection", [False, True])
def test_schema_script_preserves_caller_ownership(
    sqlite_backend: SQLiteBackend,
    transactional: bool,
    supply_connection: bool,
) -> None:
    connection = sqlite_backend.get_pool().get_connection()
    schema = """
    CREATE TABLE "script;source" (value TEXT); CREATE TABLE "script;log" (value TEXT);
    CREATE TRIGGER "script;copy" AFTER INSERT ON "script;source"
    BEGIN
        INSERT INTO "script;log" VALUES ('trigger;first');
        INSERT INTO "script;log" VALUES (NEW.value || ';second');
    END;
    INSERT INTO "script;source" VALUES ('quoted;value')
    -- The final statement has no delimiter; this comment contains a semicolon.
    """

    if transactional:
        connection.execute("BEGIN IMMEDIATE")
    sqlite_backend.create_tables(schema, connection=connection if supply_connection else None)
    assert sqlite_backend.execute('SELECT value FROM "script;log" ORDER BY rowid').rows == [
        {"value": "trigger;first"},
        {"value": "quoted;value;second"},
    ]
    if transactional:
        assert connection.in_transaction
        connection.rollback()
        assert not sqlite_backend.table_exists("script;source")
        assert not sqlite_backend.table_exists("script;log")
    else:
        observer = sqlite_backend.connect()
        try:
            assert [
                row["value"] for row in observer.execute('SELECT value FROM "script;log" ORDER BY rowid').fetchall()
            ] == ["trigger;first", "quoted;value;second"]
        finally:
            sqlite_backend.disconnect(observer)


def test_schema_failure_rolls_back_caller_writes(
    sqlite_backend: SQLiteBackend,
) -> None:
    sqlite_backend.create_tables("CREATE TABLE retained (value TEXT)")

    with pytest.raises(DatabaseError, match="Failed to create schema"):
        with sqlite_backend.transaction() as connection:
            connection.execute("INSERT INTO retained VALUES ('pending')")
            sqlite_backend.create_tables(
                "CREATE TABLE partial (value TEXT); INVALID SQL",
                connection=connection,
            )

    assert sqlite_backend.execute("SELECT value FROM retained").rows == []
    assert not sqlite_backend.table_exists("partial")

"""PostgreSQL reservation provenance DDL emitted by the usage bootstrap."""

import sqlite3
from contextlib import asynccontextmanager

import pytest
import sqlglot
from sqlglot import exp

from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_usage_tables_pg

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("schema_path", ["new", "provisional"])
async def test_pg_reservation_owner_is_nullable_on_both_bootstrap_paths(schema_path):
    statements = []

    class RecordingPool:
        pool = object()

        @asynccontextmanager
        async def transaction(self):
            yield self

        async def execute(self, sql, *_params):
            statements.append(sql)

    assert await ensure_usage_tables_pg(RecordingPool())
    columns = []
    for sql in statements:
        if "provider_usage_reservations" not in sql:
            continue
        statement = sqlglot.parse_one(sql, read="postgres")
        statement_type = exp.Create if schema_path == "new" else exp.Alter
        if isinstance(statement, statement_type):
            columns.extend(column for column in statement.find_all(exp.ColumnDef) if column.name == "insertion_owner")
    assert len(columns) == 1
    column = columns[0].copy()
    if schema_path == "provisional":
        assert column.args.get("exists") is True
    column.set("exists", False)
    table = exp.Create(this=exp.Schema(this=exp.to_table("owner_values"), expressions=[column]), kind="TABLE")
    with sqlite3.connect(":memory:") as connection:
        connection.execute(table.sql(dialect="sqlite"))
        assert connection.execute("PRAGMA table_info(owner_values)").fetchone()[2:5] == ("TEXT", 0, None)
        connection.execute("INSERT INTO owner_values DEFAULT VALUES")
        assert connection.execute("SELECT insertion_owner FROM owner_values").fetchall() == [(None,)]


@pytest.mark.parametrize("schema_path", ["new", "provisional"])
async def test_pg_reservation_source_is_nullable_and_rejects_unknown_provenance(
    schema_path: str,
) -> None:
    """Both bootstrap paths emit the same constrained, optional TEXT field."""
    statements: list[str] = []

    class _RecordingPool:
        pool = object()

        @asynccontextmanager
        async def transaction(self):
            yield self

        async def execute(self, sql: str, *_params: object) -> None:
            statements.append(sql)

    assert await ensure_usage_tables_pg(_RecordingPool())
    columns: list[exp.ColumnDef] = []
    for sql in statements:
        if "provider_usage_reservations" not in sql:
            continue
        statement = sqlglot.parse_one(sql, read="postgres")
        statement_type = exp.Create if schema_path == "new" else exp.Alter
        if not isinstance(statement, statement_type):
            continue
        columns.extend(column for column in statement.find_all(exp.ColumnDef) if column.name == "credential_source")
    assert len(columns) == 1, f"missing credential_source DDL for {schema_path} reservations"
    column = columns[0]
    if schema_path == "provisional":
        assert column.args.get("exists") is True
    column = column.copy()
    column.set("exists", False)

    # Exercise the emitted field's CHECK, not an independently rebuilt constraint.
    table = exp.Create(
        this=exp.Schema(this=exp.to_table("source_values"), expressions=[column]),
        kind="TABLE",
    )
    with sqlite3.connect(":memory:") as connection:
        connection.execute(table.sql(dialect="sqlite"))
        column_info = connection.execute("PRAGMA table_info(source_values)").fetchone()
        assert column_info[2:5] == ("TEXT", 0, None)
        connection.execute("INSERT INTO source_values DEFAULT VALUES")
        allowed = ["user", "team", "org", "server_default", "user_unmetered", None]
        for source in allowed:
            connection.execute("INSERT INTO source_values VALUES (?)", (source,))
        for source in ("unknown", "none", "", "USER", "user "):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute("INSERT INTO source_values VALUES (?)", (source,))
        assert connection.execute("SELECT credential_source FROM source_values").fetchall() == [
            (None,),
            *((source,) for source in allowed),
        ]

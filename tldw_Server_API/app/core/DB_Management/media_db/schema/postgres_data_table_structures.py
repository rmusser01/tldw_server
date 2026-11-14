"""Package-owned PostgreSQL data-tables ensure helpers."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

try:
    from loguru import logger
except ImportError:  # pragma: no cover - defensive fallback
    logger = logging.getLogger("media_db_postgres_data_table_structures")


class _PostgresDataTablesBackend(Protocol):
    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: object,
    ) -> object: ...

    def escape_identifier(self, name: str) -> str: ...

    def table_exists(self, table: str, *, connection: object) -> bool: ...

    def get_table_info(self, table: str, *, connection: object) -> list[dict[str, object]]: ...


class PostgresDataTablesDB(Protocol):
    backend: _PostgresDataTablesBackend
    _DATA_TABLES_SQL: str
    client_id: object

    def _convert_sqlite_sql_to_postgres_statements(self, sql: str) -> list[str]: ...

    def _ensure_postgres_columns(
        self,
        conn: Any,
        *,
        table: str,
        column_defs: dict[str, str],
    ) -> None: ...

    def _ensure_postgres_data_tables_columns(self, conn: Any) -> None: ...


def ensure_postgres_data_tables(db: PostgresDataTablesDB, conn: Any) -> None:
    """Ensure Data Tables schema exists on PostgreSQL."""

    statements = db._convert_sqlite_sql_to_postgres_statements(db._DATA_TABLES_SQL)
    create_tables = [s for s in statements if s.strip().upper().startswith("CREATE TABLE")]
    other_statements = [s for s in statements if s not in create_tables]

    for stmt in create_tables:
        try:
            db.backend.execute(stmt, connection=conn)
        except BackendDatabaseError as exc:
            logger.warning(
                "Could not ensure Data Tables base table on PostgreSQL: {}",
                exc,
            )

    # Some older PostgreSQL schemas may have partial data_tables definitions.
    # Ensure late-added columns exist before applying indexes that depend on them.
    db._ensure_postgres_data_tables_columns(conn)

    indexes_ready = True
    for stmt in other_statements:
        try:
            db.backend.execute(stmt, connection=conn)
        except BackendDatabaseError as exc:
            logger.warning(
                "Could not ensure Data Tables index/statement on PostgreSQL: {}",
                exc,
            )
            indexes_ready = False
    if indexes_ready:
        # Never remove an old guard after a failed replacement-index creation.
        db.backend.execute("DROP INDEX IF EXISTS ux_data_table_columns_table_position_active", connection=conn)
        db.backend.execute("DROP INDEX IF EXISTS ux_data_table_rows_table_index_active", connection=conn)


def ensure_postgres_columns(
    db: PostgresDataTablesDB,
    conn: Any,
    *,
    table: str,
    column_defs: dict[str, str],
) -> None:
    """Ensure a set of columns exist on a PostgreSQL table."""

    backend = db.backend
    ident = backend.escape_identifier

    if not backend.table_exists(table, connection=conn):
        return

    try:
        existing = {
            str(row.get("name") or "").lower()
            for row in backend.get_table_info(table, connection=conn)
        }
    except BackendDatabaseError as exc:
        logger.warning("Could not introspect PostgreSQL table {}: {}", table, exc)
        return

    for column, definition in column_defs.items():
        if column.lower() in existing:
            continue
        try:
            backend.execute(
                f"ALTER TABLE {ident(table)} "  # nosec B608
                f"ADD COLUMN IF NOT EXISTS {ident(column)} {definition}",
                connection=conn,
            )
        except BackendDatabaseError as exc:
            logger.warning(
                "Could not add PostgreSQL column {}.{}: {}",
                table,
                column,
                exc,
            )


def ensure_postgres_data_tables_columns(db: PostgresDataTablesDB, conn: Any) -> None:
    """Ensure late-added Data Tables columns and indexes exist on PostgreSQL."""

    backend = db.backend
    ident = backend.escape_identifier

    try:
        db._ensure_postgres_columns(
            conn,
            table="data_tables",
            column_defs={
                "workspace_tag": "TEXT",
                "column_hints_json": "TEXT",
                "generation_job_json": "TEXT",
                "last_modified": "TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                "version": "INTEGER NOT NULL DEFAULT 1",
                "client_id": "TEXT NOT NULL DEFAULT ''",
                "deleted": "BOOLEAN NOT NULL DEFAULT FALSE",
                "prev_version": "BIGINT",
                "merge_parent_uuid": "TEXT",
            },
        )
        db._ensure_postgres_columns(
            conn,
            table="data_table_columns",
            column_defs={
                "format": "TEXT",
                "last_modified": "TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                "version": "INTEGER NOT NULL DEFAULT 1",
                "client_id": "TEXT NOT NULL DEFAULT ''",
                "deleted": "BOOLEAN NOT NULL DEFAULT FALSE",
                "prev_version": "BIGINT",
                "merge_parent_uuid": "TEXT",
            },
        )
        db._ensure_postgres_columns(
            conn,
            table="data_table_rows",
            column_defs={
                "row_hash": "TEXT",
                "last_modified": "TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                "version": "INTEGER NOT NULL DEFAULT 1",
                "client_id": "TEXT NOT NULL DEFAULT ''",
                "deleted": "BOOLEAN NOT NULL DEFAULT FALSE",
                "prev_version": "BIGINT",
                "merge_parent_uuid": "TEXT",
            },
        )
        db._ensure_postgres_columns(
            conn,
            table="data_table_sources",
            column_defs={
                "last_modified": "TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
                "version": "INTEGER NOT NULL DEFAULT 1",
                "client_id": "TEXT NOT NULL DEFAULT ''",
                "deleted": "BOOLEAN NOT NULL DEFAULT FALSE",
                "prev_version": "BIGINT",
                "merge_parent_uuid": "TEXT",
            },
        )

        if backend.table_exists("data_tables", connection=conn):
            backend.execute(
                f"UPDATE {ident('data_tables')} "  # nosec B608
                f"SET {ident('client_id')} = %s "
                f"WHERE {ident('client_id')} IS NULL OR {ident('client_id')} = ''",
                (str(db.client_id),),
                connection=conn,
            )
            backend.execute(
                f"UPDATE {ident('data_tables')} "  # nosec B608
                f"SET {ident('last_modified')} = CURRENT_TIMESTAMP "
                f"WHERE {ident('last_modified')} IS NULL",
                connection=conn,
            )
            backend.execute(
                f"CREATE INDEX IF NOT EXISTS {ident('idx_data_tables_workspace_tag')} "  # nosec B608
                f"ON {ident('data_tables')} ({ident('workspace_tag')})",
                connection=conn,
            )
    except BackendDatabaseError as exc:
        logger.warning(
            "Could not ensure late Data Tables columns/indexes on PostgreSQL: {}",
            exc,
        )


__all__ = [
    "PostgresDataTablesDB",
    "ensure_postgres_columns",
    "ensure_postgres_data_tables",
    "ensure_postgres_data_tables_columns",
]

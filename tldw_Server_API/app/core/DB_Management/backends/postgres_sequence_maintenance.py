"""Forward-only PostgreSQL serial maintenance for fixed module inventories."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.postgres_sequences import (
    PostgresSequenceBackend,
    advance_postgres_serial,
)


class PostgresSequenceMaintenanceDB(Protocol):
    """Protocol for DB objects exposing a PostgreSQL backend."""

    backend: PostgresSequenceBackend


def sync_owned_postgres_sequences(
    db: PostgresSequenceMaintenanceDB,
    conn: Any,
    owned_columns: Collection[tuple[str, str]],
    *,
    continue_on_error: bool = False,
) -> None:
    """Advance owned serials without reusing allocated IDs hidden by tenant RLS.

    The schema owner calls this inside its existing migration transaction.
    Only a temporary lock savepoint is rolled back; caller work is untouched.
    PostgreSQL's nontransactional setval deliberately retains forward advances.
    ChaCha opts into per-table continuation; Media retains fail-fast maintenance.
    """

    backend = db.backend
    sequence_rows = backend.execute(
        """
        SELECT
            sequence_ns.nspname AS sequence_schema,
            seq.relname AS sequence_name,
            tab.relname AS table_name,
            col.attname AS column_name
        FROM pg_class seq
        JOIN pg_namespace sequence_ns ON sequence_ns.oid = seq.relnamespace
        JOIN pg_depend dep ON dep.objid = seq.oid AND dep.deptype = 'a'
        JOIN pg_class tab ON tab.oid = dep.refobjid
        JOIN pg_namespace tab_ns ON tab_ns.oid = tab.relnamespace
        JOIN pg_attribute col ON col.attrelid = tab.oid AND col.attnum = dep.refobjsubid
        WHERE seq.relkind = 'S' AND tab_ns.nspname = 'public'
        ORDER BY sequence_ns.nspname, seq.relname;
        """,
        connection=conn,
    )

    for row in sequence_rows.rows:
        table_name = row.get("table_name")
        column_name = row.get("column_name")
        sequence_schema = row.get("sequence_schema", "public")
        sequence_name = row.get("sequence_name")

        if not table_name or not column_name or not sequence_name:
            continue
        if (table_name, column_name) not in owned_columns:
            continue

        try:
            advance_postgres_serial(
                backend,
                conn,
                table_name=table_name,
                column_name=column_name,
                sequence_schema=sequence_schema,
                sequence_name=sequence_name,
            )
        except DatabaseError:
            if not continue_on_error:
                raise
            # The primitive rolls back its own savepoint, preserving caller work.
            logger.warning("Could not advance owned PostgreSQL sequence for {}.{}", table_name, column_name)

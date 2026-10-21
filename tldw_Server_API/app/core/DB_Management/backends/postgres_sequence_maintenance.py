"""Forward-only PostgreSQL serial maintenance for fixed module inventories."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol


class _SequenceQueryResult(Protocol):
    """Backend query result protocol for sequence metadata scans."""

    rows: list[dict[str, object]]


class _ScalarQueryResult(Protocol):
    """Backend query result protocol for MAX(...) lookups."""

    scalar: object


class _SequenceMaintenanceBackend(Protocol):
    """Backend protocol for PostgreSQL sequence synchronization."""

    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: object,
    ) -> _SequenceQueryResult | _ScalarQueryResult | object: ...

    def escape_identifier(self, value: str) -> str: ...


class PostgresSequenceMaintenanceDB(Protocol):
    """Protocol for DB objects exposing a PostgreSQL backend."""

    backend: _SequenceMaintenanceBackend


def sync_owned_postgres_sequences(
    db: PostgresSequenceMaintenanceDB,
    conn: Any,
    owned_columns: Collection[tuple[str, str]],
) -> None:
    """Advance owned serials without reusing allocated IDs hidden by tenant RLS.

    The schema owner calls this inside its existing migration transaction.
    Only a temporary lock savepoint is rolled back; caller work is untouched.
    PostgreSQL's nontransactional setval deliberately retains forward advances.
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

        ident = backend.escape_identifier
        qualified_sequence = f"{ident(sequence_schema)}.{ident(sequence_name)}"

        # A rollback-only savepoint releases this temporary sequence lock before
        # later schema DDL, avoiding table/sequence lock inversion with other
        # modules. setval is nontransactional; its forward advance survives.
        with conn.transaction(force_rollback=True):
            # Reassert the existing dependency without changing parameters.
            # ALTER SEQUENCE also blocks ordinary nextval/setval allocations.
            backend.execute(
                f"ALTER SEQUENCE {qualified_sequence} "  # nosec B608
                f"OWNED BY {ident('public')}.{ident(table_name)}.{ident(column_name)}",
                connection=conn,
            )
            sequence_state = backend.execute(
                f"SELECT last_value, is_called FROM {qualified_sequence}",  # nosec B608
                connection=conn,
            ).rows[0]

            max_result = backend.execute(
                (
                    f"SELECT COALESCE(MAX({ident(column_name)}), 0) AS max_id "  # nosec B608
                    f"FROM {ident(table_name)}"
                ),
                connection=conn,
            )

            max_id_raw = max_result.scalar
            try:
                max_id = int(max_id_raw or 0)
            except (TypeError, ValueError):
                max_id = 0

            last_value = int(sequence_state["last_value"])
            if max_id > last_value or (
                max_id == last_value and not sequence_state["is_called"]
            ):
                backend.execute(
                    "SELECT setval(%s, %s)",
                    (qualified_sequence, max_id),
                    connection=conn,
                )

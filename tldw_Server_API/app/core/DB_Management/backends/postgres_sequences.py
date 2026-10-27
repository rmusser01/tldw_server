"""Advance one native PostgreSQL serial without reusing allocated IDs."""

from __future__ import annotations

from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.backends.base import QueryResult


class PostgresSequenceBackend(Protocol):
    """Backend protocol for PostgreSQL sequence synchronization."""

    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: object,
    ) -> QueryResult: ...

    def escape_identifier(self, identifier: str) -> str: ...


def advance_postgres_serial(
    backend: PostgresSequenceBackend,
    conn: Any,
    *,
    table_name: str,
    column_name: str,
    sequence_schema: str,
    sequence_name: str,
) -> None:
    """Preserve high water and serialize allocation in the caller transaction.

    The caller owns this public-schema serial and chooses its error policy.
    Rolling back only the lock savepoint releases the sequence lock; any
    forward setval survives without committing the caller's pending rows.
    """
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
        if max_id > last_value or (max_id == last_value and not sequence_state["is_called"]):
            backend.execute(
                "SELECT setval(%s, %s)",
                (qualified_sequence, max_id),
                connection=conn,
            )

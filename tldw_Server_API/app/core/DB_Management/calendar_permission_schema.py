"""DB-owned schema checks for the Calendar PostgreSQL permission backfill."""

from __future__ import annotations

from typing import Any


async def postgres_calendar_rbac_tables_exist(conn: Any) -> bool:
    """Check whether the current PostgreSQL schema can accept the Calendar seed.

    Args:
        conn: Async PostgreSQL connection supplied by the migration runner.

    Returns:
        True when roles, permissions, and role_permissions all exist; False
        when any is missing. The caller owns the transaction and seed writes.
        Query failures propagate to the caller without committing or rolling back.
    """
    table_names = ["roles", "permissions", "role_permissions"]
    rows = await conn.fetch(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = current_schema()
          AND table_name = ANY($1::text[])
        """,
        table_names,
    )
    return not (set(table_names) - {str(row["table_name"]) for row in rows})

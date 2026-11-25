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


async def postgres_calendar_permissions_ready(conn: Any) -> bool:
    """Read the owner-provisioned Calendar grants without repairing serving state.

    The caller owns the transaction. Missing catalog entries, system roles or
    baseline grants require owner migration; query errors propagate unchanged.
    """
    if not await postgres_calendar_rbac_tables_exist(conn):
        return False
    expected = (
        ("admin", "calendar.read"), ("admin", "calendar.write"),
        ("admin", "calendar.sync"), ("admin", "calendar.admin"),
        ("user", "calendar.read"), ("user", "calendar.write"),
        ("user", "calendar.sync"), ("viewer", "calendar.read"),
    )
    return bool(await conn.fetchval(
        """
        SELECT NOT EXISTS (
            SELECT 1
            FROM unnest($1::text[], $2::text[]) AS required(role_name, permission_name)
            LEFT JOIN public.roles r ON r.name = required.role_name AND r.is_system IS TRUE
            LEFT JOIN public.permissions p ON p.name = required.permission_name
            LEFT JOIN public.role_permissions rp ON rp.role_id = r.id AND rp.permission_id = p.id
            WHERE rp.role_id IS NULL
        )
        """,
        [role for role, _permission in expected],
        [permission for _role, permission in expected],
    ))

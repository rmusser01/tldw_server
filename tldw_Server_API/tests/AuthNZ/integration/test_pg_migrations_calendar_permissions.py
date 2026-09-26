"""Exercise the Calendar permission backfill using the existing AuthNZ PG fixture."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_calendar_permissions_pg

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_calendar_pg_backfill_restores_grants_idempotently(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    """Repeated migrations restore Calendar role grants without duplicate mappings."""
    _client, _database_name = isolated_test_environment
    pool = await get_db_pool()
    async with pool.transaction() as connection:
        await connection.execute("DELETE FROM permissions WHERE category = $1", "calendar")

    assert await ensure_calendar_permissions_pg(pool) is True
    assert await ensure_calendar_permissions_pg(pool) is True

    async with pool.transaction() as connection:
        grants = await connection.fetch(
            """
            SELECT r.name AS role_name, p.name AS permission_name
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            WHERE p.category = $1 AND r.name = ANY($2::text[])
            """,
            "calendar",
            ["admin", "user", "viewer"],
        )

    assert sorted((row["role_name"], row["permission_name"]) for row in grants) == [
        ("admin", "calendar.admin"),
        ("admin", "calendar.read"),
        ("admin", "calendar.sync"),
        ("admin", "calendar.write"),
        ("user", "calendar.read"),
        ("user", "calendar.sync"),
        ("user", "calendar.write"),
        ("viewer", "calendar.read"),
    ]


async def test_calendar_pg_backfill_skips_incomplete_schema(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    """A missing mapping table must skip the seed rather than fail on its inserts."""
    _client, _database_name = isolated_test_environment
    pool = await get_db_pool()
    async with pool.transaction() as connection:
        await connection.execute("DROP TABLE role_permissions")

    assert await ensure_calendar_permissions_pg(pool) is False

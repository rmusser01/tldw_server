from __future__ import annotations

import asyncpg
import pytest

pytestmark = pytest.mark.integration

_PERMISSIONS = ("notes.graph.suggest", "notes.link_keyword", "keywords.create")
_WRITING_ROLES = ("admin", "user", "moderator")


@pytest.mark.asyncio
async def test_postgres_fresh_seed_matches_migration_94_and_is_idempotent(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import setup_database

    pool = await get_db_pool()
    setup_connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        await setup_connection.execute(
            "TRUNCATE user_permissions, user_roles, role_permissions, permissions, roles "
            "RESTART IDENTITY CASCADE"
        )
    finally:
        await setup_connection.close()

    await setup_database()
    await setup_database()

    permission_rows = await pool.fetch(
        "SELECT name, description, category FROM permissions WHERE name IN (?, ?, ?)",
        *_PERMISSIONS,
    )
    assert {
        (str(row["name"]), str(row["description"]), str(row["category"]))
        for row in permission_rows
    } == {
        (
            "notes.graph.suggest",
            "Generate and review Notes graph suggestions",
            "notes",
        ),
        (
            "notes.link_keyword",
            "Accept Notes keyword-link suggestions",
            "notes",
        ),
        (
            "keywords.create",
            "Create keywords while accepting suggestions",
            "keywords",
        ),
    }

    grant_rows = await pool.fetch(
        """
        SELECT r.name AS role_name, p.name AS permission_name
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE r.name IN (?, ?, ?)
          AND p.name IN (?, ?, ?)
        """,
        *_WRITING_ROLES,
        *_PERMISSIONS,
    )
    assert {
        (str(row["role_name"]), str(row["permission_name"])) for row in grant_rows
    } == {
        (role_name, permission_name)
        for role_name in _WRITING_ROLES
        for permission_name in _PERMISSIONS
    }

    readonly_rows = await pool.fetch(
        """
        SELECT r.name AS role_name, p.name AS permission_name
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE r.name IN (?, ?)
          AND p.name IN (?, ?, ?)
        """,
        "viewer",
        "reviewer",
        *_PERMISSIONS,
    )
    assert readonly_rows == []

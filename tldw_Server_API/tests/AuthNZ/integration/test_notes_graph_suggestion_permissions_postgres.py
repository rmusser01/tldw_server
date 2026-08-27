from __future__ import annotations

import asyncpg
import pytest

pytestmark = pytest.mark.integration

_PERMISSIONS = ("notes.graph.suggest", "notes.link_keyword", "keywords.create")
_WRITING_ROLES = ("admin", "user", "moderator")


async def _fetch_notes_suggestion_grants(pool) -> set[tuple[str, str]]:
    rows = await pool.fetch(
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
    return {
        (str(row["role_name"]), str(row["permission_name"])) for row in rows
    }


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


@pytest.mark.asyncio
async def test_postgres_repeated_bootstrap_preserves_revoked_notes_suggestion_grants(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import setup_database

    pool = await get_db_pool()
    setup_connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        await setup_connection.execute(
            """
            DELETE FROM role_permissions rp
            USING permissions p
            WHERE rp.permission_id = p.id
              AND p.name = ANY($1::text[])
            """,
            list(_PERMISSIONS),
        )
        await setup_connection.execute(
            """
            DELETE FROM role_permissions rp
            USING roles r, permissions p
            WHERE rp.role_id = r.id
              AND rp.permission_id = p.id
              AND r.name = 'user'
              AND p.name = 'media.read'
            """
        )
    finally:
        await setup_connection.close()

    await setup_database()
    await setup_database()

    revoked_rows = await pool.fetch(
        """
        SELECT r.name AS role_name, p.name AS permission_name
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE p.name IN (?, ?, ?)
        """,
        *_PERMISSIONS,
    )
    assert revoked_rows == []
    restored_legacy = await pool.fetchrow(
        """
        SELECT 1
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE r.name = ? AND p.name = ?
        """,
        "user",
        "media.read",
    )
    assert restored_legacy is not None


@pytest.mark.asyncio
async def test_mcp_seed_failure_rolls_back_catalog_then_retry_grants_defaults(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_runtime import (
        TldwPermissionSeeder,
    )

    pool = await get_db_pool()
    setup_connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        await setup_connection.execute(
            """
            DELETE FROM role_permissions rp
            USING permissions p
            WHERE rp.permission_id = p.id
              AND p.name = ANY($1::text[])
            """,
            list(_PERMISSIONS),
        )
        await setup_connection.execute(
            "DELETE FROM permissions WHERE name = ANY($1::text[])",
            list(_PERMISSIONS),
        )
        await setup_connection.execute(
            """
            CREATE FUNCTION tldw_test_fail_rbac_grant() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'forced RBAC grant failure';
            END;
            $$
            """
        )
        await setup_connection.execute(
            """
            CREATE TRIGGER tldw_test_fail_rbac_grant
            BEFORE INSERT ON role_permissions
            FOR EACH ROW EXECUTE FUNCTION tldw_test_fail_rbac_grant()
            """
        )

        with pytest.raises(
            TransactionError,
            match="Transaction failed during: PostgreSQL transaction",
        ):
            await TldwPermissionSeeder().seed_default_tool_permissions()

        permission_rows = await setup_connection.fetch(
            "SELECT name FROM permissions WHERE name = ANY($1::text[])",
            list(_PERMISSIONS),
        )
        assert permission_rows == []
        assert await _fetch_notes_suggestion_grants(pool) == set()
    finally:
        await setup_connection.execute(
            "DROP TRIGGER IF EXISTS tldw_test_fail_rbac_grant ON role_permissions"
        )
        await setup_connection.execute(
            "DROP FUNCTION IF EXISTS tldw_test_fail_rbac_grant()"
        )
        await setup_connection.close()

    await TldwPermissionSeeder().seed_default_tool_permissions()
    expected_grants = {
        (role_name, permission_name)
        for role_name in _WRITING_ROLES
        for permission_name in _PERMISSIONS
    }
    assert await _fetch_notes_suggestion_grants(pool) == expected_grants

    cleanup_connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        await cleanup_connection.execute(
            """
            DELETE FROM role_permissions rp
            USING permissions p
            WHERE rp.permission_id = p.id
              AND p.name = ANY($1::text[])
            """,
            list(_PERMISSIONS),
        )
    finally:
        await cleanup_connection.close()

    await TldwPermissionSeeder().seed_default_tool_permissions()
    assert await _fetch_notes_suggestion_grants(pool) == set()

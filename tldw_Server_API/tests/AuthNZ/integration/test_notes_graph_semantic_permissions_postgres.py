from __future__ import annotations

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ.permissions import NOTES_GRAPH_SEMANTIC_MANAGE

pytestmark = pytest.mark.integration

_WRITING_ROLES = ("admin", "user", "moderator")


@pytest.mark.asyncio
async def test_postgres_seed_grants_new_semantic_permission_to_existing_approved_roles(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import setup_database

    pool = await get_db_pool()
    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        await connection.execute(
            "DELETE FROM role_permissions rp USING permissions p "
            "WHERE rp.permission_id = p.id AND p.name = $1",
            NOTES_GRAPH_SEMANTIC_MANAGE,
        )
        await connection.execute(
            "DELETE FROM permissions WHERE name = $1", NOTES_GRAPH_SEMANTIC_MANAGE
        )
    finally:
        await connection.close()

    await setup_database()

    rows = await pool.fetch(
        """
        SELECT r.name AS role_name, p.name AS permission_name
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE p.name = $1
        """,
        NOTES_GRAPH_SEMANTIC_MANAGE,
    )
    assert {(str(row["role_name"]), str(row["permission_name"])) for row in rows} == {
        (role, NOTES_GRAPH_SEMANTIC_MANAGE) for role in _WRITING_ROLES
    }


@pytest.mark.asyncio
async def test_postgres_backstop_preserves_revoked_semantic_manage_grants(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import setup_database

    pool = await get_db_pool()
    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        await connection.execute(
            "DELETE FROM role_permissions rp USING permissions p "
            "WHERE rp.permission_id = p.id AND p.name = $1",
            NOTES_GRAPH_SEMANTIC_MANAGE,
        )
    finally:
        await connection.close()

    await setup_database()

    assert await pool.fetch(
        """
        SELECT 1
        FROM role_permissions rp
        JOIN permissions p ON p.id = rp.permission_id
        WHERE p.name = $1
        """,
        NOTES_GRAPH_SEMANTIC_MANAGE,
    ) == []

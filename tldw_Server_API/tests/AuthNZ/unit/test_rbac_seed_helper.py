import pytest


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_ensure_baseline_rbac_seed_sqlite_idempotent():
    import aiosqlite

    from tldw_Server_API.app.core.AuthNZ.rbac_seed import ensure_baseline_rbac_seed

    async with aiosqlite.connect(":memory:") as conn:
        conn.row_factory = aiosqlite.Row
        await conn.execute(
            """
            CREATE TABLE roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                is_system INTEGER DEFAULT 0
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                category TEXT
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE role_permissions (
                role_id INTEGER NOT NULL,
                permission_id INTEGER NOT NULL,
                PRIMARY KEY (role_id, permission_id)
            )
            """
        )
        await conn.commit()

        await ensure_baseline_rbac_seed(conn, include_mcp_permissions=True)
        await ensure_baseline_rbac_seed(conn, include_mcp_permissions=True)

        cur = await conn.execute("SELECT name FROM roles")
        roles = {row[0] for row in await cur.fetchall()}
        assert {"admin", "user", "viewer"} <= roles

        expected_permissions = {
            "media.read",
            "media.create",
            "media.delete",
            "sql.read",
            "sql.target:media_db",
            "system.configure",
            "users.manage_roles",
            "modules.read",
            "prompts.read",
            "tools.execute:*",
        }
        cur = await conn.execute("SELECT name FROM permissions")
        perms = {row[0] for row in await cur.fetchall()}
        assert expected_permissions <= perms

        cur = await conn.execute("SELECT id, name FROM roles WHERE name IN ('admin','user','viewer')")
        role_id = {row[1]: row[0] for row in await cur.fetchall()}

        cur = await conn.execute(
            """
            SELECT id, name
            FROM permissions
            WHERE name IN (
                'media.read','media.create','media.delete','system.configure',
                'users.manage_roles','sql.read','sql.target:media_db','modules.read',
                'prompts.read','tools.execute:*'
            )
            """
        )
        perm_id = {row[1]: row[0] for row in await cur.fetchall()}

        cur = await conn.execute(
            "SELECT permission_id FROM role_permissions WHERE role_id = ?",
            (role_id["user"],),
        )
        user_perm_ids = {row[0] for row in await cur.fetchall()}
        assert perm_id["media.read"] in user_perm_ids
        assert perm_id["media.create"] in user_perm_ids
        assert perm_id["sql.read"] in user_perm_ids
        assert perm_id["sql.target:media_db"] in user_perm_ids
        assert perm_id["modules.read"] in user_perm_ids
        assert perm_id["prompts.read"] in user_perm_ids

        cur = await conn.execute(
            "SELECT permission_id FROM role_permissions WHERE role_id = ?",
            (role_id["viewer"],),
        )
        viewer_perm_ids = {row[0] for row in await cur.fetchall()}
        assert perm_id["media.read"] in viewer_perm_ids

        cur = await conn.execute(
            "SELECT permission_id FROM role_permissions WHERE role_id = ?",
            (role_id["admin"],),
        )
        admin_perm_ids = {row[0] for row in await cur.fetchall()}
        for name in expected_permissions:
            assert perm_id[name] in admin_perm_ids


def test_migration_089_seeds_prompts_read_for_existing_admin_and_user_roles():
    import sqlite3

    from tldw_Server_API.app.core.AuthNZ.migrations import (
        get_authnz_migrations,
        migration_089_seed_mcp_prompts_read_permission,
    )

    assert get_authnz_migrations()[-1].version == 89

    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(
            """
            CREATE TABLE roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                is_system INTEGER DEFAULT 0
            );
            CREATE TABLE permissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                category TEXT
            );
            CREATE TABLE role_permissions (
                role_id INTEGER NOT NULL,
                permission_id INTEGER NOT NULL,
                PRIMARY KEY (role_id, permission_id)
            );
            INSERT INTO roles (name, description, is_system)
            VALUES
                ('admin', 'Administrator', 1),
                ('user', 'Standard User', 1);
            """
        )

        migration_089_seed_mcp_prompts_read_permission(conn)
        migration_089_seed_mcp_prompts_read_permission(conn)

        permission_rows = conn.execute(
            "SELECT name, description, category FROM permissions WHERE name = ?",
            ("prompts.read",),
        ).fetchall()
        assert permission_rows == [("prompts.read", "Read MCP prompts", "prompts")]

        grant_rows = conn.execute(
            """
            SELECT r.name, p.name
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            WHERE p.name = ?
            ORDER BY r.name
            """,
            ("prompts.read",),
        ).fetchall()
        assert grant_rows == [("admin", "prompts.read"), ("user", "prompts.read")]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_ensure_sqlite_rbac_tables_creates_minimal_schema():
    import aiosqlite

    from tldw_Server_API.app.core.AuthNZ.rbac_seed import ensure_sqlite_rbac_tables

    async with aiosqlite.connect(":memory:") as conn:
        await ensure_sqlite_rbac_tables(conn)
        await conn.commit()

        for table in ("roles", "permissions", "role_permissions", "user_roles"):
            cur = await conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            assert await cur.fetchone() is not None


@pytest.mark.asyncio
async def test_ensure_baseline_rbac_seed_explicit_backend_hint_skips_detection(
    monkeypatch: pytest.MonkeyPatch,
):
    import aiosqlite

    from tldw_Server_API.app.core.AuthNZ import rbac_seed

    async with aiosqlite.connect(":memory:") as conn:
        await rbac_seed.ensure_sqlite_rbac_tables(conn)
        await conn.commit()

        def _raise_if_called(_conn):
            raise AssertionError("backend auto-detection should be bypassed when hint is provided")

        monkeypatch.setattr(rbac_seed, "_is_postgres_connection", _raise_if_called)
        await rbac_seed.ensure_baseline_rbac_seed(
            conn,
            include_mcp_permissions=False,
            is_postgres=False,
        )

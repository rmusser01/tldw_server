import ast
import sqlite3
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _is_named_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == name)
        or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
    )


def test_all_production_rbac_seed_callers_own_pool_transactions() -> None:
    repository_root = Path(__file__).resolve().parents[4]
    app_root = repository_root / "tldw_Server_API" / "app"
    expected_callers = {
        "core/AuthNZ/initialize.py": 2,
        "core/MCP_unified/adapters/tldw_runtime.py": 1,
    }
    actual_callers: dict[str, int] = {}

    for path in app_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "ensure_baseline_rbac_seed" not in source:
            continue
        tree = ast.parse(source)
        seed_calls = {
            node
            for node in ast.walk(tree)
            if _is_named_call(node, "ensure_baseline_rbac_seed")
        }
        if not seed_calls:
            continue

        relative_path = str(path.relative_to(app_root))
        actual_callers[relative_path] = len(seed_calls)
        transaction_owned_calls: set[ast.AST] = set()
        for async_with in (
            node for node in ast.walk(tree) if isinstance(node, ast.AsyncWith)
        ):
            if not any(
                _is_named_call(item.context_expr, "transaction")
                for item in async_with.items
            ):
                continue
            for statement in async_with.body:
                transaction_owned_calls.update(
                    node
                    for node in ast.walk(statement)
                    if _is_named_call(node, "ensure_baseline_rbac_seed")
                )

        assert seed_calls == transaction_owned_calls, relative_path

    assert actual_callers == expected_callers

    helper_path = app_root / "core" / "AuthNZ" / "rbac_seed.py"
    helper_tree = ast.parse(helper_path.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in ast.walk(helper_tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "ensure_baseline_rbac_seed"
    )
    assert not any(_is_named_call(node, "transaction") for node in ast.walk(helper))


@pytest.mark.asyncio
async def test_ensure_baseline_rbac_seed_sqlite_idempotent() -> None:
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
        assert {"admin", "user", "moderator", "viewer"} <= roles

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
            "notifications.read",
            "notifications.control",
            "notes.graph.suggest",
            "notes.link_keyword",
            "keywords.create",
        }
        cur = await conn.execute("SELECT name FROM permissions")
        perms = {row[0] for row in await cur.fetchall()}
        assert expected_permissions <= perms

        cur = await conn.execute(
            "SELECT id, name FROM roles WHERE name IN ('admin','user','moderator','viewer','reviewer')"
        )
        role_id = {row[1]: row[0] for row in await cur.fetchall()}

        cur = await conn.execute(
            """
            SELECT id, name
            FROM permissions
            WHERE name IN (
                'media.read','media.create','media.delete','system.configure',
                'users.manage_roles','sql.read','sql.target:media_db','modules.read',
                'prompts.read','tools.execute:*','notifications.read','notifications.control'
                ,'notes.graph.suggest','notes.link_keyword','keywords.create'
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
        assert perm_id["notifications.read"] in user_perm_ids
        assert perm_id["notifications.control"] in user_perm_ids
        assert perm_id["notes.graph.suggest"] in user_perm_ids
        assert perm_id["notes.link_keyword"] in user_perm_ids
        assert perm_id["keywords.create"] in user_perm_ids

        cur = await conn.execute(
            "SELECT permission_id FROM role_permissions WHERE role_id = ?",
            (role_id["moderator"],),
        )
        moderator_perm_ids = {row[0] for row in await cur.fetchall()}
        assert {
            perm_id["notes.graph.suggest"],
            perm_id["notes.link_keyword"],
            perm_id["keywords.create"],
        } <= moderator_perm_ids

        cur = await conn.execute(
            "SELECT permission_id FROM role_permissions WHERE role_id = ?",
            (role_id["viewer"],),
        )
        viewer_perm_ids = {row[0] for row in await cur.fetchall()}
        assert perm_id["media.read"] in viewer_perm_ids
        assert perm_id["notifications.read"] in viewer_perm_ids
        assert perm_id["notifications.control"] in viewer_perm_ids
        assert perm_id["notes.graph.suggest"] not in viewer_perm_ids
        assert perm_id["notes.link_keyword"] not in viewer_perm_ids
        assert perm_id["keywords.create"] not in viewer_perm_ids

        cur = await conn.execute(
            "SELECT permission_id FROM role_permissions WHERE role_id = ?",
            (role_id["reviewer"],),
        )
        reviewer_perm_ids = {row[0] for row in await cur.fetchall()}
        assert perm_id["notifications.read"] in reviewer_perm_ids
        assert perm_id["notifications.control"] in reviewer_perm_ids

        cur = await conn.execute(
            "SELECT permission_id FROM role_permissions WHERE role_id = ?",
            (role_id["admin"],),
        )
        admin_perm_ids = {row[0] for row in await cur.fetchall()}
        for name in expected_permissions:
            assert perm_id[name] in admin_perm_ids


@pytest.mark.asyncio
async def test_sqlite_repeated_bootstrap_preserves_revoked_notes_suggestion_grants() -> None:
    import aiosqlite

    from tldw_Server_API.app.core.AuthNZ.rbac_seed import (
        ensure_baseline_rbac_seed,
        ensure_sqlite_rbac_tables,
    )

    protected_permissions = (
        "notes.graph.suggest",
        "notes.link_keyword",
        "keywords.create",
    )
    async with aiosqlite.connect(":memory:") as conn:
        await ensure_sqlite_rbac_tables(conn)
        await ensure_baseline_rbac_seed(conn, include_mcp_permissions=True)
        await conn.execute(
            """
            DELETE FROM role_permissions
            WHERE permission_id IN (
                SELECT id FROM permissions WHERE name IN (?, ?, ?)
            )
            """,
            protected_permissions,
        )
        await conn.execute(
            """
            DELETE FROM role_permissions
            WHERE role_id = (SELECT id FROM roles WHERE name = 'user')
              AND permission_id = (SELECT id FROM permissions WHERE name = 'media.read')
            """
        )

        await ensure_baseline_rbac_seed(conn, include_mcp_permissions=True)

        cur = await conn.execute(
            """
            SELECT r.name, p.name
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            WHERE p.name IN (?, ?, ?)
            """,
            protected_permissions,
        )
        assert await cur.fetchall() == []
        cur = await conn.execute(
            """
            SELECT 1
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            WHERE r.name = 'user' AND p.name = 'media.read'
            """
        )
        assert await cur.fetchone() is not None


def test_migration_089_seeds_prompts_read_for_existing_admin_and_user_roles() -> None:
    import sqlite3

    from tldw_Server_API.app.core.AuthNZ.migrations import (
        get_authnz_migrations,
        migration_089_seed_mcp_prompts_read_permission,
    )

    assert any(migration.version == 89 for migration in get_authnz_migrations())

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


def _create_version_089_rbac_database(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
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
            CREATE TABLE user_roles (
                user_id INTEGER NOT NULL,
                role_id INTEGER NOT NULL,
                granted_by INTEGER,
                expires_at TIMESTAMP,
                PRIMARY KEY (user_id, role_id)
            );
            CREATE TABLE user_permissions (
                user_id INTEGER NOT NULL,
                permission_id INTEGER NOT NULL,
                granted INTEGER NOT NULL DEFAULT 1,
                expires_at TIMESTAMP,
                PRIMARY KEY (user_id, permission_id)
            );

            INSERT INTO roles (name, description, is_system) VALUES
                ('admin', 'Administrator', 1),
                ('user', 'Standard User', 1),
                ('moderator', 'Moderator', 1),
                ('reviewer', 'Reviewer', 1),
                ('viewer', 'Viewer', 1),
                ('custom-auditor', 'Custom Auditor', 0);

            INSERT INTO users (id, username, role) VALUES
                (1, 'legacy-admin', 'admin'),
                (2, 'legacy-user', 'user'),
                (3, 'legacy-moderator', 'moderator'),
                (4, 'legacy-reviewer', 'reviewer'),
                (5, 'legacy-viewer', 'viewer'),
                (6, 'legacy-custom', 'custom-auditor'),
                (7, 'legacy-missing', 'role-that-does-not-exist');

            INSERT INTO permissions (name, description, category) VALUES
                ('notifications.control', 'Legacy notification control', 'notifications'),
                ('custom.audit', 'Custom audit permission', 'custom');

            INSERT INTO role_permissions (role_id, permission_id)
            SELECT r.id, p.id
            FROM roles r
            JOIN permissions p ON p.name = 'custom.audit'
            WHERE r.name = 'custom-auditor';

            INSERT INTO user_permissions (user_id, permission_id, granted, expires_at)
            SELECT 2, p.id, 0, '2035-01-02 03:04:05'
            FROM permissions p
            WHERE p.name = 'notifications.control';
            """
        )
        conn.commit()
    finally:
        conn.close()


def _migrate_version_089_database(db_path: Path) -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import get_authnz_migrations
    from tldw_Server_API.app.core.DB_Management.migrations import MigrationManager

    manager = MigrationManager(db_path)
    migrations = get_authnz_migrations()
    assert migrations[-1].version == 95
    for migration in migrations:
        manager.add_migration(migration)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (89, "Seed MCP prompts.read permission"),
        )

    manager.migrate()
    with sqlite3.connect(db_path) as conn:
        migrations[-1].apply(conn)
    manager.migrate()


def test_migration_090_seeds_notification_permissions_and_interactive_role_grants(tmp_path: Path) -> None:
    db_path = tmp_path / "authnz-v089.db"
    _create_version_089_rbac_database(db_path)

    with sqlite3.connect(db_path) as conn:
        deny_rows_before = conn.execute(
            """
            SELECT up.user_id, p.name, up.granted, up.expires_at
            FROM user_permissions up
            JOIN permissions p ON p.id = up.permission_id
            WHERE p.name = ?
            """,
            ("notifications.control",),
        ).fetchall()

    _migrate_version_089_database(db_path)

    with sqlite3.connect(db_path) as conn:
        permission_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM permissions WHERE category = ?",
                ("notifications",),
            ).fetchall()
        }
        assert permission_names >= {"notifications.read", "notifications.control"}

        grant_rows = conn.execute(
            """
            SELECT r.name, p.name
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            WHERE p.name IN (?, ?)
            """,
            ("notifications.read", "notifications.control"),
        ).fetchall()
        grants_by_role: dict[str, set[str]] = {}
        for role_name, permission_name in grant_rows:
            grants_by_role.setdefault(role_name, set()).add(permission_name)

        interactive_roles = {"admin", "user", "moderator", "reviewer", "viewer"}
        present_roles = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM roles WHERE name IN (?, ?, ?, ?, ?)",
                tuple(sorted(interactive_roles)),
            ).fetchall()
        }
        for role_name in present_roles:
            assert grants_by_role[role_name] >= {"notifications.read", "notifications.control"}

        deny_rows_after = conn.execute(
            """
            SELECT up.user_id, p.name, up.granted, up.expires_at
            FROM user_permissions up
            JOIN permissions p ON p.id = up.permission_id
            WHERE p.name = ?
            """,
            ("notifications.control",),
        ).fetchall()
        assert deny_rows_after == deny_rows_before == [
            (2, "notifications.control", 0, "2035-01-02 03:04:05")
        ]


def test_migration_090_backfills_only_matching_legacy_roles_without_changing_custom_grants(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "authnz-v089-custom-role.db"
    _create_version_089_rbac_database(db_path)

    with sqlite3.connect(db_path) as conn:
        custom_grants_before = conn.execute(
            """
            SELECT p.name
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            WHERE r.name = ?
            ORDER BY p.name
            """,
            ("custom-auditor",),
        ).fetchall()

    _migrate_version_089_database(db_path)

    with sqlite3.connect(db_path) as conn:
        user_role_rows = set(
            conn.execute(
                """
                SELECT u.username, r.name
                FROM user_roles ur
                JOIN users u ON u.id = ur.user_id
                JOIN roles r ON r.id = ur.role_id
                """
            ).fetchall()
        )
        assert user_role_rows == {
            ("legacy-admin", "admin"),
            ("legacy-user", "user"),
            ("legacy-moderator", "moderator"),
            ("legacy-reviewer", "reviewer"),
            ("legacy-viewer", "viewer"),
            ("legacy-custom", "custom-auditor"),
        }
        assert conn.execute(
            "SELECT 1 FROM roles WHERE name = ?",
            ("role-that-does-not-exist",),
        ).fetchone() is None

        custom_grants_after = conn.execute(
            """
            SELECT p.name
            FROM role_permissions rp
            JOIN roles r ON r.id = rp.role_id
            JOIN permissions p ON p.id = rp.permission_id
            WHERE r.name = ?
            ORDER BY p.name
            """,
            ("custom-auditor",),
        ).fetchall()
        assert custom_grants_after == custom_grants_before == [("custom.audit",)]


@pytest.mark.asyncio
async def test_ensure_sqlite_rbac_tables_creates_minimal_schema() -> None:
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
) -> None:
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

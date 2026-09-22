from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.integration

_INTERACTIVE_ROLES = ("admin", "user", "moderator", "reviewer", "viewer")
_NOTIFICATION_PERMISSIONS = {"notifications.read", "notifications.control"}


@pytest.mark.asyncio
async def test_postgres_notification_backfill_returns_false_on_transaction_error() -> None:
    from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_notification_permissions_pg,
    )

    @asynccontextmanager
    async def _failing_transaction():
        raise TransactionError("PostgreSQL transaction", "forced SQL failure")
        yield

    pool = SimpleNamespace(pool=object(), transaction=_failing_transaction)

    assert await ensure_notification_permissions_pg(pool) is False


@pytest.mark.asyncio
async def test_postgres_fresh_initialization_grants_notifications_after_roles_are_seeded(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import setup_database

    pool = await get_db_pool()
    async with pool.transaction() as conn:
        await conn.execute("DELETE FROM user_permissions")
        await conn.execute("DELETE FROM user_roles")
        await conn.execute("DELETE FROM role_permissions")
        await conn.execute("DELETE FROM permissions")
        await conn.execute("DELETE FROM roles")

    assert await pool.fetchval("SELECT COUNT(*) FROM user_permissions") == 0
    assert await pool.fetchval("SELECT COUNT(*) FROM user_roles") == 0
    assert await pool.fetchval("SELECT COUNT(*) FROM role_permissions") == 0
    assert await pool.fetchval("SELECT COUNT(*) FROM permissions") == 0
    assert await pool.fetchval("SELECT COUNT(*) FROM roles") == 0

    await setup_database()

    role_rows = await pool.fetch(
        "SELECT name FROM roles WHERE is_system = TRUE AND name IN (?, ?, ?, ?, ?)",
        *_INTERACTIVE_ROLES,
    )
    present_roles = {str(row["name"]) for row in role_rows}
    assert {"admin", "user"} <= present_roles

    grant_rows = await pool.fetch(
        """
        SELECT r.name AS role_name, p.name AS permission_name
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE r.name IN (?, ?, ?, ?, ?)
          AND p.name IN (?, ?)
        """,
        *_INTERACTIVE_ROLES,
        *sorted(_NOTIFICATION_PERMISSIONS),
    )
    grants = {(str(row["role_name"]), str(row["permission_name"])) for row in grant_rows}
    assert grants == {
        (role_name, permission_name)
        for role_name in present_roles
        for permission_name in _NOTIFICATION_PERMISSIONS
    }


@pytest.mark.asyncio
async def test_postgres_notification_backfill_is_idempotent_and_preserves_overrides(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_notification_permissions_pg,
    )
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    pool = await get_db_pool()
    for role_name, is_system in (
        ("moderator", True),
        ("reviewer", True),
        ("viewer", True),
        ("custom-auditor", False),
    ):
        await pool.execute(
            """
            INSERT INTO roles (name, description, is_system)
            VALUES (?, ?, ?)
            ON CONFLICT (name) DO NOTHING
            """,
            role_name,
            f"{role_name} role",
            is_system,
        )

    legacy_users = (
        ("notification-admin", "notification-admin@example.test", "admin"),
        ("notification-user", "notification-user@example.test", "user"),
        ("notification-moderator", "notification-moderator@example.test", "moderator"),
        ("notification-reviewer", "notification-reviewer@example.test", "reviewer"),
        ("notification-viewer", "notification-viewer@example.test", "viewer"),
        ("notification-custom", "notification-custom@example.test", "custom-auditor"),
        ("notification-missing", "notification-missing@example.test", "missing-role"),
    )
    users = UsersDB(pool)
    await users.initialize(ensure_schema=False)
    for username, email, role_name in legacy_users:
        await users.create_user(  # nosec B106 # Inert fixture hash, never used for login
            username=username,
            email=email,
            password_hash="test-password-hash",
            role=role_name,
        )

    legacy_rows = await pool.fetch(
        "SELECT username, email, role FROM users WHERE username LIKE ?",
        "notification-%",
    )
    assert {tuple(row.values()) for row in legacy_rows} == set(legacy_users)
    assert await pool.fetchval(
        """
        SELECT COUNT(*) FROM user_roles ur
        JOIN users u ON u.id = ur.user_id
        WHERE u.username LIKE ?
        """,
        "notification-%",
    ) == 0

    await pool.execute(
        """
        INSERT INTO permissions (name, description, category)
        VALUES (?, ?, ?)
        ON CONFLICT (name) DO NOTHING
        """,
        "notifications.control",
        "Legacy notification control",
        "notifications",
    )
    await pool.execute(
        """
        INSERT INTO permissions (name, description, category)
        VALUES (?, ?, ?)
        ON CONFLICT (name) DO NOTHING
        """,
        "custom.audit",
        "Custom audit permission",
        "custom",
    )
    await pool.execute(
        """
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT r.id, p.id
        FROM roles r
        JOIN permissions p ON p.name = ?
        WHERE r.name = ?
        ON CONFLICT (role_id, permission_id) DO NOTHING
        """,
        "custom.audit",
        "custom-auditor",
    )
    await pool.execute(
        """
        INSERT INTO user_permissions (user_id, permission_id, granted, expires_at)
        SELECT u.id, p.id, FALSE, ?
        FROM users u
        JOIN permissions p ON p.name = ?
        WHERE u.username = ?
        ON CONFLICT (user_id, permission_id) DO NOTHING
        """,
        datetime(2035, 1, 2, 3, 4, 5),
        "notifications.control",
        "notification-user",
    )

    deny_rows_before = await pool.fetch(
        """
        SELECT u.username, p.name, up.granted, up.expires_at
        FROM user_permissions up
        JOIN users u ON u.id = up.user_id
        JOIN permissions p ON p.id = up.permission_id
        WHERE p.name = ?
        """,
        "notifications.control",
    )
    custom_grants_before = await pool.fetch(
        """
        SELECT p.name
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE r.name = ?
        ORDER BY p.name
        """,
        "custom-auditor",
    )
    assert [
        (row["username"], row["name"], row["granted"], row["expires_at"])
        for row in deny_rows_before
    ] == [
        (
            "notification-user",
            "notifications.control",
            False,
            datetime(2035, 1, 2, 3, 4, 5),
        )
    ]

    assert await ensure_notification_permissions_pg(pool) is True
    assert await ensure_notification_permissions_pg(pool) is True

    permission_rows = await pool.fetch(
        "SELECT name FROM permissions WHERE name IN (?, ?)",
        *sorted(_NOTIFICATION_PERMISSIONS),
    )
    assert {str(row["name"]) for row in permission_rows} == _NOTIFICATION_PERMISSIONS

    grant_rows = await pool.fetch(
        """
        SELECT r.name AS role_name, p.name AS permission_name
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE r.is_system = TRUE
          AND r.name IN (?, ?, ?, ?, ?)
          AND p.name IN (?, ?)
        """,
        *_INTERACTIVE_ROLES,
        *sorted(_NOTIFICATION_PERMISSIONS),
    )
    grants = {(str(row["role_name"]), str(row["permission_name"])) for row in grant_rows}
    assert grants == {
        (role_name, permission_name)
        for role_name in _INTERACTIVE_ROLES
        for permission_name in _NOTIFICATION_PERMISSIONS
    }

    membership_rows = await pool.fetch(
        """
        SELECT u.username, r.name AS role_name
        FROM user_roles ur
        JOIN users u ON u.id = ur.user_id
        JOIN roles r ON r.id = ur.role_id
        WHERE u.username LIKE ?
        """,
        "notification-%",
    )
    assert {(str(row["username"]), str(row["role_name"])) for row in membership_rows} == {
        ("notification-admin", "admin"),
        ("notification-user", "user"),
        ("notification-moderator", "moderator"),
        ("notification-reviewer", "reviewer"),
        ("notification-viewer", "viewer"),
        ("notification-custom", "custom-auditor"),
    }

    deny_rows_after = await pool.fetch(
        """
        SELECT u.username, p.name, up.granted, up.expires_at
        FROM user_permissions up
        JOIN users u ON u.id = up.user_id
        JOIN permissions p ON p.id = up.permission_id
        WHERE p.name = ?
        """,
        "notifications.control",
    )
    assert [tuple(row.values()) for row in deny_rows_after] == [tuple(row.values()) for row in deny_rows_before]

    custom_grants_after = await pool.fetch(
        """
        SELECT p.name
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE r.name = ?
        ORDER BY p.name
        """,
        "custom-auditor",
    )
    assert [tuple(row.values()) for row in custom_grants_after] == [
        tuple(row.values()) for row in custom_grants_before
    ] == [("custom.audit",)]

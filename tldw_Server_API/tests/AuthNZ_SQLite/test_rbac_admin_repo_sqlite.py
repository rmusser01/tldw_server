"""rbac_admin_repo against a migrated SQLite AuthNZ database (TASK-13362)."""

from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.repos import rbac_admin_repo as repo
from tldw_Server_API.app.core.AuthNZ.settings import Settings


@pytest.mark.asyncio
async def test_rbac_admin_repo_round_trip_on_migrated_sqlite(tmp_path):
    db_path = tmp_path / "authnz_rbac_admin.sqlite"
    ensure_authnz_tables(db_path)
    with sqlite3.connect(db_path) as raw:  # the application pool guards user writes
        raw.execute("INSERT INTO users (id, username, email, password_hash, is_active) VALUES (51, 'rb', 'rb@x.io', 'h', 1)")
        raw.execute("INSERT INTO roles (id, name, description) VALUES (951, 'rb-editor', 'e')")
        raw.execute("INSERT INTO roles (id, name, description) VALUES (952, 'rb-viewer', 'v')")
    pool = DatabasePool(
        Settings(
            AUTH_MODE="multi_user",
            DATABASE_URL=f"sqlite:///{db_path}",
            JWT_SECRET_KEY="rbac-admin-secret-key-32-characters-minimum!",
        )
    )
    await pool.initialize()
    kw = {"is_postgres": False}
    try:
        async with pool.transaction() as conn:
            created = await repo.create_permission(conn, name="rb.read", description="d", category="rb", **kw)
            assert created["name"] == "rb.read"
            assert await repo.create_permission(conn, name="RB.READ", description="d", category="rb", **kw) is None
            granted = await repo.grant_tool_permissions(
                conn, role_id=951, permissions=[("tools.execute:rb-a", "A"), ("tools.execute:rb-b", "B")], **kw
            )
            assert [row["name"] for row in granted] == ["tools.execute:rb-a", "tools.execute:rb-b"]
            # idempotent
            await repo.grant_tool_permissions(conn, role_id=951, permissions=[("tools.execute:rb-a", "A")], **kw)
            await repo.grant_permission(conn, role_id=952, permission_id=int(created["id"]), **kw)
            await repo.add_user_role(conn, user_id=51, role_id=951, **kw)
            await repo.add_user_role(conn, user_id=51, role_id=951, **kw)
            await repo.upsert_user_override(
                conn, user_id=51, permission_id=int(created["id"]), granted=True, expires_at=None, **kw
            )
            await repo.upsert_user_override(
                conn, user_id=51, permission_id=int(created["id"]), granted=False, expires_at=None, **kw
            )

        async with pool.transaction() as conn:
            assert [r["name"] for r in await repo.list_role_tool_permissions(conn, role_id=951, **kw)] == [
                "tools.execute:rb-a",
                "tools.execute:rb-b",
            ]
            assert "rb" in await repo.permission_categories(conn, **kw)
            assert [r["name"] for r in await repo.list_permissions(conn, category="rb", **kw)] == ["rb.read"]
            total, roles = await repo.roles_page(
                conn, role_search="rb-", role_names=None, limit=1, offset=0, **kw
            )
            assert total == 2 and [r["name"] for r in roles] == ["rb-editor"]
            grants = await repo.role_permission_grants(conn, category="rb", role_ids=[951, 952], **kw)
            assert grants == [(952, int(created["id"]))]
            assert (await repo.get_role(conn, role_id=952, **kw))["name"] == "rb-viewer"
            override = await (await conn.execute(
                "SELECT granted FROM user_permissions WHERE user_id = 51 AND permission_id = ?", (int(created["id"]),)
            )).fetchone()
            assert not bool(override[0])  # the second upsert replaced the first

            assert await repo.revoke_tool_permissions(
                conn, role_id=951, names=["tools.execute:rb-a", "tools.execute:missing"], **kw
            ) == ["tools.execute:rb-a"]
            await repo.remove_user_role(conn, user_id=51, role_id=951, **kw)
            await repo.delete_user_override(conn, user_id=51, permission_id=int(created["id"]), **kw)
    finally:
        await pool.close()

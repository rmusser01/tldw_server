"""rbac_rate_limits_repo against a migrated SQLite AuthNZ database (TASK-13317 AC2)."""

from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.repos import rbac_rate_limits_repo as repo
from tldw_Server_API.app.core.AuthNZ.settings import Settings


@pytest.mark.asyncio
async def test_rate_limit_repo_round_trip_on_migrated_sqlite(tmp_path):
    db_path = tmp_path / "authnz_rate_limits.sqlite"
    ensure_authnz_tables(db_path)
    # Fixture rows go in directly: the application pool guards user writes.
    with sqlite3.connect(db_path) as raw:
        raw.execute(
            "INSERT INTO users (id, username, email, password_hash, is_active) VALUES (41, 'rl', 'rl@x.io', 'h', 1)"
        )
        raw.execute("INSERT INTO roles (id, name, description) VALUES (901, 'rl-role', 'r')")
        raw.execute("INSERT INTO roles (id, name, description) VALUES (902, 'rl-expired', 'r')")
        raw.execute("INSERT INTO user_roles (user_id, role_id) VALUES (41, 901)")
        raw.execute("INSERT INTO user_roles (user_id, role_id, expires_at) VALUES (41, 902, '2000-01-01 00:00:00')")
    pool = DatabasePool(
        Settings(
            AUTH_MODE="multi_user",
            DATABASE_URL=f"sqlite:///{db_path}",
            JWT_SECRET_KEY="rate-limits-secret-key-32-characters-minimum!",
        )
    )
    await pool.initialize()
    try:
        async with pool.transaction() as conn:
            await repo.upsert_user_limit(conn, is_postgres=False, user_id=41, resource="/api/v1/x", limit_per_min=5, burst=1)
            await repo.upsert_role_limit(conn, is_postgres=False, role_id=901, resource="/api/v1/x", limit_per_min=20, burst=4)
            await repo.upsert_role_limit(conn, is_postgres=False, role_id=902, resource="/api/v1/x", limit_per_min=1, burst=1)
            # upsert replaces rather than duplicating
            await repo.upsert_user_limit(conn, is_postgres=False, user_id=41, resource="/api/v1/x", limit_per_min=6, burst=2)

        async with pool.transaction() as conn:
            listed = await repo.list_all(conn, is_postgres=False)
            assert [(r["scope"], r["id"], r["limit_per_min"]) for r in listed] == [
                ("role", 901, 20),
                ("role", 902, 1),
                ("user", 41, 6),
            ]
            assert await repo.user_limits(conn, is_postgres=False, user_id=41) == [
                {"resource": "/api/v1/x", "limit_per_min": 6, "burst": 2}
            ]
            # Only the unexpired role counts, and its name comes from the real roles table.
            assert await repo.role_limits(conn, is_postgres=False, user_id=41) == [
                {"resource": "/api/v1/x", "limit_per_min": 20, "burst": 4, "role_name": "rl-role"}
            ]

        user_limit, role_limit = await repo.effective_limits(pool, 41, "/api/v1/x")
        assert tuple(user_limit) == (6, 2)
        assert tuple(role_limit) == (20, 4)

        async with pool.transaction() as conn:
            await repo.clear_role_limits(conn, is_postgres=False, role_id=901)
            assert [r["id"] for r in await repo.list_all(conn, is_postgres=False) if r["scope"] == "role"] == [902]
    finally:
        await pool.close()

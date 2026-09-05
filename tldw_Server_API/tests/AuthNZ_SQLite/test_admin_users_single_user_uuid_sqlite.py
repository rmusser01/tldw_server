"""Regression tests for the single-user bootstrap uuid fix.

A fresh single-user SQLite install used to bootstrap the admin row without a
uuid, which made GET /api/v1/admin/users 500 (UserSummary required a UUID).
"""

import pytest


async def _fresh_pool(tmp_path, monkeypatch, name: str):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / name
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()
    return await get_db_pool(), db_path


@pytest.mark.asyncio
async def test_ensure_single_user_admin_user_assigns_uuid_sqlite(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

    pool, _ = await _fresh_pool(tmp_path, monkeypatch, "single_user_uuid.db")
    repo = AuthnzUsersRepo(db_pool=pool)

    await repo.ensure_single_user_admin_user(user_id=1)

    row = await repo.get_user_by_id(1)
    assert row is not None
    assert row.get("uuid"), "bootstrapped single-user row must carry a uuid"


@pytest.mark.asyncio
async def test_ensure_single_user_admin_user_preserves_uuid_sqlite(
    tmp_path, monkeypatch
):
    """Repeated ensure calls (it runs at startup) must not rotate the uuid.

    The COALESCE backfill only fills NULL; an already-assigned uuid stays
    stable across restarts. (The NULL-backfill path itself only exists on
    legacy schemas where users.uuid is nullable; it was verified against a
    real legacy database and cannot be reproduced on the current NOT NULL
    schema.)
    """
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

    pool, _ = await _fresh_pool(tmp_path, monkeypatch, "single_user_stable.db")
    repo = AuthnzUsersRepo(db_pool=pool)

    await repo.ensure_single_user_admin_user(user_id=1)
    first = await repo.get_user_by_id(1)
    assert first is not None and first.get("uuid")

    await repo.ensure_single_user_admin_user(user_id=1)
    second = await repo.get_user_by_id(1)
    assert second is not None
    assert second["uuid"] == first["uuid"]

    users, total = await repo.list_users(offset=0, limit=20)
    assert total == 1
    assert users[0]["uuid"] == first["uuid"]


def test_user_summary_tolerates_null_uuid():
    """One legacy NULL-uuid row must not 500 the admin users list."""
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import UserSummary

    summary = UserSummary(
        id=1,
        uuid=None,
        username="single_user",
        email="single_user@example.local",
        role="admin",
        is_active=True,
        is_verified=True,
        created_at="2026-09-05T01:44:56",
        storage_quota_mb=5120,
        storage_used_mb=0.0,
    )
    assert summary.uuid is None

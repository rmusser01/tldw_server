"""Regression tests for the single-user bootstrap uuid fix.

A fresh single-user SQLite install used to bootstrap the admin row without a
uuid, which made GET /api/v1/admin/users 500 (UserSummary required a UUID).
"""

import pathlib
from typing import AsyncIterator

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.database import (
    DatabasePool,
    get_db_pool,
    reset_db_pool,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

pytestmark = pytest.mark.unit


@pytest_asyncio.fixture
async def sqlite_pool(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[DatabasePool]:
    """Yield a fresh SQLite AuthNZ pool and reset shared state afterwards.

    Args:
        tmp_path: Per-test directory for the throwaway SQLite database file.
        monkeypatch: Used to point AUTH_MODE/DATABASE_URL at that file.

    Yields:
        The initialized :class:`DatabasePool` for the temporary database.
    """
    db_path = tmp_path / "single_user_uuid.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    try:
        yield pool
    finally:
        await reset_db_pool()
        reset_settings()


@pytest.mark.asyncio
async def test_ensure_single_user_admin_user_assigns_uuid_sqlite(
    sqlite_pool: DatabasePool,
) -> None:
    """The bootstrapped single-user admin row must carry a uuid.

    Regression: rows created without one made UserSummary validation fail and
    the admin users list return HTTP 500.
    """
    repo = AuthnzUsersRepo(db_pool=sqlite_pool)

    await repo.ensure_single_user_admin_user(user_id=1)

    row = await repo.get_user_by_id(1)
    assert row is not None
    assert row.get("uuid"), "bootstrapped single-user row must carry a uuid"


@pytest.mark.asyncio
async def test_ensure_single_user_admin_user_preserves_uuid_sqlite(
    sqlite_pool: DatabasePool,
) -> None:
    """Repeated ensure calls (it runs at startup) must not rotate the uuid.

    The COALESCE backfill only fills NULL; an already-assigned uuid stays
    stable across restarts. (The NULL-backfill path itself only exists on
    legacy schemas where users.uuid is nullable; it was verified against a
    real legacy database and cannot be reproduced on the current NOT NULL
    schema.)
    """
    repo = AuthnzUsersRepo(db_pool=sqlite_pool)

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


def test_user_summary_tolerates_null_uuid() -> None:
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

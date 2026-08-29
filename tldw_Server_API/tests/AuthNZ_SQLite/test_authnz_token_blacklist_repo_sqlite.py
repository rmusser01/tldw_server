from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.repos.token_blacklist_repo import (
    AuthnzTokenBlacklistRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest.mark.asyncio
async def test_authnz_token_blacklist_repo_sqlite(tmp_path, monkeypatch):
    """AuthnzTokenBlacklistRepo basic helpers should work on SQLite."""
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    now = datetime.now(timezone.utc).replace(microsecond=0)
    future = now + timedelta(hours=1)

    # Seed a simple user row for FK safety
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active, is_verified, role)
            VALUES (?, ?, ?, 1, 1, 'user')
            """,
            ("blacklist-sqlite-user", "blacklist-sqlite@example.com", "hash"),
        )
    user_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", "blacklist-sqlite-user"
    )
    assert user_id is not None

    repo = AuthnzTokenBlacklistRepo(pool)

    # Insert a blacklisted token
    await repo.insert_blacklisted_token(
        jti="sqlite-test-jti",
        user_id=int(user_id),
        token_type="access",
        expires_at=future,
        reason="unit-test",
        revoked_by=42,
        ip_address="127.0.0.1",
    )

    # Active expiry lookup should find it
    expiry = await repo.get_active_expiry_for_jti("sqlite-test-jti", now=now)
    assert expiry is not None

    # Stats should count it
    stats_global = await repo.get_blacklist_stats(now=now, user_id=None)
    assert stats_global["total"] >= 1
    assert stats_global["access_tokens"] >= 1

    stats_user = await repo.get_blacklist_stats(now=now, user_id=int(user_id))
    assert stats_user["total"] >= 1

    # Cleanup with a past cutoff should retain it
    deleted_none = await repo.cleanup_expired(now=now - timedelta(hours=1))
    assert deleted_none == 0

    # Cleanup with a future cutoff should eventually remove it
    deleted_some = await repo.cleanup_expired(now=now + timedelta(days=1))
    assert deleted_some >= 1

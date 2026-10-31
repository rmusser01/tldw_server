"""The Postgres storage-quota backfill copies non-default values exactly once (spec 2 §8)."""

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ import storage_quota_backfill
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_authnz_core_tables_pg,
    ensure_storage_quota_overrides_backfill_pg,
)

pytestmark = pytest.mark.integration
KEY = "limits.storage_quota_mb"


async def _add_user(pool, name: str, quota_mb: int) -> int:
    """Insert a user with the given legacy column value; return its id.

    Uses a raw asyncpg connection because the managed pool rejects direct users writes.
    """
    conn = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        return int(await conn.fetchval(
            "INSERT INTO users (username, email, password_hash, storage_quota_mb) VALUES ($1, $2, 'x', $3) RETURNING id",
            name, f"{name}@example.com", quota_mb,
        ))
    finally:
        await conn.close()


async def _reset_marker(pool) -> None:
    """Drop the marker table; the fixture's truncate does not cover it, so it would persist across tests."""
    conn = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        await conn.execute("DROP TABLE IF EXISTS authnz_data_backfills")
    finally:
        await conn.close()


async def _override(pool, user_id: int):
    """The user's limits.storage_quota_mb value_json, or None."""
    return await pool.fetchval("SELECT value_json FROM user_config_overrides WHERE user_id = $1 AND key = $2", user_id, KEY)


async def test_pg_backfill_copies_non_default_values(test_db_pool, monkeypatch: pytest.MonkeyPatch) -> None:
    """2048 is copied; 5120 and the configured default are not."""
    monkeypatch.setattr(storage_quota_backfill, "skip_values", lambda: [5120, 10240])
    pool = test_db_pool
    assert await ensure_authnz_core_tables_pg(pool)  # the fixture DB has users but no user_config_overrides
    await _reset_marker(pool)
    u_custom = await _add_user(pool, "pgcustom", 2048)
    u_default = await _add_user(pool, "pgdefault", 5120)
    u_configured = await _add_user(pool, "pgconfigured", 10240)
    assert await ensure_storage_quota_overrides_backfill_pg(pool) is True
    assert await _override(pool, u_custom) == "2048"
    assert await _override(pool, u_default) is None and await _override(pool, u_configured) is None


async def test_pg_backfill_runs_once_and_never_resurrects(test_db_pool, monkeypatch: pytest.MonkeyPatch) -> None:
    """After the first run, a deleted override is not re-created by a second run."""
    monkeypatch.setattr(storage_quota_backfill, "skip_values", lambda: [5120])
    pool = test_db_pool
    assert await ensure_authnz_core_tables_pg(pool)
    await _reset_marker(pool)
    user_id = await _add_user(pool, "pgonce", 2048)
    assert await ensure_storage_quota_overrides_backfill_pg(pool) is True
    assert await _override(pool, user_id) == "2048"
    await pool.execute("DELETE FROM user_config_overrides WHERE user_id = $1 AND key = $2", user_id, KEY)
    assert await ensure_storage_quota_overrides_backfill_pg(pool) is True
    assert await _override(pool, user_id) is None

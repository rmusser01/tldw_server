"""TASK-15671: the account-lock columns on PostgreSQL (type, serving-pool admission, users write guard)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

_LOCK_COLUMNS = (("users", "security_locked_at"), ("users", "credentials_valid_after"), ("api_keys", "suspended_at"))


@pytest.mark.asyncio
async def test_lock_columns_exist_and_are_timestamptz_pg(isolated_test_environment) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_authnz_core_tables_pg

    pool = await get_db_pool()
    assert await ensure_authnz_core_tables_pg(pool)
    for table, column in _LOCK_COLUMNS:
        row = await pool.fetchrow(
            "SELECT data_type, is_nullable, column_default FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2",
            table,
            column,
        )
        assert row is not None, (table, column)
        assert (row["data_type"], row["is_nullable"], row["column_default"]) == (
            "timestamp with time zone",
            "YES",
            None,
        )


@pytest.mark.asyncio
async def test_raw_update_of_lock_columns_passes_the_users_write_guard_pg(isolated_test_environment) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_authnz_core_tables_pg
    from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

    pool = await get_db_pool()
    assert await ensure_authnz_core_tables_pg(pool)
    user_id = await create_authnz_test_user(pool, username="locktest", email="locktest@example.com")
    await pool.execute("UPDATE users SET security_locked_at = now() WHERE id = $1", user_id)
    await pool.execute("UPDATE users SET credentials_valid_after = now() WHERE id = $1", user_id)
    row = await pool.fetchrow("SELECT security_locked_at, credentials_valid_after FROM users WHERE id = $1", user_id)
    assert row["security_locked_at"] is not None
    assert row["credentials_valid_after"] is not None

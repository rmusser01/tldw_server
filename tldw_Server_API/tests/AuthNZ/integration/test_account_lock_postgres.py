"""TASK-15671: the account-lock columns and the lock rule on PostgreSQL.

The state tests are the SQLite module's, imported below and run here against PostgreSQL through
this module's ``lock_db`` fixture (the versioned writer and the timestamp comparisons differ per backend).
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import asyncpg
import pytest

from tldw_Server_API.tests.AuthNZ_SQLite.test_account_lock_schema_sqlite import (  # noqa: F401 - collected here
    OLD,
    OWNER_PASSWORD,
    test_clear_lock_on_an_unlocked_account_does_nothing,
    test_clear_lock_suspends_only_keys_created_in_the_last_seven_days,
    test_email_change_to_an_address_another_account_just_released_is_refused,
    test_plain_unlock_racing_a_take_back_of_a_locked_account_does_not_win,
    test_previous_address_lookup_refuses_an_ambiguous_match,
    test_set_lock_is_first_writer_wins,
    test_take_back_is_refused_and_the_lock_kept_when_another_user_holds_the_address,
    test_take_back_of_an_unlocked_account_leaves_nothing_made_while_it_ran,
    test_take_back_on_an_already_unlocked_account_still_suspends_keys_and_drops_new_sso_links,
    test_take_back_removes_sso_links_made_since_the_week_before_the_email_change_even_if_the_email_moved_again,
    test_take_back_restores_the_address_and_kills_the_password,
    test_two_unlocks_of_one_lock_only_one_wins,
    test_valid_after_is_strictly_later_than_the_lock_second_and_already_begun,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def lock_db(isolated_test_environment):
    """The owner (email ``OLD``, password ``OWNER_PASSWORD``) on PostgreSQL, and a way to age stored times."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.password_service import hash_password
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_authnz_core_tables_pg
    from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user
    from tldw_Server_API.tests.helpers.pg_env import get_pg_env

    _client, db_name = isolated_test_environment
    pool = await get_db_pool()
    assert await ensure_authnz_core_tables_pg(pool)
    user_id = await create_authnz_test_user(
        pool, username="owner", email=OLD, password_hash=hash_password(OWNER_PASSWORD)
    )

    async def age(table: str, column: str, row_id: int, delta: timedelta) -> None:
        """Move a stored time back by ``delta``, outside the AuthNZ write guard, as time passing would."""
        pg = get_pg_env()
        conn = await asyncpg.connect(
            host=pg.host, port=int(pg.port), user=pg.user, password=pg.password, database=db_name
        )
        try:
            await conn.execute(
                f"UPDATE {table} SET {column} = {column} - $1::interval WHERE id = $2",  # nosec B608 - test constants
                delta,
                row_id,
            )
        finally:
            await conn.close()

    return SimpleNamespace(pool=pool, user_id=user_id, backend="postgres", age=age)

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

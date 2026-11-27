"""TASK-15671: the account-lock columns and the lock rule on PostgreSQL.

The state tests are the SQLite module's, imported below and run here against PostgreSQL through
this module's ``lock_db`` fixture (the versioned writer and the timestamp comparisons differ per backend).
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ.account_lock import AccountSecurityLockedError
from tldw_Server_API.tests.AuthNZ_SQLite._account_lock_fixtures import lock_user, unlock_user
from tldw_Server_API.tests.AuthNZ_SQLite.test_account_lock_schema_sqlite import (  # noqa: F401 - collected here
    OLD,
    OWNER_PASSWORD,
    test_a_second_take_back_to_the_restored_address_still_replaces_the_password,
    test_clear_lock_on_an_unlocked_account_does_nothing,
    test_clear_lock_suspends_only_keys_created_in_the_last_seven_days,
    test_email_change_back_to_your_own_previous_address_is_allowed,
    test_email_change_to_an_address_another_account_just_released_is_refused,
    test_email_change_to_an_address_released_eight_days_ago_is_allowed,
    test_email_in_use_check_ignores_letter_case,
    test_plain_unlock_racing_a_take_back_of_a_locked_account_does_not_win,
    test_plain_unlock_that_wins_during_a_take_back_leaves_the_intruder_nothing,
    test_previous_address_lookup_refuses_an_ambiguous_match,
    test_recently_released_address_is_refused_in_any_letter_case,
    test_set_lock_is_first_writer_wins,
    test_take_back_is_refused_and_the_lock_kept_when_another_user_holds_the_address,
    test_take_back_is_refused_and_the_lock_kept_when_the_address_is_taken_while_it_runs,
    test_take_back_of_an_unlocked_account_leaves_nothing_made_while_it_ran,
    test_take_back_on_an_already_unlocked_account_still_suspends_keys_and_drops_new_sso_links,
    test_take_back_removes_sso_links_made_since_the_week_before_the_email_change_even_if_the_email_moved_again,
    test_take_back_restores_the_address_and_kills_the_password,
    test_take_back_that_never_wins_its_compare_and_set_gives_up_with_the_account_locked,
    test_take_back_that_restores_nothing_is_refused_and_keeps_the_lock,
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

    return SimpleNamespace(pool=pool, user_id=user_id, backend="postgres", age=age, db_name=db_name)

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


# --- enforcement (Task 3) ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mfa_challenge_started_before_an_unlock_is_refused_pg(isolated_test_environment, monkeypatch) -> None:
    """A challenge cached before the lock carries an ``issued_at`` older than the unlock's valid_after."""
    import pyotp

    from tldw_Server_API.app.core.AuthNZ import mfa_service as mfa_module
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.password_service import hash_password
    from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

    client, _db_name = isolated_test_environment
    monkeypatch.setenv("TOTP_ENCRYPTION_KEY", "")
    monkeypatch.setattr(mfa_module, "_mfa_service", None)  # a previous test's service holds its pool
    secret = pyotp.random_base32()

    async def _mfa_user() -> int:
        """The user and its TOTP, written on the app's loop (the pool is bound to it)."""
        user_id = await create_authnz_test_user(
            await get_db_pool(),
            username="mfa-owner",
            email="mfa-owner@example.com",
            password_hash=hash_password(OWNER_PASSWORD),
            is_verified=True,
        )
        mfa = mfa_module.get_mfa_service()
        await mfa.initialize()
        assert await mfa.enable_mfa(user_id, secret, ["backup-code-1"])
        return user_id

    user_id = client.portal.call(_mfa_user)
    challenge = client.post("/api/v1/auth/login", data={"username": "mfa-owner", "password": OWNER_PASSWORD})
    assert challenge.status_code == 202 and challenge.json()["mfa_required"] is True, challenge.text
    client.portal.call(lock_user, user_id)
    client.portal.call(unlock_user, user_id)

    response = client.post(
        "/api/v1/auth/mfa/login",
        json={"session_token": challenge.json()["session_token"], "mfa_token": pyotp.TOTP(secret).now()},
    )
    assert response.status_code == 400 and response.json()["detail"] == "MFA session expired or invalid"


@pytest.mark.asyncio
async def test_sign_in_waits_for_an_uncommitted_lock_and_is_refused_pg(lock_db) -> None:
    """Under READ COMMITTED a plain read would see no lock and insert the session; FOR SHARE waits for it."""
    from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
    from tldw_Server_API.tests.helpers.pg_env import get_pg_env

    pg = get_pg_env()
    locker = await asyncpg.connect(
        host=pg.host, port=int(pg.port), user=pg.user, password=pg.password, database=lock_db.db_name
    )
    try:
        lock_tx = locker.transaction()
        await lock_tx.start()
        await locker.execute("UPDATE users SET security_locked_at = now() WHERE id = $1", lock_db.user_id)
        manager = await get_session_manager()
        sign_in = asyncio.create_task(
            manager.create_session(user_id=lock_db.user_id, access_token="a", refresh_token="r")
        )
        await asyncio.sleep(0.3)
        assert not sign_in.done()
        await lock_tx.commit()
        with pytest.raises(AccountSecurityLockedError):
            await sign_in
    finally:
        await locker.close()
    assert await lock_db.pool.fetchval("SELECT COUNT(*) FROM sessions WHERE user_id = $1", lock_db.user_id) == 0

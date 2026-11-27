"""TASK-15679: the account-lock columns and the lock rule on PostgreSQL.

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
    sign_in = None
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
        if sign_in is not None:  # a failed assert above leaves it waiting on the uncommitted lock
            sign_in.cancel()
            await asyncio.gather(sign_in, return_exceptions=True)
        await locker.close()
    assert await lock_db.pool.fetchval("SELECT COUNT(*) FROM sessions WHERE user_id = $1", lock_db.user_id) == 0


async def _apps_db(lock_db) -> SimpleNamespace:
    """``lock_db`` on the pool of the running loop. Run on the app's loop (``client.portal.call``): its
    background pollers recreate a pool bound to another loop, which would close one in use."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    return SimpleNamespace(pool=await get_db_pool(), user_id=lock_db.user_id, backend="postgres")


@pytest.mark.asyncio
async def test_take_back_waits_for_an_in_flight_identity_write_and_then_overrides_it_pg(
    isolated_test_environment, lock_db
) -> None:
    """The intruder's email and password change passed its checks before the owner's take-back began; the
    newer address had unlocked first, so the account is not locked. A pool transaction plays that request."""
    from tldw_Server_API.app.core.AuthNZ.account_lock import assert_identity_write_allowed, clear_lock
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService, hash_password
    from tldw_Server_API.app.core.DB_Management.user_profile_writes import update_user_email
    from tldw_Server_API.tests.AuthNZ_SQLite.test_account_lock_schema_sqlite import (
        NEW,
        _move_email,
        _password_verifies,
        _row,
    )

    client, _db_name = isolated_test_environment
    intruder_password = "Intruder-Pass-2026!"

    async def _race() -> tuple:
        db = await _apps_db(lock_db)
        await _move_email(db, NEW)  # OLD is the previous address for 7 days
        take_back = None
        try:
            async with db.pool.transaction() as request:
                await assert_identity_write_allowed(request, db.user_id, is_postgres=True)
                take_back = asyncio.create_task(clear_lock(db.user_id, take_back=OLD))
                await asyncio.sleep(0.5)
                waited = not take_back.done()
                await update_user_email(request, backend="postgres", user_id=db.user_id, email="third@example.com")
                await PasswordService().set_password(
                    db.user_id, hash_password(intruder_password), request, is_postgres=True
                )
            valid_after = await take_back
        finally:
            if take_back is not None:
                take_back.cancel()
                await asyncio.gather(take_back, return_exceptions=True)
        return await _row(db), await _password_verifies(db, intruder_password), waited, valid_after

    row, intruder_password_works, waited, valid_after = client.portal.call(_race)
    assert (row["email"], row["security_locked_at"]) == (OLD, None)
    assert intruder_password_works is False
    assert waited is True and valid_after is not None


@pytest.mark.asyncio
async def test_identity_write_guard_lets_a_session_revoke_on_another_connection_through_pg(
    isolated_test_environment, lock_db
) -> None:
    """A password reset on PostgreSQL revokes sessions on another connection while it holds the guard's row
    lock. The blacklist rows' foreign key takes KEY SHARE on the users row, which FOR UPDATE would block until
    the request commits: the request would wait on itself."""
    from tldw_Server_API.app.core.AuthNZ.account_lock import assert_identity_write_allowed
    from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist
    from tldw_Server_API.tests.AuthNZ_SQLite.test_account_lock_schema_sqlite import _live_session_count, _session

    client, _db_name = isolated_test_environment

    async def _reset_with_inline_revoke() -> tuple[int, int]:
        db = await _apps_db(lock_db)
        await _session(db, "before-the-reset")
        async with db.pool.transaction() as request:
            await assert_identity_write_allowed(request, db.user_id, is_postgres=True)
            await asyncio.wait_for(get_token_blacklist().revoke_all_user_tokens(db.user_id, "Password reset"), 5)
        blacklisted = await db.pool.fetchval("SELECT COUNT(*) FROM token_blacklist WHERE user_id = $1", db.user_id)
        return blacklisted, await _live_session_count(db)

    assert client.portal.call(_reset_with_inline_revoke) == (2, 0)


async def _pg_key(lock_db) -> tuple[dict, "APIKeyManager"]:
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager

    manager = APIKeyManager(lock_db.pool)
    await manager.initialize()
    return await manager.create_api_key(user_id=lock_db.user_id, name="pg-lock"), manager


async def test_key_lookup_carries_owner_state_pg(lock_db) -> None:
    from tldw_Server_API.app.core.AuthNZ.account_lock import set_lock
    from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo

    created, _manager = await _pg_key(lock_db)
    repo = AuthnzApiKeysRepo(db_pool=lock_db.pool)
    identifier = created["key"].removeprefix("tldw_").split(".")[0]
    first = await repo.fetch_active_by_key_id(identifier)
    by_hash_before = await repo.fetch_active_by_hash_candidates([first["key_hash"]])
    await set_lock(lock_db.user_id)
    by_id_after = await repo.fetch_active_by_key_id(identifier)
    by_hash_after = await repo.fetch_active_by_hash_candidates([first["key_hash"]])
    for before in (first, by_hash_before):
        assert before["owner_security_locked_at"] is None and before["owner_is_active"] is True
        assert before["suspended_at"] is None
    for after in (by_id_after, by_hash_after):
        assert after["owner_security_locked_at"] is not None


async def test_suspended_key_is_invalid_pg(lock_db) -> None:
    from tldw_Server_API.app.core.AuthNZ.account_lock import set_lock

    created, manager = await _pg_key(lock_db)
    info = await manager.validate_api_key(created["key"])
    assert info["user_id"] == lock_db.user_id
    assert "owner_is_active" not in info and "owner_security_locked_at" not in info
    await set_lock(lock_db.user_id)
    assert await manager.validate_api_key(created["key"]) is None
    await unlock_user(lock_db.user_id)
    assert await manager.validate_api_key(created["key"]) is None
    suspended = await lock_db.pool.fetchval("SELECT suspended_at FROM api_keys WHERE id = $1", int(created["id"]))
    assert suspended is not None


# --- unlock through the mailed link (Task 6), at the account_lock level -----------------------------


@pytest.fixture
def unlock_audit(monkeypatch) -> list[tuple]:
    """The ``(event type, metadata)`` of every account audit event ``redeem_unlock`` writes."""
    from tldw_Server_API.app.core.AuthNZ import account_lock

    written: list[tuple] = []

    async def _record(event_type, **kwargs) -> None:
        written.append((event_type.value, kwargs.get("metadata")))

    monkeypatch.setattr(account_lock, "write_account_audit", _record)
    return written


async def _pg_user(lock_db) -> dict:
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

    return await AuthnzUsersRepo(lock_db.pool).get_user_by_id(lock_db.user_id)


def _unlock_claims(email: str, *, restores: bool) -> dict:
    """The claims of an unlock link as ``send_lock_mail`` mints them."""
    return {"email": email, "purpose": "account_unlock", "restore_email": restores}


async def _redeem(lock_db, claims: dict) -> bool:
    from tldw_Server_API.app.core.AuthNZ.account_lock import redeem_unlock

    return await redeem_unlock(await _pg_user(lock_db), claims, via="magic_link", ip_address=None, user_agent=None)


async def test_unlock_link_signs_in_with_one_session_and_clears_the_lock_pg(lock_db, unlock_audit) -> None:
    from tldw_Server_API.app.core.AuthNZ.account_lock import magic_link_email_allowed, magic_link_email_current
    from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
    from tldw_Server_API.tests.AuthNZ_SQLite.test_account_lock_schema_sqlite import _live_session_count, _session

    await _session(lock_db, "before-the-lock")
    await lock_user(lock_db.user_id)
    owner = _unlock_claims(OLD, restores=False)
    assert await magic_link_email_allowed(await _pg_user(lock_db), owner) is True
    assert await _redeem(lock_db, owner) is True
    await (await get_session_manager()).create_session(user_id=lock_db.user_id, access_token="a", refresh_token="r")
    row = await _pg_user(lock_db)
    assert row["security_locked_at"] is None and row["credentials_valid_after"] is not None
    assert await _live_session_count(lock_db) == 1
    assert await magic_link_email_current(lock_db.user_id, owner) is True
    assert unlock_audit == [("auth.account.unlocked", {"via": "magic_link", "took_back_address": False})]


async def test_link_to_previous_address_takes_the_account_back_pg(lock_db, unlock_audit) -> None:
    from tldw_Server_API.app.core.AuthNZ.account_lock import magic_link_email_allowed, magic_link_email_current
    from tldw_Server_API.tests.AuthNZ_SQLite.test_account_lock_schema_sqlite import NEW, _move_email, _password_verifies

    await _move_email(lock_db, NEW)
    await lock_user(lock_db.user_id)
    owner, intruder = _unlock_claims(OLD, restores=True), _unlock_claims(NEW, restores=False)
    locked = await _pg_user(lock_db)
    assert await magic_link_email_allowed(locked, owner) is True
    assert await magic_link_email_allowed(locked, intruder) is True
    assert await _redeem(lock_db, owner) is True
    row = await _pg_user(lock_db)
    assert (row["email"], row["previous_email"], row["security_locked_at"]) == (OLD, None, None)
    assert await _password_verifies(lock_db) is False
    assert await magic_link_email_allowed(row, intruder) is False  # the newer address's link is dead
    assert await magic_link_email_current(lock_db.user_id, owner) is True
    assert await magic_link_email_current(lock_db.user_id, intruder) is False
    assert unlock_audit == [("auth.account.unlocked", {"via": "magic_link", "took_back_address": True})]


async def test_take_back_after_the_newer_address_unlocked_first_leaves_the_intruder_nothing_pg(
    lock_db, unlock_audit
) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
    from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
    from tldw_Server_API.tests.AuthNZ_SQLite.test_account_lock_schema_sqlite import (
        NEW,
        _live_session_count,
        _move_email,
    )

    await _move_email(lock_db, NEW)
    recent, keys = await _pg_key(lock_db)  # made in the week before the lock: the unlock suspends it
    await lock_user(lock_db.user_id)
    assert await _redeem(lock_db, _unlock_claims(NEW, restores=False)) is True
    # The intruder signs in, makes a key and lifts the suspension of the recent one.
    await (await get_session_manager()).create_session(user_id=lock_db.user_id, access_token="a", refresh_token="r")
    created = await keys.create_api_key(user_id=lock_db.user_id, name="intruder")
    repo = AuthnzApiKeysRepo(db_pool=lock_db.pool)
    assert await repo.unsuspend_key_for_user(key_id=int(recent["id"]), user_id=lock_db.user_id) is True

    assert await _redeem(lock_db, _unlock_claims(OLD, restores=True)) is True
    assert (await _pg_user(lock_db))["email"] == OLD
    assert await _live_session_count(lock_db) == 0
    assert await keys.validate_api_key(created["key"]) is None
    assert await keys.validate_api_key(recent["key"]) is None
    assert [took_back["took_back_address"] for _event, took_back in unlock_audit] == [False, True]


@pytest.mark.parametrize("locked_again", [False, True], ids=["unlocked", "locked-again"])
async def test_take_back_link_wins_after_the_intruder_unlocked_and_reverted_the_email_pg(
    lock_db, unlock_audit, locked_again
) -> None:
    """The intruder unlocks first, changes the email back to the owner's address (the stored previous address
    is then the email too) and makes a key and an SSO link; the owner's take-back link still wins."""
    from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
    from tldw_Server_API.tests.AuthNZ_SQLite.test_account_lock_schema_sqlite import (
        NEW,
        _identity_links,
        _link_identity,
        _live_session_count,
        _move_email,
        _password_verifies,
    )

    await _move_email(lock_db, NEW)
    await lock_user(lock_db.user_id)
    assert await _redeem(lock_db, _unlock_claims(NEW, restores=False)) is True
    await (await get_session_manager()).create_session(user_id=lock_db.user_id, access_token="a", refresh_token="r")
    await _move_email(lock_db, OLD)  # inside 7 days, so previous_email stays OLD
    assert (await _pg_user(lock_db))["previous_email"] == OLD
    created, keys = await _pg_key(lock_db)
    await _link_identity(lock_db, "intruder")
    if locked_again:
        await lock_user(lock_db.user_id)

    assert await _redeem(lock_db, _unlock_claims(OLD, restores=True)) is True
    row = await _pg_user(lock_db)
    assert (row["email"], row["previous_email"], row["security_locked_at"]) == (OLD, None, None)
    assert await _password_verifies(lock_db) is False
    assert await _live_session_count(lock_db) == 0
    assert await keys.validate_api_key(created["key"]) is None
    assert await _identity_links(lock_db) == []
    assert [took_back["took_back_address"] for _event, took_back in unlock_audit] == [False, True]


async def test_list_users_carries_the_lock_pg(lock_db) -> None:
    from tldw_Server_API.app.core.AuthNZ.account_lock import set_lock
    from tldw_Server_API.app.core.AuthNZ.password_service import hash_password
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

    other_id = await create_authnz_test_user(
        lock_db.pool, username="other", email="other@example.com", password_hash=hash_password(OWNER_PASSWORD)
    )
    await set_lock(lock_db.user_id)
    users, total = await AuthnzUsersRepo(lock_db.pool).list_users(offset=0, limit=10)
    by_id = {u["id"]: u for u in users}
    assert total == 2
    assert by_id[lock_db.user_id]["security_locked_at"] is not None
    assert by_id[other_id]["security_locked_at"] is None

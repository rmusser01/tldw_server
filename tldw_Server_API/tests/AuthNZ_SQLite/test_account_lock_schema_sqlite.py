"""TASK-15679: the account-lock columns (migration 109, users write guard) and the lock rule.

The state tests (those taking ``lock_db``) also run against PostgreSQL: test_account_lock_postgres.py
imports them and supplies its own ``lock_db``.
"""

import asyncio
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Audit.unified_audit_service import AuditEventType
from tldw_Server_API.app.core.AuthNZ import api_key_audit, migrations
from tldw_Server_API.app.core.AuthNZ.account_lock import (
    AccountSecurityLockedError,
    StaleCredentialError,
    TakeBackUnavailableError,
    assert_account_not_locked,
    clear_lock,
    is_security_locked,
    is_user_security_locked,
    security_mail_recipients,
    set_lock,
    take_back_address,
    write_account_audit,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.email_change import EmailChangeRefused, authorize_own_email_change
from tldw_Server_API.app.core.AuthNZ.migrations import migration_109_add_account_lock_columns
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService, hash_password
from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
from tldw_Server_API.app.core.AuthNZ.repos.federated_identity_repo import FederatedIdentityRepo
from tldw_Server_API.app.core.AuthNZ.repos.identity_provider_repo import IdentityProviderRepo
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.user_profile_writes import update_user_email
from tldw_Server_API.app.core.exceptions import ServiceInitializationError
from tldw_Server_API.app.core.Utils.iso_datetime import parse_iso_utc
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

OLD, NEW = "old@example.com", "new@example.com"
OWNER_PASSWORD = "Owner-Pass-2026!"


def _db(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "users.db")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)")
    conn.execute("CREATE TABLE api_keys (id INTEGER PRIMARY KEY, user_id INTEGER)")
    return conn


def _names(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def test_migration_109_adds_the_lock_columns(tmp_path) -> None:
    db = _db(tmp_path)
    migration_109_add_account_lock_columns(db)
    assert {"security_locked_at", "credentials_valid_after"} <= set(_names(db, "users"))
    assert "suspended_at" in _names(db, "api_keys")


def test_migration_109_is_idempotent(tmp_path) -> None:
    db = _db(tmp_path)
    migration_109_add_account_lock_columns(db)
    migration_109_add_account_lock_columns(db)
    assert _names(db, "users").count("security_locked_at") == 1
    assert _names(db, "api_keys").count("suspended_at") == 1


def test_migration_109_skips_missing_tables(tmp_path) -> None:
    """Synthetic databases migrate without a users or api_keys table (cf. migrations 100, 107 and 108)."""
    with sqlite3.connect(tmp_path / "synthetic.db") as conn:
        migration_109_add_account_lock_columns(conn)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert not tables & {"users", "api_keys"}


def test_migration_109_is_registered_last() -> None:
    assert migrations.get_authnz_migrations()[-1].version == 109


@pytest.fixture
async def owner_db(tmp_path, monkeypatch, fresh_services):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "synthetic-signing-key-for-regression-tests-only")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    reset_settings()
    pool = await get_db_pool()
    user_id = await create_authnz_test_user(
        pool, username="owner", email=OLD, password_hash=hash_password(OWNER_PASSWORD)
    )
    try:
        yield SimpleNamespace(pool=pool, user_id=user_id)
    finally:
        reset_settings()


@pytest.fixture
def lock_db(owner_db):
    """The owner (email ``OLD``, password ``OWNER_PASSWORD``) on SQLite, and a way to age stored times."""

    async def age(table: str, column: str, row_id: int, delta: timedelta) -> None:
        """Move a stored time back by ``delta``, outside the AuthNZ write guard, as time passing would."""
        with sqlite3.connect(owner_db.pool.db_path) as conn:
            conn.execute(
                f"UPDATE {table} SET {column} = datetime({column}, ?) WHERE id = ?",  # nosec B608 - test constants
                (f"-{int(delta.total_seconds())} seconds", row_id),
            )

    return SimpleNamespace(pool=owner_db.pool, user_id=owner_db.user_id, backend="sqlite", age=age)


@pytest.mark.asyncio
async def test_raw_update_of_lock_columns_passes_the_users_write_guard(owner_db) -> None:
    await owner_db.pool.execute(
        "UPDATE users SET security_locked_at = CURRENT_TIMESTAMP, credentials_valid_after = CURRENT_TIMESTAMP "
        "WHERE id = ?",
        owner_db.user_id,
    )
    row = dict(
        await owner_db.pool.fetchone(
            "SELECT security_locked_at, credentials_valid_after FROM users WHERE id = ?", owner_db.user_id
        )
    )
    assert row["security_locked_at"] is not None
    assert row["credentials_valid_after"] is not None


# --- the rule, as pure functions ----------------------------------------------------------


def test_locked_user_raises_423_with_the_code() -> None:
    with pytest.raises(AccountSecurityLockedError) as exc:
        assert_account_not_locked({"security_locked_at": "2026-11-26 10:00:00"})
    assert exc.value.status_code == 423 and exc.value.detail["code"] == "account_locked"


def test_an_unreadable_lock_value_still_counts_as_locked() -> None:
    assert is_security_locked({"security_locked_at": "not a date"}) is True


@pytest.mark.parametrize(
    "stored",
    ["2026-11-26 10:00:00", "2026-11-26T10:00:00+00:00", datetime(2026, 11, 26, 10, tzinfo=timezone.utc)],
)
def test_token_issued_before_unlock_is_stale(stored) -> None:
    user = {"security_locked_at": None, "credentials_valid_after": stored}
    at = int(datetime(2026, 11, 26, 10, tzinfo=timezone.utc).timestamp())
    with pytest.raises(StaleCredentialError):
        assert_account_not_locked(user, token_iat=at - 1)
    assert_account_not_locked(user, token_iat=at)  # the same second passes (Decision 10)
    assert_account_not_locked(user)  # no token, nothing to compare


def test_mail_goes_to_the_previous_address_for_seven_days_only() -> None:
    now = datetime(2026, 11, 26, tzinfo=timezone.utc)
    recent = {"email": NEW, "previous_email": OLD, "email_changed_at": now - timedelta(days=6)}
    stale = {**recent, "email_changed_at": now - timedelta(days=8)}
    assert security_mail_recipients(recent, now=now) == [(NEW, False), (OLD, True)]
    assert security_mail_recipients(stale, now=now) == [(NEW, False)]
    assert take_back_address(recent, {"email": OLD, "restore_email": True}, now=now) == OLD
    assert take_back_address(recent, {"email": NEW, "restore_email": True}, now=now) is None
    assert take_back_address(stale, {"email": OLD, "restore_email": True}, now=now) is None


# --- the state, on a real database (also collected by test_account_lock_postgres.py) ------


async def _row(db, user_id: int | None = None) -> dict:
    return dict(await db.pool.fetchone("SELECT * FROM users WHERE id = ?", user_id or db.user_id))


async def _user(db, username: str, email: str) -> int:
    return await create_authnz_test_user(
        db.pool, username=username, email=email, password_hash=hash_password(OWNER_PASSWORD)
    )


async def _move_email(db, email: str, user_id: int | None = None) -> None:
    async with db.pool.transaction() as conn:
        await update_user_email(conn, backend=db.backend, user_id=user_id or db.user_id, email=email)


async def _key(db, name: str) -> int:
    return await AuthnzApiKeysRepo(db.pool).create_api_key_row(
        user_id=db.user_id, key_hash=uuid.uuid4().hex, key_identifier=None, key_prefix="tldw_", name=name,
        description=None, scope="read", expires_at=None, rate_limit=None, allowed_ips=None, metadata=None,
    )


async def _suspended_ids(db) -> set[int]:
    rows = await db.pool.fetchall("SELECT id FROM api_keys WHERE user_id = ? AND suspended_at IS NOT NULL", db.user_id)
    return {int(row["id"]) for row in rows}


async def _unsuspended_active_keys(db) -> list[int]:
    rows = await db.pool.fetchall(
        "SELECT id FROM api_keys WHERE user_id = ? AND status = 'active' AND suspended_at IS NULL", db.user_id
    )
    return [int(row["id"]) for row in rows]


async def _link_identity(db, subject: str) -> int:
    providers = IdentityProviderRepo(db.pool)
    await providers.ensure_tables()
    provider = await providers.create_provider(
        slug=f"idp-{subject}", provider_type="oidc", owner_scope_type="global", owner_scope_id=None, enabled=True,
        issuer=f"https://{subject}.example.com", claim_mapping={}, provisioning_policy={},
    )
    links = FederatedIdentityRepo(db.pool)
    await links.ensure_tables()
    link = await links.upsert_identity(identity_provider_id=provider["id"], external_subject=subject, user_id=db.user_id)
    return int(link["id"])


async def _identity_links(db) -> list[dict]:
    return await FederatedIdentityRepo(db.pool).list_for_user(user_id=db.user_id)


async def _session(db, tag: str) -> int:
    now = datetime.now(timezone.utc)
    return await AuthnzSessionsRepo(db.pool).create_session_record(
        user_id=db.user_id, token_hash=f"h-{tag}", refresh_token_hash=f"r-{tag}", encrypted_token="enc",
        encrypted_refresh="enc", expires_at=now + timedelta(hours=1), refresh_expires_at=now + timedelta(days=7),
        ip_address="127.0.0.1", user_agent="pytest", device_id=None, access_jti=f"aj-{tag}", refresh_jti=f"rj-{tag}",
    )


async def _live_session_count(db) -> int:
    return len(await AuthnzSessionsRepo(db.pool).get_active_sessions_for_user(db.user_id))


async def test_set_lock_is_first_writer_wins(lock_db) -> None:
    first, second = await set_lock(lock_db.user_id), await set_lock(lock_db.user_id)
    assert first is not None and second is None
    assert await is_user_security_locked(lock_db.user_id) is True
    assert await is_user_security_locked(lock_db.user_id + 1000) is False


async def test_clear_lock_suspends_only_keys_created_in_the_last_seven_days(lock_db) -> None:
    keys = {}
    for name, age in (("ten_days", timedelta(days=10)), ("six_days", timedelta(days=6)), ("one_hour", timedelta(hours=1))):
        keys[name] = await _key(lock_db, name)
        await lock_db.age("api_keys", "created_at", keys[name], age)
    await set_lock(lock_db.user_id)
    await clear_lock(lock_db.user_id)
    row = await _row(lock_db)
    assert await _suspended_ids(lock_db) == {keys["six_days"], keys["one_hour"]}
    assert row["security_locked_at"] is None and parse_iso_utc(row["credentials_valid_after"]) is not None


async def test_clear_lock_on_an_unlocked_account_does_nothing(lock_db) -> None:
    assert await clear_lock(lock_db.user_id) is None
    assert (await _row(lock_db))["credentials_valid_after"] is None


async def test_valid_after_is_strictly_later_than_the_lock_second_and_already_begun(lock_db) -> None:
    locked_at = await set_lock(lock_db.user_id)
    valid_after = await clear_lock(lock_db.user_id)
    assert int(valid_after.timestamp()) > int(locked_at.timestamp()) and valid_after <= datetime.now(timezone.utc)


async def test_two_unlocks_of_one_lock_only_one_wins(lock_db) -> None:
    await set_lock(lock_db.user_id)
    results = await asyncio.gather(clear_lock(lock_db.user_id), clear_lock(lock_db.user_id))
    assert sum(result is not None for result in results) == 1


async def test_take_back_restores_the_address_and_kills_the_password(lock_db) -> None:
    await _move_email(lock_db, NEW)
    await set_lock(lock_db.user_id)
    await clear_lock(lock_db.user_id, take_back=OLD)
    row = await _row(lock_db)
    assert row["email"] == OLD and row["previous_email"] is None and row["security_locked_at"] is None
    assert PasswordService().verify_password(OWNER_PASSWORD, row["password_hash"])[0] is False


async def test_take_back_on_an_already_unlocked_account_still_suspends_keys_and_drops_new_sso_links(lock_db) -> None:
    uid = lock_db.user_id
    await _move_email(lock_db, NEW)
    first, second = await _key(lock_db, "first"), await _key(lock_db, "second")
    await set_lock(uid)
    await clear_lock(uid)  # the intruder's unlock, through the link mailed to the newer address
    assert await AuthnzApiKeysRepo(lock_db.pool).unsuspend_key_for_user(key_id=first, user_id=uid) is True
    third = await _key(lock_db, "third")
    await _link_identity(lock_db, "intruder")
    await clear_lock(uid, take_back=OLD)
    assert await _suspended_ids(lock_db) == {first, second, third}
    assert await _identity_links(lock_db) == []
    assert (await _row(lock_db))["email"] == OLD


async def test_take_back_is_refused_and_the_lock_kept_when_another_user_holds_the_address(lock_db) -> None:
    await _move_email(lock_db, NEW)
    await _user(lock_db, "holder", OLD)
    await set_lock(lock_db.user_id)
    with pytest.raises(TakeBackUnavailableError):
        await clear_lock(lock_db.user_id, take_back=OLD)
    row = await _row(lock_db)
    assert row["email"] == NEW and row["security_locked_at"] is not None


async def test_take_back_of_an_unlocked_account_leaves_nothing_made_while_it_ran(lock_db, monkeypatch) -> None:
    """The intruder unlocked first; a key made while the take-back runs is caught by its second sweep,
    and a sign-in attempted meanwhile is refused (the take-back locked the account again first).
    """
    uid = lock_db.user_id
    await _move_email(lock_db, NEW)
    await set_lock(uid)
    await clear_lock(uid)
    sweep = AuthnzApiKeysRepo.suspend_keys_created_since
    made: dict[str, int] = {}

    async def _intruder_makes_a_key_after_the_first_sweep(self, **kwargs):
        suspended = await sweep(self, **kwargs)
        if not made:
            made["key"] = await _key(lock_db, "made-while-taking-back")
            with pytest.raises(AccountSecurityLockedError):
                await (await get_session_manager()).create_session(user_id=uid, access_token="a", refresh_token="r")
        return suspended

    monkeypatch.setattr(AuthnzApiKeysRepo, "suspend_keys_created_since", _intruder_makes_a_key_after_the_first_sweep)
    await clear_lock(uid, take_back=OLD)
    row = await _row(lock_db)
    assert "key" in made and await _unsuspended_active_keys(lock_db) == []
    assert row["security_locked_at"] is None and row["email"] == OLD


async def test_plain_unlock_racing_a_take_back_of_a_locked_account_does_not_win(lock_db, monkeypatch) -> None:
    """The other link read the lock before the take-back wrote its fresh one, and tries to clear it meanwhile."""
    uid = lock_db.user_id
    await _move_email(lock_db, NEW)
    await _session(lock_db, "intruder")
    assert await _live_session_count(lock_db) == 1
    await set_lock(uid)
    sweep = AuthnzApiKeysRepo.suspend_keys_created_since
    plain_has_read, take_back_has_locked = asyncio.Event(), asyncio.Event()
    calls = 0

    async def _interleave(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:  # the plain unlock's first sweep: it has read the original lock
            plain_has_read.set()
            await take_back_has_locked.wait()
        elif calls == 2:  # the take-back's first sweep: its fresh lock is written; let the plain unlock finish
            take_back_has_locked.set()
            await plain
        return await sweep(self, **kwargs)

    monkeypatch.setattr(AuthnzApiKeysRepo, "suspend_keys_created_since", _interleave)
    plain = asyncio.create_task(clear_lock(uid))
    await plain_has_read.wait()
    await clear_lock(uid, take_back=OLD)
    plain_unlock_result = await plain
    row = await _row(lock_db)
    assert plain_unlock_result is None
    assert row["email"] == OLD and row["security_locked_at"] is None and await _live_session_count(lock_db) == 0


async def test_take_back_removes_sso_links_made_since_the_week_before_the_email_change_even_if_the_email_moved_again(
    lock_db,
) -> None:
    uid = lock_db.user_id
    await _move_email(lock_db, NEW)
    await lock_db.age("users", "email_changed_at", uid, timedelta(days=2))
    link = await _link_identity(lock_db, "intruder")
    await lock_db.age("federated_identities", "created_at", link, timedelta(days=1))
    await _move_email(lock_db, "newer@example.com")  # inside 7 days, so previous_email stays OLD
    await lock_db.age("users", "email_changed_at", uid, timedelta(hours=1))
    await clear_lock(uid, take_back=OLD)
    assert await _identity_links(lock_db) == []


async def test_email_change_to_an_address_another_account_just_released_is_refused(lock_db) -> None:
    await _move_email(lock_db, NEW)  # the owner releases OLD
    await lock_db.age("users", "email_changed_at", lock_db.user_id, timedelta(days=1))
    other_user_id = await _user(lock_db, "other", "other@example.com")
    async with lock_db.pool.transaction() as conn:
        with pytest.raises(EmailChangeRefused) as refused:
            await authorize_own_email_change(
                db_conn=conn, is_postgres=lock_db.backend == "postgres", user_id=other_user_id,
                new_emails=(OLD,), current_password=OWNER_PASSWORD,
            )
    assert refused.value.code == "email_recently_released"


async def test_previous_address_lookup_refuses_an_ambiguous_match(lock_db) -> None:
    repo = AuthnzUsersRepo(lock_db.pool)
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    await _move_email(lock_db, NEW)
    second = await _user(lock_db, "second", OLD)
    await _move_email(lock_db, "second-new@example.com", user_id=second)
    await set_lock(lock_db.user_id)
    assert (await repo.find_locked_user_by_previous_email(OLD, since=week_ago))["id"] == lock_db.user_id
    await set_lock(second)
    assert await repo.find_locked_user_by_previous_email(OLD, since=week_ago) is None


async def _password_verifies(db, password: str = OWNER_PASSWORD) -> bool:
    return PasswordService().verify_password(password, (await _row(db))["password_hash"])[0]


async def test_plain_unlock_that_wins_during_a_take_back_leaves_the_intruder_nothing(lock_db, monkeypatch) -> None:
    """The other link runs to completion inside the take-back's first sweep and clears its fresh lock; the
    intruder then makes a key, an SSO link and a session just before the take-back's compare-and-set."""
    uid = lock_db.user_id
    await _move_email(lock_db, NEW)
    await set_lock(uid)
    sweep = AuthnzApiKeysRepo.suspend_keys_created_since
    compare_and_set = AuthnzUsersRepo.clear_security_lock
    plain: dict[str, object] = {}
    cas_calls = 0

    async def _plain_unlock_runs_in_the_first_sweep(self, **kwargs):
        if "result" not in plain:
            plain["result"] = None
            plain["result"] = await clear_lock(uid)
        return await sweep(self, **kwargs)

    async def _intruder_acts_before_the_take_backs_compare_and_set(self, user_id, **kwargs):
        nonlocal cas_calls
        cas_calls += 1
        if cas_calls == 2:  # 1 is the plain unlock's, 2 the take-back's first
            await _key(lock_db, "made-in-the-window")
            await _link_identity(lock_db, "intruder")
            await _session(lock_db, "intruder")
        return await compare_and_set(self, user_id, **kwargs)

    monkeypatch.setattr(AuthnzApiKeysRepo, "suspend_keys_created_since", _plain_unlock_runs_in_the_first_sweep)
    monkeypatch.setattr(AuthnzUsersRepo, "clear_security_lock", _intruder_acts_before_the_take_backs_compare_and_set)
    await clear_lock(uid, take_back=OLD)
    row = await _row(lock_db)
    assert plain["result"] is not None  # the other link did win its own compare-and-set
    assert await _unsuspended_active_keys(lock_db) == []
    assert await _identity_links(lock_db) == []
    assert await _live_session_count(lock_db) == 0
    assert row["email"] == OLD and row["security_locked_at"] is None
    assert await _password_verifies(lock_db) is False


async def test_take_back_that_never_wins_its_compare_and_set_gives_up_with_the_account_locked(
    lock_db, monkeypatch
) -> None:
    uid = lock_db.user_id
    await _move_email(lock_db, NEW)
    await set_lock(uid)
    compare_and_set = AuthnzUsersRepo.clear_security_lock
    attempts = 0

    async def _another_link_unlocks_first(self, user_id, **kwargs):
        """A plain unlock wins first every time: the account is left unlocked, so only the take-back's own
        final lock can leave it locked."""
        nonlocal attempts
        attempts += 1
        await self.db_pool.execute("UPDATE users SET security_locked_at = NULL WHERE id = ?", user_id)
        return await compare_and_set(self, user_id, **kwargs)

    monkeypatch.setattr(AuthnzUsersRepo, "clear_security_lock", _another_link_unlocks_first)
    result = await clear_lock(uid, take_back=OLD)
    assert result is None and attempts == 3
    assert (await _row(lock_db))["security_locked_at"] is not None


async def test_take_back_that_restores_nothing_is_refused_and_keeps_the_lock(lock_db) -> None:
    await _move_email(lock_db, NEW)
    await set_lock(lock_db.user_id)
    with pytest.raises(TakeBackUnavailableError):
        await clear_lock(lock_db.user_id, take_back="stranger@example.com")
    row = await _row(lock_db)
    assert row["email"] == NEW and row["security_locked_at"] is not None


async def test_take_back_is_refused_and_the_lock_kept_when_the_address_is_taken_while_it_runs(
    lock_db, monkeypatch
) -> None:
    await _move_email(lock_db, NEW)
    await set_lock(lock_db.user_id)
    sweep = AuthnzApiKeysRepo.suspend_keys_created_since
    taken: dict[str, int] = {}

    async def _someone_registers_the_address_during_the_first_sweep(self, **kwargs):
        if not taken:
            taken["by"] = await _user(lock_db, "late-holder", OLD)
        return await sweep(self, **kwargs)

    monkeypatch.setattr(AuthnzApiKeysRepo, "suspend_keys_created_since", _someone_registers_the_address_during_the_first_sweep)
    with pytest.raises(TakeBackUnavailableError):
        await clear_lock(lock_db.user_id, take_back=OLD)
    row = await _row(lock_db)
    assert "by" in taken and row["email"] == NEW and row["security_locked_at"] is not None


async def test_a_second_take_back_to_the_restored_address_still_replaces_the_password(lock_db) -> None:
    uid = lock_db.user_id
    await _move_email(lock_db, NEW)
    await set_lock(uid)
    await clear_lock(uid, take_back=OLD)
    async with lock_db.pool.transaction() as conn:  # a password set again after the first take-back
        await PasswordService().set_password(
            uid, hash_password(OWNER_PASSWORD), conn, is_postgres=lock_db.backend == "postgres"
        )
    assert await _password_verifies(lock_db) is True
    assert await clear_lock(uid, take_back=OLD) is not None  # the second link mailed to the same address
    row = await _row(lock_db)
    assert row["email"] == OLD and row["security_locked_at"] is None
    assert await _password_verifies(lock_db) is False


async def _email_change_refusal(db, user_id: int, address: str) -> str | None:
    """The refusal code of a self-service change to ``address`` with the right password; None when allowed."""
    async with db.pool.transaction() as conn:
        try:
            await authorize_own_email_change(
                db_conn=conn, is_postgres=db.backend == "postgres", user_id=user_id,
                new_emails=(address,), current_password=OWNER_PASSWORD,
            )
        except EmailChangeRefused as refused:
            return refused.code
    return None


async def test_email_change_to_an_address_released_eight_days_ago_is_allowed(lock_db) -> None:
    await _move_email(lock_db, NEW)
    await lock_db.age("users", "email_changed_at", lock_db.user_id, timedelta(days=8))
    other_user_id = await _user(lock_db, "other", "other@example.com")
    assert await _email_change_refusal(lock_db, other_user_id, OLD) is None


async def test_email_change_back_to_your_own_previous_address_is_allowed(lock_db) -> None:
    await _move_email(lock_db, NEW)
    assert await _email_change_refusal(lock_db, lock_db.user_id, OLD) is None


async def test_recently_released_address_is_refused_in_any_letter_case(lock_db) -> None:
    await _move_email(lock_db, NEW)
    other_user_id = await _user(lock_db, "other", "other@example.com")
    assert await _email_change_refusal(lock_db, other_user_id, "Old@Example.COM") == "email_recently_released"


async def test_email_in_use_check_ignores_letter_case(lock_db) -> None:
    repo = AuthnzUsersRepo(lock_db.pool)
    other_user_id = await _user(lock_db, "other", "other@example.com")
    assert await repo.email_in_use_by_another_user("Old@Example.COM", user_id=other_user_id) is True
    assert await repo.email_in_use_by_another_user("Old@Example.COM", user_id=lock_db.user_id) is False


# --- the audit write (SQLite only: tests/AuthNZ/conftest.py stubs audit for PostgreSQL tests) ------


async def test_audit_service_that_cannot_start_does_not_fail_the_caller(owner_db, monkeypatch) -> None:
    async def _cannot_start(user_id: int):
        raise ServiceInitializationError("audit service unavailable")

    monkeypatch.setattr(api_key_audit, "_get_or_create_audit_service", _cannot_start)
    assert await write_account_audit(AuditEventType.AUTH_ACCOUNT_LOCKED, user_id=owner_db.user_id) is None


@pytest.mark.real_audit
async def test_audit_event_is_written_and_reads_back(owner_db, real_audit_service) -> None:
    await write_account_audit(
        AuditEventType.AUTH_ACCOUNT_LOCKED, user_id=owner_db.user_id, session_id=41, metadata={"via": "test"}
    )
    service = await api_key_audit._get_or_create_audit_service(owner_db.user_id)
    events = await service.query_events(user_id=str(owner_db.user_id), event_types=[AuditEventType.AUTH_ACCOUNT_LOCKED])
    assert [(event["event_type"], event["context_session_id"]) for event in events] == [("auth.account.locked", "41")]

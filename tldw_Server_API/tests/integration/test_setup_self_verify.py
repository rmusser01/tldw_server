"""Exercise setup verification through the real versioned user writer."""

from datetime import datetime, timezone
from typing import Any

import pytest

import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import RollbackSignal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


async def _verify_in_caller_transaction(
    pool: DatabasePool, monkeypatch: pytest.MonkeyPatch, rollback: bool,
) -> None:
    """The setup caller owns commit/rollback of verification and its version."""
    monkeypatch.setattr(
        setup_endpoint.setup_manager, "get_status_snapshot", lambda: {"needs_setup": True},
    )
    users = UsersDB(pool)
    await users.initialize(ensure_schema=False)
    target = await users.create_user(  # nosec B106 - inert fixture hash, never used for login
        username="setup-user", email="setup@example.test", password_hash="test-hash", is_verified=False,
    )
    other = await users.create_user(  # nosec B106 - inert fixture hash, never used for login
        username="other-user", email="other@example.test", password_hash="test-hash", is_verified=False,
    )
    user_id = int(target["id"])
    before = dict(await pool.fetchrow("SELECT * FROM users WHERE id = ?", user_id))
    other_before = dict(await pool.fetchrow("SELECT * FROM users WHERE id = ?", int(other["id"])))
    started = datetime.now(timezone.utc)

    class CallerRollback(RollbackSignal):
        """Roll back after the endpoint returns successfully."""

    try:
        async with pool.transaction() as connection:
            result = await setup_endpoint.setup_self_verify(
                principal=AuthPrincipal(kind="user", user_id=user_id, username="setup-user"),
                db=connection, _guard=None,
            )
            assert result["success"] is True
            assert result["user_id"] == user_id
            # Read through a separate checkout: setup must not commit its caller's work.
            assert dict(await pool.fetchrow("SELECT * FROM users WHERE id = ?", user_id)) == before
            if rollback:
                raise CallerRollback
    except CallerRollback:
        pass
    finished = datetime.now(timezone.utc)
    after = dict(await pool.fetchrow("SELECT * FROM users WHERE id = ?", user_id))
    assert dict(await pool.fetchrow("SELECT * FROM users WHERE id = ?", int(other["id"]))) == other_before
    if rollback:
        assert after == before
    else:
        assert bool(after["is_verified"]) is True
        assert after["profile_version"] != before["profile_version"]
        updated_at = after["updated_at"]
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        # SQLite's canonical update_users_timestamp trigger stores whole seconds.
        lower_bound = started.replace(microsecond=0) if pool.pool is None else started
        assert lower_bound <= updated_at <= finished


@pytest.mark.parametrize("rollback", [False, True], ids=["commit", "rollback"])
@pytest.mark.asyncio
async def test_setup_self_verify_updates_sqlite(tmp_path, monkeypatch: pytest.MonkeyPatch, rollback: bool) -> None:
    """Managed SQLite persists verification only after its caller commits."""
    pool = DatabasePool(Settings(AUTH_MODE="single_user", DATABASE_URL=f"sqlite:///{tmp_path / 'users.db'}"))
    await pool.initialize()
    try:
        await _verify_in_caller_transaction(pool, monkeypatch, rollback)
    finally:
        await pool.close()


@pytest.mark.postgres
@pytest.mark.parametrize("rollback", [False, True], ids=["commit", "rollback"])
@pytest.mark.asyncio
async def test_setup_self_verify_updates_asyncpg(
    isolated_test_environment: Any, monkeypatch: pytest.MonkeyPatch, rollback: bool,
) -> None:
    """The official PostgreSQL fixture exercises actual timestamp and version writes."""
    pool = await get_db_pool()
    assert pool.pool is not None
    await _verify_in_caller_transaction(pool, monkeypatch, rollback)

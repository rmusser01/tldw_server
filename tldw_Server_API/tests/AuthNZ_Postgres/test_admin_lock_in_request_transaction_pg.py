"""The admin identity.is_locked write joins the request's PostgreSQL transaction."""

from __future__ import annotations

import uuid

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter
from tldw_Server_API.app.core.UserProfiles import update_service as update_service_module
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand

pytestmark = pytest.mark.postgres


async def _seed_user(dsn: str) -> tuple[int, str]:
    username = f"lock-{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(dsn)
    try:
        user_id = await conn.fetchval(
            "INSERT INTO users (uuid, username, email, password_hash, is_active) "
            "VALUES ($1, $2, $3, 'unused-test-hash', TRUE) RETURNING id",
            uuid.uuid4(),
            username,
            f"{username}@example.test",
        )
    finally:
        await conn.close()
    return int(user_id), username


async def _apply_lock(pool, user_id: int, conn, *, locked: bool) -> None:
    await ProfileCommandService(db_pool=pool).apply(
        ProfileUpdateCommand(
            actor_user_id=user_id,
            target_user_id=user_id,
            updates=(("identity.is_locked", locked),),
            roles=frozenset({"admin"}),
            dry_run=False,
        ),
        db_conn=conn,
        scope=None,
    )


@pytest.mark.asyncio
async def test_lock_commits_with_the_request_and_rolls_back_with_it(
    test_db_pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    limiter = RateLimiter(db_pool=test_db_pool)
    await limiter.initialize()
    monkeypatch.setattr(update_service_module, "get_rate_limiter", lambda: limiter)
    user_id, username = await _seed_user(test_db_pool.settings.DATABASE_URL)

    with pytest.raises(TransactionError):
        async with test_db_pool.transaction() as conn:
            await _apply_lock(test_db_pool, user_id, conn, locked=True)
            raise RuntimeError("request failed after the lock write")
    assert (await limiter.check_lockout(username, attempt_type="login"))[0] is False

    async with test_db_pool.transaction() as conn:
        await _apply_lock(test_db_pool, user_id, conn, locked=True)
    assert (await limiter.check_lockout(username, attempt_type="login"))[0] is True

    async with test_db_pool.transaction() as conn:
        await _apply_lock(test_db_pool, user_id, conn, locked=False)
    assert (await limiter.check_lockout(username, attempt_type="login"))[0] is False

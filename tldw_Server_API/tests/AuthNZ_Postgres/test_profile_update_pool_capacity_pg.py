"""Profile updates must fit within their existing PostgreSQL transaction slot."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import AsyncExitStack

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ.password_service import hash_password
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand

pytestmark = pytest.mark.postgres


async def _seed_user(test_db_pool, username: str, password_hash: str) -> int:
    seed = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        return await seed.fetchval(
            """
            INSERT INTO users (uuid, username, email, password_hash, is_active)
            VALUES ($1, $2, $3, $4, TRUE)
            RETURNING id
            """,
            uuid.uuid4(),
            username,
            f"{username}@example.com",
            password_hash,
        )
    finally:
        await seed.close()


@pytest.mark.asyncio
async def test_profile_preference_update_completes_when_other_pool_slots_are_busy(
    test_db_pool,
) -> None:
    """Concurrent requests holding every other slot cannot block this write."""
    user_id = await _seed_user(test_db_pool, "profile-capacity", "unused-test-hash")
    key = "preferences.chat.default_character_id"
    async with AsyncExitStack() as slots:
        connection = await slots.enter_async_context(test_db_pool.acquire())
        for _ in range(test_db_pool.pool.get_max_size() - 1):
            await slots.enter_async_context(test_db_pool.acquire())
        async with connection.transaction():
            result = await asyncio.wait_for(
                ProfileCommandService(db_pool=test_db_pool).apply(
                    ProfileUpdateCommand(
                        actor_user_id=user_id,
                        target_user_id=user_id,
                        updates=((key, "capacity-character"),),
                        roles=frozenset({"user"}),
                        dry_run=False,
                    ),
                    db_conn=connection,
                    scope=None,
                ),
                timeout=5,
            )
            assert result.applied == (key,)
            saved = await connection.fetchval(
                "SELECT value_json FROM public.user_config_overrides WHERE user_id = $1 AND key = $2",
                user_id,
                key,
            )
            assert saved == '"capacity-character"'


@pytest.mark.asyncio
async def test_own_email_change_completes_when_other_pool_slots_are_busy(test_db_pool) -> None:
    """The password check reads on the request's own connection, never a second pool slot."""
    user_id = await _seed_user(test_db_pool, "email-capacity", hash_password("Capacity-Pass-2026!"))
    async with AsyncExitStack() as slots:
        connection = await slots.enter_async_context(test_db_pool.acquire())
        for _ in range(test_db_pool.pool.get_max_size() - 1):
            await slots.enter_async_context(test_db_pool.acquire())
        async with connection.transaction():
            result = await asyncio.wait_for(
                ProfileCommandService(db_pool=test_db_pool).apply(
                    ProfileUpdateCommand(
                        actor_user_id=user_id,
                        target_user_id=user_id,
                        updates=(("identity.email", "email-capacity-new@example.com"),),
                        roles=frozenset({"user"}),
                        dry_run=False,
                        current_password="Capacity-Pass-2026!",
                    ),
                    db_conn=connection,
                    scope=None,
                ),
                timeout=5,
            )
            assert result.applied == ("identity.email",)
            row = await connection.fetchrow("SELECT email, previous_email FROM users WHERE id = $1", user_id)
            assert (row["email"], row["previous_email"]) == (
                "email-capacity-new@example.com",
                "email-capacity@example.com",
            )

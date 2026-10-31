"""Profile updates must fit within their existing PostgreSQL transaction slot."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import AsyncExitStack

import asyncpg
import pytest

from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import ProfileUpdateCommand

pytestmark = pytest.mark.postgres


@pytest.mark.asyncio
async def test_profile_preference_update_completes_when_other_pool_slots_are_busy(
    test_db_pool,
) -> None:
    """Concurrent requests holding every other slot cannot block this write."""
    seed = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        user_id = await seed.fetchval(
            """
            INSERT INTO users (uuid, username, email, password_hash, is_active)
            VALUES ($1, $2, $3, $4, TRUE)
            RETURNING id
            """,
            uuid.uuid4(),
            "profile-capacity",
            "profile-capacity@example.test",
            "unused-test-hash",
        )
    finally:
        await seed.close()
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

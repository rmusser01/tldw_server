from __future__ import annotations

import uuid

import pytest

from tldw_Server_API.app.core.UserProfiles.service import UserProfileService


pytestmark = pytest.mark.postgres


@pytest.mark.asyncio
async def test_profile_version_lock_uses_transaction_connection(test_db_pool):
    user_id = await test_db_pool.fetchval(
        """
        INSERT INTO users (uuid, username, email, password_hash, is_active)
        VALUES ($1, $2, $3, $4, TRUE)
        RETURNING id
        """,
        str(uuid.uuid4()),
        "pg-profile-version-lock",
        "pg-profile-version-lock@example.com",
        "hash",
    )
    service = UserProfileService(test_db_pool)

    async with test_db_pool.transaction() as conn:
        version = await service.get_profile_version(
            user_id=int(user_id),
            db_conn=conn,
            lock_user=True,
        )

    assert version is not None

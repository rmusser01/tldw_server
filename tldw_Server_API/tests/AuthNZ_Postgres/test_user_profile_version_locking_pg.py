from __future__ import annotations

import uuid

import asyncpg
import pytest

from tldw_Server_API.app.core.UserProfiles.version_gateway import ProfileVersionGateway

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
    gateway = ProfileVersionGateway(test_db_pool)
    backend_pool = test_db_pool.pool
    assert backend_pool is not None
    lock_failure_observed = False

    async with backend_pool.acquire(timeout=5.0) as lock_conn:
        async with lock_conn.transaction():
            version = await gateway.read_in_transaction(
                lock_conn,
                int(user_id),
                lock_user=True,
            )

            async with backend_pool.acquire(timeout=5.0) as competing_conn:
                with pytest.raises(asyncpg.exceptions.LockNotAvailableError) as raised:
                    async with competing_conn.transaction():
                        await competing_conn.execute(
                            "SET LOCAL lock_timeout = '500ms'"
                        )
                        await competing_conn.execute(
                            "UPDATE users SET profile_version = profile_version "
                            "WHERE id = $1",
                            int(user_id),
                        )
                lock_failure_observed = True
                assert raised.value.sqlstate == "55P03"

    assert version is not None
    assert lock_failure_observed is True

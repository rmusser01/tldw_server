from __future__ import annotations

import asyncpg
import pytest

from tldw_Server_API.app.core.UserProfiles.version_gateway import ProfileVersionGateway
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

pytestmark = pytest.mark.postgres


@pytest.mark.asyncio
async def test_profile_version_lock_uses_transaction_connection(test_db_pool):
    seed_hash = "hash"
    user_id = await ensure_test_user(
        test_db_pool,
        "pg-profile-version-lock",
        "pg-profile-version-lock@example.com",
        password_hash=seed_hash,
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
                        await competing_conn.fetchrow(
                            "SELECT id FROM users WHERE id = $1 FOR UPDATE",
                            int(user_id),
                        )
                lock_failure_observed = True
                assert raised.value.sqlstate == "55P03"

    assert version is not None
    assert lock_failure_observed is True

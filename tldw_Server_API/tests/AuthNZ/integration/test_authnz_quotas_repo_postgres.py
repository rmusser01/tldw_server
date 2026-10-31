from __future__ import annotations

import pytest

from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_authnz_quotas_repo_postgres_increment_and_check(test_db_pool):
    """AuthnzQuotasRepo increment helpers should work on Postgres."""
    from tldw_Server_API.app.core.AuthNZ.repos.quotas_repo import AuthnzQuotasRepo

    pool = test_db_pool

    # Seed a user and API key to satisfy FKs on vk_api_key_counters.
    user_id = await ensure_test_user(
        pool, "pg-quotas-user", "pg-quotas-user@example.com", is_verified=True
    )
    async with pool.acquire() as conn:
        api_key_id = await conn.fetchval(
            """
            INSERT INTO api_keys (user_id, key_hash, key_prefix, scope, status)
            VALUES ($1, $2, $3, $4, 'active')
            RETURNING id
            """,
            int(user_id),
            "pg-quotas-key-hash",
            "pg-quotas-prefix",
            "read",
        )
    repo = AuthnzQuotasRepo(db_pool=pool)
    # Ensure vk_* counters schema exists via the repo helper.
    await repo.ensure_schema()

    # JWT quota: limit 2, third call should be denied.
    allowed1, count1 = await repo.increment_and_check_jwt_quota(
        jti="jwt-pg-1",
        counter_type="test",
        limit=2,
        bucket=None,
    )
    allowed2, count2 = await repo.increment_and_check_jwt_quota(
        jti="jwt-pg-1",
        counter_type="test",
        limit=2,
        bucket=None,
    )
    allowed3, count3 = await repo.increment_and_check_jwt_quota(
        jti="jwt-pg-1",
        counter_type="test",
        limit=2,
        bucket=None,
    )

    assert allowed1 is True and int(count1) == 1
    assert allowed2 is True and int(count2) == 2
    assert allowed3 is False and int(count3) == 3

    # API key quota: same pattern, using the seeded api_key_id and a bucket label.
    allowed4, count4 = await repo.increment_and_check_api_key_quota(
        api_key_id=int(api_key_id),
        counter_type="audio",
        limit=2,
        bucket="unit",
    )
    allowed5, count5 = await repo.increment_and_check_api_key_quota(
        api_key_id=int(api_key_id),
        counter_type="audio",
        limit=2,
        bucket="unit",
    )
    allowed6, count6 = await repo.increment_and_check_api_key_quota(
        api_key_id=int(api_key_id),
        counter_type="audio",
        limit=2,
        bucket="unit",
    )

    assert allowed4 is True and int(count4) == 1
    assert allowed5 is True and int(count5) == 2
    assert allowed6 is False and int(count6) == 3

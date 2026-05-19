from __future__ import annotations

from datetime import datetime, timezone

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_authnz_usage_repo_insert_llm_and_rollup_postgres(test_db_pool):
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    pool = test_db_pool
    repo = AuthnzUsageRepo(pool)

    # Seed a user and API key to satisfy FKs.
    async with pool.acquire() as conn:
        user_id = await conn.fetchval(
            """
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES (gen_random_uuid(), $1, $2, $3, $4, TRUE, TRUE, 5120, 0.0)
            RETURNING id
            """,
            "pg-usage-rollup-user",
            "pg-usage-rollup-user@example.com",
            "hashed",
            "user",
        )
        key_id = await conn.fetchval(
            """
            INSERT INTO api_keys (user_id, key_hash, key_prefix, scope, status)
            VALUES ($1, $2, $3, $4, 'active')
            RETURNING id
            """,
            int(user_id),
            "pg-usage-rollup-key-hash",
            "pg-rollup-pfx",
            "read",
        )

    await repo.insert_llm_usage_log(
        user_id=int(user_id),
        key_id=int(key_id),
        endpoint="POST:/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-test",
        status=200,
        latency_ms=100,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        prompt_cost_usd=0.1,
        completion_cost_usd=0.2,
        total_cost_usd=0.3,
        currency="USD",
        estimated=False,
        request_id="pg-req-1",
    )
    await repo.insert_llm_usage_log(
        user_id=int(user_id),
        key_id=int(key_id),
        endpoint="POST:/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-test",
        status=500,
        latency_ms=50,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        prompt_cost_usd=0.05,
        completion_cost_usd=0.05,
        total_cost_usd=0.1,
        currency="USD",
        estimated=False,
        request_id="pg-req-2",
    )

    day_val = datetime.now(timezone.utc).date()
    await repo.aggregate_llm_usage_daily_for_day(day=day_val)

    row = await pool.fetchone(
        """
        SELECT requests, errors, input_tokens, output_tokens, total_tokens, total_cost_usd
        FROM llm_usage_daily
        WHERE day = $1 AND user_id = $2 AND operation = $3 AND provider = $4 AND model = $5
        """,
        day_val,
        int(user_id),
        "chat",
        "openai",
        "gpt-test",
    )
    assert row is not None
    row = dict(row)
    assert int(row["requests"]) == 2
    assert int(row["errors"]) == 1
    assert int(row["input_tokens"]) == 15
    assert int(row["output_tokens"]) == 25
    assert int(row["total_tokens"]) == 40
    assert float(row["total_cost_usd"]) == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_authnz_usage_repo_rollup_usage_daily_postgres(test_db_pool):
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    pool = test_db_pool
    repo = AuthnzUsageRepo(pool)

    # Seed a user for FK.
    async with pool.acquire() as conn:
        user_id = await conn.fetchval(
            """
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES (gen_random_uuid(), $1, $2, $3, $4, TRUE, TRUE, 5120, 0.0)
            RETURNING id
            """,
            "pg-usage-daily-user",
            "pg-usage-daily-user@example.com",
            "hashed",
            "user",
        )

    await repo.insert_usage_log(
        user_id=int(user_id),
        key_id=None,
        endpoint="GET:/rag/search",
        status=200,
        latency_ms=10,
        bytes_out=100,
        bytes_in=50,
        meta="{}",
        request_id="pg-u1-1",
    )
    await repo.insert_usage_log(
        user_id=int(user_id),
        key_id=None,
        endpoint="GET:/rag/search",
        status=404,
        latency_ms=20,
        bytes_out=200,
        bytes_in=25,
        meta="{}",
        request_id="pg-u1-2",
    )

    day_val = datetime.now(timezone.utc).date()
    await repo.aggregate_usage_daily_for_day(day=day_val)

    row = await pool.fetchone(
        """
        SELECT requests, errors, bytes_total, bytes_in_total
        FROM usage_daily
        WHERE day = $1 AND user_id = $2
        """,
        day_val,
        int(user_id),
    )
    assert row is not None
    row = dict(row)
    assert int(row["requests"]) == 2
    assert int(row["errors"]) == 1
    assert int(row["bytes_total"]) == 300
    assert int(row.get("bytes_in_total") or 0) == 75


@pytest.mark.asyncio
async def test_authnz_usage_repo_summarize_user_and_key_day_postgres(test_db_pool):
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    pool = test_db_pool
    repo = AuthnzUsageRepo(pool)

    # Seed a user and API key to satisfy FKs.
    async with pool.acquire() as conn:
        user_id = await conn.fetchval(
            """
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES (gen_random_uuid(), $1, $2, $3, $4, TRUE, TRUE, 5120, 0.0)
            RETURNING id
            """,
            "pg-usage-summarize-user",
            "pg-usage-summarize-user@example.com",
            "hashed",
            "user",
        )
        key_id = await conn.fetchval(
            """
            INSERT INTO api_keys (user_id, key_hash, key_prefix, scope, status)
            VALUES ($1, $2, $3, $4, 'active')
            RETURNING id
            """,
            int(user_id),
            "pg-usage-summarize-key-hash",
            "pg-summarize-pfx",
            "read",
        )

    await repo.insert_llm_usage_log(
        user_id=int(user_id),
        key_id=int(key_id),
        endpoint="POST:/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-test",
        status=200,
        latency_ms=10,
        prompt_tokens=5,
        completion_tokens=5,
        total_tokens=10,
        prompt_cost_usd=0.0,
        completion_cost_usd=0.0,
        total_cost_usd=0.0,
        currency="USD",
        estimated=False,
        request_id="pg-sum-u-k-1",
    )
    await repo.insert_llm_usage_log(
        user_id=int(user_id),
        key_id=None,
        endpoint="POST:/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-test",
        status=200,
        latency_ms=10,
        prompt_tokens=15,
        completion_tokens=15,
        total_tokens=30,
        prompt_cost_usd=0.0,
        completion_cost_usd=0.0,
        total_cost_usd=0.0,
        currency="USD",
        estimated=False,
        request_id="pg-sum-u-none-2",
    )

    day_val = datetime.now(timezone.utc).date()
    user_summary = await repo.summarize_user_day(user_id=int(user_id), day=day_val)
    assert int(user_summary.get("tokens") or 0) == 40

    key_summary = await repo.summarize_key_day(key_id=int(key_id), day=day_val)
    assert int(key_summary.get("tokens") or 0) == 10

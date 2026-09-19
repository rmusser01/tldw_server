from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_authnz_usage_repo_prune_postgres(test_db_pool):
    """AuthnzUsageRepo pruning helpers should work on Postgres."""
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    pool = test_db_pool

    now = datetime.utcnow().replace(microsecond=0)
    old_ts = now - timedelta(days=10)
    recent_ts = now - timedelta(days=1)

    # Seed a user to satisfy FKs on usage_daily / llm_usage_daily
    users = UsersDB(pool)
    await users.initialize(ensure_schema=False)
    user = await users.create_user(
        username="pg-usage-prune-user", email="pg-usage-prune-user@example.com",
        password_hash="hashed", role="user", is_active=True, is_verified=True,
    )
    user_id = int(user["id"])

    # Seed minimal usage rows
    await pool.execute(
        """
        INSERT INTO usage_log (ts, user_id, key_id, endpoint, status, latency_ms, bytes, bytes_in, meta, request_id)
        VALUES ($1, NULL, NULL, $2, 200, 50, 1000, 500, $3, $4)
        """,
        old_ts,
        "/old",
        "{}",
        "old-req",
    )
    await pool.execute(
        """
        INSERT INTO usage_log (ts, user_id, key_id, endpoint, status, latency_ms, bytes, bytes_in, meta, request_id)
        VALUES ($1, NULL, NULL, $2, 200, 50, 1000, 500, $3, $4)
        """,
        recent_ts,
        "/recent",
        "{}",
        "recent-req",
    )

    await pool.execute(
        """
        INSERT INTO llm_usage_log (ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms)
        VALUES ($1, NULL, NULL, $2, 'chat', 'openai', 'gpt', 200, 100)
        """,
        old_ts,
        "/llm-old",
    )
    await pool.execute(
        """
        INSERT INTO llm_usage_log (ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms)
        VALUES ($1, NULL, NULL, $2, 'chat', 'openai', 'gpt', 200, 100)
        """,
        recent_ts,
        "/llm-recent",
    )

    # usage_daily / llm_usage_daily
    await pool.execute(
        """
        INSERT INTO usage_daily (user_id, day, requests, errors, bytes_total, bytes_in_total, latency_avg_ms)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        int(user_id),
        (now.date() - timedelta(days=30)),
        10,
        1,
        1000,
        500,
        20.0,
    )
    await pool.execute(
        """
        INSERT INTO usage_daily (user_id, day, requests, errors, bytes_total, bytes_in_total, latency_avg_ms)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        int(user_id),
        now.date(),
        5,
        0,
        500,
        250,
        15.0,
    )
    await pool.execute(
        """
        INSERT INTO llm_usage_daily (
            day, user_id, operation, provider, model,
            requests, errors, input_tokens, output_tokens, total_tokens,
            total_cost_usd
        )
        VALUES ($1, $2, 'chat', 'openai', 'gpt', 10, 1, 100, 200, 300, 1.0)
        """,
        (now.date() - timedelta(days=30)),
        int(user_id),
    )
    await pool.execute(
        """
        INSERT INTO llm_usage_daily (
            day, user_id, operation, provider, model,
            requests, errors, input_tokens, output_tokens, total_tokens,
            total_cost_usd
        )
        VALUES ($1, $2, 'chat', 'openai', 'gpt', 5, 0, 50, 100, 150, 0.5)
        """,
        now.date(),
        int(user_id),
    )

    repo = AuthnzUsageRepo(pool)

    cutoff_ts = now - timedelta(days=5)
    cutoff_day = (now.date() - timedelta(days=10))

    deleted_usage = await repo.prune_usage_log_before(cutoff_ts)
    deleted_llm = await repo.prune_llm_usage_log_before(cutoff_ts)
    deleted_daily = await repo.prune_usage_daily_before(cutoff_day)
    deleted_llm_daily = await repo.prune_llm_usage_daily_before(cutoff_day)

    assert deleted_usage == 1
    assert deleted_llm == 1
    assert deleted_daily == 1
    assert deleted_llm_daily == 1

    remaining_usage = await pool.fetchval("SELECT COUNT(*) FROM usage_log")
    remaining_llm = await pool.fetchval("SELECT COUNT(*) FROM llm_usage_log")
    remaining_daily = await pool.fetchval("SELECT COUNT(*) FROM usage_daily")
    remaining_llm_daily = await pool.fetchval("SELECT COUNT(*) FROM llm_usage_daily")

    assert remaining_usage == 1
    assert remaining_llm == 1
    assert remaining_daily == 1
    assert remaining_llm_daily == 1

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_authnz_usage_repo_prune_sqlite(tmp_path, monkeypatch, fresh_services):
    """AuthnzUsageRepo pruning helpers should work on SQLite."""
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    now = datetime.now(timezone.utc)
    old_ts = now - timedelta(days=10)
    recent_ts = now - timedelta(days=1)

    async with pool.transaction() as conn:
        # usage_log
        await conn.execute(
            "INSERT INTO usage_log (ts, endpoint, status, latency_ms, bytes, bytes_in, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (old_ts.isoformat(), "/old", 200, 50, 1000, 500, "{}"),
        )
        await conn.execute(
            "INSERT INTO usage_log (ts, endpoint, status, latency_ms, bytes, bytes_in, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (recent_ts.isoformat(), "/recent", 200, 50, 1000, 500, "{}"),
        )
        # llm_usage_log
        await conn.execute(
            "INSERT INTO llm_usage_log (ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms, prompt_tokens, completion_tokens, total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated, request_id) "
            "VALUES (?, NULL, NULL, ?, 'chat', 'openai', 'gpt', 200, 100, 10, 20, 30, 0.1, 0.2, 0.3, 'USD', 0, 'old-req')",
            (old_ts.isoformat(), "/llm-old"),
        )
        await conn.execute(
            "INSERT INTO llm_usage_log (ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms, prompt_tokens, completion_tokens, total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated, request_id) "
            "VALUES (?, NULL, NULL, ?, 'chat', 'openai', 'gpt', 200, 100, 10, 20, 30, 0.1, 0.2, 0.3, 'USD', 0, 'recent-req')",
            (recent_ts.isoformat(), "/llm-recent"),
        )
        # usage_daily / llm_usage_daily
        await conn.execute(
            "INSERT INTO usage_daily (user_id, day, requests, errors, bytes_total, bytes_in_total, latency_avg_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, (now.date() - timedelta(days=30)).isoformat(), 10, 1, 1000, 500, 20.0),
        )
        await conn.execute(
            "INSERT INTO usage_daily (user_id, day, requests, errors, bytes_total, bytes_in_total, latency_avg_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, now.date().isoformat(), 5, 0, 500, 250, 15.0),
        )
        await conn.execute(
            "INSERT INTO llm_usage_daily (day, user_id, operation, provider, model, requests, errors, input_tokens, output_tokens, total_tokens, total_cost_usd) "
            "VALUES (?, ?, 'chat', 'openai', 'gpt', 10, 1, 100, 200, 300, 1.0)",
            ((now.date() - timedelta(days=30)).isoformat(), 1),
        )
        await conn.execute(
            "INSERT INTO llm_usage_daily (day, user_id, operation, provider, model, requests, errors, input_tokens, output_tokens, total_tokens, total_cost_usd) "
            "VALUES (?, ?, 'chat', 'openai', 'gpt', 5, 0, 50, 100, 150, 0.5)",
            (now.date().isoformat(), 1),
        )

    repo = AuthnzUsageRepo(pool)

    # Cutoff chosen to remove "old" but keep "recent"
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

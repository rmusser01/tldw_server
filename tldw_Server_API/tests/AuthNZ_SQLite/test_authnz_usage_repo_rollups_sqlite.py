from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_authnz_usage_repo_insert_llm_and_rollup_sqlite(tmp_path, monkeypatch, fresh_services):
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

    repo = AuthnzUsageRepo(pool)

    await repo.insert_llm_usage_log(
        user_id=1,
        key_id=None,
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
        request_id="req-1",
    )
    await repo.insert_llm_usage_log(
        user_id=1,
        key_id=None,
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
        request_id="req-2",
    )

    day_val = datetime.now(timezone.utc).date()
    await repo.aggregate_llm_usage_daily_for_day(day=day_val)

    row = await pool.fetchone(
        "SELECT requests, errors, input_tokens, output_tokens, total_tokens, total_cost_usd "
        "FROM llm_usage_daily WHERE day = ? AND user_id = ? AND operation = ? AND provider = ? AND model = ?",
        day_val.isoformat(),
        1,
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
async def test_authnz_usage_repo_rollup_usage_daily_sqlite(tmp_path, monkeypatch, fresh_services):
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

    repo = AuthnzUsageRepo(pool)

    await repo.insert_usage_log(
        user_id=1,
        key_id=None,
        endpoint="GET:/rag/search",
        status=200,
        latency_ms=10,
        bytes_out=100,
        bytes_in=50,
        meta="{}",
        request_id="u1-1",
    )
    await repo.insert_usage_log(
        user_id=1,
        key_id=None,
        endpoint="GET:/rag/search",
        status=404,
        latency_ms=20,
        bytes_out=200,
        bytes_in=25,
        meta="{}",
        request_id="u1-2",
    )

    day_val = datetime.now(timezone.utc).date()
    await repo.aggregate_usage_daily_for_day(day=day_val)

    row = await pool.fetchone(
        "SELECT requests, errors, bytes_total, bytes_in_total FROM usage_daily WHERE day = ? AND user_id = ?",
        day_val.isoformat(),
        1,
    )
    assert row is not None
    row = dict(row)
    assert int(row["requests"]) == 2
    assert int(row["errors"]) == 1
    assert int(row["bytes_total"]) == 300
    # bytes_in_total exists in current schema; tolerate missing legacy column.
    assert int(row.get("bytes_in_total") or 0) == 75


@pytest.mark.asyncio
async def test_authnz_usage_repo_summarize_user_and_key_day_sqlite(tmp_path, monkeypatch, fresh_services):
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

    repo = AuthnzUsageRepo(pool)

    await repo.insert_llm_usage_log(
        user_id=1,
        key_id=9,
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
        request_id="sum-u1-k9-1",
    )
    await repo.insert_llm_usage_log(
        user_id=1,
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
        request_id="sum-u1-none-2",
    )

    day_val = datetime.now(timezone.utc).date()
    user_summary = await repo.summarize_user_day(user_id=1, day=day_val)
    assert int(user_summary.get("tokens") or 0) == 40

    key_summary = await repo.summarize_key_day(key_id=9, day=day_val)
    assert int(key_summary.get("tokens") or 0) == 10

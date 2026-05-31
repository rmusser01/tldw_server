from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from types import SimpleNamespace
import uuid
import pytest

from tldw_Server_API.app.services.llm_usage_aggregator import (
    _aggregator_loop,
    aggregate_llm_usage_daily,
)


async def _ensure_llm_tables(pool):
    if pool.pool:
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_log (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                key_id INTEGER,
                endpoint TEXT,
                operation TEXT,
                provider TEXT,
                model TEXT,
                status INTEGER,
                latency_ms INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                total_cost_usd DOUBLE PRECISION,
                currency TEXT,
                estimated BOOLEAN,
                request_id TEXT
            )
            """
        )
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_daily (
                day DATE NOT NULL,
                user_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                requests INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                input_tokens BIGINT DEFAULT 0,
                output_tokens BIGINT DEFAULT 0,
                total_tokens BIGINT DEFAULT 0,
                total_cost_usd DOUBLE PRECISION DEFAULT 0.0,
                latency_avg_ms DOUBLE PRECISION,
                PRIMARY KEY (day, user_id, operation, provider, model)
            )
            """
        )
    else:
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                key_id INTEGER,
                endpoint TEXT,
                operation TEXT,
                provider TEXT,
                model TEXT,
                status INTEGER,
                latency_ms INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                total_cost_usd REAL,
                currency TEXT,
                estimated INTEGER,
                request_id TEXT
            )
            """
        )
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_daily (
                day DATE NOT NULL,
                user_id INTEGER NOT NULL,
                operation TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                requests INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                total_cost_usd REAL DEFAULT 0.0,
                latency_avg_ms REAL,
                PRIMARY KEY (day, user_id, operation, provider, model)
            )
            """
        )


async def _insert_llm_log(pool, *, user_id, operation, provider, model, status, latency_ms, pt, ct, cost):
    if pool.pool:
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, endpoint, operation, provider, model, status, latency_ms, prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)",
            user_id, "/x", operation, provider, model, status, latency_ms, pt, ct, pt+ct, cost, "USD", False
        )
    else:
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, endpoint, operation, provider, model, status, latency_ms, prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            user_id, "/x", operation, provider, model, status, latency_ms, pt, ct, pt+ct, cost, "USD", 0
        )


@pytest.mark.asyncio
async def test_start_llm_usage_aggregator_registers_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.services.llm_usage_aggregator as llm_usage_aggregator

    class _Settings:
        LLM_USAGE_AGGREGATOR_ENABLED = True

    worker_inventory = object()
    task = SimpleNamespace()
    stop_event = object()
    calls: list[dict[str, object]] = []

    async def _fake_start_stop_event_worker(
        inventory: object,
        **kwargs: object,
    ) -> tuple[SimpleNamespace, object]:
        calls.append({"inventory": inventory, **kwargs})
        return task, stop_event

    monkeypatch.setattr(llm_usage_aggregator, "get_settings", lambda: _Settings())
    monkeypatch.setattr(
        llm_usage_aggregator,
        "start_stop_event_worker",
        _fake_start_stop_event_worker,
    )

    returned_task = await llm_usage_aggregator.start_llm_usage_aggregator(
        worker_inventory=worker_inventory,
    )

    assert returned_task is task
    assert getattr(returned_task, "_tldw_stop_event") is stop_event
    assert len(calls) == 1
    assert calls[0]["inventory"] is worker_inventory
    assert calls[0]["name"] == "llm_usage_aggregator"
    assert calls[0]["task_name"] == "llm_usage_aggregator"
    assert calls[0]["category"] == "usage"
    assert calls[0]["shutdown_phase"] == llm_usage_aggregator.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert callable(calls[0]["coroutine_factory"])


@pytest.mark.asyncio
async def test_llm_aggregate_sqlite(monkeypatch):
    # Force SQLite single-user DB
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "agg-llm-" + uuid.uuid4().hex)
    dburl = f"sqlite:///./Databases/users_test_llmagg_{uuid.uuid4().hex}.sqlite"
    monkeypatch.setenv("DATABASE_URL", dburl)

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()

    pool = await get_db_pool()
    await _ensure_llm_tables(pool)

    await _insert_llm_log(pool, user_id=1, operation="chat", provider="openai", model="gpt-3.5-turbo", status=200, latency_ms=100, pt=100, ct=50, cost=0.05)
    await _insert_llm_log(pool, user_id=1, operation="chat", provider="openai", model="gpt-3.5-turbo", status=500, latency_ms=300, pt=20, ct=0, cost=0.01)
    await _insert_llm_log(pool, user_id=2, operation="embeddings", provider="openai", model="text-embedding-3-small", status=200, latency_ms=200, pt=200, ct=0, cost=0.02)

    day = datetime.now(timezone.utc).date().isoformat()
    await aggregate_llm_usage_daily(day=day)

    if pool.pool:
        rows = await pool.fetchall("SELECT user_id, operation, provider, model, requests, errors, total_tokens, total_cost_usd FROM llm_usage_daily WHERE day = $1", day)
        rows = [dict(r) for r in rows]
    else:
        rows = await pool.fetchall("SELECT user_id, operation, provider, model, requests, errors, total_tokens, total_cost_usd FROM llm_usage_daily WHERE day = ?", day)
        rows = [{"user_id": r[0], "operation": r[1], "provider": r[2], "model": r[3], "requests": r[4], "errors": r[5], "total_tokens": r[6], "total_cost_usd": r[7]} for r in rows]

    # Check aggregates for user 1 chat
    row1 = next((r for r in rows if r["user_id"] == 1 and r["operation"] == "chat"), None)
    assert row1 is not None
    assert int(row1["requests"]) == 2
    assert int(row1["errors"]) == 1
    assert int(row1["total_tokens"]) == 170
    assert float(row1["total_cost_usd"]) > 0.0


@pytest.mark.asyncio
async def test_aggregate_llm_usage_daily_failure_log_is_sanitized(monkeypatch):
    import tldw_Server_API.app.services.llm_usage_aggregator as llm_usage_aggregator

    class _ExplodingRepo:
        def __init__(self, _pool):
            pass

        async def aggregate_llm_usage_daily_for_day(self, *, day):
            _ = day
            raise RuntimeError("llm usage backend exploded at /tmp/llm-usage-secret-token")

    records: list[str] = []
    sink_id = llm_usage_aggregator.logger.add(
        lambda message: records.append(str(message)),
        level="DEBUG",
        format="{message} {extra}",
    )
    monkeypatch.setattr(llm_usage_aggregator, "AuthnzUsageRepo", _ExplodingRepo)

    try:
        await llm_usage_aggregator.aggregate_llm_usage_daily(db_pool=object(), day="2026-04-29")
    finally:
        llm_usage_aggregator.logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "llm_usage_daily aggregation skipped/failed" in rendered
    assert "RuntimeError" in rendered
    assert "llm usage backend exploded" not in rendered
    assert "/tmp/llm-usage-secret-token" not in rendered


@pytest.mark.asyncio
async def test_aggregator_loop_outer_failure_log_is_sanitized(monkeypatch):
    import tldw_Server_API.app.services.llm_usage_aggregator as llm_usage_aggregator

    class _Settings:
        LLM_USAGE_AGGREGATOR_ENABLED = True
        LLM_USAGE_AGGREGATOR_INTERVAL_MINUTES = 60

    async def _explode():
        raise RuntimeError("loop exit leaked /tmp/llm-loop-secret")

    records: list[str] = []
    sink_id = llm_usage_aggregator.logger.add(
        lambda message: records.append(str(message)),
        level="INFO",
        format="{message} {extra}",
    )
    monkeypatch.setattr(llm_usage_aggregator, "get_settings", lambda: _Settings())
    monkeypatch.setattr(llm_usage_aggregator, "aggregate_llm_usage_daily", _explode)

    try:
        await _aggregator_loop(asyncio.Event())
    finally:
        llm_usage_aggregator.logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "LLM usage aggregator loop exited" in rendered
    assert "RuntimeError" in rendered
    assert "loop exit leaked" not in rendered
    assert "/tmp/llm-loop-secret" not in rendered

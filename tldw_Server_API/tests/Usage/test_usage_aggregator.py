from __future__ import annotations

import os
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest
import asyncio

from tldw_Server_API.app.services.usage_aggregator import aggregate_usage_daily


async def _ensure_usage_tables(pool):
    # Create tables in either SQLite or Postgres depending on pool
    if pool.pool:
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_log (
                id SERIAL PRIMARY KEY,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                key_id INTEGER,
                endpoint TEXT,
                status INTEGER,
                latency_ms INTEGER,
                bytes INTEGER,
                meta TEXT
            )
            """
        )
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_daily (
                user_id INTEGER NOT NULL,
                day DATE NOT NULL,
                requests INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                bytes_total INTEGER DEFAULT 0,
                latency_avg_ms DOUBLE PRECISION,
                PRIMARY KEY (user_id, day)
            )
            """
        )
    else:
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                key_id INTEGER,
                endpoint TEXT,
                status INTEGER,
                latency_ms INTEGER,
                bytes INTEGER,
                meta TEXT
            )
            """
        )
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_daily (
                user_id INTEGER NOT NULL,
                day DATE NOT NULL,
                requests INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                bytes_total INTEGER DEFAULT 0,
                latency_avg_ms REAL,
                PRIMARY KEY (user_id, day)
            )
            """
        )


async def _insert_log(pool, *, user_id, status, latency_ms, bytes_val):
    if pool.pool:
        await pool.execute(
            "INSERT INTO usage_log (user_id, endpoint, status, latency_ms, bytes, meta) VALUES ($1,$2,$3,$4,$5,$6)",
            user_id,
            "/x",
            status,
            latency_ms,
            bytes_val,
            None,
        )
    else:
        await pool.execute(
            "INSERT INTO usage_log (user_id, endpoint, status, latency_ms, bytes, meta) VALUES (?,?,?,?,?,?)",
            user_id,
            "/x",
            status,
            latency_ms,
            bytes_val,
            None,
        )


@pytest.mark.asyncio
async def test_start_usage_aggregator_registers_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.services.usage_aggregator as usage_aggregator

    class _Settings:
        USAGE_LOG_ENABLED = True

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

    monkeypatch.setattr(usage_aggregator, "get_settings", lambda: _Settings())
    monkeypatch.setattr(
        usage_aggregator,
        "start_stop_event_worker",
        _fake_start_stop_event_worker,
    )

    returned_task = await usage_aggregator.start_usage_aggregator(
        worker_inventory=worker_inventory,
    )

    assert returned_task is task
    assert getattr(returned_task, "_tldw_stop_event") is stop_event
    assert len(calls) == 1
    assert calls[0]["inventory"] is worker_inventory
    assert calls[0]["name"] == "usage_aggregator"
    assert calls[0]["task_name"] == "usage_aggregator"
    assert calls[0]["category"] == "usage"
    assert calls[0]["shutdown_phase"] == usage_aggregator.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert callable(calls[0]["coroutine_factory"])


@pytest.mark.asyncio
async def test_aggregate_sqlite(monkeypatch):
    # Force single-user SQLite
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "agg-sqlite-key-1234567890")
    monkeypatch.setenv("USAGE_LOG_ENABLED", "true")
    import uuid as _uuid
    dburl = f"sqlite:///./Databases/users_test_usage_agg_{_uuid.uuid4().hex}.sqlite"
    monkeypatch.setenv("DATABASE_URL", dburl)

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()

    pool = await get_db_pool()
    await _ensure_usage_tables(pool)

    # Two users, one error, different latency/bytes
    await _insert_log(pool, user_id=1, status=200, latency_ms=100, bytes_val=10)
    await _insert_log(pool, user_id=1, status=500, latency_ms=300, bytes_val=30)
    await _insert_log(pool, user_id=2, status=200, latency_ms=200, bytes_val=50)

    day = datetime.now(timezone.utc).date().isoformat()
    await aggregate_usage_daily(day=day)

    # Verify aggregates
    if pool.pool:
        cnt_logs = await pool.fetchval("SELECT COUNT(*) FROM usage_log WHERE date(ts) = $1", day)
        rows = await pool.fetchall("SELECT user_id, day, requests, errors, bytes_total, latency_avg_ms FROM usage_daily WHERE day = $1 ORDER BY user_id", day)
    else:
        cnt_logs = await pool.fetchval("SELECT COUNT(*) FROM usage_log WHERE DATE(ts) = ?", day)
        rows = await pool.fetchall("SELECT user_id, day, requests, errors, bytes_total, latency_avg_ms FROM usage_daily WHERE day = ? ORDER BY user_id", day)

    # Normalize rows to list[dict]
    def _to_dict(r):
        if isinstance(r, dict):
            return r
        return {
            "user_id": r[0],
            "day": str(r[1]),
            "requests": r[2],
            "errors": r[3],
            "bytes_total": r[4],
            "latency_avg_ms": float(r[5]) if r[5] is not None else None,
        }

    rows = [_to_dict(r) for r in rows]
    # Should have aggregated our 3 log rows into 2 users
    assert int(cnt_logs or 0) == 3
    assert len(rows) >= 2
    assert any(r["user_id"] == 1 and r["requests"] == 2 and r["errors"] == 1 and r["bytes_total"] == 40 for r in rows)
    # Avg latency for user 1: (100 + 300) / 2 = 200
    assert any(r["user_id"] == 1 and int(round(r["latency_avg_ms"] or 0)) == 200 for r in rows)
    assert any(r["user_id"] == 2 and r["requests"] == 1 and r["errors"] == 0 and r["bytes_total"] == 50 for r in rows)


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("DATABASE_URL", "").startswith("postgresql"), reason="PostgreSQL DATABASE_URL not set")
@pytest.mark.asyncio
async def test_aggregate_postgres_branch():
    # In Postgres environment, verify PG-specific branch works without requiring cross-package fixtures

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    pool = await get_db_pool()
    await _ensure_usage_tables(pool)

    # Seed PG usage_log
    await _insert_log(pool, user_id=3, status=200, latency_ms=100, bytes_val=5)
    await _insert_log(pool, user_id=3, status=404, latency_ms=300, bytes_val=15)

    day = datetime.now(timezone.utc).date().isoformat()
    await aggregate_usage_daily(day=day)

    rows = await pool.fetchall("SELECT user_id, day, requests, errors, bytes_total, latency_avg_ms FROM usage_daily WHERE day = $1", day)
    assert any(dict(r)["user_id"] == 3 and dict(r)["requests"] == 2 and dict(r)["errors"] == 1 for r in rows)


@pytest.mark.asyncio
async def test_aggregate_usage_daily_failure_log_is_sanitized(monkeypatch):
    import tldw_Server_API.app.services.usage_aggregator as usage_aggregator

    class _ExplodingRepo:
        def __init__(self, _pool):
            pass

        async def aggregate_usage_daily_for_day(self, *, day):
            _ = day
            raise RuntimeError("usage backend exploded at /tmp/usage-secret-token")

    records: list[str] = []
    sink_id = usage_aggregator.logger.add(
        lambda message: records.append(str(message)),
        level="DEBUG",
        format="{message} {extra}",
    )
    monkeypatch.setattr(usage_aggregator, "AuthnzUsageRepo", _ExplodingRepo)

    try:
        await usage_aggregator.aggregate_usage_daily(db_pool=object(), day="2026-04-29")
    finally:
        usage_aggregator.logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "usage_daily aggregation skipped/failed" in rendered
    assert "RuntimeError" in rendered
    assert "usage backend exploded" not in rendered
    assert "/tmp/usage-secret-token" not in rendered


@pytest.mark.asyncio
async def test_aggregator_loop_outer_exception_log_is_sanitized(monkeypatch):
    import tldw_Server_API.app.services.usage_aggregator as usage_aggregator

    class _Settings:
        USAGE_LOG_ENABLED = True
        USAGE_AGGREGATOR_INTERVAL_MINUTES = 60

    async def _explode():
        raise RuntimeError("loop exploded at /tmp/usage-secret-token")

    records: list[str] = []
    sink_id = usage_aggregator.logger.add(
        lambda message: records.append(str(message)),
        level="DEBUG",
        format="{message} {extra}",
    )
    monkeypatch.setattr(usage_aggregator, "get_settings", lambda: _Settings())
    monkeypatch.setattr(usage_aggregator, "aggregate_usage_daily", _explode)

    try:
        await usage_aggregator._aggregator_loop(asyncio.Event())
    finally:
        usage_aggregator.logger.remove(sink_id)

    rendered = "\n".join(records)
    assert "Usage aggregator loop exited" in rendered
    assert "RuntimeError" in rendered
    assert "loop exploded" not in rendered
    assert "/tmp/usage-secret-token" not in rendered

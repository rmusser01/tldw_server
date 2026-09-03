from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _setup_env(tmp_path):
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-api-key"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_usage_reporting.db'}"
    os.environ["USAGE_LOG_ENABLED"] = "true"
    # Exclude all paths from middleware usage logging to avoid NULL user_id rows
    # interfering with FK-constrained aggregation during this test.
    os.environ["USAGE_LOG_EXCLUDE_PREFIXES"] = "[\"/\"]"


async def _insert_usage_rows_sqlite():
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    pool = await get_db_pool()
    # Create a test user to satisfy FK on usage_daily.user_id
    user_id = await ensure_test_user(pool, "usageuser", "usageuser@example.com")
    # Ensure usage tables exist (migrations may be skipped in some single-user SQLite setups)
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
            bytes_in INTEGER,
            meta TEXT,
            request_id TEXT
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
            bytes_in_total INTEGER DEFAULT 0,
            latency_avg_ms REAL,
            PRIMARY KEY (user_id, day)
        )
        """
    )
    # Backfill legacy tables that may have been created before bytes_in fields existed.
    usage_log_cols = {row["name"] for row in await pool.fetchall("PRAGMA table_info(usage_log)")}
    if "bytes_in" not in usage_log_cols:
        await pool.execute("ALTER TABLE usage_log ADD COLUMN bytes_in INTEGER")
    if "request_id" not in usage_log_cols:
        await pool.execute("ALTER TABLE usage_log ADD COLUMN request_id TEXT")

    usage_daily_cols = {row["name"] for row in await pool.fetchall("PRAGMA table_info(usage_daily)")}
    if "bytes_in_total" not in usage_daily_cols:
        await pool.execute("ALTER TABLE usage_daily ADD COLUMN bytes_in_total INTEGER DEFAULT 0")

    # Insert a few rows for today (ts default CURRENT_TIMESTAMP)
    # Use SQLite parameter placeholders
    await pool.execute(
        "INSERT INTO usage_log (user_id, key_id, endpoint, status, latency_ms, bytes, bytes_in, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        int(user_id),
        None,
        "/api/v1/chat/completions",
        200,
        50,
        100,
        25,
        None,
    )
    await pool.execute(
        "INSERT INTO usage_log (user_id, key_id, endpoint, status, latency_ms, bytes, bytes_in, meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        int(user_id),
        None,
        "/api/v1/embeddings",
        500,
        150,
        300,
        40,
        None,
    )
    return int(user_id)


@pytest.mark.asyncio
async def test_admin_usage_endpoints_sqlite_smoke(monkeypatch, tmp_path):
    # Arrange environment and singletons
    _setup_env(tmp_path)

    # Reset AuthNZ singletons to pick up new env
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    # Start app to run lifespan and ensure migrations
    with TestClient(app, headers=headers) as client:
        # Seed usage_log rows (SQLite)
        u_id = await _insert_usage_rows_sqlite()

        # Trigger aggregation for today
        day_str = datetime.now(timezone.utc).date().isoformat()
        r_agg = client.post(f"/api/v1/admin/usage/aggregate?day={day_str}")
        assert r_agg.status_code == 200, r_agg.text

        # Query daily for today
        r_daily = client.get(f"/api/v1/admin/usage/daily?start={day_str}&end={day_str}")
        assert r_daily.status_code == 200, r_daily.text
        daily = r_daily.json()
        assert "items" in daily and isinstance(daily["items"], list)
        # Expect at least one row for our created user
        assert any(int(row["user_id"]) == int(u_id) for row in daily["items"]), daily

        # Query top users
        r_top = client.get(f"/api/v1/admin/usage/top?start={day_str}&end={day_str}&limit=5")
        assert r_top.status_code == 200, r_top.text
        top = r_top.json()
        assert "items" in top and isinstance(top["items"], list)
        assert len(top["items"]) >= 1

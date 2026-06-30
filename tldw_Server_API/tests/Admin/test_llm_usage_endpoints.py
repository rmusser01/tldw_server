from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _setup_env(tmp_path):
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-api-key-llm"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_llm_usage_endpoints.db'}"


async def _ensure_llm_tables_and_seed():
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    pool = await get_db_pool()
    # Ensure two users exist and capture their IDs (respect FK constraints)
    import uuid as _uuid
    if pool.pool:
        # PostgreSQL-style
        await pool.execute(
            "INSERT INTO users (uuid, username, email, password_hash, is_active) VALUES ($1,$2,$3,$4,TRUE) ON CONFLICT (username) DO NOTHING",
            str(_uuid.uuid4()), "llmuser1", "llmuser1@example.com", "x"
        )
        await pool.execute(
            "INSERT INTO users (uuid, username, email, password_hash, is_active) VALUES ($1,$2,$3,$4,TRUE) ON CONFLICT (username) DO NOTHING",
            str(_uuid.uuid4()), "llmuser2", "llmuser2@example.com", "x"
        )
        u1 = await pool.fetchval("SELECT id FROM users WHERE username = $1", "llmuser1")
        u2 = await pool.fetchval("SELECT id FROM users WHERE username = $1", "llmuser2")
    else:
        # SQLite-style
        await pool.execute(
            "INSERT OR IGNORE INTO users (uuid, username, email, password_hash, is_active) VALUES (?,?,?,?,1)",
            str(_uuid.uuid4()), "llmuser1", "llmuser1@example.com", "x"
        )
        await pool.execute(
            "INSERT OR IGNORE INTO users (uuid, username, email, password_hash, is_active) VALUES (?,?,?,?,1)",
            str(_uuid.uuid4()), "llmuser2", "llmuser2@example.com", "x"
        )
        u1 = await pool.fetchval("SELECT id FROM users WHERE username = ?", "llmuser1")
        u2 = await pool.fetchval("SELECT id FROM users WHERE username = ?", "llmuser2")
    # Create tables if not exist
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
        # Seed two rows (ts default now)
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, endpoint, operation, provider, model, status, latency_ms, prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)",
            int(u1), "/api/v1/chat/completions", "chat", "openai", "gpt-3.5-turbo", 200, 120, 100, 50, 150, 0.1, "USD", False
        )
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, endpoint, operation, provider, model, status, latency_ms, prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)",
            int(u2), "/api/v1/embeddings", "embeddings", "openai", "text-embedding-3-small", 500, 250, 200, 0, 200, 0.02, "USD", True
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
            "INSERT INTO llm_usage_log (user_id, endpoint, operation, provider, model, status, latency_ms, prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            int(u1), "/api/v1/chat/completions", "chat", "openai", "gpt-3.5-turbo", 200, 120, 100, 50, 150, 0.1, "USD", 0
        )
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, endpoint, operation, provider, model, status, latency_ms, prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            int(u2), "/api/v1/embeddings", "embeddings", "openai", "text-embedding-3-small", 500, 250, 200, 0, 200, 0.02, "USD", 1
        )


async def _ensure_llm_cache_columns():
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    pool = await get_db_pool()
    columns = [
        ("cached_input_tokens", "INTEGER"),
        ("cache_write_input_tokens", "INTEGER"),
        ("cache_read_input_tokens", "INTEGER"),
        ("billable_input_tokens", "INTEGER"),
        ("estimate_source", "TEXT"),
        ("raw_usage_metadata_json", "TEXT"),
    ]
    for column, column_type in columns:
        if pool.pool:
            await pool.execute(f"ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS {column} {column_type}")
        else:
            import contextlib

            with contextlib.suppress(Exception):
                await pool.execute(f"ALTER TABLE llm_usage_log ADD COLUMN {column} {column_type}")


async def _insert_cache_usage_rows():
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    pool = await get_db_pool()
    u1 = await pool.fetchval(
        "SELECT id FROM users WHERE username = $1" if pool.pool else "SELECT id FROM users WHERE username = ?",
        "llmuser1",
    )
    if pool.pool:
        await pool.execute(
            """
            INSERT INTO llm_usage_log (
                user_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated,
                request_id, cached_input_tokens, cache_write_input_tokens, cache_read_input_tokens,
                billable_input_tokens, estimate_source, raw_usage_metadata_json
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
            """,
            int(u1), "/api/v1/chat/completions", "chat", "anthropic", "claude-3-sonnet", 200, 120,
            100, 25, 125, 0.001, "USD", False,
            "req-cache-admin", 70, 10, 70, 20, "provider_usage",
            '{"api_key":"sk-secret","cache_read_input_tokens":70}',
        )
        await pool.execute(
            """
            INSERT INTO llm_usage_log (
                user_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated,
                request_id, cached_input_tokens, cache_write_input_tokens, cache_read_input_tokens,
                billable_input_tokens, estimate_source, raw_usage_metadata_json
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
            """,
            int(u1), "/api/v1/chat/completions", "chat", "llama.cpp", "local-model", 200, 90,
            80, 30, 110, 0.0, "USD", True,
            "req-local-diagnostic", 0, 0, 0, 80, "local_diagnostic",
            '{"tldw_local_cache_diagnostics":{"provider":"llama.cpp","request_extension_keys":["cache_prompt"]}}',
        )
    else:
        await pool.execute(
            """
            INSERT INTO llm_usage_log (
                user_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated,
                request_id, cached_input_tokens, cache_write_input_tokens, cache_read_input_tokens,
                billable_input_tokens, estimate_source, raw_usage_metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            int(u1), "/api/v1/chat/completions", "chat", "anthropic", "claude-3-sonnet", 200, 120,
            100, 25, 125, 0.001, "USD", 0,
            "req-cache-admin", 70, 10, 70, 20, "provider_usage",
            '{"api_key":"sk-secret","cache_read_input_tokens":70}',
        )
        await pool.execute(
            """
            INSERT INTO llm_usage_log (
                user_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens, total_cost_usd, currency, estimated,
                request_id, cached_input_tokens, cache_write_input_tokens, cache_read_input_tokens,
                billable_input_tokens, estimate_source, raw_usage_metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            int(u1), "/api/v1/chat/completions", "chat", "llama.cpp", "local-model", 200, 90,
            80, 30, 110, 0.0, "USD", 1,
            "req-local-diagnostic", 0, 0, 0, 80, "local_diagnostic",
            '{"tldw_local_cache_diagnostics":{"provider":"llama.cpp","request_extension_keys":["cache_prompt"]}}',
        )


@pytest.mark.asyncio
async def test_llm_usage_endpoints_sqlite(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
    with TestClient(app, headers=headers) as client:
        await _ensure_llm_tables_and_seed()

        # List
        r = client.get("/api/v1/admin/llm-usage?operation=chat&limit=10")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data.get('items'), list)
        assert any(row.get('operation') == 'chat' for row in data['items'])
        assert "cached_input_tokens" in data["items"][0]
        assert "raw_usage_metadata_json" not in data["items"][0]
        assert data["pagination"]["total"] == data["total"]
        assert data["pagination"]["limit"] == 10
        assert data["pagination"]["offset"] == 0
        assert data["has_more"] == data["pagination"]["has_more"]
        assert data["next_offset"] == data["pagination"]["next_offset"]

        # Summary by user
        r2 = client.get("/api/v1/admin/llm-usage/summary?group_by=user")
        assert r2.status_code == 200
        s = r2.json()
        assert isinstance(s.get('items'), list)
        assert any('requests' in row for row in s['items'])
        assert "cached_input_tokens" in s["items"][0]
        assert "provider_usage_count" in s["items"][0]

        # Summary by provider/day with provider filter (used by provider trend sparklines)
        r2b = client.get("/api/v1/admin/llm-usage/summary?group_by=provider&group_by=day&provider=openai")
        assert r2b.status_code == 200
        s2 = r2b.json()
        assert isinstance(s2.get('items'), list)
        assert all(row.get('group_value') == 'openai' for row in s2['items'])
        assert all('group_value_secondary' in row for row in s2['items'])

        # Reject more than two group_by dimensions
        r2c = client.get("/api/v1/admin/llm-usage/summary?group_by=user&group_by=provider&group_by=day")
        assert r2c.status_code == 422

        # CSV export
        r3 = client.get("/api/v1/admin/llm-usage/export.csv?operation=chat&limit=5")
        assert r3.status_code == 200
        assert r3.text.startswith("id,ts,user_id,key_id,endpoint,operation")

        # Cache-aware reporting remains bounded and does not expose raw provider metadata.
        await _ensure_llm_cache_columns()
        await _insert_cache_usage_rows()

        r = client.get("/api/v1/admin/llm-usage?provider=anthropic&limit=10")
        assert r.status_code == 200
        items = r.json()["items"]
        row = next(item for item in items if item["request_id"] == "req-cache-admin")
        assert row["prompt_tokens"] == 100
        assert row["cached_input_tokens"] == 70
        assert row["cache_write_input_tokens"] == 10
        assert row["cache_read_input_tokens"] == 70
        assert row["billable_input_tokens"] == 20
        assert row["completion_tokens"] == 25
        assert row["estimate_source"] == "provider_usage"
        assert "raw_usage_metadata_json" not in row

        summary = client.get("/api/v1/admin/llm-usage/summary?group_by=provider&provider=anthropic")
        assert summary.status_code == 200
        summary_row = summary.json()["items"][0]
        assert summary_row["group_value"] == "anthropic"
        assert summary_row["input_tokens"] == 100
        assert summary_row["cached_input_tokens"] == 70
        assert summary_row["cache_write_input_tokens"] == 10
        assert summary_row["cache_read_input_tokens"] == 70
        assert summary_row["billable_input_tokens"] == 20
        assert summary_row["output_tokens"] == 25
        assert summary_row["provider_usage_count"] == 1
        assert summary_row["local_diagnostic_count"] == 0

        local_summary = client.get("/api/v1/admin/llm-usage/summary?group_by=provider&provider=llama.cpp")
        assert local_summary.status_code == 200
        local_row = local_summary.json()["items"][0]
        assert local_row["group_value"] == "llama.cpp"
        assert local_row["cached_input_tokens"] == 0
        assert local_row["billable_input_tokens"] == 80
        assert local_row["local_diagnostic_count"] == 1
        assert local_row["provider_usage_count"] == 0

        csv_resp = client.get("/api/v1/admin/llm-usage/export.csv?provider=anthropic&limit=10")
        assert csv_resp.status_code == 200
        header = csv_resp.text.splitlines()[0]
        assert "cached_input_tokens" in header
        assert "cache_write_input_tokens" in header
        assert "cache_read_input_tokens" in header
        assert "billable_input_tokens" in header
        assert "estimate_source" in header
        assert "raw_usage_metadata_json" not in header
        assert "sk-secret" not in csv_resp.text

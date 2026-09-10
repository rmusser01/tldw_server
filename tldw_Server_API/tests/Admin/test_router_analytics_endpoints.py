from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _setup_env(tmp_path) -> None:
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-router-analytics-endpoints"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_router_analytics_endpoints.db'}"


async def _seed_router_rows() -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    pool = await get_db_pool()
    user_id = await ensure_test_user(
        pool, "router_endpoints_user", "router_endpoints_user@example.com"
    )

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
            prompt_cost_usd REAL,
            completion_cost_usd REAL,
            total_cost_usd REAL,
            currency TEXT,
            estimated INTEGER,
            request_id TEXT,
            remote_ip TEXT,
            user_agent TEXT,
            token_name TEXT,
            conversation_id TEXT
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key_hash TEXT,
            key_prefix TEXT,
            name TEXT,
            status TEXT DEFAULT 'active'
        )
        """
    )

    cols = {row["name"] for row in await pool.fetchall("PRAGMA table_info(llm_usage_log)")}
    if "remote_ip" not in cols:
        await pool.execute("ALTER TABLE llm_usage_log ADD COLUMN remote_ip TEXT")
    if "user_agent" not in cols:
        await pool.execute("ALTER TABLE llm_usage_log ADD COLUMN user_agent TEXT")
    if "token_name" not in cols:
        await pool.execute("ALTER TABLE llm_usage_log ADD COLUMN token_name TEXT")
    if "conversation_id" not in cols:
        await pool.execute("ALTER TABLE llm_usage_log ADD COLUMN conversation_id TEXT")

    key_cols = {row["name"] for row in await pool.fetchall("PRAGMA table_info(api_keys)")}
    if "llm_budget_day_tokens" not in key_cols:
        await pool.execute("ALTER TABLE api_keys ADD COLUMN llm_budget_day_tokens INTEGER")
    if "llm_budget_month_tokens" not in key_cols:
        await pool.execute("ALTER TABLE api_keys ADD COLUMN llm_budget_month_tokens INTEGER")
    if "llm_budget_day_usd" not in key_cols:
        await pool.execute("ALTER TABLE api_keys ADD COLUMN llm_budget_day_usd REAL")
    if "llm_budget_month_usd" not in key_cols:
        await pool.execute("ALTER TABLE api_keys ADD COLUMN llm_budget_month_usd REAL")

    await pool.execute(
        """
        INSERT OR REPLACE INTO api_keys (
            id, user_id, key_hash, name, status,
            llm_budget_day_tokens, llm_budget_month_tokens, llm_budget_day_usd, llm_budget_month_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        201,
        user_id,
        "hash-201",
        "Admin",
        "active",
        500,
        5000,
        5.0,
        50.0,
    )
    await pool.execute(
        """
        INSERT OR REPLACE INTO api_keys (
            id, user_id, key_hash, name, status,
            llm_budget_day_tokens, llm_budget_month_tokens, llm_budget_day_usd, llm_budget_month_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        202,
        user_id,
        "hash-202",
        "Ops",
        "active",
        10,
        100,
        0.01,
        1.0,
    )

    await pool.execute(
        """
        INSERT INTO llm_usage_log (
            user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
            prompt_tokens, completion_tokens, total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd,
            currency, estimated, request_id, remote_ip, user_agent, token_name, conversation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        user_id,
        201,
        "/api/v1/chat/completions",
        "chat",
        "openai",
        "gpt-4o-mini",
        200,
        600,
        100,
        40,
        140,
        0.0,
        0.0,
        0.0,
        "USD",
        0,
        "router-endpoint-1",
        "127.0.0.1",
        "curl/8.8.0",
        "Admin",
        "conv-1",
    )
    await pool.execute(
        """
        INSERT INTO llm_usage_log (
            user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
            prompt_tokens, completion_tokens, total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd,
            currency, estimated, request_id, remote_ip, user_agent, token_name, conversation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        user_id,
        202,
        "/api/v1/chat/completions",
        "chat",
        "groq",
        "llama-3.3-70b",
        500,
        300,
        20,
        0,
        20,
        0.0,
        0.0,
        0.0,
        "USD",
        1,
        "router-endpoint-2",
        None,
        None,
        None,
        "conv-2",
    )


@pytest.mark.asyncio
async def test_router_analytics_endpoints_status_breakdowns_meta(monkeypatch, tmp_path):
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
    with TestClient(app, headers=headers) as client:
        await _seed_router_rows()

        status_resp = client.get("/api/v1/admin/router-analytics/status", params={"range": "1h"})
        assert status_resp.status_code == 200, status_resp.text
        status_payload = status_resp.json()
        assert status_payload["kpis"]["requests"] == 2
        assert status_payload["providers_available"] == 2
        assert status_payload["providers_online"] == 1
        assert "data_window" in status_payload

        breakdowns_resp = client.get(
            "/api/v1/admin/router-analytics/status/breakdowns",
            params={"range": "1h"},
        )
        assert breakdowns_resp.status_code == 200, breakdowns_resp.text
        breakdowns_payload = breakdowns_resp.json()
        assert isinstance(breakdowns_payload["providers"], list)
        assert any(row["key"] == "openai" for row in breakdowns_payload["providers"])
        assert any(row["key"] == "unknown" for row in breakdowns_payload["remote_ips"])

        meta_resp = client.get("/api/v1/admin/router-analytics/meta")
        assert meta_resp.status_code == 200, meta_resp.text
        meta_payload = meta_resp.json()
        assert any(option["value"] == "openai" for option in meta_payload["providers"])
        ops_option = next(option for option in meta_payload["tokens"] if option["label"] == "Ops")
        assert ops_option["value"] == "202"
        assert ops_option["key_id"] == 202

        filtered_status_resp = client.get(
            "/api/v1/admin/router-analytics/status",
            params={"range": "1h", "token_id": ops_option["value"]},
        )
        assert filtered_status_resp.status_code == 200, filtered_status_resp.text
        filtered_status_payload = filtered_status_resp.json()
        assert filtered_status_payload["kpis"]["requests"] == 1

        quota_resp = client.get("/api/v1/admin/router-analytics/quota", params={"range": "1h"})
        assert quota_resp.status_code == 200, quota_resp.text
        quota_payload = quota_resp.json()
        assert quota_payload["summary"]["keys_total"] >= 1
        assert "items" in quota_payload
        keyed = {int(row["key_id"]): row for row in quota_payload["items"]}
        assert 202 in keyed

        providers_resp = client.get("/api/v1/admin/router-analytics/providers", params={"range": "1h"})
        assert providers_resp.status_code == 200, providers_resp.text
        providers_payload = providers_resp.json()
        assert providers_payload["summary"]["providers_total"] == 2
        assert providers_payload["summary"]["providers_online"] == 1
        assert providers_payload["summary"]["failover_events"] == 1
        provider_rows = {row["provider"]: row for row in providers_payload["items"]}
        assert provider_rows["openai"]["online"] is True
        assert provider_rows["groq"]["online"] is False

        access_resp = client.get("/api/v1/admin/router-analytics/access", params={"range": "1h"})
        assert access_resp.status_code == 200, access_resp.text
        access_payload = access_resp.json()
        assert access_payload["summary"]["token_names_total"] >= 2
        assert access_payload["summary"]["remote_ips_total"] >= 1
        assert access_payload["summary"]["user_agents_total"] >= 1
        assert access_payload["summary"]["anonymous_requests"] == 1

        network_resp = client.get("/api/v1/admin/router-analytics/network", params={"range": "1h"})
        assert network_resp.status_code == 200, network_resp.text
        network_payload = network_resp.json()
        assert network_payload["summary"]["remote_ips_total"] >= 1
        assert network_payload["summary"]["endpoints_total"] >= 1
        assert network_payload["summary"]["operations_total"] >= 1
        assert network_payload["summary"]["error_requests"] == 1

        models_resp = client.get("/api/v1/admin/router-analytics/models", params={"range": "1h"})
        assert models_resp.status_code == 200, models_resp.text
        models_payload = models_resp.json()
        assert models_payload["summary"]["models_total"] == 2
        assert models_payload["summary"]["models_online"] == 1
        assert models_payload["summary"]["providers_total"] == 2
        assert models_payload["summary"]["error_requests"] == 1

        conversations_resp = client.get("/api/v1/admin/router-analytics/conversations", params={"range": "1h"})
        assert conversations_resp.status_code == 200, conversations_resp.text
        conversations_payload = conversations_resp.json()
        assert conversations_payload["summary"]["conversations_total"] == 2
        assert conversations_payload["summary"]["active_conversations"] == 1
        assert conversations_payload["summary"]["avg_requests_per_conversation"] == pytest.approx(1.0)
        assert conversations_payload["summary"]["error_requests"] == 1
        rows = {row["conversation_id"]: row for row in conversations_payload["items"]}
        assert rows["conv-1"]["success_rate_pct"] == pytest.approx(100.0)
        assert rows["conv-2"]["success_rate_pct"] == pytest.approx(0.0)

        log_resp = client.get("/api/v1/admin/router-analytics/log", params={"range": "1h"})
        assert log_resp.status_code == 200, log_resp.text
        log_payload = log_resp.json()
        assert log_payload["summary"]["requests_total"] == 2
        assert log_payload["summary"]["error_requests"] == 1
        assert log_payload["summary"]["estimated_requests"] == 1
        assert log_payload["summary"]["request_ids_total"] == 2
        assert len(log_payload["items"]) == 2
        assert log_payload["items"][0]["request_id"] == "router-endpoint-2"
        assert log_payload["items"][0]["error"] is True

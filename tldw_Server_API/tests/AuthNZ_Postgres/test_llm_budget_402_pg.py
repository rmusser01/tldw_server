import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_budget_middleware_returns_402_postgres(test_db_pool, monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_db_handling
    from tldw_Server_API.app.core.AuthNZ import api_key_manager as api_key_manager_module
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as reset_auth_settings
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app

    # Ensure multi-user mode for AuthNZ + budgets
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_auth_settings()

    pool = test_db_pool
    app_settings['CSRF_ENABLED'] = False

    # Ensure llm_usage_log table exists
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_usage_log (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            key_id INTEGER REFERENCES api_keys(id) ON DELETE SET NULL,
            endpoint TEXT,
            operation TEXT,
            provider TEXT,
            model TEXT,
            status INTEGER,
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            prompt_cost_usd DOUBLE PRECISION,
            completion_cost_usd DOUBLE PRECISION,
            total_cost_usd DOUBLE PRECISION,
            currency TEXT DEFAULT 'USD',
            estimated BOOLEAN DEFAULT FALSE,
            request_id TEXT
        )
        """
    )

    # Insert a user
    user_id = await ensure_test_user(pool, "pgbudget", "pgbudget@example.com")

    # Create a virtual key with small daily token budget
    mgr = APIKeyManager(pool)
    await mgr.initialize()

    async def _get_mgr_override():
        return mgr

    # Ensure all API-key auth paths (AuthPrincipal resolver, budget middleware, etc.)
    # reuse the same manager instance bound to the Postgres test pool.
    monkeypatch.setattr(auth_deps, "get_api_key_manager", _get_mgr_override)
    monkeypatch.setattr(api_key_manager_module, "get_api_key_manager", _get_mgr_override)
    monkeypatch.setattr(user_db_handling, "get_api_key_manager", _get_mgr_override)
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-budget-pg",
        allowed_endpoints=["chat.completions"],
        budget_day_tokens=100,
    )
    key_id = res['id']
    vkey = res['key']

    # Add usage that exceeds daily tokens
    await pool.execute(
        """
        INSERT INTO llm_usage_log (
            ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
            prompt_tokens, completion_tokens, total_tokens,
            prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
        ) VALUES (
            CURRENT_TIMESTAMP, $1, $2, 'api', 'chat', 'openai', 'gpt-4o-mini', 200, 100,
            50, 60, 110,
            0.02, 0.03, 0.05, 'USD', FALSE
        )
        """,
        user_id, key_id,
    )

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code == 402, r.text
        assert "budget_exceeded" in r.text

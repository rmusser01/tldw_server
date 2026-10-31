
import pytest

from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_virtual_keys_and_budget_postgres(test_db_pool):
    pool = test_db_pool

    # Ensure org/team tables exist for FK references in api_keys
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(64) UNIQUE,
            name VARCHAR(255) UNIQUE NOT NULL,
            slug VARCHAR(255) UNIQUE,
            owner_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS teams (
            id SERIAL PRIMARY KEY,
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255),
            description TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (org_id, name)
        )
        """
    )

    # Prepare api_keys via manager (creates table + columns)
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager(pool)
    await mgr.initialize()

    # Ensure LLM usage tables exist
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
    user_id = await ensure_test_user(pool, "pguser", "pguser@example.com")

    # Create virtual key with small budget
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-pg",
        allowed_endpoints=["chat.completions"],
        budget_day_tokens=500,
        budget_month_usd=2.0,
    )
    assert res['id'] > 0 and res['key'].startswith('tldw_')
    key_id = res['id']

    # Check budget helper (initially false)
    from tldw_Server_API.app.core.AuthNZ.virtual_keys import is_key_over_budget
    over = await is_key_over_budget(key_id)
    assert over['over'] is False

    # Add usage that exceeds day_tokens
    await pool.execute(
        """
        INSERT INTO llm_usage_log (
            ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
            prompt_tokens, completion_tokens, total_tokens,
            prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
        ) VALUES (
            CURRENT_TIMESTAMP, $1, $2, 'api', 'chat', 'openai', 'gpt-4o-mini', 200, 120,
            200, 400, 600,
            0.10, 0.20, 0.30, 'USD', FALSE
        )
        """,
        user_id, key_id,
    )
    await pool.execute(
        """
        INSERT INTO llm_usage_log (
            ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
            prompt_tokens, completion_tokens, total_tokens,
            prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
        ) VALUES (
            CURRENT_TIMESTAMP, $1, $2, 'api', 'chat', 'openai', 'gpt-4o-mini', 200, 50,
            0, 200, 200,
            0.00, 0.05, 0.05, 'USD', FALSE
        )
        """,
        user_id, key_id,
    )

    over2 = await is_key_over_budget(key_id)
    assert over2['over'] is True
    assert any(r.startswith('day_tokens_exceeded') for r in over2['reasons'])

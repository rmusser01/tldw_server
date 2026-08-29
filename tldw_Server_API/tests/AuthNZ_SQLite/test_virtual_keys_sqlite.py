import os
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_create_virtual_key_and_budget_checks(tmp_path):
    # Configure SQLite for AuthNZ
    os.environ['AUTH_MODE'] = 'single_user'
    db_path = tmp_path / 'users.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    # Reset singletons
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Create a dummy user for FK
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("bob", "bob@example.com", "x"),
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "bob")

    # Create a virtual key
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk1",
        allowed_endpoints=["chat.completions", "embeddings"],
        budget_day_tokens=1000,
        budget_month_usd=1.00,
    )
    assert res['id'] > 0 and res['key'].startswith('tldw_')
    key_id = res['id']

    # Initially not over budget
    from tldw_Server_API.app.core.AuthNZ.virtual_keys import is_key_over_budget
    result = await is_key_over_budget(key_id)
    assert result['over'] is False

    # Insert some usage: 900 tokens, $0.50
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO llm_usage_log (
                ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens,
                prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, 'api', 'chat', 'openai', 'gpt-4o-mini', 200, 100,
                400, 500, 900,
                0.2, 0.3, 0.5, 'USD', 0
            )
            """,
            (user_id, key_id),
        )

    result = await is_key_over_budget(key_id)
    assert result['over'] is False

    # Add 200 more tokens -> exceed day_tokens=1000
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO llm_usage_log (
                ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens,
                prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, 'api', 'chat', 'openai', 'gpt-4o-mini', 200, 80,
                0, 200, 200, 0.0, 0.05, 0.05, 'USD', 0
            )
            """,
            (user_id, key_id),
        )

    result = await is_key_over_budget(key_id)
    assert result['over'] is True
    assert any(r.startswith('day_tokens_exceeded') for r in result['reasons'])

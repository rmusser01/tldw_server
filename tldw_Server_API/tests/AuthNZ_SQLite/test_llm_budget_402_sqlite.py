import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _chat_stub_response():
    return {
        "id": "chatcmpl-budget",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _override_chat_deps(app, monkeypatch, chacha_db_path: Path):
    import tldw_Server_API.app.api.v1.endpoints.chat as chat_endpoint
    import tldw_Server_API.app.api.v1.schemas.chat_request_schemas as schema_chat
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    monkeypatch.setattr(chat_endpoint, "perform_chat_api_call", lambda **kwargs: _chat_stub_response())
    monkeypatch.setattr(
        chat_endpoint,
        "API_KEYS",
        {**(chat_endpoint.API_KEYS or {}), "openai": "test"},
        raising=False,
    )
    monkeypatch.setattr(
        schema_chat,
        "API_KEYS",
        {**(schema_chat.API_KEYS or {}), "openai": "test"},
        raising=False,
    )

    async def _override_chacha_db_for_user(current_user=None):
        return CharactersRAGDB(db_path=chacha_db_path, client_id="test-llm-budget")

    app.dependency_overrides[chacha_deps.get_chacha_db_for_user] = _override_chacha_db_for_user


@pytest.mark.asyncio
async def test_llm_budget_middleware_returns_402_on_overage(tmp_path):
    # Configure SQLite for AuthNZ
    os.environ['AUTH_MODE'] = 'multi_user'
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-budget-402-12345678901234567890'
    db_path = tmp_path / 'users.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    # Reset singletons and ensure schema
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Create a user
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("budgetuser", "budgetuser@example.com", "x"),
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "budgetuser")

    # Create a virtual key with small daily token budget and allow chat.completions
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    vk = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-budget",
        allowed_endpoints=["chat.completions"],
        budget_day_tokens=100,
    )
    key_id = vk['id']
    vkey = vk['key']

    # Insert usage that exceeds the daily token budget (150 >= 100)
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO llm_usage_log (
                ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens,
                prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, 'api', 'chat', 'openai', 'gpt-4o-mini', 200, 100,
                50, 100, 150,
                0.02, 0.04, 0.06, 'USD', 0
            )
            """,
            (user_id, key_id),
        )

    # Prepare TestClient and disable CSRF for this test
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code == 402, r.text
        body = r.json()
        assert body.get("error") == "budget_exceeded"
        principal = (body.get("details") or {}).get("principal") or {}
        assert principal.get("api_key_id") == key_id
        assert principal.get("user_id") == user_id


@pytest.mark.asyncio
async def test_llm_budget_allows_under_budget_chat_sqlite(tmp_path, monkeypatch):
    os.environ["AUTH_MODE"] = "multi_user"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-budget-200-12345678901234567890"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_under_budget.db'}"
    os.environ["VIRTUAL_KEYS_ENABLED"] = "true"
    os.environ["LLM_BUDGET_ENFORCE"] = "true"
    os.environ["OPENAI_API_KEY"] = "test"

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("budget_ok_user", "budget_ok@example.com", "x"),
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "budget_ok_user")

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    vk = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-budget-ok",
        allowed_endpoints=["chat.completions"],
        budget_day_tokens=1000,
    )
    vkey = vk["key"]

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps

    app_settings["CSRF_ENABLED"] = False
    _override_chat_deps(app, monkeypatch, tmp_path / "chacha_under_budget.db")

    try:
        with TestClient(app) as client:
            r = client.post(
                "/api/v1/chat/completions",
                headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
                json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert r.status_code == 200, r.text
            assert r.json().get("choices")
    finally:
        app.dependency_overrides.pop(chacha_deps.get_chacha_db_for_user, None)


@pytest.mark.asyncio
async def test_llm_budget_middleware_blocks_disallowed_endpoint_sqlite(tmp_path):
    os.environ['AUTH_MODE'] = 'multi_user'
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-endpoint-403-12345678901234567890'
    db_path = tmp_path / 'users_endpoint.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("endpointuser", "endpointuser@example.com", "x"),
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "endpointuser")

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    vk = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-endpoint",
        allowed_endpoints=["chat.completions"],
        budget_day_tokens=1000,
    )
    vkey = vk['key']

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/embeddings",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
            json={"model": "text-embedding-3-small", "input": "hello"},
        )
        assert r.status_code == 403, r.text
        assert "Endpoint 'embeddings' not allowed" in r.text


@pytest.mark.asyncio
async def test_llm_budget_middleware_enforces_usd_budgets_sqlite(tmp_path):
    os.environ['AUTH_MODE'] = 'multi_user'
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-usd-402-12345678901234567890'
    db_path = tmp_path / 'users_usd.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Create a user
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("usduser", "usduser@example.com", "x"),
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "usduser")

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()

    # Daily USD budget scenario
    vk_day = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-usd-day",
        allowed_endpoints=["chat.completions"],
        budget_day_usd=0.05,
    )
    key_day = vk_day['key']
    key_day_id = vk_day['id']

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO llm_usage_log (
                ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens,
                prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, 'api', 'chat', 'openai', 'gpt-4o-mini', 200, 120,
                50, 50, 100,
                0.03, 0.04, 0.07, 'USD', 0
            )
            """,
            (user_id, key_day_id),
        )

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": key_day, "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 402, r.text
        assert "budget_exceeded" in r.text

    # Monthly USD budget scenario (no daily limit set)
    vk_month = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-usd-month",
        allowed_endpoints=["chat.completions"],
        budget_month_usd=0.10,
    )
    key_month = vk_month['key']
    key_month_id = vk_month['id']

    month_ts = datetime.utcnow().replace(microsecond=0) - timedelta(days=2)
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO llm_usage_log (
                ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens,
                prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
            ) VALUES (
                ?, ?, ?, 'api', 'chat', 'openai', 'gpt-4o-mini', 200, 90,
                30, 30, 60,
                0.02, 0.10, 0.12, 'USD', 0
            )
            """,
            (month_ts.isoformat(), user_id, key_month_id),
        )

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": key_month, "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "ping"}]},
        )
        assert r.status_code == 402, r.text
        assert "budget_exceeded" in r.text


@pytest.mark.asyncio
async def test_llm_budget_middleware_enforces_month_tokens_sqlite(tmp_path):
    os.environ["AUTH_MODE"] = "multi_user"
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-month-tokens-12345678901234567890"
    db_path = tmp_path / "users_month_tokens.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("monthtok_user", "monthtok@example.com", "x"),
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "monthtok_user")

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    vk = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-month-tokens",
        allowed_endpoints=["chat.completions"],
        budget_month_tokens=50,
    )
    key_id = vk["id"]
    vkey = vk["key"]

    month_ts = datetime.utcnow().replace(microsecond=0) - timedelta(days=2)
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO llm_usage_log (
                ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens,
                prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
            ) VALUES (
                ?, ?, ?, 'api', 'chat', 'openai', 'gpt-4o-mini', 200, 90,
                20, 40, 60,
                0.01, 0.02, 0.03, 'USD', 0
            )
            """,
            (month_ts.isoformat(), user_id, key_id),
        )

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings["CSRF_ENABLED"] = False

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 402, r.text
        assert "budget_exceeded" in r.text


@pytest.mark.asyncio
async def test_llm_budget_middleware_returns_402_on_embeddings_overage(tmp_path):
    os.environ['AUTH_MODE'] = 'multi_user'
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-embeddings-402-12345678901234567890'
    db_path = tmp_path / 'users_embeddings.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
            ("embeduser", "embeduser@example.com", "x"),
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "embeduser")

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    vk = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-embed-budget",
        allowed_endpoints=["embeddings"],
        budget_day_tokens=50,
    )
    key_id = vk['id']
    vkey = vk['key']

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO llm_usage_log (
                ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
                prompt_tokens, completion_tokens, total_tokens,
                prompt_cost_usd, completion_cost_usd, total_cost_usd, currency, estimated
            ) VALUES (
                CURRENT_TIMESTAMP, ?, ?, 'api', 'embeddings', 'openai', 'text-embedding-3-small', 200, 80,
                30, 30, 60,
                0.01, 0.02, 0.03, 'USD', 0
            )
            """,
            (user_id, key_id),
        )

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/embeddings",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
            json={"model": "text-embedding-3-small", "input": "hello world"},
        )
        assert r.status_code == 402, r.text
        body = r.json()
        assert body.get("error") == "budget_exceeded"
        details = body.get("details") or {}
        principal = details.get("principal") or {}
        assert principal.get("api_key_id") == key_id
        assert principal.get("user_id") == user_id
        day = details.get("day") or {}
        assert day.get("tokens") in (60, "60", 60.0)
        assert day.get("usd") in (0.03, "0.03", 0.03)

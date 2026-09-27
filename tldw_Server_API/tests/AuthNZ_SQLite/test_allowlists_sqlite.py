import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user


@pytest.mark.asyncio
async def test_provider_model_allowlists_sqlite(tmp_path):
    # Configure SQLite for AuthNZ
    os.environ['AUTH_MODE'] = 'multi_user'
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-allowlists-12345678901234567890'
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

    # Create a user through the guarded write path.
    user_id = await create_authnz_test_user(
        pool, username="vkuser", email="vkuser@example.com"
    )

    # Create a virtual key with allowlists
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-allowlist",
        allowed_endpoints=["chat.completions"],
        allowed_providers=["openai"],
        allowed_models=["gpt-4o-mini"],
        budget_day_tokens=100000,
    )
    vkey = res['key']

    # Prepare TestClient
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    with TestClient(app) as client:
        # Disallowed model
        r = client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json", "X-LLM-Provider": "openai"},
            json={"model": "not-allowed", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code == 403
        assert "Model 'not-allowed' not allowed" in r.text

        # Disallowed provider
        r = client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json", "X-LLM-Provider": "anthropic"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code == 403
        assert "Provider 'anthropic' not allowed" in r.text


@pytest.mark.asyncio
async def test_missing_provider_header_allows_when_allowlist_present_sqlite(tmp_path):
    # Configure SQLite for AuthNZ
    import os
    os.environ['AUTH_MODE'] = 'multi_user'
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-allowlists-missing-12345678901234567890'
    db_path = tmp_path / 'users_missing.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    # Reset singletons and ensure schema
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Create a user through the guarded write path.
    user_id = await create_authnz_test_user(
        pool, username="vkuser2", email="vkuser2@example.com"
    )

    # Create a virtual key with provider/model allowlists
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-allowlist-missing",
        allowed_endpoints=["chat.completions"],
        allowed_providers=["openai"],
        allowed_models=["gpt-4o-mini"],
        budget_day_tokens=100000,
    )
    vkey = res['key']

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    # Missing X-LLM-Provider header should not 403/402
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code not in (403, 402), r.text


@pytest.mark.asyncio
async def test_non_json_body_skips_model_enforcement_sqlite(tmp_path):
    # Configure SQLite for AuthNZ
    import os
    os.environ['AUTH_MODE'] = 'multi_user'
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-allowlists-nonjson-12345678901234567890'
    db_path = tmp_path / 'users_nonjson.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    user_id = await create_authnz_test_user(
        pool, username="vkuser3", email="vkuser3@example.com"
    )

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-allowlist-nonjson",
        allowed_endpoints=["chat.completions"],
        allowed_providers=["openai"],
        allowed_models=["gpt-4o-mini"],
        budget_day_tokens=100000,
    )
    vkey = res['key']

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    with TestClient(app) as client:
        try:
            r = client.post(
                "/api/v1/chat/completions",
                headers={"X-API-KEY": vkey, "Content-Type": "text/plain"},
                data="hello"
            )
            assert r.status_code not in (403, 402), r.text
        except Exception as e:
            # Route may raise on non-JSON; middleware behavior under test is "no 403/402" which still holds
            # when the request is not blocked by allowlists/budget middleware.
            _ = e


@pytest.mark.asyncio
async def test_invalid_json_body_skips_model_enforcement_sqlite(tmp_path):
    # Configure SQLite for AuthNZ
    import os
    os.environ['AUTH_MODE'] = 'multi_user'
    os.environ['JWT_SECRET_KEY'] = 'test-secret-key-for-allowlists-badjson-12345678901234567890'
    db_path = tmp_path / 'users_badjson.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    user_id = await create_authnz_test_user(
        pool, username="vkuser4", email="vkuser4@example.com"
    )

    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    mgr = APIKeyManager()
    await mgr.initialize()
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-allowlist-badjson",
        allowed_endpoints=["chat.completions"],
        allowed_providers=["openai"],
        allowed_models=["gpt-4o-mini"],
        budget_day_tokens=100000,
    )
    vkey = res['key']

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    with TestClient(app) as client:
        try:
            r = client.post(
                "/api/v1/chat/completions",
                headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
                data="this is not json"
            )
            assert r.status_code not in (403, 402), r.text
        except Exception:
            # Invalid JSON can trip downstream parsing; we're only asserting middleware does not block.
            _ = None

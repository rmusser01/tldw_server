import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_model_allowlists_postgres(test_db_pool, monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_db_handling
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as reset_auth_settings
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app

    # Ensure multi-user mode for AuthNZ (virtual keys + budgets)
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_auth_settings()

    pool = test_db_pool
    app_settings['CSRF_ENABLED'] = False

    # Ensure tables used by manager and usage exist
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

    # Manager to ensure api_keys columns (explicit pool)
    mgr = APIKeyManager(pool)
    await mgr.initialize()

    # Ensure the API-key auth path used by the app reuses this manager (and
    # thus the same Postgres pool) instead of creating a separate singleton
    # bound to a different DATABASE_URL.
    async def _get_mgr_override():
        return mgr

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _get_mgr_override)
    monkeypatch.setattr(user_db_handling, "get_api_key_manager", _get_mgr_override)

    # Insert user
    user_id = await ensure_test_user(pool, "vkpg", "vkpg@example.com")

    # Create virtual key with allowlists
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-allowlist-pg",
        allowed_endpoints=["chat.completions"],
        allowed_providers=["openai"],
        allowed_models=["gpt-4o-mini"],
        budget_day_tokens=100000,
    )
    vkey = res['key']

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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_missing_provider_header_allows_when_allowlist_present_postgres(test_db_pool, monkeypatch):
    """If allowed_providers is set but X-LLM-Provider header is missing, middleware should not 403."""
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_db_handling
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as reset_auth_settings
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app

    # Ensure multi-user mode for AuthNZ
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_auth_settings()

    pool = test_db_pool
    app_settings['CSRF_ENABLED'] = False

    # Ensure minimal tables for manager
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

    # Insert user
    user_id = await ensure_test_user(pool, "vkpg-missing", "vkpg-missing@example.com")

    # Create virtual key with provider/model allowlists
    mgr = APIKeyManager(pool)
    await mgr.initialize()

    async def _get_mgr_override():
        return mgr

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _get_mgr_override)
    monkeypatch.setattr(user_db_handling, "get_api_key_manager", _get_mgr_override)
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-allowlist-missing-provider",
        allowed_endpoints=["chat.completions"],
        allowed_providers=["openai"],
        allowed_models=["gpt-4o-mini"],
        budget_day_tokens=100000,
    )
    vkey = res['key']

    with TestClient(app) as client:
        # No X-LLM-Provider header; should NOT 403 due to provider allowlist
        r = client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert r.status_code not in (403, 402), r.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_json_body_skips_model_enforcement_postgres(test_db_pool, monkeypatch):
    """With non-JSON content-type, model allowlist is skipped; ensure no 403/402 from middleware."""
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_db_handling
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as reset_auth_settings
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app

    # Ensure multi-user mode for AuthNZ
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_auth_settings()

    pool = test_db_pool
    app_settings['CSRF_ENABLED'] = False

    # Insert user
    user_id = await ensure_test_user(pool, "vkpg-nonjson", "vkpg-nonjson@example.com")

    mgr = APIKeyManager(pool)
    await mgr.initialize()

    async def _get_mgr_override():
        return mgr

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _get_mgr_override)
    monkeypatch.setattr(user_db_handling, "get_api_key_manager", _get_mgr_override)
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-allowlist-nonjson",
        allowed_endpoints=["chat.completions"],
        allowed_providers=["openai"],
        allowed_models=["gpt-4o-mini"],
        budget_day_tokens=100000,
    )
    vkey = res['key']

    async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://testserver") as client:
        r = await client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": vkey, "Content-Type": "text/plain"},
            data="hello",
        )
        # Downstream parsing may reject non-JSON, but the allowlist and budget must not.
        assert r.status_code not in (401, 403, 402), r.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_json_body_skips_model_enforcement_postgres(test_db_pool, monkeypatch):
    """Invalid JSON should not trigger model allowlist enforcement; ensure not 403/402."""
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_db_handling
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as reset_auth_settings
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app

    # Ensure multi-user mode for AuthNZ
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_auth_settings()

    pool = test_db_pool
    app_settings['CSRF_ENABLED'] = False

    # Insert user
    user_id = await ensure_test_user(pool, "vkpg-badjson", "vkpg-badjson@example.com")

    mgr = APIKeyManager(pool)
    await mgr.initialize()

    async def _get_mgr_override():
        return mgr

    monkeypatch.setattr(auth_deps, "get_api_key_manager", _get_mgr_override)
    monkeypatch.setattr(user_db_handling, "get_api_key_manager", _get_mgr_override)
    res = await mgr.create_virtual_key(
        user_id=user_id,
        name="vk-allowlist-badjson",
        allowed_endpoints=["chat.completions"],
        allowed_providers=["openai"],
        allowed_models=["gpt-4o-mini"],
        budget_day_tokens=100000,
    )
    vkey = res['key']

    async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://testserver") as client:
        r = await client.post(
            "/api/v1/chat/completions",
            headers={"X-API-KEY": vkey, "Content-Type": "application/json"},
            data="this is not json",
        )
        # A malformed body may fail downstream, but it must not trigger these guards.
        assert r.status_code not in (401, 403, 402), r.text

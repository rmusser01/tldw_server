import os

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as emb_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


def _make_app():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(emb_mod.router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_tenant_quotas_respects_profile_flag_over_mode(monkeypatch):
    """
    Ensure get_tenant_quotas uses PROFILE when set, but preserves the existing
    single-user AUTH_MODE behavior when PROFILE is unset.
    """
    # Force multi_user mode but PROFILE=single_user to simulate a desktop profile.
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("PROFILE", "single_user")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS", "5")
    reset_settings()

    # Provide a simple get_request_user override so the dependency succeeds.
    async def _fake_get_request_user():
        return User(id=1, username="tester", is_active=True)

    from fastapi import FastAPI

    app = _make_app()
    app.dependency_overrides[get_request_user] = _fake_get_request_user

    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/tenant/quotas")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    # PROFILE=single_user should disable tenant quotas regardless of AUTH_MODE.
    assert data["limit_rps"] == 0
    assert data["remaining"] is None


@pytest.mark.asyncio
async def test_tenant_quotas_falls_back_to_mode_when_profile_unset(monkeypatch):
    """
    When PROFILE is unset, behavior should match the legacy mode-based guard.
    """
    # Explicitly clear PROFILE and drive via AUTH_MODE.
    monkeypatch.delenv("PROFILE", raising=False)
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS", "5")
    reset_settings()

    async def _fake_get_request_user():
        return User(id=1, username="tester", is_active=True)

    app = _make_app()
    app.dependency_overrides[get_request_user] = _fake_get_request_user

    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/tenant/quotas")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Legacy behavior: single_user AUTH_MODE disables tenant quotas.
    assert data["limit_rps"] == 0
    assert data["remaining"] is None


@pytest.mark.asyncio
async def test_tenant_quotas_local_single_user_profile(monkeypatch):
    """
    Explicit PROFILE=local-single-user should behave like single-user mode for
    tenant quotas, even when AUTH_MODE is multi_user.
    """
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("PROFILE", "local-single-user")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS", "5")
    reset_settings()

    async def _fake_get_request_user():
        return User(id=1, username="tester", is_active=True)

    app = _make_app()
    app.dependency_overrides[get_request_user] = _fake_get_request_user

    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/tenant/quotas")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    # PROFILE=local-single-user should disable tenant quotas.
    assert data["limit_rps"] == 0
    assert data["remaining"] is None


@pytest.mark.asyncio
async def test_tenant_quota_helper_flag_disabled_preserves_legacy_mode(monkeypatch):
    """
    When EMBEDDINGS_TENANT_RPS_PROFILE_AWARE is disabled/unset, the helper
    should fall back to the legacy mode/profile guard via _is_multi_user_runtime().
    """
    # Explicitly disable the flag and drive AUTH_MODE.
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS_PROFILE_AWARE", "0")
    monkeypatch.delenv("PROFILE", raising=False)
    reset_settings()

    scope = {"type": "http", "headers": []}
    request = Request(scope)

    # Helper should mirror the legacy multi-user detection.
    assert emb_mod._should_enforce_tenant_rps(request) is emb_mod._is_multi_user_runtime()


@pytest.mark.asyncio
async def test_tenant_quota_helper_flag_enabled_single_user_profile(monkeypatch):
    """
    With the flag enabled, single-user profiles should disable tenant quotas
    even when AUTH_MODE is multi_user.
    """
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("PROFILE", "single_user")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS_PROFILE_AWARE", "1")
    reset_settings()

    scope = {"type": "http", "headers": []}
    request = Request(scope)

    assert emb_mod._should_enforce_tenant_rps(request) is False


@pytest.mark.asyncio
async def test_tenant_quota_helper_flag_enabled_single_user_principal(monkeypatch):
    """
    With the flag enabled and a multi-user profile, an explicit single-user
    principal should still disable tenant quotas.
    """
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("PROFILE", "multi-user-postgres")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS_PROFILE_AWARE", "1")
    reset_settings()

    scope = {"type": "http", "headers": []}
    request = Request(scope)

    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="single_user",
        token_type="api_key",
        jti=None,
        roles=["admin"],
        permissions=["*"],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )
    request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)

    assert emb_mod._should_enforce_tenant_rps(request) is False


@pytest.mark.asyncio
async def test_tenant_quota_helper_flag_enabled_multi_user_principal(monkeypatch):
    """
    With the flag enabled and a multi-user profile, a regular principal should
    keep tenant quotas enabled.
    """
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("PROFILE", "multi-user-postgres")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS_PROFILE_AWARE", "1")
    reset_settings()

    scope = {"type": "http", "headers": []}
    request = Request(scope)

    principal = AuthPrincipal(
        kind="user",
        user_id=123,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["user"],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )
    request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)

    assert emb_mod._should_enforce_tenant_rps(request) is True


@pytest.mark.asyncio
async def test_tenant_quotas_flag_enabled_single_user_principal_http(monkeypatch):
    """
    HTTP-level: with the flag enabled and a multi-user profile, a single-user
    principal should see tenant quotas disabled (limit_rps=0).
    """
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("PROFILE", "multi-user-postgres")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS_PROFILE_AWARE", "1")
    monkeypatch.setattr(emb_mod, "TENANT_RPS", 5, raising=False)
    reset_settings()

    # Stub Redis so the endpoint can call into it if needed.
    class _FakeRedis:
        async def get(self, *_args, **_kwargs):
            return None

    async def _fake_get_redis_client():
        return _FakeRedis()

    async def _fake_ensure_async_client_closed(_client):
        return None

    monkeypatch.setattr(emb_mod, "_get_redis_client", _fake_get_redis_client)
    monkeypatch.setattr("tldw_Server_API.app.core.Infrastructure.redis_factory.ensure_async_client_closed", _fake_ensure_async_client_closed)

    app = _make_app()

    async def _fake_get_request_user(request: Request) -> User:
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="single_user",
            token_type="api_key",
            jti=None,
            roles=["admin"],
            permissions=["*"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return User(id=1, username="tester", is_active=True)

    app.dependency_overrides[get_request_user] = _fake_get_request_user

    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/tenant/quotas")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["limit_rps"] == 0
    assert data["remaining"] is None


@pytest.mark.asyncio
async def test_tenant_quotas_flag_enabled_multi_user_principal_http(monkeypatch):
    """
    HTTP-level: with the flag enabled and a multi-user profile, a regular
    multi-user principal should see tenant quotas enabled (limit_rps>0).
    """
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("PROFILE", "multi-user-postgres")
    monkeypatch.setenv("EMBEDDINGS_TENANT_RPS_PROFILE_AWARE", "1")
    monkeypatch.setattr(emb_mod, "TENANT_RPS", 5, raising=False)
    reset_settings()

    class _FakeRedis:
        async def get(self, *_args, **_kwargs):
            return None

    async def _fake_get_redis_client():
        return _FakeRedis()

    async def _fake_ensure_async_client_closed(_client):
        return None

    monkeypatch.setattr(emb_mod, "_get_redis_client", _fake_get_redis_client)
    monkeypatch.setattr("tldw_Server_API.app.core.Infrastructure.redis_factory.ensure_async_client_closed", _fake_ensure_async_client_closed)

    app = _make_app()

    async def _fake_get_request_user(request: Request) -> User:
        principal = AuthPrincipal(
            kind="user",
            user_id=123,
            api_key_id=None,
            subject=None,
            token_type="access",
            jti=None,
            roles=["user"],
            permissions=[],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return User(id=123, username="tenant-user", is_active=True)

    app.dependency_overrides[get_request_user] = _fake_get_request_user

    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/tenant/quotas")

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["limit_rps"] == 5
    # remaining is dependent on Redis state but should be non-negative.
    assert isinstance(data["remaining"], int)

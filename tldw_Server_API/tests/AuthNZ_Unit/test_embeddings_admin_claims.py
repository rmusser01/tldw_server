from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as embeddings_mod
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE, EMBEDDINGS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(
    *,
    is_admin: bool,
    roles: Optional[list[str]] = None,
    permissions: Optional[list[str]] = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=permissions or [],
        is_admin=is_admin,
        org_ids=[1],
        team_ids=[],
    )


def _build_app(principal: Optional[AuthPrincipal], *, fail_with_401: bool = False) -> FastAPI:
    app = FastAPI()
    app.include_router(embeddings_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    async def _fake_get_request_user() -> User:
        return User(
            id=1,
            username="admin",
            is_active=True,
            roles=list(principal.roles) if principal else [],
            permissions=list(principal.permissions) if principal else [],
            is_admin=bool(principal.is_admin) if principal else False,
        )

    app.dependency_overrides[embeddings_mod.get_request_user] = _fake_get_request_user
    return app


@pytest.mark.asyncio
async def test_embeddings_metrics_401_when_principal_missing():
    app = _build_app(principal=None, fail_with_401=True)
    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/metrics")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_embeddings_metrics_403_without_permission():
    principal = _make_principal(is_admin=False, roles=["user"], permissions=[])
    app = _build_app(principal=principal)
    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/metrics")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_embeddings_metrics_200_for_admin_with_permission():
    principal = _make_principal(is_admin=True, roles=["admin"], permissions=[SYSTEM_CONFIGURE])
    app = _build_app(principal=principal)
    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/metrics")
    assert resp.status_code == 200
    assert isinstance(resp.json(), dict)


@pytest.mark.asyncio
async def test_embeddings_stage_status_403_without_embeddings_admin_permission():
    principal = _make_principal(is_admin=False, roles=["user"], permissions=[])
    app = _build_app(principal=principal)
    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/stage/status")
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_embeddings_stage_status_200_with_embeddings_admin_permission_non_admin(monkeypatch):
    principal = _make_principal(is_admin=False, roles=["user"], permissions=[EMBEDDINGS_ADMIN])
    app = _build_app(principal=principal)

    class _FakeRedis:
        async def get(self, *_args, **_kwargs):
            return None

    async def _fake_get_redis_client():
        return _FakeRedis()

    async def _fake_ensure_async_client_closed(_client):
        return None

    monkeypatch.setattr(embeddings_mod, "_get_redis_client", _fake_get_redis_client)
    monkeypatch.setattr("tldw_Server_API.app.core.Infrastructure.redis_factory.ensure_async_client_closed", _fake_ensure_async_client_closed)

    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/stage/status")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)

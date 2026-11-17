from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as emb_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.permissions import EMBEDDINGS_ADMIN


def _make_principal(
    *,
    kind: str = "user",
    is_admin: bool = False,
    roles: Optional[list[str]] = None,
    permissions: Optional[list[str]] = None,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind=kind,
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=permissions or [],
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


def _build_app_with_overrides(
    principal: Optional[AuthPrincipal],
    *,
    fail_with_401: bool = False,
) -> FastAPI:
    app = FastAPI()
    app.include_router(emb_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
        ip = request.client.host if getattr(request, "client", None) else None
        ua = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
        request_id = request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
        request.state.auth = AuthContext(
            principal=principal,
            ip=ip,
            user_agent=ua,
            request_id=request_id,
        )
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    async def _fake_get_request_user():
        return SimpleNamespace(
            id=principal.user_id if principal else 1,
            username="emb-user",
            is_active=True,
            roles=list(principal.roles) if principal else [],
            permissions=list(principal.permissions) if principal else [],
            is_admin=bool(principal.is_admin) if principal else False,
        )

    app.dependency_overrides[emb_mod.get_request_user] = _fake_get_request_user
    return app


@pytest.mark.unit
def test_embeddings_stage_status_401_when_principal_unavailable():
    app = _build_app_with_overrides(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/stage/status")

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.unit
def test_embeddings_stage_status_403_without_embeddings_admin_permission():
    principal = _make_principal(is_admin=False, roles=["user"], permissions=[])
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/stage/status")

    assert resp.status_code == 403


@pytest.mark.unit
def test_embeddings_stage_status_200_with_embeddings_admin_permission(monkeypatch):
    principal = _make_principal(is_admin=False, roles=["user"], permissions=[EMBEDDINGS_ADMIN])
    app = _build_app_with_overrides(principal=principal)

    class _FakeRedis:
        async def get(self, *_args, **_kwargs):
            return None

    async def _fake_get_redis_client():
        return _FakeRedis()

    async def _fake_ensure_async_client_closed(_client):
        return None

    monkeypatch.setattr(emb_mod, "_get_redis_client", _fake_get_redis_client)
    monkeypatch.setattr("tldw_Server_API.app.core.Infrastructure.redis_factory.ensure_async_client_closed", _fake_ensure_async_client_closed)

    with TestClient(app) as client:
        resp = client.get("/api/v1/embeddings/stage/status")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, dict)

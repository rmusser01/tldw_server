from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.main import app as _app
from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import resource_governor as rg_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


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
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_with_401: bool = False,
) -> FastAPI:
    app = FastAPI()
    app.include_router(rg_mod.router, prefix="/api/v1")

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

    # Stub governor to keep diagnostics endpoints fast and deterministic
    class _FakeGovernor:
        def capabilities(self):
            return {"backend": "stub"}

    fake_governor = _FakeGovernor()
    for target_app in (_app, rg_mod._get_app()):
        monkeypatch.setattr(target_app.state, "rg_governor", fake_governor, raising=False)

    return app


@pytest.mark.asyncio
async def test_rg_diag_capabilities_401_when_principal_unavailable(monkeypatch):
    app = _build_app_with_overrides(
        principal=None,
        monkeypatch=monkeypatch,
        fail_with_401=True,
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/resource-governor/diag/capabilities")

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.asyncio
async def test_rg_diag_capabilities_403_when_missing_admin_role(monkeypatch):
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal, monkeypatch=monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/api/v1/resource-governor/diag/capabilities")

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_rg_diag_capabilities_200_for_admin_principal(monkeypatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal, monkeypatch=monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/api/v1/resource-governor/diag/capabilities")

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"
    assert isinstance(body.get("capabilities"), dict)

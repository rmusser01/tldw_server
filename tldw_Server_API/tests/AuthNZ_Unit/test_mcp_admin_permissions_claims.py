from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as mcp_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Utils.app_services import app_services


def _build_app_with_overrides(
    principal: Optional[AuthPrincipal],
    *,
    fail_with_401: bool = False,
    monkeypatch: pytest.MonkeyPatch,
) -> FastAPI:
    app = FastAPI()
    app.include_router(mcp_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    async def _no_op_guard() -> None:
        return None

    app.dependency_overrides[mcp_mod.enforce_http_security] = _no_op_guard

    class _DummyServer:
        def __init__(self) -> None:
            self.initialized = False

        async def initialize(self) -> None:
            self.initialized = True

        async def handle_http_request(self, *_args: Any, **_kwargs: Any):
            class _Resp:
                error = None
                result = {"status": "ok"}

            return _Resp()

        async def get_metrics(self) -> Dict[str, Any]:
            return {"connections": {}, "modules": {}}

    monkeypatch.setattr(app_services(), "mcp_server", _DummyServer())

    return app


def _make_principal(
    *,
    is_admin: bool = False,
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
        org_ids=[],
        team_ids=[],
    )


@pytest.mark.asyncio
async def test_mcp_modules_health_401_when_principal_unavailable(monkeypatch: pytest.MonkeyPatch):
    app = _build_app_with_overrides(principal=None, fail_with_401=True, monkeypatch=monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/modules/health")

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.asyncio
async def test_mcp_modules_health_403_when_missing_system_logs_permission(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal, monkeypatch=monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/modules/health")

    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    assert SYSTEM_LOGS in detail


@pytest.mark.asyncio
async def test_mcp_modules_health_200_for_admin_principal(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal, monkeypatch=monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/modules/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"


@pytest.mark.asyncio
async def test_mcp_metrics_401_when_principal_unavailable(monkeypatch: pytest.MonkeyPatch):
    app = _build_app_with_overrides(principal=None, fail_with_401=True, monkeypatch=monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/metrics")

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_mcp_metrics_403_when_missing_system_logs_permission(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal, monkeypatch=monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/metrics")

    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    assert SYSTEM_LOGS in detail


@pytest.mark.asyncio
async def test_mcp_metrics_200_for_principal_with_system_logs_permission(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[SYSTEM_LOGS],
    )
    app = _build_app_with_overrides(principal=principal, monkeypatch=monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/api/v1/mcp/metrics")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("modules"), dict)

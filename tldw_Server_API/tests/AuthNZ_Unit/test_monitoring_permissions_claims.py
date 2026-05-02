from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


def _build_app_with_overrides(
    principal: Optional[AuthPrincipal],
    *,
    fail_with_401: bool = False,
) -> FastAPI:
    app = FastAPI()
    app.include_router(monitoring_mod.router, prefix="/api/v1")

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

    # Stub topic monitoring service to avoid touching real DBs
    class _FakeMonitoringService:
        def list_watchlists(self) -> list[dict]:
            return []

    def _fake_get_topic_monitoring_service() -> _FakeMonitoringService:

        return _FakeMonitoringService()

    monitoring_mod.get_topic_monitoring_service = _fake_get_topic_monitoring_service  # type: ignore[assignment]

    return app


def _build_admin_app_with_overrides(principal: AuthPrincipal) -> FastAPI:
    app = FastAPI()
    app.include_router(admin_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        del request
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
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
async def test_monitoring_watchlists_401_when_principal_unavailable():
    app = _build_app_with_overrides(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        resp = client.get("/api/v1/monitoring/watchlists")

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.asyncio
async def test_monitoring_watchlists_403_when_missing_system_logs_permission():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/monitoring/watchlists")

    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    assert SYSTEM_LOGS in detail


@pytest.mark.asyncio
async def test_monitoring_watchlists_200_for_admin_principal():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/monitoring/watchlists")

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("watchlists") == []


@pytest.mark.asyncio
async def test_monitoring_watchlists_200_for_system_logs_principal():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[SYSTEM_LOGS],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/monitoring/watchlists")

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("watchlists") == []


@pytest.mark.asyncio
async def test_monitoring_notification_settings_403_when_missing_system_logs_permission():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/monitoring/notifications/settings")

    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    assert SYSTEM_LOGS in detail


@pytest.mark.asyncio
async def test_admin_monitoring_routes_reject_system_logs_without_admin_role():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[SYSTEM_LOGS],
    )
    app = _build_admin_app_with_overrides(principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/admin/monitoring/alert-rules")

    assert resp.status_code == 403
    assert "Required role" in resp.json().get("detail", "")

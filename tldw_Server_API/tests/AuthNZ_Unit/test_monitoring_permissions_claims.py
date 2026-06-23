from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod
from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE, SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest.fixture(autouse=True)
def _single_user_test_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "monitoring-test-key")
    reset_settings()
    yield
    reset_settings()


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

    class _FakeNotificationService:
        def get_settings(self) -> dict:
            return {
                "enabled": False,
                "min_severity": "critical",
                "file": "Databases/monitoring_notifications.log",
                "webhook_url": "",
                "email_to": "",
                "smtp_host": "",
                "smtp_port": 587,
                "smtp_starttls": True,
                "smtp_user": "",
                "email_from": "",
            }

        def update_settings(self, **kwargs) -> dict:
            settings = self.get_settings()
            settings.update(kwargs)
            return settings

        def is_file_path_allowed(self, _path: str) -> bool:
            return True

        def notify(self, _alert) -> str:
            return "logged"

    monitoring_mod.get_notification_service = lambda: _FakeNotificationService()  # type: ignore[assignment]

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
async def test_monitoring_notification_settings_update_403_for_system_logs_only():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[SYSTEM_LOGS],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/monitoring/notifications/settings",
            json={"enabled": True},
        )

    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    assert SYSTEM_CONFIGURE in detail


@pytest.mark.asyncio
async def test_monitoring_notification_settings_update_200_with_system_configure():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[SYSTEM_LOGS, SYSTEM_CONFIGURE],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/monitoring/notifications/settings",
            json={"enabled": True},
        )

    assert resp.status_code == 200
    assert resp.json().get("enabled") is True


@pytest.mark.asyncio
async def test_monitoring_notification_test_403_for_system_logs_only():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[SYSTEM_LOGS],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/monitoring/notifications/test",
            json={"message": "probe", "severity": "critical"},
        )

    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    assert SYSTEM_CONFIGURE in detail


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

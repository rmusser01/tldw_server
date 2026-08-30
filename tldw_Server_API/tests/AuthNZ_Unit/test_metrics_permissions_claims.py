from __future__ import annotations

from typing import Any, Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import metrics as metrics_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

DETAILED_METRICS_PATHS = (
    "/metrics",
    "/api/v1/metrics",
    "/api/v1/metrics/text",
    "/api/v1/metrics/json",
    "/api/v1/metrics/health",
    "/api/v1/metrics/chat",
)


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
    app.include_router(metrics_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None, "Test setup error: principal must be provided when fail_with_401 is False"
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

    return app


def _main_app_with_override(principal: Optional[AuthPrincipal], *, fail_with_401: bool = False) -> FastAPI:
    from tldw_Server_API.app import main

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
        if fail_with_401:
            raise HTTPException(status_code=401, detail="Authentication required")
        assert principal is not None
        return principal

    main.app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    return main.app


@pytest.mark.unit
@pytest.mark.parametrize("path", DETAILED_METRICS_PATHS)
@pytest.mark.parametrize(
    ("principal", "fail_with_401", "expected"),
    ((None, True, 401), (_make_principal(permissions=[]), False, 403)),
)
def test_detailed_metrics_require_system_logs(
    path: str,
    principal: Optional[AuthPrincipal],
    fail_with_401: bool,
    expected: int,
):
    with TestClient(_main_app_with_override(principal, fail_with_401=fail_with_401)) as client:
        response = client.get(path)
    assert response.status_code == expected


@pytest.mark.unit
@pytest.mark.parametrize("path", DETAILED_METRICS_PATHS)
@pytest.mark.parametrize(
    "principal",
    (_make_principal(permissions=[SYSTEM_LOGS]), _make_principal(is_admin=True, roles=["admin"], permissions=[])),
)
def test_detailed_metrics_allow_permission_or_admin_bypass(path: str, principal: AuthPrincipal):
    with TestClient(_main_app_with_override(principal)) as client:
        response = client.get(path)
    assert response.status_code not in {401, 403}


@pytest.mark.unit
def test_metrics_reset_401_when_principal_unavailable():
    app = _build_app_with_overrides(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        resp = client.post("/api/v1/metrics/reset")

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.unit
def test_metrics_reset_403_when_missing_admin_role():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.post("/api/v1/metrics/reset")

    assert resp.status_code == 403


@pytest.mark.unit
def test_metrics_reset_200_for_admin_principal():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        resp = client.post("/api/v1/metrics/reset")

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "success"
    assert "message" in body


@pytest.mark.unit
def test_metrics_reset_invokes_chat_reset_hook_for_admin(monkeypatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    class StubRegistry:
        def __init__(self):
            self.was_reset = False

        def reset(self):
            self.was_reset = True

    class StubChatMetrics:
        def __init__(self):
            self.was_reset = False

        def reset_active_metrics(self):
            self.was_reset = True

    stub_registry = StubRegistry()
    stub_chat = StubChatMetrics()

    monkeypatch.setattr(metrics_mod, "get_metrics_registry", lambda: stub_registry)
    monkeypatch.setattr(metrics_mod, "get_chat_metrics", lambda: stub_chat)

    with TestClient(app) as client:
        resp = client.post("/api/v1/metrics/reset")

    assert resp.status_code == 200
    assert stub_registry.was_reset is True
    assert stub_chat.was_reset is True

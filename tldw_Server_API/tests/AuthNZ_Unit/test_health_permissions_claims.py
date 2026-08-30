from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import health
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

DETAILED_HEALTH_PATHS = (
    "/ready",
    "/health/ready",
    "/api/v1/healthz",
    "/api/v1/readyz",
    "/api/v1/health",
    "/api/v1/health/live",
    "/api/v1/health/ready",
    "/api/v1/health/metrics",
    "/api/v1/health/security",
)


def _principal(*, permissions: list[str], is_admin: bool = False) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="test-operator",
        token_type="access",
        jti=None,
        roles=["admin"] if is_admin else ["user"],
        permissions=permissions,
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


def _app(principal: Optional[AuthPrincipal]) -> FastAPI:
    from tldw_Server_API.app import main

    app = main.app

    async def _auth(request: Request) -> AuthPrincipal:
        if principal is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _auth
    return app


@pytest.mark.parametrize("path", DETAILED_HEALTH_PATHS)
@pytest.mark.parametrize(
    ("principal", "expected"),
    (
        (None, 401),
        (_principal(permissions=[]), 403),
    ),
)
def test_detailed_health_requires_system_logs(path: str, principal: Optional[AuthPrincipal], expected: int) -> None:
    with TestClient(_app(principal)) as client:
        response = client.get(path)
    assert response.status_code == expected


@pytest.mark.parametrize("path", DETAILED_HEALTH_PATHS)
@pytest.mark.parametrize("principal", (_principal(permissions=[SYSTEM_LOGS]), _principal(permissions=[], is_admin=True)))
def test_detailed_health_allows_permission_or_admin_bypass(path: str, principal: AuthPrincipal) -> None:
    with TestClient(_app(principal)) as client:
        response = client.get(path)
    assert response.status_code not in {401, 403}

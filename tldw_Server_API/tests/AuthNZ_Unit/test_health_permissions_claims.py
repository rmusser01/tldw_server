from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
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


@contextmanager
def _claim_client(principal: AuthPrincipal | None) -> Iterator[TestClient]:
    from tldw_Server_API.app import main

    app = main.app
    original_overrides = dict(app.dependency_overrides)

    async def _auth(request: Request) -> AuthPrincipal:
        if principal is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _auth
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)


@pytest.mark.parametrize("path", DETAILED_HEALTH_PATHS)
@pytest.mark.parametrize(
    ("principal", "expected"),
    (
        (None, 401),
        (_principal(permissions=[]), 403),
    ),
)
def test_detailed_health_requires_system_logs(path: str, principal: AuthPrincipal | None, expected: int) -> None:
    with _claim_client(principal) as client:
        response = client.get(path)
    assert response.status_code == expected


@pytest.mark.parametrize("path", DETAILED_HEALTH_PATHS)
@pytest.mark.parametrize(
    "principal", (_principal(permissions=[SYSTEM_LOGS]), _principal(permissions=[], is_admin=True))
)
def test_detailed_health_allows_permission_or_admin_bypass(path: str, principal: AuthPrincipal) -> None:
    with _claim_client(principal) as client:
        response = client.get(path)
    assert response.status_code not in {401, 403}


def test_health_claim_client_restores_dependency_overrides_after_each_use() -> None:
    from tldw_Server_API.app import main

    original = dict(main.app.dependency_overrides)
    for principal in (
        _principal(permissions=[]),
        _principal(permissions=[SYSTEM_LOGS]),
    ):
        with _claim_client(principal):
            assert auth_deps.get_auth_principal in main.app.dependency_overrides
        assert main.app.dependency_overrides == original

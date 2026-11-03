"""
Integration tests covering single-user claims/permissions wiring.

These tests validate that the single-user bootstrap populates admin claims and
permissions, and that claim-first dependencies (`require_permissions`,
`require_roles`, `get_auth_principal`) respect the single-user API key across
protected routes. Assumes pytest integration marker and the shared
`isolated_test_environment` fixture for DB setup/teardown.
"""

from typing import Any, Dict

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient
from loguru import logger
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings, get_settings
from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    require_permissions,
    require_roles,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


router = APIRouter()


def _create_limited_principal_override(settings, request_id: str):
    async def _override(request: Request):
        principal = AuthPrincipal(
            kind="user",
            user_id=settings.SINGLE_USER_FIXED_ID,
            api_key_id=None,
            subject="single_user",
            token_type="api_key",
            jti=None,
            roles=["user"],
            permissions=[],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )
        try:
            request.state.auth = AuthContext(
                principal=principal,
                ip="127.0.0.1",
                user_agent="pytest-agent",
                request_id=request_id,
            )
        except Exception as exc:
            logger.debug(f"Could not set request.state.auth: {exc}")
        return principal

    return _override


@router.get("/single-user/whoami/user")
async def whoami_user(
    user: User = Depends(get_request_user),
) -> Dict[str, Any]:
    """Return the single-user profile details (id, roles, permissions, is_admin) for the injected User."""
    return {
        "id": user.id,
        "roles": list(user.roles or []),
        "permissions": list(user.permissions or []),
        "is_admin": bool(getattr(user, "is_admin", False)),
    }


@router.get("/single-user/protected/perm")
async def protected_by_permission(
    principal: AuthPrincipal = Depends(require_permissions("media.read")),
) -> Dict[str, Any]:
    """Verify permission-protected endpoint returns principal_id/kind/roles/permissions/is_admin from AuthPrincipal."""
    return {
        "principal_id": principal.principal_id,
        "kind": principal.kind,
        "roles": principal.roles,
        "permissions": principal.permissions,
        "is_admin": principal.is_admin,
    }


@router.get("/single-user/protected/role")
async def protected_by_role(
    principal: AuthPrincipal = Depends(require_roles("admin")),
) -> Dict[str, Any]:
    """Verify role-protected endpoint returns principal_id/kind/roles/permissions/is_admin from AuthPrincipal."""
    return {
        "principal_id": principal.principal_id,
        "kind": principal.kind,
        "roles": principal.roles,
        "permissions": principal.permissions,
        "is_admin": principal.is_admin,
    }


def _build_single_user_app() -> FastAPI:


    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.mark.asyncio
async def test_single_user_bootstrapped_admin_uses_claims_for_permissions(
    isolated_test_environment, monkeypatch, fresh_services
):
    """Verify single-user bootstrapped admin uses claims for permissions."""
    _client, _db_name = isolated_test_environment
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_claims_key_123")
    monkeypatch.setenv("TEST_MODE", "true")

    # Ensure settings and DB pool see the single-user profile and isolated DB
    reset_settings()

    # Seed the single-user admin and primary key via the shared bootstrap helper
    ok_first = await bootstrap_single_user_profile()
    ok_second = await bootstrap_single_user_profile()
    assert ok_first is True
    assert ok_second is True

    settings = get_settings()
    single_user_key = settings.SINGLE_USER_API_KEY

    app = _build_single_user_app()
    client = TestClient(app)

    # Fetch the bootstrapped admin as a User via get_request_user
    resp_user = client.get(
        "/api/v1/single-user/whoami/user",
        headers={"X-API-KEY": single_user_key},
    )
    assert resp_user.status_code == 200
    user_payload = resp_user.json()
    assert user_payload["id"] == settings.SINGLE_USER_FIXED_ID
    assert "admin" in user_payload["roles"]
    # Single-user path exposes concrete permissions claims
    assert "media.read" in user_payload["permissions"]
    assert user_payload["is_admin"] is True

    # The same key should succeed on permission- and role-protected endpoints
    resp_perm = client.get(
        "/api/v1/single-user/protected/perm",
        headers={"X-API-KEY": single_user_key},
    )
    assert resp_perm.status_code == 200
    perm_payload = resp_perm.json()
    assert perm_payload["kind"] == "user"
    assert perm_payload["is_admin"] is True
    assert "media.read" in perm_payload["permissions"]

    resp_role = client.get(
        "/api/v1/single-user/protected/role",
        headers={"X-API-KEY": single_user_key},
    )
    assert resp_role.status_code == 200
    role_payload = resp_role.json()
    assert role_payload["kind"] == "user"
    assert role_payload["is_admin"] is True
    assert "admin" in role_payload["roles"]


@pytest.mark.asyncio
async def test_single_user_non_admin_principal_denied_on_role(
    isolated_test_environment, monkeypatch, fresh_services
):
    """Ensure non-admin single-user principal is denied role-based access."""
    # Reuse the same single-user bootstrap flow but override get_auth_principal
    # to simulate a principal with limited claims.
    _client, _db_name = isolated_test_environment
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_claims_key_limited")
    monkeypatch.setenv("TEST_MODE", "true")

    reset_settings()

    ok_first = await bootstrap_single_user_profile()
    ok_second = await bootstrap_single_user_profile()
    assert ok_first is True
    assert ok_second is True

    settings = get_settings()
    single_user_key = settings.SINGLE_USER_API_KEY

    app = _build_single_user_app()
    app.dependency_overrides[get_auth_principal] = _create_limited_principal_override(
        settings, "single-user-limited"
    )
    client = TestClient(app)

    resp_role = client.get(
        "/api/v1/single-user/protected/role",
        headers={"X-API-KEY": single_user_key},
    )
    assert resp_role.status_code == 403


@pytest.mark.asyncio
async def test_single_user_non_admin_principal_denied_on_permission(
    isolated_test_environment, monkeypatch, fresh_services
):
    """Ensure non-admin single-user principal is denied permission-based access."""
    # Same setup as above, but exercise permission-based gating.
    _client, _db_name = isolated_test_environment
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_claims_key_limited_perm")
    monkeypatch.setenv("TEST_MODE", "true")

    reset_settings()

    ok_first = await bootstrap_single_user_profile()
    ok_second = await bootstrap_single_user_profile()
    assert ok_first is True
    assert ok_second is True

    settings = get_settings()
    single_user_key = settings.SINGLE_USER_API_KEY

    app = _build_single_user_app()
    app.dependency_overrides[get_auth_principal] = _create_limited_principal_override(
        settings, "single-user-limited-perm"
    )
    client = TestClient(app)

    resp_perm = client.get(
        "/api/v1/single-user/protected/perm",
        headers={"X-API-KEY": single_user_key},
    )
    assert resp_perm.status_code == 403

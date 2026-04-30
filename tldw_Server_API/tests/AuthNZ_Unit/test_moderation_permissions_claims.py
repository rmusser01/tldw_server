from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import moderation as moderation_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
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


def _build_app(
    principal: Optional[AuthPrincipal],
    *,
    fail_with_401: bool = False,
    moderation_service: object | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(moderation_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
        request.state.auth = AuthContext(
            principal=principal,
            ip=None,
            user_agent=None,
            request_id=None,
        )
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    class _StubModerationService:
        def list_user_overrides(self) -> dict:
            return {}

        def update_settings(self, **_: object) -> dict:
            return {
                "pii_enabled": None,
                "categories_enabled": None,
                "effective": {
                    "pii_enabled": False,
                    "categories_enabled": [],
                },
            }

    service = moderation_service or _StubModerationService()
    moderation_mod.get_moderation_service = lambda: service  # type: ignore[assignment]

    return app


def test_moderation_router_uses_standard_auth_factory_aliases() -> None:
    assert moderation_mod.RequireRole is auth_deps.RequireRole
    assert moderation_mod.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(moderation_mod, "require_roles")
    assert not hasattr(moderation_mod, "require_permissions")


@pytest.mark.asyncio
async def test_moderation_users_401_when_principal_unavailable():
    app = _build_app(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/users")

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.asyncio
async def test_moderation_users_403_without_admin_or_permission():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/users")

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_moderation_users_200_for_admin_with_permission():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
    )
    app = _build_app(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/users")

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("overrides") == {}


@pytest.mark.asyncio
async def test_moderation_settings_put_403_without_admin_or_permission():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app(principal=principal)

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/moderation/settings",
            json={"pii_enabled": True, "persist": True},
        )

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_moderation_settings_put_500_for_runtime_persistence_failure():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
    )

    class _PersistFailModerationService:
        def list_user_overrides(self) -> dict:
            return {}

        def update_settings(self, **_: object) -> dict:
            return {
                "ok": False,
                "persisted": False,
                "error": "disk full",
                "error_type": "persistence",
            }

    app = _build_app(
        principal=principal,
        moderation_service=_PersistFailModerationService(),
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/moderation/settings",
            json={"pii_enabled": True, "persist": True},
        )

    assert resp.status_code == 500
    assert resp.json().get("detail") == "Failed to update moderation settings"


@pytest.mark.asyncio
async def test_moderation_settings_put_400_for_validation_failure():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
    )

    class _ValidationFailModerationService:
        def list_user_overrides(self) -> dict:
            return {}

        def update_settings(self, **_: object) -> dict:
            return {
                "ok": False,
                "persisted": False,
                "error": "categories_enabled contains invalid value",
                "error_type": "validation",
            }

    app = _build_app(
        principal=principal,
        moderation_service=_ValidationFailModerationService(),
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/moderation/settings",
            json={"categories_enabled": ["bad"], "persist": False},
        )

    assert resp.status_code == 400
    assert resp.json().get("detail") == "categories_enabled contains invalid value"

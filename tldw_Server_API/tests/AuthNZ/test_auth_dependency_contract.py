from __future__ import annotations

from typing import Any

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    AdminPrincipal,
    CurrentPrincipal,
    CurrentUserDict,
    RequireApiKeyScope,
    RequirePermission,
    RequireRole,
    ServicePrincipal,
    TokenScopeGuard,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


pytestmark = pytest.mark.unit


def _principal(
    *,
    kind: str = "user",
    user_id: int | None = 42,
    api_key_id: int | None = None,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
    subject: str | None = None,
    token_type: str | None = "access",
    is_admin: bool = False,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind=kind,
        user_id=user_id,
        api_key_id=api_key_id,
        username="phase34-user" if kind == "user" else None,
        subject=subject,
        token_type=token_type,
        roles=list(roles or []),
        permissions=list(permissions or []),
        is_admin=is_admin,
        org_ids=[7] if kind == "user" else [],
        team_ids=[11] if kind == "user" else [],
    )


async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
    principal = request.app.state.principal
    request.state.auth = AuthContext(
        principal=principal,
        ip="127.0.0.1",
        user_agent="phase34-test",
        request_id="phase34-request",
    )
    request.state._auth_user = {
        "id": principal.user_id,
        "roles": list(principal.roles),
        "permissions": list(principal.permissions),
        "is_active": True,
        "is_verified": True,
    }
    request.state.user_id = principal.user_id
    request.state.api_key_id = principal.api_key_id
    request.state.org_ids = list(principal.org_ids)
    request.state.team_ids = list(principal.team_ids)
    return principal


async def _fake_current_active_user() -> dict[str, Any]:
    return {
        "id": 99,
        "roles": ["legacy"],
        "permissions": ["legacy.read"],
        "is_active": True,
        "is_verified": True,
    }


def _build_app(principal: AuthPrincipal) -> FastAPI:
    app = FastAPI()
    app.state.principal = principal
    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.get_current_active_user] = _fake_current_active_user

    @app.get("/current-principal")
    async def current_principal(request: Request, principal: CurrentPrincipal):
        ctx = request.state.auth
        return {
            "principal_id": principal.principal_id,
            "state_principal_id": ctx.principal.principal_id,
            "user_id": principal.user_id,
            "state_user_id": request.state.user_id,
            "org_ids": request.state.org_ids,
            "team_ids": request.state.team_ids,
        }

    @app.get("/current-user-dict")
    async def current_user_dict(user: CurrentUserDict):
        return user

    @app.get("/admin")
    async def admin(principal: AdminPrincipal):
        return {"principal_id": principal.principal_id}

    @app.get("/service")
    async def service(principal: ServicePrincipal):
        return {"principal_id": principal.principal_id, "kind": principal.kind}

    @app.get("/role")
    async def role(principal: AuthPrincipal = Depends(RequireRole("editor", "admin"))):
        return {"roles": principal.roles}

    @app.get("/permission")
    async def permission(
        principal: AuthPrincipal = Depends(RequirePermission("media.read", "skills.read")),
    ):
        return {"permissions": principal.permissions}

    return app


def test_current_principal_alias_returns_principal_and_preserves_request_state() -> None:
    app = _build_app(
        _principal(roles=["user"], permissions=["media.read"], subject="user:42"),
    )

    response = TestClient(app).get("/current-principal")

    assert response.status_code == 200
    assert response.json() == {
        "principal_id": _principal(subject="user:42").principal_id,
        "state_principal_id": _principal(subject="user:42").principal_id,
        "user_id": 42,
        "state_user_id": 42,
        "org_ids": [7],
        "team_ids": [11],
    }


def test_current_user_dict_alias_honors_legacy_active_user_override() -> None:
    response = TestClient(_build_app(_principal())).get("/current-user-dict")

    assert response.status_code == 200
    assert response.json() == {
        "id": 99,
        "roles": ["legacy"],
        "permissions": ["legacy.read"],
        "is_active": True,
        "is_verified": True,
    }


def test_admin_principal_alias_preserves_admin_role_gate() -> None:
    admin_response = TestClient(
        _build_app(_principal(roles=["admin"], is_admin=True)),
    ).get("/admin")
    user_response = TestClient(
        _build_app(_principal(roles=["user"], permissions=[])),
    ).get("/admin")

    assert admin_response.status_code == 200
    assert user_response.status_code == 403


def test_service_principal_alias_preserves_service_gate() -> None:
    service_response = TestClient(
        _build_app(_principal(kind="service", user_id=None, subject="service:worker")),
    ).get("/service")
    user_response = TestClient(_build_app(_principal(kind="user"))).get("/service")

    assert service_response.status_code == 200
    assert service_response.json()["kind"] == "service"
    assert user_response.status_code == 403


def test_role_and_permission_factory_aliases_preserve_existing_semantics() -> None:
    role_response = TestClient(_build_app(_principal(roles=["editor"]))).get("/role")
    permission_response = TestClient(
        _build_app(_principal(permissions=["media.read", "skills.read"])),
    ).get("/permission")
    missing_permission_response = TestClient(
        _build_app(_principal(permissions=["media.read"])),
    ).get("/permission")

    assert role_response.status_code == 200
    assert permission_response.status_code == 200
    assert missing_permission_response.status_code == 403


def test_factory_aliases_are_documented_existing_factories() -> None:
    assert RequireRole is auth_deps.require_roles
    assert RequirePermission is auth_deps.require_permissions
    assert RequireApiKeyScope is auth_deps.require_api_key_scope
    assert TokenScopeGuard is auth_deps.require_token_scope

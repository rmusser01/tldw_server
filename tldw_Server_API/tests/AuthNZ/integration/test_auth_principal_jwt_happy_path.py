from typing import Any

import pytest
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_current_user,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

pytestmark = pytest.mark.integration


def _attach_whoami_router(app: FastAPI) -> None:
    """
    Attach a lightweight whoami endpoint that exercises both get_auth_principal
    and get_current_user in the same request.

    This is wired dynamically inside the test so that the production app
    remains unchanged outside of test contexts.
    """
    router = APIRouter()

    @router.get("/authnz/jwt-happy")
    async def whoami_jwt_happy(
        request: Request,
        principal: AuthPrincipal = Depends(get_auth_principal),
        user: dict[str, Any] = Depends(get_current_user),
        request_user: User = Depends(get_request_user),
    ) -> dict[str, Any]:
        auth_ctx = getattr(request.state, "auth", None)
        if auth_ctx is not None:
            cp = auth_ctx.principal
            state_auth_principal: dict[str, Any] | None = {
                "principal_id": getattr(cp, "principal_id", None),
                "kind": getattr(cp, "kind", None),
                "user_id": getattr(cp, "user_id", None),
                "api_key_id": getattr(cp, "api_key_id", None),
                "roles": getattr(cp, "roles", None),
                "permissions": getattr(cp, "permissions", None),
                "org_ids": getattr(cp, "org_ids", None),
                "team_ids": getattr(cp, "team_ids", None),
                "active_org_id": getattr(cp, "active_org_id", None),
                "active_team_id": getattr(cp, "active_team_id", None),
            }
        else:
            state_auth_principal = None

        return {
            "principal": {
                "principal_id": principal.principal_id,
                "kind": principal.kind,
                "user_id": principal.user_id,
                "api_key_id": principal.api_key_id,
                "roles": principal.roles,
                "permissions": principal.permissions,
                "org_ids": principal.org_ids,
                "team_ids": principal.team_ids,
                "active_org_id": principal.active_org_id,
                "active_team_id": principal.active_team_id,
            },
            "user": {
                "id": user.get("id"),
                "role": user.get("role"),
                "roles": user.get("roles"),
                "permissions": user.get("permissions"),
                "is_active": user.get("is_active"),
                "is_verified": user.get("is_verified"),
            },
            "request_user": {
                "id": getattr(request_user, "id", None),
                "roles": list(getattr(request_user, "roles", []) or []),
                "permissions": list(getattr(request_user, "permissions", []) or []),
                "is_admin": bool(getattr(request_user, "is_admin", False)),
                "org_ids": list(getattr(request_user, "org_ids", []) or []),
                "team_ids": list(getattr(request_user, "team_ids", []) or []),
                "active_org_id": getattr(request_user, "active_org_id", None),
                "active_team_id": getattr(request_user, "active_team_id", None),
            },
            "state": {
                "user_id": getattr(request.state, "user_id", None),
                "api_key_id": getattr(request.state, "api_key_id", None),
                "org_ids": getattr(request.state, "org_ids", None),
                "team_ids": getattr(request.state, "team_ids", None),
                "active_org_id": getattr(request.state, "active_org_id", None),
                "active_team_id": getattr(request.state, "active_team_id", None),
            },
            "state_auth_principal": state_auth_principal,
        }

    # Avoid attaching the router multiple times if the test reuses the app.
    paths = {getattr(r, "path", "") for r in app.router.routes}
    if "/api/v1/authnz/jwt-happy" not in paths:
        app.include_router(router, prefix="/api/v1")


@pytest.mark.asyncio
async def test_multi_user_jwt_happy_path_principal_matches_state(
    isolated_test_environment,
) -> None:
    """
    Full multi-user happy path:

    - Use the real auth endpoints to register and log in a user.
    - Call a protected endpoint that depends on both get_auth_principal and
      get_current_user.
    - Assert that AuthPrincipal mirrors request.state/user data.
    """
    client, _db_name = isolated_test_environment
    assert isinstance(client, TestClient)

    app = client.app
    assert isinstance(app, FastAPI)

    # Ensure the auxiliary whoami route is attached for this app instance.
    _attach_whoami_router(app)

    # Sanity check: we are in multi-user mode for this fixture.
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    assert settings.AUTH_MODE == "multi_user"

    # 1. Register a user
    username = "jwtuser"
    password = "Test@Pass#2024"
    email = "jwtuser@example.com"

    register_response = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": email,
            "password": password,
        },
    )
    if register_response.status_code != 200:
        pytest.fail(
            f"Registration failed: status={register_response.status_code}, "
            f"body={register_response.text}"
        )

    # 2. Login to obtain a JWT access token
    login_response = client.post(
        "/api/v1/auth/login",
        data={
            "username": username,
            "password": password,
        },
    )
    if login_response.status_code != 200:
        pytest.fail(
            f"Login failed: status={login_response.status_code}, "
            f"body={login_response.text}"
        )

    tokens = login_response.json()
    access_token = tokens.get("access_token")
    assert access_token, f"Expected access_token in login response, got: {tokens}"

    # 3. Hit the protected whoami endpoint with the Bearer token
    whoami_response = client.get(
        "/api/v1/authnz/jwt-happy",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert whoami_response.status_code == 200

    payload = whoami_response.json()
    principal = payload["principal"]
    user = payload["user"]
    request_user = payload["request_user"]
    state = payload["state"]
    state_auth_principal = payload["state_auth_principal"]

    # Basic identity consistency checks
    assert principal["kind"] == "user"
    assert principal["user_id"] is not None
    assert str(principal["user_id"]) == str(user["id"])
    assert str(principal["user_id"]) == str(request_user["id"])

    # request.state mirrors the principal's identity
    assert str(state["user_id"]) == str(principal["user_id"])
    assert state["api_key_id"] is None

    # request.state.auth.principal mirrors both principal and request.state
    assert state_auth_principal is not None
    assert state_auth_principal["kind"] == principal["kind"]
    assert str(state_auth_principal["user_id"]) == str(principal["user_id"])
    assert state_auth_principal["api_key_id"] == principal["api_key_id"]
    assert state_auth_principal["org_ids"] == principal["org_ids"]
    assert state_auth_principal["team_ids"] == principal["team_ids"]
    assert state_auth_principal["active_org_id"] == principal["active_org_id"]
    assert state_auth_principal["active_team_id"] == principal["active_team_id"]

    # Org/team membership is mirrored between principal and request.state
    assert state["org_ids"] == principal["org_ids"]
    assert state["team_ids"] == principal["team_ids"]
    assert state["active_org_id"] == principal["active_org_id"]
    assert state["active_team_id"] == principal["active_team_id"]

    # The legacy request-user compatibility object must expose the same
    # post-validation scope used by AuthPrincipal and request.state.
    assert request_user["org_ids"] == principal["org_ids"]
    assert request_user["team_ids"] == principal["team_ids"]
    assert request_user["active_org_id"] == principal["active_org_id"]
    assert request_user["active_team_id"] == principal["active_team_id"]

    # Claims on AuthPrincipal, get_current_user, and get_request_user should be aligned
    assert principal["roles"] == user["roles"] == request_user["roles"]
    assert principal["permissions"] == user["permissions"] == request_user["permissions"]

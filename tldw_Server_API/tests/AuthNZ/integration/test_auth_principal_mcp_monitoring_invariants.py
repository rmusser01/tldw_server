from typing import Any, Dict, Tuple

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext


pytestmark = pytest.mark.integration


def _install_auth_capture(app: FastAPI) -> Tuple[Dict[str, Any], Any]:
    """
    Install a lightweight wrapper around get_auth_principal that records
    principal/state alignment for the last request.

    Mirrors the helper used in media/RAG/tools/evaluations invariant tests.
    """
    captured: Dict[str, Any] = {}
    original_get_auth_principal = auth_deps.get_auth_principal

    async def _capturing_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = await original_get_auth_principal(request)

        captured["principal"] = {
            "principal_id": principal.principal_id,
            "kind": principal.kind,
            "user_id": principal.user_id,
            "api_key_id": principal.api_key_id,
            "roles": list(principal.roles),
            "permissions": list(principal.permissions),
            "org_ids": list(principal.org_ids),
            "team_ids": list(principal.team_ids),
        }
        captured["state"] = {
            "user_id": getattr(request.state, "user_id", None),
            "api_key_id": getattr(request.state, "api_key_id", None),
            "org_ids": getattr(request.state, "org_ids", None),
            "team_ids": getattr(request.state, "team_ids", None),
        }

        ctx = getattr(request.state, "auth", None)
        if isinstance(ctx, AuthContext):
            cp = ctx.principal
            captured["state_auth_principal"] = {
                "principal_id": cp.principal_id,
                "kind": cp.kind,
                "user_id": cp.user_id,
                "api_key_id": cp.api_key_id,
                "roles": list(cp.roles),
                "permissions": list(cp.permissions),
                "org_ids": list(cp.org_ids),
                "team_ids": list(cp.team_ids),
            }
        else:
            captured["state_auth_principal"] = None

        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _capturing_get_auth_principal
    return captured, original_get_auth_principal


def _restore_auth_capture(app: FastAPI, original_get_auth_principal: Any) -> None:
    try:
        app.dependency_overrides.pop(auth_deps.get_auth_principal, None)
    finally:
        auth_deps.get_auth_principal = original_get_auth_principal  # type: ignore[assignment]


def test_mcp_modules_health_jwt_principal_and_state_alignment(isolated_test_environment, monkeypatch):


    """
    Multi-user JWT happy path for MCP modules health:

    - Register and log in a user.
    - Grant system.logs permission.
    - Call /api/v1/mcp/modules/health and assert that request.state.* and
      request.state.auth.principal stay aligned with AuthPrincipal.
    """
    client, db_name = isolated_test_environment
    assert isinstance(client, TestClient)
    _ = db_name  # parity with other invariant tests

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    assert settings.AUTH_MODE == "multi_user"

    # 1. Register and log in via real auth endpoints.
    username = "mcp_invariants_user"
    password = "Str0ngP@ssw0rd!"
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": "mcp_invariants_user@example.com", "password": password},
    )
    assert reg.status_code == 200, reg.text

    login = client.post("/api/v1/auth/login", data={"username": username, "password": password})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Ensure system.logs permission is present via RBAC helper override so
    # the route passes require_permissions(SYSTEM_LOGS) regardless of backend.
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_db_handling

    _orig_get_effective_permissions = user_db_handling.get_effective_permissions

    def _patched_get_effective_permissions(user_id: int):
        perms = _orig_get_effective_permissions(user_id)
        if "system.logs" not in perms:
            perms = list(perms) + ["system.logs"]
        return perms

    monkeypatch.setattr(user_db_handling, "get_effective_permissions", _patched_get_effective_permissions)

    # 3. Stub MCP server to avoid heavy initialization and to ensure a simple, deterministic response.
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as mcp_mod

    class _DummyServer:
        def __init__(self) -> None:
            self.initialized = False

        async def initialize(self) -> None:
            self.initialized = True

        async def handle_http_request(self, *_args: Any, **_kwargs: Any):
            class _Resp:
                error = None
                result = {"status": "ok", "modules": {}}

            return _Resp()

    monkeypatch.setattr(mcp_mod, "get_mcp_server", lambda: _DummyServer())

    # 4. Install the auth capture wrapper and call /api/v1/mcp/modules/health.
    from tldw_Server_API.app.main import app as fastapi_app

    app = fastapi_app
    captured, original = _install_auth_capture(app)
    try:
        resp = client.get("/api/v1/mcp/modules/health", headers=headers)
        assert resp.status_code == 200, resp.text

        principal = captured.get("principal")
        state = captured.get("state")
        state_auth_principal = captured.get("state_auth_principal")

        assert principal is not None
        assert state is not None
        assert state_auth_principal is not None

        # AuthPrincipal kind/user identity.
        assert principal["kind"] == "user"
        assert principal["user_id"] is not None
        assert principal["api_key_id"] is None

        # request.state mirrors principal identity (user_id, no api_key_id).
        assert str(state["user_id"]) == str(principal["user_id"])
        assert state["api_key_id"] is None

        # request.state.auth.principal mirrors principal and state.
        assert state_auth_principal["kind"] == principal["kind"]
        assert str(state_auth_principal["user_id"]) == str(principal["user_id"])
        assert state_auth_principal["api_key_id"] == principal["api_key_id"]
        assert state_auth_principal["org_ids"] == principal["org_ids"]
        assert state_auth_principal["team_ids"] == principal["team_ids"]
        assert state["org_ids"] == principal["org_ids"]
        assert state["team_ids"] == principal["team_ids"]
    finally:
        _restore_auth_capture(app, original)


def test_monitoring_watchlists_jwt_principal_and_state_alignment(isolated_test_environment, monkeypatch):


    """
    Multi-user JWT happy path for monitoring watchlists:

    - Register and log in a user.
    - Grant system.logs permission.
    - Call /api/v1/monitoring/watchlists and assert that request.state.* and
      request.state.auth.principal stay aligned with AuthPrincipal.
    """
    client, db_name = isolated_test_environment
    assert isinstance(client, TestClient)
    _ = db_name  # parity with other invariant tests

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    settings = get_settings()
    assert settings.AUTH_MODE == "multi_user"

    # 1. Register and log in via real auth endpoints.
    username = "monitoring_invariants_user"
    password = "Str0ngP@ssw0rd!"
    reg = client.post(
        "/api/v1/auth/register",
        json={
            "username": username,
            "email": "monitoring_invariants_user@example.com",
            "password": password,
        },
    )
    assert reg.status_code == 200, reg.text

    login = client.post("/api/v1/auth/login", data={"username": username, "password": password})
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Ensure system.logs permission is present via RBAC helper override.
    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_db_handling

    _orig_get_effective_permissions = user_db_handling.get_effective_permissions

    def _patched_get_effective_permissions(user_id: int):
        perms = _orig_get_effective_permissions(user_id)
        if "system.logs" not in perms:
            perms = list(perms) + ["system.logs"]
        return perms

    monkeypatch.setattr(user_db_handling, "get_effective_permissions", _patched_get_effective_permissions)

    # 3. Stub monitoring service to avoid real DBs and keep the response simple.
    from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_mod

    class _FakeMonitoringService:
        def list_watchlists(self) -> list[dict]:
            return []

    def _fake_get_topic_monitoring_service() -> _FakeMonitoringService:

        return _FakeMonitoringService()

    monitoring_mod.get_topic_monitoring_service = _fake_get_topic_monitoring_service  # type: ignore[assignment]

    # 4. Ensure monitoring routes are mounted (route gating may disable them in some profiles).
    from tldw_Server_API.app.main import app as fastapi_app
    from tldw_Server_API.app.main import API_V1_PREFIX

    app = fastapi_app
    monitoring_path = f"{API_V1_PREFIX}/monitoring/watchlists"
    if not any(getattr(r, "path", None) == monitoring_path for r in iter_served_routes(app.routes)):
        app.include_router(monitoring_mod.router, prefix=f"{API_V1_PREFIX}")

    # 5. Install the auth capture wrapper and call /api/v1/monitoring/watchlists.
    captured, original = _install_auth_capture(app)
    try:
        resp = client.get("/api/v1/monitoring/watchlists", headers=headers)
        assert resp.status_code == 200, resp.text

        principal = captured.get("principal")
        state = captured.get("state")
        state_auth_principal = captured.get("state_auth_principal")

        assert principal is not None
        assert state is not None
        assert state_auth_principal is not None

        # AuthPrincipal kind/user identity.
        assert principal["kind"] == "user"
        assert principal["user_id"] is not None
        assert principal["api_key_id"] is None

        # request.state mirrors principal identity (user_id, no api_key_id).
        assert str(state["user_id"]) == str(principal["user_id"])
        assert state["api_key_id"] is None

        # request.state.auth.principal mirrors principal and state.
        assert state_auth_principal["kind"] == principal["kind"]
        assert str(state_auth_principal["user_id"]) == str(principal["user_id"])
        assert state_auth_principal["api_key_id"] == principal["api_key_id"]
        assert state_auth_principal["org_ids"] == principal["org_ids"]
        assert state_auth_principal["team_ids"] == principal["team_ids"]
        assert state["org_ids"] == principal["org_ids"]
        assert state["team_ids"] == principal["team_ids"]
    finally:
        _restore_auth_capture(app, original)

from types import SimpleNamespace
from typing import List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_mod
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_CONTROL
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(
    *,
    kind: str = "user",
    is_admin: bool = False,
    roles: List[str] | None = None,
    permissions: List[str] | None = None,
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


class _FakeControlDb:
    def get_run(self, run_id: str):
        return SimpleNamespace(
            run_id=run_id,
            tenant_id="default",
            user_id=1,
        )


def _build_app_with_overrides(
    principal: AuthPrincipal,
    *,
    user_permissions: List[str],
    monkeypatch: pytest.MonkeyPatch,
) -> FastAPI:
    app = FastAPI()
    app.include_router(workflows_mod.router)

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
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

    async def _fake_get_request_user():
        return SimpleNamespace(
            id=1,
            username="wf-control-user",
            is_active=True,
            roles=list(principal.roles),
            permissions=list(user_permissions),
            is_admin=principal.is_admin,
            tenant_id="default",
        )

    app.dependency_overrides[workflows_mod.get_request_user] = _fake_get_request_user

    async def _fake_get_db():
        return _FakeControlDb()

    app.dependency_overrides[workflows_mod._get_db] = _fake_get_db

    class _FakeEngine:
        def __init__(self, _db):
            self.db = _db
            self.paused_runs: List[str] = []

        def pause(self, run_id: str) -> None:
            self.paused_runs.append(run_id)

        def resume(self, run_id: str) -> None:  # pragma: no cover - not exercised here
            return None

        def cancel(self, run_id: str) -> None:  # pragma: no cover - not exercised here
            return None

    # Avoid invoking the real WorkflowEngine; use a lightweight stub instead.
    monkeypatch.setattr("tldw_Server_API.app.core.Workflows.WorkflowEngine", _FakeEngine, raising=True)

    return app


@pytest.mark.asyncio
async def test_workflows_control_run_forbidden_when_principal_lacks_control_permission_but_user_has(monkeypatch: pytest.MonkeyPatch):
    """
    PermissionChecker sees workflows.runs.control on the User object, but the
    AuthPrincipal lacks WORKFLOWS_RUNS_CONTROL in its permissions. The request
    must still be forbidden, demonstrating that require_permissions(WORKFLOWS_RUNS_CONTROL)
    is the effective gate for the control_run endpoint.
    """
    principal = _make_principal(
        roles=["user"],
        permissions=[],
        is_admin=False,
    )
    app = _build_app_with_overrides(
        principal,
        user_permissions=[WORKFLOWS_RUNS_CONTROL],
        monkeypatch=monkeypatch,
    )

    with TestClient(app) as client:
        resp = client.post("/api/v1/workflows/runs/run-123/pause")
    assert resp.status_code == 403
    assert WORKFLOWS_RUNS_CONTROL in resp.json().get("detail", "")


@pytest.mark.asyncio
async def test_workflows_control_run_allowed_with_control_permission(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(
        roles=["user"],
        permissions=[WORKFLOWS_RUNS_CONTROL],
        is_admin=False,
    )
    app = _build_app_with_overrides(
        principal,
        user_permissions=[WORKFLOWS_RUNS_CONTROL],
        monkeypatch=monkeypatch,
    )

    with TestClient(app) as client:
        resp = client.post("/api/v1/workflows/runs/run-123/pause")
    # The handler should succeed (no 403 from require_permissions) and
    # return the nominal success payload.
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True

from types import SimpleNamespace
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import tools as tools_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(
    *,
    kind: str = "user",
    is_admin: bool = False,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
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


def _build_app_with_overrides(principal: AuthPrincipal) -> FastAPI:
    app = FastAPI()
    app.include_router(tools_mod.router)

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
            username="tools-user",
            is_active=True,
            roles=list(principal.roles),
            permissions=list(principal.permissions),
            is_admin=principal.is_admin,
            tenant_id="default",
        )

    app.dependency_overrides[tools_mod.get_request_user] = _fake_get_request_user
    return app


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_tool_forbidden_without_permission():
    principal = _make_principal(
        roles=["user"],
        permissions=[],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)

    with TestClient(app) as client:
        resp = client.post(
            "/tools/execute",
            json={"tool_name": "echo", "arguments": {}, "dry_run": True, "idempotency_key": None},
        )
    assert resp.status_code == 403


@pytest.mark.asyncio
@pytest.mark.unit
async def test_execute_tool_allowed_with_permission(monkeypatch):
    principal = _make_principal(
        roles=["user"],
        permissions=["tools.execute:*"],
        is_admin=False,
    )
    app = _build_app_with_overrides(principal)

    class _FakeExecutor:
        def __init__(self) -> None:
            self.calls: Dict[str, Any] = {}

        async def execute(self, **kwargs):
            self.calls["execute"] = kwargs
            return {"result": {"ok": True}, "module": "fake"}

        async def list_tools(self, **_kwargs):
            return {"tools": []}

    fake_executor = _FakeExecutor()

    monkeypatch.setattr("tldw_Server_API.app.core.Tools.tool_executor.ToolExecutor", lambda: fake_executor)

    with TestClient(app) as client:
        resp = client.post(
            "/tools/execute",
            json={"tool_name": "echo", "arguments": {}, "dry_run": False, "idempotency_key": None},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True
    assert "execute" in fake_executor.calls

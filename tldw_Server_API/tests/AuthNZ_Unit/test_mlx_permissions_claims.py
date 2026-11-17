from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import mlx as mlx_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


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
    app.include_router(mlx_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None, "principal must be provided when fail_with_401 is False"
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

    async def _fake_check_rate_limit() -> None:
        # No-op to keep tests focused on RBAC behavior
        return

    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[mlx_mod.check_rate_limit] = _fake_check_rate_limit

    class _StubRegistry:
        def __init__(self) -> None:
            self.loaded = False
            self.unloaded = False

        def load(self, *args, **kwargs):
            self.loaded = True
            return {"status": "ok"}

        def unload(self):
            self.unloaded = True
            return {"status": "unloaded"}

        def status(self):
            return {"ok": True}

    def _get_stub_registry() -> _StubRegistry:
        return _StubRegistry()

    app.dependency_overrides[mlx_mod._resolve_mlx_registry] = _get_stub_registry

    return app


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/v1/llm/providers/mlx/load", {}),
        ("post", "/api/v1/llm/providers/mlx/unload", {}),
        ("get", "/api/v1/llm/providers/mlx/status", None),
    ],
)
def test_mlx_lifecycle_401_when_principal_unavailable(method: str, path: str, payload: dict | None):
    app = _build_app_with_overrides(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        if method == "post":
            resp = client.post(path, json=payload or {})
        else:
            resp = client.get(path)

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/v1/llm/providers/mlx/load", {}),
        ("post", "/api/v1/llm/providers/mlx/unload", {}),
        ("get", "/api/v1/llm/providers/mlx/status", None),
    ],
)
def test_mlx_lifecycle_403_when_missing_admin_role(method: str, path: str, payload: dict | None):
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        if method == "post":
            resp = client.post(path, json=payload or {})
        else:
            resp = client.get(path)

    assert resp.status_code == 403


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/v1/llm/providers/mlx/load", {}),
        ("post", "/api/v1/llm/providers/mlx/unload", {}),
        ("get", "/api/v1/llm/providers/mlx/status", None),
    ],
)
def test_mlx_lifecycle_200_for_admin_principal(monkeypatch, method: str, path: str, payload: dict | None):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    if path.endswith("/load"):
        # Ensure a valid default model_path so internal helpers do not raise.
        monkeypatch.setattr("tldw_Server_API.app.core.LLM_Calls.providers.mlx_provider._default_settings", lambda: {"model_path": "stub-model-path"})
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        if method == "post":
            resp = client.post(path, json=payload or {})
        else:
            resp = client.get(path)

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("backend") == "mlx"

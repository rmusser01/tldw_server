from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import vllm_management as vm
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


class _StubRepository:
    def create_instance(self, payload):  # noqa: ANN001
        _ = payload
        return {
            "instance_id": "stub-1",
            "name": "stub",
            "execution_mode": "local",
            "transport_config": {},
            "launch_spec": {"model": "stub-model"},
            "routing_policy": {},
            "declared_capabilities": {"chat": True},
            "probed_capabilities": {},
            "effective_capabilities": {},
            "desired_state": "stopped",
            "observed_state": "stopped",
            "last_known_base_url": None,
            "last_error": None,
            "executor_handle": {},
            "created_at": "2026-03-10T00:00:00+00:00",
            "updated_at": "2026-03-10T00:00:00+00:00",
        }

    def list_instances(self):
        return []

    def get_instance(self, instance_id: str):  # noqa: ARG002
        return None

    def update_instance(self, instance_id: str, patch):  # noqa: ANN001, ARG002
        _ = patch
        raise AssertionError("should not be called without admin access")

    def update_instance_runtime(self, instance_id: str, patch):  # noqa: ANN001, ARG002
        _ = patch
        raise AssertionError("should not be called without admin access")

    def delete_instance(self, instance_id: str):  # noqa: ARG002
        raise AssertionError("should not be called without admin access")

    def set_default_instance(self, instance_id: str | None) -> None:  # noqa: ARG002
        raise AssertionError("should not be called without admin access")

    def get_default_instance_id(self) -> str | None:
        return None


def _build_app_with_overrides(
    principal: Optional[AuthPrincipal],
    *,
    fail_with_401: bool = False,
) -> FastAPI:
    app = FastAPI()
    app.include_router(vm.router, prefix="/api/v1")

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

    async def _fake_check_rate_limit() -> None:
        return

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[vm.check_rate_limit] = _fake_check_rate_limit
    app.dependency_overrides[vm._resolve_vllm_repository] = _StubRepository
    return app


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/v1/llm/providers/vllm/instances", {"name": "stub", "execution_mode": "local"}),
        ("get", "/api/v1/llm/providers/vllm/instances", None),
        ("post", "/api/v1/llm/providers/vllm/default", {"instance_id": "stub-1"}),
    ],
)
def test_vllm_management_401_when_principal_unavailable(method: str, path: str, payload: dict | None):
    app = _build_app_with_overrides(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        if method == "post":
            response = client.post(path, json=payload or {})
        else:
            response = client.get(path)

    assert response.status_code == 401
    assert "Authentication required" in response.json().get("detail", "")


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/v1/llm/providers/vllm/instances", {"name": "stub", "execution_mode": "local"}),
        ("get", "/api/v1/llm/providers/vllm/instances", None),
        ("post", "/api/v1/llm/providers/vllm/default", {"instance_id": "stub-1"}),
    ],
)
def test_vllm_management_403_when_missing_admin_role(method: str, path: str, payload: dict | None):
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        if method == "post":
            response = client.post(path, json=payload or {})
        else:
            response = client.get(path)

    assert response.status_code == 403


@pytest.mark.unit
def test_vllm_management_200_for_admin_principal():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)

    with TestClient(app) as client:
        response = client.get("/api/v1/llm/providers/vllm/instances")

    assert response.status_code == 200
    assert response.json()["backend"] == "vllm"

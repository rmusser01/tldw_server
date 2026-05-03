from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import storage as storage_mod
from tldw_Server_API.app.api.v1.endpoints import storage_admin_quotas
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(*, is_admin: bool, roles: list[str] | None = None) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=roles or [],
        permissions=[],
        is_admin=is_admin,
        org_ids=[1],
        team_ids=[],
    )


def _build_app(principal: AuthPrincipal | None, *, fail_with_401: bool = False) -> FastAPI:
    app = FastAPI()
    app.include_router(storage_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    return app


class _FakeStorageService:
    async def set_user_quota(self, user_id: int, quota_mb: int) -> dict[str, Any]:
        return {
            "storage_quota_mb": quota_mb,
            "storage_used_mb": 100.0,
            "available_mb": max(0.0, float(quota_mb) - 100.0),
            "usage_percentage": 10.0,
        }

    async def set_team_quota(
        self,
        team_id: int,
        quota_mb: int,
        *,
        soft_limit_pct: int,
        hard_limit_pct: int,
    ) -> None:
        return None

    async def set_org_quota(
        self,
        org_id: int,
        quota_mb: int,
        *,
        soft_limit_pct: int,
        hard_limit_pct: int,
    ) -> None:
        return None

    async def get_team_quota(self, team_id: int) -> dict[str, Any]:
        return {
            "quota_mb": 1024,
            "used_mb": 128.0,
            "remaining_mb": 896.0,
            "usage_pct": 12.5,
            "at_soft_limit": False,
            "at_hard_limit": False,
            "has_quota": True,
        }

    async def get_org_quota(self, org_id: int) -> dict[str, Any]:
        return {
            "quota_mb": 2048,
            "used_mb": 256.0,
            "remaining_mb": 1792.0,
            "usage_pct": 12.5,
            "at_soft_limit": False,
            "at_hard_limit": False,
            "has_quota": True,
        }


@pytest.mark.asyncio
async def test_storage_admin_quota_endpoints_401_when_principal_unavailable(monkeypatch: pytest.MonkeyPatch):
    app = _build_app(principal=None, fail_with_401=True)
    with TestClient(app) as client:
        resp = client.get("/api/v1/storage/admin/quotas/org/1")
    assert resp.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("put", "/api/v1/storage/admin/quotas/user/2", {"quota_mb": 500, "soft_limit_pct": 80, "hard_limit_pct": 100}),
        ("put", "/api/v1/storage/admin/quotas/team/3", {"quota_mb": 600, "soft_limit_pct": 80, "hard_limit_pct": 100}),
        ("put", "/api/v1/storage/admin/quotas/org/4", {"quota_mb": 700, "soft_limit_pct": 80, "hard_limit_pct": 100}),
        ("get", "/api/v1/storage/admin/quotas/team/3", None),
        ("get", "/api/v1/storage/admin/quotas/org/4", None),
    ],
)
async def test_storage_admin_quota_endpoints_403_for_non_admin_principal(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
):
    principal = _make_principal(is_admin=False, roles=["user"])
    app = _build_app(principal=principal)

    async def _fake_get_service() -> _FakeStorageService:
        return _FakeStorageService()

    monkeypatch.setattr(storage_admin_quotas, "_get_storage_service", _fake_get_service)
    with TestClient(app) as client:
        resp = client.request(method.upper(), path, json=payload)
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_storage_admin_quota_endpoints_200_for_admin_role_claim(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(is_admin=False, roles=["admin"])
    app = _build_app(principal=principal)

    async def _fake_get_service() -> _FakeStorageService:
        return _FakeStorageService()

    monkeypatch.setattr(storage_admin_quotas, "_get_storage_service", _fake_get_service)

    with TestClient(app) as client:
        user_put = client.put(
            "/api/v1/storage/admin/quotas/user/2",
            json={"quota_mb": 500, "soft_limit_pct": 80, "hard_limit_pct": 100},
        )
        team_put = client.put(
            "/api/v1/storage/admin/quotas/team/3",
            json={"quota_mb": 600, "soft_limit_pct": 80, "hard_limit_pct": 100},
        )
        org_put = client.put(
            "/api/v1/storage/admin/quotas/org/4",
            json={"quota_mb": 700, "soft_limit_pct": 80, "hard_limit_pct": 100},
        )
        team_get = client.get("/api/v1/storage/admin/quotas/team/3")
        org_get = client.get("/api/v1/storage/admin/quotas/org/4")

    assert user_put.status_code == 200, user_put.text
    assert team_put.status_code == 200, team_put.text
    assert org_put.status_code == 200, org_put.text
    assert team_get.status_code == 200, team_get.text
    assert org_get.status_code == 200, org_get.text


@pytest.mark.asyncio
async def test_storage_admin_quota_endpoints_200_for_is_admin_claim(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(is_admin=True, roles=["user"])
    app = _build_app(principal=principal)

    async def _fake_get_service() -> _FakeStorageService:
        return _FakeStorageService()

    monkeypatch.setattr(storage_admin_quotas, "_get_storage_service", _fake_get_service)

    with TestClient(app) as client:
        resp = client.get("/api/v1/storage/admin/quotas/team/3")

    assert resp.status_code == 200, resp.text

from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import connectors as connectors_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


def _build_app_with_overrides(
    principal: Optional[AuthPrincipal],
    *,
    fail_with_401: bool = False,
) -> FastAPI:
    app = FastAPI()
    app.include_router(connectors_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        _ = request  # required by dependency signature
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    async def _fake_db_transaction():
        class _Dummy:
            pass

        yield _Dummy()

    app.dependency_overrides[auth_deps.get_db_transaction] = _fake_db_transaction

    return app


def _make_principal(
    *,
    is_admin: bool = False,
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
        org_ids=[],
        team_ids=[],
    )


async def _fake_policy(_: Any, __: int) -> Dict[str, Any]:
    return {
        "enabled_providers": [],
        "allowed_export_formats": [],
        "allowed_file_types": [],
        "max_file_size_mb": 0,
        "account_linking_role": "admin",
        "allowed_account_domains": [],
        "allowed_remote_paths": [],
        "denied_remote_paths": [],
        "allowed_notion_workspaces": [],
        "denied_notion_workspaces": [],
        "quotas_per_role": {},
    }


@pytest.mark.external_api
@pytest.mark.unit
@pytest.mark.asyncio
async def test_connectors_admin_policy_401_when_principal_unavailable(monkeypatch):
    app = _build_app_with_overrides(principal=None, fail_with_401=True)
    monkeypatch.setattr("tldw_Server_API.app.core.External_Sources.connectors_service.get_policy", _fake_policy, raising=True)

    with TestClient(app) as client:
        resp = client.get("/api/v1/connectors/admin/policy", params={"org_id": 1})

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_connectors_admin_policy_403_when_missing_permission(monkeypatch):
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)
    monkeypatch.setattr("tldw_Server_API.app.core.External_Sources.connectors_service.get_policy", _fake_policy, raising=True)

    with TestClient(app) as client:
        resp = client.get("/api/v1/connectors/admin/policy", params={"org_id": 1})

    assert resp.status_code == 403
    detail = resp.json().get("detail", "")
    # Either role or permission guard may fire first; assert one of the expected markers.
    assert (
        SYSTEM_CONFIGURE in detail
        or "Required role(s)" in detail
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_connectors_admin_policy_200_for_admin_principal(monkeypatch):
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[],
    )
    app = _build_app_with_overrides(principal=principal)
    monkeypatch.setattr("tldw_Server_API.app.core.External_Sources.connectors_service.get_policy", _fake_policy, raising=True)

    with TestClient(app) as client:
        resp = client.get("/api/v1/connectors/admin/policy", params={"org_id": 1})

    assert resp.status_code == 200
    body = resp.json()
    assert body["org_id"] == 1

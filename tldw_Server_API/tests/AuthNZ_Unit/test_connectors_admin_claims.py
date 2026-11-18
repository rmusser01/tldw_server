from __future__ import annotations

from typing import Any, Optional

import pytest
from fastapi import FastAPI, HTTPException, Depends
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import connectors as connectors_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE


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
    org_memberships: Optional[list[dict[str, int]]] = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(connectors_mod.router, prefix="/api/v1")

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

    async def _fake_current_user() -> dict[str, Any]:
        return {
            "id": 1,
            "role": "admin",
            "org_memberships": org_memberships if org_memberships is not None else [{"org_id": 1}],
            "is_active": True,
        }

    app.dependency_overrides[auth_deps.get_current_active_user] = _fake_current_user

    class _FakeDB:
        async def execute(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            """No-op execute for DB dependency."""
            return None

    async def _fake_get_db():
        return _FakeDB()

    app.dependency_overrides[auth_deps.get_db_transaction] = _fake_get_db

    @app.get("/api/v1/test/org-policy")
    async def _test_org_policy_endpoint(
        org_policy: dict[str, Any] = Depends(auth_deps.get_org_policy_from_principal),
    ) -> dict[str, Any]:
        # Echo back the resolved org_id so tests can assert behaviour.
        return {"org_id": int(org_policy.get("org_id"))}

    return app


@pytest.mark.asyncio
async def test_connectors_admin_policy_401(monkeypatch: pytest.MonkeyPatch):
    app = _build_app(principal=None, fail_with_401=True)
    with TestClient(app) as client:
        resp = client.get("/api/v1/connectors/admin/policy", params={"org_id": 1})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_connectors_admin_policy_403_without_admin(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(is_admin=False, roles=["user"], permissions=[])
    app = _build_app(principal=principal)
    with TestClient(app) as client:
        resp = client.get("/api/v1/connectors/admin/policy", params={"org_id": 1})
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_connectors_admin_policy_200_for_admin_with_perm(monkeypatch: pytest.MonkeyPatch):
    principal = _make_principal(is_admin=True, roles=["admin"], permissions=[SYSTEM_CONFIGURE])
    app = _build_app(principal=principal)

    async def _fake_get_policy(db, org_id):
        return {"org_id": org_id, "source": "test"}

    monkeypatch.setattr("tldw_Server_API.app.core.External_Sources.connectors_service.get_policy", _fake_get_policy)

    with TestClient(app) as client:
        resp = client.get("/api/v1/connectors/admin/policy", params={"org_id": 1})
    assert resp.status_code == 200
    assert resp.json().get("org_id") == 1


def test_connectors_policy_role_ignores_boolean_admin_without_claims():
    principal = _make_principal(is_admin=True, roles=["user"], permissions=[])
    assert connectors_mod._principal_role_for_policy(principal) == "member"


@pytest.mark.asyncio
async def test_connectors_single_user_org_policy_principal_only_paths(monkeypatch: pytest.MonkeyPatch):
    """
    Org-policy fallback is principal-only:
    - explicit single_user subject may resolve synthetic org_id=1
    - non-single-user principals without org claims are rejected
    """
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    # Even when a legacy compatibility flag is set, behavior is principal-only.
    monkeypatch.setenv("ORG_POLICY_SINGLE_USER_PRINCIPAL", "0")

    # Case A: single-user principal with no org memberships → synthetic org_id=1 allowed.
    single_principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="single_user",
        token_type="api_key",
        jti=None,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )
    app_a = _build_app(principal=single_principal, org_memberships=[])

    async def _fake_get_policy(db, org_id):
        return {"org_id": org_id, "source": "test"}

    # For the test org-policy endpoint, patch the helper used by get_org_policy_from_principal.
    monkeypatch.setattr(auth_deps, "get_policy", _fake_get_policy)

    with TestClient(app_a) as client:
        resp = client.get("/api/v1/test/org-policy")
    assert resp.status_code == 200
    assert resp.json().get("org_id") == 1

    # Case B: non-single-user principal with no org memberships → 400 from org-policy helper.
    # Important: ensure this principal has **no** org_ids so the helper cannot
    # resolve an organization from claims and must reject fallback.
    non_single_principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )
    app_b = _build_app(principal=non_single_principal, org_memberships=[])

    with TestClient(app_b) as client:
        resp = client.get("/api/v1/test/org-policy")
    assert resp.status_code == 400

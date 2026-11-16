"""Unit tests validating vector store RBAC behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


def _make_principal(
    *,
    kind: str = "user",
    is_admin: bool = False,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
) -> AuthPrincipal:
    """Create a test AuthPrincipal with optional roles/permissions and admin flag.

    Args:
        kind (str): Principal kind to assign (default: "user").
        is_admin (bool): Whether the principal should be marked as admin.
        roles (list[str] | None): Roles to attach; defaults to empty list when None.
        permissions (list[str] | None): Permissions to attach; defaults to empty list when None.

    Returns:
        AuthPrincipal: Principal populated with provided attributes and empty org/team ids.
    """
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


def _build_app_with_overrides(principal: AuthPrincipal, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """Build a FastAPI app wired with fake auth/principal adapters for testing."""
    app = FastAPI()
    app.include_router(vs_mod.router)

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

    # Override both the endpoint-local and shared auth dependency so that
    # claim-first gates (require_roles/require_permissions) and any direct
    # principal dependencies see the same test principal.
    app.dependency_overrides[vs_mod.get_auth_principal] = _fake_get_auth_principal
    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    async def _fake_get_request_user():
        return SimpleNamespace(
            id=1,
            username="vector-admin",
            is_active=True,
            roles=list(principal.roles),
            permissions=list(principal.permissions),
            is_admin=principal.is_admin,
        )

    app.dependency_overrides[vs_mod.get_request_user] = _fake_get_request_user

    class _FakeAdapter:
        async def initialize(self):
            return None

        async def health(self):
            return {"ok": True}

    async def _fake_get_adapter_for_user(_user, _dim):
        return _FakeAdapter()

    monkeypatch.setattr(vs_mod, "_get_adapter_for_user", _fake_get_adapter_for_user)
    return app


@pytest.mark.asyncio
@pytest.mark.unit
async def test_vector_stores_admin_health_forbidden_without_admin_role(monkeypatch):
    """User-role principal without permissions should receive 403 on /vector_stores/admin/health."""
    principal = _make_principal(roles=["user"], permissions=[], is_admin=False)
    app = _build_app_with_overrides(principal, monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/vector_stores/admin/health")
    assert resp.status_code == 403


@pytest.mark.asyncio
@pytest.mark.unit
async def test_vector_stores_admin_health_allowed_with_admin_role(monkeypatch):
    """Admin principal with admin role and is_admin=True receives 200 from /vector_stores/admin/health."""
    principal = _make_principal(roles=["admin"], permissions=[], is_admin=True)
    app = _build_app_with_overrides(principal, monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/vector_stores/admin/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") == True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_vector_stores_batches_non_admin_cannot_override_user_id(monkeypatch):
    """Non-admin principals may list only their own batches; user_id override should be forbidden."""
    principal = _make_principal(roles=["user"], permissions=[], is_admin=False)
    app = _build_app_with_overrides(principal, monkeypatch)

    # Patch db_list_batches to avoid touching real storage and to capture the user_id used.
    captured: dict[str, Any] = {}

    def _fake_list_batches(*, user_id: str, status=None, limit: int, offset: int):
        captured["user_id"] = user_id
        return [{"id": "b1", "user_id": user_id, "status": status or "completed"}]

    monkeypatch.setattr("tldw_Server_API.app.core.Embeddings.vector_store_batches_db.list_batches", _fake_list_batches)

    with TestClient(app) as client:
        # Without override: should succeed and use the current user's id.
        resp_self = client.get("/vector_stores/batches")
        assert resp_self.status_code == 200
        body_self = resp_self.json()
        assert isinstance(body_self.get("data"), list)
        assert captured.get("user_id") == "1"

        # Override to a different user_id should be rejected with 403 for non-admins.
        resp_other = client.get("/vector_stores/batches", params={"user_id": "2"})
        assert resp_other.status_code == 403


@pytest.mark.asyncio
@pytest.mark.unit
async def test_vector_stores_batches_rejects_boolean_admin_without_claims(monkeypatch):
    """Boolean admin flag alone must not permit cross-user batch inspection."""
    principal = _make_principal(roles=["user"], permissions=[], is_admin=True)
    app = _build_app_with_overrides(principal, monkeypatch)

    with TestClient(app) as client:
        resp = client.get("/vector_stores/batches", params={"user_id": "42"})
    assert resp.status_code == 403


@pytest.mark.asyncio
@pytest.mark.unit
async def test_vector_stores_batches_admin_can_override_user_id(monkeypatch):
    """Admin principals may override user_id and inspect other users' batches."""
    principal = _make_principal(roles=["admin"], permissions=["*"], is_admin=True)
    app = _build_app_with_overrides(principal, monkeypatch)

    captured: dict[str, Any] = {}

    def _fake_list_batches(*, user_id: str, status=None, limit: int, offset: int):
        captured["user_id"] = user_id
        return [{"id": "b2", "user_id": user_id, "status": status or "completed"}]

    monkeypatch.setattr("tldw_Server_API.app.core.Embeddings.vector_store_batches_db.list_batches", _fake_list_batches)

    with TestClient(app) as client:
        resp = client.get("/vector_stores/batches", params={"user_id": "42"})
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body.get("data"), list)
        assert captured.get("user_id") == "42"

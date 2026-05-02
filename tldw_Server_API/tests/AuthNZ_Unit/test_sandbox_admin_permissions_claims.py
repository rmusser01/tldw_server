from types import SimpleNamespace
from typing import List

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import sandbox as sandbox_mod
from tldw_Server_API.app.core.AuthNZ.permissions import ROLE_ADMIN
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


class _FakeStore:
    def list_idempotency(self, **_kwargs):
        return []

    def count_idempotency(self, **_kwargs) -> int:

        return 0

    def list_usage(self, **_kwargs):

        return []

    def count_usage(self, **_kwargs) -> int:

        return 0


class _FakeOrch:
    def __init__(self) -> None:
        self._store = _FakeStore()

    def list_runs(self, **_kwargs):

        return []

    def count_runs(self, **_kwargs) -> int:

        return 0

    def get_run_owner(self, run_id: str) -> str | None:
        return "user-1"


class _FakeService:
    def __init__(self) -> None:
        self._orch = _FakeOrch()

    def get_run(self, run_id: str):
        return SimpleNamespace(
            id=run_id,
            spec_version="v1",
            runtime=None,
            runtime_version="1.0",
            base_image="image",
            image_digest="digest",
            policy_hash="policy",
            phase=SimpleNamespace(value="completed"),
            exit_code=0,
            started_at="2024-01-01T00:00:00Z",
            finished_at="2024-01-01T00:00:01Z",
            message=None,
            resource_usage=None,
        )

    def macos_diagnostics(self):
        return {
            "host": {
                "os": "darwin",
                "arch": "arm64",
                "apple_silicon": True,
                "macos_version": "15.0",
                "supported": True,
                "reasons": [],
            },
            "helper": {
                "configured": True,
                "path": "/tmp/helper",
                "exists": True,
                "executable": True,
                "ready": True,
                "transport": "fake",
                "reasons": [],
            },
            "templates": {},
            "runtimes": {},
        }


def _build_app_with_overrides(principal: AuthPrincipal) -> FastAPI:
    app = FastAPI()
    app.include_router(sandbox_mod.router, prefix="/api/v1")

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
            username="sandbox-admin",
            is_active=True,
            roles=list(principal.roles),
            permissions=list(principal.permissions),
            is_admin=principal.is_admin,
            tenant_id="default",
        )

    app.dependency_overrides[sandbox_mod.get_request_user] = _fake_get_request_user

    return app


@pytest.mark.asyncio
async def test_sandbox_admin_runs_forbidden_for_non_admin_principal(monkeypatch):
    principal = _make_principal(
        roles=["user"],
        permissions=[],
        is_admin=False,
    )
    fake_service = _FakeService()
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/runs")
    assert resp.status_code == 403
    assert "admin" in resp.json().get("detail", "")


@pytest.mark.asyncio
async def test_sandbox_admin_runs_allowed_for_admin_principal(monkeypatch):
    principal = _make_principal(
        roles=[ROLE_ADMIN],
        permissions=[],
        is_admin=True,
    )
    fake_service = _FakeService()
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("items"), list)
    assert body.get("total") == 0
    assert body["has_more"] is False
    assert body["next_offset"] is None


@pytest.mark.asyncio
async def test_sandbox_admin_idempotency_includes_pagination_aliases(monkeypatch):
    principal = _make_principal(
        roles=[ROLE_ADMIN],
        permissions=[],
        is_admin=True,
    )
    fake_service = _FakeService()
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/idempotency")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("items"), list)
    assert body.get("total") == 0
    assert body["has_more"] is False
    assert body["next_offset"] is None


@pytest.mark.asyncio
async def test_sandbox_admin_usage_includes_pagination_aliases(monkeypatch):
    principal = _make_principal(
        roles=[ROLE_ADMIN],
        permissions=[],
        is_admin=True,
    )
    fake_service = _FakeService()
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/usage")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("items"), list)
    assert body.get("total") == 0
    assert body["has_more"] is False
    assert body["next_offset"] is None


@pytest.mark.asyncio
async def test_sandbox_admin_macos_diagnostics_allowed_for_admin_principal(monkeypatch):
    principal = _make_principal(
        roles=[ROLE_ADMIN],
        permissions=[],
        is_admin=True,
    )
    fake_service = _FakeService()
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    app = _build_app_with_overrides(principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/sandbox/admin/macos-diagnostics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["host"]["supported"] is True


@pytest.mark.unit
def test_sandbox_run_owner_rejects_legacy_admin_flag_without_claims(monkeypatch):
    fake_service = SimpleNamespace(
        _orch=SimpleNamespace(
            get_run_owner=lambda _run_id: "2",
        )
    )
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    legacy_admin_user = SimpleNamespace(
        id=1,
        roles=["user"],
        permissions=[],
        is_admin=True,
    )

    with pytest.raises(HTTPException) as excinfo:
        sandbox_mod._require_run_owner("run-1", legacy_admin_user)
    assert excinfo.value.status_code == 404


@pytest.mark.unit
def test_sandbox_run_owner_allows_admin_permission_claim(monkeypatch):
    fake_service = SimpleNamespace(
        _orch=SimpleNamespace(
            get_run_owner=lambda _run_id: "2",
        )
    )
    monkeypatch.setattr(sandbox_mod, "_service", fake_service, raising=True)

    permission_admin_user = SimpleNamespace(
        id=1,
        roles=["user"],
        permissions=["system.configure"],
        is_admin=False,
    )

    owner_id = sandbox_mod._require_run_owner("run-1", permission_admin_user)
    assert owner_id == "2"

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import jobs_admin as jobs_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


pytestmark = pytest.mark.unit


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="admin",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


def _build_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    app = FastAPI()
    app.include_router(jobs_mod.router, prefix="/api/v1")

    principal = _admin_principal()

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        request.state.auth = AuthContext(
            principal=principal,
            ip=None,
            user_agent=None,
            request_id=None,
        )
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps as audit_deps

    class _FakeAuditService:
        async def log_event(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)
            return None

    async def _fake_get_audit_service_for_user() -> _FakeAuditService:
        return _FakeAuditService()

    app.dependency_overrides[audit_deps.get_audit_service_for_user] = _fake_get_audit_service_for_user

    class _FakeJobManager:
        def __init__(self, backend: str | None = None, db_url: str | None = None) -> None:
            self.backend = backend
            self.db_url = db_url

        def prune_jobs(self, **_kwargs: Any) -> int:
            return 0

    monkeypatch.setattr(jobs_mod, "JobManager", _FakeJobManager)
    monkeypatch.delenv("JOBS_DOMAIN_SCOPED_RBAC", raising=False)
    monkeypatch.delenv("JOBS_RBAC_FORCE", raising=False)
    monkeypatch.delenv("JOBS_DOMAIN_RBAC_PRINCIPAL", raising=False)
    return app


def test_jobs_admin_router_uses_standard_role_factory_alias() -> None:
    assert jobs_mod.RequireRole is auth_deps.RequireRole
    assert not hasattr(jobs_mod, "require_roles")


def test_prune_skips_confirm_when_tldw_test_mode_is_y(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    monkeypatch.setenv("JOBS_REQUIRE_CONFIRM", "1")

    app = _build_app(monkeypatch)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/jobs/prune",
            json={
                "statuses": ["completed"],
                "older_than_days": 1,
                "domain": "tenant-a",
                "dry_run": False,
            },
        )
    assert resp.status_code == 200, resp.text


def test_prune_requires_confirm_outside_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "0")
    monkeypatch.setenv("JOBS_REQUIRE_CONFIRM", "1")

    app = _build_app(monkeypatch)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/jobs/prune",
            json={
                "statuses": ["completed"],
                "older_than_days": 1,
                "domain": "tenant-a",
                "dry_run": False,
            },
        )
    assert resp.status_code == 400
    assert "Confirmation required" in resp.json().get("detail", "")

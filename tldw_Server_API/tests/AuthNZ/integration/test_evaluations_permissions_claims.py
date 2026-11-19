from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_crud import (
    RBAC_EVALS_CREATE,
    crud_router,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext


def _build_app_with_overrides(principal: AuthPrincipal):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(crud_router, prefix="/api/v1/evaluations")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
        _ = request  # required by dependency signature
        ip = request.client.host if request.client else None
        ua = request.headers.get("User-Agent") if request.headers else None
        request_id = request.headers.get("X-Request-ID") if request.headers else None
        ctx = AuthContext(
            principal=principal,
            ip=ip,
            user_agent=ua,
            request_id=request_id,
        )
        request.state.auth = ctx
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    async def _fake_verify_api_key() -> str:
        return "vk-test"

    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_auth as eval_auth

    app.dependency_overrides[eval_auth.verify_api_key] = _fake_verify_api_key

    async def _fake_get_request_user() -> SimpleNamespace:
        return SimpleNamespace(
            id=1,
            username="eval-user",
            is_active=True,
            roles=list(principal.roles),
            permissions=list(principal.permissions),
            is_admin=principal.is_admin,
        )

    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as udh

    app.dependency_overrides[udh.get_request_user] = _fake_get_request_user
    app.dependency_overrides[eval_auth.get_eval_request_user] = _fake_get_request_user

    async def _noop_rbac_rate_limit(*_args, **_kwargs) -> None:
        return None

    app.dependency_overrides[RBAC_EVALS_CREATE] = _noop_rbac_rate_limit

    return app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluations_list_requires_auth_principal_and_api_key():
    app = _build_app_with_overrides(
        AuthPrincipal(
            kind="anonymous",
            user_id=None,
            api_key_id=None,
            subject=None,
            token_type=None,
            jti=None,
            roles=[],
            permissions=[],
            is_admin=False,
            org_ids=[],
            team_ids=[],
        )
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/evaluations/")
    assert resp.status_code in (401, 403)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluations_list_forbidden_without_claims():
    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["user"],
        permissions=[],  # no evals.read
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )
    app = _build_app_with_overrides(principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/evaluations/", headers={"Authorization": "Bearer test"})
    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluations_list_allows_with_evals_read(monkeypatch):
    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["user"],
        permissions=["evals.read"],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )
    app = _build_app_with_overrides(principal)

    def _fake_service_for_user(_user_id: int):
        class _Svc:
            async def list_evaluations(
                self,
                *,
                limit: int,
                after: str | None,
                eval_type: str | None,
                created_by: Any,
            ):
                _ = (limit, after, eval_type, created_by)
                return [], False

        return _Svc()

    from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as ues

    monkeypatch.setattr(ues, "get_unified_evaluation_service_for_user", _fake_service_for_user)

    with TestClient(app) as client:
        resp = client.get("/api/v1/evaluations/", headers={"Authorization": "Bearer test"})
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("object") == "list"
    assert isinstance(body.get("data"), list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluations_admin_cleanup_respects_claim_first_admin_gate(monkeypatch):
    """
    Sanity check that claim-first heavy-evals gating rejects non-admin principals
    when EVALS_HEAVY_ADMIN_ONLY is enabled.
    """
    from fastapi import HTTPException
    from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_auth import (
        enforce_heavy_evaluations_admin,
    )

    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["user"],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "true")
    with pytest.raises(HTTPException) as excinfo:
        enforce_heavy_evaluations_admin(principal)
    assert excinfo.value.status_code == 403


def _build_app_with_admin_cleanup(principal: AuthPrincipal):
    from fastapi import FastAPI
    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as eval_unified

    app = FastAPI()
    app.include_router(eval_unified.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:
        _ = request  # required by dependency signature
        ip = request.client.host if request.client else None
        ua = request.headers.get("User-Agent") if request.headers else None
        request_id = request.headers.get("X-Request-ID") if request.headers else None
        ctx = AuthContext(
            principal=principal,
            ip=ip,
            user_agent=ua,
            request_id=request_id,
        )
        request.state.auth = ctx
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    async def _fake_admin_principal_if_needed(request: Request) -> AuthPrincipal:
        _ = request  # required by dependency signature
        ctx = AuthContext(
            principal=principal,
            ip=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent") if request.headers else None,
            request_id=request.headers.get("X-Request-ID") if request.headers else None,
        )
        request.state.auth = ctx
        return principal

    # evaluations_unified calls get_auth_principal directly inside this helper,
    # so overriding the helper itself is the most reliable test bypass.
    app.dependency_overrides[eval_unified._get_admin_principal_if_needed] = _fake_admin_principal_if_needed

    async def _fake_verify_api_key() -> str:
        return "vk-test"

    from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_auth as eval_auth

    app.dependency_overrides[eval_auth.verify_api_key] = _fake_verify_api_key

    async def _fake_get_request_user() -> SimpleNamespace:
        return SimpleNamespace(
            id=1,
            username="eval-admin",
            is_active=True,
            roles=list(principal.roles),
            permissions=list(principal.permissions),
            is_admin=principal.is_admin,
        )

    from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as udh

    app.dependency_overrides[udh.get_request_user] = _fake_get_request_user
    app.dependency_overrides[eval_auth.get_eval_request_user] = _fake_get_request_user

    return app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluations_admin_cleanup_forbidden_without_admin_role(monkeypatch):
    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["user"],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )
    app = _build_app_with_admin_cleanup(principal)
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "true")

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/evaluations/admin/idempotency/cleanup",
            headers={"Authorization": "Bearer test"},
        )
    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evaluations_admin_cleanup_allowed_with_admin_role(monkeypatch, tmp_path):
    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )
    app = _build_app_with_admin_cleanup(principal)
    monkeypatch.setenv("EVALS_HEAVY_ADMIN_ONLY", "true")

    from pathlib import Path
    from tldw_Server_API.app.core.DB_Management import db_path_utils as dp_mod
    from tldw_Server_API.app.core.DB_Management import Evaluations_DB as eval_db_mod

    db_file = tmp_path / "evals.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_file.touch()

    def _fake_get_single_user_id() -> int:

        return 1

    def _fake_get_user_base_directory(_user_id: int) -> Path:
        return tmp_path

    def _fake_get_evaluations_db_path(_user_id: int) -> Path:
        return db_file

    monkeypatch.setattr(dp_mod.DatabasePaths, "get_single_user_id", staticmethod(_fake_get_single_user_id))
    monkeypatch.setattr(dp_mod.DatabasePaths, "get_user_base_directory", staticmethod(_fake_get_user_base_directory))
    monkeypatch.setattr(dp_mod.DatabasePaths, "get_evaluations_db_path", staticmethod(_fake_get_evaluations_db_path))

    class _FakeEvalDB:
        def __init__(self, path: str):
            self.path = path
            self.called_with_ttl: int | None = None

        def cleanup_idempotency_keys(self, ttl_hours: int) -> int:
            self.called_with_ttl = ttl_hours
            return 5

    monkeypatch.setattr(eval_db_mod, "EvaluationsDatabase", _FakeEvalDB)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/evaluations/admin/idempotency/cleanup",
            headers={"Authorization": "Bearer test"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("deleted_total") == 5
    details = body.get("details") or []
    assert details and details[0].get("deleted") == 5

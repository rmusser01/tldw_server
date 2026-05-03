from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import workflows as wf_mod
from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_RUNS_CONTROL
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


def _build_app_with_overrides(
    principal: AuthPrincipal,
    dlq_rows: List[Dict[str, Any]],
    *,
    user_permissions: List[str] | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(wf_mod.router)

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
        perms = list(user_permissions) if user_permissions is not None else list(principal.permissions)
        return SimpleNamespace(
            id=1,
            username="wf-user",
            is_active=True,
            roles=list(principal.roles),
            permissions=perms,
            is_admin=principal.is_admin,
            tenant_id="default",
        )

    app.dependency_overrides[wf_mod.get_request_user] = _fake_get_request_user

    class _FakeWorkflowsDB:
        def list_webhook_dlq_all(self, limit: int, offset: int):
            return dlq_rows[offset : offset + limit]

        def delete_webhook_dlq(self, dlq_id: int) -> None:
            for idx, row in enumerate(dlq_rows):
                if int(row.get("id")) == int(dlq_id):
                    dlq_rows.pop(idx)
                    break

    async def _fake_get_db():
        return _FakeWorkflowsDB()

    app.dependency_overrides[wf_mod._get_db] = _fake_get_db  # type: ignore[attr-defined]
    return app


@pytest.mark.asyncio
async def test_workflows_webhook_dlq_forbidden_for_non_admin_principal():
    principal = _make_principal(
        roles=["user"],
        permissions=[],
        is_admin=False,
    )
    dlq_rows: List[Dict[str, Any]] = []
    app = _build_app_with_overrides(principal, dlq_rows)

    with TestClient(app) as client:
        resp = client.get("/api/v1/workflows/webhooks/dlq")
    assert resp.status_code == 403 or resp.status_code == 404


@pytest.mark.asyncio
async def test_workflows_webhook_dlq_allowed_for_admin_role(monkeypatch):
    principal = _make_principal(
        roles=["admin"],
        permissions=["workflows.runs.control"],
        is_admin=True,
    )
    dlq_rows: List[Dict[str, Any]] = [
        {
            "id": 1,
            "tenant_id": "default",
            "run_id": "run-1",
            "url": "https://example.com/hook",
            "attempts": 0,
            "next_attempt_at": None,
            "last_error": None,
            "created_at": "2024-01-01T00:00:00Z",
            "body_json": "{}",
        }
    ]
    app = _build_app_with_overrides(principal, dlq_rows)

    with TestClient(app) as client:
        resp = client.get("/api/v1/workflows/webhooks/dlq")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body.get("items"), list)


@pytest.mark.asyncio
async def test_workflows_webhook_dlq_returns_canonical_offset_pagination() -> None:
    principal = _make_principal(
        roles=["admin"],
        permissions=["workflows.runs.control"],
        is_admin=True,
    )
    dlq_rows: List[Dict[str, Any]] = [
        {
            "id": 1,
            "tenant_id": "default",
            "run_id": "run-1",
            "url": "https://example.com/hook-1",
            "attempts": 0,
            "next_attempt_at": None,
            "last_error": None,
            "created_at": "2024-01-01T00:00:00Z",
            "body_json": "{}",
        },
        {
            "id": 2,
            "tenant_id": "default",
            "run_id": "run-2",
            "url": "https://example.com/hook-2",
            "attempts": 1,
            "next_attempt_at": None,
            "last_error": "retry",
            "created_at": "2024-01-02T00:00:00Z",
            "body_json": "{}",
        },
    ]
    app = _build_app_with_overrides(principal, dlq_rows)

    with TestClient(app) as client:
        resp = client.get("/api/v1/workflows/webhooks/dlq?limit=1&offset=0")

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert [item["id"] for item in body["items"]] == [1]
    assert body["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": None,
        "has_more": True,
        "next_offset": 1,
    }


@pytest.mark.asyncio
async def test_workflows_webhook_dlq_forbidden_when_principal_lacks_control_permission_but_user_has():
    """
    PermissionChecker sees workflows.runs.control on the User object, but the
    AuthPrincipal lacks WORKFLOWS_RUNS_CONTROL in its permissions. The request
    must still be forbidden, demonstrating that require_permissions(WORKFLOWS_RUNS_CONTROL)
    is the effective gate.
    """
    principal = _make_principal(
        roles=["user"],
        permissions=[],
        is_admin=False,
    )
    dlq_rows: List[Dict[str, Any]] = []
    app = _build_app_with_overrides(
        principal,
        dlq_rows,
        user_permissions=[WORKFLOWS_RUNS_CONTROL],
    )

    with TestClient(app) as client:
        resp = client.get("/api/v1/workflows/webhooks/dlq")
    assert resp.status_code == 403
    assert WORKFLOWS_RUNS_CONTROL in resp.json().get("detail", "")

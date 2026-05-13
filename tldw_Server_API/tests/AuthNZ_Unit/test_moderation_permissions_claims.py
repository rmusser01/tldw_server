from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import moderation as moderation_mod
from tldw_Server_API.app.core.AuthNZ.permissions import (
    MODERATION_AUDIT_READ,
    MODERATION_REVIEW_BULK_DECIDE,
    MODERATION_REVIEW_DECIDE,
    MODERATION_REVIEW_READ,
    SYSTEM_CONFIGURE,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


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
    moderation_service: object | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(moderation_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        if fail_with_401:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        assert principal is not None
        request.state.auth = AuthContext(
            principal=principal,
            ip=None,
            user_agent=None,
            request_id=None,
        )
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    class _StubModerationService:
        def list_user_overrides(self) -> dict:
            return {}

        def update_settings(self, **_: object) -> dict:
            return {
                "pii_enabled": None,
                "categories_enabled": None,
                "effective": {
                    "pii_enabled": False,
                    "categories_enabled": [],
                },
            }

    service = moderation_service or _StubModerationService()
    moderation_mod.get_moderation_service = lambda: service  # type: ignore[assignment]
    moderation_mod.get_moderation_review_service = lambda: service  # type: ignore[assignment]

    return app


def test_moderation_router_uses_standard_auth_factory_aliases() -> None:
    assert moderation_mod.RequireRole is auth_deps.RequireRole
    assert moderation_mod.RequirePermission is auth_deps.RequirePermission
    assert not hasattr(moderation_mod, "require_roles")
    assert not hasattr(moderation_mod, "require_permissions")


@pytest.mark.asyncio
async def test_moderation_users_401_when_principal_unavailable():
    app = _build_app(principal=None, fail_with_401=True)

    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/users")

    assert resp.status_code == 401
    assert "Authentication required" in resp.json().get("detail", "")


@pytest.mark.asyncio
async def test_moderation_users_403_without_admin_or_permission():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/users")

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_moderation_users_200_for_admin_with_permission():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
    )
    app = _build_app(principal=principal)

    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/users")

    assert resp.status_code == 200
    body = resp.json()
    assert body.get("overrides") == {}


@pytest.mark.asyncio
async def test_moderation_settings_put_403_without_admin_or_permission():
    principal = _make_principal(
        is_admin=False,
        roles=["user"],
        permissions=[],
    )
    app = _build_app(principal=principal)

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/moderation/settings",
            json={"pii_enabled": True, "persist": True},
        )

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_moderation_settings_put_500_for_runtime_persistence_failure():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
    )

    class _PersistFailModerationService:
        def list_user_overrides(self) -> dict:
            return {}

        def update_settings(self, **_: object) -> dict:
            return {
                "ok": False,
                "persisted": False,
                "error": "disk full",
                "error_type": "persistence",
            }

    app = _build_app(
        principal=principal,
        moderation_service=_PersistFailModerationService(),
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/moderation/settings",
            json={"pii_enabled": True, "persist": True},
        )

    assert resp.status_code == 500
    assert resp.json().get("detail") == "Failed to update moderation settings"


@pytest.mark.asyncio
async def test_moderation_settings_put_400_for_validation_failure():
    principal = _make_principal(
        is_admin=True,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
    )

    class _ValidationFailModerationService:
        def list_user_overrides(self) -> dict:
            return {}

        def update_settings(self, **_: object) -> dict:
            return {
                "ok": False,
                "persisted": False,
                "error": "categories_enabled contains invalid value",
                "error_type": "validation",
            }

    app = _build_app(
        principal=principal,
        moderation_service=_ValidationFailModerationService(),
    )

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/moderation/settings",
            json={"categories_enabled": ["bad"], "persist": False},
        )

    assert resp.status_code == 400
    assert resp.json().get("detail") == "categories_enabled contains invalid value"


@pytest.mark.asyncio
async def test_moderation_review_list_allows_reviewer_read_without_admin():
    principal = _make_principal(
        is_admin=False,
        roles=["reviewer"],
        permissions=[MODERATION_REVIEW_READ],
    )

    class _ReviewService:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def list_items(self, **_: object) -> dict:
            self.kwargs = _
            return {"items": [], "next_cursor": None, "total": 0}

    service = _ReviewService()
    app = _build_app(principal=principal, moderation_service=service)

    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/review/items?sort=oldest")

    assert resp.status_code == 200
    assert service.kwargs["sort"] == "oldest"
    assert resp.json() == {"items": [], "next_cursor": None, "total": 0}


@pytest.mark.asyncio
async def test_moderation_review_decision_requires_decide_permission_and_uses_principal_actor():
    principal = _make_principal(
        is_admin=False,
        roles=["reviewer"],
        permissions=[MODERATION_REVIEW_DECIDE],
    )

    class _ReviewService:
        def __init__(self) -> None:
            self.actor_id: str | None = None
            self.request_actor_id: str | None = None

        def record_decision(self, item_id: str, *, action: str, actor_id: str, reason: str | None = None, request_actor_id: str | None = None) -> dict:
            self.actor_id = actor_id
            self.request_actor_id = request_actor_id
            return {
                "item": {
                    "id": item_id,
                    "status": "approved",
                    "phase": "input",
                    "created_at": "2026-05-12T00:00:00Z",
                    "excerpt": "safe",
                    "effective_policy": {},
                    "matches": [],
                    "safe_fields": {"excerpt": True},
                },
                "decision": {
                    "id": "decision-1",
                    "item_id": item_id,
                    "action": action,
                    "status": "approved",
                    "previous_status": "needs_review",
                    "decided_by": actor_id,
                    "reason": reason,
                    "decided_at": "2026-05-12T00:00:00Z",
                    "undo_token": "undo-1",
                },
                "undo_token": "undo-1",
            }

    service = _ReviewService()
    app = _build_app(principal=principal, moderation_service=service)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/review/items/item-1/decision",
            json={"action": "approve", "reason": "ok", "actor_id": "spoofed"},
        )

    assert resp.status_code == 200
    assert service.actor_id == principal.principal_id
    assert service.request_actor_id == "spoofed"
    assert resp.json()["decision"]["decided_by"] == principal.principal_id


@pytest.mark.asyncio
async def test_moderation_review_decision_forbidden_with_read_only_permission():
    principal = _make_principal(
        is_admin=False,
        roles=["reviewer"],
        permissions=[MODERATION_REVIEW_READ],
    )
    app = _build_app(principal=principal)

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/review/items/item-1/decision",
            json={"action": "approve"},
        )

    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_moderation_review_bulk_and_audit_use_specific_permissions():
    bulk_principal = _make_principal(
        is_admin=False,
        roles=["reviewer"],
        permissions=[MODERATION_REVIEW_BULK_DECIDE],
    )

    class _ReviewService:
        def bulk_decision(self, *, item_ids: list[str], action: str, actor_id: str, reason: str | None = None) -> dict:
            return {"results": [{"item_id": item_ids[0], "ok": True}], "ok_count": 1, "error_count": 0}

        def list_audit(self, **_: object) -> dict:
            return {"events": [], "next_cursor": None}

    app = _build_app(principal=bulk_principal, moderation_service=_ReviewService())

    with TestClient(app) as client:
        bulk_resp = client.post(
            "/api/v1/moderation/review/bulk-decision",
            json={"item_ids": ["item-1"], "action": "dismiss"},
        )
        audit_resp = client.get("/api/v1/moderation/review/audit")

    assert bulk_resp.status_code == 200
    assert audit_resp.status_code == 403

    audit_principal = _make_principal(
        is_admin=False,
        roles=["reviewer"],
        permissions=[MODERATION_AUDIT_READ],
    )
    app = _build_app(principal=audit_principal, moderation_service=_ReviewService())

    with TestClient(app) as client:
        audit_resp = client.get("/api/v1/moderation/review/audit")

    assert audit_resp.status_code == 200

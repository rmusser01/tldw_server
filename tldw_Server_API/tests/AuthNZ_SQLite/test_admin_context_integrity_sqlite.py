from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


pytestmark = pytest.mark.unit


def _build_app(*, roles: list[str]) -> FastAPI:
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

    app = FastAPI()
    app.include_router(admin_router, prefix="/api/v1")

    async def _principal_override(request=None):  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="adminuser",
            token_type="access",
            jti=None,
            roles=roles,
            permissions=["system.configure"],
            is_admin="admin" in roles,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override
    return app


def test_admin_context_integrity_returns_boot_state() -> None:
    from tldw_Server_API.app.core.Context_Integrity.models import (
        ContextIntegrityBootState,
        ContextIntegrityFinding,
    )

    app = _build_app(roles=["admin"])
    app.state.context_integrity_boot_state = ContextIntegrityBootState(
        mode="enforce",
        degraded=False,
        manifest_sequence=7,
        manifest_digest="sha256:manifest",
        findings=(
            ContextIntegrityFinding(
                asset_id="prompt_file:rag.prompts.yaml",
                state="new_unapproved",
                severity="warning",
                summary="new",
                remediation="review",
                source_type="prompt_file",
            ),
        ),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/admin/context-integrity")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scope"] == "current_process"
    assert body["mode"] == "enforce"
    assert body["manifest_sequence"] == 7
    assert body["findings"][0]["asset_id"] == "prompt_file:rag.prompts.yaml"


def test_admin_context_integrity_is_admin_only() -> None:
    app = _build_app(roles=["user"])

    with TestClient(app) as client:
        response = client.get("/api/v1/admin/context-integrity")

    assert response.status_code == 403

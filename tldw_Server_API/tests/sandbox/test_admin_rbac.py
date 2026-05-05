from __future__ import annotations

import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client(monkeypatch) -> TestClient:


    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    # Ensure AuthNZ settings re-read after env change
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
        reset_settings()
    except Exception:
        _ = None
    existing_enable = os.environ.get("ROUTES_ENABLE", "")
    parts = [p.strip().lower() for p in existing_enable.split(",") if p.strip()]
    if "sandbox" not in parts:
        parts.append("sandbox")
    monkeypatch.setenv("ROUTES_ENABLE", ",".join(parts))
    # Build a minimal app with only the sandbox router
    from tldw_Server_API.app.api.v1.endpoints.sandbox import router as sandbox_router
    app = FastAPI()
    app.include_router(sandbox_router, prefix="/api/v1")
    return TestClient(app)


def _non_admin_dep():


    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
    return User(id=2, username="user", roles=["user"], is_admin=False)


@pytest.mark.unit
@pytest.mark.sandbox_no_auth
def test_admin_endpoints_require_admin_role(monkeypatch) -> None:
    with _client(monkeypatch) as client:
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        client.app.dependency_overrides[get_request_user] = _non_admin_dep
        for path in (
            "/api/v1/sandbox/admin/runs",
            "/api/v1/sandbox/admin/runtime-diagnostics",
            "/api/v1/sandbox/admin/macos-diagnostics",
            "/api/v1/sandbox/admin/macos-image-store/cleanup-plan",
            "/api/v1/sandbox/admin/idempotency",
            "/api/v1/sandbox/admin/usage",
        ):
            r = client.get(path)
            assert r.status_code in (401, 403), f"Expected RBAC to reject non-admin on {path}"
        r = client.post("/api/v1/sandbox/admin/macos-image-store/cleanup", json={})
        assert r.status_code in (401, 403), "Expected RBAC to reject non-admin on cleanup mutation surface"
        client.app.dependency_overrides.clear()

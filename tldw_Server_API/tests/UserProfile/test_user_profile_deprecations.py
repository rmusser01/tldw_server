from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def test_users_me_deprecation_headers(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/users/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.headers.get("Deprecation") == "true"
        assert resp.headers.get("Sunset")
        assert resp.headers.get("Link") == "</api/v1/users/me/profile>; rel=successor-version"
        payload = resp.json()
        assert payload.get("warning") == "deprecated_endpoint"
        assert payload.get("successor") == "/api/v1/users/me/profile"


def test_auth_me_deprecation_headers(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.headers.get("Deprecation") == "true"
        assert resp.headers.get("Sunset")
        assert resp.headers.get("Link") == "</api/v1/users/me/profile>; rel=successor-version"
        payload = resp.json()
        assert payload.get("warning") == "deprecated_endpoint"
        assert payload.get("successor") == "/api/v1/users/me/profile"


def test_legacy_me_endpoints_disabled(auth_headers, monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_LEGACY_USER_ME_ENDPOINTS", "false")
    with TestClient(app) as client:
        resp = client.get("/api/v1/users/me", headers=auth_headers)
        assert resp.status_code == 410
        payload = resp.json()
        assert payload.get("warning") == "deprecated_endpoint"
        assert payload.get("successor") == "/api/v1/users/me/profile"

        resp = client.get("/api/v1/auth/me", headers=auth_headers)
        assert resp.status_code == 410
        payload = resp.json()
        assert payload.get("warning") == "deprecated_endpoint"
        assert payload.get("successor") == "/api/v1/users/me/profile"


def test_users_me_rejects_missing_user_in_multi_user_mode(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        assert allow_missing is False
        raise HTTPException(status_code=404, detail="User not found")

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.principal_model.is_single_user_principal", lambda _principal: False)
    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)

    with TestClient(app) as client:
        resp = client.get("/api/v1/users/me", headers=auth_headers)

    assert resp.status_code == 404
    assert resp.json().get("detail") == "User not found"


def test_users_me_allows_missing_user_fallback_for_single_user(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        assert allow_missing is True
        return {
            "id": 1,
            "username": "single-user-fallback",
            "email": "fallback@example.invalid",
            "role": "user",
            "is_active": True,
            "is_verified": True,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.principal_model.is_single_user_principal", lambda _principal: True)
    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)

    with TestClient(app) as client:
        resp = client.get("/api/v1/users/me", headers=auth_headers)

    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("warning") == "deprecated_endpoint"
    assert payload.get("successor") == "/api/v1/users/me/profile"
    assert payload.get("username") == "single-user-fallback"

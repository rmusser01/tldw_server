from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _get_user_id(client: TestClient, auth_headers) -> int:
    resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert resp.status_code == 200
    return resp.json()["user"]["id"]


def test_user_profile_me_default(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
        assert resp.status_code == 200
        payload = resp.json()

    assert payload.get("profile_version")
    assert payload.get("catalog_version")
    assert payload.get("user", {}).get("id")
    assert "memberships" in payload
    assert "security" in payload
    assert "quotas" in payload
    assert "effective_config" in payload


def test_user_profile_sections_filter(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "identity,quotas"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        payload = resp.json()

    assert "user" in payload
    assert "quotas" in payload
    assert "memberships" not in payload
    assert "security" not in payload


def test_user_profile_effective_config(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "effective_config"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        payload = resp.json()

    effective_config = payload.get("effective_config", {})
    assert all(value is not None for value in effective_config.values())
    assert not payload.get("section_errors", {}).get("effective_config")


@pytest.mark.integration
def test_user_profile_preferences_section_returns_success(auth_headers) -> None:
    """Verify the preferences section loads without section-level errors."""
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "preferences"},
            headers=auth_headers,
        )
        assert resp.status_code == 200  # nosec B101 - pytest assertion
        payload = resp.json()

    assert isinstance(payload.get("preferences"), dict)  # nosec B101 - pytest assertion
    assert not payload.get("section_errors", {}).get("preferences")  # nosec B101 - pytest assertion


def test_admin_user_profile_default(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        resp = client.get(f"/api/v1/admin/users/{user_id}/profile", headers=auth_headers)
        assert resp.status_code == 200
        payload = resp.json()

    assert payload.get("user", {}).get("id") == user_id
    assert "memberships" in payload
    assert "security" in payload
    assert "quotas" in payload


def test_admin_user_profile_sections_filter(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"sections": "identity"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        payload = resp.json()

    assert "user" in payload
    assert "memberships" not in payload
    assert "security" not in payload
    assert "quotas" not in payload


def test_user_profile_quota_extensions(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        payload = resp.json()

    quotas = payload.get("quotas", {})
    assert "storage_quota_mb" in quotas
    assert "audio" in quotas
    assert "evaluations" in quotas
    assert "prompt_studio" in quotas
    assert "daily_minutes_limit" in quotas.get("audio", {})
    assert "limits" in quotas.get("evaluations", {})


def test_admin_profile_include_raw(auth_headers) -> None:
    with TestClient(app) as client:
        update_resp = client.patch(
            "/api/v1/users/me/profile",
            json={"updates": [{"key": "preferences.ui.theme", "value": "dark"}]},
            headers=auth_headers,
        )
        assert update_resp.status_code == 200
        user_id = _get_user_id(client, auth_headers)
        resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"include_raw": "true", "sections": "preferences"},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        payload = resp.json()

    raw_overrides = payload.get("raw_overrides", {})
    user_overrides = raw_overrides.get("user", [])
    assert any(
        entry.get("key") == "preferences.ui.theme" and entry.get("value") == "dark"
        for entry in user_overrides
    )


def test_user_profile_include_raw_forbidden(auth_headers) -> None:
    with TestClient(app) as client:
        resp = client.get(
            "/api/v1/users/me/profile",
            params={"include_raw": "true"},
            headers=auth_headers,
        )
        assert resp.status_code == 403


def test_admin_profile_mask_secrets_false(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            params={"mask_secrets": "false", "sections": "identity"},
            headers=auth_headers,
        )
        assert resp.status_code == 200


def test_user_profile_me_rejects_unverified_user(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {
            "id": 1,
            "username": "unverified-user",
            "email": "unverified@example.invalid",
            "role": "user",
            "is_active": True,
            "is_verified": False,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)

    with TestClient(app) as client:
        resp = client.get("/api/v1/users/me/profile", headers=auth_headers)

    assert resp.status_code == 403


def test_users_api_keys_reject_inactive_user(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {
            "id": 1,
            "username": "inactive-user",
            "email": "inactive@example.invalid",
            "role": "user",
            "is_active": False,
            "is_verified": True,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)

    with TestClient(app) as client:
        resp = client.get("/api/v1/users/api-keys", headers=auth_headers)

    assert resp.status_code == 403
    assert resp.json().get("detail") == "User account is inactive"


def test_users_sessions_reject_unverified_user(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {
            "id": 1,
            "username": "unverified-user",
            "email": "unverified@example.invalid",
            "role": "user",
            "is_active": True,
            "is_verified": False,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)

    with TestClient(app) as client:
        resp = client.get("/api/v1/users/sessions", headers=auth_headers)

    assert resp.status_code == 403
    assert resp.json().get("detail") == "Email verification required"


def test_users_storage_reject_inactive_user(auth_headers, monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {
            "id": 1,
            "username": "inactive-user",
            "email": "inactive@example.invalid",
            "role": "user",
            "is_active": False,
            "is_verified": True,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)

    with TestClient(app) as client:
        resp = client.get("/api/v1/users/storage", headers=auth_headers)

    assert resp.status_code == 403
    assert resp.json().get("detail") == "User account is inactive"

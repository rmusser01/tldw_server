from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.interactive_session_deps import require_interactive_session
from tldw_Server_API.app.core.AuthNZ.interactive_session import InteractiveSession
from tldw_Server_API.app.main import app


@pytest.mark.parametrize(
    "method,path,payload",
    [
        (
            "POST",
            "/api/v1/users/change-password",
            {"current_password": "Old@Pass#2024", "new_password": "New@Secure#2024!"},
        ),
        ("GET", "/api/v1/users/api-keys", None),
        ("GET", "/api/v1/users/sessions", None),
        ("POST", "/api/v1/users/sessions/revoke-all", None),
        ("GET", "/api/v1/users/storage", None),
        ("POST", "/api/v1/users/storage/recalculate", None),
        ("GET", "/api/v1/users/profile/catalog", None),
    ],
)
@pytest.mark.parametrize(
    "is_active,is_verified,expected_detail",
    [
        (False, True, "User account is inactive"),
        (True, False, "Email verification required"),
    ],
)
def test_user_management_endpoints_reject_inactive_or_unverified_users(
    auth_headers,
    monkeypatch,
    method: str,
    path: str,
    payload: dict | None,
    is_active: bool,
    is_verified: bool,
    expected_detail: str,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import users as users_endpoints

    async def _fake_resolve_user_context(_principal, *, allow_missing: bool = False):
        del allow_missing
        return {
            "id": 1,
            "username": "state-guard-user",
            "email": "state-guard@example.invalid",
            "role": "user",
            "is_active": is_active,
            "is_verified": is_verified,
            "storage_quota_mb": 5120,
            "storage_used_mb": 0.0,
            "created_at": datetime.utcnow(),
            "last_login": None,
        }

    monkeypatch.setattr(users_endpoints, "_resolve_user_context", _fake_resolve_user_context)
    # Ending sessions needs a browser session (API keys get 403 first); this test is about the
    # account-state guard, so present one.
    monkeypatch.setitem(
        app.dependency_overrides, require_interactive_session, lambda: InteractiveSession(user_id=1, session_id=1)
    )

    with TestClient(app) as client:
        request_kwargs: dict[str, object] = {"headers": auth_headers}
        if payload is not None:
            request_kwargs["json"] = payload
        resp = client.request(method, path, **request_kwargs)

    assert resp.status_code == 403
    assert resp.json().get("detail") == expected_detail

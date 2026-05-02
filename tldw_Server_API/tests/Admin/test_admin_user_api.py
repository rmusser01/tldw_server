from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _setup_env(monkeypatch, *, user_db_base: str) -> None:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("USER_DB_BASE_DIR", user_db_base)
    auth_db_path = Path(user_db_base).parent / "users_test_admin_user_api.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{auth_db_path}")
    monkeypatch.setenv("TEST_MODE", "true")


@pytest.mark.asyncio
async def test_list_users_forwards_admin_capable_filter(monkeypatch, tmp_path):
    _setup_env(monkeypatch, user_db_base=str(tmp_path / "user_dbs"))

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services import admin_users_service

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    async def _fake_list_users(
        principal,
        *,
        page: int,
        limit: int,
        role: str | None,
        admin_capable: bool,
        is_active: bool | None,
        mfa_enabled: bool | None = None,
        search: str | None,
        org_id: int | None,
    ):
        assert page == 1
        assert limit == 25
        assert role is None
        assert admin_capable is True
        assert is_active is None
        assert mfa_enabled is None
        assert search is None
        assert org_id is None
        return (
            [
                {
                    "id": 7,
                    "uuid": "123e4567-e89b-12d3-a456-426614174000",
                    "username": "alice",
                    "email": "alice@example.com",
                    "role": "user",
                    "is_active": True,
                    "is_verified": True,
                    "mfa_enabled": False,
                    "created_at": "2026-03-12T00:00:00Z",
                    "last_login": None,
                    "storage_quota_mb": 1024,
                    "storage_used_mb": 0,
                }
            ],
            1,
        )

    monkeypatch.setattr(admin_users_service, "list_users", _fake_list_users)

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}
    with TestClient(app, headers=headers) as client:
        response = client.get("/api/v1/admin/users", params={"limit": 25, "admin_capable": "true"})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 1
    assert payload["users"][0]["id"] == 7
    assert payload["pagination"]["total"] == 1
    assert payload["pagination"]["limit"] == 25
    assert payload["pagination"]["offset"] == 0
    assert payload["pagination"]["has_more"] is False
    assert payload["has_more"] is False
    assert payload["next_offset"] is None

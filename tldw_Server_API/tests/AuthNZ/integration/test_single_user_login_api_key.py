"""Single-user login should return the configured API key for auth."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.Audit.unified_audit_service import shutdown_audit_service
from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db
from tldw_Server_API.app.services.registration_service import reset_registration_service
from tldw_Server_API.tests.helpers.app_main_state import reload_app_main


pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def single_user_client(tmp_path, monkeypatch, fresh_services):
    db_path = tmp_path / "authnz_single_user.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test_single_user_api_key_123")
    monkeypatch.setenv("DEFER_HEAVY_STARTUP", "true")
    monkeypatch.setenv("TEST_MODE", "true")

    reset_settings()
    await reset_registration_service()
    await shutdown_audit_service()
    await reset_users_db()

    await bootstrap_single_user_profile()

    password = "StrongSU@9xQ4!"
    password_hash = PasswordService().hash_password(password)
    settings = get_settings()
    pool = await get_db_pool()
    async with pool.transaction() as conn:
        if getattr(pool, "pool", None):
            await conn.execute(
                "UPDATE users SET password_hash = $1 WHERE id = $2",
                password_hash,
                settings.SINGLE_USER_FIXED_ID,
            )
        else:
            await conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (password_hash, settings.SINGLE_USER_FIXED_ID),
            )

    app = reload_app_main().app

    with TestClient(app) as client:
        yield client, password, settings.SINGLE_USER_API_KEY

    reset_settings()
    await reset_registration_service()
    await shutdown_audit_service()
    await reset_users_db()


def test_single_user_login_returns_api_key_and_authenticates(single_user_client):
    client, password, api_key = single_user_client

    resp = client.post(
        "/api/v1/auth/login",
        data={"username": "single_user", "password": password},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["access_token"] == api_key
    assert data["refresh_token"] == api_key

    sessions_resp = client.get(
        "/api/v1/auth/sessions",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert sessions_resp.status_code == 200


def test_single_user_refresh_requires_api_key(single_user_client):
    client, password, api_key = single_user_client

    # Refresh with the configured API key should succeed.
    ok = client.post("/api/v1/auth/refresh", json={"refresh_token": api_key})
    assert ok.status_code == 200
    ok_json = ok.json()
    assert ok_json["access_token"] == api_key
    assert ok_json["refresh_token"] == api_key

    # Legacy single-user refresh tokens are no longer accepted.
    bad = client.post("/api/v1/auth/refresh", json={"refresh_token": "single-user-refresh-1"})
    assert bad.status_code == 401

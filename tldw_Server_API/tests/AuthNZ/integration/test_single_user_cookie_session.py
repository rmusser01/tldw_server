"""HTTP coverage for persistent single-user cookie sessions."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Audit.unified_audit_service import shutdown_audit_service
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db
from tldw_Server_API.app.services.registration_service import reset_registration_service
from tldw_Server_API.tests.helpers.app_main_state import reload_app_main

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def single_user_cookie_client(tmp_path, monkeypatch):
    db_path = tmp_path / "authnz_single_user_cookie.db"
    api_key = "test_single_user_cookie_api_key_123"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("SINGLE_USER_API_KEY", api_key)
    monkeypatch.setenv("SESSION_ENCRYPTION_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "false")
    monkeypatch.setenv("CSRF_ENABLED", "1")
    monkeypatch.setenv("DEFER_HEAVY_STARTUP", "true")
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTHNZ_FORCE_REAL_SESSION_MANAGER", "1")

    await reset_db_pool()
    await reset_session_manager()
    reset_settings()
    await reset_registration_service()
    await shutdown_audit_service()
    await reset_users_db()

    await bootstrap_single_user_profile()
    app = reload_app_main().app

    with TestClient(app) as client:
        yield client, get_settings().SINGLE_USER_API_KEY

    await reset_db_pool()
    await reset_session_manager()
    reset_settings()
    await reset_registration_service()
    await shutdown_audit_service()
    await reset_users_db()


def _mint(client: TestClient, api_key: str):
    return client.post(
        "/api/v1/auth/single-user/session",
        headers={"X-API-KEY": api_key},
    )


def test_mint_returns_no_token_and_sets_exact_cookie(single_user_cookie_client):
    client, api_key = single_user_cookie_client

    response = _mint(client, api_key)

    assert response.status_code == 200
    assert set(response.json()) == {"authenticated", "expires_at"}
    cookie = response.headers["set-cookie"]
    assert "tldw_single_user_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/api" in cookie
    assert api_key not in response.text + cookie


def test_cookie_principal_authenticates_existing_http_dependencies(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    assert _mint(client, api_key).status_code == 200

    response = client.get("/api/v1/auth/sessions")

    assert response.status_code == 200


def test_cookie_mutation_requires_csrf_but_api_key_does_not(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    mint = _mint(client, api_key)
    assert mint.status_code == 200

    assert client.delete("/api/v1/auth/single-user/session").status_code == 403
    assert _mint(client, api_key).status_code == 200


def test_cookie_only_mint_is_rejected_after_csrf_validation(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    assert _mint(client, api_key).status_code == 200
    csrf_token = client.cookies["csrf_token"]

    response = client.post(
        "/api/v1/auth/single-user/session",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 401


def test_valid_cookie_session_is_reused_on_mint(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    first = _mint(client, api_key)
    assert first.status_code == 200
    cookie_before = client.cookies["tldw_single_user_session"]

    second = _mint(client, api_key)

    assert second.status_code == 200
    assert second.json()["expires_at"] == first.json()["expires_at"]
    assert client.cookies["tldw_single_user_session"] == cookie_before


def test_csrf_disabled_refuses_cookie_mint(single_user_cookie_client, monkeypatch):
    client, api_key = single_user_cookie_client
    monkeypatch.setenv("CSRF_ENABLED", "0")

    response = _mint(client, api_key)

    assert response.status_code == 503


def test_logout_revokes_only_current_cookie_session(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    first = _mint(client, api_key)
    assert first.status_code == 200
    first_cookie = client.cookies["tldw_single_user_session"]
    first_csrf = client.cookies["csrf_token"]

    client.cookies.clear()
    second = _mint(client, api_key)
    assert second.status_code == 200
    second_cookie = client.cookies["tldw_single_user_session"]
    assert second_cookie != first_cookie

    client.cookies.clear()
    client.cookies.set("tldw_single_user_session", first_cookie, path="/api")
    client.cookies.set("csrf_token", first_csrf, path="/")
    logout = client.delete(
        "/api/v1/auth/single-user/session",
        headers={"X-CSRF-Token": first_csrf},
    )
    assert logout.status_code == 200
    assert logout.json() == {"authenticated": False}
    assert "tldw_single_user_session=" in logout.headers["set-cookie"]
    assert "Max-Age=0" in logout.headers["set-cookie"]

    client.cookies.clear()
    client.cookies.set("tldw_single_user_session", first_cookie, path="/api")
    assert client.get("/api/v1/auth/sessions").status_code == 401

    client.cookies.clear()
    client.cookies.set("tldw_single_user_session", second_cookie, path="/api")
    assert client.get("/api/v1/auth/sessions").status_code == 200

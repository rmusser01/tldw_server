"""MFA login over HTTP on SQLite: the real request-transaction path.

Regression: ``/auth/mfa/login`` took ``get_db_transaction`` (BEGIN IMMEDIATE) and then
the session manager wrote on a second connection, deadlocking on the SQLite write lock.
"""

import uuid
from pathlib import Path

import pyotp
import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_mfa_login_http_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")
    monkeypatch.setenv("TOTP_ENCRYPTION_KEY", "")
    monkeypatch.setenv("CSRF_ENABLED", "false")
    monkeypatch.delenv("REDIS_URL", raising=False)
    # Reach the real get_db_transaction instead of the test-mode adapter.
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.mfa_service import get_mfa_service
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB, reset_users_db
    from tldw_Server_API.app.core.Utils.app_services import app_services

    reset_settings()
    app_services().jwt_service = None
    await reset_db_pool()
    await reset_users_db()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    users_db = UsersDB(pool)
    await users_db.initialize()

    password = "Kq7#vLm2$Zp9wRt"
    user = await users_db.create_user(
        username="mfa-user",
        email="mfa-user@example.com",
        password_hash=PasswordService().hash_password(password),
        role="user",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    secret = pyotp.random_base32()
    mfa = get_mfa_service()
    await mfa.initialize()
    assert await mfa.enable_mfa(int(user["id"]), secret, ["backup-code-1"])

    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/auth/login", data={"username": "mfa-user", "password": password})
        assert r.status_code in (200, 202), r.text
        body = r.json()
        assert body.get("mfa_required") is True, body
        r = client.post(
            "/api/v1/auth/mfa/login",
            json={"session_token": body["session_token"], "mfa_token": pyotp.TOTP(secret).now()},
        )
        assert r.status_code == 200, r.text
        tokens = r.json()
        assert tokens["access_token"] and tokens["refresh_token"]

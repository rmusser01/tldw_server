"""MFA login over HTTP on SQLite: the real request-transaction path.

Regression: ``/auth/mfa/login`` took ``get_db_transaction`` (BEGIN IMMEDIATE) and then
the session manager wrote on a second connection, deadlocking on the SQLite write lock.
"""

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyotp
import pytest
from fastapi.testclient import TestClient
from jose import jwt as jose_jwt

from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

PASSWORD = "Kq7#vLm2$Zp9wRt"


async def _mfa_user(tmp_path, monkeypatch):
    """Return ``(pool, user_id, secret)`` for a fresh SQLite user with TOTP enabled."""
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")
    monkeypatch.setenv("TOTP_ENCRYPTION_KEY", "")
    monkeypatch.setenv("CSRF_ENABLED", "false")
    monkeypatch.delenv("REDIS_URL", raising=False)
    # Reach the real get_db_transaction instead of the test-mode adapter.
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    from tldw_Server_API.app.core.AuthNZ import mfa_service as mfa_module
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB, reset_users_db
    from tldw_Server_API.app.core.Utils.app_services import app_services

    reset_settings()
    app_services().jwt_service = None
    # The MFA service caches its pool; a previous test's would point at another file.
    monkeypatch.setattr(mfa_module, "_mfa_service", None)
    await reset_db_pool()
    await reset_users_db()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    users_db = UsersDB(pool)
    await users_db.initialize()

    user = await users_db.create_user(
        username="mfa-user",
        email="mfa-user@example.com",
        password_hash=PasswordService().hash_password(PASSWORD),
        role="user",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    secret = pyotp.random_base32()
    mfa = mfa_module.get_mfa_service()
    await mfa.initialize()
    assert await mfa.enable_mfa(int(user["id"]), secret, ["backup-code-1"])
    return pool, int(user["id"]), secret


@pytest.mark.asyncio
async def test_mfa_login_http_sqlite(tmp_path, monkeypatch):
    pool, uid, secret = await _mfa_user(tmp_path, monkeypatch)
    sessions = AuthnzSessionsRepo(pool)

    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/auth/login", data={"username": "mfa-user", "password": PASSWORD})
        assert r.status_code in (200, 202), r.text
        body = r.json()
        assert body.get("mfa_required") is True, body
        # The password step alone holds no session row.
        assert await sessions.get_active_sessions_for_user(uid) == []
        r = client.post(
            "/api/v1/auth/mfa/login",
            json={"session_token": body["session_token"], "mfa_token": pyotp.TOTP(secret).now()},
        )
        assert r.status_code == 200, r.text
        tokens = r.json()
        assert tokens["access_token"] and tokens["refresh_token"]
        live = await sessions.get_active_sessions_for_user(uid)
        assert len(live) == 1
        assert live[0]["id"] == jose_jwt.get_unverified_claims(tokens["access_token"])["session_id"]


@pytest.mark.asyncio
async def test_password_step_without_second_factor_evicts_nothing(tmp_path, monkeypatch):
    # With a cap of one, any row the password step created would evict the live session.
    monkeypatch.setenv("MAX_SESSIONS_PER_USER", "1")
    pool, uid, _secret = await _mfa_user(tmp_path, monkeypatch)
    sessions = AuthnzSessionsRepo(pool)
    now = datetime.now(timezone.utc)
    existing = await sessions.create_session_record(
        user_id=uid,
        token_hash="h-existing",
        refresh_token_hash="r-existing",
        encrypted_token="enc",
        encrypted_refresh="enc",
        expires_at=now + timedelta(hours=1),
        refresh_expires_at=now + timedelta(days=1),
        ip_address="127.0.0.1",
        user_agent="pytest",
        device_id=None,
        access_jti="aj-existing",
        refresh_jti="rj-existing",
    )

    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        r = client.post("/api/v1/auth/login", data={"username": "mfa-user", "password": PASSWORD})
        assert r.status_code == 202, r.text
        assert [row["id"] for row in await sessions.get_active_sessions_for_user(uid)] == [existing]

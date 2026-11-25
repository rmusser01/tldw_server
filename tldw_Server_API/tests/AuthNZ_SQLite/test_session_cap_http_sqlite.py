"""Session cap over HTTP on SQLite: install grouping and the session-list routes (TASK-15628)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

pytestmark = pytest.mark.asyncio

PASSWORD = "Kq7#vLm2$Zp9wRt"


@pytest.fixture
async def multi_user_client(tmp_path, monkeypatch, fresh_services):
    """Yield ``(client, user_id, password)`` for a verified user of a fresh multi-user SQLite app."""
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")
    monkeypatch.setenv("CSRF_ENABLED", "false")
    monkeypatch.delenv("REDIS_URL", raising=False)
    reset_settings()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    user_id = await create_authnz_test_user(
        pool,
        username="cap-user",
        email="cap-user@example.com",
        password_hash=PasswordService().hash_password(PASSWORD),
        is_verified=True,
    )

    from tldw_Server_API.app.main import app

    try:
        with TestClient(app) as client:
            yield client, user_id, PASSWORD
    finally:
        reset_settings()


def _login(client: TestClient, password: str, device: str | None = None) -> dict:
    headers = {"X-TLDW-Device-Id": device} if device is not None else {}
    response = client.post(
        "/api/v1/auth/login",
        data={"username": "cap-user", "password": password},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _bearer(tokens: dict) -> dict:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


async def _insert_row(user_id: int, tag: str, *, access_in: timedelta, refresh_in: timedelta) -> int:
    """Insert a session via the uncapped repo path; ``*_in`` are offsets from now."""
    now = datetime.now(timezone.utc)
    return await AuthnzSessionsRepo(await get_db_pool()).create_session_record(
        user_id=user_id,
        token_hash=f"h-{tag}",
        refresh_token_hash=f"r-{tag}",
        encrypted_token="enc",
        encrypted_refresh="enc",
        expires_at=now + access_in,
        refresh_expires_at=now + refresh_in,
        ip_address="127.0.0.1",
        user_agent="pytest",
        device_id=None,
        access_jti=f"aj-{tag}",
        refresh_jti=f"rj-{tag}",
    )


async def test_same_install_login_replaces_only_its_own_session(multi_user_client):
    client, uid, pw = multi_user_client
    a, _b, _c = (_login(client, pw, d) for d in ("dev-aaaaaaaa", "dev-bbbbbbbb", "dev-cccccccc"))
    for _ in range(5):
        _login(client, pw, "dev-dddddddd")

    rows = await AuthnzSessionsRepo(await get_db_pool()).get_active_sessions_for_user(uid)

    assert Counter(r["device_id"] for r in rows) == {
        "dev-aaaaaaaa": 1,
        "dev-bbbbbbbb": 1,
        "dev-cccccccc": 1,
        "dev-dddddddd": 1,
    }
    assert client.get("/api/v1/auth/sessions", headers=_bearer(a)).status_code == 200


async def test_users_sessions_alias_lists_and_revokes_idle_refreshable_session(multi_user_client):
    client, uid, pw = multi_user_client
    tok = _login(client, pw)
    idle = await _insert_row(uid, "idle", access_in=timedelta(days=-2), refresh_in=timedelta(days=5))
    dead = await _insert_row(uid, "dead", access_in=timedelta(days=-3), refresh_in=timedelta(hours=-1))

    for path in ("/api/v1/users/sessions", "/api/v1/auth/sessions"):
        response = client.get(path, headers=_bearer(tok))
        assert response.status_code == 200, response.text
        listed = {s["id"] for s in response.json()}
        assert idle in listed and dead not in listed

    assert client.delete(f"/api/v1/users/sessions/{idle}", headers=_bearer(tok)).status_code == 200
    assert idle not in {s["id"] for s in client.get("/api/v1/users/sessions", headers=_bearer(tok)).json()}

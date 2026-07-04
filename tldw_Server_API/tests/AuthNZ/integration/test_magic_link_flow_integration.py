"""Integration tests for magic link auth flows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import asyncpg
import pytest

from tldw_Server_API.tests.helpers.pg_env import get_pg_env

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]
pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_pg = get_pg_env()
TEST_DB_HOST = _pg.host
TEST_DB_PORT = _pg.port
TEST_DB_USER = _pg.user
TEST_DB_PASSWORD = _pg.password


class _StubEmailService:
    def __init__(self) -> None:
        self.last_token: str | None = None
        self.sent: list[dict[str, Any]] = []

    async def send_magic_link_email(
        self,
        to_email: str,
        magic_token: str,
        expires_in_minutes: int,
        username: str | None = None,
        ip_address: str | None = None,
        base_url: str | None = None,
    ) -> bool:
        self.last_token = magic_token
        self.sent.append(
            {
                "to": to_email,
                "token": magic_token,
                "expires_in_minutes": expires_in_minutes,
                "username": username,
                "ip_address": ip_address,
                "base_url": base_url,
            }
        )
        return True


async def _fetch_user_and_memberships(db_name: str, email: str):
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=db_name,
    )
    try:
        user = await conn.fetchrow("SELECT id, is_verified FROM users WHERE email = $1", email)
        memberships = []
        if user:
            memberships = await conn.fetch(
                "SELECT role FROM org_members WHERE user_id = $1",
                user["id"],
            )
        return user, memberships
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_magic_link_verify_creates_user_and_org(isolated_test_environment, monkeypatch):
    client, db_name = isolated_test_environment
    email = "magicuser@example.com"

    stub_email = _StubEmailService()
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    monkeypatch.setattr(auth, "_get_email_service", lambda: stub_email)
    monkeypatch.setattr(
        auth,
        "get_input_validator",
        lambda: SimpleNamespace(validate_email=lambda _e: (True, None)),
    )

    resp = client.post("/api/v1/auth/magic-link/request", json={"email": email})
    assert resp.status_code == 200
    assert stub_email.last_token

    verify = client.post("/api/v1/auth/magic-link/verify", json={"token": stub_email.last_token})
    assert verify.status_code == 200

    # Token is one-time; reusing should fail.
    replay = client.post("/api/v1/auth/magic-link/verify", json={"token": stub_email.last_token})
    assert replay.status_code == 400

    user, memberships = await _fetch_user_and_memberships(db_name, email)
    assert user is not None
    assert user["is_verified"] is True
    assert memberships
    assert any(m["role"] == "owner" for m in memberships)


@pytest.mark.asyncio
async def test_magic_link_request_respects_rate_limit(isolated_test_environment, monkeypatch):
    client, _db_name = isolated_test_environment
    email = "throttled@example.com"

    stub_email = _StubEmailService()
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    monkeypatch.setattr(auth, "_get_email_service", lambda: stub_email)
    monkeypatch.setattr(
        auth,
        "get_input_validator",
        lambda: SimpleNamespace(validate_email=lambda _e: (True, None)),
    )

    # First RG check (ip scope) should pass, second (email scope) should deny.
    calls = {"count": 0}
    async def _rg_stub(*args, **kwargs):
        calls["count"] += 1
        return (calls["count"] == 1), 1
    monkeypatch.setattr(auth, "_reserve_auth_rg_requests", _rg_stub)

    resp = client.post("/api/v1/auth/magic-link/request", json={"email": email})
    assert resp.status_code == 200
    assert len(stub_email.sent) == 0


@pytest.mark.asyncio
async def test_magic_link_verify_rejects_inactive_user(isolated_test_environment, monkeypatch):
    client, db_name = isolated_test_environment
    email = "inactive@example.com"

    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=db_name,
    )
    try:
        from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
        import uuid as uuid_lib

        password_hash = PasswordService().hash_password("Inactive@Pass#2024!")
        await conn.execute(
            """
            INSERT INTO users (
                uuid, username, email, password_hash, role,
                is_active, is_verified, storage_quota_mb, storage_used_mb
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            str(uuid_lib.uuid4()),
            "inactiveuser",
            email,
            password_hash,
            "user",
            False,
            True,
            5120,
            0.0,
        )
    finally:
        await conn.close()

    stub_email = _StubEmailService()
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    monkeypatch.setattr(auth, "_get_email_service", lambda: stub_email)
    monkeypatch.setattr(
        auth,
        "get_input_validator",
        lambda: SimpleNamespace(validate_email=lambda _e: (True, None)),
    )

    resp = client.post("/api/v1/auth/magic-link/request", json={"email": email})
    assert resp.status_code == 200
    assert stub_email.last_token

    verify = client.post("/api/v1/auth/magic-link/verify", json={"token": stub_email.last_token})
    assert verify.status_code == 401
    assert "inactive" in verify.json().get("detail", "").lower()

    _user, memberships = await _fetch_user_and_memberships(db_name, email)
    assert memberships == []

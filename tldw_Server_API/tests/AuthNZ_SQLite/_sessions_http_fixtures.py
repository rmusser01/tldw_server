"""Shared helpers for the sign-in alert / active-sessions HTTP tests (not a test module)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo


@dataclass(frozen=True)
class SessionsCtx:
    client: TestClient
    alice_id: int
    bob_id: int
    password: str


def login(client: TestClient, username: str, password: str, *, device: str | None = None, user_agent: str | None = None) -> dict:
    headers = {}
    if device is not None:
        headers["X-TLDW-Device-Id"] = device
    if user_agent is not None:
        headers["User-Agent"] = user_agent
    response = client.post("/api/v1/auth/login", data={"username": username, "password": password}, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def bearer(tokens: dict) -> dict:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


def api_key_for(client: TestClient, tokens: dict) -> str:
    response = client.post("/api/v1/users/api-keys", json={"name": "t", "scope": "write"}, headers=bearer(tokens))
    assert response.status_code == 201, response.text
    return response.json()["key"]


def session_id_of(tokens: dict) -> int:
    return int(get_jwt_service().decode_access_token(tokens["access_token"])["session_id"])


async def seed_session_revoked_with_reason(user_id: int, reason: str) -> int:
    """Insert a session row and revoke it with ``reason``; return its id."""
    repo = AuthnzSessionsRepo(await get_db_pool())
    now = datetime.now(timezone.utc)
    tag = uuid4().hex
    session_id = await repo.create_session_record(
        user_id=user_id,
        token_hash=f"h-{tag}",
        refresh_token_hash=f"r-{tag}",
        encrypted_token="enc",
        encrypted_refresh="enc",
        expires_at=now + timedelta(minutes=30),
        refresh_expires_at=now + timedelta(days=1),
        ip_address="127.0.0.1",
        user_agent="seed",
        device_id=f"seed-{tag[:8]}",
        access_jti=None,
        refresh_jti=None,
    )
    await repo.revoke_session_record(session_id=session_id, expected_user_id=user_id, revoked_by=user_id, reason=reason)
    return session_id


async def revoke_reason_of(session_id: int) -> str | None:
    pool = await get_db_pool()
    row = await pool.fetchone("SELECT revoke_reason FROM sessions WHERE id = ?", session_id)
    return row["revoke_reason"] if row else None

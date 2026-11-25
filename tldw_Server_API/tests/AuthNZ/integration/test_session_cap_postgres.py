"""Live-session predicate and cap-row cleanup against isolated PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytestmark = pytest.mark.integration


async def _create_user(pool, username: str) -> int:
    users_db = UsersDB(pool)
    await users_db.initialize()
    created = await users_db.create_user(
        username=username,
        email=f"{username}@example.test",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
    )
    return int(created["id"])


async def _row(repo, uid, tag, *, access_in, refresh_in) -> int:
    now = datetime.now(timezone.utc)
    return await repo.create_session_record(
        user_id=uid,
        token_hash=f"h-{tag}",
        refresh_token_hash=f"r-{tag}",
        encrypted_token="enc",
        encrypted_refresh="enc",
        expires_at=now + access_in,
        refresh_expires_at=now + refresh_in,
        ip_address="127.0.0.1",
        user_agent="pytest-pg",
        device_id=None,
        access_jti=f"aj-{tag}",
        refresh_jti=f"rj-{tag}",
    )


async def _revoke(pool, repo, sid, reason, minutes_ago=0) -> None:
    await repo.revoke_session_record(session_id=sid, revoked_by=None, reason=reason)
    await pool.execute(
        "UPDATE sessions SET revoked_at = $1 WHERE id = $2",
        (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).replace(tzinfo=None),
        sid,
    )


@pytest.mark.asyncio
async def test_live_predicate_and_cleanup_postgres(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    uid = await _create_user(pool, "pg_session_cap")
    repo = AuthnzSessionsRepo(pool)
    h, d = timedelta(hours=1), timedelta(days=1)

    live = await _row(repo, uid, "live", access_in=h, refresh_in=7 * d)
    idle = await _row(repo, uid, "idle", access_in=-2 * d, refresh_in=5 * d)
    dead = await _row(repo, uid, "dead", access_in=-3 * d, refresh_in=-2 * d)
    revoked = await _row(repo, uid, "revoked", access_in=h, refresh_in=7 * d)
    await _revoke(pool, repo, revoked, "User logout")

    assert {r["id"] for r in await repo.get_active_sessions_for_user(uid)} == {live, idle}

    cap_61 = await _row(repo, uid, "cap61", access_in=h, refresh_in=7 * d)
    cap_30 = await _row(repo, uid, "cap30", access_in=h, refresh_in=7 * d)
    await _revoke(pool, repo, cap_61, "session_cap", 61)
    await _revoke(pool, repo, cap_30, "session_cap", 30)

    await repo.cleanup_expired_sessions(session_cap_retention=timedelta(hours=1))

    rows = await pool.fetch("SELECT id FROM sessions WHERE user_id = $1", uid)
    remaining = {int(r["id"]) for r in rows}
    assert cap_61 not in remaining and cap_30 in remaining
    assert dead not in remaining and idle in remaining
    assert revoked in remaining

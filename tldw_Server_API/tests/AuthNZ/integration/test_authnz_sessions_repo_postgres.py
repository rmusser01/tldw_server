from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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


@pytest.mark.asyncio
async def test_authnz_sessions_repo_validation_and_refresh_postgres(
    isolated_test_environment,
):
    """AuthnzSessionsRepo validation/refresh helpers should work on Postgres."""
    _client, _db_name = isolated_test_environment
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    user_id = await _create_user(pool, "pg_sessions_user")

    repo = AuthnzSessionsRepo(pool)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=1)
    refresh_expires = now + timedelta(days=7)

    # Create a session record with known hashes
    session_id = await repo.create_session_record(
        user_id=user_id,
        token_hash="hash-access",
        refresh_token_hash="hash-refresh",
        encrypted_token="enc-access",
        encrypted_refresh="enc-refresh",
        expires_at=expires_at,
        refresh_expires_at=refresh_expires,
        ip_address="127.0.0.1",
        user_agent="pytest-pg",
        device_id="pg-device-1",
        access_jti="access-jti",
        refresh_jti="refresh-jti",
    )
    assert session_id > 0

    # Validation helpers: by id and by token hash
    by_id = await repo.fetch_session_for_validation_by_id(session_id)
    assert by_id is not None
    assert by_id["id"] == session_id
    assert by_id["user_id"] == user_id
    assert bool(by_id["user_active"]) is True

    by_hash = await repo.fetch_session_for_validation_by_token_hash("hash-access")
    assert by_hash is not None
    assert by_hash["id"] == session_id

    missing = await repo.fetch_session_for_validation_by_id(session_id + 1)
    assert missing is None

    # Refresh helpers: find by refresh hash candidates and update tokens
    found = await repo.find_active_session_by_refresh_hash_candidates(["does-not-exist", "hash-refresh"])
    assert found is not None
    assert found["id"] == session_id
    assert found["user_id"] == user_id
    assert found["token_hash"] == "hash-access"
    assert found["refresh_token_hash"] == "hash-refresh"

    new_expires = now + timedelta(hours=2)
    updated = await repo.update_session_tokens_for_refresh(
        session_id=found["id"],
        expected_access_hash=found["token_hash"],
        expected_refresh_hash=found["refresh_token_hash"],
        new_access_hash="hash-access-new",
        access_jti="access-jti-new",
        expires_at=new_expires,
        encrypted_access_token="enc-access-new",
        refresh_hash_update="hash-refresh-new",
        refresh_jti="refresh-jti-new",
        refresh_expires_at=refresh_expires,
        encrypted_refresh_token="enc-refresh-new",
    )
    assert updated is True

    row = await pool.fetchrow(
        """
        SELECT token_hash,
               refresh_token_hash,
               access_jti,
               refresh_jti,
               encrypted_token,
               encrypted_refresh
        FROM sessions
        WHERE id = $1
        """,
        session_id,
    )
    assert row is not None
    assert row["token_hash"] == "hash-access-new"
    assert row["refresh_token_hash"] == "hash-refresh-new"
    assert row["access_jti"] == "access-jti-new"
    assert row["refresh_jti"] == "refresh-jti-new"
    assert row["encrypted_token"] == "enc-access-new"
    assert row["encrypted_refresh"] == "enc-refresh-new"

    stale_update = await repo.update_session_tokens_for_refresh(
        session_id=found["id"],
        expected_access_hash="hash-access",
        expected_refresh_hash="hash-refresh",
        new_access_hash="hash-access-stale",
        access_jti="access-jti-stale",
        expires_at=new_expires,
        encrypted_access_token="enc-access-stale",
        refresh_hash_update="hash-refresh-stale",
        refresh_jti="refresh-jti-stale",
        refresh_expires_at=refresh_expires,
        encrypted_refresh_token="enc-refresh-stale",
    )
    assert stale_update is False


@pytest.mark.asyncio
async def test_authnz_sessions_repo_bulk_revocation_postgres(
    isolated_test_environment,
):
    """AuthnzSessionsRepo bulk revocation helpers should work on Postgres."""
    _client, _db_name = isolated_test_environment
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    user_id = await _create_user(pool, "pg_sessions_bulk_user")

    repo = AuthnzSessionsRepo(pool)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=1)
    refresh_expires = now + timedelta(days=7)

    for idx in range(2):
        await repo.create_session_record(
            user_id=user_id,
            token_hash=f"hash-access-bulk-{idx}",
            refresh_token_hash=f"hash-refresh-bulk-{idx}",
            encrypted_token=f"enc-access-bulk-{idx}",
            encrypted_refresh=f"enc-refresh-bulk-{idx}",
            expires_at=expires_at,
            refresh_expires_at=refresh_expires,
            ip_address="127.0.0.1",
            user_agent=f"pytest-pg-bulk-{idx}",
            device_id=f"pg-device-bulk-{idx}",
            access_jti=f"access-jti-bulk-{idx}",
            refresh_jti=f"refresh-jti-bulk-{idx}",
        )

    sessions = await repo.fetch_session_token_metadata_for_user(user_id)
    assert len(sessions) >= 2

    affected = await repo.mark_sessions_revoked_for_user_with_audit(
        user_id=user_id,
        revoked_by=user_id,
        reason="bulk-logout",
    )
    assert affected >= 2

    rows = await pool.fetch(
        """
        SELECT is_active, is_revoked, revoked_by, revoke_reason
        FROM sessions
        WHERE user_id = $1
        """,
        user_id,
    )
    assert rows
    for row in rows:
        is_active = row["is_active"]
        is_revoked = row["is_revoked"]
        revoked_by = row["revoked_by"]
        revoke_reason = row["revoke_reason"]
        assert not bool(is_active)
        assert bool(is_revoked)
        assert revoked_by == user_id
        assert revoke_reason == "bulk-logout"

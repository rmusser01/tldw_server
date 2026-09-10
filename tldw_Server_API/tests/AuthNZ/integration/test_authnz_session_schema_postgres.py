"""Verify production PostgreSQL bootstrap creates and upgrades session activity storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


async def _seed_user(connection: asyncpg.Connection, username: str) -> int:
    """Insert a user for session schema checks in the fixture's isolated database."""
    return int(
        await connection.fetchval(
            """
            INSERT INTO users (uuid, username, email, password_hash)
            VALUES ($1, $2, $3, 'hash')
            RETURNING id
            """,
            uuid.uuid4(),
            username,
            f"{username}@example.test",
        )
    )


@pytest.mark.asyncio
async def test_postgres_production_bootstrap_creates_usable_session_activity_schema(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    """Fresh session tables support activity reads, touches, and token refreshes."""
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_authnz_core_tables_pg,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

    pool = await get_db_pool()
    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        await connection.execute("DROP TABLE sessions")
    finally:
        await connection.close()

    assert await ensure_authnz_core_tables_pg(pool) is True

    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        user_id = await _seed_user(connection, "session-bootstrap-user")
        column = await connection.fetchrow(
            """
            SELECT data_type, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'sessions'
              AND column_name = 'last_activity'
            """
        )
    finally:
        await connection.close()

    assert column is not None
    assert column["data_type"] == "timestamp without time zone"
    assert column["column_default"] is not None

    repo = AuthnzSessionsRepo(pool)
    now = datetime.now(timezone.utc)
    session_id = await repo.create_session_record(
        user_id=user_id,
        token_hash="bootstrap-access",
        refresh_token_hash="bootstrap-refresh",
        encrypted_token="encrypted-access",
        encrypted_refresh="encrypted-refresh",
        expires_at=now + timedelta(hours=1),
        refresh_expires_at=now + timedelta(days=1),
        ip_address="127.0.0.1",
        user_agent="pytest",
        device_id="bootstrap-device",
        access_jti="bootstrap-access-jti",
        refresh_jti="bootstrap-refresh-jti",
    )

    sessions = await repo.get_active_sessions_for_user(user_id)
    assert [session["id"] for session in sessions] == [session_id]
    assert sessions[0]["last_activity"] is not None

    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        old_activity = datetime(2000, 1, 1)
        await connection.execute(
            "UPDATE sessions SET last_activity = $1 WHERE id = $2",
            old_activity,
            session_id,
        )
    finally:
        await connection.close()

    await repo.update_last_activity(session_id)
    touched_activity = await pool.fetchval(
        "SELECT last_activity FROM sessions WHERE id = ?",
        session_id,
    )
    assert touched_activity > old_activity

    refreshed = await repo.update_session_tokens_for_refresh(
        session_id=session_id,
        expected_access_hash="bootstrap-access",
        expected_refresh_hash="bootstrap-refresh",
        new_access_hash="bootstrap-access-new",
        access_jti="bootstrap-access-jti-new",
        expires_at=now + timedelta(hours=2),
        encrypted_access_token="encrypted-access-new",
        refresh_hash_update="bootstrap-refresh-new",
        refresh_jti="bootstrap-refresh-jti-new",
        refresh_expires_at=now + timedelta(days=2),
        encrypted_refresh_token="encrypted-refresh-new",
    )
    assert refreshed is True
    assert (
        await pool.fetchval(
            "SELECT last_activity FROM sessions WHERE id = ?",
            session_id,
        )
        >= touched_activity
    )


@pytest.mark.asyncio
async def test_postgres_production_bootstrap_backfills_legacy_session_rows_idempotently(
    isolated_test_environment: tuple[TestClient, str],
) -> None:
    """Repeated bootstrap backfills missing activity without replacing existing values."""
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_authnz_core_tables_pg,
    )

    pool = await get_db_pool()
    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        await connection.execute("DROP TABLE sessions")
        await connection.execute(
            """
            CREATE TABLE sessions (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash VARCHAR(64) NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        user_id = await _seed_user(connection, "session-upgrade-user")
        session_id = int(
            await connection.fetchval(
                """
                INSERT INTO sessions (user_id, token_hash, expires_at, created_at)
                VALUES ($1, 'legacy-access', '2030-01-01 00:00:00',
                        '2025-02-03 04:05:06')
                RETURNING id
                """,
                user_id,
            )
        )
    finally:
        await connection.close()

    assert await ensure_authnz_core_tables_pg(pool) is True

    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        backfilled = await connection.fetchval(
            "SELECT last_activity FROM sessions WHERE id = $1",
            session_id,
        )
        preserved = datetime(2025, 6, 7, 8, 9, 10)
        await connection.execute(
            "UPDATE sessions SET last_activity = $1 WHERE id = $2",
            preserved,
            session_id,
        )
    finally:
        await connection.close()

    assert backfilled == datetime(2025, 2, 3, 4, 5, 6)
    assert await ensure_authnz_core_tables_pg(pool) is True
    assert (
        await pool.fetchval(
            "SELECT last_activity FROM sessions WHERE id = ?",
            session_id,
        )
        == preserved
    )

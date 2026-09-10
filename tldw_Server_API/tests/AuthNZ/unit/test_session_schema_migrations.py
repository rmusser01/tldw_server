from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite
import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo

pytestmark = pytest.mark.unit


class _SQLitePool:
    pool = None

    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection

    @asynccontextmanager
    async def transaction(self):
        try:
            yield self._connection
            await self._connection.commit()
        except BaseException:
            await self._connection.rollback()
            raise


def _create_legacy_database(
    db_path: Path,
    *,
    include_last_activity: bool,
) -> None:
    last_activity_column = ", last_activity TIMESTAMP" if include_last_activity else ""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL,
                refresh_token_hash TEXT,
                encrypted_token TEXT,
                encrypted_refresh TEXT,
                expires_at TIMESTAMP NOT NULL,
                refresh_expires_at TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                device_id TEXT,
                is_active INTEGER DEFAULT 1,
                is_revoked INTEGER DEFAULT 0,
                revoked_at TIMESTAMP,
                revoked_by INTEGER,
                revoke_reason TEXT,
                access_jti TEXT,
                refresh_jti TEXT,
                created_at TIMESTAMP
                {last_activity_column}
            )
            """
        )
        conn.execute(
            "INSERT INTO users (id, username, email, password_hash) "
            "VALUES (1, 'legacy', 'legacy@example.test', 'hash')"
        )
        conn.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) "
            "VALUES (97, 'legacy current', CURRENT_TIMESTAMP)"
        )


def test_fresh_sqlite_sessions_have_last_activity_default(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh-sessions.db"

    apply_authnz_migrations(db_path, target_version=2)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1]: row for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("fresh", "fresh@example.test", "hash"),
        )
        user_id = int(conn.execute("SELECT id FROM users").fetchone()[0])
        conn.execute(
            "INSERT INTO sessions (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
            (user_id, "fresh-token", "2030-01-01 00:00:00"),
        )
        activity = conn.execute("SELECT last_activity FROM sessions").fetchone()[0]

    assert columns["last_activity"][2].upper() == "TIMESTAMP"
    assert "CURRENT_TIMESTAMP" in columns["last_activity"][4].upper()
    assert activity is not None


def test_sqlite_session_upgrade_adds_and_backfills_last_activity(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-sessions.db"
    _create_legacy_database(db_path, include_last_activity=False)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, user_id, token_hash, expires_at, created_at)
            VALUES (1, 1, 'legacy-token', '2030-01-01 00:00:00',
                    '2025-02-03 04:05:06')
            """
        )

    apply_authnz_migrations(db_path)
    apply_authnz_migrations(db_path)

    with sqlite3.connect(db_path) as conn:
        activity = conn.execute("SELECT last_activity FROM sessions WHERE id = 1").fetchone()[0]
        migration_count = conn.execute("SELECT COUNT(*) FROM schema_migrations WHERE version = 98").fetchone()[0]

    assert activity == "2025-02-03 04:05:06"
    assert migration_count == 1


def test_sqlite_session_upgrade_preserves_existing_activity_values(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "partial-sessions.db"
    _create_legacy_database(db_path, include_last_activity=True)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO sessions (
                id, user_id, token_hash, expires_at, created_at, last_activity
            ) VALUES (?, 1, ?, '2030-01-01 00:00:00', ?, ?)
            """,
            (
                (1, "preserved-token", "2025-01-01 00:00:00", "2025-06-01 00:00:00"),
                (2, "backfilled-token", "2025-02-01 00:00:00", None),
            ),
        )

    apply_authnz_migrations(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, last_activity FROM sessions ORDER BY id").fetchall()

    assert rows == [
        (1, "2025-06-01 00:00:00"),
        (2, "2025-02-01 00:00:00"),
    ]


@pytest.mark.asyncio
async def test_sqlite_session_writer_initializes_activity_after_legacy_upgrade(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "upgraded-session-writer.db"
    _create_legacy_database(db_path, include_last_activity=False)
    apply_authnz_migrations(db_path)

    connection = await aiosqlite.connect(db_path)
    try:
        repo = AuthnzSessionsRepo(_SQLitePool(connection))
        now = datetime.now(timezone.utc)
        session_id = await repo.create_session_record(
            user_id=1,
            token_hash="upgraded-access",
            refresh_token_hash="upgraded-refresh",
            encrypted_token="encrypted-access",
            encrypted_refresh="encrypted-refresh",
            expires_at=now + timedelta(hours=1),
            refresh_expires_at=now + timedelta(days=1),
            ip_address="127.0.0.1",
            user_agent="pytest",
            device_id="upgraded-device",
            access_jti="upgraded-access-jti",
            refresh_jti="upgraded-refresh-jti",
        )
        row = await (
            await connection.execute(
                "SELECT last_activity FROM sessions WHERE id = ?",
                (session_id,),
            )
        ).fetchone()
    finally:
        await connection.close()

    assert row is not None
    assert row[0] is not None

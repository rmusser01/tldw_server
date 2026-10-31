"""DB-owned AuthNZ session schema changes on caller-owned connections."""

from __future__ import annotations

import sqlite3
from typing import Any


def create_sqlite_sessions_table(conn: sqlite3.Connection) -> None:
    """Create the session table with an activity default for new rows.

    Args:
        conn: SQLite connection supplied by the AuthNZ migration runner.

    Returns:
        None. The caller owns the transaction and its commit or rollback.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)


def ensure_sqlite_session_last_activity(conn: sqlite3.Connection) -> None:
    """Add and backfill session activity without replacing existing values.

    Args:
        conn: SQLite connection supplied by the AuthNZ migration runner.

    Returns:
        None. Missing session tables are left unchanged; the caller owns the
        transaction and its commit or rollback.
    """
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("sessions",),
    ).fetchone()
    if table is None:
        return

    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    if "last_activity" not in columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN last_activity TIMESTAMP")
    if "created_at" in columns:
        conn.execute(
            "UPDATE sessions SET last_activity = COALESCE(created_at, CURRENT_TIMESTAMP) WHERE last_activity IS NULL"
        )
    else:
        conn.execute("UPDATE sessions SET last_activity = CURRENT_TIMESTAMP WHERE last_activity IS NULL")


async def ensure_postgres_session_last_activity(conn: Any) -> None:
    """Add, backfill, and default activity for an existing PostgreSQL session table.

    Args:
        conn: PostgreSQL connection in the AuthNZ bootstrap transaction.

    Returns:
        None. Existing activity values remain unchanged, and the caller owns
        the transaction and its commit or rollback.
    """
    await conn.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_activity TIMESTAMP")
    await conn.execute(
        "UPDATE sessions SET last_activity = COALESCE(created_at, CURRENT_TIMESTAMP) WHERE last_activity IS NULL"
    )
    await conn.execute("ALTER TABLE sessions ALTER COLUMN last_activity SET DEFAULT CURRENT_TIMESTAMP")

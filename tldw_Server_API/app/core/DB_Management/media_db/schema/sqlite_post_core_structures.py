"""Package-owned SQLite post-core schema helpers."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover - defensive fallback
    logger = logging.getLogger("media_db_sqlite_post_core_structures")


def ensure_sqlite_visibility_columns(db: Any, conn: sqlite3.Connection) -> None:
    """Ensure Media visibility/owner columns and indexes exist on SQLite."""

    del db

    try:
        cursor = conn.execute("PRAGMA table_info(Media)")
        columns = {row[1] for row in cursor.fetchall()}
    except sqlite3.Error as exc:
        logging.warning(f"Could not introspect Media table for visibility columns: {exc}")
        return

    try:
        index_cursor = conn.execute("PRAGMA index_list(Media)")
        indexes = {row[1] for row in index_cursor.fetchall()}
    except sqlite3.Error:
        indexes = set()

    statements: list[str] = []

    if "visibility" not in columns:
        statements.append(
            "ALTER TABLE Media ADD COLUMN visibility TEXT DEFAULT 'personal' "
            "CHECK (visibility IN ('personal', 'team', 'org'));"
        )

    if "owner_user_id" not in columns:
        statements.append("ALTER TABLE Media ADD COLUMN owner_user_id INTEGER;")

    if not indexes or "idx_media_visibility" not in indexes:
        statements.append(
            "CREATE INDEX IF NOT EXISTS idx_media_visibility ON Media(visibility);"
        )
    if not indexes or "idx_media_owner_user_id" not in indexes:
        statements.append(
            "CREATE INDEX IF NOT EXISTS idx_media_owner_user_id ON Media(owner_user_id);"
        )

    if "client_id" in columns:
        # Older ingest workers wrote a provenance label instead of the user ID.
        # Recover only canonical positive IDs, preserving every explicit owner.
        statements.append(
            "UPDATE Media SET owner_user_id = CAST(SUBSTR(client_id, 21) AS INTEGER), "
            "version = version + 1, last_modified = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE owner_user_id IS NULL "
            "AND CAST(SUBSTR(client_id, 21) AS INTEGER) > 0 "
            "AND client_id = 'media_ingest_worker:' || "
            "CAST(CAST(SUBSTR(client_id, 21) AS INTEGER) AS TEXT);"
        )

    if not statements:
        return

    try:
        conn.executescript("\n".join(statements))
    except sqlite3.Error as exc:
        logger.warning(f"Could not ensure visibility columns/indexes on Media: {exc}")


def ensure_sqlite_source_hash_column(db: Any, conn: sqlite3.Connection) -> None:
    """Ensure Media source_hash column and index exist on SQLite."""

    del db

    try:
        cursor = conn.execute("PRAGMA table_info(Media)")
        columns = {row[1] for row in cursor.fetchall()}
    except sqlite3.Error as exc:
        logging.warning(f"Could not introspect Media table for source_hash column: {exc}")
        return

    try:
        index_cursor = conn.execute("PRAGMA index_list(Media)")
        indexes = {row[1] for row in index_cursor.fetchall()}
    except sqlite3.Error:
        indexes = set()

    statements: list[str] = []

    if "source_hash" not in columns:
        statements.append("ALTER TABLE Media ADD COLUMN source_hash TEXT;")

    if not indexes or "idx_media_source_hash" not in indexes:
        statements.append(
            "CREATE INDEX IF NOT EXISTS idx_media_source_hash ON Media(source_hash);"
        )

    if not statements:
        return

    try:
        conn.executescript("\n".join(statements))
    except sqlite3.Error as exc:
        logger.warning(f"Could not ensure source_hash column/index on Media: {exc}")


def ensure_sqlite_data_tables(db: Any, conn: sqlite3.Connection) -> None:
    """Ensure Data Tables schema exists on SQLite."""

    try:
        conn.executescript(db._DATA_TABLES_SQL)
    except sqlite3.Error as exc:
        logger.warning(f"Could not ensure data_tables schema on SQLite: {exc}")


__all__ = [
    "ensure_sqlite_data_tables",
    "ensure_sqlite_source_hash_column",
    "ensure_sqlite_visibility_columns",
]

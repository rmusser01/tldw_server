"""Package-owned email schema helpers."""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)

try:
    from loguru import logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging

    logger = logging.getLogger("media_db_email_schema_structures")


_EMAIL_SEARCH_LOOKUP_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_email_participant_reverse
    ON email_message_participants(participant_id, role, email_message_id);
CREATE INDEX IF NOT EXISTS idx_email_label_reverse
    ON email_message_labels(label_id, email_message_id);
CREATE INDEX IF NOT EXISTS idx_email_messages_identity_cover ON email_messages(id, tenant_id, media_id);
CREATE INDEX IF NOT EXISTS idx_email_media_visibility ON Media(id, deleted, is_trash);
"""


def ensure_sqlite_email_schema(db: Any, conn: sqlite3.Connection) -> None:
    """Ensure SQLite email-native schema, indexes, and FTS objects exist."""

    try:
        fts_existed = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='email_fts' LIMIT 1"
            ).fetchone()
            is not None
        )
        conn.executescript(db._EMAIL_SCHEMA_SQL)
        conn.executescript(db._EMAIL_INDICES_SQL)
        conn.executescript(_EMAIL_SEARCH_LOOKUP_INDEXES)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_email_message_count_cover "
            "ON email_messages(tenant_id, internal_date, id, media_id, has_attachments, subject)"
        )
        conn.executescript(db._EMAIL_SQLITE_FTS_SQL)
        if not fts_existed:
            with suppress(sqlite3.Error):
                conn.execute("INSERT INTO email_fts(email_fts) VALUES ('rebuild')")
    except sqlite3.Error as exc:
        logger.warning("Could not ensure email-native SQLite schema (error_type={})", type(exc).__name__[:80])


def ensure_postgres_email_schema(db: Any, conn: Any) -> None:
    """Ensure PostgreSQL email-native schema and lookup indexes exist."""

    schema_statements = db._convert_sqlite_sql_to_postgres_statements(
        db._EMAIL_SCHEMA_SQL
    )
    index_statements = db._convert_sqlite_sql_to_postgres_statements(
        db._EMAIL_INDICES_SQL + _EMAIL_SEARCH_LOOKUP_INDEXES
    )
    index_statements.append(
        "CREATE INDEX IF NOT EXISTS idx_email_message_count_cover "
        "ON email_messages(tenant_id, internal_date, id, media_id, has_attachments)"
    )
    for stmt in schema_statements + index_statements:
        try:
            db.backend.execute(stmt, connection=conn)
        except BackendDatabaseError as exc:
            logger.warning(
                "Could not ensure email-native PostgreSQL schema (error_type={})",
                type(exc).__name__[:80],
            )


__all__ = [
    "ensure_postgres_email_schema",
    "ensure_sqlite_email_schema",
]

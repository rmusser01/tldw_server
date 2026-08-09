"""Package-owned SQLite claims extension helper."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover - defensive fallback
    logger = logging.getLogger("media_db_sqlite_claims_extensions")


def ensure_sqlite_claims_extensions(db: Any, conn: sqlite3.Connection) -> None:
    """Ensure SQLite claims review and monitoring extension artifacts exist."""

    try:
        table_cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Claims'"
        )
        if not table_cursor.fetchone():
            conn.executescript(db._CLAIMS_TABLE_SQL)
            return
    except sqlite3.Error as exc:
        logger.warning(f"Could not introspect Claims table for extensions: {exc}")
        return

    try:
        cursor = conn.execute("PRAGMA table_info(Claims)")
        columns = {row[1] for row in cursor.fetchall()}
    except sqlite3.Error as exc:
        logging.warning(f"Could not introspect Claims table for extension columns: {exc}")
        return

    statements: list[str] = []

    if "review_status" not in columns:
        statements.append(
            "ALTER TABLE Claims ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending';"
        )
    if "reviewer_id" not in columns:
        statements.append("ALTER TABLE Claims ADD COLUMN reviewer_id INTEGER;")
    if "review_group" not in columns:
        statements.append("ALTER TABLE Claims ADD COLUMN review_group TEXT;")
    if "reviewed_at" not in columns:
        statements.append("ALTER TABLE Claims ADD COLUMN reviewed_at DATETIME;")
    if "review_notes" not in columns:
        statements.append("ALTER TABLE Claims ADD COLUMN review_notes TEXT;")
    if "review_version" not in columns:
        statements.append(
            "ALTER TABLE Claims ADD COLUMN review_version INTEGER NOT NULL DEFAULT 1;"
        )
    if "review_reason_code" not in columns:
        statements.append("ALTER TABLE Claims ADD COLUMN review_reason_code TEXT;")
    if "claim_cluster_id" not in columns:
        statements.append("ALTER TABLE Claims ADD COLUMN claim_cluster_id INTEGER;")

    if statements:
        try:
            conn.executescript("\n".join(statements))
        except sqlite3.Error as exc:
            logger.warning(f"Could not ensure Claims extension columns: {exc}")

    try:
        conn.executescript(db._CLAIMS_TABLE_SQL)
    except sqlite3.Error as exc:
        logger.warning(f"Could not ensure Claims extension tables/indexes: {exc}")

    try:
        events_cursor = conn.execute("PRAGMA table_info(claims_monitoring_events)")
        events_columns = {row[1] for row in events_cursor.fetchall()}
        events_statements: list[str] = []
        if "delivered_at" not in events_columns:
            events_statements.append(
                "ALTER TABLE claims_monitoring_events ADD COLUMN delivered_at DATETIME;"
            )
        if events_statements:
            conn.executescript("\n".join(events_statements))
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_delivered "
            "ON claims_monitoring_events(delivered_at);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_user_created_id "
            "ON claims_monitoring_events(user_id, created_at, id);"
        )
    except sqlite3.Error as exc:
        logger.warning(f"Could not ensure delivered_at for claims_monitoring_events: {exc}")


__all__ = ["ensure_sqlite_claims_extensions"]

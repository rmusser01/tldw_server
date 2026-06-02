"""Schema ownership for document workspace storage."""

from __future__ import annotations

from typing import Any

from loguru import logger

DOCUMENT_READING_PROGRESS_TABLE = "document_reading_progress"
DOCUMENT_ANNOTATIONS_TABLE = "document_annotations"
DOCUMENT_PARSED_REFERENCES_CACHE_TABLE = "document_parsed_references_cache"

SQLITE_DOCUMENT_WORKSPACE_SQL = """
CREATE TABLE IF NOT EXISTS document_reading_progress (
    media_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    current_page INTEGER NOT NULL DEFAULT 1,
    total_pages INTEGER NOT NULL DEFAULT 1,
    zoom_level INTEGER NOT NULL DEFAULT 100,
    view_mode TEXT NOT NULL DEFAULT 'single',
    cfi TEXT,
    percentage REAL,
    last_read_at TEXT NOT NULL,
    PRIMARY KEY (media_id, user_id)
);

CREATE TABLE IF NOT EXISTS document_annotations (
    id TEXT PRIMARY KEY,
    media_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    location TEXT NOT NULL,
    text TEXT NOT NULL,
    color TEXT NOT NULL DEFAULT 'yellow',
    note TEXT,
    annotation_type TEXT NOT NULL DEFAULT 'highlight',
    chapter_title TEXT,
    percentage REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_annotations_media_user
ON document_annotations(media_id, user_id, deleted);

CREATE TABLE IF NOT EXISTS document_parsed_references_cache (
    media_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    references_json TEXT NOT NULL,
    total_detected INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (media_id, user_id, parser_version, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_doc_refs_cache_lookup
ON document_parsed_references_cache(media_id, user_id, parser_version);
"""

POSTGRES_DOCUMENT_WORKSPACE_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS document_reading_progress (
        media_id BIGINT NOT NULL,
        user_id TEXT NOT NULL,
        current_page INTEGER NOT NULL DEFAULT 1,
        total_pages INTEGER NOT NULL DEFAULT 1,
        zoom_level INTEGER NOT NULL DEFAULT 100,
        view_mode TEXT NOT NULL DEFAULT 'single',
        cfi TEXT,
        percentage DOUBLE PRECISION,
        last_read_at TEXT NOT NULL,
        PRIMARY KEY (media_id, user_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS document_annotations (
        id TEXT PRIMARY KEY,
        media_id BIGINT NOT NULL,
        user_id TEXT NOT NULL,
        location TEXT NOT NULL,
        text TEXT NOT NULL,
        color TEXT NOT NULL DEFAULT 'yellow',
        note TEXT,
        annotation_type TEXT NOT NULL DEFAULT 'highlight',
        chapter_title TEXT,
        percentage DOUBLE PRECISION,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        deleted INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_annotations_media_user
    ON document_annotations(media_id, user_id, deleted)
    """,
    """
    CREATE TABLE IF NOT EXISTS document_parsed_references_cache (
        media_id BIGINT NOT NULL,
        user_id TEXT NOT NULL,
        parser_version TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        references_json TEXT NOT NULL,
        total_detected INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (media_id, user_id, parser_version, content_hash)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_doc_refs_cache_lookup
    ON document_parsed_references_cache(media_id, user_id, parser_version)
    """,
    "ALTER TABLE document_reading_progress ADD COLUMN IF NOT EXISTS cfi TEXT",
    "ALTER TABLE document_reading_progress ADD COLUMN IF NOT EXISTS percentage DOUBLE PRECISION",
    "ALTER TABLE document_annotations ADD COLUMN IF NOT EXISTS chapter_title TEXT",
    "ALTER TABLE document_annotations ADD COLUMN IF NOT EXISTS percentage DOUBLE PRECISION",
)


def _sqlite_columns(conn: Any, table_name: str) -> set[str]:
    """Return SQLite column names for a known document workspace table."""
    cursor = conn.execute(f"PRAGMA table_info({table_name})")  # nosec B608
    columns: set[str] = set()
    for row in cursor.fetchall():
        if isinstance(row, dict):
            columns.add(str(row["name"]))
        else:
            columns.add(str(row["name"] if "name" in getattr(row, "keys", lambda: [])() else row[1]))
    return columns


def ensure_sqlite_document_workspace_schema(conn: Any) -> None:
    """Ensure SQLite document workspace tables, indexes, and additive columns exist."""
    conn.executescript(SQLITE_DOCUMENT_WORKSPACE_SQL)

    progress_columns = _sqlite_columns(conn, DOCUMENT_READING_PROGRESS_TABLE)
    if "cfi" not in progress_columns:
        conn.execute(f"ALTER TABLE {DOCUMENT_READING_PROGRESS_TABLE} ADD COLUMN cfi TEXT")  # nosec B608
        logger.info("Added cfi column to reading progress table")
    if "percentage" not in progress_columns:
        conn.execute(f"ALTER TABLE {DOCUMENT_READING_PROGRESS_TABLE} ADD COLUMN percentage REAL")  # nosec B608
        logger.info("Added percentage column to reading progress table")

    annotation_columns = _sqlite_columns(conn, DOCUMENT_ANNOTATIONS_TABLE)
    if "chapter_title" not in annotation_columns:
        conn.execute(f"ALTER TABLE {DOCUMENT_ANNOTATIONS_TABLE} ADD COLUMN chapter_title TEXT")  # nosec B608
        logger.info("Added chapter_title column to annotations table")
    if "percentage" not in annotation_columns:
        conn.execute(f"ALTER TABLE {DOCUMENT_ANNOTATIONS_TABLE} ADD COLUMN percentage REAL")  # nosec B608
        logger.info("Added percentage column to annotations table")


def _execute_postgres_statement(conn: Any, statement: str) -> None:
    if hasattr(conn, "execute"):
        conn.execute(statement)
        return
    cursor = conn.cursor()
    cursor.execute(statement)


def ensure_postgres_document_workspace_schema(conn: Any) -> None:
    """Ensure PostgreSQL document workspace tables, indexes, and additive columns exist."""
    for statement in POSTGRES_DOCUMENT_WORKSPACE_STATEMENTS:
        _execute_postgres_statement(conn, statement)


__all__ = [
    "DOCUMENT_ANNOTATIONS_TABLE",
    "DOCUMENT_PARSED_REFERENCES_CACHE_TABLE",
    "DOCUMENT_READING_PROGRESS_TABLE",
    "ensure_postgres_document_workspace_schema",
    "ensure_sqlite_document_workspace_schema",
]

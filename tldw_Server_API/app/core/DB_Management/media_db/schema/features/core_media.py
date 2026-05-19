"""Core media schema helpers."""

from __future__ import annotations

import sqlite3
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

try:
    from loguru import logger

    logging = logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging as _stdlib_logging

    logger = _stdlib_logging.getLogger("media_db_core_media_schema")
    logging = logger


_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    MEDIA_NONCRITICAL_EXCEPTIONS
)

_EXPECTED_MEDIA_COLUMNS = {
    "id",
    "url",
    "title",
    "type",
    "content",
    "author",
    "ingestion_date",
    "transcription_model",
    "is_trash",
    "trash_date",
    "vector_embedding",
    "chunking_status",
    "vector_processing",
    "content_hash",
    "source_hash",
    "uuid",
    "last_modified",
    "version",
    "org_id",
    "team_id",
    "visibility",
    "owner_user_id",
    "latest_transcription_run_id",
    "next_transcription_run_id",
    "client_id",
    "deleted",
    "prev_version",
    "merge_parent_uuid",
}

_POSTGRES_REQUIRED_TABLES = [
    "media",
    "keywords",
    "mediakeywords",
    "transcripts",
    "mediachunks",
    "unvectorizedmediachunks",
    "documentversions",
    "documentversionidentifiers",
    "documentstructureindex",
    "sync_log",
    "chunkingtemplates",
    "claims",
]

_SQLITE_ADDITIONAL_CORE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS output_templates (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    format TEXT NOT NULL,
    body TEXT NOT NULL,
    description TEXT,
    is_default INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_output_templates_user ON output_templates(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_output_templates_user_name ON output_templates(user_id, name);

CREATE TABLE IF NOT EXISTS reading_highlights (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    item_id INTEGER NOT NULL,
    quote TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    color TEXT,
    note TEXT,
    created_at TEXT NOT NULL,
    anchor_strategy TEXT NOT NULL DEFAULT 'fuzzy_quote',
    content_hash_ref TEXT,
    context_before TEXT,
    context_after TEXT,
    state TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_highlights_user_item ON reading_highlights(user_id, item_id);

CREATE TABLE IF NOT EXISTS collection_tags (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    UNIQUE (user_id, name)
);

CREATE TABLE IF NOT EXISTS content_items (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,
    origin TEXT NOT NULL,
    origin_type TEXT,
    origin_id INTEGER,
    url TEXT,
    canonical_url TEXT,
    domain TEXT,
    title TEXT,
    summary TEXT,
    notes TEXT,
    content_hash TEXT,
    word_count INTEGER,
    published_at TEXT,
    status TEXT,
    favorite INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT,
    media_id INTEGER,
    job_id INTEGER,
    run_id INTEGER,
    source_id INTEGER,
    read_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_user_canonical ON content_items(user_id, canonical_url) WHERE canonical_url IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS ux_content_items_user_hash ON content_items(user_id, content_hash) WHERE content_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_content_items_user_updated ON content_items(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_content_items_user_domain ON content_items(user_id, domain);
CREATE INDEX IF NOT EXISTS idx_content_items_job ON content_items(job_id);
CREATE INDEX IF NOT EXISTS idx_content_items_run ON content_items(run_id);

CREATE TABLE IF NOT EXISTS content_item_tags (
    item_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    UNIQUE (item_id, tag_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS content_items_fts USING fts5(
    title,
    summary,
    metadata,
    content=''
);
"""


class _SQLiteCursor(Protocol):
    def fetchall(self) -> list[Any]: ...

    def fetchone(self) -> Any: ...


class _SQLiteConnection(Protocol):
    def executescript(self, script: str) -> Any: ...

    def execute(self, query: str) -> _SQLiteCursor: ...


class _PostgresBackend(Protocol):
    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: object,
    ) -> object: ...

    def table_exists(self, table: str, *, connection: object) -> bool: ...


class SupportsSqliteCoreMediaSchema(Protocol):
    db_path_str: str
    _TABLES_SQL_V1: str
    _INDICES_SQL_V1: str
    _TRIGGERS_SQL_V1: str
    _SCHEMA_UPDATE_VERSION_SQL_V1: str
    _CLAIMS_TABLE_SQL: str
    _MEDIA_FILES_TABLE_SQL: str
    _TTS_HISTORY_TABLE_SQL: str
    _DATA_TABLES_SQL: str
    _CURRENT_SCHEMA_VERSION: int

    def _ensure_sqlite_email_schema(self, conn: _SQLiteConnection) -> None: ...

    def _ensure_fts_structures(self, conn: _SQLiteConnection) -> None: ...


class SupportsPostgresCoreMediaSchema(Protocol):
    _TABLES_SQL_V1: str
    _CLAIMS_TABLE_SQL: str
    _MEDIA_FILES_TABLE_SQL: str
    _TTS_HISTORY_TABLE_SQL: str
    _DATA_TABLES_SQL: str
    _INDICES_SQL_V1: str
    _CURRENT_SCHEMA_VERSION: int
    backend: _PostgresBackend

    def _convert_sqlite_sql_to_postgres_statements(self, sql: str) -> list[str]: ...

    def _ensure_postgres_email_schema(self, conn: Any) -> None: ...


def _build_sqlite_core_schema_script(db: SupportsSqliteCoreMediaSchema) -> str:
    return f"""
        {db._TABLES_SQL_V1}
        {db._INDICES_SQL_V1}
        {db._TRIGGERS_SQL_V1}
        {db._SCHEMA_UPDATE_VERSION_SQL_V1}
        {db._CLAIMS_TABLE_SQL}
        {db._MEDIA_FILES_TABLE_SQL}
        {db._TTS_HISTORY_TABLE_SQL}
        {db._DATA_TABLES_SQL}
        {_SQLITE_ADDITIONAL_CORE_SCHEMA_SQL}
    """


def _validate_sqlite_media_table(conn: _SQLiteConnection) -> None:
    cursor = conn.execute("PRAGMA table_info(Media)")
    columns = {row["name"] for row in cursor.fetchall()}
    if not _EXPECTED_MEDIA_COLUMNS.issubset(columns):
        missing_cols = _EXPECTED_MEDIA_COLUMNS - columns
        raise SchemaError(
            f"Validation Error: Media table is missing columns after creation: {missing_cols}"
        )


def _verify_sqlite_schema_version(
    db: SupportsSqliteCoreMediaSchema,
    conn: _SQLiteConnection,
) -> None:
    cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
    version_row = cursor.fetchone()
    version_in_db = version_row["version"] if version_row else None
    if version_in_db != db._CURRENT_SCHEMA_VERSION:
        logging.error(
            "[Schema V1] Version check failed after schema script. Found: {}",
            version_in_db if version_row else "None",
        )
        raise SchemaError("Schema version update did not take effect after schema script.")


def apply_sqlite_core_media_schema(
    db: SupportsSqliteCoreMediaSchema,
    conn: _SQLiteConnection,
) -> None:
    """Apply the SQLite base schema through the package-owned helper."""

    logging.info("Applying initial schema (Version 1) to DB: {}...", db.db_path_str)
    try:
        full_schema_script = _build_sqlite_core_schema_script(db)

        logging.debug("[Schema V1] Applying full schema script...")
        conn.executescript(full_schema_script)
        logging.debug("[Schema V1] Full schema script executed.")

        db._ensure_sqlite_email_schema(conn)

        try:
            _validate_sqlite_media_table(conn)
            logging.debug("[Schema V1] Media table structure validated successfully.")
        except (sqlite3.Error, SchemaError) as val_err:
            logging.error(
                "[Schema V1] Validation failed after table creation: {}",
                val_err,
                exc_info=True,
            )
            raise

        _verify_sqlite_schema_version(db, conn)
        logging.debug(
            "[Schema V1] Version check confirmed version is {}.",
            db._CURRENT_SCHEMA_VERSION,
        )
        logging.info(
            "[Schema V1] Core Schema V1 (incl. version update) applied successfully for DB: {}.",
            db.db_path_str,
        )

        try:
            logging.debug("[Schema V1] Applying FTS Tables...")
            db._ensure_fts_structures(conn)
            logging.info("[Schema V1] FTS Tables created successfully.")
        except (sqlite3.Error, DatabaseError) as fts_err:
            logging.error(
                "[Schema V1] Failed to create FTS tables: {}", fts_err, exc_info=True
            )

    except SchemaError:
        raise
    except sqlite3.Error as exc:
        logging.error(
            "[Schema V1] Application failed during schema script: {}",
            exc,
            exc_info=True,
        )
        raise DatabaseError(f"DB schema V1 setup failed: {exc}") from exc
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logging.error(
            "[Schema V1] Unexpected error during schema V1 application: {}",
            exc,
            exc_info=True,
        )
        raise DatabaseError(f"Unexpected error applying schema V1: {exc}") from exc


def apply_postgres_core_media_schema(
    db: SupportsPostgresCoreMediaSchema,
    conn: Any,
) -> None:
    """Apply the PostgreSQL base schema through the package-owned helper."""

    table_statements = db._convert_sqlite_sql_to_postgres_statements(db._TABLES_SQL_V1)
    table_statements += db._convert_sqlite_sql_to_postgres_statements(db._CLAIMS_TABLE_SQL)
    table_statements += db._convert_sqlite_sql_to_postgres_statements(db._MEDIA_FILES_TABLE_SQL)
    table_statements += db._convert_sqlite_sql_to_postgres_statements(db._TTS_HISTORY_TABLE_SQL)
    table_statements += db._convert_sqlite_sql_to_postgres_statements(db._DATA_TABLES_SQL)

    create_tables = [
        statement
        for statement in table_statements
        if statement.strip().upper().startswith("CREATE TABLE")
    ]
    other_table_statements = [
        statement for statement in table_statements if statement not in create_tables
    ]

    for statement in create_tables:
        logger.debug(f"Applying Postgres base table DDL: {statement[:120]}...")
        db.backend.execute(statement, connection=conn)

    for statement in other_table_statements:
        logger.debug(f"Applying Postgres base initializer DDL: {statement[:120]}...")
        db.backend.execute(statement, connection=conn)

    for table in _POSTGRES_REQUIRED_TABLES:
        if not db.backend.table_exists(table, connection=conn):
            raise SchemaError(f"Postgres schema init missing table: {table}")

    index_statements = db._convert_sqlite_sql_to_postgres_statements(db._INDICES_SQL_V1)
    for statement in index_statements:
        logger.debug(f"Applying Postgres index DDL: {statement[:120]}...")
        db.backend.execute(statement, connection=conn)

    db._ensure_postgres_email_schema(conn)
    db.backend.execute(
        "DELETE FROM schema_version WHERE version <> %s",
        (0,),
        connection=conn,
    )
    db.backend.execute(
        "INSERT INTO schema_version (version) VALUES (%s) ON CONFLICT (version) DO NOTHING",
        (0,),
        connection=conn,
    )
    db.backend.execute(
        "UPDATE schema_version SET version = %s",
        (db._CURRENT_SCHEMA_VERSION,),
        connection=conn,
    )

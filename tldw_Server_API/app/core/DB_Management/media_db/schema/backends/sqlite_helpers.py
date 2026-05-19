"""Package-native helper utilities for SQLite schema bootstrap."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator, MigrationError
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.fts import (
    ensure_sqlite_fts_structures,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.core_media import (
    apply_sqlite_core_media_schema,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

try:
    from loguru import logger

    logging = logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging as _stdlib_logging

    logger = _stdlib_logging.getLogger("media_db_sqlite_schema_bootstrap")
    logging = logger


_SQLITE_COLLECTIONS_AND_CONTENT_ITEMS_SQL = """
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


class _SupportsExecutescript(Protocol):
    def executescript(self, script: str) -> Any: ...


class SupportsSqlitePostCoreStructures(Protocol):
    _CLAIMS_TABLE_SQL: str
    _MEDIA_FILES_TABLE_SQL: str
    _TTS_HISTORY_TABLE_SQL: str
    _CURRENT_SCHEMA_VERSION: int
    db_path_str: str
    is_memory_db: bool

    def get_connection(self) -> sqlite3.Connection: ...
    def _get_db_version(self, conn: sqlite3.Connection) -> int: ...
    def _ensure_sqlite_data_tables(self, conn: _SupportsExecutescript) -> None: ...
    def _ensure_sqlite_visibility_columns(self, conn: _SupportsExecutescript) -> None: ...
    def _ensure_sqlite_source_hash_column(self, conn: _SupportsExecutescript) -> None: ...
    def _ensure_sqlite_claims_extensions(self, conn: _SupportsExecutescript) -> None: ...
    def _ensure_sqlite_email_schema(self, conn: _SupportsExecutescript) -> None: ...


def ensure_sqlite_post_core_structures(
    db: SupportsSqlitePostCoreStructures,
    conn: _SupportsExecutescript,
) -> None:
    """Ensure follow-up SQLite structures after the core media schema exists."""

    db._ensure_sqlite_data_tables(conn)
    ensure_sqlite_fts_structures(db, conn)
    conn.executescript(_SQLITE_COLLECTIONS_AND_CONTENT_ITEMS_SQL)
    db._ensure_sqlite_visibility_columns(conn)
    db._ensure_sqlite_source_hash_column(conn)
    db._ensure_sqlite_claims_extensions(conn)
    db._ensure_sqlite_email_schema(conn)


def bootstrap_sqlite_schema(db: SupportsSqlitePostCoreStructures) -> None:
    """Initialize or migrate the SQLite schema through package-owned coordination."""

    conn = db.get_connection()
    try:
        current_db_version = db._get_db_version(conn)
        target_version = db._CURRENT_SCHEMA_VERSION

        logging.info(
            "Checking DB schema. Current version: {}. Code supports: {}",
            current_db_version,
            target_version,
        )

        if current_db_version == target_version:
            logging.debug("Database schema is up to date.")
            try:
                conn.executescript(db._CLAIMS_TABLE_SQL)
                conn.executescript(db._MEDIA_FILES_TABLE_SQL)
                conn.executescript(db._TTS_HISTORY_TABLE_SQL)
                ensure_sqlite_post_core_structures(db, conn)
                logging.debug("Verified FTS tables and visibility columns exist.")
            except (sqlite3.Error, DatabaseError) as bootstrap_err:
                logging.warning(
                    "Could not verify/create FTS tables on already correct schema version: {}",
                    bootstrap_err,
                )
            return

        if current_db_version > target_version:
            raise SchemaError(
                f"Database schema version ({current_db_version}) is newer than supported by code ({target_version})."
            )

        if current_db_version == 0:
            apply_sqlite_core_media_schema(db, conn)
            final_db_version = db._get_db_version(conn)
            if final_db_version != target_version:
                raise SchemaError(
                    f"Schema applied, but final DB version is {final_db_version}, expected {target_version}."
                )
            logger.info("Database schema initialized to version {}.", target_version)
        elif current_db_version < target_version:
            try:
                if db.is_memory_db:
                    apply_sqlite_core_media_schema(db, conn)
                else:
                    conn.close()

                    migrations_dir = None
                    db_name = os.path.basename(db.db_path_str)
                    db_dir = os.path.dirname(db.db_path_str)
                    db_dir_lower = db_dir.lower()
                    if (
                        "test_" in db_name
                        or "tmp" in db_dir_lower
                        or "temp" in db_dir_lower
                        or "pytest" in db_dir_lower
                    ):
                        migrations_dir = os.path.abspath(
                            os.path.join(
                                os.path.dirname(__file__),
                                "..",
                                "..",
                                "..",
                                "migrations",
                            )
                        )
                    migrator = DatabaseMigrator(db.db_path_str, migrations_dir=migrations_dir)
                    result = migrator.migrate_to_version(target_version)

                    status = (result or {}).get("status")
                    if status in {"success", "no_migrations", "no_change"}:
                        if status == "success":
                            logger.info(
                                "Database migrated from version {} to {}",
                                result["previous_version"],
                                result["current_version"],
                            )
                        elif status == "no_migrations":
                            migrations_dir_used = (
                                (result or {}).get("migrations_dir")
                                or getattr(migrator, "migrations_dir", None)
                                or migrations_dir
                            )
                            available_versions = (result or {}).get("available_versions") or []
                            missing_versions = (result or {}).get("missing_versions") or []
                            raise SchemaError(
                                "No migration scripts available to upgrade database schema "
                                f"from version {current_db_version} to {target_version}. "
                                f"migrations_dir={migrations_dir_used}, "
                                f"discovered_versions={available_versions}, "
                                f"missing_versions={missing_versions}."
                            )
                        else:
                            logger.info(
                                "No migration scripts to apply (status={}); proceeding with FTS/setup checks",
                                status,
                            )
                        conn = sqlite3.connect(db.db_path_str, check_same_thread=False)
                        conn.row_factory = sqlite3.Row
                        final_db_version = db._get_db_version(conn)
                        if final_db_version != target_version:
                            raise SchemaError(
                                "Database schema did not reach expected version after migration run "
                                f"(status={status}, current={final_db_version}, expected={target_version})."
                            )
                        ensure_sqlite_post_core_structures(db, conn)
                    else:
                        raise SchemaError(f"Migration failed: {result}")
            except MigrationError as exc:
                raise SchemaError(f"Database migration failed: {exc}") from exc
        else:
            raise SchemaError(
                f"Migration needed from version {current_db_version} to {target_version}, but no migration path is defined."
            )
    except (DatabaseError, SchemaError, sqlite3.Error) as exc:
        logger.error(f"Schema initialization/migration failed: {exc}", exc_info=True)
        raise DatabaseError(f"Schema initialization failed: {exc}") from exc
    except MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Unexpected error during schema initialization: {exc}", exc_info=True)
        raise DatabaseError(f"Unexpected error applying schema: {exc}") from exc


__all__ = ["bootstrap_sqlite_schema", "ensure_sqlite_post_core_structures"]

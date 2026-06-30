"""Package-native helper utilities for PostgreSQL schema bootstrap."""

from __future__ import annotations

from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.media_db.errors import SchemaError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.policies import (
    ensure_postgres_policies,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.document_workspace_schema import (
    ensure_postgres_document_workspace_schema,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.core_media import (
    apply_postgres_core_media_schema,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.fts import (
    ensure_postgres_fts,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migrations import (
    run_postgres_migrations,
)

try:
    from loguru import logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging

    logger = logging.getLogger("media_db_postgres_schema_bootstrap")


class SupportsPostgresPostCoreStructures(Protocol):
    """Minimal DB surface required for Postgres post-bootstrap ensures."""

    def _ensure_postgres_collections_tables(self, conn: Any) -> None: ...
    def _ensure_postgres_tts_history(self, conn: Any) -> None: ...
    def _ensure_postgres_audio_presets(self, conn: Any) -> None: ...
    def _ensure_postgres_data_tables(self, conn: Any) -> None: ...
    def _ensure_postgres_source_hash_column(self, conn: Any) -> None: ...
    def _ensure_postgres_claims_extensions(self, conn: Any) -> None: ...
    def _ensure_postgres_email_schema(self, conn: Any) -> None: ...
    def _sync_postgres_sequences(self, conn: Any) -> None: ...
    _CURRENT_SCHEMA_VERSION: int
    backend: Any


def ensure_postgres_post_core_structures(
    db: SupportsPostgresPostCoreStructures,
    conn: Any,
) -> None:
    """Ensure non-core PostgreSQL schema structures after base bootstrap or migration."""

    db._ensure_postgres_collections_tables(conn)
    db._ensure_postgres_tts_history(conn)
    db._ensure_postgres_audio_presets(conn)
    db._ensure_postgres_data_tables(conn)
    ensure_postgres_document_workspace_schema(conn)
    db._ensure_postgres_source_hash_column(conn)
    db._ensure_postgres_claims_extensions(conn)
    db._ensure_postgres_email_schema(conn)
    db._sync_postgres_sequences(conn)
    ensure_postgres_policies(db, conn)


def bootstrap_postgres_schema(db: SupportsPostgresPostCoreStructures) -> None:
    """Initialize or migrate the PostgreSQL schema through package-owned coordination."""

    target_version = db._CURRENT_SCHEMA_VERSION
    backend = db.backend

    with backend.transaction() as conn:
        schema_exists = backend.table_exists("schema_version", connection=conn)

        if not schema_exists:
            apply_postgres_core_media_schema(db, conn)
            ensure_postgres_fts(db, conn)
            ensure_postgres_post_core_structures(db, conn)
            return

        result = backend.execute("SELECT version FROM schema_version LIMIT 1", connection=conn)
        current_version_raw = result.scalar if result else None
        current_version = int(current_version_raw or 0)

        if current_version > target_version:
            raise SchemaError(
                f"Database schema version ({current_version}) is newer than supported by code ({target_version})."
            )

        must_tables = [
            "media",
            "keywords",
            "mediakeywords",
            "transcripts",
            "mediachunks",
            "unvectorizedmediachunks",
            "documentversions",
            "documentversionidentifiers",
            "sync_log",
            "chunkingtemplates",
            "claims",
        ]
        missing = [table for table in must_tables if not backend.table_exists(table, connection=conn)]
        if missing:
            logger.warning(
                "Postgres schema_version exists but base tables missing: {}. Applying base schema.",
                missing,
            )
            apply_postgres_core_media_schema(db, conn)
            ensure_postgres_fts(db, conn)
            ensure_postgres_post_core_structures(db, conn)
            return

        if current_version < target_version:
            run_postgres_migrations(db, conn, current_version, target_version)

        ensure_postgres_fts(db, conn)
        ensure_postgres_post_core_structures(db, conn)


__all__ = [
    "bootstrap_postgres_schema",
    "ensure_postgres_document_workspace_schema",
    "ensure_postgres_post_core_structures",
]

"""Package-native helper utilities for PostgreSQL schema bootstrap."""

from __future__ import annotations

from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.media_db.errors import SchemaError
from tldw_Server_API.app.core.DB_Management.media_db.schema.document_workspace_schema import (
    ensure_postgres_document_workspace_schema,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.core_media import (
    apply_postgres_core_media_schema,
    ensure_postgres_keyword_identity,
    ensure_postgres_media_owner_identity,
    ensure_postgres_sync_log_contract,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.fts import (
    ensure_postgres_fts,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.features.policies import (
    ensure_postgres_policies,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_claims_cluster_rebuild_publication import (
    run_postgres_migrate_to_v27,
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

    _CURRENT_SCHEMA_VERSION: int
    backend: Any


_SAFE_TRANSCRIPT_TEXT_FUNCTION_SQL = """
CREATE OR REPLACE FUNCTION public.tldw_try_extract_normalized_transcript_text(
    input_text TEXT
)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
STRICT
SET search_path = pg_catalog
AS $function$
DECLARE
    parsed JSON;
    value_type TEXT;
BEGIN
    parsed := input_text::JSON;
    IF json_typeof(parsed) <> 'object' THEN
        RETURN NULL;
    END IF;
    value_type := json_typeof(parsed -> 'text');
    IF value_type IS NULL OR value_type = 'null' THEN
        RETURN '';
    END IF;
    IF value_type = 'string' THEN
        RETURN parsed ->> 'text';
    END IF;
    IF value_type = 'boolean' THEN
        IF (parsed ->> 'text')::BOOLEAN THEN
            RETURN 'True';
        END IF;
        RETURN 'False';
    END IF;
    RETURN parsed ->> 'text';
EXCEPTION
    WHEN data_exception THEN
        RETURN NULL;
END;
$function$;
"""


def _ensure_postgres_safe_transcript_extractor(
    db: SupportsPostgresPostCoreStructures,
    conn: Any,
) -> None:
    """Install the PG13-compatible non-throwing normalized-text extractor."""

    db.backend.execute(
        _SAFE_TRANSCRIPT_TEXT_FUNCTION_SQL,
        connection=conn,
    )


def ensure_postgres_post_core_structures(
    db: SupportsPostgresPostCoreStructures,
    conn: Any,
) -> None:
    """Ensure non-core PostgreSQL schema structures after base bootstrap or migration."""

    ensure_postgres_sync_log_contract(db, conn)
    ensure_postgres_media_owner_identity(db, conn)
    db._ensure_postgres_collections_tables(conn)
    db._ensure_postgres_tts_history(conn)
    db._ensure_postgres_audio_presets(conn)
    db._ensure_postgres_data_tables(conn)
    _ensure_postgres_safe_transcript_extractor(db, conn)
    ensure_postgres_document_workspace_schema(conn)
    db._ensure_postgres_source_hash_column(conn)
    db._ensure_postgres_claims_extensions(conn)
    run_postgres_migrate_to_v27(db, conn)
    db._ensure_postgres_email_schema(conn)
    # Sequence repair belongs to explicit migrations. A routine unscoped
    # handle may see no rows under forced RLS and must not rewind global IDs.
    ensure_postgres_policies(db, conn)
    # Last: it locks keywords, and requests lock media (RLS) before keywords.
    ensure_postgres_keyword_identity(db, conn)


def ensure_postgres_email_acceleration_if_missing(db: SupportsPostgresPostCoreStructures, conn: Any) -> None:
    """Retry optional acceleration only when its existing artifacts are absent."""
    ready = db.backend.execute(
        "SELECT to_regclass('public.idx_email_text_trigram') IS NOT NULL "
        "AND (SELECT count(*) FROM pg_statistic_ext s JOIN pg_namespace n ON n.oid=s.stxnamespace "
        "WHERE n.nspname=current_schema() AND s.stxname=ANY(%s))=2",
        (["email_media_visibility_stats", "email_text_condition_stats"],),
        connection=conn,
    ).scalar
    if ready is not True:
        from tldw_Server_API.app.core.DB_Management.media_db.schema.email_schema_structures import (
            _ensure_postgres_email_search_acceleration,
        )

        _ensure_postgres_email_search_acceleration(db, conn)


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

        authority = backend.execute(
            "SELECT pg_has_role(current_user,c.relowner,'USAGE') AS can_migrate "
            "FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            "WHERE n.nspname=current_schema() AND c.relname='media' AND c.relkind IN ('r','p')",
            connection=conn,
        ).first
        if authority is not None and not authority["can_migrate"]:
            if current_version != target_version:
                raise SchemaError("PostgreSQL content requires schema-owner migration before runtime startup")
            from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_runtime_validation import (
                validate_postgres_runtime_schema,
            )
            validate_postgres_runtime_schema(db, conn)
            return

        if (
            authority is not None
            and authority["can_migrate"] is True
            and current_version == target_version
        ):
            from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_runtime_validation import (
                validate_postgres_owner_reopen,
                validate_postgres_schema_catalog,
            )

            # Reopening a ready schema must not take DDL locks against live readers.
            try:
                validate_postgres_schema_catalog(db, conn)
                validate_postgres_owner_reopen(db, conn)
            except SchemaError:
                logger.debug("PostgreSQL catalog validation requires schema-owner repair")
            else:
                ensure_postgres_email_acceleration_if_missing(db, conn)
                return

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

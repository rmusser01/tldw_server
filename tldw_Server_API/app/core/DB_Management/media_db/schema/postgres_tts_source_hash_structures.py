"""Package-owned PostgreSQL TTS/source-hash ensure helpers."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_output_history_incarnation import (
    run_postgres_migrate_to_v27,
)

try:
    from loguru import logger
except ImportError:  # pragma: no cover - defensive fallback
    logger = logging.getLogger("media_db_postgres_tts_source_hash_structures")


class _PostgresStatementBackend(Protocol):
    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
        *,
        connection: object,
    ) -> object: ...

    def escape_identifier(self, name: str) -> str: ...


class PostgresTtsSourceHashDB(Protocol):
    """Protocol for DB objects exposing the backend needed by these helpers."""

    backend: _PostgresStatementBackend
    _TTS_HISTORY_TABLE_SQL: str


def ensure_postgres_tts_history(db: PostgresTtsSourceHashDB, conn: Any) -> None:
    """Ensure TTS history tables and indexes exist on PostgreSQL."""

    backend = db.backend
    try:
        backend.execute(
            (
                "CREATE TABLE IF NOT EXISTS tts_history ("
                "id BIGSERIAL PRIMARY KEY, "
                "user_id TEXT NOT NULL, "
                "created_at TIMESTAMPTZ NOT NULL, "
                "text TEXT, "
                "text_hash TEXT NOT NULL, "
                "text_length INTEGER, "
                "provider TEXT, "
                "model TEXT, "
                "voice_id TEXT, "
                "voice_name TEXT, "
                "voice_info TEXT, "
                "format TEXT, "
                "duration_ms INTEGER, "
                "generation_time_ms INTEGER, "
                "params_json TEXT, "
                "status TEXT, "
                "segments_json TEXT, "
                "favorite BOOLEAN NOT NULL DEFAULT FALSE, "
                "job_id BIGINT, "
                "output_id BIGINT, "
                "artifact_ids TEXT, "
                "artifact_deleted_at TIMESTAMPTZ, "
                "error_message TEXT, "
                "deleted BOOLEAN NOT NULL DEFAULT FALSE, "
                "deleted_at TIMESTAMPTZ"
                ")"
            ),
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tts_history_user_created "
            "ON tts_history(user_id, created_at DESC)",
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tts_history_user_favorite "
            "ON tts_history(user_id, favorite)",
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tts_history_user_provider "
            "ON tts_history(user_id, provider)",
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tts_history_user_model "
            "ON tts_history(user_id, model)",
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tts_history_user_voice_id "
            "ON tts_history(user_id, voice_id)",
            connection=conn,
        )
        backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_tts_history_user_text_hash "
            "ON tts_history(user_id, text_hash)",
            connection=conn,
        )
    except BackendDatabaseError as exc:
        logger.warning("Could not ensure tts_history table on PostgreSQL: {}", exc)
    # Receiver installation is required and must not be swallowed by legacy ensures.
    run_postgres_migrate_to_v27(db, conn)


def ensure_postgres_source_hash_column(db: PostgresTtsSourceHashDB, conn: Any) -> None:
    """Ensure Media.source_hash column and index exist on PostgreSQL."""

    backend = db.backend
    ident = backend.escape_identifier
    try:
        backend.execute(
            f"ALTER TABLE {ident('media')} ADD COLUMN IF NOT EXISTS {ident('source_hash')} TEXT",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_media_source_hash')} ON {ident('media')} ({ident('source_hash')})",
            connection=conn,
        )
    except BackendDatabaseError as exc:
        logger.warning("Could not ensure source_hash column/index on media: {}", exc)


__all__ = [
    "PostgresTtsSourceHashDB",
    "ensure_postgres_source_hash_column",
    "ensure_postgres_tts_history",
]

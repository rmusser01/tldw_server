"""PostgreSQL audio-preset schema helpers."""

from __future__ import annotations

from typing import Any, Protocol

try:
    from loguru import logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging

    logger = logging.getLogger("media_db_postgres_audio_presets")


class PostgresAudioPresetDB(Protocol):
    """Minimal DB surface required to ensure audio preset structures."""

    backend: Any


def ensure_postgres_audio_presets(db: PostgresAudioPresetDB, conn: Any) -> None:
    """Ensure audio preset tables and indexes exist on PostgreSQL."""
    try:
        db.backend.execute(
            """
            CREATE TABLE IF NOT EXISTS audio_presets (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                favorite BOOLEAN NOT NULL DEFAULT FALSE,
                is_default BOOLEAN NOT NULL DEFAULT FALSE,
                config_json TEXT NOT NULL,
                capability_assumptions_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT FALSE,
                deleted_at TEXT,
                version INTEGER NOT NULL DEFAULT 1
            )
            """,
            connection=conn,
        )
        db.backend.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audio_presets_user_kind_updated
            ON audio_presets(user_id, kind, deleted, updated_at DESC)
            """,
            connection=conn,
        )
        db.backend.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audio_presets_user_kind_favorite
            ON audio_presets(user_id, kind, favorite, deleted)
            """,
            connection=conn,
        )
        db.backend.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audio_presets_user_kind_default
            ON audio_presets(user_id, kind, is_default, deleted)
            """,
            connection=conn,
        )
    except Exception as exc:  # pragma: no cover - defensive schema bootstrap logging
        logger.warning("Could not ensure audio_presets table on PostgreSQL: {}", exc)
        raise


__all__ = ["ensure_postgres_audio_presets", "PostgresAudioPresetDB"]

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import (
    DatabasePool,
    build_postgres_in_clause,
    build_sqlite_in_clause,
)


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


@dataclass
class MediaIngestDedupeRepo:
    """Shared internal index for reusable media-ingest transcript sources."""

    db_pool: DatabasePool

    async def ensure_tables(self) -> None:
        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    """
                    CREATE TABLE IF NOT EXISTS media_ingest_dedupe_entries (
                        dedupe_key TEXT PRIMARY KEY,
                        key_type TEXT NOT NULL,
                        media_type TEXT NOT NULL,
                        source_user_id BIGINT NOT NULL,
                        source_media_id BIGINT NOT NULL,
                        source_url TEXT NULL,
                        source_hash TEXT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                await self.db_pool.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_media_ingest_dedupe_media_type
                    ON media_ingest_dedupe_entries (media_type, updated_at DESC)
                    """
                )
                return

            await self.db_pool.execute(
                """
                CREATE TABLE IF NOT EXISTS media_ingest_dedupe_entries (
                    dedupe_key TEXT PRIMARY KEY,
                    key_type TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    source_user_id INTEGER NOT NULL,
                    source_media_id INTEGER NOT NULL,
                    source_url TEXT NULL,
                    source_hash TEXT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await self.db_pool.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_media_ingest_dedupe_media_type
                ON media_ingest_dedupe_entries (media_type, updated_at DESC)
                """
            )
        except Exception as exc:
            logger.error("MediaIngestDedupeRepo.ensure_tables failed: {}", exc)
            raise

    @staticmethod
    def _normalize_datetime_for_postgres(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if row is None:
            return {}
        if isinstance(row, dict):
            return dict(row)
        try:
            return dict(row)
        except Exception:
            try:
                keys = row.keys()
                return {key: row[key] for key in keys}
            except Exception:
                return {}

    async def lookup_source(
        self,
        *,
        media_type: str,
        dedupe_keys: list[str],
    ) -> dict[str, Any] | None:
        keys = [str(key).strip() for key in dedupe_keys if str(key or "").strip()]
        if not keys:
            return None

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                placeholders, params = build_postgres_in_clause(keys, start_param=2)
                query = f"""
                    SELECT dedupe_key, key_type, media_type, source_user_id, source_media_id,
                           source_url, source_hash, created_at, updated_at
                    FROM media_ingest_dedupe_entries
                    WHERE media_type = $1
                      AND dedupe_key IN ({placeholders})
                    ORDER BY updated_at DESC
                    LIMIT 1
                """  # nosec B608
                row = await self.db_pool.fetchone(query, str(media_type), *params)
                return self._row_to_dict(row) if row else None

            placeholders, params = build_sqlite_in_clause(keys)
            query = f"""
                SELECT dedupe_key, key_type, media_type, source_user_id, source_media_id,
                       source_url, source_hash, created_at, updated_at
                FROM media_ingest_dedupe_entries
                WHERE media_type = ?
                  AND dedupe_key IN ({placeholders})
                ORDER BY updated_at DESC
                LIMIT 1
            """  # nosec B608
            row = await self.db_pool.fetchone(query, (str(media_type), *params))
            return self._row_to_dict(row) if row else None
        except Exception as exc:
            logger.error("MediaIngestDedupeRepo.lookup_source failed: {}", exc)
            raise

    async def upsert_source(
        self,
        *,
        dedupe_key: str,
        key_type: str,
        media_type: str,
        source_user_id: int,
        source_media_id: int,
        source_url: str | None = None,
        source_hash: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        dedupe_key_value = str(dedupe_key or "").strip()
        if not dedupe_key_value:
            raise ValueError("dedupe_key is required")
        key_type_value = str(key_type or "").strip().lower()
        if not key_type_value:
            raise ValueError("key_type is required")
        media_type_value = str(media_type or "").strip().lower()
        if not media_type_value:
            raise ValueError("media_type is required")

        try:
            if getattr(self.db_pool, "pool", None) is not None:
                await self.db_pool.execute(
                    """
                    INSERT INTO media_ingest_dedupe_entries (
                        dedupe_key, key_type, media_type, source_user_id, source_media_id,
                        source_url, source_hash, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
                    ON CONFLICT (dedupe_key) DO UPDATE SET
                        key_type = EXCLUDED.key_type,
                        media_type = EXCLUDED.media_type,
                        source_user_id = EXCLUDED.source_user_id,
                        source_media_id = EXCLUDED.source_media_id,
                        source_url = EXCLUDED.source_url,
                        source_hash = EXCLUDED.source_hash,
                        updated_at = EXCLUDED.updated_at
                    """,
                    dedupe_key_value,
                    key_type_value,
                    media_type_value,
                    int(source_user_id),
                    int(source_media_id),
                    _normalize_optional_text(source_url),
                    _normalize_optional_text(source_hash),
                    self._normalize_datetime_for_postgres(now),
                )
                return

            await self.db_pool.execute(
                """
                INSERT INTO media_ingest_dedupe_entries (
                    dedupe_key, key_type, media_type, source_user_id, source_media_id,
                    source_url, source_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dedupe_key) DO UPDATE SET
                    key_type = excluded.key_type,
                    media_type = excluded.media_type,
                    source_user_id = excluded.source_user_id,
                    source_media_id = excluded.source_media_id,
                    source_url = excluded.source_url,
                    source_hash = excluded.source_hash,
                    updated_at = excluded.updated_at
                """,
                (
                    dedupe_key_value,
                    key_type_value,
                    media_type_value,
                    int(source_user_id),
                    int(source_media_id),
                    _normalize_optional_text(source_url),
                    _normalize_optional_text(source_hash),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
        except Exception as exc:
            logger.error("MediaIngestDedupeRepo.upsert_source failed: {}", exc)
            raise


__all__ = ["MediaIngestDedupeRepo"]

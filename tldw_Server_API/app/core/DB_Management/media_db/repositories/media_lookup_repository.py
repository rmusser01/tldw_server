from __future__ import annotations

import json
import sqlite3
from math import ceil
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import (
    media_dedupe_url_candidates,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)


class MediaLookupRepository:
    """Repository for canonical media lookup reads."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> MediaLookupRepository:
        return cls(session=require_media_database_like(
            db,
            error_message="db_instance must be a Database object.",
        ))

    @staticmethod
    def _parse_safe_metadata(raw_value: Any) -> dict[str, Any] | None:
        """Normalize latest-version safe_metadata JSON for source-list consumers."""
        if isinstance(raw_value, dict):
            return dict(raw_value)
        if not isinstance(raw_value, str) or not raw_value.strip():
            return None
        try:
            parsed = json.loads(raw_value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def by_id(
        self,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        if not isinstance(media_id, int):
            raise InputError("media_id must be an integer.")  # noqa: TRY003

        query = "SELECT * FROM Media WHERE id = ?"
        params = [media_id]
        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"

        db = self.session
        try:
            cursor = db.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as exc:
            logger.error("Error fetching media by ID {}: {}", media_id, exc, exc_info=True)
            raise DatabaseError(f"Failed to fetch media by ID: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error fetching media by ID {}: {}",
                media_id,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected error fetching media by ID: {exc}") from exc  # noqa: TRY003

    def status_by_id(
        self,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        """Return lightweight media fields needed for source-readiness status."""
        if not isinstance(media_id, int):
            raise InputError("media_id must be an integer.")  # noqa: TRY003

        query = (
            "SELECT id, uuid, title, type, url, chunking_status, vector_processing, "
            "CASE WHEN content IS NOT NULL AND content <> '' THEN 1 ELSE 0 END AS has_content "
            "FROM Media WHERE id = ?"
        )
        params = [media_id]
        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"

        db = self.session
        try:
            cursor = db.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as exc:
            logger.error("Error fetching media status by ID {}: {}", media_id, exc, exc_info=True)
            raise DatabaseError(f"Failed to fetch media status by ID: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error fetching media status by ID {}: {}",
                media_id,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected error fetching media status by ID: {exc}") from exc  # noqa: TRY003

    def source_projection_by_id(
        self,
        media_id: int,
        *,
        max_chars: int,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return bounded text-only fields for standalone Slides generation."""
        if isinstance(media_id, bool) or not isinstance(media_id, int) or not 1 <= media_id <= 2**63 - 1:
            raise InputError("media_id must be a positive 64-bit integer.")  # noqa: TRY003
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
            raise InputError("max_chars must be a positive integer.")  # noqa: TRY003
        if owner_user_id is not None and (not isinstance(owner_user_id, str) or not owner_user_id.strip()):
            raise InputError("owner_user_id must be a non-empty string.")  # noqa: TRY003

        prefix_chars = max_chars + 1
        backend_name = getattr(getattr(self.session, "backend_type", None), "name", None)
        is_postgres = backend_name == "POSTGRESQL"
        if is_postgres and owner_user_id is None:
            raise InputError("owner_user_id is required for PostgreSQL source projections.")  # noqa: TRY003
        false_literal = "FALSE" if is_postgres else "0"
        owner_clause = ""
        owner_params: list[Any] = []
        if owner_user_id is not None:
            owner_params.append(owner_user_id.strip())
            if is_postgres:
                owner_clause = "AND COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ?"
            else:
                owner_clause = "AND (m.owner_user_id IS NULL OR CAST(m.owner_user_id AS TEXT) = ?)"
        if is_postgres:
            transcript_text = """
                CASE
                    WHEN LEFT(LTRIM(t.transcription), 1) = '{' THEN
                        COALESCE(
                            public.tldw_try_extract_normalized_transcript_text(
                                t.transcription
                            ),
                            t.transcription
                        )
                    ELSE t.transcription
                END
            """
        else:
            transcript_text = """
                CASE
                    WHEN json_valid(t.transcription)
                         AND json_type(t.transcription) = 'object'
                    THEN
                        CASE
                            WHEN json_type(t.transcription, '$.text') IS NULL
                                 OR json_type(t.transcription, '$.text') = 'null'
                            THEN ''
                            WHEN json_type(t.transcription, '$.text') = 'text'
                            THEN json_extract(t.transcription, '$.text')
                            WHEN json_type(t.transcription, '$.text') = 'true'
                            THEN 'True'
                            WHEN json_type(t.transcription, '$.text') = 'false'
                            THEN 'False'
                            ELSE CAST(json_extract(t.transcription, '$.text') AS TEXT)
                        END
                    ELSE t.transcription
                END
            """
        resolved_text = f"""
            COALESCE(
                NULLIF(TRIM({transcript_text}), ''),
                NULLIF(TRIM(dv.content), ''),
                NULLIF(TRIM(m.content), '')
            )
        """
        invalid_expression = "FALSE" if is_postgres else f"COALESCE(INSTR({resolved_text}, CHAR(0)), 0) > 0"

        query = f"""
            SELECT
                m.id AS id,
                SUBSTR(
                    {resolved_text},
                    1,
                    ?
                ) AS source_text,
                {invalid_expression} AS source_invalid
            FROM Media m
            LEFT JOIN Transcripts t
              ON t.id = COALESCE(
                    (
                        SELECT pointed_t.id
                        FROM Transcripts pointed_t
                        WHERE pointed_t.media_id = m.id
                          AND pointed_t.deleted = {false_literal}
                          AND pointed_t.transcription_run_id = m.latest_transcription_run_id
                        ORDER BY pointed_t.created_at DESC, pointed_t.id DESC
                        LIMIT 1
                    ),
                    (
                        SELECT fallback_t.id
                        FROM Transcripts fallback_t
                        WHERE fallback_t.media_id = m.id
                          AND fallback_t.deleted = {false_literal}
                        ORDER BY
                            CASE WHEN fallback_t.transcription_run_id IS NULL THEN 1 ELSE 0 END,
                            fallback_t.transcription_run_id DESC,
                            fallback_t.created_at DESC,
                            fallback_t.id DESC
                        LIMIT 1
                    )
                )
             AND t.deleted = {false_literal}
            LEFT JOIN DocumentVersions dv
              ON dv.id = (
                    SELECT selected_dv.id
                    FROM DocumentVersions selected_dv
                    WHERE selected_dv.media_id = m.id
                      AND selected_dv.deleted = {false_literal}
                    ORDER BY selected_dv.version_number DESC, selected_dv.id DESC
                    LIMIT 1
                )
             AND dv.deleted = {false_literal}
            WHERE m.id = ?
              AND m.deleted = {false_literal}
              AND m.is_trash = {false_literal}
              {owner_clause}
            LIMIT 1
        """  # nosec B608
        params = (
            prefix_chars,
            media_id,
            *owner_params,
        )

        failure_type: str | None = None
        try:
            result = self.session.execute_query(
                query,
                params,
                log_errors=False,
            ).fetchone()
            if not result:
                return None
            record = dict(result)
            invalid = record.get("source_invalid")
            if not isinstance(invalid, (bool, int)) or invalid not in (0, 1):
                raise DatabaseError("Invalid media source validation marker.")  # noqa: TRY003
            record["source_invalid"] = bool(invalid)
            return record
        except Exception as exc:  # noqa: BLE001 - source reads cross a redacted boundary
            failure_type = type(exc).__name__

        logger.error(
            "Media source projection failed for ID {} ({})",
            media_id,
            failure_type,
        )
        raise DatabaseError("Media source projection failed.")  # noqa: TRY003

    def by_uuid(
        self,
        media_uuid: str,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        if not media_uuid:
            raise InputError("media_uuid cannot be empty.")  # noqa: TRY003

        query = "SELECT * FROM Media WHERE uuid = ?"
        params = [media_uuid]
        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"

        db = self.session
        try:
            cursor = db.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as exc:
            logger.error("Error fetching media by UUID {}: {}", media_uuid, exc, exc_info=True)
            raise DatabaseError(f"Failed fetch media by UUID: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error fetching media by UUID {}: {}",
                media_uuid,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected error fetching media by UUID: {exc}") from exc  # noqa: TRY003

    def by_url(
        self,
        url: str,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        if not url:
            raise InputError("url cannot be empty or None.")  # noqa: TRY003

        url_candidates = media_dedupe_url_candidates(url)
        if not url_candidates:
            raise InputError("url cannot be empty or None.")  # noqa: TRY003

        if len(url_candidates) == 1:
            query = "SELECT * FROM Media WHERE url = ?"
            params = [url_candidates[0]]
        else:
            placeholders = ", ".join(["?"] * len(url_candidates))
            query = f"SELECT * FROM Media WHERE url IN ({placeholders})"  # nosec B608
            params = list(url_candidates)

        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"
        query += " LIMIT 1"

        db = self.session
        try:
            cursor = db.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as exc:
            logger.error("Error fetching media by URL {}: {}", url, exc, exc_info=True)
            raise DatabaseError(f"Failed fetch media by URL: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error fetching media by URL {}: {}",
                url,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected error fetching media by URL: {exc}") from exc  # noqa: TRY003

    def by_hash(
        self,
        content_hash: str,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        if not content_hash:
            raise InputError("content_hash cannot be empty or None.")  # noqa: TRY003

        query = "SELECT * FROM Media WHERE content_hash = ?"
        params = [content_hash]
        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"
        query += " LIMIT 1"

        db = self.session
        try:
            cursor = db.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as exc:
            logger.error("Error fetching media by hash {}: {}", content_hash, exc, exc_info=True)
            raise DatabaseError(f"Failed fetch media by hash: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error fetching media by hash {}: {}",
                content_hash,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected error fetching media by hash: {exc}") from exc  # noqa: TRY003

    def by_title(
        self,
        title: str,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        if not title:
            raise InputError("title cannot be empty or None.")  # noqa: TRY003

        query = "SELECT * FROM Media WHERE title = ?"
        params = [title]
        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"
        query += " ORDER BY last_modified DESC LIMIT 1"

        db = self.session
        try:
            cursor = db.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as exc:
            logger.error("Error fetching media by title {}: {}", title, exc, exc_info=True)
            raise DatabaseError(f"Failed fetch media by title: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error fetching media by title {}: {}",
                title,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected error fetching media by title: {exc}") from exc  # noqa: TRY003

    def distinct_media_types(
        self,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> list[str]:
        conditions = ["type IS NOT NULL AND type != ''"]
        if not include_deleted:
            conditions.append("deleted = 0")
        if not include_trash:
            conditions.append("is_trash = 0")

        query = f"SELECT DISTINCT type FROM Media WHERE {' AND '.join(conditions)} ORDER BY type ASC"  # nosec B608

        db = self.session
        try:
            cursor = db.execute_query(query)
            return [row["type"] for row in cursor.fetchall() if row["type"]]
        except (DatabaseError, sqlite3.Error) as exc:
            logger.error("Error fetching distinct media types: {}", exc, exc_info=True)
            raise DatabaseError(f"Failed fetch distinct media types: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                "Unexpected error fetching distinct media types: {}",
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Unexpected error fetching distinct media types: {exc}") from exc  # noqa: TRY003

    def paginated_files(
        self,
        *,
        page: int = 1,
        results_per_page: int = 50,
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        if page < 1:
            raise ValueError("Page number must be 1 or greater.")  # noqa: TRY003
        if results_per_page < 1:
            raise ValueError("Results per page must be 1 or greater.")  # noqa: TRY003

        offset = (page - 1) * results_per_page
        db = self.session
        try:
            count_cursor = db.execute_query(
                "SELECT COUNT(*) AS total_items FROM Media WHERE deleted = 0 AND is_trash = 0"
            )
            count_result = count_cursor.fetchone()
            total_items = count_result["total_items"] if count_result else 0

            results: list[dict[str, Any]] = []
            if total_items > 0:
                items_cursor = db.execute_query(
                    """
                    SELECT
                        m.id,
                        m.title,
                        m.type,
                        m.ingestion_date,
                        m.last_modified,
                        m.chunking_status,
                        latest_source_metadata.safe_metadata AS safe_metadata
                    FROM Media m
                    LEFT JOIN (
                        SELECT media_id, safe_metadata
                        FROM (
                            SELECT
                                dv.media_id,
                                dv.safe_metadata,
                                ROW_NUMBER() OVER (
                                    PARTITION BY dv.media_id
                                    ORDER BY dv.version_number DESC, dv.id DESC
                                ) AS row_number
                            FROM DocumentVersions dv
                            WHERE dv.deleted = 0
                        ) latest_document_versions
                        WHERE row_number = 1
                    ) latest_source_metadata ON latest_source_metadata.media_id = m.id
                    WHERE m.deleted = 0
                      AND m.is_trash = 0
                    ORDER BY m.last_modified DESC, m.id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (results_per_page, offset),
                )
                results = []
                for row in items_cursor.fetchall():
                    item = dict(row)
                    item["safe_metadata"] = self._parse_safe_metadata(item.get("safe_metadata"))
                    results.append(item)

            total_pages = ceil(total_items / results_per_page) if total_items > 0 else 0
            return results, total_pages, page, total_items
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            logger.error("Error fetching paginated files: {}", exc, exc_info=True)
            raise DatabaseError(f"Failed pagination query: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error("Unexpected error fetching paginated files: {}", exc, exc_info=True)
            raise DatabaseError(f"Unexpected error during pagination: {exc}") from exc  # noqa: TRY003

    def paginated_trash(
        self,
        *,
        page: int = 1,
        results_per_page: int = 10,
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        if page < 1:
            raise ValueError("Page number must be 1 or greater.")  # noqa: TRY003
        if results_per_page < 1:
            raise ValueError("Results per page must be 1 or greater.")  # noqa: TRY003

        offset = (page - 1) * results_per_page
        db = self.session
        try:
            count_cursor = db.execute_query(
                "SELECT COUNT(*) AS total_items FROM Media WHERE deleted = 0 "
                "AND is_trash = 1 AND system_operation_id IS NULL"
            )
            count_row = count_cursor.fetchone()
            total_items = count_row["total_items"] if count_row else 0

            results: list[dict[str, Any]] = []
            if total_items > 0:
                items_cursor = db.execute_query(
                    """
                    SELECT id, title, type, uuid
                    FROM Media
                    WHERE deleted = 0
                      AND is_trash = 1
                      AND system_operation_id IS NULL
                    ORDER BY trash_date DESC, last_modified DESC, id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (results_per_page, offset),
                )
                results = [dict(row) for row in items_cursor.fetchall()]

            total_pages = ceil(total_items / results_per_page) if total_items > 0 else 0
            return results, total_pages, page, total_items
        except DatabaseError:
            raise
        except sqlite3.Error as exc:
            logger.error("Error fetching paginated trash files: {}", exc, exc_info=True)
            raise DatabaseError(f"Failed trash pagination query: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error("Unexpected error fetching paginated trash files: {}", exc, exc_info=True)
            raise DatabaseError(f"Unexpected error during trash pagination: {exc}") from exc  # noqa: TRY003


__all__ = ["MediaLookupRepository"]

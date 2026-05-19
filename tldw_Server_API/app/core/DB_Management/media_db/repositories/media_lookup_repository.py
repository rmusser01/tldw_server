from __future__ import annotations

import json
from math import ceil
import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import (
    media_dedupe_url_candidates,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)


class MediaLookupRepository:
    """Repository for canonical media lookup reads."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> "MediaLookupRepository":
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
                "SELECT COUNT(*) AS total_items FROM Media WHERE deleted = 0 AND is_trash = 1"
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

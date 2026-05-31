"""Legacy media state helpers extracted from the media DB shim."""

from __future__ import annotations

import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.dedupe_urls import (
    media_dedupe_url_candidates,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError, DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)


def check_media_exists(
    db_instance: MediaDbLike,
    media_id: int | None = None,
    url: str | None = None,
    content_hash: str | None = None,
) -> int | None:
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )

    query_parts: list[str] = []
    params: list[Any] = []
    if media_id is not None:
        query_parts.append("id = ?")
        params.append(media_id)
    if url:
        url_candidates = media_dedupe_url_candidates(url)
        if len(url_candidates) == 1:
            query_parts.append("url = ?")
            params.append(url_candidates[0])
        elif url_candidates:
            placeholders = ", ".join(["?"] * len(url_candidates))
            query_parts.append(f"url IN ({placeholders})")
            params.extend(url_candidates)
    if content_hash:
        query_parts.append("content_hash = ?")
        params.append(content_hash)
    if not query_parts:
        raise ValueError("Must provide id, url, or content_hash to check.")  # noqa: TRY003

    query = f"SELECT id FROM Media WHERE ({' OR '.join(query_parts)}) AND deleted = 0 LIMIT 1"  # nosec B608
    try:
        cursor = db_instance.execute_query(query, tuple(params))
        result = cursor.fetchone()
        return result["id"] if result else None
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(f"Error checking media existence DB '{db_instance.db_path_str}'")
        raise DatabaseError(f"Failed check media existence: {exc}") from exc  # noqa: TRY003


def get_unprocessed_media(db_instance: MediaDbLike) -> list[dict]:
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    try:
        query = (
            "SELECT id, uuid, content, type, title FROM Media "
            "WHERE vector_processing = 0 AND deleted = 0 AND is_trash = 0 ORDER BY id"
        )
        cursor = db_instance.execute_query(query)
        return [dict(row) for row in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(f"Error getting unprocessed media DB '{db_instance.db_path_str}'")
        raise DatabaseError("Failed get unprocessed media") from exc  # noqa: TRY003


def mark_media_as_processed(
    db_instance: MediaDbLike,
    media_id: int,
) -> None:
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    logger.debug(
        f"Marking media {media_id} vector_processing=1 on DB '{db_instance.db_path_str}'."
    )
    try:
        with db_instance.transaction() as conn:
            media_row = db_instance._fetchone_with_connection(
                conn,
                "SELECT uuid, version, vector_processing, chunking_status FROM Media WHERE id = ? AND deleted = 0",
                (media_id,),
            )
            if not media_row:
                logger.warning(
                    f"Attempted mark media {media_id} processed, but not found/deleted."
                )
                return
            chunking_status = str(media_row["chunking_status"] or "").strip().lower()
            if media_row["vector_processing"] == 1 and chunking_status in {"completed", "complete", "done"}:
                logger.debug(f"Media {media_id} already marked as processed.")
                return

            media_uuid = media_row["uuid"]
            current_version = media_row["version"]
            next_version = current_version + 1
            now = db_instance._get_current_utc_timestamp_str()

            cursor = db_instance._execute_with_connection(
                conn,
                """
                UPDATE Media
                SET vector_processing = 1,
                    chunking_status = ?,
                    last_modified = ?,
                    version = ?,
                    client_id = ?
                WHERE id = ? AND version = ?
                """,
                ("completed", now, next_version, db_instance.client_id, media_id, current_version),
            )
            if cursor.rowcount == 0:
                raise ConflictError(
                    "Concurrent modification detected for media state update.",
                    entity="Media",
                    identifier=media_id,
                )
            payload = db_instance._fetchone_with_connection(
                conn,
                "SELECT * FROM Media WHERE id = ?",
                (media_id,),
            ) or {}
            db_instance._log_sync_event(conn, "Media", media_uuid, "update", next_version, payload)
    except (ConflictError, DatabaseError, sqlite3.Error) as exc:
        logger.exception(
            f"Error marking media {media_id} processed '{db_instance.db_path_str}'"
        )
        if isinstance(exc, (ConflictError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed mark media {media_id} processed") from exc  # noqa: TRY003


__all__ = [
    "check_media_exists",
    "get_unprocessed_media",
    "mark_media_as_processed",
]

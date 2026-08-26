"""Runtime-owned state mutation wrappers for canonical MediaDatabase methods."""

from __future__ import annotations

import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)


def update_media_reprocess_state(
    self,
    media_id: int,
    *,
    chunking_status: str | None,
    reset_vector_processing: bool,
) -> None:
    """Update media processing state with sync logging and optimistic concurrency."""
    try:
        with self.transaction() as conn:
            row = self._fetchone_with_connection(
                conn,
                "SELECT uuid, version FROM Media WHERE id = ? AND deleted = 0 "
                "AND is_trash = 0 AND system_operation_id IS NULL",
                (media_id,),
            )
            if not row:
                raise InputError(f"Media {media_id} not found or inactive.")  # noqa: TRY003, TRY301
            media_uuid = row["uuid"]
            current_version = row["version"]
            next_version = current_version + 1
            now = self._get_current_utc_timestamp_str()

            set_parts = ["last_modified = ?", "version = ?", "client_id = ?"]
            params: list[Any] = [now, next_version, self.client_id]
            payload: dict[str, Any] = {"last_modified": now}

            if chunking_status is not None:
                set_parts.append("chunking_status = ?")
                params.append(chunking_status)
                payload["chunking_status"] = chunking_status

            if reset_vector_processing:
                set_parts.append("vector_processing = ?")
                params.append(0)
                payload["vector_processing"] = 0

            update_sql = f"UPDATE Media SET {', '.join(set_parts)} WHERE id = ? AND version = ? AND system_operation_id IS NULL"  # nosec B608
            update_params = (*params, media_id, current_version)
            cursor = self._execute_with_connection(conn, update_sql, update_params)
            if cursor.rowcount == 0:
                raise ConflictError("Media", media_id)  # noqa: TRY301

            self._log_sync_event(conn, "Media", media_uuid, "update", next_version, payload)
    except (InputError, ConflictError):
        raise
    except (DatabaseError, sqlite3.Error) as exc:
        logger.error(f"Error updating reprocess state for media {media_id}: {exc}", exc_info=True)
        if isinstance(exc, DatabaseError):
            raise
        raise DatabaseError(f"Failed updating reprocess state: {exc}") from exc  # noqa: TRY003
    except MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Unexpected error updating reprocess state for media {media_id}: {exc}", exc_info=True)
        raise DatabaseError(f"Unexpected error updating reprocess state: {exc}") from exc  # noqa: TRY003


def mark_embeddings_error(self, media_id: int, error_message: str) -> None:
    """Mark embeddings processing as failed for a media item."""
    try:
        with self.transaction() as conn:
            row = self._fetchone_with_connection(
                conn,
                "SELECT uuid, version FROM Media WHERE id = ? AND deleted = 0 "
                "AND is_trash = 0 AND system_operation_id IS NULL",
                (media_id,),
            )
            if not row:
                raise InputError(f"Media {media_id} not found or inactive.")  # noqa: TRY003, TRY301
            media_uuid = row["uuid"]
            current_version = row["version"]
            next_version = current_version + 1
            now = self._get_current_utc_timestamp_str()

            safe_message = str(error_message).replace("\r", " ").replace("\n", " ").strip()
            if not safe_message:
                safe_message = "unknown error"
            max_error_len = 500
            if len(safe_message) > max_error_len:
                safe_message = f"{safe_message[: max_error_len - 3]}..."
            error_status = f"embeddings_error: {safe_message}"
            params: list[Any] = [
                now,
                next_version,
                self.client_id,
                -1,
                error_status,
            ]
            payload: dict[str, Any] = {
                "last_modified": now,
                "vector_processing": -1,
                "chunking_status": error_status,
            }

            update_sql = (
                "UPDATE Media SET last_modified = ?, version = ?, client_id = ?, "
                "vector_processing = ?, chunking_status = ? WHERE id = ? AND version = ? "
                "AND system_operation_id IS NULL"
            )
            update_params = (*params, media_id, current_version)
            cursor = self._execute_with_connection(conn, update_sql, update_params)
            if cursor.rowcount == 0:
                raise ConflictError("Media", media_id)  # noqa: TRY301

            self._log_sync_event(conn, "Media", media_uuid, "update", next_version, payload)
    except (InputError, ConflictError):
        raise
    except (DatabaseError, sqlite3.Error) as exc:
        logger.error(f"Error marking embeddings error for media {media_id}: {exc}", exc_info=True)
        if isinstance(exc, DatabaseError):
            raise
        raise DatabaseError(f"Failed marking embeddings error: {exc}") from exc  # noqa: TRY003
    except MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Unexpected error marking embeddings error for media {media_id}: {exc}", exc_info=True)
        raise DatabaseError(f"Unexpected error marking embeddings error: {exc}") from exc  # noqa: TRY003


__all__ = [
    "mark_embeddings_error",
    "update_media_reprocess_state",
]

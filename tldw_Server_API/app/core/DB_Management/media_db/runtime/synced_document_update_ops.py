"""Package-owned helper for synced document content updates."""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.collections import (
    load_collections_database_cls,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)


_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS
_COLLECTIONS_DB = load_collections_database_cls()


def apply_synced_document_content_update(
    self: Any,
    *,
    media_id: int,
    content: str,
    prompt: str | None = None,
    analysis_content: str | None = None,
    safe_metadata: str | None = None,
) -> dict[str, Any]:
    """Apply an externally-sourced document content update atomically."""
    if content is None:
        raise InputError("Content is required for synced document updates.")  # noqa: TRY003

    client_id = self.client_id
    current_time = self._get_current_utc_timestamp_str()

    try:
        with self.transaction() as conn:
            media_info = self._fetchone_with_connection(
                conn,
                "SELECT uuid, version, title, content FROM Media WHERE id = ? AND deleted = 0",
                (media_id,),
            )
            if not media_info:
                raise InputError(f"Media {media_id} not found or deleted.")  # noqa: TRY003, TRY301

            media_uuid = media_info["uuid"]
            current_media_version = media_info["version"]
            current_title = media_info["title"]
            current_content = media_info["content"]
            new_media_version = current_media_version + 1
            new_content_hash = hashlib.sha256(content.encode()).hexdigest()

            update_cursor = self._execute_with_connection(
                conn,
                """
                UPDATE Media
                SET content = ?,
                    content_hash = ?,
                    last_modified = ?,
                    version = ?,
                    client_id = ?,
                    chunking_status = 'pending',
                    vector_processing = 0
                WHERE id = ? AND version = ?
                """,
                (
                    content,
                    new_content_hash,
                    current_time,
                    new_media_version,
                    client_id,
                    media_id,
                    current_media_version,
                ),
            )
            if getattr(update_cursor, "rowcount", 0) == 0:
                raise ConflictError("Media", media_id)  # noqa: TRY301

            new_doc_version_info = self.create_document_version(
                media_id=media_id,
                content=content,
                prompt=prompt,
                analysis_content=analysis_content,
                safe_metadata=safe_metadata,
            )

            updated_media_data = self._fetchone_with_connection(
                conn,
                "SELECT * FROM Media WHERE id = ?",
                (media_id,),
            ) or {}
            updated_media_data["created_doc_ver_uuid"] = new_doc_version_info.get("uuid")
            updated_media_data["created_doc_ver_num"] = new_doc_version_info.get(
                "version_number"
            )
            self._log_sync_event(
                conn,
                "Media",
                media_uuid,
                "update",
                new_media_version,
                updated_media_data,
            )
            self._update_fts_media(
                conn,
                media_id,
                current_title,
                content,
                old_title=current_title,
                old_content=current_content,
            )

        logger.info(
            "Applied synced content update for media {}. New doc version: {}, new media version: {}",
            media_id,
            new_doc_version_info.get("version_number"),
            new_media_version,
        )
        try:
            if _COLLECTIONS_DB is not None and client_id is not None:
                _COLLECTIONS_DB.from_backend(
                    user_id=str(client_id),
                    backend=self.backend,
                ).mark_highlights_stale_if_content_changed(media_id, new_content_hash)
        except _MEDIA_NONCRITICAL_EXCEPTIONS as anch_err:
            logger.debug("Highlight re-anchoring hook (sync update) failed: {}", anch_err)
        try:
            from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
                invalidate_intra_doc_vectors,
            )

            invalidate_intra_doc_vectors(str(media_id))
        except _MEDIA_NONCRITICAL_EXCEPTIONS as rag_err:
            logger.debug(
                "Intra-doc vector invalidation skipped for media {}: {}",
                media_id,
                rag_err,
            )
    except (InputError, ConflictError, DatabaseError, sqlite3.Error, TypeError) as exc:
        logger.error(
            f"Synced content update error media {media_id}: {exc}",
            exc_info=True,
        )
        if isinstance(exc, (InputError, ConflictError, DatabaseError, TypeError)):
            raise
        raise DatabaseError(f"Synced content update failed: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            f"Unexpected synced content update error media {media_id}: {exc}",
            exc_info=True,
        )
        raise DatabaseError(f"Unexpected synced content update error: {exc}") from exc  # noqa: TRY003
    else:
        return {
            "media_id": media_id,
            "content_hash": new_content_hash,
            "new_media_version": new_media_version,
            "document_version_number": new_doc_version_info.get("version_number"),
            "document_version_uuid": new_doc_version_info.get("uuid"),
        }


__all__ = ["apply_synced_document_content_update"]

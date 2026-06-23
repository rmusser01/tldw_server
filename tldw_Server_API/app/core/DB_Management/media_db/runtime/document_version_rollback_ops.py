"""Rollback helpers owned by the package-native Media DB runtime."""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_wrappers import (
    get_document_version,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.collections import (
    load_collections_database_cls,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

try:
    from loguru import logger

    logging = logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging as _stdlib_logging

    logger = _stdlib_logging.getLogger("media_db_document_version_rollback")
    logging = logger

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS
_CollectionsDB = load_collections_database_cls()


def rollback_to_version(
    self: Any,
    media_id: int,
    target_version_number: int,
) -> dict[str, Any]:
    """Roll back the active media row to a previous document-version state."""

    if not isinstance(target_version_number, int) or target_version_number < 1:
        raise ValueError("Target version invalid.")  # noqa: TRY003

    client_id = self.client_id
    current_time = self._get_current_utc_timestamp_str()
    logger.debug(
        f"Rolling back media {media_id} to doc version {target_version_number}."
    )

    try:
        with self.transaction() as conn:
            media_info = self._fetchone_with_connection(
                conn,
                "SELECT uuid, version, title, content FROM Media WHERE id = ? AND deleted = 0",
                (media_id,),
            )
            if not media_info:
                return {"error": f"Media {media_id} not found or deleted."}

            media_uuid = media_info["uuid"]
            current_media_version = media_info["version"]
            current_title = media_info["title"]
            current_content = media_info["content"]
            new_media_version = current_media_version + 1

            target_version_data = get_document_version(
                self,
                media_id,
                target_version_number,
                True,
            )
            if target_version_data is None:
                return {
                    "error": (
                        f"Rollback target version {target_version_number} "
                        "not found or inactive."
                    )
                }

            latest_vn_row = self._fetchone_with_connection(
                conn,
                (
                    "SELECT MAX(version_number) AS latest_vn FROM DocumentVersions "
                    "WHERE media_id=? AND deleted=0"
                ),
                (media_id,),
            )
            latest_vn = latest_vn_row["latest_vn"] if latest_vn_row else None
            if latest_vn is not None and target_version_number == latest_vn:
                return {"error": "Cannot rollback to the current latest version number."}

            target_content = target_version_data.get("content")
            target_prompt = target_version_data.get("prompt")
            target_analysis = target_version_data.get("analysis_content")
            if target_content is None:
                return {"error": f"Version {target_version_number} has no content."}

            new_doc_version_info = self.create_document_version(
                media_id=media_id,
                content=target_content,
                prompt=target_prompt,
                analysis_content=target_analysis,
            )
            new_doc_version_number = new_doc_version_info.get("version_number")
            new_doc_version_uuid = new_doc_version_info.get("uuid")

            new_content_hash = hashlib.sha256(target_content.encode()).hexdigest()
            update_cursor = self._execute_with_connection(
                conn,
                """UPDATE Media SET content=?, content_hash=?, last_modified=?, version=?, client_id=?,
                   chunking_status='pending', vector_processing=0 WHERE id=? AND version=?""",
                (
                    target_content,
                    new_content_hash,
                    current_time,
                    new_media_version,
                    client_id,
                    media_id,
                    current_media_version,
                ),
            )
            if update_cursor.rowcount == 0:
                raise ConflictError("Media", media_id)  # noqa: TRY301

            updated_media_data = self._fetchone_with_connection(
                conn,
                "SELECT * FROM Media WHERE id = ?",
                (media_id,),
            ) or {}
            updated_media_data["rolled_back_to_doc_ver_uuid"] = new_doc_version_uuid
            updated_media_data["rolled_back_to_doc_ver_num"] = new_doc_version_number
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
                target_content,
                old_title=current_title,
                old_content=current_content,
            )

        logger.info(
            "Rolled back media {} to state of doc ver {}. New DocVer: {}, New MediaVer: {}",
            media_id,
            target_version_number,
            new_doc_version_number,
            new_media_version,
        )

        try:
            if _CollectionsDB is not None and client_id is not None:
                _CollectionsDB.from_backend(
                    user_id=str(client_id),
                    backend=self.backend,
                ).mark_highlights_stale_if_content_changed(media_id, new_content_hash)
        except _MEDIA_NONCRITICAL_EXCEPTIONS as anch_err:
            logging.debug(f"Highlight re-anchoring hook (rollback) failed: {anch_err}")

        try:
            from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
                invalidate_intra_doc_vectors,
            )

            invalidate_intra_doc_vectors(str(media_id))
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            pass
    except (InputError, ValueError, ConflictError, DatabaseError, sqlite3.Error, TypeError) as exc:
        logger.error(f"Rollback error media {media_id}: {exc}", exc_info=True)
        if isinstance(exc, (InputError, ValueError, ConflictError, DatabaseError, TypeError)):
            raise
        raise DatabaseError(f"DB error during rollback: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Unexpected rollback error media {media_id}: {exc}", exc_info=True)
        raise DatabaseError(f"Unexpected rollback error: {exc}") from exc  # noqa: TRY003
    else:
        return {
            "success": (
                f"Rolled back to version {target_version_number}. "
                f"State saved as new version {new_doc_version_number}."
            ),
            "new_document_version_number": new_doc_version_number,
            "new_document_version_uuid": new_doc_version_uuid,
            "new_media_version": new_media_version,
        }


__all__ = ["rollback_to_version"]

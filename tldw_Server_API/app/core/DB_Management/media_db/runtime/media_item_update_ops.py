"""Package-owned helper for user-facing media item updates."""

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
_OWNED_UPDATE_FIELDS = frozenset({"title", "author", "type", "content"})


def apply_media_metadata_patch(
    self: Any,
    media_id: int,
    title: str | None = None,
    author: str | None = None,
    keywords_add: list[str] | tuple[str, ...] = (),
) -> dict[str, int]:
    """Apply the reviewed metadata-only allowlist in one Media DB transaction."""
    if type(media_id) is not int or media_id < 1:
        raise InputError("media_id must be a positive integer")  # noqa: TRY003

    def _text(value: str | None, field: str) -> str | None:
        if value is None:
            return None
        if type(value) is not str or not value.strip() or len(value.strip()) > 500:
            raise InputError(f"{field} must be a string between 1 and 500 characters")  # noqa: TRY003
        return value.strip()

    title_value = _text(title, "title")
    author_value = _text(author, "author")
    if type(keywords_add) not in {list, tuple} or len(keywords_add) > 100:
        raise InputError("keywords_add must be a list or tuple of at most 100 strings")  # noqa: TRY003
    additions: list[str] = []
    seen_additions: set[str] = set()
    for keyword in keywords_add:
        if type(keyword) is not str or not keyword.strip() or len(keyword.strip()) > 128:
            raise InputError("keywords_add must contain strings between 1 and 128 characters")  # noqa: TRY003
        normalized = keyword.strip()
        folded = normalized.casefold()
        if folded not in seen_additions:
            additions.append(normalized)
            seen_additions.add(folded)
    if title_value is None and author_value is None and not additions:
        raise InputError("At least one metadata patch field is required")  # noqa: TRY003

    try:
        with self.transaction() as conn:
            current = self._fetchone_with_connection(
                conn,
                """
                SELECT id, uuid, title, author, content, version
                FROM Media
                WHERE id = ? AND deleted = 0 AND is_trash = 0
                """,
                (media_id,),
            )
            if not current:
                raise InputError(f"Media {media_id} not found or inactive/trashed.")  # noqa: TRY003, TRY301
            keyword_rows = self._fetchall_with_connection(
                conn,
                """
                SELECT k.keyword
                FROM Keywords k
                JOIN MediaKeywords mk ON mk.keyword_id = k.id
                WHERE mk.media_id = ? AND k.deleted = 0
                ORDER BY k.keyword
                """,
                (media_id,),
            )
            keywords = [str(row["keyword"]) for row in keyword_rows]
            seen = {keyword.casefold() for keyword in keywords}
            keywords_changed = False
            for keyword in additions:
                if keyword.casefold() not in seen:
                    keywords.append(keyword)
                    seen.add(keyword.casefold())
                    keywords_changed = True

            current_version = int(current["version"])
            title_changed = title_value is not None and title_value != current.get("title")
            author_changed = author_value is not None and author_value != current.get("author")
            if not title_changed and not author_changed and not keywords_changed:
                return {"media_id": media_id, "new_media_version": current_version}
            next_version = current_version + 1
            now = self._get_current_utc_timestamp_str()
            assignments = ["last_modified = ?", "version = ?", "client_id = ?"]
            params: list[Any] = [now, next_version, self.client_id]
            if title_changed:
                assignments.append("title = ?")
                params.append(title_value)
            if author_changed:
                assignments.append("author = ?")
                params.append(author_value)
            result = self._execute_with_connection(
                conn,
                f"UPDATE Media SET {', '.join(assignments)} WHERE id = ? AND version = ?",  # nosec B608
                (*params, media_id, current_version),
            )
            if getattr(result, "rowcount", 0) != 1:
                raise ConflictError("Media", media_id)  # noqa: TRY301

            if keywords_changed:
                self.update_keywords_for_media(media_id, keywords, conn=conn)
            if title_changed:
                self._update_fts_media(
                    conn,
                    media_id,
                    title_value,
                    current.get("content"),
                    old_title=current.get("title"),
                    old_content=current.get("content"),
                )
            updated = (
                self._fetchone_with_connection(
                    conn,
                    "SELECT * FROM Media WHERE id = ?",
                    (media_id,),
                )
                or {}
            )
            self._log_sync_event(
                conn,
                "Media",
                current["uuid"],
                "update",
                next_version,
                dict(updated),
            )
        return {"media_id": media_id, "new_media_version": next_version}
    except (InputError, ConflictError, DatabaseError, TypeError):
        raise
    except sqlite3.Error as exc:
        raise DatabaseError(f"Media metadata patch failed: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        raise DatabaseError(f"Unexpected media metadata patch error: {exc}") from exc  # noqa: TRY003


def apply_media_item_update(
    self: Any,
    *,
    media_id: int,
    fields: dict[str, Any],
    prompt: str | None = None,
    analysis_content: str | None = None,
) -> dict[str, Any]:
    """Apply a user-facing media item update and return side-effect metadata."""
    unsupported_fields = sorted(set(fields) - _OWNED_UPDATE_FIELDS)
    if unsupported_fields:
        joined = ", ".join(unsupported_fields)
        raise InputError(f"Unsupported media update field(s): {joined}")  # noqa: TRY003

    fields = {key: value for key, value in fields.items() if value is not None}
    if not fields:
        raise InputError("At least one media update field is required.")  # noqa: TRY003

    client_id = self.client_id
    current_time = self._get_current_utc_timestamp_str()
    content_provided = "content" in fields and fields["content"] is not None
    new_doc_version_info: dict[str, Any] | None = None
    content_actually_changed = False

    try:
        with self.transaction() as conn:
            media_info = self._fetchone_with_connection(
                conn,
                """
                SELECT id, uuid, title, content, content_hash, version
                FROM Media
                WHERE id = ? AND deleted = 0 AND is_trash = 0
                """,
                (media_id,),
            )
            if not media_info:
                raise InputError(f"Media {media_id} not found or inactive/trashed.")  # noqa: TRY003, TRY301

            media_uuid = media_info["uuid"]
            current_title = media_info["title"]
            current_content = media_info["content"]
            current_hash = media_info["content_hash"]
            current_media_version = media_info["version"]
            new_media_version = current_media_version + 1
            resulting_content_hash = current_hash

            set_parts = [
                "last_modified = ?",
                "version = ?",
                "client_id = ?",
            ]
            params: list[Any] = [
                current_time,
                new_media_version,
                client_id,
            ]

            if "title" in fields:
                set_parts.append("title = ?")
                params.append(fields["title"])
            if "author" in fields:
                set_parts.append("author = ?")
                params.append(fields["author"])
            if "type" in fields:
                set_parts.append("type = ?")
                params.append(fields["type"])

            new_content = fields.get("content") if content_provided else None
            if content_provided:
                new_content_hash = hashlib.sha256(new_content.encode()).hexdigest()
                resulting_content_hash = new_content_hash
                content_actually_changed = new_content_hash != current_hash
                if content_actually_changed:
                    logger.info(
                        "Content changed for media {}. Updating content and hash.",
                        media_id,
                    )
                    set_parts.extend(
                        [
                            "content = ?",
                            "content_hash = ?",
                            "chunking_status = ?",
                            "vector_processing = ?",
                        ]
                    )
                    params.extend([new_content, new_content_hash, "pending", 0])
                else:
                    logger.info(
                        "Content provided for media {} but hash is identical. " "Content field not updated.",
                        media_id,
                    )

            sql_set_clause = ", ".join(set_parts)
            update_query = f"UPDATE Media SET {sql_set_clause} WHERE id = ? AND version = ?"  # nosec B608
            update_params = tuple(params + [media_id, current_media_version])

            update_cursor = self._execute_with_connection(
                conn,
                update_query,
                update_params,
            )
            if getattr(update_cursor, "rowcount", 0) == 0:
                raise ConflictError("Media", media_id)  # noqa: TRY301

            fts_title = fields.get("title", current_title)
            fts_content = new_content if content_actually_changed else current_content
            if "title" in fields or content_actually_changed:
                self._update_fts_media(
                    conn,
                    media_id,
                    fts_title,
                    fts_content,
                    old_title=current_title,
                    old_content=current_content,
                )

            if content_provided:
                new_doc_version_info = self.create_document_version(
                    media_id=media_id,
                    content=new_content,
                    prompt=prompt,
                    analysis_content=analysis_content,
                )

            updated_row = (
                self._fetchone_with_connection(
                    conn,
                    "SELECT * FROM Media WHERE id = ?",
                    (media_id,),
                )
                or {}
            )
            updated_media_data = dict(updated_row)
            if new_doc_version_info:
                updated_media_data["created_doc_ver_uuid"] = new_doc_version_info.get("uuid")
                updated_media_data["created_doc_ver_num"] = new_doc_version_info.get("version_number")

            self._log_sync_event(
                conn,
                "Media",
                media_uuid,
                "update",
                new_media_version,
                updated_media_data,
            )

        try:
            if content_actually_changed and _COLLECTIONS_DB is not None and client_id is not None:
                _COLLECTIONS_DB.from_backend(
                    user_id=str(client_id),
                    backend=self.backend,
                ).mark_highlights_stale_if_content_changed(
                    media_id,
                    resulting_content_hash,
                )
        except _MEDIA_NONCRITICAL_EXCEPTIONS as anch_err:
            logger.debug("Highlight re-anchoring hook (media update) failed: {}", anch_err)
    except (InputError, ConflictError, DatabaseError, TypeError):
        raise
    except sqlite3.Error as exc:
        logger.error(
            f"Media item update error media {media_id}: {exc}",
            exc_info=True,
        )
        raise DatabaseError(f"Media item update failed: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            f"Unexpected media item update error media {media_id}: {exc}",
            exc_info=True,
        )
        raise DatabaseError(f"Unexpected media item update error: {exc}") from exc  # noqa: TRY003
    else:
        return {
            "media_id": media_id,
            "content_hash": resulting_content_hash,
            "new_media_version": new_media_version,
            "content_changed": content_actually_changed,
            "document_version_number": (new_doc_version_info.get("version_number") if new_doc_version_info else None),
            "document_version_uuid": (new_doc_version_info.get("uuid") if new_doc_version_info else None),
            "invalidate_rag": True,
        }


__all__ = ["apply_media_item_update", "apply_media_metadata_patch"]

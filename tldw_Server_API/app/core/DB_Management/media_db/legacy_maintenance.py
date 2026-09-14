"""Legacy maintenance helpers extracted from the media DB shim."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import yaml
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.db_migration import MigrationError
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError, DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)

_MAINTENANCE_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    BackendDatabaseError,
    ConflictError,
    DatabaseError,
    InputError,
    MigrationError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
    sqlite3.Error,
    yaml.YAMLError,
)


def empty_trash(db_instance: MediaDbLike, days_threshold: int) -> tuple[int, int]:
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    if not isinstance(days_threshold, int) or days_threshold < 0:
        raise ValueError("Days must be non-negative int.")  # noqa: TRY003

    threshold_date_str = (
        datetime.now(timezone.utc) - timedelta(days=days_threshold)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    processed_count = 0
    logger.info(
        f"Emptying trash older than {days_threshold} days ({threshold_date_str}) on DB {db_instance.db_path_str}"
    )
    try:
        cursor_find = db_instance.execute_query(
            "SELECT id, title FROM Media WHERE is_trash = 1 AND deleted = 0 "
            "AND system_operation_id IS NULL AND trash_date <= ?",
            (threshold_date_str,),
        )
        items_to_process = cursor_find.fetchall()
        if not items_to_process:
            logger.info("No items found in trash older than threshold.")
        else:
            logger.info(f"Found {len(items_to_process)} items to process.")
            for item in items_to_process:
                media_id, title = item["id"], item["title"]
                logger.debug(
                    "Processing item ID {} ({}) for sync delete from trash.",
                    media_id,
                    title,
                )
                try:
                    success = db_instance.soft_delete_media(media_id=media_id, cascade=True)
                    if success:
                        processed_count += 1
                    else:
                        logger.warning(
                            f"Failed process item ID {media_id} during trash emptying."
                        )
                except ConflictError as exc:
                    logger.warning(
                        f"Conflict processing item ID {media_id} during trash emptying: {exc}"
                    )
                except DatabaseError:
                    logger.exception(
                        f"DB error processing item ID {media_id} during trash emptying"
                    )
                except _MAINTENANCE_NONCRITICAL_EXCEPTIONS as exc:
                    logger.error(
                        f"Unexpected error processing item ID {media_id} during trash emptying: {exc}",
                        exc_info=True,
                    )

        cursor_remain = db_instance.execute_query(
            "SELECT COUNT(*) AS trash_remaining FROM Media WHERE is_trash = 1 "
            "AND deleted = 0 AND system_operation_id IS NULL"
        )
        remain_row = cursor_remain.fetchone()
        remaining_count = remain_row["trash_remaining"] if remain_row else 0
        logger.info(
            f"Trash emptying complete. Processed (sync deleted): {processed_count}. Remaining in UI trash: {remaining_count}."
        )
    except (DatabaseError, sqlite3.Error) as exc:
        logger.error(
            f"Error emptying trash DB '{db_instance.db_path_str}': {exc}",
            exc_info=True,
        )
        return 0, -1
    except _MAINTENANCE_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            f"Unexpected error emptying trash DB '{db_instance.db_path_str}': {exc}",
            exc_info=True,
        )
        return 0, -1
    else:
        return processed_count, remaining_count


def check_media_and_whisper_model(*args: Any, **kwargs: Any) -> tuple[bool, str]:
    del args, kwargs
    logger.warning("check_media_and_whisper_model is deprecated.")
    return True, "Deprecated"


def permanently_delete_item(db_instance: MediaDbLike, media_id: int) -> bool:
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )

    logger.warning(
        f"!!! PERMANENT DELETE initiated Media ID: {media_id} DB {db_instance.db_path_str}. NOT SYNCED !!!"
    )
    try:
        with db_instance.transaction() as conn:
            sel_cur = db_instance.execute_query(
                "SELECT 1 AS one FROM Media WHERE id = ? "
                "AND system_operation_id IS NULL",
                (media_id,),
                connection=conn,
            )
            row = sel_cur.fetchone()
            if not row:
                logger.warning(f"Permanent delete failed: Media {media_id} not found.")
                return False

            del_cur = db_instance.execute_query(
                "DELETE FROM Media WHERE id = ? AND system_operation_id IS NULL",
                (media_id,),
                commit=False,
                connection=conn,
            )
            deleted_count = getattr(del_cur, "rowcount", 0) or 0

            try:
                db_instance._delete_fts_media(conn, media_id)
            except _MAINTENANCE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"FTS cleanup during permanent delete skipped/failed: {exc}")

        if int(deleted_count) > 0:
            logger.info(f"Permanently deleted Media ID: {media_id}. NO sync log generated.")
            try:
                from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
                    invalidate_intra_doc_vectors,
                )

                invalidate_intra_doc_vectors(str(media_id))
            except _MAINTENANCE_NONCRITICAL_EXCEPTIONS:
                pass
            return True

        logger.error(f"Permanent delete failed unexpectedly Media {media_id}.")
    except sqlite3.Error as exc:
        logger.error(f"Error permanently deleting Media {media_id}: {exc}", exc_info=True)
        raise DatabaseError(f"Failed permanently delete item: {exc}") from exc  # noqa: TRY003
    except _MAINTENANCE_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            f"Unexpected error permanently deleting Media {media_id}: {exc}",
            exc_info=True,
        )
        raise DatabaseError(f"Unexpected permanent delete error: {exc}") from exc  # noqa: TRY003
    else:
        return False


__all__ = [
    "check_media_and_whisper_model",
    "empty_trash",
    "permanently_delete_item",
]

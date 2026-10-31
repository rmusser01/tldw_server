"""Legacy content and keyword query helpers extracted from the media DB shim."""

from __future__ import annotations

import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.keywords_repository import (
    KeywordsRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)


def get_all_content_from_database(db_instance: MediaDbLike) -> list[dict[str, Any]]:
    """Return active media rows ordered by most recent modification first."""
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    try:
        cursor = db_instance.execute_query(
            "SELECT id, uuid, content, title, author, type, url, ingestion_date, last_modified "
            "FROM Media WHERE deleted = 0 AND is_trash = 0 "
            "AND system_operation_id IS NULL ORDER BY last_modified DESC"
        )
        return [dict(item) for item in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(f"Error retrieving all content DB '{db_instance.db_path_str}'")
        raise DatabaseError("Error retrieving all content") from exc  # noqa: TRY003


def fetch_keywords_for_media(media_id: int, db_instance: MediaDbLike) -> list[str]:
    """Return active keywords attached to a single media row."""
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    logger.debug(f"Fetching keywords media_id={media_id} DB: {db_instance.db_path_str}")
    return KeywordsRepository.from_legacy_db(db_instance).fetch_for_media(media_id)


def fetch_keywords_for_media_batch(
    media_ids: list[int],
    db_instance: MediaDbLike,
) -> dict[int, list[str]]:
    """Return active keywords for each requested media row keyed by media ID."""
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    if not media_ids:
        return {}
    try:
        safe_media_ids = [int(media_id) for media_id in media_ids]
    except (ValueError, TypeError) as exc:
        raise InputError(f"media_ids must be list of integers: {exc}") from exc  # noqa: TRY003
    if not safe_media_ids:
        return {}

    keywords_map = {media_id: [] for media_id in safe_media_ids}
    placeholders = ",".join("?" * len(safe_media_ids))
    order_expr = db_instance._keyword_order_expression("k.keyword")  # type: ignore[attr-defined]
    query = (
        f"SELECT mk.media_id, k.keyword FROM MediaKeywords mk "  # nosec B608
        "JOIN Keywords k ON mk.keyword_id = k.id "
        "JOIN Media m ON mk.media_id = m.id "
        f"WHERE mk.media_id IN ({placeholders}) AND k.deleted = ? AND m.deleted = ? "
        "AND m.system_operation_id IS NULL "
        f"ORDER BY mk.media_id, {order_expr}"
    )
    params = (*safe_media_ids, False, False)
    try:
        cursor = db_instance.execute_query(query, params)
        for row in cursor.fetchall():
            media_id = row["media_id"]
            if media_id in keywords_map:
                keywords_map[media_id].append(row["keyword"])
    except (DatabaseError, sqlite3.Error) as exc:
        logger.error(f"Failed fetch keywords batch '{db_instance.db_path_str}': {exc}", exc_info=True)
        raise DatabaseError("Failed fetch keywords batch") from exc  # noqa: TRY003
    else:
        return keywords_map


__all__ = [
    "fetch_keywords_for_media",
    "fetch_keywords_for_media_batch",
    "get_all_content_from_database",
]

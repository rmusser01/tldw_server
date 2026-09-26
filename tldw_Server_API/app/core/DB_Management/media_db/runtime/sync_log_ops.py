"""Package-owned sync-log access and maintenance helpers."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def get_sync_log_entries(
    self,
    since_change_id: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return sync-log rows newer than the given change id."""
    identity = "entity_id AS entity_uuid" if getattr(self, "_sync_entity_column", None) == "entity_id" else "entity_uuid"
    # The projection is one of two fixed application-owned column names.
    query = (
        f"SELECT change_id, entity, {identity}, operation, timestamp, client_id, version, "  # nosec B608
        "org_id, team_id, payload FROM sync_log WHERE change_id > ? ORDER BY change_id ASC"
    )
    params: list[Any] = [since_change_id]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    try:
        conn = self.get_connection()
        rows = self._fetchall_with_connection(conn, query, tuple(params))
        results: list[dict[str, Any]] = []
        for row_dict in rows:
            if row_dict.get("payload"):
                try:
                    row_dict["payload"] = json.loads(row_dict["payload"])
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to decode JSON payload for sync log change_id {}",
                        row_dict.get("change_id"),
                    )
                    row_dict["payload"] = None
            results.append(row_dict)
    except DatabaseError:
        raise
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.exception("Error fetching sync log entries from DB '{}'", self.db_path_str)
        raise DatabaseError("Failed to fetch sync log entries") from exc  # noqa: TRY003
    else:
        return results


def delete_sync_log_entries(self, change_ids: list[int]) -> int:
    """Delete sync-log rows for the supplied change ids."""
    if not change_ids:
        return 0
    if not all(isinstance(cid, int) for cid in change_ids):
        raise ValueError("change_ids must be a list of integers.")  # noqa: TRY003
    placeholders = ",".join("?" * len(change_ids))
    query = f"DELETE FROM sync_log WHERE change_id IN ({placeholders})"  # nosec B608
    try:
        with self.transaction() as conn:
            cursor = self._execute_with_connection(
                conn,
                query,
                tuple(change_ids),
            )
            deleted_count = cursor.rowcount
            logger.info(
                "Deleted {} sync log entries from DB '{}'.",
                deleted_count,
                self.db_path_str,
            )
            return deleted_count
    except DatabaseError:
        raise
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.exception("Unexpected error deleting sync log entries from DB '{}'", self.db_path_str)
        raise DatabaseError(f"Unexpected error deleting sync log entries: {exc}") from exc  # noqa: TRY003


def delete_sync_log_entries_before(self, change_id_threshold: int) -> int:
    """Delete sync-log rows up to and including the supplied threshold."""
    if not isinstance(change_id_threshold, int) or change_id_threshold < 0:
        raise ValueError("change_id_threshold must be a non-negative integer.")  # noqa: TRY003
    query = "DELETE FROM sync_log WHERE change_id <= ?"
    try:
        with self.transaction() as conn:
            cursor = self._execute_with_connection(
                conn,
                query,
                (change_id_threshold,),
            )
            deleted_count = cursor.rowcount
            logger.info(
                "Deleted {} sync log entries before or at ID {} from DB '{}'.",
                deleted_count,
                change_id_threshold,
                self.db_path_str,
            )
            return deleted_count
    except DatabaseError:
        raise
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.exception(
            "Unexpected error deleting sync log entries before {} from DB '{}'",
            change_id_threshold,
            self.db_path_str,
        )
        raise DatabaseError(
            f"Unexpected error deleting sync log entries before threshold: {exc}"
        ) from exc  # noqa: TRY003


__all__ = [
    "get_sync_log_entries",
    "delete_sync_log_entries",
    "delete_sync_log_entries_before",
]

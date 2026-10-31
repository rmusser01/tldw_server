"""Chatbook account-scope count helpers for MediaDatabase."""

from __future__ import annotations

from typing import Any


def count_chatbook_scope_category(self: Any, category: str) -> int:
    """Return the active row count for a Chatbooks media-scope category."""
    queries = {
        "media_records": (
            "SELECT COUNT(*) AS count FROM Media WHERE deleted = 0 AND is_trash = 0 "
            "AND system_operation_id IS NULL"
        ),
        "media_transcripts": (
            "SELECT COUNT(*) AS count FROM Transcripts t WHERE t.deleted = 0 "
            "AND EXISTS (SELECT 1 FROM Media m WHERE m.id = t.media_id "
            "AND m.system_operation_id IS NULL)"
        ),
        "media_chunks": (
            "SELECT COUNT(*) AS count FROM UnvectorizedMediaChunks c WHERE c.deleted = 0 "
            "AND EXISTS (SELECT 1 FROM Media m WHERE m.id = c.media_id "
            "AND m.system_operation_id IS NULL)"
        ),
        "media_stored_artifacts": "SELECT COUNT(*) AS count FROM MediaFiles WHERE deleted = 0",
        "media_pointers": (
            "SELECT COUNT(*) AS count FROM Media "
            "WHERE deleted = 0 AND is_trash = 0 AND system_operation_id IS NULL "
            "AND url IS NOT NULL AND url <> ''"
        ),
    }
    query = queries.get(category)
    if query is None:
        return 0
    # Category selects a fixed allowlisted query; no user input enters SQL.
    cursor = self.execute_query(query)  # nosec B608
    row = cursor.fetchone()
    if not row:
        return 0
    try:
        value = row["count"]
    except (TypeError, KeyError, IndexError):
        value = row[0]
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def list_chatbook_scope_ids(self: Any, category: str) -> list[str]:
    """Return active row IDs for a Chatbooks media-scope category."""
    queries = {
        "media_records": (
            "SELECT id FROM Media WHERE deleted = 0 AND is_trash = 0 "
            "AND system_operation_id IS NULL ORDER BY id ASC"
        ),
    }
    query = queries.get(category)
    if query is None:
        return []
    # Category selects a fixed allowlisted query; no user input enters SQL.
    cursor = self.execute_query(query)  # nosec B608
    ids: list[str] = []
    for row in cursor.fetchall() or []:
        try:
            value = row["id"]
        except (TypeError, KeyError, IndexError):
            value = row[0]
        if value is not None:
            ids.append(str(value))
    return ids


__all__ = ["count_chatbook_scope_category", "list_chatbook_scope_ids"]

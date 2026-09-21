"""Shared sync/version utility helpers for the package-native Media DB runtime."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _generate_uuid(self: Any) -> str:
    """Return a fresh UUID4 string."""
    return str(uuid.uuid4())


def _get_current_utc_timestamp_str(self: Any) -> str:
    """Return the canonical UTC timestamp string used by Media DB sync rows."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _get_next_version(
    self: Any,
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    id_val: Any,
) -> tuple[int, int] | None:
    """Return the current and next version pair for an active record."""
    try:
        if not (_SAFE_IDENTIFIER_RE.fullmatch(table or "") and _SAFE_IDENTIFIER_RE.fullmatch(id_col or "")):
            raise DatabaseError(  # noqa: TRY003
                f"Unsafe identifier in version lookup: table={table!r}, column={id_col!r}"
            )
        query = f"SELECT version FROM {table} WHERE {id_col} = ? AND deleted = 0"  # nosec B608
        cursor = conn.execute(query, (id_val,))
        result = cursor.fetchone()
        if result:
            current_version = result["version"]
            if isinstance(current_version, int):
                return current_version, current_version + 1
            logging.error(
                "Invalid non-integer version %r found for %s %s=%r",
                current_version,
                table,
                id_col,
                id_val,
            )
            return None
    except sqlite3.Error as exc:
        logging.exception(
            "Database error fetching version for %s %s=%r",
            table,
            id_col,
            id_val,
        )
        raise DatabaseError(f"Failed to fetch current version: {exc}") from exc  # noqa: TRY003
    return None


def _log_sync_event(
    self: Any,
    conn: sqlite3.Connection,
    entity: str,
    entity_uuid: str,
    operation: str,
    version: int,
    payload: dict | None = None,
):
    """Insert a sync-log row for a completed mutation."""
    if not entity or not entity_uuid or not operation:
        logging.error("Sync log attempt with missing entity, uuid, or operation.")
        return

    current_time = self._get_current_utc_timestamp_str()
    client_id = self.client_id
    scope_org_id, scope_team_id = self._resolve_scope_ids()

    if payload:
        payload = payload.copy()
        payload.pop("vector_embedding", None)
        for key, value in list(payload.items()):
            try:
                if isinstance(value, datetime):
                    payload[key] = value.isoformat()
            except _MEDIA_NONCRITICAL_EXCEPTIONS:
                pass

    payload_json = json.dumps(payload, separators=(",", ":")) if payload else None

    try:
        if self.backend_type == BackendType.SQLITE:
            conn.execute(
                """
                INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, org_id, team_id, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (entity, entity_uuid, operation, current_time, client_id, version, scope_org_id, scope_team_id, payload_json),
            )
        else:
            identity = "entity_id" if getattr(self, "_sync_entity_column", None) == "entity_id" else "entity_uuid"
            self._execute_with_connection(
                conn,
                f"""
                INSERT INTO sync_log (entity, {identity}, operation, timestamp, client_id, version, org_id, team_id, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,  # nosec B608 -- identity is selected from two fixed names; all values remain bound.
                (entity, entity_uuid, operation, current_time, client_id, version, scope_org_id, scope_team_id, payload_json),
            )
        logging.debug(
            "Logged sync event: %s %s %s v%s at %s",
            entity,
            entity_uuid,
            operation,
            version,
            current_time,
        )
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logging.error(
            "Failed to insert sync log event for %s %s: %s",
            entity,
            entity_uuid,
            exc,
            exc_info=True,
        )
        raise DatabaseError(f"Failed to log sync event: {exc}") from exc  # noqa: TRY003


__all__ = [
    "_generate_uuid",
    "_get_current_utc_timestamp_str",
    "_get_next_version",
    "_log_sync_event",
]

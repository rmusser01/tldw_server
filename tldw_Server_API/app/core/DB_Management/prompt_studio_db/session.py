"""Session-level helpers shared by both Prompt Studio database classes.

Repositories call these on the session (the legacy database object), so they live on a
base class both implementations inherit rather than being written twice.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import DB_ERRORS
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError

# The PostgreSQL cursor wrapper re-raises backend errors as the Prompts DatabaseError.
_SESSION_ERRORS = (*DB_ERRORS, DatabaseError)


class PromptStudioSessionOps:
    """Needs ``transaction()``, ``_cursor_exec``, ``_execute``, ``backend_type`` and ``client_id``."""

    _sync_log_available: Optional[bool] = None

    def _sync_log_exists(self, conn: Any) -> bool:
        if self.backend_type == BackendType.POSTGRESQL:
            return bool(self.backend.table_exists("sync_log", connection=conn.raw_connection))
        row = self._cursor_exec(conn, "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sync_log'").fetchone()
        return row is not None

    def _log_sync_event(self, entity: str, entity_uuid: str, operation: str, payload: dict[str, Any]) -> None:
        """Best effort: the sync log is optional, and a failure here must not fail the write."""
        if not entity or not entity_uuid or not operation or self._sync_log_available is False:
            return
        try:
            with self.transaction() as conn:
                if self._sync_log_available is None:
                    self._sync_log_available = self._sync_log_exists(conn)
                if not self._sync_log_available:
                    return
                self._cursor_exec(
                    conn,
                    "INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload, timestamp)"
                    # An explicit RETURNING stops the PostgreSQL preparer appending
                    # "RETURNING id", a column sync_log does not have.
                    " VALUES (?, ?, ?, ?, 1, ?, CURRENT_TIMESTAMP) RETURNING change_id",
                    (
                        entity,
                        entity_uuid,
                        operation,
                        # Sync-log ownership is the tenant (PostgreSQL RLS keys on it); the
                        # entity rows keep the originating audit client.
                        getattr(self, "tenant_user_id", None) or self.client_id,
                        json.dumps(payload or {}, separators=(",", ":"), default=str),
                    ),
                )
        except Exception as exc:  # noqa: BLE001 - logged, never raised
            logger.warning("Failed to log Prompt Studio sync event for {}/{}: {}", entity, entity_uuid, type(exc).__name__)

    def _idem_lookup(self, entity_type: str, key: str, user_id: Optional[str]) -> Optional[int]:
        user_clause = "user_id IS NULL" if user_id is None else "user_id = ?"
        params = (entity_type, key) if user_id is None else (entity_type, key, user_id)
        try:
            row = self._execute(
                "SELECT entity_id FROM prompt_studio_idempotency"  # nosec B608 - fixed fragments
                f" WHERE entity_type = ? AND idempotency_key = ? AND {user_clause} LIMIT 1",
                params,
            ).fetchone()
        except _SESSION_ERRORS as exc:
            logger.warning("Prompt Studio idempotency lookup failed: {}", type(exc).__name__)
            return None
        return int(row[0]) if row else None

    def _idem_record(self, entity_type: str, key: str, entity_id: int, user_id: Optional[str]) -> None:
        try:
            with self.transaction() as conn:
                # Translated to ON CONFLICT DO NOTHING on PostgreSQL.
                self._cursor_exec(
                    conn,
                    "INSERT OR IGNORE INTO prompt_studio_idempotency (entity_type, idempotency_key, entity_id, user_id)"
                    " VALUES (?, ?, ?, ?)",
                    (entity_type, key, entity_id, user_id),
                )
        except _SESSION_ERRORS as exc:
            logger.warning("Prompt Studio idempotency record failed: {}", type(exc).__name__)

"""SQLite-backed ACP audit event persistence.

Provides durable audit logging for ACP sessions with configurable retention.
Uses a simple single-table schema optimized for append-heavy workloads.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS acp_audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id INTEGER NOT NULL,
    payload_json TEXT DEFAULT '{}',
    created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_acp_audit_session ON acp_audit_events(session_id);
CREATE INDEX IF NOT EXISTS idx_acp_audit_user ON acp_audit_events(user_id);
CREATE INDEX IF NOT EXISTS idx_acp_audit_ts ON acp_audit_events(created_at);
"""


class ACPAuditDB:
    """SQLite-backed audit event store with in-memory hot cache."""

    def __init__(
        self,
        db_path: str | None = None,
        hot_cache_size: int = 5000,
        retention_days: int = 30,
    ) -> None:
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "Databases", "acp_audit.db",
            )
        self._db_path = os.path.abspath(db_path)
        self._retention_days = retention_days
        self._hot_cache: deque[dict[str, Any]] = deque(maxlen=hot_cache_size)
        self._write_buffer: list[dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._conn_local = threading.local()
        self._initialized = False
        self._init_lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        conn = getattr(self._conn_local, "conn", None)
        if conn is None:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=10)
            configure_sqlite_connection(conn)
            self._conn_local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if needed (idempotent)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn = self._get_conn()
            conn.executescript(_SCHEMA_SQL)
            conn.commit()
            self._initialized = True

    def record_event(
        self,
        *,
        action: str,
        user_id: int,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an audit event. Adds to hot cache and write buffer."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": str(action),
            "user_id": int(user_id),
            "session_id": str(session_id),
            "metadata": dict(metadata or {}),
        }
        self._hot_cache.append(event)
        with self._buffer_lock:
            self._write_buffer.append(event)
        return event

    def flush_if_needed(self, threshold: int = 10) -> int:
        """Flush if buffer has reached threshold. Returns count written or 0."""
        with self._buffer_lock:
            if len(self._write_buffer) < threshold:
                return 0
        return self.flush()

    def flush(self) -> int:
        """Flush buffered events to SQLite. Returns count written."""
        with self._buffer_lock:
            if not self._write_buffer:
                return 0
            batch = list(self._write_buffer)
            self._write_buffer.clear()

        self._ensure_schema()
        conn = self._get_conn()
        try:
            conn.executemany(
                "INSERT INTO acp_audit_events (timestamp, event_type, session_id, user_id, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        e["timestamp"],
                        e["action"],
                        e["session_id"],
                        e["user_id"],
                        json.dumps(e.get("metadata", {})),
                    )
                    for e in batch
                ],
            )
            conn.commit()
            return len(batch)
        except sqlite3.Error as exc:
            logger.error("ACP audit flush failed: {}", exc)
            # Re-buffer failed events
            with self._buffer_lock:
                self._write_buffer = batch + self._write_buffer
            return 0

    def query_events(
        self,
        *,
        session_id: str | None = None,
        user_id: int | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query persisted events from SQLite."""
        self._ensure_schema()
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            conditions.append("event_type = ?")
            params.append(action)

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT timestamp, event_type, session_id, user_id, payload_json FROM acp_audit_events WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(sql, params)
        results = []
        for row in cursor.fetchall():
            try:
                payload = json.loads(row[4]) if row[4] else {}
            except (json.JSONDecodeError, TypeError):
                payload = {}
            results.append({
                "timestamp": row[0],
                "action": row[1],
                "session_id": row[2],
                "user_id": row[3],
                "metadata": payload,
            })
        return results

    def get_hot_cache(self, *, session_id: str | None = None) -> list[dict[str, Any]]:
        """Get events from in-memory hot cache."""
        if session_id:
            return [
                dict(e) for e in self._hot_cache
                if str(e.get("session_id")) == str(session_id)
            ]
        return [dict(e) for e in self._hot_cache]

    def purge_old_events(self, *, now: float | None = None) -> int:
        """Remove events older than retention_days. Returns count deleted."""
        self._ensure_schema()
        conn = self._get_conn()
        cutoff = (time.time() if now is None else float(now)) - (
            self._retention_days * 86400
        )
        cursor = conn.execute(
            "DELETE FROM acp_audit_events WHERE created_at < ?",
            (cutoff,),
        )
        conn.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info("Purged {} old ACP audit events (retention={}d)", deleted, self._retention_days)
        return deleted

    def close(self) -> None:
        """Close the thread-local connection."""
        conn = getattr(self._conn_local, "conn", None)
        if conn:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._conn_local.conn = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_audit_db: ACPAuditDB | None = None
_audit_db_lock = threading.Lock()


def _default_audit_retention_days() -> int:
    """Resolve ACP audit retention from config with a safe default."""
    raw = os.getenv("ACP_AUDIT_RETENTION_DAYS")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return 30
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.config import load_acp_sandbox_config

        return int(load_acp_sandbox_config().audit_retention_days)
    except Exception:
        return 30


def get_acp_audit_db(
    db_path: str | None = None,
    retention_days: int | None = None,
) -> ACPAuditDB:
    """Get or create the module-level ACPAuditDB singleton."""
    global _audit_db
    with _audit_db_lock:
        if _audit_db is None:
            _audit_db = ACPAuditDB(
                db_path=db_path,
                retention_days=_default_audit_retention_days()
                if retention_days is None
                else retention_days,
            )
        elif retention_days is not None:
            _audit_db._retention_days = int(retention_days)
    return _audit_db

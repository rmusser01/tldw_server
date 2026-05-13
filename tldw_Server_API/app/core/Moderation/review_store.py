"""SQLite persistence for sanitized moderation review items."""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection
from tldw_Server_API.app.core.Utils.Utils import get_project_root


_ACTION_STATUS = {
    "approve": "approved",
    "block": "blocked",
    "redact": "redacted",
    "dismiss": "dismissed",
    "escalate": "escalated",
}


def _utc_now() -> str:
    return _format_utc(datetime.now(timezone.utc))


def _utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _format_utc(value: datetime) -> str:
    normalized = value.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _hash_undo_token(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True, separators=(",", ":"))


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _cursor_offset(cursor: str | None) -> int:
    try:
        return max(0, int(cursor or 0))
    except (TypeError, ValueError):
        return 0


def default_review_db_path() -> Path:
    configured = os.getenv("MODERATION_REVIEW_DB_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path(get_project_root()) / "tldw_Server_API" / "Databases" / "moderation_review.db"


class ModerationReviewStore:
    """Small SQLite repository for sanitized moderation review data."""

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        *,
        undo_ttl_seconds: int = 15 * 60,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_review_db_path()
        self.undo_ttl_seconds = int(undo_ttl_seconds)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        configure_sqlite_connection(conn)
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS moderation_review_items (
                    id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL DEFAULT 'needs_review',
                    phase TEXT NOT NULL CHECK (phase IN ('input', 'output')),
                    source_type TEXT,
                    source_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    severity TEXT,
                    category TEXT,
                    safe_fields_json TEXT NOT NULL DEFAULT '{}',
                    excerpt TEXT NOT NULL,
                    context_json TEXT,
                    effective_policy_json TEXT NOT NULL DEFAULT '{}',
                    matches_json TEXT NOT NULL DEFAULT '[]',
                    recommended_action TEXT,
                    retention_expires_at TEXT,
                    content_redacted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_moderation_review_items_status_created
                    ON moderation_review_items(status, created_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_moderation_review_items_category
                    ON moderation_review_items(category);
                CREATE INDEX IF NOT EXISTS idx_moderation_review_items_source
                    ON moderation_review_items(source_type, source_id);
                CREATE INDEX IF NOT EXISTS idx_moderation_review_items_user
                    ON moderation_review_items(user_id);

                CREATE TABLE IF NOT EXISTS moderation_review_decisions (
                    id TEXT PRIMARY KEY,
                    item_id TEXT NOT NULL REFERENCES moderation_review_items(id) ON DELETE CASCADE,
                    action TEXT NOT NULL,
                    status TEXT NOT NULL,
                    previous_status TEXT NOT NULL,
                    reason TEXT,
                    decided_by TEXT NOT NULL,
                    decided_at TEXT NOT NULL,
                    undo_token TEXT UNIQUE,
                    undo_token_hash TEXT,
                    undo_expires_at TEXT,
                    undone_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_moderation_review_decisions_item
                    ON moderation_review_decisions(item_id, decided_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_moderation_review_decisions_undo_hash
                    ON moderation_review_decisions(undo_token_hash)
                    WHERE undo_token_hash IS NOT NULL;

                CREATE TABLE IF NOT EXISTS moderation_review_audit_events (
                    id TEXT PRIMARY KEY,
                    item_id TEXT REFERENCES moderation_review_items(id) ON DELETE SET NULL,
                    decision_id TEXT REFERENCES moderation_review_decisions(id) ON DELETE SET NULL,
                    actor_id TEXT,
                    action TEXT NOT NULL,
                    summary TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_moderation_review_audit_created
                    ON moderation_review_audit_events(created_at DESC, id DESC);
                CREATE INDEX IF NOT EXISTS idx_moderation_review_audit_item
                    ON moderation_review_audit_events(item_id, created_at DESC);
                """
            )
            self._ensure_column(conn, "moderation_review_decisions", "undo_token_hash", "TEXT")
            self._ensure_column(conn, "moderation_review_decisions", "undo_expires_at", "TEXT")
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_moderation_review_decisions_undo_hash
                    ON moderation_review_decisions(undo_token_hash)
                    WHERE undo_token_hash IS NOT NULL
                """
            )
            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, sql_type: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()  # nosec B608
        if column not in {row["name"] for row in rows}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")  # nosec B608

    def _row_to_item(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "status": row["status"],
            "phase": row["phase"],
            "source_type": row["source_type"],
            "source_id": row["source_id"],
            "user_id": row["user_id"],
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "severity": row["severity"],
            "category": row["category"],
            "safe_fields": _json_loads(row["safe_fields_json"], {}),
            "excerpt": row["excerpt"],
            "context": _json_loads(row["context_json"], {}),
            "effective_policy": _json_loads(row["effective_policy_json"], {}),
            "matches": _json_loads(row["matches_json"], []),
            "recommended_action": row["recommended_action"],
            "retention_expires_at": row["retention_expires_at"],
            "content_redacted_at": row["content_redacted_at"],
            "decision_history": [],
        }

    def _row_to_decision(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "item_id": row["item_id"],
            "action": row["action"],
            "status": row["status"],
            "previous_status": row["previous_status"],
            "reason": row["reason"],
            "decided_by": row["decided_by"],
            "decided_at": row["decided_at"],
            "undo_expires_at": row["undo_expires_at"],
            "undone_at": row["undone_at"],
            "undo_token": None,  # nosec B105
        }

    def _row_to_decision_history(self, row: sqlite3.Row, *, content_redacted_at: str | None) -> dict[str, Any]:
        expires_at = _parse_utc(row["undo_expires_at"])
        undone_at = row["undone_at"]
        undo_eligible = bool(
            row["undo_token_hash"]
            and not undone_at
            and expires_at is not None
            and expires_at >= _utc_now_dt()
        )
        return {
            "id": row["id"],
            "action": row["action"],
            "status": row["status"],
            "previous_status": row["previous_status"],
            "actor_id": row["decided_by"],
            "reason": row["reason"],
            "decided_at": row["decided_at"],
            "undo_eligible": undo_eligible,
            "undo_expires_at": row["undo_expires_at"],
            "undone_at": undone_at,
            "redaction_state": "redacted" if content_redacted_at else "not_redacted",
        }

    def _list_decision_history(self, conn: sqlite3.Connection, item_id: str, *, content_redacted_at: str | None) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT * FROM moderation_review_decisions
            WHERE item_id = ?
            ORDER BY decided_at DESC, id DESC
            """,
            (item_id,),
        ).fetchall()
        return [
            self._row_to_decision_history(row, content_redacted_at=content_redacted_at)
            for row in rows
        ]

    def _row_to_audit(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "item_id": row["item_id"],
            "decision_id": row["decision_id"],
            "actor_id": row["actor_id"],
            "action": row["action"],
            "summary": row["summary"],
            "created_at": row["created_at"],
            "metadata": _json_loads(row["metadata_json"], {}),
        }

    def _append_audit(
        self,
        conn: sqlite3.Connection,
        *,
        item_id: str | None,
        decision_id: str | None,
        actor_id: str | None,
        action: str,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        audit_id = str(uuid.uuid4())
        created_at = _utc_now()
        conn.execute(
            """
            INSERT INTO moderation_review_audit_events (
                id, item_id, decision_id, actor_id, action, summary, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (audit_id, item_id, decision_id, actor_id, action, summary, created_at, _json_dumps(metadata or {})),
        )
        row = conn.execute(
            "SELECT * FROM moderation_review_audit_events WHERE id = ?",
            (audit_id,),
        ).fetchone()
        return self._row_to_audit(row)

    def upsert_item(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Insert a sanitized review item or return the existing idempotent row."""
        now = _utc_now()
        item_id = str(payload.get("id") or uuid.uuid4())
        idempotency_key = str(payload["idempotency_key"])
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO moderation_review_items (
                    id, idempotency_key, status, phase, source_type, source_id, user_id,
                    session_id, created_at, updated_at, severity, category, safe_fields_json,
                    excerpt, context_json, effective_policy_json, matches_json, recommended_action,
                    retention_expires_at, content_redacted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    idempotency_key,
                    str(payload.get("status") or "needs_review"),
                    str(payload["phase"]),
                    payload.get("source_type"),
                    payload.get("source_id"),
                    payload.get("user_id"),
                    payload.get("session_id"),
                    str(payload.get("created_at") or now),
                    payload.get("updated_at"),
                    payload.get("severity"),
                    payload.get("category"),
                    _json_dumps(payload.get("safe_fields") or {}),
                    str(payload.get("excerpt") or "[content unavailable]"),
                    _json_dumps(payload.get("context") or {}),
                    _json_dumps(payload.get("effective_policy") or {}),
                    _json_dumps(payload.get("matches") or []),
                    payload.get("recommended_action"),
                    payload.get("retention_expires_at"),
                    payload.get("content_redacted_at"),
                ),
            )
            row = conn.execute(
                "SELECT * FROM moderation_review_items WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
            item = self._row_to_item(row)
            if cur.rowcount:
                self._append_audit(
                    conn,
                    item_id=item["id"],
                    decision_id=None,
                    actor_id=None,
                    action="item.created",
                    summary="Moderation review item created",
                    metadata={"source_type": item.get("source_type"), "phase": item.get("phase")},
                )
            conn.commit()
            return item

    def get_item(self, item_id: str, *, include_history: bool = False) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM moderation_review_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            item = self._row_to_item(row)
            if item is not None and include_history:
                item["decision_history"] = self._list_decision_history(
                    conn,
                    item_id,
                    content_redacted_at=item.get("content_redacted_at"),
                )
        return item

    def list_items(
        self,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        filters = filters or {}
        safe_limit = max(1, min(int(limit or 50), 200))
        offset = _cursor_offset(cursor)
        clauses: list[str] = []
        params: list[Any] = []
        for key in ("status", "category", "severity", "source_type", "source_id", "user_id"):
            value = filters.get(key)
            if value is None or value == "":
                continue
            clauses.append(f"{key} = ?")
            params.append(str(value))
        if filters.get("q"):
            clauses.append("(excerpt LIKE ? OR category LIKE ? OR source_id LIKE ?)")
            needle = f"%{str(filters['q'])}%"
            params.extend([needle, needle, needle])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM moderation_review_items {where}",  # nosec B608
                tuple(params),
            ).fetchone()
            rows = conn.execute(
                f"""
                SELECT * FROM moderation_review_items
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,  # nosec B608
                tuple(params + [safe_limit + 1, offset]),
            ).fetchall()
        items = [self._row_to_item(row) for row in rows[:safe_limit]]
        next_cursor = str(offset + safe_limit) if len(rows) > safe_limit else None
        return {"items": items, "next_cursor": next_cursor, "total": int(total_row["total"] if total_row else 0)}

    def record_decision(
        self,
        item_id: str,
        *,
        action: str,
        decided_by: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        status = _ACTION_STATUS.get(action)
        if status is None:
            raise ValueError(f"unsupported decision action: {action}")
        decision_id = str(uuid.uuid4())
        undo_token = secrets.token_urlsafe(32)
        undo_token_hash = _hash_undo_token(undo_token)
        decided_at_dt = _utc_now_dt()
        decided_at = _format_utc(decided_at_dt)
        undo_expires_at = _format_utc(decided_at_dt + timedelta(seconds=self.undo_ttl_seconds))
        with self._connect() as conn:
            item_row = conn.execute(
                "SELECT * FROM moderation_review_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            if item_row is None:
                raise KeyError(item_id)
            previous_status = str(item_row["status"])
            conn.execute(
                """
                INSERT INTO moderation_review_decisions (
                    id, item_id, action, status, previous_status, reason, decided_by, decided_at,
                    undo_token, undo_token_hash, undo_expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    item_id,
                    action,
                    status,
                    previous_status,
                    reason,
                    decided_by,
                    decided_at,
                    None,
                    undo_token_hash,
                    undo_expires_at,
                ),
            )
            conn.execute(
                "UPDATE moderation_review_items SET status = ?, updated_at = ? WHERE id = ?",
                (status, decided_at, item_id),
            )
            self._append_audit(
                conn,
                item_id=item_id,
                decision_id=decision_id,
                actor_id=decided_by,
                action=f"decision.{action}",
                summary=reason,
                metadata={"previous_status": previous_status, "status": status},
            )
            if action == "redact":
                self._redact_item_content_in_conn(
                    conn,
                    item_id=item_id,
                    actor_id=decided_by,
                    redacted_at=decided_at,
                )
            row = conn.execute(
                "SELECT * FROM moderation_review_decisions WHERE id = ?",
                (decision_id,),
            ).fetchone()
            conn.commit()
        decision = self._row_to_decision(row)
        decision["undo_token"] = undo_token
        return decision

    def undo_decision(self, item_id: str, *, undo_token: str, actor_id: str) -> dict[str, Any]:
        now_dt = _utc_now_dt()
        now = _format_utc(now_dt)
        undo_token_hash = _hash_undo_token(undo_token)
        with self._connect() as conn:
            decision_row = conn.execute(
                """
                SELECT rowid AS decision_rowid, * FROM moderation_review_decisions
                WHERE item_id = ? AND undo_token_hash = ? AND undone_at IS NULL
                ORDER BY decided_at DESC
                LIMIT 1
                """,
                (item_id, undo_token_hash),
            ).fetchone()
            if decision_row is None:
                raise KeyError("undo_token")
            expires_at = _parse_utc(decision_row["undo_expires_at"])
            if expires_at is None or expires_at < now_dt:
                raise ValueError("undo token expired")
            current_item = conn.execute(
                "SELECT * FROM moderation_review_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            if current_item is None:
                raise KeyError(item_id)
            later_count = conn.execute(
                """
                SELECT COUNT(*) AS count FROM moderation_review_decisions
                WHERE item_id = ? AND rowid > ?
                """,
                (item_id, decision_row["decision_rowid"]),
            ).fetchone()
            if int(later_count["count"] if later_count else 0) > 0 or str(current_item["status"]) != str(decision_row["status"]):
                raise ValueError("undo token superseded")
            previous_status = str(decision_row["previous_status"])
            conn.execute(
                "UPDATE moderation_review_decisions SET undone_at = ? WHERE id = ?",
                (now, decision_row["id"]),
            )
            conn.execute(
                "UPDATE moderation_review_items SET status = ?, updated_at = ? WHERE id = ?",
                (previous_status, now, item_id),
            )
            self._append_audit(
                conn,
                item_id=item_id,
                decision_id=decision_row["id"],
                actor_id=actor_id,
                action="decision.undo",
                summary="Decision undone",
                metadata={"restored_status": previous_status},
            )
            row = conn.execute(
                "SELECT * FROM moderation_review_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            conn.commit()
        return self._row_to_item(row)

    def _redact_item_content_in_conn(
        self,
        conn: sqlite3.Connection,
        *,
        item_id: str,
        actor_id: str,
        redacted_at: str,
    ) -> None:
        item_row = conn.execute(
            "SELECT * FROM moderation_review_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        if item_row is None:
            raise KeyError(item_id)
        item = self._row_to_item(item_row)
        safe_fields = dict(item.get("safe_fields") or {})
        safe_fields.update({"excerpt": False, "context": False, "matches": False})
        redacted_matches = []
        for match in item.get("matches") or []:
            if isinstance(match, dict):
                redacted_matches.append({**match, "sample": "[content redacted]"})
        conn.execute(
            """
            UPDATE moderation_review_items
            SET excerpt = ?, context_json = ?, matches_json = ?, safe_fields_json = ?,
                content_redacted_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                "[content redacted]",
                _json_dumps({
                    "redacted": True,
                    "message": "Context removed from moderation review item",
                }),
                _json_dumps(redacted_matches),
                _json_dumps(safe_fields),
                redacted_at,
                redacted_at,
                item_id,
            ),
        )
        self._append_audit(
            conn,
            item_id=item_id,
            decision_id=None,
            actor_id=actor_id,
            action="content.redacted",
            summary="Review content redacted",
            metadata={"redacted_at": redacted_at},
        )

    def redact_item_content(self, item_id: str, actor_id: str) -> dict[str, Any]:
        now = _utc_now()
        with self._connect() as conn:
            self._redact_item_content_in_conn(conn, item_id=item_id, actor_id=actor_id, redacted_at=now)
            row = conn.execute(
                "SELECT * FROM moderation_review_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            conn.commit()
        return self._row_to_item(row)

    def list_audit(
        self,
        *,
        item_id: str | None = None,
        decision_id: str | None = None,
        actor_id: str | None = None,
        action: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit or 50), 200))
        offset = _cursor_offset(cursor)
        clauses: list[str] = []
        params: list[Any] = []
        if item_id:
            clauses.append("item_id = ?")
            params.append(item_id)
        if decision_id:
            clauses.append("decision_id = ?")
            params.append(decision_id)
        if actor_id:
            clauses.append("actor_id = ?")
            params.append(actor_id)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if date_from:
            clauses.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            clauses.append("created_at <= ?")
            params.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM moderation_review_audit_events
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,  # nosec B608
                tuple(params + [safe_limit + 1, offset]),
            ).fetchall()
        events = [self._row_to_audit(row) for row in rows[:safe_limit]]
        next_cursor = str(offset + safe_limit) if len(rows) > safe_limit else None
        return {"events": events, "next_cursor": next_cursor}

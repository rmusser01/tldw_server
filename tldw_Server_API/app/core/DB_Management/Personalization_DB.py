from __future__ import annotations

"""
Personalization_DB

SQLite wrapper for per-user personalization data:
 - profiles
 - usage_events
 - semantic_memories
 - episodic_memories
 - topic_profiles

This module encapsulates raw SQL per project guidelines.
"""

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    resolve_trusted_database_path,
)
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UsageEvent:
    user_id: str
    type: str
    resource_id: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    timestamp: str | None = None


@dataclass
class SemanticMemory:
    user_id: str
    content: str
    tags: list[str] | None = None
    pinned: bool = False


class PersonalizationDB:
    @classmethod
    def for_path(cls, db_path: str | Path) -> "PersonalizationDB":
        return cls(Path(db_path))

    @classmethod
    def for_user(cls, user_id: str | int) -> "PersonalizationDB":
        return cls.for_path(DatabasePaths.get_personalization_db_path(user_id))

    def __init__(self, db_path: str | Path) -> None:
        try:
            resolved_path = resolve_trusted_database_path(
                db_path,
                label="personalization database",
            )
        except InvalidStoragePathError as exc:
            raise ValueError("PersonalizationDB path must stay within trusted database roots") from exc
        if not resolved_path.parent.exists():
            raise ValueError("PersonalizationDB parent directory must already exist")
        self.db_path = str(resolved_path)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            configure_sqlite_connection(conn)
        except Exception as pragma_error:
            _ = pragma_error  # proceed with defaults if pragmas fail
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS profiles (
                        user_id TEXT PRIMARY KEY,
                        enabled INTEGER NOT NULL DEFAULT 0,
                        alpha REAL NOT NULL DEFAULT 0.2,
                        beta REAL NOT NULL DEFAULT 0.6,
                        gamma REAL NOT NULL DEFAULT 0.2,
                        recency_half_life_days INTEGER NOT NULL DEFAULT 14,
                        proactive_enabled INTEGER NOT NULL DEFAULT 1,
                        proactive_frequency TEXT NOT NULL DEFAULT 'normal',
                        proactive_types TEXT,
                        quiet_hours_start TEXT,
                        quiet_hours_end TEXT,
                        response_style TEXT NOT NULL DEFAULT 'balanced',
                        preferred_format TEXT NOT NULL DEFAULT 'auto',
                        session_continuity_enabled INTEGER NOT NULL DEFAULT 1,
                        session_summaries_enabled INTEGER NOT NULL DEFAULT 1,
                        companion_reflections_enabled INTEGER NOT NULL DEFAULT 1,
                        companion_daily_reflections_enabled INTEGER NOT NULL DEFAULT 1,
                        companion_weekly_reflections_enabled INTEGER NOT NULL DEFAULT 1,
                        purged_at TEXT,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS usage_events (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        type TEXT NOT NULL,
                        resource_id TEXT,
                        tags TEXT,
                        metadata TEXT,
                        FOREIGN KEY (user_id) REFERENCES profiles(user_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_usage_user_ts ON usage_events(user_id, timestamp DESC);

                    CREATE TABLE IF NOT EXISTS semantic_memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        tags TEXT,
                        pinned INTEGER NOT NULL DEFAULT 0,
                        hidden INTEGER NOT NULL DEFAULT 0,
                        last_validated TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES profiles(user_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_semantic_user ON semantic_memories(user_id);

                    CREATE TABLE IF NOT EXISTS episodic_memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        event_id TEXT,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES profiles(user_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_episodic_user_ts ON episodic_memories(user_id, timestamp DESC);

                    CREATE TABLE IF NOT EXISTS topic_profiles (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        label TEXT NOT NULL,
                        score REAL NOT NULL DEFAULT 0,
                        centroid_embedding BLOB,
                        last_seen TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES profiles(user_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_topics_user ON topic_profiles(user_id);
                    CREATE INDEX IF NOT EXISTS idx_topics_score ON topic_profiles(user_id, score DESC);

                    CREATE TABLE IF NOT EXISTS companion_activity_events (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        source_id TEXT NOT NULL,
                        surface TEXT NOT NULL,
                        dedupe_key TEXT NOT NULL,
                        tags TEXT,
                        provenance_json TEXT NOT NULL,
                        metadata_json TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES profiles(user_id),
                        UNIQUE(user_id, dedupe_key)
                    );
                    CREATE INDEX IF NOT EXISTS idx_companion_activity_user_created
                        ON companion_activity_events(user_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS companion_knowledge_cards (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        card_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        score REAL NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'active',
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES profiles(user_id),
                        UNIQUE(user_id, card_type, title)
                    );
                    CREATE INDEX IF NOT EXISTS idx_companion_knowledge_user_score
                        ON companion_knowledge_cards(user_id, score DESC, updated_at DESC);

                    CREATE TABLE IF NOT EXISTS companion_goals (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        goal_type TEXT NOT NULL,
                        config_json TEXT NOT NULL,
                        progress_json TEXT NOT NULL,
                        origin_kind TEXT NOT NULL DEFAULT 'manual',
                        progress_mode TEXT NOT NULL DEFAULT 'manual',
                        derivation_key TEXT,
                        evidence_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (user_id) REFERENCES profiles(user_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_companion_goals_user_updated
                        ON companion_goals(user_id, updated_at DESC);
                    """
                )
                conn.commit()
            finally:
                conn.close()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add columns that may be missing in databases created before schema updates."""
        migrations: list[tuple[str, str, str]] = [
            # (table, column, column_def)
            ("profiles", "proactive_enabled", "INTEGER NOT NULL DEFAULT 1"),
            ("profiles", "proactive_frequency", "TEXT NOT NULL DEFAULT 'normal'"),
            ("profiles", "proactive_types", "TEXT"),
            ("profiles", "quiet_hours_start", "TEXT"),
            ("profiles", "quiet_hours_end", "TEXT"),
            ("profiles", "response_style", "TEXT NOT NULL DEFAULT 'balanced'"),
            ("profiles", "preferred_format", "TEXT NOT NULL DEFAULT 'auto'"),
            ("profiles", "session_continuity_enabled", "INTEGER NOT NULL DEFAULT 1"),
            ("profiles", "session_summaries_enabled", "INTEGER NOT NULL DEFAULT 1"),
            ("profiles", "companion_reflections_enabled", "INTEGER NOT NULL DEFAULT 1"),
            ("profiles", "companion_daily_reflections_enabled", "INTEGER NOT NULL DEFAULT 1"),
            ("profiles", "companion_weekly_reflections_enabled", "INTEGER NOT NULL DEFAULT 1"),
            ("profiles", "purged_at", "TEXT"),
            ("semantic_memories", "hidden", "INTEGER NOT NULL DEFAULT 0"),
            ("semantic_memories", "last_validated", "TEXT"),
            ("topic_profiles", "centroid_embedding", "BLOB"),
            ("companion_goals", "origin_kind", "TEXT NOT NULL DEFAULT 'manual'"),
            ("companion_goals", "progress_mode", "TEXT NOT NULL DEFAULT 'manual'"),
            ("companion_goals", "derivation_key", "TEXT"),
            ("companion_goals", "evidence_json", "TEXT NOT NULL DEFAULT '[]'"),
        ]
        with self._lock:
            conn = self._connect()
            try:
                for table, column, col_def in migrations:
                    existing = {
                        row[1]
                        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
                    }
                    if column not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
                conn.commit()
            finally:
                conn.close()

    # Profiles
    def list_profile_user_ids(self) -> list[str]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("SELECT user_id FROM profiles ORDER BY updated_at DESC, user_id ASC")
                return [str(row["user_id"]) for row in cur.fetchall() if str(row["user_id"]).strip()]
            finally:
                conn.close()

    def get_or_create_profile(self, user_id: str) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (str(user_id),))
                row = cur.fetchone()
                if row:
                    return {k: row[k] for k in row.keys()}
                now = _utcnow_iso()
                conn.execute(
                    """
                    INSERT INTO profiles (user_id, enabled, alpha, beta, gamma, recency_half_life_days, updated_at)
                    VALUES (?, 0, 0.2, 0.6, 0.2, 14, ?)
                    """,
                    (str(user_id), now),
                )
                conn.commit()
                return {
                    "user_id": str(user_id),
                    "enabled": 0,
                    "alpha": 0.2,
                    "beta": 0.6,
                    "gamma": 0.2,
                    "recency_half_life_days": 14,
                    "companion_reflections_enabled": 1,
                    "companion_daily_reflections_enabled": 1,
                    "companion_weekly_reflections_enabled": 1,
                    "updated_at": now,
                }
            finally:
                conn.close()

    def update_profile(self, user_id: str, **fields) -> dict[str, Any]:
        # Ensure row exists
        self.get_or_create_profile(user_id)
        if not fields:
            return self.get_or_create_profile(user_id)
        allowed = {
            "enabled", "alpha", "beta", "gamma", "recency_half_life_days",
            "proactive_enabled", "proactive_frequency", "proactive_types",
            "quiet_hours_start", "quiet_hours_end", "response_style",
            "preferred_format", "session_continuity_enabled",
            "session_summaries_enabled",
            "companion_reflections_enabled",
            "companion_daily_reflections_enabled",
            "companion_weekly_reflections_enabled",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_or_create_profile(user_id)

        # Clamp weight fields to [0.0, 1.0]
        for wk in ("alpha", "beta", "gamma"):
            if wk in updates:
                updates[wk] = max(0.0, min(1.0, float(updates[wk])))
        # Normalize alpha+beta+gamma to sum <= 1.0
        weight_keys = [k for k in ("alpha", "beta", "gamma") if k in updates]
        if weight_keys:
            # Merge with existing to check total
            current = self.get_or_create_profile(user_id)
            a = float(updates.get("alpha", current.get("alpha", 0.2)))
            b = float(updates.get("beta", current.get("beta", 0.6)))
            g = float(updates.get("gamma", current.get("gamma", 0.2)))
            total = a + b + g
            if total > 1.0 and total > 0:
                a, b, g = a / total, b / total, g / total
                # When normalizing, write back all three to keep them consistent
                updates["alpha"] = a
                updates["beta"] = b
                updates["gamma"] = g
            else:
                # No normalization needed - write back only changed keys
                if "alpha" in updates:
                    updates["alpha"] = a
                if "beta" in updates:
                    updates["beta"] = b
                if "gamma" in updates:
                    updates["gamma"] = g

        # Clamp recency_half_life_days to [1, 365]
        if "recency_half_life_days" in updates:
            updates["recency_half_life_days"] = max(1, min(365, int(updates["recency_half_life_days"])))

        # Clear purged_at when re-enabling
        if updates.get("enabled") == 1:
            updates["purged_at"] = None

        updates["updated_at"] = _utcnow_iso()
        set_clause = ", ".join([f"{k} = ?" for k in updates])
        params = list(updates.values()) + [str(user_id)]
        update_profile_sql_template = "UPDATE profiles SET {set_clause} WHERE user_id = ?"
        update_profile_sql = update_profile_sql_template.format_map(locals())  # nosec B608
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(update_profile_sql, params)
                conn.commit()
            finally:
                conn.close()
        return self.get_or_create_profile(user_id)

    # Usage events
    def insert_usage_event(self, evt: UsageEvent) -> str:
        import uuid
        self.get_or_create_profile(evt.user_id)
        eid = uuid.uuid4().hex
        ts = evt.timestamp or _utcnow_iso()
        tags_json = json.dumps(evt.tags) if evt.tags else None
        meta_json = json.dumps(evt.metadata) if evt.metadata else None
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO usage_events (id, user_id, timestamp, type, resource_id, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eid,
                        str(evt.user_id),
                        ts,
                        evt.type,
                        evt.resource_id,
                        tags_json,
                        meta_json,
                    ),
                )
                conn.commit()
                return eid
            finally:
                conn.close()

    # Memories
    def add_semantic_memory(self, mem: SemanticMemory) -> str:
        import uuid
        self.get_or_create_profile(mem.user_id)
        mid = uuid.uuid4().hex
        tags_json = json.dumps(mem.tags) if mem.tags else None
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO semantic_memories (id, user_id, content, tags, pinned, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (mid, str(mem.user_id), mem.content, tags_json, int(mem.pinned or 0), _utcnow_iso()),
                )
                conn.commit()
                return mid
            finally:
                conn.close()

    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM semantic_memories WHERE id = ? AND user_id = ?",
                    (str(memory_id), str(user_id)),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def get_memory(self, memory_id: str, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT id, content, tags, pinned, hidden, created_at FROM semantic_memories WHERE id = ? AND user_id = ?",
                    (str(memory_id), str(user_id)),
                )
                r = cur.fetchone()
                if not r:
                    return None
                return {
                    "id": r["id"],
                    "type": "semantic",
                    "content": r["content"],
                    "pinned": bool(r["pinned"]),
                    "hidden": bool(r["hidden"]),
                    "tags": (json.loads(r["tags"]) if r["tags"] else None),
                    "timestamp": datetime.fromisoformat(r["created_at"]) if r["created_at"] else None,
                }
            finally:
                conn.close()

    def update_memory(self, memory_id: str, user_id: str, **fields) -> dict[str, Any] | None:
        allowed = {"content", "pinned", "hidden", "tags"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return self.get_memory(memory_id, user_id)
        # Serialize tags to JSON
        if "tags" in updates:
            updates["tags"] = json.dumps(updates["tags"]) if updates["tags"] is not None else None
        if "pinned" in updates:
            updates["pinned"] = int(bool(updates["pinned"]))
        if "hidden" in updates:
            updates["hidden"] = int(bool(updates["hidden"]))
        set_clause = ", ".join([f"{k} = ?" for k in updates])
        params = list(updates.values()) + [str(memory_id), str(user_id)]
        update_memory_sql_template = "UPDATE semantic_memories SET {set_clause} WHERE id = ? AND user_id = ?"
        update_memory_sql = update_memory_sql_template.format_map(locals())  # nosec B608
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    update_memory_sql,
                    params,
                )
                conn.commit()
                if cur.rowcount == 0:
                    return None
            finally:
                conn.close()
        return self.get_memory(memory_id, user_id)

    def validate_memories(self, user_id: str, memory_ids: list[str]) -> int:
        """Mark memories as validated by setting last_validated timestamp. Returns count updated."""
        if not memory_ids:
            return 0
        now = _utcnow_iso()
        placeholders = ", ".join(["?"] * len(memory_ids))
        params: list[Any] = [now, str(user_id)] + [str(mid) for mid in memory_ids]
        memory_ids_clause = f"({placeholders})"
        validate_sql_template = "UPDATE semantic_memories SET last_validated = ? WHERE user_id = ? AND id IN {memory_ids_clause}"
        validate_sql = validate_sql_template.format_map(locals())  # nosec B608
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    validate_sql,
                    params,
                )
                conn.commit()
                return cur.rowcount or 0
            finally:
                conn.close()

    def list_semantic_memories(
        self,
        user_id: str,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
        include_hidden: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses = ["user_id = ?"]
        params: list[Any] = [str(user_id)]
        if not include_hidden:
            clauses.append("hidden = 0")
        if q:
            clauses.append("content LIKE ?")
            params.append(f"%{q}%")
        where = " WHERE " + " AND ".join(clauses)
        select_memories_sql_template = (
            "SELECT id, content, tags, pinned, hidden, created_at "
            "FROM semantic_memories{where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        sql = select_memories_sql_template.format_map(locals())  # nosec B608
        params_page = params + [int(limit), int(offset)]
        count_sql_template = "SELECT COUNT(*) as c FROM semantic_memories{where}"
        count_sql = count_sql_template.format_map(locals())  # nosec B608
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(sql, params_page)
                rows = cur.fetchall()
                items: list[dict[str, Any]] = []
                for r in rows:
                    items.append({
                        "id": r["id"],
                        "type": "semantic",
                        "content": r["content"],
                        "pinned": bool(r["pinned"]),
                        "hidden": bool(r["hidden"]),
                        "tags": (json.loads(r["tags"]) if r["tags"] else None),
                        "timestamp": datetime.fromisoformat(r["created_at"]) if r["created_at"] else None,
                    })
                total = int(conn.execute(count_sql, params).fetchone()[0])
                return items, total
            finally:
                conn.close()

    def export_all_memories(self, user_id: str) -> list[dict[str, Any]]:
        """Return all semantic memories for a user (for export)."""
        items, _ = self.list_semantic_memories(user_id, limit=10000, include_hidden=True)
        return items

    def bulk_add_memories(self, user_id: str, memories: list[dict[str, Any]]) -> int:
        """Bulk-insert semantic memories from import data. Returns count inserted."""
        import uuid
        self.get_or_create_profile(user_id)
        count = 0
        with self._lock:
            conn = self._connect()
            try:
                for mem in memories:
                    mid = uuid.uuid4().hex
                    tags_json = json.dumps(mem.get("tags")) if mem.get("tags") else None
                    conn.execute(
                        "INSERT INTO semantic_memories (id, user_id, content, tags, pinned, hidden, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            mid,
                            str(user_id),
                            str(mem.get("content", "")),
                            tags_json,
                            int(bool(mem.get("pinned", False))),
                            int(bool(mem.get("hidden", False))),
                            _utcnow_iso(),
                        ),
                    )
                    count += 1
                conn.commit()
                return count
            finally:
                conn.close()

    # Topics
    def upsert_topic(self, user_id: str, label: str, score: float, last_seen: str | None = None) -> None:
        import uuid
        last = last_seen or _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                # Try update first
                cur = conn.execute(
                    "UPDATE topic_profiles SET score = ?, last_seen = ? WHERE user_id = ? AND label = ?",
                    (float(score), last, str(user_id), label),
                )
                if cur.rowcount == 0:
                    conn.execute(
                        "INSERT INTO topic_profiles (id, user_id, label, score, last_seen) VALUES (?, ?, ?, ?, ?)",
                        (uuid.uuid4().hex, str(user_id), label, float(score), last),
                    )
                conn.commit()
            finally:
                conn.close()

    def topic_counts(self, user_id: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                r = conn.execute("SELECT COUNT(*) FROM topic_profiles WHERE user_id = ?", (str(user_id),)).fetchone()
                return int(r[0] if r else 0)
            finally:
                conn.close()

    def memory_counts(self, user_id: str) -> int:
        with self._lock:
            conn = self._connect()
            try:
                r = conn.execute("SELECT COUNT(*) FROM semantic_memories WHERE user_id = ?", (str(user_id),)).fetchone()
                return int(r[0] if r else 0)
            finally:
                conn.close()

    def session_count(self, user_id: str) -> int:
        """Count distinct sessions (unique event timestamps with type containing 'session')."""
        with self._lock:
            conn = self._connect()
            try:
                r = conn.execute(
                    "SELECT COUNT(DISTINCT resource_id) FROM usage_events WHERE user_id = ? AND resource_id IS NOT NULL",
                    (str(user_id),),
                ).fetchone()
                return int(r[0] if r else 0)
            finally:
                conn.close()

    def list_recent_events(self, user_id: str, limit: int = 500, offset: int = 0) -> list[dict[str, Any]]:
        """Return recent usage events, thread-safe."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    (
                        "SELECT id, timestamp, type, resource_id, tags "
                        "FROM usage_events WHERE user_id = ? "
                        "ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                    ),
                    (str(user_id), int(limit), int(offset)),
                )
                rows = cur.fetchall()
                out: list[dict[str, Any]] = []
                for r in rows:
                    tags = None
                    try:
                        tags = json.loads(r["tags"]) if r["tags"] else None
                    except Exception:
                        tags = None
                    out.append({
                        "id": r["id"],
                        "timestamp": r["timestamp"],
                        "type": r["type"],
                        "resource_id": r["resource_id"],
                        "tags": tags or [],
                    })
                return out
            finally:
                conn.close()

    # Companion
    def insert_companion_activity_event(
        self,
        *,
        user_id: str,
        event_type: str,
        source_type: str,
        source_id: str,
        surface: str,
        dedupe_key: str,
        tags: list[str] | None = None,
        provenance: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        import uuid

        self.get_or_create_profile(user_id)
        event_id = uuid.uuid4().hex
        created_at = _utcnow_iso()
        tags_json = json.dumps(tags) if tags is not None else None
        provenance_json = json.dumps(provenance or {})
        metadata_json = json.dumps(metadata) if metadata is not None else None

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO companion_activity_events (
                        id, user_id, event_type, source_type, source_id, surface,
                        dedupe_key, tags, provenance_json, metadata_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        str(user_id),
                        str(event_type),
                        str(source_type),
                        str(source_id),
                        str(surface),
                        str(dedupe_key),
                        tags_json,
                        provenance_json,
                        metadata_json,
                        created_at,
                    ),
                )
                conn.commit()
                return event_id
            finally:
                conn.close()

    def insert_companion_activity_events_bulk(
        self,
        *,
        user_id: str,
        events: list[dict[str, Any]],
    ) -> list[str]:
        """Insert multiple companion activity events in a single transaction.

        Duplicate dedupe keys are skipped so one conflict does not fail the
        entire batch.
        """
        if not events:
            return []

        import uuid

        self.get_or_create_profile(user_id)
        with self._lock:
            conn = self._connect()
            try:
                existing_dedupe_keys: set[str] = set()
                dedupe_keys = [
                    str(event.get("dedupe_key"))
                    for event in events
                    if str(event.get("dedupe_key", "")).strip()
                ]
                if dedupe_keys:
                    placeholders = ", ".join(["?"] * len(dedupe_keys))
                    existing_rows = conn.execute(
                        f"""
                        SELECT dedupe_key
                        FROM companion_activity_events
                        WHERE user_id = ? AND dedupe_key IN ({placeholders})
                        """,  # nosec B608
                        (str(user_id), *dedupe_keys),
                    ).fetchall()
                    existing_dedupe_keys = {str(row["dedupe_key"]) for row in existing_rows}

                rows: list[tuple[Any, ...]] = []
                seen_dedupe_keys: set[str] = set(existing_dedupe_keys)
                for event in events:
                    dedupe_key = str(event["dedupe_key"])
                    if dedupe_key in seen_dedupe_keys:
                        continue
                    seen_dedupe_keys.add(dedupe_key)
                    event_id = uuid.uuid4().hex
                    rows.append(
                        (
                            event_id,
                            str(user_id),
                            str(event["event_type"]),
                            str(event["source_type"]),
                            str(event["source_id"]),
                            str(event["surface"]),
                            dedupe_key,
                            json.dumps(event.get("tags")) if event.get("tags") is not None else None,
                            json.dumps(event.get("provenance") or {}),
                            json.dumps(event.get("metadata")) if event.get("metadata") is not None else None,
                            _utcnow_iso(),
                        )
                    )

                if not rows:
                    return []

                event_ids: list[str] = []
                for row in rows:
                    cursor = conn.execute(
                        """
                        INSERT OR IGNORE INTO companion_activity_events (
                            id, user_id, event_type, source_type, source_id, surface,
                            dedupe_key, tags, provenance_json, metadata_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        row,
                    )
                    if cursor.rowcount == 1:
                        event_ids.append(str(row[0]))
                conn.commit()
                return event_ids
            finally:
                conn.close()

    def get_companion_activity_event_id_by_dedupe_key(
        self,
        *,
        user_id: str,
        dedupe_key: str,
    ) -> str | None:
        """Return the event id for a user's dedupe key, if one exists."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id
                    FROM companion_activity_events
                    WHERE user_id = ? AND dedupe_key = ?
                    LIMIT 1
                    """,
                    (str(user_id), str(dedupe_key)),
                ).fetchone()
                if row is None:
                    return None
                return str(row["id"])
            finally:
                conn.close()

    def list_companion_activity_events(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT id, event_type, source_type, source_id, surface, tags,
                           provenance_json, metadata_json, created_at
                    FROM companion_activity_events
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (str(user_id), int(limit), int(offset)),
                )
                rows = cur.fetchall()
                items: list[dict[str, Any]] = []
                for row in rows:
                    items.append(
                        {
                            "id": row["id"],
                            "event_type": row["event_type"],
                            "source_type": row["source_type"],
                            "source_id": row["source_id"],
                            "surface": row["surface"],
                            "tags": json.loads(row["tags"]) if row["tags"] else [],
                            "provenance": json.loads(row["provenance_json"] or "{}"),
                            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                            "created_at": row["created_at"],
                        }
                    )
                total_row = conn.execute(
                    "SELECT COUNT(*) FROM companion_activity_events WHERE user_id = ?",
                    (str(user_id),),
                ).fetchone()
                total = int(total_row[0] if total_row else 0)
                return items, total
            finally:
                conn.close()

    def get_companion_activity_event(self, user_id: str, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, event_type, source_type, source_id, surface, tags,
                           provenance_json, metadata_json, created_at
                    FROM companion_activity_events
                    WHERE user_id = ? AND id = ?
                    LIMIT 1
                    """,
                    (str(user_id), str(event_id)),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "id": row["id"],
                    "event_type": row["event_type"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "surface": row["surface"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "provenance": json.loads(row["provenance_json"] or "{}"),
                    "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else {},
                    "created_at": row["created_at"],
                }
            finally:
                conn.close()

    def delete_companion_reflection_activity_events(self, user_id: str) -> tuple[list[str], int]:
        """Delete persisted reflection activity rows and return their ids."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT id
                    FROM companion_activity_events
                    WHERE user_id = ?
                      AND (event_type = 'companion_reflection_generated' OR source_type = 'companion_reflection')
                    """,
                    (str(user_id),),
                ).fetchall()
                reflection_ids = [str(row["id"]) for row in rows]
                if not reflection_ids:
                    return [], 0
                cur = conn.execute(
                    """
                    DELETE FROM companion_activity_events
                    WHERE user_id = ?
                      AND (event_type = 'companion_reflection_generated' OR source_type = 'companion_reflection')
                    """,
                    (str(user_id),),
                )
                conn.commit()
                return reflection_ids, int(cur.rowcount or 0)
            finally:
                conn.close()

    def upsert_companion_knowledge_card(
        self,
        *,
        user_id: str,
        card_type: str,
        title: str,
        summary: str,
        evidence: list[dict[str, Any]] | None = None,
        score: float = 0.0,
        status: str = "active",
    ) -> str:
        import uuid

        self.get_or_create_profile(user_id)
        updated_at = _utcnow_iso()
        evidence_json = json.dumps(evidence or [])

        with self._lock:
            conn = self._connect()
            try:
                existing = conn.execute(
                    """
                    SELECT id FROM companion_knowledge_cards
                    WHERE user_id = ? AND card_type = ? AND title = ?
                    """,
                    (str(user_id), str(card_type), str(title)),
                ).fetchone()
                if existing:
                    card_id = str(existing["id"])
                    conn.execute(
                        """
                        UPDATE companion_knowledge_cards
                        SET summary = ?, evidence_json = ?, score = ?, status = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (str(summary), evidence_json, float(score), str(status), updated_at, card_id),
                    )
                else:
                    card_id = uuid.uuid4().hex
                    conn.execute(
                        """
                        INSERT INTO companion_knowledge_cards (
                            id, user_id, card_type, title, summary, evidence_json, score, status, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card_id,
                            str(user_id),
                            str(card_type),
                            str(title),
                            str(summary),
                            evidence_json,
                            float(score),
                            str(status),
                            updated_at,
                        ),
                    )
                conn.commit()
                return card_id
            finally:
                conn.close()

    def list_companion_knowledge_cards(self, user_id: str, status: str | None = "active") -> list[dict[str, Any]]:
        if status is None:
            sql = (
                "SELECT id, card_type, title, summary, evidence_json, score, status, updated_at "
                "FROM companion_knowledge_cards "
                "WHERE user_id = ? "
                "ORDER BY score DESC, updated_at DESC"
            )
            params: list[Any] = [str(user_id)]
        else:
            sql = (
                "SELECT id, card_type, title, summary, evidence_json, score, status, updated_at "
                "FROM companion_knowledge_cards "
                "WHERE user_id = ? AND status = ? "
                "ORDER BY score DESC, updated_at DESC"
            )
            params = [str(user_id), str(status)]

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [
                    {
                        "id": row["id"],
                        "card_type": row["card_type"],
                        "title": row["title"],
                        "summary": row["summary"],
                        "evidence": json.loads(row["evidence_json"] or "[]"),
                        "score": float(row["score"]),
                        "status": row["status"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def get_companion_knowledge_card(self, user_id: str, card_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, card_type, title, summary, evidence_json, score, status, updated_at
                    FROM companion_knowledge_cards
                    WHERE user_id = ? AND id = ?
                    LIMIT 1
                    """,
                    (str(user_id), str(card_id)),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "id": row["id"],
                    "card_type": row["card_type"],
                    "title": row["title"],
                    "summary": row["summary"],
                    "evidence": json.loads(row["evidence_json"] or "[]"),
                    "score": float(row["score"]),
                    "status": row["status"],
                    "updated_at": row["updated_at"],
                }
            finally:
                conn.close()

    def delete_companion_knowledge_cards(self, user_id: str) -> int:
        """Delete all derived companion knowledge cards for a user."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM companion_knowledge_cards WHERE user_id = ?",
                    (str(user_id),),
                )
                conn.commit()
                return int(cur.rowcount or 0)
            finally:
                conn.close()

    def create_companion_goal(
        self,
        *,
        user_id: str,
        title: str,
        description: str | None,
        goal_type: str,
        config: dict[str, Any] | None = None,
        progress: dict[str, Any] | None = None,
        origin_kind: str = "manual",
        progress_mode: str = "manual",
        derivation_key: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
        status: str = "active",
    ) -> str:
        import uuid

        self.get_or_create_profile(user_id)
        goal_id = uuid.uuid4().hex
        now = _utcnow_iso()

        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO companion_goals (
                        id, user_id, title, description, goal_type, config_json,
                        progress_json, origin_kind, progress_mode, derivation_key,
                        evidence_json, status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        goal_id,
                        str(user_id),
                        str(title),
                        None if description is None else str(description),
                        str(goal_type),
                        json.dumps(config or {}),
                        json.dumps(progress or {}),
                        str(origin_kind),
                        str(progress_mode),
                        None if derivation_key is None else str(derivation_key),
                        json.dumps(evidence or []),
                        str(status),
                        now,
                        now,
                    ),
                )
                conn.commit()
                return goal_id
            finally:
                conn.close()

    def update_companion_goal(self, goal_id: str, user_id: str, **updates: Any) -> dict[str, Any] | None:
        allowed_fields = {
            "title",
            "description",
            "config",
            "progress",
            "origin_kind",
            "progress_mode",
            "derivation_key",
            "evidence",
            "status",
        }
        filtered_updates = {key: value for key, value in updates.items() if key in allowed_fields}

        with self._lock:
            conn = self._connect()
            try:
                if filtered_updates:
                    cur = conn.execute(
                        """
                        UPDATE companion_goals
                        SET title = CASE WHEN ? THEN ? ELSE title END,
                            description = CASE WHEN ? THEN ? ELSE description END,
                            config_json = CASE WHEN ? THEN ? ELSE config_json END,
                            progress_json = CASE WHEN ? THEN ? ELSE progress_json END,
                            origin_kind = CASE WHEN ? THEN ? ELSE origin_kind END,
                            progress_mode = CASE WHEN ? THEN ? ELSE progress_mode END,
                            derivation_key = CASE WHEN ? THEN ? ELSE derivation_key END,
                            evidence_json = CASE WHEN ? THEN ? ELSE evidence_json END,
                            status = CASE WHEN ? THEN ? ELSE status END,
                            updated_at = ?
                        WHERE id = ? AND user_id = ?
                        """,
                        (
                            int("title" in filtered_updates),
                            filtered_updates.get("title"),
                            int("description" in filtered_updates),
                            filtered_updates.get("description"),
                            int("config" in filtered_updates),
                            json.dumps(filtered_updates.get("config") or {}),
                            int("progress" in filtered_updates),
                            json.dumps(filtered_updates.get("progress") or {}),
                            int("origin_kind" in filtered_updates),
                            filtered_updates.get("origin_kind"),
                            int("progress_mode" in filtered_updates),
                            filtered_updates.get("progress_mode"),
                            int("derivation_key" in filtered_updates),
                            filtered_updates.get("derivation_key"),
                            int("evidence" in filtered_updates),
                            json.dumps(filtered_updates.get("evidence") or []),
                            int("status" in filtered_updates),
                            filtered_updates.get("status"),
                            _utcnow_iso(),
                            str(goal_id),
                            str(user_id),
                        ),
                    )
                    if (cur.rowcount or 0) == 0:
                        return None
                    conn.commit()

                row = conn.execute(
                    """
                    SELECT id, title, description, goal_type, config_json, progress_json,
                           origin_kind, progress_mode, derivation_key, evidence_json,
                           status, created_at, updated_at
                    FROM companion_goals
                    WHERE id = ? AND user_id = ?
                    """,
                    (str(goal_id), str(user_id)),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "id": row["id"],
                    "title": row["title"],
                    "description": row["description"],
                    "goal_type": row["goal_type"],
                    "config": json.loads(row["config_json"] or "{}"),
                    "progress": json.loads(row["progress_json"] or "{}"),
                    "origin_kind": row["origin_kind"],
                    "progress_mode": row["progress_mode"],
                    "derivation_key": row["derivation_key"],
                    "evidence": json.loads(row["evidence_json"] or "[]"),
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            finally:
                conn.close()

    def list_companion_goals(self, user_id: str, status: str | None = None) -> list[dict[str, Any]]:
        if status is None:
            sql = (
                "SELECT id, title, description, goal_type, config_json, progress_json, "
                "origin_kind, progress_mode, derivation_key, evidence_json, status, created_at, updated_at "
                "FROM companion_goals "
                "WHERE user_id = ? "
                "ORDER BY updated_at DESC"
            )
            params: list[Any] = [str(user_id)]
        else:
            sql = (
                "SELECT id, title, description, goal_type, config_json, progress_json, "
                "origin_kind, progress_mode, derivation_key, evidence_json, status, created_at, updated_at "
                "FROM companion_goals "
                "WHERE user_id = ? AND status = ? "
                "ORDER BY updated_at DESC"
            )
            params = [str(user_id), str(status)]

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "description": row["description"],
                        "goal_type": row["goal_type"],
                        "config": json.loads(row["config_json"] or "{}"),
                        "progress": json.loads(row["progress_json"] or "{}"),
                        "origin_kind": row["origin_kind"],
                        "progress_mode": row["progress_mode"],
                        "derivation_key": row["derivation_key"],
                        "evidence": json.loads(row["evidence_json"] or "[]"),
                        "status": row["status"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def get_companion_goal(self, goal_id: str, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, title, description, goal_type, config_json, progress_json,
                           origin_kind, progress_mode, derivation_key, evidence_json,
                           status, created_at, updated_at
                    FROM companion_goals
                    WHERE id = ? AND user_id = ?
                    LIMIT 1
                    """,
                    (str(goal_id), str(user_id)),
                ).fetchone()
                if row is None:
                    return None
                return {
                    "id": row["id"],
                    "title": row["title"],
                    "description": row["description"],
                    "goal_type": row["goal_type"],
                    "config": json.loads(row["config_json"] or "{}"),
                    "progress": json.loads(row["progress_json"] or "{}"),
                    "origin_kind": row["origin_kind"],
                    "progress_mode": row["progress_mode"],
                    "derivation_key": row["derivation_key"],
                    "evidence": json.loads(row["evidence_json"] or "[]"),
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            finally:
                conn.close()

    def delete_companion_goals_by_origin_kind(self, user_id: str, origin_kind: str) -> int:
        """Delete companion goals matching a specific origin kind."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM companion_goals WHERE user_id = ? AND origin_kind = ?",
                    (str(user_id), str(origin_kind)),
                )
                conn.commit()
                return int(cur.rowcount or 0)
            finally:
                conn.close()

    def reset_companion_goal_progress(self, user_id: str, progress_mode: str = "computed") -> int:
        """Clear computed goal progress without deleting the underlying goal rows."""
        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    UPDATE companion_goals
                    SET progress_json = ?, evidence_json = ?, updated_at = ?
                    WHERE user_id = ? AND progress_mode = ?
                    """,
                    ("{}", "[]", now, str(user_id), str(progress_mode)),
                )
                conn.commit()
                return int(cur.rowcount or 0)
            finally:
                conn.close()

    def purge_user(self, user_id: str) -> dict[str, int]:
        with self._lock:
            conn = self._connect()
            try:
                counts: dict[str, int] = {}
                for table in (
                    "usage_events",
                    "semantic_memories",
                    "episodic_memories",
                    "topic_profiles",
                    "companion_activity_events",
                    "companion_knowledge_cards",
                    "companion_goals",
                ):
                    delete_table_sql_template = "DELETE FROM {table} WHERE user_id = ?"
                    delete_table_sql = delete_table_sql_template.format_map(locals())  # nosec B608
                    cur = conn.execute(delete_table_sql, (str(user_id),))
                    counts[table] = cur.rowcount or 0
                conn.commit()
                # Reset profile to disabled and stamp purged_at
                now = _utcnow_iso()
                conn.execute(
                    "UPDATE profiles SET enabled = 0, purged_at = ?, updated_at = ? WHERE user_id = ?",
                    (now, now, str(user_id)),
                )
                conn.commit()
                return counts
            finally:
                conn.close()

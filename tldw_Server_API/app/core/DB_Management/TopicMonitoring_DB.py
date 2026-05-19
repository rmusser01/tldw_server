from __future__ import annotations

"""
TopicMonitoring_DB
Lightweight SQLite wrapper for topic monitoring storage.

Tables:
  - monitoring_watchlists
  - monitoring_watchlist_rules
  - topic_alerts

topic_alerts columns (key subset):
  - id INTEGER PRIMARY KEY AUTOINCREMENT
  - created_at TEXT (ISO 8601 UTC)
  - user_id TEXT
  - scope_type TEXT  -- 'global' | 'user' | 'team' | 'org'
  - scope_id TEXT
  - source TEXT       -- 'chat.input' | 'chat.output' | 'ingestion' | 'notes' | 'rag'
  - watchlist_id TEXT
  - rule_id TEXT
  - rule_category TEXT
  - rule_severity TEXT
  - pattern TEXT
  - source_id TEXT
  - chunk_id TEXT
  - chunk_seq INTEGER
  - text_snippet TEXT
  - metadata TEXT     -- JSON string
  - is_read INTEGER DEFAULT 0
  - read_at TEXT NULL

This module intentionally keeps SQL usage encapsulated within DB_Management, per project guidelines.
"""

import json
import os
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    resolve_trusted_database_path,
)
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    begin_immediate_if_needed,
    configure_sqlite_connection,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TopicAlert:
    user_id: str | None
    scope_type: str
    scope_id: str | None
    source: str
    watchlist_id: str
    rule_id: str | None = None
    rule_category: str | None = None
    rule_severity: str | None = None
    pattern: str = ""
    source_id: str | None = None
    chunk_id: str | None = None
    chunk_seq: int | None = None
    text_snippet: str = ""
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    is_read: int = 0
    read_at: str | None = None


@dataclass
class WatchlistRecord:
    id: str
    name: str
    description: str | None = None
    enabled: bool = True
    scope_type: str = "user"
    scope_id: str | None = None
    managed_by: str = "api"
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class WatchlistRuleRecord:
    rule_id: str
    watchlist_id: str
    pattern: str
    category: str | None = None
    severity: str | None = None
    note: str | None = None
    tags: list[str] | None = None
    created_at: str | None = None
    updated_at: str | None = None


class TopicMonitoringDB:
    def __init__(self, db_path: str = "Databases/monitoring_alerts.db") -> None:
        path_obj = Path(db_path)
        if not path_obj.is_absolute() and path_obj.parent == Path("."):
            msg = (
                "TopicMonitoringDB db_path must include a directory when using a relative path "
                f"(got {db_path!r}). Use an absolute path or include a directory component."
            )
            logger.error(msg)
            raise RuntimeError(msg)
        extra_roots = (
            [path_obj.expanduser().resolve(strict=False).parent]
            if path_obj.is_absolute()
            else None
        )
        self.db_path = str(
            resolve_trusted_database_path(
                db_path,
                label="topic monitoring db",
                extra_roots=extra_roots,
            )
        )
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            configure_sqlite_connection(conn)
        except Exception as pragma_error:
            logger.debug("TopicMonitoringDB pragma initialization failed", exc_info=pragma_error)
        return conn

    @staticmethod
    def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
        try:
            cur = conn.execute(f"PRAGMA table_info({table})")
            rows = cur.fetchall()
            return any(str(row["name"]) == column for row in rows)
        except Exception:
            return False

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monitoring_watchlists (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        scope_type TEXT NOT NULL,
                        scope_id TEXT,
                        managed_by TEXT NOT NULL DEFAULT 'api',
                        created_at TEXT,
                        updated_at TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS monitoring_watchlist_rules (
                        rule_id TEXT NOT NULL,
                        watchlist_id TEXT NOT NULL,
                        pattern TEXT NOT NULL,
                        category TEXT,
                        severity TEXT,
                        note TEXT,
                        tags TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        PRIMARY KEY (watchlist_id, rule_id),
                        FOREIGN KEY(watchlist_id) REFERENCES monitoring_watchlists(id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_monitoring_watchlists_scope ON monitoring_watchlists(scope_type, scope_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_monitoring_watchlists_enabled ON monitoring_watchlists(enabled)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS topic_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        user_id TEXT,
                        scope_type TEXT,
                        scope_id TEXT,
                        source TEXT,
                        watchlist_id TEXT,
                        rule_id TEXT,
                        rule_category TEXT,
                        rule_severity TEXT,
                        pattern TEXT,
                        source_id TEXT,
                        chunk_id TEXT,
                        chunk_seq INTEGER,
                        text_snippet TEXT,
                        metadata TEXT,
                        is_read INTEGER DEFAULT 0,
                        read_at TEXT
                    )
                    """
                )
                if self._watchlist_rules_pk_needs_migration(conn):
                    self._migrate_watchlist_rules_schema(conn)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_monitoring_watchlist_rules_watchlist ON monitoring_watchlist_rules(watchlist_id)"
                )
                if not self._column_exists(conn, "topic_alerts", "rule_id"):
                    conn.execute("ALTER TABLE topic_alerts ADD COLUMN rule_id TEXT")
                if not self._column_exists(conn, "topic_alerts", "source_id"):
                    conn.execute("ALTER TABLE topic_alerts ADD COLUMN source_id TEXT")
                if not self._column_exists(conn, "topic_alerts", "chunk_id"):
                    conn.execute("ALTER TABLE topic_alerts ADD COLUMN chunk_id TEXT")
                if not self._column_exists(conn, "topic_alerts", "chunk_seq"):
                    conn.execute("ALTER TABLE topic_alerts ADD COLUMN chunk_seq INTEGER")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_topic_alerts_created_at ON topic_alerts(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_topic_alerts_user ON topic_alerts(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_topic_alerts_watch ON topic_alerts(watchlist_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_topic_alerts_rule ON topic_alerts(rule_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_topic_alerts_source ON topic_alerts(source)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_topic_alerts_read ON topic_alerts(is_read)")
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _watchlist_rules_pk_needs_migration(conn: sqlite3.Connection) -> bool:
        try:
            rows = conn.execute("PRAGMA table_info(monitoring_watchlist_rules)").fetchall()
        except Exception:
            return False
        if not rows:
            return False
        pk_cols = [
            row["name"]
            for row in sorted(rows, key=lambda r: r["pk"])
            if row["pk"]
        ]
        return pk_cols == ["rule_id"]

    def _migrate_watchlist_rules_schema(self, conn: sqlite3.Connection) -> None:
        logger.info("Migrating monitoring_watchlist_rules schema to composite primary key")
        try:
            begin_immediate_if_needed(conn)
            conn.execute("DROP INDEX IF EXISTS idx_monitoring_watchlist_rules_watchlist")
            conn.execute("ALTER TABLE monitoring_watchlist_rules RENAME TO monitoring_watchlist_rules_old")
            conn.execute(
                """
                CREATE TABLE monitoring_watchlist_rules (
                    rule_id TEXT NOT NULL,
                    watchlist_id TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    category TEXT,
                    severity TEXT,
                    note TEXT,
                    tags TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (watchlist_id, rule_id),
                    FOREIGN KEY(watchlist_id) REFERENCES monitoring_watchlists(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                INSERT INTO monitoring_watchlist_rules (
                    rule_id, watchlist_id, pattern, category, severity, note, tags, created_at, updated_at
                )
                SELECT rule_id, watchlist_id, pattern, category, severity, note, tags, created_at, updated_at
                FROM monitoring_watchlist_rules_old
                """
            )
            conn.execute("DROP TABLE monitoring_watchlist_rules_old")
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error("Failed migrating monitoring_watchlist_rules schema: {}", exc)
            raise

    @staticmethod
    def _normalize_tags(tags: Iterable[str] | None) -> list[str] | None:
        if tags is None:
            return None
        cleaned = {str(tag).strip() for tag in tags if tag is not None and str(tag).strip()}
        return sorted(cleaned) if cleaned else []

    @staticmethod
    def _serialize_tags(tags: Iterable[str] | None) -> str | None:
        norm = TopicMonitoringDB._normalize_tags(tags)
        if norm is None:
            return None
        try:
            return json.dumps(norm, ensure_ascii=False)
        except Exception:
            return None

    @staticmethod
    def _deserialize_tags(raw: str | None) -> list[str] | None:
        if raw is None:
            return None
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(x) for x in data if str(x).strip()]
        except Exception as tags_decode_error:
            logger.debug("TopicMonitoringDB failed to deserialize tags payload", exc_info=tags_decode_error)
        return []

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        """Safely materialize sqlite3.Row (or row-like objects) as a plain dict."""
        if row is None:
            return {}
        if isinstance(row, dict):
            return dict(row)
        row_keys = getattr(row, "keys", None)
        if callable(row_keys):
            try:
                return {k: row[k] for k in row_keys()}
            except Exception as row_materialize_error:
                logger.debug("TopicMonitoringDB row key-based materialization failed", exc_info=row_materialize_error)
        try:
            return dict(row)
        except Exception:
            return {}

    def list_watchlists(
        self,
        *,
        include_rules: bool = True,
        enabled_only: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                where = " WHERE enabled = 1" if enabled_only else ""
                watchlists_sql_template = """
                    SELECT id, name, description, enabled, scope_type, scope_id, managed_by, created_at, updated_at
                    FROM monitoring_watchlists
                    {where}
                    ORDER BY name COLLATE NOCASE
                    """
                watchlists_sql = watchlists_sql_template.format_map(locals())  # nosec B608
                rows = conn.execute(
                    watchlists_sql
                ).fetchall()
                watchlists: list[dict[str, Any]] = []
                for row in rows:
                    watchlists.append(self._row_to_dict(row))
                if not include_rules or not watchlists:
                    return watchlists
                ids = [str(w["id"]) for w in watchlists if w.get("id") is not None]
                if not ids:
                    return watchlists
                placeholders = ",".join("?" for _ in ids)
                watchlist_ids_clause = f"({placeholders})"
                rules_sql_template = """
                    SELECT rule_id, watchlist_id, pattern, category, severity, note, tags, created_at, updated_at
                    FROM monitoring_watchlist_rules
                    WHERE watchlist_id IN {watchlist_ids_clause}
                    ORDER BY rule_id
                    """
                rules_sql = rules_sql_template.format_map(locals())  # nosec B608
                rule_rows = conn.execute(
                    rules_sql,
                    ids,
                ).fetchall()
                rules_by_watchlist: dict[str, list[dict[str, Any]]] = {}
                for row in rule_rows:
                    item = self._row_to_dict(row)
                    item["tags"] = self._deserialize_tags(item.get("tags"))
                    rules_by_watchlist.setdefault(str(item["watchlist_id"]), []).append(item)
                for wl in watchlists:
                    wl["rules"] = rules_by_watchlist.get(str(wl.get("id")), [])
                return watchlists
            finally:
                conn.close()

    def get_watchlist(self, watchlist_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT id, name, description, enabled, scope_type, scope_id, managed_by, created_at, updated_at
                    FROM monitoring_watchlists
                    WHERE id = ?
                    """,
                    (str(watchlist_id),),
                ).fetchone()
                if not row:
                    return None
                wl = self._row_to_dict(row)
                return wl
            finally:
                conn.close()

    def get_watchlist_by_key(
        self,
        name: str,
        scope_type: str,
        scope_id: str | None,
    ) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                if scope_id is None:
                    row = conn.execute(
                        """
                        SELECT id, name, description, enabled, scope_type, scope_id, managed_by, created_at, updated_at
                        FROM monitoring_watchlists
                        WHERE name = ? AND scope_type = ? AND scope_id IS NULL
                        """,
                        (str(name), str(scope_type)),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT id, name, description, enabled, scope_type, scope_id, managed_by, created_at, updated_at
                        FROM monitoring_watchlists
                        WHERE name = ? AND scope_type = ? AND scope_id = ?
                        """,
                        (str(name), str(scope_type), str(scope_id)),
                ).fetchone()
                if not row:
                    return None
                return self._row_to_dict(row)
            finally:
                conn.close()

    def upsert_watchlist(self, watchlist: WatchlistRecord) -> str:
        with self._lock:
            conn = self._connect()
            try:
                now = _utcnow_iso()
                row = conn.execute(
                    "SELECT id FROM monitoring_watchlists WHERE id = ?",
                    (str(watchlist.id),),
                ).fetchone()
                if row:
                    conn.execute(
                        """
                        UPDATE monitoring_watchlists
                        SET name = ?, description = ?, enabled = ?, scope_type = ?, scope_id = ?, managed_by = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            watchlist.name,
                            watchlist.description,
                            1 if watchlist.enabled else 0,
                            watchlist.scope_type,
                            watchlist.scope_id,
                            watchlist.managed_by,
                            now,
                            watchlist.id,
                        ),
                    )
                else:
                    created_at = watchlist.created_at or now
                    conn.execute(
                        """
                        INSERT INTO monitoring_watchlists (
                            id, name, description, enabled, scope_type, scope_id, managed_by, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            watchlist.id,
                            watchlist.name,
                            watchlist.description,
                            1 if watchlist.enabled else 0,
                            watchlist.scope_type,
                            watchlist.scope_id,
                            watchlist.managed_by,
                            created_at,
                            watchlist.updated_at or now,
                        ),
                    )
                conn.commit()
                return str(watchlist.id)
            finally:
                conn.close()

    def delete_watchlist(self, watchlist_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM monitoring_watchlists WHERE id = ?",
                    (str(watchlist_id),),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def replace_watchlist_rules(
        self,
        watchlist_id: str,
        rules: list[WatchlistRuleRecord],
    ) -> None:
        with self._lock:
            conn = self._connect()
            try:
                begin_immediate_if_needed(conn)
                conn.execute(
                    "DELETE FROM monitoring_watchlist_rules WHERE watchlist_id = ?",
                    (str(watchlist_id),),
                )
                now = _utcnow_iso()
                for rule in rules:
                    conn.execute(
                        """
                        INSERT INTO monitoring_watchlist_rules (
                            rule_id, watchlist_id, pattern, category, severity, note, tags, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rule.rule_id,
                            rule.watchlist_id,
                            rule.pattern,
                            rule.category,
                            rule.severity,
                            rule.note,
                            self._serialize_tags(rule.tags),
                            rule.created_at or now,
                            rule.updated_at or now,
                        ),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def insert_alert(self, alert: TopicAlert) -> int:
        with self._lock:
            conn = self._connect()
            try:
                created_at = alert.created_at or _utcnow_iso()
                metadata_json = None
                if alert.metadata is not None:
                    try:
                        metadata_json = json.dumps(alert.metadata, ensure_ascii=False)
                    except Exception:
                        metadata_json = None
                cur = conn.execute(
                    """
                    INSERT INTO topic_alerts (
                        created_at, user_id, scope_type, scope_id, source, watchlist_id,
                        rule_id, rule_category, rule_severity, pattern, source_id, chunk_id, chunk_seq,
                        text_snippet, metadata, is_read, read_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        created_at,
                        alert.user_id,
                        alert.scope_type,
                        alert.scope_id,
                        alert.source,
                        alert.watchlist_id,
                        alert.rule_id,
                        alert.rule_category,
                        alert.rule_severity,
                        alert.pattern,
                        alert.source_id,
                        alert.chunk_id,
                        alert.chunk_seq,
                        alert.text_snippet,
                        metadata_json,
                        int(alert.is_read or 0),
                        alert.read_at,
                    ),
                )
                conn.commit()
                return int(cur.lastrowid)
            finally:
                conn.close()

    def recent_duplicate_exists(
        self,
        user_id: str | None,
        watchlist_id: str,
        source: str,
        *,
        pattern: str | None = None,
        rule_id: str | None = None,
        source_id: str | None = None,
        window_seconds: int = 300,
    ) -> bool:
        """Check if an alert for the same (user, watchlist, rule/pattern, source) exists in recent window."""
        # Compute threshold timestamp
        threshold = (datetime.now(timezone.utc) - timedelta(seconds=window_seconds)).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                clauses = ["watchlist_id = ?", "source = ?", "created_at >= ?"]
                params: list[Any] = [str(watchlist_id), str(source), threshold]
                if user_id is None:
                    clauses.append("user_id IS NULL")
                else:
                    clauses.append("user_id = ?")
                    params.append(str(user_id))
                if rule_id:
                    clauses.append("rule_id = ?")
                    params.append(str(rule_id))
                elif pattern:
                    clauses.append("pattern = ?")
                    params.append(str(pattern))
                if source_id:
                    clauses.append("source_id = ?")
                    params.append(str(source_id))
                where = " AND ".join(clauses)
                cur = conn.execute(
                    f"SELECT 1 FROM topic_alerts WHERE {where} LIMIT 1",  # nosec B608
                    params,
                )
                row = cur.fetchone()
                return bool(row)
            finally:
                conn.close()

    def list_alerts(
        self,
        user_id: str | None = None,
        source: str | None = None,
        rule_category: str | None = None,
        rule_severity: str | None = None,
        scope_type: str | None = None,
        scope_id: str | None = None,
        since_iso: str | None = None,
        unread_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(str(user_id))
        if source:
            clauses.append("source = ?")
            params.append(str(source))
        if rule_category:
            clauses.append("rule_category = ?")
            params.append(str(rule_category))
        if rule_severity:
            clauses.append("rule_severity = ?")
            params.append(str(rule_severity))
        if scope_type:
            clauses.append("scope_type = ?")
            params.append(str(scope_type))
        if scope_id:
            clauses.append("scope_id = ?")
            params.append(str(scope_id))
        if since_iso:
            clauses.append("created_at >= ?")
            params.append(since_iso)
        if unread_only:
            clauses.append("is_read = 0")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        alerts_sql_template = """
            SELECT id, created_at, user_id, scope_type, scope_id, source,
                   watchlist_id, rule_id, rule_category, rule_severity, pattern,
                   source_id, chunk_id, chunk_seq, text_snippet, metadata, is_read, read_at
            FROM topic_alerts
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        sql = alerts_sql_template.format_map(locals())  # nosec B608
        params.extend([int(limit), int(offset)])
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(sql, params)
                rows = cur.fetchall()
                out: list[dict[str, Any]] = []
                for r in rows:
                    item = {key: r[key] for key in r.keys()}
                    out.append(item)
                return out
            finally:
                conn.close()

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT id, created_at, user_id, scope_type, scope_id, source,
                           watchlist_id, rule_id, rule_category, rule_severity, pattern,
                           source_id, chunk_id, chunk_seq, text_snippet, metadata, is_read, read_at
                    FROM topic_alerts
                    WHERE id = ?
                    """,
                    (int(alert_id),),
                )
                row = cur.fetchone()
                return {key: row[key] for key in row.keys()} if row else None
            finally:
                conn.close()

    def mark_read(self, alert_id: int) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                now_iso = _utcnow_iso()
                cur = conn.execute(
                    "UPDATE topic_alerts SET is_read = 1, read_at = ? WHERE id = ?",
                    (now_iso, int(alert_id)),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

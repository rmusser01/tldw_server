"""SQLite-backed ACP session persistence.

Provides durable storage for ACP session records. Returns plain dicts
to avoid circular imports with the in-memory session store layer.
"""
from __future__ import annotations

import fnmatch as _fnmatch
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)

_SCHEMA_VERSION = 13

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    agent_type TEXT NOT NULL DEFAULT 'custom',
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    cwd TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_activity_at TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    bootstrap_ready INTEGER NOT NULL DEFAULT 1,
    needs_bootstrap INTEGER NOT NULL DEFAULT 0,
    forked_from TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    mcp_servers TEXT NOT NULL DEFAULT '[]',
    persona_id TEXT,
    workspace_id TEXT,
    workspace_group_id TEXT,
    scope_snapshot_id TEXT,
    policy_snapshot_version TEXT,
    policy_snapshot_fingerprint TEXT,
    policy_snapshot_refreshed_at TEXT,
    policy_summary TEXT,
    policy_provenance_summary TEXT,
    policy_refresh_error TEXT,
    model TEXT,
    token_budget INTEGER DEFAULT NULL,
    auto_terminate_at_budget INTEGER NOT NULL DEFAULT 0,
    budget_exhausted INTEGER NOT NULL DEFAULT 0,
    ancestry_chain_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_status ON sessions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_created_agent ON sessions(created_at, agent_type);
CREATE INDEX IF NOT EXISTS idx_sessions_forked ON sessions(forked_from);

CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    raw_data TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_session_idx
    ON session_messages(session_id, message_index);

CREATE TABLE IF NOT EXISTS agent_registry (
    agent_type TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    command TEXT NOT NULL DEFAULT '',
    args TEXT NOT NULL DEFAULT '[]',
    env TEXT NOT NULL DEFAULT '{}',
    requires_api_key TEXT,
    is_default INTEGER NOT NULL DEFAULT 0,
    install_instructions TEXT NOT NULL DEFAULT '[]',
    docs_url TEXT,
    mcp_orchestration TEXT NOT NULL DEFAULT 'agent_driven',
    mcp_entry_tool TEXT NOT NULL DEFAULT 'execute',
    mcp_structured_response INTEGER NOT NULL DEFAULT 0,
    mcp_llm_provider TEXT,
    mcp_llm_model TEXT,
    mcp_max_iterations INTEGER NOT NULL DEFAULT 20,
    mcp_refresh_tools INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'api',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_health_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_type TEXT NOT NULL,
    health TEXT NOT NULL,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    details TEXT NOT NULL DEFAULT '{}',
    checked_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_health_agent_time
    ON agent_health_history(agent_type, checked_at DESC);

CREATE TABLE IF NOT EXISTS permission_policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    rules_json TEXT NOT NULL,
    conditions_json TEXT,
    org_id TEXT,
    team_id TEXT,
    priority INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS permission_decisions (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    tool_pattern TEXT NOT NULL,
    decision TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'session',
    session_id TEXT,
    persona_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_perm_dec_user ON permission_decisions(user_id);
CREATE INDEX IF NOT EXISTS idx_perm_dec_pattern ON permission_decisions(tool_pattern);

CREATE TABLE IF NOT EXISTS webhook_triggers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'generic',
    secret_encrypted TEXT NOT NULL,
    owner_user_id INTEGER NOT NULL,
    agent_config_json TEXT NOT NULL DEFAULT '{}',
    prompt_template TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_webhook_triggers_owner
    ON webhook_triggers(owner_user_id);

CREATE TABLE IF NOT EXISTS config_templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    scope TEXT NOT NULL DEFAULT 'system',
    scope_id TEXT,
    base_template_id INTEGER,
    schema_version TEXT NOT NULL DEFAULT '1',
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_config_templates_scope
    ON config_templates(scope, scope_id);
CREATE INDEX IF NOT EXISTS idx_config_templates_name
    ON config_templates(name);
"""

# Columns that are stored as INTEGER 0/1 but should be returned as bool
_BOOL_FIELDS = frozenset({
    "bootstrap_ready",
    "needs_bootstrap",
    "mcp_structured_response",
    "mcp_refresh_tools",
    "auto_terminate_at_budget",
    "budget_exhausted",
})

# Columns that are stored as JSON TEXT but should be returned as parsed objects
_JSON_LIST_FIELDS = frozenset({"tags", "mcp_servers", "ancestry_chain_json"})
_JSON_OBJECT_FIELDS = frozenset({"policy_summary", "policy_provenance_summary"})
_ALLOWED_MIGRATION_COLUMNS = {
    "sessions": {
        "policy_snapshot_version": "policy_snapshot_version TEXT",
        "policy_snapshot_fingerprint": "policy_snapshot_fingerprint TEXT",
        "policy_snapshot_refreshed_at": "policy_snapshot_refreshed_at TEXT",
        "policy_summary": "policy_summary TEXT",
        "policy_provenance_summary": "policy_provenance_summary TEXT",
        "policy_refresh_error": "policy_refresh_error TEXT",
        "model": "model TEXT",
        "token_budget": "token_budget INTEGER DEFAULT NULL",
        "auto_terminate_at_budget": "auto_terminate_at_budget INTEGER NOT NULL DEFAULT 0",
        "budget_exhausted": "budget_exhausted INTEGER NOT NULL DEFAULT 0",
        "ancestry_chain_json": "ancestry_chain_json TEXT",
    },
    "agent_registry": {
        "mcp_orchestration": "mcp_orchestration TEXT NOT NULL DEFAULT 'agent_driven'",
        "mcp_entry_tool": "mcp_entry_tool TEXT NOT NULL DEFAULT 'execute'",
        "mcp_structured_response": "mcp_structured_response INTEGER NOT NULL DEFAULT 0",
        "mcp_llm_provider": "mcp_llm_provider TEXT",
        "mcp_llm_model": "mcp_llm_model TEXT",
        "mcp_max_iterations": "mcp_max_iterations INTEGER NOT NULL DEFAULT 20",
        "mcp_refresh_tools": "mcp_refresh_tools INTEGER NOT NULL DEFAULT 0",
    },
    "permission_policies": {
        "conditions_json": "conditions_json TEXT",
    },
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_sql: str,
) -> None:
    expected_column_sql = _ALLOWED_MIGRATION_COLUMNS.get(table_name, {}).get(column_name)
    if expected_column_sql != column_sql:
        raise ValueError("Unsupported ACP session migration target")

    existing_columns = {
        str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    }
    if column_name in existing_columns:
        return
    conn.execute(f'ALTER TABLE "{table_name}" ADD COLUMN {column_sql}')


def _ensure_config_template_unique_index(conn: sqlite3.Connection) -> None:
    """Enforce deterministic template lookup within a scope."""
    duplicate = conn.execute(
        """
        SELECT name, scope, COALESCE(scope_id, '') AS normalized_scope_id, COUNT(*) AS duplicate_count
        FROM config_templates
        GROUP BY name, scope, COALESCE(scope_id, '')
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        raise sqlite3.IntegrityError(
            "Duplicate config_templates rows detected for "
            f"name={duplicate['name']!r}, scope={duplicate['scope']!r}, "
            f"scope_id={duplicate['normalized_scope_id']!r}"
        )

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_config_templates_name_scope_unique
        ON config_templates(name, scope, COALESCE(scope_id, ''))
        """
    )


class ACPSessionsDB:
    """SQLite-backed ACP session store."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "Databases", "acp_sessions.db",
            )
        self._db_path = os.path.abspath(db_path)
        self._conn_local = threading.local()
        self._initialized = False
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local SQLite connection."""
        conn: sqlite3.Connection | None = getattr(self._conn_local, "conn", None)
        if conn is None:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            configure_sqlite_connection(conn)
            self._conn_local.conn = conn
        self._ensure_schema()
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if needed (idempotent, double-checked locking)."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn: sqlite3.Connection | None = getattr(self._conn_local, "conn", None)
            if conn is None:
                return  # _get_conn will call us again after creating conn
            conn.executescript(_SCHEMA_SQL)
            # Migrate schema forward as needed
            current_version = conn.execute("PRAGMA user_version").fetchone()[0]
            if current_version < 2:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS agent_registry (
                        agent_type TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        command TEXT NOT NULL DEFAULT '',
                        args TEXT NOT NULL DEFAULT '[]',
                        env TEXT NOT NULL DEFAULT '{}',
                        requires_api_key TEXT,
                        is_default INTEGER NOT NULL DEFAULT 0,
                        install_instructions TEXT NOT NULL DEFAULT '[]',
                        docs_url TEXT,
                        mcp_orchestration TEXT NOT NULL DEFAULT 'agent_driven',
                        mcp_entry_tool TEXT NOT NULL DEFAULT 'execute',
                        mcp_structured_response INTEGER NOT NULL DEFAULT 0,
                        mcp_llm_provider TEXT,
                        mcp_llm_model TEXT,
                        mcp_max_iterations INTEGER NOT NULL DEFAULT 20,
                        mcp_refresh_tools INTEGER NOT NULL DEFAULT 0,
                        source TEXT NOT NULL DEFAULT 'api',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                """)
            if current_version < 3:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS agent_health_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_type TEXT NOT NULL,
                        health TEXT NOT NULL,
                        consecutive_failures INTEGER NOT NULL DEFAULT 0,
                        details TEXT NOT NULL DEFAULT '{}',
                        checked_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_health_agent_time
                        ON agent_health_history(agent_type, checked_at DESC);
                """)
            if current_version < 4:
                _ensure_column(
                    conn,
                    "sessions",
                    "policy_snapshot_version",
                    "policy_snapshot_version TEXT",
                )
                _ensure_column(
                    conn,
                    "sessions",
                    "policy_snapshot_fingerprint",
                    "policy_snapshot_fingerprint TEXT",
                )
                _ensure_column(
                    conn,
                    "sessions",
                    "policy_snapshot_refreshed_at",
                    "policy_snapshot_refreshed_at TEXT",
                )
                _ensure_column(
                    conn,
                    "sessions",
                    "policy_summary",
                    "policy_summary TEXT",
                )
                _ensure_column(
                    conn,
                    "sessions",
                    "policy_provenance_summary",
                    "policy_provenance_summary TEXT",
                )
                _ensure_column(
                    conn,
                    "sessions",
                    "policy_refresh_error",
                    "policy_refresh_error TEXT",
                )
            if current_version < 5:
                _ensure_column(
                    conn,
                    "agent_registry",
                    "mcp_orchestration",
                    "mcp_orchestration TEXT NOT NULL DEFAULT 'agent_driven'",
                )
                _ensure_column(
                    conn,
                    "agent_registry",
                    "mcp_entry_tool",
                    "mcp_entry_tool TEXT NOT NULL DEFAULT 'execute'",
                )
                _ensure_column(
                    conn,
                    "agent_registry",
                    "mcp_structured_response",
                    "mcp_structured_response INTEGER NOT NULL DEFAULT 0",
                )
                _ensure_column(
                    conn,
                    "agent_registry",
                    "mcp_llm_provider",
                    "mcp_llm_provider TEXT",
                )
                _ensure_column(
                    conn,
                    "agent_registry",
                    "mcp_llm_model",
                    "mcp_llm_model TEXT",
                )
                _ensure_column(
                    conn,
                    "agent_registry",
                    "mcp_max_iterations",
                    "mcp_max_iterations INTEGER NOT NULL DEFAULT 20",
                )
                _ensure_column(
                    conn,
                    "agent_registry",
                    "mcp_refresh_tools",
                    "mcp_refresh_tools INTEGER NOT NULL DEFAULT 0",
                )
            if current_version < 6:
                _ensure_column(
                    conn,
                    "sessions",
                    "model",
                    "model TEXT",
                )
            if current_version < 7:
                _ensure_column(
                    conn,
                    "sessions",
                    "token_budget",
                    "token_budget INTEGER DEFAULT NULL",
                )
                _ensure_column(
                    conn,
                    "sessions",
                    "auto_terminate_at_budget",
                    "auto_terminate_at_budget INTEGER NOT NULL DEFAULT 0",
                )
                _ensure_column(
                    conn,
                    "sessions",
                    "budget_exhausted",
                    "budget_exhausted INTEGER NOT NULL DEFAULT 0",
                )
            if current_version < 8:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS permission_policies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        rules_json TEXT NOT NULL,
                        org_id TEXT,
                        team_id TEXT,
                        priority INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );
                """)
            if current_version < 9:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS permission_decisions (
                        id TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        tool_pattern TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        scope TEXT NOT NULL DEFAULT 'session',
                        session_id TEXT,
                        persona_id TEXT,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        expires_at TEXT,
                        reason TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_perm_dec_user
                        ON permission_decisions(user_id);
                    CREATE INDEX IF NOT EXISTS idx_perm_dec_pattern
                        ON permission_decisions(tool_pattern);
                """)
            if current_version < 10:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS webhook_triggers (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        source_type TEXT NOT NULL DEFAULT 'generic',
                        secret_encrypted TEXT NOT NULL,
                        owner_user_id INTEGER NOT NULL,
                        agent_config_json TEXT NOT NULL DEFAULT '{}',
                        prompt_template TEXT NOT NULL DEFAULT '',
                        enabled INTEGER NOT NULL DEFAULT 1,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );
                    CREATE INDEX IF NOT EXISTS idx_webhook_triggers_owner
                        ON webhook_triggers(owner_user_id);
                """)
            if current_version < 11:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS config_templates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        scope TEXT NOT NULL DEFAULT 'system',
                        scope_id TEXT,
                        base_template_id INTEGER,
                        schema_version TEXT NOT NULL DEFAULT '1',
                        config_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );
                    CREATE INDEX IF NOT EXISTS idx_config_templates_scope
                        ON config_templates(scope, scope_id);
                    CREATE INDEX IF NOT EXISTS idx_config_templates_name
                        ON config_templates(name);
                """)
            if current_version < 12:
                _ensure_column(
                    conn,
                    "permission_policies",
                    "conditions_json",
                    "conditions_json TEXT",
                )
                _ensure_column(
                    conn,
                    "sessions",
                    "ancestry_chain_json",
                    "ancestry_chain_json TEXT",
                )
            if current_version < 13:
                _ensure_config_template_unique_index(conn)
            conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            conn.commit()
            self._initialized = True
            logger.debug("ACP Sessions DB schema initialized at {}", self._db_path)

    # ------------------------------------------------------------------
    # Row conversion
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row to a plain dict with deserialized fields."""
        d: dict[str, Any] = dict(row)
        for field in _BOOL_FIELDS:
            if field in d:
                d[field] = bool(d[field])
        for field in _JSON_LIST_FIELDS:
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        for field in _JSON_OBJECT_FIELDS:
            if field in d and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = None
        return d

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def register_session(
        self,
        session_id: str,
        user_id: int,
        agent_type: str = "custom",
        name: str = "",
        cwd: str = "",
        tags: list[str] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        policy_snapshot_version: str | None = None,
        policy_snapshot_fingerprint: str | None = None,
        policy_snapshot_refreshed_at: str | None = None,
        policy_summary: dict[str, Any] | None = None,
        policy_provenance_summary: dict[str, Any] | None = None,
        policy_refresh_error: str | None = None,
        forked_from: str | None = None,
        needs_bootstrap: bool = False,
        model: str | None = None,
        token_budget: int | None = None,
        auto_terminate_at_budget: bool = False,
    ) -> dict[str, Any]:
        """Insert a new session record and return it as a dict."""
        conn = self._get_conn()
        now = _utcnow_iso()
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, user_id, agent_type, name, status, cwd,
                created_at, last_activity_at,
                tags, mcp_servers,
                persona_id, workspace_id, workspace_group_id, scope_snapshot_id,
                policy_snapshot_version, policy_snapshot_fingerprint, policy_snapshot_refreshed_at,
                policy_summary, policy_provenance_summary, policy_refresh_error,
                forked_from, needs_bootstrap, model,
                token_budget, auto_terminate_at_budget
            ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id, user_id, agent_type, name, cwd,
                now, now,
                json.dumps(tags or []),
                json.dumps(mcp_servers or []),
                persona_id, workspace_id, workspace_group_id, scope_snapshot_id,
                policy_snapshot_version,
                policy_snapshot_fingerprint,
                policy_snapshot_refreshed_at,
                json.dumps(policy_summary) if policy_summary is not None else None,
                json.dumps(policy_provenance_summary)
                if policy_provenance_summary is not None
                else None,
                policy_refresh_error,
                forked_from, int(needs_bootstrap),
                model,
                token_budget,
                int(auto_terminate_at_budget),
            ),
        )
        conn.commit()
        return self.get_session(session_id)  # type: ignore[return-value]

    def update_policy_snapshot_state(
        self,
        session_id: str,
        *,
        policy_snapshot_version: str | None,
        policy_snapshot_fingerprint: str | None,
        policy_snapshot_refreshed_at: str | None,
        policy_summary: dict[str, Any] | None,
        policy_provenance_summary: dict[str, Any] | None,
        policy_refresh_error: str | None,
    ) -> None:
        """Update persisted ACP policy snapshot metadata for a session."""
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE sessions
            SET policy_snapshot_version = ?,
                policy_snapshot_fingerprint = ?,
                policy_snapshot_refreshed_at = ?,
                policy_summary = ?,
                policy_provenance_summary = ?,
                policy_refresh_error = ?,
                last_activity_at = ?
            WHERE session_id = ?
            """,
            (
                policy_snapshot_version,
                policy_snapshot_fingerprint,
                policy_snapshot_refreshed_at,
                json.dumps(policy_summary) if policy_summary is not None else None,
                json.dumps(policy_provenance_summary)
                if policy_provenance_summary is not None
                else None,
                policy_refresh_error,
                _utcnow_iso(),
                session_id,
            ),
        )
        conn.commit()

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch a single session by ID, or None if not found."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def close_session(self, session_id: str) -> None:
        """Mark a session as closed."""
        self.set_session_status(session_id, "closed")

    def set_session_status(self, session_id: str, status: str) -> None:
        """Update the status of a session."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET status = ?, last_activity_at = ? WHERE session_id = ?",
            (status, _utcnow_iso(), session_id),
        )
        conn.commit()

    def update_activity(self, session_id: str) -> None:
        """Touch the last_activity_at timestamp."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET last_activity_at = ? WHERE session_id = ?",
            (_utcnow_iso(), session_id),
        )
        conn.commit()

    def set_bootstrap_ready(self, session_id: str, ready: bool) -> None:
        """Set the bootstrap_ready flag for a session."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET bootstrap_ready = ? WHERE session_id = ?",
            (int(ready), session_id),
        )
        conn.commit()

    def clear_needs_bootstrap(self, session_id: str) -> None:
        """Clear the needs_bootstrap flag and update activity timestamp."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE sessions SET needs_bootstrap = 0, last_activity_at = ? WHERE session_id = ?",
            (_utcnow_iso(), session_id),
        )
        conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if a row was actually removed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def list_sessions(
        self,
        *,
        user_id: int | None = None,
        status: str | None = None,
        agent_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List sessions with optional filters. Returns (rows, total_count)."""
        conn = self._get_conn()
        params: list[Any] = []

        query_key = (user_id is not None, status is not None, agent_type is not None)
        match query_key:
            case (False, False, False):
                count_query = "SELECT COUNT(*) FROM sessions"
                rows_query = "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ? OFFSET ?"
            case (True, False, False):
                count_query = "SELECT COUNT(*) FROM sessions WHERE user_id = ?"
                rows_query = (
                    "SELECT * FROM sessions WHERE user_id = ? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?"
                )
                params.append(user_id)
            case (False, True, False):
                count_query = "SELECT COUNT(*) FROM sessions WHERE status = ?"
                rows_query = (
                    "SELECT * FROM sessions WHERE status = ? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?"
                )
                params.append(status)
            case (False, False, True):
                count_query = "SELECT COUNT(*) FROM sessions WHERE agent_type = ?"
                rows_query = (
                    "SELECT * FROM sessions WHERE agent_type = ? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?"
                )
                params.append(agent_type)
            case (True, True, False):
                count_query = "SELECT COUNT(*) FROM sessions WHERE user_id = ? AND status = ?"
                rows_query = (
                    "SELECT * FROM sessions WHERE user_id = ? AND status = ? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?"
                )
                params.extend([user_id, status])
            case (True, False, True):
                count_query = "SELECT COUNT(*) FROM sessions WHERE user_id = ? AND agent_type = ?"
                rows_query = (
                    "SELECT * FROM sessions WHERE user_id = ? AND agent_type = ? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?"
                )
                params.extend([user_id, agent_type])
            case (False, True, True):
                count_query = "SELECT COUNT(*) FROM sessions WHERE status = ? AND agent_type = ?"
                rows_query = (
                    "SELECT * FROM sessions WHERE status = ? AND agent_type = ? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?"
                )
                params.extend([status, agent_type])
            case _:
                count_query = (
                    "SELECT COUNT(*) FROM sessions "
                    "WHERE user_id = ? AND status = ? AND agent_type = ?"
                )
                rows_query = (
                    "SELECT * FROM sessions "
                    "WHERE user_id = ? AND status = ? AND agent_type = ? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?"
                )
                params.extend([user_id, status, agent_type])

        count_row = conn.execute(count_query, params).fetchone()
        total = count_row[0] if count_row else 0

        rows = conn.execute(
            rows_query,
            params + [limit, offset],
        ).fetchall()

        return [self._row_to_dict(r) for r in rows], total

    def aggregate_metrics_by_agent(self) -> list[dict[str, Any]]:
        """Aggregate session metrics grouped by agent_type.

        Returns a list of dicts sorted by total_tokens descending, each with:
        agent_type, session_count, active_sessions, total_prompt_tokens,
        total_completion_tokens, total_tokens, total_messages, last_used_at.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT
                agent_type,
                COUNT(*)                                       AS session_count,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_sessions,
                COALESCE(SUM(prompt_tokens), 0)                AS total_prompt_tokens,
                COALESCE(SUM(completion_tokens), 0)            AS total_completion_tokens,
                COALESCE(SUM(total_tokens), 0)                 AS total_tokens,
                COALESCE(SUM(message_count), 0)                AS total_messages,
                MAX(COALESCE(last_activity_at, created_at))    AS last_used_at
            FROM sessions
            GROUP BY agent_type
            ORDER BY total_tokens DESC
            """
        ).fetchall()
        return [
            {
                "agent_type": r["agent_type"],
                "session_count": r["session_count"],
                "active_sessions": r["active_sessions"],
                "total_prompt_tokens": r["total_prompt_tokens"],
                "total_completion_tokens": r["total_completion_tokens"],
                "total_tokens": r["total_tokens"],
                "total_messages": r["total_messages"],
                "last_used_at": r["last_used_at"],
            }
            for r in rows
        ]

    def get_session_cost_data(self) -> list[dict[str, Any]]:
        """Return per-session (agent_type, model, prompt_tokens, completion_tokens).

        Used by the service layer to compute estimated costs using the
        pricing catalog (which lives outside the DB module).
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT agent_type, model, prompt_tokens, completion_tokens FROM sessions"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_usage_stats(self, *, since_iso: str) -> list[dict[str, Any]]:
        """Aggregate per-agent token usage for sessions created on or after *since_iso*."""
        conn = self._get_conn()
        query = (
            "SELECT "
            "  agent_type, "
            "  COUNT(*) AS invocation_count, "
            "  COALESCE(SUM(total_tokens), 0) AS total_tokens, "
            "  COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
            "  COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
            "  SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count, "
            "  CASE WHEN COUNT(*) > 0 "
            "    THEN CAST(COALESCE(SUM(total_tokens), 0) AS REAL) / COUNT(*) "
            "    ELSE 0 "
            "  END AS avg_tokens_per_session "
            "FROM sessions "
            "WHERE created_at >= ? "
            "GROUP BY agent_type "
            "ORDER BY total_tokens DESC"
        )
        rows = conn.execute(query, [since_iso]).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Run history queries (date-filtered)
    # ------------------------------------------------------------------

    def list_runs(
        self,
        *,
        user_id: int | None = None,
        status: str | None = None,
        agent_type: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Query session records with optional filters including date range.

        Returns ``(rows, total_count)`` where each row is a deserialized dict.
        """
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if agent_type is not None:
            clauses.append("agent_type = ?")
            params.append(agent_type)
        if from_date is not None:
            clauses.append("created_at >= ?")
            params.append(from_date)
        if to_date is not None:
            clauses.append("created_at <= ?")
            params.append(to_date)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        count_row = conn.execute(
            f"SELECT COUNT(*) FROM sessions{where}", params,
        ).fetchone()
        total = count_row[0] if count_row else 0

        rows = conn.execute(
            f"SELECT * FROM sessions{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()

        return [self._row_to_dict(r) for r in rows], total

    def aggregate_runs(
        self,
        *,
        user_id: int | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate token usage and session counts with optional filters.

        Returns a summary dict with total_sessions, prompt_tokens,
        completion_tokens, total_tokens, and per-session cost data for the
        service layer to compute estimated costs.
        """
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if from_date is not None:
            clauses.append("created_at >= ?")
            params.append(from_date)
        if to_date is not None:
            clauses.append("created_at <= ?")
            params.append(to_date)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

        row = conn.execute(
            f"SELECT "
            f"  COUNT(*) AS total_sessions, "
            f"  COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
            f"  COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
            f"  COALESCE(SUM(total_tokens), 0) AS total_tokens "
            f"FROM sessions{where}",
            params,
        ).fetchone()

        result: dict[str, Any] = {
            "total_sessions": row["total_sessions"] if row else 0,
            "prompt_tokens": row["prompt_tokens"] if row else 0,
            "completion_tokens": row["completion_tokens"] if row else 0,
            "total_tokens": row["total_tokens"] if row else 0,
        }

        # Also return per-session cost data so the service layer can compute costs
        cost_rows = conn.execute(
            f"SELECT model, prompt_tokens, completion_tokens FROM sessions{where}",
            params,
        ).fetchall()
        result["_cost_rows"] = [dict(r) for r in cost_rows]

        return result

    # ------------------------------------------------------------------
    # Text normalization (local helper — no external imports)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text_content(value: Any) -> str | None:
        """Extract plain text from various content representations.

        Handles:
        - str: return stripped (or None if empty)
        - list: join text parts from content block lists
        - dict with type in (text, input_text, output_text): return text field
        - dict with content/text keys: recurse
        """
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("type", "")
                    if t in ("text", "input_text", "output_text"):
                        txt = item.get("text", "")
                        if txt:
                            parts.append(str(txt))
                    else:
                        # Try content/text keys
                        for key in ("content", "text"):
                            if key in item:
                                resolved = ACPSessionsDB._normalize_text_content(item[key])
                                if resolved:
                                    parts.append(resolved)
                                break
            return "\n".join(parts) if parts else None
        if isinstance(value, dict):
            d = value
            t = d.get("type", "")
            if t in ("text", "input_text", "output_text"):
                txt = d.get("text", "")
                return str(txt).strip() if txt else None
            for key in ("content", "text", "message", "output", "detail", "value"):
                resolved = ACPSessionsDB._normalize_text_content(d.get(key))
                if resolved:
                    return resolved
            return None
        return None

    # ------------------------------------------------------------------
    # Message recording
    # ------------------------------------------------------------------

    def record_prompt(
        self,
        session_id: str,
        prompt: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Record a prompt+response exchange and update token counters.

        Returns a dict with prompt_tokens, completion_tokens, total_tokens
        for this exchange, or None if the session does not exist.
        """
        conn = self._get_conn()
        session = self.get_session(session_id)
        if session is None:
            return None

        now = _utcnow_iso()

        # Use BEGIN IMMEDIATE to serialize writers, preventing
        # concurrent MAX(message_index) from picking duplicate indices.
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Determine current max message_index for this session
            row = conn.execute(
                "SELECT COALESCE(MAX(message_index), -1) FROM session_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            next_idx: int = row[0] + 1 if row else 0

            inserted = 0
            # Insert user messages from prompt
            for msg in prompt:
                role = msg.get("role", "user")
                content = self._normalize_text_content(msg.get("content")) or ""
                conn.execute(
                    "INSERT INTO session_messages (session_id, message_index, role, content, timestamp, raw_data)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, next_idx, role, content, now, json.dumps(msg)),
                )
                next_idx += 1
                inserted += 1

            # Insert assistant response
            assistant_text = self._normalize_text_content(result.get("content")) or ""
            conn.execute(
                "INSERT INTO session_messages (session_id, message_index, role, content, timestamp, raw_data)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, next_idx, "assistant", assistant_text, now, json.dumps(result)),
            )
            inserted += 1

            # Extract token usage
            usage = result.get("usage") or {}
            p_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            c_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            t_tokens = p_tokens + c_tokens

            # Update session counters
            conn.execute(
                """
                UPDATE sessions SET
                    message_count = message_count + ?,
                    prompt_tokens = prompt_tokens + ?,
                    completion_tokens = completion_tokens + ?,
                    total_tokens = total_tokens + ?,
                    last_activity_at = ?
                WHERE session_id = ?
                """,
                (inserted, p_tokens, c_tokens, t_tokens, now, session_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        return {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": t_tokens,
        }

    def get_messages(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return messages for a session ordered by message_index."""
        conn = self._get_conn()
        sql = (
            "SELECT role, content, timestamp, raw_data FROM session_messages"
            " WHERE session_id = ? ORDER BY message_index"
        )
        params: list[Any] = [session_id]
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params.append(offset)

        rows = conn.execute(sql, params).fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            if d.get("raw_data"):
                try:
                    d["raw_data"] = json.loads(d["raw_data"])
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append(d)
        return results

    def update_token_usage(
        self,
        session_id: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Directly increment token counters for a session."""
        conn = self._get_conn()
        total = prompt_tokens + completion_tokens
        conn.execute(
            """
            UPDATE sessions SET
                prompt_tokens = prompt_tokens + ?,
                completion_tokens = completion_tokens + ?,
                total_tokens = total_tokens + ?,
                last_activity_at = ?
            WHERE session_id = ?
            """,
            (prompt_tokens, completion_tokens, total, _utcnow_iso(), session_id),
        )
        conn.commit()

    def update_session_budget(
        self,
        session_id: str,
        token_budget: int | None,
        auto_terminate_at_budget: bool,
    ) -> bool:
        """Update token budget settings for a session.

        Returns True if the session was found and updated.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            UPDATE sessions
            SET token_budget = ?,
                auto_terminate_at_budget = ?,
                last_activity_at = ?
            WHERE session_id = ?
            """,
            (
                token_budget,
                int(auto_terminate_at_budget),
                _utcnow_iso(),
                session_id,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0

    def check_budget_and_terminate(self, session_id: str) -> bool:
        """Check if a session has exceeded its token budget.

        If auto_terminate_at_budget is enabled and total_tokens >= token_budget,
        marks the session as closed with budget_exhausted = 1.

        Returns True if the session was terminated due to budget exhaustion.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT total_tokens, token_budget, auto_terminate_at_budget, status "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return False
        # Skip if no budget set, auto-terminate not enabled, or already closed
        if row["token_budget"] is None or not row["auto_terminate_at_budget"]:
            return False
        if row["status"] != "active":
            return False
        if row["total_tokens"] >= row["token_budget"]:
            now = _utcnow_iso()
            conn.execute(
                "UPDATE sessions SET status = 'closed', budget_exhausted = 1, "
                "last_activity_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()
            logger.info(
                "ACP session {} auto-terminated: token budget exhausted "
                "({}/{})",
                session_id,
                row["total_tokens"],
                row["token_budget"],
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Fork
    # ------------------------------------------------------------------

    def fork_session(
        self,
        source_session_id: str,
        new_session_id: str,
        message_index: int,
        user_id: int,
        name: str | None = None,
    ) -> dict[str, Any] | None:
        """Fork a session, copying messages up to *message_index* (inclusive).

        Returns the new session dict, or None if the source does not exist
        or is not owned by *user_id*.
        """
        source = self.get_session(source_session_id)
        if source is None or source["user_id"] != user_id:
            return None

        # Create new session copying key fields from source
        self.register_session(
            session_id=new_session_id,
            user_id=user_id,
            agent_type=source.get("agent_type", "custom"),
            name=name or source.get("name", ""),
            cwd=source.get("cwd", ""),
            tags=source.get("tags"),
            mcp_servers=source.get("mcp_servers"),
            persona_id=source.get("persona_id"),
            workspace_id=source.get("workspace_id"),
            workspace_group_id=source.get("workspace_group_id"),
            scope_snapshot_id=source.get("scope_snapshot_id"),
            forked_from=source_session_id,
            needs_bootstrap=True,
            model=source.get("model"),
        )

        # Copy messages from source up to message_index (inclusive)
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT message_index, role, content, timestamp, raw_data"
            " FROM session_messages WHERE session_id = ? AND message_index <= ?"
            " ORDER BY message_index",
            (source_session_id, message_index),
        ).fetchall()

        for r in rows:
            conn.execute(
                "INSERT INTO session_messages (session_id, message_index, role, content, timestamp, raw_data)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (new_session_id, r["message_index"], r["role"], r["content"],
                 r["timestamp"], r["raw_data"]),
            )

        # Update message_count on the new session
        if rows:
            conn.execute(
                "UPDATE sessions SET message_count = ? WHERE session_id = ?",
                (len(rows), new_session_id),
            )
            conn.commit()

        return self.get_session(new_session_id)

    def get_fork_lineage(
        self,
        session_id: str,
        *,
        max_depth: int = 50,
    ) -> list[str]:
        """Walk the forked_from chain and return ancestor IDs (oldest first)."""
        conn = self._get_conn()
        ancestors: list[str] = []
        seen: set[str] = {session_id}
        current = session_id

        for _ in range(max_depth):
            row = conn.execute(
                "SELECT forked_from FROM sessions WHERE session_id = ?",
                (current,),
            ).fetchone()
            if row is None or row["forked_from"] is None:
                break
            parent = row["forked_from"]
            if parent in seen:
                break  # cycle guard
            seen.add(parent)
            ancestors.append(parent)
            current = parent

        ancestors.reverse()
        return ancestors

    # ------------------------------------------------------------------
    # Quota configuration
    # ------------------------------------------------------------------

    def configure_quotas(
        self,
        *,
        max_concurrent_per_user: int = 5,
        max_tokens_per_session: int = 1_000_000,
        session_ttl_seconds: int = 86400,
        max_session_duration_seconds: int = 14400,
    ) -> None:
        """Store quota limits as instance attributes (from server config)."""
        self._max_concurrent_per_user = max_concurrent_per_user
        self._max_tokens_per_session = max_tokens_per_session
        self._session_ttl_seconds = session_ttl_seconds
        self._max_session_duration_seconds = max_session_duration_seconds

    def check_session_quota(self, user_id: int) -> dict[str, Any] | None:
        """Check if user has reached the concurrent session limit.

        Returns an error dict if quota exceeded, None otherwise.
        """
        limit = getattr(self, "_max_concurrent_per_user", 5)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE user_id = ? AND status = 'active'",
            (user_id,),
        ).fetchone()
        count = row[0] if row else 0
        if count >= limit:
            return {
                "code": "quota_exceeded",
                "message": f"Max concurrent sessions ({limit}) exceeded",
                "current": count,
                "limit": limit,
            }
        return None

    def check_token_quota(self, session_id: str) -> dict[str, Any] | None:
        """Check if session has exceeded the token limit.

        Returns an error dict if quota exceeded, None if under limit
        or session not found.
        """
        limit = getattr(self, "_max_tokens_per_session", 1_000_000)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT total_tokens FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        if row[0] >= limit:
            return {
                "code": "token_quota_exceeded",
                "message": f"Session token limit ({limit}) exceeded",
                "current": row[0],
                "limit": limit,
            }
        return None

    def get_quota_status(
        self, user_id: int, session_id: str | None = None
    ) -> dict[str, Any]:
        """Return current quota usage stats."""
        conn = self._get_conn()
        concurrent_limit = getattr(self, "_max_concurrent_per_user", 5)
        row = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE user_id = ? AND status = 'active'",
            (user_id,),
        ).fetchone()
        active_count = row[0] if row else 0

        result: dict[str, Any] = {
            "concurrent_sessions": {
                "current": active_count,
                "limit": concurrent_limit,
            },
        }

        if session_id is not None:
            token_limit = getattr(self, "_max_tokens_per_session", 1_000_000)
            sess_row = conn.execute(
                "SELECT total_tokens FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            total = sess_row[0] if sess_row else 0
            result["session_tokens"] = {
                "current": total,
                "limit": token_limit,
            }

        return result

    def evict_expired_sessions(self) -> int:
        """Close active sessions that have exceeded TTL (since last activity)
        or max duration (since creation).

        Returns the number of sessions evicted.
        """
        ttl = getattr(self, "_session_ttl_seconds", 86400)
        max_dur = getattr(self, "_max_session_duration_seconds", 14400)

        conn = self._get_conn()
        now = datetime.now(timezone.utc)

        # Fetch active sessions and check expiry in Python
        rows = conn.execute(
            "SELECT session_id, created_at, last_activity_at FROM sessions WHERE status = 'active'"
        ).fetchall()

        expired_ids: list[str] = []
        for r in rows:
            try:
                created = datetime.fromisoformat(r["created_at"])
            except (ValueError, TypeError):
                continue
            # Check max duration (time since creation)
            age = (now - created).total_seconds()
            if age >= max_dur:
                expired_ids.append(r["session_id"])
                continue
            # Check TTL (time since last activity, fall back to creation)
            activity_ts = r["last_activity_at"] or r["created_at"]
            try:
                last_active = datetime.fromisoformat(activity_ts)
            except (ValueError, TypeError):
                last_active = created
            idle = (now - last_active).total_seconds()
            if idle >= ttl:
                expired_ids.append(r["session_id"])

        if not expired_ids:
            return 0

        now_iso = _utcnow_iso()
        for sid in expired_ids:
            conn.execute(
                "UPDATE sessions SET status = 'closed', last_activity_at = ? WHERE session_id = ?",
                (now_iso, sid),
            )
        conn.commit()
        logger.info("Evicted {} expired ACP sessions", len(expired_ids))
        return len(expired_ids)

    # ------------------------------------------------------------------
    # Agent Registry CRUD
    # ------------------------------------------------------------------

    def save_agent_entry(self, entry_dict: dict[str, Any]) -> dict[str, Any]:
        """Insert or replace an agent registry entry. Returns the saved entry."""
        conn = self._get_conn()
        now = _utcnow_iso()
        agent_type = entry_dict.get("agent_type", "")
        name = entry_dict.get("name", "")
        description = entry_dict.get("description", "")
        command = entry_dict.get("command", "")
        args = entry_dict.get("args", "[]")
        env = entry_dict.get("env", "{}")
        requires_api_key = entry_dict.get("requires_api_key")
        is_default = int(entry_dict.get("is_default", 0))
        install_instructions = entry_dict.get("install_instructions", "[]")
        docs_url = entry_dict.get("docs_url")
        mcp_orchestration = entry_dict.get("mcp_orchestration", "agent_driven")
        mcp_entry_tool = entry_dict.get("mcp_entry_tool", "execute")
        mcp_structured_response = int(bool(entry_dict.get("mcp_structured_response", 0)))
        mcp_llm_provider = entry_dict.get("mcp_llm_provider")
        mcp_llm_model = entry_dict.get("mcp_llm_model")
        mcp_max_iterations = int(entry_dict.get("mcp_max_iterations", 20))
        mcp_refresh_tools = int(bool(entry_dict.get("mcp_refresh_tools", 0)))
        source = entry_dict.get("source", "api")

        conn.execute(
            """
            INSERT INTO agent_registry (
                agent_type, name, description, command, args, env,
                requires_api_key, is_default, install_instructions, docs_url,
                mcp_orchestration, mcp_entry_tool, mcp_structured_response,
                mcp_llm_provider, mcp_llm_model, mcp_max_iterations, mcp_refresh_tools,
                source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_type) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                command = excluded.command,
                args = excluded.args,
                env = excluded.env,
                requires_api_key = excluded.requires_api_key,
                is_default = excluded.is_default,
                install_instructions = excluded.install_instructions,
                docs_url = excluded.docs_url,
                mcp_orchestration = excluded.mcp_orchestration,
                mcp_entry_tool = excluded.mcp_entry_tool,
                mcp_structured_response = excluded.mcp_structured_response,
                mcp_llm_provider = excluded.mcp_llm_provider,
                mcp_llm_model = excluded.mcp_llm_model,
                mcp_max_iterations = excluded.mcp_max_iterations,
                mcp_refresh_tools = excluded.mcp_refresh_tools,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                agent_type, name, description, command, args, env,
                requires_api_key, is_default, install_instructions, docs_url,
                mcp_orchestration, mcp_entry_tool, mcp_structured_response,
                mcp_llm_provider, mcp_llm_model, mcp_max_iterations, mcp_refresh_tools,
                source, now, now,
            ),
        )
        conn.commit()
        return self.get_agent_entry(agent_type)  # type: ignore[return-value]

    def delete_agent_entry(self, agent_type: str) -> bool:
        """Delete an agent entry. Returns True if a row was removed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM agent_registry WHERE agent_type = ?", (agent_type,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def list_agent_entries(self, source: str | None = None) -> list[dict[str, Any]]:
        """List agent entries, optionally filtered by source."""
        conn = self._get_conn()
        if source is not None:
            rows = conn.execute(
                "SELECT * FROM agent_registry WHERE source = ? ORDER BY name",
                (source,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM agent_registry ORDER BY name"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_entry(self, agent_type: str) -> dict[str, Any] | None:
        """Fetch a single agent entry by type, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM agent_registry WHERE agent_type = ?", (agent_type,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Health History
    # ------------------------------------------------------------------

    def record_health_check(
        self,
        agent_type: str,
        health: str,
        consecutive_failures: int = 0,
        details: str = "{}",
    ) -> None:
        """Record a health check result."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO agent_health_history
               (agent_type, health, consecutive_failures, details, checked_at)
               VALUES (?, ?, ?, ?, ?)""",
            (agent_type, health, consecutive_failures, details, _utcnow_iso()),
        )
        conn.commit()

    def get_health_history(
        self,
        agent_type: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get recent health check history for an agent."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT * FROM agent_health_history
               WHERE agent_type = ? ORDER BY checked_at DESC LIMIT ?""",
            (agent_type, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Permission Policy CRUD
    # ------------------------------------------------------------------

    def create_permission_policy(
        self,
        name: str,
        rules_json: str,
        priority: int = 0,
        description: str = "",
        org_id: str | None = None,
        team_id: str | None = None,
    ) -> int:
        """Insert a permission policy. Returns the new row ID."""
        conn = self._get_conn()
        now = _utcnow_iso()
        cursor = conn.execute(
            """
            INSERT INTO permission_policies
                (name, description, rules_json, org_id, team_id, priority,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, description, rules_json, org_id, team_id, priority, now, now),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_permission_policy(self, policy_id: int) -> dict[str, Any] | None:
        """Fetch a single policy by ID, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM permission_policies WHERE id = ?", (policy_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_permission_policies(self) -> list[dict[str, Any]]:
        """List all policies ordered by priority DESC, then name ASC."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM permission_policies ORDER BY priority DESC, name ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_permission_policy(self, policy_id: int, **kwargs: Any) -> bool:
        """Update fields on a permission policy.

        Accepted kwargs: name, description, rules_json, org_id, team_id, priority.
        Returns True if the row was found and updated.
        """
        allowed = {"name", "description", "rules_json", "org_id", "team_id", "priority"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = _utcnow_iso()
        set_clause = ", ".join(f"{col} = ?" for col in updates)
        values = list(updates.values()) + [policy_id]
        conn = self._get_conn()
        cursor = conn.execute(
            f"UPDATE permission_policies SET {set_clause} WHERE id = ?",
            values,
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_permission_policy(self, policy_id: int) -> bool:
        """Delete a policy by ID. Returns True if a row was removed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM permission_policies WHERE id = ?", (policy_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def resolve_permission_tier(self, tool_name: str) -> str | None:
        """Query all policies, match tool_name against rules using fnmatch.

        Returns the tier from the highest-priority matching rule, or None
        if no rule matches.
        """
        policies = self.list_permission_policies()
        best_priority = -1
        best_tier: str | None = None
        for pol in policies:
            priority = pol.get("priority", 0)
            raw = pol.get("rules_json", "[]")
            try:
                rules = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            for rule in rules:
                pattern = rule.get("tool_pattern", "")
                tier = rule.get("tier", "")
                if _fnmatch.fnmatch(tool_name.lower(), pattern.lower()):
                    if priority > best_priority:
                        best_priority = priority
                        best_tier = tier
        return best_tier

    # ------------------------------------------------------------------
    # Permission Decision CRUD
    # ------------------------------------------------------------------

    def insert_permission_decision(
        self,
        id: str,
        user_id: int,
        tool_pattern: str,
        decision: str,
        scope: str = "session",
        session_id: str | None = None,
        persona_id: str | None = None,
        reason: str | None = None,
        expires_at: str | None = None,
    ) -> None:
        """Insert a persisted permission decision."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO permission_decisions
                (id, user_id, tool_pattern, decision, scope,
                 session_id, persona_id, created_at, expires_at, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id, user_id, tool_pattern, decision, scope,
                session_id, persona_id, _utcnow_iso(), expires_at, reason,
            ),
        )
        conn.commit()

    def list_permission_decisions(
        self,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """List permission decisions, optionally filtered by user_id."""
        conn = self._get_conn()
        if user_id is not None:
            rows = conn.execute(
                "SELECT * FROM permission_decisions WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM permission_decisions ORDER BY created_at DESC"
            ).fetchall()
        now_iso = _utcnow_iso()
        results: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            # Skip expired decisions
            if d.get("expires_at") and d["expires_at"] < now_iso:
                continue
            results.append(d)
        return results

    def check_permission_decision(
        self,
        user_id: int,
        tool_name: str,
        session_id: str | None = None,
    ) -> str | None:
        """Find a matching persisted decision for user_id + tool_name.

        Returns ``'allow'`` or ``'deny'`` if a matching non-expired decision
        exists, otherwise ``None``.  Global scope is checked first, then
        session scope (if *session_id* is provided).
        """
        decisions = self.list_permission_decisions(user_id=user_id)
        for d in decisions:
            if not _fnmatch.fnmatch(tool_name, d["tool_pattern"]):
                continue
            if d["scope"] == "global":
                return d["decision"]
            if d["scope"] == "session" and d.get("session_id") == session_id:
                return d["decision"]
        return None

    def delete_permission_decision(self, decision_id: str) -> bool:
        """Delete a permission decision by ID. Returns True if a row was removed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM permission_decisions WHERE id = ?", (decision_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def get_permission_decision(self, decision_id: str) -> dict[str, Any] | None:
        """Fetch a single permission decision by ID, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM permission_decisions WHERE id = ?", (decision_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # Webhook trigger CRUD
    # ------------------------------------------------------------------

    def create_webhook_trigger(
        self,
        trigger_id: str,
        name: str,
        source_type: str,
        secret_encrypted: str,
        owner_user_id: int,
        agent_config_json: str = "{}",
        prompt_template: str = "",
        enabled: bool = True,
    ) -> None:
        """Insert a new webhook trigger row."""
        conn = self._get_conn()
        now = _utcnow_iso()
        conn.execute(
            """
            INSERT INTO webhook_triggers
                (id, name, source_type, secret_encrypted, owner_user_id,
                 agent_config_json, prompt_template, enabled,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trigger_id,
                name,
                source_type,
                secret_encrypted,
                owner_user_id,
                agent_config_json,
                prompt_template,
                1 if enabled else 0,
                now,
                now,
            ),
        )
        conn.commit()

    def list_webhook_triggers(self, owner_user_id: int) -> list[dict[str, Any]]:
        """Return all webhook triggers owned by *owner_user_id*."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM webhook_triggers WHERE owner_user_id = ? ORDER BY created_at DESC",
            (owner_user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_webhook_trigger(self, trigger_id: str) -> dict[str, Any] | None:
        """Return a single webhook trigger by id, or ``None``."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM webhook_triggers WHERE id = ?", (trigger_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    _WEBHOOK_TRIGGER_UPDATABLE_COLS = frozenset({
        "name", "source_type", "secret_encrypted", "agent_config_json",
        "prompt_template", "enabled", "updated_at",
    })

    def update_webhook_trigger(self, trigger_id: str, updates: dict[str, Any]) -> bool:
        """Update fields of a webhook trigger. Returns True if a row was modified."""
        if not updates:
            return False
        for k in updates:
            if k not in self._WEBHOOK_TRIGGER_UPDATABLE_COLS:
                raise ValueError(f"Cannot update column: {k!r}")
        conn = self._get_conn()
        updates["updated_at"] = _utcnow_iso()
        set_clauses = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [trigger_id]
        cursor = conn.execute(
            f"UPDATE webhook_triggers SET {set_clauses} WHERE id = ?",
            values,
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_webhook_trigger(self, trigger_id: str) -> bool:
        """Delete a webhook trigger. Returns True if a row was removed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM webhook_triggers WHERE id = ?", (trigger_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Config Template CRUD
    # ------------------------------------------------------------------

    def create_config_template(
        self,
        name: str,
        description: str = "",
        scope: str = "system",
        scope_id: str | None = None,
        base_template_id: int | None = None,
        schema_version: str = "1",
        config_json: str = "{}",
    ) -> int:
        """Insert a new config template. Returns the new row ID."""
        conn = self._get_conn()
        now = _utcnow_iso()
        try:
            cursor = conn.execute(
                """
                INSERT INTO config_templates
                    (name, description, scope, scope_id, base_template_id,
                     schema_version, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name, description, scope, scope_id, base_template_id,
                    schema_version, config_json, now, now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Config template already exists for this scope") from exc
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def get_config_template(self, template_id: int) -> dict[str, Any] | None:
        """Fetch a single config template by ID, or None."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM config_templates WHERE id = ?", (template_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_config_templates(
        self,
        *,
        scope: str | None = None,
        scope_id: str | None = None,
        name: str | None = None,
    ) -> list[dict[str, Any]]:
        """List config templates with optional filters.

        Filters are AND-combined when provided.
        """
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(scope_id)
        if name is not None:
            clauses.append("name = ?")
            params.append(name)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM config_templates{where} ORDER BY id",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    _CONFIG_TEMPLATE_UPDATABLE_COLS = frozenset({
        "name", "description", "scope", "scope_id", "base_template_id",
        "schema_version", "config_json",
    })

    def update_config_template(self, template_id: int, **kwargs: Any) -> bool:
        """Update fields on a config template.

        Returns True if the row was found and updated.
        """
        updates = {k: v for k, v in kwargs.items() if k in self._CONFIG_TEMPLATE_UPDATABLE_COLS}
        if not updates:
            return False
        updates["updated_at"] = _utcnow_iso()
        set_clause = ", ".join(f"{col} = ?" for col in updates)
        values = list(updates.values()) + [template_id]
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                f"UPDATE config_templates SET {set_clause} WHERE id = ?",  # nosec B608
                values,
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Config template already exists for this scope") from exc
        conn.commit()
        return cursor.rowcount > 0

    def delete_config_template(self, template_id: int) -> bool:
        """Delete a config template by ID. Returns True if a row was removed."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM config_templates WHERE id = ?", (template_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the thread-local connection."""
        conn: sqlite3.Connection | None = getattr(self._conn_local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._conn_local.conn = None

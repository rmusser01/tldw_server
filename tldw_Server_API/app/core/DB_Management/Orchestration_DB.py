"""SQLite-backed persistence for Agent Orchestration (per-user).

Stores projects, tasks, runs, reviews, workspaces, and workspace MCP
servers with full state-machine enforcement and cycle-safe dependency
tracking.

Schema versions:
    v1 — original projects/tasks/runs/reviews tables
    v2 — adds acp_workspaces, acp_workspace_mcp_servers tables;
          adds workspace_id FK to projects
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Agent_Orchestration.models import (
    ACPWorkspace,
    AgentProject,
    AgentRun,
    AgentTask,
    RunStatus,
    TaskStatus,
    is_valid_transition,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.sqlite_policy import configure_sqlite_connection

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class OrchestrationNotFoundError(ValueError):
    """Raised when a project, task, or run is not found."""


class InvalidTransitionError(ValueError):
    """Raised when a task state transition is not allowed."""


class CanonicalWorkspaceBridgeConflictError(ValueError):
    """Raised when a canonical workspace bridge would create an ambiguous link."""


_SCHEMA_VERSION = 3
CANONICAL_WORKSPACE_ID_METADATA_KEY = "canonical_workspace_id"
CANONICAL_WORKSPACE_SOURCE_METADATA_KEY = "canonical_workspace_source"
CANONICAL_WORKSPACE_LINK_STATUS_METADATA_KEY = "link_status"
CANONICAL_WORKSPACE_LINKED_STATUS = "linked"

# Base schema (v1) — applied to fresh databases
_SCHEMA_V1_SQL = """\
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    user_id INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'todo',
    agent_type TEXT,
    dependency_id INTEGER,
    reviewer_agent_type TEXT,
    max_review_attempts INTEGER NOT NULL DEFAULT 3,
    review_count INTEGER NOT NULL DEFAULT 0,
    success_criteria TEXT NOT NULL DEFAULT '',
    user_id INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (dependency_id) REFERENCES tasks(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_dependency ON tasks(dependency_id);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    session_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    agent_type TEXT,
    started_at TEXT,
    completed_at TEXT,
    result_summary TEXT NOT NULL DEFAULT '',
    error TEXT,
    token_usage TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);

CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    approved INTEGER NOT NULL,
    feedback TEXT NOT NULL DEFAULT '',
    reviewer TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_reviews_task ON reviews(task_id);
"""

# v2 additions — new tables + ALTER on projects
_SCHEMA_V2_SQL = """\
CREATE TABLE IF NOT EXISTS acp_workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    root_path TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    workspace_type TEXT NOT NULL DEFAULT 'manual',
    parent_workspace_id INTEGER REFERENCES acp_workspaces(id) ON DELETE SET NULL,
    env_vars TEXT NOT NULL DEFAULT '{}',
    git_remote_url TEXT,
    git_default_branch TEXT,
    git_current_branch TEXT,
    git_is_dirty INTEGER,
    last_health_check TEXT,
    health_status TEXT NOT NULL DEFAULT 'unknown',
    user_id INTEGER NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_acp_workspaces_user ON acp_workspaces(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_acp_workspaces_name ON acp_workspaces(user_id, name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_acp_workspaces_root ON acp_workspaces(user_id, root_path);
CREATE INDEX IF NOT EXISTS idx_acp_workspaces_parent ON acp_workspaces(parent_workspace_id);

CREATE TABLE IF NOT EXISTS acp_workspace_mcp_servers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES acp_workspaces(id) ON DELETE CASCADE,
    server_name TEXT NOT NULL,
    server_type TEXT NOT NULL DEFAULT 'stdio',
    command TEXT,
    args TEXT NOT NULL DEFAULT '[]',
    env TEXT NOT NULL DEFAULT '{}',
    url TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    UNIQUE(workspace_id, server_name)
);
CREATE INDEX IF NOT EXISTS idx_ws_mcp_workspace ON acp_workspace_mcp_servers(workspace_id);
"""

_SCHEMA_V3_SQL = """\
CREATE UNIQUE INDEX IF NOT EXISTS idx_acp_workspaces_user_canonical
ON acp_workspaces(user_id, canonical_workspace_id)
WHERE canonical_workspace_id IS NOT NULL;
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _canonical_workspace_id_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Return a normalized canonical workspace ID from workspace metadata."""
    if not isinstance(metadata, dict):
        return None
    canonical_id = str(metadata.get(CANONICAL_WORKSPACE_ID_METADATA_KEY) or "").strip()
    return canonical_id or None


def _parse_json_list(raw: str | None) -> list[Any]:
    if not raw:
        return []
    try:
        result = json.loads(raw)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


_UNSET = object()
"""Sentinel for 'no value provided' (distinct from None)."""

_SAFE_TABLE_NAMES = frozenset({"projects", "tasks", "runs", "reviews", "acp_workspaces", "acp_workspace_mcp_servers"})


def _col_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table.

    Only accepts known table names to prevent SQL injection via PRAGMA.
    """
    if table not in _SAFE_TABLE_NAMES:
        raise ValueError(f"Unknown table name: {table}")
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


class OrchestrationDB:
    """Per-user SQLite store for orchestration projects, tasks, runs, reviews, workspaces."""

    @classmethod
    def for_user(cls, user_id: int) -> OrchestrationDB:
        safe_user_id = int(user_id)
        safe_db_dir = DatabasePaths.get_user_base_directory(safe_user_id)
        return cls(user_id=safe_user_id, db_dir=safe_db_dir, _trusted_db_dir=True)

    def __init__(
        self,
        user_id: int,
        db_dir: str | Path | None = None,
        *,
        _trusted_db_dir: bool = False,
    ) -> None:
        self._user_id = int(user_id)
        self._managed_db_dir = db_dir is None or _trusted_db_dir
        if db_dir is None:
            db_dir_path = DatabasePaths.get_user_base_directory(self._user_id)
        elif _trusted_db_dir:
            db_dir_path = Path(db_dir)
        else:
            db_dir_path = Path(str(db_dir)).expanduser().resolve(strict=False)
        if not self._managed_db_dir and not db_dir_path.exists():
            raise ValueError("Custom OrchestrationDB db_dir must already exist")
        self._db_dir = db_dir_path
        self._db_path = str(self._db_dir / "orchestration.db")
        self._conn_local = threading.local()
        self._initialized = False
        self._init_lock = threading.Lock()
        self._bridge_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._conn_local, "conn", None)
        if conn is None:
            if self._managed_db_dir:
                self._db_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            configure_sqlite_connection(conn)
            self._conn_local.conn = conn
        return conn

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            conn = self._get_conn()
            current_version = conn.execute("PRAGMA user_version").fetchone()[0]

            if current_version < 1:
                # Fresh database — apply v1 base schema
                conn.executescript(_SCHEMA_V1_SQL)
                current_version = 1

            if current_version < 2:
                # Migrate v1 → v2
                self._migrate_v1_to_v2(conn)
                current_version = 2

            if current_version < 3:
                # Migrate v2 → v3
                self._migrate_v2_to_v3(conn)
                current_version = 3

            conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            conn.commit()
            self._initialized = True

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Apply v2 schema additions: workspace tables + projects.workspace_id."""
        logger.info("Orchestration DB: migrating schema v1 → v2")
        # Create new tables
        conn.executescript(_SCHEMA_V2_SQL)
        # Add workspace_id column to projects if not already present
        if not _col_exists(conn, "projects", "workspace_id"):
            conn.execute(
                "ALTER TABLE projects ADD COLUMN workspace_id INTEGER "
                "REFERENCES acp_workspaces(id) ON DELETE SET NULL"
            )
        conn.commit()

    def _migrate_v2_to_v3(self, conn: sqlite3.Connection) -> None:
        """Apply v3 canonical workspace link column and uniqueness index."""
        logger.info("Orchestration DB: migrating schema v2 → v3")
        if not _col_exists(conn, "acp_workspaces", "canonical_workspace_id"):
            conn.execute("ALTER TABLE acp_workspaces ADD COLUMN canonical_workspace_id TEXT")

        rows = conn.execute(
            "SELECT id, user_id, metadata, canonical_workspace_id FROM acp_workspaces"
        ).fetchall()
        seen: dict[tuple[int, str], int] = {}
        duplicates: list[tuple[int, str, int, int]] = []
        updates: list[tuple[str, int]] = []
        for row in rows:
            metadata = _parse_json(row["metadata"])
            canonical_id = str(row["canonical_workspace_id"] or "").strip()
            canonical_id = canonical_id or _canonical_workspace_id_from_metadata(metadata)
            if not canonical_id:
                continue
            key = (int(row["user_id"]), canonical_id)
            if key in seen:
                duplicates.append((key[0], canonical_id, seen[key], int(row["id"])))
                continue
            seen[key] = int(row["id"])
            updates.append((canonical_id, int(row["id"])))

        if duplicates:
            details = "; ".join(
                f"user_id={user_id} canonical_workspace_id={canonical_id} "
                f"workspace_ids={first_id},{duplicate_id}"
                for user_id, canonical_id, first_id, duplicate_id in duplicates
            )
            raise ValueError(
                "Duplicate canonical workspace links detected during orchestration "
                f"schema migration; remove duplicate ACP workspace metadata before retrying: {details}"
            )

        conn.executemany(
            "UPDATE acp_workspaces SET canonical_workspace_id = ? WHERE id = ?",
            updates,
        )
        conn.executescript(_SCHEMA_V3_SQL)
        conn.commit()

    # ------------------------------------------------------------------
    # Row -> dataclass helpers
    # ------------------------------------------------------------------

    def _row_to_workspace(self, row: sqlite3.Row) -> ACPWorkspace:
        return ACPWorkspace(
            id=row["id"],
            name=row["name"],
            root_path=row["root_path"],
            description=row["description"],
            workspace_type=row["workspace_type"],
            parent_workspace_id=row["parent_workspace_id"],
            env_vars=_parse_json(row["env_vars"]),
            git_remote_url=row["git_remote_url"],
            git_default_branch=row["git_default_branch"],
            git_current_branch=row["git_current_branch"],
            git_is_dirty=bool(row["git_is_dirty"]) if row["git_is_dirty"] is not None else None,
            last_health_check=row["last_health_check"],
            health_status=row["health_status"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_parse_json(row["metadata"]),
        )

    def _row_to_project(self, row: sqlite3.Row) -> AgentProject:
        columns = set(row.keys())
        return AgentProject(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            workspace_id=row["workspace_id"] if "workspace_id" in columns else None,
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_parse_json(row["metadata"]),
        )

    def _row_to_task(self, row: sqlite3.Row) -> AgentTask:
        return AgentTask(
            id=row["id"],
            project_id=row["project_id"],
            title=row["title"],
            description=row["description"],
            status=TaskStatus(row["status"]),
            agent_type=row["agent_type"],
            dependency_id=row["dependency_id"],
            reviewer_agent_type=row["reviewer_agent_type"],
            max_review_attempts=row["max_review_attempts"],
            review_count=row["review_count"],
            success_criteria=row["success_criteria"],
            user_id=row["user_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_parse_json(row["metadata"]),
        )

    def _row_to_run(self, row: sqlite3.Row) -> AgentRun:
        return AgentRun(
            id=row["id"],
            task_id=row["task_id"],
            session_id=row["session_id"],
            status=RunStatus(row["status"]),
            agent_type=row["agent_type"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            result_summary=row["result_summary"],
            error=row["error"],
            token_usage=_parse_json(row["token_usage"]),
        )

    def _row_to_mcp_server(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "server_name": row["server_name"],
            "server_type": row["server_type"],
            "command": row["command"],
            "args": _parse_json_list(row["args"]),
            "env": _parse_json(row["env"]),
            "url": row["url"],
            "enabled": bool(row["enabled"]),
        }

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    def create_workspace(
        self,
        name: str,
        root_path: str,
        description: str = "",
        workspace_type: str = "manual",
        parent_workspace_id: int | None = None,
        env_vars: dict[str, str] | None = None,
        git_remote_url: str | None = None,
        git_default_branch: str | None = None,
        git_current_branch: str | None = None,
        git_is_dirty: bool | None = None,
        health_status: str = "unknown",
        metadata: dict[str, Any] | None = None,
    ) -> ACPWorkspace:
        self._ensure_schema()
        conn = self._get_conn()
        metadata_payload = dict(metadata or {})
        canonical_workspace_id = _canonical_workspace_id_from_metadata(metadata_payload)

        # Validate parent exists if provided
        if parent_workspace_id is not None:
            if conn.execute(
                "SELECT 1 FROM acp_workspaces WHERE id = ?", (parent_workspace_id,)
            ).fetchone() is None:
                raise OrchestrationNotFoundError(
                    f"Parent workspace {parent_workspace_id} not found"
                )

        now = _now_iso()
        try:
            cur = conn.execute(
                "INSERT INTO acp_workspaces "
                "(name, root_path, description, workspace_type, parent_workspace_id, "
                " env_vars, git_remote_url, git_default_branch, git_current_branch, "
                " git_is_dirty, health_status, user_id, metadata, canonical_workspace_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    name, root_path, description, workspace_type,
                    parent_workspace_id, json.dumps(env_vars or {}),
                    git_remote_url, git_default_branch, git_current_branch,
                    int(git_is_dirty) if git_is_dirty is not None else None,
                    health_status, self._user_id,
                    json.dumps(metadata_payload), canonical_workspace_id, now,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            err = str(exc).lower()
            if "unique" in err:
                raise ValueError(
                    f"Workspace with name '{name}', root_path '{root_path}', or "
                    f"canonical_workspace_id '{canonical_workspace_id}' already exists for this user"
                ) from exc
            raise

        return ACPWorkspace(
            id=cur.lastrowid,
            name=name,
            root_path=root_path,
            description=description,
            workspace_type=workspace_type,
            parent_workspace_id=parent_workspace_id,
            env_vars=dict(env_vars or {}),
            git_remote_url=git_remote_url,
            git_default_branch=git_default_branch,
            git_current_branch=git_current_branch,
            git_is_dirty=git_is_dirty,
            health_status=health_status,
            user_id=self._user_id,
            created_at=now,
            metadata=metadata_payload,
        )

    def get_workspace(self, workspace_id: int) -> ACPWorkspace | None:
        self._ensure_schema()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM acp_workspaces WHERE id = ? AND user_id = ?",
            (workspace_id, self._user_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_workspace(row)

    def get_workspaces_by_ids(self, workspace_ids: list[int]) -> dict[int, ACPWorkspace]:
        """Return current-user workspaces keyed by ID for the provided IDs."""
        self._ensure_schema()
        unique_ids = sorted({int(workspace_id) for workspace_id in workspace_ids})
        if not unique_ids:
            return {}
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in unique_ids)
        rows = conn.execute(
            f"SELECT * FROM acp_workspaces WHERE user_id = ? AND id IN ({placeholders})",  # nosec B608
            [self._user_id, *unique_ids],
        ).fetchall()
        return {int(row["id"]): self._row_to_workspace(row) for row in rows}

    def get_workspace_by_root_path(self, root_path: str) -> ACPWorkspace | None:
        """Look up a workspace by its root_path for the current user."""
        self._ensure_schema()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM acp_workspaces WHERE root_path = ? AND user_id = ?",
            (root_path, self._user_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_workspace(row)

    def get_workspace_by_canonical_workspace_id(
        self,
        canonical_workspace_id: str,
    ) -> ACPWorkspace | None:
        """Look up a workspace linked to a canonical product workspace."""
        canonical_id = str(canonical_workspace_id or "").strip()
        if not canonical_id:
            return None
        self._ensure_schema()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM acp_workspaces "
            "WHERE canonical_workspace_id = ? AND user_id = ?",
            (canonical_id, self._user_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_workspace(row)

    def link_workspace_to_canonical(
        self,
        workspace_id: int,
        *,
        canonical_workspace_id: str,
        canonical_workspace_source: str = "workspace_playground",
        link_status: str = CANONICAL_WORKSPACE_LINKED_STATUS,
        metadata: dict[str, Any] | None = None,
    ) -> ACPWorkspace:
        """Attach canonical workspace metadata to an existing ACP workspace."""
        workspace = self.get_workspace(workspace_id)
        if workspace is None:
            raise OrchestrationNotFoundError(f"Workspace {workspace_id} not found")

        canonical_id = str(canonical_workspace_id).strip()
        existing_canonical_id = _canonical_workspace_id_from_metadata(workspace.metadata)
        if existing_canonical_id and existing_canonical_id != canonical_id:
            raise CanonicalWorkspaceBridgeConflictError(
                "ACP workspace root is already linked to a different canonical workspace."
            )

        merged_metadata = dict(workspace.metadata or {})
        merged_metadata.update(metadata or {})
        merged_metadata[CANONICAL_WORKSPACE_ID_METADATA_KEY] = canonical_id
        merged_metadata[CANONICAL_WORKSPACE_SOURCE_METADATA_KEY] = str(
            canonical_workspace_source
        )
        merged_metadata[CANONICAL_WORKSPACE_LINK_STATUS_METADATA_KEY] = str(link_status)
        conn = self._get_conn()
        try:
            cur = conn.execute(
                "UPDATE acp_workspaces SET metadata = ?, canonical_workspace_id = ?, "
                "updated_at = ? WHERE id = ? AND user_id = ? "
                "AND (canonical_workspace_id IS NULL OR canonical_workspace_id = ?)",
                (
                    json.dumps(merged_metadata),
                    canonical_id,
                    _now_iso(),
                    workspace_id,
                    self._user_id,
                    canonical_id,
                ),
            )
            if cur.rowcount == 0:
                conn.rollback()
                raise CanonicalWorkspaceBridgeConflictError(
                    "ACP workspace root is already linked to a different canonical workspace."
                )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise CanonicalWorkspaceBridgeConflictError(
                "Canonical workspace is already linked to a different ACP workspace."
            ) from exc
        updated = self.get_workspace(workspace_id)
        if updated is None:
            raise OrchestrationNotFoundError(f"Workspace {workspace_id} not found")
        return updated

    def find_or_create_canonical_workspace_bridge(
        self,
        *,
        canonical_workspace_id: str,
        canonical_workspace_source: str,
        root_path: str,
        name: str,
        description: str = "",
        env_vars: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ACPWorkspace:
        """Atomically find or create the ACP workspace for a canonical workspace."""
        canonical_id = str(canonical_workspace_id).strip()
        with self._bridge_lock:
            existing_link = self.get_workspace_by_canonical_workspace_id(canonical_id)
            if existing_link:
                if existing_link.root_path != root_path:
                    raise CanonicalWorkspaceBridgeConflictError(
                        "Canonical workspace is already linked to a different ACP workspace root."
                    )
                return existing_link

            existing_root = self.get_workspace_by_root_path(root_path)
            if existing_root:
                existing_canonical_id = _canonical_workspace_id_from_metadata(
                    existing_root.metadata
                )
                if existing_canonical_id and existing_canonical_id != canonical_id:
                    raise CanonicalWorkspaceBridgeConflictError(
                        "ACP workspace root is already linked to a different canonical workspace."
                    )
                return self.link_workspace_to_canonical(
                    existing_root.id,
                    canonical_workspace_id=canonical_id,
                    canonical_workspace_source=canonical_workspace_source,
                    metadata=metadata,
                )

            bridge_metadata = dict(metadata or {})
            bridge_metadata[CANONICAL_WORKSPACE_ID_METADATA_KEY] = canonical_id
            bridge_metadata[CANONICAL_WORKSPACE_SOURCE_METADATA_KEY] = (
                canonical_workspace_source
            )
            bridge_metadata[CANONICAL_WORKSPACE_LINK_STATUS_METADATA_KEY] = (
                CANONICAL_WORKSPACE_LINKED_STATUS
            )
            try:
                return self.create_workspace(
                    name=name,
                    root_path=root_path,
                    description=description,
                    workspace_type="manual",
                    env_vars=env_vars,
                    metadata=bridge_metadata,
                )
            except ValueError as exc:
                raise CanonicalWorkspaceBridgeConflictError(str(exc)) from exc

    def list_workspaces(
        self,
        workspace_type: str | None = None,
        health_status: str | None = None,
    ) -> list[ACPWorkspace]:
        self._ensure_schema()
        conn = self._get_conn()
        query = "SELECT * FROM acp_workspaces WHERE user_id = ?"
        params: list[Any] = [self._user_id]
        if workspace_type is not None:
            query += " AND workspace_type = ?"
            params.append(workspace_type)
        if health_status is not None:
            query += " AND health_status = ?"
            params.append(health_status)
        query += " ORDER BY created_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_workspace(r) for r in rows]

    def update_workspace(
        self,
        workspace_id: int,
        **fields: Any,
    ) -> ACPWorkspace:
        self._ensure_schema()
        conn = self._get_conn()

        # Verify ownership
        existing = conn.execute(
            "SELECT * FROM acp_workspaces WHERE id = ? AND user_id = ?",
            (workspace_id, self._user_id),
        ).fetchone()
        if existing is None:
            raise OrchestrationNotFoundError(f"Workspace {workspace_id} not found")

        # workspace_type and parent_workspace_id are immutable after creation
        allowed = {"name", "root_path", "description", "env_vars", "metadata"}
        sets: list[str] = []
        params: list[Any] = []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key in ("env_vars", "metadata"):
                sets.append(f"{key} = ?")
                params.append(json.dumps(value or {}))
                if key == "metadata":
                    sets.append("canonical_workspace_id = ?")
                    params.append(_canonical_workspace_id_from_metadata(value or {}))
            else:
                sets.append(f"{key} = ?")
                params.append(value)

        if not sets:
            return self._row_to_workspace(existing)

        sets.append("updated_at = ?")
        params.append(_now_iso())
        params.append(workspace_id)

        try:
            # Column names are constrained by the local allowlist above.
            conn.execute(
                f"UPDATE acp_workspaces SET {', '.join(sets)} WHERE id = ?",  # nosec B608
                params,
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            err = str(exc).lower()
            if "unique" in err:
                raise ValueError(
                    "Workspace name, root_path, or canonical workspace link conflicts with an existing workspace"
                ) from exc
            raise

        row = conn.execute(
            "SELECT * FROM acp_workspaces WHERE id = ?", (workspace_id,)
        ).fetchone()
        return self._row_to_workspace(row)

    def delete_workspace(self, workspace_id: int) -> bool:
        self._ensure_schema()
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM acp_workspaces WHERE id = ? AND user_id = ?",
            (workspace_id, self._user_id),
        )
        conn.commit()
        return cur.rowcount > 0

    def update_workspace_health(
        self,
        workspace_id: int,
        health_status: str,
        git_remote_url: str | None = None,
        git_default_branch: str | None = None,
        git_current_branch: str | None = None,
        git_is_dirty: bool | None = None,
        last_health_check: str | None = None,
    ) -> ACPWorkspace:
        self._ensure_schema()
        conn = self._get_conn()
        existing = conn.execute(
            "SELECT * FROM acp_workspaces WHERE id = ? AND user_id = ?",
            (workspace_id, self._user_id),
        ).fetchone()
        if existing is None:
            raise OrchestrationNotFoundError(f"Workspace {workspace_id} not found")

        now = last_health_check or _now_iso()
        conn.execute(
            "UPDATE acp_workspaces SET health_status = ?, git_remote_url = ?, "
            "git_default_branch = ?, git_current_branch = ?, git_is_dirty = ?, "
            "last_health_check = ?, updated_at = ? WHERE id = ?",
            (
                health_status, git_remote_url, git_default_branch,
                git_current_branch,
                int(git_is_dirty) if git_is_dirty is not None else None,
                now, _now_iso(), workspace_id,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM acp_workspaces WHERE id = ?", (workspace_id,)
        ).fetchone()
        return self._row_to_workspace(row)

    def list_workspace_children(self, workspace_id: int) -> list[ACPWorkspace]:
        self._ensure_schema()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM acp_workspaces WHERE parent_workspace_id = ? AND user_id = ? "
            "ORDER BY name",
            (workspace_id, self._user_id),
        ).fetchall()
        return [self._row_to_workspace(r) for r in rows]

    # ------------------------------------------------------------------
    # Workspace MCP Servers
    # ------------------------------------------------------------------

    def create_workspace_mcp_server(
        self,
        workspace_id: int,
        server_name: str,
        server_type: str = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        url: str | None = None,
        enabled: bool = True,
    ) -> dict[str, Any]:
        self._ensure_schema()
        conn = self._get_conn()

        # Verify workspace exists and belongs to user
        if conn.execute(
            "SELECT 1 FROM acp_workspaces WHERE id = ? AND user_id = ?",
            (workspace_id, self._user_id),
        ).fetchone() is None:
            raise OrchestrationNotFoundError(f"Workspace {workspace_id} not found")

        try:
            cur = conn.execute(
                "INSERT INTO acp_workspace_mcp_servers "
                "(workspace_id, server_name, server_type, command, args, env, url, enabled) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    workspace_id, server_name, server_type, command,
                    json.dumps(args or []), json.dumps(env or {}),
                    url, int(enabled),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"MCP server '{server_name}' already exists for workspace {workspace_id}"
            ) from exc

        return {
            "id": cur.lastrowid,
            "workspace_id": workspace_id,
            "server_name": server_name,
            "server_type": server_type,
            "command": command,
            "args": list(args or []),
            "env": dict(env or {}),
            "url": url,
            "enabled": enabled,
        }

    def list_workspace_mcp_servers(self, workspace_id: int) -> list[dict[str, Any]]:
        self._ensure_schema()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM acp_workspace_mcp_servers WHERE workspace_id = ? ORDER BY server_name",
            (workspace_id,),
        ).fetchall()
        return [self._row_to_mcp_server(r) for r in rows]

    def delete_workspace_mcp_server(self, workspace_id: int, server_id: int) -> bool:
        """Delete an MCP server in a single atomic query that also verifies ownership."""
        self._ensure_schema()
        conn = self._get_conn()
        cur = conn.execute(
            "DELETE FROM acp_workspace_mcp_servers "
            "WHERE id = ? AND workspace_id = ? "
            "AND EXISTS (SELECT 1 FROM acp_workspaces w "
            "            WHERE w.id = ? AND w.user_id = ?)",
            (server_id, workspace_id, workspace_id, self._user_id),
        )
        conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def create_project(
        self,
        name: str,
        description: str = "",
        workspace_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentProject:
        self._ensure_schema()
        conn = self._get_conn()

        # Validate workspace exists if provided
        if workspace_id is not None:
            if conn.execute(
                "SELECT 1 FROM acp_workspaces WHERE id = ? AND user_id = ?",
                (workspace_id, self._user_id),
            ).fetchone() is None:
                raise OrchestrationNotFoundError(f"Workspace {workspace_id} not found")

        now = _now_iso()
        cur = conn.execute(
            "INSERT INTO projects (name, description, workspace_id, user_id, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, description, workspace_id, self._user_id, json.dumps(metadata or {}), now),
        )
        conn.commit()
        return AgentProject(
            id=cur.lastrowid,
            name=name,
            description=description,
            workspace_id=workspace_id,
            user_id=self._user_id,
            created_at=now,
            metadata=dict(metadata or {}),
        )

    def get_project(self, project_id: int) -> AgentProject | None:
        self._ensure_schema()
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_project(row)

    def list_projects(self, workspace_id: int | None | object = _UNSET) -> list[AgentProject]:
        """List projects for the current user.

        Args:
            workspace_id: Filter by workspace.  Omit (default) for no filter.
                Pass ``None`` to list only unbound projects.
                Pass an int to list projects for that workspace.
        """
        self._ensure_schema()
        conn = self._get_conn()
        if workspace_id is _UNSET:
            rows = conn.execute(
                "SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC",
                (self._user_id,),
            ).fetchall()
        elif workspace_id is None:
            rows = conn.execute(
                "SELECT * FROM projects WHERE user_id = ? AND workspace_id IS NULL "
                "ORDER BY created_at DESC",
                (self._user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM projects WHERE user_id = ? AND workspace_id = ? "
                "ORDER BY created_at DESC",
                (self._user_id, workspace_id),
            ).fetchall()
        return [self._row_to_project(r) for r in rows]

    def delete_project(self, project_id: int) -> bool:
        self._ensure_schema()
        conn = self._get_conn()
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    def create_task(
        self,
        project_id: int,
        title: str,
        description: str = "",
        agent_type: str | None = None,
        dependency_id: int | None = None,
        reviewer_agent_type: str | None = None,
        max_review_attempts: int = 3,
        success_criteria: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AgentTask:
        self._ensure_schema()
        conn = self._get_conn()

        # Validate project exists
        if conn.execute("SELECT 1 FROM projects WHERE id = ?", (project_id,)).fetchone() is None:
            raise OrchestrationNotFoundError(f"Project {project_id} not found")

        # Validate dependency exists
        if dependency_id is not None:
            if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (dependency_id,)).fetchone() is None:
                raise OrchestrationNotFoundError(f"Dependency task {dependency_id} not found")

        now = _now_iso()
        cur = conn.execute(
            "INSERT INTO tasks "
            "(project_id, title, description, status, agent_type, dependency_id, "
            " reviewer_agent_type, max_review_attempts, review_count, success_criteria, "
            " user_id, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)",
            (
                project_id, title, description, TaskStatus.TODO.value,
                agent_type, dependency_id, reviewer_agent_type,
                max_review_attempts, success_criteria,
                self._user_id, json.dumps(metadata or {}), now,
            ),
        )
        conn.commit()
        return AgentTask(
            id=cur.lastrowid,
            project_id=project_id,
            title=title,
            description=description,
            status=TaskStatus.TODO,
            agent_type=agent_type,
            dependency_id=dependency_id,
            reviewer_agent_type=reviewer_agent_type,
            max_review_attempts=max_review_attempts,
            review_count=0,
            success_criteria=success_criteria,
            user_id=self._user_id,
            created_at=now,
            metadata=dict(metadata or {}),
        )

    def get_task(self, task_id: int) -> AgentTask | None:
        self._ensure_schema()
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def list_tasks(
        self,
        project_id: int,
        status: TaskStatus | None = None,
    ) -> list[AgentTask]:
        self._ensure_schema()
        conn = self._get_conn()
        if status is not None:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE project_id = ? AND status = ? ORDER BY created_at",
                (project_id, status.value),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE project_id = ? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def transition_task(self, task_id: int, new_status: TaskStatus) -> AgentTask:
        self._ensure_schema()
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise OrchestrationNotFoundError(f"Task {task_id} not found")

        current = TaskStatus(row["status"])
        if not is_valid_transition(current, new_status):
            raise InvalidTransitionError(
                f"Invalid transition from {current.value} to {new_status.value}"
            )

        now = _now_iso()
        conn.execute(
            "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
            (new_status.value, now, task_id),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_task(updated)

    def check_dependency_ready(self, task_id: int) -> bool:
        self._ensure_schema()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT dependency_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            raise OrchestrationNotFoundError(f"Task {task_id} not found")
        dep_id = row["dependency_id"]
        if dep_id is None:
            return True
        dep_row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (dep_id,),
        ).fetchone()
        if dep_row is None:
            return True  # dependency deleted
        return TaskStatus(dep_row["status"]) == TaskStatus.COMPLETE

    def _detect_cycle(self, task_id: int, dependency_id: int) -> bool:
        """Walk dependency chain from dependency_id; True if it reaches task_id."""
        self._ensure_schema()
        conn = self._get_conn()
        visited: set[int] = set()
        current = dependency_id
        while current is not None:
            if current == task_id:
                return True
            if current in visited:
                return True  # already a cycle in the chain
            visited.add(current)
            row = conn.execute(
                "SELECT dependency_id FROM tasks WHERE id = ?", (current,),
            ).fetchone()
            if row is None:
                break
            current = row["dependency_id"]
        return False

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def create_run(
        self,
        task_id: int,
        agent_type: str | None = None,
        session_id: str | None = None,
    ) -> AgentRun:
        self._ensure_schema()
        conn = self._get_conn()
        now = _now_iso()
        cur = conn.execute(
            "INSERT INTO runs (task_id, session_id, status, agent_type, started_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, session_id, RunStatus.RUNNING.value, agent_type, now),
        )
        conn.commit()
        return AgentRun(
            id=cur.lastrowid,
            task_id=task_id,
            session_id=session_id,
            status=RunStatus.RUNNING,
            agent_type=agent_type,
            started_at=now,
        )

    def complete_run(
        self,
        run_id: int,
        result_summary: str = "",
        token_usage: dict[str, Any] | None = None,
    ) -> AgentRun:
        self._ensure_schema()
        conn = self._get_conn()
        now = _now_iso()
        conn.execute(
            "UPDATE runs SET status = ?, completed_at = ?, result_summary = ?, token_usage = ? "
            "WHERE id = ?",
            (
                RunStatus.COMPLETED.value, now, result_summary,
                json.dumps(token_usage or {}), run_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_run(row)

    def fail_run(self, run_id: int, error: str = "") -> AgentRun:
        self._ensure_schema()
        conn = self._get_conn()
        now = _now_iso()
        conn.execute(
            "UPDATE runs SET status = ?, completed_at = ?, error = ? WHERE id = ?",
            (RunStatus.FAILED.value, now, error, run_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._row_to_run(row)

    def list_runs(self, task_id: int) -> list[AgentRun]:
        self._ensure_schema()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM runs WHERE task_id = ? ORDER BY started_at DESC",
            (task_id,),
        ).fetchall()
        return [self._row_to_run(r) for r in rows]

    # ------------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------------

    def submit_review(
        self,
        task_id: int,
        approved: bool,
        feedback: str = "",
        reviewer: str | None = None,
    ) -> AgentTask:
        self._ensure_schema()
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise OrchestrationNotFoundError(f"Task {task_id} not found")
        if TaskStatus(row["status"]) != TaskStatus.REVIEW:
            raise InvalidTransitionError(f"Task {task_id} is not in review status")

        now = _now_iso()
        new_review_count = row["review_count"] + 1

        # Determine new status
        if approved:
            new_status = TaskStatus.COMPLETE
        elif new_review_count >= row["max_review_attempts"]:
            new_status = TaskStatus.TRIAGE
        else:
            new_status = TaskStatus.IN_PROGRESS

        conn.execute(
            "UPDATE tasks SET status = ?, review_count = ?, updated_at = ? WHERE id = ?",
            (new_status.value, new_review_count, now, task_id),
        )
        conn.execute(
            "INSERT INTO reviews (task_id, approved, feedback, reviewer, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, int(approved), feedback, reviewer, now),
        )
        conn.commit()
        updated = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_task(updated)

    def list_reviews(self, task_id: int) -> list[dict[str, Any]]:
        self._ensure_schema()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM reviews WHERE task_id = ? ORDER BY created_at",
            (task_id,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "task_id": r["task_id"],
                "approved": bool(r["approved"]),
                "feedback": r["feedback"],
                "reviewer": r["reviewer"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_project_summary(self, project_id: int) -> dict[str, Any]:
        self._ensure_schema()
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM tasks WHERE project_id = ? GROUP BY status",
            (project_id,),
        ).fetchall()
        status_counts: dict[str, int] = {}
        total = 0
        for r in rows:
            status_counts[r["status"]] = r["cnt"]
            total += r["cnt"]
        return {
            "project_id": project_id,
            "total_tasks": total,
            "status_counts": status_counts,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._conn_local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._conn_local.conn = None

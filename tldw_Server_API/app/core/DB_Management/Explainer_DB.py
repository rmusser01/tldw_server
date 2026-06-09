"""SQLite storage for the Explainer workspace."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)


class ExplainerDatabaseError(Exception):
    """Base exception for Explainer database failures."""


class SchemaError(ExplainerDatabaseError):
    """Raised when Explainer schema initialization fails."""


class InputError(ValueError):
    """Raised for invalid Explainer input."""


class ExplainerDatabase:
    """Per-user SQLite database for Explainer sessions, nodes, sources, and citations."""

    _schema_init_paths: ClassVar[set[str]] = set()

    def __init__(self, db_path: str | Path, client_id: str = "explainer") -> None:
        if not client_id:
            raise ValueError("client_id is required")
        self.client_id = str(client_id)
        self._db_path_str = str(db_path)
        self.db_path = Path(self._db_path_str).resolve() if self._db_path_str != ":memory:" else Path(":memory:")
        if self._db_path_str != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._ensure_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Return a thread-local configured SQLite connection."""
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path_str, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            configure_sqlite_connection(conn)
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.connection = conn
        return conn

    def close_connection(self) -> None:
        """Close the thread-local SQLite connection if one exists."""
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None

    @contextmanager
    def transaction(self) -> Iterable[sqlite3.Connection]:
        """Run operations in a SQLite transaction."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def utcnow_iso() -> str:
        """Return a second-resolution UTC timestamp."""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _ensure_schema(self) -> None:
        if self._db_path_str != ":memory:" and self._db_path_str in self._schema_init_paths:
            return
        conn = self.get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS explainer_sessions (
                  id TEXT PRIMARY KEY,
                  owner_user_id TEXT NOT NULL,
                  title TEXT NOT NULL,
                  mode TEXT NOT NULL CHECK (mode IN ('goal', 'sources')),
                  status TEXT NOT NULL CHECK (status IN ('draft', 'active', 'archived', 'error')),
                  output_intent TEXT NOT NULL CHECK (output_intent IN ('explain', 'plan', 'both')),
                  grounding TEXT NOT NULL CHECK (grounding IN ('source_only', 'source_led', 'open')),
                  depth_preset TEXT NOT NULL CHECK (depth_preset IN ('quick', 'standard', 'deep')),
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  archived_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_explainer_sessions_owner_updated
                  ON explainer_sessions(owner_user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_explainer_sessions_owner_status
                  ON explainer_sessions(owner_user_id, status);

                CREATE TABLE IF NOT EXISTS explainer_nodes (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL REFERENCES explainer_sessions(id) ON DELETE CASCADE,
                  parent_id TEXT REFERENCES explainer_nodes(id) ON DELETE CASCADE,
                  ordinal INTEGER NOT NULL,
                  title TEXT NOT NULL,
                  body TEXT,
                  kind TEXT NOT NULL,
                  intent TEXT NOT NULL,
                  status TEXT NOT NULL,
                  evidence_state TEXT NOT NULL,
                  outside_knowledge_used INTEGER NOT NULL DEFAULT 0,
                  question_options_json TEXT,
                  selected_option_id TEXT,
                  selected_custom_answer TEXT,
                  generation_metadata_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  deleted_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_explainer_nodes_session_parent
                  ON explainer_nodes(session_id, parent_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_explainer_nodes_session_deleted
                  ON explainer_nodes(session_id, deleted_at);

                CREATE TABLE IF NOT EXISTS explainer_selected_sources (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL REFERENCES explainer_sessions(id) ON DELETE CASCADE,
                  owner_user_id TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  source_id TEXT NOT NULL,
                  source_type TEXT NOT NULL,
                  title TEXT NOT NULL,
                  added_at TEXT NOT NULL,
                  snapshot_version TEXT,
                  metadata_json TEXT,
                  deleted_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_explainer_sources_session
                  ON explainer_selected_sources(session_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_explainer_sources_owner_source
                  ON explainer_selected_sources(owner_user_id, source_type, source_id);

                CREATE TABLE IF NOT EXISTS explainer_citations (
                  id TEXT PRIMARY KEY,
                  session_id TEXT NOT NULL REFERENCES explainer_sessions(id) ON DELETE CASCADE,
                  node_id TEXT NOT NULL REFERENCES explainer_nodes(id) ON DELETE CASCADE,
                  owner_user_id TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  source_id TEXT NOT NULL,
                  source_type TEXT NOT NULL,
                  title TEXT NOT NULL,
                  excerpt TEXT NOT NULL,
                  location_label TEXT,
                  start_offset INTEGER,
                  end_offset INTEGER,
                  url TEXT,
                  snapshot_hash TEXT,
                  created_at TEXT NOT NULL,
                  deleted_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_explainer_citations_node
                  ON explainer_citations(node_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_explainer_citations_owner_source
                  ON explainer_citations(owner_user_id, source_type, source_id);
                """
            )
            conn.commit()
            if self._db_path_str != ":memory:":
                self._schema_init_paths.add(self._db_path_str)
        except sqlite3.Error as exc:
            conn.rollback()
            raise SchemaError(f"Failed to initialize Explainer DB schema: {exc}") from exc

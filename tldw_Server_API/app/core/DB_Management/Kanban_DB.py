# Kanban_DB.py
# Description: SQLite database management for Kanban boards, lists, and cards.
#
from __future__ import annotations

"""
Kanban_DB.py
------------

SQLite wrapper for per-user Kanban board data:
 - boards
 - lists
 - cards
 - labels
 - card_labels
 - checklists
 - checklist_items
 - comments
 - activities
 - card_links

This module encapsulates raw SQL per project guidelines.
Implements soft delete, archive, versioning, and FTS5 search.
"""

import contextlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402
import threading  # noqa: E402
import uuid as uuid_module  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from loguru import logger  # noqa: E402

from tldw_Server_API.app.core.config import settings  # noqa: E402
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths, _is_test_context  # noqa: E402
from tldw_Server_API.app.core.DB_Management.kanban_vector_search import create_kanban_vector_search  # noqa: E402
from tldw_Server_API.app.core.exceptions import StorageUnavailableError  # noqa: E402

_KANBAN_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)

_WORKFLOW_METADATA_UNSET = object()


# --- Helper Functions ---
def _utcnow_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _generate_uuid() -> str:
    """Generate a lowercase hex UUID."""
    return uuid_module.uuid4().hex.lower()


def _normalize_user_id_for_records(user_id: str) -> str:
    """Normalize user IDs for storage; in tests, prefix numeric IDs with test_user_."""
    raw = str(user_id).strip()
    if _is_test_context() and raw.isdigit():
        return f"test_user_{raw}"
    return raw


def _normalize_db_path(
    db_path: str,
    user_id: str,
) -> tuple[str, bool]:
    """Normalize db_path and always enforce user directory containment."""
    if not db_path or not str(db_path).strip():
        raise InputError("db_path is required")  # noqa: TRY003

    if db_path == ":memory:":
        return db_path, True

    try:
        resolved = Path(db_path).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise InputError(f"Invalid db_path: {db_path}") from exc  # noqa: TRY003

    # Always enforce that the database path stays within the user's base directory
    # when a user_id is provided. This guards against directory traversal or use
    # of unexpected locations.
    if user_id:
        try:
            user_dir = DatabasePaths.get_user_base_directory(user_id).resolve()
        except StorageUnavailableError as exc:
            raise KanbanDBError("storage_unavailable") from exc
        except ValueError as exc:
            raise InputError("invalid_user_id") from exc
        try:
            resolved.relative_to(user_dir)
        except ValueError as exc:
            raise InputError("db_path must be within the user database directory") from exc  # noqa: TRY003

    return str(resolved), False


def _get_int_setting(key: str, default: int) -> int:
    raw = settings.get(key)
    if raw is None:
        raw = os.getenv(key)
    try:
        value = int(str(raw).strip()) if raw is not None else default
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _kanban_card_indexable(
    *,
    user_id: str,
    card_id: int,
    expected_version: Any | None = None,
) -> bool:
    """Return True when a card exists, is not deleted/archived, and matches the expected version."""
    try:
        db_path = DatabasePaths.get_kanban_db_path(user_id)
        db = KanbanDB(db_path=str(db_path), user_id=str(user_id))
        try:
            card = db.get_card(card_id, include_deleted=True)
            if not card:
                return False
            if card.get("deleted") or card.get("archived"):
                return False
            if expected_version is not None:
                try:
                    expected_version = int(expected_version)
                except (TypeError, ValueError):
                    expected_version = None
            return not (expected_version is not None and int(card.get("version") or 0) != expected_version)
        finally:
            try:
                db.close()
            except _KANBAN_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    "Error closing db in _kanban_card_indexable during db.close(): "
                    f"user_id={user_id}, card_id={card_id}, error={exc}"
                )
    except _KANBAN_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Kanban card indexability check failed: {exc}")
        return False


# --- SQLite Connection Helpers ---
class _KanbanMemoryConnection(sqlite3.Connection):
    """SQLite connection that keeps :memory: databases alive across operations."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._allow_close = False

    def close(self) -> None:
        if self._allow_close:
            super().close()

    def force_close(self) -> None:
        self._allow_close = True
        super().close()


# --- Custom Exceptions ---
class KanbanDBError(Exception):
    """Base exception for KanbanDB related errors."""
    pass


class InputError(ValueError):
    """Exception for input validation errors."""
    pass


class ConflictError(KanbanDBError):
    """
    Indicates a conflict due to concurrent modification or unique constraint violation.

    This can occur if a record's version doesn't match an expected version during
    an update/delete operation (optimistic locking), or if an insert/update
    violates a unique constraint (e.g., duplicate client_id).
    """
    def __init__(
        self,
        message: str = "Conflict detected.",
        entity: str | None = None,
        entity_id: Any = None,
        *,
        code: str | None = None,
    ):
        super().__init__(message)
        self.entity = entity
        self.entity_id = entity_id
        self.code = code

    def __str__(self):
        base = super().__str__()
        details = []
        if self.entity:
            details.append(f"Entity: {self.entity}")
        if self.entity_id:
            details.append(f"ID: {self.entity_id}")
        return f"{base} ({', '.join(details)})" if details else base


class NotFoundError(KanbanDBError):
    """Exception when a requested resource is not found."""
    def __init__(self, message: str = "Resource not found.", entity: str | None = None, entity_id: Any = None):
        super().__init__(message)
        self.entity = entity
        self.entity_id = entity_id


# --- Database Class ---
class KanbanDB:
    """
    Manages SQLite connections and operations for the Kanban database.

    This class provides thread-safe database access with:
    - WAL mode for better concurrency
    - Foreign key enforcement
    - Optimistic locking via version field
    - Soft delete and archive support
    - FTS5 full-text search for cards

    Attributes:
        db_path (str): Path to the SQLite database file.
        user_id (str): The user ID for user-scoped operations.
    """

    # Configuration defaults (can be overridden via environment variables)
    DEFAULT_ACTIVITY_RETENTION_DAYS = 30
    MAX_BOARDS_PER_USER = 50
    MAX_LISTS_PER_BOARD = 20
    MAX_CARDS_PER_LIST = 200
    MAX_CARDS_PER_BOARD = 500
    MAX_LABELS_PER_BOARD = 20
    MAX_CHECKLISTS_PER_CARD = 20
    MAX_CHECKLIST_ITEMS_PER_CHECKLIST = 100
    MAX_COMMENTS_PER_CARD = 500
    MAX_COMMENT_SIZE = 16384  # characters
    VECTOR_INDEX_RETRY_DELAY_SECONDS = 5
    VECTOR_INDEX_MAX_RETRY_ATTEMPTS = 1
    DEFAULT_WORKFLOW_STATUSES = (
        ("todo", "To Do", 0, 0, 1),
        ("impl", "In Progress", 1, 0, 1),
        ("done", "Done", 2, 1, 1),
    )
    DEFAULT_WORKFLOW_TRANSITIONS = (
        ("todo", "impl", 1, 0, None, None, None, 0, 1),
        ("impl", "done", 1, 0, None, None, None, 0, 1),
    )

    def __init__(self, db_path: str, user_id: str) -> None:
        """
        Initialize the KanbanDB instance.

        Args:
            db_path: Path to the SQLite database file. Must resolve within the
                user-scoped database directory.
            user_id: The user ID for this database instance. In test context,
                purely numeric IDs are stored as test_user_<id>.
        """
        raw_user_id = str(user_id)
        if not raw_user_id.strip():
            raise InputError("user_id is required")  # noqa: TRY003
        self.user_id = _normalize_user_id_for_records(raw_user_id)
        self._lock = threading.RLock()
        self.db_path, self._is_memory_db = _normalize_db_path(
            db_path,
            raw_user_id,
        )
        self._memory_conn: _KanbanMemoryConnection | None = None
        self._vector_search = None
        self._vector_search_initialized = False
        self._vector_index_retry_pending: set[tuple[str, int]] = set()
        self._vector_index_retry_lock = threading.Lock()
        self._apply_limit_overrides()

        # Ensure directory exists
        if not self._is_memory_db:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

        # Initialize schema
        self._ensure_schema()
        logger.debug(f"KanbanDB initialized for user {self.user_id} at {self.db_path}")

    def _apply_limit_overrides(self) -> None:
        """Apply limit overrides from settings/environment."""
        self.DEFAULT_ACTIVITY_RETENTION_DAYS = _get_int_setting(
            "KANBAN_DEFAULT_ACTIVITY_RETENTION_DAYS",
            self.DEFAULT_ACTIVITY_RETENTION_DAYS,
        )
        self.MAX_BOARDS_PER_USER = _get_int_setting(
            "KANBAN_MAX_BOARDS_PER_USER",
            self.MAX_BOARDS_PER_USER,
        )
        self.MAX_LISTS_PER_BOARD = _get_int_setting(
            "KANBAN_MAX_LISTS_PER_BOARD",
            self.MAX_LISTS_PER_BOARD,
        )
        self.MAX_CARDS_PER_LIST = _get_int_setting(
            "KANBAN_MAX_CARDS_PER_LIST",
            self.MAX_CARDS_PER_LIST,
        )
        self.MAX_CARDS_PER_BOARD = _get_int_setting(
            "KANBAN_MAX_CARDS_PER_BOARD",
            self.MAX_CARDS_PER_BOARD,
        )
        self.MAX_LABELS_PER_BOARD = _get_int_setting(
            "KANBAN_MAX_LABELS_PER_BOARD",
            self.MAX_LABELS_PER_BOARD,
        )
        self.MAX_CHECKLISTS_PER_CARD = _get_int_setting(
            "KANBAN_MAX_CHECKLISTS_PER_CARD",
            self.MAX_CHECKLISTS_PER_CARD,
        )
        self.MAX_CHECKLIST_ITEMS_PER_CHECKLIST = _get_int_setting(
            "KANBAN_MAX_CHECKLIST_ITEMS_PER_CHECKLIST",
            self.MAX_CHECKLIST_ITEMS_PER_CHECKLIST,
        )
        self.MAX_COMMENTS_PER_CARD = _get_int_setting(
            "KANBAN_MAX_COMMENTS_PER_CARD",
            self.MAX_COMMENTS_PER_CARD,
        )
        self.MAX_COMMENT_SIZE = _get_int_setting(
            "KANBAN_MAX_COMMENT_SIZE",
            self.MAX_COMMENT_SIZE,
        )

    def _configure_connection(self, conn: sqlite3.Connection, *, enable_wal: bool = True) -> None:
        conn.row_factory = sqlite3.Row
        # Foreign keys and timeout are required for integrity and responsiveness.
        for statement in ("PRAGMA foreign_keys=ON", "PRAGMA busy_timeout=30000"):
            try:
                conn.execute(statement)
            except sqlite3.Error as e:
                logger.error(f"Failed to set critical PRAGMA option {statement}: {e}")
                raise KanbanDBError(  # noqa: TRY003
                    f"Failed to set critical PRAGMA option {statement}: {e}"
                ) from e

        # Performance pragmas are best-effort; log failures for investigation.
        try:
            if enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        except sqlite3.Error as e:
            logger.error(f"Failed to set performance PRAGMA options: {e}")

    def _connect(self) -> sqlite3.Connection:
        """Create and configure a database connection."""
        with self._lock:
            if self._is_memory_db:
                if self._memory_conn is None:
                    # Keep a single shared connection so :memory: schema/data persist.
                    conn = sqlite3.connect(
                        self.db_path,
                        timeout=30,
                        isolation_level=None,
                        check_same_thread=False,
                        factory=_KanbanMemoryConnection,
                    )
                    try:
                        self._configure_connection(conn, enable_wal=False)
                    except _KANBAN_NONCRITICAL_EXCEPTIONS:
                        try:
                            conn.force_close()
                        finally:
                            raise
                    self._memory_conn = conn
                return self._memory_conn

            conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
            try:
                self._configure_connection(conn)
            except _KANBAN_NONCRITICAL_EXCEPTIONS:
                conn.close()
                raise
            return conn

    def _ensure_schema(self) -> None:
        """Create all tables, indexes, and triggers if they don't exist."""
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(self._get_schema_sql())
                conn.commit()
                logger.debug("Kanban schema ensured")
            except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Failed to ensure schema: {e}")
                raise KanbanDBError(f"Schema creation failed: {e}") from e  # noqa: TRY003
            finally:
                conn.close()

    def close(self) -> None:
        """
        Close any persistent in-memory connection.

        This is only needed for :memory: databases. File-based databases are a
        no-op. Call this when finished with the KanbanDB instance to free
        resources.
        """
        with self._lock:
            if self._vector_search is not None:
                try:
                    self._vector_search.close()
                except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"Error closing Kanban vector search: {e}")
                finally:
                    self._vector_search = None
                    self._vector_search_initialized = False
            if self._memory_conn is None:
                return
            try:
                self._memory_conn.force_close()
            except sqlite3.Error as e:
                logger.warning(f"Error closing memory connection: {e}")
            finally:
                self._memory_conn = None

    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        try:
            self.close()
        except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
            # Ignore errors during cleanup - object is being destroyed anyway.
            logger.debug(f"KanbanDB __del__ cleanup failed: {e}")

    def __enter__(self) -> KanbanDB:
        """Enable context-manager use for automatic cleanup."""
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        """Ensure persistent resources are released on exit."""
        self.close()

    def _get_schema_sql(self) -> str:
        """Return the complete schema SQL."""
        return """
-- Boards
CREATE TABLE IF NOT EXISTS kanban_boards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    client_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    archived INTEGER DEFAULT 0,
    archived_at TIMESTAMP,
    activity_retention_days INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted INTEGER DEFAULT 0,
    deleted_at TIMESTAMP,
    version INTEGER DEFAULT 1,
    metadata JSON
);
CREATE INDEX IF NOT EXISTS idx_boards_user_archived ON kanban_boards(user_id, archived);
CREATE INDEX IF NOT EXISTS idx_boards_deleted ON kanban_boards(deleted, deleted_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_boards_client_id ON kanban_boards(user_id, client_id);

-- Lists
CREATE TABLE IF NOT EXISTS kanban_lists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,
    client_id TEXT NOT NULL,
    name TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    archived INTEGER DEFAULT 0,
    archived_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted INTEGER DEFAULT 0,
    deleted_at TIMESTAMP,
    version INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_lists_board_position ON kanban_lists(board_id, position);
CREATE INDEX IF NOT EXISTS idx_lists_board_archived ON kanban_lists(board_id, archived);
CREATE INDEX IF NOT EXISTS idx_lists_deleted ON kanban_lists(deleted, deleted_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_lists_client_id ON kanban_lists(board_id, client_id);

-- Cards
CREATE TABLE IF NOT EXISTS kanban_cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,
    list_id INTEGER NOT NULL REFERENCES kanban_lists(id) ON DELETE CASCADE,
    client_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    position INTEGER NOT NULL DEFAULT 0,
    due_date TIMESTAMP,
    due_complete INTEGER DEFAULT 0,
    start_date TIMESTAMP,
    priority TEXT CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    archived INTEGER DEFAULT 0,
    archived_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted INTEGER DEFAULT 0,
    deleted_at TIMESTAMP,
    version INTEGER DEFAULT 1,
    metadata JSON
);
CREATE INDEX IF NOT EXISTS idx_cards_board ON kanban_cards(board_id);
CREATE INDEX IF NOT EXISTS idx_cards_list_position ON kanban_cards(list_id, position);
CREATE INDEX IF NOT EXISTS idx_cards_list_archived ON kanban_cards(list_id, archived);
CREATE INDEX IF NOT EXISTS idx_cards_due_date ON kanban_cards(due_date);
CREATE INDEX IF NOT EXISTS idx_cards_priority ON kanban_cards(board_id, priority);
CREATE INDEX IF NOT EXISTS idx_cards_deleted ON kanban_cards(deleted, deleted_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cards_client_id ON kanban_cards(board_id, client_id);

-- Labels
CREATE TABLE IF NOT EXISTS kanban_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,
    name TEXT,
    color TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_labels_board ON kanban_labels(board_id);

-- Card-Label join
CREATE TABLE IF NOT EXISTS kanban_card_labels (
    card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
    label_id INTEGER NOT NULL REFERENCES kanban_labels(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (card_id, label_id)
);

-- Checklists
CREATE TABLE IF NOT EXISTS kanban_checklists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_checklists_card ON kanban_checklists(card_id);

-- Checklist Items
CREATE TABLE IF NOT EXISTS kanban_checklist_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    checklist_id INTEGER NOT NULL REFERENCES kanban_checklists(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    checked INTEGER DEFAULT 0,
    checked_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_checklist_items_checklist ON kanban_checklist_items(checklist_id);

-- Comments
CREATE TABLE IF NOT EXISTS kanban_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_comments_card ON kanban_comments(card_id);

-- Activity Log
CREATE TABLE IF NOT EXISTS kanban_activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    board_id INTEGER NOT NULL REFERENCES kanban_boards(id) ON DELETE CASCADE,
    list_id INTEGER REFERENCES kanban_lists(id) ON DELETE SET NULL,
    card_id INTEGER REFERENCES kanban_cards(id) ON DELETE SET NULL,
    user_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id INTEGER,
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_activities_board ON kanban_activities(board_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activities_list ON kanban_activities(list_id);
CREATE INDEX IF NOT EXISTS idx_activities_card ON kanban_activities(card_id);
CREATE INDEX IF NOT EXISTS idx_activities_created ON kanban_activities(created_at);

-- Workflow Policies
CREATE TABLE IF NOT EXISTS board_workflow_policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    board_id INTEGER NOT NULL UNIQUE REFERENCES kanban_boards(id) ON DELETE CASCADE,
    version INTEGER NOT NULL DEFAULT 1,
    is_paused INTEGER NOT NULL DEFAULT 0 CHECK (is_paused IN (0, 1)),
    is_draining INTEGER NOT NULL DEFAULT 0 CHECK (is_draining IN (0, 1)),
    default_lease_ttl_sec INTEGER NOT NULL DEFAULT 900 CHECK (default_lease_ttl_sec > 0),
    strict_projection INTEGER NOT NULL DEFAULT 1 CHECK (strict_projection IN (0, 1)),
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_workflow_policies_board ON board_workflow_policies(board_id);

-- Workflow Status Catalog
CREATE TABLE IF NOT EXISTS board_workflow_statuses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id INTEGER NOT NULL REFERENCES board_workflow_policies(id) ON DELETE CASCADE,
    status_key TEXT NOT NULL,
    display_name TEXT NOT NULL,
    is_terminal INTEGER NOT NULL DEFAULT 0 CHECK (is_terminal IN (0, 1)),
    sort_order INTEGER NOT NULL DEFAULT 0,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(policy_id, status_key)
);
CREATE INDEX IF NOT EXISTS idx_workflow_statuses_policy ON board_workflow_statuses(policy_id);

-- Workflow Transition Edges
CREATE TABLE IF NOT EXISTS board_workflow_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id INTEGER NOT NULL REFERENCES board_workflow_policies(id) ON DELETE CASCADE,
    from_status_key TEXT NOT NULL,
    to_status_key TEXT NOT NULL,
    requires_claim INTEGER NOT NULL DEFAULT 1 CHECK (requires_claim IN (0, 1)),
    requires_approval INTEGER NOT NULL DEFAULT 0 CHECK (requires_approval IN (0, 1)),
    approve_to_status_key TEXT,
    reject_to_status_key TEXT,
    auto_move_list_id INTEGER REFERENCES kanban_lists(id) ON DELETE SET NULL,
    max_retries INTEGER NOT NULL DEFAULT 0 CHECK(max_retries >= 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(policy_id, from_status_key, to_status_key),
    FOREIGN KEY(policy_id, from_status_key)
        REFERENCES board_workflow_statuses(policy_id, status_key) ON DELETE CASCADE,
    FOREIGN KEY(policy_id, to_status_key)
        REFERENCES board_workflow_statuses(policy_id, status_key) ON DELETE CASCADE,
    FOREIGN KEY(policy_id, approve_to_status_key)
        REFERENCES board_workflow_statuses(policy_id, status_key) ON DELETE SET NULL,
    FOREIGN KEY(policy_id, reject_to_status_key)
        REFERENCES board_workflow_statuses(policy_id, status_key) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_workflow_transitions_policy ON board_workflow_transitions(policy_id);

-- Card Workflow Runtime State
CREATE TABLE IF NOT EXISTS kanban_card_workflow_state (
    card_id INTEGER PRIMARY KEY REFERENCES kanban_cards(id) ON DELETE CASCADE,
    policy_id INTEGER NOT NULL REFERENCES board_workflow_policies(id) ON DELETE CASCADE,
    workflow_status_key TEXT NOT NULL,
    lease_owner TEXT,
    lease_expires_at TIMESTAMP,
    approval_state TEXT NOT NULL DEFAULT 'none'
        CHECK (approval_state IN ('none', 'awaiting_approval', 'approved', 'rejected')),
    pending_transition_id INTEGER REFERENCES board_workflow_transitions(id) ON DELETE SET NULL,
    retry_counters JSON,
    last_transition_at TIMESTAMP,
    last_actor TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(policy_id, workflow_status_key)
        REFERENCES board_workflow_statuses(policy_id, status_key) ON DELETE RESTRICT
);
CREATE INDEX IF NOT EXISTS idx_card_workflow_state_policy ON kanban_card_workflow_state(policy_id);
CREATE INDEX IF NOT EXISTS idx_card_workflow_state_status ON kanban_card_workflow_state(workflow_status_key);
CREATE INDEX IF NOT EXISTS idx_card_workflow_state_lease ON kanban_card_workflow_state(lease_expires_at);

-- Card Workflow Events (append-only)
CREATE TABLE IF NOT EXISTS kanban_card_workflow_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    from_status_key TEXT,
    to_status_key TEXT,
    actor TEXT NOT NULL,
    reason TEXT,
    idempotency_key TEXT NOT NULL,
    correlation_id TEXT,
    before_snapshot JSON,
    after_snapshot JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(card_id, event_type, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_card_workflow_events_card_created ON kanban_card_workflow_events(card_id, created_at DESC);

-- Card Workflow Approvals
CREATE TABLE IF NOT EXISTS kanban_card_workflow_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
    transition_id INTEGER NOT NULL REFERENCES board_workflow_transitions(id) ON DELETE CASCADE,
    state TEXT NOT NULL CHECK (state IN ('pending', 'approved', 'rejected')),
    reviewer TEXT,
    decision_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_card_workflow_approvals_card ON kanban_card_workflow_approvals(card_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_card_workflow_approvals_pending_unique
    ON kanban_card_workflow_approvals(card_id, transition_id)
    WHERE state = 'pending';

-- Card Links
CREATE TABLE IF NOT EXISTS kanban_card_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE NOT NULL,
    card_id INTEGER NOT NULL REFERENCES kanban_cards(id) ON DELETE CASCADE,
    linked_type TEXT NOT NULL CHECK (linked_type IN ('media', 'note')),
    linked_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(card_id, linked_type, linked_id)
);
CREATE INDEX IF NOT EXISTS idx_card_links_card ON kanban_card_links(card_id);
CREATE INDEX IF NOT EXISTS idx_card_links_linked ON kanban_card_links(linked_type, linked_id);

-- FTS5 for card search
CREATE VIRTUAL TABLE IF NOT EXISTS kanban_cards_fts USING fts5(
    title,
    description,
    content='kanban_cards',
    content_rowid='id'
);

-- Triggers for FTS sync (handles archive and soft delete)
-- Only index cards that are neither archived nor deleted
DROP TRIGGER IF EXISTS kanban_cards_ai;
CREATE TRIGGER kanban_cards_ai AFTER INSERT ON kanban_cards
WHEN NEW.deleted = 0 AND NEW.archived = 0 BEGIN
    INSERT INTO kanban_cards_fts(rowid, title, description)
    VALUES (NEW.id, NEW.title, NEW.description);
END;

DROP TRIGGER IF EXISTS kanban_cards_ad;
CREATE TRIGGER kanban_cards_ad AFTER DELETE ON kanban_cards BEGIN
    INSERT INTO kanban_cards_fts(kanban_cards_fts, rowid, title, description)
    VALUES ('delete', OLD.id, OLD.title, OLD.description);
END;

-- Handle content updates, archive, and soft delete/restore
DROP TRIGGER IF EXISTS kanban_cards_au;
CREATE TRIGGER kanban_cards_au AFTER UPDATE ON kanban_cards BEGIN
    -- Only remove old entry if it was previously indexed (not deleted and not archived)
    INSERT INTO kanban_cards_fts(kanban_cards_fts, rowid, title, description)
    SELECT 'delete', OLD.id, OLD.title, OLD.description
    WHERE OLD.deleted = 0 AND OLD.archived = 0;
    -- Only re-add if not deleted AND not archived
    INSERT INTO kanban_cards_fts(rowid, title, description)
    SELECT NEW.id, NEW.title, NEW.description
    WHERE NEW.deleted = 0 AND NEW.archived = 0;
END;
"""

    # =========================================================================
    # WORKFLOW OPERATIONS
    # =========================================================================

    @staticmethod
    def _coerce_bool_int(value: Any, *, default: bool = False) -> int:
        """Coerce a boolean-like value to SQLite integer form."""
        if value is None:
            return 1 if default else 0
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int):
            return 1 if value != 0 else 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return 1
            if normalized in {"0", "false", "no", "off", ""}:
                return 0
        raise InputError("Boolean-like value required")  # noqa: TRY003

    def _default_workflow_statuses(self) -> list[dict[str, Any]]:
        return [
            {
                "status_key": status_key,
                "display_name": display_name,
                "sort_order": sort_order,
                "is_terminal": is_terminal,
                "is_active": is_active,
            }
            for status_key, display_name, sort_order, is_terminal, is_active in self.DEFAULT_WORKFLOW_STATUSES
        ]

    def _default_workflow_transitions(self) -> list[dict[str, Any]]:
        return [
            {
                "from_status_key": from_status_key,
                "to_status_key": to_status_key,
                "requires_claim": requires_claim,
                "requires_approval": requires_approval,
                "approve_to_status_key": approve_to_status_key,
                "reject_to_status_key": reject_to_status_key,
                "auto_move_list_id": auto_move_list_id,
                "max_retries": max_retries,
                "is_active": is_active,
            }
            for (
                from_status_key,
                to_status_key,
                requires_claim,
                requires_approval,
                approve_to_status_key,
                reject_to_status_key,
                auto_move_list_id,
                max_retries,
                is_active,
            ) in self.DEFAULT_WORKFLOW_TRANSITIONS
        ]

    def _normalize_workflow_statuses(
        self,
        statuses: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        raw_statuses = statuses if statuses is not None else self._default_workflow_statuses()
        if not raw_statuses:
            raise InputError("At least one workflow status is required")  # noqa: TRY003

        normalized: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for index, raw in enumerate(raw_statuses):
            status_key = str(raw.get("status_key") or "").strip()
            if not status_key:
                raise InputError("status_key is required for all statuses")  # noqa: TRY003
            if status_key in seen_keys:
                raise InputError(f"Duplicate workflow status_key: {status_key}")  # noqa: TRY003
            seen_keys.add(status_key)

            display_name = str(raw.get("display_name") or "").strip() or status_key
            sort_order = int(raw.get("sort_order", index))
            normalized.append(
                {
                    "status_key": status_key,
                    "display_name": display_name,
                    "sort_order": sort_order,
                    "is_terminal": self._coerce_bool_int(raw.get("is_terminal"), default=False),
                    "is_active": self._coerce_bool_int(raw.get("is_active"), default=True),
                }
            )
        return normalized

    def _normalize_workflow_transitions(
        self,
        transitions: list[dict[str, Any]] | None,
        *,
        valid_status_keys: set[str],
    ) -> list[dict[str, Any]]:
        raw_transitions = transitions if transitions is not None else self._default_workflow_transitions()
        normalized: list[dict[str, Any]] = []
        seen_edges: set[tuple[str, str]] = set()

        for raw in raw_transitions:
            from_status_key = str(raw.get("from_status_key") or "").strip()
            to_status_key = str(raw.get("to_status_key") or "").strip()
            if not from_status_key or not to_status_key:
                raise InputError("from_status_key and to_status_key are required for transitions")  # noqa: TRY003
            if from_status_key not in valid_status_keys:
                raise InputError(f"Unknown transition from_status_key: {from_status_key}")  # noqa: TRY003
            if to_status_key not in valid_status_keys:
                raise InputError(f"Unknown transition to_status_key: {to_status_key}")  # noqa: TRY003

            edge_key = (from_status_key, to_status_key)
            if edge_key in seen_edges:
                raise InputError(f"Duplicate transition edge: {from_status_key} -> {to_status_key}")  # noqa: TRY003
            seen_edges.add(edge_key)

            approve_to_status_key = raw.get("approve_to_status_key")
            reject_to_status_key = raw.get("reject_to_status_key")
            if approve_to_status_key is not None and str(approve_to_status_key).strip() not in valid_status_keys:
                raise InputError("approve_to_status_key must reference a known status_key")  # noqa: TRY003
            if reject_to_status_key is not None and str(reject_to_status_key).strip() not in valid_status_keys:
                raise InputError("reject_to_status_key must reference a known status_key")  # noqa: TRY003

            normalized.append(
                {
                    "from_status_key": from_status_key,
                    "to_status_key": to_status_key,
                    "requires_claim": self._coerce_bool_int(raw.get("requires_claim"), default=True),
                    "requires_approval": self._coerce_bool_int(raw.get("requires_approval"), default=False),
                    "approve_to_status_key": str(approve_to_status_key).strip() if approve_to_status_key else None,
                    "reject_to_status_key": str(reject_to_status_key).strip() if reject_to_status_key else None,
                    "auto_move_list_id": raw.get("auto_move_list_id"),
                    "max_retries": max(0, int(raw.get("max_retries", 0))),
                    "is_active": self._coerce_bool_int(raw.get("is_active"), default=True),
                }
            )

        return normalized

    def _row_to_workflow_policy_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "board_id": row["board_id"],
            "version": row["version"],
            "is_paused": bool(row["is_paused"]),
            "is_draining": bool(row["is_draining"]),
            "default_lease_ttl_sec": row["default_lease_ttl_sec"],
            "strict_projection": bool(row["strict_projection"]),
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_workflow_status_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "policy_id": row["policy_id"],
            "status_key": row["status_key"],
            "display_name": row["display_name"],
            "is_terminal": bool(row["is_terminal"]),
            "sort_order": row["sort_order"],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_workflow_transition_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "policy_id": row["policy_id"],
            "from_status_key": row["from_status_key"],
            "to_status_key": row["to_status_key"],
            "requires_claim": bool(row["requires_claim"]),
            "requires_approval": bool(row["requires_approval"]),
            "approve_to_status_key": row["approve_to_status_key"],
            "reject_to_status_key": row["reject_to_status_key"],
            "auto_move_list_id": row["auto_move_list_id"],
            "max_retries": row["max_retries"],
            "is_active": bool(row["is_active"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _row_to_card_workflow_state_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        retry_counters = row["retry_counters"]
        return {
            "card_id": row["card_id"],
            "policy_id": row["policy_id"],
            "workflow_status_key": row["workflow_status_key"],
            "lease_owner": row["lease_owner"],
            "lease_expires_at": row["lease_expires_at"],
            "approval_state": row["approval_state"],
            "pending_transition_id": row["pending_transition_id"],
            "retry_counters": json.loads(retry_counters) if retry_counters else None,
            "last_transition_at": row["last_transition_at"],
            "last_actor": row["last_actor"],
            "version": row["version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _get_workflow_policy_row(self, conn: sqlite3.Connection, board_id: int) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT p.id, p.board_id, p.version, p.is_paused, p.is_draining,
                   p.default_lease_ttl_sec, p.strict_projection, p.metadata,
                   p.created_at, p.updated_at
            FROM board_workflow_policies p
            JOIN kanban_boards b ON b.id = p.board_id
            WHERE p.board_id = ? AND b.user_id = ? AND b.deleted = 0
            """,
            (board_id, self.user_id),
        ).fetchone()

    def _list_workflow_statuses_for_policy(
        self,
        conn: sqlite3.Connection,
        policy_id: int,
        *,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT id, policy_id, status_key, display_name, is_terminal, sort_order, is_active, created_at, updated_at
            FROM board_workflow_statuses
            WHERE policy_id = ?
        """
        params: list[Any] = [policy_id]
        if not include_inactive:
            sql += " AND is_active = 1"
        sql += " ORDER BY sort_order ASC, status_key ASC"
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_workflow_status_dict(row) for row in rows]

    def _list_workflow_transitions_for_policy(
        self,
        conn: sqlite3.Connection,
        policy_id: int,
        *,
        include_inactive: bool = False,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT id, policy_id, from_status_key, to_status_key, requires_claim, requires_approval,
                   approve_to_status_key, reject_to_status_key, auto_move_list_id, max_retries,
                   is_active, created_at, updated_at
            FROM board_workflow_transitions
            WHERE policy_id = ?
        """
        params: list[Any] = [policy_id]
        if not include_inactive:
            sql += " AND is_active = 1"
        sql += " ORDER BY from_status_key ASC, to_status_key ASC"
        rows = conn.execute(sql, params).fetchall()
        return [self._row_to_workflow_transition_dict(row) for row in rows]

    def _upsert_workflow_policy_internal(
        self,
        conn: sqlite3.Connection,
        *,
        board_id: int,
        statuses: list[dict[str, Any]] | None = None,
        transitions: list[dict[str, Any]] | None = None,
        is_paused: bool = False,
        is_draining: bool = False,
        default_lease_ttl_sec: int = 900,
        strict_projection: bool = True,
        metadata: dict[str, Any] | None | object = _WORKFLOW_METADATA_UNSET,
    ) -> dict[str, Any]:
        board = self._get_board_by_id(conn, board_id)
        if not board:
            raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003

        if default_lease_ttl_sec <= 0:
            raise InputError("default_lease_ttl_sec must be greater than zero")  # noqa: TRY003

        normalized_statuses = self._normalize_workflow_statuses(statuses)
        status_keys = {status["status_key"] for status in normalized_statuses}
        normalized_transitions = self._normalize_workflow_transitions(
            transitions,
            valid_status_keys=status_keys,
        )

        now = _utcnow_iso()
        policy_row = self._get_workflow_policy_row(conn, board_id)
        if metadata is _WORKFLOW_METADATA_UNSET:
            metadata_json = policy_row["metadata"] if policy_row else None
        else:
            metadata_json = json.dumps(metadata) if metadata is not None else None

        if policy_row:
            policy_id = policy_row["id"]
            conn.execute(
                """
                UPDATE board_workflow_policies
                SET version = version + 1,
                    is_paused = ?,
                    is_draining = ?,
                    default_lease_ttl_sec = ?,
                    strict_projection = ?,
                    metadata = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    self._coerce_bool_int(is_paused),
                    self._coerce_bool_int(is_draining),
                    int(default_lease_ttl_sec),
                    self._coerce_bool_int(strict_projection, default=True),
                    metadata_json,
                    now,
                    policy_id,
                ),
            )
        else:
            cur = conn.execute(
                """
                INSERT INTO board_workflow_policies
                (board_id, version, is_paused, is_draining, default_lease_ttl_sec,
                 strict_projection, metadata, created_at, updated_at)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    board_id,
                    self._coerce_bool_int(is_paused),
                    self._coerce_bool_int(is_draining),
                    int(default_lease_ttl_sec),
                    self._coerce_bool_int(strict_projection, default=True),
                    metadata_json,
                    now,
                    now,
                ),
            )
            policy_id = int(cur.lastrowid)

        for status in normalized_statuses:
            conn.execute(
                """
                INSERT INTO board_workflow_statuses
                (policy_id, status_key, display_name, is_terminal, sort_order, is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(policy_id, status_key) DO UPDATE SET
                    display_name = excluded.display_name,
                    is_terminal = excluded.is_terminal,
                    sort_order = excluded.sort_order,
                    is_active = excluded.is_active,
                    updated_at = excluded.updated_at
                """,
                (
                    policy_id,
                    status["status_key"],
                    status["display_name"],
                    status["is_terminal"],
                    status["sort_order"],
                    status["is_active"],
                    now,
                    now,
                ),
            )

        status_placeholders = ",".join("?" * len(status_keys))
        conn.execute(
            f"""
            UPDATE board_workflow_statuses
            SET is_active = 0, updated_at = ?
            WHERE policy_id = ? AND status_key NOT IN ({status_placeholders})
            """,  # nosec B608
            [now, policy_id, *status_keys],
        )

        for transition in normalized_transitions:
            conn.execute(
                """
                INSERT INTO board_workflow_transitions
                (policy_id, from_status_key, to_status_key, requires_claim, requires_approval,
                 approve_to_status_key, reject_to_status_key, auto_move_list_id, max_retries,
                 is_active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(policy_id, from_status_key, to_status_key) DO UPDATE SET
                    requires_claim = excluded.requires_claim,
                    requires_approval = excluded.requires_approval,
                    approve_to_status_key = excluded.approve_to_status_key,
                    reject_to_status_key = excluded.reject_to_status_key,
                    auto_move_list_id = excluded.auto_move_list_id,
                    max_retries = excluded.max_retries,
                    is_active = excluded.is_active,
                    updated_at = excluded.updated_at
                """,
                (
                    policy_id,
                    transition["from_status_key"],
                    transition["to_status_key"],
                    transition["requires_claim"],
                    transition["requires_approval"],
                    transition["approve_to_status_key"],
                    transition["reject_to_status_key"],
                    transition["auto_move_list_id"],
                    transition["max_retries"],
                    transition["is_active"],
                    now,
                    now,
                ),
            )

        if normalized_transitions:
            transition_clauses = " OR ".join(["(from_status_key = ? AND to_status_key = ?)"] * len(normalized_transitions))
            transition_params: list[Any] = [now, policy_id]
            for transition in normalized_transitions:
                transition_params.extend([transition["from_status_key"], transition["to_status_key"]])
            conn.execute(
                f"""
                UPDATE board_workflow_transitions
                SET is_active = 0, updated_at = ?
                WHERE policy_id = ? AND NOT ({transition_clauses})
                """,  # nosec B608
                transition_params,
            )
        else:
            conn.execute(
                "UPDATE board_workflow_transitions SET is_active = 0, updated_at = ? WHERE policy_id = ?",
                (now, policy_id),
            )

        return self._get_workflow_policy_internal(conn, board_id)

    def _ensure_workflow_policy_for_board(self, conn: sqlite3.Connection, board_id: int) -> dict[str, Any]:
        policy = self._get_workflow_policy_internal(conn, board_id)
        if policy:
            return policy
        return self._upsert_workflow_policy_internal(conn, board_id=board_id)

    def _get_workflow_policy_internal(self, conn: sqlite3.Connection, board_id: int) -> dict[str, Any] | None:
        row = self._get_workflow_policy_row(conn, board_id)
        if not row:
            return None

        policy = self._row_to_workflow_policy_dict(row)
        policy_id = int(row["id"])
        policy["statuses"] = self._list_workflow_statuses_for_policy(conn, policy_id)
        policy["transitions"] = self._list_workflow_transitions_for_policy(conn, policy_id)
        return policy

    def upsert_workflow_policy(
        self,
        *,
        board_id: int,
        statuses: list[dict[str, Any]] | None = None,
        transitions: list[dict[str, Any]] | None = None,
        is_paused: bool = False,
        is_draining: bool = False,
        default_lease_ttl_sec: int = 900,
        strict_projection: bool = True,
        metadata: dict[str, Any] | None | object = _WORKFLOW_METADATA_UNSET,
    ) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                policy = self._upsert_workflow_policy_internal(
                    conn,
                    board_id=board_id,
                    statuses=statuses,
                    transitions=transitions,
                    is_paused=is_paused,
                    is_draining=is_draining,
                    default_lease_ttl_sec=default_lease_ttl_sec,
                    strict_projection=strict_projection,
                    metadata=metadata,
                )
                conn.commit()
                return policy
            finally:
                conn.close()

    def update_workflow_policy_flags(
        self,
        *,
        board_id: int,
        is_paused: bool | None = None,
        is_draining: bool | None = None,
    ) -> dict[str, Any]:
        """Update policy control flags without rewriting statuses/transitions."""
        with self._lock:
            conn = self._connect()
            try:
                policy = self._ensure_workflow_policy_for_board(conn, board_id)
                policy_row = self._get_workflow_policy_row(conn, board_id)
                if not policy_row:
                    raise NotFoundError("Workflow policy not found", entity="workflow_policy", entity_id=board_id)  # noqa: TRY003

                next_paused = bool(policy["is_paused"]) if is_paused is None else bool(is_paused)
                next_draining = bool(policy["is_draining"]) if is_draining is None else bool(is_draining)
                now = _utcnow_iso()
                conn.execute(
                    """
                    UPDATE board_workflow_policies
                    SET version = version + 1,
                        is_paused = ?,
                        is_draining = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        self._coerce_bool_int(next_paused),
                        self._coerce_bool_int(next_draining),
                        now,
                        int(policy_row["id"]),
                    ),
                )
                updated = self._get_workflow_policy_internal(conn, board_id)
                if not updated:
                    raise KanbanDBError("workflow_policy_update_failed")  # noqa: TRY003
                conn.commit()
                return updated
            finally:
                conn.close()

    def get_workflow_policy(self, board_id: int) -> dict[str, Any] | None:
        with self._lock:
            conn = self._connect()
            try:
                policy = self._ensure_workflow_policy_for_board(conn, board_id)
                conn.commit()
                return policy
            finally:
                conn.close()

    def list_workflow_statuses(self, board_id: int) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                policy = self._ensure_workflow_policy_for_board(conn, board_id)
                statuses = self._list_workflow_statuses_for_policy(conn, int(policy["id"]))
                conn.commit()
                return statuses
            finally:
                conn.close()

    def list_workflow_transitions(self, board_id: int) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                policy = self._ensure_workflow_policy_for_board(conn, board_id)
                transitions = self._list_workflow_transitions_for_policy(conn, int(policy["id"]))
                conn.commit()
                return transitions
            finally:
                conn.close()

    def _get_card_workflow_state_row(self, conn: sqlite3.Connection, card_id: int) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT s.card_id, s.policy_id, s.workflow_status_key, s.lease_owner, s.lease_expires_at,
                   s.approval_state, s.pending_transition_id, s.retry_counters, s.last_transition_at,
                   s.last_actor, s.version, s.created_at, s.updated_at
            FROM kanban_card_workflow_state s
            JOIN kanban_cards c ON c.id = s.card_id
            JOIN kanban_boards b ON b.id = c.board_id
            WHERE s.card_id = ? AND b.user_id = ? AND b.deleted = 0 AND c.deleted = 0
            """,
            (card_id, self.user_id),
        ).fetchone()

    def _get_default_policy_status_key(self, conn: sqlite3.Connection, policy_id: int) -> str:
        row = conn.execute(
            """
            SELECT status_key
            FROM board_workflow_statuses
            WHERE policy_id = ? AND is_active = 1
            ORDER BY sort_order ASC, status_key ASC
            LIMIT 1
            """,
            (policy_id,),
        ).fetchone()
        if not row:
            raise NotFoundError("No active workflow statuses found for policy", entity="workflow_policy", entity_id=policy_id)  # noqa: TRY003
        return str(row["status_key"])

    def _ensure_card_workflow_state(self, conn: sqlite3.Connection, card_id: int) -> sqlite3.Row:
        existing_state = self._get_card_workflow_state_row(conn, card_id)
        if existing_state:
            return existing_state

        card = self._get_card_by_id(conn, card_id)
        if not card:
            raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

        policy = self._ensure_workflow_policy_for_board(conn, int(card["board_id"]))
        policy_id = int(policy["id"])
        default_status_key = self._get_default_policy_status_key(conn, policy_id)
        now = _utcnow_iso()

        try:
            conn.execute(
                """
                INSERT INTO kanban_card_workflow_state
                (card_id, policy_id, workflow_status_key, approval_state, version, created_at, updated_at)
                VALUES (?, ?, ?, 'none', 1, ?, ?)
                """,
                (card_id, policy_id, default_status_key, now, now),
            )
        except sqlite3.IntegrityError:
            # Lost a race to another initializer; fetch the row created by the winner.
            pass

        state_row = self._get_card_workflow_state_row(conn, card_id)
        if not state_row:
            raise KanbanDBError("Failed to initialize card workflow state")  # noqa: TRY003
        return state_row

    def get_card_workflow_state(self, card_id: int) -> dict[str, Any]:
        with self._lock:
            conn = self._connect()
            try:
                state_row = self._ensure_card_workflow_state(conn, card_id)
                conn.commit()
                return self._row_to_card_workflow_state_dict(state_row)
            finally:
                conn.close()

    def patch_card_workflow_state(
        self,
        *,
        card_id: int,
        workflow_status_key: str | None,
        expected_version: int,
        lease_owner: str | None,
        idempotency_key: str,
        correlation_id: str | None = None,
        last_actor: str | None = None,
    ) -> dict[str, Any]:
        if expected_version < 1:
            raise InputError("expected_version must be >= 1")  # noqa: TRY003
        if not idempotency_key or not idempotency_key.strip():
            raise InputError("idempotency_key is required")  # noqa: TRY003

        actor = (last_actor or self.user_id).strip()
        corr_id = correlation_id.strip() if correlation_id and correlation_id.strip() else None

        with self._lock:
            conn = self._connect()
            try:
                state_row = self._ensure_card_workflow_state(conn, card_id)
                current_state = self._row_to_card_workflow_state_dict(state_row)

                replay = conn.execute(
                    """
                    SELECT id
                    FROM kanban_card_workflow_events
                    WHERE card_id = ? AND event_type = 'state_patched' AND idempotency_key = ?
                    """,
                    (card_id, idempotency_key.strip()),
                ).fetchone()
                if replay:
                    conn.commit()
                    return current_state

                if int(current_state["version"]) != int(expected_version):
                    raise ConflictError(  # noqa: TRY003
                        f"version_conflict: expected {expected_version}, got {current_state['version']}",
                        entity="card_workflow_state",
                        entity_id=card_id,
                        code="version_conflict",
                    )

                next_status_key = workflow_status_key.strip() if workflow_status_key else current_state["workflow_status_key"]
                status_row = conn.execute(
                    """
                    SELECT status_key
                    FROM board_workflow_statuses
                    WHERE policy_id = ? AND status_key = ? AND is_active = 1
                    """,
                    (current_state["policy_id"], next_status_key),
                ).fetchone()
                if not status_row:
                    raise InputError("workflow_status_key is not valid for this policy")  # noqa: TRY003

                now = _utcnow_iso()
                update_cur = conn.execute(
                    """
                    UPDATE kanban_card_workflow_state
                    SET workflow_status_key = ?,
                        lease_owner = ?,
                        last_transition_at = ?,
                        last_actor = ?,
                        version = version + 1,
                        updated_at = ?
                    WHERE card_id = ? AND version = ?
                    """,
                    (
                        next_status_key,
                        lease_owner,
                        now,
                        actor,
                        now,
                        card_id,
                        expected_version,
                    ),
                )
                if update_cur.rowcount != 1:
                    raise ConflictError(  # noqa: TRY003
                        "version_conflict: workflow state update conflict",
                        entity="card_workflow_state",
                        entity_id=card_id,
                        code="version_conflict",
                    )

                updated_row = self._get_card_workflow_state_row(conn, card_id)
                if not updated_row:
                    raise KanbanDBError("Updated workflow state could not be loaded")  # noqa: TRY003
                updated_state = self._row_to_card_workflow_state_dict(updated_row)

                conn.execute(
                    """
                    INSERT INTO kanban_card_workflow_events
                    (card_id, event_type, from_status_key, to_status_key, actor, idempotency_key,
                     correlation_id, before_snapshot, after_snapshot, created_at)
                    VALUES (?, 'state_patched', ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card_id,
                        current_state["workflow_status_key"],
                        updated_state["workflow_status_key"],
                        actor,
                        idempotency_key.strip(),
                        corr_id,
                        json.dumps(current_state),
                        json.dumps(updated_state),
                        now,
                    ),
                )

                conn.commit()
                return updated_state
            finally:
                conn.close()

    def claim_card_workflow(
        self,
        *,
        card_id: int,
        owner: str,
        lease_ttl_sec: int | None = None,
        idempotency_key: str,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        if not owner or not owner.strip():
            raise InputError("owner is required")  # noqa: TRY003
        if not idempotency_key or not idempotency_key.strip():
            raise InputError("idempotency_key is required")  # noqa: TRY003
        corr_id = correlation_id.strip() if correlation_id and correlation_id.strip() else None

        with self._lock:
            conn = self._connect()
            try:
                state_row = self._ensure_card_workflow_state(conn, card_id)
                current_state = self._row_to_card_workflow_state_dict(state_row)

                replay = conn.execute(
                    """
                    SELECT id
                    FROM kanban_card_workflow_events
                    WHERE card_id = ? AND event_type = 'workflow_claimed' AND idempotency_key = ?
                    """,
                    (card_id, idempotency_key.strip()),
                ).fetchone()
                if replay:
                    conn.commit()
                    return current_state

                if lease_ttl_sec is None:
                    ttl_row = conn.execute(
                        "SELECT default_lease_ttl_sec FROM board_workflow_policies WHERE id = ?",
                        (current_state["policy_id"],),
                    ).fetchone()
                    lease_ttl_sec = int(ttl_row["default_lease_ttl_sec"]) if ttl_row else 900
                if lease_ttl_sec <= 0:
                    raise InputError("lease_ttl_sec must be greater than zero")  # noqa: TRY003

                now = _utcnow_iso()
                lease_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=lease_ttl_sec)).strftime("%Y-%m-%d %H:%M:%S")
                lease_owner = current_state["lease_owner"]
                lease_expiry = current_state["lease_expires_at"]
                owner_name = owner.strip()

                if lease_owner and lease_owner != owner_name and lease_expiry and str(lease_expiry) > now:
                    raise ConflictError(  # noqa: TRY003
                        "lease_mismatch",
                        entity="card_workflow_state",
                        entity_id=card_id,
                        code="lease_mismatch",
                    )

                update_cur = conn.execute(
                    """
                    UPDATE kanban_card_workflow_state
                    SET lease_owner = ?,
                        lease_expires_at = ?,
                        last_actor = ?,
                        version = version + 1,
                        updated_at = ?
                    WHERE card_id = ? AND version = ?
                      AND (
                        lease_owner IS NULL
                        OR lease_expires_at IS NULL
                        OR lease_expires_at <= ?
                        OR lease_owner = ?
                      )
                    """,
                    (owner_name, lease_expires_at, owner_name, now, card_id, current_state["version"], now, owner_name),
                )
                if update_cur.rowcount != 1:
                    latest_row = self._get_card_workflow_state_row(conn, card_id)
                    latest_state = self._row_to_card_workflow_state_dict(latest_row) if latest_row else None
                    if latest_state:
                        latest_owner = latest_state["lease_owner"]
                        latest_expiry = latest_state["lease_expires_at"]
                        if latest_owner and latest_owner != owner_name and latest_expiry and str(latest_expiry) > now:
                            raise ConflictError(  # noqa: TRY003
                                "lease_mismatch",
                                entity="card_workflow_state",
                                entity_id=card_id,
                                code="lease_mismatch",
                            )
                    raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id, code="version_conflict")  # noqa: TRY003

                updated_row = self._get_card_workflow_state_row(conn, card_id)
                if not updated_row:
                    raise KanbanDBError("Updated workflow state could not be loaded")  # noqa: TRY003
                updated_state = self._row_to_card_workflow_state_dict(updated_row)

                conn.execute(
                    """
                    INSERT INTO kanban_card_workflow_events
                    (card_id, event_type, from_status_key, to_status_key, actor, reason, idempotency_key,
                     correlation_id, before_snapshot, after_snapshot, created_at)
                    VALUES (?, 'workflow_claimed', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card_id,
                        current_state["workflow_status_key"],
                        updated_state["workflow_status_key"],
                        owner.strip(),
                        "claim_acquired",
                        idempotency_key.strip(),
                        corr_id,
                        json.dumps(current_state),
                        json.dumps(updated_state),
                        now,
                    ),
                )

                conn.commit()
                return updated_state
            finally:
                conn.close()

    def release_card_workflow(
        self,
        *,
        card_id: int,
        owner: str,
        idempotency_key: str,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        if not owner or not owner.strip():
            raise InputError("owner is required")  # noqa: TRY003
        if not idempotency_key or not idempotency_key.strip():
            raise InputError("idempotency_key is required")  # noqa: TRY003
        corr_id = correlation_id.strip() if correlation_id and correlation_id.strip() else None

        with self._lock:
            conn = self._connect()
            try:
                state_row = self._ensure_card_workflow_state(conn, card_id)
                current_state = self._row_to_card_workflow_state_dict(state_row)

                replay = conn.execute(
                    """
                    SELECT id
                    FROM kanban_card_workflow_events
                    WHERE card_id = ? AND event_type = 'workflow_released' AND idempotency_key = ?
                    """,
                    (card_id, idempotency_key.strip()),
                ).fetchone()
                if replay:
                    conn.commit()
                    return current_state

                if current_state["lease_owner"] and current_state["lease_owner"] != owner.strip():
                    raise ConflictError("lease_mismatch", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                now = _utcnow_iso()
                update_cur = conn.execute(
                    """
                    UPDATE kanban_card_workflow_state
                    SET lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_actor = ?,
                        version = version + 1,
                        updated_at = ?
                    WHERE card_id = ? AND version = ?
                    """,
                    (owner.strip(), now, card_id, current_state["version"]),
                )
                if update_cur.rowcount != 1:
                    raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                updated_row = self._get_card_workflow_state_row(conn, card_id)
                if not updated_row:
                    raise KanbanDBError("Updated workflow state could not be loaded")  # noqa: TRY003
                updated_state = self._row_to_card_workflow_state_dict(updated_row)

                conn.execute(
                    """
                    INSERT INTO kanban_card_workflow_events
                    (card_id, event_type, from_status_key, to_status_key, actor, reason, idempotency_key,
                     correlation_id, before_snapshot, after_snapshot, created_at)
                    VALUES (?, 'workflow_released', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card_id,
                        current_state["workflow_status_key"],
                        updated_state["workflow_status_key"],
                        owner.strip(),
                        "claim_released",
                        idempotency_key.strip(),
                        corr_id,
                        json.dumps(current_state),
                        json.dumps(updated_state),
                        now,
                    ),
                )

                conn.commit()
                return updated_state
            finally:
                conn.close()

    def transition_card_workflow(
        self,
        *,
        card_id: int,
        to_status_key: str,
        actor: str,
        expected_version: int,
        idempotency_key: str,
        correlation_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if not to_status_key or not to_status_key.strip():
            raise InputError("to_status_key is required")  # noqa: TRY003
        if not actor or not actor.strip():
            raise InputError("actor is required")  # noqa: TRY003
        if expected_version < 1:
            raise InputError("expected_version must be >= 1")  # noqa: TRY003
        if not idempotency_key or not idempotency_key.strip():
            raise InputError("idempotency_key is required")  # noqa: TRY003

        target_status_key = to_status_key.strip()
        actor_name = actor.strip()
        corr_id = correlation_id.strip() if correlation_id and correlation_id.strip() else None

        with self._lock:
            conn = self._connect()
            try:
                state_row = self._ensure_card_workflow_state(conn, card_id)
                current_state = self._row_to_card_workflow_state_dict(state_row)
                card_row = conn.execute(
                    """
                    SELECT id, board_id, list_id
                    FROM kanban_cards
                    WHERE id = ? AND deleted = 0
                    """,
                    (card_id,),
                ).fetchone()
                if not card_row:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                policy_row = conn.execute(
                    "SELECT is_paused, strict_projection FROM board_workflow_policies WHERE id = ?",
                    (current_state["policy_id"],),
                ).fetchone()
                if policy_row and bool(policy_row["is_paused"]):
                    raise ConflictError("policy_paused", entity="workflow_policy", entity_id=current_state["policy_id"])  # noqa: TRY003

                replay = conn.execute(
                    """
                    SELECT id
                    FROM kanban_card_workflow_events
                    WHERE card_id = ?
                      AND event_type IN ('workflow_transitioned', 'workflow_approval_requested')
                      AND idempotency_key = ?
                    """,
                    (card_id, idempotency_key.strip()),
                ).fetchone()
                if replay:
                    conn.commit()
                    return current_state

                if int(current_state["version"]) != int(expected_version):
                    raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                transition_row = conn.execute(
                    """
                    SELECT id, requires_claim, requires_approval, is_active, auto_move_list_id
                    FROM board_workflow_transitions
                    WHERE policy_id = ? AND from_status_key = ? AND to_status_key = ? AND is_active = 1
                    """,
                    (current_state["policy_id"], current_state["workflow_status_key"], target_status_key),
                ).fetchone()
                if not transition_row:
                    raise ConflictError("transition_not_allowed", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                now = _utcnow_iso()
                if bool(transition_row["requires_claim"]):
                    lease_owner = current_state["lease_owner"]
                    lease_expires_at = current_state["lease_expires_at"]
                    if not lease_owner or lease_owner != actor_name:
                        raise ConflictError("lease_required", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003
                    if not lease_expires_at or str(lease_expires_at) <= now:
                        raise ConflictError("lease_required", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                projected_list_id: int | None = None
                if transition_row["auto_move_list_id"] is not None:
                    projected_row = conn.execute(
                        """
                        SELECT id
                        FROM kanban_lists
                        WHERE id = ? AND board_id = ? AND deleted = 0 AND archived = 0
                        """,
                        (transition_row["auto_move_list_id"], card_row["board_id"]),
                    ).fetchone()
                    if not projected_row and bool(policy_row and policy_row["strict_projection"]):
                        raise ConflictError("projection_failed", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003
                    if projected_row:
                        projected_list_id = int(projected_row["id"])

                if bool(transition_row["requires_approval"]):
                    update_cur = conn.execute(
                        """
                        UPDATE kanban_card_workflow_state
                        SET approval_state = 'awaiting_approval',
                            pending_transition_id = ?,
                            last_actor = ?,
                            version = version + 1,
                            updated_at = ?
                        WHERE card_id = ? AND version = ?
                        """,
                        (transition_row["id"], actor_name, now, card_id, expected_version),
                    )
                    if update_cur.rowcount != 1:
                        raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                    conn.execute(
                        """
                        UPDATE kanban_card_workflow_approvals
                        SET state = 'rejected', updated_at = ?, decision_reason = COALESCE(decision_reason, 'superseded')
                        WHERE card_id = ? AND state = 'pending'
                        """,
                        (now, card_id),
                    )
                    conn.execute(
                        """
                        INSERT INTO kanban_card_workflow_approvals
                        (card_id, transition_id, state, reviewer, decision_reason, created_at, updated_at)
                        VALUES (?, ?, 'pending', NULL, ?, ?, ?)
                        """,
                        (card_id, transition_row["id"], reason, now, now),
                    )

                    updated_row = self._get_card_workflow_state_row(conn, card_id)
                    if not updated_row:
                        raise KanbanDBError("Updated workflow state could not be loaded")  # noqa: TRY003
                    updated_state = self._row_to_card_workflow_state_dict(updated_row)

                    conn.execute(
                        """
                        INSERT INTO kanban_card_workflow_events
                        (card_id, event_type, from_status_key, to_status_key, actor, reason, idempotency_key,
                         correlation_id, before_snapshot, after_snapshot, created_at)
                        VALUES (?, 'workflow_approval_requested', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            card_id,
                            current_state["workflow_status_key"],
                            target_status_key,
                            actor_name,
                            reason,
                            idempotency_key.strip(),
                            corr_id,
                            json.dumps(current_state),
                            json.dumps(updated_state),
                            now,
                        ),
                    )

                    conn.commit()
                    return updated_state

                update_cur = conn.execute(
                    """
                    UPDATE kanban_card_workflow_state
                    SET workflow_status_key = ?,
                        approval_state = 'none',
                        pending_transition_id = NULL,
                        last_transition_at = ?,
                        last_actor = ?,
                        version = version + 1,
                        updated_at = ?
                    WHERE card_id = ? AND version = ?
                    """,
                    (target_status_key, now, actor_name, now, card_id, expected_version),
                )
                if update_cur.rowcount != 1:
                    raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                if projected_list_id is not None and int(card_row["list_id"]) != projected_list_id:
                    projected_update = conn.execute(
                        """
                        UPDATE kanban_cards
                        SET list_id = ?, version = version + 1, updated_at = ?
                        WHERE id = ? AND board_id = ? AND deleted = 0 AND archived = 0
                        """,
                        (projected_list_id, now, card_id, card_row["board_id"]),
                    )
                    if projected_update.rowcount != 1 and bool(policy_row["strict_projection"]):
                        raise ConflictError("projection_failed", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                updated_row = self._get_card_workflow_state_row(conn, card_id)
                if not updated_row:
                    raise KanbanDBError("Updated workflow state could not be loaded")  # noqa: TRY003
                updated_state = self._row_to_card_workflow_state_dict(updated_row)

                conn.execute(
                    """
                    INSERT INTO kanban_card_workflow_events
                    (card_id, event_type, from_status_key, to_status_key, actor, reason, idempotency_key,
                     correlation_id, before_snapshot, after_snapshot, created_at)
                    VALUES (?, 'workflow_transitioned', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card_id,
                        current_state["workflow_status_key"],
                        updated_state["workflow_status_key"],
                        actor_name,
                        reason,
                        idempotency_key.strip(),
                        corr_id,
                        json.dumps(current_state),
                        json.dumps(updated_state),
                        now,
                    ),
                )

                conn.commit()
                return updated_state
            finally:
                conn.close()

    def decide_card_workflow_approval(
        self,
        *,
        card_id: int,
        reviewer: str,
        decision: str,
        expected_version: int,
        idempotency_key: str,
        correlation_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if not reviewer or not reviewer.strip():
            raise InputError("reviewer is required")  # noqa: TRY003
        if decision not in {"approved", "rejected"}:
            raise InputError("decision must be 'approved' or 'rejected'")  # noqa: TRY003
        if expected_version < 1:
            raise InputError("expected_version must be >= 1")  # noqa: TRY003
        if not idempotency_key or not idempotency_key.strip():
            raise InputError("idempotency_key is required")  # noqa: TRY003

        reviewer_name = reviewer.strip()
        corr_id = correlation_id.strip() if correlation_id and correlation_id.strip() else None

        with self._lock:
            conn = self._connect()
            try:
                state_row = self._ensure_card_workflow_state(conn, card_id)
                current_state = self._row_to_card_workflow_state_dict(state_row)

                replay = conn.execute(
                    """
                    SELECT id
                    FROM kanban_card_workflow_events
                    WHERE card_id = ? AND event_type = 'workflow_approval_decided' AND idempotency_key = ?
                    """,
                    (card_id, idempotency_key.strip()),
                ).fetchone()
                if replay:
                    conn.commit()
                    return current_state

                if int(current_state["version"]) != int(expected_version):
                    raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003
                if current_state["approval_state"] != "awaiting_approval" or not current_state["pending_transition_id"]:
                    raise ConflictError("approval_required", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                transition_row = conn.execute(
                    """
                    SELECT id, to_status_key, approve_to_status_key, reject_to_status_key, auto_move_list_id
                    FROM board_workflow_transitions
                    WHERE id = ?
                    """,
                    (current_state["pending_transition_id"],),
                ).fetchone()
                if not transition_row:
                    raise ConflictError("transition_not_allowed", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                if decision == "approved":
                    target_status_key = transition_row["approve_to_status_key"] or transition_row["to_status_key"]
                    approval_state = "approved"
                else:
                    target_status_key = transition_row["reject_to_status_key"] or current_state["workflow_status_key"]
                    approval_state = "rejected"

                now = _utcnow_iso()
                card_row = conn.execute(
                    """
                    SELECT id, board_id, list_id
                    FROM kanban_cards
                    WHERE id = ? AND deleted = 0
                    """,
                    (card_id,),
                ).fetchone()
                if not card_row:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003
                policy_row = conn.execute(
                    "SELECT strict_projection FROM board_workflow_policies WHERE id = ?",
                    (current_state["policy_id"],),
                ).fetchone()
                projected_list_id: int | None = None
                if decision == "approved" and transition_row["auto_move_list_id"] is not None:
                    projected_row = conn.execute(
                        """
                        SELECT id
                        FROM kanban_lists
                        WHERE id = ? AND board_id = ? AND deleted = 0 AND archived = 0
                        """,
                        (transition_row["auto_move_list_id"], card_row["board_id"]),
                    ).fetchone()
                    if not projected_row and bool(policy_row and policy_row["strict_projection"]):
                        raise ConflictError("projection_failed", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003
                    if projected_row:
                        projected_list_id = int(projected_row["id"])

                conn.execute(
                    """
                    UPDATE kanban_card_workflow_approvals
                    SET state = ?, reviewer = ?, decision_reason = ?, updated_at = ?
                    WHERE card_id = ? AND transition_id = ? AND state = 'pending'
                    """,
                    (decision, reviewer_name, reason, now, card_id, transition_row["id"]),
                )

                update_cur = conn.execute(
                    """
                    UPDATE kanban_card_workflow_state
                    SET workflow_status_key = ?,
                        approval_state = ?,
                        pending_transition_id = NULL,
                        last_transition_at = ?,
                        last_actor = ?,
                        version = version + 1,
                        updated_at = ?
                    WHERE card_id = ? AND version = ?
                    """,
                    (target_status_key, approval_state, now, reviewer_name, now, card_id, expected_version),
                )
                if update_cur.rowcount != 1:
                    raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                if projected_list_id is not None and int(card_row["list_id"]) != projected_list_id:
                    projected_update = conn.execute(
                        """
                        UPDATE kanban_cards
                        SET list_id = ?, version = version + 1, updated_at = ?
                        WHERE id = ? AND board_id = ? AND deleted = 0 AND archived = 0
                        """,
                        (projected_list_id, now, card_id, card_row["board_id"]),
                    )
                    if projected_update.rowcount != 1 and bool(policy_row and policy_row["strict_projection"]):
                        raise ConflictError("projection_failed", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                updated_row = self._get_card_workflow_state_row(conn, card_id)
                if not updated_row:
                    raise KanbanDBError("Updated workflow state could not be loaded")  # noqa: TRY003
                updated_state = self._row_to_card_workflow_state_dict(updated_row)

                conn.execute(
                    """
                    INSERT INTO kanban_card_workflow_events
                    (card_id, event_type, from_status_key, to_status_key, actor, reason, idempotency_key,
                     correlation_id, before_snapshot, after_snapshot, created_at)
                    VALUES (?, 'workflow_approval_decided', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card_id,
                        current_state["workflow_status_key"],
                        updated_state["workflow_status_key"],
                        reviewer_name,
                        reason or decision,
                        idempotency_key.strip(),
                        corr_id,
                        json.dumps(current_state),
                        json.dumps(updated_state),
                        now,
                    ),
                )

                conn.commit()
                return updated_state
            finally:
                conn.close()

    def list_card_workflow_events(
        self,
        *,
        card_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise InputError("limit must be positive")  # noqa: TRY003
        if offset < 0:
            raise InputError("offset must be non-negative")  # noqa: TRY003

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT e.id, e.card_id, e.event_type, e.from_status_key, e.to_status_key, e.actor,
                           e.reason, e.idempotency_key, e.correlation_id, e.before_snapshot, e.after_snapshot,
                           e.created_at
                    FROM kanban_card_workflow_events e
                    JOIN kanban_cards c ON c.id = e.card_id
                    JOIN kanban_boards b ON b.id = c.board_id
                    WHERE e.card_id = ? AND b.user_id = ?
                    ORDER BY e.id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (card_id, self.user_id, limit, offset),
                ).fetchall()
                events: list[dict[str, Any]] = []
                for row in rows:
                    events.append(
                        {
                            "id": row["id"],
                            "card_id": row["card_id"],
                            "event_type": row["event_type"],
                            "from_status_key": row["from_status_key"],
                            "to_status_key": row["to_status_key"],
                            "actor": row["actor"],
                            "reason": row["reason"],
                            "idempotency_key": row["idempotency_key"],
                            "correlation_id": row["correlation_id"],
                            "before_snapshot": json.loads(row["before_snapshot"]) if row["before_snapshot"] else None,
                            "after_snapshot": json.loads(row["after_snapshot"]) if row["after_snapshot"] else None,
                            "created_at": row["created_at"],
                        }
                    )
                return events
            finally:
                conn.close()

    def list_stale_workflow_claims(
        self,
        *,
        board_id: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise InputError("limit must be positive")  # noqa: TRY003
        if offset < 0:
            raise InputError("offset must be non-negative")  # noqa: TRY003

        now = _utcnow_iso()
        with self._lock:
            conn = self._connect()
            try:
                sql = """
                    SELECT s.card_id, c.board_id, c.list_id, c.title, s.workflow_status_key,
                           s.lease_owner, s.lease_expires_at, s.version, s.updated_at
                    FROM kanban_card_workflow_state s
                    JOIN kanban_cards c ON c.id = s.card_id
                    JOIN kanban_boards b ON b.id = c.board_id
                    WHERE b.user_id = ?
                      AND s.lease_owner IS NOT NULL
                      AND s.lease_expires_at IS NOT NULL
                      AND s.lease_expires_at <= ?
                """
                params: list[Any] = [self.user_id, now]
                if board_id is not None:
                    sql += " AND c.board_id = ?"
                    params.append(board_id)
                sql += " ORDER BY s.lease_expires_at ASC, s.card_id ASC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                rows = conn.execute(sql, params).fetchall()

                return [
                    {
                        "card_id": row["card_id"],
                        "board_id": row["board_id"],
                        "list_id": row["list_id"],
                        "title": row["title"],
                        "workflow_status_key": row["workflow_status_key"],
                        "lease_owner": row["lease_owner"],
                        "lease_expires_at": row["lease_expires_at"],
                        "version": row["version"],
                        "updated_at": row["updated_at"],
                    }
                    for row in rows
                ]
            finally:
                conn.close()

    def force_reassign_workflow_claim(
        self,
        *,
        card_id: int,
        new_owner: str,
        idempotency_key: str,
        correlation_id: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if not new_owner or not new_owner.strip():
            raise InputError("new_owner is required")  # noqa: TRY003
        if not idempotency_key or not idempotency_key.strip():
            raise InputError("idempotency_key is required")  # noqa: TRY003

        owner = new_owner.strip()
        corr_id = correlation_id.strip() if correlation_id and correlation_id.strip() else None
        with self._lock:
            conn = self._connect()
            try:
                state_row = self._ensure_card_workflow_state(conn, card_id)
                current_state = self._row_to_card_workflow_state_dict(state_row)

                replay = conn.execute(
                    """
                    SELECT id
                    FROM kanban_card_workflow_events
                    WHERE card_id = ? AND event_type = 'workflow_claim_reassigned' AND idempotency_key = ?
                    """,
                    (card_id, idempotency_key.strip()),
                ).fetchone()
                if replay:
                    conn.commit()
                    return current_state

                ttl_row = conn.execute(
                    "SELECT default_lease_ttl_sec FROM board_workflow_policies WHERE id = ?",
                    (current_state["policy_id"],),
                ).fetchone()
                lease_ttl_sec = int(ttl_row["default_lease_ttl_sec"]) if ttl_row else 900

                now = _utcnow_iso()
                lease_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=lease_ttl_sec)).strftime("%Y-%m-%d %H:%M:%S")
                update_cur = conn.execute(
                    """
                    UPDATE kanban_card_workflow_state
                    SET lease_owner = ?,
                        lease_expires_at = ?,
                        last_actor = ?,
                        version = version + 1,
                        updated_at = ?
                    WHERE card_id = ? AND version = ?
                    """,
                    (owner, lease_expires_at, owner, now, card_id, current_state["version"]),
                )
                if update_cur.rowcount != 1:
                    raise ConflictError("version_conflict", entity="card_workflow_state", entity_id=card_id)  # noqa: TRY003

                updated_row = self._get_card_workflow_state_row(conn, card_id)
                if not updated_row:
                    raise KanbanDBError("Updated workflow state could not be loaded")  # noqa: TRY003
                updated_state = self._row_to_card_workflow_state_dict(updated_row)

                conn.execute(
                    """
                    INSERT INTO kanban_card_workflow_events
                    (card_id, event_type, from_status_key, to_status_key, actor, reason, idempotency_key,
                     correlation_id, before_snapshot, after_snapshot, created_at)
                    VALUES (?, 'workflow_claim_reassigned', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card_id,
                        current_state["workflow_status_key"],
                        updated_state["workflow_status_key"],
                        owner,
                        reason,
                        idempotency_key.strip(),
                        corr_id,
                        json.dumps(current_state),
                        json.dumps(updated_state),
                        now,
                    ),
                )

                conn.commit()
                return updated_state
            finally:
                conn.close()

    # =========================================================================
    # BOARD OPERATIONS
    # =========================================================================

    def create_board(
        self,
        name: str,
        client_id: str,
        description: str | None = None,
        activity_retention_days: int | None = None,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create a new board.

        Args:
            name: Board name (required).
            client_id: Client-generated unique ID for idempotency.
            description: Optional board description.
            activity_retention_days: Optional override for activity retention.
            metadata: Optional JSON metadata.

        Returns:
            The created board as a dictionary.

        Raises:
            InputError: If name is empty or too long.
            ConflictError: If client_id already exists for this user.
        """
        # Validate inputs
        if not name or not name.strip():
            raise InputError("Board name is required")  # noqa: TRY003
        name = name.strip()
        if len(name) > 255:
            raise InputError("Board name must be 255 characters or less")  # noqa: TRY003
        if not client_id or not client_id.strip():
            raise InputError("client_id is required")  # noqa: TRY003
        client_id = client_id.strip()

        board_uuid = _generate_uuid()
        now = _utcnow_iso()
        metadata_json = json.dumps(metadata) if metadata else None

        with self._lock:
            conn = self._connect()
            try:
                # Check board limit
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_boards WHERE user_id = ? AND deleted = 0",
                    (self.user_id,)
                )
                count = cur.fetchone()["cnt"]
                if count >= self.MAX_BOARDS_PER_USER:
                    raise InputError(f"Maximum boards ({self.MAX_BOARDS_PER_USER}) reached")  # noqa: TRY003

                # Insert board
                cur = conn.execute(
                    """
                    INSERT INTO kanban_boards
                    (uuid, user_id, client_id, name, description, activity_retention_days,
                     metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (board_uuid, self.user_id, client_id, name, description,
                     activity_retention_days, metadata_json, now, now)
                )
                board_id = cur.lastrowid

                # Log activity
                self._log_activity_internal(
                    conn, board_id, "board_created", "board",
                    entity_id=board_id, details={"name": name}
                )

                conn.commit()

                # Fetch and return the created board
                return self._get_board_by_id(conn, board_id)

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint" in str(e) and "client_id" in str(e):
                    raise ConflictError(  # noqa: B904, TRY003
                        f"Board with client_id '{client_id}' already exists",
                        entity="board",
                        entity_id=client_id
                    )
                raise KanbanDBError(f"Database error: {e}") from e  # noqa: TRY003
            finally:
                conn.close()

    def get_board(
        self,
        board_id: int,
        include_deleted: bool = False
    ) -> dict[str, Any] | None:
        """
        Get a board by ID.

        Args:
            board_id: The board ID.
            include_deleted: If True, include soft-deleted boards.

        Returns:
            The board as a dictionary, or None if not found.
        """
        with self._lock:
            conn = self._connect()
            try:
                return self._get_board_by_id(conn, board_id, include_deleted)
            finally:
                conn.close()

    def _get_board_by_id(
        self,
        conn: sqlite3.Connection,
        board_id: int,
        include_deleted: bool = False
    ) -> dict[str, Any] | None:
        """Internal method to get a board by ID using an existing connection."""
        sql = """
            SELECT id, uuid, user_id, client_id, name, description, archived,
                   archived_at, activity_retention_days, created_at, updated_at,
                   deleted, deleted_at, version, metadata
            FROM kanban_boards
            WHERE id = ? AND user_id = ?
        """
        params: list[Any] = [board_id, self.user_id]

        if not include_deleted:
            sql += " AND deleted = 0"

        cur = conn.execute(sql, params)
        row = cur.fetchone()

        if not row:
            return None

        return self._row_to_board_dict(row)

    def _row_to_board_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a board row to a dictionary."""
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "user_id": row["user_id"],
            "client_id": row["client_id"],
            "name": row["name"],
            "description": row["description"],
            "archived": bool(row["archived"]),
            "archived_at": row["archived_at"],
            "activity_retention_days": row["activity_retention_days"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted": bool(row["deleted"]),
            "deleted_at": row["deleted_at"],
            "version": row["version"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None
        }

    def list_boards(
        self,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List boards for the user with pagination.

        Args:
            include_archived: If True, include archived boards.
            include_deleted: If True, include soft-deleted boards.
            limit: Maximum number of boards to return.
            offset: Number of boards to skip.

        Returns:
            Tuple of (list of boards, total count).
        """
        with self._lock:
            conn = self._connect()
            try:
                conditions = ["user_id = ?"]
                params: list[Any] = [self.user_id]

                if not include_archived:
                    conditions.append("archived = 0")
                if not include_deleted:
                    conditions.append("deleted = 0")

                where_clause = " AND ".join(conditions)

                # Get total count
                count_sql = f"SELECT COUNT(*) as cnt FROM kanban_boards WHERE {where_clause}"  # nosec B608
                cur = conn.execute(count_sql, params)
                total = cur.fetchone()["cnt"]

                # Get boards
                sql = """
                    SELECT id, uuid, user_id, client_id, name, description, archived,
                           archived_at, activity_retention_days, created_at, updated_at,
                           deleted, deleted_at, version, metadata
                    FROM kanban_boards
                    WHERE {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """.format_map(locals())  # nosec B608
                params.extend([limit, offset])
                cur = conn.execute(sql, params)

                boards = [self._row_to_board_dict(row) for row in cur.fetchall()]
                if boards:
                    board_ids = [board["id"] for board in boards]
                    placeholders = ",".join("?" * len(board_ids))

                    list_conditions = [f"board_id IN ({placeholders})"]
                    list_params: list[Any] = list(board_ids)
                    if not include_archived:
                        list_conditions.append("archived = 0")
                    if not include_deleted:
                        list_conditions.append("deleted = 0")
                    list_where = " AND ".join(list_conditions)
                    list_sql = """
                        SELECT board_id, COUNT(*) as cnt
                        FROM kanban_lists
                        WHERE {list_where}
                        GROUP BY board_id
                    """.format_map(locals())  # nosec B608
                    list_counts = {
                        row["board_id"]: int(row["cnt"])
                        for row in conn.execute(list_sql, list_params).fetchall()
                    }

                    card_conditions = [f"board_id IN ({placeholders})"]
                    card_params: list[Any] = list(board_ids)
                    if not include_archived:
                        card_conditions.append("archived = 0")
                    if not include_deleted:
                        card_conditions.append("deleted = 0")
                    card_where = " AND ".join(card_conditions)
                    card_sql = """
                        SELECT board_id, COUNT(*) as cnt
                        FROM kanban_cards
                        WHERE {card_where}
                        GROUP BY board_id
                    """.format_map(locals())  # nosec B608
                    card_counts = {
                        row["board_id"]: int(row["cnt"])
                        for row in conn.execute(card_sql, card_params).fetchall()
                    }

                    for board in boards:
                        board_id = board["id"]
                        board["list_count"] = list_counts.get(board_id, 0)
                        board["card_count"] = card_counts.get(board_id, 0)

                return boards, total

            finally:
                conn.close()

    def update_board(
        self,
        board_id: int,
        name: str | None = None,
        description: str | None = None,
        activity_retention_days: int | None = None,
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None
    ) -> dict[str, Any]:
        """
        Update a board.

        Args:
            board_id: The board ID to update.
            name: New name (optional).
            description: New description (optional).
            activity_retention_days: New retention days (optional).
            metadata: New metadata (optional).
            expected_version: If provided, only update if version matches (optimistic locking).

        Returns:
            The updated board.

        Raises:
            NotFoundError: If board not found.
            ConflictError: If expected_version doesn't match.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Get current board
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003

                # Check version if provided
                if expected_version is not None and board["version"] != expected_version:
                    raise ConflictError(  # noqa: TRY003
                        f"Version mismatch: expected {expected_version}, got {board['version']}",
                        entity="board",
                        entity_id=board_id
                    )

                # Build update
                updates = []
                params: list[Any] = []

                if name is not None:
                    if not name.strip():
                        raise InputError("Board name cannot be empty")  # noqa: TRY003
                    name = name.strip()
                    if len(name) > 255:
                        raise InputError("Board name must be 255 characters or less")  # noqa: TRY003
                    updates.append("name = ?")
                    params.append(name)

                if description is not None:
                    updates.append("description = ?")
                    params.append(description)

                if activity_retention_days is not None:
                    if activity_retention_days < 7 or activity_retention_days > 365:
                        raise InputError("activity_retention_days must be between 7 and 365")  # noqa: TRY003
                    updates.append("activity_retention_days = ?")
                    params.append(activity_retention_days)

                if metadata is not None:
                    updates.append("metadata = ?")
                    params.append(json.dumps(metadata))

                if not updates:
                    return board

                # Always update version and timestamp
                updates.append("version = version + 1")
                updates.append("updated_at = ?")
                params.append(_utcnow_iso())
                params.append(board_id)

                sql = f"UPDATE kanban_boards SET {', '.join(updates)} WHERE id = ?"  # nosec B608
                conn.execute(sql, params)

                # Log activity
                self._log_activity_internal(
                    conn, board_id, "board_updated", "board",
                    entity_id=board_id, details={"updated_fields": [u.split(" = ")[0] for u in updates if "version" not in u and "updated_at" not in u]}
                )

                conn.commit()

                return self._get_board_by_id(conn, board_id)

            finally:
                conn.close()

    def archive_board(self, board_id: int, archive: bool = True) -> dict[str, Any]:
        """
        Archive or unarchive a board.

        Args:
            board_id: The board ID.
            archive: True to archive, False to unarchive.

        Returns:
            The updated board.
        """
        card_ids: list[int] = []
        updated_board: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003

                now = _utcnow_iso()
                archived_at = now if archive else None

                list_rows = conn.execute(
                    """
                    SELECT id, name FROM kanban_lists
                    WHERE board_id = ? AND deleted = 0
                    """,
                    (board_id,),
                ).fetchall()
                card_rows = conn.execute(
                    """
                    SELECT id, list_id, title FROM kanban_cards
                    WHERE board_id = ? AND deleted = 0
                    """,
                    (board_id,),
                ).fetchall()
                card_ids = [row["id"] for row in card_rows]
                conn.execute(
                    """
                    UPDATE kanban_boards
                    SET archived = ?, archived_at = ?, version = version + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (1 if archive else 0, archived_at, now, board_id)
                )

                conn.execute(
                    """
                    UPDATE kanban_lists
                    SET archived = ?, archived_at = ?, version = version + 1, updated_at = ?
                    WHERE board_id = ? AND deleted = 0
                    """,
                    (1 if archive else 0, archived_at, now, board_id)
                )
                conn.execute(
                    """
                    UPDATE kanban_cards
                    SET archived = ?, archived_at = ?, version = version + 1, updated_at = ?
                    WHERE board_id = ? AND deleted = 0
                    """,
                    (1 if archive else 0, archived_at, now, board_id)
                )

                # Log activity
                action = "board_archived" if archive else "board_unarchived"
                self._log_activity_internal(
                    conn, board_id, action, "board", entity_id=board_id
                )
                list_action = "list_archived" if archive else "list_unarchived"
                for lst in list_rows:
                    self._log_activity_internal(
                        conn,
                        board_id,
                        list_action,
                        "list",
                        entity_id=lst["id"],
                        list_id=lst["id"],
                        details={"name": lst["name"]},
                    )
                card_action = "card_archived" if archive else "card_unarchived"
                for card in card_rows:
                    self._log_activity_internal(
                        conn,
                        board_id,
                        card_action,
                        "card",
                        entity_id=card["id"],
                        list_id=card["list_id"],
                        card_id=card["id"],
                        details={"title": card["title"]},
                    )

                conn.commit()

                updated_board = self._get_board_by_id(conn, board_id)

            finally:
                conn.close()

        if card_ids:
            self._sync_vector_index_for_card_ids(card_ids)
        return updated_board

    def delete_board(self, board_id: int, hard_delete: bool = False) -> bool:
        """
        Delete a board (soft delete by default).

        Args:
            board_id: The board ID.
            hard_delete: If True, permanently delete. If False, soft delete.

        Returns:
            True if deleted, False if not found.
        """
        card_ids: list[int] = []
        deleted = False
        with self._lock:
            conn = self._connect()
            try:
                board = self._get_board_by_id(conn, board_id, include_deleted=True)
                if not board:
                    return False

                if hard_delete:
                    card_rows = conn.execute(
                        "SELECT id FROM kanban_cards WHERE board_id = ?",
                        (board_id,),
                    ).fetchall()
                    card_ids = [row["id"] for row in card_rows]
                    conn.execute("DELETE FROM kanban_boards WHERE id = ?", (board_id,))
                else:
                    now = _utcnow_iso()
                    list_rows = conn.execute(
                        """
                        SELECT id, name FROM kanban_lists
                        WHERE board_id = ? AND deleted = 0
                        """,
                        (board_id,),
                    ).fetchall()
                    card_rows = conn.execute(
                        """
                        SELECT id, list_id, title FROM kanban_cards
                        WHERE board_id = ? AND deleted = 0
                        """,
                        (board_id,),
                    ).fetchall()
                    card_ids = [row["id"] for row in card_rows]
                    conn.execute(
                        """
                        UPDATE kanban_boards
                        SET deleted = 1, deleted_at = ?, version = version + 1, updated_at = ?
                        WHERE id = ?
                        """,
                        (now, now, board_id)
                    )
                    conn.execute(
                        """
                        UPDATE kanban_lists
                        SET deleted = 1, deleted_at = ?, version = version + 1, updated_at = ?
                        WHERE board_id = ? AND deleted = 0
                        """,
                        (now, now, board_id)
                    )
                    conn.execute(
                        """
                        UPDATE kanban_cards
                        SET deleted = 1, deleted_at = ?, version = version + 1, updated_at = ?
                        WHERE board_id = ? AND deleted = 0
                        """,
                        (now, now, board_id)
                    )
                    # Log activity (only for soft delete, hard delete removes everything)
                    self._log_activity_internal(
                        conn, board_id, "board_deleted", "board", entity_id=board_id
                    )
                    for lst in list_rows:
                        self._log_activity_internal(
                            conn,
                            board_id,
                            "list_deleted",
                            "list",
                            entity_id=lst["id"],
                            list_id=lst["id"],
                            details={"name": lst["name"]},
                        )
                    for card in card_rows:
                        self._log_activity_internal(
                            conn,
                            board_id,
                            "card_deleted",
                            "card",
                            entity_id=card["id"],
                            list_id=card["list_id"],
                            card_id=card["id"],
                            details={"title": card["title"]},
                        )

                conn.commit()
                deleted = True

            finally:
                conn.close()

        if deleted and card_ids:
            self._sync_vector_index_for_card_ids(card_ids)
        return deleted

    def restore_board(self, board_id: int) -> dict[str, Any]:
        """
        Restore a soft-deleted board.

        Args:
            board_id: The board ID.

        Returns:
            The restored board.

        Raises:
            NotFoundError: If board not found or not deleted.
        """
        card_ids: list[int] = []
        restored_board: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                board = self._get_board_by_id(conn, board_id, include_deleted=True)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003
                if not board["deleted"]:
                    raise InputError("Board is not deleted")  # noqa: TRY003

                now = _utcnow_iso()
                list_rows = conn.execute(
                    """
                    SELECT id, name FROM kanban_lists
                    WHERE board_id = ? AND deleted = 1
                    """,
                    (board_id,),
                ).fetchall()
                card_rows = conn.execute(
                    """
                    SELECT id, list_id, title FROM kanban_cards
                    WHERE board_id = ? AND deleted = 1
                    """,
                    (board_id,),
                ).fetchall()
                card_ids = [row["id"] for row in card_rows]

                conn.execute(
                    """
                    UPDATE kanban_boards
                    SET deleted = 0, deleted_at = NULL, version = version + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, board_id)
                )
                conn.execute(
                    """
                    UPDATE kanban_lists
                    SET deleted = 0, deleted_at = NULL, version = version + 1, updated_at = ?
                    WHERE board_id = ? AND deleted = 1
                    """,
                    (now, board_id)
                )
                conn.execute(
                    """
                    UPDATE kanban_cards
                    SET deleted = 0, deleted_at = NULL, version = version + 1, updated_at = ?
                    WHERE board_id = ? AND deleted = 1
                    """,
                    (now, board_id)
                )

                # Log activity
                self._log_activity_internal(
                    conn, board_id, "board_restored", "board", entity_id=board_id
                )
                for lst in list_rows:
                    self._log_activity_internal(
                        conn,
                        board_id,
                        "list_restored",
                        "list",
                        entity_id=lst["id"],
                        list_id=lst["id"],
                        details={"name": lst["name"]},
                    )
                for card in card_rows:
                    self._log_activity_internal(
                        conn,
                        board_id,
                        "card_restored",
                        "card",
                        entity_id=card["id"],
                        list_id=card["list_id"],
                        card_id=card["id"],
                        details={"title": card["title"]},
                    )

                conn.commit()

                restored_board = self._get_board_by_id(conn, board_id)

            finally:
                conn.close()

        if card_ids:
            self._sync_vector_index_for_card_ids(card_ids)
        return restored_board

    # =========================================================================
    # LIST OPERATIONS
    # =========================================================================

    def create_list(
        self,
        board_id: int,
        name: str,
        client_id: str,
        position: int | None = None
    ) -> dict[str, Any]:
        """
        Create a new list in a board.

        Args:
            board_id: The board ID.
            name: List name.
            client_id: Client-generated unique ID.
            position: Optional position (defaults to end).

        Returns:
            The created list.
        """
        if not name or not name.strip():
            raise InputError("List name is required")  # noqa: TRY003
        name = name.strip()
        if len(name) > 255:
            raise InputError("List name must be 255 characters or less")  # noqa: TRY003
        if not client_id or not client_id.strip():
            raise InputError("client_id is required")  # noqa: TRY003
        client_id = client_id.strip()

        list_uuid = _generate_uuid()
        now = _utcnow_iso()

        with self._lock:
            conn = self._connect()
            try:
                # Verify board exists and belongs to user
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003

                # Check list limit
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_lists WHERE board_id = ? AND deleted = 0",
                    (board_id,)
                )
                count = cur.fetchone()["cnt"]
                if count >= self.MAX_LISTS_PER_BOARD:
                    raise InputError(f"Maximum lists ({self.MAX_LISTS_PER_BOARD}) per board reached")  # noqa: TRY003

                # Get next position if not provided
                if position is None:
                    cur = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM kanban_lists WHERE board_id = ? AND deleted = 0",
                        (board_id,)
                    )
                    position = cur.fetchone()["next_pos"]

                # Insert list
                cur = conn.execute(
                    """
                    INSERT INTO kanban_lists
                    (uuid, board_id, client_id, name, position, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (list_uuid, board_id, client_id, name, position, now, now)
                )
                list_id = cur.lastrowid

                # Log activity
                self._log_activity_internal(
                    conn, board_id, "list_created", "list", entity_id=list_id,
                    list_id=list_id, details={"name": name}
                )

                conn.commit()

                return self._get_list_by_id(conn, list_id)

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint" in str(e) and "client_id" in str(e):
                    raise ConflictError(  # noqa: B904, TRY003
                        f"List with client_id '{client_id}' already exists in this board",
                        entity="list",
                        entity_id=client_id
                    )
                raise KanbanDBError(f"Database error: {e}") from e  # noqa: TRY003
            finally:
                conn.close()

    def get_list(self, list_id: int, include_deleted: bool = False) -> dict[str, Any] | None:
        """Get a list by ID."""
        with self._lock:
            conn = self._connect()
            try:
                return self._get_list_by_id(conn, list_id, include_deleted)
            finally:
                conn.close()

    def _get_list_by_id(
        self,
        conn: sqlite3.Connection,
        list_id: int,
        include_deleted: bool = False
    ) -> dict[str, Any] | None:
        """Internal method to get a list by ID."""
        sql = """
            SELECT l.id, l.uuid, l.board_id, l.client_id, l.name, l.position,
                   l.archived, l.archived_at, l.created_at, l.updated_at,
                   l.deleted, l.deleted_at, l.version
            FROM kanban_lists l
            JOIN kanban_boards b ON l.board_id = b.id
            WHERE l.id = ? AND b.user_id = ?
        """
        params: list[Any] = [list_id, self.user_id]

        if not include_deleted:
            sql += " AND l.deleted = 0"

        cur = conn.execute(sql, params)
        row = cur.fetchone()

        if not row:
            return None

        return self._row_to_list_dict(row)

    def _row_to_list_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a list row to a dictionary."""
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "board_id": row["board_id"],
            "client_id": row["client_id"],
            "name": row["name"],
            "position": row["position"],
            "archived": bool(row["archived"]),
            "archived_at": row["archived_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted": bool(row["deleted"]),
            "deleted_at": row["deleted_at"],
            "version": row["version"]
        }

    def list_lists(
        self,
        board_id: int,
        include_archived: bool = False,
        include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get all lists for a board, ordered by position.

        Args:
            board_id: The board ID.
            include_archived: If True, include archived lists.
            include_deleted: If True, include soft-deleted lists.

        Returns:
            List of lists ordered by position.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify board exists and belongs to user
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003
                if board["archived"] and not include_archived:
                    return []

                conditions = ["l.board_id = ?"]
                params: list[Any] = [board_id]

                if not include_archived:
                    conditions.append("l.archived = 0")
                if not include_deleted:
                    conditions.append("l.deleted = 0")

                where_clause = " AND ".join(conditions)

                sql = """
                    SELECT l.id, l.uuid, l.board_id, l.client_id, l.name, l.position,
                           l.archived, l.archived_at, l.created_at, l.updated_at,
                           l.deleted, l.deleted_at, l.version
                    FROM kanban_lists l
                    WHERE {where_clause}
                    ORDER BY l.position ASC
                """.format_map(locals())  # nosec B608
                cur = conn.execute(sql, params)

                return [self._row_to_list_dict(row) for row in cur.fetchall()]

            finally:
                conn.close()

    def update_list(
        self,
        list_id: int,
        name: str | None = None,
        position: int | None = None,
        expected_version: int | None = None
    ) -> dict[str, Any]:
        """Update a list."""
        with self._lock:
            conn = self._connect()
            try:
                lst = self._get_list_by_id(conn, list_id)
                if not lst:
                    raise NotFoundError("List not found", entity="list", entity_id=list_id)  # noqa: TRY003

                board = self._get_board_by_id(conn, lst["board_id"])
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=lst["board_id"])  # noqa: TRY003

                if expected_version is not None and lst["version"] != expected_version:
                    raise ConflictError(  # noqa: TRY003
                        f"Version mismatch: expected {expected_version}, got {lst['version']}",
                        entity="list",
                        entity_id=list_id
                    )

                updates = []
                params: list[Any] = []

                if name is not None:
                    if not name.strip():
                        raise InputError("List name cannot be empty")  # noqa: TRY003
                    name = name.strip()
                    if len(name) > 255:
                        raise InputError("List name must be 255 characters or less")  # noqa: TRY003
                    updates.append("name = ?")
                    params.append(name)

                if position is not None:
                    if position < 0:
                        raise InputError("List position must be >= 0")  # noqa: TRY003

                    # Clamp to valid range (end if larger than current list count - 1).
                    cur = conn.execute(
                        "SELECT COUNT(*) as cnt FROM kanban_lists WHERE board_id = ? AND deleted = 0",
                        (lst["board_id"],),
                    )
                    max_position = max(0, int(cur.fetchone()["cnt"]) - 1)
                    target_position = min(int(position), max_position)

                    if target_position != lst["position"]:
                        if target_position > lst["position"]:
                            conn.execute(
                                """
                                UPDATE kanban_lists
                                SET position = position - 1, updated_at = ?, version = version + 1
                                WHERE board_id = ?
                                  AND deleted = 0
                                  AND position > ?
                                  AND position <= ?
                                  AND id != ?
                                """,
                                (_utcnow_iso(), lst["board_id"], lst["position"], target_position, list_id),
                            )
                        else:
                            conn.execute(
                                """
                                UPDATE kanban_lists
                                SET position = position + 1, updated_at = ?, version = version + 1
                                WHERE board_id = ?
                                  AND deleted = 0
                                  AND position >= ?
                                  AND position < ?
                                  AND id != ?
                                """,
                                (_utcnow_iso(), lst["board_id"], target_position, lst["position"], list_id),
                            )
                        updates.append("position = ?")
                        params.append(target_position)

                if not updates:
                    return lst

                updates.append("version = version + 1")
                updates.append("updated_at = ?")
                params.append(_utcnow_iso())
                params.append(list_id)

                sql = f"UPDATE kanban_lists SET {', '.join(updates)} WHERE id = ?"  # nosec B608
                conn.execute(sql, params)

                # Log activity with updated fields
                updated_fields = []
                if name is not None:
                    updated_fields.append("name")
                if position is not None:
                    updated_fields.append("position")
                self._log_activity_internal(
                    conn, lst["board_id"], "list_updated", "list", entity_id=list_id,
                    list_id=list_id, details={"updated_fields": updated_fields}
                )

                conn.commit()

                return self._get_list_by_id(conn, list_id)

            finally:
                conn.close()

    def reorder_lists(self, board_id: int, list_ids: list[int]) -> list[dict[str, Any]]:
        """
        Reorder lists in a board.

        Args:
            board_id: The board ID.
            list_ids: List IDs in the desired order.

        Returns:
            Updated lists in new order.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify board exists
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003

                # Verify all lists exist and belong to the board
                cur = conn.execute(
                    f"SELECT id FROM kanban_lists WHERE board_id = ? AND deleted = 0 AND id IN ({','.join('?' * len(list_ids))})",  # nosec B608
                    [board_id] + list_ids
                )
                existing_ids = {row["id"] for row in cur.fetchall()}

                if len(existing_ids) != len(list_ids):
                    missing = set(list_ids) - existing_ids
                    raise InputError(f"Lists not found or don't belong to board: {missing}")  # noqa: TRY003

                # Update positions
                now = _utcnow_iso()
                for position, list_id in enumerate(list_ids):
                    conn.execute(
                        "UPDATE kanban_lists SET position = ?, updated_at = ?, version = version + 1 WHERE id = ?",
                        (position, now, list_id)
                    )

                # Log activity
                self._log_activity_internal(
                    conn, board_id, "lists_reordered", "board", entity_id=board_id,
                    details={"list_ids": list_ids}
                )

                conn.commit()

                return self.list_lists(board_id)

            finally:
                conn.close()

    def archive_list(self, list_id: int, archive: bool = True) -> dict[str, Any]:
        """Archive or unarchive a list."""
        card_ids: list[int] = []
        updated_list: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                lst = self._get_list_by_id(conn, list_id)
                if not lst:
                    raise NotFoundError("List not found", entity="list", entity_id=list_id)  # noqa: TRY003

                now = _utcnow_iso()
                archived_at = now if archive else None
                card_rows = conn.execute(
                    """
                    SELECT id, title FROM kanban_cards
                    WHERE list_id = ? AND deleted = 0
                    """,
                    (list_id,),
                ).fetchall()
                conn.execute(
                    """
                    UPDATE kanban_lists
                    SET archived = ?, archived_at = ?, version = version + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (1 if archive else 0, archived_at, now, list_id)
                )
                conn.execute(
                    """
                    UPDATE kanban_cards
                    SET archived = ?, archived_at = ?, version = version + 1, updated_at = ?
                    WHERE list_id = ? AND deleted = 0
                    """,
                    (1 if archive else 0, archived_at, now, list_id)
                )

                # Log activity
                action = "list_archived" if archive else "list_unarchived"
                self._log_activity_internal(
                    conn, lst["board_id"], action, "list", entity_id=list_id,
                    list_id=list_id, details={"name": lst["name"]}
                )
                card_action = "card_archived" if archive else "card_unarchived"
                for card in card_rows:
                    self._log_activity_internal(
                        conn,
                        lst["board_id"],
                        card_action,
                        "card",
                        entity_id=card["id"],
                        list_id=list_id,
                        card_id=card["id"],
                        details={"title": card["title"]},
                    )

                conn.commit()
                card_ids = [row["id"] for row in card_rows]
                updated_list = self._get_list_by_id(conn, list_id)

            finally:
                conn.close()

        if card_ids:
            self._sync_vector_index_for_card_ids(card_ids)
        return updated_list

    def delete_list(self, list_id: int, hard_delete: bool = False) -> bool:
        """Delete a list (soft delete by default)."""
        card_ids: list[int] = []
        deleted = False
        with self._lock:
            conn = self._connect()
            try:
                lst = self._get_list_by_id(conn, list_id, include_deleted=True)
                if not lst:
                    return False

                if hard_delete:
                    card_rows = conn.execute(
                        "SELECT id FROM kanban_cards WHERE list_id = ?",
                        (list_id,),
                    ).fetchall()
                    card_ids = [row["id"] for row in card_rows]
                    conn.execute("DELETE FROM kanban_lists WHERE id = ?", (list_id,))
                else:
                    now = _utcnow_iso()
                    card_rows = conn.execute(
                        """
                        SELECT id, title FROM kanban_cards
                        WHERE list_id = ? AND deleted = 0
                        """,
                        (list_id,),
                    ).fetchall()
                    card_ids = [row["id"] for row in card_rows]
                    conn.execute(
                        """
                        UPDATE kanban_lists
                        SET deleted = 1, deleted_at = ?, version = version + 1, updated_at = ?
                        WHERE id = ?
                        """,
                        (now, now, list_id)
                    )
                    conn.execute(
                        """
                        UPDATE kanban_cards
                        SET deleted = 1, deleted_at = ?, version = version + 1, updated_at = ?
                        WHERE list_id = ? AND deleted = 0
                        """,
                        (now, now, list_id)
                    )
                    # Log activity (only for soft delete)
                    self._log_activity_internal(
                        conn, lst["board_id"], "list_deleted", "list", entity_id=list_id,
                        list_id=list_id, details={"name": lst["name"]}
                    )
                    for card in card_rows:
                        self._log_activity_internal(
                            conn,
                            lst["board_id"],
                            "card_deleted",
                            "card",
                            entity_id=card["id"],
                            list_id=list_id,
                            card_id=card["id"],
                            details={"title": card["title"]},
                        )
                conn.commit()
                deleted = True

            finally:
                conn.close()

        if deleted and card_ids:
            self._sync_vector_index_for_card_ids(card_ids)
        return deleted

    def restore_list(self, list_id: int) -> dict[str, Any]:
        """Restore a soft-deleted list."""
        card_ids: list[int] = []
        restored_list: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                lst = self._get_list_by_id(conn, list_id, include_deleted=True)
                if not lst:
                    raise NotFoundError("List not found", entity="list", entity_id=list_id)  # noqa: TRY003
                if not lst["deleted"]:
                    raise InputError("List is not deleted")  # noqa: TRY003

                now = _utcnow_iso()
                card_rows = conn.execute(
                    """
                    SELECT id, title FROM kanban_cards
                    WHERE list_id = ? AND deleted = 1
                    """,
                    (list_id,),
                ).fetchall()
                card_ids = [row["id"] for row in card_rows]

                conn.execute(
                    """
                    UPDATE kanban_lists
                    SET deleted = 0, deleted_at = NULL, version = version + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, list_id)
                )
                conn.execute(
                    """
                    UPDATE kanban_cards
                    SET deleted = 0, deleted_at = NULL, version = version + 1, updated_at = ?
                    WHERE list_id = ? AND deleted = 1
                    """,
                    (now, list_id)
                )

                # Log activity
                self._log_activity_internal(
                    conn, lst["board_id"], "list_restored", "list", entity_id=list_id,
                    list_id=list_id, details={"name": lst["name"]}
                )
                for card in card_rows:
                    self._log_activity_internal(
                        conn,
                        lst["board_id"],
                        "card_restored",
                        "card",
                        entity_id=card["id"],
                        list_id=list_id,
                        card_id=card["id"],
                        details={"title": card["title"]},
                    )

                conn.commit()
                restored_list = self._get_list_by_id(conn, list_id)

            finally:
                conn.close()

        if card_ids:
            self._sync_vector_index_for_card_ids(card_ids)
        return restored_list

    # =========================================================================
    # CARD OPERATIONS
    # =========================================================================

    def create_card(
        self,
        list_id: int,
        title: str,
        client_id: str,
        description: str | None = None,
        position: int | None = None,
        due_date: str | None = None,
        start_date: str | None = None,
        priority: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Create a new card in a list.

        Args:
            list_id: The list ID.
            title: Card title.
            client_id: Client-generated unique ID.
            description: Optional description.
            position: Optional position (defaults to end).
            due_date: Optional due date (ISO format).
            start_date: Optional start date (ISO format).
            priority: Optional priority ('low', 'medium', 'high', 'urgent').
            metadata: Optional JSON metadata.

        Returns:
            The created card.
        """
        if not title or not title.strip():
            raise InputError("Card title is required")  # noqa: TRY003
        title = title.strip()
        if len(title) > 500:
            raise InputError("Card title must be 500 characters or less")  # noqa: TRY003
        if not client_id or not client_id.strip():
            raise InputError("client_id is required")  # noqa: TRY003
        client_id = client_id.strip()

        if priority and priority not in ('low', 'medium', 'high', 'urgent'):
            raise InputError("priority must be one of: low, medium, high, urgent")  # noqa: TRY003

        card_uuid = _generate_uuid()
        now = _utcnow_iso()
        metadata_json = json.dumps(metadata) if metadata else None
        card: dict[str, Any] | None = None

        with self._lock:
            conn = self._connect()
            try:
                # Get list and board_id
                lst = self._get_list_by_id(conn, list_id)
                if not lst:
                    raise NotFoundError("List not found", entity="list", entity_id=list_id)  # noqa: TRY003

                board_id = lst["board_id"]

                # Check card limit
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_cards WHERE board_id = ? AND deleted = 0",
                    (board_id,)
                )
                board_count = cur.fetchone()["cnt"]
                if board_count >= self.MAX_CARDS_PER_BOARD:
                    raise InputError(f"Maximum cards ({self.MAX_CARDS_PER_BOARD}) per board reached")  # noqa: TRY003

                # Check list limit
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_cards WHERE list_id = ? AND deleted = 0",
                    (list_id,)
                )
                count = cur.fetchone()["cnt"]
                if count >= self.MAX_CARDS_PER_LIST:
                    raise InputError(f"Maximum cards ({self.MAX_CARDS_PER_LIST}) per list reached")  # noqa: TRY003

                # Get next position if not provided
                if position is None:
                    cur = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM kanban_cards WHERE list_id = ? AND deleted = 0",
                        (list_id,)
                    )
                    position = cur.fetchone()["next_pos"]

                # Insert card
                cur = conn.execute(
                    """
                    INSERT INTO kanban_cards
                    (uuid, board_id, list_id, client_id, title, description, position,
                     due_date, start_date, priority, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (card_uuid, board_id, list_id, client_id, title, description, position,
                     due_date, start_date, priority, metadata_json, now, now)
                )
                card_id = cur.lastrowid

                # Log activity
                self._log_activity_internal(
                    conn, board_id, "card_created", "card", entity_id=card_id,
                    list_id=list_id, card_id=card_id, details={"title": title}
                )

                conn.commit()

                card = self._get_card_by_id(conn, card_id)

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint" in str(e) and "client_id" in str(e):
                    raise ConflictError(  # noqa: B904, TRY003
                        f"Card with client_id '{client_id}' already exists in this board",
                        entity="card",
                        entity_id=client_id
                    )
                raise KanbanDBError(f"Database error: {e}") from e  # noqa: TRY003
            finally:
                conn.close()

        if card:
            self._sync_vector_index_for_card_id(card["id"])
        return card

    def get_card(self, card_id: int, include_deleted: bool = False) -> dict[str, Any] | None:
        """Get a card by ID."""
        with self._lock:
            conn = self._connect()
            try:
                return self._get_card_by_id(conn, card_id, include_deleted)
            finally:
                conn.close()

    def get_card_with_details(
        self,
        card_id: int,
        include_deleted: bool = False
    ) -> dict[str, Any] | None:
        """
        Get a card with labels, checklists (including items), and comment count.

        Args:
            card_id: The card ID.
            include_deleted: If True, include soft-deleted cards.

        Returns:
            Card dictionary with labels, checklists, and comment_count included.
        """
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id, include_deleted)
                if not card:
                    return None

                card["labels"] = self._get_card_labels_for_index(conn, card_id)

                checklist_rows = conn.execute(
                    """
                    SELECT id, uuid, card_id, name, position, created_at, updated_at
                    FROM kanban_checklists
                    WHERE card_id = ?
                    ORDER BY position ASC
                    """,
                    (card_id,),
                ).fetchall()

                checklists: list[dict[str, Any]] = []
                for row in checklist_rows:
                    checklist = self._row_to_checklist_dict(row)
                    item_rows = conn.execute(
                        """
                        SELECT id, uuid, checklist_id, name, position, checked, checked_at, created_at, updated_at
                        FROM kanban_checklist_items
                        WHERE checklist_id = ?
                        ORDER BY position ASC
                        """,
                        (checklist["id"],),
                    ).fetchall()
                    items = [self._row_to_checklist_item_dict(item) for item in item_rows]
                    total = len(items)
                    checked = sum(1 for item in items if item["checked"])
                    checklist["items"] = items
                    checklist["total_items"] = total
                    checklist["checked_items"] = checked
                    checklist["progress_percent"] = round(checked / total * 100) if total > 0 else 0
                    checklists.append(checklist)

                card["checklists"] = checklists

                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_comments WHERE card_id = ? AND deleted = 0",
                    (card_id,),
                )
                card["comment_count"] = cur.fetchone()["cnt"]

                return card

            finally:
                conn.close()

    def get_cards_by_ids(
        self,
        card_ids: list[int],
        include_deleted: bool = False,
        include_archived: bool = True
    ) -> list[dict[str, Any]]:
        """
        Get multiple cards by IDs in a single query (batch fetch).

        This is more efficient than calling get_card() in a loop.

        Args:
            card_ids: List of card IDs to fetch.
            include_deleted: If True, include soft-deleted cards.
            include_archived: If True, include archived cards (default True).

        Returns:
            List of cards (may be fewer than requested if some IDs don't exist).
        """
        if not card_ids:
            return []

        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(card_ids))
                sql = """
                    SELECT c.id, c.uuid, c.board_id, c.list_id, c.client_id, c.title, c.description,
                           c.position, c.due_date, c.due_complete, c.start_date, c.priority,
                           c.archived, c.archived_at, c.created_at, c.updated_at,
                           c.deleted, c.deleted_at, c.version, c.metadata,
                           b.name as board_name, l.name as list_name
                    FROM kanban_cards c
                    JOIN kanban_boards b ON c.board_id = b.id
                    JOIN kanban_lists l ON c.list_id = l.id
                    WHERE c.id IN ({placeholders}) AND b.user_id = ?
                """.format_map(locals())  # nosec B608
                params: list[Any] = list(card_ids) + [self.user_id]

                if not include_deleted:
                    sql += " AND c.deleted = 0"
                if not include_archived:
                    sql += " AND c.archived = 0 AND b.archived = 0 AND l.archived = 0"

                cur = conn.execute(sql, params)
                rows = cur.fetchall()

                cards = []
                for row in rows:
                    card = self._row_to_card_dict(row)
                    card["board_name"] = row["board_name"]
                    card["list_name"] = row["list_name"]
                    # Get labels for this card
                    card["labels"] = self.get_card_labels(card["id"])
                    cards.append(card)

                return cards
            finally:
                conn.close()

    def _get_card_by_id(
        self,
        conn: sqlite3.Connection,
        card_id: int,
        include_deleted: bool = False
    ) -> dict[str, Any] | None:
        """Internal method to get a card by ID."""
        sql = """
            SELECT c.id, c.uuid, c.board_id, c.list_id, c.client_id, c.title, c.description,
                   c.position, c.due_date, c.due_complete, c.start_date, c.priority,
                   c.archived, c.archived_at, c.created_at, c.updated_at,
                   c.deleted, c.deleted_at, c.version, c.metadata
            FROM kanban_cards c
            JOIN kanban_boards b ON c.board_id = b.id
            WHERE c.id = ? AND b.user_id = ?
        """
        params: list[Any] = [card_id, self.user_id]

        if not include_deleted:
            sql += " AND c.deleted = 0"

        cur = conn.execute(sql, params)
        row = cur.fetchone()

        if not row:
            return None

        return self._row_to_card_dict(row)

    def _row_to_card_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a card row to a dictionary."""
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "board_id": row["board_id"],
            "list_id": row["list_id"],
            "client_id": row["client_id"],
            "title": row["title"],
            "description": row["description"],
            "position": row["position"],
            "due_date": row["due_date"],
            "due_complete": bool(row["due_complete"]),
            "start_date": row["start_date"],
            "priority": row["priority"],
            "archived": bool(row["archived"]),
            "archived_at": row["archived_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted": bool(row["deleted"]),
            "deleted_at": row["deleted_at"],
            "version": row["version"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None
        }

    def _get_vector_search(self) -> Any | None:
        """Return a cached KanbanVectorSearch instance when available."""
        with self._lock:
            if self._vector_search_initialized:
                return self._vector_search
            embedding_config = dict(settings.get("EMBEDDING_CONFIG") or {})
            user_db_base_dir = None
            if not self._is_memory_db:
                try:
                    # Keep vector storage colocated with the active Kanban DB root
                    # so tests and runtime do not drift to stale persisted stores.
                    user_db_base_dir = str(Path(self.db_path).expanduser().resolve().parent.parent)
                except (OSError, RuntimeError) as exc:
                    logger.debug(
                        f"Failed to derive USER_DB_BASE_DIR from db_path={self.db_path}: {exc}; "
                        "falling back to settings."
                    )
            if not user_db_base_dir:
                user_db_base_dir = settings.get("USER_DB_BASE_DIR")
            if user_db_base_dir:
                embedding_config["USER_DB_BASE_DIR"] = user_db_base_dir
            self._vector_search = create_kanban_vector_search(self.user_id, embedding_config)
            self._vector_search_initialized = True
            return self._vector_search

    def get_vector_search(self) -> Any | None:
        """Expose vector search for API use."""
        return self._get_vector_search()

    def _get_card_labels_for_index(self, conn: sqlite3.Connection, card_id: int) -> list[dict[str, Any]]:
        """Fetch labels for a card using an existing connection."""
        cur = conn.execute(
            """
            SELECT l.id, l.uuid, l.board_id, l.name, l.color, l.created_at, l.updated_at
            FROM kanban_labels l
            JOIN kanban_card_labels cl ON l.id = cl.label_id
            WHERE cl.card_id = ?
            ORDER BY l.name ASC
            """,
            (card_id,),
        )
        return [self._row_to_label_dict(row) for row in cur.fetchall()]

    def _get_checklist_item_names(self, conn: sqlite3.Connection, card_id: int) -> list[str]:
        """Fetch checklist item names for a card in display order."""
        cur = conn.execute(
            """
            SELECT ci.name
            FROM kanban_checklist_items ci
            JOIN kanban_checklists ch ON ci.checklist_id = ch.id
            WHERE ch.card_id = ?
            ORDER BY ch.position ASC, ci.position ASC
            """,
            (card_id,),
        )
        return [row["name"] for row in cur.fetchall() if row["name"]]

    def _fetch_card_for_vector_indexing(
        self,
        conn: sqlite3.Connection,
        card_id: int,
    ) -> dict[str, Any] | None:
        """Fetch a card with labels and checklist items for vector indexing."""
        card = self._get_card_by_id(conn, card_id, include_deleted=True)
        if not card:
            return None
        card["labels"] = self._get_card_labels_for_index(conn, card_id)
        card["checklist_items"] = self._get_checklist_item_names(conn, card_id)
        return card

    def _schedule_vector_index_retry(
        self,
        card_id: int,
        operation: str,
        *,
        retry_attempt: int,
    ) -> None:
        """Schedule a best-effort retry for vector index operations."""
        if _is_test_context():
            return
        if retry_attempt >= self.VECTOR_INDEX_MAX_RETRY_ATTEMPTS:
            return
        key = (operation, card_id)
        with self._vector_index_retry_lock:
            if key in self._vector_index_retry_pending:
                return
            self._vector_index_retry_pending.add(key)

        delay = self.VECTOR_INDEX_RETRY_DELAY_SECONDS

        def _retry() -> None:
            try:
                self._sync_vector_index_for_card_id(card_id, retry_attempt=retry_attempt + 1)
            finally:
                with self._vector_index_retry_lock:
                    self._vector_index_retry_pending.discard(key)

        timer = threading.Timer(delay, _retry)
        timer.daemon = True
        timer.start()
        logger.debug(f"Scheduled vector index retry for card {card_id} op={operation}")

    def _sync_vector_index_for_card_id(
        self,
        card_id: int,
        search: Any | None = None,
        *,
        retry_attempt: int = 0,
    ) -> None:
        """Index or remove a card in vector search based on current state."""
        vector_search = search or self._get_vector_search()
        if vector_search is None:
            return
        attempt = retry_attempt + 1
        with self._lock:
            conn = self._connect()
            try:
                card = self._fetch_card_for_vector_indexing(conn, card_id)
            finally:
                conn.close()

        if not card:
            try:
                result = vector_search.remove_card(card_id)
                if result is False:
                    logger.warning(
                        f"Vector index op=remove reported failure for card {card_id} (attempt {attempt})"
                    )
                    self._schedule_vector_index_retry(
                        card_id,
                        "remove",
                        retry_attempt=retry_attempt,
                    )
            except _KANBAN_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    f"Vector index op=remove failed for card {card_id} (attempt {attempt}): {exc}"
                )
                self._schedule_vector_index_retry(
                    card_id,
                    "remove",
                    retry_attempt=retry_attempt,
                )
            return

        if card.get("deleted") or card.get("archived"):
            try:
                result = vector_search.remove_card(card_id)
                if result is False:
                    logger.warning(
                        f"Vector index op=remove reported failure for card {card_id} (attempt {attempt})"
                    )
                    self._schedule_vector_index_retry(
                        card_id,
                        "remove",
                        retry_attempt=retry_attempt,
                    )
            except _KANBAN_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    f"Vector index op=remove failed for card {card_id} (attempt {attempt}): {exc}"
                )
                self._schedule_vector_index_retry(
                    card_id,
                    "remove",
                    retry_attempt=retry_attempt,
                )
            return

        try:
            result = vector_search.index_card(card)
            if result is False:
                logger.warning(
                    f"Vector index op=index reported failure for card {card_id} (attempt {attempt})"
                )
                self._schedule_vector_index_retry(
                    card_id,
                    "index",
                    retry_attempt=retry_attempt,
                )
        except _KANBAN_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                f"Vector index op=index failed for card {card_id} (attempt {attempt}): {exc}"
            )
            self._schedule_vector_index_retry(
                card_id,
                "index",
                retry_attempt=retry_attempt,
            )

    def _sync_vector_index_for_card_ids(self, card_ids: list[int]) -> None:
        """Sync vector index for multiple cards."""
        if not card_ids:
            return
        vector_search = self._get_vector_search()
        if vector_search is None:
            return
        for card_id in card_ids:
            self._sync_vector_index_for_card_id(card_id, search=vector_search)

    def list_cards(
        self,
        list_id: int,
        include_archived: bool = False,
        include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """
        Get all cards in a list, ordered by position.

        Args:
            list_id: The list ID.
            include_archived: If True, include archived cards.
            include_deleted: If True, include soft-deleted cards.

        Returns:
            List of cards ordered by position.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify list exists and belongs to user's board
                lst = self._get_list_by_id(conn, list_id)
                if not lst:
                    raise NotFoundError("List not found", entity="list", entity_id=list_id)  # noqa: TRY003
                if not include_archived:
                    if lst["archived"]:
                        return []
                    board = self._get_board_by_id(conn, lst["board_id"])
                    if not board:
                        raise NotFoundError("Board not found", entity="board", entity_id=lst["board_id"])  # noqa: TRY003
                    if board["archived"]:
                        return []

                conditions = ["c.list_id = ?"]
                params: list[Any] = [list_id]

                if not include_archived:
                    conditions.append("c.archived = 0")
                if not include_deleted:
                    conditions.append("c.deleted = 0")

                where_clause = " AND ".join(conditions)

                sql = """
                    SELECT c.id, c.uuid, c.board_id, c.list_id, c.client_id, c.title, c.description,
                           c.position, c.due_date, c.due_complete, c.start_date, c.priority,
                           c.archived, c.archived_at, c.created_at, c.updated_at,
                           c.deleted, c.deleted_at, c.version, c.metadata
                    FROM kanban_cards c
                    WHERE {where_clause}
                    ORDER BY c.position ASC
                """.format_map(locals())  # nosec B608
                cur = conn.execute(sql, params)

                return [self._row_to_card_dict(row) for row in cur.fetchall()]

            finally:
                conn.close()

    def _list_cards_with_summary_for_list(
        self,
        conn: sqlite3.Connection,
        list_id: int,
        *,
        include_archived: bool,
        include_deleted: bool,
    ) -> list[dict[str, Any]]:
        """
        Get cards in a list with label and checklist/comment summaries.

        Uses an existing connection to avoid extra round-trips inside nested board views.
        """
        conditions = ["c.list_id = ?"]
        params: list[Any] = [list_id]

        if not include_archived:
            conditions.append("c.archived = 0")
        if not include_deleted:
            conditions.append("c.deleted = 0")

        where_clause = " AND ".join(conditions)

        sql = """
            SELECT c.id, c.uuid, c.board_id, c.list_id, c.client_id, c.title, c.description,
                   c.position, c.due_date, c.due_complete, c.start_date, c.priority,
                   c.archived, c.archived_at, c.created_at, c.updated_at,
                   c.deleted, c.deleted_at, c.version, c.metadata,
                   (SELECT COUNT(*) FROM kanban_checklists ch WHERE ch.card_id = c.id) AS checklist_count,
                   (SELECT COUNT(*) FROM kanban_checklist_items ci
                        JOIN kanban_checklists ch ON ci.checklist_id = ch.id
                        WHERE ch.card_id = c.id) AS checklist_total,
                   (SELECT COUNT(*) FROM kanban_checklist_items ci
                        JOIN kanban_checklists ch ON ci.checklist_id = ch.id
                        WHERE ch.card_id = c.id AND ci.checked = 1) AS checklist_complete,
                   (SELECT COUNT(*) FROM kanban_comments cm
                        WHERE cm.card_id = c.id AND cm.deleted = 0) AS comment_count
            FROM kanban_cards c
            WHERE {where_clause}
            ORDER BY c.position ASC
        """.format_map(locals())  # nosec B608
        cur = conn.execute(sql, params)
        rows = cur.fetchall()

        cards: list[dict[str, Any]] = []
        card_ids: list[int] = []
        for row in rows:
            card = self._row_to_card_dict(row)
            card["checklist_count"] = int(row["checklist_count"] or 0)
            card["checklist_total"] = int(row["checklist_total"] or 0)
            card["checklist_complete"] = int(row["checklist_complete"] or 0)
            card["comment_count"] = int(row["comment_count"] or 0)
            cards.append(card)
            card_ids.append(card["id"])

        if not card_ids:
            return cards

        placeholders = ",".join("?" * len(card_ids))
        label_sql = """
            SELECT cl.card_id, l.id, l.uuid, l.board_id, l.name, l.color, l.created_at, l.updated_at
            FROM kanban_card_labels cl
            JOIN kanban_labels l ON l.id = cl.label_id
            WHERE cl.card_id IN ({placeholders})
            ORDER BY l.name ASC
        """.format_map(locals())  # nosec B608
        label_rows = conn.execute(label_sql, card_ids).fetchall()
        labels_by_card: dict[int, list[dict[str, Any]]] = {}
        for row in label_rows:
            card_id = row["card_id"]
            label = self._row_to_label_dict(row)
            labels_by_card.setdefault(card_id, []).append(label)

        for card in cards:
            card["labels"] = labels_by_card.get(card["id"], [])

        return cards

    def update_card(
        self,
        card_id: int,
        title: str | None = None,
        description: str | None = None,
        due_date: str | None = None,
        due_complete: bool | None = None,
        start_date: str | None = None,
        priority: str | None = None,
        metadata: dict[str, Any] | None = None,
        expected_version: int | None = None
    ) -> dict[str, Any]:
        """Update a card."""
        card: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                if expected_version is not None and card["version"] != expected_version:
                    raise ConflictError(  # noqa: TRY003
                        f"Version mismatch: expected {expected_version}, got {card['version']}",
                        entity="card",
                        entity_id=card_id
                    )

                updates = []
                params: list[Any] = []

                if title is not None:
                    if not title.strip():
                        raise InputError("Card title cannot be empty")  # noqa: TRY003
                    title = title.strip()
                    if len(title) > 500:
                        raise InputError("Card title must be 500 characters or less")  # noqa: TRY003
                    updates.append("title = ?")
                    params.append(title)

                if description is not None:
                    updates.append("description = ?")
                    params.append(description)

                if due_date is not None:
                    updates.append("due_date = ?")
                    params.append(due_date if due_date else None)

                if due_complete is not None:
                    updates.append("due_complete = ?")
                    params.append(1 if due_complete else 0)

                if start_date is not None:
                    updates.append("start_date = ?")
                    params.append(start_date if start_date else None)

                if priority is not None:
                    if priority and priority not in ('low', 'medium', 'high', 'urgent'):
                        raise InputError("priority must be one of: low, medium, high, urgent")  # noqa: TRY003
                    updates.append("priority = ?")
                    params.append(priority if priority else None)

                if metadata is not None:
                    updates.append("metadata = ?")
                    params.append(json.dumps(metadata))

                if not updates:
                    return card

                # Build list of updated fields for activity log
                updated_fields = []
                if title is not None:
                    updated_fields.append("title")
                if description is not None:
                    updated_fields.append("description")
                if due_date is not None:
                    updated_fields.append("due_date")
                if due_complete is not None:
                    updated_fields.append("due_complete")
                if start_date is not None:
                    updated_fields.append("start_date")
                if priority is not None:
                    updated_fields.append("priority")
                if metadata is not None:
                    updated_fields.append("metadata")

                updates.append("version = version + 1")
                updates.append("updated_at = ?")
                params.append(_utcnow_iso())
                params.append(card_id)

                sql = f"UPDATE kanban_cards SET {', '.join(updates)} WHERE id = ?"  # nosec B608
                conn.execute(sql, params)

                # Log activity
                self._log_activity_internal(
                    conn, card["board_id"], "card_updated", "card", entity_id=card_id,
                    list_id=card["list_id"], card_id=card_id,
                    details={"updated_fields": updated_fields}
                )

                conn.commit()

                card = self._get_card_by_id(conn, card_id)

            finally:
                conn.close()

        if card:
            self._sync_vector_index_for_card_id(card_id)
        return card

    def move_card(
        self,
        card_id: int,
        target_list_id: int,
        position: int | None = None
    ) -> dict[str, Any]:
        """
        Move a card to a different list.

        Args:
            card_id: The card ID.
            target_list_id: The destination list ID.
            position: Optional position in the target list (defaults to end).

        Returns:
            The updated card.
        """
        moved_card: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                target_list = self._get_list_by_id(conn, target_list_id)
                if not target_list:
                    raise NotFoundError("Target list not found", entity="list", entity_id=target_list_id)  # noqa: TRY003

                # Verify target list is in the same board
                if target_list["board_id"] != card["board_id"]:
                    raise InputError("Cannot move card to a list in a different board")  # noqa: TRY003

                # Get target position
                if position is None:
                    cur = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM kanban_cards WHERE list_id = ? AND deleted = 0",
                        (target_list_id,)
                    )
                    position = cur.fetchone()["next_pos"]

                source_list_id = card["list_id"]
                now = _utcnow_iso()
                conn.execute(
                    """
                    UPDATE kanban_cards
                    SET list_id = ?, position = ?, version = version + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (target_list_id, position, now, card_id)
                )

                # Log activity
                self._log_activity_internal(
                    conn, card["board_id"], "card_moved", "card", entity_id=card_id,
                    list_id=target_list_id, card_id=card_id,
                    details={
                        "title": card["title"],
                        "from_list_id": source_list_id,
                        "to_list_id": target_list_id
                    }
                )

                conn.commit()

                moved_card = self._get_card_by_id(conn, card_id)

            finally:
                conn.close()

        if moved_card:
            self._sync_vector_index_for_card_id(card_id)
        return moved_card

    def copy_card(
        self,
        card_id: int,
        target_list_id: int,
        new_client_id: str,
        position: int | None = None,
        new_title: str | None = None
    ) -> dict[str, Any]:
        """
        Copy a card to a list.

        Args:
            card_id: The source card ID.
            target_list_id: The destination list ID.
            new_client_id: Client-generated unique ID for the copy.
            position: Optional position in the target list.
            new_title: Optional new title (defaults to "Copy of {original}").

        Returns:
            The copied card.
        """
        copied_card: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                target_list = self._get_list_by_id(conn, target_list_id)
                if not target_list:
                    raise NotFoundError("Target list not found", entity="list", entity_id=target_list_id)  # noqa: TRY003

                # Verify target list is in the same board
                if target_list["board_id"] != card["board_id"]:
                    raise InputError("Cannot copy card to a list in a different board")  # noqa: TRY003

                # Enforce board and list limits
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_cards WHERE board_id = ? AND deleted = 0",
                    (card["board_id"],)
                )
                board_count = cur.fetchone()["cnt"]
                if board_count >= self.MAX_CARDS_PER_BOARD:
                    raise InputError(f"Maximum cards ({self.MAX_CARDS_PER_BOARD}) per board reached")  # noqa: TRY003

                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_cards WHERE list_id = ? AND deleted = 0",
                    (target_list_id,)
                )
                list_count = cur.fetchone()["cnt"]
                if list_count >= self.MAX_CARDS_PER_LIST:
                    raise InputError(f"Maximum cards ({self.MAX_CARDS_PER_LIST}) per list reached")  # noqa: TRY003

                # Generate title if not provided
                if new_title is None:
                    new_title = f"Copy of {card['title']}"

                # Get target position
                if position is None:
                    cur = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM kanban_cards WHERE list_id = ? AND deleted = 0",
                        (target_list_id,)
                    )
                    position = cur.fetchone()["next_pos"]

                card_uuid = _generate_uuid()
                now = _utcnow_iso()

                # Insert the copy
                cur = conn.execute(
                    """
                    INSERT INTO kanban_cards
                    (uuid, board_id, list_id, client_id, title, description, position,
                     due_date, start_date, priority, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (card_uuid, card["board_id"], target_list_id, new_client_id,
                     new_title, card["description"], position,
                     card["due_date"], card["start_date"], card["priority"],
                     json.dumps(card["metadata"]) if card["metadata"] else None,
                     now, now)
                )
                new_card_id = cur.lastrowid

                # Copy labels
                conn.execute(
                    """
                    INSERT INTO kanban_card_labels (card_id, label_id, created_at)
                    SELECT ?, label_id, ? FROM kanban_card_labels WHERE card_id = ?
                    """,
                    (new_card_id, now, card_id)
                )

                # Log activity
                self._log_activity_internal(
                    conn, card["board_id"], "card_copied", "card", entity_id=new_card_id,
                    list_id=target_list_id, card_id=new_card_id,
                    details={
                        "title": new_title,
                        "source_card_id": card_id,
                        "target_list_id": target_list_id
                    }
                )

                conn.commit()

                copied_card = self._get_card_by_id(conn, new_card_id)

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint" in str(e) and "client_id" in str(e):
                    raise ConflictError(  # noqa: B904, TRY003
                        f"Card with client_id '{new_client_id}' already exists",
                        entity="card",
                        entity_id=new_client_id
                    )
                raise KanbanDBError(f"Database error: {e}") from e  # noqa: TRY003
            finally:
                conn.close()

        if copied_card:
            self._sync_vector_index_for_card_id(copied_card["id"])
        return copied_card

    def reorder_cards(self, list_id: int, card_ids: list[int]) -> list[dict[str, Any]]:
        """
        Reorder cards in a list.

        Args:
            list_id: The list ID.
            card_ids: Card IDs in the desired order.

        Returns:
            Updated cards in new order.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify list exists
                lst = self._get_list_by_id(conn, list_id)
                if not lst:
                    raise NotFoundError("List not found", entity="list", entity_id=list_id)  # noqa: TRY003

                # Verify all cards exist and belong to the list
                cur = conn.execute(
                    f"SELECT id FROM kanban_cards WHERE list_id = ? AND deleted = 0 AND id IN ({','.join('?' * len(card_ids))})",  # nosec B608
                    [list_id] + card_ids
                )
                existing_ids = {row["id"] for row in cur.fetchall()}

                if len(existing_ids) != len(card_ids):
                    missing = set(card_ids) - existing_ids
                    raise InputError(f"Cards not found or don't belong to list: {missing}")  # noqa: TRY003

                # Update positions
                now = _utcnow_iso()
                for position, card_id in enumerate(card_ids):
                    conn.execute(
                        "UPDATE kanban_cards SET position = ?, updated_at = ?, version = version + 1 WHERE id = ?",
                        (position, now, card_id)
                    )

                # Log activity
                self._log_activity_internal(
                    conn, lst["board_id"], "cards_reordered", "list", entity_id=list_id,
                    list_id=list_id, details={"card_ids": card_ids}
                )

                conn.commit()

                return self.list_cards(list_id)

            finally:
                conn.close()

    def archive_card(self, card_id: int, archive: bool = True) -> dict[str, Any]:
        """Archive or unarchive a card."""
        updated_card: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                now = _utcnow_iso() if archive else None
                conn.execute(
                    """
                    UPDATE kanban_cards
                    SET archived = ?, archived_at = ?, version = version + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (1 if archive else 0, now, _utcnow_iso(), card_id)
                )

                # Log activity
                action = "card_archived" if archive else "card_unarchived"
                self._log_activity_internal(
                    conn, card["board_id"], action, "card", entity_id=card_id,
                    list_id=card["list_id"], card_id=card_id,
                    details={"title": card["title"]}
                )

                conn.commit()

                updated_card = self._get_card_by_id(conn, card_id)

            finally:
                conn.close()

        if updated_card:
            self._sync_vector_index_for_card_id(card_id)
        return updated_card

    def delete_card(self, card_id: int, hard_delete: bool = False) -> bool:
        """Delete a card (soft delete by default)."""
        deleted = False
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id, include_deleted=True)
                if not card:
                    return False

                if hard_delete:
                    conn.execute("DELETE FROM kanban_cards WHERE id = ?", (card_id,))
                else:
                    conn.execute(
                        """
                        UPDATE kanban_cards
                        SET deleted = 1, deleted_at = ?, version = version + 1, updated_at = ?
                        WHERE id = ?
                        """,
                        (_utcnow_iso(), _utcnow_iso(), card_id)
                    )
                    # Log activity (only for soft delete)
                    self._log_activity_internal(
                        conn, card["board_id"], "card_deleted", "card", entity_id=card_id,
                        list_id=card["list_id"], card_id=card_id,
                        details={"title": card["title"]}
                    )
                conn.commit()
                deleted = True

            finally:
                conn.close()

        if deleted:
            self._sync_vector_index_for_card_id(card_id)
        return deleted

    def restore_card(self, card_id: int) -> dict[str, Any]:
        """Restore a soft-deleted card."""
        restored_card: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id, include_deleted=True)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003
                if not card["deleted"]:
                    raise InputError("Card is not deleted")  # noqa: TRY003

                conn.execute(
                    """
                    UPDATE kanban_cards
                    SET deleted = 0, deleted_at = NULL, version = version + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (_utcnow_iso(), card_id)
                )

                # Log activity
                self._log_activity_internal(
                    conn, card["board_id"], "card_restored", "card", entity_id=card_id,
                    list_id=card["list_id"], card_id=card_id,
                    details={"title": card["title"]}
                )

                conn.commit()

                restored_card = self._get_card_by_id(conn, card_id)

            finally:
                conn.close()

        if restored_card:
            self._sync_vector_index_for_card_id(card_id)
        return restored_card

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================

    def search_cards(
        self,
        query: str,
        board_id: int | None = None,
        label_ids: list[int] | None = None,
        priority: str | None = None,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Search cards using FTS5.

        Args:
            query: Search query.
            board_id: Optional board ID to filter results.
            label_ids: Optional list of label IDs (cards must have ALL).
            priority: Optional priority filter.
            include_archived: Whether to include archived cards.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            Tuple of (list of search result cards with enriched data, total count).
        """
        if not query or not query.strip():
            raise InputError("Search query is required")  # noqa: TRY003

        with self._lock:
            conn = self._connect()
            try:
                # Build the search query
                raw_query = query.strip()

                # Escape FTS5 special characters by wrapping in double quotes
                # FTS5 treats quoted strings as literal phrase matches
                # First escape any internal double quotes
                fts_escaped = raw_query.replace('"', '""')
                fts_query = f'"{fts_escaped}"'

                # Escape LIKE wildcards to prevent unexpected matches
                escaped_query = raw_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                like_pattern = f"%{escaped_query}%"

                # Build label filter subquery if needed
                label_filter_sql = ""
                label_params: list[Any] = []
                if label_ids and len(label_ids) > 0:
                    label_placeholders = ",".join("?" * len(label_ids))
                    label_filter_sql = """
                        AND c.id IN (
                            SELECT card_id
                            FROM kanban_card_labels
                            WHERE label_id IN ({label_placeholders})
                            GROUP BY card_id
                            HAVING COUNT(DISTINCT label_id) = ?
                        )
                    """.format_map(locals())  # nosec B608
                    label_params.extend(label_ids)
                    label_params.append(len(label_ids))

                # Build common filters
                board_filter = "AND c.board_id = ?" if board_id else ""
                priority_filter = "AND c.priority = ?" if priority else ""
                active_board_filter = "AND b.archived = 0"
                active_list_filter = "AND l.archived = 0"
                archived_scope_filter = "AND (c.archived = 1 OR l.archived = 1 OR b.archived = 1)"

                # Strategy: FTS only indexes non-archived, non-deleted cards
                # For include_archived=True, we need to also search archived cards using LIKE
                if include_archived:
                    # Use UNION: FTS for non-archived + LIKE for archived
                    count_sql = """
                        SELECT COUNT(*) as cnt FROM (
                            -- Non-archived cards via FTS
                            SELECT c.id
                            FROM kanban_cards c
                            JOIN kanban_boards b ON c.board_id = b.id
                            JOIN kanban_lists l ON c.list_id = l.id
                            JOIN kanban_cards_fts fts ON c.id = fts.rowid
                            WHERE b.user_id = ? AND c.deleted = 0 AND c.archived = 0
                            {active_board_filter} {active_list_filter} {board_filter} {priority_filter} {label_filter_sql}
                            AND kanban_cards_fts MATCH ?
                            UNION
                            -- Archived cards via LIKE
                            SELECT c.id
                            FROM kanban_cards c
                            JOIN kanban_boards b ON c.board_id = b.id
                            JOIN kanban_lists l ON c.list_id = l.id
                            WHERE b.user_id = ? AND c.deleted = 0
                            {archived_scope_filter} {board_filter} {priority_filter} {label_filter_sql}
                            AND (c.title LIKE ? ESCAPE '\\' OR c.description LIKE ? ESCAPE '\\')
                        )
                    """.format_map(locals())  # nosec B608
                    # Build params: FTS part + archived LIKE part
                    count_params: list[Any] = [self.user_id]
                    if board_id:
                        count_params.append(board_id)
                    if priority:
                        count_params.append(priority)
                    count_params.extend(label_params)
                    count_params.append(fts_query)
                    # Archived part
                    count_params.append(self.user_id)
                    if board_id:
                        count_params.append(board_id)
                    if priority:
                        count_params.append(priority)
                    count_params.extend(label_params)
                    count_params.extend([like_pattern, like_pattern])

                    cur = conn.execute(count_sql, count_params)
                    total = cur.fetchone()["cnt"]

                    # Search query with UNION
                    sql = """
                        SELECT * FROM (
                            -- Non-archived cards via FTS
                            SELECT c.id, c.uuid, c.board_id, c.list_id, c.client_id, c.title, c.description,
                                   c.position, c.due_date, c.due_complete, c.start_date, c.priority,
                                   c.archived, c.archived_at, c.created_at, c.updated_at,
                                   c.deleted, c.deleted_at, c.version, c.metadata,
                                   b.name as board_name, l.name as list_name,
                                   1 as search_rank
                            FROM kanban_cards c
                            JOIN kanban_boards b ON c.board_id = b.id
                            JOIN kanban_lists l ON c.list_id = l.id
                            JOIN kanban_cards_fts fts ON c.id = fts.rowid
                            WHERE b.user_id = ? AND c.deleted = 0 AND c.archived = 0
                            {active_board_filter} {active_list_filter} {board_filter} {priority_filter} {label_filter_sql}
                            AND kanban_cards_fts MATCH ?
                            UNION
                            -- Archived cards via LIKE
                            SELECT c.id, c.uuid, c.board_id, c.list_id, c.client_id, c.title, c.description,
                                   c.position, c.due_date, c.due_complete, c.start_date, c.priority,
                                   c.archived, c.archived_at, c.created_at, c.updated_at,
                                   c.deleted, c.deleted_at, c.version, c.metadata,
                                   b.name as board_name, l.name as list_name,
                                   2 as search_rank
                            FROM kanban_cards c
                            JOIN kanban_boards b ON c.board_id = b.id
                            JOIN kanban_lists l ON c.list_id = l.id
                            WHERE b.user_id = ? AND c.deleted = 0
                            {archived_scope_filter} {board_filter} {priority_filter} {label_filter_sql}
                            AND (c.title LIKE ? ESCAPE '\\' OR c.description LIKE ? ESCAPE '\\')
                        )
                        ORDER BY search_rank, updated_at DESC
                        LIMIT ? OFFSET ?
                    """.format_map(locals())  # nosec B608
                    search_params: list[Any] = [self.user_id]
                    if board_id:
                        search_params.append(board_id)
                    if priority:
                        search_params.append(priority)
                    search_params.extend(label_params)
                    search_params.append(fts_query)
                    # Archived part
                    search_params.append(self.user_id)
                    if board_id:
                        search_params.append(board_id)
                    if priority:
                        search_params.append(priority)
                    search_params.extend(label_params)
                    search_params.extend([like_pattern, like_pattern])
                    search_params.extend([limit, offset])

                    cur = conn.execute(sql, search_params)
                else:
                    # Simple FTS-only search for non-archived cards
                    count_sql = """
                        SELECT COUNT(*) as cnt
                        FROM kanban_cards c
                        JOIN kanban_boards b ON c.board_id = b.id
                        JOIN kanban_lists l ON c.list_id = l.id
                        JOIN kanban_cards_fts fts ON c.id = fts.rowid
                        WHERE b.user_id = ? AND c.deleted = 0 AND c.archived = 0
                        {active_board_filter} {active_list_filter} {board_filter} {priority_filter} {label_filter_sql}
                        AND kanban_cards_fts MATCH ?
                    """.format_map(locals())  # nosec B608
                    count_params = [self.user_id]
                    if board_id:
                        count_params.append(board_id)
                    if priority:
                        count_params.append(priority)
                    count_params.extend(label_params)
                    count_params.append(fts_query)

                    cur = conn.execute(count_sql, count_params)
                    total = cur.fetchone()["cnt"]

                    sql = """
                        SELECT c.id, c.uuid, c.board_id, c.list_id, c.client_id, c.title, c.description,
                               c.position, c.due_date, c.due_complete, c.start_date, c.priority,
                               c.archived, c.archived_at, c.created_at, c.updated_at,
                               c.deleted, c.deleted_at, c.version, c.metadata,
                               b.name as board_name, l.name as list_name
                        FROM kanban_cards c
                        JOIN kanban_boards b ON c.board_id = b.id
                        JOIN kanban_lists l ON c.list_id = l.id
                        JOIN kanban_cards_fts fts ON c.id = fts.rowid
                        WHERE b.user_id = ? AND c.deleted = 0 AND c.archived = 0
                        {active_board_filter} {active_list_filter} {board_filter} {priority_filter} {label_filter_sql}
                        AND kanban_cards_fts MATCH ?
                        ORDER BY rank
                        LIMIT ? OFFSET ?
                    """.format_map(locals())  # nosec B608
                    search_params = [self.user_id]
                    if board_id:
                        search_params.append(board_id)
                    if priority:
                        search_params.append(priority)
                    search_params.extend(label_params)
                    search_params.extend([fts_query, limit, offset])

                    cur = conn.execute(sql, search_params)

                results = []
                for row in cur.fetchall():
                    card = self._row_to_card_dict(row)
                    card["board_name"] = row["board_name"]
                    card["list_name"] = row["list_name"]
                    # Get labels for this card
                    card["labels"] = self.get_card_labels(card["id"])
                    results.append(card)

                return results, total

            finally:
                conn.close()

    # =========================================================================
    # ACTIVITY OPERATIONS (for Phase 2, but schema is here)
    # =========================================================================

    def log_activity(
        self,
        board_id: int,
        action_type: str,
        entity_type: str,
        entity_id: int | None = None,
        list_id: int | None = None,
        card_id: int | None = None,
        details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Log an activity event.

        Args:
            board_id: The board ID.
            action_type: Type of action (create, update, delete, move, etc.).
            entity_type: Type of entity (board, list, card, label, etc.).
            entity_id: Optional ID of the entity.
            list_id: Optional list ID for context.
            card_id: Optional card ID for context.
            details: Optional JSON details.

        Returns:
            The created activity record.
        """
        activity_uuid = _generate_uuid()
        now = _utcnow_iso()
        details_json = json.dumps(details) if details else None

        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO kanban_activities
                    (uuid, board_id, list_id, card_id, user_id, action_type, entity_type, entity_id, details, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (activity_uuid, board_id, list_id, card_id, self.user_id,
                     action_type, entity_type, entity_id, details_json, now)
                )
                activity_id = cur.lastrowid
                conn.commit()

                # Fetch and return
                cur = conn.execute(
                    "SELECT * FROM kanban_activities WHERE id = ?",
                    (activity_id,)
                )
                row = cur.fetchone()

                return {
                    "id": row["id"],
                    "uuid": row["uuid"],
                    "board_id": row["board_id"],
                    "list_id": row["list_id"],
                    "card_id": row["card_id"],
                    "user_id": row["user_id"],
                    "action_type": row["action_type"],
                    "entity_type": row["entity_type"],
                    "entity_id": row["entity_id"],
                    "details": json.loads(row["details"]) if row["details"] else None,
                    "created_at": row["created_at"]
                }

            finally:
                conn.close()

    def _log_activity_internal(
        self,
        conn: sqlite3.Connection,
        board_id: int,
        action_type: str,
        entity_type: str,
        entity_id: int | None = None,
        list_id: int | None = None,
        card_id: int | None = None,
        details: dict[str, Any] | None = None
    ) -> None:
        """
        Internal helper to log activity while already holding a connection.

        This is called from within other operations to avoid nested locks.
        Does not commit - caller is responsible for committing.
        """
        activity_uuid = _generate_uuid()
        now = _utcnow_iso()
        details_json = json.dumps(details) if details else None

        conn.execute(
            """
            INSERT INTO kanban_activities
            (uuid, board_id, list_id, card_id, user_id, action_type, entity_type, entity_id, details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (activity_uuid, board_id, list_id, card_id, self.user_id,
             action_type, entity_type, entity_id, details_json, now)
        )

    def _fetch_activities(
        self,
        conn: sqlite3.Connection,
        board_id: int,
        list_id: int | None = None,
        card_id: int | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        action_type: str | None = None,
        entity_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        if limit < 1 or offset < 0:
            raise InputError("limit must be positive and offset must be non-negative")  # noqa: TRY003
        conditions = ["board_id = ?"]
        params: list[Any] = [board_id]

        if list_id is not None:
            conditions.append("list_id = ?")
            params.append(list_id)

        if card_id is not None:
            conditions.append("card_id = ?")
            params.append(card_id)

        if created_after:
            conditions.append("created_at >= ?")
            params.append(created_after)

        if created_before:
            conditions.append("created_at <= ?")
            params.append(created_before)

        if action_type:
            conditions.append("action_type = ?")
            params.append(action_type)

        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)

        where_clause = " AND ".join(conditions)

        count_sql = f"SELECT COUNT(*) as cnt FROM kanban_activities WHERE {where_clause}"  # nosec B608
        cur = conn.execute(count_sql, params)
        total = cur.fetchone()["cnt"]

        sql = """
            SELECT id, uuid, board_id, list_id, card_id, user_id, action_type,
                   entity_type, entity_id, details, created_at
            FROM kanban_activities
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        cur = conn.execute(sql, params + [limit, offset])

        activities = []
        for row in cur.fetchall():
            activities.append({
                "id": row["id"],
                "uuid": row["uuid"],
                "board_id": row["board_id"],
                "list_id": row["list_id"],
                "card_id": row["card_id"],
                "user_id": row["user_id"],
                "action_type": row["action_type"],
                "entity_type": row["entity_type"],
                "entity_id": row["entity_id"],
                "details": json.loads(row["details"]) if row["details"] else None,
                "created_at": row["created_at"]
            })

        return activities, total

    def get_board_activities(
        self,
        board_id: int,
        list_id: int | None = None,
        card_id: int | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        action_type: str | None = None,
        entity_type: str | None = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Get activities for a board.

        Args:
            board_id: The board ID.
            list_id: Optional filter by list.
            card_id: Optional filter by card.
            created_after: Optional timestamp filter (inclusive).
            created_before: Optional timestamp filter (inclusive).
            action_type: Optional filter by action_type.
            entity_type: Optional filter by entity_type.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            Tuple of (list of activities, total count).
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify board belongs to user
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003

                if list_id is not None:
                    lst = self._get_list_by_id(conn, list_id)
                    if not lst or lst["board_id"] != board_id:
                        raise NotFoundError("List not found for board", entity="list", entity_id=list_id)  # noqa: TRY003

                if card_id is not None:
                    card = self._get_card_by_id(conn, card_id, include_deleted=True)
                    if not card or card["board_id"] != board_id:
                        raise NotFoundError("Card not found for board", entity="card", entity_id=card_id)  # noqa: TRY003

                return self._fetch_activities(
                    conn=conn,
                    board_id=board_id,
                    list_id=list_id,
                    card_id=card_id,
                    created_after=created_after,
                    created_before=created_before,
                    action_type=action_type,
                    entity_type=entity_type,
                    limit=limit,
                    offset=offset,
                )

            finally:
                conn.close()

    def get_list_activities(
        self,
        list_id: int,
        created_after: str | None = None,
        created_before: str | None = None,
        action_type: str | None = None,
        entity_type: str | None = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Get activities for a list.

        Args:
            list_id: The list ID.
            created_after: Optional timestamp filter (inclusive).
            created_before: Optional timestamp filter (inclusive).
            action_type: Optional filter by action_type.
            entity_type: Optional filter by entity_type.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            Tuple of (list of activities, total count).
        """
        with self._lock:
            conn = self._connect()
            try:
                lst = self._get_list_by_id(conn, list_id, include_deleted=True)
                if not lst:
                    raise NotFoundError("List not found", entity="list", entity_id=list_id)  # noqa: TRY003

                return self._fetch_activities(
                    conn=conn,
                    board_id=lst["board_id"],
                    list_id=list_id,
                    created_after=created_after,
                    created_before=created_before,
                    action_type=action_type,
                    entity_type=entity_type,
                    limit=limit,
                    offset=offset,
                )
            finally:
                conn.close()

    def get_card_activities(
        self,
        card_id: int,
        created_after: str | None = None,
        created_before: str | None = None,
        action_type: str | None = None,
        entity_type: str | None = None,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Get activities for a card.

        Args:
            card_id: The card ID.
            created_after: Optional timestamp filter (inclusive).
            created_before: Optional timestamp filter (inclusive).
            action_type: Optional filter by action_type.
            entity_type: Optional filter by entity_type.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            Tuple of (list of activities, total count).
        """
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id, include_deleted=True)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                return self._fetch_activities(
                    conn=conn,
                    board_id=card["board_id"],
                    card_id=card_id,
                    created_after=created_after,
                    created_before=created_before,
                    action_type=action_type,
                    entity_type=entity_type,
                    limit=limit,
                    offset=offset,
                )
            finally:
                conn.close()

    def cleanup_old_activities(self, board_id: int | None = None) -> int:
        """
        Remove activities older than retention period.

        Args:
            board_id: Optional board ID to clean up. If None, cleans all boards.

        Returns:
            Number of activities deleted.
        """
        with self._lock:
            conn = self._connect()
            try:
                if board_id is not None:
                    # Get board-specific retention or use default
                    board = self._get_board_by_id(conn, board_id, include_deleted=True)
                    if not board:
                        return 0

                    retention_days = board.get("activity_retention_days") or self.DEFAULT_ACTIVITY_RETENTION_DAYS
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")

                    cur = conn.execute(
                        "DELETE FROM kanban_activities WHERE board_id = ? AND created_at < ?",
                        (board_id, cutoff)
                    )
                else:
                    now = datetime.now(timezone.utc)
                    board_rows = conn.execute(
                        """
                        SELECT id, COALESCE(activity_retention_days, ?) as retention_days
                        FROM kanban_boards
                        WHERE user_id = ?
                        """,
                        (self.DEFAULT_ACTIVITY_RETENTION_DAYS, self.user_id),
                    ).fetchall()
                    deleted = 0
                    for row in board_rows:
                        retention_days = int(row["retention_days"]) if row["retention_days"] else self.DEFAULT_ACTIVITY_RETENTION_DAYS
                        if retention_days <= 0:
                            retention_days = self.DEFAULT_ACTIVITY_RETENTION_DAYS
                        cutoff = (now - timedelta(days=retention_days)).strftime("%Y-%m-%d %H:%M:%S")
                        cur = conn.execute(
                            "DELETE FROM kanban_activities WHERE board_id = ? AND created_at < ?",
                            (row["id"], cutoff)
                        )
                        deleted += cur.rowcount
                    conn.commit()
                    logger.info(f"Cleaned up {deleted} old activities")
                    return deleted

                deleted = cur.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted} old activities")
                return deleted

            finally:
                conn.close()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_board_with_lists_and_cards(
        self,
        board_id: int,
        include_archived: bool = False
    ) -> dict[str, Any] | None:
        """
        Get a board with all its lists and cards nested.

        Args:
            board_id: The board ID.
            include_archived: If True, include archived items.

        Returns:
            Board with nested lists, each containing their cards.
        """
        with self._lock:
            conn = self._connect()
            try:
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    return None

                board["labels"] = self._list_labels_for_board(conn, board_id)

                # Get lists
                lists = self.list_lists(board_id, include_archived=include_archived)

                # Get cards for each list
                for lst in lists:
                    lst["cards"] = self._list_cards_with_summary_for_list(
                        conn,
                        lst["id"],
                        include_archived=include_archived,
                        include_deleted=False,
                    )
                    lst["card_count"] = len(lst["cards"])

                board["lists"] = lists
                board["total_cards"] = sum(lst["card_count"] for lst in lists)

                return board

            finally:
                conn.close()

    def get_card_count_for_list(self, list_id: int) -> int:
        """Get the count of non-deleted, non-archived cards in a list."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_cards WHERE list_id = ? AND deleted = 0 AND archived = 0",
                    (list_id,)
                )
                return cur.fetchone()["cnt"]
            finally:
                conn.close()

    def get_card_count_for_board(self, board_id: int) -> int:
        """Get the count of non-deleted cards in a board."""
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_cards WHERE board_id = ? AND deleted = 0",
                    (board_id,)
                )
                return cur.fetchone()["cnt"]
            finally:
                conn.close()

    def purge_deleted_items(self, days_old: int = 30) -> dict[str, int]:
        """
        Permanently delete items that have been soft-deleted for more than N days.

        Args:
            days_old: Number of days since deletion.

        Returns:
            Dictionary with counts of items purged per entity type.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_old)).strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            conn = self._connect()
            try:
                counts = {}

                # Delete cards first (children)
                cur = conn.execute(
                    "DELETE FROM kanban_cards WHERE deleted = 1 AND deleted_at < ?",
                    (cutoff,)
                )
                counts["cards"] = cur.rowcount

                # Delete lists
                cur = conn.execute(
                    "DELETE FROM kanban_lists WHERE deleted = 1 AND deleted_at < ?",
                    (cutoff,)
                )
                counts["lists"] = cur.rowcount

                # Delete boards
                cur = conn.execute(
                    "DELETE FROM kanban_boards WHERE deleted = 1 AND deleted_at < ? AND user_id = ?",
                    (cutoff, self.user_id)
                )
                counts["boards"] = cur.rowcount

                conn.commit()

                logger.info(f"Purged deleted items: {counts}")
                return counts

            finally:
                conn.close()

    # =========================================================================
    # LABEL OPERATIONS
    # =========================================================================

    # Predefined color palette for labels
    LABEL_COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "gray"}

    def create_label(
        self,
        board_id: int,
        name: str,
        color: str
    ) -> dict[str, Any]:
        """
        Create a new label for a board.

        Args:
            board_id: The board ID.
            name: Label name.
            color: Label color (must be from LABEL_COLORS).

        Returns:
            The created label as a dictionary.

        Raises:
            NotFoundError: If board not found.
            InputError: If validation fails.
        """
        # Validate inputs
        if not name or not name.strip():
            raise InputError("Label name is required")  # noqa: TRY003
        name = name.strip()
        if len(name) > 50:
            raise InputError("Label name must be 50 characters or less")  # noqa: TRY003

        if not color or color.lower() not in self.LABEL_COLORS:
            raise InputError(f"Invalid color. Must be one of: {', '.join(sorted(self.LABEL_COLORS))}")  # noqa: TRY003
        color = color.lower()

        label_uuid = _generate_uuid()
        now = _utcnow_iso()

        with self._lock:
            conn = self._connect()
            try:
                # Verify board exists and belongs to user
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003

                # Check label limit
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_labels WHERE board_id = ?",
                    (board_id,)
                )
                count = cur.fetchone()["cnt"]
                if count >= self.MAX_LABELS_PER_BOARD:
                    raise InputError(f"Maximum labels ({self.MAX_LABELS_PER_BOARD}) per board reached")  # noqa: TRY003

                # Insert label
                cur = conn.execute(
                    """
                    INSERT INTO kanban_labels (uuid, board_id, name, color, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (label_uuid, board_id, name, color, now, now)
                )
                label_id = cur.lastrowid

                # Log activity
                self._log_activity_internal(
                    conn, board_id, "label_created", "label", entity_id=label_id,
                    details={"name": name, "color": color}
                )

                conn.commit()

                return self._get_label_by_id(conn, label_id)

            finally:
                conn.close()

    def get_label(self, label_id: int) -> dict[str, Any] | None:
        """Get a label by ID."""
        with self._lock:
            conn = self._connect()
            try:
                return self._get_label_by_id(conn, label_id)
            finally:
                conn.close()

    def _get_label_by_id(self, conn: sqlite3.Connection, label_id: int) -> dict[str, Any] | None:
        """Internal method to get a label by ID."""
        sql = """
            SELECT l.id, l.uuid, l.board_id, l.name, l.color, l.created_at, l.updated_at
            FROM kanban_labels l
            JOIN kanban_boards b ON l.board_id = b.id
            WHERE l.id = ? AND b.user_id = ?
        """
        cur = conn.execute(sql, (label_id, self.user_id))
        row = cur.fetchone()

        if not row:
            return None

        return self._row_to_label_dict(row)

    def _row_to_label_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a label row to a dictionary."""
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "board_id": row["board_id"],
            "name": row["name"],
            "color": row["color"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }

    def _list_labels_for_board(self, conn: sqlite3.Connection, board_id: int) -> list[dict[str, Any]]:
        """Fetch labels for a board using an existing connection."""
        cur = conn.execute(
            """
            SELECT l.id, l.uuid, l.board_id, l.name, l.color, l.created_at, l.updated_at
            FROM kanban_labels l
            WHERE l.board_id = ?
            ORDER BY l.name ASC
            """,
            (board_id,),
        )
        return [self._row_to_label_dict(row) for row in cur.fetchall()]

    def list_labels(self, board_id: int) -> list[dict[str, Any]]:
        """
        Get all labels for a board.

        Args:
            board_id: The board ID.

        Returns:
            List of labels.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify board exists and belongs to user
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003

                sql = """
                    SELECT l.id, l.uuid, l.board_id, l.name, l.color, l.created_at, l.updated_at
                    FROM kanban_labels l
                    WHERE l.board_id = ?
                    ORDER BY l.name ASC
                """
                cur = conn.execute(sql, (board_id,))

                return [self._row_to_label_dict(row) for row in cur.fetchall()]

            finally:
                conn.close()

    def update_label(
        self,
        label_id: int,
        name: str | None = None,
        color: str | None = None
    ) -> dict[str, Any]:
        """Update a label."""
        updated_label: dict[str, Any] | None = None
        affected_card_ids: list[int] = []
        with self._lock:
            conn = self._connect()
            try:
                label = self._get_label_by_id(conn, label_id)
                if not label:
                    raise NotFoundError("Label not found", entity="label", entity_id=label_id)  # noqa: TRY003

                updates = []
                params: list[Any] = []

                if name is not None:
                    if not name.strip():
                        raise InputError("Label name cannot be empty")  # noqa: TRY003
                    name = name.strip()
                    if len(name) > 50:
                        raise InputError("Label name must be 50 characters or less")  # noqa: TRY003
                    updates.append("name = ?")
                    params.append(name)

                if color is not None:
                    if color.lower() not in self.LABEL_COLORS:
                        raise InputError(f"Invalid color. Must be one of: {', '.join(sorted(self.LABEL_COLORS))}")  # noqa: TRY003
                    updates.append("color = ?")
                    params.append(color.lower())

                if not updates:
                    return label

                cur = conn.execute(
                    "SELECT card_id FROM kanban_card_labels WHERE label_id = ?",
                    (label_id,),
                )
                affected_card_ids = [row["card_id"] for row in cur.fetchall()]

                # Build list of updated fields
                updated_fields = []
                if name is not None:
                    updated_fields.append("name")
                if color is not None:
                    updated_fields.append("color")

                updates.append("updated_at = ?")
                params.append(_utcnow_iso())
                params.append(label_id)

                sql = f"UPDATE kanban_labels SET {', '.join(updates)} WHERE id = ?"  # nosec B608
                conn.execute(sql, params)

                # Log activity
                self._log_activity_internal(
                    conn, label["board_id"], "label_updated", "label", entity_id=label_id,
                    details={"updated_fields": updated_fields}
                )

                conn.commit()

                updated_label = self._get_label_by_id(conn, label_id)

            finally:
                conn.close()
        if affected_card_ids:
            self._sync_vector_index_for_card_ids(affected_card_ids)
        return updated_label

    def delete_label(self, label_id: int) -> bool:
        """
        Delete a label (hard delete).

        This also removes all card_label associations.
        """
        deleted = False
        affected_card_ids: list[int] = []
        with self._lock:
            conn = self._connect()
            try:
                label = self._get_label_by_id(conn, label_id)
                if not label:
                    return False

                # Log activity before deleting
                self._log_activity_internal(
                    conn, label["board_id"], "label_deleted", "label", entity_id=label_id,
                    details={"name": label["name"], "color": label["color"]}
                )

                cur = conn.execute(
                    "SELECT card_id FROM kanban_card_labels WHERE label_id = ?",
                    (label_id,),
                )
                affected_card_ids = [row["card_id"] for row in cur.fetchall()]

                conn.execute("DELETE FROM kanban_labels WHERE id = ?", (label_id,))
                conn.commit()

                deleted = True

            finally:
                conn.close()
        if deleted and affected_card_ids:
            self._sync_vector_index_for_card_ids(affected_card_ids)
        return deleted

    def assign_label_to_card(self, card_id: int, label_id: int) -> bool:
        """
        Assign a label to a card.

        Args:
            card_id: The card ID.
            label_id: The label ID.

        Returns:
            True if assigned successfully.

        Raises:
            NotFoundError: If card or label not found.
            InputError: If label doesn't belong to the card's board.
        """
        assigned = False
        with self._lock:
            conn = self._connect()
            try:
                # Get card
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                # Get label
                label = self._get_label_by_id(conn, label_id)
                if not label:
                    raise NotFoundError("Label not found", entity="label", entity_id=label_id)  # noqa: TRY003

                # Verify label belongs to the same board as the card
                if label["board_id"] != card["board_id"]:
                    raise InputError("Label does not belong to the card's board")  # noqa: TRY003

                # Insert (ignore if already exists)
                now = _utcnow_iso()
                try:
                    conn.execute(
                        "INSERT INTO kanban_card_labels (card_id, label_id, created_at) VALUES (?, ?, ?)",
                        (card_id, label_id, now)
                    )

                    # Log activity
                    self._log_activity_internal(
                        conn, card["board_id"], "label_assigned", "card", entity_id=card_id,
                        list_id=card["list_id"], card_id=card_id,
                        details={"label_id": label_id, "label_name": label["name"]}
                    )

                    conn.commit()
                    assigned = True
                except sqlite3.IntegrityError:
                    # Already exists, that's fine
                    pass

            finally:
                conn.close()

        if assigned:
            self._sync_vector_index_for_card_id(card_id)
        return assigned

    def remove_label_from_card(self, card_id: int, label_id: int) -> bool:
        """
        Remove a label from a card.

        Args:
            card_id: The card ID.
            label_id: The label ID.

        Returns:
            True if removed, False if the association didn't exist.
        """
        removed = False
        with self._lock:
            conn = self._connect()
            try:
                # Verify card belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                # Get label info for activity log
                label = self._get_label_by_id(conn, label_id)

                cur = conn.execute(
                    "DELETE FROM kanban_card_labels WHERE card_id = ? AND label_id = ?",
                    (card_id, label_id)
                )

                if cur.rowcount > 0 and label:
                    # Log activity
                    self._log_activity_internal(
                        conn, card["board_id"], "label_removed", "card", entity_id=card_id,
                        list_id=card["list_id"], card_id=card_id,
                        details={"label_id": label_id, "label_name": label["name"]}
                    )

                conn.commit()

                removed = cur.rowcount > 0

            finally:
                conn.close()

        if removed:
            self._sync_vector_index_for_card_id(card_id)
        return removed

    def get_card_labels(self, card_id: int) -> list[dict[str, Any]]:
        """Get all labels assigned to a card."""
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                sql = """
                    SELECT l.id, l.uuid, l.board_id, l.name, l.color, l.created_at, l.updated_at
                    FROM kanban_labels l
                    JOIN kanban_card_labels cl ON l.id = cl.label_id
                    WHERE cl.card_id = ?
                    ORDER BY l.name ASC
                """
                cur = conn.execute(sql, (card_id,))

                return [self._row_to_label_dict(row) for row in cur.fetchall()]

            finally:
                conn.close()

    # =========================================================================
    # CHECKLIST OPERATIONS
    # =========================================================================

    def create_checklist(
        self,
        card_id: int,
        name: str,
        position: int | None = None
    ) -> dict[str, Any]:
        """
        Create a new checklist for a card.

        Args:
            card_id: The card ID.
            name: Checklist name.
            position: Optional position (auto-assigned if not provided).

        Returns:
            The created checklist as a dictionary.
        """
        # Validate inputs
        if not name or not name.strip():
            raise InputError("Checklist name is required")  # noqa: TRY003
        name = name.strip()
        if len(name) > 255:
            raise InputError("Checklist name must be 255 characters or less")  # noqa: TRY003

        checklist_uuid = _generate_uuid()
        now = _utcnow_iso()
        checklist: dict[str, Any] | None = None

        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                # Check checklist limit
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_checklists WHERE card_id = ?",
                    (card_id,)
                )
                count = cur.fetchone()["cnt"]
                if count >= self.MAX_CHECKLISTS_PER_CARD:
                    raise InputError(f"Maximum checklists ({self.MAX_CHECKLISTS_PER_CARD}) per card reached")  # noqa: TRY003

                # Get next position if not provided
                if position is None:
                    cur = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM kanban_checklists WHERE card_id = ?",
                        (card_id,)
                    )
                    position = cur.fetchone()["next_pos"]

                # Insert checklist
                cur = conn.execute(
                    """
                    INSERT INTO kanban_checklists (uuid, card_id, name, position, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (checklist_uuid, card_id, name, position, now, now)
                )
                checklist_id = cur.lastrowid

                # Log activity
                self._log_activity_internal(
                    conn, card["board_id"], "checklist_created", "checklist", entity_id=checklist_id,
                    list_id=card["list_id"], card_id=card_id,
                    details={"name": name}
                )

                conn.commit()

                checklist = self._get_checklist_by_id(conn, checklist_id)

            finally:
                conn.close()

        if checklist:
            self._sync_vector_index_for_card_id(card_id)
        return checklist

    def get_checklist(self, checklist_id: int) -> dict[str, Any] | None:
        """Get a checklist by ID."""
        with self._lock:
            conn = self._connect()
            try:
                return self._get_checklist_by_id(conn, checklist_id)
            finally:
                conn.close()

    def _get_checklist_by_id(self, conn: sqlite3.Connection, checklist_id: int) -> dict[str, Any] | None:
        """Internal method to get a checklist by ID."""
        sql = """
            SELECT ch.id, ch.uuid, ch.card_id, ch.name, ch.position, ch.created_at, ch.updated_at
            FROM kanban_checklists ch
            JOIN kanban_cards c ON ch.card_id = c.id
            JOIN kanban_boards b ON c.board_id = b.id
            WHERE ch.id = ? AND b.user_id = ?
        """
        cur = conn.execute(sql, (checklist_id, self.user_id))
        row = cur.fetchone()

        if not row:
            return None

        return self._row_to_checklist_dict(row)

    def _row_to_checklist_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a checklist row to a dictionary."""
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "card_id": row["card_id"],
            "name": row["name"],
            "position": row["position"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }

    def list_checklists(self, card_id: int) -> list[dict[str, Any]]:
        """
        Get all checklists for a card, ordered by position.

        Args:
            card_id: The card ID.

        Returns:
            List of checklists ordered by position.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                sql = """
                    SELECT ch.id, ch.uuid, ch.card_id, ch.name, ch.position, ch.created_at, ch.updated_at
                    FROM kanban_checklists ch
                    WHERE ch.card_id = ?
                    ORDER BY ch.position ASC
                """
                cur = conn.execute(sql, (card_id,))

                return [self._row_to_checklist_dict(row) for row in cur.fetchall()]

            finally:
                conn.close()

    def update_checklist(
        self,
        checklist_id: int,
        name: str | None = None
    ) -> dict[str, Any]:
        """Update a checklist."""
        updated_checklist: dict[str, Any] | None = None
        card_id: int | None = None
        changed = False
        with self._lock:
            conn = self._connect()
            try:
                checklist = self._get_checklist_by_id(conn, checklist_id)
                if not checklist:
                    raise NotFoundError("Checklist not found", entity="checklist", entity_id=checklist_id)  # noqa: TRY003

                # Get the card for activity logging
                card = self._get_card_by_id(conn, checklist["card_id"])

                updates = []
                params: list[Any] = []

                if name is not None:
                    if not name.strip():
                        raise InputError("Checklist name cannot be empty")  # noqa: TRY003
                    name = name.strip()
                    if len(name) > 255:
                        raise InputError("Checklist name must be 255 characters or less")  # noqa: TRY003
                    updates.append("name = ?")
                    params.append(name)

                if not updates:
                    updated_checklist = checklist
                    card_id = checklist["card_id"]
                    return updated_checklist

                updates.append("updated_at = ?")
                params.append(_utcnow_iso())
                params.append(checklist_id)

                sql = f"UPDATE kanban_checklists SET {', '.join(updates)} WHERE id = ?"  # nosec B608
                conn.execute(sql, params)
                changed = True
                card_id = checklist["card_id"]

                # Log activity
                if card:
                    self._log_activity_internal(
                        conn, card["board_id"], "checklist_updated", "checklist", entity_id=checklist_id,
                        list_id=card["list_id"], card_id=card["id"],
                        details={"name": name or checklist["name"]}
                    )

                conn.commit()

                updated_checklist = self._get_checklist_by_id(conn, checklist_id)

            finally:
                conn.close()

        if changed and card_id:
            self._sync_vector_index_for_card_id(card_id)
        return updated_checklist

    def reorder_checklists(self, card_id: int, checklist_ids: list[int]) -> list[dict[str, Any]]:
        """
        Reorder checklists on a card.

        Args:
            card_id: The card ID.
            checklist_ids: Checklist IDs in the desired order.

        Returns:
            Updated checklists in new order.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                # Verify all checklists exist and belong to the card
                cur = conn.execute(
                    f"SELECT id FROM kanban_checklists WHERE card_id = ? AND id IN ({','.join('?' * len(checklist_ids))})",  # nosec B608
                    [card_id] + checklist_ids
                )
                existing_ids = {row["id"] for row in cur.fetchall()}

                if len(existing_ids) != len(checklist_ids):
                    missing = set(checklist_ids) - existing_ids
                    raise InputError(f"Checklists not found or don't belong to card: {missing}")  # noqa: TRY003

                # Update positions
                now = _utcnow_iso()
                for position, checklist_id in enumerate(checklist_ids):
                    conn.execute(
                        "UPDATE kanban_checklists SET position = ?, updated_at = ? WHERE id = ?",
                        (position, now, checklist_id)
                    )

                # Log activity
                self._log_activity_internal(
                    conn, card["board_id"], "checklists_reordered", "card", entity_id=card_id,
                    list_id=card["list_id"], card_id=card_id,
                    details={"checklist_ids": checklist_ids}
                )

                conn.commit()

                return self.list_checklists(card_id)

            finally:
                conn.close()

    def delete_checklist(self, checklist_id: int) -> bool:
        """
        Delete a checklist (hard delete).

        This also cascades to delete all checklist items.
        """
        deleted = False
        card_id: int | None = None
        with self._lock:
            conn = self._connect()
            try:
                checklist = self._get_checklist_by_id(conn, checklist_id)
                if not checklist:
                    return False

                # Get the card for activity logging
                card = self._get_card_by_id(conn, checklist["card_id"])
                card_id = checklist["card_id"]

                # Log activity before deleting
                if card:
                    self._log_activity_internal(
                        conn, card["board_id"], "checklist_deleted", "checklist", entity_id=checklist_id,
                        list_id=card["list_id"], card_id=card["id"],
                        details={"name": checklist["name"]}
                    )

                conn.execute("DELETE FROM kanban_checklists WHERE id = ?", (checklist_id,))
                conn.commit()

                deleted = True

            finally:
                conn.close()

        if deleted and card_id:
            self._sync_vector_index_for_card_id(card_id)
        return deleted

    # =========================================================================
    # CHECKLIST ITEM OPERATIONS
    # =========================================================================

    def create_checklist_item(
        self,
        checklist_id: int,
        name: str,
        position: int | None = None,
        checked: bool = False
    ) -> dict[str, Any]:
        """
        Create a new checklist item.

        Args:
            checklist_id: The checklist ID.
            name: Item name.
            position: Optional position (auto-assigned if not provided).
            checked: Whether the item starts checked.

        Returns:
            The created checklist item as a dictionary.
        """
        # Validate inputs
        if not name or not name.strip():
            raise InputError("Checklist item name is required")  # noqa: TRY003
        name = name.strip()
        if len(name) > 500:
            raise InputError("Checklist item name must be 500 characters or less")  # noqa: TRY003

        item_uuid = _generate_uuid()
        now = _utcnow_iso()

        created_item: dict[str, Any] | None = None
        card_id: int | None = None
        with self._lock:
            conn = self._connect()
            try:
                # Verify checklist exists and belongs to user
                checklist = self._get_checklist_by_id(conn, checklist_id)
                if not checklist:
                    raise NotFoundError("Checklist not found", entity="checklist", entity_id=checklist_id)  # noqa: TRY003

                # Check item limit
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_checklist_items WHERE checklist_id = ?",
                    (checklist_id,)
                )
                count = cur.fetchone()["cnt"]
                if count >= self.MAX_CHECKLIST_ITEMS_PER_CHECKLIST:
                    raise InputError(f"Maximum items ({self.MAX_CHECKLIST_ITEMS_PER_CHECKLIST}) per checklist reached")  # noqa: TRY003

                # Get next position if not provided
                if position is None:
                    cur = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM kanban_checklist_items WHERE checklist_id = ?",
                        (checklist_id,)
                    )
                    position = cur.fetchone()["next_pos"]

                checked_at = now if checked else None

                # Get the card for activity logging
                card = self._get_card_by_id(conn, checklist["card_id"])
                card_id = checklist["card_id"]

                # Insert item
                cur = conn.execute(
                    """
                    INSERT INTO kanban_checklist_items (uuid, checklist_id, name, position, checked, checked_at, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (item_uuid, checklist_id, name, position, 1 if checked else 0, checked_at, now, now)
                )
                item_id = cur.lastrowid

                # Log activity
                if card:
                    self._log_activity_internal(
                        conn, card["board_id"], "checklist_item_created", "checklist_item", entity_id=item_id,
                        list_id=card["list_id"], card_id=card["id"],
                        details={"name": name, "checklist_name": checklist["name"]}
                    )

                conn.commit()

                created_item = self._get_checklist_item_by_id(conn, item_id)

            finally:
                conn.close()
        if created_item and card_id:
            self._sync_vector_index_for_card_id(card_id)
        return created_item

    def get_checklist_item(self, item_id: int) -> dict[str, Any] | None:
        """Get a checklist item by ID."""
        with self._lock:
            conn = self._connect()
            try:
                return self._get_checklist_item_by_id(conn, item_id)
            finally:
                conn.close()

    def _get_checklist_item_by_id(self, conn: sqlite3.Connection, item_id: int) -> dict[str, Any] | None:
        """Internal method to get a checklist item by ID."""
        sql = """
            SELECT ci.id, ci.uuid, ci.checklist_id, ci.name, ci.position, ci.checked, ci.checked_at, ci.created_at, ci.updated_at
            FROM kanban_checklist_items ci
            JOIN kanban_checklists ch ON ci.checklist_id = ch.id
            JOIN kanban_cards c ON ch.card_id = c.id
            JOIN kanban_boards b ON c.board_id = b.id
            WHERE ci.id = ? AND b.user_id = ?
        """
        cur = conn.execute(sql, (item_id, self.user_id))
        row = cur.fetchone()

        if not row:
            return None

        return self._row_to_checklist_item_dict(row)

    def _row_to_checklist_item_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a checklist item row to a dictionary."""
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "checklist_id": row["checklist_id"],
            "name": row["name"],
            "position": row["position"],
            "checked": bool(row["checked"]),
            "checked_at": row["checked_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }

    def list_checklist_items(self, checklist_id: int) -> list[dict[str, Any]]:
        """
        Get all items for a checklist, ordered by position.

        Args:
            checklist_id: The checklist ID.

        Returns:
            List of checklist items ordered by position.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify checklist exists and belongs to user
                checklist = self._get_checklist_by_id(conn, checklist_id)
                if not checklist:
                    raise NotFoundError("Checklist not found", entity="checklist", entity_id=checklist_id)  # noqa: TRY003

                sql = """
                    SELECT ci.id, ci.uuid, ci.checklist_id, ci.name, ci.position, ci.checked, ci.checked_at, ci.created_at, ci.updated_at
                    FROM kanban_checklist_items ci
                    WHERE ci.checklist_id = ?
                    ORDER BY ci.position ASC
                """
                cur = conn.execute(sql, (checklist_id,))

                return [self._row_to_checklist_item_dict(row) for row in cur.fetchall()]

            finally:
                conn.close()

    def update_checklist_item(
        self,
        item_id: int,
        name: str | None = None,
        checked: bool | None = None
    ) -> dict[str, Any]:
        """Update a checklist item."""
        updated_item: dict[str, Any] | None = None
        card_id: int | None = None
        should_reindex = name is not None
        with self._lock:
            conn = self._connect()
            try:
                item = self._get_checklist_item_by_id(conn, item_id)
                if not item:
                    raise NotFoundError("Checklist item not found", entity="checklist_item", entity_id=item_id)  # noqa: TRY003

                updates = []
                params: list[Any] = []
                now = _utcnow_iso()

                if name is not None:
                    if not name.strip():
                        raise InputError("Checklist item name cannot be empty")  # noqa: TRY003
                    name = name.strip()
                    if len(name) > 500:
                        raise InputError("Checklist item name must be 500 characters or less")  # noqa: TRY003
                    updates.append("name = ?")
                    params.append(name)

                if checked is not None:
                    updates.append("checked = ?")
                    params.append(1 if checked else 0)
                    # Update checked_at timestamp
                    if checked and not item["checked"]:
                        updates.append("checked_at = ?")
                        params.append(now)
                    elif not checked and item["checked"]:
                        updates.append("checked_at = NULL")

                if not updates:
                    return item

                updates.append("updated_at = ?")
                params.append(now)
                params.append(item_id)

                sql = f"UPDATE kanban_checklist_items SET {', '.join(updates)} WHERE id = ?"  # nosec B608
                conn.execute(sql, params)

                # Log activity for check/uncheck (important user actions)
                if checked is not None:
                    checklist = self._get_checklist_by_id(conn, item["checklist_id"])
                    if checklist:
                        card = self._get_card_by_id(conn, checklist["card_id"])
                        if card:
                            card_id = checklist["card_id"]
                            action = "checklist_item_checked" if checked else "checklist_item_unchecked"
                            self._log_activity_internal(
                                conn, card["board_id"], action, "checklist_item", entity_id=item_id,
                                list_id=card["list_id"], card_id=card["id"],
                                details={"name": item["name"]}
                            )
                    if checklist and card_id is None:
                        card_id = checklist["card_id"]
                elif should_reindex:
                    checklist = self._get_checklist_by_id(conn, item["checklist_id"])
                    if checklist:
                        card_id = checklist["card_id"]

                conn.commit()

                updated_item = self._get_checklist_item_by_id(conn, item_id)

            finally:
                conn.close()
        if should_reindex and card_id:
            self._sync_vector_index_for_card_id(card_id)
        return updated_item

    def reorder_checklist_items(self, checklist_id: int, item_ids: list[int]) -> list[dict[str, Any]]:
        """
        Reorder items in a checklist.

        Args:
            checklist_id: The checklist ID.
            item_ids: Item IDs in the desired order.

        Returns:
            Updated items in new order.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify checklist exists
                checklist = self._get_checklist_by_id(conn, checklist_id)
                if not checklist:
                    raise NotFoundError("Checklist not found", entity="checklist", entity_id=checklist_id)  # noqa: TRY003

                # Verify all items exist and belong to the checklist
                cur = conn.execute(
                    f"SELECT id FROM kanban_checklist_items WHERE checklist_id = ? AND id IN ({','.join('?' * len(item_ids))})",  # nosec B608
                    [checklist_id] + item_ids
                )
                existing_ids = {row["id"] for row in cur.fetchall()}

                if len(existing_ids) != len(item_ids):
                    missing = set(item_ids) - existing_ids
                    raise InputError(f"Items not found or don't belong to checklist: {missing}")  # noqa: TRY003

                # Update positions
                now = _utcnow_iso()
                for position, item_id in enumerate(item_ids):
                    conn.execute(
                        "UPDATE kanban_checklist_items SET position = ?, updated_at = ? WHERE id = ?",
                        (position, now, item_id)
                    )

                conn.commit()

                return self.list_checklist_items(checklist_id)

            finally:
                conn.close()

    def delete_checklist_item(self, item_id: int) -> bool:
        """Delete a checklist item (hard delete)."""
        deleted = False
        card_id: int | None = None
        with self._lock:
            conn = self._connect()
            try:
                item = self._get_checklist_item_by_id(conn, item_id)
                if not item:
                    return False

                # Get checklist and card for activity logging
                checklist = self._get_checklist_by_id(conn, item["checklist_id"])
                if checklist:
                    card_id = checklist["card_id"]
                    card = self._get_card_by_id(conn, checklist["card_id"])
                    if card:
                        self._log_activity_internal(
                            conn, card["board_id"], "checklist_item_deleted", "checklist_item", entity_id=item_id,
                            list_id=card["list_id"], card_id=card["id"],
                            details={"name": item["name"]}
                        )

                conn.execute("DELETE FROM kanban_checklist_items WHERE id = ?", (item_id,))
                conn.commit()

                deleted = True

            finally:
                conn.close()
        if deleted and card_id:
            self._sync_vector_index_for_card_id(card_id)
        return deleted

    def get_checklist_with_items(self, checklist_id: int) -> dict[str, Any] | None:
        """Get a checklist with all its items included."""
        with self._lock:
            conn = self._connect()
            try:
                checklist = self._get_checklist_by_id(conn, checklist_id)
                if not checklist:
                    return None

                checklist["items"] = self.list_checklist_items(checklist_id)
                # Calculate progress
                total = len(checklist["items"])
                checked = sum(1 for item in checklist["items"] if item["checked"])
                checklist["total_items"] = total
                checklist["checked_items"] = checked
                checklist["progress_percent"] = round(checked / total * 100) if total > 0 else 0

                return checklist

            finally:
                conn.close()

    # =========================================================================
    # COMMENT OPERATIONS
    # =========================================================================

    def create_comment(
        self,
        card_id: int,
        content: str
    ) -> dict[str, Any]:
        """
        Create a new comment on a card.

        Args:
            card_id: The card ID.
            content: Comment content (markdown supported).

        Returns:
            The created comment as a dictionary.
        """
        # Validate inputs
        if not content or not content.strip():
            raise InputError("Comment content is required")  # noqa: TRY003
        content = content.strip()
        if len(content) > self.MAX_COMMENT_SIZE:
            raise InputError(f"Comment must be {self.MAX_COMMENT_SIZE} characters or less")  # noqa: TRY003

        comment_uuid = _generate_uuid()
        now = _utcnow_iso()

        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                # Check comment limit
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_comments WHERE card_id = ? AND deleted = 0",
                    (card_id,)
                )
                count = cur.fetchone()["cnt"]
                if count >= self.MAX_COMMENTS_PER_CARD:
                    raise InputError(f"Maximum comments ({self.MAX_COMMENTS_PER_CARD}) per card reached")  # noqa: TRY003

                # Insert comment
                cur = conn.execute(
                    """
                    INSERT INTO kanban_comments (uuid, card_id, user_id, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (comment_uuid, card_id, self.user_id, content, now, now)
                )
                comment_id = cur.lastrowid

                # Log activity
                self._log_activity_internal(
                    conn, card["board_id"], "comment_created", "comment", entity_id=comment_id,
                    list_id=card["list_id"], card_id=card_id,
                    details={"preview": content[:100] if len(content) > 100 else content}
                )

                conn.commit()

                return self._get_comment_by_id(conn, comment_id)

            finally:
                conn.close()

    def get_comment(self, comment_id: int, include_deleted: bool = False) -> dict[str, Any] | None:
        """Get a comment by ID."""
        with self._lock:
            conn = self._connect()
            try:
                return self._get_comment_by_id(conn, comment_id, include_deleted)
            finally:
                conn.close()

    def _get_comment_by_id(
        self,
        conn: sqlite3.Connection,
        comment_id: int,
        include_deleted: bool = False
    ) -> dict[str, Any] | None:
        """Internal method to get a comment by ID."""
        sql = """
            SELECT cm.id, cm.uuid, cm.card_id, cm.user_id, cm.content, cm.created_at, cm.updated_at, cm.deleted
            FROM kanban_comments cm
            JOIN kanban_cards c ON cm.card_id = c.id
            JOIN kanban_boards b ON c.board_id = b.id
            WHERE cm.id = ? AND b.user_id = ?
        """
        params: list[Any] = [comment_id, self.user_id]

        if not include_deleted:
            sql += " AND cm.deleted = 0"

        cur = conn.execute(sql, params)
        row = cur.fetchone()

        if not row:
            return None

        return self._row_to_comment_dict(row)

    def _row_to_comment_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a comment row to a dictionary."""
        return {
            "id": row["id"],
            "uuid": row["uuid"],
            "card_id": row["card_id"],
            "user_id": row["user_id"],
            "content": row["content"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deleted": bool(row["deleted"])
        }

    def list_comments(
        self,
        card_id: int,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Get comments for a card with pagination.

        Args:
            card_id: The card ID.
            include_deleted: If True, include soft-deleted comments.
            limit: Maximum comments to return.
            offset: Number of comments to skip.

        Returns:
            Tuple of (comments list, total count).
        """
        if limit < 1 or offset < 0:
            raise InputError("limit must be positive and offset must be non-negative")  # noqa: TRY003
        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                conditions = ["cm.card_id = ?"]
                params: list[Any] = [card_id]

                if not include_deleted:
                    conditions.append("cm.deleted = 0")

                where_clause = " AND ".join(conditions)

                # Get total count
                count_sql = f"SELECT COUNT(*) as cnt FROM kanban_comments cm WHERE {where_clause}"  # nosec B608
                cur = conn.execute(count_sql, params)
                total = cur.fetchone()["cnt"]

                # Get paginated results
                sql = """
                    SELECT cm.id, cm.uuid, cm.card_id, cm.user_id, cm.content, cm.created_at, cm.updated_at, cm.deleted
                    FROM kanban_comments cm
                    WHERE {where_clause}
                    ORDER BY cm.created_at DESC
                    LIMIT ? OFFSET ?
                """.format_map(locals())  # nosec B608
                params.extend([limit, offset])
                cur = conn.execute(sql, params)

                comments = [self._row_to_comment_dict(row) for row in cur.fetchall()]

                return comments, total

            finally:
                conn.close()

    def update_comment(
        self,
        comment_id: int,
        content: str
    ) -> dict[str, Any]:
        """Update a comment."""
        # Validate inputs
        if not content or not content.strip():
            raise InputError("Comment content is required")  # noqa: TRY003
        content = content.strip()
        if len(content) > self.MAX_COMMENT_SIZE:
            raise InputError(f"Comment must be {self.MAX_COMMENT_SIZE} characters or less")  # noqa: TRY003

        with self._lock:
            conn = self._connect()
            try:
                comment = self._get_comment_by_id(conn, comment_id)
                if not comment:
                    raise NotFoundError("Comment not found", entity="comment", entity_id=comment_id)  # noqa: TRY003

                # Only the comment author can edit
                if comment["user_id"] != self.user_id:
                    raise InputError("You can only edit your own comments")  # noqa: TRY003

                # Get the card for activity logging
                card = self._get_card_by_id(conn, comment["card_id"])

                now = _utcnow_iso()
                conn.execute(
                    "UPDATE kanban_comments SET content = ?, updated_at = ? WHERE id = ?",
                    (content, now, comment_id)
                )

                # Log activity
                if card:
                    self._log_activity_internal(
                        conn, card["board_id"], "comment_updated", "comment", entity_id=comment_id,
                        list_id=card["list_id"], card_id=card["id"],
                        details={"preview": content[:100] if len(content) > 100 else content}
                    )

                conn.commit()

                return self._get_comment_by_id(conn, comment_id)

            finally:
                conn.close()

    def delete_comment(self, comment_id: int, hard_delete: bool = False) -> bool:
        """Delete a comment (soft delete by default)."""
        with self._lock:
            conn = self._connect()
            try:
                comment = self._get_comment_by_id(conn, comment_id, include_deleted=True)
                if not comment:
                    return False

                # Get the card for activity logging
                card = self._get_card_by_id(conn, comment["card_id"])

                if hard_delete:
                    conn.execute("DELETE FROM kanban_comments WHERE id = ?", (comment_id,))
                else:
                    now = _utcnow_iso()
                    conn.execute(
                        "UPDATE kanban_comments SET deleted = 1, updated_at = ? WHERE id = ?",
                        (now, comment_id)
                    )
                    # Log activity (only for soft delete)
                    if card:
                        self._log_activity_internal(
                            conn, card["board_id"], "comment_deleted", "comment", entity_id=comment_id,
                            list_id=card["list_id"], card_id=card["id"],
                            details={}
                        )

                conn.commit()
                return True

            finally:
                conn.close()

    # =========================================================================
    # EXPORT/IMPORT OPERATIONS
    # =========================================================================

    def export_board(
        self,
        board_id: int,
        include_archived: bool = False,
        include_deleted: bool = False
    ) -> dict[str, Any]:
        """
        Export a board with all its data as a JSON-serializable dictionary.

        Args:
            board_id: The board ID to export.
            include_archived: Include archived items.
            include_deleted: Include soft-deleted items.

        Returns:
            Complete board data including lists, cards, labels, checklists, comments.

        Raises:
            NotFoundError: If board not found.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Get the board
                board = self._get_board_by_id(conn, board_id, include_deleted=include_deleted)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003

                # Build export structure
                export_data = {
                    "format": "tldw_kanban_v1",
                    "exported_at": _utcnow_iso(),
                    "board": {
                        "uuid": board["uuid"],
                        "name": board["name"],
                        "description": board["description"],
                        "metadata": board["metadata"],
                        "archived": board["archived"],
                        "created_at": board["created_at"],
                    },
                    "labels": [],
                    "lists": [],
                }

                # Export labels
                labels = self.list_labels(board_id)
                for label in labels:
                    export_data["labels"].append({
                        "uuid": label["uuid"],
                        "name": label["name"],
                        "color": label["color"],
                        "created_at": label["created_at"],
                    })

                # Build label ID -> UUID mapping for cards
                label_id_to_uuid = {label["id"]: label["uuid"] for label in labels}

                # Export lists
                lists = self.list_lists(board_id, include_archived=include_archived, include_deleted=include_deleted)
                for lst in lists:
                    list_export = {
                        "uuid": lst["uuid"],
                        "client_id": lst["client_id"],
                        "name": lst["name"],
                        "position": lst["position"],
                        "archived": lst["archived"],
                        "created_at": lst["created_at"],
                        "cards": [],
                    }

                    # Export cards in this list
                    cards = self.list_cards(lst["id"], include_archived=include_archived, include_deleted=include_deleted)
                    for card in cards:
                        card_export = {
                            "uuid": card["uuid"],
                            "client_id": card["client_id"],
                            "title": card["title"],
                            "description": card["description"],
                            "position": card["position"],
                            "due_date": card["due_date"],
                            "due_complete": card["due_complete"],
                            "start_date": card["start_date"],
                            "priority": card["priority"],
                            "archived": card["archived"],
                            "metadata": card["metadata"],
                            "created_at": card["created_at"],
                            "label_uuids": [],
                            "checklists": [],
                            "comments": [],
                        }

                        # Get card labels
                        card_labels = self.get_card_labels(card["id"])
                        card_export["label_uuids"] = [
                            label_id_to_uuid.get(lbl["id"], lbl["uuid"])
                            for lbl in card_labels
                        ]

                        # Get checklists with items
                        checklists = self.list_checklists(card["id"])
                        for checklist in checklists:
                            checklist_export = {
                                "uuid": checklist["uuid"],
                                "name": checklist["name"],
                                "position": checklist["position"],
                                "created_at": checklist["created_at"],
                                "items": [],
                            }

                            # Get checklist items
                            items = self.list_checklist_items(checklist["id"])
                            for item in items:
                                checklist_export["items"].append({
                                    "uuid": item["uuid"],
                                    "name": item["name"],
                                    "position": item["position"],
                                    "checked": item["checked"],
                                    "checked_at": item["checked_at"],
                                    "created_at": item["created_at"],
                                })

                            card_export["checklists"].append(checklist_export)

                        # Get comments (exclude deleted by default)
                        comments, _ = self.list_comments(card["id"], include_deleted=include_deleted)
                        for comment in comments:
                            card_export["comments"].append({
                                "uuid": comment["uuid"],
                                "content": comment["content"],
                                "created_at": comment["created_at"],
                                "updated_at": comment["updated_at"],
                            })

                        list_export["cards"].append(card_export)

                    export_data["lists"].append(list_export)

                return export_data

            finally:
                conn.close()

    def import_board(
        self,
        data: dict[str, Any],
        board_name: str | None = None,
        board_client_id: str | None = None
    ) -> dict[str, Any]:
        """
        Import a board from exported JSON data.

        Supports both tldw_kanban_v1 format and Trello JSON export format.

        Args:
            data: The exported board data.
            board_name: Optional override for the board name.
            board_client_id: Optional client_id for the new board (auto-generated if not provided).

        Returns:
            The created board with summary of imported items.

        Raises:
            InputError: If the format is invalid or import fails.
        """
        # Detect format
        format_type = data.get("format")

        if format_type == "tldw_kanban_v1":
            result = self._import_tldw_format(data, board_name, board_client_id)
        elif "cards" in data and "lists" in data and "name" in data:
            # Trello format detection
            result = self._import_trello_format(data, board_name, board_client_id)
        else:
            raise InputError("Unrecognized import format. Must be tldw_kanban_v1 or Trello JSON format.")  # noqa: TRY003

        try:
            self.optimize_fts()
        except _KANBAN_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Kanban FTS optimize after import failed: {exc}")
        return result

    def _run_fts_maintenance(self, action: str) -> None:
        """Run FTS5 maintenance commands (optimize or rebuild)."""
        action_norm = str(action).strip().lower()
        if action_norm not in {"optimize", "rebuild"}:
            raise InputError("Unsupported FTS maintenance action. Use 'optimize' or 'rebuild'.")  # noqa: TRY003
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO kanban_cards_fts(kanban_cards_fts) VALUES (?)",
                    (action_norm,),
                )
                conn.commit()
            except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Kanban FTS {action_norm} failed: {e}")
                raise KanbanDBError(f"FTS {action_norm} failed: {e}") from e  # noqa: TRY003
            finally:
                conn.close()

    def optimize_fts(self) -> None:
        """Optimize the FTS index after large batch inserts."""
        self._run_fts_maintenance("optimize")

    def rebuild_fts(self) -> None:
        """Rebuild the FTS index (expensive)."""
        self._run_fts_maintenance("rebuild")

    def _import_tldw_format(
        self,
        data: dict[str, Any],
        board_name: str | None = None,
        board_client_id: str | None = None
    ) -> dict[str, Any]:
        """Import from tldw_kanban_v1 format."""
        board_data = data.get("board", {})
        if not board_data:
            raise InputError("Invalid tldw_kanban_v1 format: missing 'board' data")  # noqa: TRY003

        # Create the board
        new_board = self.create_board(
            name=board_name or board_data.get("name", "Imported Board"),
            client_id=board_client_id or _generate_uuid(),
            description=board_data.get("description"),
            metadata=board_data.get("metadata")
        )
        board_id = new_board["id"]

        # Track imported counts
        import_stats = {
            "board_id": board_id,
            "lists_imported": 0,
            "cards_imported": 0,
            "labels_imported": 0,
            "checklists_imported": 0,
            "checklist_items_imported": 0,
            "comments_imported": 0,
        }

        # Import labels and track UUID mapping
        label_uuid_to_id: dict[str, int] = {}
        for label_data in data.get("labels", []):
            try:
                label = self.create_label(
                    board_id=board_id,
                    name=label_data.get("name", "Unnamed Label"),
                    color=label_data.get("color", "gray")
                )
                label_uuid_to_id[label_data.get("uuid", "")] = label["id"]
                import_stats["labels_imported"] += 1
            except (InputError, KanbanDBError) as e:
                logger.warning(f"Failed to import label: {e}")

        # Import lists
        for list_data in data.get("lists", []):
            try:
                new_list = self.create_list(
                    board_id=board_id,
                    name=list_data.get("name", "Unnamed List"),
                    client_id=list_data.get("client_id") or _generate_uuid(),
                    position=list_data.get("position")
                )
                import_stats["lists_imported"] += 1

                # Import cards in this list
                for card_data in list_data.get("cards", []):
                    try:
                        new_card = self.create_card(
                            list_id=new_list["id"],
                            title=card_data.get("title", "Unnamed Card"),
                            client_id=card_data.get("client_id") or _generate_uuid(),
                            description=card_data.get("description"),
                            position=card_data.get("position"),
                            due_date=card_data.get("due_date"),
                            start_date=card_data.get("start_date"),
                            priority=card_data.get("priority"),
                            metadata=card_data.get("metadata")
                        )
                        import_stats["cards_imported"] += 1

                        # Assign labels
                        for label_uuid in card_data.get("label_uuids", []):
                            if label_uuid in label_uuid_to_id:
                                with contextlib.suppress(_KANBAN_NONCRITICAL_EXCEPTIONS):
                                    self.assign_label_to_card(new_card["id"], label_uuid_to_id[label_uuid])

                        # Import checklists
                        for checklist_data in card_data.get("checklists", []):
                            try:
                                new_checklist = self.create_checklist(
                                    card_id=new_card["id"],
                                    name=checklist_data.get("name", "Checklist"),
                                    position=checklist_data.get("position")
                                )
                                import_stats["checklists_imported"] += 1

                                # Import checklist items
                                for item_data in checklist_data.get("items", []):
                                    try:
                                        self.create_checklist_item(
                                            checklist_id=new_checklist["id"],
                                            name=item_data.get("name", "Item"),
                                            position=item_data.get("position"),
                                            checked=item_data.get("checked", False)
                                        )
                                        import_stats["checklist_items_imported"] += 1
                                    except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                                        logger.warning(f"Failed to import checklist item: {e}")
                            except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                                logger.warning(f"Failed to import checklist: {e}")

                        # Import comments
                        for comment_data in card_data.get("comments", []):
                            try:
                                self.create_comment(
                                    card_id=new_card["id"],
                                    content=comment_data.get("content", "")
                                )
                                import_stats["comments_imported"] += 1
                            except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                                logger.warning(f"Failed to import comment: {e}")

                    except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"Failed to import card: {e}")

            except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to import list: {e}")

        # Log activity
        self.log_activity(
            board_id=board_id,
            action_type="board_imported",
            entity_type="board",
            entity_id=board_id,
            details=import_stats
        )

        return {
            "board": self.get_board(board_id),
            "import_stats": import_stats
        }

    def _import_trello_format(
        self,
        data: dict[str, Any],
        board_name: str | None = None,
        board_client_id: str | None = None
    ) -> dict[str, Any]:
        """Import from Trello JSON export format."""
        # Create the board
        new_board = self.create_board(
            name=board_name or data.get("name", "Imported Trello Board"),
            client_id=board_client_id or _generate_uuid(),
            description=data.get("desc")
        )
        board_id = new_board["id"]

        # Track imported counts
        import_stats = {
            "board_id": board_id,
            "lists_imported": 0,
            "cards_imported": 0,
            "labels_imported": 0,
            "checklists_imported": 0,
            "checklist_items_imported": 0,
            "comments_imported": 0,
        }

        # Map Trello colors to our colors
        color_map = {
            "green": "green",
            "yellow": "yellow",
            "orange": "orange",
            "red": "red",
            "purple": "purple",
            "blue": "blue",
            "sky": "blue",
            "lime": "green",
            "pink": "pink",
            "black": "gray",
            "": "gray",
            None: "gray",
        }

        # Import labels (Trello stores these at board level)
        label_id_map: dict[str, int] = {}  # Trello ID -> our ID
        for label_data in data.get("labels", []):
            trello_color = label_data.get("color", "")
            our_color = color_map.get(trello_color, "gray")
            label_name = label_data.get("name") or our_color.title()
            try:
                label = self.create_label(
                    board_id=board_id,
                    name=label_name,
                    color=our_color
                )
                label_id_map[label_data.get("id", "")] = label["id"]
                import_stats["labels_imported"] += 1
            except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to import Trello label: {e}")

        # Build list ID map (Trello ID -> our ID)
        list_id_map: dict[str, int] = {}
        trello_lists = data.get("lists", [])

        # Sort lists by position (Trello uses 'pos' field)
        trello_lists.sort(key=lambda x: x.get("pos", 0))

        for position, list_data in enumerate(trello_lists):
            if list_data.get("closed", False):
                continue  # Skip closed/archived lists

            try:
                new_list = self.create_list(
                    board_id=board_id,
                    name=list_data.get("name", "List"),
                    client_id=list_data.get("id") or _generate_uuid(),
                    position=position
                )
                list_id_map[list_data.get("id", "")] = new_list["id"]
                import_stats["lists_imported"] += 1
            except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to import Trello list: {e}")

        # Build checklist map (Trello ID -> checklist data)
        checklist_map: dict[str, dict] = {}
        for checklist in data.get("checklists", []):
            checklist_map[checklist.get("id", "")] = checklist

        # Import cards
        trello_cards = data.get("cards", [])
        # Sort cards by position within their list
        trello_cards.sort(key=lambda x: (x.get("idList", ""), x.get("pos", 0)))

        card_positions: dict[str, int] = {}  # Track position per list

        for card_data in trello_cards:
            if card_data.get("closed", False):
                continue  # Skip closed/archived cards

            trello_list_id = card_data.get("idList", "")
            if trello_list_id not in list_id_map:
                continue  # Card's list wasn't imported

            our_list_id = list_id_map[trello_list_id]

            # Track card position within list
            if trello_list_id not in card_positions:
                card_positions[trello_list_id] = 0
            position = card_positions[trello_list_id]
            card_positions[trello_list_id] += 1

            try:
                new_card = self.create_card(
                    list_id=our_list_id,
                    title=card_data.get("name", "Card"),
                    client_id=card_data.get("id") or _generate_uuid(),
                    description=card_data.get("desc"),
                    position=position,
                    due_date=card_data.get("due"),
                )
                import_stats["cards_imported"] += 1

                # Assign labels
                for label_id in card_data.get("idLabels", []):
                    if label_id in label_id_map:
                        with contextlib.suppress(_KANBAN_NONCRITICAL_EXCEPTIONS):
                            self.assign_label_to_card(new_card["id"], label_id_map[label_id])

                # Import checklists
                for checklist_id in card_data.get("idChecklists", []):
                    if checklist_id in checklist_map:
                        checklist_data = checklist_map[checklist_id]
                        try:
                            new_checklist = self.create_checklist(
                                card_id=new_card["id"],
                                name=checklist_data.get("name", "Checklist")
                            )
                            import_stats["checklists_imported"] += 1

                            # Import checklist items
                            check_items = checklist_data.get("checkItems", [])
                            check_items.sort(key=lambda x: x.get("pos", 0))

                            for item_pos, item_data in enumerate(check_items):
                                try:
                                    self.create_checklist_item(
                                        checklist_id=new_checklist["id"],
                                        name=item_data.get("name", "Item"),
                                        position=item_pos,
                                        checked=item_data.get("state") == "complete"
                                    )
                                    import_stats["checklist_items_imported"] += 1
                                except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                                    logger.warning(f"Failed to import Trello checklist item: {e}")
                        except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                            logger.warning(f"Failed to import Trello checklist: {e}")

            except _KANBAN_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to import Trello card: {e}")

        # Log activity
        self.log_activity(
            board_id=board_id,
            action_type="board_imported_trello",
            entity_type="board",
            entity_id=board_id,
            details=import_stats
        )

        return {
            "board": self.get_board(board_id),
            "import_stats": import_stats
        }

    # =========================================================================
    # Phase 3: Bulk Operations
    # =========================================================================

    def bulk_move_cards(
        self,
        card_ids: list[int],
        target_list_id: int,
        start_position: int | None = None
    ) -> dict[str, Any]:
        """
        Move multiple cards to a target list.

        Args:
            card_ids: List of card IDs to move.
            target_list_id: The destination list ID.
            start_position: Optional starting position (cards placed sequentially from here).

        Returns:
            Dict with success status, moved count, and updated cards.
        """
        if not card_ids:
            return {"success": True, "moved_count": 0, "cards": []}

        moved_cards: list[dict[str, Any]] = []
        with self._lock:
            conn = self._connect()
            try:
                # Verify target list exists
                target_list = self._get_list_by_id(conn, target_list_id)
                if not target_list:
                    raise NotFoundError("Target list not found", entity="list", entity_id=target_list_id)  # noqa: TRY003

                board_id = target_list["board_id"]

                # Get all cards and verify they exist and belong to the same board
                placeholders = ",".join("?" * len(card_ids))
                cur = conn.execute(
                    """
                    SELECT id, board_id, list_id, title FROM kanban_cards
                    WHERE id IN ({placeholders}) AND deleted = 0
                    """.format_map(locals()),  # nosec B608
                    card_ids
                )
                cards = cur.fetchall()

                if len(cards) != len(card_ids):
                    found_ids = {c["id"] for c in cards}
                    missing = set(card_ids) - found_ids
                    raise NotFoundError(f"Cards not found: {missing}", entity="card")  # noqa: TRY003

                # Verify all cards are from the same board as target list
                for card in cards:
                    if card["board_id"] != board_id:
                        raise InputError(f"Card {card['id']} is not in the same board as the target list")  # noqa: TRY003

                # Determine starting position
                if start_position is None:
                    cur = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM kanban_cards WHERE list_id = ? AND deleted = 0",
                        (target_list_id,)
                    )
                    start_position = cur.fetchone()["next_pos"]

                # Move cards
                now = _utcnow_iso()
                moved_cards = []
                for i, card_id in enumerate(card_ids):
                    old_list_id = next(c["list_id"] for c in cards if c["id"] == card_id)
                    new_position = start_position + i

                    conn.execute(
                        """
                        UPDATE kanban_cards
                        SET list_id = ?, position = ?, updated_at = ?, version = version + 1
                        WHERE id = ?
                        """,
                        (target_list_id, new_position, now, card_id)
                    )

                    # Log activity for each card
                    card_title = next(c["title"] for c in cards if c["id"] == card_id)
                    self._log_activity_internal(
                        conn, board_id, "card_moved", "card", entity_id=card_id,
                        list_id=target_list_id, card_id=card_id,
                        details={"from_list_id": old_list_id, "to_list_id": target_list_id, "title": card_title}
                    )

                conn.commit()

                # Fetch updated cards
                for card_id in card_ids:
                    moved_cards.append(self._get_card_by_id(conn, card_id))

            finally:
                conn.close()

        self._sync_vector_index_for_card_ids(card_ids)
        return {
            "success": True,
            "moved_count": len(moved_cards),
            "cards": moved_cards,
        }

    def bulk_archive_cards(self, card_ids: list[int], archive: bool = True) -> dict[str, Any]:
        """
        Archive or unarchive multiple cards.

        Args:
            card_ids: List of card IDs.
            archive: True to archive, False to unarchive.

        Returns:
            Dict with success status and archived/unarchived count.
        """
        if not card_ids:
            return {"success": True, "archived_count" if archive else "unarchived_count": 0}

        with self._lock:
            conn = self._connect()
            try:
                # Get all cards and verify they exist
                placeholders = ",".join("?" * len(card_ids))
                cur = conn.execute(
                    """
                    SELECT id, board_id, list_id, title FROM kanban_cards
                    WHERE id IN ({placeholders}) AND deleted = 0
                    """.format_map(locals()),  # nosec B608
                    card_ids
                )
                cards = cur.fetchall()

                if len(cards) != len(card_ids):
                    found_ids = {c["id"] for c in cards}
                    missing = set(card_ids) - found_ids
                    raise NotFoundError(f"Cards not found: {missing}", entity="card")  # noqa: TRY003

                # Verify all cards belong to the same user (same board check)
                board_ids = {c["board_id"] for c in cards}
                for bid in board_ids:
                    board = self._get_board_by_id(conn, bid)
                    if not board or board["user_id"] != self.user_id:
                        raise NotFoundError("Board not found or access denied", entity="board")  # noqa: TRY003

                # Archive/unarchive cards
                now = _utcnow_iso()
                action_type = "card_archived" if archive else "card_unarchived"

                for card in cards:
                    conn.execute(
                        """
                        UPDATE kanban_cards
                        SET archived = ?, archived_at = ?, updated_at = ?, version = version + 1
                        WHERE id = ?
                        """,
                        (1 if archive else 0, now if archive else None, now, card["id"])
                    )

                    self._log_activity_internal(
                        conn, card["board_id"], action_type, "card", entity_id=card["id"],
                        list_id=card["list_id"], card_id=card["id"],
                        details={"title": card["title"]}
                    )

                conn.commit()

                result_key = "archived_count" if archive else "unarchived_count"
                result = {"success": True, result_key: len(cards)}

            finally:
                conn.close()

        self._sync_vector_index_for_card_ids(card_ids)
        return result

    def bulk_delete_cards(self, card_ids: list[int], hard_delete: bool = False) -> dict[str, Any]:
        """
        Soft or hard delete multiple cards.

        Args:
            card_ids: List of card IDs.
            hard_delete: If True, permanently delete. If False, soft delete.

        Returns:
            Dict with success status and deleted count.
        """
        if not card_ids:
            return {"success": True, "deleted_count": 0}

        with self._lock:
            conn = self._connect()
            try:
                # Get all cards and verify they exist
                placeholders = ",".join("?" * len(card_ids))
                deleted_scope_clause = "" if hard_delete else "AND deleted = 0"
                cur = conn.execute(
                    """
                    SELECT id, board_id, list_id, title FROM kanban_cards
                    WHERE id IN ({placeholders}) {deleted_scope_clause}
                    """.format_map(locals()),  # nosec B608
                    card_ids
                )
                cards = cur.fetchall()

                if len(cards) != len(card_ids):
                    found_ids = {c["id"] for c in cards}
                    missing = set(card_ids) - found_ids
                    raise NotFoundError(f"Cards not found: {missing}", entity="card")  # noqa: TRY003

                # Verify all cards belong to the same user
                board_ids = {c["board_id"] for c in cards}
                for bid in board_ids:
                    board = self._get_board_by_id(conn, bid)
                    if not board or board["user_id"] != self.user_id:
                        raise NotFoundError("Board not found or access denied", entity="board")  # noqa: TRY003

                now = _utcnow_iso()

                if hard_delete:
                    # Permanently delete
                    conn.execute(
                        f"DELETE FROM kanban_cards WHERE id IN ({placeholders})",  # nosec B608
                        card_ids
                    )
                else:
                    # Soft delete
                    for card in cards:
                        conn.execute(
                            """
                            UPDATE kanban_cards
                            SET deleted = 1, deleted_at = ?, updated_at = ?, version = version + 1
                            WHERE id = ?
                            """,
                            (now, now, card["id"])
                        )

                        self._log_activity_internal(
                            conn, card["board_id"], "card_deleted", "card", entity_id=card["id"],
                            list_id=card["list_id"], card_id=card["id"],
                            details={"title": card["title"]}
                        )

                conn.commit()

                result = {"success": True, "deleted_count": len(cards)}

            finally:
                conn.close()

        self._sync_vector_index_for_card_ids(card_ids)
        return result

    def bulk_label_cards(
        self,
        card_ids: list[int],
        add_label_ids: list[int] | None = None,
        remove_label_ids: list[int] | None = None
    ) -> dict[str, Any]:
        """
        Add and/or remove labels from multiple cards.

        Args:
            card_ids: List of card IDs.
            add_label_ids: Label IDs to add to all cards.
            remove_label_ids: Label IDs to remove from all cards.

        Returns:
            Dict with success status and updated count.
        """
        if not card_ids:
            return {"success": True, "updated_count": 0}

        add_label_ids = add_label_ids or []
        remove_label_ids = remove_label_ids or []

        if not add_label_ids and not remove_label_ids:
            return {"success": True, "updated_count": 0}

        result: dict[str, Any] = {}
        with self._lock:
            conn = self._connect()
            try:
                # Get all cards and verify they exist
                placeholders = ",".join("?" * len(card_ids))
                cur = conn.execute(
                    """
                    SELECT id, board_id, title FROM kanban_cards
                    WHERE id IN ({placeholders}) AND deleted = 0
                    """.format_map(locals()),  # nosec B608
                    card_ids
                )
                cards = cur.fetchall()

                if len(cards) != len(card_ids):
                    found_ids = {c["id"] for c in cards}
                    missing = set(card_ids) - found_ids
                    raise NotFoundError(f"Cards not found: {missing}", entity="card")  # noqa: TRY003

                # Get unique board IDs
                board_ids = {c["board_id"] for c in cards}

                # Verify labels belong to the same boards as the cards
                all_label_ids = set(add_label_ids) | set(remove_label_ids)
                if all_label_ids:
                    label_placeholders = ",".join("?" * len(all_label_ids))
                    cur = conn.execute(
                        f"SELECT id, board_id, name FROM kanban_labels WHERE id IN ({label_placeholders})",  # nosec B608
                        list(all_label_ids)
                    )
                    labels = {row["id"]: row for row in cur.fetchall()}

                    for label_id in all_label_ids:
                        if label_id not in labels:
                            raise NotFoundError(f"Label {label_id} not found", entity="label", entity_id=label_id)  # noqa: TRY003
                        if labels[label_id]["board_id"] not in board_ids:
                            raise InputError(f"Label {label_id} does not belong to the same board as the cards")  # noqa: TRY003

                now = _utcnow_iso()
                updated_count = 0

                for card in cards:
                    card_updated = False

                    # Remove labels
                    if remove_label_ids:
                        remove_placeholders = ",".join("?" * len(remove_label_ids))
                        cur = conn.execute(
                            f"DELETE FROM kanban_card_labels WHERE card_id = ? AND label_id IN ({remove_placeholders})",  # nosec B608
                            [card["id"]] + remove_label_ids
                        )
                        if cur.rowcount > 0:
                            card_updated = True
                            for label_id in remove_label_ids:
                                if label_id in labels:
                                    self._log_activity_internal(
                                        conn, card["board_id"], "label_removed", "card",
                                        entity_id=card["id"], card_id=card["id"],
                                        details={"label_name": labels[label_id]["name"], "card_title": card["title"]}
                                    )

                    # Add labels
                    if add_label_ids:
                        for label_id in add_label_ids:
                            try:
                                conn.execute(
                                    "INSERT INTO kanban_card_labels (card_id, label_id, created_at) VALUES (?, ?, ?)",
                                    (card["id"], label_id, now)
                                )
                                card_updated = True
                                self._log_activity_internal(
                                    conn, card["board_id"], "label_assigned", "card",
                                    entity_id=card["id"], card_id=card["id"],
                                    details={"label_name": labels[label_id]["name"], "card_title": card["title"]}
                                )
                            except sqlite3.IntegrityError:
                                # Label already assigned, skip
                                pass

                    if card_updated:
                        # Update card's updated_at
                        conn.execute(
                            "UPDATE kanban_cards SET updated_at = ?, version = version + 1 WHERE id = ?",
                            (now, card["id"])
                        )
                        updated_count += 1

                conn.commit()

                result = {"success": True, "updated_count": updated_count}

            finally:
                conn.close()

        if result.get("updated_count"):
            self._sync_vector_index_for_card_ids(card_ids)
        return result

    # =========================================================================
    # Phase 3: Card Filtering
    # =========================================================================

    def get_board_cards_filtered(
        self,
        board_id: int,
        label_ids: list[int] | None = None,
        priority: str | None = None,
        due_before: str | None = None,
        due_after: str | None = None,
        overdue: bool | None = None,
        has_due_date: bool | None = None,
        has_checklist: bool | None = None,
        is_complete: bool | None = None,
        include_archived: bool = False,
        include_deleted: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Get filtered cards for a board.

        Args:
            board_id: The board ID.
            label_ids: Filter by label IDs (cards must have ALL specified labels).
            priority: Filter by priority (low, medium, high, urgent).
            due_before: Filter by due date before this timestamp.
            due_after: Filter by due date after this timestamp.
            overdue: If True, only cards with due_date < now AND due_complete = 0.
            has_due_date: If True, only cards with due_date set. If False, only cards without.
            has_checklist: If True, only cards with at least one checklist.
            is_complete: If True, only cards where all checklist items are checked.
            include_archived: Include archived cards.
            include_deleted: Include soft-deleted cards.
            limit: Maximum cards to return.
            offset: Number of cards to skip.

        Returns:
            Tuple of (cards list, total count).
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify board exists and belongs to user
                board = self._get_board_by_id(conn, board_id)
                if not board:
                    raise NotFoundError("Board not found", entity="board", entity_id=board_id)  # noqa: TRY003
                if board["archived"] and not include_archived:
                    return [], 0

                # Build query
                conditions = ["c.board_id = ?"]
                params: list[Any] = [board_id]

                if not include_deleted:
                    conditions.append("c.deleted = 0")
                if not include_archived:
                    conditions.append("c.archived = 0")
                    conditions.append("l.archived = 0")

                if priority:
                    conditions.append("c.priority = ?")
                    params.append(priority)

                if due_before:
                    conditions.append("c.due_date < ?")
                    params.append(due_before)

                if due_after:
                    conditions.append("c.due_date > ?")
                    params.append(due_after)

                if overdue is True:
                    now = _utcnow_iso()
                    conditions.append("c.due_date IS NOT NULL AND c.due_date < ? AND c.due_complete = 0")
                    params.append(now)

                if has_due_date is True:
                    conditions.append("c.due_date IS NOT NULL")
                elif has_due_date is False:
                    conditions.append("c.due_date IS NULL")

                # Label filtering - cards must have ALL specified labels
                if label_ids:
                    for label_id in label_ids:
                        conditions.append(
                            "EXISTS (SELECT 1 FROM kanban_card_labels cl WHERE cl.card_id = c.id AND cl.label_id = ?)"
                        )
                        params.append(label_id)

                # Checklist filtering
                if has_checklist is True:
                    conditions.append(
                        "EXISTS (SELECT 1 FROM kanban_checklists ch WHERE ch.card_id = c.id)"
                    )
                elif has_checklist is False:
                    conditions.append(
                        "NOT EXISTS (SELECT 1 FROM kanban_checklists ch WHERE ch.card_id = c.id)"
                    )

                # Complete filtering (all checklist items checked)
                if is_complete is True:
                    conditions.append("""
                        EXISTS (SELECT 1 FROM kanban_checklists ch WHERE ch.card_id = c.id)
                        AND NOT EXISTS (
                            SELECT 1 FROM kanban_checklists ch
                            JOIN kanban_checklist_items ci ON ci.checklist_id = ch.id
                            WHERE ch.card_id = c.id AND ci.checked = 0
                        )
                    """)
                elif is_complete is False:
                    conditions.append("""
                        EXISTS (
                            SELECT 1 FROM kanban_checklists ch
                            JOIN kanban_checklist_items ci ON ci.checklist_id = ch.id
                            WHERE ch.card_id = c.id AND ci.checked = 0
                        )
                    """)

                where_clause = " AND ".join(conditions)

                # Count total
                count_sql = """
                    SELECT COUNT(*) as cnt
                    FROM kanban_cards c
                    JOIN kanban_lists l ON c.list_id = l.id
                    WHERE {where_clause}
                """.format_map(locals())  # nosec B608
                cur = conn.execute(count_sql, params)
                total = cur.fetchone()["cnt"]

                # Get paginated results
                query_sql = """
                    SELECT c.* FROM kanban_cards c
                    JOIN kanban_lists l ON c.list_id = l.id
                    WHERE {where_clause}
                    ORDER BY c.list_id, c.position
                    LIMIT ? OFFSET ?
                """.format_map(locals())  # nosec B608
                cur = conn.execute(query_sql, params + [limit, offset])
                rows = cur.fetchall()

                cards = [self._row_to_card_dict(row) for row in rows]

                return cards, total

            finally:
                conn.close()

    # =========================================================================
    # Phase 3: Toggle All Checklist Items
    # =========================================================================

    def toggle_all_checklist_items(self, checklist_id: int, checked: bool) -> dict[str, Any]:
        """
        Check or uncheck all items in a checklist.

        Args:
            checklist_id: The checklist ID.
            checked: True to check all, False to uncheck all.

        Returns:
            Updated checklist with items.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Get the checklist
                checklist = self.get_checklist(checklist_id)
                if not checklist:
                    raise NotFoundError("Checklist not found", entity="checklist", entity_id=checklist_id)  # noqa: TRY003

                # Get the card to verify ownership and get board_id
                card = self._get_card_by_id(conn, checklist["card_id"])
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=checklist["card_id"])  # noqa: TRY003

                now = _utcnow_iso()

                # Update all items
                if checked:
                    conn.execute(
                        """
                        UPDATE kanban_checklist_items
                        SET checked = 1, checked_at = ?, updated_at = ?
                        WHERE checklist_id = ? AND checked = 0
                        """,
                        (now, now, checklist_id)
                    )
                else:
                    conn.execute(
                        """
                        UPDATE kanban_checklist_items
                        SET checked = 0, checked_at = NULL, updated_at = ?
                        WHERE checklist_id = ? AND checked = 1
                        """,
                        (now, checklist_id)
                    )

                # Log activity
                action_type = "checklist_items_all_checked" if checked else "checklist_items_all_unchecked"
                self._log_activity_internal(
                    conn, card["board_id"], action_type, "checklist",
                    entity_id=checklist_id, card_id=card["id"],
                    details={"checklist_name": checklist["name"], "checked": checked}
                )

                conn.commit()

                # Return updated checklist with items
                return self.get_checklist_with_items(checklist_id)

            finally:
                conn.close()

    # =========================================================================
    # Phase 3: Enhanced Card Copy (with checklists)
    # =========================================================================

    def copy_card_with_checklists(
        self,
        card_id: int,
        target_list_id: int,
        new_client_id: str,
        position: int | None = None,
        new_title: str | None = None,
        copy_checklists: bool = True,
        copy_labels: bool = True
    ) -> dict[str, Any]:
        """
        Copy a card to a list, optionally including checklists.

        Args:
            card_id: The source card ID.
            target_list_id: The destination list ID.
            new_client_id: Client-generated unique ID for the copy.
            position: Optional position in the target list.
            new_title: Optional new title (defaults to "Copy of {original}").
            copy_checklists: Whether to copy checklists (default True).
            copy_labels: Whether to copy labels (default True).

        Returns:
            The copied card with checklists if copied.
        """
        copied_card: dict[str, Any] | None = None
        with self._lock:
            conn = self._connect()
            try:
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError("Card not found", entity="card", entity_id=card_id)  # noqa: TRY003

                target_list = self._get_list_by_id(conn, target_list_id)
                if not target_list:
                    raise NotFoundError("Target list not found", entity="list", entity_id=target_list_id)  # noqa: TRY003

                # Verify target list is in the same board
                if target_list["board_id"] != card["board_id"]:
                    raise InputError("Cannot copy card to a list in a different board")  # noqa: TRY003

                # Enforce board and list limits
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_cards WHERE board_id = ? AND deleted = 0",
                    (card["board_id"],)
                )
                board_count = cur.fetchone()["cnt"]
                if board_count >= self.MAX_CARDS_PER_BOARD:
                    raise InputError(f"Maximum cards ({self.MAX_CARDS_PER_BOARD}) per board reached")  # noqa: TRY003

                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM kanban_cards WHERE list_id = ? AND deleted = 0",
                    (target_list_id,)
                )
                list_count = cur.fetchone()["cnt"]
                if list_count >= self.MAX_CARDS_PER_LIST:
                    raise InputError(f"Maximum cards ({self.MAX_CARDS_PER_LIST}) per list reached")  # noqa: TRY003

                # Generate title if not provided
                if new_title is None:
                    new_title = f"Copy of {card['title']}"

                # Get target position
                if position is None:
                    cur = conn.execute(
                        "SELECT COALESCE(MAX(position), -1) + 1 as next_pos FROM kanban_cards WHERE list_id = ? AND deleted = 0",
                        (target_list_id,)
                    )
                    position = cur.fetchone()["next_pos"]

                card_uuid = _generate_uuid()
                now = _utcnow_iso()

                # Insert the copy
                cur = conn.execute(
                    """
                    INSERT INTO kanban_cards
                    (uuid, board_id, list_id, client_id, title, description, position,
                     due_date, start_date, priority, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (card_uuid, card["board_id"], target_list_id, new_client_id,
                     new_title, card["description"], position,
                     card["due_date"], card["start_date"], card["priority"],
                     json.dumps(card["metadata"]) if card["metadata"] else None,
                     now, now)
                )
                new_card_id = cur.lastrowid

                # Copy labels
                if copy_labels:
                    conn.execute(
                        """
                        INSERT INTO kanban_card_labels (card_id, label_id, created_at)
                        SELECT ?, label_id, ? FROM kanban_card_labels WHERE card_id = ?
                        """,
                        (new_card_id, now, card_id)
                    )

                # Copy checklists
                checklists_copied = 0
                items_copied = 0
                if copy_checklists:
                    # Get source checklists
                    cur = conn.execute(
                        "SELECT * FROM kanban_checklists WHERE card_id = ? ORDER BY position",
                        (card_id,)
                    )
                    source_checklists = cur.fetchall()

                    for checklist in source_checklists:
                        checklist_uuid = _generate_uuid()
                        cur = conn.execute(
                            """
                            INSERT INTO kanban_checklists (uuid, card_id, name, position, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (checklist_uuid, new_card_id, checklist["name"], checklist["position"], now, now)
                        )
                        new_checklist_id = cur.lastrowid
                        checklists_copied += 1

                        # Copy checklist items
                        cur = conn.execute(
                            "SELECT * FROM kanban_checklist_items WHERE checklist_id = ? ORDER BY position",
                            (checklist["id"],)
                        )
                        source_items = cur.fetchall()

                        for item in source_items:
                            item_uuid = _generate_uuid()
                            conn.execute(
                                """
                                INSERT INTO kanban_checklist_items
                                (uuid, checklist_id, name, position, checked, checked_at, created_at, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (item_uuid, new_checklist_id, item["name"], item["position"],
                                 item["checked"], item["checked_at"], now, now)
                            )
                            items_copied += 1

                # Log activity
                self._log_activity_internal(
                    conn, card["board_id"], "card_copied", "card", entity_id=new_card_id,
                    list_id=target_list_id, card_id=new_card_id,
                    details={
                        "title": new_title,
                        "source_card_id": card_id,
                        "target_list_id": target_list_id,
                        "checklists_copied": checklists_copied,
                        "checklist_items_copied": items_copied
                    }
                )

                conn.commit()

                copied_card = self._get_card_by_id(conn, new_card_id)

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint" in str(e) and "client_id" in str(e):
                    raise ConflictError(  # noqa: B904, TRY003
                        f"Card with client_id '{new_client_id}' already exists",
                        entity="card",
                        entity_id=new_client_id
                    )
                raise KanbanDBError(f"Database error: {e}") from e  # noqa: TRY003
            finally:
                conn.close()

        if copied_card:
            self._sync_vector_index_for_card_id(copied_card["id"])
        return copied_card

    # ==========================================================================
    # Card Links Methods (Phase 5: Content Integration)
    # ==========================================================================

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        """Return True when table_name exists in the target SQLite DB."""
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _check_link_target_exists(self, linked_type: str, linked_id: str) -> bool | None:
        """
        Check whether a linked target exists.

        Returns:
            True: target exists
            False: target definitively does not exist
            None: target could not be validated (DB/table unavailable, fallback mode)
        """
        linked_id = str(linked_id).strip()
        if not linked_id:
            return False

        if linked_type == "media":
            db_path = DatabasePaths.get_media_db_path(self.user_id)
            table_name = "Media"
            sql = (
                "SELECT 1 FROM Media "
                "WHERE COALESCE(deleted, 0) = 0 "
                "AND (CAST(id AS TEXT) = ? OR uuid = ? OR client_id = ?) "
                "LIMIT 1"
            )
            params = (linked_id, linked_id, linked_id)
        elif linked_type == "note":
            db_path = DatabasePaths.get_chacha_db_path(self.user_id)
            table_name = "notes"
            sql = (
                "SELECT 1 FROM notes "
                "WHERE COALESCE(deleted, 0) = 0 "
                "AND (id = ? OR client_id = ?) "
                "LIMIT 1"
            )
            params = (linked_id, linked_id)
        else:
            return False

        if not db_path.exists():
            logger.debug(
                f"Skipping link target validation for {linked_type}:{linked_id} - "
                f"database file not found at {db_path}"
            )
            return None

        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            try:
                if not self._table_exists(conn, table_name):
                    logger.debug(
                        f"Skipping link target validation for {linked_type}:{linked_id} - "
                        f"table {table_name} not found in {db_path}"
                    )
                    return None
                row = conn.execute(sql, params).fetchone()
                return row is not None
            finally:
                conn.close()
        except sqlite3.Error as e:
            logger.warning(
                f"Link target validation unavailable for {linked_type}:{linked_id}: {e}"
            )
            return None

    def _validate_link_target_or_raise(self, linked_type: str, linked_id: str) -> None:
        """
        Validate linked media/note target when possible.

        Validation is strict only when the target content DB/table is available.
        If unavailable, validation gracefully falls back to permissive mode.
        """
        exists = self._check_link_target_exists(linked_type=linked_type, linked_id=linked_id)
        if exists is False:
            raise NotFoundError(
                f"Linked {linked_type} target not found: {linked_id}",
                entity=linked_type,
                entity_id=linked_id,
            )

    def add_card_link(
        self,
        card_id: int,
        linked_type: str,
        linked_id: str
    ) -> dict[str, Any]:
        """
        Add a link from a card to a media item or note.

        Args:
            card_id: The card to add the link to.
            linked_type: Type of linked content ('media' or 'note').
            linked_id: ID of the linked content.

        Returns:
            The created link.

        Raises:
            NotFoundError: If the card doesn't exist.
            InputError: If linked_type is invalid.
            ConflictError: If the link already exists.
        """
        if linked_type not in ("media", "note"):
            raise InputError(f"Invalid linked_type: {linked_type}. Must be 'media' or 'note'.")  # noqa: TRY003

        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError(f"Card {card_id} not found", entity="card", entity_id=card_id)  # noqa: TRY003

                self._validate_link_target_or_raise(linked_type=linked_type, linked_id=linked_id)

                link_uuid = _generate_uuid()
                now = _utcnow_iso()

                cur = conn.execute(
                    """
                    INSERT INTO kanban_card_links (uuid, card_id, linked_type, linked_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (link_uuid, card_id, linked_type, linked_id, now)
                )
                link_id = cur.lastrowid

                # Log activity
                self._log_activity_internal(
                    conn, card["board_id"], "link_added", "card_link", entity_id=link_id,
                    card_id=card_id, list_id=card["list_id"],
                    details={"linked_type": linked_type, "linked_id": linked_id}
                )

                conn.commit()

                return {  # noqa: TRY300
                    "id": link_id,
                    "card_id": card_id,
                    "linked_type": linked_type,
                    "linked_id": linked_id,
                    "created_at": now
                }

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint" in str(e):
                    raise ConflictError(  # noqa: B904, TRY003
                        f"Link already exists from card {card_id} to {linked_type}:{linked_id}",
                        entity="card_link"
                    )
                raise KanbanDBError(f"Database error: {e}") from e  # noqa: TRY003
            finally:
                conn.close()

    def get_card_links(
        self,
        card_id: int,
        linked_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Get all links for a card.

        Args:
            card_id: The card to get links for.
            linked_type: Optional filter by type ('media' or 'note').

        Returns:
            List of card links.

        Raises:
            NotFoundError: If the card doesn't exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError(f"Card {card_id} not found", entity="card", entity_id=card_id)  # noqa: TRY003

                if linked_type:
                    cur = conn.execute(
                        """
                        SELECT id, card_id, linked_type, linked_id, created_at
                        FROM kanban_card_links
                        WHERE card_id = ? AND linked_type = ?
                        ORDER BY created_at DESC
                        """,
                        (card_id, linked_type)
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT id, card_id, linked_type, linked_id, created_at
                        FROM kanban_card_links
                        WHERE card_id = ?
                        ORDER BY created_at DESC
                        """,
                        (card_id,)
                    )

                return [dict(row) for row in cur.fetchall()]

            finally:
                conn.close()

    def get_linked_content_counts(self, card_id: int) -> dict[str, int]:
        """
        Get counts of linked content by type for a card.

        Args:
            card_id: The card to get link counts for.

        Returns:
            Dict with counts by type: {"media": N, "note": M}

        Raises:
            NotFoundError: If the card doesn't exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError(f"Card {card_id} not found", entity="card", entity_id=card_id)  # noqa: TRY003

                cur = conn.execute(
                    """
                    SELECT linked_type, COUNT(*) as count
                    FROM kanban_card_links
                    WHERE card_id = ?
                    GROUP BY linked_type
                    """,
                    (card_id,)
                )

                counts = {"media": 0, "note": 0}
                for row in cur.fetchall():
                    counts[row["linked_type"]] = row["count"]

                return counts

            finally:
                conn.close()

    def remove_card_link(
        self,
        card_id: int,
        linked_type: str,
        linked_id: str
    ) -> bool:
        """
        Remove a link from a card.

        Args:
            card_id: The card to remove the link from.
            linked_type: Type of linked content.
            linked_id: ID of the linked content.

        Returns:
            True if the link was removed, False if it didn't exist.

        Raises:
            NotFoundError: If the card doesn't exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError(f"Card {card_id} not found", entity="card", entity_id=card_id)  # noqa: TRY003

                cur = conn.execute(
                    """
                    DELETE FROM kanban_card_links
                    WHERE card_id = ? AND linked_type = ? AND linked_id = ?
                    """,
                    (card_id, linked_type, linked_id)
                )

                if cur.rowcount > 0:
                    # Log activity
                    self._log_activity_internal(
                        conn, card["board_id"], "link_removed", "card_link",
                        card_id=card_id, list_id=card["list_id"],
                        details={"linked_type": linked_type, "linked_id": linked_id}
                    )
                    conn.commit()
                    return True

                return False

            finally:
                conn.close()

    def remove_card_link_by_id(self, link_id: int) -> bool:
        """
        Remove a card link by its ID.

        Args:
            link_id: The link ID to remove.

        Returns:
            True if the link was removed, False if it didn't exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Get link details first for activity logging
                cur = conn.execute(
                    """
                    SELECT cl.*, c.board_id, c.list_id
                    FROM kanban_card_links cl
                    JOIN kanban_cards c ON cl.card_id = c.id
                    JOIN kanban_boards b ON c.board_id = b.id
                    WHERE cl.id = ? AND b.user_id = ?
                    """,
                    (link_id, self.user_id)
                )
                link = cur.fetchone()

                if not link:
                    return False

                conn.execute("DELETE FROM kanban_card_links WHERE id = ?", (link_id,))

                # Log activity
                self._log_activity_internal(
                    conn, link["board_id"], "link_removed", "card_link",
                    card_id=link["card_id"], list_id=link["list_id"],
                    details={"linked_type": link["linked_type"], "linked_id": link["linked_id"]}
                )
                conn.commit()
                return True

            finally:
                conn.close()

    def remove_card_link_by_id_for_card(self, card_id: int, link_id: int) -> bool:
        """
        Remove a card link by its ID, scoped to a specific card.

        Args:
            card_id: The card ID the link must belong to.
            link_id: The link ID to remove.

        Returns:
            True if the link was removed, False if it didn't exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT cl.*, c.board_id, c.list_id
                    FROM kanban_card_links cl
                    JOIN kanban_cards c ON cl.card_id = c.id
                    JOIN kanban_boards b ON c.board_id = b.id
                    WHERE cl.id = ? AND cl.card_id = ? AND b.user_id = ?
                    """,
                    (link_id, card_id, self.user_id)
                )
                link = cur.fetchone()

                if not link:
                    return False

                conn.execute("DELETE FROM kanban_card_links WHERE id = ?", (link_id,))

                self._log_activity_internal(
                    conn, link["board_id"], "link_removed", "card_link",
                    card_id=link["card_id"], list_id=link["list_id"],
                    details={"linked_type": link["linked_type"], "linked_id": link["linked_id"]}
                )
                conn.commit()
                return True

            finally:
                conn.close()

    def bulk_add_card_links(
        self,
        card_id: int,
        links: list[dict[str, str]]
    ) -> dict[str, Any]:
        """
        Add multiple links to a card at once.

        Args:
            card_id: The card to add links to.
            links: List of dicts with "linked_type" and "linked_id".

        Returns:
            Dict with added_count, skipped_count, and links list.

        Raises:
            NotFoundError: If the card doesn't exist.
            InputError: If any linked_type is invalid.
        """
        # Validate all linked_types first
        for link in links:
            if link.get("linked_type") not in ("media", "note"):
                raise InputError(  # noqa: TRY003
                    f"Invalid linked_type: {link.get('linked_type')}. Must be 'media' or 'note'."
                )

        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError(f"Card {card_id} not found", entity="card", entity_id=card_id)  # noqa: TRY003

                added_links = []
                skipped_count = 0
                now = _utcnow_iso()

                for link in links:
                    link_uuid = _generate_uuid()
                    linked_type = link["linked_type"]
                    linked_id = link["linked_id"]

                    self._validate_link_target_or_raise(linked_type=linked_type, linked_id=linked_id)

                    try:
                        cur = conn.execute(
                            """
                            INSERT INTO kanban_card_links (uuid, card_id, linked_type, linked_id, created_at)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (link_uuid, card_id, linked_type, linked_id, now)
                        )
                        link_id = cur.lastrowid
                        added_links.append({
                            "id": link_id,
                            "card_id": card_id,
                            "linked_type": linked_type,
                            "linked_id": linked_id,
                            "created_at": now
                        })
                    except sqlite3.IntegrityError:
                        # Duplicate link, skip
                        skipped_count += 1

                if added_links:
                    # Log activity
                    self._log_activity_internal(
                        conn, card["board_id"], "links_bulk_added", "card",
                        card_id=card_id, list_id=card["list_id"],
                        details={"added_count": len(added_links), "skipped_count": skipped_count}
                    )

                conn.commit()

                return {
                    "added_count": len(added_links),
                    "skipped_count": skipped_count,
                    "links": added_links
                }

            finally:
                conn.close()

    def bulk_remove_card_links(
        self,
        card_id: int,
        links: list[dict[str, str]]
    ) -> dict[str, Any]:
        """
        Remove multiple links from a card at once.

        Args:
            card_id: The card to remove links from.
            links: List of dicts with "linked_type" and "linked_id".

        Returns:
            Dict with removed_count.

        Raises:
            NotFoundError: If the card doesn't exist.
        """
        with self._lock:
            conn = self._connect()
            try:
                # Verify card exists and belongs to user
                card = self._get_card_by_id(conn, card_id)
                if not card:
                    raise NotFoundError(f"Card {card_id} not found", entity="card", entity_id=card_id)  # noqa: TRY003

                removed_count = 0

                for link in links:
                    linked_type = link.get("linked_type")
                    linked_id = link.get("linked_id")

                    cur = conn.execute(
                        """
                        DELETE FROM kanban_card_links
                        WHERE card_id = ? AND linked_type = ? AND linked_id = ?
                        """,
                        (card_id, linked_type, linked_id)
                    )
                    removed_count += cur.rowcount

                if removed_count > 0:
                    # Log activity
                    self._log_activity_internal(
                        conn, card["board_id"], "links_bulk_removed", "card",
                        card_id=card_id, list_id=card["list_id"],
                        details={"removed_count": removed_count}
                    )
                    conn.commit()

                return {"removed_count": removed_count}

            finally:
                conn.close()

    def get_cards_by_linked_content(
        self,
        linked_type: str,
        linked_id: str,
        include_archived: bool = False,
        include_deleted: bool = False
    ) -> list[dict[str, Any]]:
        """
        Find all cards that link to a specific media item or note.

        This is the bidirectional lookup - given a media item or note ID,
        find all Kanban cards that reference it.

        Args:
            linked_type: Type of content ('media' or 'note').
            linked_id: ID of the content.
            include_archived: Include archived cards.
            include_deleted: Include soft-deleted cards.

        Returns:
            List of cards with board/list context and link info.
        """
        with self._lock:
            conn = self._connect()
            try:
                query = """
                    SELECT
                        c.id,
                        c.title,
                        c.description,
                        c.board_id,
                        b.name as board_name,
                        c.list_id,
                        l.name as list_name,
                        c.position,
                        c.archived as is_archived,
                        c.deleted as is_deleted,
                        cl.id as link_id,
                        cl.created_at as linked_at
                    FROM kanban_card_links cl
                    JOIN kanban_cards c ON cl.card_id = c.id
                    JOIN kanban_boards b ON c.board_id = b.id
                    JOIN kanban_lists l ON c.list_id = l.id
                    WHERE cl.linked_type = ?
                      AND cl.linked_id = ?
                      AND b.user_id = ?
                """

                params: list[Any] = [linked_type, linked_id, self.user_id]

                if not include_archived:
                    query += " AND c.archived = 0"

                if not include_deleted:
                    query += " AND c.deleted = 0"

                query += " ORDER BY cl.created_at DESC"

                cur = conn.execute(query, params)
                return [dict(row) for row in cur.fetchall()]

            finally:
                conn.close()

    def get_card_counts_for_lists(self, list_ids: list[int]) -> dict[int, int]:
        """
        Get card counts for multiple lists in a single query.

        Args:
            list_ids: List of list IDs to get card counts for.

        Returns:
            Dict mapping list_id -> card count.
        """
        if not list_ids:
            return {}

        with self._lock:
            conn = self._connect()
            try:
                placeholders = ",".join("?" * len(list_ids))
                cur = conn.execute(
                    """
                    SELECT list_id, COUNT(*) as count
                    FROM kanban_cards
                    WHERE list_id IN ({placeholders}) AND deleted = 0
                    GROUP BY list_id
                    """.format_map(locals()),  # nosec B608
                    list_ids
                )

                return {row["list_id"]: row["count"] for row in cur.fetchall()}

            finally:
                conn.close()

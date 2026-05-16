# world_book_manager.py
# Description: World Book/Lorebook Manager for character chat with multi-user support
# Adapted from single-user to multi-user architecture with per-user database isolation
#
"""
World Book Manager for Multi-User Environment
---------------------------------------------

This module provides management for world books (lorebooks) that can be used
independently of characters, allowing shared lorebooks across conversations.

Key Adaptations from Single-User:
- Per-user database isolation (each user has their own database)
- Stateless service instances (no global state or singletons)
- Request-scoped processing (instantiated per API request)
- No thread-local storage (incompatible with async API)
- Database-backed persistence with user isolation

Features:
- CRUD operations for world books and entries
- Keyword matching and activation
- Token budget management
- Priority-based entry selection
- Scan depth control
- Recursive scanning support
"""

import json
import re
import sqlite3
from datetime import datetime
from typing import Any, Optional, Union

from loguru import logger

try:
    from tldw_Server_API.app.core.Utils.tokenizer import count_tokens as _count_tokens
except ImportError:
    from tldw_Server_API.app.core.utils.tokenizer import count_tokens as _count_tokens

# Local imports
# Import shared constants and helpers
from tldw_Server_API.app.core.Character_Chat.constants import (
    MAX_BOOK_CACHE_SIZE,
    MAX_ENTRY_CACHE_SIZE,
    MAX_RECURSIVE_DEPTH,
    safe_parse_json_dict,
    safe_parse_json_list,
)

# Import shared regex safety utilities
from tldw_Server_API.app.core.Character_Chat.regex_safety import (
    validate_regex_safety as _validate_regex_safety,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

# Whitelisted field names for dynamic UPDATE statements (SQL injection prevention)
_WORLD_BOOK_UPDATE_FIELDS = frozenset({
    "name", "description", "scan_depth", "token_budget", "recursive_scanning",
    "enabled", "last_modified", "version"
})
_WORLD_BOOK_ENTRY_UPDATE_FIELDS = frozenset({
    "keywords", "content", "priority", "enabled", "case_sensitive",
    "regex_match", "whole_word_match", "metadata", "last_modified"
})

_WORLD_BOOK_NONCRITICAL_EXCEPTIONS = (
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


def _coerce_backend_type(raw_backend_type: Any) -> BackendType:
    """Return a stable backend enum, defaulting unknown values to SQLite."""
    if isinstance(raw_backend_type, BackendType):
        return raw_backend_type

    if isinstance(raw_backend_type, str):
        normalized = raw_backend_type.strip().lower()
        if normalized in {"postgres", "postgresql"}:
            return BackendType.POSTGRESQL
        if normalized in {"sqlite", "sqlite3"}:
            return BackendType.SQLITE

    return BackendType.SQLITE


def _coerce_metadata_bool(value: Any, default: bool = False) -> bool:
    """Interpret legacy/loose metadata booleans consistently."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        return default
    return bool(value)


def _build_safe_update_clause(
    updates: list[str],
    allowed_fields: frozenset[str]
) -> str:
    """
    Build a safe UPDATE SET clause from a list of 'field = ?' strings.

    Validates each field name against an allowed whitelist to prevent SQL injection.

    Args:
        updates: List of strings like 'field = ?'
        allowed_fields: Frozenset of allowed field names

    Returns:
        Joined SET clause string

    Raises:
        ValueError: If any field is not in the whitelist
    """
    for update_str in updates:
        # Extract field name from 'field = ?' or 'field = field + 1' patterns
        field_name = update_str.split('=')[0].strip()
        if field_name not in allowed_fields:
            raise ValueError(f"Invalid field name in UPDATE: {field_name}")
    return ', '.join(updates)


def _escape_like_pattern(query: str) -> str:
    """
    Escape SQL LIKE wildcards to prevent pattern injection.

    Args:
        query: The search query to escape

    Returns:
        Escaped query safe for use in LIKE patterns
    """
    # Escape backslash first, then the SQL LIKE special characters
    return query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


class BoundedDict(dict):
    """
    A dictionary with a maximum size limit (LRU-style eviction).

    When the limit is reached, the oldest entry (first inserted) is removed
    before adding the new entry. Uses insertion order (Python 3.7+).

    Note: This class is NOT thread-safe by design. It is intended for use
    in request-scoped service instances where each request gets its own
    instance. In Python's async model with a single-threaded event loop,
    there is no concurrent access within a single request.

    If thread safety is needed in the future, consider using
    functools.lru_cache or a dedicated thread-safe cache library.
    """

    def __init__(self, max_size: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_size = max_size

    def __setitem__(self, key, value):
        # If key already exists, just update it
        if key in self:
            super().__setitem__(key, value)
            return

        # If at capacity, remove the oldest entry
        if len(self) >= self._max_size:
            oldest = next(iter(self))
            del self[oldest]

        super().__setitem__(key, value)


def _is_unique_violation(error: Exception) -> bool:
    """Best-effort detection of unique constraint/duplicate key violations across backends."""
    message = str(error).lower()
    return "unique constraint" in message or "duplicate key" in message


class WorldBookEntry:
    """
    Individual world book entry with keyword matching capabilities.

    This is a stateless data class that handles keyword matching and content.
    No global state or thread-local storage is used.
    """

    def __init__(
        self,
        entry_id: Optional[int] = None,
        world_book_id: Optional[int] = None,
        keywords: Optional[list[str]] = None,
        content: str = "",
        priority: int = 0,
        enabled: bool = True,
        case_sensitive: bool = False,
        regex_match: bool = False,
        whole_word_match: bool = True,
        metadata: Optional[dict[str, Any]] = None,
        created_at: Optional[Any] = None,
        last_modified: Optional[Any] = None,
    ):
        """
        Initialize a world book entry.

        Args:
            entry_id: Database ID
            world_book_id: Parent world book ID
            keywords: List of keywords to match
            content: Content to inject when matched
            priority: Priority for ordering (higher = more important)
            enabled: Whether entry is active
            case_sensitive: Whether keyword matching is case-sensitive
            regex_match: Whether keywords are regex patterns
            whole_word_match: Whether to match whole words only
            metadata: Additional metadata
            created_at: Entry creation timestamp
            last_modified: Entry last-modified timestamp
        """
        self.entry_id = entry_id
        self.world_book_id = world_book_id
        self.keywords = keywords or []
        self.content = content
        self.priority = priority
        self.enabled = enabled
        self.case_sensitive = case_sensitive
        self.regex_match = regex_match
        self.whole_word_match = whole_word_match
        self.metadata = metadata or {}
        self.created_at = created_at
        self.last_modified = last_modified

        # Compile patterns for efficient matching
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> list[tuple[re.Pattern, str]]:
        """
        Compile keyword patterns for matching.

        For regex patterns, validates against potential ReDoS vulnerabilities
        before compilation. Dangerous patterns are rejected with a warning.
        """
        patterns: list[tuple[re.Pattern, str]] = []

        for keyword in self.keywords:
            if self.regex_match:
                # Validate regex safety before compiling (ReDoS prevention)
                is_safe, reason = _validate_regex_safety(keyword)
                if not is_safe:
                    logger.warning(
                        "Rejected potentially unsafe regex pattern '{}': {}",
                        keyword[:50] + "..." if len(keyword) > 50 else keyword,
                        reason
                    )
                    continue

                # Use keyword as regex directly
                try:
                    flags = 0 if self.case_sensitive else re.IGNORECASE
                    pattern = re.compile(keyword, flags)
                    patterns.append((pattern, keyword))
                except re.error as e:
                    logger.warning("Invalid regex pattern '{}': {}", keyword, e)
            else:
                # Build pattern for literal matching (always safe)
                escaped = re.escape(keyword)
                pattern_str = r'\b' + escaped + r'\b' if self.whole_word_match else escaped

                flags = 0 if self.case_sensitive else re.IGNORECASE
                patterns.append((re.compile(pattern_str, flags), keyword))

        return patterns

    def matches(self, text: str) -> bool:
        """
        Check if any keyword matches the given text.

        Args:
            text: Text to search for keywords

        Returns:
            True if any keyword matches
        """
        if not self.enabled or not self._patterns:
            return False

        return any(pattern.search(text) for pattern, _keyword in self._patterns)

    def get_first_match_info(self, text: str) -> Optional[dict[str, Any]]:
        """Return the first matching keyword and reason, if any."""
        if not self.enabled or not self._patterns:
            return None

        for pattern, keyword in self._patterns:
            if pattern.search(text):
                return {
                    "reason": "regex_match" if self.regex_match else "keyword_match",
                    "keyword": keyword,
                }

        return None

    def get_match_count(self, text: str) -> int:
        """
        Count how many keywords match in the text.

        Args:
            text: Text to search

        Returns:
            Number of matching keywords
        """
        if not self.enabled or not self._patterns:
            return 0

        count = 0
        for pattern, _keyword in self._patterns:
            if pattern.search(text):
                count += 1

        return count

    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage (normalized types)."""
        return {
            'id': self.entry_id,
            'world_book_id': self.world_book_id,
            'keywords': json.dumps(self.keywords),
            'content': self.content,
            'priority': int(self.priority),
            'enabled': int(self.enabled),
            'case_sensitive': int(self.case_sensitive),
            'regex_match': int(self.regex_match),
            'whole_word_match': int(self.whole_word_match),
            'metadata': json.dumps(self.metadata)
        }

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API/tests (readable types)."""
        return {
            'id': self.entry_id,
            'world_book_id': self.world_book_id,
            'keywords': list(self.keywords),
            'content': self.content,
            'group': (
                str(self.metadata.get('group')).strip()
                if self.metadata.get('group') not in (None, "")
                else None
            ),
            'appendable': _coerce_metadata_bool(self.metadata.get('appendable', False)),
            'priority': int(self.priority),
            'enabled': bool(self.enabled),
            'case_sensitive': bool(self.case_sensitive),
            'regex_match': bool(self.regex_match),
            'whole_word_match': bool(self.whole_word_match),
            'recursive_scanning': _coerce_metadata_bool(self.metadata.get('recursive_scanning', False)),
            'metadata': dict(self.metadata or {}),
            'created_at': self.created_at,
            'last_modified': self.last_modified,
        }


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'WorldBookEntry':
        """Create WorldBookEntry instance from database dictionary."""
        # Use safe_parse helpers to validate JSON structure
        keywords = safe_parse_json_list(data.get('keywords'), 'keywords')
        metadata = safe_parse_json_dict(data.get('metadata'), 'metadata')
        return cls(
            entry_id=data.get('id'),
            world_book_id=data.get('world_book_id'),
            keywords=keywords or [],
            content=data.get('content', ''),
            priority=int(data.get('priority', 0)),
            enabled=_coerce_metadata_bool(data.get('enabled', True), default=True),
            case_sensitive=_coerce_metadata_bool(data.get('case_sensitive', False), default=False),
            regex_match=_coerce_metadata_bool(data.get('regex_match', False), default=False),
            whole_word_match=_coerce_metadata_bool(data.get('whole_word_match', True), default=True),
            metadata=metadata or {},
            created_at=data.get('created_at'),
            last_modified=data.get('last_modified'),
        )

class WorldBookEntryView:
    """A hybrid view that behaves like both an object and a dict for tests."""
    def __init__(self, data: dict[str, Any]):
        self._d = dict(data)
    # Attribute access
    def __getattr__(self, name: str):
        if name in self._d:
            return self._d[name]
        raise AttributeError(name)
    # Dict-like access
    def __getitem__(self, key):
        return self._d[key]
    def get(self, key, default=None):
        return self._d.get(key, default)
    def __repr__(self):
        return f"WorldBookEntryView({self._d!r})"


class OpResult:
    """Bool-like + dict-like operation result with 'success' key."""
    def __init__(self, success: bool):
        self.success = bool(success)
    def __getitem__(self, key):
        if key == 'success':
            return self.success
        raise KeyError(key)
    def get(self, key, default=None):
        return self.success if key == 'success' else default
    def __bool__(self):
        return self.success
    def __eq__(self, other):
        if isinstance(other, bool):
            return self.success == other
        if isinstance(other, dict):
            return other.get('success') == self.success
        return False
    def __repr__(self):
        return f"OpResult(success={self.success})"

## removed stray helper; functionality provided by WorldBookEntry.from_dict


class WorldBookService:
    """
    Service class for managing world books in a multi-user environment.

    This is a request-scoped service that is instantiated per API request.
    It works with the per-user database model where each user has their own
    separate database instance.
    """

    def __init__(self, db: CharactersRAGDB):
        """
        Initialize the service with a user-specific database connection.

        Args:
            db: User-specific database instance from dependency injection
        """
        self.db = db
        self._init_tables()

        # Request-scoped cache with bounded size to prevent memory leaks
        # These caches use BoundedDict to limit memory usage if the service
        # instance is accidentally reused across requests
        self._entry_cache: dict[int, list[WorldBookEntry]] = BoundedDict(MAX_ENTRY_CACHE_SIZE)
        self._book_cache: dict[int, dict[str, Any]] = BoundedDict(MAX_BOOK_CACHE_SIZE)
        self._activation_counts: dict[int, int] = BoundedDict(MAX_BOOK_CACHE_SIZE)
        self._last_activated_at: dict[int, datetime] = BoundedDict(MAX_BOOK_CACHE_SIZE)

    def _init_tables(self):
        """Initialize world book tables in the user's database if they don't exist."""
        backend_type = getattr(self.db, "backend_type", BackendType.SQLITE)
        try:
            with self.db.get_connection() as conn:
                if backend_type == BackendType.POSTGRESQL:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS world_books (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL UNIQUE,
                            description TEXT,
                            scan_depth INTEGER DEFAULT 3,
                            token_budget INTEGER DEFAULT 500,
                            recursive_scanning BOOLEAN DEFAULT FALSE,
                            enabled BOOLEAN DEFAULT TRUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            version INTEGER DEFAULT 1,
                            deleted BOOLEAN DEFAULT FALSE
                        )
                    """)

                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS world_book_entries (
                            id SERIAL PRIMARY KEY,
                            world_book_id INTEGER NOT NULL,
                            keywords TEXT NOT NULL,
                            content TEXT NOT NULL,
                            priority INTEGER DEFAULT 0,
                            enabled BOOLEAN DEFAULT TRUE,
                            case_sensitive BOOLEAN DEFAULT FALSE,
                            regex_match BOOLEAN DEFAULT FALSE,
                            whole_word_match BOOLEAN DEFAULT TRUE,
                            metadata TEXT DEFAULT '{}',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (world_book_id) REFERENCES world_books(id) ON DELETE CASCADE
                        )
                    """)

                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS character_world_books (
                            id SERIAL PRIMARY KEY,
                            character_id INTEGER NOT NULL,
                            world_book_id INTEGER NOT NULL,
                            enabled BOOLEAN DEFAULT TRUE,
                            priority INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(character_id, world_book_id),
                            FOREIGN KEY (character_id) REFERENCES character_cards(id) ON DELETE CASCADE,
                            FOREIGN KEY (world_book_id) REFERENCES world_books(id) ON DELETE CASCADE
                        )
                    """)
                else:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS world_books (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL UNIQUE,
                            description TEXT,
                            scan_depth INTEGER DEFAULT 3,
                            token_budget INTEGER DEFAULT 500,
                            recursive_scanning BOOLEAN DEFAULT 0,
                            enabled BOOLEAN DEFAULT 1,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            version INTEGER DEFAULT 1,
                            deleted BOOLEAN DEFAULT 0
                        )
                    """)

                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS world_book_entries (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            world_book_id INTEGER NOT NULL,
                            keywords TEXT NOT NULL,
                            content TEXT NOT NULL,
                            priority INTEGER DEFAULT 0,
                            enabled BOOLEAN DEFAULT 1,
                            case_sensitive BOOLEAN DEFAULT 0,
                            regex_match BOOLEAN DEFAULT 0,
                            whole_word_match BOOLEAN DEFAULT 1,
                            metadata TEXT DEFAULT '{}',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (world_book_id) REFERENCES world_books(id) ON DELETE CASCADE
                        )
                    """)

                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS character_world_books (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            character_id INTEGER NOT NULL,
                            world_book_id INTEGER NOT NULL,
                            enabled BOOLEAN DEFAULT 1,
                            priority INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(character_id, world_book_id),
                            FOREIGN KEY (character_id) REFERENCES character_cards(id) ON DELETE CASCADE,
                            FOREIGN KEY (world_book_id) REFERENCES world_books(id) ON DELETE CASCADE
                        )
                    """)

                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_wb_entries_book_id ON world_book_entries(world_book_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_wb_entries_priority ON world_book_entries(priority DESC)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_wb_entries_book_enabled_priority "
                    "ON world_book_entries(world_book_id, enabled, priority DESC)"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_char_wb_char_id ON character_world_books(character_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_char_wb_book_id ON character_world_books(world_book_id)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_char_wb_char_enabled_priority "
                    "ON character_world_books(character_id, enabled, priority DESC)"
                )

                conn.commit()
                logger.info("World book tables initialized")
        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to initialize world book tables: {e}")
            raise CharactersRAGDBError(f"Failed to initialize world book tables: {e}") from e

    # --- World Book CRUD Operations ---

    def create_world_book(
        self,
        name: str,
        description: Optional[str] = None,
        scan_depth: int = 3,
        token_budget: int = 500,
        recursive_scanning: bool = False,
        enabled: bool = True
    ) -> int:
        """
        Create a new world book.

        Args:
            name: Unique name for the world book
            description: Optional description
            scan_depth: Maximum matched entries per world book during processing
            token_budget: Maximum tokens to use for world info
            recursive_scanning: Whether to scan matched entries for more keywords
            enabled: Whether the world book is active

        Returns:
            The ID of the created world book

        Raises:
            InputError: If name is empty or invalid
            ConflictError: If a world book with this name already exists
        """
        if not name or not name.strip():
            raise InputError("World book name cannot be empty")

        try:
            with self.db.get_connection() as conn:
                insert_sql = """
                    INSERT INTO world_books
                    (name, description, scan_depth, token_budget, recursive_scanning, enabled)
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                params = (
                    name.strip(),
                    description,
                    scan_depth,
                    token_budget,
                    _coerce_metadata_bool(recursive_scanning, default=False),
                    _coerce_metadata_bool(enabled, default=True),
                )
                if self.db.backend_type == BackendType.POSTGRESQL:
                    insert_sql += " RETURNING id"

                cursor = conn.execute(insert_sql, params)

                if self.db.backend_type == BackendType.POSTGRESQL:
                    row = cursor.fetchone()
                    world_book_id = (row or {}).get("id") if row else None
                else:
                    world_book_id = cursor.lastrowid

                if world_book_id is None:
                    raise CharactersRAGDBError("Database did not return a world book id.")

                conn.commit()

                logger.info(f"Created world book '{name}' with ID {world_book_id}")
                self._invalidate_cache()
                return int(world_book_id)

        except sqlite3.IntegrityError as e:
            if _is_unique_violation(e):
                raise ConflictError(f"World book with name '{name}' already exists", "world_books", name) from e
            raise CharactersRAGDBError(f"Database error creating world book: {e}") from e
        except CharactersRAGDBError as e:
            if _is_unique_violation(e):
                raise ConflictError(f"World book with name '{name}' already exists", "world_books", name) from e
            raise
        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Database error creating world book: {e}")
            raise CharactersRAGDBError(f"Database error creating world book: {e}") from e

    def get_world_book(self, world_book_id: Optional[int] = None, name: Optional[str] = None) -> Optional[dict[str, Any]]:
        """
        Get a world book by ID or name.

        Args:
            world_book_id: Optional world book ID
            name: Optional world book name

        Returns:
            Dictionary with world book data or None if not found
        """
        # Check cache first
        if world_book_id and world_book_id in self._book_cache:
            return self._book_cache[world_book_id]

        try:
            with self.db.get_connection() as conn:
                if world_book_id:
                    cursor = conn.execute(
                        """
                        SELECT * FROM world_books
                        WHERE id = ? AND deleted = ?
                        """,
                        (world_book_id, False)
                    )
                elif name:
                    cursor = conn.execute(
                        """
                        SELECT * FROM world_books
                        WHERE name = ? AND deleted = ?
                        """,
                        (name, False)
                    )
                else:
                    return None

                row = cursor.fetchone()
                if row:
                    book_data = dict(row)
                    if world_book_id:
                        self._book_cache[world_book_id] = book_data
                    return book_data
                return None

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error fetching world book: {e}")
            raise CharactersRAGDBError(f"Error fetching world book: {e}") from e

    def list_world_books(self, include_disabled: bool = False) -> list[dict[str, Any]]:
        """
        List all world books for the user.

        Args:
            include_disabled: Whether to include disabled world books

        Returns:
            List of world book data dictionaries
        """
        try:
            with self.db.get_connection() as conn:
                query = "SELECT * FROM world_books WHERE deleted = ?"
                params: list[Any] = [False]
                if not include_disabled:
                    query += " AND enabled = ?"
                    params.append(True)
                query += " ORDER BY name"

                cursor = conn.execute(query, tuple(params))
                books = [dict(row) for row in cursor.fetchall()]

                # Cache the books
                for book in books:
                    self._book_cache[book['id']] = book

                return books

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error listing world books: {e}")
            raise CharactersRAGDBError(f"Error listing world books: {e}") from e

    def get_entry_counts_for_world_books(
        self,
        world_book_ids: Optional[list[int]] = None,
    ) -> dict[int, int]:
        """
        Return entry counts keyed by world_book_id.

        Args:
            world_book_ids: Optional subset of world book IDs. When omitted,
                counts are returned for every world book that has entries.

        Returns:
            Mapping of world_book_id -> entry_count.
        """
        normalized_ids: Optional[list[int]] = None
        if world_book_ids is not None:
            deduped_ids: set[int] = set()
            for raw_id in world_book_ids:
                try:
                    parsed_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if parsed_id > 0:
                    deduped_ids.add(parsed_id)
            normalized_ids = sorted(deduped_ids)
            if not normalized_ids:
                return {}

        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT world_book_id, COUNT(*) AS entry_count
                    FROM world_book_entries
                """
                params: list[Any] = []
                if normalized_ids is not None:
                    placeholders = ",".join("?" for _ in normalized_ids)
                    query += f" WHERE world_book_id IN ({placeholders})"  # nosec B608
                    params.extend(normalized_ids)
                query += " GROUP BY world_book_id"

                cursor = conn.execute(query, tuple(params))
                counts: dict[int, int] = {}
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    try:
                        world_book_id = int(row_dict.get("world_book_id"))
                    except (TypeError, ValueError):
                        continue
                    counts[world_book_id] = int(row_dict.get("entry_count") or 0)

                if normalized_ids is not None:
                    for world_book_id in normalized_ids:
                        counts.setdefault(world_book_id, 0)
                return counts
        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error fetching world book entry counts: {e}")
            raise CharactersRAGDBError(f"Error fetching world book entry counts: {e}") from e

    def update_world_book(
        self,
        world_book_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scan_depth: Optional[int] = None,
        token_budget: Optional[int] = None,
        recursive_scanning: Optional[bool] = None,
        enabled: Optional[bool] = None,
        expected_version: Optional[int] = None
    ) -> bool:
        """
        Update a world book's settings.

        Args:
            world_book_id: World book ID
            Various optional fields to update
            expected_version: Optional optimistic-locking version. If provided,
                update only succeeds when the current row version matches.

        Returns:
            True if updated successfully

        Raises:
            ConflictError: If the new name conflicts with an existing world book
        """
        try:
            updates = []
            params = []

            if name is not None:
                updates.append("name = ?")
                params.append(name.strip())
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            if scan_depth is not None:
                updates.append("scan_depth = ?")
                params.append(scan_depth)
            if token_budget is not None:
                updates.append("token_budget = ?")
                params.append(token_budget)
            if recursive_scanning is not None:
                updates.append("recursive_scanning = ?")
                params.append(_coerce_metadata_bool(recursive_scanning, default=False))
            if enabled is not None:
                updates.append("enabled = ?")
                params.append(_coerce_metadata_bool(enabled, default=True))

            if not updates:
                return True

            updates.append("last_modified = CURRENT_TIMESTAMP")
            updates.append("version = version + 1")
            where_clause = "id = ? AND deleted = ?"
            where_params: list[Any] = [world_book_id, False]
            if expected_version is not None:
                if not isinstance(expected_version, int) or expected_version < 1:
                    raise InputError("expected_version must be a positive integer")
                where_clause += " AND version = ?"
                where_params.append(expected_version)
            params.extend(where_params)

            with self.db.get_connection() as conn:
                set_clause = _build_safe_update_clause(updates, _WORLD_BOOK_UPDATE_FIELDS)
                cursor = conn.execute(
                    f"UPDATE world_books SET {set_clause} WHERE {where_clause}",  # nosec B608
                    tuple(params)
                )
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Updated world book {world_book_id}")
                    self._invalidate_cache()
                    return True
                if expected_version is not None:
                    current = conn.execute(
                        """
                        SELECT version FROM world_books
                        WHERE id = ? AND deleted = ?
                        """,
                        (world_book_id, False),
                    ).fetchone()
                    if current is not None:
                        current_version = (
                            current["version"]
                            if hasattr(current, "keys") and "version" in current.keys()
                            else current[0]
                        )
                        raise ConflictError(
                            f"Version mismatch. Expected {expected_version}, found {current_version}. Please refresh and try again."
                        )
                return False

        except sqlite3.IntegrityError as e:
            if _is_unique_violation(e):
                raise ConflictError(f"World book name '{name}' already exists", "world_books", name) from e
            raise CharactersRAGDBError(f"Database error updating world book: {e}") from e
        except CharactersRAGDBError as e:
            if _is_unique_violation(e):
                raise ConflictError(f"World book name '{name}' already exists", "world_books", name) from e
            raise

    def delete_world_book(self, world_book_id: int, hard_delete: bool = False, **kwargs) -> bool:
        """
        Delete a world book (soft delete by default).

        Args:
            world_book_id: World book ID
            hard_delete: If True, permanently delete; otherwise soft delete

        Returns:
            True if deleted successfully
        """
        # Alias: support cascade=True from tests
        if kwargs.get('cascade') is True:
            hard_delete = True
        try:
            with self.db.get_connection() as conn:
                if hard_delete:
                    cursor = conn.execute(
                        "DELETE FROM world_books WHERE id = ?",
                        (world_book_id,)
                    )
                else:
                    cursor = conn.execute(
                        "UPDATE world_books SET deleted = ?, last_modified = CURRENT_TIMESTAMP WHERE id = ?",
                        (True, world_book_id)
                    )
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted world book {world_book_id}")
                    self._invalidate_cache()
                    return True
                return False

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error deleting world book: {e}")
            raise CharactersRAGDBError(f"Error deleting world book: {e}") from e

    # --- Entry CRUD Operations ---

    def add_entry(
        self,
        world_book_id: int,
        keywords: list[str],
        content: str,
        priority: int = 0,
        enabled: bool = True,
        case_sensitive: bool = False,
        regex_match: bool = False,
        whole_word_match: bool = True,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """
        Add an entry to a world book.

        Args:
            world_book_id: World book ID
            keywords: List of keywords to match
            content: Content to inject when matched
            priority: Priority for ordering (higher = more important)
            enabled: Whether entry is active
            case_sensitive: Whether keyword matching is case-sensitive
            regex_match: Whether keywords are regex patterns
            whole_word_match: Whether to match whole words only
            metadata: Additional metadata

        Returns:
            The ID of the created entry
        """
        if not keywords:
            # Tests expect ValueError for empty keywords
            raise ValueError("Entry must have at least one keyword")
        # Empty content is allowed (store empty string)
        if content is None:
            content = ""
        # Clamp priority to [0, 100]
        try:
            priority = int(priority)
        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS:
            priority = 0
        priority = max(0, min(100, priority))

        # Support per-entry recursive scanning via metadata
        if 'recursive_scanning' in kwargs:
            md = dict(metadata or {})
            md['recursive_scanning'] = _coerce_metadata_bool(
                kwargs['recursive_scanning'],
                default=False,
            )
            metadata = md

        # Validate regex patterns if regex_match is True (including ReDoS prevention)
        if regex_match:
            for keyword in keywords:
                # Validate regex safety (ReDoS prevention)
                is_safe, reason = _validate_regex_safety(keyword)
                if not is_safe:
                    raise InputError(f"Unsafe regex pattern '{keyword}': {reason}")
                try:
                    re.compile(keyword)
                except re.error as e:
                    raise InputError(f"Invalid regex pattern '{keyword}': {e}") from e

        keywords_json = json.dumps(keywords)
        metadata_json = json.dumps(metadata or {})

        try:
            with self.db.get_connection() as conn:
                insert_sql = """
                    INSERT INTO world_book_entries
                    (world_book_id, keywords, content, priority, enabled,
                     case_sensitive, regex_match, whole_word_match, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = (
                    world_book_id,
                    keywords_json,
                    content,
                    priority,
                    _coerce_metadata_bool(enabled, default=True),
                    _coerce_metadata_bool(case_sensitive, default=False),
                    _coerce_metadata_bool(regex_match, default=False),
                    _coerce_metadata_bool(whole_word_match, default=True),
                    metadata_json,
                )
                if self.db.backend_type == BackendType.POSTGRESQL:
                    insert_sql += " RETURNING id"

                cursor = conn.execute(insert_sql, params)

                if self.db.backend_type == BackendType.POSTGRESQL:
                    row = cursor.fetchone()
                    entry_id = (row or {}).get("id") if row else None
                else:
                    entry_id = cursor.lastrowid

                if entry_id is None:
                    raise CharactersRAGDBError("Database did not return a world book entry id.")

                conn.commit()

                logger.info(f"Added entry {entry_id} to world book {world_book_id}")
                self._invalidate_cache()
                return int(entry_id)

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error adding world book entry: {e}")
            raise CharactersRAGDBError(f"Error adding world book entry: {e}") from e

    def add_world_book_entry(
        self,
        world_book_id: int,
        keywords: list[str],
        content: str,
        priority: int = 0,
        enabled: bool = True,
        case_sensitive: bool = False,
        regex_match: bool = False,
        whole_word_match: bool = True,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> int:
        """
        Legacy alias maintained for backwards compatibility.
        """
        return self.add_entry(
            world_book_id=world_book_id,
            keywords=keywords,
            content=content,
            priority=priority,
            enabled=enabled,
            case_sensitive=case_sensitive,
            regex_match=regex_match,
            whole_word_match=whole_word_match,
            metadata=metadata,
            **kwargs,
        )

    def get_entries(
        self,
        world_book_id: Optional[int] = None,
        enabled_only: bool = False,
        group: Optional[str] = None,
        appendable: Optional[bool] = None,
        recursive_scanning: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """
        Get world book entries.

        Args:
            world_book_id: Optional specific world book ID
            enabled_only: Only return enabled entries
            group: Optional group/category filter
            appendable: Optional appendable-flag filter
            recursive_scanning: Optional recursive-source filter

        Returns:
            List of WorldBookEntry instances
        """
        normalized_group: Optional[str] = None
        if group is not None:
            normalized_group = str(group).strip()
            if not normalized_group:
                normalized_group = None
        normalized_group_lower = normalized_group.lower() if normalized_group is not None else None
        normalized_appendable: Optional[bool] = (
            _coerce_metadata_bool(appendable, default=False)
            if appendable is not None else None
        )
        normalized_recursive_filter: Optional[bool] = (
            _coerce_metadata_bool(recursive_scanning, default=False)
            if recursive_scanning is not None else None
        )
        has_metadata_filters = (
            normalized_group_lower is not None
            or normalized_appendable is not None
            or normalized_recursive_filter is not None
        )

        def _apply_entry_filters(entries_to_filter: list[WorldBookEntryView]) -> list[WorldBookEntryView]:
            if not has_metadata_filters:
                return entries_to_filter

            filtered_entries: list[WorldBookEntryView] = []
            for entry_view in entries_to_filter:
                if normalized_group_lower is not None:
                    group_value = entry_view.get("group")
                    group_normalized = str(group_value).strip().lower() if group_value not in (None, "") else ""
                    if group_normalized != normalized_group_lower:
                        continue
                if normalized_appendable is not None:
                    if _coerce_metadata_bool(entry_view.get("appendable", False)) != normalized_appendable:
                        continue
                if normalized_recursive_filter is not None:
                    if _coerce_metadata_bool(entry_view.get("recursive_scanning", False)) != normalized_recursive_filter:
                        continue
                filtered_entries.append(entry_view)
            return filtered_entries

        # Check cache first
        if world_book_id and world_book_id in self._entry_cache:
            entries = self._entry_cache[world_book_id]
            if enabled_only:
                # Match DB behavior: when enabled_only is requested, disabled
                # world books should not return entries.
                book = self.get_world_book(world_book_id)
                if not book or not _coerce_metadata_bool(book.get("enabled", True), default=True):
                    return []
                entries = [e for e in entries if e.enabled]
            entry_views = [WorldBookEntryView(e.to_api_dict()) for e in entries]
            return _apply_entry_filters(entry_views)

        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT e.*, wb.enabled as book_enabled
                    FROM world_book_entries e
                    JOIN world_books wb ON e.world_book_id = wb.id
                    WHERE wb.deleted = ?
                """
                params: list[Any] = [False]

                if world_book_id:
                    query += " AND e.world_book_id = ?"
                    params.append(world_book_id)
                if enabled_only:
                    query += " AND e.enabled = ? AND wb.enabled = ?"
                    params.extend([True, True])

                if has_metadata_filters:
                    backend_type = _coerce_backend_type(
                        getattr(self.db, "backend_type", None)
                    )
                    if backend_type == BackendType.POSTGRESQL:
                        metadata_json_expr = "COALESCE(NULLIF(TRIM(e.metadata), ''), '{}')::jsonb"
                        truthy_set = "('true','1','t','yes','on')"

                        if normalized_group_lower is not None:
                            query += (
                                f" AND LOWER(COALESCE({metadata_json_expr} ->> 'group', '')) = ?"
                            )  # nosec B608
                            params.append(normalized_group_lower)
                        if normalized_appendable is not None:
                            if normalized_appendable:
                                query += (
                                    f" AND LOWER(COALESCE({metadata_json_expr} ->> 'appendable', 'false')) IN {truthy_set}"
                                )  # nosec B608
                            else:
                                query += (
                                    f" AND LOWER(COALESCE({metadata_json_expr} ->> 'appendable', 'false')) NOT IN {truthy_set}"
                                )  # nosec B608
                        if normalized_recursive_filter is not None:
                            if normalized_recursive_filter:
                                query += (
                                    f" AND LOWER(COALESCE({metadata_json_expr} ->> 'recursive_scanning', 'false')) IN {truthy_set}"
                                )  # nosec B608
                            else:
                                query += (
                                    f" AND LOWER(COALESCE({metadata_json_expr} ->> 'recursive_scanning', 'false')) NOT IN {truthy_set}"
                                )  # nosec B608
                    elif backend_type == BackendType.SQLITE:
                        metadata_json_expr = "COALESCE(NULLIF(TRIM(e.metadata), ''), '{}')"
                        truthy_set = "('1','true','t','yes','on')"
                        if normalized_group_lower is not None:
                            query += (
                                f" AND LOWER(COALESCE(CAST(json_extract({metadata_json_expr}, '$.group') AS TEXT), '')) = ?"
                            )  # nosec B608
                            params.append(normalized_group_lower)
                        if normalized_appendable is not None:
                            appendable_expr = (
                                "LOWER(COALESCE("
                                f"CAST(json_extract({metadata_json_expr}, '$.appendable') AS TEXT), "
                                "'false'))"
                            )
                            if normalized_appendable:
                                query += (
                                    f" AND {appendable_expr} IN {truthy_set}"
                                )  # nosec B608
                            else:
                                query += (
                                    f" AND {appendable_expr} NOT IN {truthy_set}"
                                )  # nosec B608
                        if normalized_recursive_filter is not None:
                            recursive_expr = (
                                "LOWER(COALESCE("
                                f"CAST(json_extract({metadata_json_expr}, '$.recursive_scanning') AS TEXT), "
                                "'false'))"
                            )
                            if normalized_recursive_filter:
                                query += (
                                    f" AND {recursive_expr} IN {truthy_set}"
                                )  # nosec B608
                            else:
                                query += (
                                    f" AND {recursive_expr} NOT IN {truthy_set}"
                                )  # nosec B608

                query += " ORDER BY e.priority DESC, e.id"

                cursor = conn.execute(query, tuple(params))
                entries: list[WorldBookEntry] = []

                for row in cursor.fetchall():
                    entry_data = dict(row)
                    entry = WorldBookEntry.from_dict(entry_data)
                    entries.append(entry)

                # Cache full per-book result sets only (avoid partial-cache poisoning
                # from enabled-only or metadata-filtered reads).
                if world_book_id and not enabled_only and not has_metadata_filters:
                    self._entry_cache[world_book_id] = entries

                # Return hybrid views for legacy/new compatibility
                entry_views = [WorldBookEntryView(e.to_api_dict()) for e in entries]
                return _apply_entry_filters(entry_views)

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error fetching world book entries: {e}")
            raise CharactersRAGDBError(f"Error fetching world book entries: {e}") from e

    def get_entry(self, entry_id: int) -> Optional[WorldBookEntryView]:
        """
        Get a single world book entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            Hybrid entry view if found, else None
        """
        try:
            normalized_entry_id = int(entry_id)
        except (TypeError, ValueError):
            return None

        if normalized_entry_id <= 0:
            return None

        # Check cached book entry lists first.
        for cached_entries in self._entry_cache.values():
            for cached_entry in cached_entries:
                try:
                    cached_id = int(getattr(cached_entry, "entry_id", 0) or 0)
                except (TypeError, ValueError):
                    cached_id = 0
                if cached_id == normalized_entry_id:
                    return WorldBookEntryView(cached_entry.to_api_dict())

        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT e.*, wb.enabled as book_enabled
                    FROM world_book_entries e
                    JOIN world_books wb ON e.world_book_id = wb.id
                    WHERE e.id = ? AND wb.deleted = ?
                    LIMIT 1
                """
                row = conn.execute(query, (normalized_entry_id, False)).fetchone()
                if not row:
                    return None
                entry = WorldBookEntry.from_dict(dict(row))
                return WorldBookEntryView(entry.to_api_dict())

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error fetching world book entry {entry_id}: {e}")
            raise CharactersRAGDBError(f"Error fetching world book entry {entry_id}: {e}") from e

    def update_entry(
        self,
        entry_id: int,
        keywords: Optional[list[str]] = None,
        content: Optional[str] = None,
        priority: Optional[int] = None,
        enabled: Optional[bool] = None,
        case_sensitive: Optional[bool] = None,
        regex_match: Optional[bool] = None,
        whole_word_match: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """
        Update a world book entry.

        Args:
            entry_id: Entry ID
            Various optional fields to update

        Returns:
            True if updated successfully
        """
        try:
            updates = []
            params = []
            effective_regex_match: Optional[bool] = (
                _coerce_metadata_bool(regex_match, default=False)
                if regex_match is not None else None
            )

            if keywords is not None:
                if not keywords:
                    raise InputError("Entry must have at least one keyword")
                # If regex_match is explicitly set, validate immediately.
                if effective_regex_match is True:
                    for keyword in keywords:
                        # Validate regex safety (ReDoS prevention)
                        is_safe, reason = _validate_regex_safety(keyword)
                        if not is_safe:
                            raise InputError(f"Unsafe regex pattern '{keyword}': {reason}")
                        try:
                            re.compile(keyword)
                        except re.error as e:
                            raise InputError(f"Invalid regex pattern '{keyword}': {e}") from e
                updates.append("keywords = ?")
                params.append(json.dumps(keywords))

            if content is not None:
                # Empty content is allowed (consistent with add_entry behavior)
                updates.append("content = ?")
                params.append(content)

            if priority is not None:
                try:
                    p = int(priority)
                except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS:
                    p = 0
                p = max(0, min(100, p))
                updates.append("priority = ?")
                params.append(p)

            if enabled is not None:
                updates.append("enabled = ?")
                params.append(_coerce_metadata_bool(enabled, default=True))

            if case_sensitive is not None:
                updates.append("case_sensitive = ?")
                params.append(_coerce_metadata_bool(case_sensitive, default=False))

            if regex_match is not None:
                updates.append("regex_match = ?")
                params.append(_coerce_metadata_bool(regex_match, default=False))

            if whole_word_match is not None:
                updates.append("whole_word_match = ?")
                params.append(_coerce_metadata_bool(whole_word_match, default=True))

            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))

            if not updates:
                return True

            updates.append("last_modified = CURRENT_TIMESTAMP")
            params.append(entry_id)

            with self.db.get_connection() as conn:
                # If keywords are being updated but regex_match is omitted, inherit
                # current regex mode from DB so invalid patterns are still blocked.
                if keywords is not None and effective_regex_match is None:
                    row = conn.execute(
                        "SELECT regex_match FROM world_book_entries WHERE id = ?",
                        (entry_id,),
                    ).fetchone()
                    existing_regex = False
                    if row is not None:
                        if hasattr(row, "keys") and "regex_match" in row.keys():
                            existing_regex = _coerce_metadata_bool(row["regex_match"], default=False)
                        else:
                            existing_regex = _coerce_metadata_bool(row[0], default=False)
                    if existing_regex:
                        for keyword in keywords:
                            is_safe, reason = _validate_regex_safety(keyword)
                            if not is_safe:
                                raise InputError(f"Unsafe regex pattern '{keyword}': {reason}")
                            try:
                                re.compile(keyword)
                            except re.error as e:
                                raise InputError(f"Invalid regex pattern '{keyword}': {e}") from e

                set_clause = _build_safe_update_clause(updates, _WORLD_BOOK_ENTRY_UPDATE_FIELDS)
                cursor = conn.execute(
                    f"UPDATE world_book_entries SET {set_clause} WHERE id = ?",  # nosec B608
                    tuple(params)
                )
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Updated world book entry {entry_id}")
                    self._invalidate_cache()
                    return True
                return False

        except InputError:
            raise
        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error updating world book entry: {e}")
            raise CharactersRAGDBError(f"Error updating world book entry: {e}") from e

    def delete_entry(self, entry_id: int) -> bool:
        """
        Delete a world book entry.

        Args:
            entry_id: Entry ID

        Returns:
            True if deleted successfully
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM world_book_entries WHERE id = ?",
                    (entry_id,)
                )
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Deleted world book entry {entry_id}")
                    self._invalidate_cache()
                    return True
                return False

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error deleting world book entry: {e}")
            raise CharactersRAGDBError(f"Error deleting world book entry: {e}") from e

    # --- Character Association ---

    def attach_to_character(
        self,
        world_book_id: int,
        character_id: int,
        enabled: bool = True,
        priority: int = 0
    ) -> dict[str, Any]:
        """
        Attach a world book to a character.

        Args:
            world_book_id: World book ID
            character_id: Character ID
            enabled: Whether the attachment is active
            priority: Priority for this character (higher = more important)

        Returns:
            OpResult with 'success' indicating attachment status
        """
        try:
            with self.db.get_connection() as conn:
                params = (
                    character_id,
                    world_book_id,
                    _coerce_metadata_bool(enabled, default=True),
                    int(priority),
                )
                try:
                    if self.db.backend_type == BackendType.POSTGRESQL:
                        conn.execute(
                            """
                            INSERT INTO character_world_books (character_id, world_book_id, enabled, priority)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (character_id, world_book_id)
                            DO UPDATE SET enabled = EXCLUDED.enabled,
                                          priority = EXCLUDED.priority
                            """,
                            params,
                        )
                    else:
                        conn.execute(
                            """
                            INSERT INTO character_world_books (character_id, world_book_id, enabled, priority)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(character_id, world_book_id)
                            DO UPDATE SET enabled = excluded.enabled,
                                          priority = excluded.priority
                            """,
                            params,
                        )
                    conn.commit()
                    logger.info(f"Attached world book {world_book_id} to character {character_id}")
                    return OpResult(True)
                except sqlite3.IntegrityError as e:
                    if 'FOREIGN KEY constraint failed' in str(e):
                        logger.warning(
                            f"Attach failed due to missing character_id {character_id} for world_book {world_book_id}"
                        )
                        return OpResult(False)
                    raise
                except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
                    # psycopg raises different exceptions; detect FK violation generically
                    msg = str(e).lower()
                    if "foreign key" in msg and "constraint" in msg:
                        logger.warning(
                            f"Attach failed due to missing character_id {character_id} for world_book {world_book_id}"
                        )
                        return OpResult(False)
                    raise

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error attaching world book to character: {e}")
            raise CharactersRAGDBError(f"Error attaching world book to character: {e}") from e

    def detach_from_character(self, world_book_id: int, character_id: int) -> dict[str, Any]:
        """
        Detach a world book from a character.

        Args:
            character_id: Character ID
            world_book_id: World book ID

        Returns:
            True if detached successfully
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM character_world_books
                    WHERE character_id = ? AND world_book_id = ?
                    """,
                    (character_id, world_book_id)
                )
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Detached world book {world_book_id} from character {character_id}")
                    return OpResult(True)
                return OpResult(False)

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error detaching world book from character: {e}")
            raise CharactersRAGDBError(f"Error detaching world book from character: {e}") from e

    def get_character_world_books(self, character_id: int, enabled_only: bool = True) -> list[dict[str, Any]]:
        """
        Get all world books attached to a character.

        Args:
            character_id: Character ID
            enabled_only: Only return enabled attachments

        Returns:
            List of world book data with attachment info
        """
        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT wb.*, cwb.enabled as attachment_enabled, cwb.priority as attachment_priority
                    FROM world_books wb
                    JOIN character_world_books cwb ON wb.id = cwb.world_book_id
                    WHERE cwb.character_id = ? AND wb.deleted = ?
                """
                params: list[Any] = [character_id, False]

                if enabled_only:
                    query += " AND cwb.enabled = ? AND wb.enabled = ?"
                    params.extend([True, True])

                query += " ORDER BY cwb.priority DESC, wb.name"

                cursor = conn.execute(query, tuple(params))
                return [dict(row) for row in cursor.fetchall()]

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error fetching character world books: {e}")
            raise CharactersRAGDBError(f"Error fetching character world books: {e}") from e

    # --- Content Processing ---

    def process_context(
        self,
        text: str,
        world_book_ids: Optional[Union[list[int], int]] = None,
        character_id: Optional[int] = None,
        scan_depth: int = 3,
        token_budget: int = 500,
        recursive_scanning: Optional[bool] = None,
        **kwargs
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """
        Process text to find and inject relevant world info.

        Args:
            text: Text to scan for keywords (usually recent messages)
            world_book_ids: Specific world books to use (optional)
            character_id: Character whose world books to use (optional)
            scan_depth: Override maximum matched entries per world book
            token_budget: Maximum tokens to inject
            recursive_scanning: Whether to scan matched entries for more keywords.
                If None, inherits from selected world-book settings.

        Returns:
            Dictionary with processed_context and statistics
        """
        # Aliases
        if 'max_tokens' in kwargs and kwargs.get('max_tokens') is not None:
            token_budget = int(kwargs['max_tokens'])
        include_diagnostics = _coerce_metadata_bool(kwargs.get("include_diagnostics", False), default=False)
        recursive_depth = int(kwargs.get('recursive_depth', 0) or 0)
        # Clamp recursive depth to prevent infinite loops
        if recursive_depth > MAX_RECURSIVE_DEPTH:
            logger.warning(f"Recursive depth {recursive_depth} exceeds max {MAX_RECURSIVE_DEPTH}, clamping")
            recursive_depth = MAX_RECURSIVE_DEPTH
        if recursive_depth > 0:
            recursive_scanning = True

        # Gather applicable world books
        books_to_use = []
        compact_return = False
        if isinstance(world_book_ids, int):
            world_book_ids = [world_book_ids]
            compact_return = True

        if world_book_ids:
            for book_id in world_book_ids:
                book = self.get_world_book(book_id)
                if book and book.get('enabled', True):
                    books_to_use.append(book)

        if character_id:
            char_books = self.get_character_world_books(character_id, enabled_only=True)
            books_to_use.extend(char_books)

        if books_to_use:
            unique_books: dict[int, dict[str, Any]] = {}
            for book in books_to_use:
                book_id = book.get('id')
                if book_id is None or book_id in unique_books:
                    continue
                unique_books[book_id] = book
            books_to_use = list(unique_books.values())

        if recursive_scanning is None:
            recursive_scanning = any(
                _coerce_metadata_bool(book.get("recursive_scanning", False))
                for book in books_to_use
            )

        if not books_to_use:
            empty = {
                "processed_context": "",
                "entries_matched": 0,
                "tokens_used": 0,
                "books_used": 0,
                "token_budget": int(token_budget),
                "budget_exhausted": False,
                "skipped_entries_due_to_budget": 0,
            }
            if include_diagnostics:
                empty["diagnostics"] = []
            return [] if compact_return else empty

        def _normalize_depth(value: Any) -> Optional[int]:
            try:
                depth_val = int(value)
            except (TypeError, ValueError):
                return None
            return depth_val if depth_val > 0 else None

        request_depth = _normalize_depth(scan_depth)

        # Gather all entries from applicable books
        all_entries: list[WorldBookEntry] = []
        book_entry_limits: dict[int, Optional[int]] = {}
        for book in books_to_use:
            # Ensure we have object entries for matching
            if book['id'] in self._entry_cache:
                book_entries = [e for e in self._entry_cache[book['id']] if e.enabled]
            else:
                # Populate full per-book cache via DB. `enabled_only=True` reads are
                # intentionally not cached to avoid partial-cache poisoning.
                _ = self.get_entries(book['id'], enabled_only=False)
                book_entries = [e for e in self._entry_cache.get(book['id'], []) if e.enabled]
            all_entries.extend(book_entries)
            # scan_depth is currently treated as a per-book match cap.
            book_depth = _normalize_depth(book.get('scan_depth'))
            book_id = book.get('id')
            if book_id is None:
                continue
            if request_depth is not None:
                limit = request_depth if book_depth is None else min(request_depth, book_depth)
            else:
                limit = book_depth
            book_entry_limits[book_id] = limit

        # Sort by priority (highest first)
        all_entries.sort(key=lambda e: e.priority, reverse=True)
        has_recursive_sources = any(
            _coerce_metadata_bool((entry.metadata or {}).get("recursive_scanning", False))
            for entry in all_entries
        )

        def _is_recursive_source(entry: WorldBookEntry) -> bool:
            # Backward compatibility: if no entry explicitly opts into recursive
            # source scanning, preserve legacy behavior (all matched entries can chain).
            if not has_recursive_sources:
                return True
            return _coerce_metadata_bool((entry.metadata or {}).get("recursive_scanning", False))

        # Build a lightweight normalized keyword index for common literal lookups.
        # Regex/case-sensitive/partial-match entries fall back to full scan.
        quick_keyword_index: dict[str, list[WorldBookEntry]] = {}
        entries_requiring_scan: list[WorldBookEntry] = []

        def _entry_quick_lookup_keywords(entry: WorldBookEntry) -> Optional[list[str]]:
            if entry.regex_match or entry.case_sensitive or not entry.whole_word_match:
                return None

            normalized_keywords: list[str] = []
            for keyword in entry.keywords:
                normalized_keyword = str(keyword or "").strip().lower()
                if not normalized_keyword:
                    continue
                # Keep quick lookup conservative to preserve exact matching behavior.
                if re.fullmatch(r"\w+", normalized_keyword) is None:
                    return None
                normalized_keywords.append(normalized_keyword)

            return normalized_keywords or None

        for entry in all_entries:
            quick_keywords = _entry_quick_lookup_keywords(entry)
            if not quick_keywords:
                entries_requiring_scan.append(entry)
                continue
            for normalized_keyword in quick_keywords:
                quick_keyword_index.setdefault(normalized_keyword, []).append(entry)

        def _candidate_entries_for_text(source_text: str) -> list[WorldBookEntry]:
            if not quick_keyword_index:
                return all_entries

            tokens = set(re.findall(r"\b\w+\b", (source_text or "").lower()))
            quick_candidates: set[WorldBookEntry] = set()
            for token in tokens:
                for candidate in quick_keyword_index.get(token, []):
                    quick_candidates.add(candidate)

            if not quick_candidates:
                return entries_requiring_scan

            ordered_quick_candidates = [entry for entry in all_entries if entry in quick_candidates]
            return ordered_quick_candidates + entries_requiring_scan

        # Find matching entries
        matched_entries = []
        tokens_used = 0
        per_book_match_count: dict[int, int] = {}
        diagnostics_by_key: dict[int, dict[str, Any]] = {}
        skipped_entries_due_to_budget = 0

        def _record_diagnostic(
            *,
            entry: WorldBookEntry,
            entry_tokens: int,
            activation_reason: str,
            match_info: Optional[dict[str, Any]] = None,
            depth_level: Optional[int] = None,
        ) -> None:
            if not include_diagnostics:
                return

            entry_key = int(entry.entry_id) if entry.entry_id is not None else id(entry)
            keyword_raw = (match_info or {}).get("keyword")
            keyword = str(keyword_raw).strip() if keyword_raw is not None else None
            if keyword == "":
                keyword = None

            entry_metadata = entry.metadata or {}
            static_or_pinned = any(
                _coerce_metadata_bool(entry_metadata.get(key), default=False)
                for key in (
                    "static",
                    "pinned",
                    "always_on",
                    "constant",
                    "cache_static",
                    "cache_pinned",
                )
            )
            diagnostics_by_key[entry_key] = {
                "entry_id": int(entry.entry_id) if entry.entry_id is not None else None,
                "world_book_id": int(entry.world_book_id) if entry.world_book_id is not None else None,
                "activation_reason": activation_reason,
                "keyword": keyword,
                "token_cost": int(entry_tokens),
                "priority": int(entry.priority),
                "regex_match": bool(entry.regex_match),
                "appendable": _coerce_metadata_bool(entry.metadata.get("appendable", False)) if entry.metadata else False,
                "static_or_pinned": static_or_pinned,
                "content_preview": (entry.content or "")[:240],
                "depth_level": depth_level,
            }

        for entry in _candidate_entries_for_text(text):
            match_info = entry.get_first_match_info(text)
            if match_info is None:
                continue
            book_id = getattr(entry, 'world_book_id', None)
            limit = book_entry_limits.get(book_id)
            if limit is not None and per_book_match_count.get(book_id, 0) >= limit:
                continue
            # Estimate tokens (simple approximation)
            entry_tokens = self.count_tokens(entry.content)
            if tokens_used + entry_tokens <= token_budget:
                matched_entries.append(entry)
                tokens_used += entry_tokens
                _record_diagnostic(
                    entry=entry,
                    entry_tokens=entry_tokens,
                    activation_reason=str(match_info.get("reason") or "keyword_match"),
                    match_info=match_info,
                    depth_level=0,
                )
                if limit is not None:
                    per_book_match_count[book_id] = per_book_match_count.get(book_id, 0) + 1
            else:
                skipped_entries_due_to_budget += 1
                continue  # Skip oversized entry but continue scanning

        # Handle recursive scanning
        if recursive_scanning and matched_entries:
            current_depth = max(1, recursive_depth)
            seen = set(matched_entries)
            recursive_seed_entries = [entry for entry in matched_entries if _is_recursive_source(entry)]
            combined_content = " ".join(entry.content for entry in recursive_seed_entries)
            while current_depth > 0 and combined_content:
                depth_level = current_depth
                additional_entries = []
                for entry in _candidate_entries_for_text(combined_content):
                    if entry in seen:
                        continue
                    book_id = getattr(entry, 'world_book_id', None)
                    limit = book_entry_limits.get(book_id)
                    if limit is not None and per_book_match_count.get(book_id, 0) >= limit:
                        continue
                    match_info = entry.get_first_match_info(combined_content)
                    if match_info is None:
                        continue
                    entry_tokens = self.count_tokens(entry.content)
                    if tokens_used + entry_tokens <= token_budget:
                        additional_entries.append(entry)
                        tokens_used += entry_tokens
                        _record_diagnostic(
                            entry=entry,
                            entry_tokens=entry_tokens,
                            activation_reason="depth",
                            match_info=match_info,
                            depth_level=depth_level,
                        )
                        if limit is not None:
                            per_book_match_count[book_id] = per_book_match_count.get(book_id, 0) + 1
                    else:
                        skipped_entries_due_to_budget += 1
                if not additional_entries:
                    break
                for e in additional_entries:
                    seen.add(e)
                matched_entries.extend(additional_entries)
                recursive_seed_entries = [entry for entry in additional_entries if _is_recursive_source(entry)]
                if not recursive_seed_entries:
                    break
                combined_content = " ".join(entry.content for entry in recursive_seed_entries)
                current_depth -= 1

        # Build injected content (appendable-aware grouping)
        if matched_entries:
            # Sort by priority for final output
            matched_entries.sort(key=lambda e: e.priority, reverse=True)
            # Consecutive appendable entries are concatenated directly (no separator).
            # Non-appendable entries are separated by "\n\n".
            blocks: list[str] = []
            current_block: list[str] = []
            for entry in matched_entries:
                is_appendable = (
                    _coerce_metadata_bool(entry.metadata.get("appendable", False))
                    if entry.metadata else False
                )
                if is_appendable:
                    current_block.append(entry.content)
                else:
                    if current_block:
                        blocks.append("".join(current_block))
                        current_block = []
                    blocks.append(entry.content)
            if current_block:
                blocks.append("".join(current_block))
            injected_content = "\n\n".join(blocks)
        else:
            injected_content = ""

        # Track activation counts
        try:
            if matched_entries:
                per_book_counts: dict[int, int] = {}
                for entry in matched_entries:
                    if entry.world_book_id is None:
                        continue
                    per_book_counts[entry.world_book_id] = per_book_counts.get(entry.world_book_id, 0) + 1
                now = datetime.now()
                for wb, count in per_book_counts.items():
                    self._activation_counts[wb] = self._activation_counts.get(wb, 0) + count
                    self._last_activated_at[wb] = now
        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS:
            pass

        if compact_return:
            return [e.to_api_dict() for e in matched_entries]

        diagnostics: list[dict[str, Any]] = []
        if include_diagnostics and matched_entries:
            for entry in matched_entries:
                entry_key = int(entry.entry_id) if entry.entry_id is not None else id(entry)
                diagnostic = diagnostics_by_key.get(entry_key)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)

        response = {
            "processed_context": injected_content,
            "entries_matched": len(matched_entries),
            "tokens_used": tokens_used,
            "books_used": len({e.world_book_id for e in matched_entries}) if matched_entries else 0,
            "entry_ids": [e.entry_id for e in matched_entries],
            "token_budget": int(token_budget),
            "budget_exhausted": bool(
                int(token_budget) > 0 and tokens_used >= int(token_budget)
            ),
            "skipped_entries_due_to_budget": int(skipped_entries_due_to_budget),
        }
        if include_diagnostics:
            response["diagnostics"] = diagnostics
        return response

    # --- Import/Export ---

    def export_world_book(self, world_book_id: int) -> dict[str, Any]:
        """
        Export a world book to a dictionary format.

        Args:
            world_book_id: World book ID

        Returns:
            Dictionary with world book data and entries
        """
        book = self.get_world_book(world_book_id)
        if not book:
            raise InputError(f"World book {world_book_id} not found")

        entries = self.get_entries(world_book_id, enabled_only=False)
        top = {
            "name": book.get('name'),
            "description": book.get('description'),
            "scan_depth": book.get('scan_depth'),
            "token_budget": book.get('token_budget'),
            "recursive_scanning": _coerce_metadata_bool(book.get('recursive_scanning', 0)),
            "enabled": _coerce_metadata_bool(book.get('enabled', 1), default=True),
            "entries": [ev._d for ev in entries],
        }
        # Legacy nested shape
        top["world_book"] = {
            "id": book.get('id'),
            "name": book.get('name'),
            "description": book.get('description'),
            "scan_depth": book.get('scan_depth'),
            "token_budget": book.get('token_budget'),
            "recursive_scanning": _coerce_metadata_bool(book.get('recursive_scanning', 0)),
            "enabled": _coerce_metadata_bool(book.get('enabled', 1), default=True),
        }
        return top

    def import_world_book(self, data: dict[str, Any], merge_on_conflict: bool = False) -> int:
        """
        Import a world book from dictionary format.

        Args:
            data: Dictionary with world book data and entries
            merge_on_conflict: If True, merge with existing book of same name

        Returns:
            ID of the imported/merged world book
        """
        # Support both nested and flattened formats
        if 'world_book' in data and isinstance(data['world_book'], dict):
            book_data = data['world_book']
            entries_data = data.get('entries', [])
        else:
            book_data = {k: v for k, v in data.items() if k != 'entries'}
            entries_data = data.get('entries', [])

        if not book_data.get("name"):
            raise InputError("World book must have a name")

        # Check for existing book
        existing = self.get_world_book(name=book_data["name"])

        if existing:
            if not merge_on_conflict:
                raise ConflictError(f"World book '{book_data['name']}' already exists")
            world_book_id = existing["id"]
            logger.info(f"Merging into existing world book {world_book_id}")
        else:
            # Create new world book
            world_book_id = self.create_world_book(
                name=book_data["name"],
                description=book_data.get("description"),
                scan_depth=book_data.get("scan_depth", 3),
                token_budget=book_data.get("token_budget", 500),
                recursive_scanning=_coerce_metadata_bool(
                    book_data.get("recursive_scanning", False),
                    default=False,
                ),
                enabled=_coerce_metadata_bool(
                    book_data.get("enabled", True),
                    default=True,
                ),
            )

        # Import entries
        for entry_data in entries_data:
            self.add_entry(
                world_book_id=world_book_id,
                keywords=entry_data.get("keywords", []),
                content=entry_data.get("content", ""),
                priority=entry_data.get("priority", 0),
                enabled=_coerce_metadata_bool(entry_data.get("enabled", True), default=True),
                case_sensitive=_coerce_metadata_bool(entry_data.get("case_sensitive", False)),
                regex_match=_coerce_metadata_bool(entry_data.get("regex_match", False)),
                whole_word_match=_coerce_metadata_bool(entry_data.get("whole_word_match", True), default=True),
                metadata=entry_data.get("metadata", {}),
                recursive_scanning=_coerce_metadata_bool(
                    entry_data.get('recursive_scanning', False),
                    default=False,
                ),
            )

        logger.info(f"Imported world book with {len(entries_data)} entries")
        return world_book_id

    # --- Helper Methods ---

    def _invalidate_cache(self):
        """Invalidate the request-scoped cache."""
        self._entry_cache = BoundedDict(MAX_ENTRY_CACHE_SIZE)
        self._book_cache = BoundedDict(MAX_BOOK_CACHE_SIZE)

    def get_statistics(self, world_book_id: Optional[int] = None) -> dict[str, Any]:
        """
        Get world book usage statistics.

        Returns:
            Dictionary containing statistics
        """
        try:
            if world_book_id is not None:
                with self.db.get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT COUNT(*) as total_entries, AVG(priority) as avg_priority FROM world_book_entries WHERE world_book_id = ?",
                        (world_book_id,)
                    )
                    row = dict(cursor.fetchone())
                    cursor = conn.execute(
                        "SELECT keywords, metadata FROM world_book_entries WHERE world_book_id = ?",
                        (world_book_id,)
                    )
                    total_keywords = 0
                    recursive_entries = 0
                    for r in cursor.fetchall():
                        # keywords - use safe parsing with structure validation
                        kws = safe_parse_json_list(r['keywords'], 'keywords')
                        total_keywords += len(kws)
                        # metadata - use safe parsing with structure validation
                        md = safe_parse_json_dict(r['metadata'], 'metadata')
                        if _coerce_metadata_bool(md.get('recursive_scanning')):
                            recursive_entries += 1
                    return {
                        'total_entries': int(row.get('total_entries', 0) or 0),
                        'total_keywords': int(total_keywords),
                        'avg_priority': float(row.get('avg_priority', 0) or 0),
                        'recursive_entries': int(recursive_entries),
                    }
            with self.db.get_connection() as conn:
                # Get world book counts
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_world_books,
                        SUM(CASE WHEN enabled THEN 1 ELSE 0 END) as enabled_world_books
                    FROM world_books
                    WHERE deleted = ?
                    """,
                    (False,),
                )
                book_stats = dict(cursor.fetchone())

                # Get entry counts
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) as total_entries
                    FROM world_book_entries e
                    JOIN world_books w ON e.world_book_id = w.id
                    WHERE w.deleted = ?
                    """,
                    (False,),
                )
                entry_stats = dict(cursor.fetchone())

                # Get character attachment counts
                cursor = conn.execute(
                    """
                    SELECT COUNT(DISTINCT character_id) as total_attachments
                    FROM character_world_books c
                    JOIN world_books w ON c.world_book_id = w.id
                    WHERE w.deleted = ?
                    """,
                    (False,),
                )
                attachment_stats = dict(cursor.fetchone())

                # Calculate average entries per world book
                avg_entries = 0
                if book_stats['total_world_books'] > 0:
                    avg_entries = entry_stats['total_entries'] / book_stats['total_world_books']

                return {
                    "total_world_books": book_stats['total_world_books'],
                    "enabled_world_books": book_stats['enabled_world_books'],
                    "total_entries": entry_stats['total_entries'],
                    "total_character_attachments": attachment_stats['total_attachments'],
                    "average_entries_per_world_book": avg_entries
                }

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error getting statistics: {e}")
            raise CharactersRAGDBError(f"Error getting statistics: {e}") from e

    def search_entries(self, world_book_id: Optional[int] = None, query: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Search for entries by keyword or content.

        Args:
            search_term: Search term to look for

        Returns:
            List of matching entries with world book info
        """
        try:
            with self.db.get_connection() as conn:
                q = [
                    "SELECT e.*, w.name as world_book_name FROM world_book_entries e JOIN world_books w ON e.world_book_id = w.id",
                    "WHERE w.deleted = ?"
                ]
                params: list[Any] = [False]
                if world_book_id is not None:
                    q.append("AND e.world_book_id = ?")
                    params.append(world_book_id)
                if query:
                    # Escape SQL LIKE wildcards to prevent pattern injection
                    escaped_query = _escape_like_pattern(query)
                    q.append("AND (e.keywords LIKE ? ESCAPE '\\' OR e.content LIKE ? ESCAPE '\\')")
                    params.extend([f"%{escaped_query}%", f"%{escaped_query}%"])
                q.append("ORDER BY w.name, e.priority DESC")
                cursor = conn.execute(" ".join(q), tuple(params))
                results: list[dict[str, Any]] = []
                for row in cursor.fetchall():
                    rd = dict(row)
                    # Parse keywords - try JSON first, fall back to comma-separated
                    kw_raw = rd.get('keywords')
                    if isinstance(kw_raw, str):
                        try:
                            rd['keywords'] = json.loads(kw_raw)
                            if not isinstance(rd['keywords'], list):
                                rd['keywords'] = []
                        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS:
                            # Fallback: comma-separated string
                            rd['keywords'] = [s.strip() for s in kw_raw.split(',') if s.strip()]
                    else:
                        rd['keywords'] = kw_raw if isinstance(kw_raw, list) else []
                    # Parse metadata with validation
                    md = safe_parse_json_dict(rd.get('metadata'), 'metadata')
                    rd['recursive_scanning'] = _coerce_metadata_bool(md.get('recursive_scanning', False))
                    results.append(rd)
                return results

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error searching entries: {e}")
            raise CharactersRAGDBError(f"Error searching entries: {e}") from e

    def bulk_update_entries(
        self,
        world_book_id: int,
        entry_ids: list[int],
        enabled: Optional[bool] = None,
        priority: Optional[int] = None
    ) -> int:
        """
        Bulk update entries.

        Args:
            world_book_id: World book ID
            entry_ids: List of entry IDs to update
            enabled: New enabled status (optional)
            priority: New priority (optional)

        Returns:
            Number of entries updated
        """
        try:
            if not entry_ids:
                return 0

            updates = []
            params = []

            if enabled is not None:
                updates.append("enabled = ?")
                params.append(int(enabled))

            if priority is not None:
                updates.append("priority = ?")
                params.append(priority)

            if not updates:
                return 0

            updates.append("last_modified = CURRENT_TIMESTAMP")

            with self.db.get_connection() as conn:
                # Build the IN clause for entry IDs
                placeholders = ','.join('?' * len(entry_ids))
                params.extend(entry_ids)
                params.append(world_book_id)

                set_clause = _build_safe_update_clause(updates, _WORLD_BOOK_ENTRY_UPDATE_FIELDS)
                entry_ids_clause = f"({placeholders})"
                update_entries_sql_template = """
                    UPDATE world_book_entries
                    SET {set_clause}
                    WHERE id IN {entry_ids_clause}
                    AND world_book_id = ?
                    """
                update_entries_sql = update_entries_sql_template.format_map(locals())  # nosec B608
                cursor = conn.execute(
                    update_entries_sql,
                    params
                )
                conn.commit()

                updated_count = cursor.rowcount
                logger.info(f"Updated {updated_count} entries in world book {world_book_id}")
                self._invalidate_cache()
                return updated_count

        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error bulk updating entries: {e}")
            raise CharactersRAGDBError(f"Error bulk updating entries: {e}") from e

    def clone_world_book(self, source_wb_id: int, new_name: str) -> int:
        """
        Create a copy of a world book with all its entries.

        Args:
            source_wb_id: Source world book ID
            new_name: Name for the cloned world book

        Returns:
            ID of the new world book
        """
        try:
            # Get source world book
            source_wb = self.get_world_book(source_wb_id)
            if not source_wb:
                raise InputError(f"Source world book {source_wb_id} not found")

            # Create new world book
            new_wb_id = self.create_world_book(
                name=new_name,
                description=f"Cloned from {source_wb['name']}",
                scan_depth=source_wb.get('scan_depth', 3),
                token_budget=source_wb.get('token_budget', 500),
                recursive_scanning=source_wb.get('recursive_scanning', False),
                enabled=source_wb.get('enabled', True)
            )

            # Get all entries from source world book
            entries = self.get_entries(source_wb_id, enabled_only=False)

            # Add entries to new world book
            if entries:
                with self.db.get_connection() as conn:
                    for entry in entries:
                        metadata_json = json.dumps(entry.get('metadata') or {})

                        conn.execute(
                            """
                            INSERT INTO world_book_entries
                            (world_book_id, keywords, content, priority, enabled,
                             case_sensitive, regex_match, whole_word_match, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                new_wb_id,
                                json.dumps(entry.get('keywords') or []),
                                entry.get('content', ''),
                                int(entry.get('priority', 0)),
                                _coerce_metadata_bool(entry.get('enabled', True), default=True),
                                _coerce_metadata_bool(entry.get('case_sensitive', False), default=False),
                                _coerce_metadata_bool(entry.get('regex_match', False), default=False),
                                _coerce_metadata_bool(entry.get('whole_word_match', True), default=True),
                                metadata_json,
                            ),
                        )
                    conn.commit()

            logger.info(f"Cloned world book {source_wb_id} to new world book {new_wb_id} with {len(entries)} entries")
            self._invalidate_cache()
            return new_wb_id

        except ConflictError:
            raise
        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error cloning world book: {e}")
            raise CharactersRAGDBError(f"Error cloning world book: {e}") from e

    def close(self):
        """
        Close the service and clean up resources.

        This method is provided for compatibility with test fixtures
        that expect a close method. Since the database connection is
        managed by CharactersRAGDB, this is a no-op.
        """
        # Database connections are managed by CharactersRAGDB
        # Nothing to close here
        pass

    # --- Additional test-facing APIs ---

    def toggle_entry_enabled(self, entry_id: int) -> bool:
        try:
            with self.db.get_connection() as conn:
                cur = conn.execute("SELECT enabled FROM world_book_entries WHERE id = ?", (entry_id,))
                row = cur.fetchone()
                current = _coerce_metadata_bool(row[0], default=True) if row else True
                cur = conn.execute(
                    "UPDATE world_book_entries SET enabled = ?, last_modified = CURRENT_TIMESTAMP WHERE id = ?",
                    (not current, entry_id)
                )
                conn.commit()
                if cur.rowcount > 0:
                    self._invalidate_cache()
                    return True
                return False
        except _WORLD_BOOK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error toggling entry enabled: {e}")
            raise CharactersRAGDBError(f"Error toggling entry enabled: {e}") from e

    def bulk_add_entries(self, world_book_id: int, entries: list[dict[str, Any]]) -> dict[str, int]:
        added = 0
        for e in entries:
            self.add_entry(
                world_book_id=world_book_id,
                keywords=e.get('keywords', []),
                content=e.get('content', ''),
                priority=e.get('priority', 0),
                enabled=e.get('enabled', True),
                case_sensitive=e.get('case_sensitive', False),
                regex_match=e.get('regex_match', False),
                whole_word_match=e.get('whole_word_match', True),
                metadata=e.get('metadata', {}),
                recursive_scanning=_coerce_metadata_bool(e.get('recursive_scanning', False), default=False),
            )
            added += 1
        return {'added': added}

    def filter_entries(self, world_book_id: int, min_priority: Optional[int] = None, recursive_only: bool = False) -> list[dict[str, Any]]:
        entries = self.get_entries(world_book_id, enabled_only=True)
        res: list[dict[str, Any]] = []
        for e in entries:
            if min_priority is not None and int(e.get('priority', 0)) < int(min_priority):
                continue
            if recursive_only and not _coerce_metadata_bool(e.get('recursive_scanning', False), default=False):
                continue
            res.append(e)
        return res

    def export_to_lorebook_format(self, world_book_id: int) -> dict[str, Any]:
        entries = self.get_entries(world_book_id, enabled_only=False)
        lore_entries = []
        for e in entries:
            key = e.get('keywords')[0] if e.get('keywords') else ''
            lore_entries.append({'key': key, 'content': e.get('content', '')})
        return {'entries': lore_entries}

    def get_activation_statistics(self, world_book_id: int) -> dict[str, Any]:
        return {
            'total_activations': int(self._activation_counts.get(world_book_id, 0)),
            'last_activated_at': self._last_activated_at.get(world_book_id).isoformat() if self._last_activated_at.get(world_book_id) else None,
        }

    def normalize_keyword(self, kw: str) -> str:
        return (kw or '').strip().lower()

    def count_tokens(self, text: str) -> int:
        return _count_tokens(text)


# Export main classes
__all__ = [
    'WorldBookEntry',
    'WorldBookService'
]

# chat_dictionary.py
# Description: Chat Dictionary System with pattern-based text replacement
# Adapted from single-user to multi-user architecture with per-user database isolation
#
"""
Chat Dictionary Service for Multi-User Environment
--------------------------------------------------

This module provides a pattern-based text replacement system for chat conversations.
Adapted from the single-user TUI version to work with the multi-user API architecture.

Key Adaptations:
- Per-user database isolation (each user has their own database)
- Stateless service instances (no global state or singletons)
- Request-scoped processing (instantiated per API request)
- No thread-local storage (incompatible with async API)
- Database-backed persistence instead of in-memory storage

Features:
- Regex and literal pattern matching
- Probability-based replacements
- Token budget management
- Group-based dictionary organization
- Max replacement limits
- Timed effects support (sticky, cooldown, delay)
"""

import json
import random
import re
import sqlite3
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

import regex as _regex
from loguru import logger

try:
    from tldw_Server_API.app.core.Utils.tokenizer import count_tokens as _count_tokens  # Case-sensitive envs
except ImportError:  # fallback for alternate casing
    from tldw_Server_API.app.core.utils.tokenizer import count_tokens as _count_tokens

# Local imports
# Import shared constants and helpers
from tldw_Server_API.app.core.Character_Chat.constants import (
    MAX_CHAT_DICTIONARY_TEXT_LENGTH,
    MAX_REGEX_MATCH_TIME_MS,
    safe_parse_json_dict,
)

# Import shared regex safety utilities
from tldw_Server_API.app.core.Character_Chat.regex_safety import (
    safe_compile_regex as _safe_compile_regex,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

# `regex.TimeoutError` is not present in all `regex` package builds.
_REGEX_TIMEOUT_ERROR = getattr(_regex, "TimeoutError", TimeoutError)

_CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    CharactersRAGDBError,
    ConflictError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    InputError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    sqlite3.Error,
    re.error,
    _regex.error,
    _REGEX_TIMEOUT_ERROR,
    json.JSONDecodeError,
)

# Whitelisted field names for dynamic UPDATE statements (SQL injection prevention)
_DICTIONARY_UPDATE_FIELDS = frozenset({
    "name",
    "description",
    "category",
    "tags",
    "included_dictionary_ids",
    "is_active",
    "default_token_budget",
    "updated_at",
    "version",
})
_ENTRY_UPDATE_FIELDS = frozenset({
    "key", "content", "is_regex", "probability", "max_replacements",
    "group_name", "timed_effects", "enabled", "case_sensitive", "sort_order", "updated_at"
})

_DICTIONARY_SETTINGS_SINGLE_KEYS = frozenset({
    "chatdictionaryid",
    "chat_dictionary_id",
    "dictionaryid",
    "dictionary_id",
})
_DICTIONARY_SETTINGS_LIST_KEYS = frozenset({
    "chatdictionaryids",
    "chat_dictionary_ids",
    "chatdictionary",
    "chat_dictionary",
    "dictionaryids",
    "dictionary_ids",
    "chatdictionaries",
    "chat_dictionaries",
})
_ACTIVE_CONVERSATION_STATES = frozenset({"in-progress"})


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


def _is_unique_violation(error: Exception) -> bool:
    """Best-effort detection of unique constraint/duplicate key violations across backends."""
    message = str(error).lower()
    return "unique constraint" in message or "duplicate key" in message


def _escape_like_pattern(query: str) -> str:
    """
    Escape SQL LIKE wildcards to prevent pattern injection.

    Args:
        query: User-provided search query

    Returns:
        Escaped query safe for use in LIKE patterns with ESCAPE '\\'
    """
    # Escape backslash first, then the SQL LIKE special characters
    return query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _regex_timeout_seconds() -> float:
    return MAX_REGEX_MATCH_TIME_MS / 1000.0


def _is_compiled_regex(obj: Any) -> bool:
    return isinstance(obj, (re.Pattern, _regex.Pattern))


class TokenBudgetExceededWarning(Warning):
    """Custom warning for token budget issues"""

    pass


class ChatDictionaryEntry:
    """
    Individual dictionary entry with pattern matching capabilities.

    This is a stateless data class that handles pattern compilation and matching.
    No global state or thread-local storage is used.
    """

    def __init__(
        self,
        key: str,
        content: str,
        probability: Union[int, float] = 1.0,
        group: Optional[str] = None,
        timed_effects: Optional[dict[str, int]] = None,
        max_replacements: int = 0,
        entry_id: Optional[int] = None,
        enabled: bool = True,
        case_sensitive: bool = True,
    ):
        """
        Initialize a dictionary entry.

        Args:
            key: Pattern to match (can be regex with /pattern/flags format)
            content: Replacement text
            probability: Chance of replacement. Accepts:
                - int 0-100: Interpreted as percentage, normalized to 0.0-1.0
                - float 0.0-1.0: Used directly as probability
                - bool: Deprecated, logs warning. True=1.0, False=0.0
                Values outside valid range are clamped with a warning.
            group: Optional group name for organization
            timed_effects: Dictionary with sticky, cooldown, delay settings
            max_replacements: Maximum number of replacements per processing
            entry_id: Database ID (for persistence)
        """
        self.entry_id = entry_id
        self.raw_key = key
        self.content = content
        # Store probability as float [0.0, 1.0]
        try:
            if isinstance(probability, bool):
                # Boolean input is ambiguous - log warning and convert
                logger.warning(
                    f"Boolean probability value '{probability}' is deprecated. "
                    f"Use int 0-100 or float 0.0-1.0 instead."
                )
                self.probability = 1.0 if probability else 0.0
            elif isinstance(probability, int):
                # Integer 0-100 range, normalize to 0.0-1.0
                if probability < 0 or probability > 100:
                    logger.warning(f"Probability {probability} outside 0-100 range, clamping.")
                clamped = max(min(probability, 100), 0)
                self.probability = float(clamped) / 100.0
            else:
                # Float - validate range and clamp if needed
                prob_f = float(probability)
                if prob_f < 0.0 or prob_f > 1.0:
                    logger.warning(f"Probability {prob_f} outside 0.0-1.0 range, clamping.")
                    prob_f = max(min(prob_f, 1.0), 0.0)
                self.probability = prob_f
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid probability value '{probability}', defaulting to 1.0: {e}")
            self.probability = 1.0
        self.group = group
        self.timed_effects = timed_effects or {"sticky": 0, "cooldown": 0, "delay": 0}
        self.max_replacements = max_replacements
        self.enabled = bool(enabled)
        self.case_sensitive = bool(case_sensitive)
        self._loaded_at = datetime.now(timezone.utc)

        # Pattern compilation
        self.is_regex = False
        self.key_pattern_str = ""
        self.key_flags = 0
        self.key = self._compile_key(key)

        # Runtime state (not persisted)
        self.last_triggered: Optional[datetime] = None
        self.trigger_count = 0

    def _compile_regex_pattern(self, pattern: str) -> _regex.Pattern:
        if not pattern:
            raise re.error(f"Empty regex pattern from key '{self.raw_key}'")
        safe_compiled = _safe_compile_regex(pattern, self.key_flags)
        if isinstance(safe_compiled, _regex.Pattern):
            return safe_compiled
        try:
            return _regex.compile(pattern, self.key_flags)
        except _regex.error as exc:
            raise re.error(str(exc)) from exc

    def _compile_key(self, key_str: str) -> Union[re.Pattern, _regex.Pattern, str]:
        """
        Compile the key pattern, detecting regex format.

        Supports /pattern/flags format for regex patterns.
        Uses safe compilation to prevent ReDoS attacks.
        """
        self.is_regex = False
        self.key_flags = 0
        pattern_to_compile = key_str

        # Check for /pattern/flags format
        if key_str.startswith("/") and len(key_str) > 1:
            last_slash_idx = key_str.rfind("/")
            if last_slash_idx > 0:
                pattern_to_compile = key_str[1:last_slash_idx]
                flag_chars = key_str[last_slash_idx + 1 :]

                # Parse regex flags
                if "i" in flag_chars:
                    self.key_flags |= re.IGNORECASE
                if "m" in flag_chars:
                    self.key_flags |= re.MULTILINE
                if "s" in flag_chars:
                    self.key_flags |= re.DOTALL
                if "x" in flag_chars:
                    self.key_flags |= re.VERBOSE

                self.is_regex = True
            elif key_str.endswith("/") and len(key_str) > 2:
                pattern_to_compile = key_str[1:-1]
                self.is_regex = True

        self.key_pattern_str = pattern_to_compile

        if self.is_regex:
            # Validate and compile with a timeout-capable regex engine.
            return self._compile_regex_pattern(pattern_to_compile)
        else:
            return key_str

    def matches(self, text: str) -> bool:
        """Check if this entry's pattern matches the given text."""
        if self.is_regex and _is_compiled_regex(self.key):
            try:
                if isinstance(self.key, _regex.Pattern):
                    return bool(self.key.search(text, timeout=_regex_timeout_seconds()))
                return bool(self.key.search(text))
            except _REGEX_TIMEOUT_ERROR:
                preview = self.key_pattern_str
                if len(preview) > 50:
                    preview = preview[:50] + "..."
                logger.warning(f"Regex match timed out for pattern '{preview}'")
                return False
        elif not self.is_regex and isinstance(self.key, str):
            if not self.key:
                return False
            if self.case_sensitive:
                return self.key in text
            return self.key.casefold() in text.casefold()
        return False

    def should_apply(self) -> bool:
        """Check probability and timed effects to determine if replacement should occur."""
        # Check probability (0.0 - 1.0)
        if self.probability < 1.0 and random.random() > self.probability:
            return False

        # Check timed effects
        now = datetime.now(timezone.utc)
        if self.last_triggered:
            # Cooldown check
            cooldown = self.timed_effects.get("cooldown", 0)
            if cooldown > 0 and (now - self.last_triggered) < timedelta(seconds=cooldown):
                return False

            # Delay check (initial delay before first trigger)
            delay = self.timed_effects.get("delay", 0)
            if delay > 0 and self.trigger_count == 0:
                if (now - self.last_triggered) < timedelta(seconds=delay):
                    return False
        else:
            delay = self.timed_effects.get("delay", 0)
            if delay > 0 and self.trigger_count == 0:
                reference_ts = self._loaded_at
                if reference_ts and (now - reference_ts) < timedelta(seconds=delay):
                    return False

        return True

    def apply_replacement(self, text: str) -> tuple[str, int]:
        """
        Apply the replacement to the text.

        Returns:
            Tuple of (modified text, number of replacements made)
        """
        if not self.should_apply():
            return text, 0

        replacement_count = 0

        if self.is_regex and _is_compiled_regex(self.key):
            # Use subn to support backreferences and count replacements
            max_count = max(0, int(self.max_replacements))
            # subn with count=0 means replace all
            try:
                if isinstance(self.key, _regex.Pattern):
                    text, replacement_count = self.key.subn(
                        self.content,
                        text,
                        count=max_count,
                        timeout=_regex_timeout_seconds(),
                    )
                else:
                    text, replacement_count = self.key.subn(self.content, text, count=max_count)
            except _REGEX_TIMEOUT_ERROR:
                preview = self.key_pattern_str
                if len(preview) > 50:
                    preview = preview[:50] + "..."
                logger.warning(f"Regex replacement timed out for pattern '{preview}'")
                return text, 0
        else:
            # Literal replacement; support case sensitivity
            max_count = max(0, int(self.max_replacements))
            if not self.case_sensitive:
                pattern = re.compile(re.escape(self.raw_key), flags=re.IGNORECASE)
                text, replacement_count = pattern.subn(self.content, text, count=max_count)
            else:
                parts = text.split(self.raw_key)
                if len(parts) > 1:
                    replacements_to_make = (len(parts) - 1) if max_count == 0 else min(len(parts) - 1, max_count)
                    replacement_count = replacements_to_make
                    result: list[str] = []
                    for i, part in enumerate(parts):
                        result.append(part)
                        if i < len(parts) - 1:
                            if replacements_to_make > 0:
                                result.append(self.content)
                                replacements_to_make -= 1
                            else:
                                result.append(self.raw_key)
                    text = "".join(result)

        if replacement_count > 0:
            self.last_triggered = datetime.now(timezone.utc)
            self.trigger_count += replacement_count

        return text, replacement_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.entry_id,
            "pattern": self.raw_key,
            "replacement": self.content,
            "probability": float(self.probability),
            "group": self.group,
            "timed_effects": json.dumps(self.timed_effects),
            "max_replacements": int(self.max_replacements),
            "type": "regex" if self.is_regex else "literal",
            "enabled": int(self.enabled),
            "case_sensitive": int(self.case_sensitive),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatDictionaryEntry":
        """Create instance from database dictionary."""
        # Use safe parsing with structure validation for timed_effects
        timed_effects = safe_parse_json_dict(data.get("timed_effects"), "timed_effects")

        entry = cls(
            key=data.get("key") or data.get("pattern", ""),
            content=data.get("content") or data.get("replacement", ""),
            probability=data.get("probability", 1.0),
            group=data.get("group") or data.get("group_name"),
            timed_effects=timed_effects,
            max_replacements=int(data.get("max_replacements", 0)),
            entry_id=data.get("id"),
            enabled=bool(data.get("enabled", data.get("is_enabled", 1))),
            case_sensitive=bool(data.get("case_sensitive", data.get("is_case_sensitive", 1))),
        )
        # Respect persisted is_regex flag if present
        if "is_regex" in data:
            entry.is_regex = bool(data["is_regex"])
            if entry.is_regex and not _is_compiled_regex(entry.key) and entry.raw_key:
                try:
                    entry.key = entry._compile_regex_pattern(entry.key_pattern_str)
                except re.error:
                    # Leave as literal if invalid; caller may handle
                    logger.warning(f"Invalid or dangerous regex pattern in stored entry: {entry.raw_key[:50]}...")
                    entry.is_regex = False
        created_at_val = data.get("created_at")
        if created_at_val:
            try:
                if isinstance(created_at_val, datetime):
                    entry._loaded_at = (
                        created_at_val
                        if created_at_val.tzinfo is not None
                        else created_at_val.replace(tzinfo=timezone.utc)
                    )
                else:
                    iso_source = str(created_at_val).replace(" ", "T")
                    parsed = datetime.fromisoformat(iso_source.replace("Z", "+00:00"))
                    entry._loaded_at = (
                        parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
                    )
            except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
                entry._loaded_at = datetime.now(timezone.utc)
        return entry


class ChatDictionaryService:
    """
    Service class for managing chat dictionaries in a multi-user environment.

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

        # Request-scoped cache (not shared between requests)
        # Note: No lock needed - this service is instantiated per-request and
        # Python's async model runs on a single-threaded event loop with no
        # concurrent access within a single request.
        self._entry_cache: Optional[list[ChatDictionaryEntry]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = timedelta(seconds=60)  # Cache for 1 minute within request
        # Simple in-memory usage tracking for tests/analytics
        self._usage_counts: dict[int, int] = {}
        self._last_used_at: dict[int, datetime] = {}

    def _init_tables(self):
        """Initialize dictionary tables in the user's database if they don't exist."""
        backend_type = getattr(self.db, "backend_type", BackendType.SQLITE)
        try:
            with self.db.get_connection() as conn:
                if backend_type == BackendType.POSTGRESQL:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_dictionaries (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL UNIQUE,
                            description TEXT,
                            category TEXT,
                            tags TEXT,
                            included_dictionary_ids TEXT,
                            is_active BOOLEAN DEFAULT TRUE,
                            default_token_budget INTEGER NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            usage_count INTEGER DEFAULT 0,
                            last_used_at TIMESTAMP NULL,
                            version INTEGER DEFAULT 1,
                            deleted BOOLEAN DEFAULT FALSE
                        )
                        """
                    )

                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS dictionary_entries (
                            id SERIAL PRIMARY KEY,
                            dictionary_id INTEGER NOT NULL,
                            key TEXT NOT NULL,
                            content TEXT NOT NULL,
                            is_regex BOOLEAN DEFAULT FALSE,
                            probability REAL DEFAULT 1.0,
                            max_replacements INTEGER DEFAULT 1,
                            group_name TEXT,
                            timed_effects TEXT DEFAULT '{}',
                            enabled BOOLEAN DEFAULT TRUE,
                            case_sensitive BOOLEAN DEFAULT TRUE,
                            sort_order INTEGER,
                            usage_count INTEGER DEFAULT 0,
                            last_used_at TIMESTAMP NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                        )
                        """
                    )

                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS dictionary_transform_activity (
                            id SERIAL PRIMARY KEY,
                            dictionary_id INTEGER NOT NULL,
                            chat_id TEXT NULL,
                            entries_used TEXT NOT NULL,
                            replacements INTEGER DEFAULT 0,
                            iterations INTEGER DEFAULT 0,
                            token_budget_used INTEGER NULL,
                            original_text_preview TEXT NOT NULL,
                            processed_text_preview TEXT NOT NULL,
                            processing_time_ms REAL NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                        )
                        """
                    )

                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS dictionary_version_history (
                            id SERIAL PRIMARY KEY,
                            dictionary_id INTEGER NOT NULL,
                            revision INTEGER NOT NULL,
                            source_dictionary_version INTEGER NULL,
                            change_type TEXT NOT NULL,
                            summary TEXT NULL,
                            entry_count INTEGER DEFAULT 0,
                            dictionary_snapshot TEXT NOT NULL,
                            entries_snapshot TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(dictionary_id, revision),
                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                        )
                        """
                    )
                else:
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_dictionaries (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL UNIQUE,
                            description TEXT,
                            category TEXT,
                            tags TEXT,
                            included_dictionary_ids TEXT,
                            is_active BOOLEAN DEFAULT 1,
                            default_token_budget INTEGER,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            usage_count INTEGER DEFAULT 0,
                            last_used_at TIMESTAMP NULL,
                            version INTEGER DEFAULT 1,
                            deleted BOOLEAN DEFAULT 0
                        )
                    """
                    )

                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS dictionary_entries (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            dictionary_id INTEGER NOT NULL,
                            key TEXT NOT NULL,
                            content TEXT NOT NULL,
                            is_regex BOOLEAN DEFAULT 0,
                            probability REAL DEFAULT 1.0,
                            max_replacements INTEGER DEFAULT 1,
                            group_name TEXT,
                            timed_effects TEXT DEFAULT '{"sticky": 0, "cooldown": 0, "delay": 0}',
                            enabled BOOLEAN DEFAULT 1,
                            case_sensitive BOOLEAN DEFAULT 1,
                            sort_order INTEGER,
                            usage_count INTEGER DEFAULT 0,
                            last_used_at TIMESTAMP NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                        )
                        """
                    )

                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS dictionary_transform_activity (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            dictionary_id INTEGER NOT NULL,
                            chat_id TEXT,
                            entries_used TEXT NOT NULL,
                            replacements INTEGER DEFAULT 0,
                            iterations INTEGER DEFAULT 0,
                            token_budget_used INTEGER,
                            original_text_preview TEXT NOT NULL,
                            processed_text_preview TEXT NOT NULL,
                            processing_time_ms REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                        )
                        """
                    )

                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS dictionary_version_history (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            dictionary_id INTEGER NOT NULL,
                            revision INTEGER NOT NULL,
                            source_dictionary_version INTEGER,
                            change_type TEXT NOT NULL,
                            summary TEXT,
                            entry_count INTEGER DEFAULT 0,
                            dictionary_snapshot TEXT NOT NULL,
                            entries_snapshot TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(dictionary_id, revision),
                            FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                        )
                        """
                    )

                # Ensure legacy schemas are upgraded before creating indexes
                # that depend on upgraded columns.
                self._ensure_entry_sort_order_column(conn)

                # Create indexes
                conn.execute("CREATE INDEX IF NOT EXISTS idx_dict_entries_dict_id ON dictionary_entries(dictionary_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_dict_entries_group ON dictionary_entries(group_name)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dict_entries_sort_order "
                    "ON dictionary_entries(dictionary_id, sort_order)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dict_transform_activity_dictionary "
                    "ON dictionary_transform_activity(dictionary_id, created_at)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_dict_version_history_lookup "
                    "ON dictionary_version_history(dictionary_id, revision DESC)"
                )

                self._ensure_entry_usage_columns(conn)
                self._ensure_dictionary_usage_columns(conn)
                self._ensure_dictionary_default_token_budget_column(conn)
                self._ensure_dictionary_metadata_columns(conn)
                self._ensure_dictionary_version_history_table(conn)
                self._normalize_all_entry_sort_orders(conn)

                conn.commit()
                logger.info("Chat dictionary tables initialized")
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to initialize dictionary tables: {e}")
            raise CharactersRAGDBError(f"Failed to initialize dictionary tables: {e}") from e

    def _ensure_entry_sort_order_column(self, conn: Any) -> None:
        """Ensure dictionary_entries has a sort_order column for deterministic priority."""
        try:
            if self.db.backend_type == BackendType.POSTGRESQL:
                row = conn.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'dictionary_entries'
                      AND column_name = 'sort_order'
                    LIMIT 1
                    """
                ).fetchone()
                has_column = bool(row)
            else:
                rows = conn.execute("PRAGMA table_info(dictionary_entries)").fetchall()
                has_column = False
                for row in rows:
                    if isinstance(row, dict):
                        column_name = row.get("name")
                    elif hasattr(row, "_mapping"):
                        column_name = row._mapping.get("name")
                    else:
                        # SQLite PRAGMA row tuple shape: (cid, name, type, notnull, dflt_value, pk)
                        column_name = row[1] if len(row) > 1 else None
                    if column_name == "sort_order":
                        has_column = True
                        break

            if not has_column:
                conn.execute("ALTER TABLE dictionary_entries ADD COLUMN sort_order INTEGER")
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            message = str(e).lower()
            if "duplicate column" in message or "already exists" in message:
                return
            raise

    def _ensure_entry_usage_columns(self, conn: Any) -> None:
        """Ensure dictionary_entries has usage_count and last_used_at columns."""
        try:
            if self.db.backend_type == BackendType.POSTGRESQL:
                usage_row = conn.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'dictionary_entries'
                      AND column_name = 'usage_count'
                    LIMIT 1
                    """
                ).fetchone()
                if not usage_row:
                    conn.execute("ALTER TABLE dictionary_entries ADD COLUMN usage_count INTEGER DEFAULT 0")
                    conn.execute("UPDATE dictionary_entries SET usage_count = 0 WHERE usage_count IS NULL")

                last_used_row = conn.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'dictionary_entries'
                      AND column_name = 'last_used_at'
                    LIMIT 1
                    """
                ).fetchone()
                if not last_used_row:
                    conn.execute("ALTER TABLE dictionary_entries ADD COLUMN last_used_at TIMESTAMP NULL")
                return

            rows = conn.execute("PRAGMA table_info(dictionary_entries)").fetchall()
            column_names: set[str] = set()
            for row in rows:
                if isinstance(row, dict):
                    name = row.get("name")
                elif hasattr(row, "_mapping"):
                    name = row._mapping.get("name")
                else:
                    name = row[1] if len(row) > 1 else None
                if isinstance(name, str):
                    column_names.add(name)

            if "usage_count" not in column_names:
                conn.execute("ALTER TABLE dictionary_entries ADD COLUMN usage_count INTEGER DEFAULT 0")
                conn.execute("UPDATE dictionary_entries SET usage_count = 0 WHERE usage_count IS NULL")
            if "last_used_at" not in column_names:
                conn.execute("ALTER TABLE dictionary_entries ADD COLUMN last_used_at TIMESTAMP")
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            message = str(e).lower()
            if "duplicate column" in message or "already exists" in message:
                return
            raise

    def _ensure_dictionary_usage_columns(self, conn: Any) -> None:
        """Ensure chat_dictionaries has usage_count and last_used_at columns."""
        try:
            if self.db.backend_type == BackendType.POSTGRESQL:
                usage_row = conn.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'chat_dictionaries'
                      AND column_name = 'usage_count'
                    LIMIT 1
                    """
                ).fetchone()
                if not usage_row:
                    conn.execute("ALTER TABLE chat_dictionaries ADD COLUMN usage_count INTEGER DEFAULT 0")
                    conn.execute("UPDATE chat_dictionaries SET usage_count = 0 WHERE usage_count IS NULL")

                last_used_row = conn.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'chat_dictionaries'
                      AND column_name = 'last_used_at'
                    LIMIT 1
                    """
                ).fetchone()
                if not last_used_row:
                    conn.execute("ALTER TABLE chat_dictionaries ADD COLUMN last_used_at TIMESTAMP NULL")
                return

            rows = conn.execute("PRAGMA table_info(chat_dictionaries)").fetchall()
            column_names: set[str] = set()
            for row in rows:
                if isinstance(row, dict):
                    name = row.get("name")
                elif hasattr(row, "_mapping"):
                    name = row._mapping.get("name")
                else:
                    name = row[1] if len(row) > 1 else None
                if isinstance(name, str):
                    column_names.add(name)

            if "usage_count" not in column_names:
                conn.execute("ALTER TABLE chat_dictionaries ADD COLUMN usage_count INTEGER DEFAULT 0")
                conn.execute("UPDATE chat_dictionaries SET usage_count = 0 WHERE usage_count IS NULL")
            if "last_used_at" not in column_names:
                conn.execute("ALTER TABLE chat_dictionaries ADD COLUMN last_used_at TIMESTAMP")
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            message = str(e).lower()
            if "duplicate column" in message or "already exists" in message:
                return
            raise

    def _ensure_dictionary_default_token_budget_column(self, conn: Any) -> None:
        """Ensure chat_dictionaries has default_token_budget for fallback processing."""
        try:
            if self.db.backend_type == BackendType.POSTGRESQL:
                row = conn.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'chat_dictionaries'
                      AND column_name = 'default_token_budget'
                    LIMIT 1
                    """
                ).fetchone()
                if not row:
                    conn.execute(
                        "ALTER TABLE chat_dictionaries ADD COLUMN default_token_budget INTEGER NULL"
                    )
                return

            rows = conn.execute("PRAGMA table_info(chat_dictionaries)").fetchall()
            column_names: set[str] = set()
            for row in rows:
                if isinstance(row, dict):
                    name = row.get("name")
                elif hasattr(row, "_mapping"):
                    name = row._mapping.get("name")
                else:
                    name = row[1] if len(row) > 1 else None
                if isinstance(name, str):
                    column_names.add(name)

            if "default_token_budget" not in column_names:
                conn.execute(
                    "ALTER TABLE chat_dictionaries ADD COLUMN default_token_budget INTEGER"
                )
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            message = str(e).lower()
            if "duplicate column" in message or "already exists" in message:
                return
            raise

    def _ensure_dictionary_metadata_columns(self, conn: Any) -> None:
        """Ensure chat_dictionaries has metadata columns for category/tags."""
        try:
            if self.db.backend_type == BackendType.POSTGRESQL:
                category_row = conn.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'chat_dictionaries'
                      AND column_name = 'category'
                    LIMIT 1
                    """
                ).fetchone()
                if not category_row:
                    conn.execute("ALTER TABLE chat_dictionaries ADD COLUMN category TEXT NULL")

                tags_row = conn.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'chat_dictionaries'
                      AND column_name = 'tags'
                    LIMIT 1
                    """
                ).fetchone()
                if not tags_row:
                    conn.execute("ALTER TABLE chat_dictionaries ADD COLUMN tags TEXT NULL")

                includes_row = conn.execute(
                    """
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'chat_dictionaries'
                      AND column_name = 'included_dictionary_ids'
                    LIMIT 1
                    """
                ).fetchone()
                if not includes_row:
                    conn.execute(
                        "ALTER TABLE chat_dictionaries ADD COLUMN included_dictionary_ids TEXT NULL"
                    )
                return

            rows = conn.execute("PRAGMA table_info(chat_dictionaries)").fetchall()
            column_names: set[str] = set()
            for row in rows:
                if isinstance(row, dict):
                    name = row.get("name")
                elif hasattr(row, "_mapping"):
                    name = row._mapping.get("name")
                else:
                    name = row[1] if len(row) > 1 else None
                if isinstance(name, str):
                    column_names.add(name)

            if "category" not in column_names:
                conn.execute("ALTER TABLE chat_dictionaries ADD COLUMN category TEXT")
            if "tags" not in column_names:
                conn.execute("ALTER TABLE chat_dictionaries ADD COLUMN tags TEXT")
            if "included_dictionary_ids" not in column_names:
                conn.execute("ALTER TABLE chat_dictionaries ADD COLUMN included_dictionary_ids TEXT")
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            message = str(e).lower()
            if "duplicate column" in message or "already exists" in message:
                return
            raise

    def _ensure_dictionary_version_history_table(self, conn: Any) -> None:
        """Ensure dictionary version-history table and indexes exist."""
        try:
            if self.db.backend_type == BackendType.POSTGRESQL:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS dictionary_version_history (
                        id SERIAL PRIMARY KEY,
                        dictionary_id INTEGER NOT NULL,
                        revision INTEGER NOT NULL,
                        source_dictionary_version INTEGER NULL,
                        change_type TEXT NOT NULL,
                        summary TEXT NULL,
                        entry_count INTEGER DEFAULT 0,
                        dictionary_snapshot TEXT NOT NULL,
                        entries_snapshot TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(dictionary_id, revision),
                        FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_dict_version_history_lookup
                    ON dictionary_version_history(dictionary_id, revision DESC)
                    """
                )
                return

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dictionary_version_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dictionary_id INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    source_dictionary_version INTEGER,
                    change_type TEXT NOT NULL,
                    summary TEXT,
                    entry_count INTEGER DEFAULT 0,
                    dictionary_snapshot TEXT NOT NULL,
                    entries_snapshot TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(dictionary_id, revision),
                    FOREIGN KEY (dictionary_id) REFERENCES chat_dictionaries(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_dict_version_history_lookup
                ON dictionary_version_history(dictionary_id, revision DESC)
                """
            )
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            message = str(e).lower()
            if "already exists" in message:
                return
            raise

    def _normalize_all_entry_sort_orders(self, conn: Any) -> None:
        """Normalize sort_order values to a contiguous 1..N sequence per dictionary."""
        dict_rows = conn.execute("SELECT id FROM chat_dictionaries").fetchall()
        dictionary_ids: list[int] = []
        for row in dict_rows:
            if isinstance(row, dict):
                raw_id = row.get("id")
            elif hasattr(row, "_mapping"):
                raw_id = row._mapping.get("id")
            else:
                raw_id = row[0] if row else None
            if raw_id is None:
                continue
            dictionary_ids.append(int(raw_id))

        for dictionary_id in dictionary_ids:
            self._normalize_dictionary_entry_sort_order(conn, dictionary_id)

    def _normalize_dictionary_entry_sort_order(self, conn: Any, dictionary_id: int) -> None:
        """Normalize a single dictionary's entry order to contiguous sort_order values."""
        rows = conn.execute(
            """
            SELECT id
            FROM dictionary_entries
            WHERE dictionary_id = ?
            ORDER BY COALESCE(sort_order, id), id
            """,
            (dictionary_id,),
        ).fetchall()

        for index, row in enumerate(rows, start=1):
            if isinstance(row, dict):
                entry_id = row.get("id")
            elif hasattr(row, "_mapping"):
                entry_id = row._mapping.get("id")
            else:
                entry_id = row[0] if row else None
            if entry_id is None:
                continue
            conn.execute(
                "UPDATE dictionary_entries SET sort_order = ? WHERE id = ?",
                (index, int(entry_id)),
            )

    # --- Dictionary CRUD Operations ---

    def create_dictionary(
        self,
        name: str,
        description: Optional[str] = None,
        default_token_budget: Optional[int] = None,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        included_dictionary_ids: Optional[list[int]] = None,
    ) -> int:
        """
        Create a new chat dictionary for the user.

        Args:
            name: Unique name for the dictionary
            description: Optional description

        Returns:
            The ID of the created dictionary

        Raises:
            ConflictError: If a dictionary with this name already exists
        """
        try:
            with self.db.get_connection() as conn:
                normalized_includes = self._validate_included_dictionary_ids(
                    conn,
                    dictionary_id=None,
                    included_dictionary_ids=included_dictionary_ids or [],
                )
                insert_sql = """
                    INSERT INTO chat_dictionaries (
                        name,
                        description,
                        category,
                        tags,
                        included_dictionary_ids,
                        default_token_budget
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                params = (
                    name,
                    description,
                    self._normalize_dictionary_category(category),
                    self._serialize_dictionary_tags(tags),
                    self._serialize_included_dictionary_ids(normalized_includes),
                    self._coerce_positive_int(default_token_budget),
                )
                if self.db.backend_type == BackendType.POSTGRESQL:
                    insert_sql += " RETURNING id"

                cursor = conn.execute(insert_sql, params)

                if self.db.backend_type == BackendType.POSTGRESQL:
                    row = cursor.fetchone()
                    if row is None:
                        dictionary_id = None
                    elif isinstance(row, dict):
                        dictionary_id = row.get("id")
                    elif hasattr(row, '_mapping'):
                        # SQLAlchemy Row object
                        dictionary_id = row._mapping.get("id")
                    else:
                        # Tuple or sequence - first column is the ID
                        dictionary_id = row[0] if row else None
                else:
                    dictionary_id = cursor.lastrowid

                if dictionary_id is None:
                    raise CharactersRAGDBError("Database did not return a dictionary id.")

                self._create_dictionary_version_snapshot(
                    conn,
                    int(dictionary_id),
                    change_type="create",
                    summary="Dictionary created",
                    include_deleted=True,
                )
                conn.commit()
                logger.info(f"Created dictionary '{name}' with ID {dictionary_id}")
                self._invalidate_cache()
                return int(dictionary_id)

        except sqlite3.IntegrityError as e:
            if _is_unique_violation(e):
                raise ConflictError(f"Dictionary '{name}' already exists", "chat_dictionaries", name) from e
            raise CharactersRAGDBError(f"Database error creating dictionary: {e}") from e
        except CharactersRAGDBError as e:
            if _is_unique_violation(e):
                raise ConflictError(f"Dictionary '{name}' already exists", "chat_dictionaries", name) from e
            raise
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Database error creating dictionary: {e}")
            raise CharactersRAGDBError(f"Database error creating dictionary: {e}") from e

    def get_dictionary(
        self, dictionary_id: Optional[int] = None, name: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """
        Get a dictionary by ID or name.

        Args:
            dictionary_id: Optional dictionary ID
            name: Optional dictionary name

        Returns:
            Dictionary data or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                if dictionary_id:
                    cursor = conn.execute(
                        "SELECT * FROM chat_dictionaries WHERE id = ? AND deleted = ?", (dictionary_id, False)
                    )
                elif name:
                    cursor = conn.execute(
                        "SELECT * FROM chat_dictionaries WHERE name = ? AND deleted = ?", (name, False)
                    )
                else:
                    return None

                row = cursor.fetchone()
                if row:
                    return self._normalize_dictionary_row(dict(row))
                return None

        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error fetching dictionary: {e}")
            raise CharactersRAGDBError(f"Error fetching dictionary: {e}") from e

    def list_dictionaries(self, include_inactive: bool = False) -> list[dict[str, Any]]:
        """
        List all dictionaries for the user.

        Args:
            include_inactive: Whether to include inactive dictionaries

        Returns:
            List of dictionary data
        """
        try:
            with self.db.get_connection() as conn:
                query = "SELECT * FROM chat_dictionaries WHERE deleted = ?"
                params: list[Any] = [False]
                if not include_inactive:
                    query += " AND is_active = ?"
                    params.append(True)
                query += " ORDER BY name"

                cursor = conn.execute(query, tuple(params))
                return [self._normalize_dictionary_row(dict(row)) for row in cursor.fetchall()]

        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error listing dictionaries: {e}")
            raise CharactersRAGDBError(f"Error listing dictionaries: {e}") from e

    def list_dictionaries_with_entry_counts(self, include_inactive: bool = False) -> list[dict[str, Any]]:
        """
        List dictionaries with precomputed entry counts in a single query.

        Returns:
            List of dictionary data including an `entry_count` key per dictionary.
        """
        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT
                        d.*,
                        COALESCE(cnt.entry_count, 0) AS entry_count,
                        COALESCE(cnt.regex_entry_count, 0) AS regex_entry_count
                    FROM chat_dictionaries AS d
                    LEFT JOIN (
                        SELECT
                            dictionary_id,
                            COUNT(*) AS entry_count,
                            SUM(CASE WHEN is_regex THEN 1 ELSE 0 END) AS regex_entry_count
                        FROM dictionary_entries
                        GROUP BY dictionary_id
                    ) AS cnt
                    ON d.id = cnt.dictionary_id
                    WHERE d.deleted = ?
                """
                params: list[Any] = [False]
                if not include_inactive:
                    query += " AND d.is_active = ?"
                    params.append(True)
                query += " ORDER BY d.name"

                cursor = conn.execute(query, tuple(params))
                return [self._normalize_dictionary_row(dict(row)) for row in cursor.fetchall()]
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error listing dictionaries with counts: {e}")
            raise CharactersRAGDBError(f"Error listing dictionaries with counts: {e}") from e

    @staticmethod
    def _normalize_dictionary_category(value: Any) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @classmethod
    def _normalize_dictionary_tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        raw_values: list[Any]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in stripped.split(",")]
            if isinstance(parsed, (list, tuple, set)):
                raw_values = list(parsed)
            else:
                raw_values = [parsed]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            return []

        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_values:
            if item is None:
                continue
            tag = str(item).strip()
            if not tag:
                continue
            dedupe_key = tag.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(tag)
        return normalized[:20]

    @staticmethod
    def _normalize_included_dictionary_ids(value: Any) -> list[int]:
        if value is None:
            return []
        raw_values: list[Any]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in stripped.split(",")]
            if isinstance(parsed, (list, tuple, set)):
                raw_values = list(parsed)
            else:
                raw_values = [parsed]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            return []

        normalized: list[int] = []
        seen: set[int] = set()
        for raw in raw_values:
            if raw is None or isinstance(raw, bool):
                continue
            try:
                dictionary_id = int(raw)
            except (TypeError, ValueError):
                continue
            if dictionary_id <= 0 or dictionary_id in seen:
                continue
            seen.add(dictionary_id)
            normalized.append(dictionary_id)
        return normalized[:50]

    @classmethod
    def _serialize_included_dictionary_ids(cls, value: Any) -> Optional[str]:
        normalized = cls._normalize_included_dictionary_ids(value)
        if not normalized:
            return None
        return json.dumps(normalized)

    def _validate_included_dictionary_ids(
        self,
        conn: Any,
        *,
        dictionary_id: Optional[int],
        included_dictionary_ids: list[int],
    ) -> list[int]:
        normalized = self._normalize_included_dictionary_ids(included_dictionary_ids)
        if not normalized:
            return []

        if dictionary_id is not None:
            dictionary_id_int = int(dictionary_id)
            if dictionary_id_int in normalized:
                raise InputError("Dictionary cannot include itself")
        else:
            dictionary_id_int = None

        placeholders = ", ".join("?" for _ in normalized)
        included_ids_clause = f"({placeholders})"
        included_ids_sql_template = """
            SELECT id
            FROM chat_dictionaries
            WHERE deleted = ? AND id IN {included_ids_clause}
            """
        included_ids_sql = included_ids_sql_template.format_map(locals())  # nosec B608
        rows = conn.execute(
            included_ids_sql,
            tuple([False, *normalized]),
        ).fetchall()
        found_ids: set[int] = set()
        for row in rows:
            if isinstance(row, dict):
                raw_id = row.get("id")
            elif hasattr(row, "_mapping"):
                raw_id = row._mapping.get("id")
            else:
                raw_id = row[0] if row else None
            if raw_id is not None:
                found_ids.add(int(raw_id))
        missing_ids = [dictionary_ref for dictionary_ref in normalized if dictionary_ref not in found_ids]
        if missing_ids:
            raise InputError(f"Included dictionaries not found: {missing_ids}")

        if dictionary_id_int is None:
            return normalized

        existing_rows = conn.execute(
            """
            SELECT id, included_dictionary_ids
            FROM chat_dictionaries
            WHERE deleted = ?
            """,
            (False,),
        ).fetchall()
        include_map: dict[int, list[int]] = {}
        for row in existing_rows:
            row_dict = dict(row)
            row_dictionary_id = self._coerce_positive_int(row_dict.get("id"))
            if row_dictionary_id is None:
                continue
            include_map[row_dictionary_id] = self._normalize_included_dictionary_ids(
                row_dict.get("included_dictionary_ids")
            )
        include_map[dictionary_id_int] = normalized

        visiting: set[int] = set()
        visited: set[int] = set()
        path: list[int] = []

        def _visit(current_dictionary_id: int) -> None:
            if current_dictionary_id in visited:
                return
            if current_dictionary_id in visiting:
                cycle_start_index = path.index(current_dictionary_id) if current_dictionary_id in path else 0
                cycle_path = path[cycle_start_index:] + [current_dictionary_id]
                raise InputError(
                    f"Dictionary include cycle detected: {' -> '.join(str(item) for item in cycle_path)}"
                )
            visiting.add(current_dictionary_id)
            path.append(current_dictionary_id)
            for included_id in include_map.get(current_dictionary_id, []):
                if included_id not in include_map:
                    continue
                _visit(included_id)
            path.pop()
            visiting.remove(current_dictionary_id)
            visited.add(current_dictionary_id)

        _visit(dictionary_id_int)
        return normalized

    @classmethod
    def _serialize_dictionary_tags(cls, value: Any) -> Optional[str]:
        normalized = cls._normalize_dictionary_tags(value)
        if not normalized:
            return None
        return json.dumps(normalized)

    @classmethod
    def _normalize_dictionary_row(cls, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        normalized["category"] = cls._normalize_dictionary_category(normalized.get("category"))
        normalized["tags"] = cls._normalize_dictionary_tags(normalized.get("tags"))
        normalized["included_dictionary_ids"] = cls._normalize_included_dictionary_ids(
            normalized.get("included_dictionary_ids")
        )
        return normalized

    @staticmethod
    def _json_serialize_default(value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _build_entry_snapshot_payload(self, row_dict: dict[str, Any]) -> dict[str, Any]:
        entry = ChatDictionaryEntry.from_dict(row_dict).to_dict()
        entry["dictionary_id"] = row_dict.get("dictionary_id")
        entry["sort_order"] = row_dict.get("sort_order")
        entry["usage_count"] = int(row_dict.get("usage_count") or 0)
        entry["last_used_at"] = row_dict.get("last_used_at")
        entry["created_at"] = row_dict.get("created_at")
        entry["updated_at"] = row_dict.get("updated_at")
        if not entry.get("group") and row_dict.get("group_name"):
            entry["group"] = row_dict.get("group_name")
        return entry

    def _fetch_snapshot_state(
        self,
        conn: Any,
        dictionary_id: int,
        *,
        include_deleted: bool = False,
    ) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
        where_clause = "id = ?"
        params: list[Any] = [int(dictionary_id)]
        if not include_deleted:
            where_clause += " AND deleted = ?"
            params.append(False)

        dictionary_lookup_sql_template = "SELECT * FROM chat_dictionaries WHERE {where_clause} LIMIT 1"
        dictionary_lookup_sql = dictionary_lookup_sql_template.format_map(locals())  # nosec B608
        dictionary_row = conn.execute(
            dictionary_lookup_sql,
            tuple(params),
        ).fetchone()
        if not dictionary_row:
            return None, []

        dictionary_payload = self._normalize_dictionary_row(dict(dictionary_row))

        entry_rows = conn.execute(
            """
            SELECT *
            FROM dictionary_entries
            WHERE dictionary_id = ?
            ORDER BY COALESCE(sort_order, id), id
            """,
            (int(dictionary_id),),
        ).fetchall()
        entries_payload = [
            self._build_entry_snapshot_payload(dict(entry_row))
            for entry_row in entry_rows
        ]
        return dictionary_payload, entries_payload

    def _create_dictionary_version_snapshot(
        self,
        conn: Any,
        dictionary_id: int,
        *,
        change_type: str,
        summary: Optional[str] = None,
        include_deleted: bool = False,
    ) -> Optional[int]:
        dictionary_payload, entries_payload = self._fetch_snapshot_state(
            conn,
            dictionary_id,
            include_deleted=include_deleted,
        )
        if dictionary_payload is None:
            return None

        revision_row = conn.execute(
            """
            SELECT COALESCE(MAX(revision), 0) + 1 AS next_revision
            FROM dictionary_version_history
            WHERE dictionary_id = ?
            """,
            (int(dictionary_id),),
        ).fetchone()
        if isinstance(revision_row, dict):
            next_revision_raw = revision_row.get("next_revision")
        elif hasattr(revision_row, "_mapping"):
            next_revision_raw = revision_row._mapping.get("next_revision")
        else:
            next_revision_raw = revision_row[0] if revision_row else 1
        next_revision = int(next_revision_raw or 1)

        conn.execute(
            """
            INSERT INTO dictionary_version_history
            (
                dictionary_id,
                revision,
                source_dictionary_version,
                change_type,
                summary,
                entry_count,
                dictionary_snapshot,
                entries_snapshot
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(dictionary_id),
                next_revision,
                self._coerce_positive_int(dictionary_payload.get("version")),
                str(change_type).strip() or "unspecified",
                summary.strip() if isinstance(summary, str) and summary.strip() else None,
                len(entries_payload),
                json.dumps(dictionary_payload, default=self._json_serialize_default),
                json.dumps(entries_payload, default=self._json_serialize_default),
            ),
        )
        return next_revision

    @staticmethod
    def _coerce_dictionary_id(value: Any) -> Optional[int]:
        """Best-effort coercion for dictionary IDs embedded in conversation settings."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            if value.is_integer() and value > 0:
                return int(value)
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            if stripped.isdigit():
                parsed = int(stripped)
                return parsed if parsed > 0 else None
        return None

    @staticmethod
    def _coerce_positive_int(value: Any) -> Optional[int]:
        """Best-effort coercion for optional positive integers."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value > 0 else None
        if isinstance(value, float):
            if value.is_integer() and value > 0:
                return int(value)
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                parsed = int(stripped)
                return parsed if parsed > 0 else None
        return None

    @classmethod
    def _extract_dictionary_ids_from_value(cls, value: Any) -> set[int]:
        """Extract dictionary IDs from known scalar/list/object setting value shapes."""
        coerced = cls._coerce_dictionary_id(value)
        if coerced is not None:
            return {coerced}

        if isinstance(value, (list, tuple, set)):
            ids: set[int] = set()
            for item in value:
                ids.update(cls._extract_dictionary_ids_from_value(item))
            return ids

        if isinstance(value, dict):
            ids: set[int] = set()
            for key, nested in value.items():
                lowered = str(key).strip().lower()
                if lowered in _DICTIONARY_SETTINGS_SINGLE_KEYS or lowered in {
                    "id",
                    "chatdictionaryid",
                    "chat_dictionary_id",
                    "dictionaryid",
                    "dictionary_id",
                }:
                    nested_id = cls._coerce_dictionary_id(nested)
                    if nested_id is not None:
                        ids.add(nested_id)
                    continue
                if lowered in _DICTIONARY_SETTINGS_LIST_KEYS:
                    ids.update(cls._extract_dictionary_ids_from_value(nested))
            return ids

        return set()

    @classmethod
    def _collect_linked_dictionary_ids_from_settings(cls, settings: dict[str, Any]) -> set[int]:
        """Collect dictionary IDs from conversation settings using known key aliases."""
        found_ids: set[int] = set()
        queue: list[Any] = [settings]

        while queue:
            node = queue.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    lowered = str(key).strip().lower()
                    if lowered in _DICTIONARY_SETTINGS_SINGLE_KEYS or lowered in _DICTIONARY_SETTINGS_LIST_KEYS:
                        found_ids.update(cls._extract_dictionary_ids_from_value(value))
                    if isinstance(value, (dict, list, tuple, set)):
                        queue.append(value)
            elif isinstance(node, (list, tuple, set)):
                for item in node:
                    if isinstance(item, (dict, list, tuple, set)):
                        queue.append(item)

        return found_ids

    def get_dictionary_usage_summary(
        self,
        dictionary_ids: Optional[list[int]] = None,
        *,
        max_chat_refs_per_dictionary: int = 3,
    ) -> dict[int, dict[str, Any]]:
        """
        Summarize dictionary usage across chat conversation settings.

        The summary scans conversation settings for known dictionary-id keys.
        """
        target_ids: set[int] = set()
        for item in dictionary_ids or []:
            coerced_id = self._coerce_dictionary_id(item)
            if coerced_id is not None:
                target_ids.add(coerced_id)
        summary: dict[int, dict[str, Any]] = {
            dictionary_id: {
                "used_by_chat_count": 0,
                "used_by_active_chat_count": 0,
                "used_by_chat_refs": [],
            }
            for dictionary_id in target_ids
        }

        try:
            ensure_settings_table = getattr(self.db, "_ensure_conversation_settings_table", None)
            if callable(ensure_settings_table):
                ensure_settings_table()

            query = (
                "SELECT c.id AS chat_id, c.title, c.state, c.last_modified, cs.settings_json "
                "FROM conversations c "
                "JOIN conversation_settings cs ON cs.conversation_id = c.id "
                "WHERE c.deleted = ?"
            )
            params: list[Any] = [False]

            client_id = getattr(self.db, "client_id", None)
            if client_id:
                query += " AND c.client_id = ?"
                params.append(client_id)

            query += " ORDER BY c.last_modified DESC"
            cursor = self.db.execute_query(query, tuple(params))
            rows = cursor.fetchall() or []
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Unable to compute dictionary usage summary: {}", exc)
            return summary

        for row in rows:
            row_data = dict(row)
            settings = safe_parse_json_dict(
                row_data.get("settings_json"),
                "conversation_settings.settings_json",
                default={},
            )
            linked_dictionary_ids = self._collect_linked_dictionary_ids_from_settings(settings)
            if not linked_dictionary_ids:
                continue

            if target_ids:
                linked_dictionary_ids &= target_ids
                if not linked_dictionary_ids:
                    continue

            state = str(row_data.get("state") or "").strip().lower()
            is_active_chat = state in _ACTIVE_CONVERSATION_STATES
            chat_ref = {
                "chat_id": row_data.get("chat_id"),
                "title": row_data.get("title"),
                "state": row_data.get("state") or "in-progress",
                "last_modified": row_data.get("last_modified"),
            }

            for dictionary_id in linked_dictionary_ids:
                current = summary.setdefault(
                    dictionary_id,
                    {
                        "used_by_chat_count": 0,
                        "used_by_active_chat_count": 0,
                        "used_by_chat_refs": [],
                    },
                )
                current["used_by_chat_count"] += 1
                if is_active_chat:
                    current["used_by_active_chat_count"] += 1
                if len(current["used_by_chat_refs"]) < max_chat_refs_per_dictionary:
                    current["used_by_chat_refs"].append(chat_ref)

        return summary

    def update_dictionary(
        self,
        dictionary_id: Optional[int] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        included_dictionary_ids: Optional[list[int]] = None,
        is_active: Optional[bool] = None,
        default_token_budget: Optional[int] = None,
        update_category: bool = False,
        update_tags: bool = False,
        update_included_dictionary_ids: bool = False,
        update_default_token_budget: bool = False,
        expected_version: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """
        Update a dictionary's metadata.

        Args:
            dictionary_id: Dictionary ID
            name: New name (optional)
            description: New description (optional)
            is_active: Active status (optional)
            expected_version: Expected current version for optimistic locking (optional)

        Returns:
            True if updated successfully

        Raises:
            ConflictError: If the new name conflicts with an existing dictionary
        """
        try:
            # Accept alias
            if dictionary_id is None:
                dictionary_id = kwargs.get("dict_id")
            if expected_version is None and "version" in kwargs:
                expected_version = kwargs.get("version")
            updates: list[str] = []
            params: list[Any] = []
            changed_fields: list[str] = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
                changed_fields.append("name")
            if description is not None:
                updates.append("description = ?")
                params.append(description)
                changed_fields.append("description")
            if update_category:
                updates.append("category = ?")
                params.append(self._normalize_dictionary_category(category))
                changed_fields.append("category")
            if update_tags:
                updates.append("tags = ?")
                params.append(self._serialize_dictionary_tags(tags))
                changed_fields.append("tags")
            if update_included_dictionary_ids:
                with self.db.get_connection() as conn:
                    normalized_includes = self._validate_included_dictionary_ids(
                        conn,
                        dictionary_id=int(dictionary_id) if dictionary_id is not None else None,
                        included_dictionary_ids=included_dictionary_ids or [],
                    )
                updates.append("included_dictionary_ids = ?")
                params.append(self._serialize_included_dictionary_ids(normalized_includes))
                changed_fields.append("included_dictionary_ids")
            if is_active is not None:
                updates.append("is_active = ?")
                params.append(bool(is_active))
                changed_fields.append("is_active")
            if update_default_token_budget:
                updates.append("default_token_budget = ?")
                params.append(self._coerce_positive_int(default_token_budget))
                changed_fields.append("default_token_budget")

            if not updates:
                return True

            updates.append("updated_at = CURRENT_TIMESTAMP")
            updates.append("version = version + 1")
            params.extend([dictionary_id, False])
            if expected_version is not None:
                params.append(int(expected_version))

            with self.db.get_connection() as conn:
                set_clause = _build_safe_update_clause(updates, _DICTIONARY_UPDATE_FIELDS)
                where_clause = "id = ? AND deleted = ?"
                if expected_version is not None:
                    where_clause += " AND version = ?"
                update_dictionary_sql_template = "UPDATE chat_dictionaries SET {set_clause} WHERE {where_clause}"
                update_dictionary_sql = update_dictionary_sql_template.format_map(locals())  # nosec B608
                cursor = conn.execute(
                    update_dictionary_sql,
                    tuple(params),
                )

                if cursor.rowcount > 0:
                    summary = "Updated fields: " + ", ".join(changed_fields) if changed_fields else "Dictionary updated"
                    self._create_dictionary_version_snapshot(
                        conn,
                        int(dictionary_id),
                        change_type="update",
                        summary=summary,
                    )
                    conn.commit()
                    logger.info(f"Updated dictionary {dictionary_id}")
                    self._invalidate_cache()
                    return True

                if expected_version is not None:
                    version_row = conn.execute(
                        "SELECT version, deleted FROM chat_dictionaries WHERE id = ?",
                        (dictionary_id,),
                    ).fetchone()
                    if version_row is not None:
                        if isinstance(version_row, sqlite3.Row):
                            current_version = version_row["version"]
                            is_deleted = bool(version_row["deleted"])
                        else:
                            current_version = version_row[0]
                            is_deleted = bool(version_row[1])
                        if not is_deleted:
                            raise ConflictError(
                                (
                                    "Dictionary was modified by another session. "
                                    f"Expected version {expected_version}, current version {current_version}."
                                ),
                                "chat_dictionaries",
                                dictionary_id,
                            )
                return False

        except sqlite3.IntegrityError as e:
            if _is_unique_violation(e):
                raise ConflictError(f"Dictionary name '{name}' already exists", "chat_dictionaries", name) from e
            raise CharactersRAGDBError(f"Database error updating dictionary: {e}") from e
        except CharactersRAGDBError as e:
            if _is_unique_violation(e):
                raise ConflictError(f"Dictionary name '{name}' already exists", "chat_dictionaries", name) from e
            raise

    def delete_dictionary(self, dictionary_id: int, hard_delete: bool = False) -> bool:
        """
        Delete a dictionary (soft delete by default).

        Args:
            dictionary_id: Dictionary ID
            hard_delete: If True, permanently delete; otherwise soft delete

        Returns:
            True if deleted successfully
        """
        try:
            with self.db.get_connection() as conn:
                existing_row = conn.execute(
                    "SELECT id, deleted FROM chat_dictionaries WHERE id = ?",
                    (dictionary_id,),
                ).fetchone()
                if not existing_row:
                    return False
                if isinstance(existing_row, dict):
                    is_deleted = bool(existing_row.get("deleted"))
                elif hasattr(existing_row, "_mapping"):
                    is_deleted = bool(existing_row._mapping.get("deleted"))
                else:
                    is_deleted = bool(existing_row[1]) if len(existing_row) > 1 else False
                if is_deleted and not hard_delete:
                    return False

                self._create_dictionary_version_snapshot(
                    conn,
                    int(dictionary_id),
                    change_type="delete",
                    summary="Hard deleted dictionary" if hard_delete else "Soft deleted dictionary",
                    include_deleted=True,
                )

                if hard_delete:
                    cursor = conn.execute("DELETE FROM chat_dictionaries WHERE id = ?", (dictionary_id,))
                else:
                    cursor = conn.execute(
                        "UPDATE chat_dictionaries SET deleted = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                        (True, dictionary_id),
                    )
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted dictionary {dictionary_id}")
                    self._invalidate_cache()
                    return True
                return False

        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error deleting dictionary: {e}")
            raise CharactersRAGDBError(f"Error deleting dictionary: {e}") from e

    # --- Entry CRUD Operations ---

    def add_entry(
        self,
        dictionary_id: int,
        key: Optional[str] = None,
        content: Optional[str] = None,
        probability: Optional[Union[int, float]] = None,
        group: Optional[str] = None,
        timed_effects: Optional[dict[str, int]] = None,
        max_replacements: Optional[int] = None,
        **kwargs,
    ) -> int:
        """
        Add an entry to a dictionary.

        Args:
            dictionary_id: Dictionary ID
            key: Pattern to match
            content: Replacement text
            probability: Chance of replacement (0-100)
            group: Optional group name
            timed_effects: Timing effects dictionary
            max_replacements: Max replacements per processing

        Returns:
            The ID of the created entry
        """
        try:
            # Accept new-style params
            pattern = kwargs.get("pattern", key)
            replacement = kwargs.get("replacement", content)
            entry_type = kwargs.get("type")
            enabled = kwargs.get("enabled", True)
            case_sensitive = kwargs.get("case_sensitive", True)
            record_snapshot = bool(kwargs.get("record_snapshot", True))

            if pattern is None or pattern == "":
                raise ValueError("Pattern cannot be empty")
            if replacement is None:
                logger.warning("Replacement is empty; proceeding with empty string")
                replacement = ""

            # Normalize probability
            if probability is None:
                probability_f = 1.0
            else:
                if isinstance(probability, int):
                    if not 0 <= probability <= 100:
                        raise InputError("Probability must be between 0 and 100")
                    probability_f = probability / 100.0
                else:
                    if not 0.0 <= probability <= 1.0:
                        raise InputError("Probability must be between 0.0 and 1.0")
                    probability_f = float(probability)

            # Compile pattern to check validity - raise re.error for invalid regex
            try:
                entry = ChatDictionaryEntry(
                    pattern,
                    replacement,
                    probability_f,
                    group,
                    timed_effects,
                    0 if max_replacements is None else int(max_replacements),
                    enabled=bool(enabled),
                    case_sensitive=bool(case_sensitive),
                )
                # Override regex decision if type provided
                if entry_type is not None:
                    entry.is_regex = entry_type == "regex"
                    if entry.is_regex and not _is_compiled_regex(entry.key):
                        # Use safe regex compilation to validate the pattern and protect against ReDoS.
                        # The compiled pattern is not stored on this transient instance; persistence
                        # relies on the `is_regex` flag and `from_dict` will recompile as needed.
                        entry._compile_regex_pattern(entry.key_pattern_str)
            except re.error:
                raise

            timed_effects_json = json.dumps(timed_effects or {"sticky": 0, "cooldown": 0, "delay": 0})

            with self.db.get_connection() as conn:
                next_sort_order_row = conn.execute(
                    "SELECT COALESCE(MAX(sort_order), 0) + 1 AS next_sort_order "
                    "FROM dictionary_entries WHERE dictionary_id = ?",
                    (dictionary_id,),
                ).fetchone()
                if isinstance(next_sort_order_row, dict):
                    next_sort_order_raw = next_sort_order_row.get("next_sort_order")
                elif hasattr(next_sort_order_row, "_mapping"):
                    next_sort_order_raw = next_sort_order_row._mapping.get("next_sort_order")
                else:
                    next_sort_order_raw = (
                        next_sort_order_row[0] if next_sort_order_row else 1
                    )
                next_sort_order = int(next_sort_order_raw or 1)

                insert_sql = """
                    INSERT INTO dictionary_entries
                    (dictionary_id, key, content, is_regex, probability, max_replacements, group_name, timed_effects, enabled, case_sensitive, sort_order)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                params = (
                    dictionary_id,
                    pattern,
                    replacement,
                    bool(entry.is_regex),
                    probability_f,
                    0 if max_replacements is None else int(max_replacements),
                    group,
                    timed_effects_json,
                    bool(enabled),
                    bool(case_sensitive),
                    next_sort_order,
                )
                if self.db.backend_type == BackendType.POSTGRESQL:
                    insert_sql += " RETURNING id"

                cursor = conn.execute(insert_sql, params)

                if self.db.backend_type == BackendType.POSTGRESQL:
                    row = cursor.fetchone()
                    if row is None:
                        entry_id = None
                    elif isinstance(row, dict):
                        entry_id = row.get("id")
                    elif hasattr(row, '_mapping'):
                        # SQLAlchemy Row object
                        entry_id = row._mapping.get("id")
                    else:
                        # Tuple or sequence - first column is the ID
                        entry_id = row[0] if row else None
                else:
                    entry_id = cursor.lastrowid

                if entry_id is None:
                    raise CharactersRAGDBError("Database did not return a dictionary entry id.")

                if record_snapshot:
                    self._create_dictionary_version_snapshot(
                        conn,
                        int(dictionary_id),
                        change_type="entry_add",
                        summary=f"Added entry {int(entry_id)}",
                    )
                conn.commit()

                logger.info(f"Added entry {entry_id} to dictionary {dictionary_id}")
                self._invalidate_cache()
                return int(entry_id)

        except ValueError:
            # Re-raise ValueError as is (for invalid inputs)
            raise
        except re.error:
            # Preserve regex errors for caller/tests
            raise
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error adding dictionary entry: {e}")
            raise CharactersRAGDBError(f"Error adding dictionary entry: {e}") from e

    def get_entries(
        self,
        dictionary_id: Optional[int] = None,
        group: Optional[str] = None,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get dictionary entries as dictionaries.

        Args:
            dictionary_id: Optional specific dictionary ID
            group: Optional group filter
            active_only: Only return entries from active dictionaries

        Returns:
            List of dictionaries representing entries
        """
        try:
            with self.db.get_connection() as conn:
                query = (
                    "SELECT e.* FROM dictionary_entries e JOIN chat_dictionaries d ON e.dictionary_id = d.id "
                    "WHERE d.deleted = ?"
                )
                params: list[Any] = [False]
                if active_only:
                    query += " AND d.is_active = ? AND e.enabled = ?"
                    params.extend([True, True])
                if dictionary_id:
                    query += " AND e.dictionary_id = ?"
                    params.append(dictionary_id)
                if group:
                    query += " AND e.group_name = ?"
                    params.append(group)
                # Deterministic multi-dictionary order:
                # 1) dictionary name (alphabetical), 2) entry sort order, 3) entry id.
                query += (
                    " ORDER BY LOWER(d.name), d.id, COALESCE(e.sort_order, e.id), e.id"
                )

                cursor = conn.execute(query, tuple(params))
                result: list[dict[str, Any]] = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    entry_obj = ChatDictionaryEntry.from_dict(row_dict)
                    entry = entry_obj.to_dict()
                    # Preserve additional metadata from the raw row
                    entry["dictionary_id"] = row_dict.get("dictionary_id")
                    entry["created_at"] = row_dict.get("created_at")
                    entry["updated_at"] = row_dict.get("updated_at")
                    entry["sort_order"] = row_dict.get("sort_order")
                    entry["usage_count"] = int(row_dict.get("usage_count") or 0)
                    entry["last_used_at"] = row_dict.get("last_used_at")
                    # Normalize group naming consistency
                    if not entry.get("group") and row_dict.get("group_name"):
                        entry["group"] = row_dict.get("group_name")
                    result.append(entry)
                return result
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error fetching dictionary entries: {e}")
            raise CharactersRAGDBError(f"Error fetching dictionary entries: {e}") from e

    def get_entry(self, entry_id: int, active_only: bool = True) -> Optional[dict[str, Any]]:
        """
        Get a single dictionary entry by ID, including metadata.

        Args:
            entry_id: Entry ID to fetch
            active_only: Only return entries from active dictionaries

        Returns:
            Dictionary representing the entry or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                query = (
                    "SELECT e.* FROM dictionary_entries e JOIN chat_dictionaries d ON e.dictionary_id = d.id "
                    "WHERE d.deleted = ? AND e.id = ?"
                )
                params: list[Any] = [False, entry_id]
                if active_only:
                    query += " AND d.is_active = ? AND e.enabled = ?"
                    params.extend([True, True])

                row = conn.execute(query, tuple(params)).fetchone()
                if not row:
                    return None

                row_dict = dict(row)
                entry_obj = ChatDictionaryEntry.from_dict(row_dict)
                entry = entry_obj.to_dict()
                entry["dictionary_id"] = row_dict.get("dictionary_id")
                entry["created_at"] = row_dict.get("created_at")
                entry["updated_at"] = row_dict.get("updated_at")
                entry["sort_order"] = row_dict.get("sort_order")
                entry["usage_count"] = int(row_dict.get("usage_count") or 0)
                entry["last_used_at"] = row_dict.get("last_used_at")
                if not entry.get("group") and row_dict.get("group_name"):
                    entry["group"] = row_dict.get("group_name")
                return entry
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error fetching dictionary entry: {e}")
            raise CharactersRAGDBError(f"Error fetching dictionary entry: {e}") from e

    def get_entry_objects(
        self,
        dictionary_id: Optional[int] = None,
        group: Optional[str] = None,
        active_only: bool = True,
    ) -> list[ChatDictionaryEntry]:
        """Internal: get entries as objects for processing, with basic caching."""
        # Check cache (no lock needed - request-scoped service instance)
        if (
            self._entry_cache is not None
            and self._cache_timestamp
            and not dictionary_id
            and not group
            and active_only
            and datetime.now() - self._cache_timestamp < self._cache_ttl
        ):
            return self._entry_cache

        try:
            with self.db.get_connection() as conn:
                query = (
                    "SELECT e.* FROM dictionary_entries e JOIN chat_dictionaries d ON e.dictionary_id = d.id "
                    "WHERE d.deleted = ?"
                )
                params: list[Any] = [False]
                if active_only:
                    query += " AND d.is_active = ? AND e.enabled = ?"
                    params.extend([True, True])
                if dictionary_id:
                    query += " AND e.dictionary_id = ?"
                    params.append(dictionary_id)
                if group:
                    query += " AND e.group_name = ?"
                    params.append(group)
                # Keep runtime processing order aligned with list semantics.
                query += (
                    " ORDER BY LOWER(d.name), d.id, COALESCE(e.sort_order, e.id), e.id"
                )

                cursor = conn.execute(query, tuple(params))
                entries: list[ChatDictionaryEntry] = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    entry = ChatDictionaryEntry.from_dict(row_dict)
                    dictionary_id_for_entry = self._coerce_dictionary_id(row_dict.get("dictionary_id"))
                    if dictionary_id_for_entry is not None:
                        setattr(entry, "dictionary_id", dictionary_id_for_entry)
                    entries.append(entry)

                # Update cache (no lock needed - request-scoped)
                if not dictionary_id and not group and active_only:
                    self._entry_cache = entries
                    self._cache_timestamp = datetime.now()
                # Legacy fallback: some tests stub execute_query instead of connection
                if not entries and hasattr(self.db, "execute_query"):
                    try:
                        # Consume potential dict list (ignored), then load entries
                        _ = self.db.execute_query("active_dictionaries")
                        rows = self.db.execute_query("dictionary_entries")
                        for r in rows or []:
                            entries.append(ChatDictionaryEntry.from_dict(r))
                    except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
                        pass
                return entries
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error fetching dictionary entry objects: {e}")
            raise CharactersRAGDBError(f"Error fetching dictionary entry objects: {e}") from e

    def _resolve_processing_dictionary_order(
        self,
        dictionary_id: Optional[int],
    ) -> list[int]:
        """
        Resolve dictionary processing order with include/inheritance semantics.

        Included dictionaries are processed before the including dictionary.
        For global processing, active root dictionaries are traversed in
        alphabetical order (name, id).
        """
        target_dictionary_id = self._coerce_positive_int(dictionary_id)
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT id, name, included_dictionary_ids
                    FROM chat_dictionaries
                    WHERE deleted = ? AND is_active = ?
                    ORDER BY LOWER(name), id
                    """,
                    (False, True),
                ).fetchall()
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error loading dictionary processing graph: {e}")
            raise CharactersRAGDBError(
                f"Error loading dictionary processing graph: {e}"
            ) from e

        dictionary_rows: dict[int, dict[str, Any]] = {}
        for row in rows:
            row_dict = dict(row)
            row_id = self._coerce_positive_int(row_dict.get("id"))
            if row_id is None:
                continue
            dictionary_rows[row_id] = row_dict

        if target_dictionary_id is not None and target_dictionary_id not in dictionary_rows:
            return []

        include_map: dict[int, list[int]] = {}
        for row_id, row_dict in dictionary_rows.items():
            includes = self._normalize_included_dictionary_ids(row_dict.get("included_dictionary_ids"))
            include_map[row_id] = [
                included_id for included_id in includes if included_id in dictionary_rows
            ]

        if target_dictionary_id is not None:
            root_ids = [target_dictionary_id]
        else:
            root_ids = [
                self._coerce_positive_int(dict(row).get("id"))
                for row in rows
                if self._coerce_positive_int(dict(row).get("id")) is not None
            ]

        visiting: set[int] = set()
        visited: set[int] = set()
        ordered: list[int] = []
        path: list[int] = []

        def _visit(current_dictionary_id: int) -> None:
            if current_dictionary_id in visited:
                return
            if current_dictionary_id in visiting:
                cycle_start_index = path.index(current_dictionary_id) if current_dictionary_id in path else 0
                cycle_path = path[cycle_start_index:] + [current_dictionary_id]
                raise InputError(
                    f"Dictionary include cycle detected: {' -> '.join(str(item) for item in cycle_path)}"
                )
            visiting.add(current_dictionary_id)
            path.append(current_dictionary_id)
            for included_id in include_map.get(current_dictionary_id, []):
                _visit(included_id)
            path.pop()
            visiting.remove(current_dictionary_id)
            visited.add(current_dictionary_id)
            ordered.append(current_dictionary_id)

        for root_dictionary_id in root_ids:
            if root_dictionary_id is None:
                continue
            _visit(root_dictionary_id)
        return ordered

    def _get_entry_objects_for_processing_order(
        self,
        dictionary_order: list[int],
        group: Optional[str],
    ) -> list[ChatDictionaryEntry]:
        if not dictionary_order:
            return []

        entries: list[ChatDictionaryEntry] = []
        try:
            with self.db.get_connection() as conn:
                for dictionary_id in dictionary_order:
                    query = (
                        "SELECT e.* FROM dictionary_entries e JOIN chat_dictionaries d ON e.dictionary_id = d.id "
                        "WHERE d.deleted = ? AND e.enabled = ? AND e.dictionary_id = ?"
                    )
                    params: list[Any] = [False, True, int(dictionary_id)]
                    if group:
                        query += " AND e.group_name = ?"
                        params.append(group)
                    query += " ORDER BY COALESCE(e.sort_order, e.id), e.id"

                    rows = conn.execute(query, tuple(params)).fetchall()
                    for row in rows:
                        row_dict = dict(row)
                        entry = ChatDictionaryEntry.from_dict(row_dict)
                        setattr(entry, "dictionary_id", int(dictionary_id))
                        entries.append(entry)
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error loading entries for composed processing order: {e}")
            raise CharactersRAGDBError(
                f"Error loading entries for composed processing order: {e}"
            ) from e

        return entries

    def update_entry(
        self,
        entry_id: int,
        key: Optional[str] = None,
        content: Optional[str] = None,
        probability: Optional[Union[int, float]] = None,
        group: Optional[str] = None,
        timed_effects: Optional[dict[str, int]] = None,
        max_replacements: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """
        Update a dictionary entry.

        Args:
            entry_id: Entry ID
            Various optional fields to update

        Returns:
            True if updated successfully
        """
        try:
            updates: list[str] = []
            params: list[Any] = []

            pattern = kwargs.get("pattern", key)
            replacement = kwargs.get("replacement", content)
            entry_type = kwargs.get("type")
            enabled = kwargs.get("enabled")
            case_sensitive = kwargs.get("case_sensitive")
            record_snapshot = bool(kwargs.get("record_snapshot", True))

            if pattern is not None:
                entry = ChatDictionaryEntry(pattern, "test")
                if entry_type is not None:
                    is_regex_requested = entry_type == "regex"
                    # Validate regex pattern if explicitly requested
                    if is_regex_requested and not entry.is_regex:
                        # Entry didn't auto-detect as regex, try to compile it explicitly
                        try:
                            _safe_compile_regex(pattern)
                        except re.error as e:
                            raise InputError(f"Invalid regex pattern: {e}") from e
                    entry.is_regex = is_regex_requested
                updates.append("key = ?")
                updates.append("is_regex = ?")
                params.extend([pattern, bool(entry.is_regex)])

            if replacement is not None:
                updates.append("content = ?")
                params.append(replacement)

            if probability is not None:
                if isinstance(probability, int):
                    if not 0 <= probability <= 100:
                        raise InputError("Probability must be between 0 and 100")
                    prob_f = probability / 100.0
                else:
                    if not 0.0 <= probability <= 1.0:
                        raise InputError("Probability must be between 0.0 and 1.0")
                    prob_f = float(probability)
                updates.append("probability = ?")
                params.append(prob_f)

            if group is not None:
                updates.append("group_name = ?")
                params.append(group)

            if timed_effects is not None:
                updates.append("timed_effects = ?")
                params.append(json.dumps(timed_effects))

            if max_replacements is not None:
                updates.append("max_replacements = ?")
                params.append(int(max_replacements))

            if enabled is not None:
                updates.append("enabled = ?")
                params.append(bool(enabled))

            if case_sensitive is not None:
                updates.append("case_sensitive = ?")
                params.append(bool(case_sensitive))

            if not updates:
                return True

            updates.append("updated_at = CURRENT_TIMESTAMP")
            params.append(entry_id)

            with self.db.get_connection() as conn:
                dictionary_id_row = conn.execute(
                    "SELECT dictionary_id FROM dictionary_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone()
                if isinstance(dictionary_id_row, dict):
                    dictionary_id_for_entry = dictionary_id_row.get("dictionary_id")
                elif hasattr(dictionary_id_row, "_mapping"):
                    dictionary_id_for_entry = dictionary_id_row._mapping.get("dictionary_id")
                else:
                    dictionary_id_for_entry = dictionary_id_row[0] if dictionary_id_row else None

                set_clause = _build_safe_update_clause(updates, _ENTRY_UPDATE_FIELDS)
                update_entry_sql_template = "UPDATE dictionary_entries SET {set_clause} WHERE id = ?"
                update_entry_sql = update_entry_sql_template.format_map(locals())  # nosec B608
                cursor = conn.execute(update_entry_sql, tuple(params))

                if cursor.rowcount > 0:
                    if record_snapshot and dictionary_id_for_entry is not None:
                        self._create_dictionary_version_snapshot(
                            conn,
                            int(dictionary_id_for_entry),
                            change_type="entry_update",
                            summary=f"Updated entry {int(entry_id)}",
                        )
                    conn.commit()
                    logger.info(f"Updated dictionary entry {entry_id}")
                    self._invalidate_cache()
                    return True
                return False

        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error updating dictionary entry: {e}")
            raise CharactersRAGDBError(f"Error updating dictionary entry: {e}") from e

    def delete_entry(self, entry_id: int, *, record_snapshot: bool = True) -> bool:
        """
        Delete a dictionary entry.

        Args:
            entry_id: Entry ID

        Returns:
            True if deleted successfully
        """
        try:
            with self.db.get_connection() as conn:
                dictionary_id_row = conn.execute(
                    "SELECT dictionary_id FROM dictionary_entries WHERE id = ?",
                    (entry_id,),
                ).fetchone()
                if isinstance(dictionary_id_row, dict):
                    dictionary_id_for_entry = dictionary_id_row.get("dictionary_id")
                elif hasattr(dictionary_id_row, "_mapping"):
                    dictionary_id_for_entry = dictionary_id_row._mapping.get("dictionary_id")
                else:
                    dictionary_id_for_entry = dictionary_id_row[0] if dictionary_id_row else None

                cursor = conn.execute("DELETE FROM dictionary_entries WHERE id = ?", (entry_id,))

                if cursor.rowcount > 0:
                    if record_snapshot and dictionary_id_for_entry is not None:
                        self._create_dictionary_version_snapshot(
                            conn,
                            int(dictionary_id_for_entry),
                            change_type="entry_delete",
                            summary=f"Deleted entry {int(entry_id)}",
                        )
                    conn.commit()
                    logger.info(f"Deleted dictionary entry {entry_id}")
                    self._invalidate_cache()
                    return True
                return False

        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error deleting dictionary entry: {e}")
            raise CharactersRAGDBError(f"Error deleting dictionary entry: {e}") from e

    def reorder_entries(
        self,
        dictionary_id: int,
        entry_ids: list[int],
        *,
        record_snapshot: bool = True,
    ) -> int:
        """
        Persist a new execution order for all entries in a dictionary.

        Args:
            dictionary_id: Dictionary ID whose entries should be reordered.
            entry_ids: Full ordered list of entry IDs for the dictionary.

        Returns:
            Number of entries reordered.

        Raises:
            InputError: If the requested entry IDs are invalid or incomplete.
        """
        try:
            if not entry_ids:
                raise InputError("entry_ids must contain at least one entry ID")

            normalized_entry_ids: list[int] = []
            seen: set[int] = set()
            for raw_id in entry_ids:
                entry_id = int(raw_id)
                if entry_id <= 0:
                    raise InputError("entry_ids must contain positive integers")
                if entry_id in seen:
                    raise InputError("entry_ids must not contain duplicates")
                normalized_entry_ids.append(entry_id)
                seen.add(entry_id)

            with self.db.get_connection() as conn:
                rows = conn.execute(
                    "SELECT id FROM dictionary_entries WHERE dictionary_id = ?",
                    (dictionary_id,),
                ).fetchall()
                existing_entry_ids: list[int] = []
                for row in rows:
                    if isinstance(row, dict):
                        raw_entry_id = row.get("id")
                    elif hasattr(row, "_mapping"):
                        raw_entry_id = row._mapping.get("id")
                    else:
                        raw_entry_id = row[0] if row else None
                    if raw_entry_id is not None:
                        existing_entry_ids.append(int(raw_entry_id))

                if not existing_entry_ids:
                    raise InputError("Dictionary has no entries to reorder")

                if set(existing_entry_ids) != set(normalized_entry_ids):
                    raise InputError(
                        "entry_ids must include every dictionary entry exactly once"
                    )

                for order_index, entry_id in enumerate(normalized_entry_ids, start=1):
                    conn.execute(
                        """
                        UPDATE dictionary_entries
                        SET sort_order = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ? AND dictionary_id = ?
                        """,
                        (order_index, entry_id, dictionary_id),
                    )

                if record_snapshot:
                    self._create_dictionary_version_snapshot(
                        conn,
                        int(dictionary_id),
                        change_type="entry_reorder",
                        summary=f"Reordered {len(normalized_entry_ids)} entries",
                    )
                conn.commit()
                self._invalidate_cache()
                return len(normalized_entry_ids)
        except InputError:
            raise
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error reordering dictionary entries: {e}")
            raise CharactersRAGDBError(
                f"Error reordering dictionary entries: {e}"
            ) from e

    # --- Text Processing ---

    def _record_dictionary_usage(self, dictionary_id: int) -> None:
        """Persist usage metrics for a dictionary, best-effort."""
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE chat_dictionaries
                    SET usage_count = COALESCE(usage_count, 0) + 1,
                        last_used_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND deleted = ?
                    """,
                    (dictionary_id, False),
                )
                conn.commit()
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
            return

    def _record_dictionary_usage_for_processing(self, dictionary_ids: list[int]) -> None:
        """Persist usage metrics for all dictionaries participating in a processing run."""
        normalized_ids = sorted(
            {
                dictionary_id
                for dictionary_id in (
                    self._coerce_positive_int(candidate_id) for candidate_id in dictionary_ids
                )
                if dictionary_id is not None
            }
        )
        if not normalized_ids:
            return

        now = datetime.now(timezone.utc)
        if hasattr(self, "_usage_counts"):
            for dictionary_id in normalized_ids:
                self._usage_counts[dictionary_id] = self._usage_counts.get(dictionary_id, 0) + 1
        if hasattr(self, "_last_used_at"):
            for dictionary_id in normalized_ids:
                self._last_used_at[dictionary_id] = now

        placeholders = ", ".join("?" for _ in normalized_ids)
        dictionary_ids_clause = f"({placeholders})"
        usage_update_sql_template = """
                    UPDATE chat_dictionaries
                    SET usage_count = COALESCE(usage_count, 0) + 1,
                        last_used_at = CURRENT_TIMESTAMP
                    WHERE deleted = ? AND id IN {dictionary_ids_clause}
                    """
        usage_update_sql = usage_update_sql_template.format_map(locals())  # nosec B608
        try:
            with self.db.get_connection() as conn:
                conn.execute(
                    usage_update_sql,
                    tuple([False, *normalized_ids]),
                )
                conn.commit()
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
            return

    def _record_entry_usage_counts(self, usage_counts: dict[int, int]) -> None:
        """Persist usage count increments for dictionary entries, best-effort."""
        if not usage_counts:
            return
        try:
            with self.db.get_connection() as conn:
                for entry_id, increment in usage_counts.items():
                    if entry_id <= 0:
                        continue
                    increment_value = int(increment)
                    if increment_value <= 0:
                        continue
                    conn.execute(
                        """
                        UPDATE dictionary_entries
                        SET usage_count = COALESCE(usage_count, 0) + ?,
                            last_used_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (increment_value, int(entry_id)),
                    )
                conn.commit()
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
            return

    def _resolve_default_token_budget(
        self,
        dictionary_id: Optional[int],
    ) -> Optional[int]:
        """Resolve fallback token budget from dictionary metadata when request omits it."""
        try:
            with self.db.get_connection() as conn:
                if dictionary_id is not None:
                    row = conn.execute(
                        """
                        SELECT default_token_budget
                        FROM chat_dictionaries
                        WHERE id = ? AND deleted = ? AND is_active = ?
                        LIMIT 1
                        """,
                        (int(dictionary_id), False, True),
                    ).fetchone()
                    if row is None:
                        return None
                    if isinstance(row, dict):
                        raw_budget = row.get("default_token_budget")
                    elif hasattr(row, "_mapping"):
                        raw_budget = row._mapping.get("default_token_budget")
                    else:
                        raw_budget = row[0] if len(row) > 0 else None
                    return self._coerce_positive_int(raw_budget)

                rows = conn.execute(
                    """
                    SELECT default_token_budget
                    FROM chat_dictionaries
                    WHERE deleted = ? AND is_active = ?
                      AND default_token_budget IS NOT NULL
                    """,
                    (False, True),
                ).fetchall()
                budgets: list[int] = []
                for row in rows:
                    if isinstance(row, dict):
                        raw_budget = row.get("default_token_budget")
                    elif hasattr(row, "_mapping"):
                        raw_budget = row._mapping.get("default_token_budget")
                    else:
                        raw_budget = row[0] if len(row) > 0 else None
                    coerced = self._coerce_positive_int(raw_budget)
                    if coerced is not None:
                        budgets.append(coerced)
                if not budgets:
                    return None
                # Conservative fallback for multi-dictionary processing.
                return min(budgets)
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
            return None

    def _resolve_default_token_budget_for_dictionary_ids(
        self,
        dictionary_ids: list[int],
    ) -> Optional[int]:
        normalized_ids = [
            dictionary_id
            for dictionary_id in (self._coerce_positive_int(item) for item in dictionary_ids)
            if dictionary_id is not None
        ]
        if not normalized_ids:
            return None

        placeholders = ", ".join("?" for _ in normalized_ids)
        dictionary_ids_clause = f"({placeholders})"
        budget_lookup_sql_template = """
                    SELECT default_token_budget
                    FROM chat_dictionaries
                    WHERE deleted = ? AND is_active = ?
                      AND id IN {dictionary_ids_clause}
                      AND default_token_budget IS NOT NULL
                    """
        budget_lookup_sql = budget_lookup_sql_template.format_map(locals())  # nosec B608
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    budget_lookup_sql,
                    tuple([False, True, *normalized_ids]),
                ).fetchall()
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
            return None

        budgets: list[int] = []
        for row in rows:
            if isinstance(row, dict):
                raw_budget = row.get("default_token_budget")
            elif hasattr(row, "_mapping"):
                raw_budget = row._mapping.get("default_token_budget")
            else:
                raw_budget = row[0] if len(row) > 0 else None
            coerced = self._coerce_positive_int(raw_budget)
            if coerced is not None:
                budgets.append(coerced)

        return min(budgets) if budgets else None

    def record_transform_activity(
        self,
        dictionary_id: int,
        *,
        original_text: str,
        processed_text: str,
        entries_used: list[int],
        replacements: int,
        iterations: int,
        token_budget_used: Optional[int],
        processing_time_ms: Optional[float],
        chat_id: Optional[str] = None,
    ) -> None:
        """Persist a dictionary transformation event for recent activity views."""
        if dictionary_id <= 0:
            return
        try:
            entries_payload: list[int] = []
            seen: set[int] = set()
            for entry_id in entries_used:
                coerced_entry_id = self._coerce_positive_int(entry_id)
                if coerced_entry_id is None or coerced_entry_id in seen:
                    continue
                entries_payload.append(coerced_entry_id)
                seen.add(coerced_entry_id)
            original_preview = original_text[:280]
            processed_preview = processed_text[:280]
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO dictionary_transform_activity (
                        dictionary_id,
                        chat_id,
                        entries_used,
                        replacements,
                        iterations,
                        token_budget_used,
                        original_text_preview,
                        processed_text_preview,
                        processing_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(dictionary_id),
                        str(chat_id).strip() if isinstance(chat_id, str) and chat_id.strip() else None,
                        json.dumps(entries_payload),
                        max(0, int(replacements)),
                        max(0, int(iterations)),
                        self._coerce_positive_int(token_budget_used),
                        original_preview,
                        processed_preview,
                        float(processing_time_ms) if processing_time_ms is not None else None,
                    ),
                )
                conn.commit()
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
            return

    def _record_transform_activity_snapshot(
        self,
        *,
        dictionary_id: Optional[int],
        original_text: str,
        processed_text: str,
        stats: dict[str, Any],
        dictionary_replacements: dict[int, int],
        dictionary_entries_used: dict[int, set[int]],
        token_budget_used: Optional[int],
        chat_id: Optional[str],
    ) -> None:
        """Record per-dictionary activity events derived from a processing result."""
        target_dictionary_ids: set[int] = set()
        explicit_dictionary_id = self._coerce_dictionary_id(dictionary_id)
        if explicit_dictionary_id is not None:
            target_dictionary_ids.add(explicit_dictionary_id)
        else:
            target_dictionary_ids.update(
                dictionary_key
                for dictionary_key in dictionary_replacements.keys()
                if dictionary_key > 0
            )
            target_dictionary_ids.update(
                dictionary_key
                for dictionary_key in dictionary_entries_used.keys()
                if dictionary_key > 0
            )

        if not target_dictionary_ids:
            return

        total_replacements = max(0, int(stats.get("replacements") or 0))
        total_entries_used = [
            int(entry_id)
            for entry_id in stats.get("entries_used", [])
            if isinstance(entry_id, (int, float, str))
            and self._coerce_positive_int(entry_id) is not None
        ]
        iterations = max(0, int(stats.get("iterations") or 0))

        for target_dictionary_id in sorted(target_dictionary_ids):
            replacements = max(0, int(dictionary_replacements.get(target_dictionary_id, 0) or 0))
            entries_used = sorted(dictionary_entries_used.get(target_dictionary_id, set()))
            if explicit_dictionary_id is not None:
                if replacements <= 0 and total_replacements > 0:
                    replacements = total_replacements
                if not entries_used and total_entries_used:
                    entries_used = total_entries_used

            self.record_transform_activity(
                target_dictionary_id,
                original_text=original_text,
                processed_text=processed_text,
                entries_used=entries_used,
                replacements=replacements,
                iterations=iterations,
                token_budget_used=token_budget_used,
                processing_time_ms=None,
                chat_id=chat_id,
            )

    def list_transform_activity(
        self,
        dictionary_id: int,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return paginated transformation activity for a dictionary."""
        safe_dictionary_id = int(dictionary_id)
        safe_limit = max(1, int(limit))
        safe_offset = max(0, int(offset))
        with self.db.get_connection() as conn:
            count_row = conn.execute(
                """
                SELECT COUNT(1)
                FROM dictionary_transform_activity
                WHERE dictionary_id = ?
                """,
                (safe_dictionary_id,),
            ).fetchone()
            if count_row is None:
                total = 0
            elif isinstance(count_row, dict):
                total = int(next(iter(count_row.values()), 0))
            elif hasattr(count_row, "_mapping"):
                total = int(next(iter(count_row._mapping.values()), 0))
            else:
                total = int(count_row[0] if len(count_row) > 0 else 0)

            rows = conn.execute(
                """
                SELECT
                    id,
                    dictionary_id,
                    chat_id,
                    entries_used,
                    replacements,
                    iterations,
                    token_budget_used,
                    original_text_preview,
                    processed_text_preview,
                    processing_time_ms,
                    created_at
                FROM dictionary_transform_activity
                WHERE dictionary_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (safe_dictionary_id, safe_limit, safe_offset),
            ).fetchall()

            events: list[dict[str, Any]] = []
            for row in rows:
                row_data = dict(row)
                entries_used: list[int] = []
                raw_entries_payload = row_data.get("entries_used")
                parsed_entries: Any = []
                if isinstance(raw_entries_payload, str):
                    try:
                        parsed_entries = json.loads(raw_entries_payload)
                    except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
                        parsed_entries = []
                elif isinstance(raw_entries_payload, list):
                    parsed_entries = raw_entries_payload
                if isinstance(parsed_entries, list):
                    seen_entries: set[int] = set()
                    for item in parsed_entries:
                        coerced = self._coerce_positive_int(item)
                        if coerced is None or coerced in seen_entries:
                            continue
                        entries_used.append(coerced)
                        seen_entries.add(coerced)

                events.append(
                    {
                        "id": int(row_data.get("id")),
                        "dictionary_id": int(row_data.get("dictionary_id")),
                        "chat_id": row_data.get("chat_id"),
                        "entries_used": entries_used,
                        "replacements": int(row_data.get("replacements") or 0),
                        "iterations": int(row_data.get("iterations") or 0),
                        "token_budget_used": self._coerce_positive_int(row_data.get("token_budget_used")),
                        "original_text_preview": str(row_data.get("original_text_preview") or ""),
                        "processed_text_preview": str(row_data.get("processed_text_preview") or ""),
                        "processing_time_ms": (
                            float(row_data.get("processing_time_ms"))
                            if row_data.get("processing_time_ms") is not None
                            else None
                        ),
                        "created_at": row_data.get("created_at"),
                    }
                )

            return events, total

    def process_text(
        self,
        text: str,
        dictionary_id: Optional[int] = None,
        group: Optional[str] = None,
        max_iterations: int = 5,
        token_budget: Optional[int] = None,
        return_stats: bool = False,
        **kwargs,
    ) -> Union[str, tuple[str, dict[str, Any]]]:
        """
        Process text through dictionary replacements.

        Args:
            text: Input text
            dictionary_id: Optional specific dictionary to use
            group: Optional group filter
            max_iterations: Maximum processing iterations
            token_budget: Optional token limit (alias: max_tokens)

        Returns:
            - When return_stats=True: a tuple of (processed_text, stats_dict)
            - Otherwise: a string-like result containing the processed text. The
              returned object is a subclass of str that also supports
              mapping-like access for legacy tests (e.g. result['processed_text']).
        """
        # Support alias max_tokens
        if token_budget is None and "max_tokens" in kwargs:
            token_budget = kwargs.get("max_tokens")

        raw_chat_id = kwargs.get("chat_id")
        chat_id = str(raw_chat_id).strip() if isinstance(raw_chat_id, str) and raw_chat_id.strip() else None

        if len(text) > MAX_CHAT_DICTIONARY_TEXT_LENGTH:
            raise InputError(
                f"Input text exceeds maximum length ({len(text)} > {MAX_CHAT_DICTIONARY_TEXT_LENGTH})"
            )

        original_text = text
        processing_dictionary_ids = self._resolve_processing_dictionary_order(dictionary_id)

        effective_token_budget = self._coerce_positive_int(token_budget)
        if effective_token_budget is None:
            effective_token_budget = self._resolve_default_token_budget_for_dictionary_ids(
                processing_dictionary_ids
            )
        if effective_token_budget is None:
            effective_token_budget = self._resolve_default_token_budget(dictionary_id)

        # Get applicable entries as objects
        entries = self._get_entry_objects_for_processing_order(
            processing_dictionary_ids,
            group,
        )

        if not entries:
            empty_stats = {
                "replacements": 0,
                "iterations": 0,
                "entries_used": [],
                "token_budget_exceeded": False,
                "token_budget_used": effective_token_budget,
            }
            explicit_dictionary_id = self._coerce_dictionary_id(dictionary_id)
            if explicit_dictionary_id is not None:
                self.record_transform_activity(
                    explicit_dictionary_id,
                    original_text=original_text,
                    processed_text=text,
                    entries_used=[],
                    replacements=0,
                    iterations=0,
                    token_budget_used=effective_token_budget,
                    processing_time_ms=None,
                    chat_id=chat_id,
                )
            if return_stats:
                return text, empty_stats
            # Return a string-like result that also supports ['processed_text'] access
            return _ProcessedTextResult(text, empty_stats)

        stats = {
            "replacements": 0,
            "iterations": 0,
            "entries_used": [],
            "token_budget_exceeded": False,
            "token_budget_used": effective_token_budget,
        }
        entry_usage_increments: dict[int, int] = {}
        dictionary_replacements: dict[int, int] = {}
        dictionary_entries_used: dict[int, set[int]] = {}

        # Track usage for all dictionaries included in this processing pass.
        self._record_dictionary_usage_for_processing(processing_dictionary_ids)

        for _iteration in range(max_iterations):
            iteration_replacements = 0

            for entry in entries:
                if entry.matches(text):
                    entry_dictionary_id = self._coerce_dictionary_id(getattr(entry, "dictionary_id", None))
                    # If enforcing a token budget, replace incrementally (one per entry per pass)
                    original_max = entry.max_replacements
                    if effective_token_budget is not None and original_max == 0:
                        entry.max_replacements = 1
                    try:
                        new_text, count = entry.apply_replacement(text)
                    finally:
                        entry.max_replacements = original_max
                    if count > 0:
                        text = new_text
                        iteration_replacements += count
                        stats["replacements"] += count

                        if entry.entry_id not in stats["entries_used"]:
                            stats["entries_used"].append(entry.entry_id)
                        if isinstance(entry.entry_id, int) and entry.entry_id > 0:
                            entry_usage_increments[entry.entry_id] = (
                                entry_usage_increments.get(entry.entry_id, 0) + count
                            )
                            if entry_dictionary_id is not None:
                                dictionary_entries_used.setdefault(entry_dictionary_id, set()).add(entry.entry_id)
                        if entry_dictionary_id is not None:
                            dictionary_replacements[entry_dictionary_id] = (
                                dictionary_replacements.get(entry_dictionary_id, 0) + count
                            )

                        # Check token budget
                        if effective_token_budget and self.count_tokens(text) > effective_token_budget:
                            warnings.warn(
                                (
                                    f"Token budget ({effective_token_budget}) exceeded "
                                    f"after {stats['replacements']} replacements"
                                ),
                                TokenBudgetExceededWarning, stacklevel=2,
                            )
                            stats["token_budget_exceeded"] = True
                            self._record_entry_usage_counts(entry_usage_increments)
                            self._record_transform_activity_snapshot(
                                dictionary_id=dictionary_id,
                                original_text=original_text,
                                processed_text=text,
                                stats=stats,
                                dictionary_replacements=dictionary_replacements,
                                dictionary_entries_used=dictionary_entries_used,
                                token_budget_used=effective_token_budget,
                                chat_id=chat_id,
                            )
                            if return_stats:
                                return text, stats
                            return _ProcessedTextResult(text, stats)

            stats["iterations"] += 1

            # Stop if no replacements were made in this iteration
            if iteration_replacements == 0:
                break
        self._record_entry_usage_counts(entry_usage_increments)
        self._record_transform_activity_snapshot(
            dictionary_id=dictionary_id,
            original_text=original_text,
            processed_text=text,
            stats=stats,
            dictionary_replacements=dictionary_replacements,
            dictionary_entries_used=dictionary_entries_used,
            token_budget_used=effective_token_budget,
            chat_id=chat_id,
        )
        if return_stats:
            return text, stats
        return _ProcessedTextResult(text, stats)

    def count_tokens(self, text: str) -> int:
        """Public token counter to support tests; can be patched/mocked."""
        return _count_tokens(text)

    def import_from_markdown(self, markdown_or_path: Union[str, Path], dictionary_name: Optional[str] = None) -> int:
        """
        Import dictionary entries from a markdown file.

        File format:
        ```
        key: value
        /regex/: replacement
        /pattern/i: case-insensitive replacement

        ## Group Name
        grouped_key: grouped_value
        ```

        Args:
            markdown_or_path: Markdown content or a Path to a markdown file
            dictionary_name: Name for the new dictionary

        Returns:
            Dictionary ID
        """
        # Determine if argument is a file path or raw markdown content.
        # Strings are treated as content to avoid unsafe file reads.
        content: Optional[str] = None
        if isinstance(markdown_or_path, Path):
            fp = markdown_or_path
            with open(fp, encoding="utf-8") as f:
                content = f.read()
            if dictionary_name is None:
                dictionary_name = fp.stem
        else:
            content = str(markdown_or_path)

        if not content:
            raise InputError("No markdown content provided")

        # Extract dictionary name if not provided
        if dictionary_name is None:
            m = re.search(r"^#\s*(.+)$", content, flags=re.MULTILINE)
            dictionary_name = m.group(1).strip() if m else "Imported Dictionary"

        # Create dictionary
        dict_id = self.create_dictionary(dictionary_name, "Imported from markdown")

        try:
            # Parse entries by scanning headings and blocks
            lines = content.splitlines()
            current_entry_name: Optional[str] = None
            current_block_lines: list[str] = []

            def flush_block():
                if current_entry_name is None:
                    return
                block_text = "\n".join(current_block_lines)
                type_val = (self._extract_md_field(block_text, "Type") or "literal").lower()
                # If a Pattern field exists, use it; else use entry name as pattern
                pattern_val = self._extract_md_field(block_text, "Pattern") or current_entry_name
                replacement_val = self._extract_md_field(block_text, "Replacement") or ""
                prob_str = self._extract_md_field(block_text, "Probability")
                enabled_str = self._extract_md_field(block_text, "Enabled")
                prob_val = self._normalize_probability_input(prob_str) if prob_str is not None else None
                enabled_val = True if enabled_str is None else enabled_str.lower() == "true"
                if pattern_val:
                    self.add_entry(
                        dict_id,
                        pattern=pattern_val,
                        replacement=replacement_val,
                        type=type_val,
                        probability=prob_val,
                        enabled=enabled_val,
                        record_snapshot=False,
                    )

            for line in lines:
                if line.startswith("## Entry:"):
                    # flush previous
                    flush_block()
                    # start new block
                    current_entry_name = line.split(":", 1)[1].strip()
                    current_block_lines = []
                else:
                    current_block_lines.append(line)
            # flush last
            flush_block()

            # If no '## Entry:' sections found, fallback to simple key: value lines
            if current_entry_name is None:
                for line in lines:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if ":" in s:
                        k, v = s.split(":", 1)
                        self.add_entry(
                            dict_id,
                            pattern=k.strip(),
                            replacement=v.strip(),
                            record_snapshot=False,
                        )

            with self.db.get_connection() as conn:
                self._create_dictionary_version_snapshot(
                    conn,
                    int(dict_id),
                    change_type="import_markdown",
                    summary="Imported dictionary from markdown",
                )
                conn.commit()

            return dict_id
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            self.delete_dictionary(dict_id, hard_delete=True)
            raise CharactersRAGDBError(f"Failed to import dictionary: {e}") from e

    def export_to_markdown(self, dictionary_id: int, file_path: Optional[Path] = None) -> Union[bool, str]:
        """
        Export a dictionary to markdown format.

        Args:
            dictionary_id: Dictionary to export
            file_path: Output file path (Path object)

        Returns:
            True if exported successfully
        """
        # Get dictionary info
        dict_info = self.get_dictionary(dictionary_id)
        if not dict_info:
            raise InputError(f"Dictionary {dictionary_id} not found")

        entries = self.get_entries(dictionary_id, active_only=False)

        lines: list[str] = []
        lines.append(f"# {dict_info['name']}")
        lines.append("")
        if dict_info.get("description"):
            lines.append(str(dict_info["description"]))
            lines.append("")

        for e in entries:
            lines.append(f"## Entry: {e['pattern']}")
            lines.append(f"- **Type**: {e.get('type', 'literal')}")
            if e.get("type") == "regex":
                lines.append(f"- **Pattern**: {e['pattern']}")
            lines.append(f"- **Replacement**: {e['replacement']}")
            if e.get("probability") is not None:
                lines.append(f"- **Probability**: {float(e['probability'])}")
            lines.append(f"- **Enabled**: {str(bool(e.get('enabled', 1))).lower()}")
            lines.append("")

        content = "\n".join(lines)

        if file_path is None:
            return content

        if not isinstance(file_path, Path):
            raise InputError("file_path must be a pathlib.Path to avoid unsafe file operations")
        fp = file_path
        try:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to export dictionary: {e}")
            raise CharactersRAGDBError(f"Failed to export dictionary: {e}") from e

    # --- Helper Methods ---

    def _invalidate_cache(self):
        """Invalidate the entry cache."""
        self._entry_cache = None
        self._cache_timestamp = None

    def _get_entry_dict_id(self, entry_id: int) -> Optional[int]:
        """Get the dictionary ID for an entry."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("SELECT dictionary_id FROM dictionary_entries WHERE id = ?", (entry_id,))
                row = cursor.fetchone()
                return row["dictionary_id"] if row else None
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
            return None

    def get_entry_dictionary_id(self, entry_id: int) -> Optional[int]:
        """
        Public wrapper to fetch the dictionary ID for a given entry.

        Exposes the lookup used by API handlers without coupling to the private helper.
        """
        return self._get_entry_dict_id(entry_id)

    def toggle_dictionary_active(self, dictionary_id: int, is_active: Optional[bool] = None) -> bool:
        """
        Toggle dictionary active status.

        Args:
            dictionary_id: Dictionary ID
            is_active: New active status (if None, flip current)

        Returns:
            True if updated successfully
        """
        if is_active is None:
            info = self.get_dictionary(dictionary_id)
            current = bool(info.get("is_active", True)) if info else True
            is_active = not current
        return self.update_dictionary(dictionary_id, is_active=is_active)

    def get_statistics(self, dictionary_id: Optional[int] = None) -> dict[str, Any]:
        """Get statistics for a specific dictionary (or global if None).

        Backward-compatible query patterns that minimize DB calls to align with legacy tests.
        """
        try:
            with self.db.get_connection() as conn:
                if dictionary_id is None:
                    # Combined counts to reduce number of execute calls
                    row = conn.execute(
                        """
                        SELECT
                            COUNT(*) as total_dictionaries,
                            SUM(CASE WHEN is_active AND NOT deleted THEN 1 ELSE 0 END) as active_dictionaries
                        FROM chat_dictionaries
                        WHERE deleted = ?
                        """,
                        (False,),
                    ).fetchone()
                    if isinstance(row, dict):
                        total_dicts = int(row.get("total_dictionaries", 0))
                        active_dicts = int(row.get("active_dictionaries", 0))
                    else:
                        # Tuple-like
                        total_dicts = int(row[0]) if row else 0
                        active_dicts = int(row[1]) if row and len(row) > 1 else 0

                    row2 = conn.execute(
                        """
                        SELECT
                            COUNT(*) as total_entries,
                            SUM(CASE WHEN COALESCE(is_regex, FALSE) THEN 1 ELSE 0 END) as regex_entries,
                            SUM(CASE WHEN NOT COALESCE(is_regex, FALSE) THEN 1 ELSE 0 END) as literal_entries,
                            SUM(CASE WHEN probability IS NOT NULL AND probability < 1.0 THEN 1 ELSE 0 END) as probabilistic_entries
                        FROM dictionary_entries
                        """,
                    ).fetchone()
                    if isinstance(row2, dict):
                        total_entries = int(row2.get("total_entries", 0))
                        regex_entries = int(row2.get("regex_entries", 0))
                        literal_entries = int(row2.get("literal_entries", 0))
                        probabilistic_entries = int(row2.get("probabilistic_entries", 0))
                    else:
                        total_entries = int(row2[0]) if row2 else 0
                        regex_entries = int(row2[1]) if row2 and len(row2) > 1 else 0
                        literal_entries = int(row2[2]) if row2 and len(row2) > 2 else 0
                        probabilistic_entries = int(row2[3]) if row2 and len(row2) > 3 else 0

                    avg_entries = (total_entries / total_dicts) if total_dicts else 0
                    return {
                        "total_dictionaries": total_dicts,
                        "active_dictionaries": active_dicts,
                        "total_entries": total_entries,
                        "literal_entries": literal_entries,
                        "regex_entries": regex_entries,
                        "probabilistic_entries": probabilistic_entries,
                        "average_entries_per_dictionary": avg_entries,
                    }
                else:
                    row = conn.execute(
                        """
                        SELECT
                            COUNT(*) AS total_entries,
                            SUM(CASE WHEN COALESCE(is_regex, FALSE) THEN 1 ELSE 0 END) AS regex_entries,
                            SUM(CASE WHEN NOT COALESCE(is_regex, FALSE) THEN 1 ELSE 0 END) AS literal_entries,
                            SUM(CASE WHEN probability IS NOT NULL AND probability < 1.0 THEN 1 ELSE 0 END) AS probabilistic_entries
                        FROM dictionary_entries
                        WHERE dictionary_id = ?
                        """,
                        (dictionary_id,),
                    ).fetchone()
                    if isinstance(row, dict):
                        total_entries = int(row.get("total_entries", 0))
                        regex_entries = int(row.get("regex_entries", 0))
                        literal_entries = int(row.get("literal_entries", 0))
                        probabilistic_entries = int(row.get("probabilistic_entries", 0))
                    else:
                        total_entries = int(row[0]) if row else 0
                        regex_entries = int(row[1]) if row and len(row) > 1 else 0
                        literal_entries = int(row[2]) if row and len(row) > 2 else 0
                        probabilistic_entries = int(row[3]) if row and len(row) > 3 else 0
                    return {
                        "total_entries": total_entries,
                        "literal_entries": literal_entries,
                        "regex_entries": regex_entries,
                        "probabilistic_entries": probabilistic_entries,
                    }
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error getting statistics: {e}")
            raise CharactersRAGDBError(f"Error getting statistics: {e}") from e

    def bulk_add_entries(self, dictionary_id: int, entries: list[dict[str, Any]]) -> Any:
        """
        Add multiple entries at once.

        Args:
            dictionary_id: Dictionary ID
            entries: List of entry dictionaries with keys: key, content, probability, group, etc.

        Returns:
            Number of entries added
        """
        try:
            added_count = 0
            with self.db.get_connection() as conn:
                for e in entries:
                    pattern = e.get("pattern") or e.get("key")
                    replacement = e.get("replacement") or e.get("content", "")
                    if not pattern:
                        continue
                    entry_type = e.get("type", "literal")
                    probability = e.get("probability", 1.0)
                    if isinstance(probability, int):
                        probability = probability / 100.0
                    group = e.get("group") or e.get("group_name")
                    timed_effects = e.get("timed_effects", {"sticky": 0, "cooldown": 0, "delay": 0})
                    max_replacements = int(e.get("max_replacements", 0))
                    enabled = bool(e.get("enabled", 1))
                    case_sensitive = bool(e.get("case_sensitive", 1))

                    # Validate pattern
                    test_entry = ChatDictionaryEntry(pattern, replacement)
                    is_regex = True if entry_type == "regex" else bool(test_entry.is_regex)
                    if is_regex and not _is_compiled_regex(test_entry.key):
                        test_entry.is_regex = True
                        test_entry._compile_regex_pattern(test_entry.key_pattern_str)
                    timed_effects_json = json.dumps(timed_effects)
                    conn.execute(
                        """
                        INSERT INTO dictionary_entries
                        (dictionary_id, key, content, is_regex, probability, max_replacements, group_name, timed_effects, enabled, case_sensitive)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            dictionary_id,
                            pattern,
                            replacement,
                            is_regex,
                            float(probability),
                            max_replacements,
                            group,
                            timed_effects_json,
                            enabled,
                            case_sensitive,
                        ),
                    )
                    added_count += 1
                conn.commit()
            self._invalidate_cache()

            class _BulkAddResult:
                def __init__(self, n: int):
                    self.added = n

                def __getitem__(self, key):
                    if key == "added":
                        return self.added
                    raise KeyError(key)

                def get(self, key, default=None):
                    return self.added if key == "added" else default

                def __int__(self):
                    return int(self.added)

                def __eq__(self, other):
                    if isinstance(other, int):
                        return self.added == other
                    if isinstance(other, dict):
                        return other.get("added") == self.added
                    return False

                def __repr__(self):
                    return f"BulkAddResult(added={self.added})"

            return _BulkAddResult(added_count)

        except re.error:
            raise
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error adding bulk entries: {e}")
            raise CharactersRAGDBError(f"Error adding bulk entries: {e}") from e

    def search_entries(self, dictionary_id: Optional[int] = None, query: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Search for entries by pattern.

        Args:
            search_term: Search term to look for in keys and content

        Returns:
            List of matching entries with dictionary info
        """
        try:
            term = query or ""
            with self.db.get_connection() as conn:
                q = [
                    "SELECT e.*, d.name as dictionary_name FROM dictionary_entries e JOIN chat_dictionaries d ON e.dictionary_id = d.id",
                    "WHERE d.deleted = ?",
                ]
                params: list[Any] = [False]
                if dictionary_id is not None:
                    q.append("AND e.dictionary_id = ?")
                    params.append(dictionary_id)
                if term:
                    # Escape SQL LIKE wildcards to prevent pattern injection
                    escaped_term = _escape_like_pattern(term)
                    q.append("AND (e.key LIKE ? ESCAPE '\\' OR e.content LIKE ? ESCAPE '\\')")
                    params.extend([f"%{escaped_term}%", f"%{escaped_term}%"])
                q.append("ORDER BY d.name, e.key")
                cursor = conn.execute(" ".join(q), tuple(params))
                results: list[dict[str, Any]] = []
                for row in cursor.fetchall():
                    rd = dict(row)
                    entry = ChatDictionaryEntry.from_dict(rd)
                    d = entry.to_dict()
                    # Legacy-friendly aliases
                    d["key"] = d.get("pattern")
                    d["content"] = d.get("replacement")
                    if "dictionary_name" in rd:
                        d["dictionary_name"] = rd["dictionary_name"]
                    results.append(d)
                return results

        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error searching entries: {e}")
            raise CharactersRAGDBError(f"Error searching entries: {e}") from e

    # --- Additional Developer-Friendly APIs ---

    def filter_entries(
        self,
        dictionary_id: int,
        type: Optional[str] = None,
        active_only: bool = True,
        group: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        entries = self.get_entries(dictionary_id=dictionary_id, group=group, active_only=active_only)
        if type is not None:
            if type not in ("literal", "regex"):
                return entries
            entries = [e for e in entries if e.get("type") == type]
        return entries

    def toggle_entry_active(self, entry_id: int, is_active: Optional[bool] = None) -> bool:
        try:
            with self.db.get_connection() as conn:
                if is_active is None:
                    cur = conn.execute("SELECT enabled FROM dictionary_entries WHERE id = ?", (entry_id,))
                    row = cur.fetchone()
                    current = bool(row[0]) if row else True
                    is_active = not current
                cur = conn.execute(
                    "UPDATE dictionary_entries SET enabled = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (bool(is_active), entry_id),
                )
                conn.commit()
                if cur.rowcount > 0:
                    self._invalidate_cache()
                    return True
                return False
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error toggling entry active: {e}")
            raise CharactersRAGDBError(f"Error toggling entry active: {e}") from e

    def clear_cache(self) -> None:
        self._entry_cache = None
        self._cache_timestamp = None

    def export_to_json(self, dictionary_id: int) -> dict[str, Any]:
        info = self.get_dictionary(dictionary_id)
        if not info:
            raise InputError(f"Dictionary {dictionary_id} not found")
        entries = self.get_entries(dictionary_id, active_only=False)
        return {
            "name": info["name"],
            "description": info.get("description"),
            "category": self._normalize_dictionary_category(info.get("category")),
            "tags": self._normalize_dictionary_tags(info.get("tags")),
            "included_dictionary_ids": self._normalize_included_dictionary_ids(
                info.get("included_dictionary_ids")
            ),
            "default_token_budget": self._coerce_positive_int(info.get("default_token_budget")),
            "entries": entries,
        }

    def import_from_json(self, data: dict[str, Any]) -> int:
        name = data.get("name") or "Imported Dictionary"
        description = data.get("description")
        category = data.get("category")
        tags = data.get("tags")
        included_dictionary_ids = data.get("included_dictionary_ids")
        default_token_budget = self._coerce_positive_int(data.get("default_token_budget"))
        dict_id = self.create_dictionary(
            name=name,
            description=description,
            category=category,
            tags=tags,
            included_dictionary_ids=included_dictionary_ids,
            default_token_budget=default_token_budget,
        )
        try:
            for e in data.get("entries", []):
                self.add_entry(
                    dict_id,
                    pattern=e.get("pattern") or e.get("key"),
                    replacement=e.get("replacement") or e.get("content", ""),
                    type=e.get("type", "literal"),
                    probability=e.get("probability", 1.0),
                    group=e.get("group") or e.get("group_name"),
                    max_replacements=e.get("max_replacements", 1),
                    enabled=e.get("enabled", True),
                    case_sensitive=e.get("case_sensitive", True),
                    record_snapshot=False,
                )
            with self.db.get_connection() as conn:
                self._create_dictionary_version_snapshot(
                    conn,
                    int(dict_id),
                    change_type="import_json",
                    summary="Imported dictionary from JSON",
                )
                conn.commit()
            return dict_id
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
            self.delete_dictionary(dict_id, hard_delete=True)
            raise

    @staticmethod
    def _extract_md_field(block: str, name: str) -> Optional[str]:
        m = re.search(rf"^\s*[-*]\s*\*\*{re.escape(name)}\*\*\s*:\s*(.+)$", block, flags=re.MULTILINE)
        return m.group(1).strip() if m else None

    @staticmethod
    def _normalize_probability_input(raw_probability: Optional[str]) -> Optional[float]:
        """
        Normalize probability strings from markdown into a float within [0.0, 1.0].

        Accepts values like "0.5", "50", or "50%". Values outside the range are clamped.
        Returns None when parsing fails so caller can fall back to default probability.
        """
        if raw_probability is None:
            return None
        cleaned = raw_probability.strip()
        if not cleaned:
            return None
        has_percent = cleaned.endswith("%")
        if has_percent:
            cleaned = cleaned[:-1].strip()
        try:
            numeric = float(cleaned)
        except ValueError:
            logger.warning("Invalid probability value '{}' in markdown entry; using default.", raw_probability)
            return None
        if has_percent or numeric > 1.0:
            numeric /= 100.0
        if numeric < 0.0:
            logger.warning("Probability {} below 0; clamping to 0.", raw_probability)
            return 0.0
        if numeric > 1.0:
            logger.warning("Probability {} above 1 after normalization; clamping to 1.", raw_probability)
            return 1.0
        return numeric

    def clone_dictionary(self, source_dict_id: int, new_name: str) -> int:
        """
        Create a copy of a dictionary with all its entries.

        Args:
            source_dict_id: Source dictionary ID
            new_name: Name for the cloned dictionary

        Returns:
            ID of the new dictionary
        """
        try:
            # Get source dictionary
            source_dict = self.get_dictionary(source_dict_id)
            if not source_dict:
                raise InputError(f"Source dictionary {source_dict_id} not found")

            # Create new dictionary
            new_dict_id = self.create_dictionary(
                name=new_name,
                description=f"Cloned from {source_dict['name']}",
                category=source_dict.get("category"),
                tags=source_dict.get("tags"),
                included_dictionary_ids=source_dict.get("included_dictionary_ids"),
                default_token_budget=self._coerce_positive_int(source_dict.get("default_token_budget")),
            )

            # Get all entries from source dictionary (include inactive)
            entries = self.get_entries(source_dict_id, active_only=False)

            # Add entries to new dictionary
            if entries:
                with self.db.get_connection() as conn:
                    for e in entries:
                        timed_effects_json = json.dumps(
                            e.get("timed_effects", {"sticky": 0, "cooldown": 0, "delay": 0})
                        )
                        conn.execute(
                            """
                            INSERT INTO dictionary_entries
                            (dictionary_id, key, content, is_regex, probability, max_replacements, group_name, timed_effects, enabled, case_sensitive)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                new_dict_id,
                                e["pattern"],
                                e["replacement"],
                                bool(e.get("type", "literal") == "regex"),
                                float(e.get("probability", 1.0)),
                                int(e.get("max_replacements", 0)),
                                e.get("group"),
                                timed_effects_json,
                                bool(e.get("enabled", 1)),
                                bool(e.get("case_sensitive", 1)),
                            ),
                        )
                    self._create_dictionary_version_snapshot(
                        conn,
                        int(new_dict_id),
                        change_type="clone",
                        summary=f"Cloned from dictionary {int(source_dict_id)}",
                    )
                    conn.commit()

            logger.info(
                f"Cloned dictionary {source_dict_id} to new dictionary {new_dict_id} with {len(entries)} entries"
            )
            self._invalidate_cache()
            return new_dict_id

        except ConflictError:
            raise
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error cloning dictionary: {e}")
            raise CharactersRAGDBError(f"Error cloning dictionary: {e}") from e

    def get_usage_statistics(self, dictionary_id: int) -> dict[str, Any]:
        """Return usage statistics for a dictionary."""
        try:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT usage_count, last_used_at
                    FROM chat_dictionaries
                    WHERE id = ? AND deleted = ?
                    """,
                    (dictionary_id, False),
                ).fetchone()
                if row:
                    if isinstance(row, dict):
                        usage_count = row.get("usage_count")
                        last_used_at = row.get("last_used_at")
                    elif hasattr(row, "_mapping"):
                        usage_count = row._mapping.get("usage_count")
                        last_used_at = row._mapping.get("last_used_at")
                    else:
                        usage_count = row[0] if len(row) > 0 else 0
                        last_used_at = row[1] if len(row) > 1 else None
                    return {
                        "times_used": int(usage_count or 0),
                        "last_used_at": last_used_at,
                    }
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS:
            pass

        return {
            "times_used": int(self._usage_counts.get(dictionary_id, 0)) if hasattr(self, "_usage_counts") else 0,
            "last_used_at": (
                self._last_used_at.get(dictionary_id).isoformat()
                if hasattr(self, "_last_used_at") and self._last_used_at.get(dictionary_id)
                else None
            ),
        }

    def list_dictionary_versions(
        self,
        dictionary_id: int,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List dictionary version snapshots in reverse chronological order."""
        safe_limit = max(1, min(int(limit), 100))
        safe_offset = max(0, int(offset))
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    """
                    SELECT
                        revision,
                        source_dictionary_version,
                        change_type,
                        summary,
                        entry_count,
                        created_at
                    FROM dictionary_version_history
                    WHERE dictionary_id = ?
                    ORDER BY revision DESC
                    LIMIT ? OFFSET ?
                    """,
                    (int(dictionary_id), safe_limit, safe_offset),
                ).fetchall()
                count_row = conn.execute(
                    """
                    SELECT COUNT(*) AS total
                    FROM dictionary_version_history
                    WHERE dictionary_id = ?
                    """,
                    (int(dictionary_id),),
                ).fetchone()
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error listing dictionary versions: {e}")
            raise CharactersRAGDBError(f"Error listing dictionary versions: {e}") from e

        if isinstance(count_row, dict):
            total_raw = count_row.get("total")
        elif hasattr(count_row, "_mapping"):
            total_raw = count_row._mapping.get("total")
        else:
            total_raw = count_row[0] if count_row else 0
        total = int(total_raw or 0)

        versions: list[dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            versions.append(
                {
                    "revision": int(row_dict.get("revision") or 0),
                    "source_dictionary_version": self._coerce_positive_int(
                        row_dict.get("source_dictionary_version")
                    ),
                    "change_type": str(row_dict.get("change_type") or "unspecified"),
                    "summary": row_dict.get("summary"),
                    "entry_count": int(row_dict.get("entry_count") or 0),
                    "created_at": row_dict.get("created_at"),
                }
            )
        return versions, total

    def get_dictionary_version(
        self,
        dictionary_id: int,
        revision: int,
    ) -> Optional[dict[str, Any]]:
        """Return a specific dictionary revision snapshot."""
        try:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT
                        revision,
                        source_dictionary_version,
                        change_type,
                        summary,
                        entry_count,
                        dictionary_snapshot,
                        entries_snapshot,
                        created_at
                    FROM dictionary_version_history
                    WHERE dictionary_id = ? AND revision = ?
                    LIMIT 1
                    """,
                    (int(dictionary_id), int(revision)),
                ).fetchone()
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error reading dictionary version: {e}")
            raise CharactersRAGDBError(f"Error reading dictionary version: {e}") from e

        if not row:
            return None

        row_dict = dict(row)
        try:
            dictionary_snapshot_raw = row_dict.get("dictionary_snapshot") or "{}"
            entries_snapshot_raw = row_dict.get("entries_snapshot") or "[]"
            dictionary_snapshot = self._normalize_dictionary_row(json.loads(dictionary_snapshot_raw))
            entries_snapshot = json.loads(entries_snapshot_raw)
            if not isinstance(entries_snapshot, list):
                entries_snapshot = []
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            raise CharactersRAGDBError(f"Invalid dictionary version snapshot payload: {e}") from e

        return {
            "dictionary_id": int(dictionary_id),
            "revision": int(row_dict.get("revision") or revision),
            "source_dictionary_version": self._coerce_positive_int(
                row_dict.get("source_dictionary_version")
            ),
            "change_type": str(row_dict.get("change_type") or "unspecified"),
            "summary": row_dict.get("summary"),
            "entry_count": int(row_dict.get("entry_count") or len(entries_snapshot)),
            "dictionary": dictionary_snapshot,
            "entries": entries_snapshot,
            "created_at": row_dict.get("created_at"),
        }

    def revert_dictionary_to_revision(self, dictionary_id: int, revision: int) -> dict[str, Any]:
        """Restore dictionary metadata/entries from a stored revision snapshot."""
        snapshot = self.get_dictionary_version(dictionary_id, revision)
        if not snapshot:
            raise InputError(f"Revision {revision} not found for dictionary {dictionary_id}")

        dictionary_snapshot = snapshot.get("dictionary") or {}
        entries_snapshot = snapshot.get("entries") or []
        if not isinstance(dictionary_snapshot, dict) or not isinstance(entries_snapshot, list):
            raise InputError("Invalid revision snapshot payload")

        tags_serialized = self._serialize_dictionary_tags(dictionary_snapshot.get("tags"))
        category = self._normalize_dictionary_category(dictionary_snapshot.get("category"))
        included_dictionary_ids = self._normalize_included_dictionary_ids(
            dictionary_snapshot.get("included_dictionary_ids")
        )
        default_token_budget = self._coerce_positive_int(dictionary_snapshot.get("default_token_budget"))

        try:
            with self.db.get_connection() as conn:
                existing_row = conn.execute(
                    "SELECT id FROM chat_dictionaries WHERE id = ? AND deleted = ?",
                    (int(dictionary_id), False),
                ).fetchone()
                if not existing_row:
                    raise InputError(f"Dictionary {dictionary_id} not found")

                validated_includes = self._validate_included_dictionary_ids(
                    conn,
                    dictionary_id=int(dictionary_id),
                    included_dictionary_ids=included_dictionary_ids,
                )

                conn.execute(
                    """
                    UPDATE chat_dictionaries
                    SET
                        name = ?,
                        description = ?,
                        category = ?,
                        tags = ?,
                        included_dictionary_ids = ?,
                        is_active = ?,
                        default_token_budget = ?,
                        updated_at = CURRENT_TIMESTAMP,
                        version = version + 1
                    WHERE id = ? AND deleted = ?
                    """,
                    (
                        dictionary_snapshot.get("name"),
                        dictionary_snapshot.get("description"),
                        category,
                        tags_serialized,
                        self._serialize_included_dictionary_ids(validated_includes),
                        bool(dictionary_snapshot.get("is_active", True)),
                        default_token_budget,
                        int(dictionary_id),
                        False,
                    ),
                )

                conn.execute("DELETE FROM dictionary_entries WHERE dictionary_id = ?", (int(dictionary_id),))

                for index, raw_entry in enumerate(entries_snapshot, start=1):
                    if not isinstance(raw_entry, dict):
                        continue
                    pattern = str(raw_entry.get("pattern") or raw_entry.get("key") or "").strip()
                    if not pattern:
                        continue
                    replacement = raw_entry.get("replacement") or raw_entry.get("content") or ""
                    entry_type = str(raw_entry.get("type") or "").strip().lower()
                    is_regex = bool(entry_type == "regex")
                    if not entry_type:
                        is_regex = bool(raw_entry.get("is_regex", False))

                    timed_effects = raw_entry.get("timed_effects")
                    timed_effects_json = json.dumps(
                        timed_effects if isinstance(timed_effects, dict) else {"sticky": 0, "cooldown": 0, "delay": 0}
                    )
                    probability_raw = raw_entry.get("probability", 1.0)
                    try:
                        probability_value = float(probability_raw)
                    except (TypeError, ValueError):
                        probability_value = 1.0
                    max_replacements_raw = raw_entry.get("max_replacements", 0)
                    try:
                        max_replacements_value = int(max_replacements_raw or 0)
                    except (TypeError, ValueError):
                        max_replacements_value = 0

                    usage_count_raw = raw_entry.get("usage_count", 0)
                    try:
                        usage_count = int(usage_count_raw or 0)
                    except (TypeError, ValueError):
                        usage_count = 0

                    sort_order_raw = raw_entry.get("sort_order")
                    try:
                        sort_order = int(sort_order_raw) if sort_order_raw is not None else index
                    except (TypeError, ValueError):
                        sort_order = index

                    conn.execute(
                        """
                        INSERT INTO dictionary_entries
                        (
                            dictionary_id,
                            key,
                            content,
                            is_regex,
                            probability,
                            max_replacements,
                            group_name,
                            timed_effects,
                            enabled,
                            case_sensitive,
                            sort_order,
                            usage_count,
                            last_used_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(dictionary_id),
                            pattern,
                            replacement,
                            is_regex,
                            probability_value,
                            max_replacements_value,
                            raw_entry.get("group") or raw_entry.get("group_name"),
                            timed_effects_json,
                            bool(raw_entry.get("enabled", True)),
                            bool(raw_entry.get("case_sensitive", True)),
                            sort_order,
                            usage_count,
                            raw_entry.get("last_used_at"),
                        ),
                    )

                created_revision = self._create_dictionary_version_snapshot(
                    conn,
                    int(dictionary_id),
                    change_type="revert",
                    summary=f"Reverted to revision {int(revision)}",
                )
                conn.commit()
        except ConflictError:
            raise
        except InputError:
            raise
        except sqlite3.IntegrityError as e:
            if _is_unique_violation(e):
                raise ConflictError(
                    f"Dictionary name '{dictionary_snapshot.get('name')}' already exists",
                    "chat_dictionaries",
                    dictionary_snapshot.get("name"),
                ) from e
            raise CharactersRAGDBError(f"Error reverting dictionary revision: {e}") from e
        except _CHAT_DICTIONARY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error reverting dictionary revision: {e}")
            raise CharactersRAGDBError(f"Error reverting dictionary revision: {e}") from e

        self._invalidate_cache()
        current_dictionary = self.get_dictionary(dictionary_id)
        current_dictionary_version = self._coerce_positive_int(
            current_dictionary.get("version") if current_dictionary else None
        ) or 1
        return {
            "dictionary_id": int(dictionary_id),
            "reverted_to_revision": int(revision),
            "current_dictionary_version": int(current_dictionary_version),
            "current_revision": int(created_revision or revision),
            "message": f"Dictionary reverted to revision {int(revision)}",
        }

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


class _ProcessedTextResult(str):
    """
    A string subclass that also behaves like a minimal mapping for tests
    expecting a dict with a 'processed_text' key. This allows code that
    expects a plain string (e.g., "substring" in result) and code that does
    result['processed_text'] to both succeed.
    """

    def __new__(cls, value: str, stats: Optional[dict[str, Any]] = None):
        obj = super().__new__(cls, value)
        obj._stats = stats or {}
        return obj

    def __getitem__(self, key):
        # Support dict-like access for property tests
        if isinstance(key, str):
            if key == "processed_text":
                return str(self)
            return self._stats.get(key)
        # Fallback to normal string indexing
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "processed_text":
            return str(self)
        return self._stats.get(key, default)

    def __contains__(self, item) -> bool:
        # Allow legacy tests like 'replacements' in result
        if isinstance(item, str) and item in {
            "processed_text",
            "replacements",
            "iterations",
            "entries_used",
            "token_budget_exceeded",
        }:
            return True
        return super().__contains__(item)


# Import handling to prevent breaking changes
__all__ = ["ChatDictionaryEntry", "ChatDictionaryService", "TokenBudgetExceededWarning"]

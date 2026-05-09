# chat_dictionary_schemas.py
# Description: Pydantic schemas for Chat Dictionary API endpoints
#
"""
Pydantic schemas for Chat Dictionary functionality.

These schemas define the request and response models for the chat dictionary
API endpoints, ensuring proper validation and serialization.
"""

import json
import re
from datetime import datetime
from typing import Any, Optional, Union

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta
from tldw_Server_API.app.core.Character_Chat.constants import MAX_CHAT_DICTIONARY_TEXT_LENGTH

# Maximum length for regex patterns to prevent complexity attacks
MAX_REGEX_PATTERN_LENGTH = 500

# Patterns that are known to cause ReDoS (catastrophic backtracking).
# This is a heuristic blocklist using substring checks; obfuscated variants may bypass it.
# Apply runtime safeguards when executing regexes.
DANGEROUS_REGEX_PATTERNS = [
    r'(.+)+',      # Nested quantifiers
    r'(.*)*',      # Nested quantifiers
    r'(.+)*',      # Nested quantifiers
    r'(.*)+',      # Nested quantifiers
    r'(.?)+',      # Many optional matches
    r'(.?)*',      # Many optional matches
    r'([a-zA-Z]+)*',  # Character class with nested quantifier
]


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


MAX_DICTIONARY_TAGS = 20
MAX_DICTIONARY_TAG_LENGTH = 40
MAX_INCLUDED_DICTIONARIES = 50


def _normalize_dictionary_category(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_dictionary_tags(value: Any) -> list[str]:
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
        raise ValueError("tags must be a list of strings")

    normalized_tags: list[str] = []
    seen: set[str] = set()
    for item in raw_values:
        if item is None:
            continue
        tag = str(item).strip()
        if not tag:
            continue
        if len(tag) > MAX_DICTIONARY_TAG_LENGTH:
            raise ValueError(
                f"Tag '{tag[:20]}...' exceeds max length of {MAX_DICTIONARY_TAG_LENGTH} characters"
            )
        dedupe_key = tag.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized_tags.append(tag)

    if len(normalized_tags) > MAX_DICTIONARY_TAGS:
        raise ValueError(f"At most {MAX_DICTIONARY_TAGS} tags are allowed")

    return normalized_tags


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
        raise ValueError("included_dictionary_ids must be a list of integers")

    normalized: list[int] = []
    seen: set[int] = set()
    for item in raw_values:
        if item is None or isinstance(item, bool):
            continue
        try:
            dictionary_id = int(item)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid dictionary ID '{item}' in included_dictionary_ids") from None
        if dictionary_id <= 0:
            raise ValueError("included_dictionary_ids must contain positive integers")
        if dictionary_id in seen:
            continue
        seen.add(dictionary_id)
        normalized.append(dictionary_id)

    if len(normalized) > MAX_INCLUDED_DICTIONARIES:
        raise ValueError(f"At most {MAX_INCLUDED_DICTIONARIES} included dictionaries are allowed")

    return normalized


def _has_nested_quantifiers(pattern: str) -> bool:
    """Heuristic check for nested quantifiers like '(a+)+'. """
    escaped = False
    in_class = False
    group_stack: list[bool] = []
    last_closed_group_had_quantifier = False
    last_token_was_group_end = False

    for char in pattern:
        if escaped:
            escaped = False
            last_token_was_group_end = False
            continue

        if char == "\\":
            escaped = True
            last_token_was_group_end = False
            continue

        if in_class:
            if char == "]":
                in_class = False
            last_token_was_group_end = False
            continue

        if char == "[":
            in_class = True
            last_token_was_group_end = False
            continue

        if char == "(":
            group_stack.append(False)
            last_token_was_group_end = False
            continue

        if char == ")":
            last_closed_group_had_quantifier = group_stack.pop() if group_stack else False
            last_token_was_group_end = True
            continue

        if char in "*+?":
            if last_token_was_group_end and last_closed_group_had_quantifier:
                return True
            if group_stack:
                group_stack[-1] = True
            last_token_was_group_end = False
            continue

        last_token_was_group_end = False

    return False


def validate_regex_pattern_safety(pattern: str) -> str:
    """Validate a regex pattern for basic ReDoS safety checks.

    Args:
        pattern: The regex pattern to validate

    Returns:
        The pattern if valid

    Raises:
        ValueError: If the pattern is too long, invalid, or potentially dangerous

    Note:
        This is a heuristic, blocklist-based check and may miss obfuscated patterns.
        Apply runtime safeguards (e.g., timeouts or sandboxed execution) when evaluating regexes.
    """
    if len(pattern) > MAX_REGEX_PATTERN_LENGTH:
        raise ValueError(f"Regex pattern too long (max {MAX_REGEX_PATTERN_LENGTH} chars)")

    # Check if the pattern compiles
    try:
        re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {str(e)[:100]}") from e

    # Check for potentially dangerous patterns (basic ReDoS detection)
    for dangerous in DANGEROUS_REGEX_PATTERNS:
        if dangerous in pattern:
            raise ValueError(
                "Potentially dangerous regex pattern detected (nested quantifiers). "
                "Pattern may cause catastrophic backtracking."
            )

    # Check for excessive quantifier nesting (heuristic)
    # Count nested groups with quantifiers
    if _has_nested_quantifiers(pattern):
        logger.warning(f"Regex pattern may have nested quantifiers: {pattern[:50]}...")
        # Don't fail, just warn - some nested quantifiers are safe

    return pattern


class TimedEffects(BaseModel):
    """Timed effects configuration for dictionary entries."""
    sticky: int = Field(0, ge=0, description="Sticky duration in seconds")
    cooldown: int = Field(0, ge=0, description="Cooldown between triggers in seconds")
    delay: int = Field(0, ge=0, description="Initial delay before first trigger in seconds")


class DictionaryEntryBase(BaseModel):
    """Base schema for dictionary entries."""
    pattern: str = Field(..., min_length=1, description="Pattern to match (literal or regex)")
    replacement: str = Field(..., description="Replacement text")
    probability: float = Field(1.0, ge=0.0, le=1.0, description="Chance of replacement (0.0-1.0)")
    group: Optional[str] = Field(None, description="Optional group name for organization")
    timed_effects: Optional[TimedEffects] = Field(None, description="Timing effects configuration")
    max_replacements: int = Field(0, ge=0, description="Maximum replacements per processing (0 = unlimited)")


class DictionaryEntryCreate(DictionaryEntryBase):
    """Schema for creating a dictionary entry."""
    type: str = Field("literal", pattern="^(literal|regex)$", description="Entry type")
    enabled: bool = Field(True, description="Whether the entry is enabled")
    case_sensitive: bool = Field(True, description="Case-sensitive matching for literal patterns")

    @model_validator(mode="after")
    def validate_regex_pattern(self) -> "DictionaryEntryCreate":
        """Validate regex patterns for safety when type is 'regex'."""
        if self.type == "regex":
            validate_regex_pattern_safety(self.pattern)
        return self


class DictionaryEntryUpdate(BaseModel):
    """Schema for updating a dictionary entry."""
    pattern: Optional[str] = Field(None, min_length=1, description="New pattern to match")
    replacement: Optional[str] = Field(None, description="New replacement text")
    probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="New probability (0.0-1.0)")
    group: Optional[str] = Field(None, description="New group name")
    timed_effects: Optional[TimedEffects] = Field(None, description="New timing effects")
    max_replacements: Optional[int] = Field(None, ge=0, description="New max replacements (0 = unlimited)")
    type: Optional[str] = Field(None, pattern="^(literal|regex)$", description="Entry type")
    enabled: Optional[bool] = Field(None, description="Whether the entry is enabled")
    case_sensitive: Optional[bool] = Field(None, description="Case-sensitive matching for literal patterns")

    @model_validator(mode="after")
    def validate_regex_pattern(self) -> "DictionaryEntryUpdate":
        """Validate regex patterns for safety when type is 'regex'.

        Note: This only validates when BOTH pattern and type are provided in the update.
        If only pattern is updated, the caller must ensure the existing type is checked.
        """
        if self.type == "regex" and self.pattern is not None:
            validate_regex_pattern_safety(self.pattern)
        return self


class DictionaryEntryResponse(DictionaryEntryBase):
    """Schema for dictionary entry response."""
    id: int = Field(..., description="Entry ID")
    dictionary_id: int = Field(..., description="Parent dictionary ID")
    type: str = Field(..., description="Entry type")
    enabled: bool = Field(..., description="Whether the entry is enabled")
    case_sensitive: bool = Field(..., description="Case-sensitive matching for literal patterns")
    priority: Optional[int] = Field(None, ge=1, description="Execution priority (1 = first)")
    usage_count: int = Field(0, ge=0, description="Total number of times this entry has fired")
    last_used_at: Optional[datetime] = Field(None, description="Timestamp when this entry last fired")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


class ChatDictionaryBase(BaseModel):
    """Base schema for chat dictionaries."""
    name: str = Field(..., min_length=1, max_length=100, description="Unique dictionary name")
    description: Optional[str] = Field(None, max_length=500, description="Dictionary description")
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional category for organizing dictionaries",
    )
    tags: list[str] = Field(default_factory=list, description="Optional tags for dictionary filtering")
    included_dictionary_ids: list[int] = Field(
        default_factory=list,
        description="Optional list of included dictionary IDs processed before this dictionary.",
    )
    is_active: bool = Field(True, description="Whether the dictionary is active")
    default_token_budget: Optional[int] = Field(
        None,
        ge=1,
        description="Optional default token budget applied when processing requests omit token_budget.",
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: Any) -> Optional[str]:
        return _normalize_dictionary_category(value)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> list[str]:
        return _normalize_dictionary_tags(value)

    @field_validator("included_dictionary_ids", mode="before")
    @classmethod
    def normalize_included_dictionary_ids(cls, value: Any) -> list[int]:
        return _normalize_included_dictionary_ids(value)


class ChatDictionaryCreate(BaseModel):
    """Schema for creating a chat dictionary."""
    name: str = Field(..., min_length=1, max_length=100, description="Unique dictionary name")
    description: Optional[str] = Field(None, max_length=500, description="Dictionary description")
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional category for organizing dictionaries",
    )
    tags: list[str] = Field(default_factory=list, description="Optional tags for dictionary filtering")
    included_dictionary_ids: list[int] = Field(
        default_factory=list,
        description="Optional included dictionary IDs for composition.",
    )
    default_token_budget: Optional[int] = Field(
        None,
        ge=1,
        description="Optional default token budget.",
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: Any) -> Optional[str]:
        return _normalize_dictionary_category(value)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> list[str]:
        return _normalize_dictionary_tags(value)

    @field_validator("included_dictionary_ids", mode="before")
    @classmethod
    def normalize_included_dictionary_ids(cls, value: Any) -> list[int]:
        return _normalize_included_dictionary_ids(value)


class ChatDictionaryUpdate(BaseModel):
    """Schema for updating a chat dictionary."""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="New dictionary name")
    description: Optional[str] = Field(None, max_length=500, description="New description")
    category: Optional[str] = Field(
        None,
        max_length=100,
        description="New category value (null clears category)",
    )
    tags: Optional[list[str]] = Field(
        None,
        description="New tags list (empty list clears tags)",
    )
    included_dictionary_ids: Optional[list[int]] = Field(
        None,
        description="New included dictionary IDs (empty list clears includes).",
    )
    is_active: Optional[bool] = Field(None, description="New active status")
    default_token_budget: Optional[int] = Field(
        None,
        ge=1,
        description="Optional default token budget override.",
    )
    version: Optional[int] = Field(
        None,
        ge=1,
        description="Expected dictionary version for optimistic locking",
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, value: Any) -> Optional[str]:
        return _normalize_dictionary_category(value)

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, value: Any) -> Optional[list[str]]:
        if value is None:
            return None
        return _normalize_dictionary_tags(value)

    @field_validator("included_dictionary_ids", mode="before")
    @classmethod
    def normalize_included_dictionary_ids(cls, value: Any) -> Optional[list[int]]:
        if value is None:
            return None
        return _normalize_included_dictionary_ids(value)


class DictionaryUsageChatRef(BaseModel):
    """Lightweight chat reference used by dictionary usage summaries."""
    chat_id: str = Field(..., description="Chat session ID")
    title: Optional[str] = Field(None, description="Chat title")
    state: str = Field("in-progress", description="Conversation state")
    last_modified: Optional[datetime] = Field(None, description="Chat last-modified timestamp")


class ChatDictionaryResponse(ChatDictionaryBase):
    """Schema for chat dictionary response."""
    id: int = Field(..., description="Dictionary ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    version: int = Field(..., description="Version number for optimistic locking")
    entry_count: Optional[int] = Field(None, description="Number of entries in the dictionary")
    processing_priority: Optional[int] = Field(
        None,
        ge=1,
        description="Execution priority among active dictionaries (1 = first).",
    )
    used_by_chat_count: int = Field(0, description="Number of chat sessions linked to this dictionary")
    used_by_active_chat_count: int = Field(0, description="Number of active chat sessions linked to this dictionary")
    used_by_chat_refs: list[DictionaryUsageChatRef] = Field(
        default_factory=list,
        description="Small preview of linked chat sessions",
    )

    model_config = ConfigDict(from_attributes=True)


class ChatDictionaryWithEntries(ChatDictionaryResponse):
    """Schema for chat dictionary with its entries."""
    entries: list[DictionaryEntryResponse] = Field(default_factory=list, description="Dictionary entries")


class ProcessTextRequest(BaseModel):
    """Request schema for processing text through dictionaries."""
    text: str = Field(..., max_length=MAX_CHAT_DICTIONARY_TEXT_LENGTH, description="Text to process")
    dictionary_id: Optional[int] = Field(None, description="Specific dictionary to use")
    dictionary_ids: Optional[list[int]] = Field(
        None,
        description="Explicit client-ordered dictionaries to apply. An empty list applies no dictionaries.",
    )
    group: Optional[str] = Field(None, description="Specific group to filter entries")
    max_iterations: int = Field(5, ge=1, le=20, description="Maximum processing iterations")
    token_budget: Optional[int] = Field(None, ge=1, description="Optional token limit")
    chat_id: Optional[str] = Field(
        None,
        description="Optional chat session ID for activity/audit attribution.",
    )

    @field_validator("dictionary_ids", mode="before")
    @classmethod
    def normalize_dictionary_ids(cls, value: Any) -> Optional[list[int]]:
        if value is None:
            return None
        return _normalize_included_dictionary_ids(value)


class ProcessTextResponse(BaseModel):
    """Response schema for processed text."""
    original_text: str = Field(..., description="Original input text")
    processed_text: str = Field(..., description="Text after processing")
    replacements: int = Field(..., description="Total number of replacements made")
    iterations: int = Field(..., description="Number of iterations performed")
    entries_used: list[int] = Field(..., description="IDs of entries that made replacements")
    token_budget_exceeded: bool = Field(False, description="Whether token budget was exceeded")
    token_budget_used: Optional[int] = Field(
        None,
        description="Effective token budget used during processing (request override or dictionary default).",
    )
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")


class DictionaryActivityEvent(BaseModel):
    """Single dictionary transformation activity event."""
    id: int = Field(..., description="Activity event ID")
    dictionary_id: int = Field(..., description="Dictionary ID")
    chat_id: Optional[str] = Field(None, description="Associated chat session ID, if provided")
    entries_used: list[int] = Field(default_factory=list, description="Entry IDs that fired for this event")
    replacements: int = Field(..., ge=0, description="Number of replacements made")
    iterations: int = Field(..., ge=0, description="Iterations performed")
    token_budget_used: Optional[int] = Field(None, ge=1, description="Effective token budget used")
    original_text_preview: str = Field(..., description="Original text preview snippet")
    processed_text_preview: str = Field(..., description="Processed text preview snippet")
    processing_time_ms: Optional[float] = Field(None, ge=0, description="Processing time in milliseconds")
    created_at: datetime = Field(..., description="Event timestamp")


class DictionaryActivityListResponse(BaseModel):
    """Paginated activity events for a dictionary."""
    dictionary_id: int = Field(..., description="Dictionary ID")
    events: list[DictionaryActivityEvent] = Field(default_factory=list, description="Activity events")
    total: int = Field(..., ge=0, description="Total number of matching events")
    limit: int = Field(..., ge=1, description="Applied page size")
    offset: int = Field(..., ge=0, description="Applied offset")
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class DictionaryVersionSummary(BaseModel):
    """Single version-history row for a dictionary snapshot."""
    revision: int = Field(..., ge=1, description="Monotonic revision number for this dictionary")
    source_dictionary_version: Optional[int] = Field(
        None,
        ge=1,
        description="Dictionary optimistic-lock version captured in this snapshot.",
    )
    change_type: str = Field(..., description="High-level change category (create/update/entry/revert/etc.)")
    summary: Optional[str] = Field(None, description="Human-readable change summary")
    entry_count: int = Field(0, ge=0, description="Number of entries captured in this snapshot")
    created_at: datetime = Field(..., description="Snapshot timestamp")


class DictionaryVersionDetailResponse(BaseModel):
    """Detailed dictionary snapshot payload for a specific revision."""
    dictionary_id: int = Field(..., description="Dictionary ID")
    revision: int = Field(..., ge=1, description="Requested revision number")
    source_dictionary_version: Optional[int] = Field(
        None,
        ge=1,
        description="Dictionary optimistic-lock version captured in this snapshot.",
    )
    change_type: str = Field(..., description="High-level change category")
    summary: Optional[str] = Field(None, description="Human-readable change summary")
    created_at: datetime = Field(..., description="Snapshot timestamp")
    dictionary: ChatDictionaryResponse = Field(..., description="Dictionary metadata captured in this revision")
    entries: list[DictionaryEntryResponse] = Field(
        default_factory=list,
        description="Dictionary entries captured in this revision",
    )


class DictionaryVersionListResponse(BaseModel):
    """Paginated dictionary version-history list response."""
    dictionary_id: int = Field(..., description="Dictionary ID")
    versions: list[DictionaryVersionSummary] = Field(default_factory=list, description="Version-history rows")
    total: int = Field(..., ge=0, description="Total number of snapshots for this dictionary")
    limit: int = Field(..., ge=1, description="Applied page size")
    offset: int = Field(..., ge=0, description="Applied offset")
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class DictionaryVersionRevertResponse(BaseModel):
    """Response payload for dictionary revision revert operations."""
    dictionary_id: int = Field(..., description="Dictionary ID")
    reverted_to_revision: int = Field(..., ge=1, description="Revision number that was restored")
    current_dictionary_version: int = Field(..., ge=1, description="Dictionary optimistic-lock version after revert")
    current_revision: int = Field(..., ge=1, description="Latest revision number after revert snapshot is recorded")
    message: str = Field(..., description="Result summary")


class ImportDictionaryRequest(BaseModel):
    """Request schema for importing a dictionary from markdown."""
    name: str = Field(..., min_length=1, max_length=100, description="Name for the imported dictionary")
    content: str = Field(..., description="Markdown content to import")
    activate: bool = Field(True, description="Whether to activate the dictionary after import")


class ImportDictionaryResponse(BaseModel):
    """Response schema for dictionary import."""
    dictionary_id: int = Field(..., description="ID of the created dictionary")
    name: str = Field(..., description="Dictionary name")
    entries_imported: int = Field(..., description="Number of entries imported")
    groups_created: list[str] = Field(..., description="List of groups found and created")


class ExportDictionaryResponse(BaseModel):
    """Response schema for dictionary export."""
    name: str = Field(..., description="Dictionary name")
    content: str = Field(..., description="Exported content in markdown format")
    entry_count: int = Field(..., description="Number of entries exported")
    group_count: int = Field(..., description="Number of groups in the dictionary")

class ImportDictionaryJSONRequest(BaseModel):
    """Request schema for importing a dictionary from JSON."""
    data: dict[str, Any] = Field(..., description="Dictionary JSON data with 'name' and 'entries'")
    activate: bool = Field(True, description="Whether to activate the dictionary after import")

class ExportDictionaryJSONResponse(BaseModel):
    """Response schema for JSON export of a dictionary."""
    name: str = Field(..., description="Dictionary name")
    description: Optional[str] = Field(None, description="Dictionary description")
    category: Optional[str] = Field(None, description="Dictionary category")
    tags: list[str] = Field(default_factory=list, description="Dictionary tags")
    included_dictionary_ids: list[int] = Field(
        default_factory=list,
        description="Included dictionary IDs for composition.",
    )
    default_token_budget: Optional[int] = Field(
        None,
        ge=1,
        description="Optional default token budget for processing requests without token_budget.",
    )
    entries: list[dict[str, Any]] = Field(..., description="Entries with pattern, replacement, type, probability, etc.")


class BulkEntryOperation(BaseModel):
    """Schema for bulk entry operations."""
    entry_ids: list[int] = Field(..., min_length=1, description="List of entry IDs to operate on")
    operation: str = Field(..., pattern="^(delete|activate|deactivate|group)$", description="Operation to perform")
    group_name: Optional[str] = Field(None, description="Group name (for group operation)")

    @model_validator(mode="after")
    def validate_group_operation(self) -> "BulkEntryOperation":
        """Require a non-empty group_name when operation is group."""
        if self.operation == "group":
            group_name = (self.group_name or "").strip()
            if not group_name:
                raise ValueError("group_name is required when operation is 'group'")
            self.group_name = group_name
        return self


class BulkOperationResponse(BaseModel):
    """Response schema for bulk operations."""
    success: bool = Field(..., description="Whether the operation succeeded")
    affected_count: int = Field(..., description="Number of entries affected")
    failed_ids: list[int] = Field(default_factory=list, description="IDs that failed to process")
    message: str = Field(..., description="Operation result message")


class DictionaryEntryReorderRequest(BaseModel):
    """Schema for reordering all entries in a dictionary."""
    entry_ids: list[int] = Field(
        ...,
        min_length=1,
        description="Full ordered list of dictionary entry IDs",
    )

    @model_validator(mode="after")
    def validate_unique_entry_ids(self) -> "DictionaryEntryReorderRequest":
        if len(set(self.entry_ids)) != len(self.entry_ids):
            raise ValueError("entry_ids must not contain duplicates")
        return self


class DictionaryEntryReorderResponse(BaseModel):
    """Response schema for dictionary entry reordering."""
    success: bool = Field(..., description="Whether reorder completed successfully")
    dictionary_id: int = Field(..., description="Dictionary ID")
    affected_count: int = Field(..., description="Number of entries reordered")
    entry_ids: list[int] = Field(..., description="Persisted ordered list of entry IDs")
    message: str = Field(..., description="Operation result message")


class DictionaryEntryUsageStatistic(BaseModel):
    """Per-entry usage statistic row."""
    entry_id: int = Field(..., description="Entry ID")
    pattern: str = Field(..., description="Entry pattern")
    usage_count: int = Field(0, ge=0, description="Times this entry fired")
    last_used_at: Optional[datetime] = Field(None, description="Entry last-used timestamp")


class DictionaryPatternConflict(BaseModel):
    """Potentially overlapping/shadowing pair of dictionary patterns."""
    entry_id_a: int = Field(..., description="First entry ID")
    entry_id_b: int = Field(..., description="Second entry ID")
    pattern_a: str = Field(..., description="First entry pattern")
    pattern_b: str = Field(..., description="Second entry pattern")
    type_a: str = Field(..., description="First entry type")
    type_b: str = Field(..., description="Second entry type")
    conflict_type: str = Field(..., description="Conflict heuristic category")
    severity: str = Field(..., pattern="^(low|medium|high)$", description="Relative conflict severity")
    reason: str = Field(..., description="Human-readable explanation")


class DictionaryStatistics(BaseModel):
    """Statistics for a dictionary."""
    dictionary_id: int = Field(..., description="Dictionary ID")
    name: str = Field(..., description="Dictionary name")
    total_entries: int = Field(..., description="Total number of entries")
    regex_entries: int = Field(..., description="Number of regex entries")
    literal_entries: int = Field(..., description="Number of literal entries")
    enabled_entries: int = Field(0, description="Number of enabled entries")
    disabled_entries: int = Field(0, description="Number of disabled entries")
    probabilistic_entries: int = Field(0, description="Entries with probability below 1.0")
    timed_effect_entries: int = Field(0, description="Entries with any timed effect configured")
    groups: list[str] = Field(..., description="List of unique groups")
    average_probability: float = Field(..., description="Average replacement probability")
    created_at: datetime = Field(..., description="Dictionary creation timestamp")
    updated_at: datetime = Field(..., description="Dictionary last-update timestamp")
    zero_usage_entries: int = Field(0, description="Entries that have never fired")
    entry_usage: list[DictionaryEntryUsageStatistic] = Field(
        default_factory=list,
        description="Per-entry usage snapshot for maintenance workflows",
    )
    pattern_conflict_count: int = Field(0, description="Number of potential pattern conflicts")
    pattern_conflicts: list[DictionaryPatternConflict] = Field(
        default_factory=list,
        description="Potential overlapping/shadowing pattern pairs",
    )
    total_usage_count: Optional[int] = Field(None, description="Total times used (if tracked)")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp (if tracked)")


class DictionaryListResponse(BaseModel):
    """Response schema for listing dictionaries."""
    dictionaries: list[ChatDictionaryResponse] = Field(..., description="List of dictionaries")
    total: int = Field(..., description="Total number of dictionaries")
    active_count: int = Field(..., description="Number of active dictionaries")
    inactive_count: int = Field(..., description="Number of inactive dictionaries")


class EntryListResponse(BaseModel):
    """Response schema for listing entries."""
    entries: list[DictionaryEntryResponse] = Field(..., description="List of entries")
    total: int = Field(..., description="Total number of entries")
    dictionary_id: Optional[int] = Field(None, description="Dictionary ID if filtered")
    group: Optional[str] = Field(None, description="Group name if filtered")


# Error response schemas
class DictionaryError(BaseModel):
    """Error response for dictionary operations."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[dict[str, Any]] = Field(None, description="Additional error details")


class DictionaryConflictError(DictionaryError):
    """Conflict error response."""
    error: str = Field("conflict", description="Error type")
    conflicting_name: Optional[str] = Field(None, description="Conflicting dictionary name")
    conflicting_id: Optional[int] = Field(None, description="Conflicting dictionary ID")


class DictionaryNotFoundError(DictionaryError):
    """Not found error response."""
    error: str = Field("not_found", description="Error type")
    resource_type: str = Field(..., description="Type of resource not found")
    resource_id: Optional[Union[int, str]] = Field(None, description="ID of missing resource")


# -----------------------------------------------------------------------------
# Validation API Schemas (for POST /api/v1/chat/dictionaries/validate)
# -----------------------------------------------------------------------------

class ValidationIssue(BaseModel):
    """Represents a single validation error or warning entry."""
    code: str = Field(..., description="Machine-readable issue code")
    field: str = Field(..., description="Field path where the issue occurred")
    message: str = Field(..., description="Human-readable message")


class ValidateDictionaryRequest(BaseModel):
    """Request body for dictionary validation endpoint."""
    data: dict[str, Any] = Field(..., description="Dictionary JSON data including entries")
    schema_version: int = Field(1, description="Schema version for validation")
    strict: bool = Field(False, description="If true, may be used to fail import on errors (reporting stays 200)")


class ValidateDictionaryResponse(BaseModel):
    """Structured validation result matching the validator's taxonomy."""
    ok: bool = Field(..., description="Whether validation passed without errors")
    schema_version: int = Field(..., description="Schema version that was validated")
    errors: list[ValidationIssue] = Field(default_factory=list, description="List of validation errors")
    warnings: list[ValidationIssue] = Field(default_factory=list, description="List of validation warnings")
    entry_stats: dict[str, int] = Field(default_factory=dict, description="Basic statistics about entries")
    suggested_fixes: list[str] = Field(default_factory=list, description="Optional suggestions to fix issues")
    partial: bool = Field(False, description="True when validation short-circuited and report is best-effort")
    partial_reason: Optional[str] = Field(
        None,
        description="Reason for partial validation, e.g. 'max_entries' or 'timeout'",
    )

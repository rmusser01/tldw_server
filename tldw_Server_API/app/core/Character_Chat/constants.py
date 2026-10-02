# constants.py
"""
Shared constants for the Character_Chat module.

This module centralizes magic numbers and configuration defaults used across
the Character_Chat subsystem to ensure consistency and ease of maintenance.
"""

# =============================================================================
# Regex Safety Constants
# =============================================================================
MAX_REGEX_LENGTH = 500
"""Maximum length for regex patterns to prevent ReDoS attacks."""

MAX_REGEX_COMPILE_TIME_MS = 100
"""Maximum time in milliseconds for regex compilation (heuristic-based)."""

MAX_REGEX_MATCH_TIME_MS = 100
"""Maximum time in milliseconds for regex match/replace operations."""

MAX_CHAT_DICTIONARY_TEXT_LENGTH = 400_000
"""Maximum input length for chat dictionary processing."""


# =============================================================================
# Cache Size Constants
# =============================================================================
MAX_ENTRY_CACHE_SIZE = 1000
"""Maximum number of world book entries to cache per service instance."""

MAX_BOOK_CACHE_SIZE = 100
"""Maximum number of world books to cache per service instance."""

MAX_DICTIONARY_CACHE_SIZE = 50
"""Maximum number of dictionaries to cache per service instance."""


# =============================================================================
# Recursion and Iteration Limits
# =============================================================================
MAX_RECURSIVE_DEPTH = 10
"""Maximum recursive scanning depth for world books to prevent infinite loops."""

MAX_BATCH_ITERATIONS = 1000
"""Maximum batch iterations for bulk operations (e.g., message deletion)."""


# =============================================================================
# Streaming Constants
# =============================================================================
MAX_STREAMING_CHUNKS = 10000
"""Maximum number of chunks to stream in a single response."""

MAX_STREAMING_BYTES = 10 * 1024 * 1024  # 10 MB
"""Maximum total bytes to stream in a single response."""


# =============================================================================
# Tool Calls Constants
# =============================================================================
MAX_TOOL_CALLS_SIZE = 100 * 1024  # 100 KB
"""Maximum size of tool_calls JSON metadata to store."""

MAX_TOOL_CALLS_COUNT = 50
"""Maximum number of tool calls per message."""


# =============================================================================
# Rate Limiting Defaults
# =============================================================================
RATE_LIMIT_WINDOW_SECONDS = 60
"""Default rate limit window in seconds."""

DEFAULT_RATE_LIMIT_OPS = 100
"""Default maximum operations per rate limit window."""

DEFAULT_MAX_CHARACTERS_PER_USER = 1000
"""Default maximum characters per user."""

DEFAULT_MAX_CHATS_PER_USER = 100000
"""Default maximum chat sessions (total) per user."""

DEFAULT_MAX_MESSAGES_PER_CHAT = 1000
"""Default maximum messages per chat session."""

DEFAULT_MAX_IMPORT_SIZE_MB = 10
"""Default maximum character import size in megabytes."""

DEFAULT_CHAT_COMPLETIONS_PER_MINUTE = 20
"""Default maximum chat completions per minute."""

DEFAULT_MESSAGE_SENDS_PER_MINUTE = 60
"""Default maximum message sends per minute."""


# =============================================================================
# Message and Content Limits
# =============================================================================
MAX_MESSAGE_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
"""Maximum size for message image attachments."""

MAX_PERSIST_CONTENT_LENGTH = 1_000_000  # 1 MB
"""Maximum length for persisted assistant content."""

MAX_PNG_METADATA_BYTES = 512 * 1024  # 512 KB
"""Maximum decompressed size for PNG text metadata chunks."""


# =============================================================================
# Throttle Cache Constants
# =============================================================================
THROTTLE_CACHE_MAX_KEYS = 10000
"""Maximum keys in the throttle cache."""

THROTTLE_STALE_SECONDS = 3600
"""Time in seconds before throttle entries become stale."""


# =============================================================================
# JSON Structure Validation Limits
# =============================================================================
MAX_JSON_LIST_ITEMS = 1000
"""Maximum number of items in a JSON list field."""

MAX_JSON_STRING_ITEM_LENGTH = 10000
"""Maximum length of individual string items in a JSON list."""

MAX_JSON_DICT_KEYS = 100
"""Maximum number of keys in a JSON dict field."""


# =============================================================================
# JSON Validation Helpers
# =============================================================================
from typing import Any


def validate_json_list(
    data: Any,
    field_name: str,
    max_items: int = MAX_JSON_LIST_ITEMS,
    max_item_length: int = MAX_JSON_STRING_ITEM_LENGTH,
) -> list[str]:
    """
    Validate that parsed JSON is a list of strings.

    Args:
        data: The parsed JSON data to validate
        field_name: Name of the field for error messages
        max_items: Maximum number of items allowed
        max_item_length: Maximum length of each string item

    Returns:
        The validated list of strings, or empty list if invalid

    Raises:
        ValueError: If validation fails and data is not recoverable
    """
    if data is None:
        return []

    if not isinstance(data, list):
        raise ValueError(f"Field '{field_name}' must be a list, got {type(data).__name__}")

    if len(data) > max_items:
        raise ValueError(f"Field '{field_name}' has too many items ({len(data)} > {max_items})")

    result = []
    for i, item in enumerate(data):
        if not isinstance(item, str):
            # Try to convert to string for robustness
            try:
                item = str(item)
            except Exception:
                raise ValueError(
                    f"Field '{field_name}' item {i} must be a string, got {type(item).__name__}"
                ) from None
        if len(item) > max_item_length:
            raise ValueError(
                f"Field '{field_name}' item {i} exceeds max length ({len(item)} > {max_item_length})"
            )
        result.append(item)

    return result


def validate_json_dict(
    data: Any,
    field_name: str,
    max_keys: int = MAX_JSON_DICT_KEYS,
) -> dict[str, Any]:
    """
    Validate that parsed JSON is a dictionary.

    Args:
        data: The parsed JSON data to validate
        field_name: Name of the field for error messages
        max_keys: Maximum number of keys allowed

    Returns:
        The validated dictionary, or empty dict if invalid

    Raises:
        ValueError: If validation fails and data is not recoverable
    """
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Field '{field_name}' must be a dict, got {type(data).__name__}")

    if len(data) > max_keys:
        raise ValueError(f"Field '{field_name}' has too many keys ({len(data)} > {max_keys})")

    return data


def safe_parse_json_list(
    data: Any,
    field_name: str,
    default: list[str] = None,
) -> list[str]:
    """
    Safely parse and validate a JSON list field, returning default on error.

    This is a lenient version that logs warnings instead of raising exceptions.

    Args:
        data: The data to validate (already parsed or raw string)
        field_name: Name of the field for logging
        default: Default value to return on error

    Returns:
        Validated list or default value
    """
    if default is None:
        default = []

    import json

    from loguru import logger

    if data is None:
        return default

    # Parse if it's a string
    if isinstance(data, str):
        if not data.strip():
            return default
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in field '{field_name}': {e}")
            return default

    # Validate structure
    try:
        return validate_json_list(data, field_name)
    except ValueError as e:
        logger.warning(f"JSON validation failed for '{field_name}': {e}")
        return default


def safe_parse_json_dict(
    data: Any,
    field_name: str,
    default: dict[str, Any] = None,
) -> dict[str, Any]:
    """
    Safely parse and validate a JSON dict field, returning default on error.

    This is a lenient version that logs warnings instead of raising exceptions.

    Args:
        data: The data to validate (already parsed or raw string)
        field_name: Name of the field for logging
        default: Default value to return on error

    Returns:
        Validated dict or default value
    """
    if default is None:
        default = {}

    import json

    from loguru import logger

    if data is None:
        return default

    # Parse if it's a string
    if isinstance(data, str):
        if not data.strip():
            return default
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in field '{field_name}': {e}")
            return default

    # Validate structure
    try:
        return validate_json_dict(data, field_name)
    except ValueError as e:
        logger.warning(f"JSON validation failed for '{field_name}': {e}")
        return default


# =============================================================================
# Default Assistant Character
# =============================================================================
DEFAULT_CHARACTER_NAME = "Helpful AI Assistant"
"""Name of the character card chats fall back to when none is chosen.

Not the ``{{char}}`` placeholder default (``character_utils.DEFAULT_CHARACTER_NAME``).
"""

DEFAULT_CHARACTER_DESCRIPTION = "A default, friendly assistant created automatically by the system."
"""Description given to the default assistant card when it is first created."""

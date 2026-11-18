"""
Character utility functions for text processing and UI helpers.

This module contains utility functions for character chat operations,
including placeholder replacement and UI-related helper functions.
"""

import re
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.Utils.chat_sender import USER_SENDER_ALIASES

# Default placeholder names used across Character_Chat helpers
DEFAULT_CHARACTER_NAME = "Character"
DEFAULT_USER_NAME = "User"


def replace_placeholders(text: Optional[str], char_name: Optional[str], user_name: Optional[str]) -> str:
    """Replaces predefined placeholders in a text string.

    The function substitutes placeholders like '{{char}}', '{{user}}',
    '{{random_user}}', '<USER>', and '<CHAR>' with the provided character
    and user names. If names are not provided, default values ("Character", "User")
    are used. Returns an empty string if the input text is None or empty.
    Args:
        text (Optional[str]): The input string, possibly containing placeholders.
        char_name (Optional[str]): The name of the character to substitute for
            '{{char}}' and '<CHAR>'. Defaults to DEFAULT_CHARACTER_NAME if None.
        user_name (Optional[str]): The name of the user to substitute for
            '{{user}}', '{{random_user}}', and '<USER>'. Defaults to DEFAULT_USER_NAME if None.

    Returns:
        str: The text with placeholders replaced. If the input `text` is None or
        an empty string, an empty string is returned.
    """
    if not text:
        return ""
    char_name_actual = str(char_name) if char_name is not None else DEFAULT_CHARACTER_NAME
    user_name_actual = str(user_name) if user_name is not None else DEFAULT_USER_NAME
    replacements = {
        '{{char}}': char_name_actual,
        '{{user}}': user_name_actual,
        '{{random_user}}': user_name_actual,
        '<USER>': user_name_actual,
        '<CHAR>': char_name_actual,
    }
    processed_text = text
    for placeholder, value in replacements.items():
        processed_text = processed_text.replace(placeholder, value)
    return processed_text


def replace_user_placeholder(history: list[tuple[Optional[str], Optional[str]]], user_name: Optional[str]) -> list[
    tuple[Optional[str], Optional[str]]]:
    """Replace {{user}} placeholders in chat history with the actual user name.

    Args:
        history: List of tuples containing (user_message, bot_message) pairs
        user_name: The name to replace {{user}} with, defaults to "User" if None

    Returns:
        Updated history with placeholders replaced
    """
    user_name_actual = user_name if user_name else DEFAULT_USER_NAME
    updated_history = []
    for user_msg, bot_msg in history:
        updated_user_msg = user_msg.replace("{{user}}", user_name_actual) if user_msg else None
        updated_bot_msg = bot_msg.replace("{{user}}", user_name_actual) if bot_msg else None
        updated_history.append((updated_user_msg, updated_bot_msg))
    return updated_history


def extract_character_id_from_ui_choice(choice: str) -> int:
    """Extract character ID from a UI choice string.

    Args:
        choice: A string in format 'Name (ID: 123)' or just a numeric ID

    Returns:
        The extracted character ID as an integer

    Raises:
        ValueError: If the choice format is invalid or ID cannot be parsed
    """
    logger.debug(f"Choice received for ID extraction: {choice}")
    if not choice:
        raise ValueError("No choice provided for character ID extraction.")
    match = re.search(r'\(ID:\s*(\d+)\s*\)$', choice)
    if match:
        character_id_str = match.group(1)
    else:
        character_id_str = choice.strip()
        if not character_id_str.isdigit():
            raise ValueError(f"Invalid choice format: '{choice}'. Expected 'Name (ID: 123)' or just a numeric ID.")
    try:
        character_id = int(character_id_str)
        logger.debug(f"Extracted character ID: {character_id}")
        return character_id
    except ValueError:
        raise ValueError(f"Could not parse character ID from: '{character_id_str}' (derived from '{choice}')") from None


def get_character_list_for_ui(db: CharactersRAGDB, limit: int = 1000) -> list[dict[str, Any]]:
    """Fetches a simplified list of characters suitable for UI display.

    Retrieves character IDs and names from the database, sorts them by name
    (case-insensitive), and returns them in a format commonly used for UI
    dropdowns or lists.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        limit (int): The maximum number of characters to fetch. Defaults to 1000.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
        contains 'id' (character ID) and 'name' (character name).
        Returns an empty list if an error occurs or no characters are found.
    """
    try:
        all_chars = db.list_character_cards(limit=limit)
        ui_list = [{"id": char.get("id"), "name": char.get("name")} for char in all_chars if
                   char.get("id") and char.get("name")]
        return sorted(ui_list, key=lambda x: (x["name"] or "").lower())  # Ensure name is not None for lower()
    except CharactersRAGDBError as e:
        logger.error(f"Database error fetching character list for UI: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching character list for UI: {e}", exc_info=True)
        return []


# --- Sender to Role Normalization ---

def sanitize_sender_name(name: Optional[str]) -> str:
    """Sanitize a display name for use as the DB 'sender' field.

    - Replace spaces with underscores
    - Strip characters that commonly interfere with parsing or storage
      (angle brackets, pipes, slashes, backslashes)
    - Preserve original casing (tests and UI display expect this)
    """
    if not name:
        return "Assistant"
    s = str(name)
    s = s.replace(" ", "_")
    for ch in ("<", ">", "|", "\\", "/"):
        s = s.replace(ch, "")
    return s

# --- Sender alias constants (centralized for reuse) ---
# Expose canonical alias sets for consistent sender-role handling across modules.
CHAR_SENDER_ALIASES = {"assistant", "bot", "ai", "character"}
SYSTEM_ALIASES = {"system", "narrator", "commentary", "metadata"}
TOOL_ALIASES = {"tool", "assistant_tool", "function"}
NON_CHARACTER_SENDER_ALIASES = set().union(SYSTEM_ALIASES, TOOL_ALIASES)

def map_sender_to_role(
    sender: Optional[str],
    character_name: Optional[str],
    default_role: str = "assistant",
) -> str:
    """Map stored DB sender strings to OpenAI-compatible roles.

    Args:
        sender: Raw sender as stored in DB (may be character name or alias)
        character_name: Character name for this conversation (used as assistant alias)
        default_role: Role to return when sender is unknown (defaults to "assistant")

    Returns:
        One of: "user", "assistant", "system", "tool"
    """
    allowed_defaults = {"user", "assistant", "system", "tool"}
    if default_role not in allowed_defaults:
        default_role = "assistant"
    s = (sender or "").strip().lower()
    char_lower = (character_name or "").strip().lower()
    sanitized_char_lower = ""
    if character_name:
        try:
            sanitized_char_lower = sanitize_sender_name(character_name).strip().lower()
        except Exception:
            sanitized_char_lower = ""
    if not s:
        return default_role
    if s in USER_SENDER_ALIASES:
        return "user"
    if s in TOOL_ALIASES or any(s.startswith(f"{p}:") for p in TOOL_ALIASES):
        return "tool"
    if s in SYSTEM_ALIASES or any(s.startswith(f"{p}:") for p in SYSTEM_ALIASES):
        return "system"
    if (
        s in CHAR_SENDER_ALIASES
        or (char_lower and s == char_lower)
        or (sanitized_char_lower and s == sanitized_char_lower)
    ):
        return "assistant"
    return default_role

# Optional explicit exports for constants
__all__ = [
    "replace_placeholders",
    "replace_user_placeholder",
    "extract_character_id_from_ui_choice",
    "get_character_list_for_ui",
    "map_sender_to_role",
    "sanitize_sender_name",
    # constants
    "USER_SENDER_ALIASES",
    "CHAR_SENDER_ALIASES",
    "SYSTEM_ALIASES",
    "TOOL_ALIASES",
    "NON_CHARACTER_SENDER_ALIASES",
    "DEFAULT_CHARACTER_NAME",
    "DEFAULT_USER_NAME",
]

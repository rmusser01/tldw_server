"""
Character chat operations module.

This module contains functions for managing chat sessions and messages. It is
the canonical implementation used by the FastAPI endpoints and the public
facade.
"""

import base64
import random
import re
import time
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, Optional, Union

from loguru import logger
from PIL import Image

from tldw_Server_API.app.core.Character_Chat.constants import MAX_PERSIST_CONTENT_LENGTH
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

_CHAR_CHAT_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

from .character_db import load_character_and_image
from .character_utils import (
    NON_CHARACTER_SENDER_ALIASES as _NON_CHARACTER_SENDER_ALIASES,
)
from .character_utils import (
    USER_SENDER_ALIASES as _LEGACY_USER_SENDER_ALIASES,
)
from .character_utils import (
    replace_placeholders,
)

# Aliases are sourced from character_utils for consistency across modules.
_DEFAULT_FIRST_MESSAGE_TEMPLATE = "Hello, I am {{char}}. How can I help you, {{user}}?"
_PLACEHOLDER_TOKENS_FOR_LENGTH = (
    "{{char}}",
    "{{user}}",
    "{{random_user}}",
    "<USER>",
    "<CHAR>",
)


def _content_length_for_guardrails(text: str) -> int:
    """Estimate content length for guardrails, discounting placeholder tokens."""
    if not text:
        return 0
    normalized = text
    for token in _PLACEHOLDER_TOKENS_FOR_LENGTH:
        normalized = normalized.replace(token, "X")
    return len(normalized)


def _extract_message_attachments(msg_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a normalized attachment list for UI consumption."""

    attachments: dict[int, dict[str, Any]] = {}

    def _normalize_payload(position: int, data: Any, mime: Optional[str]) -> None:
        if data is None:
            return
        payload = data.tobytes() if isinstance(data, memoryview) else data
        payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        attachments[position] = {
            "position": position,
            "data": base64.b64encode(payload_bytes).decode("ascii"),
            "encoding": "base64",
            "size": len(payload_bytes),
            "mime_type": mime,
            "message_id": msg_data.get("id"),
        }

    primary_data = msg_data.get("image_data")
    primary_mime = msg_data.get("image_mime_type")
    if primary_data is not None:
        _normalize_payload(0, primary_data, primary_mime)

    for image_entry in msg_data.get("images") or []:
        if not isinstance(image_entry, dict):
            continue
        position_raw = image_entry.get("position")
        try:
            position_int = int(position_raw) if position_raw is not None else len(attachments)
        except (TypeError, ValueError):
            position_int = len(attachments)
        _normalize_payload(
            position_int,
            image_entry.get("image_data"),
            image_entry.get("image_mime_type"),
        )

    return [attachments[idx] for idx in sorted(attachments.keys())]


def _infer_character_sender_aliases(
    db_messages: list[dict[str, Any]],
    user_aliases: set[str],
    primary_char_name: str,
) -> list[str]:
    """Infer legacy character sender aliases stored on existing messages."""

    alias_counter: Counter[str] = Counter()
    first_candidate: Optional[str] = None
    primary_lower = primary_char_name.lower()

    for msg in db_messages:
        sender = msg.get("sender")
        if not isinstance(sender, str):
            continue
        normalized = sender.strip()
        if not normalized:
            continue
        lower = normalized.lower()

        if lower == primary_lower:
            continue
        if lower in user_aliases:
            continue
        if lower in _NON_CHARACTER_SENDER_ALIASES:
            continue
        if normalized.startswith("["):
            continue

        alias_counter[normalized] += 1
        if first_candidate is None:
            first_candidate = normalized

    inferred_aliases: list[str] = []
    for alias, count in alias_counter.items():
        if alias.lower() == primary_lower:
            continue
        if count >= 2 or alias == first_candidate:
            inferred_aliases.append(alias)
    return inferred_aliases


def _compute_additional_char_aliases(
    db_messages: list[dict[str, Any]],
    char_name_from_card: str,
    user_name_for_placeholders: Optional[str],
    actual_user_sender_id_in_db: str,
) -> tuple[list[str], list[str]]:
    user_aliases = {alias.lower() for alias in _LEGACY_USER_SENDER_ALIASES}
    if actual_user_sender_id_in_db:
        user_aliases.add(actual_user_sender_id_in_db.lower())
    if user_name_for_placeholders:
        user_aliases.add(str(user_name_for_placeholders).strip().lower())

    candidate_aliases = _infer_character_sender_aliases(db_messages, user_aliases, char_name_from_card)
    if not candidate_aliases:
        return [], []

    inferred_user_aliases: list[str] = []
    filtered_char_aliases: list[str] = []
    normalized_alias_map = {alias.lower(): alias for alias in candidate_aliases}

    # Pre-compute lowercase senders for efficient neighbour lookups
    sender_sequence: list[Optional[str]] = []
    for msg in db_messages:
        sender = msg.get("sender")
        sender_sequence.append(str(sender).strip().lower() if isinstance(sender, str) else None)

    def _char_neighbor_set(current_alias: str) -> set[str]:
        neighbors = {char_name_from_card.strip().lower()}
        for other_alias in candidate_aliases:
            normalized_other = other_alias.strip().lower()
            if normalized_other and normalized_other != current_alias:
                neighbors.add(normalized_other)
        return neighbors

    for alias_lower, alias_original in normalized_alias_map.items():
        if not alias_lower:
            continue

        char_context = 0
        user_context = 0
        char_neighbors = _char_neighbor_set(alias_lower)

        for idx, sender_lower in enumerate(sender_sequence):
            if sender_lower != alias_lower:
                continue
            prev_lower = sender_sequence[idx - 1] if idx > 0 else None
            next_lower = sender_sequence[idx + 1] if idx + 1 < len(sender_sequence) else None

            if prev_lower and prev_lower in user_aliases:
                user_context += 1
            if next_lower and next_lower in user_aliases:
                user_context += 1

            if prev_lower and prev_lower in char_neighbors:
                char_context += 1
            if next_lower and next_lower in char_neighbors:
                char_context += 1

        # Always treat inferred aliases as character aliases; do not promote to user.
        # Multi-speaker or multi-character sessions should not be reclassified as user.
        filtered_char_aliases.append(alias_original)

    return filtered_char_aliases, inferred_user_aliases


def process_db_messages_to_ui_history(
    db_messages: list[dict[str, Any]],
    char_name_from_card: str,
    user_name_for_placeholders: Optional[str],
    actual_user_sender_id_in_db: str = "User",
    actual_char_sender_id_in_db: Optional[str] = None,
    additional_char_sender_ids: Optional[Iterable[str]] = None,
    additional_user_sender_ids: Optional[Iterable[str]] = None,
    char_first_message: Optional[str] = None,
    keep_trailing_user: bool = True,
) -> list[tuple[Optional[str], Optional[str]]]:
    """Convert database messages to UI-friendly paired chat history format."""

    rich_history = process_db_messages_to_rich_ui_history(
        db_messages=db_messages,
        char_name_from_card=char_name_from_card,
        user_name_for_placeholders=user_name_for_placeholders,
        actual_user_sender_id_in_db=actual_user_sender_id_in_db,
        actual_char_sender_id_in_db=actual_char_sender_id_in_db,
        additional_char_sender_ids=additional_char_sender_ids,
        additional_user_sender_ids=additional_user_sender_ids,
        char_first_message=char_first_message,
        keep_trailing_user=keep_trailing_user,
    )

    processed_history: list[tuple[Optional[str], Optional[str]]] = []
    for turn in rich_history:
        user_content = turn["user"]["content"] if turn["user"] else None
        character_content = None
        if turn["character"]:
            character_content = turn["character"]["content"]
        elif turn["non_character"]:
            character_content = turn["non_character"]["content"]
        processed_history.append((user_content, character_content))
    return processed_history


def process_db_messages_to_rich_ui_history(
    db_messages: list[dict[str, Any]],
    char_name_from_card: str,
    user_name_for_placeholders: Optional[str],
    actual_user_sender_id_in_db: str = "User",
    actual_char_sender_id_in_db: Optional[str] = None,
    additional_char_sender_ids: Optional[Iterable[str]] = None,
    additional_user_sender_ids: Optional[Iterable[str]] = None,
    char_first_message: Optional[str] = None,
    keep_trailing_user: bool = False,
) -> list[dict[str, Optional[dict[str, Any]]]]:
    """Convert database messages to UI-friendly structures including metadata."""

    char_sender_identifier = (
        actual_char_sender_id_in_db if actual_char_sender_id_in_db else char_name_from_card
    )
    user_identifiers = {alias.lower() for alias in _LEGACY_USER_SENDER_ALIASES}
    non_character_aliases = {alias.lower() for alias in _NON_CHARACTER_SENDER_ALIASES}
    char_identifiers = {
        char_name_from_card.lower(),
        "assistant",
        "bot",
        "ai",
        "character",
    }

    if actual_user_sender_id_in_db:
        user_identifiers.add(actual_user_sender_id_in_db.lower())
    if char_sender_identifier:
        char_identifiers.add(str(char_sender_identifier).strip().lower())
    if additional_char_sender_ids:
        for alias in additional_char_sender_ids:
            if not alias:
                continue
            alias_norm = str(alias).strip().lower()
            if alias_norm:
                char_identifiers.add(alias_norm)
    if additional_user_sender_ids:
        for alias in additional_user_sender_ids:
            if not alias:
                continue
            alias_norm = str(alias).strip().lower()
            if alias_norm:
                user_identifiers.add(alias_norm)
    if user_name_for_placeholders:
        placeholder_alias = str(user_name_for_placeholders).strip().lower()
        if placeholder_alias:
            user_identifiers.add(placeholder_alias)

    user_msg_detail_buffer: Optional[dict[str, Any]] = None

    explicit_char_sender_lower = None
    if actual_char_sender_id_in_db:
        explicit_char_sender_lower = str(actual_char_sender_id_in_db).strip().lower()

    char_first_message_normalized = None
    if isinstance(char_first_message, str):
        char_first_message_normalized = char_first_message.strip()

    def _resolve_ambiguous_sender(
        message_index: int,
        processed_text: str,
        pending_user_buffer: bool,
        character_messages_seen: int,
        user_messages_seen: int,
        next_sender_lower: Optional[str],
    ) -> str:
        if pending_user_buffer:
            return "character"
        processed_text_normalized = processed_text.strip()
        if (
            char_first_message_normalized
            and processed_text_normalized
            and processed_text_normalized == char_first_message_normalized
            and character_messages_seen == 0
        ):
            return "character"
        if user_messages_seen > character_messages_seen:
            return "character"
        if character_messages_seen > user_messages_seen:
            return "user"
        if next_sender_lower:
            # If the following sender is clearly a user (and not a character),
            # then this ambiguous message is best treated as a character reply.
            if (
                next_sender_lower in user_identifiers
                and next_sender_lower not in char_identifiers
            ):
                return "character"
            # If the following sender is clearly a character (and not a user),
            # then this ambiguous message should be a user message to maintain alternation.
            if (
                next_sender_lower in char_identifiers
                and next_sender_lower not in user_identifiers
            ):
                return "user"
        if message_index == 0 and char_first_message_normalized:
            return "character"
        # Truly ambiguous case - log for debugging
        logger.debug(
            f"Ambiguous sender at message index {message_index}, defaulting to 'character'. "
            f"user_seen={user_messages_seen}, char_seen={character_messages_seen}"
        )
        return "character"

    character_messages_seen = 0
    user_messages_seen = 0
    total_messages = len(db_messages)

    rich_history: list[dict[str, Optional[dict[str, Any]]]] = []

    for idx, msg_data in enumerate(db_messages):
        sender = msg_data.get("sender")
        content = msg_data.get("content", "")

        processed_content = replace_placeholders(content, char_name_from_card, user_name_for_placeholders)

        sender_normalized = str(sender).strip() if isinstance(sender, str) else ""
        sender_lower = sender_normalized.lower()

        is_non_character_sender = False
        if sender_lower:
            # Allow explicit character sender override to avoid misclassifying
            # characters named like reserved system/tool aliases (including prefix forms).
            if explicit_char_sender_lower and sender_lower == explicit_char_sender_lower:
                is_non_character_sender = False
            elif sender_lower in non_character_aliases:
                is_non_character_sender = True
            else:
                for alias in non_character_aliases:
                    if sender_lower.startswith(f"{alias}:"):
                        is_non_character_sender = True
                        break

        if is_non_character_sender:
            formatted_content = processed_content
            if sender_normalized:
                if formatted_content:
                    formatted_content = f"[{sender_normalized}] {processed_content}"
                else:
                    formatted_content = f"[{sender_normalized}]"
            detail = {
                "id": msg_data.get("id"),
                "sender": sender_normalized or sender,
                "content": formatted_content,
                "raw_sender": sender,
                "attachments": _extract_message_attachments(msg_data),
                "metadata": {
                    "timestamp": msg_data.get("timestamp"),
                    "ranking": msg_data.get("ranking"),
                    "version": msg_data.get("version"),
                },
            }
            if user_msg_detail_buffer is not None:
                rich_history.append(
                    {
                        "user": user_msg_detail_buffer,
                        "character": None,
                        "non_character": detail,
                    }
                )
                user_msg_detail_buffer = None
            else:
                rich_history.append(
                    {
                        "user": None,
                        "character": None,
                        "non_character": detail,
                    }
                )
            continue

        in_user_identifiers = sender_lower in user_identifiers if sender_lower else False
        in_char_identifiers = sender_lower in char_identifiers if sender_lower else False

        next_sender_lower: Optional[str] = None
        if idx + 1 < total_messages:
            next_sender_value = db_messages[idx + 1].get("sender")
            if isinstance(next_sender_value, str):
                next_sender_lower = next_sender_value.strip().lower()

        sender_role: Optional[str]
        if in_char_identifiers and not in_user_identifiers:
            sender_role = "character"
        elif in_user_identifiers and not in_char_identifiers:
            sender_role = "user"
        elif in_char_identifiers and in_user_identifiers:
            # Ambiguous: sender matches both user and character aliases.
            # Prefer classifying as a user message unless we have an explicit
            # character sender id and the alias truly belongs to the character.
            # This avoids misclassifying real user messages when the character
            # name collides with user aliases (e.g. char_name == "user").
            _placeholder_alias = str(user_name_for_placeholders).strip().lower() if isinstance(user_name_for_placeholders, str) else ""
            _char_sender_norm = str(char_sender_identifier or "").strip().lower()
            if (
                actual_char_sender_id_in_db  # explicit character sender id provided
                and _placeholder_alias
                and _char_sender_norm
                and _placeholder_alias == _char_sender_norm
                and sender_lower == _char_sender_norm
            ):
                sender_role = "character"
            elif (
                char_first_message_normalized
                and isinstance(processed_content, str)
                and processed_content.strip() == char_first_message_normalized
                and character_messages_seen == 0
            ):
                # Honor explicit first-message hint: treat as character's greeting
                sender_role = "character"
            elif sender_lower in user_identifiers and not actual_char_sender_id_in_db:
                # Default to user when ambiguous only if there is no explicit
                # character sender id provided by the caller.
                sender_role = "user"
            else:
                # Resolve tie using context (parity/neighbor hints)
                sender_role = _resolve_ambiguous_sender(
                    message_index=idx,
                    processed_text=processed_content,
                    pending_user_buffer=user_msg_detail_buffer is not None,
                    character_messages_seen=character_messages_seen,
                    user_messages_seen=user_messages_seen,
                    next_sender_lower=next_sender_lower,
                )
        elif in_user_identifiers:
            sender_role = "user"
        elif in_char_identifiers:
            sender_role = "character"
        else:
            sender_role = None

        if sender_role == "character":
            detail = {
                "id": msg_data.get("id"),
                "sender": sender_normalized or sender,
                "content": processed_content,
                "raw_sender": sender,
                "attachments": _extract_message_attachments(msg_data),
                "metadata": {
                    "timestamp": msg_data.get("timestamp"),
                    "ranking": msg_data.get("ranking"),
                    "version": msg_data.get("version"),
                },
            }
            if user_msg_detail_buffer is not None:
                rich_history.append(
                    {
                        "user": user_msg_detail_buffer,
                        "character": detail,
                        "non_character": None,
                    }
                )
                user_msg_detail_buffer = None
            else:
                rich_history.append(
                    {
                        "user": None,
                        "character": detail,
                        "non_character": None,
                    }
                )
            if sender_lower:
                char_identifiers.add(sender_lower)
            character_messages_seen += 1
            continue

        if sender_role == "user":
            detail = {
                "id": msg_data.get("id"),
                "sender": sender_normalized or sender,
                "content": processed_content,
                "raw_sender": sender,
                "attachments": _extract_message_attachments(msg_data),
                "metadata": {
                    "timestamp": msg_data.get("timestamp"),
                    "ranking": msg_data.get("ranking"),
                    "version": msg_data.get("version"),
                },
            }
            if user_msg_detail_buffer is not None:
                rich_history.append(
                    {
                        "user": user_msg_detail_buffer,
                        "character": None,
                        "non_character": None,
                    }
                )
            user_msg_detail_buffer = detail
            user_messages_seen += 1
            continue

        # Unknown sender handling - preserve legacy behaviour by treating as character
        # Redact both sender and content in logs to avoid leaking sensitive data
        try:
            _len = len(processed_content) if isinstance(processed_content, str) else 0
        except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS:
            _len = 0
        # Hash sender name for traceability without exposing PII
        try:
            sender_hash = hash(sender) & 0xFFFFFFFF  # 32-bit positive hash for log reference
        except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS:
            sender_hash = hash(str(sender)) & 0xFFFFFFFF
        logger.warning("Message from unknown sender (hash={:08x}, content_length={})", sender_hash, _len)
        sender_label = str(sender or "")
        if sender_label.strip():
            formatted_content = f"[{sender_label}] {processed_content}"
        else:
            formatted_content = processed_content
        detail = {
            "id": msg_data.get("id"),
            "sender": sender_normalized or sender,
            "content": formatted_content,
            "raw_sender": sender,
            "attachments": _extract_message_attachments(msg_data),
            "metadata": {
                "timestamp": msg_data.get("timestamp"),
                "ranking": msg_data.get("ranking"),
                "version": msg_data.get("version"),
            },
        }
        if user_msg_detail_buffer is not None:
            rich_history.append(
                {
                    "user": user_msg_detail_buffer,
                    "character": detail,
                    "non_character": None,
                }
            )
            user_msg_detail_buffer = None
        else:
            rich_history.append(
                {
                    "user": None,
                    "character": detail,
                    "non_character": None,
                }
            )
        character_messages_seen += 1

    # Optionally append trailing user-only turn if requested
    if keep_trailing_user and user_msg_detail_buffer is not None:
        rich_history.append(
            {
                "user": user_msg_detail_buffer,
                "character": None,
                "non_character": None,
            }
        )

    return rich_history


def load_chat_and_character(
    db: CharactersRAGDB,
    conversation_id_str: str,
    user_name: Optional[str],
    messages_limit: int = 2000,
) -> tuple[
    Optional[dict[str, Any]],
    list[tuple[Optional[str], Optional[str]]],
    Optional[Image.Image],
]:
    """Load an existing chat conversation and associated character data."""

    logger.debug(
        'Loading chat/conversation ID: {}, User: {}, Msg Limit: {}',
        conversation_id_str,
        user_name,
        messages_limit,
    )
    try:
        conversation_data = db.get_conversation_by_id(conversation_id_str)
        if not conversation_data:
            logger.warning("No conversation found with ID: {}", conversation_id_str)
            return None, [], None

        character_id = conversation_data.get("character_id")
        if not character_id:
            logger.error(
                'Conversation {} has no character_id associated.',
                conversation_id_str,
            )
            raw_db_messages = db.get_messages_for_conversation(
                conversation_id_str,
                limit=messages_limit,
                order_by_timestamp="ASC",
            )
            additional_aliases, extra_user_aliases = _compute_additional_char_aliases(
                raw_db_messages,
                "Unknown Character",
                user_name,
                actual_user_sender_id_in_db="User",
            )
            processed_ui_history = process_db_messages_to_ui_history(
                raw_db_messages,
                "Unknown Character",
                user_name,
                actual_user_sender_id_in_db="User",
                actual_char_sender_id_in_db="Unknown Character",
                additional_char_sender_ids=additional_aliases,
                additional_user_sender_ids=extra_user_aliases,
            )
            return None, processed_ui_history, None

        char_data, _, img = load_character_and_image(db, character_id, user_name)

        if not char_data:
            logger.warning(
                'No character card found for char_id {} (from conv {})',
                character_id,
                conversation_id_str,
            )
            raw_db_messages = db.get_messages_for_conversation(
                conversation_id_str,
                limit=messages_limit,
                order_by_timestamp="ASC",
            )
            additional_aliases, extra_user_aliases = _compute_additional_char_aliases(
                raw_db_messages,
                "Unknown Character",
                user_name,
                actual_user_sender_id_in_db="User",
            )
            processed_ui_history = process_db_messages_to_ui_history(
                raw_db_messages,
                "Unknown Character",
                user_name,
                actual_user_sender_id_in_db="User",
                actual_char_sender_id_in_db="Unknown Character",
                additional_char_sender_ids=additional_aliases,
                additional_user_sender_ids=extra_user_aliases,
            )
            return None, processed_ui_history, img

        char_name_from_card = char_data.get("name", "Character")
        raw_db_messages = db.get_messages_for_conversation(
            conversation_id_str,
            limit=messages_limit,
            order_by_timestamp="ASC",
        )
        additional_aliases, extra_user_aliases = _compute_additional_char_aliases(
            raw_db_messages,
            char_name_from_card,
            user_name,
            actual_user_sender_id_in_db="User",
        )
        processed_ui_history = process_db_messages_to_ui_history(
            raw_db_messages,
            char_name_from_card,
            user_name,
            actual_user_sender_id_in_db="User",
            actual_char_sender_id_in_db=char_name_from_card,
            additional_char_sender_ids=additional_aliases,
            additional_user_sender_ids=extra_user_aliases,
            char_first_message=char_data.get("first_message"),
        )

        return char_data, processed_ui_history, img

    except CharactersRAGDBError as exc:
        logger.error(
            'Database error in load_chat_and_character for conversation ID {}: {}',
            conversation_id_str,
            exc,
        )
        return None, [], None
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error in load_chat_and_character for conv ID {}: {}',
            conversation_id_str,
            exc,
            exc_info=True,
        )
        return None, [], None


def start_new_chat_session(
    db: CharactersRAGDB,
    character_id: int,
    user_name: Optional[str],
    custom_title: Optional[str] = None,
    greeting_strategy: Optional[str] = None,
    alternate_index: Optional[int] = None,
) -> tuple[
    Optional[str],
    Optional[dict[str, Any]],
    Optional[list[tuple[Optional[str], Optional[str]]]],
    Optional[Image.Image],
]:
    """Start a new chat session with the specified character."""

    logger.debug("Starting new chat session for character_id: {}, user: {}", character_id, user_name)

    original_first_message_content: Optional[str] = None
    original_alternate_greetings: Optional[list[str]] = None

    def _normalize_alt_greeting(value: Any) -> Optional[str]:
        """Normalize value (Any) to UTF-8 text or None (Optional[str]); bytes/memoryview are decoded with errors replaced."""
        if isinstance(value, str):
            return value
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, (bytes, bytearray)):
            try:
                return bytes(value).decode("utf-8")
            except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS:
                return bytes(value).decode("utf-8", errors="replace")
        return None

    try:
        raw_char_data_for_first_message = db.get_character_card_by_id(character_id)
        if raw_char_data_for_first_message:
            original_first_message_content = raw_char_data_for_first_message.get("first_message")
            ag = raw_char_data_for_first_message.get("alternate_greetings")
            if isinstance(ag, list):
                # Normalize bytes-like greetings to text
                normalized: list[str] = []
                for entry in ag:
                    greeting = _normalize_alt_greeting(entry)
                    if isinstance(greeting, str):
                        normalized.append(greeting)
                original_alternate_greetings = normalized
        else:
            logger.warning(
                'Could not load raw character data for ID {} to get original first message.',
                character_id,
            )
    except CharactersRAGDBError as exc:
        logger.warning(
            'DB error fetching raw character data for ID {}: {}. Proceeding with caution.',
            character_id,
            exc,
        )

    char_data, initial_ui_history, img = load_character_and_image(db, character_id, user_name)

    if not char_data:
        logger.error("Failed to load character_id {} to start new chat session.", character_id)
        return None, None, None, None

    char_name = char_data.get("name", "Character")
    conv_title = custom_title if custom_title else f"Chat with {char_name} ({time.strftime('%Y-%m-%d %H:%M')})"

    conversation_id_val: Optional[str] = None
    try:
        conv_payload = {
            "character_id": character_id,
            "title": conv_title,
        }
        conversation_id_val = db.add_conversation(conv_payload)

        if not conversation_id_val:
            logger.error(
                'Failed to create conversation record in DB for character {}.',
                char_name,
            )
            return None, char_data, initial_ui_history, img

        logger.info(
            "Created new conversation ID: {} for character '{}'.",
            conversation_id_val,
            char_name,
        )

        def _restore_first_message_template(text: str) -> str:
            restored = text
            user_token = str(user_name).strip() if user_name is not None else ""
            if user_token:
                restored = re.sub(
                    rf"(?<!\w){re.escape(user_token)}(?!\w)",
                    "{{user}}",
                    restored,
                )
            char_token = str(char_name).strip() if char_name is not None else ""
            if char_token:
                restored = re.sub(
                    rf"(?<!\w){re.escape(char_token)}(?!\w)",
                    "{{char}}",
                    restored,
                )
            return restored

        message_to_store_in_db: Optional[str] = original_first_message_content

        # Apply optional greeting strategy selection using raw (unprocessed) fields
        selected_alt: Optional[str] = None
        try:
            if greeting_strategy in {"alternate_random", "alternate_index"} and original_alternate_greetings:
                if greeting_strategy == "alternate_random":
                    if original_alternate_greetings:
                        selected_alt = random.choice(original_alternate_greetings)
                elif greeting_strategy == "alternate_index":
                    if isinstance(alternate_index, int) and alternate_index >= 0 and alternate_index < len(original_alternate_greetings):
                        selected_alt = original_alternate_greetings[alternate_index]
        except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as _sel_err:
            logger.debug("Alternate greeting selection failed: {}", _sel_err)

        if isinstance(selected_alt, str) and selected_alt.strip():
            message_to_store_in_db = selected_alt

        if message_to_store_in_db is None:
            fallback_processed: Optional[str] = None
            if initial_ui_history and initial_ui_history[0] and initial_ui_history[0][1]:
                fallback_processed = initial_ui_history[0][1]
            elif char_data.get("first_message"):
                fallback_processed = char_data["first_message"]

            if isinstance(fallback_processed, str) and fallback_processed.strip():
                processed_default = replace_placeholders(
                    _DEFAULT_FIRST_MESSAGE_TEMPLATE,
                    char_name,
                    user_name,
                ).strip()
                if fallback_processed.strip() == processed_default:
                    message_to_store_in_db = _DEFAULT_FIRST_MESSAGE_TEMPLATE
                    logger.warning(
                        'Storing default first_message template for char {} in new conversation {} as raw version was not available.',
                        char_name,
                        conversation_id_val,
                    )
                else:
                    message_to_store_in_db = _restore_first_message_template(fallback_processed)
                    logger.warning(
                        'Storing reconstructed first_message template for char {} in new conversation {} as raw version was not available.',
                        char_name,
                        conversation_id_val,
                    )

        if message_to_store_in_db:
            db.add_message(
                {
                    "conversation_id": conversation_id_val,
                    "sender": char_name,
                    "content": message_to_store_in_db,
                }
            )
            logger.debug(
                "Added character's first message to new conversation {}.",
                conversation_id_val,
            )
            # Ensure returned initial_ui_history aligns with the stored message
            try:
                processed = replace_placeholders(message_to_store_in_db, char_name, user_name)
                if initial_ui_history:
                    initial_ui_history[0] = (None, processed)
                else:
                    initial_ui_history = [(None, processed)]
            except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Failed to update initial_ui_history for conversation {conversation_id_val}: {e}")
        else:
            logger.warning(
                'Character {} (ID: {}) has no first message to add to new conversation {}.',
                char_name,
                character_id,
                conversation_id_val,
            )
            if not (initial_ui_history and initial_ui_history[0] and initial_ui_history[0][1]):
                initial_ui_history = []

        return conversation_id_val, char_data, initial_ui_history, img

    except (CharactersRAGDBError, InputError, ConflictError) as exc:
        logger.error(
            'Error during new chat session creation for char {}: {}',
            char_name,
            exc,
        )
        return conversation_id_val, char_data, initial_ui_history, img
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected error in start_new_chat_session: {}", exc, exc_info=True)
        return conversation_id_val, char_data, initial_ui_history, img


def list_character_conversations(
    db: CharactersRAGDB,
    character_id: int,
    limit: int = 50,
    offset: int = 0,
    client_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """List active conversations for a given character."""

    try:
        return db.get_conversations_for_character(
            character_id,
            limit=limit,
            offset=offset,
            client_id=client_id,
        )
    except CharactersRAGDBError as exc:
        logger.error(
            'Failed to list conversations for character ID {}: {}',
            character_id,
            exc,
        )
        return []
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error listing conversations for char ID {}: {}',
            character_id,
            exc,
            exc_info=True,
        )
        return []


def get_conversation_metadata(db: CharactersRAGDB, conversation_id: str) -> Optional[dict[str, Any]]:
    """Retrieve metadata for a specific conversation."""

    try:
        return db.get_conversation_by_id(conversation_id)
    except CharactersRAGDBError as exc:
        logger.error(
            'Failed to get metadata for conversation ID {}: {}',
            conversation_id,
            exc,
        )
        return None
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error getting conversation metadata for ID {}: {}',
            conversation_id,
            exc,
            exc_info=True,
        )
        return None


def update_conversation_metadata(
    db: CharactersRAGDB,
    conversation_id: str,
    update_data: dict[str, Any],
    expected_version: int,
) -> bool:
    """Update conversation metadata with optimistic locking."""

    try:
        valid_update_keys = {"title", "rating", "state", "topic_label", "cluster_id", "source", "external_ref"}
        payload_to_db = {k: v for k, v in update_data.items() if k in valid_update_keys}

        if not payload_to_db:
            logger.warning(
                'No valid fields to update for conversation ID {} from data: {}',
                conversation_id,
                update_data,
            )

        if "topic_label" in payload_to_db:
            raw_label = payload_to_db.get("topic_label")
            normalized_label = str(raw_label).strip() if raw_label is not None else ""
            latest_message = db.get_latest_message_for_conversation(conversation_id)
            latest_message_id = latest_message.get("id") if latest_message else None
            if normalized_label:
                payload_to_db["topic_label"] = normalized_label
                payload_to_db["topic_label_source"] = "manual"
                payload_to_db["topic_last_tagged_at"] = datetime.now(timezone.utc).isoformat()
                payload_to_db["topic_last_tagged_message_id"] = latest_message_id
            else:
                payload_to_db["topic_label"] = None
                payload_to_db["topic_label_source"] = None
                payload_to_db["topic_last_tagged_at"] = None
                payload_to_db["topic_last_tagged_message_id"] = None

        return db.update_conversation(conversation_id, payload_to_db, expected_version)
    except (CharactersRAGDBError, InputError, ConflictError) as exc:
        logger.error(
            'Failed to update metadata for conversation ID {}: {}',
            conversation_id,
            exc,
        )
        return False
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error updating conversation metadata for ID {}: {}',
            conversation_id,
            exc,
            exc_info=True,
        )
        return False


def delete_conversation_by_id(
    db: CharactersRAGDB,
    conversation_id: str,
    expected_version: int,
) -> bool:
    """Soft-delete a conversation using optimistic locking."""

    try:
        success = db.soft_delete_conversation(conversation_id, expected_version)
        if success:
            logger.info("Conversation {} soft-deleted successfully.", conversation_id)
        return success
    except (CharactersRAGDBError, ConflictError) as exc:
        logger.error(
            'Failed to remove conversation ID {}: {}',
            conversation_id,
            exc,
        )
        return False
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected error removing conversation ID {}: {}", conversation_id, exc, exc_info=True)
        return False


def search_conversations_by_title_query(
    db: CharactersRAGDB,
    title_query: str,
    character_id: Optional[int] = None,
    limit: int = 10,
    client_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Search conversations by title."""

    try:
        return db.search_conversations_by_title(
            title_query,
            character_id=character_id,
            limit=limit,
            client_id=client_id,
        )
    except CharactersRAGDBError as exc:
        logger.error(
            "Failed to search conversations with query '{}': {}",
            title_query,
            exc,
        )
        return []
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error searching conversations: {}',
            exc,
            exc_info=True,
        )
        return []


def _bump_conversation_metadata(
    db: CharactersRAGDB,
    conversation_id: str,
    max_attempts: int = 3,
) -> None:
    """Best-effort metadata bump with retry to handle concurrent updates."""
    for attempt in range(1, max_attempts + 1):
        try:
            conversation_meta = db.get_conversation_by_id(conversation_id)
        except CharactersRAGDBError as meta_exc:
            logger.warning(
                'Non-fatal: unable to fetch conversation {} for metadata update: {}',
                conversation_id,
                meta_exc,
            )
            return

        if not conversation_meta:
            return

        expected_version = conversation_meta.get("version")
        if not isinstance(expected_version, int):
            return

        try:
            db.update_conversation(conversation_id, {}, expected_version)
            return
        except ConflictError as conv_exc:
            if attempt >= max_attempts:
                logger.debug(
                    'Non-fatal: failed to bump conversation metadata for {} after {} attempts: {}',
                    conversation_id,
                    attempt,
                    conv_exc,
                )
                return
        except CharactersRAGDBError as conv_exc:
            logger.debug(
                'Non-fatal: failed to bump conversation metadata for {}: {}',
                conversation_id,
                conv_exc,
            )
            return


def post_message_to_conversation(
    db: CharactersRAGDB,
    conversation_id: str,
    character_name: str,
    message_content: str,
    is_user_message: bool,
    message_id: Optional[str] = None,
    parent_message_id: Optional[str] = None,
    ranking: Optional[int] = None,
    image_data: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
    sender_override: Optional[str] = None,
) -> Optional[str]:
    """Post a new message to a specified conversation."""

    if not conversation_id:
        logger.error("Cannot post message: conversation_id is required.")
        raise InputError("conversation_id is required for posting a message.")
    if not character_name and not is_user_message and not sender_override:
        logger.error("Cannot post character message: character_name is required.")
        raise InputError("character_name is required for character messages.")

    sender_name = sender_override or ("User" if is_user_message else character_name)
    if not sender_name:
        logger.error("Cannot post message: sender name could not be determined.")
        raise InputError("sender name could not be determined for message.")

    if not message_content and not image_data:
        logger.error("Cannot post message: Message must have text content or image data.")
        raise InputError("Message must have text content or image data.")

    # Optional preflight content size guard (mirrors DB constraints for clearer API errors)
    if message_content:
        max_content_len = MAX_PERSIST_CONTENT_LENGTH
        try:
            configured_max = settings.get("MAX_PERSIST_CONTENT_LENGTH", None)
            if configured_max is not None:
                max_content_len = int(configured_max)
        except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        if max_content_len:
            effective_len = _content_length_for_guardrails(message_content)
            if effective_len > max_content_len:
                raise InputError(
                    f"Message content exceeds maximum length of {max_content_len} characters"
                )

    # Optional preflight image size guard (mirrors DB constraints for clearer API errors)
    if image_data is not None:
        try:
            max_bytes = int(settings.get("MAX_MESSAGE_IMAGE_BYTES", 5 * 1024 * 1024))
        except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS:
            max_bytes = 5 * 1024 * 1024
        raw = image_data.tobytes() if isinstance(image_data, memoryview) else image_data
        if isinstance(raw, (bytes, bytearray)) and len(raw) > max_bytes:
            raise InputError(f"Image attachment exceeds maximum size of {max_bytes} bytes")

    msg_payload = {
        "id": message_id,
        "conversation_id": conversation_id,
        "sender": sender_name,
        "content": message_content,
        "parent_message_id": parent_message_id,
        "ranking": ranking,
        "image_data": image_data,
        "image_mime_type": image_mime_type,
    }

    try:
        message_id = db.add_message(msg_payload)
        if message_id:
            logger.info(
                "Posted message ID {} from '{}' to conversation {}.",
                message_id,
                sender_name,
                conversation_id,
            )
            _bump_conversation_metadata(db, conversation_id)
            try:
                from tldw_Server_API.app.core.Chat.conversation_enrichment import schedule_auto_tagging

                schedule_auto_tagging(db, conversation_id)
            except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    'Auto-tagging trigger skipped for conversation {}: {}',
                    conversation_id,
                    exc,
                )
        else:
            logger.error(
                "Failed to post message from '{}' to conversation {} (DB returned no ID without error).",
                sender_name,
                conversation_id,
            )
        return message_id
    except (CharactersRAGDBError, InputError, ConflictError) as exc:
        logger.error(
            "Error posting message from '{}' to conversation {}: {}",
            sender_name,
            conversation_id,
            exc,
        )
        raise
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error posting message to conv {}: {}',
            conversation_id,
            exc,
            exc_info=True,
        )
        # Always wrap unexpected exceptions to maintain consistent exception types
        raise CharactersRAGDBError(f"Unexpected error posting message: {exc}") from exc


def retrieve_message_details(
    db: CharactersRAGDB,
    message_id: str,
    character_name_for_placeholders: str,
    user_name_for_placeholders: Optional[str],
) -> Optional[dict[str, Any]]:
    """Retrieve a specific message by its ID and process placeholder content."""

    try:
        message_data = db.get_message_by_id(message_id)
        if not message_data:
            return None

        if "content" in message_data and isinstance(message_data["content"], str):
            message_data["content"] = replace_placeholders(
                message_data["content"],
                character_name_for_placeholders,
                user_name_for_placeholders,
            )
        return message_data
    except CharactersRAGDBError as exc:
        logger.error("Failed to retrieve message ID {}: {}", message_id, exc)
        return None
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error retrieving message ID {}: {}',
            message_id,
            exc,
            exc_info=True,
        )
        return None


def retrieve_conversation_messages_for_ui(
    db: CharactersRAGDB,
    conversation_id: str,
    character_name: str,
    user_name: Optional[str],
    limit: int = 2000,
    offset: int = 0,
    order: str = "ASC",
    rich_output: bool = False,
    char_first_message_hint: Optional[str] = None,
    character_id: Optional[int] = None,
) -> Union[list[tuple[Optional[str], Optional[str]]], list[dict[str, Optional[dict[str, Any]]]]]:
    """Retrieve and process conversation messages for UI display.

    When ``rich_output`` is True the response contains message metadata and attachments.
    """

    order_upper = order.upper()
    if order_upper not in ["ASC", "DESC"]:
        logger.warning("Invalid order '{}' for message retrieval. Defaulting to ASC.", order)
        order_upper = "ASC"

    try:
        raw_db_messages = db.get_messages_for_conversation(
            conversation_id,
            limit=limit,
            offset=offset,
            order_by_timestamp=order_upper,
        )

        messages_for_processing = (
            list(reversed(raw_db_messages)) if order_upper == "DESC" else raw_db_messages
        )

        additional_aliases, extra_user_aliases = _compute_additional_char_aliases(
            messages_for_processing,
            character_name,
            user_name,
            actual_user_sender_id_in_db="User",
        )

        # Optionally accept a pre-fetched character first_message to avoid a DB hit
        char_first_message_processed: Optional[str] = None
        if isinstance(char_first_message_hint, str) and char_first_message_hint.strip():
            char_first_message_processed = replace_placeholders(
                char_first_message_hint,
                character_name,
                user_name,
            ).strip()
        else:
            char_card = None
            if isinstance(character_id, int) and character_id > 0:
                try:
                    char_card = db.get_character_card_by_id(character_id)
                except CharactersRAGDBError:
                    char_card = None
                except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS:
                    char_card = None
            if char_card is None:
                try:
                    conversation = db.get_conversation_by_id(conversation_id)
                    conv_char_id = conversation.get("character_id") if conversation else None
                    if isinstance(conv_char_id, int) and conv_char_id > 0:
                        char_card = db.get_character_card_by_id(conv_char_id)
                except CharactersRAGDBError:
                    char_card = None
                except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS:
                    char_card = None
            try:
                if char_card is None:
                    char_card = db.get_character_card_by_name(character_name)
                if char_card and isinstance(char_card.get("first_message"), str):
                    char_first_message_processed = replace_placeholders(
                        char_card["first_message"],
                        character_name,
                        user_name,
                    ).strip()
            except CharactersRAGDBError:
                char_first_message_processed = None
            except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS:
                char_first_message_processed = None

        rich_history = process_db_messages_to_rich_ui_history(
            messages_for_processing,
            char_name_from_card=character_name,
            user_name_for_placeholders=user_name,
            actual_user_sender_id_in_db="User",
            actual_char_sender_id_in_db=character_name,
            additional_char_sender_ids=additional_aliases,
            additional_user_sender_ids=extra_user_aliases,
            char_first_message=char_first_message_processed,
            keep_trailing_user=True,
        )

        if order_upper == "DESC":
            rich_history = list(reversed(rich_history))

        if rich_output:
            return rich_history

        processed_ui_history = []
        for turn in rich_history:
            user_content = turn["user"]["content"] if turn["user"] else None
            character_content = None
            if turn["character"]:
                character_content = turn["character"]["content"]
            elif turn["non_character"]:
                character_content = turn["non_character"]["content"]
            processed_ui_history.append((user_content, character_content))
        return processed_ui_history

    except CharactersRAGDBError as exc:
        logger.error(
            'Failed to retrieve and process messages for conversation ID {}: {}',
            conversation_id,
            exc,
        )
        return []
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error retrieving UI messages for conversation {}: {}',
            conversation_id,
            exc,
            exc_info=True,
        )
        return []


def edit_message_content(
    db: CharactersRAGDB,
    message_id: str,
    new_content: str,
    expected_version: int,
) -> bool:
    """Update the text content of a specific message."""

    try:
        update_payload = {"content": new_content}
        success = db.update_message(message_id, update_payload, expected_version)
        if success:
            logger.info("Edited message {}", message_id)
        else:
            logger.warning(
                'Failed to edit message {} - version mismatch or message not found',
                message_id,
            )
        return bool(success)
    except (CharactersRAGDBError, InputError, ConflictError) as exc:
        logger.error("Failed to edit content for message ID {}: {}", message_id, exc)
        return False
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error editing message content for ID {}: {}',
            message_id,
            exc,
            exc_info=True,
        )
        return False


def set_message_ranking(
    db: CharactersRAGDB,
    message_id: str,
    ranking: int,
    expected_version: int,
) -> bool:
    """Set or update the ranking of a specific message."""

    update_payload = {"ranking": ranking}
    try:
        return bool(db.update_message(message_id, update_payload, expected_version))
    except (CharactersRAGDBError, InputError, ConflictError) as exc:
        logger.error("Failed to set ranking for message ID {}: {}", message_id, exc)
        return False
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error setting message ranking for ID {}: {}',
            message_id,
            exc,
            exc_info=True,
        )
        return False


def remove_message_from_conversation(
    db: CharactersRAGDB,
    message_id: str,
    expected_version: int,
) -> bool:
    """Soft-delete a message from a conversation."""

    try:
        return bool(db.soft_delete_message(message_id, expected_version))
    except (CharactersRAGDBError, ConflictError) as exc:
        logger.error("Failed to remove message ID {}: {}", message_id, exc)
        return False
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error removing message ID {}: {}',
            message_id,
            exc,
            exc_info=True,
        )
        return False


def find_messages_in_conversation(
    db: CharactersRAGDB,
    conversation_id: str,
    search_query: str,
    character_name_for_placeholders: str,
    user_name_for_placeholders: Optional[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search for messages within a specific conversation by content."""

    try:
        found_messages = db.search_messages_by_content(
            content_query=search_query,
            conversation_id=conversation_id,
            limit=limit,
        )

        processed_results = []
        for msg_data in found_messages:
            if "content" in msg_data and isinstance(msg_data["content"], str):
                msg_data["content"] = replace_placeholders(
                    msg_data["content"],
                    character_name_for_placeholders,
                    user_name_for_placeholders,
                )
            processed_results.append(msg_data)
        return processed_results
    except CharactersRAGDBError as exc:
        logger.error(
            "Failed to search messages in conversation ID {} for '{}': {}",
            conversation_id,
            search_query,
            exc,
        )
        return []
    except _CHAR_CHAT_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(
            'Unexpected error searching messages in conversation {}: {}',
            conversation_id,
            exc,
            exc_info=True,
        )
        return []

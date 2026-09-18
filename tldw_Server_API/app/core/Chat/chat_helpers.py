# chat_helpers.py
# Description: Helper functions for chat endpoint to improve modularity and maintainability
#
# Imports
import asyncio
import datetime
import json
import random
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Utils.image_validation import estimate_decoded_size, get_max_base64_bytes

#######################################################################################################################
#
# Character Name Sanitization:

# Characters to remove from character names for OpenAI API compatibility
# OpenAI requires name to match pattern ^[^\s<|\\/>]+$ (no spaces or special chars)
_UNSAFE_NAME_CHARS = frozenset(' \t\n\r<>|\\/"\'`:;!@#$%^&*()+=[]{}')


def _sanitize_character_name(name: str, replacement: str = '_') -> str:
    """
    Sanitize a character name for OpenAI API compatibility.

    Args:
        name: Raw character name
        replacement: Character to use for replacements (default: underscore)

    Returns:
        Sanitized name safe for OpenAI 'name' field, or empty string if all chars removed
    """
    if not isinstance(name, str):
        return ''

    # Replace unsafe characters
    sanitized = ''.join(
        replacement if c in _UNSAFE_NAME_CHARS else c
        for c in name
    )

    # Remove consecutive replacement chars and leading/trailing replacements
    while replacement + replacement in sanitized:
        sanitized = sanitized.replace(replacement + replacement, replacement)
    sanitized = sanitized.strip(replacement)

    # Truncate to reasonable length (64 chars max for OpenAI)
    return sanitized[:64]


#######################################################################################################################
#
# Request Validation Functions:

async def validate_request_payload(
    request_data: ChatCompletionRequest,
    max_messages: int = 1000,
    max_images: int = 10,
    max_text_length: int = 400_000,
    *,
    enforce_image_max_bytes: bool = False,
    max_image_bytes: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """
    Validate the chat completion request payload.

    Args:
        request_data: The chat completion request
        max_messages: Maximum allowed messages
        max_images: Maximum allowed images
        max_text_length: Maximum text length per message
        enforce_image_max_bytes: Whether to enforce decoded byte limits on data URI images
        max_image_bytes: Optional max decoded bytes for data URI images (defaults to config)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check messages list
    if not request_data.messages:
        return False, "Messages list cannot be empty."

    if len(request_data.messages) > max_messages:
        return False, f"Too many messages (max {max_messages}, got {len(request_data.messages)})."

    # Count images across all messages
    total_image_parts = 0
    for msg_idx, msg_model in enumerate(request_data.messages):
        if isinstance(msg_model.content, list):
            for part_idx, part in enumerate(msg_model.content):
                part_type = getattr(part, 'type', None)
                if part_type is None and isinstance(part, dict):
                    part_type = part.get("type")
                if part_type == 'image_url':
                    total_image_parts += 1
                    if enforce_image_max_bytes:
                        url_val = _extract_image_url(part)
                        if not url_val:
                            return False, f"Image at index {part_idx} in message {msg_idx} is missing a data URI."
                        if isinstance(url_val, str) and url_val.startswith("data:image"):
                            ok, error = _validate_image_size(url_val, max_image_bytes)
                            if not ok:
                                return False, f"Image at index {part_idx} in message {msg_idx} too large ({error})."
                elif part_type == 'text':
                    text_content = part.get("text") if isinstance(part, dict) else getattr(part, 'text', None)
                    if isinstance(text_content, str) and len(text_content) > max_text_length:
                        return False, f"Text part at index {part_idx} in message {msg_idx} too long."
        elif isinstance(msg_model.content, str) and len(msg_model.content) > max_text_length:
            return False, f"Message at index {msg_idx} text too long."

    if total_image_parts > max_images:
        return False, f"Too many images in request (max {max_images}, found {total_image_parts})."

    return True, None


def _extract_image_url(part: Any) -> Optional[str]:
    """Best-effort extraction of image_url.url from content parts."""
    if part is None:
        return None
    image_url = None
    image_url = part.get("image_url") if isinstance(part, dict) else getattr(part, "image_url", None)
    if isinstance(image_url, dict):
        return image_url.get("url")
    return getattr(image_url, "url", None)


def _validate_image_size(url_val: str, max_image_bytes: Optional[int]) -> tuple[bool, str]:
    """Validate data URI size using decoded-byte estimate."""
    if not isinstance(url_val, str) or not url_val.startswith("data:image"):
        return False, "invalid data URI"
    header, sep, data = url_val.partition(",")
    if not sep or "base64" not in header.lower():
        return False, "invalid base64 data URI"
    cleaned = "".join(data.split())
    try:
        estimated = estimate_decoded_size(cleaned)
    except Exception:
        return False, "unable to estimate size"
    limit = int(max_image_bytes or get_max_base64_bytes())
    if estimated > limit:
        return False, f"max {limit} bytes"
    return True, ""


#######################################################################################################################
#
# Character and Conversation Context Functions:

async def get_or_create_character_context(
    db: CharactersRAGDB,
    character_id: Optional[str],
    loop
) -> tuple[Optional[dict[str, Any]], Optional[int]]:
    """
    Get or create character context for the chat.

    Args:
        db: Database instance
        character_id: Optional character ID (string or numeric)
        loop: Event loop for async operations

    Returns:
        Tuple of (character_card, character_db_id)
    """
    character_card = None
    final_character_db_id = None

    if character_id:
        try:
            # Try as integer first
            char_id_int = int(character_id)
            character_card = await asyncio.to_thread(db.get_character_card_by_id, char_id_int)
        except ValueError:
            # Not an integer, try by name
            character_card = await asyncio.to_thread(db.get_character_card_by_name, character_id)

        if character_card:
            final_character_db_id = character_card['id']
            logger.info(f"Loaded character '{character_card['name']}' (ID: {final_character_db_id})")

    # Fall back to default character if needed
    if not character_card:
        character_card = await asyncio.to_thread(db.get_character_card_by_name, DEFAULT_CHARACTER_NAME)
        if character_card:
            final_character_db_id = character_card['id']
            logger.info(f"Using default character '{DEFAULT_CHARACTER_NAME}' (ID: {final_character_db_id})")
        else:
            character_card, final_character_db_id = await _ensure_default_character(db, loop)

    return character_card, final_character_db_id


async def _ensure_default_character(
    db: CharactersRAGDB,
    loop,
) -> tuple[Optional[dict[str, Any]], Optional[int]]:
    """
    Ensure the default character exists for the current client.

    Returns:
        Tuple of (character_card, character_db_id)
    """

    def _create_default() -> tuple[Optional[dict[str, Any]], Optional[int]]:
        try:
            existing = db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
            if existing:
                return existing, existing.get("id")

            payload = {
                "name": DEFAULT_CHARACTER_NAME,
                "description": "Automatically created default assistant persona.",
                "system_prompt": "You are a helpful AI assistant.",
                "personality": "Supportive and concise.",
                "scenario": "General assistance",
                "first_message": "Hello! I'm your Helpful AI Assistant. How can I help you today?",
                "creator_notes": "Auto-generated default character.",
                "creator": "System",
                "tags": json.dumps(["default", "assistant"]),
                "client_id": getattr(db, "client_id", None),
            }
            if not payload["client_id"]:
                logger.warning(
                    "Cannot create default character '{}' because client_id is missing on DB instance.",
                    DEFAULT_CHARACTER_NAME,
                )
                return None, None

            new_id = db.add_character_card(payload)
            if not new_id:
                logger.error("add_character_card returned None while creating default character '{}'.", DEFAULT_CHARACTER_NAME)
                return None, None

            created = db.get_character_card_by_id(new_id)
            if created:
                logger.info("Created default character '{}' with ID {}.", DEFAULT_CHARACTER_NAME, new_id)
                return created, created.get("id")

            # Fallback lookup by name if direct fetch failed
            fetched = db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
            if fetched:
                return fetched, fetched.get("id")
            return None, None
        except (CharactersRAGDBError, InputError, ConflictError) as db_error:
            logger.error("Failed to ensure default character '{}': {}", DEFAULT_CHARACTER_NAME, db_error)
            return None, None
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Unexpected error ensuring default character '{}': {}", DEFAULT_CHARACTER_NAME, exc)
            return None, None

    return await asyncio.to_thread(_create_default)


async def get_or_create_conversation(
    db: CharactersRAGDB,
    conversation_id: Optional[str],
    character_id: int,
    character_name: str,
    client_id: str,
    loop
) -> tuple[str, bool]:
    """
    Get existing conversation or create a new one with proper concurrency handling.

    Args:
        db: Database instance
        conversation_id: Optional existing conversation ID
        character_id: Character database ID
        character_name: Character name for title
        client_id: Client identifier
        loop: Event loop

    Returns:
        Tuple of (conversation_id, was_created)
    """
    was_created = False

    if conversation_id:
        # Verify existing conversation
        conv_details = await asyncio.to_thread(db.get_conversation_by_id, conversation_id)
        if conv_details:
            # Validate ownership and character match
            if (conv_details.get('character_id') == character_id and
                conv_details.get('client_id') == client_id):
                return conversation_id, False
            else:
                logger.warning(
                    f"Conversation {conversation_id} mismatch - expected char:{character_id}, "
                    f"client:{client_id}, got char:{conv_details.get('character_id')}, "
                    f"client:{conv_details.get('client_id')}"
                )
                conversation_id = None

    # Create new conversation if needed with retry logic for race conditions
    if not conversation_id:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        title = f"{character_name} ({timestamp})"
        conv_data = {
            'character_id': character_id,
            'title': title,
            'client_id': client_id
        }

        # Try to create with retry on conflict - with overall timeout
        max_retries = 3
        overall_timeout = 5.0  # Maximum 5 seconds for all retries
        start_time = datetime.datetime.now(datetime.timezone.utc).timestamp()

        for attempt in range(max_retries):
            # Check overall timeout
            elapsed = datetime.datetime.now(datetime.timezone.utc).timestamp() - start_time
            if elapsed >= overall_timeout:
                logger.error(f"Conversation creation timed out after {elapsed:.2f}s")
                raise TimeoutError(f"Conversation creation timed out after {overall_timeout}s")

            try:
                def _create_conversation_sync() -> Optional[str]:
                    with db.transaction():
                        return db.add_conversation(conv_data)

                conversation_id = await asyncio.to_thread(_create_conversation_sync)
                was_created = True
                logger.info(f"Created new conversation '{conversation_id}' for character {character_id}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    # Add small random delay to reduce collision probability
                    await asyncio.sleep(0.05 + random.random() * 0.1)
                    # Update timestamp for uniqueness
                    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                    title = f"{character_name} ({timestamp})"
                    conv_data['title'] = title
                else:
                    logger.error(f"Failed to create conversation after {max_retries} attempts: {e}")
                    raise

    return conversation_id, was_created


#######################################################################################################################
#
# History Loading Functions:

async def load_conversation_history(
    db: CharactersRAGDB,
    conversation_id: str,
    character_card: Optional[dict[str, Any]],
    limit: int = 20,
    loop = None
) -> list[dict[str, Any]]:
    """
    Load conversation history from database.

    Args:
        db: Database instance
        conversation_id: Conversation ID
        character_card: Character card for context
        limit: Maximum messages to load
        loop: Event loop

    Returns:
        List of messages in OpenAI format
    """
    if not loop:
        import asyncio
        loop = asyncio.get_running_loop()

    historical_messages = []

    try:
        # Load messages from database
        raw_history = await asyncio.to_thread(db.get_messages_for_conversation,
            conversation_id,
            limit,
            0,
            "ASC"
        )

        for db_msg in raw_history:
            sender_val = str(db_msg.get("sender", "") or "")
            sender_lower = sender_val.lower()
            if sender_lower == "user":
                role = "user"
            elif sender_lower == "tool":
                role = "tool"
            else:
                role = "assistant"

            # Build message content
            msg_parts = []

            # Add text content
            text_content = db_msg.get("content", "")
            if text_content:
                # Apply placeholder replacement if using character context (skip tool outputs)
                if character_card and role != "tool":
                    from tldw_Server_API.app.core.Character_Chat.Character_Chat_Lib_facade import replace_placeholders
                    char_name = character_card.get('name', "Assistant")
                    text_content = replace_placeholders(text_content, char_name, "User")
                msg_parts.append({"type": "text", "text": text_content})

            # Add images if present (supports multiple attachments)
            raw_images = db_msg.get("images") or []
            if (not raw_images) and db_msg.get("image_data") and db_msg.get("image_mime_type"):
                raw_images = [{
                    "position": 0,
                    "image_data": db_msg.get("image_data"),
                    "image_mime_type": db_msg.get("image_mime_type"),
                }]

            if raw_images:
                import base64
                for image_entry in raw_images:
                    try:
                        img_bytes = image_entry.get("image_data")
                        if isinstance(img_bytes, memoryview):
                            img_bytes = img_bytes.tobytes()
                        if not img_bytes:
                            continue
                        img_mime = image_entry.get("image_mime_type") or db_msg.get("image_mime_type") or "image/png"
                        b64_img = await loop.run_in_executor(None, base64.b64encode, img_bytes)
                        msg_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_mime};base64,{b64_img.decode('utf-8')}"}
                        })
                    except Exception as e:
                        logger.warning(f"Error encoding image from history (msg_id {db_msg.get('id')}): {e}")

            # Create message entry
            if msg_parts:
                hist_entry: dict[str, Any] = {"role": role}

                if len(msg_parts) == 1:
                    sole_part = msg_parts[0]
                    if sole_part.get("type") == "text":
                        hist_entry["content"] = sole_part.get("text", "")
                    else:
                        hist_entry["content"] = [sole_part]
                else:
                    hist_entry["content"] = msg_parts

                # Add sanitized character name for assistant messages
                if role == "assistant" and character_card and character_card.get('name'):
                    safe_name = _sanitize_character_name(character_card.get('name', ''))
                    if safe_name:
                        hist_entry["name"] = safe_name

                historical_messages.append(hist_entry)

        logger.info(f"Loaded {len(historical_messages)} historical messages for conversation {conversation_id}")

    except Exception as e:
        logger.error(f"Error loading conversation history: {e}", exc_info=True)

    return historical_messages


#######################################################################################################################
#
# Message Processing Functions:

async def prepare_llm_messages(
    request_messages: list[Any],
    historical_messages: list[dict[str, Any]],
    character_card: Optional[dict[str, Any]] = None
) -> list[dict[str, Any]]:
    """
    Prepare messages for LLM API call.

    Args:
        request_messages: Messages from the request
        historical_messages: Historical conversation messages
        character_card: Optional character context

    Returns:
        List of messages ready for LLM
    """
    llm_messages = []

    # Add historical messages
    llm_messages.extend(historical_messages)

    # Process current request messages
    for msg_model in request_messages:
        if msg_model.role == "system":
            continue  # System messages handled separately

        msg_dict = msg_model.model_dump(exclude_none=True)

        # Add character name for assistant messages (sanitized for OpenAI compatibility)
        if msg_model.role == "assistant" and character_card and character_card.get('name'):
            # OpenAI requires name to match pattern ^[^\\s<|\\\/>]+$ (no spaces or special chars)
            name = _sanitize_character_name(character_card.get('name', ''))
            if name:  # Only add if name is not empty after sanitization
                msg_dict["name"] = name

        llm_messages.append(msg_dict)

    return llm_messages


def extract_system_message(
    request_messages: list[Any],
    character_card: Optional[dict[str, Any]] = None
) -> Optional[str]:
    """
    Extract system message from request or character card.

    Args:
        request_messages: Messages from the request
        character_card: Optional character card with system prompt

    Returns:
        System message string or None
    """
    # Check for explicit system message in request
    for msg in request_messages:
        if msg.role == 'system' and isinstance(msg.content, str):
            return msg.content

    # Fall back to character system prompt
    if character_card and character_card.get('system_prompt'):
        return character_card.get('system_prompt')

    return None


#######################################################################################################################
#
# Response Processing Functions:

def extract_response_content(llm_response: Any) -> Optional[str]:
    """
    Extract text content from LLM response.

    Args:
        llm_response: Response from LLM API

    Returns:
        Extracted text content or None
    """
    if isinstance(llm_response, str):
        return llm_response
    elif isinstance(llm_response, dict):
        # OpenAI-style response
        choices = llm_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content")

    return None


#######################################################################################################################
#
# Provider Validation Functions:

def validate_provider_configuration(
    provider: str,
    api_keys: dict[str, str]
) -> tuple[bool, Optional[str]]:
    """
    Validate that a provider is properly configured.

    Args:
        provider: Provider name
        api_keys: Dictionary of API keys

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
    except Exception:
        def provider_requires_api_key(_provider: str) -> bool:  # type: ignore[misc]
            return True

    if provider_requires_api_key(provider):
        api_key = api_keys.get(provider)
        if not api_key:
            return False, f"API key for provider '{provider}' is missing or not configured."

    return True, None


#
# End of chat_helpers.py
#######################################################################################################################

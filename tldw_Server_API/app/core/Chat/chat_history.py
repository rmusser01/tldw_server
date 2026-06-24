# chat_history.py
# Description: Chat history persistence and content preparation helpers.
"""
Utility functions for persisting chat history, exporting conversations, and
preparing media content for chat interactions.

Note:
  - The primary chat persistence path for `/api/v1/chat/completions` now lives in
    the Chat service module (`chat_service.build_context_and_messages` and
    related helpers).
  - `save_chat_history_to_db_wrapper` is retained for legacy callers and
    Character Chat utilities; new code should prefer the chat module's unified
    conversation/message persistence pipeline.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import time
from datetime import datetime
from typing import Any, Optional, Union

from loguru import logger  # noqa: F401 - retained for future use/debug parity

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.core.Chat.message_utils import should_persist_message_role
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.Utils.Utils import generate_unique_filename, logging

HistoryList = list[Union[tuple[Optional[str], Optional[str]], dict[str, Any]]]
MediaContent = Optional[dict[str, Any]]


def save_chat_history_to_db_wrapper(
    db: CharactersRAGDB,
    chatbot_history: list[dict[str, Any]],
    conversation_id: str | None,
    media_content_for_char_assoc: dict[str, Any] | None,
    media_name_for_char_assoc: str | None = None,
    character_name_for_chat: str | None = None,
) -> tuple[str | None, str]:
    """
    Persist a chat history into the ChaChaNotes database, creating or updating a conversation record.

    This function is maintained for legacy integrations and Character Chat
    utilities. For new features built on `/api/v1/chat/completions`, prefer
    using the Chat module's conversation/message persistence via
    `chat_service.build_context_and_messages` and `_save_message_turn_to_db`.
    """
    log_counter("save_chat_history_to_db_attempt")
    start_time = time.time()
    logging.info(
        "Saving chat history (OpenAI format). Conversation ID: %s, Character: %s, Num messages: %s",
        conversation_id,
        character_name_for_chat,
        len(chatbot_history),
    )

    try:
        associated_character_id: int | None = None
        final_character_name_for_title = "Unknown Character"

        char_lookup_name = character_name_for_chat or media_name_for_char_assoc

        if not char_lookup_name and media_content_for_char_assoc:
            content_details = media_content_for_char_assoc.get("content")
            if isinstance(content_details, str):
                try:
                    content_details = json.loads(content_details)
                except json.JSONDecodeError:
                    content_details = {}
            if isinstance(content_details, dict):
                char_lookup_name = content_details.get("title")

        if char_lookup_name:
            try:
                character = db.get_character_card_by_name(char_lookup_name)
                if character:
                    associated_character_id = character["id"]
                    final_character_name_for_title = character["name"]
                    logging.info(
                        "Chat will be associated with specific character '%s' (ID: %s).",
                        final_character_name_for_title,
                        associated_character_id,
                    )
                else:
                    logging.error(
                        "Intended specific character '%s' not found in DB. Chat save aborted.",
                        char_lookup_name,
                    )
                    return conversation_id, (
                        f"Error: Specific character '{char_lookup_name}' intended for this chat was not found. "
                        "Cannot save chat."
                    )
            except CharactersRAGDBError as exc:
                logging.error("DB error looking up specific character '%s': %s", char_lookup_name, exc)
                return conversation_id, f"DB error finding specific character: {exc}"
        else:
            logging.info("No specific character name for chat. Using %s.", DEFAULT_CHARACTER_NAME)
            try:
                default_char = db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
                if default_char:
                    associated_character_id = default_char["id"]
                    final_character_name_for_title = default_char["name"]
                    logging.info(
                        "Chat will be associated with '%s' (ID: %s).",
                        DEFAULT_CHARACTER_NAME,
                        associated_character_id,
                    )
                else:
                    logging.error(
                        "'%s' is missing from the DB and no specific character was provided. Chat save aborted.",
                        DEFAULT_CHARACTER_NAME,
                    )
                    return conversation_id, (
                        f"Error: Critical - '{DEFAULT_CHARACTER_NAME}' is missing. Cannot save chat."
                    )
            except CharactersRAGDBError as exc:
                logging.error("DB error looking up '%s': %s", DEFAULT_CHARACTER_NAME, exc)
                return conversation_id, f"DB error finding '{DEFAULT_CHARACTER_NAME}': {exc}"

        if associated_character_id is None:
            logging.critical("Logic error: associated_character_id is None after character lookup. Chat save aborted.")
            return conversation_id, "Critical internal error: Could not determine character for chat."

        current_conversation_id = conversation_id
        is_new_conversation = not current_conversation_id

        if is_new_conversation:
            conv_title_base = f"Chat with {final_character_name_for_title}"
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            conversation_title = f"{conv_title_base} ({timestamp_str})"

            conv_data = {
                "character_id": associated_character_id,
                "title": conversation_title,
                "client_id": db.client_id,
            }

            try:
                current_conversation_id = db.add_conversation(conv_data)
                if not current_conversation_id:
                    return None, "Failed to create new conversation in DB."
                logging.info(
                    "Created new conversation %s for character '%s'.",
                    current_conversation_id,
                    final_character_name_for_title,
                )
            except (InputError, ConflictError, CharactersRAGDBError) as exc:
                logging.error("Error creating new conversation: %s", exc, exc_info=True)
                return None, f"Error creating conversation: {exc}"
        else:
            logging.info(
                "Resaving history for existing conv ID: %s. Char context ID: %s ('%s')",
                current_conversation_id,
                associated_character_id,
                final_character_name_for_title,
            )

        try:
            if not chatbot_history:
                logging.warning("Chatbot history is empty; nothing to save. Returning current conversation ID.")
                return current_conversation_id, "No chat history to save."

            conv_version_for_update: int | None = None
            conv_title_for_update: str | None = None

            with db.transaction():
                if not is_new_conversation:
                    existing_conv_details = db.get_conversation_by_id(current_conversation_id)
                    if not existing_conv_details:
                        logging.error("Cannot resave: Conversation %s not found.", current_conversation_id)
                        return (
                            current_conversation_id,
                            f"Error: Conversation {current_conversation_id} not found for resaving.",
                        )

                    if existing_conv_details.get("character_id") != associated_character_id:
                        existing_char = db.get_character_card_by_id(existing_conv_details.get("character_id"))
                        existing_char_name = (
                            existing_char.get("name")
                            if existing_char
                            else f"ID {existing_conv_details.get('character_id')}"
                        )
                        logging.error(
                            "Cannot resave: Conversation %s (for char '%s') does not match current character context '%s' (ID: %s).",
                            current_conversation_id,
                            existing_char_name,
                            final_character_name_for_title,
                            associated_character_id,
                        )
                        return (
                            current_conversation_id,
                            "Error: Mismatch in character association for resaving chat. The conversation belongs to a different character.",
                        )

                    conv_version_for_update = existing_conv_details.get("version")
                    conv_title_for_update = existing_conv_details.get("title")
                    existing_messages_for_replacement = db.get_messages_for_conversation(
                        current_conversation_id, limit=10000, order_by_timestamp="ASC"
                    )
                    logging.info(
                        "Found %s existing messages to soft-delete for conv %s.",
                        len(existing_messages_for_replacement),
                        current_conversation_id,
                    )
                    for msg in existing_messages_for_replacement:
                        db.soft_delete_message(msg["id"], msg["version"])

                message_save_count = 0
                for index, message_obj in enumerate(chatbot_history):
                    sender = message_obj.get("role")
                    if not should_persist_message_role(sender):
                        logging.debug("Skipping message with role '%s' at index %s", sender, index)
                        continue

                    text_content_parts: list[str] = []
                    image_data_bytes: bytes | None = None
                    image_mime_type_str: str | None = None
                    image_entries: list[tuple[bytes, str]] = []
                    content_data = message_obj.get("content")

                    if isinstance(content_data, str):
                        text_content_parts.append(content_data)
                    elif isinstance(content_data, list):
                        for part in content_data:
                            part_type = part.get("type") if isinstance(part, dict) else None
                            if part_type == "text":
                                text_content_parts.append(part.get("text", ""))
                            elif part_type == "image_url":
                                image_url_dict = part.get("image_url", {})
                                url_str = image_url_dict.get("url", "") if isinstance(image_url_dict, dict) else ""
                                if url_str.startswith("data:") and ";base64," in url_str:
                                    try:
                                        header, b64_data = url_str.split(";base64,", 1)
                                        image_mime_type = header.split("data:", 1)[1] if "data:" in header else None
                                        if image_mime_type:
                                            decoded_bytes = base64.b64decode(b64_data)
                                            image_entries.append((decoded_bytes, image_mime_type))
                                            if image_data_bytes is None:
                                                image_data_bytes = decoded_bytes
                                                image_mime_type_str = image_mime_type
                                            logging.debug(
                                                "Decoded image for saving (MIME: %s, Size: %s) for msg %s in conv %s",
                                                image_mime_type,
                                                len(decoded_bytes) if decoded_bytes else 0,
                                                index,
                                                current_conversation_id,
                                            )
                                        else:
                                            logging.warning("Could not parse MIME type from data URI: %s", url_str[:60])
                                    except Exception as exc:
                                        logging.warning(
                                            "Failed to decode Base64 image at index %s for conv %s: %s",
                                            index,
                                            current_conversation_id,
                                            exc,
                                        )
                                elif url_str:
                                    logging.debug("Storing non-data image URL as text: %s", url_str)
                                    text_content_parts.append(f"<Image URL: {url_str}>")
                    elif content_data is not None:
                        logging.warning("Unsupported message content type at index %s: %s", index, type(content_data))
                        text_content_parts.append(f"<Unsupported content type: {type(content_data)}>")

                    final_text_content = "\n".join(text_content_parts).strip()
                    if not final_text_content and not image_data_bytes and not image_entries:
                        logging.warning(
                            "Skipping empty message (no text or decodable image) at index %s for conv %s",
                            index,
                            current_conversation_id,
                        )
                        continue

                    payload = {
                        "conversation_id": current_conversation_id,
                        "sender": sender,
                        "content": final_text_content,
                        "image_data": image_data_bytes,
                        "image_mime_type": image_mime_type_str,
                        "client_id": db.client_id,
                    }
                    if image_entries:
                        payload["images"] = [{"data": b, "mime": m} for b, m in image_entries]
                    db.add_message(payload)
                    message_save_count += 1
                logging.info(
                    "Successfully saved %s messages to conversation %s.", message_save_count, current_conversation_id
                )

                # Update conversation metadata with retry logic for version conflicts
                # This handles TOCTOU race conditions where another process may have
                # updated the conversation between our version fetch and update
                if not is_new_conversation and conv_version_for_update is not None:
                    max_retries = 3
                    for retry_attempt in range(max_retries):
                        try:
                            if retry_attempt > 0:
                                # Re-fetch conversation to get fresh version
                                fresh_conv = db.get_conversation_by_id(current_conversation_id)
                                if fresh_conv:
                                    conv_version_for_update = fresh_conv.get("version")
                                    conv_title_for_update = fresh_conv.get("title")
                                else:
                                    logging.warning(
                                        "Conversation %s disappeared during retry. Messages saved, metadata update skipped.",
                                        current_conversation_id,
                                    )
                                    break
                            db.update_conversation(
                                current_conversation_id,
                                {"title": conv_title_for_update},
                                conv_version_for_update,
                            )
                            break  # Success
                        except ConflictError:
                            if retry_attempt < max_retries - 1:
                                logging.debug(
                                    "Version conflict on conversation %s (attempt %d/%d), retrying...",
                                    current_conversation_id,
                                    retry_attempt + 1,
                                    max_retries,
                                )
                            else:
                                # Final attempt failed, log warning but continue
                                # Messages were saved successfully
                                logging.warning(
                                    "Version conflict updating conversation %s metadata after %d retries. "
                                    "Messages were saved successfully.",
                                    current_conversation_id,
                                    max_retries,
                                )

                if message_save_count > 0:
                    try:
                        from tldw_Server_API.app.core.Chat.conversation_enrichment import schedule_auto_tagging

                        schedule_auto_tagging(db, current_conversation_id)
                    except Exception as exc:
                        logging.debug(
                            "Auto-tagging trigger skipped for conversation %s: %s",
                            current_conversation_id,
                            exc,
                        )

        except (InputError, ConflictError, CharactersRAGDBError) as exc:
            logging.error("Error saving messages to conversation %s: %s", current_conversation_id, exc, exc_info=True)
            return current_conversation_id, f"Error saving messages: {exc}"

        save_duration = time.time() - start_time
        log_histogram("save_chat_history_to_db_duration", save_duration)
        log_counter("save_chat_history_to_db_success")
        return current_conversation_id, "Chat history saved successfully!"

    except Exception as exc:
        log_counter("save_chat_history_to_db_error", labels={"error": str(exc)})
        error_message = f"Failed to save chat history due to an unexpected error: {exc}"
        logging.error(error_message, exc_info=True)
        return conversation_id, error_message


def save_chat_history(
    history: HistoryList,
    conversation_id: str | None,
    media_content: MediaContent,
    db_instance: CharactersRAGDB | None = None,
) -> str | None:
    """
    Export chat history into a uniquely named JSON file and return the path.
    """
    log_counter("save_chat_history_attempt")
    start_time = time.time()
    try:
        content, conversation_name = generate_chat_history_content(
            history,
            conversation_id,
            media_content,
            db_instance=db_instance,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_conversation_name = re.sub(r"[^a-zA-Z0-9_-]", "_", conversation_name)
        base_filename = f"{safe_conversation_name}_{timestamp}.json"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        unique_filename = generate_unique_filename(os.path.dirname(temp_file_path), base_filename)
        final_path = os.path.join(os.path.dirname(temp_file_path), unique_filename)
        os.rename(temp_file_path, final_path)

        save_duration = time.time() - start_time
        log_histogram("save_chat_history_duration", save_duration)
        log_counter("save_chat_history_success")
        return final_path
    except Exception as exc:
        log_counter("save_chat_history_error", labels={"error": str(exc)})
        logging.error("Error saving chat history: %s", exc)
        return None


def get_conversation_name(
    conversation_id: str | None,
    db_instance: CharactersRAGDB | None = None,
) -> str | None:
    """
    Retrieve a conversation title from the database, if available.
    """
    if db_instance and conversation_id:
        try:
            conversation = db_instance.get_conversation_by_id(conversation_id)
            if conversation and conversation.get("title"):
                return conversation["title"]
        except Exception as exc:
            logging.warning(
                "Could not fetch conversation title from DB for %s: %s",
                conversation_id,
                exc,
            )
    return None


def generate_chat_history_content(
    history: HistoryList,
    conversation_id: str | None,
    media_content: MediaContent,
    db_instance: CharactersRAGDB | None = None,
) -> tuple[str, str]:
    """
    Generate JSON content summarising the conversation and return it along with a name.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    conversation_name = None

    if conversation_id:
        conversation_name = get_conversation_name(conversation_id, db_instance)

    if not conversation_name:
        media_name_extracted = extract_media_name(media_content)
        conversation_name = f"{media_name_extracted}-chat-{timestamp}" if media_name_extracted else f"chat-{timestamp}"

    chat_data: dict[str, Any] = {
        "conversation_id": conversation_id,
        "conversation_name": conversation_name,
        "timestamp": timestamp,
        "history": [],
    }

    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg is not None:
                chat_data["history"].append({"role": "user", "content": user_msg})
            if bot_msg is not None:
                chat_data["history"].append({"role": "assistant", "content": bot_msg})
        elif isinstance(item, dict) and "role" in item and "content" in item:
            chat_data["history"].append(item)
        else:
            logging.warning("Unexpected item format in history for JSON export: %s", item)

    return json.dumps(chat_data, indent=2), conversation_name


def extract_media_name(media_content: MediaContent) -> str | None:
    """
    Attempt to derive a human-readable media name from the provided content payload.
    """
    if not media_content:
        return None

    content_field = media_content.get("content")
    parsed_content = None

    if isinstance(content_field, str):
        try:
            parsed_content = json.loads(content_field)
        except json.JSONDecodeError:
            parsed_content = {"name": content_field}
    elif isinstance(content_field, dict):
        parsed_content = content_field

    if isinstance(parsed_content, dict):
        name = (
            parsed_content.get("title")
            or parsed_content.get("name")
            or parsed_content.get("media_title")
            or parsed_content.get("webpage_title")
        )
        if name:
            return name

    name_top_level = media_content.get("title") or media_content.get("name") or media_content.get("media_title")
    if name_top_level:
        return name_top_level

    logging.warning(
        "Could not extract a clear media name from media_content: %s",
        str(media_content)[:200],
    )
    return None


def update_chat_content(
    selected_item: str | None,
    use_content: bool,
    use_summary: bool,
    use_prompt: bool,
    item_mapping: dict[str, str],
    db_instance: CharactersRAGDB,
) -> tuple[dict[str, str], list[str]]:
    """
    Fetch content from stored notes to assemble media context for chat requests.
    """
    log_counter("update_chat_content_attempt")
    start_time = time.time()
    logging.debug("Debug - Update Chat Content - Selected Item: %s", selected_item)

    output_media_content_for_chat: dict[str, str] = {}
    selected_parts_names: list[str] = []

    if selected_item and selected_item in item_mapping:
        note_id = item_mapping[selected_item]

        try:
            note_data = db_instance.get_note_by_id(note_id)
        except CharactersRAGDBError as exc:
            logging.error("Error fetching note %s for chat content: %s", note_id, exc, exc_info=True)
            note_data = None
        except Exception as exc:
            logging.error("Unexpected error fetching note %s: %s", note_id, exc, exc_info=True)
            note_data = None

        if note_data:
            raw_note_content_field = note_data.get("content", "")
            structured_content_from_note: dict[str, str] = {}

            if (
                isinstance(raw_note_content_field, str)
                and raw_note_content_field.strip().startswith("{")
                and raw_note_content_field.strip().endswith("}")
            ):
                try:
                    structured_content_from_note = json.loads(raw_note_content_field)
                except json.JSONDecodeError:
                    logging.warning(
                        "Failed to parse note content as JSON for note %s. Treating as plain text.",
                        note_id,
                    )

            if not structured_content_from_note and isinstance(raw_note_content_field, str):
                structured_content_from_note["content"] = raw_note_content_field

            if use_content:
                content_text = structured_content_from_note.get("content")
                if content_text:
                    output_media_content_for_chat["content"] = content_text
                    selected_parts_names.append("content")

            if use_summary:
                summary_text = structured_content_from_note.get("summary")
                if summary_text:
                    output_media_content_for_chat["summary"] = summary_text
                    selected_parts_names.append("summary")

            if use_prompt:
                prompt_text = structured_content_from_note.get("prompt")
                if prompt_text:
                    output_media_content_for_chat["prompt"] = prompt_text
                    selected_parts_names.append("prompt")

            if not selected_parts_names:
                fallback_text = structured_content_from_note.get("content")
                if fallback_text:
                    output_media_content_for_chat["content"] = fallback_text
                    selected_parts_names.append("content")
        else:
            logging.warning("No note data found for selected item '%s' (note id: %s).", selected_item, note_id)
    else:
        logging.warning("Selected item missing or not present in mapping: %s", selected_item)
        log_counter(
            "update_chat_content_error",
            labels={"error": "No item selected or item not in mapping"},
        )

    update_duration = time.time() - start_time
    log_histogram("update_chat_content_duration", update_duration)
    log_counter("update_chat_content_success" if selected_parts_names else "update_chat_content_noop")

    return output_media_content_for_chat, selected_parts_names


__all__ = [
    "save_chat_history_to_db_wrapper",
    "save_chat_history",
    "get_conversation_name",
    "generate_chat_history_content",
    "extract_media_name",
    "update_chat_content",
    "DEFAULT_CHARACTER_NAME",
]

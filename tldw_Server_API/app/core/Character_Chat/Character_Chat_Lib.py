# Character_Chat_Lib.py
# # Description: Library containing functions relating to character management,
#
# Imports
import base64
import binascii
import io
import json
import os
import re
import time  # For default titles, etc.
from typing import Dict, List, Optional, Tuple, Any, Union, Set

import yaml
#
# Third-Party Libraries
from PIL import Image  # For image processing
from loguru import logger
# from PIL.Image import Image as PILImage # More specific for type hints if needed
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError, InputError
#
###############################################
#
# Placeholder functions:

def replace_placeholders(text: Optional[str], char_name: Optional[str], user_name: Optional[str]) -> str:
    """Replaces predefined placeholders in a text string.

    The function substitutes placeholders like '{{char}}', '{{user}}',
    '{{random_user}}', '<USER>', and '<CHAR>' with the provided character
    and user names. If names are not provided, default values ("Character", "User")
    are used. Returns an empty string if the input text is None or empty.

    Args:
        text (Optional[str]): The input string, possibly containing placeholders.
        char_name (Optional[str]): The name of the character to substitute for
            '{{char}}' and '<CHAR>'. Defaults to "Character" if None.
        user_name (Optional[str]): The name of the user to substitute for
            '{{user}}', '{{random_user}}', and '<USER>'. Defaults to "User" if None.

    Returns:
        str: The text with placeholders replaced. If the input `text` is None or
        an empty string, an empty string is returned.
    """
    if not text:
        return ""
    char_name_actual = char_name if char_name is not None else "Character"
    user_name_actual = user_name if user_name is not None else "User"
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


def replace_user_placeholder(history: List[Tuple[Optional[str], Optional[str]]], user_name: Optional[str]) -> List[
    Tuple[Optional[str], Optional[str]]]:
    user_name_actual = user_name if user_name else "User"
    updated_history = []
    for user_msg, bot_msg in history:
        updated_user_msg = user_msg.replace("{{user}}", user_name_actual) if user_msg else None
        updated_bot_msg = bot_msg.replace("{{user}}", user_name_actual) if bot_msg else None
        updated_history.append((updated_user_msg, updated_bot_msg))
    return updated_history


#
# End of Placeholder functions
#################################################################################

#################################################################################
#
# Functions for character interaction (DB focused):

# Helper for preparing data for DB (especially image)
def _prepare_character_data_for_db_storage(
    input_data: Dict[str, Any],
    is_update: bool = False
) -> Dict[str, Any]:
    """
    Prepares character data (from a dictionary, often derived from Pydantic model)
    for DB insertion/update.
    Handles 'image_base64' to 'image' (bytes) conversion.
    Ensures JSON-like fields are Python objects.
    """
    db_data = input_data.copy() # Work on a copy

    # Handle image_base64: convert to bytes for 'image' field, or set 'image' to None
    if 'image_base64' in db_data:
        base64_str = db_data.pop('image_base64')
        if base64_str and isinstance(base64_str, str):
            try:
                if ',' in base64_str and base64_str.startswith("data:image"):
                    base64_str = base64_str.split(',', 1)[1]
                db_data['image'] = base64.b64decode(base64_str, validate=True)
            except (binascii.Error, ValueError) as e: # ValueError for invalid padding etc.
                logger.error(f"Invalid image_base64 data for character: {e}")
                # Raise an InputError that API can catch and convert to 400
                raise InputError(f"Invalid image_base64 data: {e}")
        else: # image_base64 was None or empty string
            db_data['image'] = None # Explicitly set image to None to remove it if updating
    elif not is_update and 'image' not in db_data : # For create, if image_base64 and image not provided
        db_data['image'] = None

    # Ensure JSON fields are Python objects (lists/dicts).
    # The DB layer (add_character_card, update_character_card) expects Python objects
    # and will serialize them using _ensure_json_string_from_mixed.
    # If input_data comes from a Pydantic model with proper validators, this might be redundant,
    # but it's a good safeguard if raw dicts are passed.
    for field_name in ["alternate_greetings", "tags"]:
        if field_name in db_data and isinstance(db_data[field_name], str):
            try:
                db_data[field_name] = json.loads(db_data[field_name])
                if not isinstance(db_data[field_name], list): # Should be a list
                    logger.warning(f"Field '{field_name}' was a JSON string but not a list. Resetting to empty list.")
                    db_data[field_name] = []
            except json.JSONDecodeError:
                logger.warning(f"Field '{field_name}' is not valid JSON string. Treating as simple tag list if appropriate or ignoring.")
                # This logic depends on how you want to handle malformed JSON strings for tags/greetings
                # For now, if it's not a list after trying to parse, make it an empty list.
                if not isinstance(db_data[field_name], list):
                     db_data[field_name] = [] # Default to empty list for safety
        elif field_name in db_data and db_data[field_name] is None:
             db_data[field_name] = [] # Store None as empty list

    if "extensions" in db_data and isinstance(db_data["extensions"], str):
        try:
            db_data["extensions"] = json.loads(db_data["extensions"])
            if not isinstance(db_data["extensions"], dict): # Should be a dict
                logger.warning("Field 'extensions' was a JSON string but not a dict. Resetting to empty dict.")
                db_data["extensions"] = {}
        except json.JSONDecodeError:
            logger.warning("Field 'extensions' is not valid JSON string. Resetting to empty dict.")
            db_data["extensions"] = {}
    elif "extensions" in db_data and db_data["extensions"] is None:
        db_data["extensions"] = {} # Store None as empty dict

    return db_data


def get_character_list_for_ui(db: CharactersRAGDB, limit: int = 1000) -> List[Dict[str, Any]]:
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


def extract_character_id_from_ui_choice(choice: str) -> int:
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
        raise ValueError(f"Could not parse character ID from: '{character_id_str}' (derived from '{choice}')")


def load_character_and_image(
        db: CharactersRAGDB,
        character_id: int,
        user_name: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], Optional[str]]], Optional[Image.Image]]:
    """Loads character data, initial message, and image from the database.

    Retrieves a character's details by ID, processes its text fields (like
    description, first message) by replacing placeholders, and loads its
    associated image. The character's first message is formatted as the
    initial entry in a chat history list.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        character_id (int): The ID of the character to load.
        user_name (Optional[str]): The current user's name, used for
            placeholder replacement in character data.

    Returns:
        Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], Optional[str]]], Optional[Image.Image]]:
        A tuple containing:
            - Optional[Dict[str, Any]]: The character's data as a dictionary,
              with placeholders replaced. None if character not found or error.
            - List[Tuple[Optional[str], Optional[str]]]: The initial chat history,
              containing the character's first message as `(None, first_message)`.
              An empty list if no character or first message, or on error.
            - Optional[Image.Image]: The character's image as a PIL Image object.
              None if no image is associated or an error occurs.
    """
    logger.debug(f"Loading character and image for ID: {character_id}, User: {user_name}")
    try:
        char_data = db.get_character_card_by_id(character_id)
        if not char_data:
            logger.warning(f"No character data found for ID: {character_id}")
            return None, [], None

        char_name_from_card = char_data.get('name', 'Character')  # Fallback name

        # Replace placeholders in character data fields
        # These are fields from the DB schema
        fields_to_process = [
            'description', 'personality', 'scenario', 'system_prompt',
            'post_history_instructions', 'first_message', 'message_example',
            'creator_notes'  # 'alternate_greetings' and 'tags' are lists
        ]
        for field in fields_to_process:
            if field in char_data and char_data[field] and isinstance(char_data[field], str):
                char_data[field] = replace_placeholders(char_data[field], char_name_from_card, user_name)

        if 'alternate_greetings' in char_data and isinstance(char_data['alternate_greetings'], list):
            char_data['alternate_greetings'] = [
                replace_placeholders(ag, char_name_from_card, user_name)
                for ag in char_data['alternate_greetings'] if isinstance(ag, str)
            ]

        # The 'first_message' field from DB corresponds to 'first_mes' from old card spec
        first_mes_content = char_data.get('first_message')  # Already processed by placeholders if it was a string
        if not first_mes_content:  # Provide a generic greeting if first_message is empty
            first_mes_content = replace_placeholders(f"Hello, I am {{char}}. How can I help you, {{user}}?",
                                                     char_name_from_card, user_name)

        # Initial chat history is just the character's first message
        chat_history: List[Tuple[Optional[str], Optional[str]]] = [(None, first_mes_content)]

        img = None
        if char_data.get('image') and isinstance(char_data['image'], bytes):  # DB stores image as BLOB
            try:
                image_data_bytes = char_data['image']
                img = Image.open(io.BytesIO(image_data_bytes)).convert("RGBA")
                logger.debug(f"Successfully loaded image for character '{char_name_from_card}'")
            except Exception as e:
                logger.error(f"Error processing image for character '{char_name_from_card}' (ID: {character_id}): {e}")

        return char_data, chat_history, img

    except CharactersRAGDBError as e:
        logger.error(f"Database error in load_character_and_image for ID {character_id}: {e}")
        return None, [], None
    except Exception as e:
        logger.error(f"Unexpected error in load_character_and_image for ID {character_id}: {e}", exc_info=True)
        return None, [], None


def process_db_messages_to_ui_history(
        db_messages: List[Dict[str, Any]],
        char_name_from_card: str,
        user_name_for_placeholders: Optional[str],
        actual_user_sender_id_in_db: str = "User",
        actual_char_sender_id_in_db: Optional[str] = None
) -> List[Tuple[Optional[str], Optional[str]]]:
    """Converts database messages to UI-friendly paired chat history format.

    Takes a list of message dictionaries (as stored in the database) and
    transforms them into a list of (user_message, bot_message) tuples.
    It handles placeholder replacement in message content and correctly pairs
    consecutive messages from the same sender or different senders.

    Args:
        db_messages (List[Dict[str, Any]]): A list of message dictionaries
            retrieved from the database, ordered by timestamp. Each dictionary
            should contain at least 'sender' and 'content'.
        char_name_from_card (str): The name of the character, used for
            placeholder replacement and identifying character messages.
        user_name_for_placeholders (Optional[str]): The user's name, for
            placeholder replacement.
        actual_user_sender_id_in_db (str): The identifier string used for
            the user's messages in the 'sender' field of `db_messages`.
            Defaults to "User".
        actual_char_sender_id_in_db (Optional[str]): The identifier string used
            for the character's messages in the 'sender' field. If None,
            `char_name_from_card` is used. Defaults to None.

    Returns:
        List[Tuple[Optional[str], Optional[str]]]: A list of tuples, where each
        tuple represents a turn or a sequence of messages. The format is
        (user_message_content, bot_message_content). If a turn only has one
        type of message, the other will be None (e.g., (user_msg, None) or
        (None, bot_msg)).
    """
    processed_history: List[Tuple[Optional[str], Optional[str]]] = []
    # If char_sender_id is not provided, use the character's name from the card
    char_sender_identifier = actual_char_sender_id_in_db if actual_char_sender_id_in_db else char_name_from_card
    user_msg_buffer: Optional[str] = None

    for msg_data in db_messages:
        sender = msg_data.get('sender')
        content = msg_data.get('content', '')  # DB content should not be None

        # Replace placeholders in the content from DB
        processed_content = replace_placeholders(content, char_name_from_card, user_name_for_placeholders)

        if sender == actual_user_sender_id_in_db:
            if user_msg_buffer is not None:
                # This implies two user messages in a row. Append the previous one with no bot response.
                processed_history.append((user_msg_buffer, None))
            user_msg_buffer = processed_content
        elif sender == char_sender_identifier:
            if user_msg_buffer is not None:  # User message was waiting, pair it
                processed_history.append((user_msg_buffer, processed_content))
                user_msg_buffer = None
            else:  # Bot message starts the turn or follows another bot message
                processed_history.append((None, processed_content))
        else:
            logger.warning(f"Message from unknown sender '{sender}': {processed_content[:50]}...")
            # Treat as a system/narrator message, append as bot message
            if user_msg_buffer is not None:
                processed_history.append((user_msg_buffer, f"[{sender}] {processed_content}"))
                user_msg_buffer = None
            else:
                processed_history.append((None, f"[{sender}] {processed_content}"))

    # If the last message processed was from the user, it's still in the buffer
    if user_msg_buffer is not None:
        processed_history.append((user_msg_buffer, None))

    return processed_history


def load_chat_and_character(
        db: CharactersRAGDB,
        conversation_id_str: str,
        user_name: Optional[str],
        messages_limit: int = 2000  # Added parameter with default
) -> Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], Optional[str]]], Optional[Image.Image]]:
    """Loads an existing chat conversation and associated character data.

    Retrieves a conversation by its ID, fetches the associated character's
    details (including image and performing placeholder replacements), and
    loads the conversation's message history. The history is processed into
    a UI-friendly paired format.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        conversation_id_str (str): The ID of the conversation to load.
        user_name (Optional[str]): The current user's name, for placeholder
            replacement in character data and message content.
        messages_limit (int): The maximum number of messages to retrieve for
            the conversation. Defaults to 2000.

    Returns:
        Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], Optional[str]]], Optional[Image.Image]]:
        A tuple containing:
            - Optional[Dict[str, Any]]: The character's data, with placeholders
              replaced. None if character or conversation not found or error.
            - List[Tuple[Optional[str], Optional[str]]]: The chat history in
              UI format `(user_msg, bot_msg)`. Empty list on error or if no messages.
            - Optional[Image.Image]: The character's image. None if no image
              or error.
        Returns (None, [], None) if the conversation itself is not found.
    """
    logger.debug(f"Loading chat/conversation ID: {conversation_id_str}, User: {user_name}, Msg Limit: {messages_limit}")
    try:
        conversation_data = db.get_conversation_by_id(conversation_id_str)
        if not conversation_data:
            logger.warning(f"No conversation found with ID: {conversation_id_str}")
            return None, [], None

        character_id = conversation_data.get('character_id')
        if not character_id:
            logger.error(f"Conversation {conversation_id_str} has no character_id associated.")
            # Attempt to load messages anyway, but character data will be missing.
            raw_db_messages = db.get_messages_for_conversation(conversation_id_str, limit=messages_limit,
                                                               # Use parameter
                                                               order_by_timestamp="ASC")
            processed_ui_history = process_db_messages_to_ui_history(raw_db_messages, "Unknown Character", user_name)
            return None, processed_ui_history, None

        # Load character data and image. Initial history from this call is just the first_message, not used here.
        char_data, _, img = load_character_and_image(db, character_id, user_name)

        if not char_data:
            logger.warning(f"No character card found for char_id {character_id} (from conv {conversation_id_str})")
            # Load messages with a placeholder character name
            raw_db_messages = db.get_messages_for_conversation(conversation_id_str, limit=messages_limit,
                                                               # Use parameter
                                                               order_by_timestamp="ASC")
            processed_ui_history = process_db_messages_to_ui_history(raw_db_messages, "Unknown Character", user_name)
            return None, processed_ui_history, img  # img might be None if char_data was None

        char_name_from_card = char_data.get('name', 'Character')  # Should be valid if char_data exists

        # Fetch all messages for this conversation
        raw_db_messages = db.get_messages_for_conversation(conversation_id_str, limit=messages_limit,  # Use parameter
                                                           order_by_timestamp="ASC")

        # Convert DB messages to UI history format.
        # The application layer that calls db.add_message needs to set sender consistently.
        # Convention: User messages in DB have sender "User".
        # Convention: Character messages in DB have sender char_data['name'] (i.e., char_name_from_card).
        processed_ui_history = process_db_messages_to_ui_history(
            raw_db_messages,
            char_name_from_card,
            user_name,
            actual_user_sender_id_in_db="User",
            actual_char_sender_id_in_db=char_name_from_card
        )

        return char_data, processed_ui_history, img

    except CharactersRAGDBError as e:
        logger.error(f"Database error in load_chat_and_character for conversation ID {conversation_id_str}: {e}")
        return None, [], None
    except Exception as e:
        logger.error(f"Unexpected error in load_chat_and_character for conv ID {conversation_id_str}: {e}",
                     exc_info=True)
        return None, [], None


def load_character_wrapper(
        db: CharactersRAGDB,
        character_id_or_ui_choice: Union[int, str],
        user_name: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], Optional[str]]], Optional[Image.Image]]:
    """Wraps character loading to accept either an ID or a UI choice string.

    This function serves as a convenience wrapper around
    `load_character_and_image`. It first parses `character_id_or_ui_choice`
    to an integer ID if it's a string, then calls the main loading function.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        character_id_or_ui_choice (Union[int, str]): Either an integer
            character ID or a UI selection string (e.g., "Name (ID: 123)").
        user_name (Optional[str]): The current user's name, for placeholder
            replacement.

    Returns:
        Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], Optional[str]]], Optional[Image.Image]]:
        The result from `load_character_and_image`:
            (character_data, initial_chat_history, character_image).

    Raises:
        ValueError: If `character_id_or_ui_choice` is an invalid string format
            or not an int/string.
        Exception: Propagates other exceptions from underlying functions.
    """
    try:
        char_id_int: int
        if isinstance(character_id_or_ui_choice, str):
            char_id_int = extract_character_id_from_ui_choice(character_id_or_ui_choice)
        elif isinstance(character_id_or_ui_choice, int):
            char_id_int = character_id_or_ui_choice
        else:
            raise ValueError("character_id_or_ui_choice must be int or string.")

        return load_character_and_image(db, char_id_int, user_name)
    except ValueError as e:  # Catch errors from extract_character_id_from_ui_choice or type check
        logger.error(f"Error in load_character_wrapper with input '{character_id_or_ui_choice}': {e}")
        raise  # Re-raise for the caller to handle
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"Unexpected error in load_character_wrapper for '{character_id_or_ui_choice}': {e}",
                     exc_info=True)
        raise


# --- NEW Character CRUD Interop Functions ---
def create_new_character_from_data(db: CharactersRAGDB, character_payload: Dict[str, Any]) -> Optional[int]:
    """
    Creates a new character in the database from a dictionary payload.
    Handles image data (if 'image_base64' is present).
    Propagates DB errors (InputError, ConflictError, CharactersRAGDBError).
    """
    try:
        if 'name' not in character_payload or not character_payload['name']:
            raise InputError("Character 'name' is required and cannot be empty.")

        # Check for existing character by name before preparing and adding
        # This is good practice for the interop layer, even if API layer also checks.
        existing_char = db.get_character_card_by_name(character_payload['name'])
        if existing_char:
            raise ConflictError(
                f"Character with name '{character_payload['name']}' already exists (ID: {existing_char['id']}).")

        db_ready_data = _prepare_character_data_for_db_storage(character_payload, is_update=False)
        char_id = db.add_character_card(db_ready_data)
        if char_id:
            logger.info(f"Character '{db_ready_data['name']}' created with ID: {char_id}")
        return char_id
    except (InputError, ConflictError, CharactersRAGDBError) as e:
        logger.error(f"Error creating new character: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating character: {e}", exc_info=True)
        raise CharactersRAGDBError(f"Unexpected error creating character: {e}")


def get_character_details(db: CharactersRAGDB, character_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves full character details by ID."""
    try:
        return db.get_character_card_by_id(character_id)
    except CharactersRAGDBError as e:
        logger.error(f"Database error getting character {character_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting character {character_id}: {e}", exc_info=True)
        raise CharactersRAGDBError(f"Unexpected error getting character: {e}")


def update_existing_character_details(
        db: CharactersRAGDB,
        character_id: int,
        update_payload: Dict[str, Any],
        expected_version: int
) -> bool:
    """
    Updates an existing character's details.
    Handles image data and ensures JSON fields are correctly formatted for the DB.
    Propagates DB errors.
    """
    try:
        if not update_payload:  # No actual data to update, DB layer handles this as a "touch"
            logger.info(
                f"No specific fields to update for character ID {character_id}, DB layer will touch if version matches.")
            # Still call update_character_card as it handles version bump and last_modified
            return db.update_character_card(character_id, {}, expected_version)

        # If name is being updated, check for conflict with other characters.
        new_name = update_payload.get('name')
        if new_name is not None:
            current_char = db.get_character_card_by_id(character_id)  # Fetch current to compare name
            if not current_char:
                # This case should ideally be caught by API layer before calling this
                raise InputError(f"Character with ID {character_id} not found for update.")
            if new_name != current_char.get('name'):
                existing_char_with_new_name = db.get_character_card_by_name(new_name)
                if existing_char_with_new_name and existing_char_with_new_name.get('id') != character_id:
                    raise ConflictError(
                        f"Another character with name '{new_name}' already exists (ID: {existing_char_with_new_name['id']}).")

        db_ready_data = _prepare_character_data_for_db_storage(update_payload, is_update=True)

        # The db.update_character_card should only receive fields that are actually changing
        # or all fields if that's how the Pydantic model worked.
        # _prepare_character_data_for_db_storage returns a full dict, but db.update_character_card
        # filters based on updatable fields.
        success = db.update_character_card(character_id, db_ready_data, expected_version)
        if success:
            logger.info(f"Character ID {character_id} updated successfully.")
        return success
    except (InputError, ConflictError, CharactersRAGDBError) as e:
        logger.error(f"Error updating character {character_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating character {character_id}: {e}", exc_info=True)
        raise CharactersRAGDBError(f"Unexpected error updating character: {e}")


def delete_character_from_db(db: CharactersRAGDB, character_id: int, expected_version: int) -> bool:
    """Soft-deletes a character from the database."""
    try:
        success = db.soft_delete_character_card(character_id, expected_version)
        if success:
            logger.info(f"Character ID {character_id} soft-deleted successfully.")
        return success
    except (ConflictError, CharactersRAGDBError) as e:
        logger.error(f"Error soft-deleting character {character_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error soft-deleting character {character_id}: {e}", exc_info=True)
        raise CharactersRAGDBError(f"Unexpected error deleting character: {e}")


def search_characters_by_query_text(
        db: CharactersRAGDB,
        search_term: str,
        limit: int = 10
) -> List[Dict[str, Any]]:
    """Searches character cards using FTS based on the search term."""
    try:
        return db.search_character_cards(search_term, limit=limit)
    except CharactersRAGDBError as e:
        logger.error(f"Error searching characters for '{search_term}': {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error searching characters: {e}", exc_info=True)
        raise CharactersRAGDBError(f"Unexpected error searching characters: {e}")

# --- End of Character CRUD Interop Functions ---

#
# Character Book parsing (copied from original, assumed correct for V2 spec)
#
def parse_character_book(book_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parses character book data from a V2 character card structure.

    This function processes the 'character_book' section of a character card,
    extracting its properties and entries according to the V2 specification.

    Args:
        book_data (Dict[str, Any]): A dictionary representing the
            'character_book' field from a character card's data node.

    Returns:
        Dict[str, Any]: A dictionary containing the parsed character book data,
        including 'name', 'description', 'entries', and other book properties.
        Entries are parsed into a structured list.
    """
    parsed_book = {
        'name': book_data.get('name', ''),
        'description': book_data.get('description', ''),
        'scan_depth': book_data.get('scan_depth'),
        'token_budget': book_data.get('token_budget'),
        'recursive_scanning': book_data.get('recursive_scanning', False),
        'extensions': book_data.get('extensions', {}),
        'entries': []
    }

    for entry_raw in book_data.get('entries', []):
        if not isinstance(entry_raw, dict):
            logger.warning(f"Skipping non-dict entry in character_book: {entry_raw}")
            continue

        # Ensure required fields for an entry are present
        if not entry_raw.get('keys') or not isinstance(entry_raw['keys'], list) or \
                'content' not in entry_raw or \
                'enabled' not in entry_raw or \
                'insertion_order' not in entry_raw:
            logger.warning(
                f"Skipping invalid character_book entry due to missing core fields: {entry_raw.get('name', 'N/A')}")
            continue

        parsed_entry = {
            'keys': entry_raw['keys'],
            'content': entry_raw['content'],
            'extensions': entry_raw.get('extensions', {}),
            'enabled': entry_raw['enabled'],
            'insertion_order': entry_raw['insertion_order'],
            'case_sensitive': entry_raw.get('case_sensitive', False),
            'name': entry_raw.get('name', ''),
            'priority': entry_raw.get('priority'),
            'id': entry_raw.get('id'),  # Can be None
            'comment': entry_raw.get('comment', ''),
            'selective': entry_raw.get('selective', False),
            'secondary_keys': entry_raw.get('secondary_keys', []),
            'constant': entry_raw.get('constant', False),
            'position': entry_raw.get('position', 'before_char')  # Default if not specified
        }
        parsed_book['entries'].append(parsed_entry)
    return parsed_book


#
#################################################################################
# Importing and Parsing External Card/Chat Formats
#

# FIXME
def extract_json_from_image_file(image_file_input: Union[str, bytes, io.BytesIO]) -> Optional[str]:
    """Extracts 'chara' metadata (Base64 encoded JSON) from an image file.

    Typically used for PNG character cards (e.g., TavernAI format) that embed
    character data in a 'chara' metadata chunk.

    Args:
        image_file_input (Union[str, bytes, io.BytesIO]): The image data,
            which can be a file path (str), raw image bytes, or a BytesIO stream.

    Returns:
        Optional[str]: The decoded JSON string from the 'chara' metadata if
        found, valid, and successfully decoded. Returns None if the 'chara'
        metadata is not found, the image cannot be processed, or if there's
        an error during decoding or JSON validation.
    """
    img_obj: Optional[Image.Image] = None
    file_name_for_log = "image_stream"
    image_source_to_use: Optional[io.BytesIO] = None

    try:
        if isinstance(image_file_input, str) and os.path.exists(image_file_input):
            file_name_for_log = image_file_input
            with open(image_file_input, 'rb') as f_bytes:
                image_source_to_use = io.BytesIO(f_bytes.read())
        elif isinstance(image_file_input, bytes):
            image_source_to_use = io.BytesIO(image_file_input)
        elif hasattr(image_file_input, 'read'):  # File-like object
            if hasattr(image_file_input, 'name') and image_file_input.name:
                file_name_for_log = image_file_input.name
            image_file_input.seek(0)
            image_source_to_use = io.BytesIO(image_file_input.read())
            image_file_input.seek(0)  # Reset original stream pointer
        else:
            logger.error("extract_json_from_image_file: Invalid input type. Must be file path, bytes, or BytesIO.")
            return None

        if not image_source_to_use: return None

        logger.debug(f"Attempting to extract JSON from image: {file_name_for_log}")

        img_obj = Image.open(image_source_to_use)

        # Primarily for PNG cards (TavernAI, SillyTavern convention)
        if img_obj.format != 'PNG':
            logger.warning(
                f"Image '{file_name_for_log}' is not in PNG format (format: {img_obj.format}). 'chara' metadata extraction may fail or not be applicable.")


        # 'text' attribute in Pillow Image objects holds metadata chunks.
        # For PNGs, these are tEXt, zTXt, or iTXt chunks.
        if hasattr(img_obj, 'info') and isinstance(img_obj.info, dict) and 'chara' in img_obj.info:
            chara_base64_str = img_obj.info['chara']
            try:
                decoded_chara_json_str = base64.b64decode(chara_base64_str).decode('utf-8')
                json.loads(decoded_chara_json_str)  # Validate it's JSON
                logger.info(f"Successfully extracted and decoded 'chara' JSON from '{file_name_for_log}'.")
                return decoded_chara_json_str
            except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as decode_err:
                logger.error(
                    f"Error decoding 'chara' metadata from '{file_name_for_log}': {decode_err}. Content (start): {str(chara_base64_str)[:100]}...")
                return None  # Explicitly return None on decode error
            except Exception as e:  # Catch any other unexpected error during decode/load
                logger.error(f"Unexpected error during 'chara' processing from '{file_name_for_log}': {e}",
                             exc_info=True)
                return None
        else:
            logger.debug(
                f"'chara' key not found in image metadata for '{file_name_for_log}'. Available metadata keys: {list(img_obj.info.keys()) if isinstance(img_obj.info, dict) else 'N/A'}")
            return None

    except FileNotFoundError:
        logger.error(f"Image file not found for JSON extraction: {file_name_for_log}")
    except IOError as e:  # Catches PIL.UnidentifiedImageError and other file I/O issues
        logger.error(f"Cannot open or read image file (or not a valid image): {file_name_for_log}. Error: {e}",
                     exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error extracting JSON from image '{file_name_for_log}': {e}", exc_info=True)
    finally:
        if img_obj:
            img_obj.close()
        if image_source_to_use:
            image_source_to_use.close()
    return None


def parse_v2_card(card_data_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parses a V2 character card (spec_version '2.0') JSON data.

    This function takes a dictionary representing a V2 character card,
    extracts relevant fields from its 'data' node (or root if 'data' is
    absent but structure is V2-like), and maps them to a new dictionary
    with keys corresponding to the application's database schema.
    It assumes basic structural validity (e.g., presence of key fields)
    may have been checked by a prior validation step.

    Args:
        card_data_json (Dict[str, Any]): The dictionary parsed from a V2
            character card JSON. This should be the entire card object.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the parsed and
        mapped character data (e.g., 'first_mes' becomes 'first_message').
        Returns None if essential V2 fields are missing or if an unexpected
        error occurs during parsing.
    """
    try:
        # data_node can be 'data' or root for some V2 variants (parsing flexibility)
        data_node = card_data_json.get('data', card_data_json)
        if not isinstance(data_node, dict):
            logger.error("V2 card 'data' node is missing or not a dictionary during parsing.")
            return None

        # Required fields in the source V2 card (using original spec names for parsing)
        # This parsing function relies on these fields existing as per V2 spec.
        required_spec_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
        for field in required_spec_fields:
            if field not in data_node or data_node[field] is None:
                logger.error(f"Missing required field '{field}' in V2 card data node during parsing.")
                return None

        # Map to DB schema names
        parsed_data = {
            'name': data_node['name'],
            'description': data_node['description'],
            'personality': data_node['personality'],
            'scenario': data_node['scenario'],
            'first_message': data_node['first_mes'],
            'message_example': data_node['mes_example'],

            'creator_notes': data_node.get('creator_notes', ''),
            'system_prompt': data_node.get('system_prompt', ''),
            'post_history_instructions': data_node.get('post_history_instructions', ''),
            'alternate_greetings': data_node.get('alternate_greetings', []),
            'tags': data_node.get('tags', []),
            'creator': data_node.get('creator', ''),
            'character_version': data_node.get('character_version', ''),
            'extensions': data_node.get('extensions', {}),
            'image_base64': data_node.get('char_image') or data_node.get('image')
        }

        if 'character_book' in data_node and isinstance(data_node['character_book'], dict):
            if not isinstance(parsed_data['extensions'], dict):
                parsed_data['extensions'] = {}
            parsed_data['extensions']['character_book'] = parse_character_book(data_node['character_book'])

        # Log spec/version from top level if present, for info, but parsing proceeds based on data_node content.
        spec = card_data_json.get('spec')
        spec_version = card_data_json.get('spec_version')
        if spec and spec != 'chara_card_v2':
            logger.warning(f"Parsing V2-like card with unexpected 'spec': {spec}.")
        if spec_version and spec_version != '2.0':
            logger.warning(f"Parsing V2-like card with 'spec_version': {spec_version} (expected '2.0').")

        return parsed_data
    except KeyError as e:
        logger.error(f"Missing key during V2 card parsing: {e}")
    except Exception as e:
        logger.error(f"Error parsing V2 card data: {e}", exc_info=True)
    return None


def parse_v1_card(card_data_json: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parses a V1 character card (flat JSON) into a V2-like structure.

    This function converts a V1 character card, which has a flat JSON structure,
    into a dictionary format that aligns with the V2 card structure and the
    application's database schema. Fields are renamed (e.g., 'first_mes' to
    'first_message'), and any non-standard V1 fields are collected into an
    'extensions' dictionary.

    Args:
        card_data_json (Dict[str, Any]): The dictionary parsed from a V1
            character card JSON.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the parsed and
        mapped character data. Returns None if an unexpected error occurs.

    Raises:
        ValueError: If any of the required V1 fields ('name', 'description',
            'personality', 'scenario', 'first_mes', 'mes_example') are missing
            from `card_data_json`.
    """
    try:
        # Required fields in the source V1 card (using original spec names)
        required_spec_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
        for field in required_spec_fields:
            if field not in card_data_json:  # V1 cards are flat, check directly in card_data_json
                raise ValueError(f"Missing required field in V1 card: {field}")

        # Map to DB schema names
        v2_like_data: Dict[str, Any] = {
            'name': card_data_json['name'],
            'description': card_data_json['description'],
            'personality': card_data_json['personality'],
            'scenario': card_data_json['scenario'],
            'first_message': card_data_json['first_mes'],  # Map first_mes -> first_message
            'message_example': card_data_json['mes_example'],  # Map mes_example -> message_example

            'creator_notes': card_data_json.get('creator_notes', ''),
            'system_prompt': card_data_json.get('system_prompt', ''),
            'post_history_instructions': card_data_json.get('post_history_instructions', ''),
            'alternate_greetings': card_data_json.get('alternate_greetings', []),
            'tags': card_data_json.get('tags', []),  # Ensure tags is a list
            'creator': card_data_json.get('creator', ''),
            'character_version': card_data_json.get('character_version', ''),
            'extensions': {},  # Initialize extensions
            'image_base64': card_data_json.get('char_image') or card_data_json.get('image')
        }

        # Collect any non-standard V1 fields into 'extensions'
        standard_v1_keys_mapped_or_known = set(required_spec_fields + [
            'creator_notes', 'system_prompt', 'post_history_instructions',
            'alternate_greetings', 'tags', 'creator', 'character_version', 'char_image', 'image'
        ])

        extra_extensions = {}
        for key, value in card_data_json.items():
            if key not in standard_v1_keys_mapped_or_known:
                extra_extensions[key] = value

        if extra_extensions:
            if isinstance(v2_like_data.get('extensions'), dict):
                v2_like_data['extensions'].update(extra_extensions)
            else:  # Should be a dict due to initialization
                v2_like_data['extensions'] = extra_extensions

        if v2_like_data['extensions'] is None:  # Defensive
            v2_like_data['extensions'] = {}

        return v2_like_data
    except ValueError:  # Re-raise from missing required fields check
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing V1 card: {e}", exc_info=True)
    return None


#
#################################################################################
# Character card parsing & Validation functions
# These validate the *structure* of the card data, typically after parsing from JSON.

def validate_character_book(book_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates the structure and content of a 'character_book' dictionary.

    Checks for required fields, correct data types, and valid values within
    the character book data, including its entries. This is typically part
    of validating a V2 character card.

    Args:
        book_data (Dict[str, Any]): The character book dictionary to validate.
            This usually comes from the 'character_book' field of a character
            card's 'data' node.

    Returns:
        Tuple[bool, List[str]]: A tuple where:
            - The first element (bool) is `True` if the book data is valid,
              `False` otherwise.
            - The second element (List[str]) is a list of error messages
              describing validation failures. Empty if valid.
    """
    validation_messages = []

    # Optional fields with expected types
    optional_fields = {
        'name': str,
        'description': str,
        'scan_depth': (int, float),
        'token_budget': (int, float),
        'recursive_scanning': bool,
        'extensions': dict,
        # 'entries' is technically required if 'character_book' exists
    }

    for field, expected_type in optional_fields.items():
        if field in book_data:
            if not isinstance(book_data[field], expected_type):
                validation_messages.append(
                    f"Field 'character_book.{field}' must be of type '{expected_type.__name__ if isinstance(expected_type, type) else expected_type}'.")

    # 'entries' is required if character_book itself is present
    if 'entries' not in book_data or not isinstance(book_data['entries'], list):
        validation_messages.append(
            "Field 'character_book.entries' is required and must be a list if 'character_book' is defined.")
        return False, validation_messages  # Cannot proceed without entries

    # Validate each entry in 'entries'
    entries = book_data.get('entries', [])
    entry_ids: Set[Union[int, float]] = set()  # Store IDs to check for uniqueness
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            validation_messages.append(f"Entry {idx} in 'character_book.entries' is not a dictionary.")
            continue
        is_valid_entry, entry_messages = validate_character_book_entry(entry, idx, entry_ids)
        if not is_valid_entry:
            validation_messages.extend(entry_messages)

    is_valid = len(validation_messages) == 0
    return is_valid, validation_messages


def validate_character_book_entry(entry: Dict[str, Any], idx: int, entry_ids: Set[Union[int, float]]) -> Tuple[
    bool, List[str]]:
    """Validates a single entry within a 'character_book.entries' list.

    Checks an individual character book entry for required fields (like 'keys',
    'content'), correct data types, valid 'position' values, constraints
    related to 'selective' entries, and uniqueness of 'id' if present.

    Args:
        entry (Dict[str, Any]): The character book entry dictionary to validate.
        idx (int): The index of the entry within its parent list, used for
            error reporting.
        entry_ids (Set[Union[int, float]]): A set of 'id' values encountered so
            far within the same character book, used to check for uniqueness of
            the current entry's 'id'. This set will be updated by this function
            if the current entry has a unique, valid ID.

    Returns:
        Tuple[bool, List[str]]: A tuple where:
            - The first element (bool) is `True` if the entry is valid,
              `False` otherwise.
            - The second element (List[str]) is a list of error messages
              describing validation failures. Empty if valid.
    """
    validation_messages = []
    required_fields_entry = {
        'keys': list,
        'content': str,
        # 'extensions': dict, # Extensions can be missing
        'enabled': bool,
        'insertion_order': (int, float)
    }

    for field, expected_type in required_fields_entry.items():
        if field not in entry:
            validation_messages.append(f"Entry {idx}: Missing required field '{field}'.")
        elif not isinstance(entry[field], expected_type):
            validation_messages.append(
                f"Entry {idx}: Field '{field}' must be of type '{expected_type.__name__ if isinstance(expected_type, type) else expected_type}'.")
        elif field == 'content' and not entry[field].strip() and entry[
            field] is not None:  # Allow None content if type check allows, but not empty string
            validation_messages.append(
                f"Entry {idx}: Field 'content' cannot be an empty or whitespace-only string if present.")
        elif field == 'keys' and not entry[field]:  # Must have at least one key
            validation_messages.append(f"Entry {idx}: Field 'keys' cannot be empty.")

    # Optional fields
    optional_fields_entry = {
        'extensions': dict,
        'case_sensitive': bool,
        'name': str,
        'priority': (int, float),
        'id': (int, float),  # ID can be int or float (number)
        'comment': str,
        'selective': bool,
        'secondary_keys': list,
        'constant': bool,
        'position': str  # Should be 'before_char' or 'after_char' or 'after_prompt' etc.
    }

    for field, expected_type in optional_fields_entry.items():
        if field in entry and entry[field] is not None and not isinstance(entry[field],
                                                                          expected_type):  # Check type only if field is present and not None
            validation_messages.append(
                f"Entry {idx}: Field '{field}' must be of type '{expected_type.__name__ if isinstance(expected_type, type) else expected_type}'.")

    # Validate 'position' value if present
    if 'position' in entry and entry['position'] is not None:
        # This list might need to be expanded based on spec (e.g. SillyTavern lorebook positions)
        valid_positions = ['before_char', 'after_char', 'after_prompt', 'before_history']
        if entry['position'] not in valid_positions:
            validation_messages.append(
                f"Entry {idx}: Field 'position' ('{entry['position']}') is not a recognized value (e.g., {', '.join(valid_positions)}).")

    # Validate 'secondary_keys' if 'selective' is True
    if entry.get('selective') is True:  # Check for explicit True
        if 'secondary_keys' not in entry or not isinstance(entry.get('secondary_keys'), list):
            validation_messages.append(f"Entry {idx}: 'secondary_keys' must be a list when 'selective' is True.")
        elif not entry.get('secondary_keys'):  # If list exists, it must not be empty for selective=true
            validation_messages.append(f"Entry {idx}: 'secondary_keys' cannot be empty when 'selective' is True.")

    # Validate 'keys' list elements (must be non-empty strings)
    if 'keys' in entry and isinstance(entry['keys'], list):
        for i, key_val in enumerate(entry['keys']):
            if not isinstance(key_val, str) or not key_val.strip():
                validation_messages.append(f"Entry {idx}: Element {i} in 'keys' must be a non-empty string.")

    # Validate 'secondary_keys' list elements (must be non-empty strings)
    if 'secondary_keys' in entry and isinstance(entry.get('secondary_keys'), list):
        for i, skey_val in enumerate(entry['secondary_keys']):
            if not isinstance(skey_val, str) or not skey_val.strip():
                validation_messages.append(f"Entry {idx}: Element {i} in 'secondary_keys' must be a non-empty string.")

    # Validate 'id' uniqueness
    if 'id' in entry and entry['id'] is not None:
        entry_id_val = entry['id']
        if entry_id_val in entry_ids:
            validation_messages.append(
                f"Entry {idx}: Duplicate 'id' value '{entry_id_val}'. Each entry 'id' in a book must be unique.")
        else:
            entry_ids.add(entry_id_val)

    # Validate 'extensions' keys are namespaced (convention)
    if 'extensions' in entry and isinstance(entry.get('extensions'), dict):
        for ext_key in entry['extensions'].keys():
            if not isinstance(ext_key, str) or ('/' not in ext_key and '_' not in ext_key and ':' not in ext_key):
                validation_messages.append(
                    f"Entry {idx}: Extension key '{ext_key}' in 'extensions' should be namespaced (e.g., 'myorg/mykey') to prevent conflicts.")

    is_valid = len(validation_messages) == 0
    return is_valid, validation_messages


def validate_v2_card(card_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates a character card dictionary against the V2 specification.

    Checks top-level fields like 'spec' and 'spec_version', the presence and
    type of the 'data' node, and required/optional fields within 'data'.
    It also invokes `validate_character_book` if a 'character_book' is present.

    Args:
        card_data (Dict[str, Any]): The full character card dictionary (parsed
            from JSON) to validate.

    Returns:
        Tuple[bool, List[str]]: A tuple where:
            - The first element (bool) is `True` if the card is valid according
              to V2 spec, `False` otherwise.
            - The second element (List[str]) is a list of error messages
              describing validation failures. Empty if valid.
    """
    validation_messages = []

    # Check top-level fields for full V2 spec compliance
    if 'spec' not in card_data:
        validation_messages.append("Missing 'spec' field (expected 'chara_card_v2' for V2 spec).")
    elif card_data['spec'] != 'chara_card_v2':
        validation_messages.append(f"Invalid 'spec' value: '{card_data['spec']}'. Expected 'chara_card_v2'.")

    if 'spec_version' not in card_data:
        validation_messages.append("Missing 'spec_version' field (expected '2.0' for V2 spec).")
    else:
        try:
            # Spec version should be a string like "2.0"
            if isinstance(card_data['spec_version'], str):
                spec_version_float = float(
                    card_data['spec_version'])  # TODO: More robust version comparison if needed (e.g., major.minor)
                if spec_version_float < 2.0:
                    validation_messages.append(
                        f"'spec_version' must be '2.0' or higher. Found '{card_data['spec_version']}'.")
            else:
                validation_messages.append(
                    f"Invalid 'spec_version' format: {card_data['spec_version']}. Must be a string (e.g., '2.0').")
        except ValueError:
            validation_messages.append(
                f"Invalid 'spec_version' format: {card_data['spec_version']}. Must be a number as a string (e.g., '2.0').")

    if 'data' not in card_data or not isinstance(card_data.get('data'), dict):  # Use .get for safety before isinstance
        validation_messages.append(
            "Missing 'data' field, or it's not a dictionary. V2 spec requires character data under a 'data' key.")
        # If 'data' is missing, further checks on data_node will likely fail or be irrelevant.
        # However, some V2 cards might be flat if spec is missing, so we don't hard return here
        # unless spec explicitly stated V2. The calling function will decide based on results.
        data_node = {}  # Avoid None for data_node if it's missing for subsequent checks to not error out
    else:
        data_node = card_data['data']

    # Required fields in 'data' node
    required_data_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
    for field in required_data_fields:
        if field not in data_node:
            validation_messages.append(f"Missing required field in 'data': '{field}'.")
        elif not isinstance(data_node[field], str):
            validation_messages.append(f"Field 'data.{field}' must be a string.")
        elif field in ['name', 'first_mes'] and not data_node[field].strip():
            validation_messages.append(f"Field 'data.{field}' cannot be empty or just whitespace.")

    # Optional fields with expected types in 'data' node
    optional_data_fields = {
        'creator_notes': str,
        'system_prompt': str,
        'post_history_instructions': str,
        'alternate_greetings': list,
        'tags': list,
        'creator': str,
        'character_version': str,
        'extensions': dict,
        'character_book': dict,
        'char_image': str,
        'image': str,
    }

    for field, expected_type in optional_data_fields.items():
        if field in data_node and data_node[field] is not None:
            if not isinstance(data_node[field], expected_type):
                validation_messages.append(f"Field 'data.{field}' must be of type '{expected_type.__name__}'.")
            elif field == 'extensions' and isinstance(data_node[field], dict):  # Check only if it's a dict
                for ext_key in data_node[field].keys():
                    if not isinstance(ext_key, str) or (
                            '/' not in ext_key and '_' not in ext_key and ':' not in ext_key):
                        validation_messages.append(
                            f"Extension key '{ext_key}' in 'data.extensions' should be namespaced (e.g., 'myorg/mykey').")

    if 'alternate_greetings' in data_node and isinstance(data_node.get('alternate_greetings'), list):
        for idx, greeting in enumerate(data_node['alternate_greetings']):
            if not isinstance(greeting, str) or not greeting.strip():
                validation_messages.append(f"Element {idx} in 'data.alternate_greetings' must be a non-empty string.")

    if 'tags' in data_node and isinstance(data_node.get('tags'), list):
        for idx, tag_val in enumerate(data_node['tags']):
            if not isinstance(tag_val, str) or not tag_val.strip():
                validation_messages.append(f"Element {idx} in 'data.tags' must be a non-empty string.")

    if 'character_book' in data_node and data_node['character_book'] is not None:
        if isinstance(data_node['character_book'], dict):
            is_valid_book, book_messages = validate_character_book(data_node['character_book'])
            if not is_valid_book:
                validation_messages.extend(book_messages)
        else:
            validation_messages.append("'data.character_book' must be a dictionary if present.")

    is_valid = len(validation_messages) == 0
    return is_valid, validation_messages


def import_character_card_from_json_string(json_content_str: str) -> Optional[Dict[str, Any]]:
    """Imports and parses a character card from a JSON string.

    This function attempts to parse a character card from the provided JSON
    string. It automatically detects whether the card is V1 or V2 format.
    For V2 cards (identified by 'spec' or 'spec_version' fields, or
    heuristically by a 'data' node), it performs V2 structural validation.
    If V2 processing fails or is not applicable, it attempts V1 parsing.
    The parsed data is then mapped to a dictionary with keys corresponding
    to the application's database schema.

    Args:
        json_content_str (str): A string containing the character card data
            in JSON format.

    Returns:
        Optional[Dict[str, Any]]: A dictionary with character data mapped to
        DB schema field names (e.g., 'first_message', 'message_example') if
        parsing and validation are successful. Returns None if the JSON is
        invalid, the card structure is unrecognized, critical fields are
        missing (like 'name' after parsing), or any other parsing error occurs.
    """
    if not json_content_str or not json_content_str.strip():
        logger.error("JSON content string is empty or whitespace.")
        return None
    try:
        card_data_dict = json.loads(json_content_str.strip())

        parsed_card: Optional[Dict[str, Any]] = None

        # Determine if V2 validation should be attempted
        is_explicit_v2_spec = card_data_dict.get('spec') == 'chara_card_v2'
        # Consider "2.0", "2.1", etc. as valid V2 versions for initial check
        is_explicit_v2_version_str = str(card_data_dict.get('spec_version', ''))
        is_explicit_v2_version = is_explicit_v2_version_str.startswith("2.")

        has_data_node_heuristic = isinstance(card_data_dict.get('data'), dict) and \
                                  'name' in card_data_dict['data']  # Heuristic for implicit V2

        attempt_v2_processing = is_explicit_v2_spec or is_explicit_v2_version or \
                                (has_data_node_heuristic and not is_explicit_v2_spec and not is_explicit_v2_version)

        if attempt_v2_processing:
            logger.debug("Attempting V2 validation based on card structure/spec.")
            is_valid_v2_struct, v2_errors = validate_v2_card(card_data_dict)

            if not is_valid_v2_struct:
                logger.error(f"V2 Card structural validation failed: {'; '.join(v2_errors)}.")
                if is_explicit_v2_spec or is_explicit_v2_version:
                    logger.error("Card explicitly declared as V2 but failed V2 structural validation. Import aborted.")
                    return None
                else:  # Implicit V2 guess failed validation
                    logger.warning(
                        "Heuristically identified V2 card failed V2 structural validation. Will attempt V1 parsing as fallback.")
                    # No 'return None' here, proceed to V1 attempt below
            else:  # V2 structural validation passed
                logger.info("V2 Card structural validation passed. Attempting to parse as V2 character card.")
                parsed_card = parse_v2_card(card_data_dict)
                if not parsed_card:
                    logger.warning(
                        "V2 parsing failed despite passing V2 structural validation. This might indicate an issue with the parser or an edge case. Attempting V1 parsing as fallback.")
                    # `parsed_card` is None, will fall through to V1 attempt

        # Fallback to V1 if V2 processing was not attempted, or if it was attempted but `parsed_card` is still None
        if parsed_card is None:
            logger.info("Attempting to parse as V1 character card.")
            try:
                # parse_v1_card raises ValueError if required fields are missing, or returns None on other errors
                parsed_card = parse_v1_card(card_data_dict)
            except ValueError as ve_v1:
                logger.error(f"V1 card parsing error (likely missing required V1 fields): {ve_v1}")
                parsed_card = None  # Ensure parsed_card is None on this error

        # Final check and return
        if parsed_card and parsed_card.get('name'):  # Name is fundamental
            logger.info(f"Successfully parsed card: '{parsed_card.get('name')}'")
            return parsed_card
        else:
            if parsed_card and not parsed_card.get('name'):
                logger.error("Parsed card is missing 'name'. Import failed.")
            else:  # parsed_card is None
                logger.error("All parsing attempts (V2 and V1) failed to produce a valid card.")
            return None

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error from string: {e}. Content (start): {json_content_str[:150]}...")
    except Exception as e:  # Catch any other unexpected errors during the process
        logger.error(f"Unexpected error parsing card from JSON string: {e}", exc_info=True)
    return None


def load_character_card_from_string_content(content_str: str) -> Optional[Dict[str, Any]]:
    """Loads a character card from various string formats (JSON, Markdown).

    This function parses character card data from a string. It supports:
    1.  Direct JSON content.
    2.  Markdown with YAML frontmatter (requires PyYAML).
    3.  Markdown with a JSON code block (e.g., ```json ... ```).

    The extracted JSON data is then processed by
    `import_character_card_from_json_string` for V1/V2 detection,
    validation, and mapping to DB schema names. This function *only* parses
    the string; it does not save anything to a database.

    Args:
        content_str (str): The string content potentially containing character
            card data in JSON, YAML frontmatter, or a JSON code block.

    Returns:
        Optional[Dict[str, Any]]: A dictionary with parsed character data mapped
        to DB schema field names if successful. Returns None if no valid card
        data can be extracted, if parsing fails, or if required dependencies
        like PyYAML (for YAML frontmatter) are missing and needed.

    Raises:
        ImportError: If PyYAML is required for parsing YAML frontmatter but is
            not installed. This exception is propagated.
    """
    if not content_str or not content_str.strip():
        logger.error("Cannot load character card from empty or whitespace string content.")
        return None

    try:
        content = content_str.replace("\ufeff", "").lstrip()  # Remove BOM, leading whitespace
        logger.debug(f"Attempting to load card from string content (start): {repr(content[:70])}")

        json_card_data_str: Optional[str] = None

        if content.startswith('{'):  # Likely direct JSON
            json_card_data_str = content
        elif content.startswith('---'):  # Likely Markdown with YAML frontmatter
            try:
                # Regex to match YAML front matter strictly at the start, allowing for optional whitespace before ---
                yaml_match = re.match(r"^\s*---\s*\n(.*?)\n\s*---\s*", content, re.DOTALL)
                if yaml_match:
                    yaml_content = yaml_match.group(1).strip()
                    # Convert YAML to JSON string for consistent parsing by import_character_card_from_json_string
                    card_dict_from_yaml = yaml.safe_load(yaml_content)
                    if isinstance(card_dict_from_yaml, dict):
                        json_card_data_str = json.dumps(card_dict_from_yaml)
                    else:
                        logger.error("YAML frontmatter did not parse into a dictionary.")
                else:  # If frontmatter malformed, check for JSON block in the rest of the content
                    logger.debug("Markdown frontmatter not found or malformed, checking for JSON code block.")
            except ImportError: # PyYAML not installed
                logger.error("PyYAML is required for loading YAML front matter. Install it via 'pip install PyYAML'.")
                raise # Re-raise to notify caller of missing dependency
            except yaml.YAMLError as ye:
                logger.error(f"Error parsing YAML frontmatter: {ye}")
                # Fall through

        if not json_card_data_str:  # If not direct JSON or YAML processed, look for JSON code block
            # Regex to find a JSON code block (```json ... ``` or ``` ... ```)
            # DOTALL allows . to match newlines, IGNORECASE for 'json' tag
            pattern = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
            match = pattern.search(content)
            if match:
                json_card_data_str = match.group(1).strip()
                logger.debug("Extracted JSON from code block.")
            else:
                if not content.startswith('{'):  # Only error if it wasn't direct JSON to begin with
                    logger.error(
                        "No valid character card data found: not direct JSON, no YAML frontmatter, and no JSON code block.")
                    return None

        if not json_card_data_str:
            logger.error("Could not extract JSON string from the provided content.")
            return None

        return import_character_card_from_json_string(json_card_data_str)

    except ImportError:  # Specifically for PyYAML
        raise  # Let it propagate so user knows dependency is missing
    except Exception as e:
        logger.error(
            f"Unexpected error in load_character_card_from_string_content: {e}. Content (start): {content_str[:100]}",
            exc_info=True)
    return None


def import_and_save_character_from_file(
        db: CharactersRAGDB,
        file_input: Union[str, io.BytesIO, bytes]
) -> Optional[int]:
    """Imports a character card from a file, saves it to DB, and returns ID.

    This function handles multiple input types for character cards:
    - Text files (e.g., .json, .md) containing character data.
    - Image files (e.g., .png, .webp) with embedded 'chara' JSON metadata.
    - Raw bytes or BytesIO streams representing either text or image content.

    It extracts the character data, parses and validates it (V1/V2 auto-detection),
    handles an embedded or base64 image, and then saves the character to the
    database using `db.add_character_card`.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        file_input (Union[str, io.BytesIO, bytes]): The source of the character
            card. Can be a file path (str), a BytesIO stream, or raw bytes.

    Returns:
        Optional[int]: The database ID of the newly imported character if
        successful. Returns the ID of an existing character if a conflict
        (e.g., duplicate name) occurs and the character already exists.
        Returns None if the import or save process fails for any other reason
        (e.g., file not found, invalid format, DB error).

    Raises:
        ImportError: If PyYAML is required for parsing (e.g., Markdown with
            YAML frontmatter in a text file input) but is not installed.
    """
    parsed_card_dict: Optional[Dict[str, Any]] = None
    filename_for_log = "input_stream"
    image_bytes_for_db: Optional[bytes] = None  # Specific for this function's image handling

    try:
        card_json_str: Optional[str] = None  # String content of the card data
        # Logic to extract card_json_str and image_bytes_for_db (if input is image)
        # This part is complex and depends on whether file_input is path, bytes, or stream
        # and whether it's text or image. The original logic is kept for this part.
        if isinstance(file_input, str):  # File path
            filename_for_log = file_input
            if not os.path.exists(filename_for_log): raise FileNotFoundError(f"File not found: {filename_for_log}")
            _, ext = os.path.splitext(filename_for_log.lower())
            if ext in ['.png', '.webp']:
                with open(filename_for_log, 'rb') as f_img:
                    image_bytes_for_db = f_img.read()
                card_json_str = extract_json_from_image_file(io.BytesIO(image_bytes_for_db))
                if not card_json_str:
                    logger.error(f"Image file {filename_for_log} but no char JSON metadata.")
                    return None
            else:  # Assume text file
                with open(filename_for_log, 'r', encoding='utf-8') as f_text:
                    card_json_str = f_text.read()
        elif isinstance(file_input, bytes):
            try:  # Try as image first
                temp_image_stream = io.BytesIO(file_input)
                card_json_str = extract_json_from_image_file(temp_image_stream)
                if card_json_str:
                    image_bytes_for_db = file_input  # It was an image with chara data
                else:
                    card_json_str = file_input.decode('utf-8')  # Try as text
            except (UnicodeDecodeError, Exception):
                card_json_str = None  # Could not decode as text
            if card_json_str is None:  # Failed both image and text attempt from bytes
                logger.error("Input bytes are not valid image with chara or UTF-8 text.")
                return None
        elif hasattr(file_input, 'read'):  # File-like object
            if hasattr(file_input, 'name') and file_input.name: filename_for_log = file_input.name
            file_input.seek(0);
            stream_bytes = file_input.read();
            file_input.seek(0)
            try:  # Try as image
                temp_image_stream = io.BytesIO(stream_bytes)
                card_json_str = extract_json_from_image_file(temp_image_stream)
                if card_json_str:
                    image_bytes_for_db = stream_bytes
                else:
                    card_json_str = stream_bytes.decode('utf-8')
            except (UnicodeDecodeError, Exception):
                card_json_str = None
            if card_json_str is None:
                logger.error(f"Stream {filename_for_log} not valid image with chara or UTF-8 text.")
                return None
        else:
            raise ValueError("Invalid file_input type.")

        if not card_json_str:
            logger.error(f"Could not obtain character card JSON string from: {filename_for_log}")
            return None

        parsed_card_dict = load_character_card_from_string_content(card_json_str)
        if not parsed_card_dict:
            return None
        if not parsed_card_dict.get('name'):
            raise InputError("Parsed card data is missing 'name'.")

        # Prepare payload for DB, including image handling
        # If image_bytes_for_db was already set (from image file input), use that.
        # Otherwise, _prepare_character_data_for_db_storage will check 'image_base64' from parsed_card_dict.
        payload_for_db_helper = parsed_card_dict.copy()
        if image_bytes_for_db:  # If image came directly from file, don't use base64 from JSON
            payload_for_db_helper.pop('image_base64', None)  # Remove if present
            payload_for_db_helper['image'] = image_bytes_for_db  # Use direct bytes

        # Use the helper for final db_payload creation
        db_payload = _prepare_character_data_for_db_storage(payload_for_db_helper, is_update=False)

        # Call db.add_character_card directly
        char_id = db.add_character_card(db_payload)
        if char_id: logger.info(f"Imported character '{db_payload['name']}' with DB ID: {char_id}")
        return char_id
    except ConflictError as ce:
        logger.warning(f"Conflict importing character: {ce}.")
        if parsed_card_dict and parsed_card_dict.get('name'):
            existing_char = db.get_character_card_by_name(parsed_card_dict['name'])
            if existing_char and existing_char.get('id'): return existing_char['id']
        raise  # Re-raise ConflictError so API layer can give 409
    except (InputError, CharactersRAGDBError) as e:
        logger.error(f"Error importing character from {filename_for_log}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"File not found during import: {filename_for_log}")
        raise InputError(f"File not found: {filename_for_log}")
    except ImportError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error importing char from {filename_for_log}: {e}", exc_info=True)
        raise CharactersRAGDBError(f"Unexpected error importing character: {e}")


def load_chat_history_from_file_and_save_to_db(
        db: CharactersRAGDB,
        file_path_or_obj: Union[str, io.BytesIO],
        user_name_for_placeholders: Optional[str] = "User",
        default_user_sender_in_db: str = "User"
) -> Tuple[Optional[str], Optional[int]]:
    """Loads chat history from a JSON file and saves it to the database.

    This function imports chat history from a JSON file, typically in
    TavernAI or SillyTavern format. It identifies the character mentioned
    in the log, verifies their existence in the database, creates a new
    conversation record, and then adds all messages from the log to this
    new conversation. Placeholders in message content can be replaced using
    `user_name_for_placeholders` and the character's name.

    Expected JSON formats include:
    ```json
    {
      "char_name": "Character Name", // or "character", or "name"
      "user_name": "User Name",      // optional, used for placeholders
      "history": {                   // or "chat"
        "internal": [
          ["User message 1", "Char message 1"],
          ["User message 2", "Char message 2"]
        ],
        "visible": [ /* similar structure, 'internal' usually preferred */ ]
      }
    }
    ```
    Or simpler formats like:
    ```json
    {
      "character": "CharName",
      "history": [["user_msg1", "bot_msg1"], ...]
    }
    ```

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        file_path_or_obj (Union[str, io.BytesIO]): The path to the JSON chat
            log file, or a file-like object (e.g., BytesIO) containing the
            JSON content.
        user_name_for_placeholders (Optional[str]): The name to use for user
            placeholders (e.g., `{{user}}`) found in the log messages.
            Defaults to "User".
        default_user_sender_in_db (str): The identifier to use for the 'sender'
            field when saving user messages to the database. Defaults to "User".

    Returns:
        Tuple[Optional[str], Optional[int]]: A tuple containing:
            - Optional[str]: The ID of the newly created conversation in the
              database if successful.
            - Optional[int]: The ID of the character associated with the chat
              if successful.
        Returns (None, None) if any part of the process fails (e.g., file
        error, JSON parsing error, character not found in DB, DB error).
    """
    filename_for_log = "chat_log_stream"
    try:
        content_str: str
        if isinstance(file_path_or_obj, str):
            filename_for_log = file_path_or_obj
            with open(file_path_or_obj, 'r', encoding='utf-8') as f:
                content_str = f.read()
        elif hasattr(file_path_or_obj, 'read'):  # File-like object
            if hasattr(file_path_or_obj, 'name') and file_path_or_obj.name:
                filename_for_log = file_path_or_obj.name
            file_path_or_obj.seek(0)
            raw_bytes = file_path_or_obj.read()
            content_str = raw_bytes.decode('utf-8') if isinstance(raw_bytes, bytes) else str(raw_bytes)
        else:
            raise ValueError("Invalid input for chat history: must be file path or file-like object.")

        chat_data_dict = json.loads(content_str)

        # Extract character name (flexible key search)
        char_name_from_log = chat_data_dict.get('char_name') or \
                             chat_data_dict.get('character') or \
                             chat_data_dict.get('name')  # Some formats might use 'name' for char

        if not char_name_from_log:
            logger.error(f"Chat log '{filename_for_log}' is missing character name ('char_name' or 'character').")
            return None, None

        # Extract history pairs (flexible key search for history structure)
        history_pairs_raw: Optional[List[List[str]]] = None
        if 'history' in chat_data_dict:
            if isinstance(chat_data_dict['history'], list):  # Simple list of pairs
                history_pairs_raw = chat_data_dict['history']
            elif isinstance(chat_data_dict['history'], dict):  # Tavern/SillyTavern structure
                history_pairs_raw = chat_data_dict['history'].get('internal') or chat_data_dict['history'].get(
                    'visible')
        elif 'chat' in chat_data_dict:  # Alternative key for history
            if isinstance(chat_data_dict['chat'], list):
                history_pairs_raw = chat_data_dict['chat']
            elif isinstance(chat_data_dict['chat'], dict):
                history_pairs_raw = chat_data_dict['chat'].get('internal') or chat_data_dict['chat'].get('visible')

        if not history_pairs_raw or not isinstance(history_pairs_raw, list):
            logger.error(f"Chat log '{filename_for_log}' is missing valid 'history' (list of message pairs).")
            return None, None

        # Validate and clean history pairs
        history_pairs: List[Tuple[Optional[str], Optional[str]]] = []
        for pair_idx, raw_pair in enumerate(history_pairs_raw):
            if isinstance(raw_pair, list) and len(raw_pair) >= 1 and len(raw_pair) <= 2:
                user_m = str(raw_pair[0]) if raw_pair[0] is not None else None
                bot_m = str(raw_pair[1]) if len(raw_pair) > 1 and raw_pair[1] is not None else None
                # Skip pairs where both are None or effectively empty after stripping
                if (user_m and user_m.strip()) or (bot_m and bot_m.strip()):
                    history_pairs.append((user_m, bot_m))
            else:
                logger.warning(
                    f"Skipping malformed message pair at index {pair_idx} in '{filename_for_log}': {raw_pair}")

        if not history_pairs:
            logger.error(f"No valid message pairs found in chat log '{filename_for_log}'.")
            return None, None

        # Find character in DB
        character_db_entry = db.get_character_card_by_name(char_name_from_log)
        if not character_db_entry or not character_db_entry.get('id'):
            logger.error(
                f"Character '{char_name_from_log}' from chat log '{filename_for_log}' not found in the database.")
            return None, None

        character_id_from_db: int = character_db_entry['id']
        actual_char_name_from_db = character_db_entry.get('name', char_name_from_log)  # Prefer DB name

        # Create a new conversation for this imported chat
        conv_title = f"Imported Chat: {actual_char_name_from_db} ({time.strftime('%Y-%m-%d %H:%M')})"
        new_conv_id = db.add_conversation({
            'character_id': character_id_from_db,
            'title': conv_title
        })

        if not new_conv_id:
            logger.error(f"Failed to create a new conversation in DB for chat with '{actual_char_name_from_db}'.")
            return None, None

        logger.info(
            f"Created new conversation (ID: {new_conv_id}) for imported chat with '{actual_char_name_from_db}'.")

        with db.transaction():
            for user_msg_str, char_msg_str in history_pairs:
                log_user_name = chat_data_dict.get('user_name') or user_name_for_placeholders

                if user_msg_str and user_msg_str.strip():
                    processed_user_msg = replace_placeholders(user_msg_str, actual_char_name_from_db, log_user_name)
                    db.add_message({
                        'conversation_id': new_conv_id,
                        'sender': default_user_sender_in_db,
                        'content': processed_user_msg
                    })

                if char_msg_str and char_msg_str.strip():
                    processed_char_msg = replace_placeholders(char_msg_str, actual_char_name_from_db, log_user_name)
                    db.add_message({
                        'conversation_id': new_conv_id,
                        'sender': actual_char_name_from_db,
                        'content': processed_char_msg
                    })
            logger.info(f"Successfully imported {len(history_pairs)} message pairs into conversation ID {new_conv_id}.")

        return new_conv_id, character_id_from_db

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from chat log '{filename_for_log}': {e}")
    except ValueError as ve:
        logger.error(f"Invalid data or format in chat log '{filename_for_log}': {ve}")
    except CharactersRAGDBError as dbe:
        logger.error(f"Database error during chat history import from '{filename_for_log}': {dbe}")
    except Exception as e:
        logger.error(f"Unexpected error importing chat history from '{filename_for_log}': {e}", exc_info=True)

    return None, None


#################################################################################
# Conversation and Message Management Functions
#################################################################################
# NOTE: This section focuses on Conversation and Message entities.

# --- Conversation Management ---

def start_new_chat_session(
    db: CharactersRAGDB,
    character_id: int,
    user_name: Optional[str], # For placeholder replacement in initial/retrieved messages
    custom_title: Optional[str] = None
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[List[Tuple[Optional[str], Optional[str]]]], Optional[Image.Image]]:
    """Starts a new chat session with a specified character.

    This function performs the following steps:
    1.  Loads the character's data and image. Raw character data is fetched first
        to obtain the original 'first_message' content for DB storage. Then,
        `load_character_and_image` is called to get placeholder-processed
        character data for UI and the initial UI history.
    2.  Creates a new conversation record in the database associated with the
        character. A title is generated if `custom_title` is not provided.
    3.  Adds the character's original (raw, if available, otherwise processed)
        first message to this new conversation in the database.
    4.  Returns the new conversation ID, the processed character data for UI,
        the initial UI chat history (containing the processed first message),
        and the character's image.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        character_id (int): The ID of the character to start a chat with.
        user_name (Optional[str]): The current user's name, used for placeholder
            replacement in the character's data and first message (for UI).
        custom_title (Optional[str]): An optional custom title for the new
            conversation. If None, a default title is generated.

    Returns:
        Tuple[Optional[str], Optional[Dict[str, Any]], Optional[List[Tuple[Optional[str], Optional[str]]]], Optional[Image.Image]]:
        A tuple containing:
            - Optional[str]: The ID of the newly created conversation. None on failure.
            - Optional[Dict[str, Any]]: Processed character data for UI. None if
              character loading fails.
            - Optional[List[Tuple[Optional[str], Optional[str]]]]: Initial UI chat
              history `[(None, processed_first_message)]`. Can be empty or None
              if character/first message issues.
            - Optional[Image.Image]: The character's image. None if no image or
              loading fails.
        If character loading fails initially, (None, None, None, None) is returned.
        If conversation creation or message adding fails, some elements might
        still be populated from the successful character load.
    """
    logger.debug(f"Starting new chat session for character_id: {character_id}, user: {user_name}")

    original_first_message_content: Optional[str] = None
    try:
        # 1. Get raw character data first for the original first_message content
        raw_char_data_for_first_message = db.get_character_card_by_id(character_id)
        if raw_char_data_for_first_message:
            original_first_message_content = raw_char_data_for_first_message.get('first_message')
        else:
            logger.warning(f"Could not load raw character data for ID {character_id} to get original first message. Will rely on processed version if available.")
    except CharactersRAGDBError as e:
        logger.warning(f"DB error fetching raw character data for ID {character_id}: {e}. Proceeding with caution.")


    # 2. Load character for UI processing (placeholders, image etc.)
    # This char_data will have its fields (like 'first_message') processed with placeholders.
    char_data, initial_ui_history, img = load_character_and_image(db, character_id, user_name)

    if not char_data:
        logger.error(f"Failed to load character_id {character_id} (for UI processing) to start new chat session.")
        return None, None, None, None

    char_name = char_data.get('name', 'Character') # Should be valid if char_data exists

    # Create a title for the conversation
    conv_title = custom_title if custom_title else f"Chat with {char_name} ({time.strftime('%Y-%m-%d %H:%M')})"

    conversation_id_val: Optional[str] = None # Ensure it's defined for return in except block
    try:
        # Add conversation to DB
        conv_payload = {
            'character_id': character_id,
            'title': conv_title,
        }
        conversation_id_val = db.add_conversation(conv_payload)

        if not conversation_id_val:
            logger.error(f"Failed to create conversation record in DB for character {char_name}.")
            return None, char_data, initial_ui_history, img

        logger.info(f"Created new conversation ID: {conversation_id_val} for character '{char_name}'.")

        # Determine the first message content to store in the DB for the new conversation
        message_to_store_in_db: Optional[str] = original_first_message_content

        if message_to_store_in_db is None: # Fallback if raw fetch failed but processed one exists
            if initial_ui_history and initial_ui_history[0] and initial_ui_history[0][1]:
                # This is already processed. Storing processed message if raw isn't available.
                # This implies the char_data['first_message'] from load_character_and_image
                message_to_store_in_db = initial_ui_history[0][1]
                logger.warning(f"Storing processed first message for char {char_name} in new conversation {conversation_id_val} as raw version was not available.")
            elif char_data.get('first_message'): # Another fallback to the processed field from char_data
                 message_to_store_in_db = char_data['first_message']
                 logger.warning(f"Storing processed first_message from char_data for char {char_name} in new conversation {conversation_id_val}.")


        if message_to_store_in_db:
            db.add_message({
                'conversation_id': conversation_id_val,
                'sender': char_name, # Character's name as sender
                'content': message_to_store_in_db, # Stored raw preferably, or processed as fallback
            })
            logger.debug(f"Added character's first message to new conversation {conversation_id_val}.")
        else:
            logger.warning(f"Character {char_name} (ID: {character_id}) has no first message to add to new conversation {conversation_id_val}.")
            # Ensure initial_ui_history is empty if no first message was effectively determined for UI
            if not (initial_ui_history and initial_ui_history[0] and initial_ui_history[0][1]):
                 initial_ui_history = []

        return conversation_id_val, char_data, initial_ui_history, img

    except (CharactersRAGDBError, InputError, ConflictError) as e:
        logger.error(f"Error during new chat session creation for char {char_name}: {e}")
        return conversation_id_val, char_data, initial_ui_history, img
    except Exception as e:
        logger.error(f"Unexpected error in start_new_chat_session: {e}", exc_info=True)
        return conversation_id_val, char_data, initial_ui_history, img


def list_character_conversations(db: CharactersRAGDB, character_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Lists active conversations for a given character.

    Retrieves a paginated list of conversation metadata dictionaries associated
    with the specified character ID. Conversations are typically ordered by
    their last modification time in descending order by the DB layer.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        character_id (int): The ID of the character whose conversations are to be listed.
        limit (int): The maximum number of conversations to return. Defaults to 50.
        offset (int): The number of conversations to skip (for pagination).
            Defaults to 0.

    Returns:
        List[Dict[str, Any]]: A list of conversation metadata dictionaries.
        Each dictionary typically includes 'id', 'title', 'character_id',
        'last_modified_at', etc. Returns an empty list if no conversations
        are found or an error occurs.
    """
    try:
        return db.get_conversations_for_character(character_id, limit=limit, offset=offset)
    except CharactersRAGDBError as e:
        logger.error(f"Failed to list conversations for character ID {character_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error listing conversations for char ID {character_id}: {e}", exc_info=True)
        return []


def get_conversation_metadata(db: CharactersRAGDB, conversation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves metadata for a specific conversation.

    Fetches the metadata associated with a given conversation ID from the database.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        conversation_id (str): The ID of the conversation to retrieve.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the conversation's
        metadata (e.g., 'id', 'title', 'character_id', 'last_modified_at',
        'version') if found. Returns None if the conversation does not exist
        or an error occurs.
    """
    try:
        return db.get_conversation_by_id(conversation_id)
    except CharactersRAGDBError as e:
        logger.error(f"Failed to get metadata for conversation ID {conversation_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting conversation metadata for ID {conversation_id}: {e}", exc_info=True)
        return None


def update_conversation_metadata(db: CharactersRAGDB, conversation_id: str, update_data: Dict[str, Any], expected_version: int) -> bool:
    """Updates metadata for a specific conversation.

    Allows modification of permissible fields of a conversation's metadata,
    such as 'title' or 'rating'. Uses optimistic locking via `expected_version`.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        conversation_id (str): The ID of the conversation to update.
        update_data (Dict[str, Any]): A dictionary containing the fields to
            update and their new values. Only specific keys (e.g., 'title',
            'rating') are typically allowed for update by this function.
        expected_version (int): The version number of the conversation record
            that the client expects to be updating, for optimistic concurrency control.

    Returns:
        bool: `True` if the update was successful. `False` if the update failed,
        which could be due to a version mismatch (ConflictError), invalid input,
        database error, or if no valid fields were provided for update.
    """
    try:
        # Ensure client_id is not in update_data, as db layer handles it.
        # Also, character_id, root_id, etc., are typically not changed via this simple update.
        valid_update_keys = {'title', 'rating'} # Define what's permissible to update via this func
        payload_to_db = {k: v for k, v in update_data.items() if k in valid_update_keys}

        if not payload_to_db:
            logger.warning(f"No valid fields to update for conversation ID {conversation_id} from data: {update_data}")
            # Depending on desired behavior, could return True (if version matches, effectively a "touch")
            # or False. db.update_conversation will still bump version if payload is empty.
            # Let's proceed, db layer will handle empty payload by just bumping version.
            pass

        return db.update_conversation(conversation_id, payload_to_db, expected_version)
    except (CharactersRAGDBError, InputError, ConflictError) as e:
        logger.error(f"Failed to update metadata for conversation ID {conversation_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error updating conversation metadata for ID {conversation_id}: {e}", exc_info=True)
        return False


def delete_conversation_by_id(db: CharactersRAGDB, conversation_id: str, expected_version: int) -> bool:
    """Soft-deletes a conversation from the database.

    Marks a conversation as deleted rather than physically removing it.
    Uses optimistic locking via `expected_version`.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        conversation_id (str): The ID of the conversation to soft-delete.
        expected_version (int): The version number of the conversation record
            that the client expects to be deleting, for optimistic concurrency control.

    Returns:
        bool: `True` if the conversation was successfully soft-deleted.
        `False` if the deletion failed, e.g., due to a version mismatch
        (ConflictError), or a database error.
    """
    try:
        return db.soft_delete_conversation(conversation_id, expected_version)
    except (CharactersRAGDBError, ConflictError) as e:
        logger.error(f"Failed to delete conversation ID {conversation_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting conversation ID {conversation_id}: {e}", exc_info=True)
        return False


def search_conversations_by_title_query(db: CharactersRAGDB, title_query: str, character_id: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Searches for conversations by their title.

    Performs a search for conversations whose titles match (partially or fully,
    depending on DB implementation) the `title_query`. Can be filtered by
    `character_id`.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        title_query (str): The text to search for in conversation titles.
        character_id (Optional[int]): If provided, limits the search to
            conversations associated with this character ID. Defaults to None.
        limit (int): The maximum number of matching conversations to return.
            Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of conversation metadata dictionaries
        for matching conversations. Returns an empty list if no matches are
        found or an error occurs.
    """
    try:
        return db.search_conversations_by_title(title_query, character_id=character_id, limit=limit)
    except CharactersRAGDBError as e:
        logger.error(f"Failed to search conversations with query '{title_query}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching conversations: {e}", exc_info=True)
        return []

# --- Message Management ---

def post_message_to_conversation(
    db: CharactersRAGDB,
    conversation_id: str,
    character_name: str, # The actual name of the character involved, used if is_user_message is False
    message_content: str,
    is_user_message: bool,
    parent_message_id: Optional[str] = None,
    ranking: Optional[int] = None,
    image_data: Optional[bytes] = None,
    image_mime_type: Optional[str] = None
) -> Optional[str]:
    """Posts a new message to a specified conversation.

    Adds a message to the database for the given `conversation_id`. The message
    content is stored raw (placeholders are typically not replaced at this stage).
    The sender is determined by `is_user_message`: if `True`, sender is "User";
    otherwise, it's the provided `character_name`.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        conversation_id (str): The ID of the conversation to post the message to.
        character_name (str): The name of the character involved in the
            conversation. Used as the sender if `is_user_message` is `False`.
        message_content (str): The text content of the message.
        is_user_message (bool): `True` if the message is from the user, `False`
            if it's from the character.
        parent_message_id (Optional[str]): The ID of the parent message if this
            message is a reply in a threaded structure. Defaults to None.
        ranking (Optional[int]): An optional ranking for the message (e.g., for
            alternative AI responses). Defaults to None.
        image_data (Optional[bytes]): Optional binary image data associated with
            the message. Defaults to None.
        image_mime_type (Optional[str]): The MIME type of the `image_data`
            (e.g., "image/png"), required if `image_data` is provided.
            Defaults to None.

    Returns:
        Optional[str]: The ID of the newly created message if successful.
        Returns None if the message posting fails.

    Raises:
        InputError: If `conversation_id` is missing, or if `character_name` is
            missing for a character message, or if both `message_content` and
            `image_data` are missing.
        CharactersRAGDBError: For other database-related errors during message addition.
        ConflictError: If a conflict occurs (e.g. parent_message_id not found,
            depending on DB constraints).
    """
    if not conversation_id:
        logger.error("Cannot post message: conversation_id is required.")
        raise InputError("conversation_id is required for posting a message.") # Raise to signal client error
    if not character_name and not is_user_message:
        logger.error("Cannot post character message: character_name is required.")
        raise InputError("character_name is required for character messages.")

    sender_name = "User" if is_user_message else character_name

    # Ensure content or image is present, as per DB layer check
    if not message_content and not image_data:
        logger.error("Cannot post message: Message must have text content or image data.")
        raise InputError("Message must have text content or image data.")


    msg_payload = {
        'conversation_id': conversation_id,
        'sender': sender_name,
        'content': message_content,
        'parent_message_id': parent_message_id,
        'ranking': ranking,
        'image_data': image_data,
        'image_mime_type': image_mime_type,
    }

    try:
        message_id = db.add_message(msg_payload)
        if message_id:
            logger.info(f"Posted message ID {message_id} from '{sender_name}' to conversation {conversation_id}.")
        else:
            # This case should ideally be covered by exceptions from db.add_message
            logger.error(f"Failed to post message from '{sender_name}' to conversation {conversation_id} (DB returned no ID without error).")
        return message_id
    except (CharactersRAGDBError, InputError, ConflictError) as e: # InputError, ConflictError from DB layer
        logger.error(f"Error posting message from '{sender_name}' to conversation {conversation_id}: {e}")
        raise # Re-raise client-correctable or conflict errors
    except Exception as e:
        logger.error(f"Unexpected error posting message to conv {conversation_id}: {e}", exc_info=True)
        # For unexpected errors, convert to a library-specific error or return None
        # depending on desired API contract for unhandled exceptions.
        # For now, re-raising as a generic error or letting it propagate if not caught by CharactersRAGDBError.
        # To be safe, wrap in CharactersRAGDBError if it's not one already.
        if not isinstance(e, CharactersRAGDBError):
             raise CharactersRAGDBError(f"Unexpected error posting message: {e}") from e
        raise


def retrieve_message_details(
    db: CharactersRAGDB,
    message_id: str,
    character_name_for_placeholders: str,
    user_name_for_placeholders: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Retrieves a specific message by its ID and processes its content.

    Fetches a message from the database using its unique ID. If the message
    has text content, placeholders within it are replaced using the provided
    character and user names.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        message_id (str): The ID of the message to retrieve.
        character_name_for_placeholders (str): The name of the character,
            used for replacing '{{char}}' placeholders in the message content.
        user_name_for_placeholders (Optional[str]): The name of the user,
            used for replacing '{{user}}' placeholders.

    Returns:
        Optional[Dict[str, Any]]: A dictionary containing the message's data
        (e.g., 'id', 'sender', 'content', 'timestamp', 'image_data') if found.
        The 'content' field will have placeholders replaced. Returns None if
        the message is not found or an error occurs.
    """
    try:
        message_data = db.get_message_by_id(message_id)
        if not message_data:
            return None

        if 'content' in message_data and isinstance(message_data['content'], str):
            message_data['content'] = replace_placeholders(
                message_data['content'],
                character_name_for_placeholders,
                user_name_for_placeholders
            )
        return message_data
    except CharactersRAGDBError as e:
        logger.error(f"Failed to retrieve message ID {message_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error retrieving message ID {message_id}: {e}", exc_info=True)
        return None


def retrieve_conversation_messages_for_ui(
    db: CharactersRAGDB,
    conversation_id: str,
    character_name: str,
    user_name: Optional[str],
    limit: int = 2000,
    offset: int = 0,
    order: str = "ASC"
) -> List[Tuple[Optional[str], Optional[str]]]:
    """Retrieves and processes conversation messages for UI display.

    Fetches messages for a given conversation from the database, then
    processes them using `process_db_messages_to_ui_history` into a
    UI-friendly paired format (user_message, bot_message). Placeholder
    replacement is applied to message content during this processing.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        conversation_id (str): The ID of the conversation whose messages
            are to be retrieved.
        character_name (str): The name of the character involved in the
            conversation. Used for placeholder replacement and identifying
            character messages during processing.
        user_name (Optional[str]): The name of the user, for placeholder
            replacement.
        limit (int): The maximum number of messages to retrieve (before
            pairing). Defaults to 2000.
        offset (int): The number of messages to skip (for pagination, applied
            before pairing). Defaults to 0.
        order (str): The order in which to retrieve messages by timestamp
            ("ASC" for ascending, "DESC" for descending). Defaults to "ASC".

    Returns:
        List[Tuple[Optional[str], Optional[str]]]: A list of
        (user_message, bot_message) tuples, suitable for UI display.
        Returns an empty list if an error occurs or no messages are found.
    """
    order_upper = order.upper()
    if order_upper not in ["ASC", "DESC"]:
        logger.warning(f"Invalid order '{order}' for message retrieval. Defaulting to ASC.")
        order_upper = "ASC"

    try:
        raw_db_messages = db.get_messages_for_conversation(
            conversation_id,
            limit=limit,
            offset=offset,
            order_by_timestamp=order_upper
        )

        processed_ui_history = process_db_messages_to_ui_history(
            raw_db_messages,
            char_name_from_card=character_name,
            user_name_for_placeholders=user_name,
            actual_user_sender_id_in_db="User", # Convention from post_message_to_conversation
            actual_char_sender_id_in_db=character_name # Convention from post_message_to_conversation
        )
        return processed_ui_history

    except CharactersRAGDBError as e:
        logger.error(f"Failed to retrieve and process messages for conversation ID {conversation_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error retrieving UI messages for conversation {conversation_id}: {e}", exc_info=True)
        return []


def edit_message_content(
    db: CharactersRAGDB,
    message_id: str,
    new_content: str, # Raw content; placeholders processed on display
    expected_version: int
) -> bool:
    """Updates the text content of a specific message.

    Modifies the 'content' field of an existing message in the database.
    The `new_content` is stored raw; placeholder replacement happens
    during display/retrieval. Uses optimistic locking via `expected_version`.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        message_id (str): The ID of the message to edit.
        new_content (str): The new raw text content for the message.
        expected_version (int): The version number of the message record
            that the client expects to be updating, for optimistic concurrency control.

    Returns:
        bool: `True` if the message content was successfully updated. `False`
        if the update failed, e.g., due to a version mismatch (ConflictError),
        invalid input, or a database error.
    """
    update_payload = {'content': new_content}
    try:
        return db.update_message(message_id, update_payload, expected_version)
    except (CharactersRAGDBError, InputError, ConflictError) as e:
        logger.error(f"Failed to edit content for message ID {message_id}: {e}")
        return False # Or re-raise depending on API contract for these errors
    except Exception as e:
        logger.error(f"Unexpected error editing message content for ID {message_id}: {e}", exc_info=True)
        return False


def set_message_ranking(
    db: CharactersRAGDB,
    message_id: str,
    ranking: int,
    expected_version: int
) -> bool:
    """Sets or updates the ranking of a specific message.

    Modifies the 'ranking' field of an existing message. This can be used, for
    example, to rate alternative AI responses. Uses optimistic locking via
    `expected_version`.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        message_id (str): The ID of the message whose ranking is to be set.
        ranking (int): The new ranking value for the message.
        expected_version (int): The version number of the message record
            that the client expects to be updating, for optimistic concurrency control.

    Returns:
        bool: `True` if the message ranking was successfully updated. `False`
        if the update failed, e.g., due to a version mismatch (ConflictError),
        invalid input, or a database error.
    """
    update_payload = {'ranking': ranking}
    try:
        return db.update_message(message_id, update_payload, expected_version)
    except (CharactersRAGDBError, InputError, ConflictError) as e:
        logger.error(f"Failed to set ranking for message ID {message_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error setting message ranking for ID {message_id}: {e}", exc_info=True)
        return False

def remove_message_from_conversation(
    db: CharactersRAGDB,
    message_id: str,
    expected_version: int
) -> bool:
    """Soft-deletes a message from a conversation.

    Marks a message as deleted in the database rather than physically removing it.
    Uses optimistic locking via `expected_version`.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        message_id (str): The ID of the message to soft-delete.
        expected_version (int): The version number of the message record
            that the client expects to be deleting, for optimistic concurrency control.

    Returns:
        bool: `True` if the message was successfully soft-deleted. `False`
        if the deletion failed, e.g., due to a version mismatch
        (ConflictError), or a database error.
    """
    try:
        return db.soft_delete_message(message_id, expected_version)
    except (CharactersRAGDBError, ConflictError) as e:
        logger.error(f"Failed to remove message ID {message_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error removing message ID {message_id}: {e}", exc_info=True)
        return False


def find_messages_in_conversation(
    db: CharactersRAGDB,
    conversation_id: str,
    search_query: str,
    character_name_for_placeholders: str,
    user_name_for_placeholders: Optional[str],
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Searches for messages within a specific conversation by content.

    Finds messages in the given `conversation_id` whose text content matches
    (partially or fully, depending on DB implementation) the `search_query`.
    Placeholder replacement is applied to the 'content' field of the
    found messages.

    Args:
        db (CharactersRAGDB): An instance of the character database manager.
        conversation_id (str): The ID of the conversation to search within.
        search_query (str): The text to search for in message content.
        character_name_for_placeholders (str): The name of the character,
            used for replacing '{{char}}' placeholders in the content of
            found messages.
        user_name_for_placeholders (Optional[str]): The name of the user,
            used for replacing '{{user}}' placeholders.
        limit (int): The maximum number of matching messages to return.
            Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of message data dictionaries for matching
        messages. The 'content' field in each dictionary will have placeholders
        replaced. Returns an empty list if no matches are found or an error occurs.
    """
    try:
        found_messages = db.search_messages_by_content(
            content_query=search_query,
            conversation_id=conversation_id,
            limit=limit
        )

        processed_results = []
        for msg_data in found_messages:
            if 'content' in msg_data and isinstance(msg_data['content'], str):
                msg_data['content'] = replace_placeholders(
                    msg_data['content'],
                    character_name_for_placeholders,
                    user_name_for_placeholders
                )
            processed_results.append(msg_data)
        return processed_results
    except CharactersRAGDBError as e:
        logger.error(f"Failed to search messages in conversation ID {conversation_id} for '{search_query}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error searching messages in conversation {conversation_id}: {e}", exc_info=True)
        return []

# End of Conversation and Message Management Functions
#################################################################################


#
# End of File
########################################################################################################################

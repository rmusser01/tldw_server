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
    """
    Replace placeholders in the given text with appropriate values.
    """
    if not text:  # Guard against None or empty string
        return ""  # Return empty string if input is None or empty

    # Ensure char_name and user_name are strings, even if None initially
    char_name_actual = char_name if char_name is not None else "Character"
    user_name_actual = user_name if user_name is not None else "User"

    replacements = {
        '{{char}}': char_name_actual,
        '{{user}}': user_name_actual,
        '{{random_user}}': user_name_actual,  # As per original logic
        '<USER>': user_name_actual,  # Common alternative
        '<CHAR>': char_name_actual,  # Common alternative
    }

    processed_text = text
    for placeholder, value in replacements.items():
        processed_text = processed_text.replace(placeholder, value)
    return processed_text


def replace_user_placeholder(history: List[Tuple[Optional[str], Optional[str]]], user_name: Optional[str]) -> List[
    Tuple[Optional[str], Optional[str]]]:
    """
    Replaces all instances of '{{user}}' in the chat history with the actual user name.
    This function processes the List[Tuple(user_msg, bot_msg)] format.
    """
    user_name_actual = user_name if user_name else "User"  # Default name if none provided

    updated_history = []
    for user_msg, bot_msg in history:
        updated_user_msg = None
        if user_msg:
            updated_user_msg = user_msg.replace("{{user}}", user_name_actual)

        updated_bot_msg = None
        if bot_msg:
            updated_bot_msg = bot_msg.replace("{{user}}", user_name_actual)
        updated_history.append((updated_user_msg, updated_bot_msg))
    return updated_history


#
# End of Placeholder functions
#################################################################################

#################################################################################
#
# Functions for character interaction (DB focused):

def get_character_list_for_ui(db: CharactersRAGDB, limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Fetches a list of characters (ID and name) suitable for UI dropdowns.
    """
    try:
        # Assuming CharactersRAGDB.list_character_cards returns more fields,
        # we select only what's needed.
        all_chars = db.list_character_cards(limit=limit)  # Use parameter
        ui_list = [{"id": char.get("id"), "name": char.get("name")} for char in all_chars if
                   char.get("id") and char.get("name")]
        return sorted(ui_list, key=lambda x: x["name"].lower() if x["name"] else "")
    except CharactersRAGDBError as e:
        logger.error(f"Database error fetching character list for UI: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching character list for UI: {e}", exc_info=True)
        return []


def extract_character_id_from_ui_choice(choice: str) -> int:
    """
    Extract the character ID from a UI dropdown-like selection string.
    Example: "My Character (ID: 123)" -> 123
    Also handles if `choice` is just an integer string.
    """
    logger.debug(f"Choice received for ID extraction: {choice}")
    if not choice:
        raise ValueError("No choice provided for character ID extraction.")

    # Regex to find (ID: <numbers>) at the end of the string
    match = re.search(r'\(ID:\s*(\d+)\s*\)$', choice)
    if match:
        character_id_str = match.group(1)
    else:
        # If no match, assume the whole string might be an ID
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
    """
    Load a character, its first message as initial chat history, and its image from the database.
    Performs placeholder replacement on relevant character fields and the first message.
    The output chat_history is List[Tuple[user_message, bot_message]]
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
    """
    Processes a list of message dictionaries from the DB into the UI's paired chat history format.
    Handles placeholder replacement. Assumes messages are ordered by timestamp.
    Output format: List of (user_message, bot_message_or_none_if_user_is_last)
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
    """
    Load an existing chat (conversation) and its associated character data and image.
    Chat history is returned in the List[Tuple[user_msg, bot_msg]] format.
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
    """Wrapper function to load character and image using either an ID or a UI choice string."""
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


#
# Character Book parsing (copied from original, assumed correct for V2 spec)
#
def parse_character_book(book_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the character book data from a V2 character card.
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
    """
    Extracts 'chara' metadata (base64 encoded JSON) from a PNG image.
    Input can be file path, raw bytes, or a BytesIO stream.
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
    """
    Parse a V2 character card (spec_version '2.0').
    Outputs a dictionary with keys matching the DB schema (e.g., 'first_message').
    Assumes basic structural validation (e.g., presence of 'data' node if 'spec' implies V2)
    has already been done by the caller or a higher-level validation function.
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
    """
    Convert a V1 card (flat JSON) into a V2-like dictionary structure,
    with keys matching the DB schema.
    Raises ValueError if required V1 fields are missing.
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
# Character card parsing & Validation functions (Copied from original):
# These validate the *structure* of the card data, typically after parsing from JSON.

def validate_character_book(book_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate the 'character_book' field in the character card.
    (Copied from original Character_Chat_Lib)
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
    """
    Validate an entry in the 'character_book.entries' list.
    (Copied from original Character_Chat_Lib)
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
    """
    Validate a character card according to the V2 specification.
    (Copied from original Character_Chat_Lib, assumes card_data is the full card object)
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
    """
    Import and parse a character card from a JSON string. Validates V2 structure first if applicable.
    Detects V1 vs V2. Returns a dictionary mapped to DB schema names, or None on failure.
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
    """
    Load a character card from a string (JSON, or Markdown with YAML/JSON block).
    Returns a parsed card dictionary (mapped to DB schema names), or None on failure.
    This function *only* parses the string content, it does not save to DB.
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
            except ImportError:
                logger.error("PyYAML is required for loading YAML front matter. Install it via 'pip install PyYAML'.")
                # Fall through to check for JSON block if YAML is not available or fails
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
        file_input: Union[str, io.BytesIO, bytes]  # File path, BytesIO stream, or raw bytes
) -> Optional[int]:
    """
    Loads character card from a JSON/MD text file, or a PNG/WEBP image file with embedded data.
    Parses and validates the card data, and saves to the database.
    Returns the character_id from the database if successful, otherwise None.
    """
    parsed_card_dict: Optional[Dict[str, Any]] = None
    image_bytes_for_db: Optional[bytes] = None  # This will hold the avatar image for the DB
    filename_for_log = "input_stream"

    try:
        # 1. Determine input type and get card JSON string and potentially image bytes
        if isinstance(file_input, str):  # File path
            filename_for_log = file_input
            if not os.path.exists(filename_for_log):
                logger.error(f"File not found: {filename_for_log}")
                return None

            _, ext = os.path.splitext(filename_for_log.lower())
            if ext in ['.png', '.webp']:  # Image file
                with open(filename_for_log, 'rb') as f_img:
                    image_bytes_for_db = f_img.read()  # The file itself is the image
                card_json_str = extract_json_from_image_file(io.BytesIO(image_bytes_for_db))
                if not card_json_str:
                    logger.warning(
                        f"No character JSON data extracted from image file: {filename_for_log}. Image itself will be used if JSON is found elsewhere or card has default image handling.")
                    # If no JSON in image, card_json_str will be None. Parsing might happen from a text file later if this function is adapted
                    # For current design, if image has no JSON, it must be a text file for card data.
                    # If the intent is to load image AND then separately load a JSON file, this function would need changes.
                    # Assuming for now: if image, JSON must be in it or it's an error for *this specific path*.
                    # Re-evaluating based on typical use: if image file, json expected inside.
                    # If no JSON inside, we don't then try to read the image file as text.
                    # So, if card_json_str is None here, and it's an image, then we lack card data.
                    if ext in ['.png', '.webp'] and not card_json_str:  # Explicitly state no card data from image
                        logger.error(
                            f"Image file {filename_for_log} provided, but no character JSON metadata found within it.")
                        return None
            else:  # Assume text file (JSON/MD)
                with open(filename_for_log, 'r', encoding='utf-8') as f_text:
                    card_json_str = f_text.read()

        elif isinstance(file_input, bytes):  # Raw bytes input
            try:
                temp_image_stream = io.BytesIO(file_input)
                potential_json_from_bytes_img = extract_json_from_image_file(temp_image_stream)
                if potential_json_from_bytes_img:
                    card_json_str = potential_json_from_bytes_img
                    image_bytes_for_db = file_input
                else:
                    logger.debug("Input bytes not an image with chara data, or not an image; trying as text.")
                    card_json_str = file_input.decode('utf-8')
                    # If it was an image but without chara, image_bytes_for_db would still be None.
                    # We could try to set image_bytes_for_db = file_input here if we confirm it IS an image
                    # even if chara extraction failed. For now, only if chara data is from image.
            except UnicodeDecodeError:
                logger.error("Input bytes are not valid UTF-8 text and didn't yield chara data as an image.")
                return None
            except Exception as e_bytes_img:
                logger.debug(f"Input bytes not processed as image ({e_bytes_img}), trying as text.")
                try:
                    card_json_str = file_input.decode('utf-8')
                except UnicodeDecodeError:
                    logger.error("Input bytes are not valid UTF-8 text.")
                    return None

        elif hasattr(file_input, 'read'):  # File-like object (e.g., BytesIO from upload)
            if hasattr(file_input, 'name') and file_input.name: filename_for_log = file_input.name
            file_input.seek(0)
            stream_bytes = file_input.read()
            file_input.seek(0)

            try:
                temp_image_stream_from_obj = io.BytesIO(stream_bytes)
                potential_json_from_stream_img = extract_json_from_image_file(temp_image_stream_from_obj)
                if potential_json_from_stream_img:
                    card_json_str = potential_json_from_stream_img
                    image_bytes_for_db = stream_bytes
                else:
                    logger.debug(
                        f"Stream {filename_for_log} not an image with chara data, or not an image; trying as text.")
                    card_json_str = stream_bytes.decode('utf-8')
            except UnicodeDecodeError:
                logger.error(
                    f"Stream content for {filename_for_log} is not valid UTF-8 and didn't yield chara from image.")
                return None
            except Exception as e_stream_img:
                logger.debug(f"Stream {filename_for_log} not processed as image ({e_stream_img}), trying as text.")
                try:
                    card_json_str = stream_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    logger.error(f"Stream content for {filename_for_log} is not valid UTF-8 text.")
                    return None
        else:
            logger.error("Invalid file_input type. Must be file path, BytesIO, or bytes.")
            return None

        if not card_json_str:
            logger.error(f"Could not obtain character card JSON string from input: {filename_for_log}")
            return None

        # 2. Parse and Validate the extracted JSON string.
        # load_character_card_from_string_content now incorporates validation.
        parsed_card_dict = load_character_card_from_string_content(card_json_str)
        if not parsed_card_dict:  # This means parsing or validation failed.
            logger.error(f"Failed to parse or validate character data from content of: {filename_for_log}")
            return None

        # 3. Post-parsing check (essential fields on the *parsed and mapped* dictionary)
        if not parsed_card_dict.get('name'):
            logger.error(
                "Character import failed: 'name' is missing in the successfully parsed and DB-schema-mapped card data.")
            return None
        # Add more critical field checks here on `parsed_card_dict` if needed.

        # 4. Handle image if it's base64 in the JSON and not already set from image file
        if not image_bytes_for_db and parsed_card_dict.get('image_base64'):
            try:
                image_bytes_for_db = base64.b64decode(str(parsed_card_dict['image_base64']))
                logger.debug("Decoded base64 image from card JSON.")
            except Exception as e_b64:
                logger.warning(f"Failed to decode base64 image string from card data: {e_b64}")
                # Keep image_bytes_for_db as None

        # 5. Prepare the payload for the database, using DB schema field names
        db_payload = {
            'name': parsed_card_dict['name'],
            'description': parsed_card_dict.get('description'),
            'personality': parsed_card_dict.get('personality'),
            'scenario': parsed_card_dict.get('scenario'),
            'system_prompt': parsed_card_dict.get('system_prompt'),
            'image': image_bytes_for_db,
            'post_history_instructions': parsed_card_dict.get('post_history_instructions'),
            'first_message': parsed_card_dict.get('first_message'),
            'message_example': parsed_card_dict.get('message_example'),
            'creator_notes': parsed_card_dict.get('creator_notes'),
            'alternate_greetings': parsed_card_dict.get('alternate_greetings', []),
            'tags': parsed_card_dict.get('tags', []),
            'creator': parsed_card_dict.get('creator'),
            'character_version': parsed_card_dict.get('character_version'),
            'extensions': parsed_card_dict.get('extensions', {})
        }

        if not isinstance(db_payload['alternate_greetings'], list): db_payload['alternate_greetings'] = []
        if not isinstance(db_payload['tags'], list): db_payload['tags'] = []
        if not isinstance(db_payload['extensions'], dict): db_payload['extensions'] = {}

        # 6. Add to database
        char_id = db.add_character_card(db_payload)
        if char_id:
            logger.info(f"Successfully imported character '{db_payload['name']}' with DB ID: {char_id}")
        else:
            logger.error(
                f"Failed to save character '{db_payload['name']}' to DB (add_character_card returned None without error).")  # Should ideally not happen
        return char_id

    except ConflictError as ce:
        logger.warning(f"Conflict importing character: {ce}. Name likely already exists.")
        if parsed_card_dict and parsed_card_dict.get(
                'name'):  # parsed_card_dict might be None if error happened before it was set
            existing_char = db.get_character_card_by_name(parsed_card_dict['name'])
            if existing_char and existing_char.get('id'):
                logger.info(f"Character '{parsed_card_dict['name']}' already exists with ID {existing_char['id']}.")
                return existing_char['id']
        return None
    except (CharactersRAGDBError, InputError) as db_e:
        logger.error(f"Database or input error importing character from {filename_for_log}: {db_e}")
    except ImportError as imp_err:
        logger.error(f"Import error during character import: {imp_err}. A required library might be missing.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error importing character from {filename_for_log}: {e}", exc_info=True)
    return None


def load_chat_history_from_file_and_save_to_db(
        db: CharactersRAGDB,
        file_path_or_obj: Union[str, io.BytesIO],
        user_name_for_placeholders: Optional[str] = "User",  # Used if placeholders are in old log
        default_user_sender_in_db: str = "User"  # How to label user messages in the DB
) -> Tuple[Optional[str], Optional[int]]:
    """
    Loads chat history from a JSON file (TavernAI/SillyTavern format),
    finds/verifies the character in the DB, creates a new conversation,
    and adds all messages to this conversation in the database.

    Expected JSON format:
    {
      "char_name": "Character Name", // or "character"
      "user_name": "User Name", // optional
      "history": { // or "chat"
        "internal": [ ["User message 1", "Char message 1"], ["User message 2", "Char message 2"] ],
        "visible": [ ... ] // same structure
      }
    }
    Or simpler: {"history": [["user_msg1", "bot_msg1"], ...], "character": "CharName"}


    Returns (new_conversation_id_str, character_id_int) or (None, None) on failure.
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

#
# End of File
########################################################################################################################

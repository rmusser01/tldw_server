# Character_Chat_Lib.py
# Description: Functions for character chat cards.
#
# Imports
import json
import io
import base64
import os
import re
import time
from typing import Dict, Any, Optional, List, Tuple, Union, cast
#
# External Imports
from PIL import Image
#
# Local imports
from App_Function_Libraries.DB.DB_Manager import get_character_card_by_id, get_character_chat_by_id
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from App_Function_Libraries.Utils.Utils import logging


#
# Constants
####################################################################################################
#
# Functions
# Using https://github.com/malfoyslastname/character-card-spec-v2 as the standard for v2 character cards

###############################################
#
# Placeholder functions:

def replace_placeholders(text: str, char_name: str, user_name: str) -> str:
    """
    Replace placeholders in the given text with appropriate values.

    Args:
        text (str): The text containing placeholders.
        char_name (str): The name of the character.
        user_name (str): The name of the user.

    Returns:
        str: The text with placeholders replaced.
    """
    replacements = {
        '{{char}}': char_name,
        '{{user}}': user_name,
        '{{random_user}}': user_name  # Assuming random_user is the same as user for simplicity
    }

    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    return text

def replace_user_placeholder(history, user_name):
    """
    Replaces all instances of '{{user}}' in the chat history with the actual user name.

    Args:
        history (list): The current chat history as a list of tuples (user_message, bot_message).
        user_name (str): The name entered by the user.

    Returns:
        list: Updated chat history with placeholders replaced.
    """
    if not user_name:
        user_name = "User"  # Default name if none provided

    updated_history = []
    for user_msg, bot_msg in history:
        # Replace in user message
        if user_msg:
            user_msg = user_msg.replace("{{user}}", user_name)
        # Replace in bot message
        if bot_msg:
            bot_msg = bot_msg.replace("{{user}}", user_name)
        updated_history.append((user_msg, bot_msg))
    return updated_history

#
# End of Placeholder functions
#################################################################################

#################################################################################
#
# Functions for character card processing:

def extract_character_id(choice: str) -> int:
    """Extract the character ID from the dropdown selection string."""
    log_counter("extract_character_id_attempt")
    try:
        logging.debug(f"Choice received: {choice}")  # Debugging line
        if not choice:
            raise ValueError("No choice provided.")

        if '(ID: ' not in choice or ')' not in choice:
            raise ValueError(f"Invalid choice format: {choice}")

        # Extract the ID part
        id_part = choice.split('(ID: ')[1]
        character_id = int(id_part.rstrip(')'))

        logging.debug(f"Extracted character ID: {character_id}")  # Debugging line
        log_counter("extract_character_id_success")
        return character_id
    except Exception as e:
        log_counter("extract_character_id_error", labels={"error": str(e)})
        raise


def load_character_wrapper(character_id: int, user_name: str) -> Tuple[Dict[str, Any], List[Tuple[Optional[str], str]], Optional[Image.Image]]:
    """Wrapper function to load character and image using the extracted ID."""
    log_counter("load_character_wrapper_attempt")
    start_time = time.time()
    try:
        char_data, chat_history, img = load_character_and_image(character_id, user_name)
        load_duration = time.time() - start_time
        log_histogram("load_character_wrapper_duration", load_duration)
        log_counter("load_character_wrapper_success")
        return char_data, chat_history, img
    except Exception as e:
        log_counter("load_character_wrapper_error", labels={"error": str(e)})
        raise

def parse_character_book(book_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the character book data from a V2 character card.

    Args:
        book_data (Dict[str, Any]): The raw character book data from the character card.

    Returns:
        Dict[str, Any]: The parsed and structured character book data.
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

    for entry in book_data.get('entries', []):
        parsed_entry = {
            'keys': entry['keys'],
            'content': entry['content'],
            'extensions': entry.get('extensions', {}),
            'enabled': entry['enabled'],
            'insertion_order': entry['insertion_order'],
            'case_sensitive': entry.get('case_sensitive', False),
            'name': entry.get('name', ''),
            'priority': entry.get('priority'),
            'id': entry.get('id'),
            'comment': entry.get('comment', ''),
            'selective': entry.get('selective', False),
            'secondary_keys': entry.get('secondary_keys', []),
            'constant': entry.get('constant', False),
            'position': entry.get('position')
        }
        parsed_book['entries'].append(parsed_entry)

    return parsed_book

def load_character_and_image(character_id: int, user_name: str) -> Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], str]], Optional[Image.Image]]:
    """
    Load a character and its associated image based on the character ID.

    Args:
        character_id (int): The ID of the character to load.
        user_name (str): The name of the user, used for placeholder replacement.

    Returns:
        Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], str]], Optional[Image.Image]]:
        A tuple containing the character data, chat history, and character image (if available).
    """
    log_counter("load_character_and_image_attempt")
    start_time = time.time()
    try:
        char_data = get_character_card_by_id(character_id)
        if not char_data:
            log_counter("load_character_and_image_no_data")
            logging.warning(f"No character data found for ID: {character_id}")
            return None, [], None

        # Replace placeholders in character data
        for field in ['first_mes', 'mes_example', 'scenario', 'description', 'personality']:
            if field in char_data:
                char_data[field] = replace_placeholders(char_data[field], char_data['name'], user_name)

        # Replace placeholders in first_mes
        first_mes = char_data.get('first_mes', "Hello! I'm ready to chat.")
        first_mes = replace_placeholders(first_mes, char_data['name'], user_name)

        chat_history = [(None, first_mes)] if first_mes else []

        img = None
        if char_data.get('image'):
            try:
                image_data = base64.b64decode(char_data['image'])
                img = Image.open(io.BytesIO(image_data)).convert("RGBA")
                log_counter("load_character_image_success")
            except Exception as e:
                log_counter("load_character_image_error", labels={"error": str(e)})
                logging.error(f"Error processing image for character '{char_data['name']}': {e}")

        load_duration = time.time() - start_time
        log_histogram("load_character_and_image_duration", load_duration)
        log_counter("load_character_and_image_success")
        return char_data, chat_history, img

    except Exception as e:
        log_counter("load_character_and_image_error", labels={"error": str(e)})
        logging.error(f"Error in load_character_and_image: {e}")
        return None, [], None

def load_chat_and_character(chat_id: int, user_name: str) -> Tuple[Optional[Dict[str, Any]], List[Tuple[str, str]], Optional[Image.Image]]:
    """
    Load a chat and its associated character, including the character image and process templates.

    Args:
        chat_id (int): The ID of the chat to load.
        user_name (str): The name of the user.

    Returns:
        Tuple[Optional[Dict[str, Any]], List[Tuple[str, str]], Optional[Image.Image]]:
        A tuple containing the character data, processed chat history, and character image (if available).
    """
    log_counter("load_chat_and_character_attempt")
    start_time = time.time()
    try:
        # Load the chat
        chat = get_character_chat_by_id(chat_id)
        if not chat:
            log_counter("load_chat_and_character_no_chat")
            logging.warning(f"No chat found with ID: {chat_id}")
            return None, [], None

        # Load the associated character
        character_id = chat['character_id']
        char_data = get_character_card_by_id(character_id)
        if not char_data:
            log_counter("load_chat_and_character_no_character")
            logging.warning(f"No character found for chat ID: {chat_id}")
            return None, chat['chat_history'], None

        # Process the chat history
        processed_history = process_chat_history(chat['chat_history'], char_data['name'], user_name)

        # Load the character image
        img = None
        if char_data.get('image'):
            try:
                image_data = base64.b64decode(char_data['image'])
                img = Image.open(io.BytesIO(image_data)).convert("RGBA")
                log_counter("load_chat_character_image_success")
            except Exception as e:
                log_counter("load_chat_character_image_error", labels={"error": str(e)})
                logging.error(f"Error processing image for character '{char_data['name']}': {e}")

        # Process character data templates
        for field in ['first_mes', 'mes_example', 'scenario', 'description', 'personality']:
            if field in char_data:
                char_data[field] = replace_placeholders(char_data[field], char_data['name'], user_name)

        load_duration = time.time() - start_time
        log_histogram("load_chat_and_character_duration", load_duration)
        log_counter("load_chat_and_character_success")
        return char_data, processed_history, img

    except Exception as e:
        log_counter("load_chat_and_character_error", labels={"error": str(e)})
        logging.error(f"Error in load_chat_and_character: {e}")
        return None, [], None


def extract_json_from_image(image_file):
    # Get filename safely
    file_name = getattr(image_file, 'name', 'unknown_file')
    logging.debug(f"Attempting to extract JSON from image: {file_name}")

    log_counter("extract_json_from_image_attempt")
    start_time = time.time()
    try:
        # Ensure we're working from the start of the file
        if hasattr(image_file, 'seek'):
            image_file.seek(0)

        with Image.open(image_file) as img:
            logging.debug("Image opened successfully")
            metadata = img.text
            if 'chara' in metadata:
                logging.debug("Found 'chara' in image metadata")
                chara_content = metadata['chara']
                logging.debug(f"Content of 'chara' metadata (first 100 chars): {chara_content[:100]}...")
                try:
                    decoded_content = base64.b64decode(chara_content).decode('utf-8')
                    logging.debug(f"Decoded content (first 100 chars): {decoded_content[:100]}...")
                    log_counter("extract_json_from_image_metadata_success")
                    return decoded_content
                except Exception as e:
                    logging.error(f"Error decoding base64 content: {e}")
                    log_counter("extract_json_from_image_decode_error", labels={"error": str(e)})

            logging.warning("'chara' not found in metadata, attempting to find JSON data in image bytes")
            # Preserve PNG format check and conversion
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')  # Maintain PNG format enforcement
            img_bytes = img_byte_arr.getvalue()
            img_str = img_bytes.decode('latin1')

            # Search for JSON-like structures in the image bytes
            json_start = img_str.find('{')
            json_end = img_str.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                possible_json = img_str[json_start:json_end + 1]
                try:
                    json.loads(possible_json)
                    logging.debug("Found JSON data in image bytes")
                    log_counter("extract_json_from_image_bytes_success")
                    return possible_json
                except json.JSONDecodeError:
                    logging.debug("No valid JSON found in image bytes")
                    log_counter("extract_json_from_image_invalid_json")

            logging.warning("No JSON data found in the image")
            log_counter("extract_json_from_image_no_json_found")
    except Exception as e:
        log_counter("extract_json_from_image_error", labels={"error": str(e)})
        logging.error(f"Error extracting JSON from image: {e}")
    finally:
        # Reset file pointer if it exists
        if hasattr(image_file, 'seek'):
            image_file.seek(0)

    extract_duration = time.time() - start_time
    log_histogram("extract_json_from_image_duration", extract_duration)
    return None


def load_chat_history(file):
    log_counter("load_chat_history_attempt")
    start_time = time.time()
    try:
        content = file.read().decode('utf-8')
        chat_data = json.loads(content)

        # Extract history and character name from the loaded data
        history = chat_data.get('history') or chat_data.get('messages')
        character_name = chat_data.get('character') or chat_data.get('character_name')

        if not history or not character_name:
            log_counter("load_chat_history_incomplete_data")
            logging.error("Chat history or character name missing in the imported file.")
            return None, None

        load_duration = time.time() - start_time
        log_histogram("load_chat_history_duration", load_duration)
        log_counter("load_chat_history_success")
        return history, character_name
    except Exception as e:
        log_counter("load_chat_history_error", labels={"error": str(e)})
        logging.error(f"Error loading chat history: {e}")
        return None, None


def process_chat_history(chat_history: List[Tuple[str, str]], char_name: str, user_name: str) -> List[Tuple[str, str]]:
    """
    Process the chat history to replace placeholders in both user and character messages.

    Args:
        chat_history (List[Tuple[str, str]]): The chat history.
        char_name (str): The name of the character.
        user_name (str): The name of the user.

    Returns:
        List[Tuple[str, str]]: The processed chat history.
    """
    log_counter("process_chat_history_attempt")
    start_time = time.time()
    try:
        processed_history = []
        for user_msg, char_msg in chat_history:
            if user_msg:
                user_msg = replace_placeholders(user_msg, char_name, user_name)
            if char_msg:
                char_msg = replace_placeholders(char_msg, char_name, user_name)
            processed_history.append((user_msg, char_msg))

        process_duration = time.time() - start_time
        log_histogram("process_chat_history_duration", process_duration)
        log_counter("process_chat_history_success", labels={"message_count": len(chat_history)})
        return processed_history
    except Exception as e:
        log_counter("process_chat_history_error", labels={"error": str(e)})
        logging.error(f"Error processing chat history: {e}")
        raise

#
#################################################################################
# Character card parsing & Validation functions:
#

def parse_v2_card(card_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse a version 2 character card. Ensures the spec_version is '2.0' and required fields are present.
    Optionally parses additional fields like creator_notes, system_prompt, etc.
    """
    try:
        # Validate the spec_version
        if card_data.get('spec_version') != '2.0':
            logging.warning(f"Unsupported V2 spec version: {card_data.get('spec_version')}")
            return None

        # 'data' should hold the actual card information.
        data = card_data.get('data', {})
        # List of fields that must be present in V2.
        required_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
        for field in required_fields:
            if field not in data:
                logging.error(f"Missing required field in V2 card: {field}")
                return None

        # Build the parsed data dict including both required and optional fields.
        parsed_data = {
            'name': data['name'],
            'description': data['description'],
            'personality': data['personality'],
            'scenario': data['scenario'],
            'first_mes': data['first_mes'],
            'mes_example': data['mes_example'],
            'creator_notes': data.get('creator_notes', ''),
            'system_prompt': data.get('system_prompt', ''),
            'post_history_instructions': data.get('post_history_instructions', ''),
            'alternate_greetings': data.get('alternate_greetings', []),
            'tags': data.get('tags', []),
            'creator': data.get('creator', ''),
            'character_version': data.get('character_version', ''),
            'extensions': data.get('extensions', {})
        }

        # If a character book is provided, parse it (assuming parse_character_book is defined elsewhere).
        if 'character_book' in data:
            parsed_data['character_book'] = parse_character_book(data['character_book'])

        return parsed_data

    except KeyError as e:
        logging.error(f"Missing key in V2 card structure: {e}")
    except Exception as e:
        logging.error(f"Error parsing V2 card: {e}")
    return None


def parse_v1_card(card_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a V1 card into a format compatible with V2. Checks for required fields and moves any
    additional fields into the 'extensions' key.
    """
    # List of fields required in a V1 card.
    required_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
    for field in required_fields:
        if field not in card_data:
            logging.error(f"Missing required field in V1 card: {field}")
            raise ValueError(f"Missing required field in V1 card: {field}")

    # Create a new dict in V2 format using available V1 data.
    v2_data: Dict[str, Union[str, List[str], Dict[str, Any]]] = {
        'name': card_data['name'],
        'description': card_data['description'],
        'personality': card_data['personality'],
        'scenario': card_data['scenario'],
        'first_mes': card_data['first_mes'],
        'mes_example': card_data['mes_example'],
        'creator_notes': cast(str, card_data.get('creator_notes', '')),
        'system_prompt': cast(str, card_data.get('system_prompt', '')),
        'post_history_instructions': cast(str, card_data.get('post_history_instructions', '')),
        'alternate_greetings': cast(List[str], card_data.get('alternate_greetings', [])),
        'tags': cast(List[str], card_data.get('tags', [])),
        'creator': cast(str, card_data.get('creator', '')),
        'character_version': cast(str, card_data.get('character_version', '')),
        'extensions': {}
    }

    # Place any additional (non-standard) V1 fields into the 'extensions' dictionary.
    for key, value in card_data.items():
        if key not in v2_data:
            v2_data['extensions'][key] = value

    return v2_data


def validate_character_book(book_data):
    """
    Validate the 'character_book' field in the character card.

    Args:
        book_data (dict): The character book data.

    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean indicating validity and a list of validation messages.
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
        'entries': list
    }

    for field, expected_type in optional_fields.items():
        if field in book_data:
            if not isinstance(book_data[field], expected_type):
                validation_messages.append(f"Field 'character_book.{field}' must be of type '{expected_type}'.")
    # 'entries' is required
    if 'entries' not in book_data or not isinstance(book_data['entries'], list):
        validation_messages.append("Field 'character_book.entries' is required and must be a list.")
        return False, validation_messages

    # Validate each entry in 'entries'
    entries = book_data.get('entries', [])
    entry_ids = set()
    for idx, entry in enumerate(entries):
        is_valid_entry, entry_messages = validate_character_book_entry(entry, idx, entry_ids)
        if not is_valid_entry:
            validation_messages.extend(entry_messages)

    is_valid = len(validation_messages) == 0
    return is_valid, validation_messages

def validate_character_book_entry(entry, idx, entry_ids):
    """
    Validate an entry in the 'character_book.entries' list.

    Args:
        entry (dict): The entry data.
        idx (int): The index of the entry in the list.
        entry_ids (set): A set of existing entry IDs for uniqueness checking.

    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean indicating validity and a list of validation messages.
    """
    validation_messages = []
    required_fields = {
        'keys': list,
        'content': str,
        'extensions': dict,
        'enabled': bool,
        'insertion_order': (int, float)
    }

    for field, expected_type in required_fields.items():
        if field not in entry:
            validation_messages.append(f"Entry {idx}: Missing required field '{field}'.")
        elif not isinstance(entry[field], expected_type):
            validation_messages.append(f"Entry {idx}: Field '{field}' must be of type '{expected_type}'.")
        elif field == 'content' and not entry[field].strip():
            validation_messages.append(f"Entry {idx}: Field 'content' cannot be empty.")
        elif field == 'keys' and not entry[field]:
            validation_messages.append(f"Entry {idx}: Field 'keys' cannot be empty.")

    # Optional fields
    optional_fields = {
        'case_sensitive': bool,
        'name': str,
        'priority': (int, float),
        'id': (int, float),
        'comment': str,
        'selective': bool,
        'secondary_keys': list,
        'constant': bool,
        'position': str  # Should be 'before_char' or 'after_char'
    }

    for field, expected_type in optional_fields.items():
        if field in entry and not isinstance(entry[field], expected_type):
            validation_messages.append(f"Entry {idx}: Field '{field}' must be of type '{expected_type}'.")

    # Validate 'position' value if present
    if 'position' in entry:
        if entry['position'] not in ['before_char', 'after_char']:
            validation_messages.append(f"Entry {idx}: Field 'position' must be 'before_char' or 'after_char'.")

    # Validate 'secondary_keys' if 'selective' is True
    if entry.get('selective', False):
        if 'secondary_keys' not in entry or not isinstance(entry['secondary_keys'], list):
            validation_messages.append(f"Entry {idx}: 'secondary_keys' must be a list when 'selective' is True.")
        elif not entry['secondary_keys']:
            validation_messages.append(f"Entry {idx}: 'secondary_keys' cannot be empty when 'selective' is True.")

    # Validate 'keys' list elements
    if 'keys' in entry and isinstance(entry['keys'], list):
        for i, key in enumerate(entry['keys']):
            if not isinstance(key, str) or not key.strip():
                validation_messages.append(f"Entry {idx}: Element {i} in 'keys' must be a non-empty string.")

    # Validate 'secondary_keys' list elements
    if 'secondary_keys' in entry and isinstance(entry['secondary_keys'], list):
        for i, key in enumerate(entry['secondary_keys']):
            if not isinstance(key, str) or not key.strip():
                validation_messages.append(f"Entry {idx}: Element {i} in 'secondary_keys' must be a non-empty string.")

    # Validate 'id' uniqueness
    if 'id' in entry:
        entry_id = entry['id']
        if entry_id in entry_ids:
            validation_messages.append(f"Entry {idx}: \Duplicate 'id' value '{entry_id}'.\ Each entry 'id' must be unique.")
        else:
            entry_ids.add(entry_id)

    # Validate 'extensions' keys are namespaced
    if 'extensions' in entry and isinstance(entry['extensions'], dict):
        for key in entry['extensions'].keys():
            if '/' not in key and '_' not in key:
                validation_messages.append \
                    (f"Entry {idx}: Extension key '{key}' in 'extensions' should be namespaced to prevent conflicts.")

    is_valid = len(validation_messages) == 0
    return is_valid, validation_messages

def validate_v2_card(card_data):
    """
    Validate a character card according to the V2 specification.

    Args:
        card_data (dict): The parsed character card data.

    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean indicating validity and a list of validation messages.
    """
    validation_messages = []

    # Check top-level fields
    if 'spec' not in card_data:
        validation_messages.append("Missing 'spec' field.")
    elif card_data['spec'] != 'chara_card_v2':
        validation_messages.append(f"Invalid 'spec' value: {card_data['spec']}. Expected 'chara_card_v2'.")

    if 'spec_version' not in card_data:
        validation_messages.append("Missing 'spec_version' field.")
    else:
        # Ensure 'spec_version' is '2.0' or higher
        try:
            spec_version = float(card_data['spec_version'])
            if spec_version < 2.0:
                validation_messages.append \
                    (f"'spec_version' must be '2.0' or higher. Found '{card_data['spec_version']}'.")
        except ValueError:
            validation_messages.append \
                (f"Invalid 'spec_version' format: {card_data['spec_version']}. Must be a number as a string.")

    if 'data' not in card_data:
        validation_messages.append("Missing 'data' field.")
        return False, validation_messages  # Cannot proceed without 'data' field

    data = card_data['data']

    # Required fields in 'data'
    required_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
    for field in required_fields:
        if field not in data:
            validation_messages.append(f"Missing required field in 'data': '{field}'.")
        elif not isinstance(data[field], str):
            validation_messages.append(f"Field '{field}' must be a string.")
        elif not data[field].strip():
            validation_messages.append(f"Field '{field}' cannot be empty.")

    # Optional fields with expected types
    optional_fields = {
        'creator_notes': str,
        'system_prompt': str,
        'post_history_instructions': str,
        'alternate_greetings': list,
        'tags': list,
        'creator': str,
        'character_version': str,
        'extensions': dict,
        'character_book': dict  # If present, should be a dict
    }

    for field, expected_type in optional_fields.items():
        if field in data:
            if not isinstance(data[field], expected_type):
                validation_messages.append(f"Field '{field}' must be of type '{expected_type.__name__}'.")
            elif field == 'extensions':
                # Validate that extensions keys are properly namespaced
                for key in data[field].keys():
                    if '/' not in key and '_' not in key:
                        validation_messages.append \
                            (f"Extension key '{key}' in 'extensions' should be namespaced to prevent conflicts.")

    # If 'alternate_greetings' is present, check that it's a list of non-empty strings
    if 'alternate_greetings' in data and isinstance(data['alternate_greetings'], list):
        for idx, greeting in enumerate(data['alternate_greetings']):
            if not isinstance(greeting, str) or not greeting.strip():
                validation_messages.append(f"Element {idx} in 'alternate_greetings' must be a non-empty string.")

    # If 'tags' is present, check that it's a list of non-empty strings
    if 'tags' in data and isinstance(data['tags'], list):
        for idx, tag in enumerate(data['tags']):
            if not isinstance(tag, str) or not tag.strip():
                validation_messages.append(f"Element {idx} in 'tags' must be a non-empty string.")

    # Validate 'extensions' field
    if 'extensions' in data and not isinstance(data['extensions'], dict):
        validation_messages.append("Field 'extensions' must be a dictionary.")

    # Validate 'character_book' if present
    if 'character_book' in data:
        is_valid_book, book_messages = validate_character_book(data['character_book'])
        if not is_valid_book:
            validation_messages.extend(book_messages)

    is_valid = len(validation_messages) == 0
    return is_valid, validation_messages


def import_character_card_json(json_content: str) -> Optional[Dict[str, Any]]:
    """
    Import and parse a character card from JSON. If the card is in V2 format (i.e. it contains
    "spec": "chara_card_v2"), it is processed with parse_v2_card; otherwise, it is assumed to be a V1 card
    and converted via parse_v1_card.
    """
    try:
        # Remove any leading/trailing whitespace and log the preview.
        json_content = json_content.strip()
        logging.debug(f"\nJSON content (first 100 chars): {json_content[:100]}...")

        # Attempt to load the JSON.
        card_data = json.loads(json_content)
        logging.debug(f"\nParsed JSON data keys: {list(card_data.keys())[:100]}")

        # Check if it is a V2 card.
        if card_data.get('spec') == 'chara_card_v2':
            logging.info("Detected V2 character card")
            parsed_card = parse_v2_card(card_data)
            if parsed_card is None:
                logging.error("Failed to parse V2 character card")
            return parsed_card
        else:
            logging.info("Assuming V1 character card")
            return parse_v1_card(card_data)

    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        logging.error(f"Problematic JSON content (first 100 chars): {json_content[:100]}...")
    except Exception as e:
        logging.error(f"Unexpected error parsing JSON: {e}")
    return None


def load_character_card(file) -> Optional[Dict[str, Any]]:
    """
    Load a character card from a file that may be in Markdown or JSON format.

    The function supports:
      - Direct JSON files (content starts with '{')
      - Markdown files containing YAML front matter (delimited by '---')
      - Markdown files containing a code block with JSON (delimited by triple backticks)

    Args:
        file: A file-like object or a string containing either the character card data or a path to the file.

    Returns:
        A dictionary representing the character card, or None if an error occurs.

    Raises:
        ImportError: If YAML front matter is detected but PyYAML is not installed.
        ValueError: If no valid character card data is found.
    """
    log_counter("load_character_card_attempt")
    start_time = time.time()
    try:
        # Determine if 'file' is a file-like object, a file path, or already the content.
        if hasattr(file, 'read'):
            # file-like object (opened file, BytesIO, etc.)
            content = file.read()
            # Reset pointer if applicable
            if hasattr(file, 'seek'):
                file.seek(0)
        elif isinstance(file, str) and os.path.exists(file):
            # If 'file' is a string and it is a valid path, open and read it.
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            # Otherwise, assume the string is the actual content.
            content = file

        # If content is bytes, decode it.
        if isinstance(content, bytes):
            content = content.decode('utf-8')

        # Remove BOM (if present) and any leading whitespace/newlines.
        content = content.replace("\ufeff", "").lstrip()

        # Log a snippet of the content for debugging.
        logging.debug("\n\nFile content start: " + repr(content[:50]))

        # If the content is a JSON object.
        if content.startswith('{'):
            card_data = json.loads(content)
        # If the content starts with YAML front matter.
        elif content.startswith('---'):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML is required for loading YAML front matter from Markdown files. Install it via 'pip install PyYAML'."
                )
            # Use regex to match YAML front matter at the very start.
            yaml_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if yaml_match:
                yaml_content = yaml_match.group(1).strip()
                card_data = yaml.safe_load(yaml_content)
            else:
                raise ValueError("Invalid Markdown front matter format.")
        else:
            # Look for a JSON code block enclosed in triple backticks.
            pattern = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
            match = pattern.search(content)
            if match:
                json_str = match.group(1)
                card_data = json.loads(json_str)
            else:
                raise ValueError("No valid character card data found in the provided file.")

        load_duration = time.time() - start_time
        log_histogram("load_character_card_duration", load_duration)
        log_counter("load_character_card_success")
        return card_data

    except Exception as e:
        log_counter("load_character_card_error", labels={"error": str(e)})
        logging.error(f"Error loading character card: {e}")
        return None

# # Usage:
# with open('my_character_card.md', 'rb') as card_file:
#     card_data = load_character_card(card_file)
# # Validation
# if card_data:
#     is_valid, messages = validate_v2_card(card_data)
#     if is_valid:
#         print("Character card is valid!")
#     else:
#         print("Validation issues:", messages)
# else:
#     print("Failed to load character card.")

#
# End of File
####################################################################################################

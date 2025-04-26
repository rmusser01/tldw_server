# Character_Interaction_Library.py
# Description: Library for character card import functions
#
# Imports
import os
import time
from datetime import datetime
import re
import tempfile
import uuid
import json
import io
import base64
from typing import Dict, Any, Optional, List, Tuple
import zipfile
#
# External Imports
from PIL import Image
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.Character_Chat.Character_Chat_Lib import validate_character_book, validate_v2_card, \
    replace_placeholders, replace_user_placeholder, extract_json_from_image, load_chat_and_character, \
    load_chat_history, load_character_and_image, extract_character_id, load_character_wrapper, \
    load_character_card, parse_v2_card, parse_v1_card
from PoC_Version.App_Function_Libraries.Chat.Chat_Functions import chat, approximate_token_count
from PoC_Version.App_Function_Libraries.DB.Character_Chat_DB import (
    add_character_card,
    get_character_cards,
    get_character_card_by_id,
    add_character_chat,
    get_character_chats,
    get_character_chat_by_id,
    update_character_chat,
    delete_character_chat,
    delete_character_card,
    update_character_card, search_character_chats, save_chat_history_to_character_db,
)
from PoC_Version.App_Function_Libraries.TTS.TTS_Providers import generate_audio
from PoC_Version.App_Function_Libraries.Utils.Utils import sanitize_user_input, format_api_name, global_api_endpoints, \
    default_api_endpoint, load_and_log_configs, logging
#
#######################################################################################################################
#
# Functions:

########################################################
#
# Character card import functions:

def import_character_card(file) -> Any:
    if file is None:
        logging.error("No file provided for character card import.")
        return None, gr.update(), "No file provided for character card import"

    try:
        # If the file is an image (PNG or WEBP), extract JSON from the image.
        if file.name.lower().endswith(('.png', '.webp')):
            logging.debug("File is an image. Extracting JSON data from the image.")
            json_data = extract_json_from_image(file)
            if not json_data:
                logging.error("No character card data found in the image.")
                return None, gr.update(), (
                    "No character card data found in the image. "
                    "This might not be a valid character card image."
                )
            logging.debug("Parsing character card JSON extracted from image.")
            card_data = import_character_card_json(json_data)

        # If the file is JSON or Markdown, use the load_character_card helper.
        elif file.name.lower().endswith(('.json', '.md', '.markdown')):
            logging.debug("File is JSON or Markdown. Loading character card data.")
            card_data = load_character_card(file)
            if not card_data:
                logging.error("Failed to parse character card data from file.")
                return None, gr.update(), (
                    "Failed to parse character card data. "
                    "The file might not contain valid character information."
                )
            # If the returned data is a JSON string, parse it.
            if isinstance(card_data, str):
                logging.debug("Character card data is a string. Parsing as JSON.")
                card_data = import_character_card_json(card_data)
            else:
                logging.debug("Character card data is already a dictionary.")

        else:
            logging.error("Unsupported file type.")
            return None, gr.update(), (
                "Unsupported file type. Please upload a PNG/WebP image, a JSON file, or a Markdown file."
            )

        if not card_data:
            logging.error("Failed to parse character card data after processing.")
            return None, gr.update(), (
                "Failed to parse character card data. The file might not contain valid character information."
            )

        # If the file was an image, save the image data in the card data.
        if file.name.lower().endswith(('.png', '.webp')):
            logging.debug("Processing image to embed image data into card_data.")
            with Image.open(file) as img:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                card_data['image'] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Save the character card to the database.
        logging.debug("Saving character card to the database.")
        character_id = add_character_card(card_data)
        if character_id:
            logging.info(f"Character card '{card_data.get('name', 'Unknown')}' imported successfully with id: {character_id}.")
            characters = get_character_cards()
            character_names = [char['name'] for char in characters]
            return card_data, gr.update(choices=character_names), f"Character card '{card_data['name']}' imported successfully."
        else:
            logging.error(f"Failed to save character card '{card_data.get('name', 'Unknown')}'. It may already exist.")
            return None, gr.update(), f"Failed to save character card '{card_data.get('name', 'Unknown')}'. It may already exist."
    except Exception as e:
        logging.error(f"Error importing character card: {e}")
        return None, gr.update(), f"Error importing character card: {e}"


def import_character_card_json(json_content: str) -> Optional[Dict[str, Any]]:
    """
    Import and parse a character card from JSON. If the card is in V2 format (i.e. it contains
    "spec": "chara_card_v2"), it is processed with parse_v2_card; otherwise, it is assumed to be a V1 card
    and converted via parse_v1_card.
    """
    try:
        # Remove any leading/trailing whitespace and log the preview.
        json_content = json_content.strip()
        logging.debug(f"JSON content (first 100 chars): {json_content[:100]}...")

        # Attempt to load the JSON.
        card_data = json.loads(json_content)
        logging.debug(f"Parsed JSON data keys(first 100 chars): {list(card_data.keys())[:100]}")

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

#
# End of Character card import functions
####################################################

####################################################
#
# Character card export functions

def export_character_as_json(character_id):
    character = get_character_card_by_id(character_id)
    if character:
        # Remove the 'id' field from the character data
        character_data = {k: v for k, v in character.items() if k != 'id'}

        # Convert image to base64 if it exists
        if 'image' in character_data and character_data['image']:
            image_data = base64.b64decode(character_data['image'])
            img = Image.open(io.BytesIO(image_data))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            character_data['image'] = base64.b64encode(buffered.getvalue()).decode('utf-8')

        json_data = json.dumps(character_data, indent=2)
        return json_data
    return None

def export_all_characters_as_zip():
    characters = get_character_cards()
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.zip') as temp_zip:
        with zipfile.ZipFile(temp_zip, 'w') as zf:
            for character in characters:
                character_data = {k: v for k, v in character.items() if k != 'id'}

                # Convert image to base64 if it exists
                if 'image' in character_data and character_data['image']:
                    image_data = base64.b64decode(character_data['image'])
                    img = Image.open(io.BytesIO(image_data))
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    character_data['image'] = base64.b64encode(buffered.getvalue()).decode('utf-8')
                json_data = json.dumps(character_data, indent=2)
                zf.writestr(f"{character['name']}.json", json_data)
        return temp_zip.name

def export_single_character(character_selection):
    if not character_selection:
        return None, "No character selected."

    character_id = int(character_selection.split('(ID: ')[1].rstrip(')'))
    json_data = export_character_as_json(character_id)

    if json_data:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as temp_file:
            temp_file.write(json_data)
            return temp_file.name, f"Character '{character_selection.split(' (ID:')[0]}' exported successfully."
    else:
        return None, f"Failed to export character '{character_selection.split(' (ID:')[0]}'."

def export_all_characters():
    zip_path = export_all_characters_as_zip()
    return zip_path, "All characters exported successfully."

#
# End of Character card export functions
####################################################

####################################################
#
# Gradio tabs

def create_character_card_interaction_tab():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None
    with gr.TabItem("Chat with a Character Card", visible=True):
        with gr.Blocks() as interface:
            gr.Markdown("# Chat with a Character Card")
            with gr.Row():
                with gr.Column(scale=1):
                    # Checkbox to Decide Whether to Save Chats by Default
                    loaded_config = load_and_log_configs()
                    auto_save_value = loaded_config['auto-save']['save_character_chats'] or 'False'
                    auto_save_checkbox = gr.Checkbox(label="Save chats automatically", value=auto_save_value)
                    chat_media_name = gr.Textbox(label="Custom Chat Name (optional)", visible=True)
                    save_chat_history_to_db = gr.Button("Save Chat History to Database")
                    save_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=2):
                    gr.Markdown("## Search and Load Existing Chats")
                    chat_search_query = gr.Textbox(
                        label="Search Chats",
                        placeholder="Enter chat name or keywords to search"
                    )
                    chat_search_button = gr.Button("Search Chats")
                    chat_search_dropdown = gr.Dropdown(label="Search Results", choices=[], visible=False)
                    load_chat_button = gr.Button("Load Selected Chat", visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    character_image = gr.Image(label="Character Image", type="pil")
                    character_card_upload = gr.File(
                        label="Upload Character Card (PNG, WEBP, JSON)",
                        file_types=[".png", ".webp", ".json", ".md"]
                    )
                    import_card_button = gr.Button("Import Character Card")
                    load_characters_button = gr.Button("Load Existing Characters")
                    character_dropdown = gr.Dropdown(label="Select Character", choices=[])
                    user_name_input = gr.Textbox(label="Your Name", placeholder="Enter your name here")
                    # Refactored API selection dropdown
                    api_name_input = gr.Dropdown(
                        choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                        value=default_value,
                        label="API for Interaction (Mandatory)"
                    )
                    api_key_input = gr.Textbox(
                        label="API Key (if not set in Config_Files/config.txt)",
                        placeholder="Enter your API key here", type="password"
                    )
                    temperature_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, value=0.7, step=0.05, label="Temperature"
                    )
                    with gr.Row():
                        minp_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.00, step=0.01, label="Min-P"
                        )
                    with gr.Row():
                        maxp_slider = gr.Slider(
                            minimum=0.00, maximum=1.0, value=1.00, step=0.01, label="Top-P"
                        )
                    with gr.Row():
                        topk_slider = gr.Slider(
                            minimum=0, maximum=200, value=0, step=1, label="Top-K"
                        )
                    chat_file_upload = gr.File(label="Upload Chat History JSON", visible=True)
                    import_chat_button = gr.Button("Import Chat History")

                with gr.Column(scale=2):
                    chat_history = gr.Chatbot(label="Conversation", height=800)
                    with gr.Row():
                        streaming = gr.Checkbox(label="Enable streaming", value=True, interactive=True)
                    user_input = gr.Textbox(label="Your message")
                    send_message_button = gr.Button("Send Message")
                    with gr.Row():
                        speak_button = gr.Button("Speak Response")
                        tts_status = gr.Textbox(label="TTS Status", interactive=False)
                    with gr.Row():
                        audio_output = gr.Audio(label="Generated Audio", visible=False)
                    with gr.Row():
                        answer_for_me_button = gr.Button("Answer for Me")
                        continue_talking_button = gr.Button("Continue Talking")
                    regenerate_button = gr.Button("Regenerate Last Message")
                    with gr.Row():
                        token_count_display = gr.Number(label="Approximate Token Count", value=0, interactive=False)
                        clear_chat_button = gr.Button("Clear Chat")
                    save_snapshot_button = gr.Button("Save Chat Snapshot")
                    update_chat_dropdown = gr.Dropdown(label="Select Chat to Update", choices=[], visible=False)
                    load_selected_chat_button = gr.Button("Load Selected Chat", visible=False)
                    update_chat_button = gr.Button("Update Selected Chat", visible=False)

            # States
            character_data = gr.State(None)
            user_name = gr.State("")
            selected_chat_id = gr.State(None)  # To track the selected chat for updates

            # Callback Functions

            def search_existing_chats(query):
                logging.info(f"Searching for chats with query: {query}")
                results, message = search_character_chats(query)
                if results:
                    # Format search results for dropdown
                    formatted_results = [
                        f"{chat['conversation_name']} (ID: {chat['id']})" for chat in results
                    ]
                else:
                    formatted_results = []
                return formatted_results, message

            def load_selected_chat_from_search(selected_chat, user_name):
                logging.info(f"Loading selected chat: {selected_chat}")
                if not selected_chat:
                    return None, [], None, "No chat selected."

                try:
                    chat_id_match = re.search(r'\(ID:\s*(\d+)\)', selected_chat)
                    if not chat_id_match:
                        return None, [], None, "Invalid chat selection format."

                    chat_id = int(chat_id_match.group(1))

                    # Use the new function to load chat and character data
                    char_data, chat_history, img = load_chat_and_character(chat_id, user_name)

                    if not char_data:
                        return None, [], None, "Failed to load character data for the selected chat."

                    return char_data, chat_history, img, f"Chat '{selected_chat}' loaded successfully."
                except Exception as e:
                    logging.error(f"Error loading selected chat: {e}")
                    return None, [], None, f"Error loading chat: {e}"

            def import_chat_history(file, current_history, char_data, user_name_val):
                """
                Imports chat history from a file, replacing '{{user}}' with the actual user name.

                Args:
                    file (file): The uploaded chat history file.
                    current_history (list): The current chat history.
                    char_data (dict): The current character data.
                    user_name_val (str): The user's name.

                Returns:
                    tuple: Updated chat history, updated character data, and a status message.
                """
                logging.info(f"Importing chat history from file: {file.name}")
                loaded_history, char_name = load_chat_history(file)
                if loaded_history is None:
                    return current_history, char_data, "Failed to load chat history."

                # Replace '{{user}}' in the loaded chat history
                loaded_history = replace_user_placeholder(loaded_history, user_name_val)

                # Check if the loaded chat is for the current character
                if char_data and char_data.get('name') != char_name:
                    return current_history, char_data, (
                        f"Warning: Loaded chat is for character '{char_name}', "
                        f"but current character is '{char_data.get('name')}'. Chat not imported."
                    )

                # If no character is selected, try to load the character from the chat
                if not char_data:
                    characters = get_character_cards()
                    character = next((char for char in characters if char['name'] == char_name), None)
                    if character:
                        char_data = character
                        # Replace '{{user}}' in the first_message if necessary
                        if character.get('first_message'):
                            character['first_message'] = character['first_message'].replace("{{user}}",
                                                                                            user_name_val if user_name_val else "User")
                    else:
                        return current_history, char_data, (
                            f"Warning: Character '{char_name}' not found. Please select the character manually."
                        )

                return loaded_history, char_data, f"Chat history for '{char_name}' imported successfully."

            def character_chat_wrapper(
                    message, history, char_data, api_endpoint, api_key,
                    temperature, user_name_val, auto_save, streaming, minp, maxp, topk_slider
            ):
                if not char_data:
                    return history, "Please select a character first."

                # Sanitize the initial history to ensure no None values
                sanitized_history = []
                for entry in history:
                    if entry is None:
                        sanitized_history.append(("", ""))  # Replace None with an empty tuple
                    elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                        # Ensure both elements are strings
                        user_msg = entry[0] if entry[0] is not None else ""
                        bot_msg = entry[1] if entry[1] is not None else ""
                        sanitized_history.append((user_msg, bot_msg))
                    else:
                        # If the entry is invalid, replace it with an empty tuple
                        sanitized_history.append(("", ""))
                history = sanitized_history

                user_name_val = user_name_val or "User"
                char_name = char_data.get('name', 'AI Assistant')

                # Prepare the character's background information
                char_background = f"""
                Name: {char_name}
                Description: {char_data.get('description', 'N/A')}
                Personality: {char_data.get('personality', 'N/A')}
                Scenario: {char_data.get('scenario', 'N/A')}
                """

                # Prepare the system prompt
                system_message = f"""You are roleplaying as {char_name}. {char_data.get('system_prompt', '')}"""

                # Prepare chat context
                media_content = {
                    'id': char_name,
                    'title': char_name,
                    'content': char_background,
                    'description': char_data.get('description', ''),
                    'personality': char_data.get('personality', ''),
                    'scenario': char_data.get('scenario', '')
                }
                selected_parts = ['description', 'personality', 'scenario']

                prompt = char_data.get('post_history_instructions', '')

                # Sanitize and format user message
                user_message = sanitize_user_input(message)
                user_message = replace_placeholders(user_message, char_name, user_name_val)
                full_message = f"{user_name_val}: {user_message}"

                # Generate bot response
                logging.debug(f"Generating response; Calling chat function with message: {full_message}")
                bot_message = chat(
                    full_message,
                    history,
                    media_content,
                    selected_parts,
                    api_endpoint,
                    api_key,
                    prompt,
                    temperature,
                    system_message,
                    streaming,
                    minp=minp,
                    maxp=maxp,
                    topk=topk_slider
                )

                # Handle streaming response
                if streaming:
                    history.append((user_message, ""))  # Append user message with an empty bot response
                    full_response = ""
                    for chunk in bot_message:
                        full_response += chunk
                        history[-1] = (user_message, full_response)
                        yield history, ""  # Yield updated history and empty status

                    # After streaming is complete, handle auto-save
                    save_status = ""
                    if auto_save:
                        character_id = char_data.get('id')
                        if character_id:
                            conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            add_character_chat(character_id, conversation_name, history)
                            save_status = "Chat auto-saved."
                        else:
                            save_status = "Character ID not found; chat not saved."
                    yield history, save_status
                else:
                    # For non-streaming, append the full bot response to the history
                    bot_message = replace_placeholders(bot_message, char_name, user_name_val)
                    history.append((user_message, bot_message))
                    logging.debug(f"Updated history (non-streaming): {history}")

                    # Auto-save if enabled
                    save_status = ""
                    if auto_save:
                        character_id = char_data.get('id')
                        if character_id:
                            conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            add_character_chat(character_id, conversation_name, history)
                            save_status = "Chat auto-saved."
                        else:
                            save_status = "Character ID not found; chat not saved."

                    return history, save_status

            def validate_chat_history(chat_history: List[Tuple[Optional[str], str]]) -> bool:
                """
                Validate the chat history format and content.

                Args:
                    chat_history: List of message tuples (user_message, bot_message)

                Returns:
                    bool: True if valid, False if invalid
                """
                if not isinstance(chat_history, list):
                    return False

                for entry in chat_history:
                    if not isinstance(entry, tuple) or len(entry) != 2:
                        return False
                    # First element can be None (for system messages) or str
                    if not (entry[0] is None or isinstance(entry[0], str)):
                        return False
                    # Second element (bot response) must be str and not empty
                    if not isinstance(entry[1], str) or not entry[1].strip():
                        return False

                return True

            def sanitize_conversation_name(name: str) -> str:
                """
                Sanitize the conversation name.

                Args:
                    name: Raw conversation name

                Returns:
                    str: Sanitized conversation name
                """
                # Remove any non-alphanumeric characters except spaces and basic punctuation
                sanitized = re.sub(r'[^a-zA-Z0-9\s\-_.]', '', name)
                # Limit length
                sanitized = sanitized[:100]
                # Ensure it's not empty
                if not sanitized.strip():
                    sanitized = f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                return sanitized

            def save_chat_history_to_db_wrapper(
                    chat_history: List[Tuple[Optional[str], str]],
                    conversation_id: str,
                    media_content: Dict,
                    chat_media_name: str,
                    char_data: Dict,
                    auto_save: bool
            ) -> Tuple[str, str]:
                """
                Save chat history to the database with validation.

                Args:
                    chat_history: List of message tuples
                    conversation_id: Current conversation ID
                    media_content: Media content metadata
                    chat_media_name: Custom name for the chat
                    char_data: Character data dictionary
                    auto_save: Auto-save flag

                Returns:
                    Tuple[str, str]: (status message, detail message)
                """
                try:
                    # Basic input validation
                    if not chat_history:
                        return "No chat history to save.", ""

                    if not validate_chat_history(chat_history):
                        return "Invalid chat history format.", "Please ensure the chat history is valid."

                    if not char_data:
                        return "No character selected.", "Please select a character first."

                    character_id = char_data.get('id')
                    if not character_id:
                        return "Invalid character data: No character ID found.", ""

                    # Sanitize and prepare conversation name
                    conversation_name = sanitize_conversation_name(
                        chat_media_name if chat_media_name.strip()
                        else f"Chat with {char_data.get('name', 'Unknown')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

                    # Save to the database using your existing function
                    chat_id = save_chat_history_to_character_db(
                        character_id=character_id,
                        conversation_name=conversation_name,
                        chat_history=chat_history
                    )

                    if chat_id:
                        success_message = (
                            f"Chat saved successfully!\n"
                            f"ID: {chat_id}\n"
                            f"Name: {conversation_name}\n"
                            f"Messages: {len(chat_history)}"
                        )
                        return success_message, ""
                    else:
                        return "Failed to save chat to database.", "Database operation failed."

                except Exception as e:
                    logging.error(f"Error saving chat history: {str(e)}", exc_info=True)
                    return f"Error saving chat: {str(e)}", "Please check the logs for more details."

            def update_character_info(name):
                return load_character_and_image(name, user_name.value)

            def on_character_select(name, user_name_val):
                logging.debug(f"Character selected: {name}")
                char_data, chat_history, img = load_character_and_image(name, user_name_val)
                return char_data, chat_history, img

            def clear_chat_history(char_data, user_name_val):
                """
                Clears the chat history and initializes it with the character's first message,
                replacing the '{{user}}' placeholder with the actual user name.

                Args:
                    char_data (dict): The current character data.
                    user_name_val (str): The user's name.

                Returns:
                    tuple: Updated chat history and the unchanged char_data.
                """
                if char_data and 'first_message' in char_data and char_data['first_message']:
                    # Replace '{{user}}' in the first_message
                    first_message = char_data['first_message'].replace("{{user}}",
                                                                       user_name_val if user_name_val else "User")
                    # Initialize chat history with the updated first_message
                    return [(None, first_message)], char_data
                else:
                    # If no first_message is defined, simply clear the chat
                    return [], char_data

            def regenerate_last_message(
                    history, char_data, api_endpoint, api_key,
                    temperature, user_name_val, auto_save, streaming_checkbox, minp, maxp):
                """
                Regenerates the last bot message by removing it and resending the corresponding user message.

                Args:
                    history (list): The current chat history as a list of tuples (user_message, bot_message).
                    char_data (dict): The current character data.
                    api_endpoint (str): The API endpoint to use for the LLM.
                    api_key (str): The API key for authentication.
                    temperature (float): The temperature setting for the LLM.
                    user_name_val (str): The user's name.
                    auto_save (bool): Flag indicating whether to auto-save the chat.

                Returns:
                    tuple: Updated chat history and a save status message.
                """
                try:
                    streaming = streaming_checkbox.value if hasattr(streaming_checkbox, 'value') else streaming_checkbox
                    if not history:
                        return history, "No messages to regenerate."

                    last_entry = history[-1]
                    last_user_message, last_bot_message = last_entry

                    # Check if the last bot message exists
                    if last_bot_message is None:
                        return history, "The last message is not from the bot."

                    # Remove the last bot message
                    new_history = history[:-1]

                    # Resend the last user message to generate a new bot response
                    if not last_user_message:
                        return new_history, "No user message to regenerate the bot response."

                    # Prepare the character's background information
                    char_name = char_data.get('name', 'AI Assistant')
                    char_background = f"""
                    Name: {char_name}
                    Description: {char_data.get('description', 'N/A')}
                    Personality: {char_data.get('personality', 'N/A')}
                    Scenario: {char_data.get('scenario', 'N/A')}
                    """

                    # Prepare the system prompt for character impersonation
                    system_message = f"""You are roleplaying as {char_name}, the character described below. Respond to the user's messages in character, maintaining the personality and background provided. Do not break character or refer to yourself as an AI. Always refer to yourself as "{char_name}" and refer to the user as "{user_name_val}".
        
                    {char_background}
        
                    Additional instructions: {char_data.get('post_history_instructions', '')}
                    """

                    # Prepare media_content and selected_parts
                    media_content = {
                        'id': char_name,
                        'title': char_name,
                        'content': char_background,
                        'description': char_data.get('description', ''),
                        'personality': char_data.get('personality', ''),
                        'scenario': char_data.get('scenario', '')
                    }
                    selected_parts = ['description', 'personality', 'scenario']

                    prompt = char_data.get('post_history_instructions', '')

                    # Prepare the input for the chat function
                    full_message = f"{user_name_val}: {last_user_message}" if last_user_message else f"{user_name_val}: "

                    # Call the chat function to get a new bot message
                    bot_message = chat(
                        full_message,
                        new_history,
                        media_content,
                        selected_parts,
                        api_endpoint,
                        api_key,
                        prompt,
                        temperature,
                        system_message,
                        streaming,
                        minp=minp,
                        maxp=maxp,
                        topk=topk_slider
                    )

                    # Handle streaming response
                    if streaming:
                        full_response = ""
                        for chunk in bot_message:
                            full_response += chunk
                            yield new_history + [(last_user_message, full_response)], ""  # Yield updated history and empty status

                        # After streaming is complete, update history and handle auto-save
                        new_history.append((last_user_message, full_response))
                    else:
                        # Replace placeholders in bot message
                        full_response = replace_placeholders(bot_message, char_name, user_name_val)
                        # Update history
                        new_history.append((last_user_message, full_response))

                        # Auto-save if enabled
                        save_status = ""
                        if auto_save:
                            character_id = char_data.get('id')
                            if character_id:
                                conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                                add_character_chat(character_id, conversation_name, new_history)
                                save_status = "Chat auto-saved."
                            else:
                                save_status = "Character ID not found; chat not saved."

                        return new_history, save_status

                except Exception as e:
                    save_status = f"Error regenerating message: {str(e)}"
                    return history, save_status  # Return original history if an error occurs

            def save_untracked_chat_action(history, char_data):
                if not char_data or not history:
                    return "No chat to save or character not selected."

                character_id = char_data.get('id')
                if not character_id:
                    return "Character ID not found."

                conversation_name = f"Snapshot {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                chat_id = add_character_chat(character_id, conversation_name, history, is_snapshot=True)
                if chat_id:
                    return f"Chat snapshot saved successfully with ID {chat_id}."
                else:
                    return "Failed to save chat snapshot."

            def select_chat_for_update():
                # Fetch all chats for the selected character
                if character_data.value:
                    character_id = character_data.value.get('id')
                    if character_id:
                        chats = get_character_chats(character_id)
                        chat_choices = [
                            f"{chat['conversation_name']} (ID: {chat['id']})" for chat in chats
                        ]
                        return gr.update(choices=chat_choices), None
                return gr.update(choices=[]), "No character selected."

            def load_selected_chat(chat_selection):
                if not chat_selection:
                    return [], "No chat selected."

                try:
                    chat_id = int(chat_selection.split('(ID: ')[1].rstrip(')'))
                    chat = get_character_chat_by_id(chat_id)
                    if chat:
                        history = chat['chat_history']
                        selected_chat_id.value = chat_id  # Update the selected_chat_id state
                        return history, f"Loaded chat '{chat['conversation_name']}' successfully."
                    else:
                        return [], "Chat not found."
                except Exception as e:
                    logging.error(f"Error loading selected chat: {e}")
                    return [], f"Error loading chat: {e}"

            def update_chat(chat_id, updated_history):
                success = update_character_chat(chat_id, updated_history)
                if success:
                    return "Chat updated successfully."
                else:
                    return "Failed to update chat."

            def continue_talking(
                    history, char_data, api_endpoint, api_key,
                    temperature, user_name_val, auto_save, streaming_checkbox, minp, maxp
            ):
                """
                Causes the character to continue the conversation or think out loud.
                """
                streaming = streaming_checkbox.value if hasattr(streaming_checkbox, 'value') else streaming_checkbox
                if not char_data:
                    return history, "Please select a character first."

                user_name_val = user_name_val or "User"
                char_name = char_data.get('name', 'AI Assistant')

                # Prepare the character's background information
                char_background = f"""
                Name: {char_name}
                Description: {char_data.get('description', 'N/A')}
                Personality: {char_data.get('personality', 'N/A')}
                Scenario: {char_data.get('scenario', 'N/A')}
                """

                # Prepare the system prompt
                system_message = f"""You are roleplaying as {char_name}. {char_data.get('system_prompt', '')}
                If the user does not respond, continue expressing your thoughts or continue the conversation by thinking out loud. If thinking out loud, prefix the message with "Thinking: "."""

                # Prepare chat context
                media_content = {
                    'id': char_name,
                    'title': char_name,
                    'content': char_background,
                    'description': char_data.get('description', ''),
                    'personality': char_data.get('personality', ''),
                    'scenario': char_data.get('scenario', '')
                }
                selected_parts = ['description', 'personality', 'scenario']

                prompt = char_data.get('post_history_instructions', '')

                # Simulate empty user input
                user_message = ""

                # Generate bot response
                bot_message = chat(
                    user_message,
                    history,
                    media_content,
                    selected_parts,
                    api_endpoint,
                    api_key,
                    prompt,
                    temperature,
                    system_message,
                    streaming,
                    minp=minp,
                    maxp=maxp,
                    topk=topk_slider
                )

                # Handle streaming response
                if streaming:
                    history.append((None, ""))  # Append empty user message with an empty bot response
                    full_response = ""
                    for chunk in bot_message:
                        full_response += chunk
                        yield history + [(None, full_response)], ""  # Yield updated history and empty status

                    # After streaming is complete, update history and handle auto-save
                    history.append((None, full_response))
                else:
                    # Replace placeholders in bot message
                    full_response = replace_placeholders(bot_message, char_name, user_name_val)

                # Update history
                history.append((None, full_response))

                # Auto-save if enabled
                save_status = ""
                if auto_save:
                    character_id = char_data.get('id')
                    if character_id:
                        conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        add_character_chat(character_id, conversation_name, history)
                        save_status = "Chat auto-saved."
                    else:
                        save_status = "Character ID not found; chat not saved."

                return history, save_status

            def answer_for_me(
                    history, char_data, api_endpoint, api_key,
                    temperature, user_name_val, auto_save, streaming_checkbox, minp, maxp):
                """
                Generates a likely user response and continues the conversation.
                """
                streaming = streaming_checkbox.value if hasattr(streaming_checkbox, 'value') else streaming_checkbox
                if not char_data:
                    return history, "Please select a character first."

                user_name_val = user_name_val or "User"
                char_name = char_data.get('name', 'AI Assistant')

                # Prepare the character's background information
                char_background = f"""
                Name: {char_name}
                Description: {char_data.get('description', 'N/A')}
                Personality: {char_data.get('personality', 'N/A')}
                Scenario: {char_data.get('scenario', 'N/A')}
                """

                # Prepare system message for generating user's response
                system_message_user = f"""You are simulating the user {user_name_val}. Based on the conversation so far, generate a natural and appropriate response that {user_name_val} might say next. The response should fit the context and flow of the conversation. ONLY SPEAK FOR {user_name_val}."""

                # Prepare chat context
                media_content = {
                    'id': char_name,
                    'title': char_name,
                    'content': char_background,
                    'description': char_data.get('description', ''),
                    'personality': char_data.get('personality', ''),
                    'scenario': char_data.get('scenario', '')
                }
                selected_parts = ['description', 'personality', 'scenario']

                # Generate user response
                user_response = chat(
                    "",  # No new message
                    history,
                    media_content,
                    selected_parts,
                    api_endpoint,
                    api_key,
                    custom_prompt="",
                    temperature=temperature,
                    system_message=system_message_user,
                    streaming=streaming,
                    minp=minp,
                    maxp=maxp,
                    topk=topk_slider
                )

                if streaming:
                    history.append(("", ""))  # Append empty user message
                    full_user_response = ""
                    for chunk in user_response:
                        full_user_response += chunk
                        history[-1] = (full_user_response, "")
                        yield history, ""  # Yield updated history and empty status

                    # Now generate the character's response to this user response
                    system_message_bot = f"""You are roleplaying as {char_name}. {char_data.get('system_prompt', '')}"""

                    bot_message = chat(
                        f"{user_name_val}: {full_user_response}",
                        history[:-1],
                        media_content,
                        selected_parts,
                        api_endpoint,
                        api_key,
                        custom_prompt=char_data.get('post_history_instructions', ''),
                        temperature=temperature,
                        system_message=system_message_bot,
                        streaming=streaming,
                        minp=minp,
                        maxp=maxp,
                        topk=topk_slider
                    )

                    full_bot_response = ""
                    for chunk in bot_message:
                        full_bot_response += chunk
                        history[-1] = (full_user_response, full_bot_response)
                        yield history, ""  # Yield updated history and empty status

                    # After streaming is complete, handle auto-save
                    save_status = ""
                    if auto_save:
                        character_id = char_data.get('id')
                        if character_id:
                            conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            add_character_chat(character_id, conversation_name, history)
                            save_status = "Chat auto-saved."
                        else:
                            save_status = "Character ID not found; chat not saved."
                    yield history, save_status
                else:
                    # Append the generated user response to history
                    history.append((user_response, None))

                # Now generate the character's response to this user response
                # Prepare the system message for the character
                system_message_bot = f"""You are roleplaying as {char_name}. {char_data.get('system_prompt', '')}"""

                bot_message = chat(
                    f"{user_name_val}: {user_response}",
                    history[:-1],
                    media_content,
                    selected_parts,
                    api_endpoint,
                    api_key,
                    custom_prompt=char_data.get('post_history_instructions', ''),
                    temperature=temperature,
                    system_message=system_message_bot,
                    streaming=streaming,
                    minp=minp,
                    maxp=maxp,
                    topk=topk_slider
                )

                # Replace placeholders in bot message
                bot_message = replace_placeholders(bot_message, char_name, user_name_val)

                # Update history with bot's response
                history[-1] = (user_response, bot_message)

                # Auto-save if enabled
                save_status = ""
                if auto_save:
                    character_id = char_data.get('id')
                    if character_id:
                        conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        add_character_chat(character_id, conversation_name, history)
                        save_status = "Chat auto-saved."
                    else:
                        save_status = "Character ID not found; chat not saved."

                return history, save_status


            # Define States for conversation_id and media_content, which are required for saving chat history
            conversation_id = gr.State(str(uuid.uuid4()))
            media_content = gr.State({})
            # TTS Generation and Playback
            def speak_last_response(chat_history):
                """Generate audio for the last response and return the audio file"""
                logging.debug("Starting speak_last_response")
                try:
                    if not chat_history or len(chat_history) == 0:
                        return "No messages to speak", None

                    last_message = chat_history[-1][1]
                    logging.debug(f"Last message to speak: {last_message}")

                    # Generate audio file
                    audio_file = generate_audio(
                        text=last_message,
                        provider="openai",
                        output_file=f"response_{int(time.time())}.mp3",  # Unique filename
                        api_key=None
                    )

                    if audio_file and os.path.exists(audio_file):
                        return "Audio ready", audio_file
                    return "Audio generation failed", None

                except Exception as e:
                    logging.error(f"Error in speak_last_response: {str(e)}")
                    return f"Error: {str(e)}", None

            # Button Callbacks
            speak_button.click(
                fn=speak_last_response,
                inputs=[chat_history],
                outputs=[tts_status, audio_output],
                api_name="speak_response"
            ).then(
                lambda: gr.update(visible=True),
                outputs=audio_output
            )
            # Add the new button callbacks here
            answer_for_me_button.click(
                fn=answer_for_me,
                inputs=[
                    chat_history,
                    character_data,
                    api_name_input,
                    api_key_input,
                    temperature_slider,
                    user_name_input,
                    auto_save_checkbox,
                    streaming,
                    minp_slider,
                    maxp_slider,
                    topk_slider
                ],
                outputs=[chat_history, save_status]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chat_history],
                outputs=[token_count_display]
            )

            continue_talking_button.click(
                fn=continue_talking,
                inputs=[
                    chat_history,
                    character_data,
                    api_name_input,
                    api_key_input,
                    temperature_slider,
                    user_name_input,
                    auto_save_checkbox,
                    streaming,
                    minp_slider,
                    maxp_slider,
                    topk_slider
                ],
                outputs=[chat_history, save_status]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chat_history],
                outputs=[token_count_display]
            )

            import_card_button.click(
                fn=import_character_card,
                inputs=[character_card_upload],
                outputs=[character_data, character_dropdown, save_status]
            )

            load_characters_button.click(
                fn=lambda: gr.update(choices=[f"{char['name']} (ID: {char['id']})" for char in get_character_cards()]),
                outputs=character_dropdown
            )

            # FIXME user_name_val = validate_user_name(user_name_val)
            clear_chat_button.click(
                fn=clear_chat_history,
                inputs=[character_data, user_name_input],
                outputs=[chat_history, character_data]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chat_history],
                outputs=[token_count_display]
            )

            character_dropdown.change(
                fn=extract_character_id,
                inputs=[character_dropdown],
                outputs=character_data
            ).then(
                fn=load_character_wrapper,
                inputs=[character_data, user_name_input],
                outputs=[character_data, chat_history, character_image]
            )

            send_message_button.click(
                fn=character_chat_wrapper,
                inputs=[
                    user_input,
                    chat_history,
                    character_data,
                    api_name_input,
                    api_key_input,
                    temperature_slider,
                    user_name_input,
                    auto_save_checkbox,
                    streaming,
                    minp_slider,
                    maxp_slider,
                    topk_slider
                ],
                outputs=[chat_history, save_status],
            ).then(
                lambda: "", outputs=user_input  # Clear the input box after sending
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chat_history],
                outputs=[token_count_display]
            )

            regenerate_button.click(
                fn=regenerate_last_message,
                inputs=[
                    chat_history,
                    character_data,
                    api_name_input,
                    api_key_input,
                    temperature_slider,
                    user_name_input,
                    auto_save_checkbox,
                    streaming,
                    minp_slider,
                    maxp_slider,
                    topk_slider
                ],
                outputs=[chat_history, save_status]
            ).then(
                lambda history: approximate_token_count(history) if history is not None else 0,
                inputs=[chat_history],
                outputs=[token_count_display]
            )

            import_chat_button.click(
                fn=lambda: gr.update(visible=True),
                outputs=chat_file_upload
            )

            chat_file_upload.change(
                fn=import_chat_history,
                inputs=[chat_file_upload, chat_history, character_data, user_name_input],
                outputs=[chat_history, character_data, save_status]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chat_history],
                outputs=[token_count_display]
            )

            save_chat_history_to_db.click(
                fn=save_chat_history_to_db_wrapper,
                inputs=[
                    chat_history,
                    conversation_id,
                    media_content,
                    chat_media_name,
                    character_data,
                    auto_save_checkbox  # Pass the auto_save state
                ],
                outputs=[conversation_id, save_status]
            )

            # Populate the update_chat_dropdown based on selected character
            character_dropdown.change(
                fn=select_chat_for_update,
                inputs=[],
                outputs=[update_chat_dropdown, save_status]
            )

            load_selected_chat_button.click(
                fn=load_selected_chat,
                inputs=[update_chat_dropdown],
                outputs=[chat_history, save_status]
            )

            save_snapshot_button.click(
                fn=save_untracked_chat_action,
                inputs=[chat_history, character_data],
                outputs=save_status
            )

            update_chat_button.click(
                fn=update_chat,
                inputs=[selected_chat_id, chat_history],
                outputs=save_status
            )

            # Search Chats
            chat_search_button.click(
                fn=search_existing_chats,
                inputs=[chat_search_query],
                outputs=[chat_search_dropdown, save_status]
            ).then(
                fn=lambda choices, msg: gr.update(choices=choices, visible=True) if choices else gr.update(visible=False),
                inputs=[chat_search_dropdown, save_status],
                outputs=[chat_search_dropdown]
            )

            # Load Selected Chat from Search
            load_chat_button.click(
                fn=load_selected_chat_from_search,
                inputs=[chat_search_dropdown, user_name_input],
                outputs=[character_data, chat_history, character_image, save_status]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chat_history],
                outputs=[token_count_display]
            )

            # Show Load Chat Button when a chat is selected
            chat_search_dropdown.change(
                fn=lambda selected: gr.update(visible=True) if selected else gr.update(visible=False),
                inputs=[chat_search_dropdown],
                outputs=[load_chat_button]
            )

            return character_data, chat_history, user_input, user_name, character_image


def create_character_chat_mgmt_tab():
    with gr.TabItem("Character Chat Management", visible=True):
        gr.Markdown("# Character Chat Management")

        with gr.Row():
            # Left Column: Character Import and Chat Management
            with gr.Column(scale=1):
                gr.Markdown("## Import Characters")
                character_files = gr.File(
                    label="Upload Character Files (PNG, WEBP, JSON)",
                    file_types=[".png", ".webp", ".json", ".md"],
                    file_count="multiple"
                )
                import_characters_button = gr.Button("Import Characters")
                import_status = gr.Markdown("")

            # Right Column: Character Selection and Image Display
            with gr.Column(scale=2):
                gr.Markdown("## Select Character")
                characters = get_character_cards()
                character_choices = [f"{char['name']} (ID: {char['id']})" for char in characters]
                load_characters_button = gr.Button("Load Existing Characters")
                select_character = gr.Dropdown(label="Select Character", choices=character_choices, interactive=True)
                character_image = gr.Image(label="Character Image", type="pil", interactive=False)

                gr.Markdown("## Search Conversations")
                search_query = gr.Textbox(label="Search Conversations", placeholder="Enter search keywords")
                search_button = gr.Button("Search")
                search_results = gr.Dropdown(label="Search Results", choices=[], visible=False)
                search_status = gr.Markdown("", visible=True)

        with gr.Row():
            gr.Markdown("## Chat Management")
            select_chat = gr.Dropdown(label="Select Chat", choices=[], visible=False, interactive=True)
            load_chat_button = gr.Button("Load Selected Chat", visible=False)
            conversation_list = gr.Dropdown(label="Select Conversation", choices=[])
            conversation_mapping = gr.State({})

        with gr.Tabs():
            with gr.TabItem("Edit", visible=True):
                chat_content = gr.TextArea(label="Chat/Character Content (JSON)", lines=20, max_lines=50)
                save_button = gr.Button("Save Changes")
                export_chat_button = gr.Button("Export Current Conversation", variant="secondary")
                export_all_chats_button = gr.Button("Export All Character Conversations", variant="secondary")
                export_file = gr.File(label="Downloaded File", visible=False)
                export_status = gr.Markdown("")
                delete_button = gr.Button("Delete Conversation/Character", variant="stop")

            with gr.TabItem("Preview", visible=True):
                chat_preview = gr.HTML(label="Chat/Character Preview")
        result_message = gr.Markdown("")

        # Callback Functions

        def load_character_image(character_selection):
            if not character_selection:
                return None

            try:
                character_id = int(character_selection.split('(ID: ')[1].rstrip(')'))
                character = get_character_card_by_id(character_id)
                if character and 'image' in character:
                    image_data = base64.b64decode(character['image'])
                    img = Image.open(io.BytesIO(image_data))
                    return img
            except Exception as e:
                logging.error(f"Error loading character image: {e}")

            return None

        def search_conversations_or_characters(query, selected_character):
            if not query.strip():
                return gr.update(choices=[], visible=False), "Please enter a search query."

            try:
                # Extract character ID from the selected character
                character_id = None
                if selected_character:
                    character_id = int(selected_character.split('(ID: ')[1].rstrip(')'))

                # Search Chats using FTS5, filtered by character_id if provided
                chat_results, chat_message = search_character_chats(query, character_id)

                # Format chat results
                formatted_chat_results = [
                    f"Chat: {chat['conversation_name']} (ID: {chat['id']})" for chat in chat_results
                ]

                # If no character is selected, also search for characters
                if not character_id:
                    characters = get_character_cards()
                    filtered_characters = [
                        char for char in characters
                        if query.lower() in char['name'].lower()
                    ]
                    formatted_character_results = [
                        f"Character: {char['name']} (ID: {char['id']})" for char in filtered_characters
                    ]
                else:
                    formatted_character_results = []

                # Combine results
                all_choices = formatted_chat_results + formatted_character_results

                if all_choices:
                    return gr.update(choices=all_choices, visible=True), chat_message
                else:
                    return gr.update(choices=[], visible=False), f"No results found for '{query}'."

            except Exception as e:
                logging.error(f"Error during search: {e}")
                return gr.update(choices=[], visible=False), f"Error occurred during search: {e}"

        def load_conversation_or_character(selected, conversation_mapping):
            if not selected or selected not in conversation_mapping:
                return "", "<p>No selection made.</p>"

            selected_id = conversation_mapping[selected]
            if selected.startswith("Chat:"):
                chat = get_character_chat_by_id(selected_id)
                if chat:
                    json_content = json.dumps({
                        "conversation_id": chat['id'],
                        "conversation_name": chat['conversation_name'],
                        "messages": chat['chat_history']
                    }, indent=2)

                    html_preview = create_chat_preview_html(chat['chat_history'])
                    return json_content, html_preview
            elif selected.startswith("Character:"):
                character = get_character_card_by_id(selected_id)
                if character:
                    json_content = json.dumps({
                        "id": character['id'],
                        "name": character['name'],
                        "description": character['description'],
                        "personality": character['personality'],
                        "scenario": character['scenario'],
                        "post_history_instructions": character['post_history_instructions'],
                        "first_mes": character['first_mes'],
                        "mes_example": character['mes_example'],
                        "creator_notes": character.get('creator_notes', ''),
                        "system_prompt": character.get('system_prompt', ''),
                        "tags": character.get('tags', []),
                        "creator": character.get('creator', ''),
                        "character_version": character.get('character_version', ''),
                        "extensions": character.get('extensions', {})
                    }, indent=2)

                    html_preview = create_character_preview_html(character)
                    return json_content, html_preview

            return "", "<p>Unable to load the selected item.</p>"

        def validate_content(selected, content):
            try:
                data = json.loads(content)
                if selected.startswith("Chat:"):
                    assert "conversation_id" in data and "messages" in data
                elif selected.startswith("Character:"):
                    assert "id" in data and "name" in data
                return True, data
            except Exception as e:
                return False, f"Invalid JSON: {e}"

        def save_conversation_or_character(selected, conversation_mapping, content):
            if not selected or selected not in conversation_mapping:
                return "Please select an item to save.", "<p>No changes made.</p>"

            is_valid, result = validate_content(selected, content)
            if not is_valid:
                return f"Error: {result}", "<p>No changes made due to validation error.</p>"

            selected_id = conversation_mapping[selected]

            if selected.startswith("Chat:"):
                success = update_character_chat(selected_id, result['messages'])
                return ("Chat updated successfully." if success else "Failed to update chat."), ("<p>Chat updated.</p>" if success else "<p>Failed to update chat.</p>")
            elif selected.startswith("Character:"):
                success = update_character_card(selected_id, result)
                return ("Character updated successfully." if success else "Failed to update character."), ("<p>Character updated.</p>" if success else "<p>Failed to update character.</p>")

            return "Unknown item type.", "<p>No changes made.</p>"

        def delete_conversation_or_character(selected, conversation_mapping):
            if not selected or selected not in conversation_mapping:
                return "Please select an item to delete.", "<p>No changes made.</p>", gr.update(choices=[])

            selected_id = conversation_mapping[selected]

            if selected.startswith("Chat:"):
                success = delete_character_chat(selected_id)
            elif selected.startswith("Character:"):
                success = delete_character_card(selected_id)
            else:
                return "Unknown item type.", "<p>No changes made.</p>", gr.update()

            if success:
                updated_choices = [choice for choice in conversation_mapping.keys() if choice != selected]
                conversation_mapping.value.pop(selected, None)
                return f"{selected.split(':')[0]} deleted successfully.", f"<p>{selected.split(':')[0]} deleted.</p>", gr.update(choices=updated_choices)
            else:
                return f"Failed to delete {selected.split(':')[0].lower()}.", f"<p>Failed to delete {selected.split(':')[0].lower()}.</p>", gr.update()

        def populate_chats(character_selection):
            if not character_selection:
                return gr.update(choices=[], visible=False), "Please select a character first."

            try:
                character_id = int(character_selection.split('(ID: ')[1].rstrip(')'))
                chats = get_character_chats(character_id=character_id)

                if not chats:
                    return gr.update(choices=[], visible=False), f"No chats found for the selected character."

                formatted_chats = [f"{chat['conversation_name']} (ID: {chat['id']})" for chat in chats]
                return gr.update(choices=formatted_chats, visible=True), f"Found {len(formatted_chats)} chat(s)."
            except Exception as e:
                logging.error(f"Error populating chats: {e}")
                return gr.update(choices=[], visible=False), f"Error occurred: {e}"

        def load_chat_from_character(selected_chat):
            if not selected_chat:
                return "", "<p>No chat selected.</p>"

            try:
                chat_id = int(selected_chat.split('(ID: ')[1].rstrip(')'))
                chat = get_character_chat_by_id(chat_id)
                if not chat:
                    return "", "<p>Selected chat not found.</p>"

                json_content = json.dumps({
                    "conversation_id": chat['id'],
                    "conversation_name": chat['conversation_name'],
                    "messages": chat['chat_history']
                }, indent=2)

                html_preview = create_chat_preview_html(chat['chat_history'])
                return json_content, html_preview
            except Exception as e:
                logging.error(f"Error loading chat: {e}")
                return "", f"<p>Error loading chat: {e}</p>"

        def create_chat_preview_html(chat_history):
            html_preview = "<div style='max-height: 500px; overflow-y: auto;'>"
            for user_msg, bot_msg in chat_history:
                user_style = "background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 5px;"
                bot_style = "background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;"
                html_preview += f"<div style='{user_style}'><strong>User:</strong> {user_msg}</div>"
                html_preview += f"<div style='{bot_style}'><strong>Bot:</strong> {bot_msg}</div>"
            html_preview += "</div>"
            return html_preview

        def create_character_preview_html(character):
            return f"""
            <div>
                <h2>{character['name']}</h2>
                <p><strong>Description:</strong> {character['description']}</p>
                <p><strong>Personality:</strong> {character['personality']}</p>
                <p><strong>Scenario:</strong> {character['scenario']}</p>
                <p><strong>First Message:</strong> {character['first_mes']}</p>
                <p><strong>Example Message:</strong> {character['mes_example']}</p>
                <p><strong>Post History Instructions:</strong> {character['post_history_instructions']}</p>
                <p><strong>System Prompt:</strong> {character.get('system_prompt', 'N/A')}</p>
                <p><strong>Tags:</strong> {', '.join(character.get('tags', []))}</p>
                <p><strong>Creator:</strong> {character.get('creator', 'N/A')}</p>
                <p><strong>Version:</strong> {character.get('character_version', 'N/A')}</p>
            </div>
            """

        def import_multiple_characters(files):
            if not files:
                return "No files provided for character import."

            results = []
            for file in files:
                result, _, message = import_character_card(file)
                if result:
                    results.append(f"Imported: {result['name']}")
                else:
                    results.append(f"Failed: {file.name} - {message}")

            # Refresh character choices
            characters = get_character_cards()
            character_choices = [f"{char['name']} (ID: {char['id']})" for char in characters]
            select_character.choices = character_choices

            return "Import results:\n" + "\n".join(results)

        def export_current_conversation(selected_chat):
            if not selected_chat:
                return "Please select a conversation to export.", None

            try:
                chat_id = int(selected_chat.split('(ID: ')[1].rstrip(')'))
                chat = get_character_chat_by_id(chat_id)

                if not chat:
                    return "Selected chat not found.", None

                # Ensure chat_history is properly parsed
                chat_history = chat['chat_history']
                if isinstance(chat_history, str):
                    chat_history = json.loads(chat_history)

                export_data = {
                    "conversation_id": chat['id'],
                    "conversation_name": chat['conversation_name'],
                    "character_id": chat['character_id'],
                    "chat_history": chat_history,
                    "exported_at": datetime.now().isoformat()
                }

                # Convert to JSON string
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

                # Create file name
                file_name = f"conversation_{chat['id']}_{chat['conversation_name']}.json"

                # Return file for download
                return "Conversation exported successfully!", (file_name, json_str, "application/json")

            except Exception as e:
                logging.error(f"Error exporting conversation: {e}")
                return f"Error exporting conversation: {str(e)}", None

        def export_all_character_conversations(character_selection):
            if not character_selection:
                return "Please select a character first.", None

            try:
                character_id = int(character_selection.split('(ID: ')[1].rstrip(')'))
                character = get_character_card_by_id(character_id)
                chats = get_character_chats(character_id=character_id)

                if not chats:
                    return "No conversations found for this character.", None

                # Process chat histories
                conversations = []
                for chat in chats:
                    chat_history = chat['chat_history']
                    if isinstance(chat_history, str):
                        chat_history = json.loads(chat_history)

                    conversations.append({
                        "conversation_id": chat['id'],
                        "conversation_name": chat['conversation_name'],
                        "chat_history": chat_history
                    })

                export_data = {
                    "character": {
                        "id": character['id'],
                        "name": character['name']
                    },
                    "conversations": conversations,
                    "exported_at": datetime.now().isoformat()
                }

                # Convert to JSON string
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

                # Create file name
                file_name = f"all_conversations_{character['name']}_{character['id']}.json"

                # Return file for download
                return "All conversations exported successfully!", (file_name, json_str, "application/json")

            except Exception as e:
                logging.error(f"Error exporting all conversations: {e}")
                return f"Error exporting conversations: {str(e)}", None

        # Register new callback for character import
        import_characters_button.click(
            fn=import_multiple_characters,
            inputs=[character_files],
            outputs=[import_status]
        ).then(
            fn=lambda: gr.update(choices=[f"{char['name']} (ID: {char['id']})" for char in get_character_cards()]),
            outputs=select_character
        )

        # Register Callback Functions with Gradio Components
        search_button.click(
            fn=search_conversations_or_characters,
            inputs=[search_query, select_character],
            outputs=[search_results, search_status]
        )

        search_results.change(
            fn=load_conversation_or_character,
            inputs=[search_results, conversation_mapping],
            outputs=[chat_content, chat_preview]
        )

        save_button.click(
            fn=save_conversation_or_character,
            inputs=[conversation_list, conversation_mapping, chat_content],
            outputs=[result_message, chat_preview]
        )

        delete_button.click(
            fn=delete_conversation_or_character,
            inputs=[conversation_list, conversation_mapping],
            outputs=[result_message, chat_preview, conversation_list]
        )

        select_character.change(
            fn=load_character_image,
            inputs=[select_character],
            outputs=[character_image]
        ).then(
            fn=populate_chats,
            inputs=[select_character],
            outputs=[select_chat, search_status]
        )

        select_chat.change(
            fn=load_chat_from_character,
            inputs=[select_chat],
            outputs=[chat_content, chat_preview]
        )

        load_chat_button.click(
            fn=load_chat_from_character,
            inputs=[select_chat],
            outputs=[chat_content, chat_preview]
        )

        load_characters_button.click(
            fn=lambda: gr.update(choices=[f"{char['name']} (ID: {char['id']})" for char in get_character_cards()]),
            outputs=select_character
            ).then(
                lambda choices: print(f"Dropdown choices: {choices}"),  # Debugging line
                inputs=[select_character],
                outputs=None
            )

        export_chat_button.click(
            fn=export_current_conversation,
            inputs=[select_chat],
            outputs=[export_status, export_file]
        )

        export_all_chats_button.click(
            fn=export_all_character_conversations,
            inputs=[select_character],
            outputs=[export_status, export_file]
        )

        return (
            character_files, import_characters_button, import_status,
            search_query, search_button, search_results, search_status,
            select_character, select_chat, load_chat_button,
            conversation_list, conversation_mapping,
            chat_content, save_button, delete_button,
            chat_preview, result_message, character_image
        )

def create_custom_character_card_tab():
    with gr.TabItem("Create a New Character Card", visible=True):
        gr.Markdown("# Create a New Character Card (v2)")

        with gr.Row():
            with gr.Column():
                # Input fields for character card data
                name_input = gr.Textbox(label="Name", placeholder="Enter character name")
                description_input = gr.TextArea(label="Description", placeholder="Enter character description")
                personality_input = gr.TextArea(label="Personality", placeholder="Enter character personality")
                scenario_input = gr.TextArea(label="Scenario", placeholder="Enter character scenario")
                first_mes_input = gr.TextArea(label="First Message", placeholder="Enter the first message")
                mes_example_input = gr.TextArea(label="Example Messages", placeholder="Enter example messages")
                creator_notes_input = gr.TextArea(label="Creator Notes", placeholder="Enter notes for the creator")
                system_prompt_input = gr.TextArea(label="System Prompt", placeholder="Enter system prompt")
                post_history_instructions_input = gr.TextArea(label="Post History Instructions", placeholder="Enter post history instructions")
                alternate_greetings_input = gr.TextArea(
                    label="Alternate Greetings (one per line)",
                    placeholder="Enter alternate greetings, one per line"
                )
                tags_input = gr.Textbox(label="Tags", placeholder="Enter tags, separated by commas")
                creator_input = gr.Textbox(label="Creator", placeholder="Enter creator name")
                character_version_input = gr.Textbox(label="Character Version", placeholder="Enter character version")
                extensions_input = gr.TextArea(
                    label="Extensions (JSON)",
                    placeholder="Enter extensions as JSON (optional)"
                )
                image_input = gr.Image(label="Character Image", type="pil")

                # Buttons
                save_button = gr.Button("Save Character Card")
                download_button = gr.Button("Download Character Card")
                download_image_button = gr.Button("Download Character Card as Image")

                # Output status and outputs
                save_status = gr.Markdown("")
                download_output = gr.File(label="Download Character Card", interactive=False)
                download_image_output = gr.File(label="Download Character Card as Image", interactive=False)

        # Import PngInfo
        from PIL.PngImagePlugin import PngInfo

        # Callback Functions
        def build_character_card(
            name, description, personality, scenario, first_mes, mes_example,
            creator_notes, system_prompt, post_history_instructions,
            alternate_greetings_str, tags_str, creator, character_version,
            extensions_str
        ):
            # Parse alternate_greetings from multiline string
            alternate_greetings = [line.strip() for line in alternate_greetings_str.strip().split('\n') if line.strip()]

            # Parse tags from comma-separated string
            tags = [tag.strip() for tag in tags_str.strip().split(',') if tag.strip()]

            # Parse extensions from JSON string
            try:
                extensions = json.loads(extensions_str) if extensions_str.strip() else {}
            except json.JSONDecodeError as e:
                extensions = {}
                logging.error(f"Error parsing extensions JSON: {e}")

            # Build the character card dictionary according to V2 spec
            character_card = {
                'spec': 'chara_card_v2',
                'spec_version': '2.0',
                'data': {
                    'name': name,
                    'description': description,
                    'personality': personality,
                    'scenario': scenario,
                    'first_mes': first_mes,
                    'mes_example': mes_example,
                    'creator_notes': creator_notes,
                    'system_prompt': system_prompt,
                    'post_history_instructions': post_history_instructions,
                    'alternate_greetings': alternate_greetings,
                    'tags': tags,
                    'creator': creator,
                    'character_version': character_version,
                    'extensions': extensions,
                }
            }
            return character_card

        def validate_character_card_data(character_card):
            """
            Validates the character card data using the extended validation logic.
            """
            is_valid, validation_messages = validate_v2_card(character_card)
            return is_valid, validation_messages

        def save_character_card(
            name, description, personality, scenario, first_mes, mes_example,
            creator_notes, system_prompt, post_history_instructions,
            alternate_greetings_str, tags_str, creator, character_version,
            extensions_str, image
        ):
            # Build the character card
            character_card = build_character_card(
                name, description, personality, scenario, first_mes, mes_example,
                creator_notes, system_prompt, post_history_instructions,
                alternate_greetings_str, tags_str, creator, character_version,
                extensions_str
            )

            # Validate the character card
            is_valid, validation_messages = validate_character_card_data(character_card)
            if not is_valid:
                # Return validation errors
                validation_output = "Character card validation failed:\n"
                validation_output += "\n".join(validation_messages)
                return validation_output

            # If image is provided, encode it to base64
            if image:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                character_card['data']['image'] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            # Save character card to database
            character_id = add_character_card(character_card['data'])
            if character_id:
                return f"Character card '{name}' saved successfully."
            else:
                return f"Failed to save character card '{name}'. It may already exist."

        def download_character_card(
            name, description, personality, scenario, first_mes, mes_example,
            creator_notes, system_prompt, post_history_instructions,
            alternate_greetings_str, tags_str, creator, character_version,
            extensions_str, image
        ):
            # Build the character card
            character_card = build_character_card(
                name, description, personality, scenario, first_mes, mes_example,
                creator_notes, system_prompt, post_history_instructions,
                alternate_greetings_str, tags_str, creator, character_version,
                extensions_str
            )

            # Validate the character card
            is_valid, validation_messages = validate_character_card_data(character_card)
            if not is_valid:
                # Return validation errors
                validation_output = "Character card validation failed:\n"
                validation_output += "\n".join(validation_messages)
                return gr.update(value=None), validation_output  # Return None for the file output

            # If image is provided, include it as base64
            if image:
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                character_card['data']['image'] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

            # Convert to JSON string
            json_str = json.dumps(character_card, indent=2)

            # Write the JSON to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as temp_file:
                temp_file.write(json_str)
                temp_file_path = temp_file.name

            # Return the file path and clear validation output
            return temp_file_path, ""

        def download_character_card_as_image(
            name, description, personality, scenario, first_mes, mes_example,
            creator_notes, system_prompt, post_history_instructions,
            alternate_greetings_str, tags_str, creator, character_version,
            extensions_str, image
        ):
            # Build the character card
            character_card = build_character_card(
                name, description, personality, scenario, first_mes, mes_example,
                creator_notes, system_prompt, post_history_instructions,
                alternate_greetings_str, tags_str, creator, character_version,
                extensions_str
            )

            # Validate the character card
            is_valid, validation_messages = validate_character_card_data(character_card)
            if not is_valid:
                # Return validation errors
                validation_output = "Character card validation failed:\n"
                validation_output += "\n".join(validation_messages)
                return gr.update(value=None), validation_output  # Return None for the file output

            # Convert the character card JSON to a string
            json_str = json.dumps(character_card, indent=2)

            # Encode the JSON string to base64
            chara_content = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')

            # Create PNGInfo object to hold metadata
            png_info = PngInfo()
            png_info.add_text('chara', chara_content)

            # If image is provided, use it; otherwise, create a blank image
            if image:
                img = image.copy()
            else:
                # Create a default blank image
                img = Image.new('RGB', (512, 512), color='white')

            # Save the image to a temporary file with metadata
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.png') as temp_file:
                img.save(temp_file, format='PNG', pnginfo=png_info)
                temp_file_path = temp_file.name

            # Return the file path and clear validation output
            return temp_file_path, ""

        # Include the validate_v2_card function here (from previous code)

        # Button Callbacks
        save_button.click(
            fn=save_character_card,
            inputs=[
                name_input, description_input, personality_input, scenario_input,
                first_mes_input, mes_example_input, creator_notes_input, system_prompt_input,
                post_history_instructions_input, alternate_greetings_input, tags_input,
                creator_input, character_version_input, extensions_input, image_input
            ],
            outputs=[save_status]
        )

        download_button.click(
            fn=download_character_card,
            inputs=[
                name_input, description_input, personality_input, scenario_input,
                first_mes_input, mes_example_input, creator_notes_input, system_prompt_input,
                post_history_instructions_input, alternate_greetings_input, tags_input,
                creator_input, character_version_input, extensions_input, image_input
            ],
            outputs=[download_output, save_status]
        )

        download_image_button.click(
            fn=download_character_card_as_image,
            inputs=[
                name_input, description_input, personality_input, scenario_input,
                first_mes_input, mes_example_input, creator_notes_input, system_prompt_input,
                post_history_instructions_input, alternate_greetings_input, tags_input,
                creator_input, character_version_input, extensions_input, image_input
            ],
            outputs=[download_image_output, save_status]
        )


def create_character_card_validation_tab():
    with gr.TabItem("Validate Character Card", visible=True):
        gr.Markdown("# Validate Character Card (v2)")
        gr.Markdown("Upload a character card (PNG, WEBP, or JSON) to validate whether it conforms to the Character Card V2 specification.")

        with gr.Row():
            with gr.Column():
                # File uploader
                file_upload = gr.File(
                    label="Upload Character Card (PNG, WEBP, JSON)",
                    file_types=[".png", ".webp", ".json", ".md"]
                )
                # Validation button
                validate_button = gr.Button("Validate Character Card")
                # Output area for validation results
                validation_output = gr.Markdown("")

        # Callback Functions
        def validate_character_card(file):
            if file is None:
                return "No file provided for validation."

            try:
                if file.name.lower().endswith(('.png', '.webp')):
                    json_data = extract_json_from_image(file)
                    if not json_data:
                        return "Failed to extract JSON data from the image. The image might not contain embedded character card data."
                elif file.name.lower().endswith('.json'):
                    with open(file.name, 'r', encoding='utf-8') as f:
                        json_data = f.read()
                else:
                    return "Unsupported file type. Please upload a PNG, WEBP, or JSON file."

                # Parse the JSON content
                try:
                    card_data = json.loads(json_data)
                except json.JSONDecodeError as e:
                    return f"JSON decoding error: {e}"

                # Validate the character card
                is_valid, validation_messages = validate_v2_card(card_data)

                # Prepare the validation output
                if is_valid:
                    return "Character card is valid according to the V2 specification."
                else:
                    # Concatenate all validation error messages
                    validation_output = "Character card validation failed:\n"
                    validation_output += "\n".join(validation_messages)
                    return validation_output

            except Exception as e:
                logging.error(f"Error validating character card: {e}")
                return f"An unexpected error occurred during validation: {e}"

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
                        validation_messages.append(f"'spec_version' must be '2.0' or higher. Found '{card_data['spec_version']}'.")
                except ValueError:
                    validation_messages.append(f"Invalid 'spec_version' format: {card_data['spec_version']}. Must be a number as a string.")

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
                                validation_messages.append(f"Extension key '{key}' in 'extensions' should be namespaced to prevent conflicts.")

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

        # Button Callback
        validate_button.click(
            fn=validate_character_card,
            inputs=[file_upload],
            outputs=[validation_output]
        )


def create_export_characters_tab():
    with gr.TabItem("Export Characters", visible=True):
        gr.Markdown("# Export Characters")
        gr.Markdown("Export character cards individually as JSON files or all together as a ZIP file.")

        with gr.Row():
            with gr.Column(scale=1):
                # Dropdown to select a character for individual export
                characters = get_character_cards()
                character_choices = [f"{char['name']} (ID: {char['id']})" for char in characters]
                export_character_dropdown = gr.Dropdown(
                    label="Select Character to Export",
                    choices=character_choices
                )
                load_characters_button = gr.Button("Load Existing Characters")
                export_single_button = gr.Button("Export Selected Character")
                export_all_button = gr.Button("Export All Characters")

            with gr.Column(scale=1):
                # Output components
                export_output = gr.File(label="Exported Character(s)", interactive=False)
                export_status = gr.Markdown("")

# FIXME
        def export_single_character_wrapper(character_selection):
            file_path, status_message = export_single_character(character_selection)
            if file_path:
                return gr.update(value=file_path), status_message
            else:
                return gr.update(value=None), status_message

        def export_all_characters_wrapper():
            zip_path = export_all_characters_as_zip()
            characters = get_character_cards()
            exported_characters = [char['name'] for char in characters]
            status_message = f"Exported {len(exported_characters)} characters successfully:\n" + "\n".join(exported_characters)
            return gr.update(value=zip_path), status_message

        # Event listeners
        load_characters_button.click(
            fn=lambda: gr.update(choices=[f"{char['name']} (ID: {char['id']})" for char in get_character_cards()]),
            outputs=export_character_dropdown
        )

        export_single_button.click(
            fn=export_single_character_wrapper,
            inputs=[export_character_dropdown],
            outputs=[export_output, export_status]
        )

        export_all_button.click(
            fn=export_all_characters_wrapper,
            inputs=[],
            outputs=[export_output, export_status]
        )

    return export_character_dropdown, load_characters_button, export_single_button, export_all_button, export_output, export_status

#
# End of Character_Chat_tab.py
#######################################################################################################################

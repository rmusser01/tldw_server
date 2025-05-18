# Character_Interaction_tab.py
# Description: This file contains the functions that are used for Character Interactions in the Gradio UI.
#
# Imports
import base64
import io
import uuid
import json
from typing import List, Dict, Tuple, Union, Optional, Any
#
# External Imports
import gradio as gr
from PIL import Image
#
# Local Imports
from App_Function_Libraries.Chat.Chat_Functions import load_characters, save_chat_history_to_db_wrapper
from App_Function_Libraries.DB.DB_Manager import add_character_card, get_character_cards
from App_Function_Libraries.Gradio_UI.Chat_ui import chat_wrapper
from App_Function_Libraries.Gradio_UI.Writing_tab import generate_writing_feedback
from App_Function_Libraries.Utils.Utils import default_api_endpoint, format_api_name, global_api_endpoints, logging
from App_Function_Libraries.Character_Chat.Character_Chat_Lib import load_character_card, parse_v2_card, parse_v1_card
#
########################################################################################################################
#
# Single-Character chat Functions:
# FIXME - add these functions to the Personas library

def chat_with_character(user_message, history, char_data, api_name_input, api_key):
    if char_data is None:
        return history, "Please import a character card first."

    bot_message = generate_writing_feedback(user_message, char_data['name'], "Overall", api_name_input,
                                            api_key)
    history.append((user_message, bot_message))
    return history, ""


def import_character_card(file):
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
            # If the returned data is a dict, and it contains the raw card keys,
            # then process it to produce the final card format.
            elif isinstance(card_data, dict):
                if card_data.get("spec") == "chara_card_v2":
                    logging.debug("Detected raw V2 character card data; parsing it.")
                    card_data = parse_v2_card(card_data)
                # If you have V1 cards that need parsing, you can add:
                # else:
                #     card_data = parse_v1_card(card_data)
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
        else:
            # **Optional:** If your final card data is expected to always have an "image" key,
            # you might set it to an empty string if not provided.
            if 'image' not in card_data:
                card_data['image'] = ""

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
        logging.debug(f"Parsed JSON data keys: {list(card_data.keys())}")

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
        logging.error(f"Problematic JSON content (first 500 chars): {json_content[:500]}...")
    except Exception as e:
        logging.error(f"Unexpected error parsing JSON: {e}")
    return None


def extract_json_from_image(image_file):
    logging.debug(f"Attempting to extract JSON from image: {image_file.name}")
    try:
        with Image.open(image_file) as img:
            logging.debug("Image opened successfully")
            metadata = img.info
            if 'chara' in metadata:
                logging.debug("Found 'chara' in image metadata")
                chara_content = metadata['chara']
                logging.debug(f"Content of 'chara' metadata (first 100 chars): {chara_content[:100]}...")
                try:
                    decoded_content = base64.b64decode(chara_content).decode('utf-8')
                    logging.debug(f"Decoded content (first 100 chars): {decoded_content[:100]}...")
                    return decoded_content
                except Exception as e:
                    logging.error(f"Error decoding base64 content: {e}")

            logging.debug("'chara' not found in metadata, checking for base64 encoded data")
            raw_data = img.tobytes()
            possible_json = raw_data.split(b'{', 1)[-1].rsplit(b'}', 1)[0]
            if possible_json:
                try:
                    decoded = base64.b64decode(possible_json).decode('utf-8')
                    if decoded.startswith('{') and decoded.endswith('}'):
                        logging.debug("Found and decoded base64 JSON data")
                        return '{' + decoded + '}'
                except Exception as e:
                    logging.error(f"Error decoding base64 data: {e}")

            logging.warning("No JSON data found in the image")
    except Exception as e:
        logging.error(f"Error extracting JSON from image: {e}")
    return None


def load_chat_history(file):
    try:
        content = file.read().decode('utf-8')
        chat_data = json.loads(content)
        return chat_data['history'], chat_data['character']
    except Exception as e:
        logging.error(f"Error loading chat history: {e}")
        return None, None


#
# End of X
######################################################################################################################
#
# Multi-Character Chat Interface

# FIXME - refactor and move these functions to the Character_Chat library so that it uses the same functions
def character_interaction_setup():
    characters = load_characters()
    return characters, [], None, None


def extract_character_response(response: Union[str, Tuple]) -> str:
    if isinstance(response, tuple):
        # If it's a tuple, try to extract the first string element
        for item in response:
            if isinstance(item, str):
                return item.strip()
        # If no string found, return a default message
        return "I'm not sure how to respond."
    elif isinstance(response, str):
        # If it's already a string, just return it
        return response.strip()
    else:
        # For any other type, return a default message
        return "I'm having trouble forming a response."

# def process_character_response(response: str) -> str:
#     # Remove any leading explanatory text before the first '---'
#     parts = response.split('---')
#     if len(parts) > 1:
#         return '---' + '---'.join(parts[1:])
#     return response.strip()
def process_character_response(response: Union[str, Tuple]) -> str:
    if isinstance(response, tuple):
        response = ' '.join(str(item) for item in response if isinstance(item, str))

    if isinstance(response, str):
        # Remove any leading explanatory text before the first '---'
        parts = response.split('---')
        if len(parts) > 1:
            return '---' + '---'.join(parts[1:])
        return response.strip()
    else:
        return "I'm having trouble forming a response."

def character_turn(characters: Dict, conversation: List[Tuple[str, str]],
                   current_character: str, other_characters: List[str],
                   api_endpoint: str, api_key: str, temperature: float,
                   scenario: str = "") -> Tuple[List[Tuple[str, str]], str]:
    if not current_character or current_character not in characters:
        return conversation, current_character

    if not conversation and scenario:
        conversation.append(("Scenario", scenario))

    current_char = characters[current_character]
    other_chars = [characters[char] for char in other_characters if char in characters and char != current_character]

    prompt = f"{current_char['name']}'s personality: {current_char['personality']}\n"
    for char in other_chars:
        prompt += f"{char['name']}'s personality: {char['personality']}\n"
    prompt += "Conversation so far:\n" + "\n".join([f"{sender}: {message}" for sender, message in conversation])
    prompt += f"\n\nHow would {current_char['name']} respond?"

    try:
        response = chat_wrapper(prompt, conversation, {}, [], api_endpoint, api_key, "", None, False, temperature, "")
        processed_response = process_character_response(response)
        conversation.append((current_char['name'], processed_response))
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        conversation.append((current_char['name'], error_message))

    return conversation, current_character


def character_interaction(character1: str, character2: str, api_endpoint: str, api_key: str,
                          num_turns: int, scenario: str, temperature: float,
                          user_interjection: str = "") -> List[str]:
    characters = load_characters()
    char1 = characters[character1]
    char2 = characters[character2]
    conversation = []
    current_speaker = char1
    other_speaker = char2

    # Add scenario to the conversation start
    if scenario:
        conversation.append(f"Scenario: {scenario}")

    for turn in range(num_turns):
        # Construct the prompt for the current speaker
        prompt = f"{current_speaker['name']}'s personality: {current_speaker['personality']}\n"
        prompt += f"{other_speaker['name']}'s personality: {other_speaker['personality']}\n"
        prompt += f"Conversation so far:\n" + "\n".join(
            [msg if isinstance(msg, str) else f"{msg[0]}: {msg[1]}" for msg in conversation])

        # Add user interjection if provided
        if user_interjection and turn == num_turns // 2:
            prompt += f"\n\nUser interjection: {user_interjection}\n"
            conversation.append(f"User: {user_interjection}")

        prompt += f"\n\nHow would {current_speaker['name']} respond?"

        # FIXME - figure out why the double print is happening
        # Get response from the LLM
        response = chat_wrapper(prompt, conversation, {}, [], api_endpoint, api_key, "", None, False, temperature, "")

        # Add the response to the conversation
        conversation.append((current_speaker['name'], response))

        # Switch speakers
        current_speaker, other_speaker = other_speaker, current_speaker

    # Convert the conversation to a list of strings for output
    return [f"{msg[0]}: {msg[1]}" if isinstance(msg, tuple) else msg for msg in conversation]


def create_multiple_character_chat_tab():
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
    with gr.TabItem("Multi-Character Chat", visible=True):
        characters, conversation, current_character, other_character = character_interaction_setup()

        with gr.Blocks() as character_interaction:
            gr.Markdown("# Multi-Character Chat")

            with gr.Row():
                num_characters = gr.Dropdown(label="Number of Characters", choices=["2", "3", "4"], value="2")
                character_selectors = [gr.Dropdown(label=f"Character {i + 1}", choices=list(characters.keys())) for i in
                                       range(4)]

            # Refactored API selection dropdown
            api_endpoint = gr.Dropdown(
                choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                value=default_value,
                label="API for Interaction (Optional)"
            )
            api_key = gr.Textbox(label="API Key (if required)", type="password")
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, step=0.1, value=0.7)
            scenario = gr.Textbox(label="Scenario (optional)", lines=3)

            chat_display = gr.Chatbot(label="Character Interaction")
            current_index = gr.State(0)

            next_turn_btn = gr.Button("Next Turn")
            narrator_input = gr.Textbox(label="Narrator Input", placeholder="Add a narration or description...")
            add_narration_btn = gr.Button("Add Narration")
            error_box = gr.Textbox(label="Error Messages", visible=False)
            reset_btn = gr.Button("Reset Conversation")
            chat_media_name = gr.Textbox(label="Custom Chat Name(optional)", visible=True)
            save_chat_history_to_db = gr.Button("Save Chat History to DataBase")

            def update_character_selectors(num):
                return [gr.update(visible=True) if i < int(num) else gr.update(visible=False) for i in range(4)]

            num_characters.change(
                update_character_selectors,
                inputs=[num_characters],
                outputs=character_selectors
            )

            def reset_conversation():
                return [], 0, gr.update(value=""), gr.update(value="")

            def take_turn(conversation, current_index, char1, char2, char3, char4, api_endpoint, api_key, temperature,
                          scenario):
                char_selectors = [char for char in [char1, char2, char3, char4] if char]  # Remove None values
                num_chars = len(char_selectors)

                if num_chars == 0:
                    return conversation, current_index  # No characters selected, return without changes

                if not conversation:
                    conversation = []
                    if scenario:
                        conversation.append(("Scenario", scenario))

                current_character = char_selectors[current_index % num_chars]
                next_index = (current_index + 1) % num_chars

                prompt = f"Character speaking: {current_character}\nOther characters: {', '.join(char for char in char_selectors if char != current_character)}\n"
                prompt += "Generate the next part of the conversation, including character dialogues and actions. Characters should speak in first person."

                response, new_conversation, _ = chat_wrapper(prompt, conversation, {}, [], api_endpoint, api_key, "",
                                                             None, False, temperature, "")

                # Format the response
                formatted_lines = []
                for line in response.split('\n'):
                    if ':' in line:
                        speaker, text = line.split(':', 1)
                        formatted_lines.append(f"**{speaker.strip()}**: {text.strip()}")
                    else:
                        formatted_lines.append(line)

                formatted_response = '\n'.join(formatted_lines)

                # Update the last message in the conversation with the formatted response
                if new_conversation:
                    new_conversation[-1] = (new_conversation[-1][0], formatted_response)
                else:
                    new_conversation.append((current_character, formatted_response))

                return new_conversation, next_index

            def add_narration(narration, conversation):
                if narration:
                    conversation.append(("Narrator", narration))
                return conversation, ""

            def take_turn_with_error_handling(conversation, current_index, char1, char2, char3, char4, api_endpoint,
                                              api_key, temperature, scenario):
                try:
                    new_conversation, next_index = take_turn(conversation, current_index, char1, char2, char3, char4,
                                                             api_endpoint, api_key, temperature, scenario)
                    return new_conversation, next_index, gr.update(visible=False, value="")
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    return conversation, current_index, gr.update(visible=True, value=error_message)

            # Define States for conversation_id and media_content, which are required for saving chat history
            media_content = gr.State({})
            conversation_id = gr.State(str(uuid.uuid4()))

            next_turn_btn.click(
                take_turn_with_error_handling,
                inputs=[chat_display, current_index] + character_selectors + [api_endpoint, api_key, temperature,
                                                                              scenario],
                outputs=[chat_display, current_index, error_box]
            )

            add_narration_btn.click(
                add_narration,
                inputs=[narrator_input, chat_display],
                outputs=[chat_display, narrator_input]
            )

            reset_btn.click(
                reset_conversation,
                outputs=[chat_display, current_index, scenario, narrator_input]
            )

        # FIXME - Implement saving chat history to database; look at Chat_UI.py for reference
        save_chat_history_to_db.click(
            save_chat_history_to_db_wrapper,
            inputs=[chat_display, conversation_id, media_content, chat_media_name],
            outputs=[conversation_id, gr.Textbox(label="Save Status")]
        )

        return character_interaction

#
# End of Multi-Character chat tab
########################################################################################################################
#
# Narrator-Controlled Conversation Tab

# From `Fuzzlewumper` on Reddit.
def create_narrator_controlled_conversation_tab():
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
    with gr.TabItem("Narrator-Controlled Conversation", visible=True):
        gr.Markdown("# Narrator-Controlled Conversation")

        with gr.Row():
            with gr.Column(scale=1):
                # Refactored API selection dropdown
                api_endpoint = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Chat Interaction (Optional)"
                )
                api_key = gr.Textbox(label="API Key (if required)", type="password")
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.0, step=0.1, value=0.7)

            with gr.Column(scale=2):
                narrator_input = gr.Textbox(
                    label="Narrator Input",
                    placeholder="Set the scene or provide context...",
                    lines=3
                )

        character_inputs = []
        for i in range(4):  # Allow up to 4 characters
            with gr.Row():
                name = gr.Textbox(label=f"Character {i + 1} Name")
                description = gr.Textbox(label=f"Character {i + 1} Description", lines=3)
                character_inputs.append((name, description))

        conversation_display = gr.Chatbot(label="Conversation", height=400)
        user_input = gr.Textbox(label="Your Input (optional)", placeholder="Add your own dialogue or action...")

        with gr.Row():
            generate_btn = gr.Button("Generate Next Interaction")
            reset_btn = gr.Button("Reset Conversation")
            chat_media_name = gr.Textbox(label="Custom Chat Name(optional)", visible=True)
            save_chat_history_to_db = gr.Button("Save Chat History to DataBase")

        error_box = gr.Textbox(label="Error Messages", visible=False)

        # Define States for conversation_id and media_content, which are required for saving chat history
        conversation_id = gr.State(str(uuid.uuid4()))
        media_content = gr.State({})

        def generate_interaction(conversation, narrator_text, user_text, api_endpoint, api_key, temperature,
                                 *character_data):
            try:
                characters = [{"name": name.strip(), "description": desc.strip()}
                              for name, desc in zip(character_data[::2], character_data[1::2])
                              if name.strip() and desc.strip()]

                if not characters:
                    raise ValueError("At least one character must be defined.")

                prompt = f"Narrator: {narrator_text}\n\n"
                for char in characters:
                    prompt += f"Character '{char['name']}': {char['description']}\n"
                prompt += "\nGenerate the next part of the conversation, including character dialogues and actions. "
                prompt += "Characters should speak in first person. "
                if user_text:
                    prompt += f"\nIncorporate this user input: {user_text}"
                prompt += "\nResponse:"

                response, conversation, _ = chat_wrapper(prompt, conversation, {}, [], api_endpoint, api_key, "", None,
                                                         False, temperature, "")

                # Format the response
                formatted_lines = []
                for line in response.split('\n'):
                    if ':' in line:
                        speaker, text = line.split(':', 1)
                        formatted_lines.append(f"**{speaker.strip()}**: {text.strip()}")
                    else:
                        formatted_lines.append(line)

                formatted_response = '\n'.join(formatted_lines)

                # Update the last message in the conversation with the formatted response
                if conversation:
                    conversation[-1] = (conversation[-1][0], formatted_response)
                else:
                    conversation.append((None, formatted_response))

                return conversation, gr.update(value=""), gr.update(value=""), gr.update(visible=False, value="")
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                return conversation, gr.update(), gr.update(), gr.update(visible=True, value=error_message)

        def reset_conversation():
            return [], gr.update(value=""), gr.update(value=""), gr.update(visible=False, value="")

        generate_btn.click(
            generate_interaction,
            inputs=[conversation_display, narrator_input, user_input, api_endpoint, api_key, temperature] +
                   [input for char_input in character_inputs for input in char_input],
            outputs=[conversation_display, narrator_input, user_input, error_box]
        )

        reset_btn.click(
            reset_conversation,
            outputs=[conversation_display, narrator_input, user_input, error_box]
        )

        # FIXME - Implement saving chat history to database; look at Chat_UI.py for reference
        save_chat_history_to_db.click(
            save_chat_history_to_db_wrapper,
            inputs=[conversation_display, conversation_id, media_content, chat_media_name],
            outputs=[conversation_id, gr.Textbox(label="Save Status")]
        )


    return api_endpoint, api_key, temperature, narrator_input, conversation_display, user_input, generate_btn, reset_btn, error_box

#
# End of Narrator-Controlled Conversation tab
########################################################################################################################
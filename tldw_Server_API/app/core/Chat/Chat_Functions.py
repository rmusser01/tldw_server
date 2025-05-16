# Chat_Functions.py
# Description: Chat functions for interacting with the LLMs as chatbots
#
# Imports
import base64
import json
import os
import random
import re
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
#
# 3rd-party Libraries
import requests

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
#
# Local Imports
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError, ChatConfigurationError, ChatAPIError, \
    ChatProviderError, ChatRateLimitError, ChatAuthenticationError
from tldw_Server_API.app.core.DB_Management.DB_Manager import start_new_conversation, delete_messages_in_conversation, save_message
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError, ConflictError, CharactersRAGDBError
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_openai, chat_with_anthropic, chat_with_cohere, \
    chat_with_groq, chat_with_openrouter, chat_with_deepseek, chat_with_mistral, chat_with_huggingface, chat_with_google
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls_Local import chat_with_aphrodite, chat_with_local_llm, chat_with_ollama, \
    chat_with_kobold, chat_with_llama, chat_with_oobabooga, chat_with_tabbyapi, chat_with_vllm, chat_with_custom_openai, \
    chat_with_custom_openai_2
from tldw_Server_API.app.core.Utils.Utils import generate_unique_filename, load_and_log_configs, logging
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
#
####################################################################################################
#
# Functions:

def approximate_token_count(history):
    try:
        total_text = ''
        for user_msg, bot_msg in history:
            if user_msg:
                total_text += user_msg + ' '
            if bot_msg:
                total_text += bot_msg + ' '
        total_tokens = len(total_text.split())
        return total_tokens
    except Exception as e:
        logging.error(f"Error calculating token count: {str(e)}")
        return 0


def chat_api_call(api_endpoint, api_key=None, input_data=None, prompt=None, temp=None, system_message=None, streaming=None, minp=None, maxp=None, model=None, topk=None, topp=None):
    """
    Acts as a sink/router to call various LLM API providers.

    Args:
        api_endpoint (str): The name of the API provider (e.g., 'openai', 'anthropic'). Case-insensitive.
        api_key (str, optional): The API key for the provider. Defaults to None (provider function may load from config).
        input_data (any, optional): The primary input data for the LLM (e.g., text, file path). Defaults to None.
        prompt (str, optional): The user's prompt or instruction. Often combined with input_data. Defaults to None.
        temp (float, optional): Temperature parameter for sampling. Defaults to None (provider function may load from config).
        system_message (str, optional): System-level instructions for the LLM. Defaults to None (provider function may load from config or use a default).
        streaming (bool, optional): Whether to enable streaming response. Defaults to None (provider function may load from config).
        minp (float, optional): Minimum probability threshold (provider specific). Defaults to None.
        maxp (float, optional): Maximum probability, often equivalent to top_p (provider specific, e.g., OpenAI, Groq). Defaults to None.
        model (str, optional): The specific model name to use. Defaults to None (provider function may load from config).
        topk (int, optional): Top-K sampling parameter (provider specific). Defaults to None.
        topp (float, optional): Top-P sampling parameter (provider specific, different from maxp for some). Defaults to None.

    Returns:
        The response from the API provider, which could be a string, a generator for streaming, or an error message.
    """
    endpoint_lower = api_endpoint.lower()
    logging.info(f"Chat API Call - Routing to endpoint: {endpoint_lower}")
    log_counter("chat_api_call_attempt", labels={"api_endpoint": endpoint_lower})
    start_time = time.time()

    try:
        # Log API key securely (first/last chars) only if it exists
        if api_key and isinstance(api_key, str) and len(api_key) > 8:
             logging.info(f"Debug - Chat API Call - API Key: {api_key[:4]}...{api_key[-4:]}")
        elif api_key:
             logging.info(f"Debug - Chat API Call - API Key: Provided (length <= 8)")
        else:
             logging.info(f"Debug - Chat API Call - API Key: Not Provided")

        # --- Routing Logic (Calls to backend functions) ---
        # The backend functions (chat_with_...) are now expected to raise exceptions
        # (like requests.exceptions.HTTPError, SDK errors, or potentially our custom ones)
        # on failure.
        if endpoint_lower == 'openai':
            response = chat_with_openai(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt' to 'custom_prompt_arg'
                temp=temp,
                system_message=system_message,
                streaming=streaming,
                maxp=maxp, # OpenAI uses 'top_p' internally, handled by chat_with_openai
                model=model
            )

        elif endpoint_lower == 'anthropic':
            # No need to load config here, let chat_with_anthropic handle it
            response = chat_with_anthropic(
                api_key=api_key,
                input_data=input_data,
                model=model,
                custom_prompt_arg=prompt, # Map 'prompt'
                system_prompt=system_message, # Map 'system_message'
                streaming=streaming,
                temp=temp,
                topp=topp, 
                topk=topk
                # chat_with_anthropic handles retries internally
            )

        elif endpoint_lower == "cohere":
            response = chat_with_cohere(
                api_key=api_key,
                input_data=input_data,
                model=model,
                custom_prompt_arg=prompt, # Map 'prompt'
                system_prompt=system_message, # Map 'system_message'
                temp=temp,
                streaming=streaming,
                topp=topp, 
                topk=topk
            )

        elif endpoint_lower == "groq":
            response = chat_with_groq(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming,
                maxp=maxp # Groq uses 'top_p' internally, handled by chat_with_groq
            )

        elif endpoint_lower == "openrouter":
            response = chat_with_openrouter(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming,
                top_p=topp, # Map 'topp' to 'top_p'
                top_k=topk, # Map 'topk' to 'top_k'
                minp=minp  # Pass 'minp'
            )

        elif endpoint_lower == "deepseek":
            response = chat_with_deepseek(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming,
                topp=topp 
            )

        elif endpoint_lower == "mistral":
            response = chat_with_mistral(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming,
                topp=topp, 
                model=model
            )

        elif endpoint_lower == "google":
            response = chat_with_google(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming,
                topp=topp, 
                topk=topk
            )

        elif endpoint_lower == "huggingface":
            response = chat_with_huggingface(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                system_prompt=system_message, # Map 'system_message'
                temp=temp,
                streaming=streaming
            )

        elif endpoint_lower == "llama.cpp":
             # chat_with_llama expects api_url as 4th arg, pass None to use config default
            response = chat_with_llama(
                input_data=input_data,
                custom_prompt=prompt, # Map 'prompt' to 'custom_prompt'
                temp=temp,
                api_url=None, # Let the function load from config
                api_key=api_key,
                system_prompt=system_message, # Map 'system_message'
                streaming=streaming,
                top_k=topk, # Map 'topk'
                top_p=topp, # Map 'topp'
                min_p=minp  # Map 'minp'
            )

        elif endpoint_lower == "kobold":
            response = chat_with_kobold(
                input_data=input_data,
                api_key=api_key,
                custom_prompt_input=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message, # Pass system_message
                streaming=streaming,
                top_k=topk, # Map 'topk'
                top_p=topp # Map 'topp'
            )

        elif endpoint_lower == "ooba":
             # chat_with_oobabooga expects api_url as 5th arg, pass None
            response = chat_with_oobabooga(
                input_data=input_data,
                api_key=api_key,
                custom_prompt=prompt, # Map 'prompt'
                system_prompt=system_message, # Map 'system_message'
                api_url=None, # Let the function load from config
                streaming=streaming,
                temp=temp,
                top_p=topp # Map 'topp'
            )

        elif endpoint_lower == "tabbyapi":
            response = chat_with_tabbyapi(
                input_data=input_data,
                custom_prompt_input=prompt, # Map 'prompt'
                system_message=system_message,
                api_key=api_key, # Pass api_key
                temp=temp,
                streaming=streaming, # Pass streaming
                top_k=topk, # Pass topk
                top_p=topp, # Pass topp
                min_p=minp # Pass minp
            )

        elif endpoint_lower == "vllm":
             # chat_with_vllm expects api_url as 4th arg, pass None
            response = chat_with_vllm(
                input_data=input_data,
                custom_prompt_input=prompt, # Map 'prompt'
                api_key=api_key,
                vllm_api_url=None, # Let the function load from config
                model=model, # Pass model
                system_prompt=system_message, # Map 'system_message'
                temp=temp,
                streaming=streaming,
                minp=minp, # Pass minp
                topp=topp, # Pass topp
                topk=topk # Pass topk
             )

        elif endpoint_lower == "local-llm":
            response = chat_with_local_llm(
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming, # Pass streaming
                top_k=topk, # Pass topk
                top_p=topp, # Pass topp
                min_p=minp # Pass minp
            )

        elif endpoint_lower == "ollama":
             # chat_with_ollama expects api_url as 3rd arg, pass None
            response = chat_with_ollama(
                input_data=input_data,
                custom_prompt=prompt, # Map 'prompt'
                api_url=None, # Let the function load from config
                api_key=api_key,
                temp=temp,
                system_message=system_message,
                model=model, # Pass model
                streaming=streaming, # Pass streaming
                top_p=topp # Pass topp
            )

        elif endpoint_lower == "aphrodite":
            response = chat_with_aphrodite(
                api_key=api_key, # Pass api_key
                input_data=input_data,
                custom_prompt=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming, # Pass streaming
                topp=topp, # Pass topp
                minp=minp, # Pass minp
                topk=topk, # Pass topk
                model=model # Pass model
            )

        elif endpoint_lower == "custom-openai-api":
            response = chat_with_custom_openai(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming, # Pass streaming
                maxp=maxp, # Pass maxp (maps to top_p internally)
                model=model, # Pass model
                minp=minp, # Pass minp
                topk=topk # Pass topk
            )

        elif endpoint_lower == "custom-openai-api-2":
             # NOTE: chat_with_custom_openai_2 doesn't accept maxp, minp, topk in its signature
            response = chat_with_custom_openai_2(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming, # Pass streaming
                model=model # Pass model
            )

        else:
            # --- Error for unsupported endpoint ---
            logging.error(f"Unsupported API endpoint requested: {api_endpoint}")
            raise ValueError(f"Unsupported API endpoint: {api_endpoint}")

        # --- Success Logging and Return ---
        call_duration = time.time() - start_time
        log_histogram("chat_api_call_duration", call_duration, labels={"api_endpoint": endpoint_lower})
        log_counter("chat_api_call_success", labels={"api_endpoint": endpoint_lower})
        # Avoid logging potentially huge responses
        if isinstance(response, str):
             logging.debug(f"Debug - Chat API Call - Response (first 500 chars): {response[:500]}...")
        elif hasattr(response, '__iter__') and not isinstance(response, (str, bytes, dict)):
             logging.debug(f"Debug - Chat API Call - Response: Streaming Generator")
        else:
             logging.debug(f"Debug - Chat API Call - Response Type: {type(response)}")

        return response # Return successful response

    # --- Exception Mapping ---
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, 'status_code', 500)
        error_text = getattr(e.response, 'text', str(e))
        log_message_base = f"{endpoint_lower} API call failed with status {status_code}"

        # Log safely first
        try:
            # Use % formatting for safety if loguru + f-string + json is problematic
            logging.error("%s. Details: %s", log_message_base, error_text[:500], exc_info=False)
            # Alternatively, keep f-string but be mindful:
            # logging.error(f"{log_message_base}. Details: {error_text[:500]}...", exc_info=False)
        except Exception as log_e:
            logging.error(f"Error during logging HTTPError details: {log_e}")  # Log the logging error itself

        # Now, raise the appropriate custom exception based on status code
        detail_message = f"API call to {endpoint_lower} failed with status {status_code}. Response: {error_text[:200]}"  # Truncate details

        if status_code == 401:
            raise ChatAuthenticationError(provider=endpoint_lower,
                                          message=f"Authentication failed for {endpoint_lower}. Check API key. Detail: {error_text[:200]}")
        elif status_code == 429:
            raise ChatRateLimitError(provider=endpoint_lower,
                                     message=f"Rate limit exceeded for {endpoint_lower}. Detail: {error_text[:200]}")
        elif 400 <= status_code < 500:
            raise ChatBadRequestError(provider=endpoint_lower,
                                      message=f"Bad request to {endpoint_lower} (Status {status_code}). Detail: {error_text[:200]}")
        # Consider 5xx errors as provider errors
        elif 500 <= status_code < 600:
            raise ChatProviderError(provider=endpoint_lower,
                                    message=f"Error from {endpoint_lower} server (Status {status_code}). Detail: {error_text[:200]}",
                                    status_code=status_code)
        else:  # Catch-all for unexpected HTTP statuses
            raise ChatAPIError(provider=endpoint_lower,
                               message=f"Unexpected HTTP status {status_code} from {endpoint_lower}. Detail: {error_text[:200]}",
                               status_code=status_code)


    except requests.exceptions.RequestException as e:
        logging.error(f"Network error connecting to {endpoint_lower}: {e}", exc_info=False)
        # Raise a custom exception for network errors too
        raise ChatProviderError(provider=endpoint_lower, message=f"Network error contacting {endpoint_lower}: {e}",
                                status_code=504)  # Use 504 Gateway Timeout

    except (ValueError, TypeError, KeyError) as e:
        logging.error(f"Value/Type/Key error during chat API call setup for {endpoint_lower}: {e}", exc_info=True)
        # Raise a configuration or bad request error
        error_type = "Configuration/Parameter Error"
        status = 400
        if "Unsupported API endpoint" in str(e):
            error_type = "Unsupported API"
            status = 501  # Not Implemented might be better? Or 400 still ok.
            raise ChatConfigurationError(provider=endpoint_lower, message=f"Unsupported API endpoint: {endpoint_lower}")
        else:
            raise ChatBadRequestError(provider=endpoint_lower, message=f"{error_type} for {endpoint_lower}: {e}")

    # --- Final Catch-all ---
    except Exception as e:
        # Log the unexpected error
        logging.exception(
            f"Unexpected internal error in chat_api_call for {endpoint_lower}: {e}")
        # Raise a generic ChatAPIError
        raise ChatAPIError(provider=endpoint_lower,
                           message=f"An unexpected internal error occurred in chat_api_call for {endpoint_lower}: {str(e)}",
                           status_code=500)


def chat(message, history, media_content, selected_parts, api_endpoint, api_key, custom_prompt, temperature,
         system_message=None, streaming=False, minp=None, maxp=None, model=None, topp=None, topk=None, chatdict_entries=None, max_tokens=500, strategy="sorted_evenly"):
    log_counter("chat_attempt", labels={"api_endpoint": api_endpoint})
    start_time = time.time()
    try:
        logging.info(f"Debug - Chat Function - Message: {message}")
        logging.info(f"Debug - Chat Function - Media Content: {media_content}")
        logging.info(f"Debug - Chat Function - Selected Parts: {selected_parts}")
        logging.info(f"Debug - Chat Function - API Endpoint: {api_endpoint}")
        # logging.info(f"Debug - Chat Function - Prompt: {prompt}")

        # Ensure selected_parts is a list
        if not isinstance(selected_parts, (list, tuple)):
            selected_parts = [selected_parts] if selected_parts else []

        # logging.debug(f"Debug - Chat Function - Selected Parts (after check): {selected_parts}")

        # Handle Chat Dictionary processing
        if chatdict_entries:
            processed_input = process_user_input(
                message,
                chatdict_entries,
                max_tokens=max_tokens,
                strategy=strategy
            )
            message = processed_input

        # Combine the selected parts of the media content
        combined_content = "\n\n".join(
            [f"{part.capitalize()}: {media_content.get(part, '')}" for part in selected_parts if part in media_content])
        # Print first 500 chars
        # logging.debug(f"Debug - Chat Function - Combined Content: {combined_content[:500]}...")

        # Prepare the input for the API
        input_data = f"{combined_content}\n\n" if combined_content else ""
        for old_message, old_response in history:
            input_data += f"{old_message}\nAssistant: {old_response}\n\n"
        input_data += f"{message}\n"

        if system_message:
            print(f"System message: {system_message}")
            logging.debug(f"Debug - Chat Function - System Message: {system_message}")
        temperature = float(temperature) if temperature else 0.7
        temp = temperature

        logging.debug(f"Debug - Chat Function - Temperature: {temperature}")
        logging.debug(f"Debug - Chat Function - API Key: {api_key[:10]}")
        logging.debug(f"Debug - Chat Function - Prompt: {custom_prompt}")

        # Use the existing API request code based on the selected endpoint
        response = chat_api_call(api_endpoint, api_key, input_data, custom_prompt, temp, system_message, streaming, minp, maxp, model, topp, topk)

        if streaming:
            logging.debug(f"Debug - Chat Function - Response: {response}")
            return response
        else:
            chat_duration = time.time() - start_time
            log_histogram("chat_duration", chat_duration, labels={"api_endpoint": api_endpoint})
            log_counter("chat_success", labels={"api_endpoint": api_endpoint})
            logging.debug(f"Debug - Chat Function - Response: {response}")
            loaded_config_data = load_and_log_configs()
            post_gen_replacement = loaded_config_data['chat_dictionaries']['post_gen_replacement']
            if post_gen_replacement:
                post_gen_replacement_dict = loaded_config_data['chat_dictionaries']['post_gen_replacement_dict']
                chatdict_entries = parse_user_dict_markdown_file(post_gen_replacement_dict)
                response = process_user_input(
                    response,
                    chatdict_entries,
                    # max_tokens=max_tokens(5000 default),
                    # strategy="sorted_evenly" (default)
                )
            return response
    except Exception as e:
        log_counter("chat_error", labels={"api_endpoint": api_endpoint, "error": str(e)})
        logging.error(f"Error in chat function: {str(e)}")
        return f"An error occurred: {str(e)}"


def save_chat_history_to_db_wrapper(
    db: CharactersRAGDB,
    chatbot: List[Tuple[Optional[str], Optional[str]]],
    conversation_id: Optional[str],
    media_content: Optional[Dict[str, Any]],
    media_name: Optional[str] = None,
    character_name_for_chat: Optional[str] = None
) -> Tuple[Optional[str], str]:
    log_counter("save_chat_history_to_db_attempt")
    start_time = time.time()
    logging.info(f"Attempting to save chat history. Conversation ID: {conversation_id}, Character Name for Chat: {character_name_for_chat}, Media Name: {media_name}")

    try:
        # The DB connection is managed by the CharactersRAGDB instance (`db`)
        # No need for direct `get_db_connection` or manual sqlite3 corruption checks here.
        # The `db` instance methods will raise exceptions if issues occur.

        associated_character_id: Optional[int] = None
        final_character_name_for_title = "Unknown Character" # For conversation title

        # Determine character_id
        # Priority: 1. character_name_for_chat, 2. media_name, 3. title from media_content
        char_lookup_name = character_name_for_chat
        if not char_lookup_name and media_name:
            char_lookup_name = media_name
        if not char_lookup_name and media_content:
            content_details = media_content.get('content')  # This might be a JSON string or a dict
            if isinstance(content_details, str):
                try:
                    content_details = json.loads(content_details)
                except json.JSONDecodeError:
                    content_details = {}  # Not a JSON string
            if isinstance(content_details, dict):
                char_lookup_name = content_details.get('title')  # Fallback to title from media content

        if char_lookup_name:
            try:
                character = db.get_character_card_by_name(char_lookup_name)
                if character:
                    associated_character_id = character['id']
                    final_character_name_for_title = character['name'] # Use the actual name from DB
                    logging.info(f"Chat will be associated with specific character '{final_character_name_for_title}' (ID: {associated_character_id}).")
                else:
                    # If a specific character was named but not found, this is an error.
                    # The user intended a character chat, but the character is missing.
                    logging.error(f"Intended specific character '{char_lookup_name}' not found in DB. Chat save aborted.")
                    return conversation_id, f"Error: Specific character '{char_lookup_name}' intended for this chat was not found. Cannot save chat."
            except CharactersRAGDBError as e:
                logging.error(f"DB error looking up specific character '{char_lookup_name}': {e}")
                return conversation_id, f"DB error finding specific character: {e}"
        else: # No specific character was named, try to use the Default Character
            logging.info("No specific character name provided for chat. Attempting to use Default Character.")
            try:
                default_char = db.get_character_card_by_name(DEFAULT_CHARACTER_NAME)
                if default_char:
                    associated_character_id = default_char['id']
                    final_character_name_for_title = default_char['name']
                    logging.info(f"Chat will be associated with '{DEFAULT_CHARACTER_NAME}' (ID: {associated_character_id}).")
                else:
                    # This is a critical state: no specific char, and default char is missing (should have been created by DB dep)
                    logging.error(f"'{DEFAULT_CHARACTER_NAME}' is missing from the DB and no specific character was provided. Chat save aborted.")
                    return conversation_id, f"Error: Critical - '{DEFAULT_CHARACTER_NAME}' is missing. Cannot save chat."
            except CharactersRAGDBError as e:
                logging.error(f"DB error looking up '{DEFAULT_CHARACTER_NAME}': {e}")
                return conversation_id, f"DB error finding '{DEFAULT_CHARACTER_NAME}': {e}"

        # Ensure we have a character_id to proceed
        if associated_character_id is None:
             # This should be an unreachable state if the logic above is correct.
             logging.critical(f"Logic error: associated_character_id is None after character lookup. Chat save aborted.")
             return conversation_id, "Critical internal error: Could not determine character for chat."

        current_conversation_id = conversation_id

        if not current_conversation_id: # New conversation
            # Use final_character_name_for_title for the conversation title
            conv_title_base = f"Chat with {final_character_name_for_title}"

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            conversation_title = f"{conv_title_base} ({timestamp_str})"

            conv_data = {
                'character_id': associated_character_id,
                'title': conversation_title,
                # 'root_id' will be set to new conv_id by add_conversation if not provided
                'client_id': db.client_id  # Use client_id from the DB instance
            }
            try:
                current_conversation_id = db.add_conversation(conv_data)
                if not current_conversation_id:  # Should not happen if add_conversation raises on failure
                    return None, "Failed to create new conversation in DB."
                logging.info(
                    f"Created new conversation with ID: {current_conversation_id} for character ID: {associated_character_id}")
            except (InputError, ConflictError, CharactersRAGDBError) as e:
                logging.error(f"Error creating new conversation: {e}", exc_info=True)
                return None, f"Error creating conversation: {e}"

            # Save all messages from chatbot history to the new conversation
            for user_msg, assistant_msg in chatbot:
                try:
                    if user_msg:
                        db.add_message({
                            'conversation_id': current_conversation_id,
                            'sender': 'user',
                            'content': user_msg,
                            'client_id': db.client_id
                        })
                    if assistant_msg:
                        db.add_message({
                            'conversation_id': current_conversation_id,
                            'sender': 'assistant',
                            'content': assistant_msg,
                            'client_id': db.client_id
                        })
                except (InputError, ConflictError, CharactersRAGDBError) as e:
                    logging.error(f"Error saving message to new conversation {current_conversation_id}: {e}",
                                  exc_info=True)
                    # Decide: stop all, or try to save other messages? For now, return error.
                    return current_conversation_id, f"Error saving messages: {e}"
            logging.info(f"Saved all messages from chatbot history to new conversation {current_conversation_id}")

        else:  # Existing conversation_id provided - "resaving" the history
            logging.info(
                f"Resaving history for existing conversation ID: {current_conversation_id}. Character ID: {associated_character_id}")
            # This implies replacing all messages for the given conversation_id with the new 'chatbot' history.
            # This is complex with versioning.
            # 1. Soft-delete all existing messages for this conversation.
            # 2. Add all messages from the 'chatbot' list.
            try:
                with db.transaction():
                    existing_conv_details = db.get_conversation_by_id(current_conversation_id)
                    if not existing_conv_details:
                        logging.error(f"Cannot resave: Conversation {current_conversation_id} not found.")
                        return current_conversation_id, f"Error: Conversation {current_conversation_id} not found for resaving."

                    # Important: Ensure the existing conversation being updated belongs to the character context we're in.
                    # This prevents accidentally overwriting a chat from Character A if the current UI context is Character B (or Default).
                    if existing_conv_details.get('character_id') != associated_character_id:
                        # Fetch names for better logging
                        existing_char_of_conv = db.get_character_card_by_id(existing_conv_details.get('character_id'))
                        existing_char_name = existing_char_of_conv['name'] if existing_char_of_conv else "ID "+str(existing_conv_details.get('character_id'))

                        logging.error(f"Cannot resave: Conversation {current_conversation_id} (for char '{existing_char_name}') does not match current character context '{final_character_name_for_title}' (ID: {associated_character_id}).")
                        return current_conversation_id, "Error: Mismatch in character association for resaving chat. The conversation belongs to a different character."

                    existing_messages = db.get_messages_for_conversation(current_conversation_id, limit=10000, order_by_timestamp="ASC")
                    logging.info(f"Found {len(existing_messages)} existing messages to soft-delete for conv {current_conversation_id}.")
                    for msg in existing_messages:
                        db.soft_delete_message(msg['id'], msg['version'])
                    logging.info(f"Soft-deleted existing messages for conv {current_conversation_id}.")

                    # Add new messages from chatbot history
                    for user_msg, assistant_msg in chatbot:
                        if user_msg:
                            db.add_message({
                                'conversation_id': current_conversation_id,
                                'sender': 'user',
                                'content': user_msg,
                                'client_id': db.client_id
                            })
                        if assistant_msg:
                            db.add_message({
                                'conversation_id': current_conversation_id,
                                'sender': 'assistant',
                                'content': assistant_msg,
                                'client_id': db.client_id
                            })
                    logging.info(f"Added new set of messages from chatbot history to conv {current_conversation_id}.")

                    # Update conversation metadata (last_modified, version)
                    # Re-fetch conversation details to get the latest version after message changes (though messages don't directly alter conv version)
                    # This is more about bumping the conversation's own version and last_modified timestamp.
                    conv_details_for_update = db.get_conversation_by_id(current_conversation_id)
                    if conv_details_for_update:
                        db.update_conversation(
                            current_conversation_id,
                            {'title': conv_details_for_update.get('title')}, # Pass some data to ensure update call structure is met
                            conv_details_for_update['version']
                        )
                    else: # Should not happen if previous checks passed
                        logging.error(f"Conversation {current_conversation_id} disappeared before final metadata update.")


            except (InputError, ConflictError, CharactersRAGDBError) as e:
                logging.error(f"Error resaving messages for conversation {current_conversation_id}: {e}", exc_info=True)
                return current_conversation_id, f"Error resaving messages: {e}"

        save_duration = time.time() - start_time
        log_histogram("save_chat_history_to_db_duration", save_duration)
        log_counter("save_chat_history_to_db_success")

        return current_conversation_id, "Chat history saved successfully!"

    except Exception as e:
        log_counter("save_chat_history_to_db_error", labels={"error": str(e)})
        error_message = f"Failed to save chat history due to an unexpected error: {str(e)}"
        logging.error(error_message, exc_info=True)
        return conversation_id, error_message


def save_chat_history(history, conversation_id, media_content):
    log_counter("save_chat_history_attempt")
    start_time = time.time()
    try:
        content, conversation_name = generate_chat_history_content(history, conversation_id, media_content)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_conversation_name = re.sub(r'[^a-zA-Z0-9_-]', '_', conversation_name)
        base_filename = f"{safe_conversation_name}_{timestamp}.json"

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Generate a unique filename
        unique_filename = generate_unique_filename(os.path.dirname(temp_file_path), base_filename)
        final_path = os.path.join(os.path.dirname(temp_file_path), unique_filename)

        # Rename the temporary file to the unique filename
        os.rename(temp_file_path, final_path)

        save_duration = time.time() - start_time
        log_histogram("save_chat_history_duration", save_duration)
        log_counter("save_chat_history_success")
        return final_path
    except Exception as e:
        log_counter("save_chat_history_error", labels={"error": str(e)})
        logging.error(f"Error saving chat history: {str(e)}")
        return None


def get_conversation_name(conversation_id: Optional[str], db_instance: Optional[CharactersRAGDB] = None) -> Optional[str]:
    """
    Helper to get conversation name. Tries DB first if instance provided, then falls back.
    """
    if db_instance and conversation_id:
        try:
            conversation = db_instance.get_conversation_by_id(conversation_id)
            if conversation and conversation.get('title'):
                return conversation['title']
        except Exception as e:
            logging.warning(f"Could not fetch conversation title from DB for {conversation_id}: {e}")
    # Fallback or if no DB instance provided
    # This part of the original logic is unclear how it worked without DB.
    # For now, returning None if not found in DB.
    return None


def generate_chat_history_content(history, conversation_id, media_content,
                                  db_instance: Optional[CharactersRAGDB] = None):
    # Modified to potentially use db_instance for conversation_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Try to get conversation name from DB if possible
    conversation_name = None
    if conversation_id:
        conversation_name = get_conversation_name(conversation_id, db_instance)

    if not conversation_name:  # Fallback logic
        media_name_extracted = extract_media_name(media_content)  # media_content is the original complex object
        if media_name_extracted:
            conversation_name = f"{media_name_extracted}-chat-{timestamp}"
        else:
            conversation_name = f"chat-{timestamp}"

    chat_data = {
        "conversation_id": conversation_id,  # Can be None if new chat not yet saved to DB
        "conversation_name": conversation_name,
        "timestamp": timestamp,
        "history": [],
        # The original history format seemed to be a list of tuples (user, bot) or just a list of messages
        # The new DB stores messages individually. This JSON should reflect the 'chatbot' structure if it's for UI.
        # Assuming 'history' is like chatbot: List[Tuple[Optional[str], Optional[str]]]
    }

    current_turn = []
    for item in history:  # Iterating through the provided history structure
        if isinstance(item, tuple) and len(item) == 2:  # Expected (user_msg, bot_msg)
            user_msg, bot_msg = item
            if user_msg is not None:
                chat_data["history"].append({"role": "user", "content": user_msg})
            if bot_msg is not None:
                chat_data["history"].append(
                    {"role": "assistant", "content": bot_msg})  # Changed "bot" to "assistant" for consistency
        elif isinstance(item, dict) and "role" in item and "content" in item:  # Already in desired format
            chat_data["history"].append(item)
        else:
            logging.warning(f"Unexpected item format in history for JSON export: {item}")

    return json.dumps(chat_data, indent=2), conversation_name  # Return the derived/fetched name


def extract_media_name(media_content: Optional[Dict[str, Any]]):  # media_content is the original complex object
    if not media_content or not isinstance(media_content, dict):
        return None

    # Try to get from 'content' which might be a JSON string or a dict
    content_field = media_content.get('content')
    parsed_content = None

    if isinstance(content_field, str):
        try:
            parsed_content = json.loads(content_field)
        except json.JSONDecodeError:
            logging.warning("Failed to parse media_content['content'] JSON string in extract_media_name")
            # It might be a plain string title itself, or not what we expect
            # For now, if it's a non-JSON string, we don't assume it's the name.
            parsed_content = {}
    elif isinstance(content_field, dict):
        parsed_content = content_field

    if isinstance(parsed_content, dict):
        # Check common keys for a title or name
        name = parsed_content.get('title') or \
               parsed_content.get('name') or \
               parsed_content.get('media_title') or \
               parsed_content.get('webpage_title')
        if name: return name

    # Fallback to top-level keys in media_content itself if 'content' didn't yield a name
    name_top_level = media_content.get('title') or \
                     media_content.get('name') or \
                     media_content.get('media_title')
    if name_top_level: return name_top_level

    logging.warning(f"Could not extract a clear media name from media_content: {str(media_content)[:200]}")
    return None


def update_chat_content(
        selected_item: Optional[str],
        use_content: bool,
        use_summary: bool,
        use_prompt: bool,
        item_mapping: Dict[str, str],  # Maps display name (selected_item) to a note_id (media_id)
        db_instance: CharactersRAGDB  # Changed: Pass the DB instance
) -> Tuple[Dict[str, str], List[str]]:  # Returns dict of content strings, and list of selected part names
    log_counter("update_chat_content_attempt")
    start_time = time.time()
    logging.debug(f"Debug - Update Chat Content - Selected Item: {selected_item}")
    # ... other debug logs ...

    # This function's purpose seems to be to fetch content (possibly from a 'note' in the new DB)
    # and prepare it for the 'chat' function's 'media_content' input.
    # The 'media_content' that 'chat' receives is a simple dict of strings: {'summary': '...', 'content': '...'}

    output_media_content_for_chat: Dict[str, str] = {}  # This will be passed to chat()
    selected_parts_names: List[str] = []

    if selected_item and selected_item in item_mapping:
        note_id = item_mapping[selected_item]  # Assuming media_id from mapping is a note_id (UUID string)

        try:
            note_data = db_instance.get_note_by_id(note_id)
        except CharactersRAGDBError as e:
            logging.error(f"Error fetching note {note_id} for chat content: {e}", exc_info=True)
            note_data = None
        except Exception as e_gen:  # Catch any other unexpected error during DB fetch
            logging.error(f"Unexpected error fetching note {note_id}: {e_gen}", exc_info=True)
            note_data = None

        if note_data:
            # The content of the note ('note_data.content') might be:
            # 1. A plain string (e.g., the main transcript/content).
            # 2. A JSON string containing structured data like {"content": "...", "summary": "...", "prompt": "..."}.

            raw_note_content_field = note_data.get('content', '')  # The actual text from notes.content
            structured_content_from_note: Dict[str, str] = {}

            # Try to parse raw_note_content_field as JSON
            if isinstance(raw_note_content_field, str) and \
                    raw_note_content_field.strip().startswith('{') and \
                    raw_note_content_field.strip().endswith('}'):
                try:
                    parsed_json = json.loads(raw_note_content_field)
                    if isinstance(parsed_json, dict):
                        # Filter to ensure only string values are taken for safety
                        structured_content_from_note = {k: str(v) for k, v in parsed_json.items() if
                                                        isinstance(v, (str, int, float, bool))}
                        logging.debug(f"Parsed note's content field (ID: {note_id}) as JSON.")
                    else:
                        # JSON, but not a dict. Treat main content as the raw string.
                        structured_content_from_note['content'] = raw_note_content_field
                        logging.debug(
                            f"Note's content field (ID: {note_id}) was JSON but not a dict. Using raw string for 'content'.")
                except json.JSONDecodeError:
                    # Not valid JSON, treat it as the main 'content' part
                    structured_content_from_note['content'] = raw_note_content_field
                    logging.debug(f"Note's content field (ID: {note_id}) is not JSON. Using raw string for 'content'.")
            else:  # Not a JSON string, treat as main 'content'
                structured_content_from_note['content'] = raw_note_content_field
                logging.debug(f"Note's content field (ID: {note_id}) is a plain string. Using for 'content'.")

            # Populate `output_media_content_for_chat` based on `use_` flags and what's in `structured_content_from_note`
            if use_content and "content" in structured_content_from_note:
                output_media_content_for_chat["content"] = structured_content_from_note["content"]
                selected_parts_names.append("content")

            if use_summary and "summary" in structured_content_from_note:
                output_media_content_for_chat["summary"] = structured_content_from_note["summary"]
                selected_parts_names.append("summary")
            elif use_summary and "content" in structured_content_from_note and "summary" not in output_media_content_for_chat:
                # Fallback: if summary requested but not explicitly present, use first N words of content as summary?
                # For now, only include if explicitly present.
                logging.debug("Summary requested but not found in structured note content.")

            if use_prompt and "prompt" in structured_content_from_note:
                output_media_content_for_chat["prompt"] = structured_content_from_note["prompt"]
                selected_parts_names.append("prompt")

            # Add note title as a part, if not already taken by 'content', 'summary', or 'prompt'
            # and if a relevant use_ flag is true (e.g., use_content implies use_title if no other content)
            # This part is a bit ambiguous from the original. Let's assume title is just metadata for now.
            # Or, if note_data['title'] is meaningful as a 'part':
            # if use_title_flag and "title" not in selected_parts_names:
            #    output_media_content_for_chat["title_from_note"] = note_data.get('title', '')
            #    selected_parts_names.append("title_from_note")

            # Debug logging of what was prepared
            logging.debug(f"Prepared media content for chat from note {note_id}:")
            for key, value in output_media_content_for_chat.items():
                logging.debug(f"  {key} (first 100 chars): {str(value)[:100]}")
            logging.debug(f"Selected part names for chat: {selected_parts_names}")

        else:  # Note not found
            logging.warning(f"Note ID {note_id} (from selected_item '{selected_item}') not found in DB.")
            # Return empty, as per original fallback
            output_media_content_for_chat = {}
            selected_parts_names = []

    else:  # No item selected or item not in mapping
        log_counter("update_chat_content_error", labels={"error": str("No item selected or item not in mapping")})
        logging.debug(f"Debug - Update Chat Content - No item selected or item not in mapping: {selected_item}")
        output_media_content_for_chat = {}
        selected_parts_names = []

    update_duration = time.time() - start_time
    log_histogram("update_chat_content_duration", update_duration)
    log_counter("update_chat_content_success" if selected_parts_names else "update_chat_content_noop")

    return output_media_content_for_chat, selected_parts_names

#
# End of Chat functions
#######################################################################################################################


#######################################################################################################################
#
# Chat Dictionary Functions

def parse_user_dict_markdown_file(file_path):
    """
    Parse a Markdown file with custom termination symbol for multi-line values.
    """
    logging.debug(f"Parsing user dictionary file: {file_path}")
    replacement_dict = {}
    current_key = None
    current_value = []
    termination_pattern = re.compile(r'^\s*---@@@---\s*$')

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Check for termination pattern first
            if termination_pattern.match(line):
                if current_key:
                    replacement_dict[current_key] = '\n'.join(current_value).strip()
                    current_key, current_value = None, []
                continue

            # Match key lines only when not in multi-line mode
            if not current_key:
                key_value_match = re.match(r'^\s*([^:\n]+?)\s*:\s*(.*?)\s*$', line)
                if key_value_match:
                    key, value = key_value_match.groups()
                    if value.strip() == '|':
                        current_key = key.strip()
                        current_value = []
                    else:
                        replacement_dict[key.strip()] = value.strip()
                continue

            # Processing multi-line content
            if current_key:
                # Strip trailing whitespace but preserve leading spaces
                cleaned_line = line.rstrip('\n\r')
                current_value.append(cleaned_line)

        # Handle any remaining multi-line value at EOF
        if current_key:
            replacement_dict[current_key] = '\n'.join(current_value).strip()

    logging.debug(f"Parsed entries: {replacement_dict}")
    return replacement_dict


class ChatDictionary:
    def __init__(self, key, content, probability=100, group=None, timed_effects=None, max_replacements=1):
        self.key = self.compile_key(key)
        self.content = content
        self.probability = probability
        self.group = group
        self.timed_effects = timed_effects or {"sticky": 0, "cooldown": 0, "delay": 0}
        self.last_triggered = None  # Track when it was last triggered (for timed effects)
        self.max_replacements = max_replacements  # New: Limit replacements

    @staticmethod
    def compile_key(key):
        # Compile regex if wrapped with "/" delimiters
        if key.startswith("/") and key.endswith("/"):
            return re.compile(key[1:-1], re.IGNORECASE)
        return key

    def matches(self, text):
        # Match either regex or plain text
        if isinstance(self.key, re.Pattern):
            return self.key.search(text) is not None
        return self.key in text


# Strategy for inclusion
def apply_strategy(entries, strategy="sorted_evenly"):
    logging.debug(f"Applying strategy: {strategy}")
    if strategy == "sorted_evenly":
        return sorted(entries, key=lambda e: e.key)
    elif strategy == "character_lore_first":
        return sorted(entries, key=lambda e: (e.group != "character", e.key))
    elif strategy == "global_lore_first":
        return sorted(entries, key=lambda e: (e.group != "global", e.key))


# Probability modification of inclusion
def filter_by_probability(entries):
    return [entry for entry in entries if random.randint(1, 100) <= entry.probability]


# Group Scoring - Situation where multiple entries are triggered in different groups in a single message
def group_scoring(entries: List[ChatDictionary]) -> List[ChatDictionary]:
    logging.debug(f"Group scoring for {len(entries)} entries")
    if not entries: return []

    grouped_entries: Dict[Optional[str], List[ChatDictionary]] = {}
    for entry in entries:
        grouped_entries.setdefault(entry.group, []).append(entry)

    selected_entries: List[ChatDictionary] = []
    for group, group_entries_list in grouped_entries.items():
        if not group_entries_list: continue
        # Scoring: prefer entries with more specific (longer) keys first, then by original order if keys are same pattern
        # This is a simple heuristic. More complex scoring could be length of key string, or complexity of regex.
        # For now, using the length of the raw key string as a proxy for specificity.
        # If all keys in a group are plain strings, this prefers longer string matches.
        # If regexes are mixed, this is less reliable.
        # Max based on key_raw length
        best_entry_in_group = max(group_entries_list, key=lambda e: len(str(e.key_raw)) if e.key_raw else 0)
        selected_entries.append(best_entry_in_group)

    logging.debug(f"Selected {len(selected_entries)} entries after group scoring.")
    return selected_entries

# Timed Effects
def apply_timed_effects(entry, current_time):
    logging.debug(f"Applying timed effects for entry: {entry.key}")
    if entry.timed_effects["delay"] > 0:
        if entry.last_triggered is None or current_time - entry.last_triggered < timedelta(seconds=entry.timed_effects["delay"]):
            return False
    if entry.timed_effects["cooldown"] > 0:
        if entry.last_triggered and current_time - entry.last_triggered < timedelta(seconds=entry.timed_effects["cooldown"]):
            return False
    entry.last_triggered = current_time
    return True

# Context/Token Budget Mgmt
def calculate_token_usage(entries):
    logging.debug(f"Calculating token usage for {len(entries)} entries")
    return sum(len(entry.content.split()) for entry in entries)

def enforce_token_budget(entries, max_tokens):
    total_tokens = 0
    valid_entries = []
    for entry in entries:
        tokens = len(entry.content.split())
        if total_tokens + tokens <= max_tokens:
            valid_entries.append(entry)
            total_tokens += tokens
    return valid_entries

# Match whole words
def match_whole_words(entries, text):
    matched_entries = []
    for entry in entries:
        if re.search(rf'\b{entry.key}\b', text):
            matched_entries.append(entry)
            logging.debug(f"Chat Dictionary: Matched entry: {entry.key}")
    return matched_entries

class TokenBudgetExceededWarning(Warning):
    """Custom warning for token budget issues"""
    pass

# Token Budget Mgmt
def alert_token_budget_exceeded(entries, max_tokens):
    token_usage = calculate_token_usage(entries)
    logging.debug(f"Token usage: {token_usage}, Max tokens: {max_tokens}")
    if token_usage > max_tokens:
        warning_msg = f"Alert: Token budget exceeded! Used: {token_usage}, Allowed: {max_tokens}"
        warnings.warn(TokenBudgetExceededWarning(warning_msg))
        print(warning_msg)

# Single Replacement Function
def apply_replacement_once(text, entry):
    """
    Replaces the 'entry.key' in 'text' exactly once (if found).
    Returns the new text and the number of replacements actually performed.
    """
    logging.debug(f"Applying replacement for entry: {entry.key}")
    if isinstance(entry.key, re.Pattern):
        replaced_text, replaced_count = re.subn(entry.key, entry.content, text, count=1)
    else:
        # Use regex to replace case-insensitively and match whole words
        pattern = re.compile(rf'\b{re.escape(entry.key)}\b', re.IGNORECASE)
        replaced_text, replaced_count = re.subn(pattern, entry.content, text, count=1)
    return replaced_text, replaced_count

# Chat Dictionary Pipeline
def process_user_input(user_input, entries, max_tokens=5000, strategy="sorted_evenly"):
    current_time = datetime.now()

    try:
        # 1. Match entries using regex or plain text
        matched_entries = []
        logging.debug(f"Chat Dictionary: Matching entries for user input: {user_input}")
        for entry in entries:
            try:
                if entry.matches(user_input):
                    matched_entries.append(entry)
            except re.error as e:
                log_counter("chat_dict_regex_error", labels={"key": entry.key})
                logging.error(f"Invalid regex pattern in entry: {entry.key}. Error: {str(e)}")
                continue  # Skip this entry but continue processing others

        logging.debug(f"Matched entries after filtering: {[e.key for e in matched_entries]}")
        # 2. Apply group scoring
        try:
            logging.debug(f"Chat Dictionary: Applying group scoring for {len(matched_entries)} entries")
            matched_entries = group_scoring(matched_entries)
        except Exception as e:
            log_counter("chat_dict_group_scoring_error")
            logging.error(f"Error in group scoring: {str(e)}")
            matched_entries = []  # Fallback to empty list

        # 3. Apply probability filter
        try:
            logging.debug(f"Chat Dictionary: Filtering by probability for {len(matched_entries)} entries")
            matched_entries = filter_by_probability(matched_entries)
        except Exception as e:
            log_counter("chat_dict_probability_error")
            logging.error(f"Error in probability filtering: {str(e)}")
            matched_entries = []  # Fallback to empty list

        # 4. Apply timed effects
        try:
            logging.debug("Chat Dictionary: Applying timed effects")
            matched_entries = [entry for entry in matched_entries if apply_timed_effects(entry, current_time)]
        except Exception as e:
            log_counter("chat_dict_timed_effects_error")
            logging.error(f"Error applying timed effects: {str(e)}")
            matched_entries = []  # Fallback to empty list

        # 5. Enforce token budget
        try:
            logging.debug(f"Chat Dictionary: Enforcing token budget for {len(matched_entries)} entries")
            matched_entries = enforce_token_budget(matched_entries, max_tokens)
        except TokenBudgetExceededWarning as e:
            log_counter("chat_dict_token_limit")
            logging.warning(str(e))
            matched_entries = []  # Fallback to empty list
        except Exception as e:
            log_counter("chat_dict_token_budget_error")
            logging.error(f"Error enforcing token budget: {str(e)}")
            matched_entries = []  # Fallback to empty list

        # Alert if token budget exceeded
        try:
            alert_token_budget_exceeded(matched_entries, max_tokens)
        except Exception as e:
            log_counter("chat_dict_token_alert_error")
            logging.error(f"Error in token budget alert: {str(e)}")

        # Apply replacement strategy
        try:
            logging.debug("Chat Dictionary: Applying replacement strategy")
            matched_entries = apply_strategy(matched_entries, strategy)
        except Exception as e:
            log_counter("chat_dict_strategy_error")
            logging.error(f"Error applying strategy: {str(e)}")
            matched_entries = []  # Fallback to empty list

        # Generate output with single replacement per match
        for entry in matched_entries:
            logging.debug("Chat Dictionary: Applying replacements")
            try:
                if entry.max_replacements > 0:
                    user_input, replaced_count = apply_replacement_once(user_input, entry)
                    logging.debug(f"Replaced {replaced_count} occurrences of '{entry.key}' with '{entry.content}'")
                    if replaced_count > 0:
                        entry.max_replacements -= 1
            except Exception as e:
                log_counter("chat_dict_replacement_error", labels={"key": entry.key})
                logging.error(f"Error applying replacement for entry {entry.key}: {str(e)}")
                continue  # Skip this replacement but continue processing others

    except Exception as e:
        log_counter("chat_dict_processing_error")
        logging.error(f"Critical error in process_user_input: {str(e)}")
        # Return original input if critical failure occurs
        return user_input

    return user_input

# Example Usage:
# 1. Load entries from a Markdown file
# entries = parse_user_dict_markdown_file('chat_dict.md')
# 2. Process user input with the entries
# processed_input = process_user_input(user_input, entries)
# print(processed_input)


#
# End of Chat Dictionary functions
#######################################################################################################################


#######################################################################################################################
#
# Character Card Functions

CHARACTERS_FILE = Path('', 'Helper_Scripts', 'Character_Cards', 'Characters.json')


def save_character(
        db: CharactersRAGDB,  # Changed: Pass the DB instance
        character_data: Dict[str, Any],  # Original character data from input
        expected_version: Optional[int] = None  # For updates, if version is known
) -> Optional[int]:  # Returns character_id or None on failure
    log_counter("save_character_attempt")
    start_time = time.time()

    char_name = character_data.get('name')
    if not char_name:
        logging.error("Character name is required to save.")
        return None

    # Prepare data for DB (map/transform if needed)
    db_card_data = {
        'name': char_name,
        'description': character_data.get('description'),
        'personality': character_data.get('personality'),
        'scenario': character_data.get('scenario'),
        'system_prompt': character_data.get('system_prompt', character_data.get('system')),  # common alternative key
        'post_history_instructions': character_data.get('post_history_instructions',
                                                        character_data.get('post_history')),
        'first_message': character_data.get('first_message', character_data.get('mes_example_greeting')),
        'message_example': character_data.get('message_example', character_data.get('mes_example')),
        'creator_notes': character_data.get('creator_notes'),
        'alternate_greetings': character_data.get('alternate_greetings'),  # Should be list or JSON string
        'tags': character_data.get('tags'),  # Should be list or JSON string
        'creator': character_data.get('creator'),
        'character_version': character_data.get('character_version'),
        'extensions': character_data.get('extensions')  # Should be dict or JSON string
    }

    # Handle image: convert base64 to bytes if present
    if 'image' in character_data and character_data['image']:
        try:
            # Assuming character_data['image'] is base64 string. Remove data URL prefix if present.
            img_b64_data = character_data['image']
            if ',' in img_b64_data:  # e.g. data:image/png;base64,xxxxx
                img_b64_data = img_b64_data.split(',', 1)[1]
            db_card_data['image'] = base64.b64decode(img_b64_data)
        except Exception as e_img:
            logging.error(f"Error decoding character image for {char_name}: {e_img}")
            db_card_data['image'] = None  # Or skip setting it
    else:
        db_card_data['image'] = None  # Ensure it's None if not provided or empty

    # Remove None values from db_card_data to avoid inserting NULLs where defaults or existing values are preferred
    # However, CharactersRAGDB add/update methods should handle None for optional fields correctly.
    # db_card_data = {k: v for k, v in db_card_data.items() if v is not None}

    try:
        # Check if character exists for an "upsert" like behavior
        existing_char = db.get_character_card_by_name(char_name)

        char_id = None
        if existing_char:
            logging.info(f"Character '{char_name}' found (ID: {existing_char['id']}). Attempting update.")
            current_db_version = existing_char['version']
            if expected_version is not None and expected_version != current_db_version:
                logging.error(
                    f"Version mismatch for character '{char_name}'. Expected {expected_version}, DB has {current_db_version}.")
                raise ConflictError(
                    f"Version mismatch for update. Expected {expected_version}, got {current_db_version}",
                    entity="character_cards", entity_id=existing_char['id'])

            # Use current_db_version as expected_version for the update call
            # Merge: Ensure fields not in character_data but in existing_char are preserved if desired
            # For now, db_card_data contains all fields to be set.
            # If a field is None in db_card_data, it will be set to NULL in DB.
            # If character_data omits a field, it's None in db_card_data, so it updates to NULL.
            # This is standard update behavior. If you want partial updates (only update provided fields),
            # then db_card_data should only contain non-None values from character_data.

            # Let's make db_card_data only contain fields that are present in the input character_data
            # so it acts as a partial update for existing characters.
            update_payload = {}
            for key, value in character_data.items():  # Iterate over original input data
                if key == 'name': continue  # Name is for lookup, not update here
                if key in db_card_data:  # Check if it's a mapped key
                    # Use the mapped value from db_card_data (e.g. image bytes)
                    update_payload[key] = db_card_data[key]

            if not update_payload:
                logging.info(
                    f"No updatable fields provided for existing character '{char_name}'. Skipping update, but returning ID.")
                char_id = existing_char['id']  # No actual update, but considered "saved"
            elif db.update_character_card(existing_char['id'], update_payload, current_db_version):
                char_id = existing_char['id']
                logging.info(f"Character '{char_name}' (ID: {char_id}) updated successfully.")
            else:  # update_character_card returned False (should raise on error)
                logging.error(
                    f"Update failed for character '{char_name}' (ID: {existing_char['id']}) for unknown reason.")
                # This path should ideally not be hit if update_character_card raises ConflictError or other DB errors

        else:  # Character does not exist, add new
            logging.info(f"Character '{char_name}' not found. Attempting to add new.")
            # For add_character_card, ensure all required fields in db_card_data are present, or handle defaults
            # add_character_card will set client_id, version, timestamps.
            # We must provide 'name'. Other text fields can be None/empty. Image can be None.
            # JSON fields should be None if not provided, or valid JSON string / list / dict.
            char_id = db.add_character_card(db_card_data)
            if char_id:
                logging.info(f"Character '{char_name}' added successfully with ID: {char_id}.")
            else:  # add_character_card returned None (should raise on error)
                logging.error(f"Failed to add new character '{char_name}'.")

        save_duration = time.time() - start_time
        if char_id:
            log_histogram("save_character_duration", save_duration)
            log_counter("save_character_success")
            return char_id
        else:
            # This path means neither update nor add succeeded in setting char_id
            log_counter("save_character_error_unspecified")
            logging.error(f"Save character operation for '{char_name}' did not result in a character ID.")
            return None

    except ConflictError as e_conflict:
        log_counter("save_character_error_conflict", labels={"error": str(e_conflict)})
        logging.error(f"Conflict error saving character '{char_name}': {e_conflict}")
        # Re-raise or return None. For now, return None for simplicity in this wrapper.
        return None
    except (InputError, CharactersRAGDBError) as e_db:
        log_counter("save_character_error_db", labels={"error": str(e_db)})
        logging.error(f"Database error saving character '{char_name}': {e_db}", exc_info=True)
        return None
    except Exception as e_gen:
        log_counter("save_character_error_generic", labels={"error": str(e_gen)})
        logging.error(f"Generic error saving character '{char_name}': {e_gen}", exc_info=True)
        return None


def load_characters(db: CharactersRAGDB) -> Dict[str, Dict[str, Any]]:  # Returns dict keyed by char name
    log_counter("load_characters_attempt")
    start_time = time.time()
    characters_map: Dict[str, Dict[str, Any]] = {}
    try:
        # list_character_cards returns List[Dict[str, Any]]
        all_cards_list = db.list_character_cards(limit=10000)  # Assuming not too many cards for now

        for card_dict in all_cards_list:
            char_name = card_dict.get('name')
            if char_name:
                # Convert image BLOB back to base64 string for compatibility if needed by UI
                if 'image' in card_dict and isinstance(card_dict['image'], bytes):
                    try:
                        # You might want to store the image format or assume one (e.g., png)
                        card_dict['image_base64'] = base64.b64encode(card_dict['image']).decode('utf-8')
                        # del card_dict['image'] # Optionally remove the bytes version
                    except Exception as e_img_enc:
                        logging.warning(f"Could not encode image for character {char_name}: {e_img_enc}")
                        card_dict['image_base64'] = None

                # The old code had 'image_path'. This is not directly stored.
                # If 'image_path' is needed, it implies images are also saved to disk by save_character,
                # which the new DB logic doesn't do (it stores BLOB).
                # For now, 'image_path' won't be present unless explicitly added.

                characters_map[char_name] = card_dict
            else:
                logging.warning(f"Character card found with no name (ID: {card_dict.get('id')}). Skipping.")

        load_duration = time.time() - start_time
        log_histogram("load_characters_duration", load_duration)
        log_counter("load_characters_success", labels={"character_count": len(characters_map)})
        logging.info(f"Loaded {len(characters_map)} characters from DB.")
        return characters_map

    except CharactersRAGDBError as e_db:
        log_counter("load_characters_error_db", labels={"error": str(e_db)})
        logging.error(f"Database error loading characters: {e_db}", exc_info=True)
        return {}
    except Exception as e_gen:
        log_counter("load_characters_error_generic", labels={"error": str(e_gen)})
        logging.error(f"Generic error loading characters: {e_gen}", exc_info=True)
        return {}


def get_character_names(db: CharactersRAGDB) -> List[str]:
    log_counter("get_character_names_attempt")
    start_time = time.time()
    names: List[str] = []
    try:
        all_cards = db.list_character_cards(limit=10000)  # Fetch all, then extract names
        for card in all_cards:
            if card.get('name'):
                names.append(card['name'])

        names.sort()  # Optional: sort names alphabetically

        get_names_duration = time.time() - start_time
        log_histogram("get_character_names_duration", get_names_duration)
        log_counter("get_character_names_success", labels={"name_count": len(names)})
        return names
    except CharactersRAGDBError as e_db:
        log_counter("get_character_names_error_db", labels={"error": str(e_db)})
        logging.error(f"Database error getting character names: {e_db}", exc_info=True)
        return []
    except Exception as e_gen:
        log_counter("get_character_names_error_generic", labels={"error": str(e_gen)})
        logging.error(f"Generic error getting character names: {e_gen}", exc_info=True)
        return []

#
# End of Chat.py
##########################################################################################################################

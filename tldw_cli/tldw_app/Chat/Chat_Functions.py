# Chat_Functions.py
# Description: Chat functions for interacting with the LLMs as chatbots
#
# Imports
import base64
import json
import logging
import os
import random
import re
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
#
# External Imports
#
# Local Imports
from ..Metrics.metrics_logger import log_counter, log_histogram
from ..api.LLM_API_Calls import chat_with_openai, chat_with_anthropic, chat_with_huggingface, \
    chat_with_google, chat_with_openrouter, chat_with_groq, chat_with_deepseek, chat_with_cohere, chat_with_mistral
from ..api.LLM_API_Calls_Local import chat_with_aphrodite, chat_with_kobold, chat_with_oobabooga, \
    chat_with_llama, chat_with_tabbyapi, chat_with_local_llm, chat_with_ollama, chat_with_vllm, chat_with_custom_openai, \
    chat_with_custom_openai_2
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

        # --- Routing Logic ---
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
                topp=topp, # Pass 'topp'
                topk=topk  # Pass 'topk'
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
                topp=topp, # Pass 'topp'
                topk=topk  # Pass 'topk'
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
                topp=topp # Pass 'topp'
            )

        elif endpoint_lower == "mistral":
            response = chat_with_mistral(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt, # Map 'prompt'
                temp=temp,
                system_message=system_message,
                streaming=streaming,
                topp=topp, # Pass 'topp'
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
                topp=topp, # Pass 'topp'
                topk=topk  # Pass 'topk'
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
        return response

    except Exception as e:
        # --- Error Logging and Return ---
        call_duration = time.time() - start_time
        log_histogram("chat_api_call_duration", call_duration, labels={"api_endpoint": endpoint_lower}) # Log duration even on error
        log_counter("chat_api_call_error", labels={"api_endpoint": endpoint_lower, "error_type": type(e).__name__, "error_message": str(e)})
        logging.error(f"Error in chat_api_call for endpoint '{endpoint_lower}': {str(e)}", exc_info=True) # Include traceback
        # Return a user-friendly error message
        return f"An error occurred while calling the '{endpoint_lower}' API: {str(e)}"


def chat(message, history, media_content, selected_parts, api_endpoint, api_key, custom_prompt, temperature,
         system_message=None, streaming=False, minp=None, maxp=None, model=None, topp=None, topk=None, chatdict_entries=None, max_tokens=500, strategy="sorted_evenly"):
    # FIXME - move to chat_api_call, and add another metric_name for chat lengths (track entire convo)
    log_counter("chat_attempt", labels={"api_endpoint": api_endpoint})
    start_time = time.time()
    try:
        logging.info(f"Debug - Chat Function - Message: {message}")
        logging.info(f"Debug - Chat Function - Media Content: {media_content}")
        logging.info(f"Debug - Chat Function - Selected Parts: {selected_parts}")
        logging.info(f"Debug - Chat Function - API Endpoint: {api_endpoint}")
        # logging.info(f"Debug - Chat Function - Prompt: {prompt}")

        # --- Input Processing ---
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
            loaded_config_data = "loaded_config_data" #FIXME - load_and_log_configs()
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



def extract_media_name(media_content):
    if isinstance(media_content, dict):
        content = media_content.get('content', {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                logging.warning("Failed to parse media_content JSON string")
                return None

        # Try to extract title from the content
        if isinstance(content, dict):
            return content.get('title') or content.get('name')

    logging.warning(f"Unexpected media_content format: {type(media_content)}")
    return None

#
# End of Chat functions
#######################################################################################################################


#######################################################################################################################
#
# Chat Dictionary Functions
def parse_user_dict_markdown_file(file_path):
    """
    Parse a Markdown-like file with key-value pairs.
    Supports multi-line values indicated by 'key: |' and terminated by '---@@@---'.

    Args:
        file_path (str): The path to the file to parse.

    Returns:
        dict: A dictionary containing the parsed key-value pairs.
              Multi-line values are stored as single strings with preserved newlines.
    """
    logging.debug(f"Parsing user dictionary file: {file_path}")
    replacement_dict = {}
    current_key = None
    current_value = []
    # Regex for the multi-line termination symbol (allows surrounding whitespace)
    termination_pattern = re.compile(r'^\s*---@@@---\s*$')
    # Regex for key-value pairs. Allows for optional value part.
    # Group 1: Key (non-greedy, anything before ':')
    # Group 2: Value (everything after ':', can be empty)
    key_value_pattern = re.compile(r'^\s*([^:\n]+?)\s*:\s*(.*?)\s*$')

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1): # Use enumerate for better error messages

                # 1. Check for termination pattern FIRST
                if termination_pattern.match(line):
                    if current_key:
                        # Finish the current multi-line value
                        replacement_dict[current_key] = '\n'.join(current_value).strip()
                        logging.debug(f"Line {line_num}: Ended multi-line value for key '{current_key}'")
                        current_key, current_value = None, []
                    else:
                        # Termination symbol found unexpectedly
                        logging.warning(f"Line {line_num}: Found termination symbol '---@@@---' outside of a multi-line value block. Ignoring.")
                    continue # Process next line

                # 2. If currently processing a multi-line value, append the line
                if current_key:
                    # Append the raw line, stripping only the trailing newline/CR
                    # Preserves leading/trailing whitespace on the line itself
                    cleaned_line = line.rstrip('\n\r')
                    current_value.append(cleaned_line)
                    continue # Process next line

                # 3. If not in multi-line mode, try to match a new key-value pair
                key_value_match = key_value_pattern.match(line)
                if key_value_match:
                    # *** THE FIX IS HERE: Check if key_value_match is not None ***
                    key, value = key_value_match.groups()
                    key = key.strip()
                    value_stripped = value.strip() # Check the stripped value

                    if not key: # Skip if key is empty after stripping
                         logging.warning(f"Line {line_num}: Ignoring line with empty key: {line.strip()}")
                         continue

                    if value_stripped == '|':
                        # Start of a multi-line value
                        if current_key: # Should not happen based on logic order, but good practice
                             logging.warning(f"Line {line_num}: Started new multi-line key '{key}' while previous key '{current_key}' was active. Finishing previous.")
                             replacement_dict[current_key] = '\n'.join(current_value).strip()

                        current_key = key
                        current_value = []
                        logging.debug(f"Line {line_num}: Started multi-line value for key '{key}'")
                    else:
                        # Single-line value (or empty value)
                        replacement_dict[key] = value_stripped # Use the stripped value part
                        logging.debug(f"Line {line_num}: Parsed single-line key='{key}', value='{value_stripped}'")

                # else: # Optional: Handle lines that don't match any expected format
                    # if line.strip(): # Only log if the line isn't just whitespace
                    #     logging.warning(f"Line {line_num}: Ignoring line, does not match key: value pattern or multi-line continuation: {line.strip()}")


        # Handle any remaining multi-line value at End Of File
        if current_key:
            logging.debug(f"EOF: Finalizing multi-line value for key '{current_key}'")
            replacement_dict[current_key] = '\n'.join(current_value).strip()

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return {} # Return empty dict on error
    except Exception as e:
        logging.error(f"An error occurred while parsing {file_path}: {e}")
        return {} # Return empty dict on error


    logging.debug(f"Parsed {len(replacement_dict)} entries from {file_path}") # More informative log
    # logging.debug(f"Parsed entries: {replacement_dict}") # Can be very verbose
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
def group_scoring(entries):
    logging.debug(f"Group scoring for {len(entries)} entries")
    grouped_entries = {}
    for entry in entries:
        grouped_entries.setdefault(entry.group, []).append(entry)

    selected_entries = []
    for group, group_entries in grouped_entries.items():
        selected_entries.append(max(group_entries, key=lambda e: len(re.findall(e.key, e.content)) if e.key else 0))

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

def save_character(character_data):
    log_counter("save_character_attempt")
    start_time = time.time()
    characters_file = os.path.join(os.path.dirname(__file__), '..', 'Helper_Scripts', 'Character_Cards', 'Characters.json')
    characters_dir = os.path.dirname(characters_file)

    try:
        if os.path.exists(characters_file):
            with open(characters_file, 'r') as f:
                characters = json.load(f)
        else:
            characters = {}

        char_name = character_data['name']

        # Save the image separately if it exists
        if 'image' in character_data:
            img_data = base64.b64decode(character_data['image'])
            img_filename = f"{char_name.replace(' ', '_')}.png"
            img_path = os.path.join(characters_dir, img_filename)
            with open(img_path, 'wb') as f:
                f.write(img_data)
            character_data['image_path'] = os.path.abspath(img_path)
            del character_data['image']  # Remove the base64 image data from the JSON

        characters[char_name] = character_data

        with open(characters_file, 'w') as f:
            json.dump(characters, f, indent=2)

        save_duration = time.time() - start_time
        log_histogram("save_character_duration", save_duration)
        log_counter("save_character_success")
        logging.info(f"Character '{char_name}' saved successfully.")
    except Exception as e:
        log_counter("save_character_error", labels={"error": str(e)})
        logging.error(f"Error saving character: {str(e)}")


def load_characters():
    log_counter("load_characters_attempt")
    start_time = time.time()
    try:
        characters_file = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..', '..',
            'Helper_Scripts', 'Character_Cards', 'Characters.json'
        ))
        if os.path.exists(characters_file):
            with open(characters_file, 'r') as f:
                characters = json.load(f)
            logging.info(f"Loaded characters from {characters_file}")
            logging.info(f"Loaded {len(characters)} characters from {characters_file}")
            load_duration = time.time() - start_time
            log_histogram("load_characters_duration", load_duration)
            log_counter("load_characters_success", labels={"character_count": len(characters)})
            return characters
        else:
            logging.warning(f"Characters file not found: {characters_file}")
            return {}
    except Exception as e:
        log_counter("load_characters_error", labels={"error": str(e)})
        return {}


def get_character_names():
    log_counter("get_character_names_attempt")
    start_time = time.time()
    try:
        characters = load_characters()
        names = list(characters.keys())
        get_names_duration = time.time() - start_time
        log_histogram("get_character_names_duration", get_names_duration)
        log_counter("get_character_names_success", labels={"name_count": len(names)})
        return names
    except Exception as e:
        log_counter("get_character_names_error", labels={"error": str(e)})
        logging.error(f"Error getting character names: {str(e)}")
        return []

#
# End of Chat.py
##########################################################################################################################

# Local_LLM_API_Calls_Lib.py
#########################################
# Local LLM API Calls Library
# This library is used to perform 'Local' API calls to LLM endpoints.
#
####
# Imports
import json
import logging
import os
from typing import Any, Generator, Union, List, Dict, Optional
#
# 3rd-Party Imports
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from tldw_cli.tldw_app.config import get_setting


#
# Local Imports
#
#
####################
# Function List
# FIXME - UPDATE
# 1. chat_with_local_llm(text, custom_prompt_arg)
# 2. chat_with_llama(api_url, text, token, custom_prompt)
# 3. chat_with_kobold(api_url, text, kobold_api_token, custom_prompt)
# 4. chat_with_oobabooga(api_url, text, ooba_api_token, custom_prompt)
# 5. chat_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg)
# 6. chat_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt)
# 7. save_summary_to_file(summary, file_path)
#
#
####################
# Import necessary libraries
# Import Local
#
#######################################################################################################################
# Function Definitions
#

def _safe_cast(value: Any, cast_to: type, default: Any = None) -> Any:
    """Safely casts value to specified type, returning default on failure."""
    if value is None:
        return default
    try:
        # Handle boolean strings explicitly
        if cast_to is bool and isinstance(value, str):
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            else:
                # Try casting numeric strings to bool (e.g., "1", "0")
                try:
                    return bool(int(value))
                except ValueError:
                    pass # Fall through to general error if not "true"/"false" or numeric string
        return cast_to(value)
    except (ValueError, TypeError):
        logging.warning(f"Could not cast '{value}' to {cast_to}. Using default: {default}")
        return default

def extract_text_from_segments(segments: List[Dict]) -> str:
    # (Keep existing implementation)
    logging.debug(f"Segments received: {segments}")
    logging.debug(f"Type of segments: {type(segments)}")
    text = ""
    if isinstance(segments, list):
        for segment in segments:
            # logging.debug(f"Current segment: {segment}") # Can be verbose
            # logging.debug(f"Type of segment: {type(segment)}")
            if isinstance(segment, dict) and 'Text' in segment:
                text += segment['Text'] + " "
            elif isinstance(segment, dict) and 'text' in segment: # Adding flexibility for key case
                 text += segment['text'] + " "
            else:
                logging.warning(f"Skipping segment due to missing 'Text' key or wrong type: {segment}")
    elif isinstance(segments, str): # Allow passing a pre-joined string
        logging.debug("Segments received as a single string.")
        text = segments
    else:
        logging.warning(f"Unexpected type of 'segments': {type(segments)}. Trying to convert to string.")
        text = str(segments) # Attempt conversion

    return text.strip()

def chat_with_local_llm(input_data, custom_prompt_arg, temp, system_message=None, streaming=False, top_k=None, top_p=None, min_p=None):
    try:
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Local LLM: Loading json data for Chat request")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("Local LLM: Using provided string data for Chat request")
            data = input_data

        logging.debug(f"Local LLM: Loaded data: {data}")
        logging.debug(f"Local LLM: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Local LLM: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Invalid input data format")

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)
        elif streaming is None:
            streaming = False

        if isinstance(top_k, int):
            top_k = int(top_k)
            logging.debug(f"Local LLM: Using top_k: {top_k}")
        elif top_k is None:
            top_k = "load_and_log_configs" # FIXME - load_and_log_configs().get('local_llm', {}).get('top_k', 100)
            logging.debug(f"Local LLM: Using top_k from config: {top_k}")

        if isinstance(top_p, float):
            top_p = float(top_p)
            logging.debug(f"Local LLM: Using top_p: {top_p}")
        elif top_p is None:
            top_p = "load_and_log_configs" # FIXME - load_and_log_configs().get('local_llm', {}).get('top_p', 0.95)
            logging.debug(f"Local LLM: Using top_p from config: {top_p}")

        if isinstance(min_p, float):
            min_p = float(min_p)
            logging.debug(f"Local LLM: Using min_p: {min_p}")
        elif min_p is None:
            min_p = "load_and_log_configs" # FIXME - load_and_log_configs().get('local_llm', {}).get('min_p', 0.05)
            logging.debug(f"Local LLM: Using min_p from config: {min_p}")

        local_llm_system_message = "You are a helpful AI assistant."

        if system_message is None:
            system_message = local_llm_system_message

        local_llm_max_tokens = 4096

        headers = {
            'Content-Type': 'application/json'
        }

        logging.debug("Local LLM: Preparing data + prompt for submittal")
        local_llm_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": local_llm_prompt
                }
            ],
            "max_tokens": local_llm_max_tokens,
            "temperature": temp,
            "stream": streaming,
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p
        }

        local_api_timeout = "loaded_and_config_data" # FIXME - loaded_config_data['local_llm']['api_timeout']
        logging.debug("Local LLM: Posting request")
        response = requests.post('http://127.0.0.1:8080/v1/chat/completions', headers=headers, json=data, timeout=local_api_timeout)

        if response.status_code == 200:
            if streaming:
                logging.debug("Local LLM: Processing streaming response")

                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith('data:'):
                                data_str = decoded_line[len('data:'):].strip()
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    if 'choices' in data_json and len(data_json['choices']) > 0:
                                        delta = data_json['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            content = delta['content']
                                            yield content
                                except json.JSONDecodeError:
                                    logging.error(f"Local LLM: Error decoding JSON from line: {decoded_line}")
                                    continue
                return stream_generator()
            else:
                logging.debug("Local LLM: Processing non-streaming response")
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Local LLM: Summarization successful")
                    logging.info("Local LLM: Summarization successful.")
                    return summary
                else:
                    logging.warning("Local LLM: Summary not found in the response data")
                    return "Local LLM: Summary not available"
        else:
            logging.error(f"Local LLM: Request failed with status code {response.status_code}")
            print("Local LLM: Failed to process Chat response:", response.text)
            return f"Local LLM: Failed to process Chat response, status code {response.status_code}"
    except Exception as e:
        logging.debug(f"Local LLM: Error in processing: {str(e)}")
        print("Error occurred while processing Chat request with Local LLM:", str(e))
        return f"Local LLM: Error occurred while processing Chat response: {str(e)}"


# --- Refactored chat_with_llama ---
def chat_with_llama(input_data, custom_prompt, temp=None, api_url=None, api_key=None, system_prompt=None, streaming=None, top_k=None, top_p=None, min_p=None):
    """Interacts with the Llama.cpp server API using settings from config."""
    log = logging.getLogger(__name__)
    provider_section_key = "llama_cpp" # Key used in [api_settings.*] in config.toml

    try:
        # --- Load Settings ---

        # API Key: Priority -> argument > env var > config fallback
        api_key_value = api_key # Prioritize argument
        if not api_key_value:
            api_key_env_var_name = get_setting("api_settings", f"{provider_section_key}.api_key_env_var", "LLAMA_CPP_API_KEY")
            api_key_value = os.environ.get(api_key_env_var_name)
            if api_key_value:
                logging.info(f"Llama.cpp: Using API key from environment variable {api_key_env_var_name}")
            # else: # Optional fallback (less secure)
            #     api_key_value = get_setting("api_settings", f"{provider_section_key}.api_key")
            #     if api_key_value: logging.warning("Llama.cpp: Using API key from config file.")

        if api_key_value:
            logging.debug(f"Llama.cpp: Using API Key: {api_key_value[:5]}...{api_key_value[-5:]}")
        else:
            logging.info("Llama.cpp: No API key provided or found. Proceeding without Authorization header.")
            api_key_value = "" # Ensure it's a string for header logic later

        # API URL: Priority -> argument > config default
        api_url_value = api_url
        if not api_url_value:
            api_url_value = get_setting("api_settings", f"{provider_section_key}.api_url", "http://localhost:8080/completion") # Default endpoint
        logging.debug(f"Llama.cpp: Using API URL: {api_url_value}")

        if not api_url_value:
            logging.error("Llama.cpp: API URL is required but not provided or configured.")
            return "Llama.cpp: API URL not found or is empty."

        # Streaming: Priority -> argument > config default
        if streaming is None:
            streaming_cfg = get_setting("api_settings", f"{provider_section_key}.streaming", False)
            streaming_value = _safe_cast(streaming_cfg, bool, False)
        else:
            streaming_value = _safe_cast(streaming, bool, False)
        logging.debug(f"Llama.cpp: Streaming mode: {streaming_value}")
        # Keep validation (good practice)
        if not isinstance(streaming_value, bool):
             raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming_value).__name__}")

        # Temperature: Priority -> argument > config default
        if temp is None:
            temp_cfg = get_setting("api_settings", f"{provider_section_key}.temperature", 0.7)
            temp_value = _safe_cast(temp_cfg, float, 0.7)
        else:
             temp_value = _safe_cast(temp, float, 0.7)
        logging.debug(f"Llama.cpp: Using temperature: {temp_value}")

        # Top K: Priority -> argument > config default
        if top_k is None:
            top_k_cfg = get_setting("api_settings", f"{provider_section_key}.top_k", 40)
            top_k_value = _safe_cast(top_k_cfg, int, 40)
        else:
            top_k_value = _safe_cast(top_k, int, 40)
        logging.debug(f"Llama.cpp: Using top_k: {top_k_value}")
        if not isinstance(top_k_value, int):
             raise ValueError(f"Invalid type for 'top_k': Expected an int, got {type(top_k_value).__name__}")

        # Top P: Priority -> argument > config default
        if top_p is None:
            top_p_cfg = get_setting("api_settings", f"{provider_section_key}.top_p", 0.95)
            top_p_value = _safe_cast(top_p_cfg, float, 0.95)
        else:
            top_p_value = _safe_cast(top_p, float, 0.95)
        logging.debug(f"Llama.cpp: Using top_p: {top_p_value}")
        if not isinstance(top_p_value, float):
             raise ValueError(f"Invalid type for 'top_p': Expected a float, got {type(top_p_value).__name__}")

        # Min P: Priority -> argument > config default
        if min_p is None:
            min_p_cfg = get_setting("api_settings", f"{provider_section_key}.min_p", 0.05)
            min_p_value = _safe_cast(min_p_cfg, float, 0.05)
        else:
            min_p_value = _safe_cast(min_p, float, 0.05)
        logging.debug(f"Llama.cpp: Using min_p: {min_p_value}")
        if not isinstance(min_p_value, float):
            raise ValueError(f"Invalid type for 'min_p': Expected a float, got {type(min_p_value).__name__}")

        # System Prompt: Priority -> argument > config default (optional) > hardcoded default
        system_prompt_value = system_prompt
        if system_prompt_value is None:
            # system_prompt_value = get_setting("api_settings", f"{provider_section_key}.system_prompt", "You are a helpful AI assistant.") # Optionally load from config
             system_prompt_value = "You are a helpful AI assistant." # Default if not passed or configured
        logging.debug(f"Llama.cpp: Using system prompt: {system_prompt_value[:100]}...")

        # Max Tokens: Load from config
        max_tokens_cfg = get_setting("api_settings", f"{provider_section_key}.max_tokens", 4096)
        max_tokens_value = _safe_cast(max_tokens_cfg, int, 4096)
        logging.debug(f"Llama.cpp: Using max_tokens (n_predict): {max_tokens_value}")

        # Timeout, Retries, Delay: Load from config
        timeout_cfg = get_setting("api_settings", f"{provider_section_key}.timeout", 300)
        api_timeout = _safe_cast(timeout_cfg, int, 300)
        retries_cfg = get_setting("api_settings", f"{provider_section_key}.retries", 1)
        retry_count = _safe_cast(retries_cfg, int, 1)
        delay_cfg = get_setting("api_settings", f"{provider_section_key}.retry_delay", 2)
        retry_delay = _safe_cast(delay_cfg, (int, float), 2) # Factor can be float
        logging.debug(f"Llama.cpp: Timeout={api_timeout}, Retries={retry_count}, Delay={retry_delay}")


        # --- Prepare Request ---
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        # Add Authorization header only if a key was provided and is non-empty
        if api_key_value:
            headers['Authorization'] = f'Bearer {api_key_value}'
            logging.debug("Llama.cpp: Added Authorization header.")

        # Combine prompts (adjust if your server uses different fields)
        # The original code put custom_prompt first, then input_data. Adapt if needed.
        # Common pattern: System prompt is separate, user prompt contains task + input data.
        combined_user_prompt = f"{custom_prompt}\n\n---\n\n{input_data}"
        logging.debug(f"Llama.cpp: Combined User Prompt (first 500): {combined_user_prompt[:500]}...")


        # Prepare data payload according to llama.cpp server API schema
        # Check your specific llama.cpp server version/docs for exact parameters
        # Common parameters for /completion endpoint:
        data = {
            "prompt": combined_user_prompt,    # The user's full prompt
            "system_prompt": system_prompt_value, # If supported by your endpoint/model template
            'temperature': temp_value,
            'top_k': top_k_value,
            'top_p': top_p_value,
            'min_p': min_p_value,
            'n_predict': max_tokens_value,  # Max tokens to generate
            'stream': streaming_value,
            # --- Optional parameters (uncomment/adjust based on your server config) ---
            # 'n_keep': 0,           # Tokens from prompt to keep in context
            # 'stop': ["\n", "User:"], # List of stop strings
            # 'tfs_z': 1.0,
            # 'typical_p': 1.0,
            # 'repeat_penalty': 1.1,
            # 'repeat_last_n': 64,
            # 'presence_penalty': 0.0,
            # 'frequency_penalty': 0.0,
            # 'mirostat': 0,          # Mirostat sampling mode (0=off, 1=v1, 2=v2)
            # 'mirostat_tau': 5.0,
            # 'mirostat_eta': 0.1,
            # 'grammar': '',          # GBNF grammar string
            # 'seed': -1,             # Random seed (-1 for random)
            # 'ignore_eos': False,    # Ignore End-Of-Sequence token
            # 'logit_bias': [],       # Adjust likelihood of specific tokens [[token_id, bias], ...]
        }
        # Remove None values potentially introduced if args were None initially
        # Though defaults should prevent this now.
        # data = {k: v for k, v in data.items() if v is not None}

        logging.debug(f"Llama.cpp: Sending data payload: {{k: v for k, v in data.items() if k not in ['prompt', 'system_prompt']}}") # Avoid logging full prompt


        # --- Execute Request ---
        session = requests.Session()
        retry_strategy = Retry(
            total=retry_count,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504], # Retry on server errors too
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # Mount for both http and https, although local is likely http
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        logging.debug(f"Llama.cpp: Submitting request to {api_url_value} with timeout {api_timeout}s")

        response = session.post(
            api_url_value,
            headers=headers,
            json=data,
            stream=streaming_value, # Use the validated boolean value
            timeout=api_timeout
        )
        logging.debug(f"Llama.cpp: Response Status Code: {response.status_code}")

        # --- Process Response ---
        if response.status_code == 200:
            if streaming_value:
                logging.debug("Llama.cpp: Processing streaming response")

                # --- Stream Generator ---
                # Note: llama.cpp stream format might differ slightly from OpenAI's
                # Check the specific format your server sends. Common format is SSE `data: {...}`
                def stream_generator():
                    buffer = ""
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            # Standard SSE format check
                            if decoded_line.startswith('data:'):
                                data_str = decoded_line[len('data:'):].strip()
                                if data_str == '[DONE]': # OpenAI specific, llama.cpp might not send this
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    # Extract content based on llama.cpp's JSON structure for /completion stream
                                    content_chunk = data_json.get('content', '')
                                    if content_chunk:
                                        yield content_chunk
                                    # Check if generation stopped (llama.cpp specific field)
                                    if data_json.get('stop', False) or data_json.get('stopped_eos', False) or data_json.get('stopped_word', False):
                                         logging.debug(f"Llama.cpp stream stopped. Reason: {data_json}")
                                         break
                                except json.JSONDecodeError:
                                    logging.error(f"Llama.cpp: Error decoding JSON from stream line: {decoded_line}")
                                    continue
                                except Exception as e_stream:
                                     logging.error(f"Llama.cpp: Error processing stream chunk {data_str}: {e_stream}")
                                     continue # Try next line
                            # Handle potential non-SSE lines if necessary
                            # else: logging.warning(f"Llama.cpp: Received non-SSE line in stream: {decoded_line}")
                return stream_generator()
            else:
                # --- Non-Streaming Response ---
                try:
                    response_data = response.json()
                    # logging.debug(f"Llama.cpp: Non-streaming Response Data: {response_data}") # Careful logging
                    # Extract content based on llama.cpp's JSON structure for /completion
                    generated_text = response_data.get('content', '').strip()
                    if generated_text:
                        logging.info("Llama.cpp: Chat request successful (non-streaming).")
                        # logging.debug(f"Llama.cpp: Generated text: {generated_text[:200]}...")
                        return generated_text
                    else:
                        logging.warning("Llama.cpp: 'content' field not found or empty in response.")
                        return "Llama.cpp: No content found in response."
                except json.JSONDecodeError:
                    logging.error(f"Llama.cpp: Failed to decode JSON response: {response.text[:500]}...")
                    return "Llama.cpp: Failed to parse JSON response."
        else:
            # --- Handle HTTP Errors ---
            error_msg = f"Llama.cpp: API request failed. Status: {response.status_code}, Body: {response.text[:500]}..."
            logging.error(error_msg)
            return f"Llama.cpp: API Request Failed ({response.status_code})"

    # --- Exception Handling ---
    except requests.exceptions.Timeout:
        logging.error(f"Llama.cpp: Request timed out after {api_timeout} seconds connecting to {api_url_value}.")
        return f"Llama.cpp: Request Timed Out ({api_url_value})"
    except requests.exceptions.RequestException as e:
        logging.error(f"Llama.cpp: RequestException: {e}", exc_info=True)
        return f"Llama.cpp: Network or Request Error: {e}"
    except ValueError as e: # Catch specific validation errors
         logging.error(f"Llama.cpp: Configuration or Value Error: {e}", exc_info=True)
         return f"Llama.cpp: Invalid configuration value: {e}"
    except Exception as e:
        logging.error(f"Llama.cpp: Unexpected error: {e}", exc_info=True)
        return f"Llama.cpp: Unexpected error occurred: {e}"


# System prompts not supported through API requests.
# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def chat_with_kobold(input_data, api_key=None, custom_prompt_input=None, temp=None, system_message=None, streaming=None, top_k=None, top_p=None):
    """Interacts with the Kobold API using settings from config."""
    log = logging.getLogger(__name__)
    provider_section_key = "koboldcpp" # Key used in [api_settings.*] in config.toml
    logging.debug(f"Kobold ({provider_section_key}): Chat request process starting...")

    try:
        # --- Load Settings ---
        # API Key: Check argument, then config (though Kobold rarely uses keys)
        kobold_api_key = api_key # Prioritize argument
        if not kobold_api_key:
            # Kobold doesn't typically use env vars, check config directly if needed
            # key_from_config = get_setting("api_settings", f"{provider_section_key}.api_key")
            # if key_from_config:
            #    kobold_api_key = key_from_config
            #    logging.info("Kobold: Using API key from config file (uncommon for Kobold).")
            pass # No key needed usually
        if kobold_api_key:
             logging.warning(f"Kobold: API Key provided ('{kobold_api_key[:2]}...'), but Kobold usually doesn't require one.")
        else:
             logging.debug("Kobold: No API Key provided (as expected).")

        # API URL: Load from config
        # IMPORTANT: Defaulting to the standard non-streaming endpoint
        default_api_url = "http://localhost:5001/api/v1/generate"
        kobold_api_url = get_setting("api_settings", f"{provider_section_key}.api_url", default_api_url)
        if not isinstance(kobold_api_url, str) or not kobold_api_url.startswith(('http://', 'https://')):
             logging.error(f"Kobold: Invalid API URL configured: '{kobold_api_url}'. Falling back to default: {default_api_url}")
             kobold_api_url = default_api_url
        logging.debug(f"Kobold: Using API URL: {kobold_api_url}")

        # Streaming: Priority -> argument > config default. FORCE TO FALSE LATER.
        requested_streaming = streaming # Store the user's request
        if requested_streaming is None:
            streaming_cfg = get_setting("api_settings", f"{provider_section_key}.streaming", False)
            requested_streaming = bool(streaming_cfg) if not isinstance(streaming_cfg, str) else streaming_cfg.lower() == "true"
        else:
             requested_streaming = bool(requested_streaming)
        logging.debug(f"Kobold: Requested streaming: {requested_streaming}")

        # --- FORCE STREAMING OFF ---
        # As noted in original code and common knowledge, Kobold streaming is non-standard.
        # Force it off for reliability using the standard endpoint.
        use_streaming = False
        if requested_streaming:
            logging.warning("Kobold: Streaming requested, but forcing OFF due to non-standard Kobold streaming API. Using non-streaming endpoint.")
        logging.debug(f"Kobold: Effective streaming mode: {use_streaming}")


        # Temperature: Priority -> argument > config default
        if temp is None:
            temp_cfg = get_setting("api_settings", f"{provider_section_key}.temperature", 0.7)
            loaded_temp = _safe_cast(temp_cfg, float, 0.7)
        else:
             loaded_temp = _safe_cast(temp, float, 0.7)
        logging.debug(f"Kobold: Using temperature: {loaded_temp}")

        # Top K: Priority -> argument > config default
        if top_k is None:
            top_k_cfg = get_setting("api_settings", f"{provider_section_key}.top_k", 50)
            loaded_top_k = _safe_cast(top_k_cfg, int, 50)
        else:
             loaded_top_k = _safe_cast(top_k, int, 50)
        # Kobold API might expect 0 to disable, Textual default might be different
        if loaded_top_k <= 0:
             logging.debug("Kobold: Top-K is <= 0, effectively disabling it.")
             # loaded_top_k = 0 # Ensure it's exactly 0 if needed
        logging.debug(f"Kobold: Using top_k: {loaded_top_k}")

        # Top P: Priority -> argument > config default
        if top_p is None:
            top_p_cfg = get_setting("api_settings", f"{provider_section_key}.top_p", 0.9)
            loaded_top_p = _safe_cast(top_p_cfg, float, 0.9)
        else:
             loaded_top_p = _safe_cast(top_p, float, 0.9)
        logging.debug(f"Kobold: Using top_p: {loaded_top_p}")

        # Max Tokens: Load from config (map to max_context_length)
        max_tokens_cfg = get_setting("api_settings", f"{provider_section_key}.max_tokens", 2048)
        kobold_max_tokens = _safe_cast(max_tokens_cfg, int, 2048)
        logging.debug(f"Kobold: Using max_tokens (for max_context_length): {kobold_max_tokens}")

        # System Message: Log that it's likely not used by standard Kobold endpoint
        if system_message:
            logging.warning(f"Kobold: System message provided, but standard Kobold API '/generate' may not use it: '{system_message[:100]}...'")

        # Timeout, Retries, Delay: Load from config
        timeout_cfg = get_setting("api_settings", f"{provider_section_key}.timeout", 300)
        api_timeout = _safe_cast(timeout_cfg, int, 300)
        retries_cfg = get_setting("api_settings", f"{provider_section_key}.retries", 1)
        retry_count = _safe_cast(retries_cfg, int, 1)
        delay_cfg = get_setting("api_settings", f"{provider_section_key}.retry_delay", 2)
        retry_delay = _safe_cast(delay_cfg, float, 2.0) # Use float for backoff_factor
        logging.debug(f"Kobold: Timeout={api_timeout}, Retries={retry_count}, DelayFactor={retry_delay}")


        # --- Input Data Processing ---
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug(f"Kobold: Loading json data from file: {input_data}")
            try:
                with open(input_data, 'r', encoding='utf-8') as file:
                    data = json.load(file)
            except json.JSONDecodeError as e:
                 logging.error(f"Kobold: Failed to decode JSON from {input_data}: {e}. Treating as plain text.")
                 # Fallback: read as text if JSON fails
                 with open(input_data, 'r', encoding='utf-8') as file:
                     data = file.read()
            except Exception as e:
                 logging.error(f"Kobold: Failed to read file {input_data}: {e}. Returning error.")
                 return f"Kobold: Error reading input file: {e}"
        else:
            logging.debug("Kobold: Using provided input data directly.")
            data = input_data

        # Check for pre-existing summary or extract text
        if isinstance(data, dict) and 'summary' in data:
            logging.debug("Kobold: Summary already exists in the loaded data.")
            return data['summary']
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            logging.error(f"Kobold: Invalid input data format after loading: {type(data)}")
            raise ValueError("Kobold: Invalid input data format")
        logging.debug(f"Kobold: Text for prompt (first 500 chars): {text[:500]}...")

        # --- Prepare Request ---
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        # No Authorization header needed typically

        # Combine custom prompt (if any) with the extracted text
        kobold_prompt = f"{custom_prompt_input}\n\n{text}" if custom_prompt_input else text
        logging.debug(f"Kobold: Final prompt being sent (first 500 chars): {kobold_prompt[:500]}...")

        payload = {
            "prompt": kobold_prompt,
            "temperature": loaded_temp,
            "top_p": loaded_top_p,
            "top_k": loaded_top_k,
            # "rep_penalty": 1.0, # Example of other Kobold params if needed
            "max_context_length": kobold_max_tokens, # Map config max_tokens
             # "stream": use_streaming, # Explicitly OMIT stream for standard endpoint
             # Add other specific Kobold params here if necessary
        }
        # Optionally remove top_k if it's 0, depending on API behavior
        # if loaded_top_k <= 0:
        #     payload.pop("top_k", None)


        # --- Execute Request (Non-Streaming Only) ---
        logging.info(f"Kobold: Submitting non-streaming request to {kobold_api_url}")
        session = requests.Session()
        retry_strategy = Retry(
            total=retry_count,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504], # Added 500 Internal Server Error
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # Mount for http only, as Kobold is usually local
        session.mount("http://", adapter)

        response = session.post(
            kobold_api_url, headers=headers, json=payload, timeout=api_timeout
        )
        logging.debug(f"Kobold: Response Status Code: {response.status_code}")

        if response.status_code == 200:
            try:
                response_data = response.json()
                # logging.debug(f"Kobold: Raw Response Data: {response_data}") # Can be large
            except json.JSONDecodeError as e:
                logging.error(f"Kobold: Error parsing JSON response: {e}. Response text: {response.text[:500]}...")
                return f"Kobold: Error parsing JSON response."

            # Check Kobold's specific response structure
            if response_data and 'results' in response_data and isinstance(response_data['results'], list) and len(response_data['results']) > 0:
                 result_text = response_data['results'][0].get('text', '').strip()
                 if result_text:
                      logging.info("Kobold: Chat request successful.")
                      # logging.debug(f"Kobold: Returning text: {result_text[:200]}...")
                      return result_text # Return the string directly
                 else:
                      logging.warning("Kobold: 'text' field missing or empty in results.")
                      return "Kobold: Response missing 'text' field."
            else:
                logging.error(f"Kobold: Expected 'results' structure not found in API response: {response_data}")
                return "Kobold: Unexpected response structure from API."
        else:
            logging.error(f"Kobold: API request failed. Status: {response.status_code}, Body: {response.text[:500]}...")
            return f"Kobold: API request failed. Status: {response.status_code}"

    # --- Exception Handling ---
    except requests.exceptions.RequestException as e:
        logging.error(f"Kobold: RequestException: {e}", exc_info=True)
        return f"Kobold: Network or Request Error: {e}"
    except ValueError as e: # Catch specific ValueErrors like invalid input format
         logging.error(f"Kobold: ValueError: {e}", exc_info=True)
         return f"Kobold: Data processing error: {e}"
    except Exception as e:
        logging.error(f"Kobold: Unexpected error: {e}", exc_info=True)
        return f"Kobold: Unexpected error occurred: {e}"


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
def chat_with_oobabooga(input_data, api_key=None, custom_prompt=None, system_prompt=None, api_url=None, streaming=None, temp=None, top_p=None):
    """Interacts with an Oobabooga API endpoint using settings from config."""
    logging.debug("Oobabooga: Chat process starting...")
    provider_section_key = "oobabooga" # Key used in [api_settings.*] in config.toml

    try:
        # --- Load Settings ---
        # API Key: Priority -> argument > env var (name from config) > config file key (fallback)
        ooba_api_key_to_use = api_key # Prioritize argument
        if not ooba_api_key_to_use:
            api_key_env_var_name = get_setting("api_settings", f"{provider_section_key}.api_key_env_var", "OOBABOOGA_API_KEY")
            ooba_api_key_to_use = os.environ.get(api_key_env_var_name)
            if ooba_api_key_to_use:
                logging.info(f"Oobabooga: Using API key from environment variable {api_key_env_var_name}")
            # else: # Optional: Fallback to reading directly from config (less secure)
            #     ooba_api_key_to_use = get_setting("api_settings", f"{provider_section_key}.api_key")
            #     if ooba_api_key_to_use:
            #         logging.warning("Oobabooga: Using API key found directly in config file (less secure).")

        if ooba_api_key_to_use:
             logging.debug(f"Oobabooga: Using API Key: {ooba_api_key_to_use[:5]}...{ooba_api_key_to_use[-5:]}")
        else:
             logging.info("Oobabooga: No API key provided or configured. Proceeding without Authorization header.")

        # API URL: Priority -> argument > config default
        api_url_to_use = api_url
        if not api_url_to_use:
            api_url_to_use = get_setting("api_settings", f"{provider_section_key}.api_url", "http://localhost:5000/v1/chat/completions") # Default URL
        logging.debug(f"Oobabooga: Using API URL: {api_url_to_use}")

        # Validate URL format
        if not isinstance(api_url_to_use, str) or not api_url_to_use.startswith(('http://', 'https://')):
            logging.error(f"Oobabooga: Invalid API URL configured or provided: {api_url_to_use}")
            return f"Oobabooga: Invalid API URL configured: {api_url_to_use}"

        # Model: Priority -> argument > config default (though Ooba might ignore this)
        # Ooba usually uses the model loaded in its UI, but we pass it if the API accepts it.
        model_to_use = get_setting("api_settings", f"{provider_section_key}.model") # No argument for model in this func signature
        logging.debug(f"Oobabooga: Model from config (may be ignored by server): {model_to_use}")

        # Streaming: Priority -> argument > config default
        if streaming is None:
            streaming_cfg = get_setting("api_settings", f"{provider_section_key}.streaming", False)
            streaming_to_use = _safe_cast(streaming_cfg, bool, False)
        else:
             streaming_to_use = bool(streaming) # Ensure boolean if passed as arg
        logging.debug(f"Oobabooga: Streaming mode: {streaming_to_use}")

        # Temperature: Priority -> argument > config default
        if temp is None:
            temp_cfg = get_setting("api_settings", f"{provider_section_key}.temperature", 0.7)
            temp_to_use = _safe_cast(temp_cfg, float, 0.7)
        else:
             temp_to_use = _safe_cast(temp, float, 0.7)
        logging.debug(f"Oobabooga: Using temperature: {temp_to_use}")

        # Top_p: Priority -> argument > config default
        if top_p is None:
            top_p_cfg = get_setting("api_settings", f"{provider_section_key}.top_p", 0.9)
            top_p_to_use = _safe_cast(top_p_cfg, float, 0.9)
        else:
             top_p_to_use = _safe_cast(top_p, float, 0.9)
        logging.debug(f"Oobabooga: Using top_p: {top_p_to_use}")

        # Max Tokens: Load from config
        max_tokens_cfg = get_setting("api_settings", f"{provider_section_key}.max_tokens", 4096)
        max_tokens_to_use = _safe_cast(max_tokens_cfg, int, 4096)
        logging.debug(f"Oobabooga: Using max_tokens: {max_tokens_to_use}")

        # System Message: Priority -> argument > fixed default
        if system_prompt is None:
            # Could load from config: get_setting("api_settings", f"{provider_section_key}.system_prompt", "Default")
            system_prompt_to_use = "You are a helpful AI assistant."
        else:
            system_prompt_to_use = system_prompt
        logging.debug(f"Oobabooga: Using system message: {system_prompt_to_use[:100]}...")

        # Timeout, Retries, Delay: Load from config
        timeout_cfg = get_setting("api_settings", f"{provider_section_key}.timeout", 300)
        api_timeout = _safe_cast(timeout_cfg, int, 300)
        retries_cfg = get_setting("api_settings", f"{provider_section_key}.retries", 1)
        retry_count = _safe_cast(retries_cfg, int, 1)
        delay_cfg = get_setting("api_settings", f"{provider_section_key}.retry_delay", 2)
        retry_delay = _safe_cast(delay_cfg, int, 2) # Can be float too
        logging.debug(f"Oobabooga: Timeout={api_timeout}, Retries={retry_count}, Delay={retry_delay}")

        # --- Prepare Request ---
        # Headers - only add Auth if key exists
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if ooba_api_key_to_use:
            headers['Authorization'] = f'Bearer {ooba_api_key_to_use}'
            logging.debug("Oobabooga: Added Authorization header.")

        # Prepare prompt and messages
        # Combine input_data (history+current message) with custom_prompt if provided
        final_user_prompt = f"{input_data}"
        if custom_prompt:
             final_user_prompt += f"\n\n{custom_prompt}"
        logging.debug(f"Oobabooga: Final user prompt (first 500): {final_user_prompt[:500]}...")

        messages = [
            {"role": "system", "content": system_prompt_to_use},
            {"role": "user", "content": final_user_prompt}
        ]

        # Prepare API payload for OpenAI compatible endpoint
        data = {
            # "mode": "chat", # Not needed for OpenAI endpoint
            # "character": "Example", # Not needed for OpenAI endpoint
            "model": model_to_use, # Pass model name if API uses it
            "messages": messages,
            "stream": streaming_to_use,
            "top_p": top_p_to_use,
            "temperature": temp_to_use,
            "max_tokens": max_tokens_to_use,
            # Add other OpenAI compatible params if needed and supported by Ooba
            # e.g., "top_k": top_k_to_use, "presence_penalty": ..., "frequency_penalty": ...
        }
        # Remove model from payload if it's empty or None, as Ooba might error
        if not model_to_use:
            data.pop("model", None)
            logging.debug("Oobabooga: Removed empty model key from payload.")


        # --- Execute Request ---
        session = requests.Session()
        retry_strategy = Retry(
            total=retry_count,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504], # Retry on server errors too
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # Mount for both http and https, as URL might be either
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        if streaming_to_use:
            logging.debug(f"Oobabooga: Posting streaming request to {api_url_to_use}")
            response = session.post(
                api_url_to_use,
                headers=headers,
                json=data,
                stream=True,
                timeout=api_timeout
            )
            logging.debug(f"Oobabooga: Response Status Code: {response.status_code}")
            response.raise_for_status() # Raise HTTP errors

            # --- Stream Processing ---
            # Keep existing generator, check JSON structure matches OpenAI stream format
            def stream_generator():
                collected_messages = "" # To potentially log full response later if needed
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data: '):
                            content = decoded_line[len('data: '):]
                            if content == '[DONE]':
                                logging.debug("Oobabooga: Received [DONE] marker.")
                                break
                            try:
                                data_chunk = json.loads(content)
                                # Standard OpenAI stream format
                                if 'choices' in data_chunk and data_chunk['choices']:
                                    delta = data_chunk['choices'][0].get('delta', {})
                                    if 'content' in delta and delta['content'] is not None: # Check content exists and is not null
                                        chunk = delta['content']
                                        collected_messages += chunk
                                        yield chunk
                                    # Handle function calls or other delta types if needed
                                else:
                                     logging.warning(f"Oobabooga: Received stream chunk without expected 'choices' structure: {data_chunk}")
                            except json.JSONDecodeError as e:
                                logging.error(f"Oobabooga: JSON decode error in stream: {e} - Line: '{decoded_line}'")
                                continue
                # logging.debug(f"Oobabooga: Streaming finished. Full text length: {len(collected_messages)}")
            return stream_generator() # Return the generator

        else: # Non-streaming
            logging.debug(f"Oobabooga: Posting non-streaming request to {api_url_to_use}")
            response = session.post(
                api_url_to_use,
                headers=headers,
                json=data,
                timeout=api_timeout
            )
            logging.debug(f"Oobabooga: Response Status Code: {response.status_code}")

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    # logging.debug(f"Oobabooga: Response Data: {response_data}") # Careful logging
                except json.JSONDecodeError as e:
                    logging.error(f"Oobabooga: Failed to decode JSON response: {e} - Response Text: {response.text[:500]}...")
                    return "Oobabooga: Error decoding API response."

                # Parse standard OpenAI response structure
                if 'choices' in response_data and response_data['choices']:
                    message_content = response_data['choices'][0].get('message', {}).get('content')
                    if message_content:
                        summary = message_content.strip()
                        logging.info("Oobabooga: Chat request successful.")
                        # logging.debug(f"Oobabooga: Summary (first 500 chars): {summary[:500]}...")
                        return summary
                    else:
                        logging.warning("Oobabooga: 'content' field missing or empty in response choices.")
                        return "Oobabooga: Response format missing content."
                else:
                    logging.warning("Oobabooga: 'choices' field missing or empty in response.")
                    return "Oobabooga: Response format missing choices."
            else:
                error_msg = f"Oobabooga: API request failed. Status: {response.status_code}, Body: {response.text[:500]}..."
                logging.error(error_msg)
                return f"Oobabooga: Failed request. Status: {response.status_code}"

    # --- Exception Handling ---
    except requests.exceptions.Timeout:
        logging.error(f"Oobabooga: Request timed out after {api_timeout} seconds to {api_url_to_use}.", exc_info=True)
        return f"Oobabooga: Request timed out."
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Oobabooga: Connection error to {api_url_to_use}: {e}", exc_info=True)
        return f"Oobabooga: Cannot connect to API endpoint: {api_url_to_use}."
    except requests.exceptions.RequestException as e:
        logging.error(f"Oobabooga: RequestException: {e}", exc_info=True)
        return f"Oobabooga: Network or Request Error: {e}"
    except json.JSONDecodeError as e: # Catch potential errors decoding input data if it's JSON
        logging.error(f"Oobabooga: Error decoding input JSON data: {e}", exc_info=True)
        return f"Oobabooga: Error decoding input JSON data."
    except ValueError as e: # Catch validation errors (like bad URL or bool conversion)
        logging.error(f"Oobabooga: Value error during setup: {e}", exc_info=True)
        return f"Oobabooga: Configuration or Input Error: {e}"
    except Exception as e:
        logging.error(f"Oobabooga: Unexpected error: {e}", exc_info=True)
        return f"Oobabooga: Unexpected error occurred: {e}"


def chat_with_tabbyapi(
    input_data,
    custom_prompt_input,
    system_message=None,
    api_key=None, # Argument for API key
    temp=None,    # Argument for temperature
    streaming=None,# Argument for streaming
    top_k=None,   # Argument for top_k
    top_p=None,   # Argument for top_p
    min_p=None,   # Argument for min_p (check if Tabby supports)
    model=None    # Argument for model override
):
    log = logging.getLogger(__name__)
    provider_section_key = "tabbyapi" # Key used in [api_settings.*] in config.toml
    logging.debug(f"{provider_section_key.capitalize()}: Chat request process starting...")

    try:
        # --- Load Settings ---
        # API Key: Priority -> argument > env var (name from config) > config file key (fallback)
        final_tabby_api_key = api_key # Prioritize argument
        if not final_tabby_api_key:
            api_key_env_var_name = get_setting("api_settings", f"{provider_section_key}.api_key_env_var")
            if api_key_env_var_name: # Check if env var name is configured
                final_tabby_api_key = os.environ.get(api_key_env_var_name)
                if final_tabby_api_key:
                    logging.info(f"{provider_section_key.capitalize()}: Using API key from environment variable {api_key_env_var_name}")
                # else: # Optional fallback to config file key
                    # final_tabby_api_key = get_setting("api_settings", f"{provider_section_key}.api_key")
                    # if final_tabby_api_key: logging.warning(f"{provider_section_key.capitalize()}: Using API key from config (less secure).")

        if final_tabby_api_key:
            logging.debug(f"{provider_section_key.capitalize()}: Using API Key: {final_tabby_api_key[:5]}...{final_tabby_api_key[-5:]}")
        else:
            logging.info(f"{provider_section_key.capitalize()}: No API key provided or configured (may be optional).")

        # API URL: Load from config
        default_api_url = "http://localhost:8000/v1/chat/completions" # Sensible default
        api_url_cfg = get_setting("api_settings", f"{provider_section_key}.api_url", default_api_url)
        final_api_url = str(api_url_cfg) if api_url_cfg else default_api_url
        # Basic URL validation
        if not final_api_url.startswith(("http://", "https://")):
            logging.error(f"{provider_section_key.capitalize()}: Invalid API URL configured: '{final_api_url}'. Using default: {default_api_url}")
            final_api_url = default_api_url
        logging.debug(f"{provider_section_key.capitalize()}: Using API URL: {final_api_url}")

        # Model: Priority -> argument > config default
        final_tabby_model = model
        if not final_tabby_model:
            final_tabby_model = get_setting("api_settings", f"{provider_section_key}.model", "TabbyML/DeepseekCoder-6.7B") # Example default
        logging.debug(f"{provider_section_key.capitalize()}: Using model: {final_tabby_model}")

        # Streaming: Priority -> argument > config default
        if streaming is None:
            streaming_cfg = get_setting("api_settings", f"{provider_section_key}.streaming", False)
            final_streaming = str(streaming_cfg).lower() == 'true' if isinstance(streaming_cfg, str) else bool(streaming_cfg)
        else:
             final_streaming = bool(streaming) # Ensure boolean
        logging.debug(f"{provider_section_key.capitalize()}: Streaming mode: {final_streaming}")

        # Temperature: Priority -> argument > config default
        if temp is None:
            temp_cfg = get_setting("api_settings", f"{provider_section_key}.temperature", 0.1)
            final_temp = _safe_cast(temp_cfg, float, 0.1)
        else:
             final_temp = _safe_cast(temp, float, 0.1)
        logging.debug(f"{provider_section_key.capitalize()}: Using temperature: {final_temp}")

        # Top K: Priority -> argument > config default
        if top_k is None:
            top_k_cfg = get_setting("api_settings", f"{provider_section_key}.top_k", 50)
            final_top_k = _safe_cast(top_k_cfg, int, 50)
        else:
             final_top_k = _safe_cast(top_k, int, 50)
        if final_top_k <= 0: final_top_k = None # Many APIs disable top_k if <= 0
        logging.debug(f"{provider_section_key.capitalize()}: Using top_k: {final_top_k}")

        # Top P: Priority -> argument > config default
        if top_p is None:
            top_p_cfg = get_setting("api_settings", f"{provider_section_key}.top_p", 0.95)
            final_top_p = _safe_cast(top_p_cfg, float, 0.95)
        else:
             final_top_p = _safe_cast(top_p, float, 0.95)
        logging.debug(f"{provider_section_key.capitalize()}: Using top_p: {final_top_p}")

        # Min P: Priority -> argument > config default (often not supported)
        if min_p is None:
            min_p_cfg = get_setting("api_settings", f"{provider_section_key}.min_p", 0.0)
            final_min_p = _safe_cast(min_p_cfg, float, 0.0)
        else:
             final_min_p = _safe_cast(min_p, float, 0.0)
        logging.debug(f"{provider_section_key.capitalize()}: Using min_p: {final_min_p}") # Log it, but may not be used

        # Max Tokens: Load from config
        max_tokens_cfg = get_setting("api_settings", f"{provider_section_key}.max_tokens", 256)
        final_max_tokens = _safe_cast(max_tokens_cfg, int, 256)
        logging.debug(f"{provider_section_key.capitalize()}: Using max_tokens: {final_max_tokens}")

        # System Message: Priority -> argument > config default
        if system_message is None:
            final_system_message = get_setting("api_settings", f"{provider_section_key}.system_prompt", "You are a helpful AI assistant.")
        else:
            final_system_message = system_message
        logging.debug(f"{provider_section_key.capitalize()}: Using system message: {final_system_message[:100]}...")

        # Timeout, Retries, Delay: Load from config
        timeout_cfg = get_setting("api_settings", f"{provider_section_key}.timeout", 120)
        final_api_timeout = _safe_cast(timeout_cfg, int, 120)
        retries_cfg = get_setting("api_settings", f"{provider_section_key}.retries", 1)
        final_retry_count = _safe_cast(retries_cfg, int, 1)
        delay_cfg = get_setting("api_settings", f"{provider_section_key}.retry_delay", 2)
        final_retry_delay = _safe_cast(delay_cfg, (float, int), 2)
        logging.debug(f"{provider_section_key.capitalize()}: Timeout={final_api_timeout}, Retries={final_retry_count}, Delay={final_retry_delay}")

        # --- Prepare Request ---
        # Combine input_data and custom_prompt_input
        # Assuming input_data is the main text and custom_prompt_input modifies the request
        if isinstance(input_data, list): # Handle list input if necessary
             input_text = " ".join(item.get("Text", "") for item in input_data) # Basic example
        else:
             input_text = str(input_data)

        combined_user_content = f"{input_text}"
        if custom_prompt_input:
            combined_user_content = f"{custom_prompt_input}\n\n{input_text}" # Prepend custom prompt
        logging.debug(f"{provider_section_key.capitalize()}: Combined user content (first 500): {combined_user_content[:500]}...")

        headers = {'Content-Type': 'application/json'}
        if final_tabby_api_key:
            headers['Authorization'] = f'Bearer {final_tabby_api_key}'

        payload = {
            'model': final_tabby_model,
            'messages': [
                {'role': 'system', 'content': final_system_message},
                {'role': 'user', 'content': combined_user_content}
            ],
            'temperature': final_temp,
            'max_tokens': final_max_tokens,
            'top_p': final_top_p,
            # Only include top_k if it has a value (not None)
            # 'min_p': final_min_p, # Include only if TabbyAPI OpenAI endpoint supports it
            'stream': final_streaming
        }
        # Conditionally add top_k
        if final_top_k is not None and final_top_k > 0:
             payload['top_k'] = final_top_k
        # Conditionally add min_p if supported
        # if final_min_p is not None and final_min_p > 0:
        #    payload['min_p'] = final_min_p

        # Clean payload of None values if needed by API
        # payload = {k: v for k, v in payload.items() if v is not None}
        logging.debug(f"{provider_section_key.capitalize()}: Request payload: {json.dumps(payload, indent=2)}")

        # --- Execute Request ---
        session = requests.Session()
        retry_strategy = Retry(
            total=final_retry_count,
            backoff_factor=final_retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # Mount for both http and https, as local URLs can be either
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        if final_streaming:
            logging.debug(f"{provider_section_key.capitalize()}: Posting streaming request to {final_api_url}")
            try:
                response = session.post(
                    final_api_url,
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=final_api_timeout
                )
                response.raise_for_status() # Raise HTTP errors

                # --- Stream Processing Generator ---
                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith('data: '):
                                data_line = decoded_line[len('data: '):]
                                if data_line == '[DONE]':
                                    break
                                try:
                                    data_json = json.loads(data_line)
                                    # Check structure - assuming OpenAI compatible stream
                                    if 'choices' in data_json and len(data_json['choices']) > 0:
                                        delta = data_json['choices'][0].get('delta', {})
                                        content = delta.get('content', '')
                                        if content:
                                            yield content
                                except json.JSONDecodeError as e:
                                    logging.error(f"{provider_section_key.capitalize()}: Failed to parse JSON streamed data: {str(e)} Line: '{decoded_line}'")
                            # else: # Optionally log non-data lines
                            #     logging.debug(f"{provider_section_key.capitalize()}: Received non-data line: {decoded_line}")
                return stream_generator()

            except requests.exceptions.RequestException as e:
                logging.error(f"{provider_section_key.capitalize()}: Streaming request failed: {e}", exc_info=True)
                # Need to yield an error message compatible with streaming consumer
                def error_stream():
                    yield f"{provider_section_key.capitalize()}: Streaming request error: {str(e)}"
                return error_stream()
            except Exception as e:
                logging.error(f"{provider_section_key.capitalize()}: Unexpected error during streaming request: {e}", exc_info=True)
                def error_stream():
                    yield f"{provider_section_key.capitalize()}: Unexpected streaming error: {str(e)}"
                return error_stream()
        else: # Non-streaming
            logging.debug(f"{provider_section_key.capitalize()}: Posting non-streaming request to {final_api_url}")
            try:
                response = session.post(
                    final_api_url,
                    headers=headers,
                    json=payload,
                    timeout=final_api_timeout
                )
                logging.debug(f"{provider_section_key.capitalize()}: Response Status: {response.status_code}")
                response.raise_for_status() # Raise HTTP errors
                response_json = response.json()
                # logging.debug(f"{provider_section_key.capitalize()}: Response JSON: {response_json}") # Careful logging

                # Validate response structure (assuming OpenAI compatible)
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    message_content = response_json['choices'][0].get('message', {}).get('content', '')
                    if message_content:
                        logging.info(f"{provider_section_key.capitalize()}: Chat request successful.")
                        return message_content.strip()
                    else:
                        logging.warning(f"{provider_section_key.capitalize()}: 'content' missing in response choice message.")
                        return f"{provider_section_key.capitalize()}: Response missing content."
                else:
                    logging.warning(f"{provider_section_key.capitalize()}: 'choices' array missing or empty in response.")
                    return f"{provider_section_key.capitalize()}: Invalid response structure (no choices)."

            except requests.exceptions.Timeout:
                 logging.error(f"{provider_section_key.capitalize()}: Request timed out after {final_api_timeout} seconds.")
                 return f"{provider_section_key.capitalize()}: Request Timed Out"
            except requests.exceptions.RequestException as e:
                logging.error(f"{provider_section_key.capitalize()}: Request failed: {e}", exc_info=True)
                return f"{provider_section_key.capitalize()}: Request Error: {str(e)}"
            except json.JSONDecodeError:
                logging.error(f"{provider_section_key.capitalize()}: Failed to decode JSON response. Body: {response.text[:500]}...")
                return f"{provider_section_key.capitalize()}: Invalid JSON response from server."
            except Exception as e:
                logging.error(f"{provider_section_key.capitalize()}: Unexpected error during non-streaming request: {e}", exc_info=True)
                return f"{provider_section_key.capitalize()}: Unexpected error: {str(e)}"

    # --- Outer Exception Handling ---
    except Exception as e:
        logging.error(f"{provider_section_key.capitalize()}: Unexpected error in setup or logic: {e}", exc_info=True)
        # Decide how to return error (yield for streaming, string for non-streaming)
        error_msg = f"{provider_section_key.capitalize()}: Unexpected error in setup: {str(e)}"
        if streaming is None: # If streaming determination failed early
             return error_msg
        elif final_streaming: # Check the determined streaming state
            def error_stream(): yield error_msg
            return error_stream()
        else:
            return error_msg


def chat_with_aphrodite(api_key: Optional[str] = None,
                        input_data: Union[str, dict, list] = None,
                        custom_prompt: Optional[str] = None,
                        temp: Optional[float] = None,
                        system_message: Optional[str] = None,
                        streaming: Optional[bool] = None,
                        topp: Optional[float] = None, # Corresponds to top_p
                        minp: Optional[float] = None,
                        topk: Optional[int] = None,
                        model: Optional[str] = None):
    """Interacts with an Aphrodite-engine compatible API using settings from config."""
    log = logging.getLogger(__name__)
    provider_section_key = "aphrodite" # Key used in [api_settings.*] in config.toml

    try:
        # --- Load Settings ---
        # API Key: Priority -> argument > env var (name from config) > config file key (fallback)
        aphrodite_api_key = api_key # Prioritize argument
        if not aphrodite_api_key:
            api_key_env_var_name = get_setting("api_settings", f"{provider_section_key}.api_key_env_var", "APHRODITE_API_KEY")
            aphrodite_api_key = os.environ.get(api_key_env_var_name)
            if aphrodite_api_key:
                logging.info(f"Aphrodite: Using API key from environment variable {api_key_env_var_name}")
            # else: # Optional: Fallback to reading directly from config
            #     aphrodite_api_key = get_setting("api_settings", f"{provider_section_key}.api_key")
            #     if aphrodite_api_key:
            #         logging.warning("Aphrodite: Using API key found directly in config file (less secure).")

        # Aphrodite often runs without auth, so key might be optional
        if aphrodite_api_key:
            logging.debug(f"Aphrodite: Using API Key: {aphrodite_api_key[:5]}...{aphrodite_api_key[-5:]}")
        else:
            logging.info("Aphrodite: No API key provided or configured. Proceeding without Authorization header.")

        # API URL: Load from config (no argument override planned here)
        api_url = get_setting("api_settings", f"{provider_section_key}.api_url")
        if not api_url:
            logging.error("Aphrodite: API URL not found in config ([api_settings.aphrodite].api_url).")
            return "Aphrodite: API URL not configured."
        logging.debug(f"Aphrodite: Using API URL: {api_url}")

        # Model: Priority -> argument > config default
        aphrodite_model = model
        if not aphrodite_model:
            aphrodite_model = get_setting("api_settings", f"{provider_section_key}.model", "aphrodite-engine") # Example default
        logging.debug(f"Aphrodite: Using model: {aphrodite_model}")

        # Streaming: Priority -> argument > config default
        if streaming is None:
            streaming_cfg = get_setting("api_settings", f"{provider_section_key}.streaming", False)
            streaming = bool(streaming_cfg) if not isinstance(streaming_cfg, str) else streaming_cfg.lower() == "true"
        else:
            streaming = bool(streaming)
        logging.debug(f"Aphrodite: Streaming mode: {streaming}")

        # Temperature: Priority -> argument > config default
        if temp is None:
            temp_cfg = get_setting("api_settings", f"{provider_section_key}.temperature", 0.7)
            temp_value = _safe_cast(temp_cfg, float, 0.7)
        else:
            temp_value = _safe_cast(temp, float, 0.7)
        logging.debug(f"Aphrodite: Using temperature: {temp_value}")

        # Top_p (topp): Priority -> argument > config default
        if topp is None:
            topp_cfg = get_setting("api_settings", f"{provider_section_key}.top_p", 0.95)
            top_p_value = _safe_cast(topp_cfg, float, 0.95)
        else:
            top_p_value = _safe_cast(topp, float, 0.95)
        logging.debug(f"Aphrodite: Using top_p: {top_p_value}")

        # Min_p (minp): Priority -> argument > config default
        if minp is None:
            minp_cfg = get_setting("api_settings", f"{provider_section_key}.min_p", 0.05)
            min_p_value = _safe_cast(minp_cfg, float, 0.05)
        else:
            min_p_value = _safe_cast(minp, float, 0.05)
        logging.debug(f"Aphrodite: Using min_p: {min_p_value}")

        # Top_k (topk): Priority -> argument > config default
        if topk is None:
            topk_cfg = get_setting("api_settings", f"{provider_section_key}.top_k", 50)
            top_k_value = _safe_cast(topk_cfg, int, 50)
        else:
            top_k_value = _safe_cast(topk, int, 50)
        # Aphrodite/OpenAI compatible APIs often use 0 or -1 to disable top_k
        if top_k_value <= 0:
            logging.debug("Aphrodite: top_k disabled (<= 0).")
            top_k_value = None # Set to None to potentially omit from payload if API prefers that
        else:
            logging.debug(f"Aphrodite: Using top_k: {top_k_value}")

        # Max Tokens: Load from config
        max_tokens_cfg = get_setting("api_settings", f"{provider_section_key}.max_tokens", 4096)
        max_tokens = _safe_cast(max_tokens_cfg, int, 4096)
        logging.debug(f"Aphrodite: Using max_tokens: {max_tokens}")

        # System Message: Priority -> argument > config default > hardcoded default
        if system_message is None:
            system_message = get_setting("api_settings", f"{provider_section_key}.system_prompt") # Check config first
            if system_message is None: # If still None, use hardcoded default
                 system_message = "You are a helpful AI assistant."
        logging.debug(f"Aphrodite: Using system message: {system_message[:100]}...")

        # Timeout, Retries, Delay: Load from config
        timeout_cfg = get_setting("api_settings", f"{provider_section_key}.timeout", 300)
        api_timeout = _safe_cast(timeout_cfg, int, 300)
        retries_cfg = get_setting("api_settings", f"{provider_section_key}.retries", 1)
        retry_count = _safe_cast(retries_cfg, int, 1)
        delay_cfg = get_setting("api_settings", f"{provider_section_key}.retry_delay", 2)
        retry_delay = _safe_cast(delay_cfg, (int, float), 2) # Delay can be float
        logging.debug(f"Aphrodite: Timeout={api_timeout}, Retries={retry_count}, Delay={retry_delay}")

        # --- Prepare Request ---
        headers = {'Content-Type': 'application/json'}
        if aphrodite_api_key:
            headers['Authorization'] = f'Bearer {aphrodite_api_key}'

        # Combine input data and prompt
        aphrodite_prompt = f"{input_data}\n\n{custom_prompt}" if custom_prompt else str(input_data)
        logging.debug(f"Aphrodite: Combined prompt (first 500 chars): {aphrodite_prompt[:500]}...")

        # Construct payload, omitting None values where appropriate
        data = {
            "model": aphrodite_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": aphrodite_prompt}
            ],
            "temperature": temp_value,
            "stream": streaming,
            "top_p": top_p_value,
            "min_p": min_p_value, # Include if API supports it
            "max_tokens": max_tokens,
        }
        # Only include top_k if it has a positive value
        if top_k_value is not None and top_k_value > 0:
             data["top_k"] = top_k_value

        logging.debug(f"Aphrodite: Payload: { {k: v for k, v in data.items() if k != 'messages'} }") # Log payload without messages

        # --- Execute Request ---
        session = requests.Session()
        retry_strategy = Retry(
            total=retry_count,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504], # Standard retry codes
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter) # Mount for http
        session.mount("https://", adapter) # Mount for https (if Aphrodite uses it)

        if streaming:
            logging.debug(f"Aphrodite: Posting streaming request to {api_url}")
            response = session.post(
                api_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=api_timeout
            )
            logging.debug(f"Aphrodite: Response Status (Streaming): {response.status_code}")
            response.raise_for_status()

            # --- Stream Processing Generator ---
            def stream_generator():
                collected_messages = "" # To potentially reconstruct full message if needed later
                try:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8").strip()
                            if decoded_line == "": continue
                            if decoded_line.startswith("data:"):
                                data_str = decoded_line[len("data: "):]
                                if data_str == "[DONE]": break
                                try:
                                    data_json = json.loads(data_str)
                                    # OpenAI compatible stream format
                                    chunk = data_json["choices"][0]["delta"].get("content", "")
                                    if chunk: # Only yield non-empty chunks
                                        collected_messages += chunk
                                        yield chunk
                                except json.JSONDecodeError:
                                    logging.error(f"Aphrodite: Error decoding streaming JSON: {decoded_line}")
                                except (KeyError, IndexError) as e:
                                     logging.error(f"Aphrodite: Unexpected stream format ({e}): {decoded_line}")
                                except Exception as stream_e:
                                     logging.error(f"Aphrodite: Unknown error processing stream chunk: {stream_e}")
                finally:
                    # Ensure response is closed even if generator isn't fully consumed
                    response.close()
                    logging.debug("Aphrodite: Streaming response closed.")
            return stream_generator()

        else: # Non-streaming
            logging.debug(f"Aphrodite: Posting non-streaming request to {api_url}")
            response = session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=api_timeout
            )
            logging.debug(f"Aphrodite: Response Status: {response.status_code}")
            # logging.debug(f"Aphrodite: Full Response Body: {response.text[:1000]}...") # Log snippet

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    message_content = response_data['choices'][0].get('message', {}).get('content')
                    if message_content:
                         chat_response = message_content.strip()
                         logging.info("Aphrodite: Chat request successful.")
                         # logging.debug(f"Aphrodite: Chat response: {chat_response[:200]}...")
                         return chat_response
                    else:
                         logging.warning("Aphrodite: 'message' or 'content' missing in response choice.")
                         return "Aphrodite: Response structure missing content."
                else:
                    logging.warning("Aphrodite: 'choices' array missing or empty in response.")
                    return "Aphrodite: Response structure missing choices."
            else:
                logging.error(f"Aphrodite: Chat request failed. Status: {response.status_code}, Body: {response.text[:500]}...")
                return f"Aphrodite: Failed request. Status: {response.status_code}"

    # --- Exception Handling ---
    except requests.exceptions.RequestException as e:
        logging.error(f"Aphrodite: RequestException: {e}", exc_info=True)
        return f"Aphrodite: Network or Request Error: {e}"
    except json.JSONDecodeError as e:
        logging.error(f"Aphrodite: Error decoding JSON response: {e}", exc_info=True)
        return f"Aphrodite: Error parsing API response."
    except Exception as e:
        logging.error(f"Aphrodite: Unexpected error: {e}", exc_info=True)
        return f"Aphrodite: Unexpected error occurred: {e}"


def chat_with_ollama(input_data: str,  # Type hint input
                     custom_prompt: str, # Type hint input
                     api_url: Optional[str] = None, # Use Optional
                     api_key: Optional[str] = None, # Use Optional
                     temp: Optional[float] = None,  # Use Optional
                     system_message: Optional[str] = None, # Use Optional
                     model: Optional[str] = None,     # Use Optional
                     streaming: Optional[bool] = None, # Use Optional
                     top_p: Optional[float] = None,   # Use Optional
                     top_k: Optional[int] = None      # Add top_k parameter
                     ) -> Union[str, Generator[str, None, None], None]: # Type hint return

    # https://github.com/ollama/ollama/blob/main/docs/openai.md
    provider_section_key = "ollama" # Key for [api_settings.ollama] in config.toml

    try:
        # --- Load Settings ---
        logging.debug(f"Starting chat_with_ollama for model: {model or 'Default'}")

        # API Key: (Ollama usually doesn't need one, but follow pattern)
        ollama_api_key = api_key # Argument first
        if not ollama_api_key:
            api_key_env_var = get_setting("api_settings", f"{provider_section_key}.api_key_env_var", "OLLAMA_API_KEY")
            ollama_api_key = os.environ.get(api_key_env_var)
            if ollama_api_key:
                logging.info(f"Ollama: Using API key from env var {api_key_env_var}")
            # No fallback to config file read for keys by default

        if ollama_api_key:
             logging.debug(f"Ollama: Using API Key: {ollama_api_key[:5]}...{ollama_api_key[-5:]}")
        else:
             logging.debug("Ollama: No API key provided or found. Proceeding without Authorization header.")

        # API URL: Argument > config > default
        api_url_value = api_url
        if not api_url_value:
            api_url_value = get_setting("api_settings", f"{provider_section_key}.api_url", "http://localhost:11434/v1/chat/completions")
        if not api_url_value:
            logging.error("Ollama: API URL not found in argument or config.")
            return "Ollama: API URL Not Configured."
        logging.debug(f"Ollama: Using API URL: {api_url_value}")

        # Model: Argument > config > default
        ollama_model = model
        if not ollama_model:
            ollama_model = get_setting("api_settings", f"{provider_section_key}.model", "llama3:latest")
        if not ollama_model:
            logging.error("Ollama: Model name not found in argument or config.")
            return "Ollama: Model Not Configured."
        logging.debug(f"Ollama: Using model: {ollama_model}")

        # Streaming: Argument > config > default
        if streaming is None:
            streaming_cfg = get_setting("api_settings", f"{provider_section_key}.streaming", False)
            streaming_value = _safe_cast(streaming_cfg, bool, False)
        else:
            streaming_value = bool(streaming)
        logging.debug(f"Ollama: Streaming mode: {streaming_value}")

        # Temperature: Argument > config > default
        if temp is None:
            temp_cfg = get_setting("api_settings", f"{provider_section_key}.temperature", 0.7)
            temp_value = _safe_cast(temp_cfg, float, 0.7)
        else:
            temp_value = _safe_cast(temp, float, 0.7)
        logging.debug(f"Ollama: Using temperature: {temp_value}")

        # Top P: Argument > config > default
        if top_p is None:
            topp_cfg = get_setting("api_settings", f"{provider_section_key}.top_p", 0.9)
            top_p_value = _safe_cast(topp_cfg, float, 0.9)
        else:
            top_p_value = _safe_cast(top_p, float, 0.9)
        logging.debug(f"Ollama: Using top_p: {top_p_value}")

        # Top K: Argument > config > default
        if top_k is None:
            topk_cfg = get_setting("api_settings", f"{provider_section_key}.top_k", 40)
            top_k_value = _safe_cast(topk_cfg, int, 40)
        else:
             top_k_value = _safe_cast(top_k, int, 40)
        logging.debug(f"Ollama: Using top_k: {top_k_value}")


        # Max Tokens: Load from config
        max_tokens_cfg = get_setting("api_settings", f"{provider_section_key}.max_tokens", 4096)
        max_tokens_value = _safe_cast(max_tokens_cfg, int, 4096)
        logging.debug(f"Ollama: Using max_tokens: {max_tokens_value}")

        # System Message: Argument > default
        if system_message is None:
            # system_message = get_setting("api_settings", f"{provider_section_key}.system_prompt", "You are helpful.") # Option to load from config
            system_message_value = "You are a helpful AI assistant"
        else:
             system_message_value = system_message
        logging.debug(f"Ollama: Using system message: {system_message_value[:100]}...")

        # Timeout, Retries, Delay: Load from config
        timeout_cfg = get_setting("api_settings", f"{provider_section_key}.timeout", 300)
        api_timeout_value = _safe_cast(timeout_cfg, int, 300)
        retries_cfg = get_setting("api_settings", f"{provider_section_key}.retries", 1)
        retry_count_value = _safe_cast(retries_cfg, int, 1)
        delay_cfg = get_setting("api_settings", f"{provider_section_key}.retry_delay", 2)
        retry_delay_value = _safe_cast(delay_cfg, (int, float), 2) # Can be float for backoff
        logging.debug(f"Ollama: Timeout={api_timeout_value}, Retries={retry_count_value}, Delay={retry_delay_value}")

        # --- Prepare Request ---
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        # Add Authorization header ONLY if a key was found
        if ollama_api_key:
            headers['Authorization'] = f'Bearer {ollama_api_key}'
            logging.debug("Ollama: Added Authorization header.")

        # Combine prompt parts
        # Ensure input_data and custom_prompt are strings
        input_str = str(input_data) if input_data is not None else ""
        prompt_str = str(custom_prompt) if custom_prompt is not None else ""
        # Handle potential combination logic if needed, here just user message
        # Example: ollama_user_prompt = f"{prompt_str}\n\n{input_str}"
        ollama_user_prompt = input_str # Assuming input_data contains the full context + user message
        if prompt_str: # If custom prompt provided, append it clearly
             ollama_user_prompt += f"\n\n--- Custom Instructions ---\n{prompt_str}"

        logging.debug(f"Ollama: Final User Prompt (first 500): {ollama_user_prompt[:500]}...")

        data_payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_message_value},
                {"role": "user",   "content": ollama_user_prompt}
            ],
            "options": { # Ollama specific options block for OpenAI endpoint
                 "temperature": temp_value,
                 "top_p": top_p_value,
                 "top_k": top_k_value,
                 "num_predict": max_tokens_value # Map max_tokens to num_predict for Ollama options
                 # Add other Ollama-specific options here if needed, e.g., stop sequences
            },
            "stream": streaming_value,
            # "max_tokens": max_tokens_value # max_tokens outside options might also work for some versions
        }
        logging.debug(f"Ollama: Payload: { {k: v for k, v in data_payload.items() if k != 'messages'} } Messages: ...") # Avoid logging full prompt

        # --- Execute Request ---
        session = requests.Session()
        retry_strategy = Retry(
            total=retry_count_value,
            backoff_factor=retry_delay_value,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"] # Explicitly allow retries for POST
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        logging.debug(f"Ollama: Sending POST to {api_url_value}")
        response = session.post(
            api_url_value,
            headers=headers,
            json=data_payload,
            stream=streaming_value,
            timeout=api_timeout_value
        )
        logging.debug(f"Ollama: Response Status Code: {response.status_code}")
        response.raise_for_status() # Raise HTTP errors (4xx, 5xx)

        # --- Handle Response ---
        if streaming_value:
            logging.debug("Ollama: Processing streaming response.")
            # We return a generator of text chunks
            def stream_generator() -> Generator[str, None, None]:
                buffer = ""
                try:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        decoded_line = line.decode('utf-8').strip()
                        logging.debug(f"Ollama Stream Raw Line: {decoded_line}") # Verbose stream log

                        # Ollama's OpenAI endpoint uses SSE format
                        if decoded_line.startswith("data:"):
                            json_str = decoded_line[len("data:"):].strip()
                            if json_str == "[DONE]":
                                logging.debug("Ollama Stream: Received [DONE]")
                                break
                            try:
                                data_json = json.loads(json_str)
                                # Structure: { "model": "...", "created_at": "...", "message": { "role": "assistant", "content": "..." }, "done": false }
                                # Or final message: { ... "done": true, "total_duration": ..., ... }
                                delta = data_json.get("message", {}).get("content")
                                if delta:
                                    logging.debug(f"Ollama Stream Chunk: {delta!r}")
                                    yield delta
                                if data_json.get("done"):
                                     logging.debug("Ollama Stream: Received done=true message.")
                                     # Optional: Yield any final stats if needed?
                                     break # Stop processing on done=true
                            except json.JSONDecodeError:
                                logging.error(f"Ollama: JSON decode error in streaming chunk: {decoded_line}")
                                continue
                            except Exception as stream_ex:
                                logging.error(f"Ollama: Error processing stream data: {stream_ex} - Data: {decoded_line}", exc_info=True)
                                continue
                        elif decoded_line: # Log lines not starting with data:
                             logging.warning(f"Ollama Stream: Received non-data line: {decoded_line}")

                except requests.exceptions.ChunkedEncodingError as cee:
                     logging.error(f"Ollama: ChunkedEncodingError during streaming: {cee}", exc_info=True)
                     yield "\n[STREAM ERROR: Connection issue]"
                except Exception as gen_ex:
                    logging.error(f"Ollama: Unexpected error during stream iteration: {gen_ex}", exc_info=True)
                    yield "\n[STREAM ERROR: Unexpected issue]"
                finally:
                     response.close() # Ensure connection is closed
                     logging.debug("Ollama: Stream generator finished.")

            return stream_generator()

        else: # Non-streaming
            logging.debug("Ollama: Processing non-streaming response.")
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logging.error(f"Ollama: Failed to parse JSON response: {str(e)}. Body: {response.text[:500]}...")
                return f"Ollama: JSON parse error: {str(e)}"

            # logging.debug(f"Ollama: Non-streaming API Response Data: {response_data}") # Careful with large logs

            # Extract content from OpenAI compatible structure
            # Example structure: { "model": "...", "created_at":"...", "message": { "role": "assistant", "content": "..." }, "done": true, ... }
            final_text = None
            if isinstance(response_data, dict):
                 message_content = response_data.get("message", {})
                 if isinstance(message_content, dict):
                      final_text = message_content.get("content")

            if final_text is not None and isinstance(final_text, str):
                logging.info("Ollama: Chat request successful (non-stream).")
                return final_text.strip()
            else:
                logging.error(f"Ollama: Could not find 'message.content' string in non-streaming response. Data: {response_data}")
                return "Ollama: API response structure unexpected (non-stream)."

    # --- Exception Handling ---
    except requests.exceptions.Timeout:
        logging.error(f"Ollama: Request timed out after {api_timeout_value} seconds.")
        return f"Ollama: Request Timed Out"
    except requests.exceptions.ConnectionError as ce:
         logging.error(f"Ollama: Connection error to {api_url_value}: {ce}", exc_info=True)
         return f"Ollama: Connection Error - Is Ollama running at {api_url_value}?"
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Ollama: HTTP Request error: {req_err}", exc_info=True)
        # Try to include response text if available
        error_detail = str(req_err)
        if hasattr(req_err, 'response') and req_err.response is not None:
             error_detail += f" | Response: {req_err.response.status_code} - {req_err.response.text[:200]}..."
        return f"Ollama: HTTP Error: {error_detail}"
    except ValueError as ve: # Catch validation errors like bad boolean cast
         logging.error(f"Ollama: Value error during setup: {ve}", exc_info=True)
         return f"Ollama: Configuration Value Error: {ve}"
    except Exception as ex:
        logging.error(f"Ollama: Unexpected error in chat_with_ollama: {ex}", exc_info=True)
        return f"Ollama: Unexpected Exception: {ex}"


def chat_with_vllm(
    input_data: Union[str, dict, list],
    custom_prompt_input: str,
    api_key: Optional[str] = None, # Allow None
    vllm_api_url: Optional[str] = None, # Allow None
    model: Optional[str] = None, # Allow None
    system_prompt: Optional[str] = None, # Allow None
    temp: Optional[float] = None, # Allow None
    streaming: Optional[bool] = None, # Allow None
    minp: Optional[float] = None, # Allow None
    topp: Optional[float] = None, # Allow None
    topk: Optional[int] = None # Allow None
) -> Union[str, Generator[Any, Any, None], Any]: # Use Union for type hint clarity

    log = logging.getLogger(__name__)
    provider_section_key = "vllm" # Key for [api_settings.vllm] in config.toml
    logging.debug(f"vLLM: Chat request started with provider key '{provider_section_key}'")

    try:
        # --- Load Settings ---
        # API Key: Priority -> argument > env var > config (optional)
        vllm_api_key = api_key # Prioritize argument
        if not vllm_api_key:
            api_key_env_var_name = get_setting("api_settings", f"{provider_section_key}.api_key_env_var", "VLLM_API_KEY")
            vllm_api_key = os.environ.get(api_key_env_var_name)
            if vllm_api_key:
                logging.info(f"vLLM: Using API key from environment variable {api_key_env_var_name}")
            # else: # Optional fallback
            #     vllm_api_key = get_setting("api_settings", f"{provider_section_key}.api_key")
            #     if vllm_api_key: logging.warning("vLLM: Using API key from config file.")

        if vllm_api_key:
             logging.debug(f"vLLM: Using API Key: {vllm_api_key[:5]}...{vllm_api_key[-5:]}")
        else:
             logging.info("vLLM: No API Key provided or configured (may not be required).")

        # API URL: Priority -> argument > config
        api_url_to_use = vllm_api_url
        if not api_url_to_use:
             api_url_to_use = get_setting("api_settings", f"{provider_section_key}.api_url", "http://localhost:8000/v1/chat/completions")
        if not api_url_to_use: # Final check if config is also missing/empty
            logging.error("vLLM: API URL not provided via argument or config.")
            return "vLLM: API URL is required but not configured."
        logging.debug(f"vLLM: Using API URL: {api_url_to_use}")

        # Model: Priority -> argument > config (often optional for vLLM request itself)
        model_to_use = model
        if not model_to_use:
            model_to_use = get_setting("api_settings", f"{provider_section_key}.model", None) # Default None if not in config
        logging.debug(f"vLLM: Using model parameter: {model_to_use or 'Server Default'}") # Log clearly if None

        # Streaming: Priority -> argument > config
        if streaming is None:
            streaming_cfg = get_setting("api_settings", f"{provider_section_key}.streaming", False)
            streaming_to_use = _safe_cast(streaming_cfg, bool, False)
        else:
            streaming_to_use = _safe_cast(streaming, bool, False) # Ensure boolean
        logging.debug(f"vLLM: Streaming mode: {streaming_to_use}")

        # Temperature: Priority -> argument > config
        if temp is None:
            temp_cfg = get_setting("api_settings", f"{provider_section_key}.temperature", 0.7)
            temp_to_use = _safe_cast(temp_cfg, float, 0.7)
        else:
            temp_to_use = _safe_cast(temp, float, 0.7)
        logging.debug(f"vLLM: Using temperature: {temp_to_use}")

        # Top-P (topp): Priority -> argument > config
        if topp is None:
            topp_cfg = get_setting("api_settings", f"{provider_section_key}.top_p", 0.95)
            top_p_to_use = _safe_cast(topp_cfg, float, 0.95)
        else:
            top_p_to_use = _safe_cast(topp, float, 0.95)
        logging.debug(f"vLLM: Using top_p: {top_p_to_use}")

        # Top-K (topk): Priority -> argument > config
        if topk is None:
            topk_cfg = get_setting("api_settings", f"{provider_section_key}.top_k", 50)
            top_k_to_use = _safe_cast(topk_cfg, int, 50)
        else:
            top_k_to_use = _safe_cast(topk, int, 50)
        logging.debug(f"vLLM: Using top_k: {top_k_to_use}")

        # Min-P (minp): Priority -> argument > config
        if minp is None:
            minp_cfg = get_setting("api_settings", f"{provider_section_key}.min_p", 0.05)
            min_p_to_use = _safe_cast(minp_cfg, float, 0.05)
        else:
            min_p_to_use = _safe_cast(minp, float, 0.05)
        logging.debug(f"vLLM: Using min_p: {min_p_to_use}")

        # Max Tokens: Load from config
        max_tokens_cfg = get_setting("api_settings", f"{provider_section_key}.max_tokens", 4096)
        max_tokens_to_use = _safe_cast(max_tokens_cfg, int, 4096)
        logging.debug(f"vLLM: Using max_tokens: {max_tokens_to_use}")

        # System Prompt: Priority -> argument > default
        if system_prompt is None:
            # system_prompt_to_use = get_setting("api_settings", f"{provider_section_key}.system_prompt", "Default prompt")
            system_prompt_to_use = "You are a helpful AI assistant."
        else:
             system_prompt_to_use = system_prompt
        logging.debug(f"vLLM: Using system prompt: {system_prompt_to_use[:100]}...")

        # Timeout, Retries, Delay: Load from config
        timeout_cfg = get_setting("api_settings", f"{provider_section_key}.timeout", 300)
        api_timeout = _safe_cast(timeout_cfg, int, 300)
        retries_cfg = get_setting("api_settings", f"{provider_section_key}.retries", 1)
        retry_count = _safe_cast(retries_cfg, int, 1)
        delay_cfg = get_setting("api_settings", f"{provider_section_key}.retry_delay", 2)
        retry_delay = _safe_cast(delay_cfg, float, 2.0) # Use float for backoff_factor
        logging.debug(f"vLLM: Timeout={api_timeout}, Retries={retry_count}, Delay={retry_delay}")


        # --- Input Data Handling ---
        # (Assuming input_data is text or needs extraction)
        # If input_data can be a file path, you'd handle that here.
        # This example assumes input_data is ready-to-use text content.
        if not isinstance(input_data, str):
             # Minimal handling for non-string input; adjust as needed
             logging.warning(f"vLLM: Received non-string input_data (type: {type(input_data)}). Attempting string conversion.")
             input_text = str(input_data)
        else:
             input_text = input_data

        combined_user_content = f"{custom_prompt_input}\n\n{input_text}" if custom_prompt_input else input_text
        logging.debug(f"vLLM: Combined user content (first 500): {combined_user_content[:500]}...")

        # --- Prepare Request ---
        headers = {"Content-Type": "application/json"}
        if vllm_api_key:
            headers["Authorization"] = f"Bearer {vllm_api_key}"
            logging.debug("vLLM: Added Authorization header.")

        # Build payload, omitting model if it's None or empty
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt_to_use},
                {"role": "user", "content": combined_user_content},
            ],
            "temperature": temp_to_use,
            "stream": streaming_to_use,
            "top_p": top_p_to_use,
            "top_k": top_k_to_use if top_k_to_use >= 0 else -1, # vLLM uses -1 to disable top_k
            # "min_p": min_p_to_use, # Check if vLLM OpenAI endpoint supports min_p
            "max_tokens": max_tokens_to_use,
        }
        if model_to_use: # Only include model if specified
             payload["model"] = model_to_use
        # Remove None values if the API doesn't like them (though OpenAI endpoints usually ignore extras)
        # payload = {k: v for k, v in payload.items() if v is not None}

        logging.debug(f"vLLM: Payload prepared (excluding messages): {{k: v for k, v in payload.items() if k != 'messages'}}")

        # --- Execute Request ---
        session = requests.Session()
        retry_strategy = Retry(
            total=retry_count,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504], # Consider adding 500
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        if streaming_to_use:
            logging.debug(f"vLLM: Posting streaming request to {api_url_to_use}")
            response = session.post(
                url=api_url_to_use,
                headers=headers,
                json=payload,
                stream=True,
                timeout=api_timeout
            )
            logging.debug(f"vLLM: Stream Response Status: {response.status_code}")
            response.raise_for_status() # Raise HTTP errors

            # --- Stream Processing Generator ---
            def stream_generator():
                for line in response.iter_lines():
                    line = line.decode("utf-8").strip()
                    if line == "": continue
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]": break
                        try:
                            data_json = json.loads(data_str)
                            # Standard OpenAI SSE format
                            if 'choices' in data_json and data_json['choices']:
                                delta = data_json['choices'][0].get('delta', {})
                                chunk = delta.get('content', '')
                                if chunk: # Only yield non-empty chunks
                                     yield chunk
                            else:
                                logging.warning(f"vLLM: Unexpected SSE data structure: {data_json}")
                        except json.JSONDecodeError:
                            logging.error(f"vLLM: Error decoding JSON from SSE line: {line}")
                            continue
                        except Exception as stream_err:
                             logging.error(f"vLLM: Error processing SSE chunk: {stream_err} - Line: {line}", exc_info=True)
                             continue # Skip malformed chunk
            logging.info("vLLM: Returning stream generator.")
            return stream_generator()

        else: # Non-streaming
            logging.debug(f"vLLM: Posting non-streaming request to {api_url_to_use}")
            response = session.post(
                url=api_url_to_use,
                headers=headers,
                json=payload,
                timeout=api_timeout
            )
            logging.debug(f"vLLM: Response Status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                # logging.debug(f"vLLM: Response Data: {response_data}") # Careful logging large responses
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    # Handle potential variations in response structure
                    message_content = response_data['choices'][0].get('message', {}).get('content')
                    if message_content:
                         chat_response = message_content.strip()
                         logging.info("vLLM: Chat request successful.")
                         return chat_response
                    else:
                         logging.warning("vLLM: 'content' not found in response message.")
                         return "vLLM: Chat response content missing."
                else:
                    logging.warning("vLLM: 'choices' array not found or empty in response data.")
                    return "vLLM: Chat response format unexpected (no choices)."
            else:
                logging.error(f"vLLM: Chat request failed. Status: {response.status_code}, Body: {response.text[:500]}...")
                return f"vLLM: Failed request. Status: {response.status_code}"

    # --- Exception Handling ---
    except requests.exceptions.RequestException as e:
        logging.error(f"vLLM: RequestException: {e}", exc_info=True)
        return f"vLLM: Network or Request Error: {e}"
    except json.JSONDecodeError as e:
        # This might happen if the non-streaming response isn't valid JSON
        logging.error(f"vLLM: Error decoding JSON response: {e}. Response text: {response.text[:500]}...", exc_info=True)
        return f"vLLM: Error parsing API response."
    except Exception as e:
        logging.error(f"vLLM: Unexpected error in chat_with_vllm: {e}", exc_info=True)
        # Determine return type based on streaming flag for consistency
        error_message = f"vLLM: Unexpected error occurred: {e}"
        if 'streaming_to_use' in locals() and streaming_to_use:
             def error_generator():
                  yield error_message
             return error_generator()
        else:
             return error_message


def chat_with_custom_openai(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=None, maxp=None, model=None, minp=None, topk=None):
    """Interacts with a custom OpenAI-compatible API using settings from config."""
    log = logging.getLogger(__name__)
    # --- Use the key defined in config.toml [api_settings.custom] ---
    provider_section_key = "custom"

    try:
        # --- Load Settings ---
        # API Key: Priority -> argument > env var (name from config) > config file key (fallback)
        custom_api_key = api_key # Prioritize argument
        if not custom_api_key:
            api_key_env_var_name = get_setting("api_settings", f"{provider_section_key}.api_key_env_var", "CUSTOM_API_KEY")
            custom_api_key = os.environ.get(api_key_env_var_name)
            if custom_api_key:
                logging.info(f"Custom OpenAI: Using API key from environment variable {api_key_env_var_name}")
            # Optional: Fallback to reading directly from config (less secure)
            # else:
            #     custom_api_key = get_setting("api_settings", f"{provider_section_key}.api_key")
            #     if custom_api_key:
            #         logging.warning("Custom OpenAI: Using API key found directly in config file (less secure).")

        if not custom_api_key:
            logging.error("Custom OpenAI: API key not found in argument, environment variable, or config.")
            return "Custom OpenAI: API Key Not Provided/Found/Configured."
        logging.debug(f"Custom OpenAI: Using API Key: {custom_api_key[:5]}...{custom_api_key[-5:]}")

        # API URL: Load from config (No argument for this one)
        default_api_url = "http://localhost:1234/v1/chat/completions" # Sensible default
        custom_api_url = get_setting("api_settings", f"{provider_section_key}.api_url", default_api_url)
        if not custom_api_url:
             logging.error(f"Custom OpenAI: API URL not configured in [api_settings.{provider_section_key}].")
             return f"Custom OpenAI: API URL is not configured."
        logging.debug(f"Custom OpenAI: Using API URL: {custom_api_url}")

        # Model: Priority -> argument > config default
        loaded_model = model
        if not loaded_model:
            loaded_model = get_setting("api_settings", f"{provider_section_key}.model", "custom-model-alpha")
        logging.debug(f"Custom OpenAI: Using model: {loaded_model}")

        # Streaming: Priority -> argument > config default
        if streaming is None:
            streaming_cfg = get_setting("api_settings", f"{provider_section_key}.streaming", False)
            streaming_loaded = str(streaming_cfg).lower() == 'true' if isinstance(streaming_cfg, str) else bool(streaming_cfg)
        else:
             streaming_loaded = bool(streaming) # Ensure boolean
        logging.debug(f"Custom OpenAI: Streaming mode: {streaming_loaded}")

        # Temperature: Priority -> argument > config default
        if temp is None:
            temp_cfg = get_setting("api_settings", f"{provider_section_key}.temperature", 0.7)
            temp_loaded = _safe_cast(temp_cfg, float, 0.7)
        else:
             temp_loaded = _safe_cast(temp, float, 0.7)
        logging.debug(f"Custom OpenAI: Using temperature: {temp_loaded}")

        # Top_p (maxp): Priority -> argument > config default
        if maxp is None:
            maxp_cfg = get_setting("api_settings", f"{provider_section_key}.top_p", 1.0)
            top_p_value = _safe_cast(maxp_cfg, float, 1.0)
        else:
             top_p_value = _safe_cast(maxp, float, 1.0)
        logging.debug(f"Custom OpenAI: Using top_p: {top_p_value}")

        # Min_p (minp): Priority -> argument > config default
        if minp is None:
            minp_cfg = get_setting("api_settings", f"{provider_section_key}.min_p", 0.0)
            min_p_value = _safe_cast(minp_cfg, float, 0.0)
        else:
            min_p_value = _safe_cast(minp, float, 0.0)
        logging.debug(f"Custom OpenAI: Using min_p: {min_p_value}")

        # Top_k (topk): Priority -> argument > config default
        if topk is None:
            topk_cfg = get_setting("api_settings", f"{provider_section_key}.top_k", 0) # Default 0 often disables it
            top_k_value = _safe_cast(topk_cfg, int, 0)
        else:
            top_k_value = _safe_cast(topk, int, 0)
        logging.debug(f"Custom OpenAI: Using top_k: {top_k_value}")

        # Max Tokens: Load from config
        max_tokens_cfg = get_setting("api_settings", f"{provider_section_key}.max_tokens", 4096)
        max_tokens_loaded = _safe_cast(max_tokens_cfg, int, 4096)
        logging.debug(f"Custom OpenAI: Using max_tokens: {max_tokens_loaded}")

        # System Message: Priority -> argument > fixed default
        if system_message is None:
            system_message = "You are a helpful AI assistant." # Fixed default
        logging.debug(f"Custom OpenAI: Using system message: {system_message[:100]}...")

        # Timeout, Retries, Delay: Load from config
        timeout_cfg = get_setting("api_settings", f"{provider_section_key}.timeout", 120)
        api_timeout_loaded = _safe_cast(timeout_cfg, int, 120)
        retries_cfg = get_setting("api_settings", f"{provider_section_key}.retries", 2)
        retry_count_loaded = _safe_cast(retries_cfg, int, 2)
        delay_cfg = get_setting("api_settings", f"{provider_section_key}.retry_delay", 5)
        retry_delay_loaded = _safe_cast(delay_cfg, int, 5) # Or float
        logging.debug(f"Custom OpenAI: Timeout={api_timeout_loaded}, Retries={retry_count_loaded}, Delay={retry_delay_loaded}")

        # --- Prepare Request ---
        headers = {
            'Authorization': f'Bearer {custom_api_key}',
            'Content-Type': 'application/json'
        }
        # Combine input data and custom prompt
        combined_prompt = f"{input_data}\n\n{custom_prompt_arg}" if custom_prompt_arg else input_data
        logging.debug(f"Custom OpenAI: Combined prompt (first 500 chars): {combined_prompt[:500]}...")

        data = {
            "model": loaded_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": combined_prompt}
            ],
            "max_tokens": max_tokens_loaded,
            "temperature": temp_loaded,
            "stream": streaming_loaded,
            "top_p": top_p_value,
            "min_p": min_p_value,
             # Only include top_k if it's likely supported and not the disable value (e.g., 0 or -1)
             # Check your specific custom API's documentation
            # "top_k": top_k_value if top_k_value > 0 else None # Example: only include if > 0
        }
        # Add top_k conditionally if API supports it and value is meaningful
        if top_k_value > 0: # Adjust condition based on API spec (e.g., maybe >= 0)
             data["top_k"] = top_k_value
             logging.debug(f"Custom OpenAI: Including top_k={top_k_value} in payload")


        # --- Execute Request ---
        session = requests.Session()
        retry_strategy = Retry(
            total=retry_count_loaded,
            backoff_factor=retry_delay_loaded,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        # Mount for both http and https, as custom URLs might use either
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        if streaming_loaded:
            logging.debug(f"Custom OpenAI: Posting streaming request to {custom_api_url}")
            response = session.post(
                custom_api_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=api_timeout_loaded
            )
            # logging.debug(f"Custom OpenAI: Raw Response Status: {response.status_code}")
            response.raise_for_status()

            # --- Stream Processing ---
            def stream_generator() -> Generator[str, None, None]: # Added type hint
                try:
                    for line in response.iter_lines():
                        if not line: continue
                        decoded_line = line.decode("utf-8").strip()

                        if decoded_line.startswith("data:"):
                            data_str = decoded_line[len("data:") :].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                chunk = data_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if chunk: # Only yield non-empty chunks
                                     yield chunk
                            except json.JSONDecodeError:
                                logging.error(f"Custom OpenAI: Error decoding streaming JSON: '{data_str}'")
                            except (IndexError, KeyError):
                                logging.error(f"Custom OpenAI: Unexpected streaming JSON structure: '{data_str}'")
                finally:
                     # Ensure response is closed even if generator isn't fully consumed
                     response.close()
                     logging.debug("Custom OpenAI: Streaming response closed.")

            return stream_generator() # Return the generator

        else: # Non-streaming
            logging.debug(f"Custom OpenAI: Posting non-streaming request to {custom_api_url}")
            response = session.post(
                custom_api_url,
                headers=headers,
                json=data,
                timeout=api_timeout_loaded
            )
            logging.debug(f"Custom OpenAI: Response Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    # logging.debug(f"Custom OpenAI: Response Data: {response_data}")
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        chat_response = response_data['choices'][0].get('message', {}).get('content', '').strip()
                        if chat_response:
                            logging.info("Custom OpenAI: Chat request successful.")
                            return chat_response
                        else:
                            logging.warning("Custom OpenAI: 'content' field missing or empty in response choice.")
                            return "Custom OpenAI: Response content empty."
                    else:
                        logging.warning("Custom OpenAI: 'choices' array missing or empty in response data.")
                        return "Custom OpenAI: Response format unexpected (no choices)."
                except json.JSONDecodeError as e:
                     logging.error(f"Custom OpenAI: Error decoding non-streaming JSON response: {e}", exc_info=True)
                     logging.debug(f"Custom OpenAI: Raw response text: {response.text[:500]}...") # Log raw text on decode error
                     return "Custom OpenAI: Error parsing API response."
            else:
                logging.error(f"Custom OpenAI: Chat request failed. Status: {response.status_code}, Body: {response.text[:500]}...")
                return f"Custom OpenAI: Failed request. Status: {response.status_code}"

    # --- Exception Handling ---
    except requests.exceptions.RequestException as e:
        logging.error(f"Custom OpenAI: RequestException: {e}", exc_info=True)
        return f"Custom OpenAI: Network or Request Error: {e}"
    except Exception as e:
        logging.error(f"Custom OpenAI: Unexpected error: {e}", exc_info=True)
        return f"Custom OpenAI: Unexpected error occurred: {e}"


# --- Refactored chat_with_custom_openai_2 ---
def chat_with_custom_openai_2(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=None, maxp=None, model=None, minp=None, topk=None):
    """Interacts with a second custom OpenAI-compatible API using settings from config."""
    log = logging.getLogger(__name__)
    provider_section_key = "custom_2" # Key used in [api_settings.*] in config.toml
    logging.info(f"Custom OpenAI API-2: Chat request initiated.")

    try:
        # --- Load Settings ---
        # API Key: Priority -> argument > env var (name from config) > config file key (fallback)
        api_key_to_use = api_key # Prioritize argument
        if not api_key_to_use:
            api_key_env_var_name = get_setting("api_settings", f"{provider_section_key}.api_key_env_var", "CUSTOM_2_API_KEY")
            api_key_to_use = os.environ.get(api_key_env_var_name)
            if api_key_to_use:
                logging.info(f"Custom OpenAI API-2: Using API key from environment variable {api_key_env_var_name}")
            # else: # Optional fallback
            #     api_key_to_use = get_setting("api_settings", f"{provider_section_key}.api_key")
            #     if api_key_to_use: logging.warning("Custom OpenAI API-2: Using API key found directly in config (less secure).")

        if not api_key_to_use:
            logging.error("Custom OpenAI API-2: API key not found in argument, environment, or config.")
            return "Custom OpenAI API-2: API Key Not Provided/Found/Configured."
        logging.debug(f"Custom OpenAI API-2: Using API Key: {api_key_to_use[:5]}...{api_key_to_use[-5:]}")

        # API URL: Load from config (Required)
        api_url = get_setting("api_settings", f"{provider_section_key}.api_url")
        if not api_url:
            logging.error(f"Custom OpenAI API-2: API URL not found in config [api_settings.{provider_section_key}].")
            return f"Custom OpenAI API-2: API URL not configured."
        logging.debug(f"Custom OpenAI API-2: Using API URL: {api_url}")

        # Model: Priority -> argument > config default
        api_model = model
        if not api_model:
            api_model = get_setting("api_settings", f"{provider_section_key}.model", "custom-model-gamma")
        logging.debug(f"Custom OpenAI API-2: Using model: {api_model}")

        # Streaming: Priority -> argument > config default
        if streaming is None:
            streaming_cfg = get_setting("api_settings", f"{provider_section_key}.streaming", False)
            api_streaming = _safe_cast(streaming_cfg, bool, False)
        else:
             api_streaming = _safe_cast(streaming, bool, False)
        logging.debug(f"Custom OpenAI API-2: Streaming mode: {api_streaming}")

        # Temperature: Priority -> argument > config default
        if temp is None:
            temp_cfg = get_setting("api_settings", f"{provider_section_key}.temperature", 0.7)
            api_temp = _safe_cast(temp_cfg, float, 0.7)
        else:
             api_temp = _safe_cast(temp, float, 0.7)
        logging.debug(f"Custom OpenAI API-2: Using temperature: {api_temp}")

        # --- Load Sampling Params (top_p, min_p, top_k) ---
        # Map UI args (maxp, minp, topk) to config keys (top_p, min_p, top_k)
        # Priority: Argument -> Config -> Default (usually 0, 1.0, or None depending on API)

        # Top_p (from maxp argument)
        if maxp is None:
             top_p_cfg = get_setting("api_settings", f"{provider_section_key}.top_p", 1.0)
             api_top_p = _safe_cast(top_p_cfg, float, 1.0)
        else:
             api_top_p = _safe_cast(maxp, float, 1.0)
        logging.debug(f"Custom OpenAI API-2: Using top_p: {api_top_p}")

        # Min_p (from minp argument)
        if minp is None:
             min_p_cfg = get_setting("api_settings", f"{provider_section_key}.min_p", 0.0)
             api_min_p = _safe_cast(min_p_cfg, float, 0.0)
        else:
             api_min_p = _safe_cast(minp, float, 0.0)
        logging.debug(f"Custom OpenAI API-2: Using min_p: {api_min_p}")

        # Top_k (from topk argument)
        if topk is None:
             top_k_cfg = get_setting("api_settings", f"{provider_section_key}.top_k", 0) # Default 0 often disables top_k
             api_top_k = _safe_cast(top_k_cfg, int, 0)
        else:
             api_top_k = _safe_cast(topk, int, 0)
        logging.debug(f"Custom OpenAI API-2: Using top_k: {api_top_k}")

        # Max Tokens: Load from config
        max_tokens_cfg = get_setting("api_settings", f"{provider_section_key}.max_tokens", 4096)
        api_max_tokens = _safe_cast(max_tokens_cfg, int, 4096)
        logging.debug(f"Custom OpenAI API-2: Using max_tokens: {api_max_tokens}")

        # System Message: Priority -> argument > fixed default
        if system_message is None:
            system_message = "You are a helpful AI assistant."
        logging.debug(f"Custom OpenAI API-2: Using system message: {system_message[:100]}...")

        # Timeout, Retries, Delay: Load from config
        timeout_cfg = get_setting("api_settings", f"{provider_section_key}.timeout", 120)
        api_timeout = _safe_cast(timeout_cfg, int, 120)
        retries_cfg = get_setting("api_settings", f"{provider_section_key}.retries", 2)
        retry_count = _safe_cast(retries_cfg, int, 2)
        delay_cfg = get_setting("api_settings", f"{provider_section_key}.retry_delay", 5)
        retry_delay = _safe_cast(delay_cfg, float, 5.0) # Use float for backoff_factor
        logging.debug(f"Custom OpenAI API-2: Timeout={api_timeout}, Retries={retry_count}, Delay={retry_delay}")

        # --- Prepare Request ---
        headers = {
            'Authorization': f'Bearer {api_key_to_use}',
            'Content-Type': 'application/json'
        }
        # Combine input_data and custom_prompt_arg
        openai_prompt = f"{input_data}\n\n{custom_prompt_arg}" if custom_prompt_arg else input_data
        logging.debug(f"Custom OpenAI API-2: Combined prompt (first 500 chars): {openai_prompt[:500]}...")

        # Construct payload using loaded values
        # Add top_p, min_p, top_k if the target API supports them
        data = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": openai_prompt}
            ],
            "max_tokens": api_max_tokens,
            "temperature": api_temp,
            "stream": api_streaming,
            "top_p": api_top_p, # Include if API supports it
            # "min_p": api_min_p, # Include if API supports it (less common)
            # "top_k": api_top_k, # Include if API supports it
        }
        # Clean payload: remove keys with None values if the API dislikes them
        # data = {k: v for k, v in data.items() if v is not None}

        # --- Execute Request ---
        session = requests.Session()
        retry_strategy = Retry(
            total=retry_count,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter) # Mount for http
        session.mount("https://", adapter) # Also mount for https if URL could be either

        if api_streaming:
            logging.debug(f"Custom OpenAI API-2: Posting streaming request to {api_url}")
            response = session.post(
                api_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=api_timeout
            )
            # logging.debug(f"Custom OpenAI API-2: Raw Response Status: {response.status_code}")
            response.raise_for_status()

            # --- Stream Processing Generator ---
            def stream_generator():
                collected_messages = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8").strip()
                    if line == "": continue
                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]": break
                        try:
                            data_json = json.loads(data_str)
                            # Adapt parsing based on actual API stream format
                            chunk = data_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if chunk: # Only yield non-empty chunks
                                collected_messages += chunk
                                yield chunk
                        except (json.JSONDecodeError, IndexError, KeyError) as e:
                            logging.error(f"Custom OpenAI API-2: Error parsing stream chunk: {e} on line: {line}")
                            continue
                # Optionally yield the full response at the end if needed by caller
                # yield collected_messages
            return stream_generator()

        else: # Non-streaming
            logging.debug(f"Custom OpenAI API-2: Posting non-streaming request to {api_url}")
            response = session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=api_timeout
            )
            logging.debug(f"Custom OpenAI API-2: Response Status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                # logging.debug(f"Custom OpenAI API-2: Response Data: {response_data}")
                # Adapt parsing based on actual API response format
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    message_content = response_data['choices'][0].get('message', {}).get('content')
                    if message_content:
                        chat_response = message_content.strip()
                        logging.info("Custom OpenAI API-2: Chat request successful.")
                        # logging.debug(f"Custom OpenAI API-2: Chat response: {chat_response[:200]}...")
                        return chat_response
                    else:
                         logging.warning("Custom OpenAI API-2: 'content' missing in response choice message.")
                         return "Custom OpenAI API-2: Chat response content missing."
                else:
                    logging.warning("Custom OpenAI API-2: 'choices' array missing or empty in response data.")
                    return "Custom OpenAI API-2: Chat response format unexpected (no choices)."
            else:
                logging.error(f"Custom OpenAI API-2: Chat request failed. Status: {response.status_code}, Body: {response.text[:500]}...")
                return f"Custom OpenAI API-2: Failed request. Status: {response.status_code}"

    # --- Exception Handling ---
    except requests.exceptions.RequestException as e:
        logging.error(f"Custom OpenAI API-2: RequestException: {e}", exc_info=True)
        return f"Custom OpenAI API-2: Network or Request Error: {e}"
    except json.JSONDecodeError as e:
        logging.error(f"Custom OpenAI API-2: Error decoding JSON response: {e}", exc_info=True)
        return f"Custom OpenAI API-2: Error parsing API response."
    except Exception as e:
        logging.error(f"Custom OpenAI API-2: Unexpected error: {e}", exc_info=True)
        return f"Custom OpenAI API-2: Unexpected error occurred: {e}"


def save_summary_to_file(summary, file_path):
    logging.debug("Now saving summary to file...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    summary_file_path = os.path.join(os.path.dirname(file_path), base_name + '_summary.txt')
    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
    logging.debug("Opening summary file for writing, *segments.json with *_summary.txt")
    with open(summary_file_path, 'w') as file:
        file.write(summary)
    logging.info(f"Summary saved to file: {summary_file_path}")

#
#
#######################################################################################################################




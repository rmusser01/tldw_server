# Summarization_General_Lib.py
#########################################
# General Summarization Library
# This library is used to perform summarization.
#
####
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. summarize_with_openai(api_key, file_path, custom_prompt_arg)
# 3. summarize_with_anthropic(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
# 4. summarize_with_cohere(api_key, file_path, model, custom_prompt_arg)
# 5. summarize_with_groq(api_key, file_path, model, custom_prompt_arg)
#
#
####################
# Import necessary libraries
import inspect
import json
import os
import time
from typing import Optional, Any, Generator, Literal
#
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

#
# Import Local
from PoC_Version.App_Function_Libraries.Audio.Audio_Transcription_Lib import convert_to_wav, speech_to_text
from PoC_Version.App_Function_Libraries.Chunk_Lib import semantic_chunking, rolling_summarize, recursive_summarize_chunks, \
    improved_chunking_process
from PoC_Version.App_Function_Libraries.Audio.Diarization_Lib import combine_transcription_and_diarization
from PoC_Version.App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_llama, summarize_with_kobold, \
    summarize_with_oobabooga, summarize_with_tabbyapi, summarize_with_vllm, summarize_with_local_llm, \
    summarize_with_ollama, summarize_with_custom_openai, summarize_with_custom_openai_2
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_media_to_database
from PoC_Version.App_Function_Libraries.Utils.Utils import (load_and_log_configs, sanitize_filename, clean_youtube_url,
                                                create_download_directory, is_valid_url, logging)
from PoC_Version.App_Function_Libraries.Video_DL_Ingestion_Lib import download_video, extract_video_info
#
#######################################################################################################################
# Function Definitions
#

loaded_config_data = load_and_log_configs()
openai_api_key = loaded_config_data.get('openai_api', {}).get('api_key', None)

def summarize(
    input_data: str,
    custom_prompt_arg: Optional[str],
    api_name: str,
    api_key: Optional[str],
    temp: Optional[float],
    system_message: Optional[str],
    streaming: Optional[bool] = False
):
    try:
        logging.debug(f"api_name type: {type(api_name)}, value: {api_name}")
        if api_name.lower() == "openai":
            return summarize_with_openai(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "anthropic":
            return summarize_with_anthropic(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "cohere":
            return summarize_with_cohere(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "google":
            return summarize_with_google(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "groq":
            return summarize_with_groq(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "huggingface":
            return summarize_with_huggingface(api_key, input_data, custom_prompt_arg, temp, streaming)
        elif api_name.lower() == "openrouter":
            return summarize_with_openrouter(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "deepseek":
            return summarize_with_deepseek(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "mistral":
            return summarize_with_mistral(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "llama.cpp":
            return summarize_with_llama(input_data, custom_prompt_arg, api_key, temp, system_message, streaming)
        elif api_name.lower() == "kobold":
            return summarize_with_kobold(input_data, api_key, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "ooba":
            return summarize_with_oobabooga(input_data, api_key, custom_prompt_arg, system_message,
                                                                      temp=None, api_url=None, streaming=False)
        elif api_name.lower() == "tabbyapi":
            return summarize_with_tabbyapi(input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "vllm":
            return summarize_with_vllm(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "local-llm":
            return summarize_with_local_llm(input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "huggingface":
            return summarize_with_huggingface(api_key, input_data, custom_prompt_arg, temp, streaming)#system_message)
        elif api_name.lower() == "custom-openai-api":
            return summarize_with_custom_openai(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "custom-openai-api-2":
            return summarize_with_custom_openai(api_key, input_data, custom_prompt_arg, temp, system_message, streaming)
        elif api_name.lower() == "ollama":
            return summarize_with_ollama(input_data, custom_prompt_arg, None, api_key, temp, system_message, streaming)
        else:
            return f"Error: Invalid API Name {api_name}"

    except Exception as e:
        logging.error(f"Error in summarize function: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"


def extract_text_from_segments(segments):
    logging.debug(f"Segments received: {segments}")
    logging.debug(f"Type of segments: {type(segments)}")

    text = ""

    if isinstance(segments, list):
        for segment in segments:
            logging.debug(f"Current segment: {segment}")
            logging.debug(f"Type of segment: {type(segment)}")
            if 'Text' in segment:
                text += segment['Text'] + " "
            else:
                logging.warning(f"Skipping segment due to missing 'Text' key: {segment}")
    else:
        logging.warning(f"Unexpected type of 'segments': {type(segments)}")

    return text.strip()


def summarize_with_openai(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    try:
        # API key validation
        if not api_key or api_key.strip() == "":
            logging.info("OpenAI Summarize: API key not provided as parameter")
            logging.info("OpenAI Summarize: Attempting to use API key from config file")
            loaded_config_data = load_and_log_configs()
            api_key = loaded_config_data.get('openai_api', {}).get('api_key', "")
            logging.debug(f"OpenAI Summarize: Using API key from config file: {api_key[:5]}...{api_key[-5:]}")

        if not api_key or api_key.strip() == "":
            logging.error("OpenAI: #2 API key not found or is empty")
            return "OpenAI: API Key Not Provided/Found in Config file or is empty"

        openai_api_key = api_key
        logging.debug(f"OpenAI: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        # Input data handling
        logging.debug(f"OpenAI: Raw input data type: {type(input_data)}")
        logging.debug(f"OpenAI: Raw input data (first 500 chars): {str(input_data)[:500]}...")

        if isinstance(input_data, str):
            if input_data.strip().startswith('{'):
                # It's likely a JSON string
                logging.debug("OpenAI: Parsing provided JSON string data for summarization")
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(f"OpenAI: Error parsing JSON string: {str(e)}")
                    return f"OpenAI: Error parsing JSON input: {str(e)}"
            elif os.path.isfile(input_data):
                logging.debug("OpenAI: Loading JSON data from file for summarization")
                with open(input_data, 'r') as file:
                    data = json.load(file)
            else:
                logging.debug("OpenAI: Using provided string data for summarization")
                data = input_data
        else:
            data = input_data

        logging.debug(f"OpenAI: Processed data type: {type(data)}")
        logging.debug(f"OpenAI: Processed data (first 500 chars): {str(data)[:500]}...")

        # Text extraction
        if isinstance(data, dict):
            if 'summary' in data:
                logging.debug("OpenAI: Summary already exists in the loaded data")
                return data['summary']
            elif 'segments' in data:
                text = extract_text_from_segments(data['segments'])
            else:
                text = json.dumps(data)  # Convert dict to string if no specific format
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError(f"OpenAI: Invalid input data format: {type(data)}")

        logging.debug(f"OpenAI: Extracted text (first 500 chars): {text[:500]}...")
        logging.debug(f"OpenAI: Custom prompt: {custom_prompt_arg}")

        config_settings = load_and_log_configs()
        openai_model = config_settings['openai_api']['model'] or "gpt-4o"
        logging.debug(f"OpenAI: Using model: {openai_model}")

        headers = {
            'Authorization': f'Bearer {openai_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")
        logging.debug("openai: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        if temp is None:
            temp = 0.7
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."
        temp = float(temp)
        data = {
            "model": openai_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": openai_prompt}
            ],
            # FIXME - Set a Max tokens value in config file for each API
            "max_tokens": 4096,
            "temperature": temp,
            "stream": streaming
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['openai_api']['api_retries']
            retry_delay = loaded_config_data['openai_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Custom OpenAI API-2: Posting request")
            logging.debug("OpenAI: Posting streaming request")
            response = session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            def stream_generator():
                collected_messages = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8").strip()

                    if line == "":
                        continue

                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            chunk = data_json["choices"][0]["delta"].get("content", "")
                            collected_messages += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            logging.error(f"OpenAI: Error decoding JSON from line: {line}")
                            continue

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['openai_api']['api_retries']
            retry_delay = loaded_config_data['openai_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("OpenAI: Posting request")
            response = session.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("OpenAI: Summarization successful")
                    logging.debug(f"OpenAI: Summary (first 500 chars): {summary[:500]}...")
                    return summary
                else:
                    logging.warning("OpenAI: Summary not found in the response data")
                    return "OpenAI: Summary not available"
            else:
                logging.error(f"OpenAI: Summarization failed with status code {response.status_code}")
                logging.error(f"OpenAI: Error response: {response.text}")
                return f"OpenAI: Failed to process summary. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"OpenAI: Error decoding JSON: {str(e)}", exc_info=True)
        return f"OpenAI: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"OpenAI: Error making API request: {str(e)}", exc_info=True)
        return f"OpenAI: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"OpenAI: Unexpected error: {str(e)}", exc_info=True)
        return f"OpenAI: Unexpected error occurred: {str(e)}"


def summarize_with_anthropic(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False, max_retries=3, retry_delay=5):
    logging.debug("Anthropic: Summarization process starting...")
    try:
        logging.debug("Anthropic: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            anthropic_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                anthropic_api_key = api_key
                logging.info("Anthropic: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                anthropic_api_key = loaded_config_data['anthropic_api'].get('api_key')
                if anthropic_api_key:
                    logging.info("Anthropic: Using API key from config file")
                else:
                    logging.warning("Anthropic: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not anthropic_api_key or not anthropic_api_key.strip():
            logging.error("Anthropic: No valid API key available")
            return "Anthropic: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Anthropic: Using API Key: {anthropic_api_key[:5]}...{anthropic_api_key[-5:]}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("AnthropicAI: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("AnthropicAI: Using provided string data for summarization")
            data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"AnthropicAI: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"AnthropicAI: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Anthropic: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Anthropic: Invalid input data format")

        if temp is None:
            temp = 0.1
        temp = float(temp)

        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'x-api-key': anthropic_api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

        anthropic_prompt = custom_prompt_arg
        logging.debug(f"Anthropic: Prompt is {anthropic_prompt}")
        user_message = {
            "role": "user",
            "content": f"{text} \n\n\n\n{anthropic_prompt}"
        }

        model = loaded_config_data['anthropic_api']['model']

        data = {
            "model": model,
            "max_tokens": 4096,  # max possible tokens to return
            "messages": [user_message],
            "stop_sequences": ["\n\nHuman:"],
            "temperature": temp,
            "top_k": 0,
            "top_p": 1.0,
            "metadata": {
                "user_id": "example_user_id",
            },
            "stream": streaming,
            "system": system_message
        }

        for attempt in range(max_retries):
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = loaded_config_data['anthropic_api']['api_retries']
                retry_delay = loaded_config_data['anthropic_api']['api_retry_delay']

                # Configure the retry strategy
                retry_strategy = Retry(
                    total=retry_count,  # Total number of retries
                    backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                    status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
                )

                # Create the adapter
                adapter = HTTPAdapter(max_retries=retry_strategy)

                # Mount adapters for both HTTP and HTTPS
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                logging.debug("Anthropic: Posting request to API")
                response = requests.post(
                    'https://api.anthropic.com/v1/messages',
                    headers=headers,
                    json=data,
                    stream=streaming
                )

                # Check if the status code indicates success
                if response.status_code == 200:
                    if streaming:
                        # Handle streaming response
                        def stream_generator():
                            collected_text = ""
                            event_type = None
                            for line in response.iter_lines():
                                line = line.decode('utf-8').strip()
                                if line == '':
                                    continue
                                if line.startswith('event:'):
                                    event_type = line[len('event:'):].strip()
                                elif line.startswith('data:'):
                                    data_str = line[len('data:'):].strip()
                                    if data_str == '[DONE]':
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        if event_type == 'content_block_delta' and data_json.get('type') == 'content_block_delta':
                                            delta = data_json.get('delta', {})
                                            text_delta = delta.get('text', '')
                                            collected_text += text_delta
                                            yield text_delta
                                    except json.JSONDecodeError:
                                        logging.error(f"Anthropic: Error decoding JSON from line: {line}")
                                        continue
                            # Optionally, return the full collected text at the end
                            # yield collected_text
                        return stream_generator()
                    else:
                        # Non-streaming response
                        logging.debug("Anthropic: Post submittal successful")
                        response_data = response.json()
                        try:
                            # Extract the assistant's reply from the 'content' field
                            content_blocks = response_data.get('content', [])
                            summary = ''
                            for block in content_blocks:
                                if block.get('type') == 'text':
                                    summary += block.get('text', '')
                            summary = summary.strip()
                            logging.debug("Anthropic: Summarization successful")
                            logging.debug(f"Anthropic: Summary (first 500 chars): {summary[:500]}...")
                            return summary
                        except Exception as e:
                            logging.debug("Anthropic: Unexpected data in response")
                            logging.error(f"Unexpected response format from Anthropic API: {response.text}")
                            return None
                elif response.status_code == 500:  # Handle internal server error specifically
                    logging.debug("Anthropic: Internal server error")
                    logging.error("Internal server error from API. Retrying may be necessary.")
                    time.sleep(retry_delay)
                else:
                    logging.debug(f"Anthropic: Failed to summarize, status code {response.status_code}: {response.text}")
                    logging.error(f"Failed to process summary, status code {response.status_code}: {response.text}")
                    return None

            except requests.RequestException as e:
                logging.error(f"Anthropic: Network error during attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return f"Anthropic: Network error: {str(e)}"
    except FileNotFoundError as e:
        logging.error(f"Anthropic: File not found: {input_data}")
        return f"Anthropic: File not found: {input_data}"
    except json.JSONDecodeError as e:
        logging.error(f"Anthropic: Invalid JSON format in file: {input_data}")
        return f"Anthropic: Invalid JSON format in file: {input_data}"
    except Exception as e:
        logging.error(f"Anthropic: Error in processing: {str(e)}")
        return f"Anthropic: Error occurred while processing summary with Anthropic: {str(e)}"


# Summarize with Cohere
def summarize_with_cohere(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    logging.debug("Cohere: Summarization process starting...")
    try:
        logging.debug("Cohere: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            cohere_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                cohere_api_key = api_key
                logging.info("Cohere: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                cohere_api_key = loaded_config_data['cohere_api'].get('api_key')
                if cohere_api_key:
                    logging.info("Cohere: Using API key from config file")
                else:
                    logging.warning("Cohere: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not cohere_api_key or not cohere_api_key.strip():
            logging.error("Cohere: No valid API key available")
            return "Cohere: API Key Not Provided/Found in Config file or is empty"

        if custom_prompt_arg is None:
            custom_prompt_arg = ""

        if system_message is None:
            system_message = ""

        logging.debug(f"Cohere: Using API Key: {cohere_api_key[:5]}...{cohere_api_key[-5:] if cohere_api_key else None}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Cohere: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("Cohere: Using provided string data for summarization")
            data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"Cohere: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"Cohere: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Cohere: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Cohere: Invalid input data format")

        cohere_model = loaded_config_data['cohere_api']['model']

        if temp is None:
            temp = 0.3
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'Authorization': f'Bearer {cohere_api_key}'
        }

        cohere_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug(f"Cohere: Prompt being sent is {cohere_prompt}")

        data = {
            "preamble": system_message,
            "message": cohere_prompt,
            "model": cohere_model,
#            "connectors": [{"id": "web-search"}],
            "temperature": temp,
            "streaming": streaming
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['cohere_api']['api_retries']
            retry_delay = loaded_config_data['cohere_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Cohere: Submitting streaming request to API endpoint")
            response = session.post(
                'https://api.cohere.ai/v1/chat',
                headers=headers,
                json=data,
                stream=True  # Enable response streaming
            )
            response.raise_for_status()

            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        try:
                            data_json = json.loads(decoded_line)
                            if 'response' in data_json:
                                chunk = data_json['response']
                                yield chunk
                            elif 'token' in data_json:
                                # For token-based streaming (if applicable)
                                chunk = data_json['token']
                                yield chunk
                            elif 'text' in data_json:
                                # For text-based streaming
                                chunk = data_json['text']
                                yield chunk
                            else:
                                logging.debug(f"Cohere: Unhandled streaming data: {data_json}")
                        except json.JSONDecodeError:
                            logging.error(f"Cohere: Error decoding JSON from line: {decoded_line}")
                            continue

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['cohere_api']['api_retries']
            retry_delay = loaded_config_data['cohere_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Cohere: Submitting request to API endpoint")
            response = session.post('https://api.cohere.ai/v1/chat', headers=headers, json=data)
            response_data = response.json()
            logging.debug(f"API Response Data: {response_data}")

            if response.status_code == 200:
                if 'text' in response_data:
                    summary = response_data['text'].strip()
                    logging.debug("Cohere: Summarization successful")
                    return summary
                elif 'response' in response_data:
                    # Adjust if the API returns 'response' field instead of 'text'
                    summary = response_data['response'].strip()
                    logging.debug("Cohere: Summarization successful")
                    return summary
                else:
                    logging.error("Cohere: Expected data not found in API response.")
                    return "Cohere: Expected data not found in API response."
            else:
                logging.error(f"Cohere: API request failed with status code {response.status_code}: {response.text}")
                return f"Cohere: API request failed: {response.text}"

    except Exception as e:
        logging.error(f"Cohere: Error in processing: {str(e)}", exc_info=True)
        return f"Cohere: Error occurred while processing summary with Cohere: {str(e)}"



# https://console.groq.com/docs/quickstart
def summarize_with_groq(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    logging.debug("Groq: Summarization process starting...")
    try:
        logging.debug("Groq: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            groq_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                groq_api_key = api_key
                logging.info("Groq: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                groq_api_key = loaded_config_data['groq_api'].get('api_key')
                if groq_api_key:
                    logging.info("Groq: Using API key from config file")
                else:
                    logging.warning("Groq: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not groq_api_key or not groq_api_key.strip():
            logging.error("Groq: No valid API key available")
            return "Groq: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Groq: Using API Key: {groq_api_key[:5]}...{groq_api_key[-5:]}")

        # Input data handling
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Groq: Loading JSON data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("Groq: Using provided string data for summarization")
            data = input_data

        # Debug logging to identify sent data
        logging.debug(f"Groq: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"Groq: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            logging.debug("Groq: Summary already exists in the loaded data")
            return data['summary']

        # Text extraction
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Groq: Invalid input data format")

        # Set the model to be used
        groq_model = loaded_config_data['groq_api']['model']

        if temp is None:
            temp = 0.2
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'Authorization': f'Bearer {groq_api_key}',
            'Content-Type': 'application/json'
        }

        groq_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug(f"Groq: Prompt being sent is {groq_prompt}")

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": groq_prompt,
                }
            ],
            "model": groq_model,
            "temperature": temp,
            "stream": streaming
        }

        logging.debug("Groq: Submitting request to API endpoint")
        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['groq_api']['api_retries']
            retry_delay = loaded_config_data['groq_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=data,
                stream=True  # Enable response streaming
            )
            response.raise_for_status()

            def stream_generator():
                collected_messages = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8").strip()

                    if line == "":
                        continue

                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            chunk = data_json["choices"][0]["delta"].get("content", "")
                            collected_messages += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            logging.error(f"Groq: Error decoding JSON from line: {line}")
                            continue
                # Optionally, you can return the full collected message at the end
                # yield collected_messages

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['groq_api']['api_retries']
            retry_delay = loaded_config_data['groq_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=data
            )

            response_data = response.json()
            logging.debug("API Response Data: %s", response_data)

            if response.status_code == 200:
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Groq: Summarization successful")
                    return summary
                else:
                    logging.error("Groq: Expected data not found in API response.")
                    return "Groq: Expected data not found in API response."
            else:
                logging.error(f"Groq: API request failed with status code {response.status_code}: {response.text}")
                return f"Groq: API request failed: {response.text}"

    except Exception as e:
        logging.error("Groq: Error in processing: %s", str(e), exc_info=True)
        return f"Groq: Error occurred while processing summary with Groq: {str(e)}"


def summarize_with_openrouter(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False,):
    import requests
    import json
    global openrouter_model, openrouter_api_key
    try:
        logging.debug("OpenRouter: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            openrouter_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                openrouter_api_key = api_key
                logging.info("OpenRouter: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                openrouter_api_key = loaded_config_data['openrouter_api'].get('api_key')
                if openrouter_api_key:
                    logging.info("OpenRouter: Using API key from config file")
                else:
                    logging.warning("OpenRouter: No API key found in config file")

        # Model Selection validation
        logging.debug("OpenRouter: Validating model selection")
        loaded_config_data = load_and_log_configs()
        openrouter_model = loaded_config_data['openrouter_api']['model']
        logging.debug(f"OpenRouter: Using model from config file: {openrouter_model}")

        # Final check to ensure we have a valid API key
        if not openrouter_api_key or not openrouter_api_key.strip():
            logging.error("OpenRouter: No valid API key available")
            raise ValueError("No valid Anthropic API key available")
    except Exception as e:
        logging.error("OpenRouter: Error in processing: %s", str(e))
        return f"OpenRouter: Error occurred while processing config file with OpenRouter: {str(e)}"

    logging.debug(f"OpenRouter: Using API Key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:]}")

    logging.debug(f"OpenRouter: Using Model: {openrouter_model}")

    if isinstance(input_data, str) and os.path.isfile(input_data):
        logging.debug("OpenRouter: Loading json data for summarization")
        with open(input_data, 'r') as file:
            data = json.load(file)
    else:
        logging.debug("OpenRouter: Using provided string data for summarization")
        data = input_data

    # DEBUG - Debug logging to identify sent data
    logging.debug(f"OpenRouter: Loaded data: {data[:500]}...(snipped to first 500 chars)")
    logging.debug(f"OpenRouter: Type of data: {type(data)}")

    if isinstance(data, dict) and 'summary' in data:
        # If the loaded data is a dictionary and already contains a summary, return it
        logging.debug("OpenRouter: Summary already exists in the loaded data")
        return data['summary']

    # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
    if isinstance(data, list):
        segments = data
        text = extract_text_from_segments(segments)
    elif isinstance(data, str):
        text = data
    else:
        raise ValueError("OpenRouter: Invalid input data format")

    openrouter_prompt = f"{input_data} \n\n\n\n{custom_prompt_arg}"

    if temp is None:
        temp = 0.1
    temp = float(temp)
    if system_message is None:
        system_message = "You are a helpful AI assistant who does whatever the user requests."

    if streaming:
        try:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['openrouter_api']['api_retries']
            retry_delay = loaded_config_data['openrouter_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("OpenRouter: Submitting streaming request to API endpoint")
            # Make streaming request
            response = session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Accept": "text/event-stream",  # Important for streaming
                },
                data=json.dumps({
                    "model": openrouter_model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": openrouter_prompt}
                    ],
                    #"max_tokens": 4096,
                    #"top_p": 1.0,
                    "temperature": temp,
                    "stream": True
                }),
                stream=True  # Enable streaming in requests
            )

            if response.status_code == 200:
                full_response = ""
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        # Remove "data: " prefix and parse JSON
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_str = line[6:]  # Remove "data: " prefix
                            if json_str.strip() == '[DONE]':
                                break
                            try:
                                json_data = json.loads(json_str)
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    delta = json_data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        print(content, end='', flush=True)  # Print streaming output
                                        full_response += content
                            except json.JSONDecodeError:
                                continue

                logging.debug("openrouter: Streaming completed successfully")
                return full_response.strip()
            else:
                error_msg = f"openrouter: Streaming API request failed with status code {response.status_code}: {response.text}"
                logging.error(error_msg)
                return error_msg

        except Exception as e:
            error_msg = f"openrouter: Error occurred while processing stream: {str(e)}"
            logging.error(error_msg)
            return error_msg
    else:
        try:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['openrouter_api']['api_retries']
            retry_delay = loaded_config_data['openrouter_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("OpenRouter: Submitting request to API endpoint")
            print("OpenRouter: Submitting request to API endpoint")
            response = session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                },
                data=json.dumps({
                    "model": openrouter_model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": openrouter_prompt}
                    ],
                    #"max_tokens": 4096,
                    #"top_p": 1.0,
                    "temperature": temp,
                    #"stream": streaming
                })
            )

            response_data = response.json()
            logging.debug("API Response Data: %s", response_data)

            if response.status_code == 200:
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("openrouter: Summarization successful")
                    print("openrouter: Summarization successful.")
                    return summary
                else:
                    logging.error("openrouter: Expected data not found in API response.")
                    return "openrouter: Expected data not found in API response."
            else:
                logging.error(f"openrouter:  API request failed with status code {response.status_code}: {response.text}")
                return f"openrouter: API request failed: {response.text}"
        except Exception as e:
            logging.error("openrouter: Error in processing: %s", str(e))
            return f"openrouter: Error occurred while processing summary with openrouter: {str(e)}"


def summarize_with_huggingface(api_key, input_data, custom_prompt_arg, temp=None, streaming=False,):
    # https://huggingface.co/docs/api-inference/tasks/chat-completion
    loaded_config_data = load_and_log_configs()
    logging.debug("HuggingFace: Summarization process starting...")
    try:
        logging.debug("HuggingFace: Loading and validating configurations")
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            huggingface_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                huggingface_api_key = api_key
                logging.info("HuggingFace: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                huggingface_api_key = loaded_config_data['huggingface_api'].get('api_key')
                logging.debug(f"HuggingFace: API key from config: {huggingface_api_key[:5]}...{huggingface_api_key[-5:]}")
                if huggingface_api_key:
                    logging.info("HuggingFace: Using API key from config file")
                else:
                    logging.warning("HuggingFace: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not huggingface_api_key or not huggingface_api_key.strip():
            logging.error("HuggingFace: No valid API key available")
            # You might want to raise an exception here or handle this case as appropriate for your application
            # FIXME
            # For example: raise ValueError("No valid Anthropic API key available")

        logging.debug(f"HuggingFace: Using API Key: {huggingface_api_key[:5]}...{huggingface_api_key[-5:]}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("HuggingFace: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("HuggingFace: Using provided string data for summarization")
            data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"HuggingFace: Loaded data: {data[:500]}...(snipped to first 500 chars)")
        logging.debug(f"HuggingFace: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("HuggingFace: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("HuggingFace: Invalid input data format")

        headers = {
            "Authorization": f"Bearer {huggingface_api_key}"
        }
        huggingface_model = loaded_config_data['huggingface_api']['model']
        API_URL = f"https://api-inference.huggingface.co/models/{huggingface_model}"
        if temp is None:
            temp = 0.1
        temp = float(temp)
        huggingface_prompt = f"{custom_prompt_arg}\n\n\n{text}"
        logging.debug(f"HuggingFace: Prompt being sent is {huggingface_prompt}")
        data_payload = {
            "inputs": huggingface_prompt,
            "max_tokens": 4096,
            "stream": streaming,
            "temperature": temp
        }

        logging.debug("HuggingFace: Submitting request...")
        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['huggingface_api']['api_retries']
            retry_delay = loaded_config_data['huggingface_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(API_URL, headers=headers, json=data_payload, stream=True)
            response.raise_for_status()

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
                                if 'token' in data_json:
                                    token_text = data_json['token'].get('text', '')
                                    yield token_text
                                elif 'generated_text' in data_json:
                                    # Some models may send the full generated text
                                    generated_text = data_json['generated_text']
                                    yield generated_text
                                else:
                                    logging.debug(f"HuggingFace: Unhandled streaming data: {data_json}")
                            except json.JSONDecodeError:
                                logging.error(f"HuggingFace: Error decoding JSON from line: {decoded_line}")
                                continue
                # Optionally, yield the final collected text
                # yield collected_text

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['huggingface_api']['api_retries']
            retry_delay = loaded_config_data['huggingface_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(API_URL, headers=headers, json=data_payload)

            if response.status_code == 200:
                response_json = response.json()
                logging.debug(f"HuggingFace: Response JSON: {response_json}")
                if isinstance(response_json, dict) and 'generated_text' in response_json:
                    chat_response = response_json['generated_text'].strip()
                elif isinstance(response_json, list) and len(response_json) > 0 and 'generated_text' in response_json[0]:
                    chat_response = response_json[0]['generated_text'].strip()
                else:
                    logging.error("HuggingFace: Expected 'generated_text' in response")
                    return "HuggingFace: Expected 'generated_text' in API response."

                logging.debug("HuggingFace: Summarization successful")
                return chat_response
            else:
                logging.error(f"HuggingFace: Summarization failed with status code {response.status_code}: {response.text}")
                return f"HuggingFace: Failed to process summary. Status code: {response.status_code}"

    except Exception as e:
        logging.error("HuggingFace: Error in processing: %s", str(e), exc_info=True)
        return f"HuggingFace: Error occurred while processing summary with HuggingFace: {str(e)}"


def summarize_with_deepseek(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    # https://api-docs.deepseek.com/api/create-chat-completion
    logging.debug("DeepSeek: Summarization process starting...")
    try:
        logging.debug("DeepSeek: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            deepseek_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                deepseek_api_key = api_key
                logging.info("DeepSeek: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                deepseek_api_key = loaded_config_data['deepseek_api'].get('api_key')
                if deepseek_api_key:
                    logging.info("DeepSeek: Using API key from config file")
                else:
                    logging.warning("DeepSeek: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not deepseek_api_key or not deepseek_api_key.strip():
            logging.error("DeepSeek: No valid API key available")
            return "DeepSeek: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"DeepSeek: Using API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:]}")

        # Input data handling
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("DeepSeek: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("DeepSeek: Using provided string data for summarization")
            data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"DeepSeek: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"DeepSeek: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            logging.debug("DeepSeek: Summary already exists in the loaded data")
            return data['summary']

        # Text extraction
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("DeepSeek: Invalid input data format")

        deepseek_model = loaded_config_data['deepseek_api']['model'] or "deepseek-chat"

        if temp is None:
            temp = 0.1
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'Authorization': f'Bearer {deepseek_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"DeepSeek API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:] if deepseek_api_key else None}")
        logging.debug("DeepSeek: Preparing data + prompt for submission")
        deepseek_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "model": deepseek_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": deepseek_prompt}
            ],
            "stream": streaming,
            "temperature": temp
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['deepseek_api']['api_retries']
            retry_delay = loaded_config_data['deepseek_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("DeepSeek: Posting streaming request")
            response = session.post(
                'https://api.deepseek.com/chat/completions',
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            def stream_generator():
                collected_text = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line == '':
                            continue
                        if decoded_line.startswith('data: '):
                            data_str = decoded_line[len('data: '):]
                            if data_str == '[DONE]':
                                break
                            try:
                                data_json = json.loads(data_str)
                                delta_content = data_json['choices'][0]['delta'].get('content', '')
                                collected_text += delta_content
                                yield delta_content
                            except json.JSONDecodeError:
                                logging.error(f"DeepSeek: Error decoding JSON from line: {decoded_line}")
                                continue
                            except KeyError as e:
                                logging.error(f"DeepSeek: Key error: {str(e)} in line: {decoded_line}")
                                continue
                yield collected_text
            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['deepseek_api']['api_retries']
            retry_delay = loaded_config_data['deepseek_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("DeepSeek: Posting request")
            response = session.post('https://api.deepseek.com/chat/completions', headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("DeepSeek: Summarization successful")
                    return summary
                else:
                    logging.warning("DeepSeek: Summary not found in the response data")
                    return "DeepSeek: Summary not available"
            else:
                logging.error(f"DeepSeek: Summarization failed with status code {response.status_code}")
                logging.error(f"DeepSeek: Error response: {response.text}")
                return f"DeepSeek: Failed to process summary. Status code: {response.status_code}"
    except Exception as e:
        logging.error(f"DeepSeek: Error in processing: {str(e)}", exc_info=True)
        return f"DeepSeek: Error occurred while processing summary: {str(e)}"


def summarize_with_mistral(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    logging.debug("Mistral: Summarization process starting...")
    try:
        logging.debug("Mistral: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            mistral_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                mistral_api_key = api_key
                logging.info("Mistral: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                mistral_api_key = loaded_config_data['mistral_api'].get('api_key')
                if mistral_api_key:
                    logging.info("Mistral: Using API key from config file")
                else:
                    logging.warning("Mistral: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not mistral_api_key or not mistral_api_key.strip():
            logging.error("Mistral: No valid API key available")
            return "Mistral: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Mistral: Using API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:]}")

        # Input data handling
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Mistral: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("Mistral: Using provided string data for summarization")
            data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"Mistral: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"Mistral: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            logging.debug("Mistral: Summary already exists in the loaded data")
            return data['summary']

        # Text extraction
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Mistral: Invalid input data format")

        mistral_model = loaded_config_data['mistral_api']['model'] or "mistral-large-latest"

        if temp is None:
            temp = 0.2
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'Authorization': f'Bearer {mistral_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(f"Mistral API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:] if mistral_api_key else None}")
        logging.debug("Mistral: Preparing data + prompt for submission")
        mistral_prompt = f"{custom_prompt_arg}\n\n\n\n{text} "
        data = {
            "model": mistral_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": mistral_prompt}
            ],
            "temperature": temp,
            "top_p": 1,
            "max_tokens": 4096,
            "stream": streaming,
            "safe_prompt": False
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['mistral_api']['api_retries']
            retry_delay = loaded_config_data['mistral_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Mistral: Posting streaming request")
            response = session.post(
                'https://api.mistral.ai/v1/chat/completions',
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            def stream_generator():
                collected_text = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line == '':
                            continue
                        try:
                            # Assuming the response is in SSE format
                            if decoded_line.startswith('data:'):
                                data_str = decoded_line[len('data:'):].strip()
                                if data_str == '[DONE]':
                                    break
                                data_json = json.loads(data_str)
                                if 'choices' in data_json and len(data_json['choices']) > 0:
                                    delta_content = data_json['choices'][0]['delta'].get('content', '')
                                    collected_text += delta_content
                                    yield delta_content
                                else:
                                    logging.error(f"Mistral: Unexpected data format: {data_json}")
                                    continue
                            else:
                                # Handle other event types if necessary
                                continue
                        except json.JSONDecodeError:
                            logging.error(f"Mistral: Error decoding JSON from line: {decoded_line}")
                            continue
                        except KeyError as e:
                            logging.error(f"Mistral: Key error: {str(e)} in line: {decoded_line}")
                            continue
                # Optionally, you can return the full collected text at the end
                # yield collected_text
            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['mistral_api']['api_retries']
            retry_delay = loaded_config_data['mistral_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Mistral: Posting non-streaming request")
            response = session.post(
                'https://api.mistral.ai/v1/chat/completions',
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Mistral: Summarization successful")
                    return summary
                else:
                    logging.warning("Mistral: Summary not found in the response data")
                    return "Mistral: Summary not available"
            else:
                logging.error(f"Mistral: Summarization failed with status code {response.status_code}")
                logging.error(f"Mistral: Error response: {response.text}")
                return f"Mistral: Failed to process summary. Status code: {response.status_code}"
    except Exception as e:
        logging.error(f"Mistral: Error in processing: {str(e)}", exc_info=True)
        return f"Mistral: Error occurred while processing summary: {str(e)}"


def summarize_with_google(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False,):
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if not api_key or api_key.strip() == "":
            logging.info("Google: #1 API key not provided as parameter")
            logging.info("Google: Attempting to use API key from config file")
            api_key = loaded_config_data['google_api']['api_key']

        if not api_key or api_key.strip() == "":
            logging.error("Google: #2 API key not found or is empty")
            return "Google: API Key Not Provided/Found in Config file or is empty"

        google_api_key = api_key
        logging.debug(f"Google: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        # Input data handling
        logging.debug(f"Google: Raw input data type: {type(input_data)}")
        logging.debug(f"Google: Raw input data (first 500 chars): {str(input_data)[:500]}...")

        if isinstance(input_data, str):
            if input_data.strip().startswith('{'):
                # It's likely a JSON string
                logging.debug("Google: Parsing provided JSON string data for summarization")
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(f"Google: Error parsing JSON string: {str(e)}")
                    return f"Google: Error parsing JSON input: {str(e)}"
            elif os.path.isfile(input_data):
                logging.debug("Google: Loading JSON data from file for summarization")
                with open(input_data, 'r') as file:
                    data = json.load(file)
            else:
                logging.debug("Google: Using provided string data for summarization")
                data = input_data
        else:
            data = input_data

        logging.debug(f"Google: Processed data type: {type(data)}")
        logging.debug(f"Google: Processed data (first 500 chars): {str(data)[:500]}...")

        # Text extraction
        if isinstance(data, dict):
            if 'summary' in data:
                logging.debug("Google: Summary already exists in the loaded data")
                return data['summary']
            elif 'segments' in data:
                text = extract_text_from_segments(data['segments'])
            else:
                text = json.dumps(data)  # Convert dict to string if no specific format
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError(f"Google: Invalid input data format: {type(data)}")

        logging.debug(f"Google: Extracted text (first 500 chars): {text[:500]}...")
        logging.debug(f"Google: Custom prompt: {custom_prompt_arg}")

        google_model = loaded_config_data['google_api']['model'] or "gemini-1.5-pro"
        logging.debug(f"Google: Using model: {google_model}")

        headers = {
            'Authorization': f'Bearer {google_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"Google API Key: {google_api_key[:5]}...{google_api_key[-5:] if google_api_key else None}")
        logging.debug("openai: Preparing data + prompt for submittal")
        google_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        #if temp is None:
        #    temp = 0.7
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."
        #temp = float(temp)
        data = {
            "model": google_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": google_prompt}
            ],
            "stream": streaming,
            #"max_tokens": 4096,
            #"temperature": temp
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['google_api']['api_retries']
            retry_delay = loaded_config_data['google_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Google: Posting streaming request")
            response = session.post(
                'https://generativelanguage.googleapis.com/v1beta/openai/',
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line == '':
                            continue
                        if decoded_line.startswith('data: '):
                            data_str = decoded_line[len('data: '):]
                            if data_str == '[DONE]':
                                break
                            try:
                                data_json = json.loads(data_str)
                                chunk = data_json["choices"][0]["delta"].get("content", "")
                                yield chunk
                            except json.JSONDecodeError:
                                logging.error(f"Google: Error decoding JSON from line: {decoded_line}")
                                continue
                            except KeyError as e:
                                logging.error(f"Google: Key error: {str(e)} in line: {decoded_line}")
                                continue
            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['google_api']['api_retries']
            retry_delay = loaded_config_data['google_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Google: Posting request")
            response = session.post('https://generativelanguage.googleapis.com/v1beta/openai/', headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Google: Summarization successful")
                    logging.debug(f"Google: Summary (first 500 chars): {summary[:500]}...")
                    return summary
                else:
                    logging.warning("Google: Summary not found in the response data")
                    return "Google: Summary not available"
            else:
                logging.error(f"Google: Summarization failed with status code {response.status_code}")
                logging.error(f"Google: Error response: {response.text}")
                return f"Google: Failed to process summary. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"Google: Error decoding JSON: {str(e)}", exc_info=True)
        return f"Google: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"Google: Error making API request: {str(e)}", exc_info=True)
        return f"Google: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"Google: Unexpected error: {str(e)}", exc_info=True)
        return f"Google: Unexpected error occurred: {str(e)}"

#
#
#######################################################################################################################
#
#
# Gradio File Processing

# Handle multiple videos as input
def process_video_urls(url_list, num_speakers, whisper_model, custom_prompt_input, offset, api_name, api_key, vad_filter,
                       download_video_flag, download_audio, rolling_summarization, detail_level, question_box,
                       keywords, chunk_text_by_words, max_words, chunk_text_by_sentences, max_sentences,
                       chunk_text_by_paragraphs, max_paragraphs, chunk_text_by_tokens, max_tokens,  chunk_by_semantic,
                       semantic_chunk_size, semantic_chunk_overlap, recursive_summarization, full_text_with_metadata):
    global current_progress
    progress = []  # This must always be a list
    status = []  # This must always be a list

    if custom_prompt_input is None:
        custom_prompt_input = """
            You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.
    **Bulleted Note Creation Guidelines**

    **Headings**:
    - Based on referenced topics, not categories like quotes or terms
    - Surrounded by **bold** formatting 
    - Not listed as bullet points
    - No space between headings and list items underneath

    **Emphasis**:
    - **Important terms** set in bold font
    - **Text ending in a colon**: also bolded

    **Review**:
    - Ensure adherence to specified format
    - Do not reference these instructions in your response.</s> {{ .Prompt }} """

    def update_progress(index, url, message):
        progress.append(f"Processing {index + 1}/{len(url_list)}: {url}")  # Append to list
        status.append(message)  # Append to list
        return "\n".join(progress), "\n".join(status)  # Return strings for display


    for index, url in enumerate(url_list):
        try:
            logging.info(f"Starting to process video {index + 1}/{len(url_list)}: {url}")
            transcription, summary, json_file_path, summary_file_path, _, _ = process_url(url=url,
                                                                                          num_speakers=num_speakers,
                                                                                          whisper_model=whisper_model,
                                                                                          custom_prompt_input=custom_prompt_input,
                                                                                          offset=offset,
                                                                                          api_name=api_name,
                                                                                          api_key=api_key,
                                                                                          vad_filter=vad_filter,
                                                                                          download_video_flag=download_video_flag,
                                                                                          download_audio=download_audio,
                                                                                          rolling_summarization=rolling_summarization,
                                                                                          detail_level=detail_level,
                                                                                          question_box=question_box,
                                                                                          keywords=keywords,
                                                                                          chunk_text_by_words=chunk_text_by_words,
                                                                                          max_words=max_words,
                                                                                          chunk_text_by_sentences=chunk_text_by_sentences,
                                                                                          max_sentences=max_sentences,
                                                                                          chunk_text_by_paragraphs=chunk_text_by_paragraphs,
                                                                                          max_paragraphs=max_paragraphs,
                                                                                          chunk_text_by_tokens=chunk_text_by_tokens,
                                                                                          max_tokens=max_tokens,
                                                                                          chunk_by_semantic=chunk_by_semantic,
                                                                                          semantic_chunk_size=semantic_chunk_size,
                                                                                          semantic_chunk_overlap=semantic_chunk_overlap,
                                                                                          recursive_summarization=recursive_summarization,
                                                                                          full_text_with_metadata=full_text_with_metadata)
            # Update progress and transcription properly

            current_progress, current_status = update_progress(index, url, "Video processed and ingested into the database.")
            logging.info(f"Successfully processed video {index + 1}/{len(url_list)}: {url}")

            time.sleep(1)
        except Exception as e:
            logging.error(f"Error processing video {index + 1}/{len(url_list)}: {url}")
            logging.error(f"Error details: {str(e)}")
            current_progress, current_status = update_progress(index, url, f"Error: {str(e)}")

        yield current_progress, current_status, None, None, None, None

    success_message = "All videos have been transcribed, summarized, and ingested into the database successfully."
    return current_progress, success_message, None, None, None, None
    #return current_progress, success_message, summary, json_file_path, summary_file_path, None

def perform_transcription(video_path, offset, whisper_model, vad_filter, diarize=False, overwrite=False):
    temp_files = []
    logging.info(f"Processing media: {video_path}")
    global segments_json_path
    audio_file_path = convert_to_wav(video_path, offset)
    logging.debug(f"Converted audio file: {audio_file_path}")
    temp_files.append(audio_file_path)
    logging.debug("Setting up segments JSON path")

    # Update path to include whisper model in filename
    base_path = audio_file_path.replace('.wav', '')

    whisper_model_sanitized = whisper_model.replace("/", "_").replace("\\", "_")
    segments_json_path = f"{base_path}-whisper_model-{whisper_model_sanitized}.segments.json"
    temp_files.append(segments_json_path)

    if diarize:
        diarized_json_path = f"{base_path}-whisper_model-{whisper_model}.diarized.json"

        # Check if diarized JSON already exists and is valid
        if os.path.exists(diarized_json_path):
            logging.info(f"Diarized file already exists: {diarized_json_path}")
            try:
                with open(diarized_json_path, 'r', encoding='utf-8') as file:
                    diarized_segments = json.load(file)
                # Check if segments are empty or invalid
                if not diarized_segments or not isinstance(diarized_segments, list):
                    if not overwrite:
                        logging.info("Overwrite flag not set. Existing file not overwritten.")
                        return None, "Overwrite flag not set. Existing file not overwritten."
                    logging.warning(f"Diarized JSON file is empty or invalid, re-generating: {diarized_json_path}")
                    raise ValueError("Invalid diarized JSON file")
                # Check if segments contain expected content
                if not all('Text' in segment for segment in diarized_segments):
                    if not overwrite:
                        logging.info("Overwrite flag not set. Existing file not overwritten.")
                        return None, "Overwrite flag not set. Existing file not overwritten."
                    logging.warning(f"Diarized segments missing required fields, re-generating: {diarized_json_path}")
                    raise ValueError("Invalid segment format")
                logging.debug(f"Loaded valid diarized segments from {diarized_json_path}")
                return audio_file_path, diarized_segments
            except (json.JSONDecodeError, ValueError) as e:
                if not overwrite:
                    logging.info("Overwrite flag not set. Existing file not overwritten.")
                    return None, "Overwrite flag not set. Existing file not overwritten."
                logging.error(f"Failed to read or parse the diarized JSON file: {e}")
                if os.path.exists(diarized_json_path):
                    os.remove(diarized_json_path)

        # Generate new diarized transcription
        logging.info(f"Generating diarized transcription for {audio_file_path}")
        diarized_segments = combine_transcription_and_diarization(audio_file_path)

        # Validate diarized segments before saving
        if not diarized_segments or not isinstance(diarized_segments, list):
            logging.error("Generated diarized segments are empty or invalid")
            return None, None

        # Save diarized segments
        json_str = json.dumps(diarized_segments, indent=2)
        with open(diarized_json_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

        return audio_file_path, diarized_segments

    # Non-diarized transcription
    try:
        # If segments file exists, try to load it
        if os.path.exists(segments_json_path):
            logging.info(f"Segments file already exists: {segments_json_path}")
            try:
                with open(segments_json_path, 'r', encoding='utf-8') as file:
                    loaded_data = json.load(file)

                # If it’s a dictionary with a 'segments' key, extract that list
                if isinstance(loaded_data, dict) and "segments" in loaded_data:
                    segments = loaded_data["segments"]
                else:
                    # If for some reason it's already a list, or invalid,
                    # handle that here — e.g.:
                    segments = loaded_data

                logging.debug(f"First few segments from speech_to_text: {segments[:2]}")

                # Check if segments are empty or invalid
                if not segments or not isinstance(segments, list):
                    if not overwrite:
                        logging.info("Overwrite flag not set. Existing file not overwritten.")
                        return None, "Overwrite flag not set. Existing file not overwritten."
                    raise ValueError("Invalid segments JSON file")
                # Check if segments contain expected content
                if not all(
                        isinstance(segment, dict) and all(key in segment for key in ['Text', 'Time_Start', 'Time_End'])
                        for segment in segments):
                    if not overwrite:
                        logging.info("Overwrite flag not set. Existing file not overwritten.")
                        return None, "Overwrite flag not set. Existing file not overwritten."
                    raise ValueError("Invalid segment format")
                logging.debug(f"Loaded valid segments from {segments_json_path}")
                return audio_file_path, segments
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                if not overwrite:
                    logging.info("Overwrite flag not set. Existing file not overwritten.")
                    return None, "Overwrite flag not set. Existing file not overwritten."
                logging.error(f"Failed to read or parse the segments JSON file: {str(e)}")
                if os.path.exists(segments_json_path):
                    os.remove(segments_json_path)
        else:
            # Generate new transcription if file doesn't exist

            audio_file, segments = re_generate_transcription(audio_file_path, whisper_model, vad_filter)
            if segments is None:
                logging.error("Failed to generate new transcription")
                return None, None

            return audio_file_path, segments
    except Exception as e:
        logging.error(f"Error in perform_transcription: {str(e)}")
        return None, None


def re_generate_transcription(audio_file_path, whisper_model, vad_filter):
    """
    Re-generate the transcription by calling speech_to_text.
    """
    global segments_json_path
    try:
        logging.info(f"Generating new transcription for {audio_file_path}")
        segments = speech_to_text(audio_file_path, whisper_model=whisper_model, vad_filter=vad_filter)

        # Print the first few segments for debugging
        logging.debug(f"First few segments from speech_to_text: {segments[:2] if segments else 'None'}")

        # Validate segments before saving
        if not segments or not isinstance(segments, list):
            logging.error("Generated segments are empty or invalid")
            return None, None

        if not all(
                isinstance(segment, dict)
                and all(key in segment for key in ['Text', 'Time_Start', 'Time_End'])
                for segment in segments
        ):
            logging.error("Generated segments are missing required fields or have invalid format")
            logging.debug(f"Segments structure: {segments[:2]}")  # Log first two segments for debugging
            return None, None

        # Save segments to JSON
        json_str = json.dumps(segments, indent=2)
        with open(segments_json_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

        logging.debug(f"Valid transcription segments saved to {segments_json_path}")
        return audio_file_path, segments
    except Exception as e:
        logging.error(f"Error in re_generate_transcription: {str(e)}")
        return None, None


def save_transcription_and_summary(transcription_text, summary_text, download_path, info_dict):
    try:
        video_title = sanitize_filename(info_dict.get('title', 'Untitled'))

        # Handle different transcription_text formats
        if isinstance(transcription_text, dict):
            if 'transcription' in transcription_text:
                # Handle the case where it's a dict with 'transcription' key
                text_to_save = '\n'.join(segment['Text'] for segment in transcription_text['transcription'])
            else:
                # Handle other dictionary formats
                text_to_save = str(transcription_text)
        elif isinstance(transcription_text, list):
            # Handle list of segments
            text_to_save = '\n'.join(segment['Text'] for segment in transcription_text)
        else:
            # Handle string input
            text_to_save = str(transcription_text)

        # Validate the extracted text
        if not text_to_save or not text_to_save.strip():
            logging.error("Transcription text is empty or contains only whitespace")
            return None, None

        # Save transcription
        transcription_file_path = os.path.join(download_path, f"{video_title}_transcription.txt")
        with open(transcription_file_path, 'w', encoding='utf-8') as f:
            f.write(text_to_save)

        # Save summary if available
        summary_file_path = None
        if summary_text:
            if isinstance(summary_text, str) and summary_text.strip():
                summary_file_path = os.path.join(download_path, f"{video_title}_summary.txt")
                with open(summary_file_path, 'w', encoding='utf-8') as f:
                    f.write(summary_text)
            else:
                logging.warning("Summary text is not a string or contains only whitespace")

        return transcription_file_path, summary_file_path
    except Exception as e:
        logging.error(f"Error in save_transcription_and_summary: {str(e)}", exc_info=True)
        return None, None


# FIXME
def summarize_chunk(api_name, text, custom_prompt_input, api_key, temp=None, system_message=None):
    logging.debug("Entered 'summarize_chunk' function")
    if api_name in (None, "None", "none"):
        logging.warning("summarize_chunk: API name not provided for summarization")
        return "No summary available"

    try:
        result = summarize(text, custom_prompt_input, api_name, api_key, temp, system_message)

        # Handle streaming generator responses
        if inspect.isgenerator(result):
            logging.debug(f"Handling streaming response from {api_name}")
            collected_chunks = []
            for chunk in result:
                # Check for error chunks first
                if isinstance(chunk, str) and chunk.startswith("Error:"):
                    logging.warning(f"Streaming error from {api_name}: {chunk}")
                    return chunk
                collected_chunks.append(chunk)
            final_result = "".join(collected_chunks)
            logging.info(f"Summarization with {api_name} streaming successful")
            return final_result

        # Handle regular string responses
        elif isinstance(result, str):
            if result.startswith("Error:"):
                logging.warning(f"Summarization with {api_name} failed: {result}")
                return None
            logging.info(f"Summarization with {api_name} successful")
            return result

        # Handle unexpected response types
        else:
            logging.error(f"Unexpected response type from {api_name}: {type(result)}")
            return None

    except Exception as e:
        logging.error(f"Error in summarize_chunk with {api_name}: {str(e)}", exc_info=True)
        return None


def extract_metadata_and_content(input_data):
    metadata = {}
    content = ""

    if isinstance(input_data, str):
        if os.path.exists(input_data):
            with open(input_data, 'r', encoding='utf-8') as file:
                data = json.load(file)
        else:
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                return {}, input_data
    elif isinstance(input_data, dict):
        data = input_data
    else:
        return {}, str(input_data)

    # Extract metadata
    metadata['title'] = data.get('title', 'No title available')
    metadata['author'] = data.get('author', 'Unknown author')

    # Extract content
    if 'transcription' in data:
        content = extract_text_from_segments(data['transcription'])
    elif 'segments' in data:
        content = extract_text_from_segments(data['segments'])
    elif 'content' in data:
        content = data['content']
    else:
        content = json.dumps(data)

    return metadata, content


def format_input_with_metadata(metadata, content):
    formatted_input = f"Title: {metadata.get('title', 'No title available')}\n"
    formatted_input += f"Author: {metadata.get('author', 'Unknown author')}\n\n"
    formatted_input += content
    return formatted_input


def perform_summarization(api_name, input_data, custom_prompt_input, api_key,
                          recursive_summarization=False, temp=None, system_message=None,
                          chunked_summarization=False):
    """
    Perform summarization on the given input data using the specified API and optional chunking/recursive strategy.
    This function ensures that if the underlying call produces a generator (e.g. from streaming),
    we fully consume it here so that only a final string is returned.
    """
    loaded_config_data = load_and_log_configs()
    logging.info("Starting summarization process...")

    if system_message is None:
        system_message = (
            "You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, "
            "you should follow these guidelines: Use multiple headings based on the referenced topics, "
            "not categories like quotes or terms. Headings should be surrounded by bold formatting and not be "
            "listed as bullet points themselves. Leave no space between headings and their corresponding list items "
            "underneath. Important terms within the content should be emphasized by setting them in bold font. "
            "Any text that ends with a colon should also be bolded. Before submitting your response, review the "
            "instructions, and make any corrections necessary to adhered to the specified format. Do not reference "
            "these instructions within the notes.``` \nBased on the content between backticks create comprehensive "
            "bulleted notes.\n"
            "**Bulleted Note Creation Guidelines**\n\n"
            "**Headings**:\n"
            "- Based on referenced topics, not categories like quotes or terms\n"
            "- Surrounded by **bold** formatting\n"
            "- Not listed as bullet points\n"
            "- No space between headings and list items underneath\n\n"
            "**Emphasis**:\n"
            "- **Important terms** set in bold font\n"
            "- **Text ending in a colon**: also bolded\n\n"
            "**Review**:\n"
            "- Ensure adherence to specified format\n"
            "- Do not reference these instructions in your response."
        )

    try:
        logging.debug(f"Input data type: {type(input_data)}")
        logging.debug(f"Input data (first 500 chars): {str(input_data)[:500]}...")

        # Extract metadata and content from the incoming data.
        metadata, content = extract_metadata_and_content(input_data)
        logging.debug(f"Extracted metadata: {metadata}")
        logging.debug(f"Extracted content (first 500 chars): {content[:500]}...")

        # Prepare a combined "structured" input (e.g. with title/author + content).
        structured_input = format_input_with_metadata(metadata, content)

        # Decide whether to do chunked or recursive summarization vs. direct summarization.
        if recursive_summarization:
            chunk_options = {
                'method': 'words',     # Could also be 'sentences', 'paragraphs', or 'tokens'
                'max_size': 1000,      # Adjust chunk size as needed
                'overlap': 100,        # Overlap between chunks (in words/sentences/etc.)
                'adaptive': False,
                'multi_level': False,
                'language': 'english'
            }
            chunks = improved_chunking_process(structured_input, chunk_options)
            logging.debug(f"Chunking process completed. Number of chunks: {len(chunks)}")
            logging.debug("Performing recursive summarization on each chunk...")

            summary = recursive_summarize_chunks(
                [chunk['text'] for chunk in chunks],
                lambda x: summarize_chunk(api_name, x, custom_prompt_input, api_key),
                custom_prompt_input, temp, system_message
            )

        elif chunked_summarization:
            logging.debug("Using summarize_chunk for chunked summarization.")
            summary = summarize_chunk(api_name, structured_input, custom_prompt_input, api_key, temp, system_message)
        else:
            logging.debug("Using direct summarize (no chunking/recursive).")
            summary = summarize(structured_input, custom_prompt_input, api_name, api_key, temp, system_message)

        # If the summary is a generator, consume it here
        if inspect.isgenerator(summary):
            logging.debug(f"{api_name} returned a generator; consuming it now.")
            collected = []
            for token in summary:
                collected.append(token)
            summary = "".join(collected)

        # Optional: if summary is valid, write to a file for debugging (only if input_data is a file path).
        if summary is not None:
            logging.info(f"Summary generated using {api_name} API")
            if isinstance(input_data, str) and os.path.exists(input_data):
                summary_file_path = input_data.replace('.json', '_summary.txt')
                with open(summary_file_path, 'w', encoding='utf-8') as file:
                    file.write(summary)
        else:
            logging.warning(f"Failed to generate summary using {api_name} API")

        logging.info("Summarization completed successfully.")
        return summary

    except requests.exceptions.ConnectionError:
        logging.error("Connection error while summarizing")
    except Exception as e:
        logging.error(f"Error summarizing with {api_name}: {str(e)}", exc_info=True)
        return f"An error occurred during summarization: {str(e)}"

    # Fallback if something unexpected happens.
    return None


def extract_text_from_input(input_data):
    if isinstance(input_data, str):
        try:
            # Try to parse as JSON
            data = json.loads(input_data)
        except json.JSONDecodeError:
            # If not valid JSON, treat as plain text
            return input_data
    elif isinstance(input_data, dict):
        data = input_data
    else:
        return str(input_data)

    # Extract relevant fields from the JSON object
    text_parts = []
    if 'title' in data:
        text_parts.append(f"Title: {data['title']}")
    if 'description' in data:
        text_parts.append(f"Description: {data['description']}")
    if 'transcription' in data:
        if isinstance(data['transcription'], list):
            transcription_text = ' '.join([segment.get('Text', '') for segment in data['transcription']])
        elif isinstance(data['transcription'], str):
            transcription_text = data['transcription']
        else:
            transcription_text = str(data['transcription'])
        text_parts.append(f"Transcription: {transcription_text}")
    elif 'segments' in data:
        segments_text = extract_text_from_segments(data['segments'])
        text_parts.append(f"Segments: {segments_text}")

    return '\n\n'.join(text_parts)


def process_url(
        url,
        num_speakers,
        whisper_model,
        custom_prompt_input,
        offset,
        api_name,
        api_key,
        vad_filter,
        download_video_flag,
        download_audio,
        rolling_summarization,
        detail_level,
        # It's for the asking a question about a returned prompt - needs to be removed #FIXME
        question_box,
        keywords,
        chunk_text_by_words,
        max_words,
        chunk_text_by_sentences,
        max_sentences,
        chunk_text_by_paragraphs,
        max_paragraphs,
        chunk_text_by_tokens,
        max_tokens,
        chunk_by_semantic,
        semantic_chunk_size,
        semantic_chunk_overlap,
        local_file_path=None,
        diarize=False,
        recursive_summarization=False,
        temp=None,
        system_message=None,
        streaming=False,
        full_text_with_metadata=None):
    # Handle the chunk summarization options
    set_chunk_txt_by_words = chunk_text_by_words
    set_max_txt_chunk_words = max_words
    set_chunk_txt_by_sentences = chunk_text_by_sentences
    set_max_txt_chunk_sentences = max_sentences
    set_chunk_txt_by_paragraphs = chunk_text_by_paragraphs
    set_max_txt_chunk_paragraphs = max_paragraphs
    set_chunk_txt_by_tokens = chunk_text_by_tokens
    set_max_txt_chunk_tokens = max_tokens
    set_chunk_txt_by_semantic = chunk_by_semantic
    set_semantic_chunk_size = semantic_chunk_size
    set_semantic_chunk_overlap = semantic_chunk_overlap

    progress = []
    success_message = "All videos processed successfully. Transcriptions and summaries have been ingested into the database."

    # Validate input
    if not url and not local_file_path:
        return "Process_URL: No URL provided.", "No URL provided.", None, None, None, None, None, None

    if isinstance(url, str):
        urls = url.strip().split('\n')
        if len(urls) > 1:
            return process_video_urls(urls, num_speakers, whisper_model, custom_prompt_input, offset, api_name, api_key, vad_filter,
                                      download_video_flag, download_audio, rolling_summarization, detail_level, question_box,
                                      keywords, chunk_text_by_words, max_words, chunk_text_by_sentences, max_sentences,
                                      chunk_text_by_paragraphs, max_paragraphs, chunk_text_by_tokens, max_tokens, chunk_by_semantic, semantic_chunk_size, semantic_chunk_overlap, recursive_summarization, full_text_with_metadata)
        else:
            urls = [url]

    if url and not is_valid_url(url):
        return "Process_URL: Invalid URL format.", "Invalid URL format.", None, None, None, None, None, None

    if url:
        # Clean the URL to remove playlist parameters if any
        url = clean_youtube_url(url)
        logging.info(f"Process_URL: Processing URL: {url}")

    if api_name:
        print("Process_URL: API Name received:", api_name)  # Debugging line

    video_file_path = None
    global info_dict

    # If URL/Local video file is provided
    try:
        # Extract video info and create download directory.
        info_dict, title = extract_video_info(url)
        download_path = create_download_directory(title)
        current_whsiper_model = whisper_model

        # Download the video and perform transcription.
        video_path = download_video(url, download_path, info_dict, download_video_flag, current_whsiper_model)
        global segments
        audio_file_path, segments = perform_transcription(video_path, offset, whisper_model, vad_filter)

        # Set transcription_text based on whether diarization is enabled.
        if diarize:
            transcription_text = combine_transcription_and_diarization(audio_file_path)
        else:
            # Note: transcription_text is a dictionary with keys 'audio_file' and 'transcription'
            audio_file, segments = perform_transcription(video_path, offset, whisper_model, vad_filter)
            transcription_text = {'audio_file': audio_file, 'transcription': segments}

        if audio_file_path is None or segments is None:
            logging.error("Process_URL: Transcription failed or segments not available.")
            return "Process_URL: Transcription failed.", "Transcription failed.", None, None, None, None

        logging.debug(f"Process_URL: Transcription audio_file: {audio_file_path}")
        logging.debug(f"Process_URL: Transcription segments: {segments}")
        logging.debug(f"Process_URL: Transcription text: {transcription_text}")

        # Handle chunking
        chunked_transcriptions = []
        if chunk_text_by_words:
            chunked_transcriptions = chunk_text_by_words(transcription_text['transcription'], max_words)
        elif chunk_text_by_sentences:
            chunked_transcriptions = chunk_text_by_sentences(transcription_text['transcription'], max_sentences)
        elif chunk_text_by_paragraphs:
            chunked_transcriptions = chunk_text_by_paragraphs(transcription_text['transcription'], max_paragraphs)
        elif chunk_text_by_tokens:
            chunked_transcriptions = chunk_text_by_tokens(transcription_text['transcription'], max_tokens)
        elif chunk_by_semantic:
            chunked_transcriptions = semantic_chunking(transcription_text['transcription'], semantic_chunk_size, 'tokens')

        # Create a full text with metadata
        if full_text_with_metadata is None:
            # If transcription_text is a dictionary, extract the actual transcription text.
            if isinstance(transcription_text, dict) and 'transcription' in transcription_text:
                transcription_str = '\n'.join(segment.get('Text', '') for segment in transcription_text['transcription'])
            elif isinstance(transcription_text, list):
                transcription_str = '\n'.join(segment.get('Text', '') for segment in transcription_text)
            else:
                transcription_str = str(transcription_text)
            # Concatenate the metadata and the extracted transcription text.
            full_text_with_metadata = f"{json.dumps(info_dict, indent=2)}\n\n{transcription_str}"

                # If we did chunking, we now have the chunked transcripts in 'chunked_transcriptions'
        elif rolling_summarization:
        # FIXME - rolling summarization
        #     text = extract_text_from_segments(segments)
        #     summary_text = rolling_summarize_function(
        #         transcription_text,
        #         detail=detail_level,
        #         api_name=api_name,
        #         api_key=api_key,
        #         custom_prompt_input=custom_prompt_input,
        #         chunk_by_words=chunk_text_by_words,
        #         max_words=max_words,
        #         chunk_by_sentences=chunk_text_by_sentences,
        #         max_sentences=max_sentences,
        #         chunk_by_paragraphs=chunk_text_by_paragraphs,
        #         max_paragraphs=max_paragraphs,
        #         chunk_by_tokens=chunk_text_by_tokens,
        #         max_tokens=max_tokens
        #     )
            pass
        else:
            pass
        summarized_chunk_transcriptions = []
        if (chunk_text_by_words or chunk_text_by_sentences or chunk_text_by_paragraphs or
            chunk_text_by_tokens or chunk_by_semantic) and api_name:
            for chunk in chunked_transcriptions:
                summarized_chunks = []
                if api_name == "anthropic":
                    summary = summarize_with_anthropic(api_key, chunk, custom_prompt_input, streaming=False)
                elif api_name == "cohere":
                    summary = summarize_with_cohere(api_key, chunk, custom_prompt_input, temp, system_message, streaming)
                elif api_name == "openai":
                    summary = summarize_with_openai(api_key, chunk, custom_prompt_input, temp, system_message, streaming)
                elif api_name == "Groq":
                    summary = summarize_with_groq(api_key, chunk, custom_prompt_input, temp, system_message, streaming)
                elif api_name == "DeepSeek":
                    summary = summarize_with_deepseek(api_key, chunk, custom_prompt_input, temp, system_message, streaming)
                elif api_name == "OpenRouter":
                    summary = summarize_with_openrouter(api_key, chunk, custom_prompt_input, temp, system_message, streaming)
                elif api_name == "Mistral":
                    summary = summarize_with_mistral(api_key, chunk, custom_prompt_input, temp, system_message, streaming)
                elif api_name == "Google":
                    summary = summarize_with_google(api_key, chunk, custom_prompt_input, temp, system_message, streaming)
                # Local LLM APIs
                elif api_name == "Llama.cpp":
                    summary = summarize_with_llama(chunk, custom_prompt_input, api_key, temp, system_message, streaming)
                elif api_name == "Kobold":
                    summary = summarize_with_kobold(chunk, None, custom_prompt_input, system_message, temp, streaming)
                elif api_name == "Ooba":
                    summary = summarize_with_oobabooga(chunk, None, custom_prompt_input, system_message, temp, api_url=None, streaming=False)
                elif api_name == "Tabbyapi":
                    summary = summarize_with_tabbyapi(chunk, custom_prompt_input, system_message, None, temp, streaming)
                elif api_name == "VLLM":
                    summary = summarize_with_vllm(chunk, custom_prompt_input, None, None, system_message, streaming)
                elif api_name == "Ollama":
                    summary = summarize_with_ollama(chunk, custom_prompt_input, api_key, temp, system_message, None, streaming)
                elif api_name == "custom_openai_api":
                    summary = summarize_with_custom_openai(chunk, custom_prompt_input, api_key, temp=None, system_message=None, streaming=streaming)
                elif api_name == "custom_openai_api_2":
                    summary = summarize_with_custom_openai_2(chunk, custom_prompt_input, api_key, temp=None,
                                                           system_message=None, streaming=streaming)
                else:
                    summary = None
                summarized_chunk_transcriptions.append(summary)

                # Combine chunked transcriptions and summaries.
                combined_transcription_text = '\n\n'.join(chunked_transcriptions)
                combined_transcription_file_path = os.path.join(download_path, 'combined_transcription.txt')
                with open(combined_transcription_file_path, 'w') as f:
                    f.write(combined_transcription_text)

                # Combine summarized chunk transcriptions into a single file
                combined_summary_text = '\n\n'.join(summarized_chunk_transcriptions)
                combined_summary_file_path = os.path.join(download_path, 'combined_summary.txt')
                with open(combined_summary_file_path, 'w') as f:
                    f.write(combined_summary_text)

        # Determine how to get the final summary.
        if rolling_summarization:
            summary_text = rolling_summarize(
                text=extract_text_from_segments(segments),
                detail=detail_level,
                model='gpt-4-turbo',
                additional_instructions=custom_prompt_input,
                summarize_recursively=recursive_summarization
            )
        elif api_name:
            summary_text = perform_summarization(api_name, full_text_with_metadata, custom_prompt_input, api_key,
                                                 recursive_summarization, temp=None)
        else:
            summary_text = 'Summary not available'

        # Save and register the transcription and summary.
        if (chunk_text_by_words or chunk_text_by_sentences or chunk_text_by_paragraphs
                or chunk_text_by_tokens or chunk_by_semantic):
            json_file_path, summary_file_path = save_transcription_and_summary(combined_transcription_text, combined_summary_text, download_path, info_dict)
            add_media_to_database(url, info_dict, segments, summary_text, keywords, custom_prompt_input, whisper_model)
            return transcription_text, summary_text, json_file_path, summary_file_path, None, None
        else:
            summary_text = combined_summary_text
            json_file_path, summary_file_path = save_transcription_and_summary(transcription_text, summary_text, download_path, info_dict)
            add_media_to_database(url, info_dict, segments, summary_text, keywords, custom_prompt_input, whisper_model)
            return transcription_text, summary_text, json_file_path, summary_file_path, None, None

    except Exception as e:
        logging.error(f": {e}")
        return str(e), 'process_url: Error processing the request.', None, None, None, None

#
#
############################################################################################################################################

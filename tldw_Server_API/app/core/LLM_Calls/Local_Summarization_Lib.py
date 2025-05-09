# Local_Summarization_Lib.py
#########################################
# Local Summarization Library
# This library is used to perform summarization with a 'local' inference engine.
#
####
#
####################
# Function List
# FIXME - UPDATE Function Arguments
# 1. summarize_with_local_llm(text, custom_prompt_arg)
# 2. summarize_with_llama(api_url, text, token, custom_prompt)
# 3. summarize_with_kobold(api_url, text, kobold_api_token, custom_prompt)
# 4. summarize_with_oobabooga(api_url, text, ooba_api_token, custom_prompt)
# 5. summarize_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg)
# 6. summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt)
# 7. save_summary_to_file(summary, file_path)
#
###############################
# Import necessary libraries
import json
import os
# Import 3rd-party Libraries
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
#
# Import Local Libraries
from tldw_Server_API.app.core.Utils.Utils import load_and_log_configs, extract_text_from_segments, logging
#
#######################################################################################################################
# Function Definitions
#

summarizer_prompt = """
                    <s>You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.
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
                        - Do not reference these instructions in your response.</s> {{ .Prompt }}
                    """


def summarize_with_local_llm(input_data, custom_prompt_arg, temp, system_message=None, streaming=False):
    try:
        logging.debug("openai: Using provided string data for summarization")
        data = input_data

        logging.debug(f"Local LLM: Loaded data: {data}")
        logging.debug(f"Local LLM: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Local LLM: Summary already exists in the loaded data")
            return data['summary']

        temp = temp or 0.7
        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Invalid input data format")

        if system_message is None:
            system_message = "You are a helpful AI assistant."

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
            "max_tokens": 4096,
            "temperature": temp,
            "stream": streaming
        }

        logging.debug("Local LLM: Posting request")
        response = requests.post(
            'http://127.0.0.1:8080/v1/chat/completions',
            headers=headers,
            json=data,
        )

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
            return f"Local LLM: Failed to process summary, status code {response.status_code}"
    except Exception as e:
        logging.error(f"Local LLM: Error in processing: {str(e)}")
        return f"Local LLM: Error occurred while processing summary: {str(e)}"


def summarize_with_llama(input_data, custom_prompt, api_key=None, temp=None, system_message=None, streaming=False):
    try:
        logging.debug("Llama.cpp: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            llama_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                llama_api_key = api_key
                logging.info("Llama.cpp: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                llama_api_key = loaded_config_data['llama_api']['api_key']
                if llama_api_key:
                    logging.info("Llama.cpp: Using API key from config file")
                else:
                    logging.warning("Llama.cpp: No API key found in config file")

        logging.info("llama.cpp: Attempting to use API URL from config file")
        api_url = loaded_config_data['llama_api']['api_ip']
        logging.debug(f"Llama: Using API URL: {api_url}")

        # Load transcript
        logging.debug("Llama.cpp: Using provided string data for summarization")
        data = input_data

        logging.debug(f"Llama.cpp Summarize: Loaded data: {data}")
        logging.debug(f"Llama.cpp Summarize: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Llama.cpp Summarize: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Llama.cpp Summarize: Invalid input data format")

        # Prepare headers
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if llama_api_key and len(llama_api_key) > 5:
            headers['Authorization'] = f'Bearer {llama_api_key}'

        # Prepare system message and prompt
        if system_message is None:
            system_message = "You are a helpful AI assistant."
        logging.debug(f"Llama Summarize: System Prompt being sent is {system_message}")

        if custom_prompt is None:
            llama_prompt = f"{summarizer_prompt}\n\n{text}"
        else:
            llama_prompt = f"{custom_prompt}\n\n{text}"

        logging.debug(f"Llama Summarize: Prompt being sent is {llama_prompt[:500]}...")

        # Temperature handling
        if temp is None:
            # Check config
            if 'temperature' in loaded_config_data['llama_api']:
                temp = loaded_config_data['llama_api']['temperature']
                temp = float(temp)
            else:
                temp = 0.7
        logging.debug(f"Llama: Using temperature: {temp}")

        # Check for max tokens
        if 'max_tokens' in loaded_config_data['llama_api']:
            max_tokens = loaded_config_data['llama_api']['max_tokens']
            max_tokens = int(max_tokens)
        else:
            max_tokens = 4096
        logging.debug(f"Llama: Using max tokens: {max_tokens}")

        # Check for streaming
        if not isinstance(streaming, bool):
            if 'streaming' in loaded_config_data['llama_api']:
                streaming = loaded_config_data['llama_api']['streaming']
                streaming = bool(streaming)
        logging.debug(f"Llama: Streaming mode: {streaming}")

        # Prepare data payload
        data = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": llama_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": streaming
        }

        # Create a session
        session = requests.Session()

        # Load config values
        retry_count = loaded_config_data['llama_api']['api_retries']
        retry_delay = loaded_config_data['llama_api']['api_retry_delay']

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
        logging.debug("Llama: Submitting request to API endpoint")
        response = session.post(api_url, headers=headers, json=data, stream=streaming)

        if response.status_code == 200:
            if streaming:
                logging.debug("Llama: Processing streaming response")

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
                                    logging.error(f"Llama: Error decoding JSON from line: {decoded_line}")
                                    continue
                return stream_generator()
            else:
                logging.debug("Llama.cpp Summarizer: Processing non-streaming response")
                response_data = response.json()
                if 'content' in response_data and len(response_data['content']) > 0:
                    logging.debug(response_data)
                    summary = response_data['content'].strip()
                    logging.debug("llama: Summarization successful")
                    print("Summarization successful.")
                    return summary
                else:
                    logging.error("Llama: No choices in response data")
                    return "Llama: No choices in response data"
        else:
            logging.error(f"Llama: API request failed with status code {response.status_code}: {response.text}")
            return f"Llama: API request failed: {response.text}"

    except Exception as e:
        logging.error(f"Llama: Error in processing: {str(e)}")
        return f"Llama: Error occurred while processing summary with Llama: {str(e)}"


# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def summarize_with_kobold(input_data, api_key, custom_prompt_input,  system_message=None, temp=None, kobold_api_ip="http://127.0.0.1:5001/api/v1/generate", streaming=False):
    logging.debug("Kobold: Summarization process starting...")
    try:
        logging.debug("Kobold: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            kobold_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                kobold_api_key = api_key
                logging.info("Kobold: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                kobold_api_key = loaded_config_data['api_keys'].get('kobold')
                if kobold_api_key:
                    logging.info("Kobold: Using API key from config file")
                else:
                    logging.warning("Kobold: No API key found in config file")
            # Get the Streaming API IP from the config
            kobold_openai_api_IP = loaded_config_data['local_api_ip']['kobold_openai']

        logging.debug(f"Kobold: Using API Key: {kobold_api_key[:5]}...{kobold_api_key[-5:]}")

        logging.debug("Kobold.cpp: Using provided string data for summarization")
        data = input_data

        logging.debug(f"Kobold.cpp: Loaded data: {data}")
        logging.debug(f"Kobold.cpp: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Kobold.cpp: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Kobold.cpp: Invalid input data format")

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if custom_prompt_input is None:
            kobold_prompt = f"{summarizer_prompt}\n\n\n\n{text}"
        else:
            kobold_prompt = f"{custom_prompt_input}\n\n\n\n{text}"

        logging.debug(f"Kobold summarization: Prompt being sent is {kobold_prompt}")

        # Construct the data payload
        data_payload = {
            "max_context_length": 8096,
            "max_length": 4096,
            "prompt": kobold_prompt,
            "temperature": 0.7,
            "stream": streaming,
            # Include other parameters if needed
            # "top_p": 0.9,
            # "top_k": 100,
            # "rep_penalty": 1.0,
        }

        logging.debug("Kobold Summarization: Submitting request to API endpoint")
        print("Kobold Summarization: Submitting request to API endpoint")
        kobold_api_ip = loaded_config_data['local_api_ip']['kobold']

        if streaming:
            logging.debug("Kobold Summarization: Streaming mode enabled")
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = loaded_config_data['kobold_api']['api_retries']
                retry_delay = loaded_config_data['kobold_api']['api_retry_delay']

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
                # Send the request with streaming enabled
                response = session.post(
                    kobold_openai_api_IP, headers=headers, json=data_payload, stream=True
                )
                logging.debug(
                    "Kobold Summarization: API Response Status Code: %d",
                    response.status_code,
                )

                if response.status_code == 200:
                    # Process the streamed response
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            logging.debug(
                                "Kobold: Received streamed data: %s", decoded_line
                            )
                            # OpenAI API streams data prefixed with 'data: '
                            if decoded_line.startswith('data: '):
                                content = decoded_line[len('data: '):].strip()
                                if content == '[DONE]':
                                    break
                                try:
                                    data_chunk = json.loads(content)
                                    if 'choices' in data_chunk and len(data_chunk['choices']) > 0:
                                        delta = data_chunk['choices'][0].get('delta', {})
                                        text = delta.get('content', '')
                                        if text:
                                            yield text
                                    else:
                                        logging.error(
                                            "Kobold: Expected data not found in streamed response."
                                        )
                                except json.JSONDecodeError as e:
                                    logging.error(
                                        "Kobold: Error decoding streamed JSON: %s", str(e)
                                    )
                            else:
                                logging.debug("Kobold: Ignoring line: %s", decoded_line)
                else:
                    logging.error(
                        f"Kobold: API request failed with status code {response.status_code}: {response.text}"
                    )
                    yield f"Kobold: API request failed: {response.text}"
            except Exception as e:
                logging.error("Kobold: Error in processing: %s", str(e))
                yield f"Kobold: Error occurred while processing summary with Kobold: {str(e)}"
        else:
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = loaded_config_data['kobold_api']['api_retries']
                retry_delay = loaded_config_data['kobold_api']['api_retry_delay']

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
                    kobold_api_ip, headers=headers, json=data_payload
                )
                logging.debug(
                    "Kobold Summarization: API Response Status Code: %d",
                    response.status_code,
                )

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        logging.debug("Kobold: API Response Data: %s", response_data)

                        if (
                            response_data
                            and 'results' in response_data
                            and len(response_data['results']) > 0
                        ):
                            summary = response_data['results'][0]['text'].strip()
                            logging.debug("Kobold: Summarization successful")
                            return summary
                        else:
                            logging.error("Expected data not found in API response.")
                            return "Expected data not found in API response."
                    except ValueError as e:
                        logging.error(
                            "Kobold: Error parsing JSON response: %s", str(e)
                        )
                        return f"Error parsing JSON response: {str(e)}"
                else:
                    logging.error(
                        f"Kobold: API request failed with status code {response.status_code}: {response.text}"
                    )
                    return f"Kobold: API request failed: {response.text}"
            except Exception as e:
                logging.error("Kobold: Error in processing: %s", str(e))
                return f"Kobold: Error occurred while processing summary with Kobold: {str(e)}"
    except Exception as e:
        logging.error("Kobold: Error in processing: %s", str(e))
        return f"Kobold: Error occurred while processing summary with Kobold: {str(e)}"


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
def summarize_with_oobabooga(input_data, api_key, custom_prompt, system_message=None, temp=None, api_url=None, streaming=False):
    logging.debug("Oobabooga: Summarization process starting...")
    try:
        logging.debug("Oobabooga: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        ooba_api_key = None

        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                ooba_api_key = api_key
                logging.info("Oobabooga: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                ooba_api_key = loaded_config_data['ooba_api']['api_key']
                if ooba_api_key:
                    logging.info("Oobabooga: Using API key from config file")
                else:
                    logging.warning("Oobabooga: No API key found in config file")

        if not api_url:
            api_url = loaded_config_data['ooba_api']['api_ip']
            logging.debug(f"Oobabooga: Using API URL from config file: {api_url}")

        if not isinstance(api_url, str) or not api_url.startswith(('http://', 'https://')):
            logging.error(f"Invalid API URL configured: {api_url}")
            return "Oobabooga: Invalid API URL configured"
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if ooba_api_key:
            headers['Authorization'] = f'Bearer {ooba_api_key}'
            logging.debug(f"Oobabooga: Using API Key: {ooba_api_key[:5]}...{ooba_api_key[-5:]}")
        else:
            logging.debug("Oobabooga: No API key provided")

        # Input data handling
        if isinstance(input_data, str):
            if input_data.strip().startswith('{'):
                try:
                    data = json.loads(input_data)
                    logging.debug("Oobabooga: Parsed JSON string input")
                except json.JSONDecodeError as e:
                    logging.error(f"Oobabooga: Error parsing JSON string: {str(e)}")
                    return f"Oobabooga: Error parsing JSON input: {str(e)}"
            else:
                data = input_data
                logging.debug("Oobabooga: Using provided string data")
        else:
            data = input_data

        logging.debug(f"Oobabooga: Processed data type: {type(data)}")

        # Check for existing summary
        if isinstance(data, dict) and 'summary' in data:
            logging.debug("Oobabooga: Summary already exists")
            return data['summary']

        # Text extraction
        if isinstance(data, dict):
            if 'segments' in data:
                text = extract_text_from_segments(data['segments'])
            else:
                text = json.dumps(data)
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Oobabooga: Invalid input data format")

        # Construct prompt
        summarizer_prompt = "Please summarize the following text:"  # Define this if not already
        if custom_prompt is None:
            custom_prompt = summarizer_prompt
        ooba_prompt = f"{text}\n\n\n\n{custom_prompt}"
        logging.debug(f"Oobabooga: Prompt being sent is {ooba_prompt[:500]}...")

        # System message handling
        if system_message is None:
            system_message = "You are a helpful AI assistant."

        # Temperature handling
        if temp is None:
            # Check config
            if 'temperature' in loaded_config_data['ooba_api']:
                temp = loaded_config_data['ooba_api']['temperature']
            else:
                temp = 0.7

        # Prepare API payload
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": ooba_prompt}
        ]
        data = {
            "mode": "chat",
            "messages": messages,
            "stream": streaming,
            "temperature": temp,
        }

        if streaming:
            logging.debug("Oobabooga: Streaming mode enabled")
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['ooba_api']['api_retries']
            retry_delay = loaded_config_data['ooba_api']['api_retry_delay']

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
            response = session.post(api_url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            try:
                def stream_generator():
                    collected_messages = ""
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith('data: '):
                                content = decoded_line[len('data: '):]
                                if content == '[DONE]':
                                    break
                                try:
                                    data_chunk = json.loads(content)
                                    if 'choices' in data_chunk and data_chunk['choices']:
                                        delta = data_chunk['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            chunk = delta['content']
                                            collected_messages += chunk
                                            yield chunk
                                except json.JSONDecodeError as e:
                                    logging.error(f"JSON decode error: {str(e)}")
                                    continue

                return stream_generator()
            except requests.RequestException as e:
                logging.error(f"Error streaming summary with Oobabooga: {e}")
                return f"Error summarizing with Oobabooga: {str(e)}"
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['ooba_api']['api_retries']
            retry_delay = loaded_config_data['ooba_api']['api_retry_delay']

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
            logging.debug("Oobabooga: Posting request")
            response = session.post(api_url, headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                logging.debug("Ooba API request successful")
                logging.debug(response_data)
                if 'choices' in response_data and response_data['choices']:
                    logging.debug("Ooba API: Summarization successful")
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug(f"Ooba API: Summary (first 500 chars): {summary[:500]}...")
                    return summary
                else:
                    error_msg = f"Ooba API request failed: {response.status_code} - {response.text}"
                    logging.error(error_msg)
                    return error_msg
            else:
                logging.error(f"Ooba API: Summarization failed with status code {response.status_code}")
                logging.error(f"Ooba API: Error response: {response.text}")
                return f"Ooba API: Failed to process summary. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"Ooba API: Error decoding JSON: {str(e)}", exc_info=True)
        return f"Ooba API: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"Ooba API: Error making API request: {str(e)}", exc_info=True)
        return f"Ooba API: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"Ooba API: Unexpected error: {str(e)}", exc_info=True)
        return f"Ooba API: Unexpected error occurred: {str(e)}"


def summarize_with_tabbyapi(
    input_data,
    custom_prompt_input,
    system_message=None,
    api_key=None,
    temp=None,
    api_IP="http://127.0.0.1:5000/v1/chat/completions",
    streaming=False
):
    logging.debug("TabbyAPI: Summarization process starting...")
    try:
        logging.debug("TabbyAPI: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            tabby_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                tabby_api_key = api_key
                logging.info("TabbyAPI: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                tabby_api_key = loaded_config_data['api_keys'].get('tabby')
                if tabby_api_key:
                    logging.info("TabbyAPI: Using API key from config file")
                else:
                    logging.warning("TabbyAPI: No API key found in config file")

        # Set API IP and model from config.txt
        tabby_api_ip = loaded_config_data['local_api_ip']['tabby']
        tabby_model = loaded_config_data['models']['tabby']
        if temp is None:
            temp = 0.7

        logging.debug(f"TabbyAPI: Using API Key: {tabby_api_key[:5]}...{tabby_api_key[-5:] if tabby_api_key else 'None'}")

        # Process input data
        logging.debug("TabbyAPI: Using provided data for summarization")
        data = input_data

        logging.debug(f"TabbyAPI: Loaded data: {data}")
        logging.debug(f"TabbyAPI: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            logging.debug("TabbyAPI: Summary already exists in the loaded data")
            return data['summary']

        # Extract text for summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Invalid input data format")

        if system_message is None:
            system_message = "You are a helpful AI assistant."

        if custom_prompt_input is None:
            custom_prompt_input = f"{summarizer_prompt}\n\n\n\n{text}"
        else:
            custom_prompt_input = f"{custom_prompt_input}\n\n\n\n{text}"

        headers = {
            'Content-Type': 'application/json'
        }
        if tabby_api_key:
            headers['Authorization'] = f'Bearer {tabby_api_key}'

        data2 = {
            'model': tabby_model,
            'messages': [
                {'role': 'system',
                 'content': system_message
                 },
                {'role': 'user',
                 'content': custom_prompt_input
                 }
            ],
            'temperature': temp,
            'max_tokens': 4096,
            "min_tokens": 0,
            #'top_p': 1.0,
            #'top_k': 0,
            #'frequency_penalty': 0,
            #'presence_penalty': 0.0,
            #"repetition_penalty": 1.0,
            "stream": streaming
        }

        if streaming:
            logging.debug("TabbyAPI: Streaming mode enabled")
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = loaded_config_data['tabby_api']['api_retries']
                retry_delay = loaded_config_data['tabby_api']['api_retry_delay']

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
                response = session.post(tabby_api_ip, headers=headers, json=data2, stream=True)
                response.raise_for_status()
                # Process the streamed response
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data: '):
                            data_line = decoded_line[len('data: '):]
                            if data_line == '[DONE]':
                                break
                            try:
                                data_json = json.loads(data_line)
                                if 'choices' in data_json and len(data_json['choices']) > 0:
                                    delta = data_json['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError as e:
                                logging.error(f"TabbyAPI: Failed to parse JSON streamed data: {str(e)}")
                        else:
                            logging.debug(f"TabbyAPI: Received non-data line: {decoded_line}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error summarizing with TabbyAPI: {e}")
                yield f"Error summarizing with TabbyAPI: {str(e)}"
            except Exception as e:
                logging.error(f"Unexpected error in summarize_with_tabbyapi: {e}")
                yield f"Unexpected error in summarization process: {str(e)}"
        else:
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = loaded_config_data['tabby_api']['api_retries']
                retry_delay = loaded_config_data['tabby_api']['api_retry_delay']

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
                response = session.post(tabby_api_ip, headers=headers, json=data2)
                response.raise_for_status()
                response_json = response.json()

                # Validate the response structure
                if all(key in response_json for key in ['id', 'choices', 'created', 'model', 'object', 'usage']):
                    logging.info("TabbyAPI: Received a valid 200 response")
                    summary = response_json['choices'][0].get('message', {}).get('content', '')
                    return summary
                else:
                    logging.error("TabbyAPI: Received a 200 response, but the structure is invalid")
                    return "Error: Received an invalid response structure from TabbyAPI."

            except requests.exceptions.RequestException as e:
                logging.error(f"Error summarizing with TabbyAPI: {e}")
                return f"Error summarizing with TabbyAPI: {str(e)}"
            except json.JSONDecodeError:
                logging.error("TabbyAPI: Received an invalid JSON response")
                return "TabbyAPI: Error: Received an invalid JSON response from TabbyAPI."
            except Exception as e:
                logging.error(f"Unexpected error in summarize_with_tabbyapi: {e}")
                return f"TabbyAPI: Unexpected error in summarization process: {str(e)}"

    except Exception as e:
        logging.error(f"TabbyAPI: Unexpected error in summarize_with_tabbyapi: {e}")
        if streaming:
            yield f"TabbyAPI: Unexpected error in summarization process: {str(e)}"
        else:
            return f"TabbyAPI: Unexpected error in summarization process: {str(e)}"


def summarize_with_vllm(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    try:
        # API key validation
        if not api_key or api_key.strip() == "":
            logging.info("vLLM Summarize: API key not provided as parameter")
            logging.info("vLLM Summarize: Attempting to use API key from config file")
            loaded_config_data = load_and_log_configs()
            api_key = loaded_config_data.get('vllm_api', {}).get('api_key', "")
            logging.debug(f"vLLM Summarize: Using API key from config file: {api_key[:5]}...{api_key[-5:]}")

        if not api_key or api_key.strip() == "":
            logging.error("vLLM Summarize: API key not found or is empty")
            logging.debug("vLLM Summarize: API Key Not Provided/Found in Config file or is empty")

        logging.debug(f"vLLM Summarize: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        # Input data handling
        logging.debug(f"vLLM Summarize: Raw input data type: {type(input_data)}")
        logging.debug(f"vLLM Summarize: Raw input data (first 500 chars): {str(input_data)[:500]}...")

        if isinstance(input_data, str):
            if input_data.strip().startswith('{'):
                # It's likely a JSON string
                logging.debug("vLLM Summarize: Parsing provided JSON string data for summarization")
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(f"vLLM Summarize: Error parsing JSON string: {str(e)}")
                    return f"vLLM Summarize: Error parsing JSON input: {str(e)}"
            else:
                logging.debug("vLLM Summarize: Using provided string data for summarization")
                data = input_data
        else:
            data = input_data

        logging.debug(f"vLLM Summarize: Processed data type: {type(data)}")
        logging.debug(f"vLLM Summarize: Processed data (first 500 chars): {str(data)[:500]}...")

        # Text extraction
        if isinstance(data, dict):
            if 'summary' in data:
                logging.debug("vLLM Summarize: Summary already exists in the loaded data")
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
            raise ValueError(f"vLLM Summarize: Invalid input data format: {type(data)}")

        logging.debug(f"vLLM Summarize: Extracted text (first 500 chars): {text[:500]}...")
        logging.debug(f"vLLM Summarize: Custom prompt: {custom_prompt_arg}")

        config_settings = load_and_log_configs()
        vllm_model = config_settings['vllm_api']['model']
        logging.debug(f"vLLM Summarize: Using model: {vllm_model}")

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"vLLM API Key: {api_key[:5]}...{api_key[-5:] if api_key else None}")
        logging.debug("vLLM Summarize: Preparing data + prompt for submittal")
        user_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        if temp is None:
            temp = load_and_log_configs()['vllm_api']['temperature']
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."
        temp = float(temp)

        # Set max tokens
        max_tokens = load_and_log_configs()['vllm_api']['max_tokens']
        max_tokens = int(max_tokens)
        logging.debug(f"vLLM Summarize: Using max tokens: {max_tokens}")

        # Prepare data payload
        data = {
            "model": vllm_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": streaming,
        }

        # Setup URL
        url = load_and_log_configs()['vllm_api']['api_ip']

        # Handle streaming
        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['vllm_api']['api_retries']
            retry_delay = loaded_config_data['vllm_api']['api_retry_delay']

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
                url,
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
        # Handle non-streaming
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['vllm_api']['api_retries']
            retry_delay = loaded_config_data['vllm_api']['api_retry_delay']

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
            logging.debug("vLLM Summarization: Posting request")
            response = session.post(url, headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("vLLM Summarization: Summarization successful")
                    logging.debug(f"vLLM Summarization: Summary (first 500 chars): {summary[:500]}...")
                    return summary
                else:
                    logging.warning("vLLM Summarization: Summary not found in the response data")
                    return "vLLM Summarization: Summary not available"
            else:
                logging.error(f"vLLM Summarization: Summarization failed with status code {response.status_code}")
                logging.error(f"vLLM Summarization: Error response: {response.text}")
                return f"vLLM Summarization: Failed to process summary. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"vLLM Summarization: Error decoding JSON: {str(e)}", exc_info=True)
        return f"vLLM Summarization: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"vLLM Summarization: Error making API request: {str(e)}", exc_info=True)
        return f"vLLM Summarization: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"vLLM Summarization: Unexpected error: {str(e)}", exc_info=True)
        return f"vLLM Summarization: Unexpected error occurred: {str(e)}"


def summarize_with_ollama(
    input_data,
    custom_prompt,
    api_url=None,
    api_key=None,
    temp=None,
    system_message=None,
    model=None,
    max_retries=5,
    retry_delay=20,
    streaming=False,
    top_p=None
):
    """
    Summarizes text via the Ollama API, returning:
      - a generator (if streaming=True)
      - a single string (if streaming=False)
    Follows the same style as chat_with_llama / chat_with_ollama.
    """
    # 1) Load config
    try:
        logging.debug("Ollama: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if not loaded_config_data:
            logging.error("summarize_with_ollama: Could not load config data.")
            return "Ollama: No config data found."

        ollama_config = loaded_config_data.get("ollama_api", {})
    except Exception as e:
        logging.error(f"summarize_with_ollama: Error loading config: {e}")
        return f"Ollama: Error loading config: {str(e)}"

    # 2) Determine API Key
    try:
        if not api_key or not api_key.strip():
            # Use config if parameter not given
            api_key = ollama_config.get("api_key", "")
            if not api_key:
                logging.warning("Ollama: No API key found in config or param.")
        else:
            logging.info("Ollama: Using API key provided as parameter.")

        # 3) Determine API URL
        if not api_url or not api_url.strip():
            api_url = ollama_config.get("api_url", "")
        if not api_url:
            logging.error("Ollama: API URL not found or is empty.")
            return "Ollama: API URL not found in config or parameter."

        # 4) Determine Model
        if not model:
            model = ollama_config.get("model", None)
        if not model:
            logging.error("Ollama: Model not provided or found in config.")
            return "Ollama: Model not found in config or parameter."

        # 5) Determine Temperature
        if temp is None:
            temp = ollama_config.get("temperature", 0.7)
        if isinstance(temp, str):
            temp = float(temp)
        temp = float(temp)
        logging.info(f"Ollama: Temp is {temp}")

        # 6) Determine top_p
        if top_p is None:
            top_p = ollama_config.get("top_p", 0.95)
        if isinstance(top_p, str):
            top_p = float(top_p)
        top_p = float(top_p)

        # 7) Determine streaming
        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)
        elif streaming is None:
            streaming = ollama_config.get('streaming', False)
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': expected bool, got {type(streaming).__name__}")
        logging.debug(f"Ollama: streaming = {streaming}")

        # 8) Determine system_message
        if not system_message:
            system_message = "You are a helpful AI assistant."

        # 9) Parse/prepare input_data
        logging.debug("Ollama: Attempting to parse the input data for summarization.")
        text_content = None
        # Input data is not a file path, treat as raw text or list
        if isinstance(input_data, dict) and 'summary' in input_data:
            logging.debug("Ollama: 'summary' already present in input_data dict.")
            return input_data['summary']
        elif isinstance(input_data, list):
            text_content = extract_text_from_segments(input_data)
        elif isinstance(input_data, str):
            text_content = input_data
        else:
            raise ValueError("Ollama: Invalid input_data format; must be str or list/dict with summary.")

        if not text_content:
            logging.error("Ollama: No valid text content found to summarize.")
            return "Ollama: Could not extract text for summarization."

        # 10) Build final prompt
        ollama_prompt = f"{custom_prompt}\n\n{text_content}"
        logging.debug(f"Ollama: Summarization prompt:\n{ollama_prompt}")

        # 11) Prepare request
        max_tokens = int(ollama_config.get("max_tokens", 500))
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        # Optionally set Authorization
        if api_key and len(api_key) > 3:
            headers['Authorization'] = f'Bearer {api_key}'
        else:
            logging.debug("Ollama: No valid API key to set in headers.")

        data_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": ollama_prompt}
            ],
            "temperature": temp,
            "stream": streaming,
            "top_p": top_p,
            "max_tokens": max_tokens
        }

        local_api_timeout = int(loaded_config_data['ollama_api']['api_timeout'])
        logging.info(f"Ollama: Using local API timeout: {local_api_timeout} seconds")

        # 12) Attempt request with retries

        try:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['ollama_api']['api_retries']
            retry_delay = loaded_config_data['ollama_api']['api_retry_delay']

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
            logging.debug(f"Ollama Summarize request being sent")
            response = session.post(
                api_url,
                headers=headers,
                json=data_payload,
                stream=streaming,
                timeout=local_api_timeout
            )
            response.raise_for_status()  # Raise HTTPError if not 2xx
        except requests.exceptions.Timeout:
            logging.error("Ollama: Request timed out.")
            return "Ollama: Request timed out."
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"Ollama: HTTP error occurred: {http_err}")
            return f"Ollama: HTTP error: {http_err}"
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Ollama: Request exception: {req_err}")
            return f"Ollama: Request exception: {req_err}"
        except Exception as e:
            logging.error(f"Ollama: Unexpected error: {str(e)}")
            return f"Ollama: Unexpected error: {str(e)}"

        # 13) Handle streaming or non-streaming
        if streaming:
            # Return a generator that yields partial text chunks
            logging.debug("Ollama: Processing streaming response.")
            def stream_generator():
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded_line = line.decode('utf-8').strip()
                    try:
                        # Some Ollama builds prefix with "data: ...".
                        # If that's the case, parse accordingly:
                        if decoded_line.startswith("data:"):
                            decoded_line = decoded_line[len("data:"):].strip()
                            if decoded_line == "[DONE]":
                                break
                        json_chunk = json.loads(decoded_line)
                        if "response" in json_chunk:
                            yield json_chunk["response"]
                        if json_chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        logging.error(f"Ollama: JSON decode error on line: {decoded_line}")
                        continue
            return stream_generator()
        else:
            # Non-streaming => parse entire JSON once and return the text
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = loaded_config_data['ollama_api']['api_retries']
                retry_delay = loaded_config_data['ollama_api']['api_retry_delay']

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
                response_data = response.json() #corrected object to get an json method(not avaliable in the session object)
            except json.JSONDecodeError:
                logging.error("Ollama: Failed to parse JSON response.")
                return "Ollama: JSON parse error from summarization API."

            logging.debug(f"Ollama: Full JSON response: {response_data}")
            # Attempt to retrieve final summary
            summary = None
            if 'response' in response_data and response_data['response']:
                summary = response_data['response'].strip()
            elif 'choices' in response_data and len(response_data['choices']) > 0:
                choice = response_data['choices'][0]
                content = choice.get('message', {}).get('content', "").strip()
                if content:
                    summary = content

            if summary:
                logging.debug("Ollama: Summarization request successful.")
                return summary
            else:
                logging.error("Ollama: 'response' or 'choices' not found in JSON.")
                return "Ollama: Summarization API response missing text."

    except Exception as e:
        logging.error(f"Ollama Summarize: Exception: {str(e)}")
        return f"Ollama: Error occurred while summarizing: {str(e)}"


def summarize_with_custom_openai(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    loaded_config_data = load_and_log_configs()
    custom_openai_api_key = api_key
    try:
        # API key validation
        if not custom_openai_api_key:
            logging.info("Custom OpenAI API: API key not provided as parameter")
            logging.info("Custom OpenAI API: Attempting to use API key from config file")
            custom_openai_api_key = loaded_config_data['custom_openai_api']['api_key']

        if not custom_openai_api_key:
            logging.error("Custom OpenAI API: API key not found or is empty")
            return "Custom OpenAI API: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Custom OpenAI API: Using API Key: {custom_openai_api_key[:5]}...{custom_openai_api_key[-5:]}")

        # Input data handling
        logging.debug(f"Custom OpenAI API: Raw input data type: {type(input_data)}")
        logging.debug(f"Custom OpenAI API: Raw input data (first 500 chars): {str(input_data)[:500]}...")

        if isinstance(input_data, str):
            if input_data.strip().startswith('{'):
                # It's likely a JSON string
                logging.debug("Custom OpenAI API: Parsing provided JSON string data for summarization")
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(f"Custom OpenAI API: Error parsing JSON string: {str(e)}")
                    data = input_data
                    pass
            else:
                logging.debug("Custom OpenAI API: Using provided string data for summarization")
                data = input_data
        else:
            data = input_data

        logging.debug(f"Custom OpenAI API: Processed data type: {type(data)}")
        logging.debug(f"Custom OpenAI API: Processed data (first 500 chars): {str(data)[:500]}...")

        # Text extraction
        if isinstance(data, dict):
            if 'summary' in data:
                logging.debug("Custom OpenAI API: Summary already exists in the loaded data")
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
            raise ValueError(f"Custom OpenAI API: Invalid input data format: {type(data)}")

        logging.debug(f"Custom OpenAI API: Extracted text (first 500 chars): {text[:500]}...")
        logging.debug(f"Custom OpenAI API: Custom prompt: {custom_prompt_arg}")

        if input_data is None:
            input_data = f"{summarizer_prompt}\n\n\n\n{text}"
        else:
            input_data = f"{input_data}\n\n\n\n{text}"

        # Model Selection
        custom_openai_model = loaded_config_data['custom_openai_api']['model']
        logging.debug(f"Custom OpenAI API: Using model: {custom_openai_model}")

        # Set max tokens
        max_tokens = loaded_config_data['custom_openai_api']['max_tokens']
        max_tokens = int(max_tokens)
        logging.debug(f"Custom OpenAI API: Using max tokens: {max_tokens}")

        # Set temperature
        if temp is None:
            temp = load_and_log_configs()['custom_openai_api']['temperature']
        temp = float(temp)

        # Set system message
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        # Set Streaming
        if streaming is None:
            streaming = load_and_log_configs()['custom_openai_api']['streaming']

        # Set API URL
        custom_openai_api_url = loaded_config_data['custom_openai_api']['api_ip']
        logging.debug(f"Custom OpenAI API: Using API URL: {custom_openai_api_url}")

        logging.debug("Custom OpenAI API: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"

        # Set headers
        headers = {
            'Authorization': f'Bearer {custom_openai_api_key}',
            'Content-Type': 'application/json'
        }

        # Payload setup
        data = {
            "model": custom_openai_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": openai_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": streaming
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['custom_openai_api']['api_retries']
            retry_delay = loaded_config_data['custom_openai_api']['api_retry_delay']

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
                custom_openai_api_url,
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
                yield collected_messages
            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['custom_openai_api']['api_retries']
            retry_delay = loaded_config_data['custom_openai_api']['api_retry_delay']

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
            logging.debug("Custom OpenAI API: Posting request")
            response = session.post(custom_openai_api_url, headers=headers, json=data)
            logging.debug(f"Custom OpenAI API full API response data: {response}")
            if response.status_code == 200:
                response_data = response.json()
                logging.debug(response_data)
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    chat_response = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Custom OpenAI API: Chat Sent successfully")
                    logging.debug(f"Custom OpenAI API: Chat response: {chat_response}")
                    return chat_response
                else:
                    logging.warning("Custom OpenAI API: Chat response not found in the response data")
                    return "Custom OpenAI API: Chat not available"
            else:
                logging.error(f"Custom OpenAI API: Chat request failed with status code {response.status_code}")
                logging.error(f"Custom OpenAI API: Error response: {response.text}")
                return f"OpenAI: Failed to process chat response. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"Custom OpenAI API: Error decoding JSON: {str(e)}", exc_info=True)
        return f"Custom OpenAI API: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"Custom OpenAI API: Error making API request: {str(e)}", exc_info=True)
        return f"Custom OpenAI API: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"Custom OpenAI API: Unexpected error: {str(e)}", exc_info=True)
        return f"Custom OpenAI API: Unexpected error occurred: {str(e)}"


def summarize_with_custom_openai_2(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    loaded_config_data = load_and_log_configs()
    custom_openai_api_key = api_key
    try:
        # API key validation
        if not custom_openai_api_key:
            logging.info("Custom OpenAI API-2: API key not provided as parameter")
            logging.info("Custom OpenAI API-2: Attempting to use API key from config file")
            custom_openai_api_key = loaded_config_data['custom_openai_api_2']['api_key']

        if not custom_openai_api_key:
            logging.error("Custom OpenAI API-2: API key not found or is empty")
            return "Custom OpenAI API-2: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Custom OpenAI API: Using API Key: {custom_openai_api_key[:5]}...{custom_openai_api_key[-5:]}")

        # Input data handling
        logging.debug(f"Custom OpenAI API-2: Raw input data type: {type(input_data)}")
        logging.debug(f"Custom OpenAI API-2: Raw input data (first 500 chars): {str(input_data)[:500]}...")

        if isinstance(input_data, str):
            if input_data.strip().startswith('{'):
                # It's likely a JSON string
                logging.debug("Custom OpenAI API-2: Parsing provided JSON string data for summarization")
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(f"Custom OpenAI API-2: Error parsing JSON string: {str(e)}")
                    data = input_data
                    pass
            else:
                logging.debug("Custom OpenAI API-2: Using provided string data for summarization")
                data = input_data
        else:
            data = input_data

        logging.debug(f"Custom OpenAI API-2: Processed data type: {type(data)}")
        logging.debug(f"Custom OpenAI API-2: Processed data (first 500 chars): {str(data)[:500]}...")

        # Text extraction
        if isinstance(data, dict):
            if 'summary' in data:
                logging.debug("Custom OpenAI API-2: Summary already exists in the loaded data")
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
            raise ValueError(f"Custom OpenAI API-2: Invalid input data format: {type(data)}")

        logging.debug(f"Custom OpenAI API-2: Extracted text (first 500 chars): {text[:500]}...")
        logging.debug(f"Custom OpenAI API-2: Custom prompt: {custom_prompt_arg}")

        if input_data is None:
            input_data = f"{summarizer_prompt}\n\n\n\n{text}"
        else:
            input_data = f"{input_data}\n\n\n\n{text}"

        # Model Selection
        custom_openai_model = loaded_config_data['custom_openai_api_2']['model']
        logging.debug(f"Custom OpenAI API-2: Using model: {custom_openai_model}")

        # Set max tokens
        max_tokens = loaded_config_data['custom_openai_api_2']['max_tokens']
        max_tokens = int(max_tokens)
        logging.debug(f"Custom OpenAI API: Using max tokens: {max_tokens}")

        # Set temperature
        if temp is None:
            temp = load_and_log_configs()['custom_openai_api_2']['temperature']
        temp = float(temp)

        # Set system message
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        # Set Streaming
        if streaming is None:
            streaming = load_and_log_configs()['custom_openai_api_2']['streaming']

        # Set API URL
        custom_openai_api_url = loaded_config_data['custom_openai_api_2']['api_ip']
        logging.debug(f"Custom OpenAI API-2: Using API URL: {custom_openai_api_url}")

        logging.debug("Custom OpenAI API-2: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"

        # Set headers
        headers = {
            'Authorization': f'Bearer {custom_openai_api_key}',
            'Content-Type': 'application/json'
        }

        # Payload setup
        data = {
            "model": custom_openai_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": openai_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temp,
            "stream": streaming
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['custom_openai_api_2']['api_retries']
            retry_delay = loaded_config_data['custom_openai_api_2']['api_retry_delay']

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
                custom_openai_api_url,
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
                            logging.error(f"Custom OpenAI API-2: Error decoding JSON from line: {line}")
                            continue
                yield collected_messages
            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['custom_openai_api_2']['api_retries']
            retry_delay = loaded_config_data['custom_openai_api_2']['api_retry_delay']

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
            response = session.post(custom_openai_api_url, headers=headers, json=data)
            logging.debug(f"Custom OpenAI API-2 full API response data: {response}")
            if response.status_code == 200:
                response_data = response.json()
                logging.debug(response_data)
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    chat_response = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Custom OpenAI API-2: Chat Sent successfully")
                    logging.debug(f"Custom OpenAI API-2: Chat response: {chat_response}")
                    return chat_response
                else:
                    logging.warning("Custom OpenAI API-2: Chat response not found in the response data")
                    return "Custom OpenAI API-2: Chat not available"
            else:
                logging.error(f"Custom OpenAI API-2: Chat request failed with status code {response.status_code}")
                logging.error(f"Custom OpenAI API-2: Error response: {response.text}")
                return f"OpenAI: Failed to process chat response. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"Custom OpenAI API-2: Error decoding JSON: {str(e)}", exc_info=True)
        return f"Custom OpenAI API-2: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"Custom OpenAI API-2: Error making API request: {str(e)}", exc_info=True)
        return f"Custom OpenAI API-2: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"Custom OpenAI API-2: Unexpected error: {str(e)}", exc_info=True)
        return f"Custom OpenAI API-2: Unexpected error occurred: {str(e)}"


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

# Local_Summarization_Lib.py
#########################################
# Local Summarization Library
# This library is used to perform summarization with a 'local' inference engine.
#
####
from typing import Any, Generator

from requests.adapters import HTTPAdapter
from urllib3 import Retry

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
from PoC_Version.App_Function_Libraries.Utils.Utils import *
#
#######################################################################################################################
# Function Definitions
#

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
            top_k = load_and_log_configs().get('local_llm', {}).get('top_k', 100)
            logging.debug(f"Local LLM: Using top_k from config: {top_k}")

        if isinstance(top_p, float):
            top_p = float(top_p)
            logging.debug(f"Local LLM: Using top_p: {top_p}")
        elif top_p is None:
            top_p = load_and_log_configs().get('local_llm', {}).get('top_p', 0.95)
            logging.debug(f"Local LLM: Using top_p from config: {top_p}")

        if isinstance(min_p, float):
            min_p = float(min_p)
            logging.debug(f"Local LLM: Using min_p: {min_p}")
        elif min_p is None:
            min_p = load_and_log_configs().get('local_llm', {}).get('min_p', 0.05)
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

        local_api_timeout = loaded_config_data['local_llm']['api_timeout']
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


def chat_with_llama(input_data, custom_prompt, temp, api_url=None, api_key=None, system_prompt=None, streaming=False, top_k=None, top_p=None, min_p=None):
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if api_key is None:
            logging.info("llama.cpp: API key not provided as parameter")
            logging.info("llama.cpp: Attempting to use API key from config file")
            api_key = loaded_config_data['llama_api']['api_key']

        if api_key is None or api_key.strip() == "":
            logging.info("llama.cpp: API key not found or is empty")

        logging.debug(f"llama.cpp: Using API Key: {api_key[:5]}...{api_key[-5:]}")


        if api_url is None:
            logging.info("llama.cpp: API URL not provided as parameter")
            logging.info("llama.cpp: Attempting to use API URL from config file")
            api_url = loaded_config_data['llama_api']['api_ip']

        if api_url is None or api_url.strip() == "":
            logging.info("llama.cpp: API URL not found or is empty")
            return "llama.cpp: API URL not found or is empty"

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('llama_api', {}).get('streaming', False)
            logging.debug("Llama.cpp: Streaming mode enabled")
        else:
            logging.debug("Llama.cpp: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        if isinstance(top_k, int):
            top_k = int(top_k)
            logging.debug(f"Llama.cpp: Using top_k: {top_k}")
        elif top_k is None:
            top_k = load_and_log_configs().get('llama_api', {}).get('top_k', 100)
            top_k = int(top_k)
            logging.debug(f"Llama.cpp: Using top_k from config: {top_k}")
        if not isinstance(streaming, int):
            raise ValueError(f"Invalid type for 'top_k': Expected an int, got {type(streaming).__name__}")

        if isinstance(top_p, float):
            top_p = float(top_p)
            logging.debug(f"Llama.cpp: Using top_p: {top_p}")
        elif top_p is None:
            top_p = load_and_log_configs().get('llama_api', {}).get('top_p', 0.95)
            top_p = float(top_p)
            logging.debug(f"Llama.cpp: Using top_p from config: {top_p}")
        if not isinstance(streaming, int):
            raise ValueError(f"Invalid type for 'top_p': Expected a float, got {type(streaming).__name__}")

        if isinstance(min_p, float):
            min_p = float(min_p)
            logging.debug(f"Llama.cpp: Using min_p: {min_p}")
        elif min_p is None:
            min_p = load_and_log_configs().get('llama_api', {}).get('min_p', 0.05)
            min_p = float(min_p)
            logging.debug(f"Llama.cpp: Using min_p from config: {min_p}")
        if not isinstance(streaming, int):
            raise ValueError(f"Invalid type for 'min_p': Expected a float, got {type(streaming).__name__}")

        local_llm_system_message = "You are a helpful AI assistant."

        if system_prompt is None:
            system_prompt = local_llm_system_message

        max_tokens_llama = int(loaded_config_data['llama_api']['max_tokens'])

        # Prepare headers
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if len(api_key) > 5:
            headers['Authorization'] = f'Bearer {api_key}'

        logging.debug(f"Llama.cpp: System prompt being used is: {system_prompt}")
        logging.debug(f"Llama.cpp: User prompt being used is: {custom_prompt}")

        llama_prompt = f"{custom_prompt} \n\n\n\n{input_data}"
        logging.debug(f"llama: Prompt being sent is {llama_prompt}")

        data = {
            "prompt": f"{llama_prompt}",
            "system_prompt": f"{system_prompt}",
            'temperature': temp,
            'top_k': top_k,
            'top_p': top_p,
            'min_p': min_p,
            'n_predict': max_tokens_llama,
            #'n_keep': '0',
            'stream': streaming,
            #'stop': '["\n"]',
            #'tfs_z': '1.0',
            #'repeat_penalty': '1.1',
            #'repeat_last_n': '64',
            #'presence_penalty': '0.0',
            #'frequency_penalty': '0.0',
            #'mirostat': '0',
            #'grammar': '0',
            #'json_schema': '0',
            #'ignore_eos': 'false',
            #'logit_bias': [],
            #'n_probs': '0',
            #'min_keep': '0',
            #'samplers': '["top_k", "tfs_z", "typical_p", "top_p", "min_p", "temperature"]',

        }

        local_api_timeout = loaded_config_data['llama_api']['api_timeout']
        local_api_timeout = int(local_api_timeout)
        logging.debug(f"llama.cpp: Submitting request to API endpoint with a timeout of {local_api_timeout} seconds")

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

        response = session.post(api_url, headers=headers, json=data, stream=streaming, timeout=local_api_timeout)
        logging.debug(f"Llama.cpp: API Response Data: {response}")
        if response.status_code == 200:
            if streaming:
                logging.debug("llama.cpp: Processing streaming response")

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
                response_data = response.json()
                logging.debug(f"API Response Data: {response_data}")

                if response.status_code == 200:
                    # if 'X' in response_data:
                    logging.debug(response_data)
                    summary = response_data['content'].strip()
                    logging.debug("Llama: Chat request successful")
                    print("Llama: Chat request successful.")
                    return summary
                else:
                    logging.error(f"Llama: API Chat request failed with status code {response.status_code}: {response.text}")
                    return f"Llama: API Chat request failed: {response.text}"

    except Exception as e:
        logging.error(f"Llama: Error in processing: {e}")
        return f"Llama: Error occurred while processing summary with llama: {str(e)}"


# System prompts not supported through API requests.
# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def chat_with_kobold(input_data, api_key, custom_prompt_input, temp=None, system_message=None, streaming=False, top_k=None, top_p=None):
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
                kobold_api_key = loaded_config_data['kobold_api'].get('api_key')
                if kobold_api_key:
                    logging.info("Kobold: Using API key from config file")
                else:
                    logging.warning("Kobold: No API key found in config file")

        logging.debug(f"Kobold: Using API Key: {kobold_api_key[:5]}...{kobold_api_key[-5:]}")

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('kobold_api', {}).get('streaming', False)
            logging.debug("Kobold.cpp: Streaming mode enabled")
        else:
            logging.debug("Kobold.cpp: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        if isinstance(top_k, int):
            top_k = int(top_k)
            logging.debug(f"Kobold.cpp: Using top_k: {top_k}")
        elif top_k is None:
            top_k = load_and_log_configs().get('kobold_api', {}).get('top_k', 100)
            top_k = int(top_k)
            logging.debug(f"Kobold.cpp: Using top_k from config: {top_k}")
        if not isinstance(streaming, int):
            raise ValueError(f"Invalid type for 'top_k': Expected an int, got {type(streaming).__name__}")

        if isinstance(top_p, float):
            top_p = float(top_p)
            logging.debug(f"Kobold.cpp: Using top_p: {top_p}")
        elif top_p is None:
            top_p = load_and_log_configs().get('kobold_api', {}).get('top_p', 0.95)
            top_p = float(top_p)
            logging.debug(f"Kobold.cpp: Using top_p from config: {top_p}")
        if not isinstance(streaming, int):
            raise ValueError(f"Invalid type for 'top_p': Expected a float, got {type(streaming).__name__}")

        if not isinstance(temp, float):
            temp = load_and_log_configs().get('kobold_api', {}).get('temperature', 0.7)
            logging.debug(f"Kobold.cpp: Using temperature from config: {temp}")
        if not isinstance(temp, float):
            raise ValueError(f"Invalid type for 'temp': Expected a float, got {type(streaming).__name__}")

        kobold_max_tokens = int(loaded_config_data['kobold_api']['max_tokens'])

        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Kobold.cpp: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
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

        kobold_prompt = f"{custom_prompt_input}\n\n{text}"
        logging.debug(f"kobold: Prompt being sent is {kobold_prompt}")

        data = {
            "prompt": kobold_prompt,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            #"rep_penalty": 1.0,
            "stream": streaming,
            "max_context_length": kobold_max_tokens,
        }

        logging.debug("kobold: Submitting request to API endpoint")
        logging.info("kobold: Submitting request to API endpoint")
        kobold_api_ip = loaded_config_data['kobold_api']['api_ip']
        local_api_timeout = loaded_config_data['local_llm']['api_timeout']

        # FIXME - Kobold uses non-standard streaming bullshit
        streaming = False
        if streaming:
            logging.debug("Kobold Summarization: Streaming mode enabled")
            try:
                # Send the request with streaming enabled
                # Get the Streaming API IP from the config
                kobold_openai_api_IP = loaded_config_data['kobold_api']['api_streaming_ip']
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
                    kobold_openai_api_IP, headers=headers, json=data, stream=True, timeout=local_api_timeout
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
                                "fKobold: Received streamed data: {decoded_line}"
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
                                        f"Kobold: Error decoding streamed JSON: {str(e)}"
                                    )
                            else:
                                logging.debug(f"Kobold: Ignoring line: {decoded_line}")
                else:
                    logging.error(
                        f"Kobold: API request failed with status code {response.status_code}: {response.text}"
                    )
                    yield f"Kobold: API request failed: {response.text}"
            except Exception as e:
                logging.error(f"Kobold: Error in processing: {str(e)}")
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
                    kobold_api_ip, headers=headers, json=data, timeout=local_api_timeout
                )
                logging.debug(
                    "Kobold Summarization: API Response Status Code: %d",
                    response.status_code,
                )

                # Debugging: Print the API response
                logging.debug(f"API Response: {response.text}")

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        logging.debug(f"Kobold: API Response Data: {response_data}")

                        # Debugging: Print the parsed response data
                        logging.debug(f"Parsed Response Data: {response_data}")

                        if (
                                response_data
                                and 'results' in response_data
                                and len(response_data['results']) > 0
                        ):
                            summary = response_data['results'][0]['text'].strip()
                            logging.debug("Kobold: Chat request successful")
                            logging.debug(f"Kobold: Returning summary: {summary}")
                            yield summary
                            return
                        else:
                            logging.error("Expected data not found in API response.")
                            return "Expected data not found in API response."
                    except ValueError as e:
                        logging.error(
                            f"Kobold: Error parsing JSON response: {str(e)}"
                        )
                        yield f"Error parsing JSON response: {str(e)}"
                        return
                else:
                    logging.error(
                        f"Kobold: API request failed with status code {response.status_code}: {response.text}"
                    )
                    yield f"Kobold: API request failed: {response.text}"
                    return
            except Exception as e:
                logging.error(f"kobold: Error in processing: {str(e)}")
                yield f"kobold: Error occurred while processing chat response with kobold: {str(e)}"
                return
    except Exception as e:
        logging.error(f"kobold: Error in processing: {str(e)}")
        return f"kobold: Error occurred while processing chat response with kobold: {str(e)}"


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
def chat_with_oobabooga(input_data, api_key, custom_prompt, system_prompt=None, api_url=None, streaming=False, temp=None, top_p=None):
    logging.debug("Oobabooga: Summarization process starting...")
    try:
        logging.debug("Oobabooga: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        ooba_api_key = None

        # Config data check
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")

        # system prompt check
        if system_prompt is None:
            ooba_system_prompt = "You are a helpful AI assistant that provides accurate and concise information."
            system_prompt = ooba_system_prompt

        # Streaming check
        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('ooba_api', {}).get('streaming', False)
            logging.debug("Oobabooga: Streaming mode enabled")
        else:
            logging.debug("Oobabooga: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # API Key
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

        # top_p handling
        if isinstance(top_p, float):
            top_p = float(top_p)
            logging.debug(f"Ooba: Using top_p: {top_p}")
        elif top_p is None:
            top_p = load_and_log_configs().get('ooba_api', {}).get('top_p', 0.95)
            logging.debug(f"Ooba: Using top_p from config: {top_p}")
        if not isinstance(top_p, float):
            raise ValueError(f"Invalid type for 'top_p': Expected a float, got {type(streaming).__name__}")

        if temp is None:
            temp = load_and_log_configs().get('ooba_api', {}).get('temperature', 0.7)
            logging.debug(f"Ooba: Using temperature from config: {temp}")
        if not isinstance(temp, float):
            raise ValueError(f"Invalid type for 'temp': Expected a float, got {type(streaming).__name__}")

        # API URL Handling
        if not api_url:
            api_url = loaded_config_data['ooba_api']['api_ip']
            logging.debug(f"Oobabooga: Using API URL from config file: {api_url}")

        if not isinstance(api_url, str) or not api_url.startswith(('http://', 'https://')):
            logging.error(f"Invalid API URL configured: {api_url}")
            return "Oobabooga: Invalid API URL configured"

        ooba_max_tokens = int(loaded_config_data['ooba_api']['max_tokens'])

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        ooba_prompt = f"{input_data}" + f"\n\n\n\n{custom_prompt}"
        logging.debug(f"ooba: Prompt being sent is {ooba_prompt}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ooba_prompt}
        ]

        # Prepare API payload
        data = {
            "mode": "chat",
            "character": "Example",
            "messages": messages,
            "stream": streaming,
            "top_p": top_p,
            "temperature": temp,
            "max_tokens": ooba_max_tokens,
        }

        local_api_timeout = loaded_config_data['local_llm']['api_timeout']
        # If the user has set streaming to True:
        if streaming:
            logging.debug("Oobabooga chat: Streaming mode enabled")
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
            response = session.post(api_url, headers=headers, json=data, stream=True, timeout=local_api_timeout)
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
            logging.debug("Oobabooga Chat: Posting request (non-streaming)")
            response = session.post(api_url, headers=headers, json=data, timeout=local_api_timeout)

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


def chat_with_tabbyapi(
    input_data,
    custom_prompt_input,
    system_message=None,
    api_key=None,
    temp=None,
    streaming=False,
    top_k=None,
    top_p=None,
    min_p=None,
):
    logging.debug("TabbyAPI: Chat request process starting...")
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
                tabby_api_key = loaded_config_data['tabby_api'].get('api_key')
                if tabby_api_key:
                    logging.info("TabbyAPI: Using API key from config file")
                else:
                    logging.warning("TabbyAPI: No API key found in config file")

        # Streaming
        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('tabby_api', {}).get('streaming', False)
            logging.debug("TabbyAPI: Streaming mode enabled")
        else:
            logging.debug("TabbyAPI: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # Set API IP and model from config.txt
        tabby_api_ip = loaded_config_data['tabby_api']['api_ip']
        tabby_model = loaded_config_data['tabby_api']['model']

        if isinstance(temp, float):
            temp = float(temp)
            logging.debug(f"TabbyAPI: Using temperature: {temp}")
        elif temp is None:
            temp = loaded_config_data.get('tabby_api', {}).get('temperature', 0.7)
            logging.debug(f"TabbyAPI: Using temperature from config: {temp}")

        if isinstance(top_k, int):
            top_k = int(top_k)
            logging.debug(f"TabbyAPI: Using top_k: {top_k}")
        elif top_k is None:
            top_k = loaded_config_data.get('tabby_api', {}).get('top_k', 100)
            logging.debug(f"TabbyAPI: Using top_k from config: {top_k}")
        if not isinstance(top_k, int):
            raise ValueError(f"Invalid type for 'top_k': Expected an int, got {type(top_k).__name__}")

        if isinstance(top_p, float):
            top_p = float(top_p)
            logging.debug(f"TabbyAPI: Using top_p: {top_p}")
        elif top_p is None:
            top_p = loaded_config_data.get('tabby_api', {}).get('top_p', 0.95)
            logging.debug(f"TabbyAPI: Using top_p from config: {top_p}")
        if not isinstance(top_p, float):
            raise ValueError(f"Invalid type for 'top_p': Expected a float, got {type(top_p).__name__}")

        if isinstance(min_p, float):
            min_p = float(min_p)
            logging.debug(f"TabbyAPI: Using min_p: {min_p}")
        elif min_p is None:
            min_p = loaded_config_data.get('tabby_api', {}).get('min_p', 0.05)
            logging.debug(f"TabbyAPI: Using min_p from config: {min_p}")
        if not isinstance(min_p, float):
            raise ValueError(f"Invalid type for 'min_p': Expected a float, got {type(min_p).__name__}")

        logging.debug(f"TabbyAPI: Using API Key: {tabby_api_key[:5]}...{tabby_api_key[-5:] if tabby_api_key else 'None'}")

        tabby_api_system_message = "You are a helpful AI assistant."

        if system_message is None:
            system_message = tabby_api_system_message

        if custom_prompt_input is None:
            custom_prompt_input = f"{input_data}"
        else:
            custom_prompt_input = f"{custom_prompt_input}\n\n{input_data}"

        tabby_max_tokens = int(loaded_config_data['tabby_api']['max_tokens'])

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
            'max_tokens': tabby_max_tokens,
            "min_tokens": 0,
            'top_p': top_p,
            'top_k': top_k,
            #'frequency_penalty': 0,
            #'presence_penalty': 0.0,
            #"repetition_penalty": 1.0,
            "stream": streaming
        }

        local_api_timeout = loaded_config_data['local_llm']['api_timeout']
        if streaming:
            logging.debug("TabbyAPI: Streaming mode enabled for chat request")
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
                response = session.post(tabby_api_ip, headers=headers, json=data2, stream=True, timeout=local_api_timeout)
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
                logging.error(f"TabbyAPI: Error making chat request with TabbyAPI: {e}")
                yield f"TabbyAPI: Error making chat request with TabbyAPI: {str(e)}"
            except Exception as e:
                logging.error(f"Unexpected error in making chat request with summarize_with_tabbyapi: {e}")
                yield f"TabbyAPI: Unexpected error in making chat request: {str(e)}"
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
                response = session.post(tabby_api_ip, headers=headers, json=data2, timeout=local_api_timeout)
                response.raise_for_status()
                response_json = response.json()

                # Validate the response structure
                if all(key in response_json for key in ['id', 'choices', 'created', 'model', 'object', 'usage']):
                    logging.info("TabbyAPI: Received a valid 200 response")
                    chat_response = response_json['choices'][0].get('message', {}).get('content', '')
                    return chat_response
                else:
                    logging.error("TabbyAPI: Received a 200 response, but the structure is invalid")
                    return "Error: Received an invalid response structure from TabbyAPI."

            except requests.exceptions.RequestException as e:
                logging.error(f"Error making chat request with TabbyAPI: {e}")
                return f"Error making chat request with TabbyAPI: {str(e)}"
            except json.JSONDecodeError:
                logging.error("TabbyAPI: Received an invalid JSON response")
                return "TabbyAPI: Error: Received an invalid JSON response from TabbyAPI."
            except Exception as e:
                logging.error(f"Unexpected error in chat_with_tabbyapi: {e}")
                return f"TabbyAPI: Unexpected error in chat request process: {str(e)}"

    except Exception as e:
        logging.error(f"TabbyAPI: Unexpected error in chat_with_tabbyapi: {e}")
        if streaming:
            yield f"TabbyAPI: Unexpected error in chat request process: {str(e)}"
        else:
            return f"TabbyAPI: Unexpected error in chat request process: {str(e)}"


def chat_with_aphrodite(api_key, input_data, custom_prompt, temp=None, system_message=None, streaming=None,
                        topp=None, minp=None, topk=None, model=None):
    loaded_config_data = load_and_log_configs()
    logging.info("Aphrodite Chat: Function entered")
    logging.debug("Aphrodite Chat: Loading and validating configurations")
    try:
        # API key validation
        if not api_key:
            logging.info("Aphrodite Chat: API key not provided as parameter")
            logging.info("Aphrodite Chat: Attempting to use API key from config file")
            aphrodite_api_key = loaded_config_data['aphrodite_api']['api_key']

        if not api_key or aphrodite_api_key or aphrodite_api_key == "":
            logging.info("Aphrodite: API key not found or is empty")

        logging.debug(f"Aphrodite: Using API Key: {aphrodite_api_key[:5]}...{aphrodite_api_key[-5:]}")

        url = loaded_config_data['aphrodite_api']['api_ip']

        if not model:
            model = loaded_config_data['aphrodite_api']['model']
        if not isinstance(model, str):
            raise ValueError(f"Aphrodite Chat: Invalid type for 'model': Expected a string, got {type(model).__name__}")

        # Temperature
        if isinstance(temp, float):
            temp = float(temp)
            logging.debug(f"Aphrodite Chat: Using temperature: {temp}")
        elif temp is None:
            temp = loaded_config_data['aphrodite_api']['temperature']
            logging.debug(f"Aphrodite Chat: Using temperature from config: {temp}")
        if not isinstance(temp, float):
            raise ValueError(f"Aphrodite Chat: Invalid type for 'temp': Expected a float, got {type(temp).__name__}")

        # Min-P
        if isinstance(minp, float):
            minp = float(minp)
            logging.debug(f"Aphrodite Chat: Using Min-P: {minp}")
        elif minp is None:
            minp = loaded_config_data['aphrodite_api']['min_p']
            logging.debug(f"Aphrodite Chat: Using Min-P from config: {minp}")
        if not isinstance(minp, float):
            raise ValueError(f"Aphrodite Chat: Invalid type for 'min_p': Expected a float, got {type(minp).__name__}")

        # Top-P
        if isinstance(topp, float):
            topp = float(topp)
            logging.debug(f"Aphrodite Chat: Using Top-P: {topp}")
        elif topp is None:
            topp = loaded_config_data['aphrodite_api']['top_p']
            logging.debug(f"Aphrodite Chat: Using Top-P from config: {topp}")
        if not isinstance(topp, float):
            raise ValueError(f"Aphrodite Chat: Invalid type for 'top_p': Expected a float, got {type(topp).__name__}")

        # Top-K
        if isinstance(topk, int):
            topk = int(topk)
            logging.debug(f"Aphrodite Chat: Using Top-K: {topk}")
        elif topk is None:
            topk = loaded_config_data['aphrodite_api']['top_k']
            logging.debug(f"Aphrodite Chat: Using Top-K from config: {topk}")
        if not isinstance(temp, float):
            raise ValueError(f"Aphrodite Chat: Invalid type for 'top_k': Expected an int, got {type(temp).__name__}")

        # Streaming
        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('aphrodite_api', {}).get('streaming', False)
            logging.debug("Aphrodite: Streaming mode enabled")
        else:
            logging.debug("Aphrodite: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Aphrodite Chat: Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # Model
        if model is None:
            aphrodite_model = loaded_config_data['aphrodite_api']['model'] or "gpt-4o"
            logging.debug(f"Aphrodite Chat: Using model: {aphrodite_model}")

        logging.debug(f"Aphrodite Chat: Custom prompt: {custom_prompt}")

        headers = {
            'Authorization': f'Bearer {aphrodite_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"Aphrodite API Key: {aphrodite_api_key[:5]}...{aphrodite_api_key[-5:] if aphrodite_api_key else None}")
        logging.debug("Aphrodite Chat: Preparing data + prompt for submittal")
        aphrodite_prompt = f"{input_data} \n\n\n\n{custom_prompt}"
        aphrodite_system_message = "You are a helpful AI assistant who does whatever the user requests."

        if system_message is None:
            system_message = aphrodite_system_message

        aphrodite_max_tokens = int(loaded_config_data['aphrodite_api']['max_tokens'])

        data = {
            "model": aphrodite_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": aphrodite_prompt}
            ],
            "max_completion_tokens": 4096,
            "temperature": temp,
            "stream": streaming,
            "top_p": topp,
            "top_k": topk,
            "min_p": minp,
            "max_tokens": aphrodite_max_tokens,
        }
        local_api_timeout = loaded_config_data['local_llm']['api_timeout']

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['aphrodite_api']['api_retries']
            retry_delay = loaded_config_data['aphrodite_api']['api_retry_delay']

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
            logging.debug("Aphrodite Chat: Posting request (streaming")
            response = session.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                stream=True,
                timeout=local_api_timeout
            )
            logging.debug(f"OpenAI: Response text: {response.text}")
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
                            logging.error(f"Aphrodite Chat: Error decoding JSON from line: {line}")
                            continue

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['aphrodite_api']['api_retries']
            retry_delay = loaded_config_data['aphrodite_api']['api_retry_delay']

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
            logging.debug("Aphrodite Chat: Posting request (non-streaming")
            response = session.post(url, headers=headers, json=data, timeout=local_api_timeout)
            logging.debug(f"Full API response data: {response}")
            if response.status_code == 200:
                response_data = response.json()
                logging.debug(response_data)
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    chat_response = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Aphrodite Chat: Chat Sent successfully")
                    logging.debug(f"Aphrodite Chat: Chat response: {chat_response}")
                    return chat_response
                else:
                    logging.warning("Aphrodite Chat: Chat response not found in the response data")
                    return "Aphrodite Chat: Chat not available"
            else:
                logging.error(f"Aphrodite Chat: Chat request failed with status code {response.status_code}")
                logging.error(f"Aphrodite Chat: Error response: {response.text}")
                return f"Aphrodite Chat: Failed to process chat response. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"Aphrodite Chat: Error decoding JSON: {str(e)}", exc_info=True)
        return f"Aphrodite Chat: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"Aphrodite Chat: Error making API request: {str(e)}", exc_info=True)
        return f"Aphrodite Chat: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"Aphrodite Chat: Unexpected error: {str(e)}", exc_info=True)
        return f"Aphrodite Chat: Unexpected error occurred: {str(e)}"


def chat_with_ollama(input_data, custom_prompt, api_url=None, api_key=None,
                     temp=None, system_message=None, model=None, max_retries=5, retry_delay=20, streaming=False,
                     top_p=None):
    # https://github.com/ollama/ollama/blob/main/docs/openai.md
    # 1. Load config
    loaded_config_data = load_and_log_configs()
    try:
        # ----------------------------------------------------------------
        # 2. Validate and retrieve API Key, API URL, Model from parameters or config
        # ----------------------------------------------------------------
        if not api_key or not api_key.strip():
            # Attempt to load from config
            api_key = loaded_config_data['ollama_api'].get('api_key', "")
            if not api_key:
                logging.warning("Ollama: No API key found in config or parameter; continuing without Authorization.")

        # Set model from parameter or config
        if model is None:
            model = loaded_config_data['ollama_api']['model']
            if model is None:
                logging.error("Ollama: Model not found in config file")
                return "Ollama: Model not found in config file"

        # Set api_url from parameter or config
        if api_url is None:
            api_url = loaded_config_data['ollama_api']['api_url']
            if api_url is None:
                logging.error("Ollama: API URL not found in config file")
                return "Ollama: API URL not found in config file"

        # ----------------------------------------------------------------
        # 3. Validate streaming, top_p, etc.
        # ----------------------------------------------------------------
        # streaming
        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert 1 => True, 0 => False
        elif streaming is None:
            streaming = loaded_config_data['ollama_api']['streaming']
            streaming = bool(streaming)
            logging.debug(f"Ollama: Streaming mode is {streaming}")
        else:
            logging.debug("Ollama: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': must be bool, got {type(streaming).__name__}")

        # top_p
        if top_p is None:
            top_p = loaded_config_data['ollama_api'].get('top_p', 0.9)
            top_p = float(top_p)
        if not isinstance(top_p, float):
            raise ValueError(f"Invalid type for 'top_p': must be float, got {type(top_p).__name__}")

        # If user provides temperature as str or None, fallback to config default
        if temp is None:
            temp = loaded_config_data['ollama_api'].get('temperature', 0.7)
        # Convert if user gave it as string
        if isinstance(temp, str):
            temp = float(temp)

        # system_message
        if not system_message:
            system_message = "You are a helpful AI assistant"

        # ----------------------------------------------------------------
        # 4. Prepare the final prompt and request data
        # ----------------------------------------------------------------
        # Combine system + user messages similarly to llama.cpp style
        ollama_prompt = f"{custom_prompt}\n\n{input_data}"
        logging.debug(f"Ollama: Final prompt to send:\n{ollama_prompt}")

        ollama_max_tokens = int(loaded_config_data['ollama_api']['max_tokens'])

        data_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": ollama_prompt}
            ],
            "temperature": temp,
            "top_p": top_p,
            "stream": streaming,
            # Possibly set a max_tokens from config as well:
            "max_tokens": int(loaded_config_data['ollama_api'].get('max_tokens', 4096))
        }

        # Prepare headers; some Ollama instances accept optional Bearer tokens
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if api_key and len(api_key) > 5:
            headers['Authorization'] = f'Bearer {api_key}'

        # Timeout from config or fallback
        local_api_timeout = int(loaded_config_data['ollama_api']['api_timeout'])
        if local_api_timeout is None:
            local_api_timeout = 900
            logging.debug(f"Ollama: Using default timeout: {local_api_timeout} seconds")
        # ----------------------------------------------------------------
        # 5. Perform the request with optional retries
        # ----------------------------------------------------------------
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
            logging.debug(f"Ollama: Sending POST to {api_url}")
            response = requests.post(
                api_url,
                headers=headers,
                json=data_payload,
                stream=streaming,
                timeout=local_api_timeout
            )
            response.raise_for_status()  # Raise on 4xx/5xx

        except requests.exceptions.Timeout:
            logging.error(f"Ollama: Request timed out.")
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Ollama: HTTP error: {req_err}")
            return f"Ollama: HTTP error: {str(req_err)}"

        # ----------------------------------------------------------------
        # 6. Handle streaming vs. non-streaming
        # ----------------------------------------------------------------
        if streaming:
            logging.debug("Ollama: Processing streaming response.")
            # We return a generator of text chunks
            def stream_generator():
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded_line = line.decode('utf-8').strip()

                    # If Ollama returns lines like "data: { ...json... }"
                    # you may need to parse them similarly:
                    if decoded_line.startswith("data:"):
                        json_str = decoded_line[len("data:"):].strip()
                        if json_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(json_str)
                            # The structure might differ in your case
                            # Some versions might have data_json["response"]
                            # or data_json["choices"][0]["message"]["content"]
                            if "response" in data_json:
                                yield data_json["response"]
                            elif "choices" in data_json and data_json["choices"]:
                                chunk = data_json["choices"][0].get("message", {}).get("content", "")
                                if chunk:
                                    yield chunk
                            # If data_json has a 'done': True, we break
                            if data_json.get("done"):
                                break
                        except json.JSONDecodeError:
                            logging.error(f"Ollama: JSON decode error in chunk: {decoded_line}")
                            continue

            return stream_generator()

        else:
            # Non-streaming: parse entire JSON once
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logging.error(f"Ollama: Failed to parse JSON: {str(e)}")
                return f"Ollama: JSON parse error: {str(e)}"

            logging.debug(f"API Response Data: {response_data}")

            final_text = None
            # Try to parse a known field
            if "response" in response_data:
                final_text = response_data["response"].strip()
            elif (
                "choices" in response_data
                and response_data["choices"]
                and "message" in response_data["choices"][0]
            ):
                final_text = response_data["choices"][0]["message"]["content"].strip()

            if final_text:
                logging.debug("Ollama: Chat request successful (non-stream).")
                return final_text
            else:
                logging.error("Ollama: Could not find text in response_data.")
                return "Ollama: API response did not contain expected text."

    except Exception as ex:
        logging.error(f"Ollama: Exception occurred: {ex}")
        return f"Ollama: Exception: {str(ex)}"



def chat_with_vllm(
        input_data: Union[str, dict, list], custom_prompt_input: str, api_key: str = None,
        vllm_api_url: str = None, model: str = None, system_prompt: str = None,
        temp: float =None, streaming: bool = False, minp: float = None, topp: float = None, topk=None) -> str | \
                                                                                                          Generator[
                                                                                                              Any, Any, None] | Any:
    logging.debug("vLLM: Chat request being made...")
    try:
        logging.debug("vLLM: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            return "vLLM: Failed to load configuration data"
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                vllm_api_key = api_key
                logging.info("vLLM: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                vllm_api_key = loaded_config_data['vllm_api'].get('api_key')
                if vllm_api_key:
                    logging.info("vLLM: Using API key from config file")
                else:
                    logging.warning("vLLM: No API key found in config file")
            if 'api_ip' in loaded_config_data['vllm_api']:
                vllm_api_url = loaded_config_data['vllm_api']['api_ip']
                logging.info(f"vLLM: Using API URL from config file: {vllm_api_url}")
            else:
                logging.error("vLLM: API URL not found in config file")

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('vllm_api', {}).get('streaming', False)
            logging.debug("vllm: Streaming mode enabled")
        else:
            logging.debug("vllm: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # Model
        model = model or loaded_config_data['vllm_api']['model']

        # Top-K
        if isinstance(topk, int):
            topk = int(topk)
            logging.debug(f"vLLM: Using Top-K: {topk}")
        elif topk is None:
            topk = loaded_config_data['vllm_api']['top_k']
            topk = int(topk)
            logging.debug(f"vLLM: Using Top-K from config: {topk}")
        if not isinstance(topk, int):
            raise ValueError(f"Invalid type for 'top_k': Expected an int, got {type(topk).__name__}")

        # Top-P
        if isinstance(topp, float):
            topp = float(topp)
            logging.debug(f"vLLM: Using Top-P: {topp}")
        elif topp is None:
            topp = loaded_config_data['vllm_api']['top_p']
            topp = float(topp)
            logging.debug(f"vLLM: Using Top-P from config: {topp}")
        if not isinstance(topp, float):
            raise ValueError(f"Invalid type for 'top_p': Expected a float, got {type(topp).__name__}")

        # Min-P
        if isinstance(minp, float):
            minp = float(minp)
            logging.debug(f"vLLM: Using Min-P: {minp}")
        elif minp is None:
            minp = loaded_config_data['vllm_api']['min_p']
            minp = float(minp)
            logging.debug(f"vLLM: Using Min-P from config: {minp}")
        if not isinstance(minp, float):
            raise ValueError(f"Invalid type for 'min_p': Expected a float, got {type(minp).__name__}")

        # Temperature
        if isinstance(temp, float):
            temp = float(temp)
            logging.debug(f"vLLM: Using temperature: {temp}")
        elif temp is None:
            temp = loaded_config_data['vllm_api']['temperature']
            temp = float(temp)
            logging.debug(f"vLLM: Using temperature from config: {temp}")
        if not isinstance(temp, float):
            raise ValueError(f"Invalid type for 'temp': Expected a float, got {type(temp).__name__}")

        logging.debug(f"vLLM: Using API Key: {vllm_api_key[:5]}...{vllm_api_key[-5:] if vllm_api_key else 'None'}")

        vllm_system_prompt = "You are a helpful AI assistant."
        if system_prompt is None:
            system_prompt = vllm_system_prompt

        # Prepare the API request
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{custom_prompt_input}\n\n{input_data}"},
            ],
            "temperature": temp,
            "stream": streaming,
            "top_p": topp,
            "top_k": topk,
            "min_p": minp
        }

        # URL validation
        if not vllm_api_url:
            vllm_api_url = loaded_config_data['vllm_api']['api_ip']
        logging.debug(f"vLLM: Sending request to {vllm_api_url}")

        local_api_timeout = loaded_config_data['local_llm']['api_timeout']

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
            logging.debug("OpenAI: Posting request (streaming")
            response = session.post(
                url=vllm_api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=local_api_timeout
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
                            logging.error(f"vLLM: Error decoding JSON from line: {line}")
                            continue

            return stream_generator()
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
            logging.debug("vLLM: Posting request (non-streaming")
            response = session.post(vllm_api_url, headers=headers, json=payload, timeout=local_api_timeout)
            logging.debug(f"Full API response data: {response}")
            if response.status_code == 200:
                response_data = response.json()
                logging.debug(response_data)
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    chat_response = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("vLLM: Chat Sent successfully")
                    logging.debug(f"vLLM: Chat response: {chat_response}")
                    return chat_response
                else:
                    logging.warning("openai: Chat response not found in the response data")
                    return "openai: Chat not available"
            else:
                logging.error(f"vLLM: Chat request failed with status code {response.status_code}")
                logging.error(f"vLLM: Error response: {response.text}")
                return f"OpenAI: Failed to process chat response. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"vLLM: Error decoding JSON: {str(e)}", exc_info=True)
        return f"vLLM: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"vLLM: Error making API request: {str(e)}", exc_info=True)
        return f"vLLM: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"vLLM: Unexpected error: {str(e)}", exc_info=True)
        return f"vLLM: Unexpected error occurred: {str(e)}"


def chat_with_custom_openai(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False, maxp=None, model=None, minp=None, topk=None):
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

        # Model Selection
        custom_openai_model = loaded_config_data['custom_openai_api']['model']
        custom_openai_model = str(custom_openai_model)
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
            streaming = bool(streaming)
        if streaming is True:
            logging.debug("Custom OpenAI API: Streaming mode enabled")
        else:
            logging.debug("Custom OpenAI API: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # Set Top_p
        if maxp is None:
            maxp = loaded_config_data['custom_openai_api']['top_p']
            maxp = float(maxp)

        # Set Min_p
        if minp is None:
            minp = loaded_config_data['custom_openai_api']['min_p']
            minp = float(minp)

        # Set model
        if model is None:
            openai_model = loaded_config_data['custom_openai_api']['model']
            logging.debug(f"OpenAI: Using model: {openai_model}")

        # Set max tokens
        custom_openai_max_tokens = loaded_config_data['custom_openai_api']['max_tokens']
        custom_openai_max_tokens = int(custom_openai_max_tokens)

        # Set API URL
        custom_openai_api_url = loaded_config_data['custom_openai_api']['api_ip']

        logging.debug("Custom OpenAI API: Preparing data + prompt for submittal")
        openai_prompt = f"{input_data} \n\n\n\n{custom_prompt_arg}"


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
            "max_tokens": custom_openai_max_tokens,
            "temperature": temp,
            "stream": streaming,
            "top_p": maxp,
            "min_p": minp,
        }

        # Set API Timeout value
        local_api_timeout = loaded_config_data['local_llm']['api_timeout']

        # Set API Retry value
        # FIXME: Implement API Retry value

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
                stream=True,
                timeout=local_api_timeout
            )
            response.raise_for_status()
            logging.debug(f"Custom OpenAI API: Response text: {response.text}")
            
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
            response = session.post(custom_openai_api_url, headers=headers, json=data, timeout=local_api_timeout)
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


def chat_with_custom_openai_2(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False, maxp=None, model=None, minp=None, topk=None):
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

        logging.debug(f"Custom OpenAI API-2: Using API Key: {custom_openai_api_key[:5]}...{custom_openai_api_key[-5:]}")

        # Input data handling
        logging.debug(f"Custom OpenAI API-2: Raw input data type: {type(input_data)}")
        logging.debug(f"Custom OpenAI API-2: Raw input data (first 500 chars): {str(input_data)[:500]}...")

        # Model Selection
        custom_openai_model = loaded_config_data['custom_openai_api_2']['model']
        custom_openai_model = str(custom_openai_model)
        logging.debug(f"Custom OpenAI API-2: Using model: {custom_openai_model}")

        # Set max tokens
        max_tokens = loaded_config_data['custom_openai_api_2']['max_tokens']
        max_tokens = int(max_tokens)
        logging.debug(f"Custom OpenAI API-2: Using max tokens: {max_tokens}")

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
            streaming = bool(streaming)
        if streaming is True:
            logging.debug("Custom OpenAI API-2: Streaming mode enabled")
        else:
            logging.debug("Custom OpenAI API-2: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # Set Top_p
        if maxp is None:
            maxp = loaded_config_data['custom_openai_api_2']['top_p']
            maxp = float(maxp)

        # Set Min_p
        if minp is None:
            minp = loaded_config_data['custom_openai_api_2']['min_p']
            minp = float(minp)

        # Set model
        if model is None:
            openai_model = loaded_config_data['custom_openai_api_2']['model']
            logging.debug(f"OpenAI: Using model: {openai_model}")

        # Set max tokens
        custom_openai_max_tokens = loaded_config_data['custom_openai_api_2']['max_tokens']
        custom_openai_max_tokens = int(custom_openai_max_tokens)

        # Set API URL
        custom_openai_api_url = loaded_config_data['custom_openai_api_2']['api_ip']

        logging.debug("Custom OpenAI API: Preparing data + prompt for submittal")
        openai_prompt = f"{input_data} \n\n\n\n{custom_prompt_arg}"

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
            retry_count = loaded_config_data['custom_openai_2_api']['api_retries']
            retry_delay = loaded_config_data['custom_openai_2_api']['api_retry_delay']

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
            retry_count = loaded_config_data['custom_openai_2_api']['api_retries']
            retry_delay = loaded_config_data['custom_openai_2_api']['api_retry_delay']

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
            response = requests.post(custom_openai_api_url, headers=headers, json=data)
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




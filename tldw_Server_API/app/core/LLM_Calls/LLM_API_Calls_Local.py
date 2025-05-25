# Local_LLM_API_Calls_Lib.py
#########################################
# Local LLM API Calls Library
# This library is used to perform 'Local' API calls to LLM endpoints.
#
####
import json
import os
from typing import Any, Generator, Union, Dict, Optional, List

import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatProviderError, ChatBadRequestError, ChatConfigurationError
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.config import load_settings


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

def _extract_text_from_message_content(content: Union[str, List[Dict[str, Any]]], provider_name: str, msg_index: int) -> str:
    """Extracts and concatenates text parts from a message's content, logging warnings for images."""
    text_parts = []
    has_image = False
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                has_image = True
    if has_image:
        logging.warning(
            f"{provider_name}: Message at index {msg_index} contained image_url parts. "
            f"This provider/function currently only processes text. Image content will be ignored."
        )
    return "\n".join(text_parts).strip()

# Most local LLMs with OpenAI-compatible endpoints (like LM Studio, Jan.ai, many Ollama setups)
# can use a generic handler.
def _chat_with_openai_compatible_local_server(
        api_base_url: str,
        model_name: Optional[str],
        input_data: List[Dict[str, Any]],  # This is messages_payload
        api_key: Optional[str] = None,
        temp: Optional[float] = None,
        system_message: Optional[str] = None, # This will be prepended to messages by this function
        streaming: Optional[bool] = False,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None, # e.g. {"type": "json_object"}
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        user_identifier: Optional[str] = None, # maps to 'user' in OpenAI spec
        provider_name: str = "Local OpenAI-Compatible Server",
        timeout: int = 120,
        api_retries: int = 1,
        api_retry_delay: int = 1
):
    logging.debug(f"{provider_name}: Chat request starting. API Base: {api_base_url}, Model: {model_name}")

    headers = {'Content-Type': 'application/json'}
    if api_key: # Some local servers might use a key
        headers['Authorization'] = f'Bearer {api_key}'

    api_messages = []
    if system_message:
        # OpenAI standard practice is to put system message as the first message
        api_messages.append({"role": "system", "content": system_message})

    # Process input_data (messages_payload from chat_api_call)
    images_present_in_payload = False
    for msg in input_data:
        api_messages.append(msg) # Pass the message object as is
        if isinstance(msg.get("content"), list):
            for part in msg.get("content", []):
                if part.get("type") == "image_url":
                    images_present_in_payload = True
                    break
    if images_present_in_payload:
        logging.info(f"{provider_name}: Multimodal content (images) detected in messages payload. "
                     f"Ensure the target model ({model_name or 'default model'}) and server support vision.")

    payload: Dict[str, Any] = {
        "messages": api_messages,
        "stream": streaming,
    }
    if model_name: payload["model"] = model_name
    if temp is not None: payload["temperature"] = temp
    if top_p is not None: payload["top_p"] = top_p
    if top_k is not None: payload["top_k"] = top_k # OpenAI spec doesn't have top_k for chat, but some servers might
    if min_p is not None: payload["min_p"] = min_p # Not standard OpenAI, but some servers might support
    if max_tokens is not None: payload["max_tokens"] = max_tokens
    if n is not None: payload["n"] = n
    if stop is not None: payload["stop"] = stop
    if presence_penalty is not None: payload["presence_penalty"] = presence_penalty
    if frequency_penalty is not None: payload["frequency_penalty"] = frequency_penalty
    if logit_bias is not None: payload["logit_bias"] = logit_bias
    if seed is not None: payload["seed"] = seed
    if response_format is not None: payload["response_format"] = response_format
    if tools is not None: payload["tools"] = tools
    if tool_choice is not None: payload["tool_choice"] = tool_choice
    if logprobs is not None: payload["logprobs"] = logprobs
    if top_logprobs is not None: # Can only be used if logprobs is true
        if logprobs:
            payload["top_logprobs"] = top_logprobs
        else:
            logging.warning(f"{provider_name}: top_logprobs provided without logprobs=True. Ignoring top_logprobs.")
    if user_identifier is not None: payload["user"] = user_identifier


    # Construct full API URL for chat completions
    chat_completions_path = "v1/chat/completions" # Standard OpenAI path
    full_api_url = api_base_url.rstrip('/') + "/" + chat_completions_path.lstrip('/')

    logging.debug(
        f"{provider_name}: Posting to {full_api_url}. Payload keys: {list(payload.keys())}")
    logging.debug(f"{provider_name} Payload details (excluding messages): {{k: v for k, v in payload.items() if k != 'messages'}}")


    try:
        session = requests.Session()
        # Configure retries
        retry_strategy = Retry(
            total=api_retries,
            backoff_factor=api_retry_delay,
            status_forcelist=[429, 500, 502, 503, 504], # Retry on these HTTP status codes
            allowed_methods=["POST"] # Important: Retry POST requests
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        if streaming:
            # Add a bit more timeout for the initial connection for streaming
            response = session.post(full_api_url, headers=headers, json=payload, stream=True, timeout=timeout + 60)
            response.raise_for_status()
            logging.debug(f"{provider_name}: Streaming response received.")

            def stream_generator():
                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.strip():
                            yield line + "\n\n" # Pass through raw SSE line
                except requests.exceptions.ChunkedEncodingError as e_chunked:
                    logging.error(f"{provider_name}: ChunkedEncodingError during stream: {e_chunked}", exc_info=False)
                    error_content = {"error": {"message": f"Stream connection error: {str(e_chunked)}", "type": "stream_error", "code": "chunked_encoding_error"}}
                    yield f"data: {json.dumps(error_content)}\n\n"
                except Exception as e_stream:
                    logging.error(f"{provider_name}: Error during stream iteration: {e_stream}", exc_info=True)
                    error_content = {"error": {"message": f"Stream iteration error: {str(e_stream)}", "type": "stream_error", "code": "iteration_error"}}
                    yield f"data: {json.dumps(error_content)}\n\n"
                finally:
                    if response:
                        response.close()
                    # It's common for OpenAI streams to end with this
                    # Yield it here to ensure the stream always terminates correctly for the client
                    yield "data: [DONE]\n\n"
            return stream_generator()
        else:
            response = session.post(full_api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            logging.debug(f"{provider_name}: Non-streaming request successful.")
            return response_data
    except requests.exceptions.HTTPError as e_http:
        # Logged by a higher level, but good to note here too
        logging.error(f"{provider_name}: HTTP Error: {getattr(e_http.response, 'status_code', 'N/A')} - {getattr(e_http.response, 'text', str(e_http))[:500]}", exc_info=False)
        raise # Re-raise to be caught by chat_api_call's handler
    except requests.RequestException as e_req:
        logging.error(f"{provider_name}: Request Exception: {e_req}", exc_info=True)
        raise ChatProviderError(provider=provider_name, message=f"Network error making request to {provider_name}: {e_req}", status_code=503) # 503 Service Unavailable
    except (ValueError, KeyError, TypeError) as e_data: # Issues with payload construction or response parsing
        logging.error(f"{provider_name}: Data processing or configuration error: {e_data}", exc_info=True)
        raise ChatBadRequestError(provider=provider_name, message=f"{provider_name} data or configuration error: {e_data}")


def chat_with_local_llm(
        input_data: List[Dict[str, Any]],
        temp: Optional[float] = None,
        system_message: Optional[str] = None,
        streaming: Optional[bool] = False,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        # Note: custom_prompt_arg is in PROVIDER_PARAM_MAP but OpenAI compatible servers expect prompts in messages.
        # It's better handled by the `chat` function by prepending to the user message if needed.
        # For now, we assume it's already part of input_data or handled by system_message.
        custom_prompt_arg: Optional[str] = None, # Mapped from 'prompt'
         # Adding other OpenAI compatible params from your map if this server type is meant to be generic OpenAI
        response_format: Optional[Dict[str, str]] = None,
        n: Optional[int] = None,
        user_identifier: Optional[str] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    loaded_config_data = load_settings()
    cfg_section = 'local_llm' # Generic section for "local-llm" type
    cfg = loaded_config_data.get(cfg_section, {})

    api_base_url = cfg.get('api_ip', 'http://127.0.0.1:8080') # Default from config
    api_key = cfg.get('api_key') # Local servers might not need a key

    current_model = model or cfg.get('model')
    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
    current_max_tokens = max_tokens if max_tokens is not None else int(cfg.get('max_tokens', 4096))
    current_seed = seed if seed is not None else cfg.get('seed')
    current_stop = stop if stop is not None else cfg.get('stop')
    current_response_format = response_format if response_format is not None else cfg.get('response_format')
    current_n = n if n is not None else cfg.get('n')
    current_user_identifier = user_identifier if user_identifier is not None else cfg.get('user_identifier')
    current_logit_bias = logit_bias if logit_bias is not None else cfg.get('logit_bias')
    current_presence_penalty = presence_penalty if presence_penalty is not None else cfg.get('presence_penalty')
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else cfg.get('frequency_penalty')
    current_logprobs = logprobs if logprobs is not None else cfg.get('logprobs')
    current_top_logprobs = top_logprobs if top_logprobs is not None else cfg.get('top_logprobs')
    current_tools = tools if tools is not None else cfg.get('tools')
    current_tool_choice = tool_choice if tool_choice is not None else cfg.get('tool_choice')


    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if isinstance(current_logprobs, str): current_logprobs = current_logprobs.lower() == "true"

    if custom_prompt_arg:
        logging.info(f"{cfg_section}: 'custom_prompt_arg' received. Ensure it's incorporated into 'input_data' or 'system_message' by the calling function if intended for the prompt, as this handler uses OpenAI message format.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=api_key,
        temp=current_temp,
        system_message=system_message,
        streaming=current_streaming,
        max_tokens=current_max_tokens,
        top_p=current_top_p,
        top_k=current_top_k,
        min_p=current_min_p,
        n=current_n,
        stop=current_stop,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        logit_bias=current_logit_bias,
        seed=current_seed,
        response_format=current_response_format,
        tools=current_tools,
        tool_choice=current_tool_choice,
        logprobs=current_logprobs,
        top_logprobs=current_top_logprobs,
        user_identifier=current_user_identifier,
        provider_name=cfg_section.capitalize(),
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )



def chat_with_llama(
        input_data: List[Dict[str, Any]],
        api_key: Optional[str] = None, # from map
        custom_prompt: Optional[str] = None,  # from map, Mapped from 'prompt'
        temp: Optional[float] = None, # from map, generic name is 'temperature'
        system_prompt: Optional[str] = None,  # from map, Mapped from 'system_message'
        streaming: Optional[bool] = False, # from map
        model: Optional[str] = None, # from map
        top_k: Optional[int] = None, # from map
        top_p: Optional[float] = None, # from map
        min_p: Optional[float] = None, # from map
        n_predict: Optional[int] = None, # from map, mapped from max_tokens
        seed: Optional[int] = None, # from map
        stop: Optional[Union[str, List[str]]] = None, # from map
        response_format: Optional[Dict[str, str]] = None, # from map
        logit_bias: Optional[Dict[str, float]] = None, # from map
        n_probs: Optional[int] = None, # from map (maps to 'n' in generic, but llama.cpp might use it for top_logprobs count if logprobs enabled)
                                       # FIXME: n_probs might need specific handling if it's not #completions
        presence_penalty: Optional[float] = None, # from map
        frequency_penalty: Optional[float] = None, # from map
        # api_url is tricky. Your notes say "positional argument".
        # If chat_api_call is the sole entry, this needs to be passed via kwargs if mapped,
        # or loaded from config if not passed. Let's assume it's primarily from config for now.
        api_url: Optional[str] = None # This is specific to this function's call from API_CALL_HANDLERS if special handling exists
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    loaded_config_data = load_settings()
    cfg = loaded_config_data.get('llama_api', {})

    current_api_base_url = api_url or cfg.get('api_ip')
    if not current_api_base_url:
        raise ChatConfigurationError(provider="llama.cpp", message="Llama.cpp API URL/IP is required but not found in config or arguments.")

    current_api_key = api_key or cfg.get('api_key')
    current_model = model or cfg.get('model')
    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7)) # llama.cpp native name is temperature
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
    current_max_tokens = n_predict if n_predict is not None else int(cfg.get('max_tokens', cfg.get('n_predict', 4096))) # use n_predict if passed
    current_seed = seed if seed is not None else cfg.get('seed')
    current_stop = stop if stop is not None else cfg.get('stop')
    current_response_format = response_format if response_format is not None else cfg.get('response_format')
    current_logit_bias = logit_bias if logit_bias is not None else cfg.get('logit_bias')
    current_presence_penalty = presence_penalty if presence_penalty is not None else cfg.get('presence_penalty')
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else cfg.get('frequency_penalty')

    # Handle n_probs: If it's meant to be OpenAI's 'n' (number of choices)
    # For llama.cpp, if it's mimicking OpenAI, 'n' is the param.
    # If n_probs is for logprobs count, it's usually top_logprobs.
    # Assuming n_probs maps to generic 'n' from your map for now.
    current_n = n_probs if n_probs is not None else cfg.get('n', cfg.get('n_probs'))


    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt:
        logging.info("Llama.cpp: 'custom_prompt' received. Ensure it's incorporated into 'input_data' or 'system_prompt' by the calling function.")

    # Assuming llama.cpp server uses an OpenAI-compatible endpoint
    return _chat_with_openai_compatible_local_server(
        api_base_url=current_api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp,
        system_message=system_prompt, # system_prompt is the mapped name for system_message
        streaming=current_streaming,
        max_tokens=current_max_tokens,
        top_p=current_top_p,
        top_k=current_top_k,
        min_p=current_min_p,
        n=current_n, # Pass n (mapped from n_probs)
        stop=current_stop,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        logit_bias=current_logit_bias,
        seed=current_seed,
        response_format=current_response_format,
        # tools, tool_choice, logprobs, top_logprobs, user_identifier could be added if llama.cpp supports them via OpenAI compat layer
        provider_name="Llama.cpp",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )



# System prompts not supported through API requests.
# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def chat_with_kobold(
        input_data: List[Dict[str, Any]],
        api_key: Optional[str] = None,
        custom_prompt_input: Optional[str] = None, # Mapped from 'prompt'
        temp: Optional[float] = None, # Mapped from 'temp'
        system_message: Optional[str] = None, # Mapped
        streaming: Optional[bool] = False, # Mapped
        model: Optional[str] = None, # Mapped
        top_k: Optional[int] = None, # Mapped
        top_p: Optional[float] = None, # Mapped
        max_length: Optional[int] = None, # Mapped from 'max_tokens'
        stop_sequence: Optional[Union[str, List[str]]] = None, # Mapped from 'stop'
        num_responses: Optional[int] = None, # Mapped from 'n'
        seed: Optional[int] = None # Mapped from 'seed'
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    logging.debug("KoboldAI (Native): Chat request starting...")
    loaded_config_data = load_settings()
    cfg = loaded_config_data.get('kobold_api', {})

    current_api_key = api_key or cfg.get('api_key')
    api_url = cfg.get('api_ip') # URL for /api/v1/generate
    # Kobold's native /api/v1/generate doesn't take 'model' in payload, it's server-fixed.
    # The 'model' param from chat_api_call is noted here if cfg needs it for other reasons.
    # cfg_model = model or cfg.get('model') # if needed for logic, not for payload

    if not api_url:
        raise ChatConfigurationError(provider="kobold", message="KoboldAI API URL (api_ip) is required but not found.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7)) # Kobold native 'temp'
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    current_max_length = max_length if max_length is not None else int(cfg.get('max_length', 200))
    current_stop_sequence = stop_sequence if stop_sequence is not None else cfg.get('stop_sequence')
    current_num_responses = num_responses if num_responses is not None else cfg.get('num_responses')
    current_seed = seed if seed is not None else cfg.get('seed')

    # Kobold native streaming for /generate is not standard SSE and can be complex.
    # Original code forced it to False. Maintaining that unless KoboldCPP has improved this significantly
    # for the native endpoint and it's easy to parse.
    # If KoboldCPP offers an OpenAI compatible streaming endpoint, that's usually preferred.
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    if current_streaming:
        logging.warning("KoboldAI (Native): Streaming with /api/v1/generate is often non-standard. "
                        "Consider using KoboldCpp's OpenAI compatible endpoint (/v1) for reliable streaming. Forcing non-streaming for native.")
        current_streaming = False

    max_context_length = int(cfg.get('max_context_length', 2048)) # Kobold uses max_context_length for context window
    timeout = int(cfg.get('api_timeout', 180))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))


    # Construct a single prompt string from messages_payload for Kobold's native API
    full_prompt_parts = []
    if system_message: # Prepend system message if provided
        full_prompt_parts.append(system_message)

    for i, msg in enumerate(input_data):
        # role = msg.get("role", "user") # Kobold native doesn't use roles in prompt string explicitly
        text_content = _extract_text_from_message_content(msg.get("content"), "KoboldAI (Native)", i)
        # Simple concatenation. For better results, specific formatting (e.g., "User: ...", "Assistant: ...")
        # might be needed depending on how the model used with Kobold was trained.
        full_prompt_parts.append(text_content)

    if custom_prompt_input: # This was mapped from 'prompt' in chat_api_call
        # The 'chat' function is expected to build the user's message, including any 'custom_prompt' from its own args.
        # If custom_prompt_input here is *another* layer, decide how to use it.
        # Assuming it might be a final instruction to append:
        logging.info("KoboldAI (Native): Appending 'custom_prompt_input' to the prompt.")
        full_prompt_parts.append(custom_prompt_input)

    final_prompt_string = "\n\n".join(filter(None, full_prompt_parts)).strip() # filter(None,...) removes empty strings

    headers = {'Content-Type': 'application/json'}
    if current_api_key: headers['X-Api-Key'] = current_api_key # Some Kobold forks might use this

    payload: Dict[str, Any] = {
        "prompt": final_prompt_string,
        "max_context_length": max_context_length, # Context window size
        "max_length": current_max_length,         # Max tokens to generate
        # Parameters from your map / common Kobold params
        "temperature": current_temp,
        "top_p": current_top_p,
        "top_k": current_top_k,
        # "stream": current_streaming, # Will be False due to above logic
    }
    # Add other params if they are not None
    if current_stop_sequence is not None: payload['stop_sequence'] = current_stop_sequence # List of strings
    if current_num_responses is not None: payload['n'] = current_num_responses # Number of responses
    if current_seed is not None: payload['seed'] = current_seed

    # Kobold specific params (can be added from cfg if needed and supported)
    if cfg.get('rep_pen') is not None: payload['rep_pen'] = float(cfg['rep_pen'])
    # Other kobold params: typical_p, tfs, top_a, etc. could be added from cfg

    logging.debug(f"KoboldAI (Native): Posting to {api_url}. Prompt (first 200 chars): '{final_prompt_string[:200]}...'")
    logging.debug(f"KoboldAI (Native) Payload details: {payload}")


    try:
        session = requests.Session()
        retry_strategy = Retry(total=api_retries, backoff_factor=api_retry_delay, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        response = session.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        response_data = response.json()

        if response_data and 'results' in response_data and len(response_data['results']) > 0:
            # Kobold /generate usually returns a list of results, each with 'text'
            # If n > 1, there might be multiple. For now, taking the first.
            generated_text = response_data['results'][0].get('text', '').strip()
            logging.debug("KoboldAI (Native): Chat request successful.")
            # To make it somewhat OpenAI-like for the dispatcher, wrap in a choices structure.
            # This assumes non-streaming. Streaming would need a generator yielding SSE-like events.
            return {"choices": [{"message": {"role": "assistant", "content": generated_text}, "finish_reason": "stop"}]} # Assuming "stop"
        else:
            logging.error(f"KoboldAI (Native): Unexpected response structure: {response_data}")
            raise ChatProviderError(provider="kobold", message=f"Unexpected response structure from KoboldAI (Native): {str(response_data)[:200]}")

    except requests.exceptions.HTTPError as e_http:
        logging.error(f"KoboldAI (Native): HTTP Error: {getattr(e_http.response, 'status_code', 'N/A')} - {getattr(e_http.response, 'text', str(e_http))[:500]}", exc_info=False)
        raise
    except requests.RequestException as e_req:
        logging.error(f"KoboldAI (Native): Request Exception: {e_req}", exc_info=True)
        raise ChatProviderError(provider="kobold", message=f"Network error calling KoboldAI (Native): {e_req}", status_code=503)
    except (ValueError, KeyError, TypeError) as e_data:
        logging.error(f"KoboldAI (Native): Data or configuration error: {e_data}", exc_info=True)
        raise ChatBadRequestError(provider="kobold", message=f"KoboldAI (Native) config/data error: {e_data}")


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
# Oobabooga with OpenAI extension
def chat_with_oobabooga(
    input_data: List[Dict[str, Any]],
    api_key: Optional[str] = None, # from map
    custom_prompt: Optional[str] = None,  # from map, Mapped from 'prompt'
    temp: Optional[float] = None, # from map, generic name 'temperature'
    system_prompt: Optional[str] = None,  # from map, Mapped from 'system_message'
    streaming: Optional[bool] = False, # from map
    model: Optional[str] = None, # from map
    top_k: Optional[int] = None, # from map
    top_p: Optional[float] = None, # from map (ooba might use 'top_p')
    min_p: Optional[float] = None, # from map
    max_tokens: Optional[int] = None, # from map
    seed: Optional[int] = None, # from map
    stop: Optional[Union[str, List[str]]] = None, # from map
    response_format: Optional[Dict[str, str]] = None, # from map
    n: Optional[int] = None, # from map
    user_identifier: Optional[str] = None, # from map
    logit_bias: Optional[Dict[str, float]] = None, # from map
    presence_penalty: Optional[float] = None, # from map
    frequency_penalty: Optional[float] = None, # from map
    api_url: Optional[str] = None # Specific, not from generic map unless handled
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    loaded_config_data = load_settings()
    cfg = loaded_config_data.get('ooba_api', {})

    current_api_base_url = api_url or cfg.get('api_ip')
    if not current_api_base_url:
        raise ChatConfigurationError(provider="ooba", message="Oobabooga API URL/IP is required.")

    # Oobabooga's OpenAI extension usually doesn't require an API key, but can be passed if set
    current_api_key = api_key or cfg.get('api_key')
    current_model = model or cfg.get('model') # Model loaded in Ooba, can be passed in payload

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7)) # ooba native 'temperature'
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = top_p if top_p is not None else cfg.get('top_p') # Ooba uses top_p
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
    current_max_tokens = max_tokens if max_tokens is not None else int(cfg.get('max_tokens', 4096))
    current_seed = seed if seed is not None else cfg.get('seed')
    current_stop = stop if stop is not None else cfg.get('stop')
    current_response_format = response_format if response_format is not None else cfg.get('response_format')
    current_n = n if n is not None else cfg.get('n')
    current_user_identifier = user_identifier if user_identifier is not None else cfg.get('user_identifier')
    current_logit_bias = logit_bias if logit_bias is not None else cfg.get('logit_bias')
    current_presence_penalty = presence_penalty if presence_penalty is not None else cfg.get('presence_penalty')
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else cfg.get('frequency_penalty')

    timeout = int(cfg.get('api_timeout', 180)) # Ooba can be slow
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt:
        logging.info("Oobabooga: 'custom_prompt' received. Ensure it's incorporated into 'input_data' or 'system_prompt'.")

    # Oobabooga with OpenAI extension uses the generic OpenAI compatible handler
    return _chat_with_openai_compatible_local_server(
        api_base_url=current_api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp,
        system_message=system_prompt, # system_prompt maps to system_message
        streaming=current_streaming,
        max_tokens=current_max_tokens,
        top_p=current_top_p,
        top_k=current_top_k,
        min_p=current_min_p,
        n=current_n,
        stop=current_stop,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        logit_bias=current_logit_bias,
        seed=current_seed,
        response_format=current_response_format,
        user_identifier=current_user_identifier,
        # tools, tool_choice, logprobs, top_logprobs might be supported by some ooba setups
        provider_name="Oobabooga (OpenAI Extension)",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# TabbyAPI (seems OpenAI compatible)
def chat_with_tabbyapi(
    input_data: List[Dict[str, Any]],
    api_key: Optional[str] = None, # from map
    custom_prompt_input: Optional[str] = None, # from map ('prompt')
    temp: Optional[float] = None, # from map (mapped to 'temperature' in generic)
    system_message: Optional[str] = None, # from map
    streaming: Optional[bool] = False, # from map
    model: Optional[str] = None, # from map
    top_k: Optional[int] = None, # from map
    top_p: Optional[float] = None, # from map
    min_p: Optional[float] = None, # from map
    max_tokens: Optional[int] = None, # from map
    seed: Optional[int] = None, # from map
    stop: Optional[Union[str, List[str]]] = None # from map
    # TabbyAPI PROVIDER_PARAM_MAP is missing:
    # response_format, n, user_identifier, logit_bias, presence_penalty, frequency_penalty,
    # logprobs, top_logprobs, tools, tool_choice.
    # Add them to signature if TabbyAPI (OpenAI compatible) supports them.
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    loaded_config_data = load_settings()
    cfg = loaded_config_data.get('tabby_api', {})

    api_base_url = cfg.get('api_ip')
    if not api_base_url:
        raise ChatConfigurationError(provider="tabbyapi", message="TabbyAPI URL (api_ip) is required.")

    current_api_key = api_key or cfg.get('api_key')
    current_model = model or cfg.get('model')
    # TabbyAPI's map uses 'temp' for 'temperature', which matches generic.
    # So, current_temp = temperature (from chat_api_call's temp arg)
    # Your map has: 'temp': 'temperature'. This seems like an error in the map, it should be:
    # 'temperature': 'temperature' (if tabby expects 'temperature') or
    # 'temp': 'temp' (if tabby expects 'temp' and generic is 'temp').
    # Assuming generic is 'temp' and TabbyAPI also takes 'temp' or 'temperature'.
    # If PROVIDER_PARAM_MAP has 'temp':'temperature', then chat_api_call passes `temp` as `temperature` kwarg.
    # Let's assume tabby expects 'temperature' based on common OpenAI compat.
    # The `temp` param in this function signature comes from `PROVIDER_PARAM_MAP` key `temperature` and its value `temp`
    # This is confusing. Let's assume `temp` is the actual parameter name for Tabby from your map.
    current_temp_val = temp if temp is not None else float(cfg.get('temperature', cfg.get('temp', 0.7)))


    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
    current_max_tokens = max_tokens if max_tokens is not None else int(cfg.get('max_tokens', 4096))
    current_seed = seed if seed is not None else cfg.get('seed')
    current_stop = stop if stop is not None else cfg.get('stop')

    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt_input:
        logging.info("TabbyAPI: 'custom_prompt_input' received. Ensure incorporated if needed.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp_val, # Use the mapped 'temp' value
        system_message=system_message,
        streaming=current_streaming,
        max_tokens=current_max_tokens,
        top_p=current_top_p,
        top_k=current_top_k,
        min_p=current_min_p,
        seed=current_seed,
        stop=current_stop,
        provider_name="TabbyAPI",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
        # Add other OpenAI params here if TabbyAPI supports them
    )


# vLLM (OpenAI compatible)
def chat_with_vllm(
    input_data: List[Dict[str, Any]],
    api_key: Optional[str] = None, # from map
    custom_prompt_input: Optional[str] = None, # from map ('prompt')
    # vLLM's map has 'temp':'temperature', 'system_prompt':'system_message' etc.
    # These are the provider-specific names this function receives.
    temperature: Optional[float] = None, # from map (mapped from generic 'temp')
    system_prompt: Optional[str] = None,   # from map (mapped from generic 'system_message')
    streaming: Optional[bool] = False,   # from map
    model: Optional[str] = None,         # from map
    top_k: Optional[int] = None,         # from map
    top_p: Optional[float] = None,         # from map (mapped from generic 'topp')
    min_p: Optional[float] = None,         # from map (mapped from generic 'minp')
    max_tokens: Optional[int] = None,      # from map
    seed: Optional[int] = None,          # from map
    stop: Optional[Union[str, List[str]]] = None, # from map
    response_format: Optional[Dict[str, str]] = None, # from map
    n: Optional[int] = None,             # from map
    logit_bias: Optional[Dict[str, float]] = None, # from map
    presence_penalty: Optional[float] = None, # from map
    frequency_penalty: Optional[float] = None, # from map
    logprobs: Optional[bool] = None,     # from map
    user_identifier: Optional[str] = None, # from map
    vllm_api_url: Optional[str] = None # Specific config, not from generic map typically
                                       # Could be loaded from cfg or passed if chat_api_call handles it
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    loaded_config_data = load_settings()
    cfg = loaded_config_data.get('vllm_api', {})

    # vllm_api_url is a specific argument for this function if it's set up in API_CALL_HANDLERS call
    # otherwise, it falls back to config.
    current_api_base_url = vllm_api_url or cfg.get('api_ip')
    if not current_api_base_url:
        raise ChatConfigurationError(provider="vllm", message="vLLM API URL (api_ip / vllm_api_url) is required.")

    current_api_key = api_key or cfg.get('api_key') # vLLM might not require a key
    current_model = model or cfg.get('model')

    current_temp = temperature if temperature is not None else float(cfg.get('temperature', 0.7)) # func arg 'temperature' is vLLM's name
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = top_p if top_p is not None else cfg.get('top_p') # func arg 'top_p' is vLLM's name
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
    current_max_tokens = max_tokens if max_tokens is not None else int(cfg.get('max_tokens', 4096))
    current_seed = seed if seed is not None else cfg.get('seed')
    current_stop = stop if stop is not None else cfg.get('stop')
    current_response_format = response_format if response_format is not None else cfg.get('response_format')
    current_n = n if n is not None else cfg.get('n')
    current_logit_bias = logit_bias if logit_bias is not None else cfg.get('logit_bias')
    current_presence_penalty = presence_penalty if presence_penalty is not None else cfg.get('presence_penalty')
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else cfg.get('frequency_penalty')
    current_logprobs = logprobs if logprobs is not None else cfg.get('logprobs')
    # top_logprobs: vLLM PROVIDER_PARAM_MAP has 'logprobs' but not 'top_logprobs'.
    # If vLLM supports top_logprobs, it should be added to the map and this func's signature.
    # Assuming for now it's not explicitly mapped for vLLM.
    current_user_identifier = user_identifier if user_identifier is not None else cfg.get('user_identifier')


    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if isinstance(current_logprobs, str): current_logprobs = current_logprobs.lower() == "true"
    if custom_prompt_input:
        logging.info("vLLM: 'custom_prompt_input' received. Ensure incorporated if needed.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=current_api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp, # Pass vLLM's 'temperature'
        system_message=system_prompt, # Pass vLLM's 'system_prompt'
        streaming=current_streaming,
        max_tokens=current_max_tokens,
        top_p=current_top_p, # Pass vLLM's 'top_p'
        top_k=current_top_k,
        min_p=current_min_p, # Pass vLLM's 'min_p'
        n=current_n,
        stop=current_stop,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        logit_bias=current_logit_bias,
        seed=current_seed,
        response_format=current_response_format,
        logprobs=current_logprobs,
        # top_logprobs=current_top_logprobs, # Add if vLLM supports and mapped
        user_identifier=current_user_identifier,
        provider_name="vLLM",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
        # tools, tool_choice for vLLM? If supported, add to map and pass.
    )


# Aphrodite (seems to be an OpenAI compatible engine)
def chat_with_aphrodite(
    input_data: List[Dict[str, Any]],
    api_key: Optional[str] = None, # from map
    custom_prompt: Optional[str] = None,  # from map ('prompt')
    # Aphrodite's map uses 'temp':'temperature', etc.
    temperature: Optional[float] = None, # from map (mapped from generic 'temp')
    system_message: Optional[str] = None, # from map
    streaming: Optional[bool] = False,   # from map
    model: Optional[str] = None,         # from map
    top_k: Optional[int] = None,         # from map
    top_p: Optional[float] = None,         # from map (mapped from generic 'topp')
    min_p: Optional[float] = None,         # from map (mapped from generic 'minp')
    max_tokens: Optional[int] = None,      # from map
    seed: Optional[int] = None,          # from map
    stop: Optional[Union[str, List[str]]] = None, # from map
    response_format: Optional[Dict[str, str]] = None, # from map
    n: Optional[int] = None,             # from map
    logit_bias: Optional[Dict[str, float]] = None, # from map
    presence_penalty: Optional[float] = None, # from map
    frequency_penalty: Optional[float] = None, # from map
    logprobs: Optional[bool] = None,     # from map
    user_identifier: Optional[str] = None # from map
    # top_logprobs, tools, tool_choice not in Aphrodite's map currently
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    loaded_config_data = load_settings()
    cfg = loaded_config_data.get('aphrodite_api', {})

    api_base_url = cfg.get('api_ip')
    if not api_base_url:
        raise ChatConfigurationError(provider="aphrodite", message="Aphrodite API URL (api_ip) is required.")

    current_api_key = api_key or cfg.get('api_key')
    # Aphrodite might require a key if it's a hosted service or proxying to OpenAI
    if not current_api_key and "127.0.0.1" not in api_base_url and "localhost" not in api_base_url:
        logging.warning("Aphrodite: API key is missing and URL doesn't look local. This might be required.")

    current_model = model or cfg.get('model')
    if not current_model: # Model is usually required for OpenAI compatible
        # Some servers might have a default, but it's better to be explicit.
        logging.warning("Aphrodite: Model name is not specified. The server might use a default or fail.")


    current_temp = temperature if temperature is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
    current_max_tokens = max_tokens if max_tokens is not None else int(cfg.get('max_tokens', 4096))
    current_seed = seed if seed is not None else cfg.get('seed')
    current_stop = stop if stop is not None else cfg.get('stop')
    current_response_format = response_format if response_format is not None else cfg.get('response_format')
    current_n = n if n is not None else cfg.get('n')
    current_logit_bias = logit_bias if logit_bias is not None else cfg.get('logit_bias')
    current_presence_penalty = presence_penalty if presence_penalty is not None else cfg.get('presence_penalty')
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else cfg.get('frequency_penalty')
    current_logprobs = logprobs if logprobs is not None else cfg.get('logprobs')
    current_user_identifier = user_identifier if user_identifier is not None else cfg.get('user_identifier')

    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if isinstance(current_logprobs, str): current_logprobs = current_logprobs.lower() == "true"
    if custom_prompt:
        logging.info("Aphrodite: 'custom_prompt' received. Ensure incorporated if needed.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp, # Aphrodite receives 'temperature'
        system_message=system_message, # Aphrodite receives 'system_message'
        streaming=current_streaming,
        max_tokens=current_max_tokens,
        top_p=current_top_p, # Aphrodite receives 'top_p'
        top_k=current_top_k,
        min_p=current_min_p, # Aphrodite receives 'min_p'
        n=current_n,
        stop=current_stop,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        logit_bias=current_logit_bias,
        seed=current_seed,
        response_format=current_response_format,
        logprobs=current_logprobs,
        user_identifier=current_user_identifier,
        provider_name="Aphrodite Engine",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# Ollama (with OpenAI compatible endpoint)
def chat_with_ollama(
    input_data: List[Dict[str, Any]],
    api_key: Optional[str] = None, # from map, Ollama doesn't use key but map has it
    custom_prompt: Optional[str] = None,  # from map ('prompt')
    # Ollama map: 'temp':'temperature', 'system_message':'system_message', 'topp':'top_p', etc.
    temperature: Optional[float] = None,  # from map (mapped from generic 'temp')
    system_message: Optional[str] = None, # from map
    model: Optional[str] = None,          # from map
    streaming: Optional[bool] = False,    # from map
    top_p: Optional[float] = None,          # from map (mapped from generic 'topp')
    top_k: Optional[int] = None,          # from map
    # Ollama specific params from map, ensure they are OpenAI compatible if passed to generic func
    num_predict: Optional[int] = None,      # from map (mapped from generic 'max_tokens')
    seed: Optional[int] = None,             # from map
    stop: Optional[Union[str, List[str]]] = None, # from map
    format_str: Optional[str] = None,       # from map (mapped from generic 'response_format', e.g. "json")
                                            # _chat_with_openai_compatible_local_server expects dict {"type": "json_object"}
    presence_penalty: Optional[float] = None, # from map
    frequency_penalty: Optional[float] = None, # from map
    # api_url is specific for Ollama if passed directly, else from config
    api_url: Optional[str] = None,
    # Missing from Ollama PROVIDER_PARAM_MAP that _openai_compatible_server handles:
    # logit_bias, n (num_choices), user_identifier, logprobs, top_logprobs, tools, tool_choice, min_p
    # Add to signature and pass if Ollama supports them.
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    loaded_config_data = load_settings()
    cfg = loaded_config_data.get('ollama_api', {})

    current_api_base_url = api_url or cfg.get('api_url') # api_url from args takes precedence
    if not current_api_base_url:
        raise ChatConfigurationError(provider="ollama", message="Ollama API URL (api_url) is required.")

    current_api_key = api_key # Ollama generally doesn't use an API key, but pass if provided
    current_model = model or cfg.get('model')
    if not current_model:
        raise ChatConfigurationError(provider="ollama", message="Ollama model name is required.")

    current_temp = temperature if temperature is not None else float(cfg.get('temperature', 0.7)) # Ollama uses 'temperature'
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = top_p if top_p is not None else cfg.get('top_p') # Ollama uses 'top_p'
    current_top_k = top_k if top_k is not None else cfg.get('top_k') # Ollama uses 'top_k'
    current_max_tokens = num_predict if num_predict is not None else int(cfg.get('num_predict', cfg.get('max_tokens', 4096))) # Ollama uses 'num_predict'
    current_seed = seed if seed is not None else cfg.get('seed') # Ollama uses 'seed'
    current_stop = stop if stop is not None else cfg.get('stop') # Ollama uses 'stop' (list of strings)

    # Handle response_format for Ollama:
    # PROVIDER_PARAM_MAP has 'response_format': 'format' (e.g., "json")
    # _chat_with_openai_compatible_local_server expects {'type': 'json_object'}
    # We need to translate `format_str` (which is "json" from mapping) to the dict format.
    ollama_response_format_dict: Optional[Dict[str, str]] = None
    actual_format_str = format_str if format_str is not None else cfg.get('format')
    if actual_format_str and actual_format_str.lower() == 'json':
        ollama_response_format_dict = {"type": "json_object"}
    elif actual_format_str: # if other formats are supported by Ollama and have an OpenAI equivalent
        logging.warning(f"Ollama: Unsupported format string '{actual_format_str}' from 'format_str'. Only 'json' is translated to OpenAI's response_format dict.")


    current_presence_penalty = presence_penalty if presence_penalty is not None else cfg.get('presence_penalty') # Ollama uses 'presence_penalty'
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else cfg.get('frequency_penalty') # Ollama uses 'frequency_penalty'

    # Ollama also supports other native parameters like 'num_ctx', 'tfs_z', 'mirostat', etc.
    # These would need to be added to PROVIDER_PARAM_MAP and then to this function if full coverage is desired.
    # For now, focusing on OpenAI compatible ones.

    timeout = int(cfg.get('api_timeout', 300)) # Ollama can be slow
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt:
        logging.info("Ollama: 'custom_prompt' received. Ensure incorporated if needed.")

    # Ollama's /v1/chat/completions endpoint is OpenAI compatible
    return _chat_with_openai_compatible_local_server(
        api_base_url=current_api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key, # Pass along, though Ollama might not use it
        temp=current_temp,
        system_message=system_message,
        streaming=current_streaming,
        max_tokens=current_max_tokens, # map num_predict to max_tokens for OpenAI server
        top_p=current_top_p,
        top_k=current_top_k,
        # min_p is not in Ollama's map, pass if supported and added
        stop=current_stop,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        # logit_bias not in Ollama's map, pass if supported
        seed=current_seed,
        response_format=ollama_response_format_dict, # Pass translated format
        # n (num_choices) not in Ollama's map, pass if supported
        # user_identifier not in Ollama's map, pass if supported
        # logprobs, top_logprobs not in Ollama's map, pass if supported
        # tools, tool_choice not in Ollama's map, pass if supported by Ollama's OpenAI endpoint
        provider_name="Ollama",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# Custom OpenAI API 1
def chat_with_custom_openai(
    input_data: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    custom_prompt_arg: Optional[str] = None, # Mapped from 'prompt'
    temp: Optional[float] = None, # Mapped from generic 'temp'
    system_message: Optional[str] = None,
    streaming: Optional[bool] = False,
    model: Optional[str] = None,
    # PROVIDER_PARAM_MAP for custom-openai-api specific names:
    maxp: Optional[float] = None,             # Mapped from 'maxp' (likely top_p)
    minp: Optional[float] = None,             # Mapped from 'minp'
    topk: Optional[int] = None,               # Mapped from 'topk'
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    response_format: Optional[Dict[str, str]] = None,
    n: Optional[int] = None,
    user_identifier: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    loaded_config_data = load_settings()
    cfg_section = 'custom_openai_api'
    cfg = loaded_config_data.get(cfg_section, {})

    api_base_url = cfg.get('api_ip')
    if not api_base_url:
        raise ChatConfigurationError(provider=cfg_section, message=f"{cfg_section} API URL (api_ip) required.")

    current_api_key = api_key or cfg.get('api_key')
    if not current_api_key: # Custom OpenAI usually implies a key
        raise ChatConfigurationError(provider=cfg_section, message=f"{cfg_section} API Key required.")

    current_model = model or cfg.get('model')
    if not current_model:
        raise ChatConfigurationError(provider=cfg_section, message=f"{cfg_section} Model required.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', cfg.get('temp', 0.7))) # Mapped param is 'temp'
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = maxp if maxp is not None else cfg.get('top_p', cfg.get('maxp')) # Mapped param is 'maxp'
    current_top_k = topk if topk is not None else cfg.get('top_k', cfg.get('topk')) # Mapped param is 'topk'
    current_min_p = minp if minp is not None else cfg.get('min_p', cfg.get('minp')) # Mapped param is 'minp'
    current_max_tokens = max_tokens if max_tokens is not None else int(cfg.get('max_tokens', 4096))
    current_seed = seed if seed is not None else cfg.get('seed')
    current_stop = stop if stop is not None else cfg.get('stop')
    current_response_format = response_format if response_format is not None else cfg.get('response_format')
    current_n = n if n is not None else cfg.get('n')
    current_user_identifier = user_identifier if user_identifier is not None else cfg.get('user_identifier', cfg.get('user'))
    current_logit_bias = logit_bias if logit_bias is not None else cfg.get('logit_bias')
    current_presence_penalty = presence_penalty if presence_penalty is not None else cfg.get('presence_penalty')
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else cfg.get('frequency_penalty')
    current_logprobs = logprobs if logprobs is not None else cfg.get('logprobs')
    current_top_logprobs = top_logprobs if top_logprobs is not None else cfg.get('top_logprobs')
    current_tools = tools if tools is not None else cfg.get('tools')
    current_tool_choice = tool_choice if tool_choice is not None else cfg.get('tool_choice')

    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if isinstance(current_logprobs, str): current_logprobs = current_logprobs.lower() == "true"
    if custom_prompt_arg:
        logging.info(f"{cfg_section}: 'custom_prompt_arg' received. Ensure incorporated if needed.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp, # Use 'temp' as mapped
        system_message=system_message,
        streaming=current_streaming,
        max_tokens=current_max_tokens,
        top_p=current_top_p, # Use 'maxp' as mapped top_p
        top_k=current_top_k, # Use 'topk' as mapped
        min_p=current_min_p, # Use 'minp' as mapped
        n=current_n,
        stop=current_stop,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        logit_bias=current_logit_bias,
        seed=current_seed,
        response_format=current_response_format,
        tools=current_tools,
        tool_choice=current_tool_choice,
        logprobs=current_logprobs,
        top_logprobs=current_top_logprobs,
        user_identifier=current_user_identifier,
        provider_name=cfg_section.capitalize(),
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# Custom OpenAI API 2
def chat_with_custom_openai_2(
    input_data: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    custom_prompt_arg: Optional[str] = None, # Mapped from 'prompt'
    temp: Optional[float] = None, # Mapped from generic 'temp'
    system_message: Optional[str] = None,
    streaming: Optional[bool] = False,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    response_format: Optional[Dict[str, str]] = None,
    n: Optional[int] = None,
    user_identifier: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None
    # This custom API 2 map is missing top_k, min_p, max_p (top_p) compared to custom 1.
    # Assuming it doesn't support them or they are set server-side.
):
    if model and (model.lower() == "none" or model.strip() == ""): model = None
    loaded_config_data = load_settings()
    cfg_section = 'custom_openai_api_2'
    cfg = loaded_config_data.get(cfg_section, {})

    api_base_url = cfg.get('api_ip')
    if not api_base_url:
        raise ChatConfigurationError(provider=cfg_section, message=f"{cfg_section} API URL (api_ip) required.")

    current_api_key = api_key or cfg.get('api_key')
    if not current_api_key:
        raise ChatConfigurationError(provider=cfg_section, message=f"{cfg_section} API Key required.")

    current_model = model or cfg.get('model')
    if not current_model:
        raise ChatConfigurationError(provider=cfg_section, message=f"{cfg_section} Model required.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', cfg.get('temp', 0.7))) # Mapped param is 'temp'
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_max_tokens = max_tokens if max_tokens is not None else int(cfg.get('max_tokens', 4096))
    current_seed = seed if seed is not None else cfg.get('seed')
    current_stop = stop if stop is not None else cfg.get('stop')
    current_response_format = response_format if response_format is not None else cfg.get('response_format')
    current_n = n if n is not None else cfg.get('n')
    current_user_identifier = user_identifier if user_identifier is not None else cfg.get('user_identifier', cfg.get('user'))
    current_logit_bias = logit_bias if logit_bias is not None else cfg.get('logit_bias')
    current_presence_penalty = presence_penalty if presence_penalty is not None else cfg.get('presence_penalty')
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else cfg.get('frequency_penalty')
    current_logprobs = logprobs if logprobs is not None else cfg.get('logprobs')
    current_top_logprobs = top_logprobs if top_logprobs is not None else cfg.get('top_logprobs')
    current_tools = tools if tools is not None else cfg.get('tools')
    current_tool_choice = tool_choice if tool_choice is not None else cfg.get('tool_choice')

    # Parameters from custom-openai-api-1 that are NOT in custom-openai-api-2's map:
    # maxp (top_p), minp, topk. These will be None if not in map and passed to generic server.
    # Check config for these too, in case they are set there for this specific custom API.
    current_top_p = cfg.get('top_p', cfg.get('maxp'))
    current_top_k = cfg.get('top_k', cfg.get('topk'))
    current_min_p = cfg.get('min_p', cfg.get('minp'))


    timeout = int(cfg.get('api_timeout', 120))
    # Original code referenced 'custom_openai_2_api' for retry config for this one.
    # Let's try to be consistent with section name 'custom_openai_api_2' for all its configs.
    retry_cfg = loaded_config_data.get('custom_openai_2_api', cfg) # Fallback to main cfg if specific retry section missing
    api_retries = int(retry_cfg.get('api_retries', cfg.get('api_retries', 1)))
    api_retry_delay = int(retry_cfg.get('api_retry_delay', cfg.get('api_retry_delay', 1)))


    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if isinstance(current_logprobs, str): current_logprobs = current_logprobs.lower() == "true"
    if custom_prompt_arg:
        logging.info(f"{cfg_section}: 'custom_prompt_arg' received. Ensure incorporated if needed.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp, # Use 'temp' as mapped
        system_message=system_message,
        streaming=current_streaming,
        max_tokens=current_max_tokens,
        top_p=current_top_p, # Pass from config if available
        top_k=current_top_k, # Pass from config if available
        min_p=current_min_p, # Pass from config if available
        n=current_n,
        stop=current_stop,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        logit_bias=current_logit_bias,
        seed=current_seed,
        response_format=current_response_format,
        tools=current_tools,
        tool_choice=current_tool_choice,
        logprobs=current_logprobs,
        top_logprobs=current_top_logprobs,
        user_identifier=current_user_identifier,
        provider_name=cfg_section.capitalize(),
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )



def save_summary_to_file(summary: str, file_path: str): # Type hinting
    logging.debug("Now saving summary to file...")
    try:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        # Ensure the path is safe and within an expected directory if necessary
        summary_file_path = os.path.join(os.path.dirname(file_path), base_name + '_summary.txt')
        os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
        logging.debug(f"Opening summary file for writing: {summary_file_path}")
        with open(summary_file_path, 'w', encoding='utf-8') as file: # Added encoding
            file.write(summary)
        logging.info(f"Summary saved to file: {summary_file_path}")
    except Exception as e:
        logging.error(f"Error saving summary to file '{summary_file_path}': {e}", exc_info=True)
        # Depending on context, might want to re-raise or handle

#
#
#######################################################################################################################




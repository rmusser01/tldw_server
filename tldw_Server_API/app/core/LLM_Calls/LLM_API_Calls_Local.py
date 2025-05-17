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
from tldw_Server_API.app.core.Utils.Utils import logging, extract_text_from_segments, load_and_log_configs, \
    loaded_config_data


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
        system_message: Optional[str] = None,
        streaming: Optional[bool] = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,  # Some servers support min_p
        max_tokens: Optional[int] = None,
        provider_name: str = "Local OpenAI-Compatible Server",
        timeout: int = 120,
        api_retries: int = 1,
        api_retry_delay: int = 1
):
    logging.debug(f"{provider_name}: Chat request starting. API Base: {api_base_url}")

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})

    # Process input_data (messages_payload)
    # For true OpenAI compatibility, we can pass multimodal content.
    # However, we need to be mindful if the *specific local model* supports it.
    # For now, let's assume the structure is passed through.
    # If a local server specifically needs only text, pre-processing would be needed.
    images_present = False
    for msg in input_data:
        api_messages.append(msg)  # Pass the message object as is
        if isinstance(msg.get("content"), list):
            for part in msg.get("content", []):
                if part.get("type") == "image_url":
                    images_present = True
                    break
    if images_present:
        logging.info(f"{provider_name}: Multimodal content (images) detected in messages. "
                     f"Ensure the target model and server ({model_name or 'default model'}) support vision.")

    payload = {
        "messages": api_messages,
        "stream": streaming,
    }
    if model_name: payload["model"] = model_name  # Some servers require it, others ignore it if model is fixed
    if temp is not None: payload["temperature"] = temp
    if top_p is not None: payload["top_p"] = top_p
    if top_k is not None: payload["top_k"] = top_k
    if min_p is not None: payload["min_p"] = min_p  # OpenAI spec doesn't have min_p, but some local servers do
    if max_tokens is not None: payload["max_tokens"] = max_tokens

    # Construct full API URL for chat completions
    # Ensure no double slashes if api_base_url ends with / and path starts with /
    chat_completions_path = "v1/chat/completions"
    full_api_url = api_base_url.rstrip('/') + "/" + chat_completions_path.lstrip('/')

    logging.debug(
        f"{provider_name}: Posting to {full_api_url} with payload: { {k: v for k, v in payload.items() if k != 'messages'} }")

    try:
        if streaming:
            session = requests.Session()
            retry_strategy = Retry(total=api_retries, backoff_factor=api_retry_delay,
                                   status_forcelist=[429, 500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            response = session.post(full_api_url, headers=headers, json=payload, stream=True,
                                    timeout=timeout + 60)  # Longer timeout for streaming
            response.raise_for_status()
            logging.debug(f"{provider_name}: Streaming response received.")

            def stream_generator():
                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if line and line.strip():
                            yield line + "\n\n"  # Pass through raw SSE line
                    yield "data: [DONE]\n\n"
                except requests.exceptions.ChunkedEncodingError as e:
                    logging.error(f"{provider_name}: ChunkedEncodingError during stream: {e}", exc_info=True)
                    error_payload = json.dumps(
                        {"error": {"message": f"Stream connection error: {str(e)}", "type": "stream_error"}})
                    yield f"data: {error_payload}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    logging.error(f"{provider_name}: Error during stream iteration: {e}", exc_info=True)
                    error_payload = json.dumps(
                        {"error": {"message": f"Stream iteration error: {str(e)}", "type": "stream_error"}})
                    yield f"data: {error_payload}\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    response.close()

            return stream_generator()
        else:
            session = requests.Session()
            retry_strategy = Retry(total=api_retries, backoff_factor=api_retry_delay,
                                   status_forcelist=[429, 500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            response = session.post(full_api_url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            logging.debug(f"{provider_name}: Non-streaming request successful.")
            return response_data  # Return the full JSON response
    except requests.exceptions.HTTPError as e:
        logging.error(f"{provider_name}: HTTP Error: {e.response.status_code} - {e.response.text}", exc_info=True)
        # Re-raise to be caught by chat_api_call's handler
        raise
    except requests.RequestException as e:
        logging.error(f"{provider_name}: Request Exception: {e}", exc_info=True)
        raise ChatProviderError(provider=provider_name, message=f"Network error: {e}", status_code=504)
    except (ValueError, KeyError, TypeError) as e:
        logging.error(f"{provider_name}: Configuration or data error: {e}", exc_info=True)
        raise ChatBadRequestError(provider=provider_name, message=f"{provider_name} config/data error: {e}") from e


def chat_with_local_llm(  # Generic OpenAI-compatible endpoint, e.g., LM Studio
        input_data: List[Dict[str, Any]],  # Mapped from 'input_data'
        custom_prompt_arg: Optional[str] = None,  # Mapped from 'prompt', largely ignored
        temp: Optional[float] = None,
        system_message: Optional[str] = None,
        streaming: Optional[bool] = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        model: Optional[str] = None  # Model name can be passed
):
    loaded_config_data = load_and_log_configs()
    cfg = loaded_config_data.get('local_llm', {})  # Renamed 'local_llm_api' to 'local_llm' for consistency

    api_base_url = cfg.get('api_ip', 'http://127.0.0.1:8080')  # Default to common local proxy
    # API key might not be needed for many local servers
    api_key = cfg.get('api_key')

    current_model = model or cfg.get('model')  # Use model from args, then config
    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
    current_max_tokens = int(cfg.get('max_tokens', 4096))
    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt_arg:
        logging.warning("Local LLM: 'custom_prompt_arg' is generally ignored; prompts should be in 'input_data'.")

    # This now uses the generic OpenAI compatible handler
    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=api_key,  # Often None for local
        temp=current_temp,
        system_message=system_message,
        streaming=current_streaming,
        top_k=current_top_k,
        top_p=current_top_p,
        min_p=current_min_p,
        max_tokens=current_max_tokens,
        provider_name="Local LLM (generic OpenAI compatible)",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


def chat_with_llama(  # llama.cpp server with OpenAI compatible endpoint
        input_data: List[Dict[str, Any]],  # Mapped from 'input_data'
        custom_prompt: Optional[str] = None,  # Mapped from 'prompt', ignored
        temp: Optional[float] = None,
        api_url: Optional[str] = None,  # Passed positionally if not None
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,  # Mapped from 'system_message'
        streaming: Optional[bool] = False,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        model: Optional[str] = None  # Model name for the payload
):
    loaded_config_data = load_and_log_configs()
    cfg = loaded_config_data.get('llama_api', {})

    # api_url is a positional argument for llama.cpp in PROVIDER_PARAM_MAP, not a generic one.
    # This means it's not passed from chat_api_call unless specifically handled.
    # The map doesn't have 'api_url'. Assuming it's loaded from config here.
    current_api_base_url = api_url or cfg.get('api_ip')  # api_url from args takes precedence
    current_api_key = api_key or cfg.get('api_key')  # API Key from args, then config
    current_model = model or cfg.get('model')  # Model from args, then config

    if not current_api_base_url:
        raise ChatConfigurationError(provider="llama.cpp", message="Llama.cpp API URL is required but not found.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
    current_max_tokens = int(cfg.get('max_tokens', 4096))
    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt:
        logging.warning("Llama.cpp: 'custom_prompt' is generally ignored; prompts should be in 'input_data'.")

    # llama.cpp server provides an OpenAI-compatible /v1/chat/completions endpoint
    return _chat_with_openai_compatible_local_server(
        api_base_url=current_api_base_url,
        model_name=current_model,  # Pass model name
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp,
        system_message=system_prompt,  # Use system_prompt which maps to system_message
        streaming=current_streaming,
        top_k=current_top_k,
        top_p=current_top_p,
        min_p=current_min_p,
        max_tokens=current_max_tokens,
        provider_name="Llama.cpp",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# System prompts not supported through API requests.
# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def chat_with_kobold(  # KoboldAI native API (/api/v1/generate)
        input_data: List[Dict[str, Any]],  # Mapped from 'input_data'
        api_key: Optional[str] = None,  # Mapped from 'api_key'
        custom_prompt_input: Optional[str] = None,  # Mapped from 'prompt', can be part of prompt construction
        temp: Optional[float] = None,
        system_message: Optional[str] = None,  # Mapped from 'system_message'
        streaming: Optional[bool] = False,  # Note: Kobold native streaming is tricky
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        model: Optional[str] = None  # Model for Kobold if selectable via API (often fixed by server)
):
    logging.debug("KoboldAI (Native): Chat request starting...")
    loaded_config_data = load_and_log_configs()
    cfg = loaded_config_data.get('kobold_api', {})

    current_api_key = api_key or cfg.get('api_key')  # Kobold often doesn't need a key
    api_url = cfg.get('api_ip')  # URL for /api/v1/generate
    # Kobold's /api/v1/generate doesn't typically use a "model" param in request body, it's fixed by server.
    # The 'model' param from chat_api_call might be used for logging or if a specific Kobold setup allows selection.

    if not api_url:
        raise ChatConfigurationError(provider="kobold", message="KoboldAI API URL is required but not found.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    # Kobold native streaming for /generate is not standard SSE.
    # The original code had a FIXME for Kobold streaming and then set it to False.
    # For reliable integration, usually use non-streaming or their OpenAI compatible endpoint if available.
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    if current_streaming:
        logging.warning(
            "KoboldAI (Native): Streaming with /api/v1/generate is often non-standard or unsupported. Forcing non-streaming.")
        current_streaming = False  # Override to False due to complexity

    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    max_context_length = int(cfg.get('max_context_length', 2048))  # Kobold uses max_context_length
    max_length = int(cfg.get('max_length', 200))  # Max tokens to generate in Kobold
    timeout = int(cfg.get('api_timeout', 180))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"

    # Construct a single prompt string from messages_payload for Kobold's native API
    full_prompt_parts = []
    if system_message:
        full_prompt_parts.append(system_message)

    for i, msg in enumerate(input_data):
        role = msg.get("role", "user")
        text_content = _extract_text_from_message_content(msg.get("content"), "KoboldAI (Native)", i)
        # Kobold doesn't use roles in the prompt string, just concatenates.
        # Could add "User: " / "Assistant: " prefixes if the model is tuned for it.
        # For simplicity, just concatenating content.
        full_prompt_parts.append(text_content)

    # The 'custom_prompt_input' (from PROVIDER_PARAM_MAP 'prompt') might be a final instruction.
    # Chat_Functions.chat() already incorporates its 'custom_prompt' into the last user message.
    # So, custom_prompt_input here might be redundant or intended for a different purpose if Kobold had it.
    # For now, assuming it's already in input_data.
    if custom_prompt_input:
        logging.info(
            "KoboldAI (Native): 'custom_prompt_input' provided. It's assumed this content is already part of 'input_data' by the calling 'chat' function.")
        # If it was meant to be appended *again*: full_prompt_parts.append(custom_prompt_input)

    final_prompt_string = "\n\n".join(full_prompt_parts).strip()

    headers = {'Content-Type': 'application/json'}
    if current_api_key: headers['X-Api-Key'] = current_api_key  # Some Kobold forks might use this

    payload = {
        "prompt": final_prompt_string,
        "temperature": current_temp,
        "top_p": current_top_p,
        "top_k": current_top_k,
        "max_context_length": max_context_length,
        "max_length": max_length,
        "stream": current_streaming,  # Will be False here
        # Other Kobold specific params: rep_pen, etc. can be added from cfg if needed
    }
    if cfg.get('rep_pen') is not None: payload['rep_pen'] = float(cfg['rep_pen'])

    logging.debug(
        f"KoboldAI (Native): Posting to {api_url} with prompt (first 200 chars): {final_prompt_string[:200]}...")

    try:
        session = requests.Session()
        retry_strategy = Retry(total=api_retries, backoff_factor=api_retry_delay,
                               status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        response = session.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        response_data = response.json()

        if response_data and 'results' in response_data and len(response_data['results']) > 0:
            generated_text = response_data['results'][0]['text'].strip()
            logging.debug("KoboldAI (Native): Chat request successful.")
            # Kobold returns just the completion, wrap it to look like an OpenAI choice for consistency if used by chat_api_call
            return {"choices": [{"message": {"role": "assistant", "content": generated_text}}]}
        else:
            logging.error(f"KoboldAI (Native): Unexpected response structure: {response_data}")
            raise ChatProviderError(provider="kobold", message="Unexpected response structure from KoboldAI.")

    except requests.exceptions.HTTPError as e:
        logging.error(f"KoboldAI (Native): HTTP Error: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise
    except requests.RequestException as e:
        logging.error(f"KoboldAI (Native): Request Exception: {e}", exc_info=True)
        raise ChatProviderError(provider="kobold", message=f"Network error: {e}", status_code=504)
    except (ValueError, KeyError, TypeError) as e:
        logging.error(f"KoboldAI (Native): Data or config error: {e}", exc_info=True)
        raise ChatBadRequestError(provider="kobold", message=f"KoboldAI config/data error: {e}") from e


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
# Oobabooga with OpenAI extension
def chat_with_oobabooga(
    input_data: List[Dict[str, Any]],     # Mapped from 'input_data'
    custom_prompt: Optional[str] = None,  # Mapped from 'prompt', ignored
    temp: Optional[float] = None,
    api_url: Optional[str] = None,        # Passed positionally
    api_key: Optional[str] = None,        # Mapped, Ooba might not use it
    system_prompt: Optional[str] = None,  # Mapped from 'system_message'
    streaming: Optional[bool] = False,
    top_p: Optional[float] = None,
    model: Optional[str] = None          # Model name for the payload
):
    loaded_config_data = load_and_log_configs()
    cfg = loaded_config_data.get('ooba_api', {}) # Ensure this section exists in your config

    current_api_base_url = api_url or cfg.get('api_ip') # api_url from args takes precedence
    # Oobabooga's OpenAI extension usually doesn't require an API key
    current_api_key = api_key or cfg.get('api_key')
    current_model = model or cfg.get('model') # Model loaded in Ooba, can be passed in payload

    if not current_api_base_url:
        raise ChatConfigurationError(provider="ooba", message="Oobabooga API URL is required but not found.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    # Ooba's OpenAI ext might support other params like top_k, max_tokens
    current_top_k = cfg.get('top_k')
    current_max_tokens = int(cfg.get('max_tokens', 4096))
    timeout = int(cfg.get('api_timeout', 180)) # Ooba can be slow
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt:
        logging.warning("Oobabooga: 'custom_prompt' is generally ignored; prompts should be in 'input_data'.")

    # Oobabooga with OpenAI extension uses the generic handler
    return _chat_with_openai_compatible_local_server(
        api_base_url=current_api_base_url,
        model_name=current_model, # Pass the model name loaded in Ooba
        input_data=input_data,
        api_key=current_api_key, # Usually None
        temp=current_temp,
        system_message=system_prompt,
        streaming=current_streaming,
        top_p=current_top_p,
        top_k=current_top_k,
        max_tokens=current_max_tokens,
        provider_name="Oobabooga (OpenAI Extension)",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# TabbyAPI (seems OpenAI compatible)
def chat_with_tabbyapi(
    input_data: List[Dict[str, Any]],             # Mapped from 'input_data'
    custom_prompt_input: Optional[str] = None,    # Mapped from 'prompt', ignored
    api_key: Optional[str] = None,                # Mapped
    temp: Optional[float] = None,
    system_message: Optional[str] = None,         # Mapped
    streaming: Optional[bool] = False,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    model: Optional[str] = None                   # Mapped
):
    loaded_config_data = load_and_log_configs()
    cfg = loaded_config_data.get('tabby_api', {})

    api_base_url = cfg.get('api_ip') # Assuming api_ip is the base URL
    current_api_key = api_key or cfg.get('api_key')
    current_model = model or cfg.get('model')

    if not api_base_url:
        raise ChatConfigurationError(provider="tabbyapi", message="TabbyAPI URL is required.")
    if not current_model:
        raise ChatConfigurationError(provider="tabbyapi", message="TabbyAPI model is required.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
    current_max_tokens = int(cfg.get('max_tokens', 4096))
    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt_input:
        logging.warning("TabbyAPI: 'custom_prompt_input' is generally ignored.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp,
        system_message=system_message,
        streaming=current_streaming,
        top_k=current_top_k,
        top_p=current_top_p,
        min_p=current_min_p,
        max_tokens=current_max_tokens,
        provider_name="TabbyAPI",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# vLLM (OpenAI compatible)
def chat_with_vllm(
    input_data: List[Dict[str, Any]],             # Mapped from 'input_data'
    custom_prompt_input: Optional[str] = None,    # Mapped from 'prompt', ignored
    api_key: Optional[str] = None,                # Mapped
    temp: Optional[float] = None,
    system_prompt: Optional[str] = None,          # Mapped from 'system_message'
    streaming: Optional[bool] = False,
    topp: Optional[float] = None,                 # Mapped from 'topp' (top_p)
    topk: Optional[int] = None,                   # Mapped from 'topk'
    minp: Optional[float] = None,                 # Mapped from 'minp'
    model: Optional[str] = None,                  # Mapped
    vllm_api_url: Optional[str] = None            # Specific config, not from generic map
):
    loaded_config_data = load_and_log_configs()
    cfg = loaded_config_data.get('vllm_api', {})

    current_api_base_url = vllm_api_url or cfg.get('api_ip') # vllm_api_url from specific arg if provided
    current_api_key = api_key or cfg.get('api_key') # vLLM might not require a key
    current_model = model or cfg.get('model') # Model served by vLLM

    if not current_api_base_url:
        raise ChatConfigurationError(provider="vllm", message="vLLM API URL is required.")
    # vLLM model name is often part of the OpenAI payload not the URL path, but good to have if fixed.

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = topp # maps from 'topp'
    current_top_k = topk # maps from 'topk'
    current_min_p = minp # maps from 'minp'
    current_max_tokens = int(cfg.get('max_tokens', 4096))
    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt_input:
        logging.warning("vLLM: 'custom_prompt_input' is generally ignored.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=current_api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp,
        system_message=system_prompt, # Use system_prompt which maps to system_message
        streaming=current_streaming,
        top_k=current_top_k,
        top_p=current_top_p,
        min_p=current_min_p,
        max_tokens=current_max_tokens,
        provider_name="vLLM",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# Aphrodite (seems to be an OpenAI compatible engine)
def chat_with_aphrodite(
    input_data: List[Dict[str, Any]],     # Mapped
    custom_prompt: Optional[str] = None,  # Mapped from 'prompt', ignored
    api_key: Optional[str] = None,        # Mapped
    temp: Optional[float] = None,
    system_message: Optional[str] = None, # Mapped
    streaming: Optional[bool] = False,
    topp: Optional[float] = None,
    minp: Optional[float] = None,
    topk: Optional[int] = None,
    model: Optional[str] = None           # Mapped
):
    loaded_config_data = load_and_log_configs()
    cfg = loaded_config_data.get('aphrodite_api', {})

    api_base_url = cfg.get('api_ip') # Assuming api_ip is the base URL
    current_api_key = api_key or cfg.get('api_key')
    current_model = model or cfg.get('model')

    if not api_base_url:
        raise ChatConfigurationError(provider="aphrodite", message="Aphrodite API URL is required.")
    if not current_api_key: # Aphrodite usually requires an OpenAI key if it's an engine using it
        logging.warning("Aphrodite: API key is missing. This might be required.")
    if not current_model:
        raise ChatConfigurationError(provider="aphrodite", message="Aphrodite model name is required.")


    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = topp # maps from 'topp'
    current_top_k = topk # maps from 'topk'
    current_min_p = minp # maps from 'minp'
    current_max_tokens = int(cfg.get('max_tokens', 4096))
    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt:
        logging.warning("Aphrodite: 'custom_prompt' is generally ignored.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp,
        system_message=system_message,
        streaming=current_streaming,
        top_k=current_top_k,
        top_p=current_top_p,
        min_p=current_min_p,
        max_tokens=current_max_tokens,
        provider_name="Aphrodite Engine",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# Ollama (with OpenAI compatible endpoint)
def chat_with_ollama(
    input_data: List[Dict[str, Any]],     # Mapped
    custom_prompt: Optional[str] = None,  # Mapped from 'prompt', ignored
    api_url: Optional[str] = None,        # Positional argument
    api_key: Optional[str] = None,        # Mapped, Ollama usually doesn't need it
    temp: Optional[float] = None,
    system_message: Optional[str] = None, # Mapped
    model: Optional[str] = None,          # Mapped
    streaming: Optional[bool] = False,
    top_p: Optional[float] = None         # Mapped from 'topp'
):
    loaded_config_data = load_and_log_configs()
    cfg = loaded_config_data.get('ollama_api', {})

    current_api_base_url = api_url or cfg.get('api_url') # api_url from args takes precedence
    current_api_key = api_key or cfg.get('api_key') # Usually None for Ollama
    current_model = model or cfg.get('model')

    if not current_api_base_url:
        raise ChatConfigurationError(provider="ollama", message="Ollama API URL is required.")
    if not current_model:
        raise ChatConfigurationError(provider="ollama", message="Ollama model name is required.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    # Ollama also supports top_k, min_p, max_tokens - can add from cfg if needed
    current_top_k = cfg.get('top_k')
    current_min_p = cfg.get('min_p')
    current_max_tokens = int(cfg.get('max_tokens', 4096))
    timeout = int(cfg.get('api_timeout', 300)) # Ollama can be slow
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt:
        logging.warning("Ollama: 'custom_prompt' is generally ignored.")

    # Ollama's /v1/chat/completions endpoint is OpenAI compatible
    return _chat_with_openai_compatible_local_server(
        api_base_url=current_api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp,
        system_message=system_message,
        streaming=current_streaming,
        top_p=current_top_p,
        top_k=current_top_k,
        min_p=current_min_p,
        max_tokens=current_max_tokens,
        provider_name="Ollama",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# Custom OpenAI API 1
def chat_with_custom_openai(
    input_data: List[Dict[str, Any]],         # Mapped
    custom_prompt_arg: Optional[str] = None,  # Mapped from 'prompt', ignored
    api_key: Optional[str] = None,            # Mapped
    temp: Optional[float] = None,
    system_message: Optional[str] = None,     # Mapped
    streaming: Optional[bool] = False,
    maxp: Optional[float] = None,             # Mapped from 'maxp' (top_p)
    minp: Optional[float] = None,             # Mapped from 'minp'
    topk: Optional[int] = None,               # Mapped from 'topk'
    model: Optional[str] = None               # Mapped
):
    loaded_config_data = load_and_log_configs()
    cfg = loaded_config_data.get('custom_openai_api', {}) # Section name from old code

    api_base_url = cfg.get('api_ip') # Assuming api_ip is the base URL
    current_api_key = api_key or cfg.get('api_key')
    current_model = model or cfg.get('model')

    if not api_base_url:
        raise ChatConfigurationError(provider="custom-openai-api", message="Custom OpenAI API URL required.")
    if not current_api_key:
        raise ChatConfigurationError(provider="custom-openai-api", message="Custom OpenAI API Key required.")
    if not current_model:
        raise ChatConfigurationError(provider="custom-openai-api", message="Custom OpenAI Model required.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_p = maxp # maps from 'maxp'
    current_top_k = topk # maps from 'topk'
    current_min_p = minp # maps from 'minp'
    current_max_tokens = int(cfg.get('max_tokens', 4096))
    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt_arg:
        logging.warning("Custom OpenAI API: 'custom_prompt_arg' is generally ignored.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp,
        system_message=system_message,
        streaming=current_streaming,
        top_k=current_top_k,
        top_p=current_top_p,
        min_p=current_min_p,
        max_tokens=current_max_tokens,
        provider_name="Custom OpenAI API",
        timeout=timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay
    )


# Custom OpenAI API 2
def chat_with_custom_openai_2(
    input_data: List[Dict[str, Any]],         # Mapped
    custom_prompt_arg: Optional[str] = None,  # Mapped from 'prompt', ignored
    api_key: Optional[str] = None,            # Mapped
    temp: Optional[float] = None,
    system_message: Optional[str] = None,     # Mapped
    streaming: Optional[bool] = False,
    model: Optional[str] = None               # Mapped
    # Original didn't take maxp, minp, topk
):
    loaded_config_data = load_and_log_configs()
    # Note: Original code referenced 'custom_openai_2_api' for retry config, but 'custom_openai_api_2' for others.
    # Using 'custom_openai_api_2' consistently for the section name.
    cfg = loaded_config_data.get('custom_openai_api_2', {})

    api_base_url = cfg.get('api_ip')
    current_api_key = api_key or cfg.get('api_key')
    current_model = model or cfg.get('model')

    if not api_base_url:
        raise ChatConfigurationError(provider="custom-openai-api-2", message="Custom OpenAI API-2 URL required.")
    if not current_api_key:
        raise ChatConfigurationError(provider="custom-openai-api-2", message="Custom OpenAI API-2 Key required.")
    if not current_model:
        raise ChatConfigurationError(provider="custom-openai-api-2", message="Custom OpenAI API-2 Model required.")

    current_temp = temp if temp is not None else float(cfg.get('temperature', 0.7))
    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_max_tokens = int(cfg.get('max_tokens', 4096))
    current_top_p = cfg.get('top_p') # This API might not use it, but check config
    current_top_k = cfg.get('top_k')
    current_min_p = cfg.get('min_p')
    timeout = int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1)) # Using consistent retry config key
    api_retry_delay = int(cfg.get('api_retry_delay', 1))


    if isinstance(current_streaming, str): current_streaming = current_streaming.lower() == "true"
    if custom_prompt_arg:
        logging.warning("Custom OpenAI API-2: 'custom_prompt_arg' is generally ignored.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
        temp=current_temp,
        system_message=system_message,
        streaming=current_streaming,
        max_tokens=current_max_tokens,
        top_p=current_top_p, # Pass if configured, server might ignore
        top_k=current_top_k, # Pass if configured
        min_p=current_min_p, # Pass if configured
        provider_name="Custom OpenAI API-2",
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




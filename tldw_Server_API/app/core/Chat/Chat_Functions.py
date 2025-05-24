# Chat_Functions.py
# Description: Chat functions for interacting with the LLMs as chatbots
"""
This module provides a comprehensive set of functions and classes for managing chat interactions,
character data, and chat dictionaries in a multimodal chatbot system. It includes functionality
for handling API calls, saving and loading chat history, managing character cards, and processing
user input with chat dictionaries.

Key Features:
- Chat API call handling with error management and multimodal support.
- Chat history saving and exporting to JSON.
- Character card management, including saving, loading, and updating character data.
- Chat dictionary processing for keyword-based text replacement and token budget management.
"""
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
from typing import List, Dict, Any, Tuple, Optional, Union
#
# 3rd-party Libraries
import requests
import logging

from loguru import logger

#
# Local Imports
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ResponseFormat
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError, ChatConfigurationError, ChatAPIError, \
    ChatProviderError, ChatRateLimitError, ChatAuthenticationError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError, ConflictError, CharactersRAGDBError
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_openai, chat_with_anthropic, chat_with_cohere, \
    chat_with_groq, chat_with_openrouter, chat_with_deepseek, chat_with_mistral, chat_with_huggingface, chat_with_google
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls_Local import chat_with_aphrodite, chat_with_local_llm, chat_with_ollama, \
    chat_with_kobold, chat_with_llama, chat_with_oobabooga, chat_with_tabbyapi, chat_with_vllm, chat_with_custom_openai, \
    chat_with_custom_openai_2
from tldw_Server_API.app.core.Utils.Utils import generate_unique_filename, logging
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.config import load_and_log_configs
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

# FIXME - Validate below
# 1. Dispatch table for handler functions
API_CALL_HANDLERS = {
    'openai': chat_with_openai,
    'anthropic': chat_with_anthropic,
    'cohere': chat_with_cohere,
    'groq': chat_with_groq,
    'openrouter': chat_with_openrouter,
    'deepseek': chat_with_deepseek,
    'mistral': chat_with_mistral,
    'google': chat_with_google,
    'huggingface': chat_with_huggingface,
    'llama.cpp': chat_with_llama,
    'kobold': chat_with_kobold,
    'ooba': chat_with_oobabooga,
    'tabbyapi': chat_with_tabbyapi,
    'vllm': chat_with_vllm,
    'local-llm': chat_with_local_llm,
    'ollama': chat_with_ollama,
    'aphrodite': chat_with_aphrodite,
    'custom-openai-api': chat_with_custom_openai,
    'custom-openai-api-2': chat_with_custom_openai_2,
}
"""
A dispatch table mapping API endpoint names (e.g., 'openai') to their
corresponding handler functions (e.g., `chat_with_openai`). This is used by
`chat_api_call` to route requests to the appropriate LLM provider.
FIXME: The mappings and handlers should be validated for correctness.
"""

# 2. Parameter mapping for each provider
# Maps generic chat_api_call param name to provider-specific param name
PROVIDER_PARAM_MAP = {
    'openai': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'maxp': 'maxp',
        'model': 'model',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
    },
    'anthropic': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_prompt',
        'streaming': 'streaming',
        'model': 'model',
        'topp': 'topp',
        'topk': 'topk',
        'tools': 'tools',
        #'tool_choice': 'tool_choice',
        'max_tokens': 'max_tokens',  # Anthropic uses max_tokens
        'stop': 'stop_sequences',  # Anthropic uses stop_sequences
    },
    'cohere': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_prompt',
        'streaming': 'streaming',
        'model': 'model',
        'topp': 'topp',
        'topk': 'topk',
        'tools': 'tools',
        #'tool_choice': 'tool_choice',
        'max_tokens': 'max_tokens',
        'stop': 'stop_sequences',
        'seed': 'seed',
        'n': 'num_generations',
        'frequency_penalty': 'frequency_penalty',
        'presence_penalty': 'presence_penalty',
    },
    'groq': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'maxp': 'maxp',
        'model':'model', # Groq also uses top_p, handled by chat_with_groq
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'openrouter': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'minp',
        'model':'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
    },
    'deepseek': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'topp',
        'model':'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',  # if supported
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'mistral': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'topp',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'random_seed',  # Mistral uses random_seed
        'topk': 'top_k',  # Mistral uses top_k
    },
    'google': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'topp',
        'topk': 'topk',
        'tools': 'tools',
        #'tool_choice': 'tool_choice',
        'model':'model',
        'max_tokens': 'max_output_tokens',
        'stop': 'stop_sequences',  # List of strings
        'n': 'candidate_count',
    },
    'huggingface': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'model':'model',
        'max_tokens': 'max_new_tokens',  # Common for TGI
        'topp': 'top_p',
        'topk': 'top_k',
        'seed': 'seed',
        'stop': 'stop',  # often 'stop_sequences'
    },
    'llama.cpp': { # Has api_url as a positional argument which needs special handling if not None
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt',
        'temp': 'temperature',
        'system_message': 'system_prompt',
        'streaming': 'stream',
        'topp': 'top_p', 'topk': 'top_k',
        'minp': 'min_p',
        'model':'model',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'max_tokens': 'n_predict', # Common for llama.cpp server
        'seed': 'seed',
        'stop': 'stop', # list of strings
        'response_format': 'response_format', # if OpenAI compatible endpoint
        'logit_bias': 'logit_bias',
        'n': 'n_probs', # FIXME: n_probs mapping might not be direct.
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'kobold': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_input',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'topp': 'top_p',
        'topk': 'top_k',
        'model':'model',
        'max_tokens': 'max_length',  # or 'max_context_length'
        'stop': 'stop_sequence',  # Often a list
        'n': 'num_responses',
        'seed': 'seed',
    },
    'ooba': { # api_url also a consideration like llama.cpp
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt',
        'temp': 'temperature',
        'system_message': 'system_prompt', # often part of messages or specific param
        'streaming': 'stream',
        'topp': 'top_p',
        'model':'model',
        'topk': 'top_k',
        'minp': 'min_p',
        'max_tokens': 'max_tokens', # or 'max_new_tokens'
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'tabbyapi': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_input',
        'temp': 'temperature',
        'system_message': 'system_message',
        'streaming': 'stream',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'model':'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
    },
    'vllm': { # vllm_api_url consideration
                'api_key': 'api_key', 'messages_payload': 'input_data', 'prompt': 'custom_prompt_input',
        'temp': 'temperature', 'system_message': 'system_prompt', 'streaming': 'stream',
        'topp': 'top_p', 'topk': 'top_k', 'minp': 'min_p', 'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'user_identifier': 'user',
    },
    'local-llm': {
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temperature',
        'system_message': 'system_message',
        'streaming': 'stream',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'model':'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
    },
    'ollama': { # api_url consideration
        'api_key': 'api_key', # api_key is not used by ollama directly, url is more important
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt', # This is 'prompt' for generate, 'messages' for chat
        'temp': 'temperature',
        'system_message': 'system', # Part of request body
        'streaming': 'stream',
        'topp': 'top_p',
        'topk': 'top_k',
        'model': 'model',
        'max_tokens': 'num_predict', # For generate endpoint, chat might be different
        'seed': 'seed',
        'stop': 'stop', # list of strings
        'response_format': 'format', # 'json' string
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
    },
    'aphrodite': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt',
        'temp': 'temperature',
        'system_message': 'system_message',
        'streaming': 'stream',
        'topp': 'top_p',
        'topk': 'top_k',
        'minp': 'min_p',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'user_identifier': 'user',
    },
    'custom-openai-api': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'maxp': 'maxp',
        'minp':'minp',
        'topk':'topk',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
    },
    'custom-openai-api-2': {
        'api_key': 'api_key',
        'messages_payload': 'input_data',
        'prompt': 'custom_prompt_arg',
        'temp': 'temp',
        'system_message': 'system_message',
        'streaming': 'streaming',
        'model': 'model',
        'max_tokens': 'max_tokens',
        'seed': 'seed',
        'stop': 'stop',
        'response_format': 'response_format',
        'n': 'n',
        'user_identifier': 'user',
        'tools': 'tools',
        'tool_choice': 'tool_choice',
        'logit_bias': 'logit_bias',
        'presence_penalty': 'presence_penalty',
        'frequency_penalty': 'frequency_penalty',
        'logprobs': 'logprobs',
        'top_logprobs': 'top_logprobs',
    },
    # Add other providers here
}
"""
Maps generic parameter names used in `chat_api_call` to provider-specific
parameter names for each LLM API. This allows `chat_api_call` to use a
consistent interface while adapting to the idiosyncrasies of different providers.
FIXME: The mappings should be validated for correctness and completeness for each provider.
"""

def chat_api_call(
    api_endpoint: str,
    messages_payload: List[Dict[str, Any]], # CHANGED from input_data, prompt
    api_key: Optional[str] = None,
    temp: Optional[float] = None,
    system_message: Optional[str] = None, # Still passed separately, some providers might use it, others expect it in messages_payload
    streaming: Optional[bool] = None,
    minp: Optional[float] = None,
    maxp: Optional[float] = None, # Often maps to top_p
    model: Optional[str] = None,
    topk: Optional[int] = None,
    topp: Optional[float] = None, # Often maps to top_p
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    response_format: Optional[Dict[str, str]] = None,  # Expects {'type': 'text' | 'json_object'}
    n: Optional[int] = None,
    user_identifier: Optional[str] = None  # Renamed from 'user' to avoid conflict with 'user' role in messages
    ):
    """
    Acts as a unified dispatcher to call various LLM API providers.

    This function routes chat requests to the appropriate LLM provider based on
    `api_endpoint`. It uses `API_CALL_HANDLERS` to find the correct handler
    function and `PROVIDER_PARAM_MAP` to translate generic parameters to
    provider-specific ones.

    Args:
        api_endpoint: The identifier for the target LLM provider (e.g., "openai", "anthropic").
        messages_payload: A list of message objects (OpenAI format: `{'role': ..., 'content': ...}`)
                          representing the conversation history and current user message.
        api_key: The API key for the specified provider.
        temp: Temperature for sampling, controlling randomness.
        system_message: An optional system-level instruction for the LLM. How this is
                        used depends on the provider; some prepend it to messages, others
                        have a dedicated parameter.
        streaming: Whether to stream the response from the LLM.
        minp: Minimum probability for token sampling (nucleus sampling related).
        maxp: Maximum probability for token sampling (often maps to `top_p`).
        model: The specific model to use for the LLM provider.
        topk: Top-K sampling parameter.
        topp: Top-P (nucleus) sampling parameter.
        logprobs: Whether to return log probabilities of tokens.
        top_logprobs: Number of top log probabilities to return.
        logit_bias: A dictionary to bias token generation probabilities.
        presence_penalty: Penalty for new tokens based on their presence in the text so far.
        frequency_penalty: Penalty for new tokens based on their frequency in the text so far.
        tools: A list of tools the model may call.
        tool_choice: Controls which tool the model should call.
        max_tokens: The maximum number of tokens to generate in the response.
        seed: A seed for deterministic generation, if supported.
        stop: A string or list of strings that, when generated, will cause the LLM to stop.
        response_format: Specifies the format of the response (e.g., `{'type': 'json_object'}`).
        n: The number of chat completion choices to generate.
        user_identifier: An identifier for the end-user, for tracking or moderation purposes.

    Returns:
        The LLM's response. This can be a string for non-streaming responses or
        a generator for streaming responses. The exact type depends on the
        underlying provider's handler function.

    Raises:
        ValueError: If the `api_endpoint` is unsupported or if there's a parameter issue.
        ChatAuthenticationError: If authentication with the provider fails (e.g., invalid API key).
        ChatRateLimitError: If the provider's rate limit is exceeded.
        ChatBadRequestError: If the request to the provider is malformed or invalid.
        ChatProviderError: If the provider's server returns an error or there's a network issue.
        ChatConfigurationError: If there's a configuration issue for the specified provider.
        ChatAPIError: For other unexpected API-related errors.
        requests.exceptions.HTTPError: Propagated from underlying HTTP requests if not caught and re-raised.
        requests.exceptions.RequestException: For network errors during the request.
    """
    endpoint_lower = api_endpoint.lower()
    logging.info(f"Chat API Call - Routing to endpoint: {endpoint_lower}")
    log_counter("chat_api_call_attempt", labels={"api_endpoint": endpoint_lower})
    start_time = time.time()

    handler = API_CALL_HANDLERS.get(endpoint_lower)
    if not handler:
        logging.error(f"Unsupported API endpoint requested: {api_endpoint}")
        raise ValueError(f"Unsupported API endpoint: {api_endpoint}")

    params_map = PROVIDER_PARAM_MAP.get(endpoint_lower, {})
    call_kwargs = {}

    # Construct kwargs for the handler function based on the map
    # This requires careful mapping and ensuring the handler functions are adapted.

    # Generic parameters available from chat_api_call signature
    available_generic_params = {
        'api_key': api_key,
        'messages_payload': messages_payload, # This is the core change
        'temp': temp,
        'system_message': system_message,
        'streaming': streaming,
        'minp': minp,
        'maxp': maxp, # Will be mapped to top_p by some providers
        'model': model,
        'topk': topk,
        'topp': topp, # Will be mapped to top_p by some providers
        'logprobs': logprobs,
        'top_logprobs': top_logprobs,
        'logit_bias': logit_bias,
        'presence_penalty': presence_penalty,
        'frequency_penalty': frequency_penalty,
        'tools': tools,
        'tool_choice': tool_choice,
        'max_tokens': max_tokens,
        'seed': seed,
        'stop': stop,
        'response_format': response_format,
        'n': n,
        'user_identifier': user_identifier
    }

    for generic_param_name, provider_param_name in params_map.items():
        if generic_param_name in available_generic_params and available_generic_params[generic_param_name] is not None:
            call_kwargs[provider_param_name] = available_generic_params[generic_param_name]
        if generic_param_name == 'prompt' and endpoint_lower == 'cohere':
             pass # Specific handling for Cohere's prompt is assumed to be within chat_with_cohere

    if call_kwargs.get(params_map.get('api_key', 'api_key')) and isinstance(call_kwargs.get(params_map.get('api_key', 'api_key')), str) and len(call_kwargs.get(params_map.get('api_key', 'api_key'))) > 8:
         logging.info(f"Debug - Chat API Call - API Key: {call_kwargs[params_map.get('api_key', 'api_key')][:4]}...{call_kwargs[params_map.get('api_key', 'api_key')][-4:]}")

    try:
        logging.debug(f"Calling handler {handler.__name__} with kwargs: { {k: (type(v) if k != params_map.get('api_key') else 'key_hidden') for k,v in call_kwargs.items()} }")
        response = handler(**call_kwargs)

        call_duration = time.time() - start_time
        log_histogram("chat_api_call_duration", call_duration, labels={"api_endpoint": endpoint_lower})
        log_counter("chat_api_call_success", labels={"api_endpoint": endpoint_lower})

        if isinstance(response, str):
             logging.debug(f"Debug - Chat API Call - Response (first 500 chars): {response[:500]}...")
        elif hasattr(response, '__iter__') and not isinstance(response, (str, bytes, dict)):
             logging.debug(f"Debug - Chat API Call - Response: Streaming Generator")
        else:
             logging.debug(f"Debug - Chat API Call - Response Type: {type(response)}")
        return response

    # --- Exception Mapping (copied from your original, ensure it's still relevant) ---
    except requests.exceptions.HTTPError as e:
        status_code = getattr(e.response, 'status_code', 500)
        error_text = getattr(e.response, 'text', str(e))
        log_message_base = f"{endpoint_lower} API call failed with status {status_code}"

        # Log safely first
        try:
            logging.error("%s. Details: %s", log_message_base, error_text[:500], exc_info=False)
        except Exception as log_e:
            logging.error(f"Error during logging HTTPError details: {log_e}")

        detail_message = f"API call to {endpoint_lower} failed with status {status_code}. Response: {error_text[:200]}"
        if status_code == 401:
            raise ChatAuthenticationError(provider=endpoint_lower,
                                          message=f"Authentication failed for {endpoint_lower}. Check API key. Detail: {error_text[:200]}")
        elif status_code == 429:
            raise ChatRateLimitError(provider=endpoint_lower,
                                     message=f"Rate limit exceeded for {endpoint_lower}. Detail: {error_text[:200]}")
        elif 400 <= status_code < 500:
            raise ChatBadRequestError(provider=endpoint_lower,
                                      message=f"Bad request to {endpoint_lower} (Status {status_code}). Detail: {error_text[:200]}")
        elif 500 <= status_code < 600:
            raise ChatProviderError(provider=endpoint_lower,
                                    message=f"Error from {endpoint_lower} server (Status {status_code}). Detail: {error_text[:200]}",
                                    status_code=status_code)
        else:
            raise ChatAPIError(provider=endpoint_lower,
                               message=f"Unexpected HTTP status {status_code} from {endpoint_lower}. Detail: {error_text[:200]}",
                               status_code=status_code)
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error connecting to {endpoint_lower}: {e}", exc_info=False)
        raise ChatProviderError(provider=endpoint_lower, message=f"Network error: {e}", status_code=504)
    except (ChatAuthenticationError, ChatRateLimitError, ChatBadRequestError, ChatConfigurationError, ChatProviderError,
            ChatAPIError) as e_chat_direct:
        # This catches cases where the handler itself has already processed an error
        # (e.g. non-HTTP error, or it decided to raise a specific Chat*Error type)
        # and raises one of our custom exceptions.
        logging.error(
            f"Handler for {endpoint_lower} directly raised: {type(e_chat_direct).__name__} - {e_chat_direct.message}",
            exc_info=True if e_chat_direct.status_code >= 500 else False)
        raise e_chat_direct  # Re-raise the specific error
    except (ValueError, TypeError, KeyError) as e:
        logging.error(f"Value/Type/Key error during chat API call setup for {endpoint_lower}: {e}", exc_info=True)
        error_type = "Configuration/Parameter Error"
        if "Unsupported API endpoint" in str(e):
            raise ChatConfigurationError(provider=endpoint_lower, message=f"Unsupported API endpoint: {endpoint_lower}")
        else:
            raise ChatBadRequestError(provider=endpoint_lower, message=f"{error_type} for {endpoint_lower}: {e}")
    except Exception as e:
        logging.exception(
            f"Unexpected internal error in chat_api_call for {endpoint_lower}: {e}")
        raise ChatAPIError(provider=endpoint_lower,
                           message=f"An unexpected internal error occurred in chat_api_call for {endpoint_lower}: {str(e)}",
                           status_code=500)


# FIXME - thing is fucking big.
# Break it down into smaller, logical async helper functions. For example:
#
#     _authenticate_request(Token)
#
#     _validate_request_payload(request_data)
#
#     _get_or_create_conversation_context(chat_db, request_data, character_card)
#
#     _load_chat_history(chat_db, conversation_id, character_card)
#
#     _prepare_llm_messages(request_data, historical_messages, character_card, template_name)
#
#     _handle_llm_call_and_response(loop, llm_call_func, request_data, chat_db, final_conversation_id, character_card)
def chat(
    message: str,
    history: List[Dict[str, Any]],
    media_content: Optional[Dict[str, str]],
    selected_parts: List[str],
    api_endpoint: str,
    api_key: Optional[str],
    custom_prompt: Optional[str],
    temperature: float,
    system_message: Optional[str] = None,
    streaming: bool = False,
    minp: Optional[float] = None,
    maxp: Optional[float] = None,
    model: Optional[str] = None,
    topp: Optional[float] = None,
    topk: Optional[int] = None,
    chatdict_entries: Optional[List[Any]] = None, # Should be List[ChatDictionary]
    max_tokens: int = 500,
    strategy: str = "sorted_evenly",
    current_image_input: Optional[Dict[str, str]] = None,
    image_history_mode: str = "tag_past",
    llm_max_tokens: Optional[int] = None,
    llm_seed: Optional[int] = None,
    llm_stop: Optional[Union[str, List[str]]] = None,
    llm_response_format: Optional[ResponseFormat] = None,
    llm_n: Optional[int] = None,
    llm_user_identifier: Optional[str] = None,
    llm_logprobs: Optional[bool] = None,
    llm_top_logprobs: Optional[int] = None,
    llm_logit_bias: Optional[Dict[str, float]] = None,
    llm_presence_penalty: Optional[float] = None,
    llm_frequency_penalty: Optional[float] = None,
    llm_tools: Optional[List[Dict[str, Any]]] = None,
    llm_tool_choice: Optional[Union[str, Dict[str, Any]]] = None
) -> Union[str, Any]: # Any for streaming generator
    """
    Orchestrates a chat interaction with an LLM, handling message processing,
    RAG, multimodal content, and chat dictionary features.

    This function prepares the `messages_payload` in OpenAI format, including
    history, current user message (with optional RAG and image), and then
    calls `chat_api_call` to get the LLM's response.

    Args:
        message: The current text message from the user.
        history: A list of previous messages in OpenAI format
                 (e.g., `[{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]`).
                 Content can be simple text or a list of multimodal parts.
        media_content: A dictionary containing RAG content (e.g., `{'summary': '...', 'transcript': '...'}`).
        selected_parts: A list of keys from `media_content` to include as RAG.
        api_endpoint: Identifier for the target LLM provider.
        api_key: API key for the provider.
        custom_prompt: An additional prompt/instruction to prepend to the user's current message.
        temperature: LLM sampling temperature.
        system_message: A system-level instruction for the LLM. Passed to `chat_api_call`.
        streaming: Whether to stream the LLM response.
        minp: Min-P sampling parameter for the LLM.
        maxp: Max-P (often Top-P) sampling parameter for the LLM.
        model: The specific LLM model to use.
        topp: Top-P (nucleus) sampling parameter for the LLM.
        topk: Top-K sampling parameter for the LLM.
        chatdict_entries: A list of `ChatDictionary` objects for keyword replacement/expansion.
        max_tokens: Max tokens for chat dictionary content processing (not LLM response).
        strategy: Strategy for applying chat dictionary entries (e.g., "sorted_evenly").
        current_image_input: An optional dictionary for the current image being sent by the user,
                             in the format `{'base64_data': '...', 'mime_type': 'image/png'}`.
        image_history_mode: How to handle images from past messages:
                            "send_all": Send all past images.
                            "send_last_user_image": Send only the last image sent by a user.
                            "tag_past": Replace past images with a textual tag (e.g., "<image: prior_history.png>").
                            "ignore_past": Do not include any past images.
        llm_max_tokens: Max tokens for the LLM to generate in its response.
        llm_seed: Seed for LLM generation.
        llm_stop: Stop sequence(s) for LLM generation.
        llm_response_format: Desired response format from LLM (e.g., JSON object).
                             Pydantic `ResponseFormat` model instance.
        llm_n: Number of LLM completion choices to generate.
        llm_user_identifier: User identifier for LLM API call.
        llm_logprobs: Whether LLM should return log probabilities.
        llm_top_logprobs: Number of top log probabilities for LLM to return.
        llm_logit_bias: Logit bias for LLM token generation.
        llm_presence_penalty: Presence penalty for LLM generation.
        llm_frequency_penalty: Frequency penalty for LLM generation.
        llm_tools: Tools for LLM function calling.
        llm_tool_choice: Tool choice for LLM function calling.

    Returns:
        The LLM's response, either as a string (non-streaming) or a generator
        (streaming). In case of an error during chat processing, a string
        containing an error message is returned.

    Raises:
        Catches internal exceptions and returns an error message string.
        Exceptions from `chat_api_call` might propagate if not handled by its own try-except blocks.
    """
    log_counter("chat_attempt_multimodal", labels={"api_endpoint": api_endpoint, "image_mode": image_history_mode})
    start_time = time.time()

    try:
        logging.info(f"Debug - Chat Function - Input Text: '{message}', Image provided: {'Yes' if current_image_input else 'No'}")
        logging.info(f"Debug - Chat Function - History length: {len(history)}, Image History Mode: {image_history_mode}")

        # Ensure selected_parts is a list
        if not isinstance(selected_parts, (list, tuple)):
            selected_parts = [selected_parts] if selected_parts else []

        # Process message with Chat Dictionary (text only for now)
        processed_text_message = message
        if chatdict_entries and message:
            processed_text_message = process_user_input(
                message, chatdict_entries, max_tokens=max_tokens, strategy=strategy
            )

        # --- Construct messages payload for the LLM API (OpenAI format) ---
        llm_messages_payload: List[Dict[str, Any]] = []

        # PHILOSOPHY:
        # `chat()` prepares the `llm_messages_payload` (user/assistant turns with multimodal content).
        # `chat()` also collects the `system_message`.
        # `chat_api_call()` receives both `llm_messages_payload` and the separate `system_message`.
        # `chat_api_call()` then dispatches these to the specific provider function (e.g., `chat_with_openai`).
        # The provider function (e.g., `chat_with_openai`) is responsible for:
        #   1. Taking the `messages` (which is `llm_messages_payload`).
        #   2. Taking the `system_message` parameter.
        #   3. If `system_message` is provided, *it* prepends `{"role": "system", "content": system_message}`
        #      to the `messages` list *if* that's how its API works (like OpenAI).
        #   4. Or, if its API takes system message as a separate top-level parameter (like Anthropic's `system_prompt`),
        #      it uses it directly there.
        # This way, `chat()` doesn't need to know the specifics of each API for system prompts


        # 2. Process History (now expecting list of OpenAI message dicts)
        last_user_image_url_from_history: Optional[str] = None

        for hist_msg_obj in history:
            role = hist_msg_obj.get("role")
            original_content = hist_msg_obj.get("content") # This can be str or list of parts

            processed_hist_content_parts = []

            if isinstance(original_content, str): # Simple text history message
                processed_hist_content_parts.append({"type": "text", "text": original_content})
            elif isinstance(original_content, list): # Already structured content
                for part in original_content:
                    if part.get("type") == "text":
                        processed_hist_content_parts.append(part)
                    elif part.get("type") == "image_url":
                        image_url_data = part.get("image_url", {}).get("url", "") # data URI
                        if image_history_mode == "send_all":
                            processed_hist_content_parts.append(part)
                            if role == "user": last_user_image_url_from_history = image_url_data
                        elif image_history_mode == "send_last_user_image" and role == "user":
                            last_user_image_url_from_history = image_url_data # Track, add later
                        elif image_history_mode == "tag_past":
                            mime_type_part = "image"
                            if image_url_data.startswith("data:image/") and ";base64," in image_url_data:
                                try: mime_type_part = image_url_data.split(';base64,')[0].split('/')[-1]
                                except: pass
                            processed_hist_content_parts.append({"type": "text", "text": f"<image: prior_history.{mime_type_part}>"})
                        # "ignore_past": do nothing, image part is skipped

            if processed_hist_content_parts: # Add if content remains
                llm_messages_payload.append({"role": role, "content": processed_hist_content_parts})

        # Handle "send_last_user_image" - append it to the last user message in payload if applicable
        if image_history_mode == "send_last_user_image" and last_user_image_url_from_history:
            appended_to_last = False
            for i in range(len(llm_messages_payload) -1, -1, -1): # Iterate backwards
                if llm_messages_payload[i]["role"] == "user":
                    # Ensure content is a list
                    if not isinstance(llm_messages_payload[i]["content"], list):
                        llm_messages_payload[i]["content"] = [{"type": "text", "text": str(llm_messages_payload[i]["content"])}]

                    # Avoid duplicates if already processed (e.g., if history was already "send_all" style)
                    is_duplicate = any(p.get("type") == "image_url" and p.get("image_url", {}).get("url") == last_user_image_url_from_history for p in llm_messages_payload[i]["content"])
                    if not is_duplicate:
                        llm_messages_payload[i]["content"].append({"type": "image_url", "image_url": {"url": last_user_image_url_from_history}})
                    appended_to_last = True
                    break
            if not appended_to_last: # No user message in history, or image already there
                 logging.debug(f"Could not append last_user_image_from_history, no suitable prior user message or already present. Image: {last_user_image_url_from_history[:60]}...")


        # 3. Add RAG Content (prepended to current user's text)
        rag_text_prefix = ""
        if media_content and selected_parts:
            rag_text_prefix = "\n\n".join(
                [f"{part.capitalize()}: {media_content.get(part, '')}" for part in selected_parts if media_content.get(part)]
            ).strip()
            if rag_text_prefix:
                rag_text_prefix += "\n\n---\n\n"

        # 4. Construct Current User Message (text + optional new image)
        current_user_content_parts: List[Dict[str, Any]] = []

        # Combine RAG, custom_prompt (if it's for current turn's text), and processed_text_message
        # Deciding where `custom_prompt` goes: if it's a direct instruction for *this* turn,
        # it should be part of the user's text. If it's more like a persona or ongoing rule,
        # it's better in `system_message`. Let's assume it's for this turn.
        final_text_for_current_message = processed_text_message
        if custom_prompt: # Prepend custom_prompt if it exists
            final_text_for_current_message = f"{custom_prompt}\n\n{final_text_for_current_message}"

        final_text_for_current_message = f"{rag_text_prefix}{final_text_for_current_message}".strip()

        if final_text_for_current_message:
            current_user_content_parts.append({"type": "text", "text": final_text_for_current_message})

        if current_image_input and current_image_input.get('base64_data') and current_image_input.get('mime_type'):
            image_url = f"data:{current_image_input['mime_type']};base64,{current_image_input['base64_data']}"
            current_user_content_parts.append({"type": "image_url", "image_url": {"url": image_url}})

        if not current_user_content_parts: # Should only happen if message, custom_prompt, RAG, and image are all empty/None
             logging.warning("Current user message has no text or image content parts. Sending a placeholder.")
             current_user_content_parts.append({"type": "text", "text": "(No user input for this turn)"})

        llm_messages_payload.append({"role": "user", "content": current_user_content_parts})

        # Temperature and other LLM params
        temperature_float = 0.7
        try: temperature_float = float(temperature) if temperature is not None else 0.7
        except ValueError: logging.warning(f"Invalid temperature '{temperature}', using 0.7.")

        logging.debug(f"Debug - Chat Function - Final LLM Payload (structure, image data truncated):")
        for i, msg_p in enumerate(llm_messages_payload):
            content_log = []
            if isinstance(msg_p.get("content"), list):
                for part_idx, part_c in enumerate(msg_p["content"]):
                    if part_c.get("type") == "text": content_log.append(f"text: '{part_c['text'][:30]}...'")
                    elif part_c.get("type") == "image_url": content_log.append(f"image: '{part_c['image_url']['url'][:40]}...'")
            logging.debug(f"  Msg {i}: Role: {msg_p['role']}, Content: [{', '.join(content_log)}]")

        logging.debug(f"Debug - Chat Function - Temperature: {temperature}")
        logging.debug(f"Debug - Chat Function - API Key: {api_key[:10]}")
        logging.debug(f"Debug - Chat Function - Prompt: {custom_prompt}")

        # --- Call the LLM via the updated chat_api_call ---
        response = chat_api_call(
            api_endpoint=api_endpoint,
            api_key=api_key,
            messages_payload=llm_messages_payload,
            temp=temperature_float,
            system_message=system_message,
            streaming=streaming,
            minp=minp, maxp=maxp, model=model, topp=topp, topk=topk,
            # Pass through new params from ChatCompletionRequest
            max_tokens=llm_max_tokens,
            seed=llm_seed,
            stop=llm_stop,
            response_format=llm_response_format.model_dump() if llm_response_format else None,
            n=llm_n,
            user_identifier=llm_user_identifier,
            logprobs=llm_logprobs,
            top_logprobs=llm_top_logprobs,
            logit_bias=llm_logit_bias,
            presence_penalty=llm_presence_penalty,
            frequency_penalty=llm_frequency_penalty,
            tools=llm_tools,
            tool_choice=llm_tool_choice
        )

        if streaming:
            logging.debug("Chat Function - Response: Streaming Generator")
            return response
        else:
            chat_duration = time.time() - start_time
            log_histogram("chat_duration_multimodal", chat_duration, labels={"api_endpoint": api_endpoint})
            log_counter("chat_success_multimodal", labels={"api_endpoint": api_endpoint})
            logging.debug(f"Chat Function - Response (first 500 chars): {str(response)[:500]}")

            loaded_config_data = load_and_log_configs()
            post_gen_replacement_config = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement')
            if post_gen_replacement_config and isinstance(response, str):
                post_gen_replacement_dict_path = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement_dict')
                if post_gen_replacement_dict_path and os.path.exists(post_gen_replacement_dict_path):
                    try:
                        parsed_dict_entries = parse_user_dict_markdown_file(post_gen_replacement_dict_path)
                        if parsed_dict_entries:
                            post_gen_chat_dict_objects = [
                                ChatDictionary(key=k, content=str(v)) for k, v in parsed_dict_entries.items()
                            ]
                            if post_gen_chat_dict_objects:
                                response = process_user_input(response, post_gen_chat_dict_objects)
                                # The original warning log can be removed or changed to a debug log if successfully applied.
                                logging.debug(
                                    f"Response after post-gen replacement (first 500 chars): {str(response)[:500]}")
                            else:
                                logging.debug("Post-gen dictionary parsed but resulted in no ChatDictionary objects.")
                        else:
                            logging.debug(
                                f"Post-gen replacement dictionary at {post_gen_replacement_dict_path} was empty or failed to parse.")

                        logging.debug(f"Response after post-gen replacement (first 500 chars): {str(response)[:500]}")
                    except Exception as e_post_gen:
                        logging.error(f"Error during post-generation replacement: {e_post_gen}", exc_info=True)
                else:
                    logging.warning("Post-gen replacement enabled but dict file not found/configured.")
            return response

    except Exception as e:
        log_counter("chat_error_multimodal", labels={"api_endpoint": api_endpoint, "error": str(e)})
        logging.error(f"Error in multimodal chat function: {str(e)}", exc_info=True)
        # Consider if the error format should change from just a string
        return f"An error occurred in the chat function: {str(e)}"


def save_chat_history_to_db_wrapper(
    db: CharactersRAGDB,
    chatbot_history: List[Dict[str, Any]], # CHANGED: Expects List of OpenAI message objects
    conversation_id: Optional[str],
    # Renamed for clarity: these are for identifying the character for association,
    # not for the content of the messages themselves.
    media_content_for_char_assoc: Optional[Dict[str, Any]],
    media_name_for_char_assoc: Optional[str] = None,
    character_name_for_chat: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Saves or updates a chat conversation in the database.

    This function handles associating the chat with a character (either specified,
    derived from media, or a default character), creating a new conversation entry
    if `conversation_id` is None, or updating an existing conversation by
    soft-deleting old messages and adding new ones.

    The `chatbot_history` is expected in OpenAI's message format: a list of
    dictionaries, each with 'role' and 'content' keys. Multimodal content
    (text and images as base64 data URIs) within the 'content' field is supported.

    Args:
        db: An instance of `CharactersRAGDB` for database operations.
        chatbot_history: The chat history as a list of OpenAI message objects.
                         Each object is a dict: `{'role': str, 'content': Union[str, List[Dict]]}`.
                         Content lists support `{'type': 'text', 'text': str}` and
                         `{'type': 'image_url', 'image_url': {'url': 'data:image/...;base64,...'}}`.
        conversation_id: The ID of an existing conversation to update. If None,
                         a new conversation is created.
        media_content_for_char_assoc: Optional dictionary containing media details.
                                      Used to derive a character name for association
                                      if `character_name_for_chat` or `media_name_for_char_assoc`
                                      are not provided. Expected to have a 'content' key
                                      which might be a JSON string or dict with a 'title'.
        media_name_for_char_assoc: Optional name of media to associate with a character.
                                   Used if `character_name_for_chat` is not provided.
        character_name_for_chat: Optional name of the character for this chat.
                                 If provided, the chat is associated with this character.

    Returns:
        A tuple containing:
        - `Optional[str]`: The conversation ID (new or existing). None on critical failure
                           to create a conversation entry.
        - `str`: A status message indicating success or failure.
    """
    log_counter("save_chat_history_to_db_attempt")
    start_time = time.time()
    logging.info(f"Saving chat history (OpenAI format). Conversation ID: {conversation_id}, Character: {character_name_for_chat}, Num messages: {len(chatbot_history)}")

    try:
        # The DB connection is managed by the CharactersRAGDB instance (`db`)
        # No need for direct `get_db_connection` or manual sqlite3 corruption checks here.
        # The `db` instance methods will raise exceptions if issues occur.

        associated_character_id: Optional[int] = None
        final_character_name_for_title = "Unknown Character" # For conversation title

        # --- Character Association Logic (largely same as your provided version) ---
        char_lookup_name = character_name_for_chat
        if not char_lookup_name and media_name_for_char_assoc:
            char_lookup_name = media_name_for_char_assoc

        # Fallback to media_content_for_char_assoc to derive char_lookup_name if others are None
        if not char_lookup_name and media_content_for_char_assoc:
            content_details = media_content_for_char_assoc.get('content')
            if isinstance(content_details, str):
                try: content_details = json.loads(content_details)
                except json.JSONDecodeError: content_details = {}
            if isinstance(content_details, dict):
                char_lookup_name = content_details.get('title')

        if char_lookup_name:
            try:
                character = db.get_character_card_by_name(char_lookup_name)
                if character:
                    associated_character_id = character['id']
                    final_character_name_for_title = character['name']
                    logging.info(f"Chat will be associated with specific character '{final_character_name_for_title}' (ID: {associated_character_id}).")
                else:
                    logging.error(f"Intended specific character '{char_lookup_name}' not found in DB. Chat save aborted.")
                    return conversation_id, f"Error: Specific character '{char_lookup_name}' intended for this chat was not found. Cannot save chat."
            except CharactersRAGDBError as e:
                logging.error(f"DB error looking up specific character '{char_lookup_name}': {e}")
                return conversation_id, f"DB error finding specific character: {e}"
        else:
            logging.info("No specific character name for chat. Using Default Character.")
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
        # --- End Character Association ---

        current_conversation_id = conversation_id
        is_new_conversation = not current_conversation_id

        # --- Create or Prepare Conversation ---
        if is_new_conversation:
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
                logging.info(f"Created new conv ID: {current_conversation_id} for char ID: {associated_character_id} ('{final_character_name_for_title}')")
            except (InputError, ConflictError, CharactersRAGDBError) as e:
                logging.error(f"Error creating new conversation: {e}", exc_info=True)
                return None, f"Error creating conversation: {e}"
        else: # Resaving existing conversation
            logging.info(f"Resaving history for existing conv ID: {current_conversation_id}. Char context ID: {associated_character_id} ('{final_character_name_for_title}')")
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
            except (InputError, ConflictError, CharactersRAGDBError) as e:
                logging.error(f"Error preparing existing conversation {current_conversation_id} for resave: {e}", exc_info=True)
                return current_conversation_id, f"Error during resave prep: {e}"
        # --- End Create or Prepare Conversation ---

        # --- Save Messages (Handles new OpenAI format) ---
        try:
            # Ensure transaction wraps message saving, especially for new conversations or full resaves.
            # For resaves, the transaction is already started above.
            with db.transaction() if is_new_conversation else db.transaction(): # No-op if already in transaction for resave
                message_save_count = 0
                for i, message_obj in enumerate(chatbot_history):
                    sender = message_obj.get("role")
                    if not sender or sender == "system": # Don't save system prompts as messages
                        logging.debug(f"Skipping message with role '{sender}' at index {i}")
                        continue

                    text_content_parts = []
                    image_data_bytes: Optional[bytes] = None
                    image_mime_type_str: Optional[str] = None

                    content_data = message_obj.get("content")

                    if isinstance(content_data, str): # Simple text content (e.g., from older history or some assistant responses)
                        text_content_parts.append(content_data)
                    elif isinstance(content_data, list): # OpenAI multimodal content list
                        for part in content_data:
                            part_type = part.get("type")
                            if part_type == "text":
                                text_content_parts.append(part.get("text", ""))
                            elif part_type == "image_url":
                                image_url_dict = part.get("image_url", {})
                                url_str = image_url_dict.get("url", "")
                                if url_str.startswith("data:") and ";base64," in url_str:
                                    try:
                                        header, b64_data = url_str.split(";base64,", 1)
                                        image_mime_type_str = header.split("data:", 1)[1] if "data:" in header else None
                                        if image_mime_type_str: # Ensure mime type was found
                                            image_data_bytes = base64.b64decode(b64_data)
                                            logging.debug(f"Decoded image for saving (MIME: {image_mime_type_str}, Size: {len(image_data_bytes) if image_data_bytes else 0}) for msg {i} in conv {current_conversation_id}")
                                        else:
                                            logging.warning(f"Could not parse MIME type from data URI: {url_str[:60]}...")
                                            text_content_parts.append("<Error: Malformed image data URI in history>")
                                    except Exception as e_b64:
                                        logging.error(f"Error decoding base64 image from history for msg {i} in conv {current_conversation_id}: {e_b64}")
                                        text_content_parts.append("<Error: Failed to decode image data from history>")
                                else:
                                    # If it's a non-data URL, store it as text.
                                    logging.debug(f"Storing non-data image URL as text: {url_str}")
                                    text_content_parts.append(f"<Image URL: {url_str}>")
                    else:
                        logging.warning(f"Unsupported message content type at index {i}: {type(content_data)}")
                        text_content_parts.append(f"<Unsupported content type: {type(content_data)}>")

                    final_text_content = "\n".join(text_content_parts).strip()

                    # A message must have either text or an image to be saved.
                    if not final_text_content and not image_data_bytes:
                        logging.warning(f"Skipping empty message (no text or decodable image) at index {i} for conv {current_conversation_id}")
                        continue

                    db.add_message({
                        'conversation_id': current_conversation_id,
                        'sender': sender, # 'user' or 'assistant'
                        'content': final_text_content,
                        'image_data': image_data_bytes,
                        'image_mime_type': image_mime_type_str,
                        'client_id': db.client_id
                        # timestamp, version, ranking, etc., handled by db.add_message
                    })
                    message_save_count +=1
                logging.info(f"Successfully saved {message_save_count} messages to conversation {current_conversation_id}.")

                # If resaving (not a new conversation), update conversation's last_modified and version
                if not is_new_conversation:
                    conv_details_for_update = db.get_conversation_by_id(current_conversation_id)
                    if conv_details_for_update:
                        db.update_conversation(
                            current_conversation_id,
                            {'title': conv_details_for_update.get('title')}, # Keep title, just bump version/timestamp
                            conv_details_for_update['version']
                        )
                    else:
                        logging.error(f"Conversation {current_conversation_id} disappeared before final metadata update during resave.")


        except (InputError, ConflictError, CharactersRAGDBError) as e:
            logging.error(f"Error saving messages to conversation {current_conversation_id}: {e}", exc_info=True)
            return current_conversation_id, f"Error saving messages: {e}"
        # --- End Save Messages ---

        save_duration = time.time() - start_time
        log_histogram("save_chat_history_to_db_duration", save_duration)
        log_counter("save_chat_history_to_db_success")

        return current_conversation_id, "Chat history saved successfully!"

    except Exception as e:
        log_counter("save_chat_history_to_db_error", labels={"error": str(e)})
        error_message = f"Failed to save chat history due to an unexpected error: {str(e)}"
        logging.error(error_message, exc_info=True)
        return conversation_id, error_message


# FIXME - turn into export function
def save_chat_history(
    history: List[Union[Tuple[Optional[str], Optional[str]], Dict[str, Any]]],
    conversation_id: Optional[str],
    media_content: Optional[Dict[str, Any]],
    db_instance: Optional[CharactersRAGDB] = None # Added for generate_chat_history_content
) -> Optional[str]:
    """
    Saves chat history to a uniquely named JSON file in a temporary directory.

    FIXME: This function is marked to be potentially turned into a more generic
    export function. Currently, it generates content using
    `generate_chat_history_content` and saves it locally.

    Args:
        history: The chat history. Can be a list of (user_msg, bot_msg) tuples
                 or a list of OpenAI message dicts.
        conversation_id: The ID of the conversation, used for naming the file
                         and potentially fetching details via `db_instance`.
        media_content: Optional dictionary containing media details, used by
                       `generate_chat_history_content` to derive a conversation name
                       if `conversation_id` doesn't yield one.
        db_instance: Optional `CharactersRAGDB` instance passed to
                     `generate_chat_history_content` to fetch conversation title.

    Returns:
        The absolute path to the saved JSON file, or None if an error occurred.
    """
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
    Retrieves the title of a conversation from the database.

    Args:
        conversation_id: The ID of the conversation.
        db_instance: An optional instance of `CharactersRAGDB` to use for DB lookup.

    Returns:
        The conversation title as a string if found, otherwise None.
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


def generate_chat_history_content(
    history: List[Union[Tuple[Optional[str], Optional[str]], Dict[str, Any]]],
    conversation_id: Optional[str],
    media_content: Optional[Dict[str, Any]],
    db_instance: Optional[CharactersRAGDB] = None
) -> Tuple[str, str]:
    """
    Generates JSON content representing the chat history and determines a conversation name.

    The conversation name is fetched from the database using `conversation_id` if
    `db_instance` is provided. Otherwise, it's derived from `media_content` or
    a timestamp. The history is formatted into a list of role-content dictionaries.

    Args:
        history: The chat history. Can be a list of (user_msg, bot_msg) tuples
                 or a list of OpenAI message dicts (`{'role': ..., 'content': ...}`).
        conversation_id: Optional ID of the conversation.
        media_content: Optional dictionary with media details, used for deriving
                       conversation name as a fallback.
        db_instance: Optional `CharactersRAGDB` instance for fetching conversation title.

    Returns:
        A tuple containing:
        - `str`: JSON string of the chat data.
        - `str`: The derived or fetched conversation name.
    """
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

def extract_media_name(media_content: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Extracts a name or title from a media content dictionary.

    It attempts to find a name by looking at common keys like 'title', 'name',
    'media_title', 'webpage_title' within the `media_content` dictionary itself
    or within a nested 'content' field (which might be a JSON string or a sub-dictionary).

    Args:
        media_content: A dictionary potentially containing media metadata.

    Returns:
        The extracted name as a string if found, otherwise None.
    """
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

# FIXME
# update_chat_content Note Parsing:
#     Issue: raw_note_content_field can be a plain string or a JSON string. This dual nature can be brittle.
#     Improvement: Enforce a consistent structure for notes.content in the database if it's meant to hold structured data. If it's always JSON, then json.loads can be used directly (with error handling). If it can be either, the current logic is a necessary workaround but adds complexity.
def update_chat_content(
        selected_item: Optional[str],
        use_content: bool,
        use_summary: bool,
        use_prompt: bool,
        item_mapping: Dict[str, str],
        db_instance: CharactersRAGDB
) -> Tuple[Dict[str, str], List[str]]:
    """
    Fetches content from a database 'note' to be used as RAG context in a chat.

    The function retrieves a note by ID (derived from `selected_item` via
    `item_mapping`). It then extracts parts like 'content', 'summary', and 'prompt'
    from the note's data based on the boolean flags `use_content`, `use_summary`,
    and `use_prompt`.

    The note's 'content' field in the database can be a plain string or a JSON string
    containing structured data (e.g., `{"content": "...", "summary": "..."}`).
    This function attempts to parse it accordingly.

    FIXME: The dual nature of `note_data.content` (plain string or JSON string)
    can be brittle. Enforcing a consistent structure for notes.content in the
    database (e.g., always JSON if structured data is intended) would be an
    improvement.

    Args:
        selected_item: The display name of the item selected by the user.
        use_content: Boolean flag to include the main 'content' part.
        use_summary: Boolean flag to include the 'summary' part.
        use_prompt: Boolean flag to include the 'prompt' part.
        item_mapping: A dictionary mapping display names (like `selected_item`)
                      to note IDs (UUID strings).
        db_instance: An instance of `CharactersRAGDB` for database operations.

    Returns:
        A tuple containing:
        - `Dict[str, str]`: A dictionary where keys are part names (e.g., "content",
                            "summary") and values are their string content. This is
                            intended for the `media_content` argument of the `chat` function.
        - `List[str]`: A list of selected part names (e.g., ["content", "summary"]).
    """
    log_counter("update_chat_content_attempt")
    start_time = time.time()
    logging.debug(f"Debug - Update Chat Content - Selected Item: {selected_item}")

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

def parse_user_dict_markdown_file(file_path: str) -> Dict[str, str]:
    """
    Parses a user-defined dictionary from a markdown-like file.

    The file format supports:
    - Single-line entries: `key: value`
    - Multi-line entries:
      ```
      key: |
      This is a
      multi-line value.
      ---@@@---
      ```
    Keys and single-line values are stripped of leading/trailing whitespace.
    Multi-line values preserve internal whitespace and newlines until the
    terminator `---@@@---` is encountered on its own line (stripped).
    Lines starting with a key pattern override previous multi-line contexts.

    Args:
        file_path: The path to the markdown dictionary file.

    Returns:
        A dictionary where keys are strings and values are the corresponding
        content strings. Returns an empty dictionary if the file is not found
        or an error occurs during parsing.
    """
    logger.debug(f"Parsing user dictionary file: {file_path}")
    replacement_dict: Dict[str, str] = {}
    current_key: Optional[str] = None
    current_value_lines: List[str] = []

    new_key_pattern = re.compile(r'^\s*([^:\n]+?)\s*:(.*)$')
    termination_pattern = re.compile(r'^\s*---@@@---\s*$')

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line_content_original in enumerate(file, 1):
                line_for_logic = line_content_original.strip()  # Use for terminator/blank checks

                if termination_pattern.match(line_for_logic):
                    if current_key:
                        replacement_dict[current_key] = '\n'.join(current_value_lines).strip()  # Final strip after join
                        logger.trace(f"L{line_number}: Terminated multi-line for '{current_key}'.")
                        current_key, current_value_lines = None, []
                    continue

                new_key_match = new_key_pattern.match(line_content_original)  # Match on original line

                if new_key_match:
                    if current_key:  # Finalize previous multi-line key if one was active
                        replacement_dict[current_key] = '\n'.join(current_value_lines).strip()
                        logger.trace(f"L{line_number}: New key, finalized old '{current_key}'.")

                    potential_new_key = new_key_match.group(1).strip()
                    value_part_after_colon = new_key_match.group(2).strip()  # Strip this part

                    if value_part_after_colon == '|':
                        current_key = potential_new_key
                        current_value_lines = []
                        logger.trace(f"L{line_number}: Starting multi-line for '{current_key}'.")
                    else:
                        replacement_dict[potential_new_key] = value_part_after_colon
                        logger.trace(f"L{line_number}: Parsed single-line key '{potential_new_key}'.")
                        current_key, current_value_lines = None, []  # Reset
                    continue

                if current_key:
                    # For multi-line content, append the line with only its trailing newline removed.
                    # Leading/internal whitespace should be preserved until the final .strip() after .join().
                    current_value_lines.append(line_content_original.rstrip('\n\r'))

            if current_key:  # Finalize any pending multi-line value at EOF
                replacement_dict[current_key] = '\n'.join(current_value_lines).strip()
                logger.debug(f"Finalizing last multi-line key '{current_key}' at EOF.")

    except FileNotFoundError:  # ...
        logger.error(f"Chat dictionary file not found: {file_path}")
        return {}
    except Exception as e:  # ...
        logger.error(f"Error parsing chat dictionary file {file_path}: {e}", exc_info=True)
        return {}

    logger.debug(f"Finished parsing chat dictionary. Keys: {list(replacement_dict.keys())}")
    return replacement_dict


class ChatDictionary:
    """
    Represents an entry in the chat dictionary for keyword replacement or expansion.

    Attributes:
        key_raw (str): The original key string (can be plain text or /regex/).
        key (Union[str, re.Pattern]): The compiled key, either a string or a regex pattern.
        content (str): The content to replace the key with or to use in expansion.
        probability (int): The probability (0-100) of this entry being triggered.
        group (Optional[str]): An optional group name for scoring and selection logic.
        timed_effects (Dict[str, int]): Dictionary for timed effects:
            - "sticky" (int): Duration in seconds the effect remains active (0 if not sticky). (Currently not directly used in provided pipeline)
            - "cooldown" (int): Duration in seconds before this entry can be triggered again.
            - "delay" (int): Duration in seconds to wait after trigger conditions met before applying.
        last_triggered (Optional[datetime]): Timestamp of when this entry was last triggered.
        max_replacements (int): Maximum number of times this entry can replace its key in a single input.
    """
    def __init__(self, key: str, content: str, probability: int = 100, group: Optional[str] = None,
                 timed_effects: Optional[Dict[str, int]] = None, max_replacements: int = 1):
        """
        Initializes a ChatDictionary entry.

        Args:
            key: The key to match (plain text or /regex/).
            content: The replacement content.
            probability: Probability of triggering (0-100). Defaults to 100.
            group: Optional group name. Defaults to None.
            timed_effects: Optional dictionary for timed effects.
                           Defaults to `{"sticky": 0, "cooldown": 0, "delay": 0}`.
            max_replacements: Max number of replacements for this entry per input. Defaults to 1.
        """
        self.key_raw = key
        self.key = self.compile_key(key)
        self.content = content
        self.probability = probability
        self.group = group
        self.timed_effects = timed_effects or {"sticky": 0, "cooldown": 0, "delay": 0}
        self.last_triggered: Optional[datetime] = None
        self.max_replacements = max_replacements

    @staticmethod
    def compile_key(key: str) -> Union[str, re.Pattern]:
        """
        Compiles a key string into a regex pattern if it's wrapped in '/'.

        Args:
            key: The key string. If it starts and ends with '/', it's treated as a regex.

        Returns:
            A compiled `re.Pattern` if the key is a regex, otherwise the original string.
        """
        if key.startswith("/") and key.endswith("/"):
            return re.compile(key[1:-1], re.IGNORECASE)
        return key

    def matches(self, text: str) -> bool:
        """
        Checks if this entry's key matches the given text.

        Args:
            text: The text to check for a match.

        Returns:
            True if a match is found, False otherwise.
        """
        if isinstance(self.key, re.Pattern):
            return self.key.search(text) is not None
        # For plain string keys, `process_user_input` uses `match_whole_words`
        # which applies \b for whole word matching. This method is a simpler check.
        # Consider if this direct `in` check is sufficient or if it should also use whole-word.
        # For now, it reflects the direct check before `match_whole_words` in the pipeline.
        return self.key in text


def apply_strategy(entries: List[ChatDictionary], strategy: str = "sorted_evenly") -> List[ChatDictionary]:
    """
    Sorts chat dictionary entries based on a given strategy.

    Strategies:
    - "sorted_evenly": Sorts entries alphabetically by their raw key.
    - "character_lore_first": Sorts "character" group entries first, then others, then by key.
    - "global_lore_first": Sorts "global" group entries first, then others, then by key.

    Args:
        entries: A list of `ChatDictionary` objects.
        strategy: The sorting strategy name. Defaults to "sorted_evenly".

    Returns:
        A new list of sorted `ChatDictionary` objects.
    """
    logging.debug(f"Applying strategy: {strategy}")
    if strategy == "sorted_evenly":
        return sorted(entries, key=lambda e: str(e.key_raw)) # Ensure key_raw is string for sort
    elif strategy == "character_lore_first":
        return sorted(entries, key=lambda e: (e.group != "character", str(e.key_raw)))
    elif strategy == "global_lore_first":
        return sorted(entries, key=lambda e: (e.group != "global", str(e.key_raw)))
    return entries # Fallback if strategy not recognized


def filter_by_probability(entries: List[ChatDictionary]) -> List[ChatDictionary]:
    """
    Filters a list of ChatDictionary entries based on their probability.

    Each entry has a `probability` attribute (0-100). This function
    includes an entry if a random number between 1 and 100 is less than
    or equal to its probability.

    Args:
        entries: A list of `ChatDictionary` objects.

    Returns:
        A new list containing only the entries that passed the probability check.
    """
    return [entry for entry in entries if random.randint(1, 100) <= entry.probability]


# Group Scoring - Situation where multiple entries are triggered in different groups in a single message
def group_scoring(entries: List[ChatDictionary]) -> List[ChatDictionary]:
    """
    Selects entries based on group scoring rules.

    - Entries without a group (group is None) are all included if matched.
    - For entries within the same named group, only the "best" entry (currently
      defined as the one with the longest raw key string) is selected from that group.

    Args:
        entries: A list of `ChatDictionary` objects that have already matched.

    Returns:
        A new list of selected `ChatDictionary` objects after group scoring.
    """
    logging.debug(f"Group scoring for {len(entries)} entries")
    if not entries: return []

    grouped_entries: Dict[Optional[str], List[ChatDictionary]] = {}
    for entry in entries:
        grouped_entries.setdefault(entry.group, []).append(entry)

    selected_entries: List[ChatDictionary] = []
    for group_name, group_entries_list in grouped_entries.items():
        if not group_entries_list: continue

        if group_name is None:  # For the default group (None)
            # Add all entries instead of just the "best" one.
            # This allows multiple ungrouped keywords to be processed if they all match.
            selected_entries.extend(group_entries_list)
        else:
            # For named groups, keep the original behavior of selecting the best.
            best_entry_in_group = max(group_entries_list, key=lambda e: len(str(e.key_raw)) if e.key_raw else 0)
            selected_entries.append(best_entry_in_group)

    logging.debug(f"Selected {len(selected_entries)} entries after group scoring.")
    # Ensure the order is somewhat predictable if multiple entries come from the None group
    # The apply_strategy step later will sort them.
    return selected_entries


def apply_timed_effects(entry: ChatDictionary, current_time: datetime) -> bool:
    """
    Applies timed effects (delay, cooldown) to a ChatDictionary entry.

    - Delay: If `entry.timed_effects["delay"]` is positive, the entry is
      invalid if the time since `last_triggered` (or from epoch if never triggered)
      is less than the delay.
    - Cooldown: If `entry.timed_effects["cooldown"]` is positive, the entry is
      invalid if it was `last_triggered` and the time since then is less than
      the cooldown.

    If the entry is considered valid after checks, its `last_triggered` time is
    updated to `current_time`.

    Args:
        entry: The `ChatDictionary` entry to check.
        current_time: The current `datetime` object.

    Returns:
        True if the entry is valid after timed effect checks, False otherwise.
    """
    logging.debug(f"Applying timed effects for entry: {entry.key_raw}") # Use key_raw for logging
    if entry.timed_effects["delay"] > 0:
        # If never triggered, assume it's valid for delay unless delay is from program start
        # For simplicity, if last_triggered is None, it passes delay check.
        # A more complex interpretation might involve first_seen time.
        # Current logic: delay is from last trigger. If never triggered, passes delay.
        if entry.last_triggered is not None and \
           current_time - entry.last_triggered < timedelta(seconds=entry.timed_effects["delay"]):
            logging.debug(f"Entry {entry.key_raw} delayed.")
            return False
    if entry.timed_effects["cooldown"] > 0:
        if entry.last_triggered and \
           current_time - entry.last_triggered < timedelta(seconds=entry.timed_effects["cooldown"]):
            logging.debug(f"Entry {entry.key_raw} on cooldown.")
            return False

    # If checks pass, update last_triggered (conceptually, this happens if it *would* be used)
    # The actual update of last_triggered for active use is often done after selection.
    # Here, we return true, and `process_user_input` will update `last_triggered` for used entries.
    # For this function's purpose (filtering), we don't update here but assume it would be if selected.
    return True


def calculate_token_usage(entries: List[ChatDictionary]) -> int:
    """
    Calculates the approximate total token usage for a list of ChatDictionary entries.

    Token usage for each entry is estimated by splitting its `content` by spaces.

    Args:
        entries: A list of `ChatDictionary` objects.

    Returns:
        The total approximate token count for all entries' content.
    """
    logging.debug(f"Calculating token usage for {len(entries)} entries")
    return sum(len(entry.content.split()) for entry in entries)


def enforce_token_budget(entries: List[ChatDictionary], max_tokens: int) -> List[ChatDictionary]:
    """
    Filters a list of ChatDictionary entries to fit within a maximum token budget.

    Entries are added to the returned list one by one, accumulating their
    token count, until the `max_tokens` budget is reached. Entries are processed
    in their given order.

    Args:
        entries: A list of `ChatDictionary` objects, typically already sorted by priority/strategy.
        max_tokens: The maximum allowed total tokens for the content of selected entries.

    Returns:
        A new list of `ChatDictionary` objects whose combined content token count
        does not exceed `max_tokens`.
    """
    total_tokens = 0
    valid_entries = []
    for entry in entries:
        tokens = len(entry.content.split())
        if total_tokens + tokens <= max_tokens:
            valid_entries.append(entry)
            total_tokens += tokens
        else:
            logging.debug(f"Token budget exceeded with entry {entry.key_raw}. Total tokens: {total_tokens + tokens}, Max: {max_tokens}")
            break # Stop adding entries once budget is full
    return valid_entries


def match_whole_words(entries: List[ChatDictionary], text: str) -> List[ChatDictionary]:
    """
    Filters entries by matching their keys against text, ensuring whole word matches for string keys.

    - If an entry's key is a compiled regex, `re.search()` is used.
    - If an entry's key is a plain string, it's matched as a whole word
      (using `\\b` word boundaries) case-insensitively.

    Args:
        entries: A list of `ChatDictionary` objects.
        text: The input text to match against.

    Returns:
        A new list of `ChatDictionary` objects that matched the text.
    """
    matched_entries = []
    for entry in entries:
        if isinstance(entry.key, re.Pattern): # Compiled regex
            if entry.key.search(text):
                matched_entries.append(entry)
                logging.debug(f"Chat Dictionary: Matched regex entry: {entry.key.pattern}")
        elif isinstance(entry.key, str): # Plain string key
            # Ensure whole word match for plain strings, case-insensitive
            if re.search(rf'\b{re.escape(entry.key)}\b', text, re.IGNORECASE):
                matched_entries.append(entry)
                logging.debug(f"Chat Dictionary: Matched string entry: {entry.key}")
    return matched_entries

class TokenBudgetExceededWarning(Warning):
    """Custom warning for token budget issues"""
    pass


def alert_token_budget_exceeded(entries: List[ChatDictionary], max_tokens: int):
    """
    Checks if the token usage of selected entries exceeds the budget and issues a warning.

    Args:
        entries: A list of `ChatDictionary` objects selected for use.
        max_tokens: The maximum allowed token budget.
    """
    token_usage = calculate_token_usage(entries)
    logging.debug(f"Token usage: {token_usage}, Max tokens: {max_tokens}")
    if token_usage > max_tokens:
        warning_msg = f"Alert: Token budget exceeded for chat dictionary! Used: {token_usage}, Allowed: {max_tokens}"
        warnings.warn(TokenBudgetExceededWarning(warning_msg))
        print(warning_msg)

def apply_replacement_once(text: str, entry: ChatDictionary) -> Tuple[str, int]:
    """
    Replaces the first occurrence of an entry's key in text with its content.

    - If `entry.key` is a regex pattern, `re.subn()` with `count=1` is used.
    - If `entry.key` is a string, a case-insensitive whole-word regex is
      constructed and used with `re.subn()` with `count=1`.

    Args:
        text: The input text where replacement should occur.
        entry: The `ChatDictionary` entry providing the key and content.

    Returns:
        A tuple containing:
        - `str`: The text after the first replacement (or original text if no match).
        - `int`: The number of replacements made (0 or 1).
    """
    logging.debug(f"Applying replacement for entry: {entry.key_raw} with content: {entry.content[:50]}... in text: {text[:50]}...")
    if isinstance(entry.key, re.Pattern):
        replaced_text, replaced_count = entry.key.subn(entry.content, text, count=1)
    else: # Plain string key
        pattern = re.compile(rf'\b{re.escape(str(entry.key))}\b', re.IGNORECASE) # Ensure entry.key is str
        replaced_text, replaced_count = pattern.subn(entry.content, text, count=1)
    return replaced_text, replaced_count


def process_user_input(
    user_input: str,
    entries: List[ChatDictionary],
    max_tokens: int = 5000,
    strategy: str = "sorted_evenly"
) -> str:
    """
    Processes user input by applying a series of chat dictionary transformations.

    The pipeline includes:
    1. Matching entries against the input text (regex and whole-word string matching).
    2. Applying group scoring to select among matched entries from the same group.
    3. Filtering entries by probability.
    4. Applying timed effects (delay, cooldown).
    5. Enforcing a token budget for the content of selected entries.
    6. Alerting if the token budget is exceeded by the (potentially filtered) entries.
    7. Sorting the final set of entries based on the chosen strategy.
    8. Applying replacements: each selected entry replaces its key in the user input
       (respecting `entry.max_replacements`).

    If any step in the pipeline encounters a significant error, it may log the error
    and continue with a potentially reduced set of entries or, in critical cases,
    return the original `user_input`.

    Args:
        user_input: The text input from the user.
        entries: A list of `ChatDictionary` objects to apply.
        max_tokens: The maximum token budget for the combined content of applied entries.
                    Defaults to 5000.
        strategy: The strategy for sorting entries before replacement.
                  Defaults to "sorted_evenly".

    Returns:
        The processed user input string after all applicable transformations.
        Returns the original input if critical errors occur.
    """
    current_time = datetime.now()
    original_input_for_fallback = user_input # Save for critical error case
    temp_user_input = user_input

    try:
        # 1. Match entries (uses refined match_whole_words for strings)
        logging.debug(f"Chat Dictionary: Initial matching for: {user_input[:100]}")
        # The original `entry.matches()` is a simple check. `match_whole_words` is more robust.
        # The original `process_user_input` had `entry.matches(user_input)` then later `match_whole_words`.
        # Consolidating to `match_whole_words` as the primary matching mechanism.
        try:
            # Ensure entries are ChatDictionary instances
            valid_initial_entries = [e for e in entries if isinstance(e, ChatDictionary)]
            if len(valid_initial_entries) != len(entries):
                logging.warning("Some provided entries were not ChatDictionary instances and were skipped.")

            matched_entries = match_whole_words(valid_initial_entries, user_input)
        except re.error as e:
            log_counter("chat_dict_regex_error", labels={"key": "compilation_phase"}) # Generic key
            logging.error(f"Invalid regex pattern during initial matching. Error: {str(e)}")
            matched_entries = []
        except Exception as e_match:
            log_counter("chat_dict_match_error")
            logging.error(f"Error during initial matching: {str(e_match)}", exc_info=True)
            matched_entries = []


        logging.debug(f"Matched entries after initial filtering: {[e.key_raw for e in matched_entries]}")

        # 2. Apply group scoring
        try:
            logging.debug(f"Chat Dictionary: Applying group scoring for {len(matched_entries)} entries")
            matched_entries = group_scoring(matched_entries)
        except Exception as e_gs: # More specific exception if defined (ChatProcessingError)
            log_counter("chat_dict_group_scoring_error")
            logging.error(f"Error in group scoring: {str(e)}")
            matched_entries = []  # Fallback to empty list

        # 3. Apply probability filter
        try:
            logging.debug(f"Chat Dictionary: Filtering by probability for {len(matched_entries)} entries")
            matched_entries = filter_by_probability(matched_entries)
        except Exception as ChatProcessingError:
            log_counter("chat_dict_probability_error")
            logging.error(f"Error in probability filtering: {str(e)}")
            matched_entries = []  # Fallback to empty list

        # 4. Apply timed effects (filter out those not ready)
        # And update last_triggered for those that *will* be used
        active_timed_entries = []
        try:
            logging.debug("Chat Dictionary: Applying timed effects")
            for entry in matched_entries:
                if apply_timed_effects(entry, current_time): # Checks if eligible
                    active_timed_entries.append(entry)
            matched_entries = active_timed_entries
        except Exception as e_time:
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
        except Exception as ChatProcessingError:
            log_counter("chat_dict_token_budget_error")
            logging.error(f"Error enforcing token budget: {str(e)}")
            matched_entries = []  # Fallback to empty list

        # Alert if token budget exceeded
        try:
            alert_token_budget_exceeded(matched_entries, max_tokens)
        except Exception as ChatProcessingError:
            log_counter("chat_dict_token_alert_error")
            logging.error(f"Error in token budget alert: {str(e)}")

        # Apply replacement strategy
        try:
            logging.debug("Chat Dictionary: Applying replacement strategy")
            matched_entries = apply_strategy(matched_entries, strategy)
        except Exception as ChatProcessingError:
            log_counter("chat_dict_strategy_error")
            logging.error(f"Error applying strategy: {str(e)}")
            matched_entries = []  # Fallback to empty list

        # Generate output with single replacement per match
        for entry in matched_entries:
            try:
                logging.debug("Chat Dictionary: Applying replacements")
                # Use a copy of max_replacements for this run if needed, or modify original for state
                replacements_done_for_this_entry = 0
                # Original code had `entry.max_replacements > 0` check outside loop.
                # If multiple replacements are allowed by one entry definition:
                current_max_replacements = entry.max_replacements # Use current value
                while current_max_replacements > 0:
                    temp_user_input, replaced_count = apply_replacement_once(temp_user_input, entry)
                    if replaced_count > 0:
                        replacements_done_for_this_entry += 1
                        current_max_replacements -= 1
                        # Update last_triggered for entries that actually made a replacement
                        entry.last_triggered = current_time
                    else:
                        break # No more matches for this key
                if replacements_done_for_this_entry > 0:
                     logging.debug(f"Replaced {replacements_done_for_this_entry} occurrences of '{entry.key_raw}'")

            except Exception as e_replace:
                log_counter("chat_dict_replacement_error", labels={"key": entry.key_raw})
                logging.error(f"Error applying replacement for entry {entry.key_raw}: {str(e_replace)}", exc_info=True)
                continue

    except Exception as e_crit: # Catch-all for ChatProcessingError or other unexpected issues
        log_counter("chat_dict_processing_error")
        logging.error(f"Critical error in process_user_input: {str(e_crit)}", exc_info=True)
        return original_input_for_fallback # Return original input on critical failure

    return temp_user_input

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

def save_character(
        db: CharactersRAGDB,
        character_data: Dict[str, Any],
        expected_version: Optional[int] = None
) -> Optional[int]:
    """
    Saves (adds or updates) a character card in the database.

    If a character with the same name already exists, it attempts to update it.
    An `expected_version` can be provided for optimistic concurrency control during updates.
    If the character does not exist, a new one is added.

    Image data in `character_data['image']` is expected as a base64 encoded string;
    it will be decoded to bytes before saving.

    Args:
        db: An instance of `CharactersRAGDB` for database operations.
        character_data: A dictionary containing the character's attributes.
                        Required key: 'name'.
                        Optional keys: 'description', 'personality', 'scenario',
                        'system_prompt' (or 'system'), 'post_history_instructions' (or 'post_history'),
                        'first_message' (or 'mes_example_greeting'), 'message_example' (or 'mes_example'),
                        'creator_notes', 'alternate_greetings' (list/JSON), 'tags' (list/JSON),
                        'creator', 'character_version', 'extensions' (dict/JSON), 'image' (base64 string).
        expected_version: Optional integer. If provided for an update, the character's
                          current version in the DB must match this value.

    Returns:
        The character's ID (integer) if successfully saved or updated.
        Returns None if the name is missing, or if a DB error, conflict,
        or other exception occurs.
    """
    log_counter("save_character_attempt")
    start_time = time.time()

    char_name = character_data.get('name')
    if not char_name:
        logging.error("Character name is required to save.")
        return None

    db_card_data_full = { # Template for all possible fields for add_character_card
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
            db_card_data_full['image'] = base64.b64decode(img_b64_data)
        except Exception as e_img:
            logging.error(f"Error decoding character image for {char_name}: {e_img}")
            db_card_data_full['image'] = None
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
                    f"Version mismatch for character '{char_name}'. Expected {expected_version}, DB has {current_db_version}",
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
            for key, input_value in character_data.items():
                if key == 'name': continue # Name is for lookup, not part of update payload itself
                if key == 'image': # Handle image separately due to b64 decode
                    update_payload['image'] = db_card_data_full['image'] # Use processed image
                elif key in db_card_data_full: # Check if it's a known/mapped field
                     # Use the value from character_data, allowing explicit nulls if desired by client
                    update_payload[key] = input_value
                # else: field not in db_card_data_full, so ignore

            # Map alternate keys if primary not in update_payload but alternate was
            if 'system' in character_data and 'system_prompt' not in character_data:
                update_payload['system_prompt'] = character_data.get('system')
            if 'post_history' in character_data and 'post_history_instructions' not in character_data:
                update_payload['post_history_instructions'] = character_data.get('post_history')
            if 'mes_example_greeting' in character_data and 'first_message' not in character_data:
                update_payload['first_message'] = character_data.get('mes_example_greeting')
            if 'mes_example' in character_data and 'message_example' not in character_data:
                update_payload['message_example'] = character_data.get('mes_example')


            if not update_payload:
                logging.info(
                    f"No updatable fields provided for existing character '{char_name}'. Skipping update, but returning ID.")
                char_id = existing_char['id']  # No actual update, but considered "saved"
            elif db.update_character_card(existing_char['id'], update_payload, current_db_version):
                char_id = existing_char['id']
                logging.info(f"Character '{char_name}' (ID: {char_id}) updated successfully.")
            # db.update_character_card should raise on failure rather than return False
        else:
            logging.info(f"Character '{char_name}' not found. Attempting to add new.")
            # Use db_card_data_full for adding, as it contains all potential fields
            char_id = db.add_character_card(db_card_data_full)
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


def load_characters(db: CharactersRAGDB) -> Dict[str, Dict[str, Any]]:
    """
    Loads all character cards from the database.

    The image data (if present) is converted from a database BLOB to a
    base64 encoded string and stored in the 'image_base64' key of each
    character's dictionary.

    Args:
        db: An instance of `CharactersRAGDB` for database operations.

    Returns:
        A dictionary where keys are character names and values are dictionaries
        representing the character cards. Returns an empty dictionary if an
        error occurs or no characters are found.
    """
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
    """
    Retrieves a sorted list of all character names from the database.

    Args:
        db: An instance of `CharactersRAGDB` for database operations.

    Returns:
        A list of character names, sorted alphabetically.
        Returns an empty list if an error occurs or no characters are found.
    """
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
# End of Chat_Functions.py
##########################################################################################################################

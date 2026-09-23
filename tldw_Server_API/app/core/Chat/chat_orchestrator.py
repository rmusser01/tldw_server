# chat_orchestrator.py
# Description: Core chat orchestration functions for LLM interactions
"""
This module provides the core chat orchestration functionality, including
the chat_api_call / chat_api_call_async dispatchers for the configured
LLM providers.
"""
#
# Imports
import time
from typing import Any, Callable, Optional, Union

#
# 3rd-party Libraries
from loguru import logger

#
# Local Imports
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.Chat.chat_logging import (
    exception_summary,
    response_summary,
    text_summary,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    perform_chat_api_call,
    perform_chat_api_call_async,
)
from tldw_Server_API.app.core.Chat.orchestrator.error_mapping import (
    map_stream_error,
)
from tldw_Server_API.app.core.Chat.orchestrator.provider_resolution import (
    resolve_provider,
)
from tldw_Server_API.app.core.Chat.orchestrator.stream_execution import (
    execute_stream,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.exceptions import (
    NetworkError,
    RetryExhaustedError,
)
from tldw_Server_API.app.core.LLM_Calls.deprecation import log_legacy_once
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    get_http_status_from_exception as _llm_get_http_status_from_exception,
)
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.testing import is_truthy as _shared_is_truthy

_CHAT_ORCHESTRATOR_COERCE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)

_CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)

try:
    from requests.exceptions import RequestException as _REQUESTS_REQUEST_EXCEPTION
except ImportError:
    _REQUESTS_REQUEST_EXCEPTION = None

try:
    from httpx import HTTPError as _HTTPX_HTTP_ERROR
    from httpx import RequestError as _HTTPX_REQUEST_ERROR
except ImportError:
    _HTTPX_HTTP_ERROR = None
    _HTTPX_REQUEST_ERROR = None

_CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS = (
    *_CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS,
    # _is_network_exception explicitly classifies these two, and the handler has a
    # ChatProviderError(504) branch for them -- but neither was catchable here, so both
    # escaped chat_api_call unmapped and that branch was dead.
    NetworkError,
    RetryExhaustedError,
    *((_REQUESTS_REQUEST_EXCEPTION,) if _REQUESTS_REQUEST_EXCEPTION else ()),
    *((_HTTPX_REQUEST_ERROR,) if _HTTPX_REQUEST_ERROR else ()),
    *((_HTTPX_HTTP_ERROR,) if _HTTPX_HTTP_ERROR else ()),
)

#
####################################################################################################
#
# Error Message Sanitization
#

def _sanitize_error_for_client(error_text: str, max_length: int = 100) -> str:
    """
    Sanitize error messages before sending to clients to prevent information leakage.

    This removes potentially sensitive information like:
    - API keys or tokens
    - Internal URLs
    - Stack traces
    - Detailed error responses from upstream providers

    Args:
        error_text: Raw error text
        max_length: Maximum length of sanitized message

    Returns:
        Sanitized error message safe for client consumption
    """
    if not error_text:
        return "Unknown error"

    # Convert to string if needed
    error_str = str(error_text)

    # Remove potential sensitive patterns
    import re

    # Remove anything that looks like an API key or token
    error_str = re.sub(r'(api[_-]?key|token|secret|password|auth)["\']?\s*[:=]\s*["\']?[^\s"\']+', '[REDACTED]', error_str, flags=re.IGNORECASE)

    # Remove URLs with potential sensitive query params
    error_str = re.sub(r'https?://[^\s]+', '[URL]', error_str)

    # Remove file paths
    error_str = re.sub(r'(/[^\s:]+)+', '[PATH]', error_str)

    # Remove stack trace patterns
    error_str = re.sub(r'File "[^"]+", line \d+', '', error_str)
    error_str = re.sub(r'Traceback \(most recent call last\):', '', error_str)

    # Truncate and clean up
    error_str = ' '.join(error_str.split())  # Normalize whitespace
    if len(error_str) > max_length:
        error_str = error_str[:max_length] + "..."

    return error_str or "An error occurred"


def _get_http_status_from_exception(exc: Exception) -> Optional[int]:
    """Delegate to the canonical extractor in LLM_Calls.

    This was a third copy of the same walk (response.status_code -> response.status ->
    exc.status_code -> exc.status -> message text). Two of the three copies carried a
    double-escaped regex that could never match, so an upstream status carried only in
    the message was silently lost. Kept as a thin alias so the single in-module caller
    and any patch points keep working.
    """
    return _llm_get_http_status_from_exception(exc)


def _get_http_error_text(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if text is None:
            text = getattr(response, "content", None)
            if isinstance(text, (bytes, bytearray)):
                try:
                    text = text.decode("utf-8", errors="replace")
                except _CHAT_ORCHESTRATOR_COERCE_EXCEPTIONS:
                    text = None
        if text is not None:
            return str(text)
    response_text = getattr(exc, "response_text", None)
    if response_text:
        return str(response_text)
    return str(exc)


def _is_network_exception(exc: Exception) -> bool:
    if isinstance(exc, (NetworkError, RetryExhaustedError)):
        return True
    module = getattr(exc.__class__, "__module__", "")
    name = exc.__class__.__name__
    if module.startswith("requests"):
        return "RequestException" in name or "ConnectionError" in name or "Timeout" in name
    if module.startswith("httpx"):
        return "RequestError" in name or "Connect" in name or "Timeout" in name
    return False

#
####################################################################################################
#
# Token Counting
#

def approximate_token_count(history):
    """
    Approximate the token count for a chat history.

    Args:
        history: Chat history in various formats

    Returns:
        Approximate token count
    """
    try:
        total_text = ''
        for user_msg, bot_msg in history:
            if user_msg:
                total_text += user_msg + ' '
            if bot_msg:
                total_text += bot_msg + ' '
        total_tokens = len(total_text.split())
        return total_tokens
    except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(error_type=type(e).__name__).error("Error calculating token count")
        return 0

#
####################################################################################################
#
# Main Chat API Call Dispatcher
#

def chat_api_call(
    api_endpoint: str,
    messages_payload: list[dict[str, Any]], # CHANGED from input_data, prompt
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
    logit_bias: Optional[dict[str, float]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, list[str]]] = None,
    response_format: Optional[dict[str, str]] = None,  # Expects {'type': 'text' | 'json_object'}
    n: Optional[int] = None,
    user_identifier: Optional[str] = None,  # Renamed from 'user' to avoid conflict with 'user' role in messages
    # Provider-specific extensions (e.g., Bedrock guardrails)
    extra_headers: Optional[dict[str, str]] = None,
    extra_body: Optional[dict[str, Any]] = None,
    inference_prefix_cache_intent: Optional[dict[str, Any]] = None,
    # Optional preloaded config to reduce repeated IO in hot paths
    app_config: Optional[dict[str, Any]] = None,
    # Marks api_key/app_config as one authoritative runtime snapshot.
    credentials_resolved: bool = False,
    # Opaque execution capability proving the authoritative snapshot.
    _provider_call_credentials: ProviderCallCredentials | None = None,
    # Testing hooks
    http_client_factory: Optional[Callable[[int], Any]] = None,
    http_fetcher: Optional[Callable[..., Any]] = None,
    ):
    """
    Acts as a unified dispatcher to call various LLM API providers.

    This function routes chat requests to the adapter registry based on
    `api_endpoint` while preserving the legacy signature and error mapping.

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
        HTTP client errors from upstream provider handlers (status errors or network failures).
    """
    resolved_endpoint = resolve_provider(model=model, provider=api_endpoint)
    endpoint_lower = resolved_endpoint.lower()
    logger.info(f"Chat API Call - Routing to endpoint: {endpoint_lower}")
    log_counter("chat_api_call_attempt", labels={"api_endpoint": endpoint_lower})
    start_time = time.time()
    log_legacy_once(
        "chat_orchestrator.chat_api_call",
        "chat_orchestrator.chat_api_call is deprecated; use chat_service.perform_chat_api_call instead.",
    )

    call_kwargs = {
        "api_endpoint": resolved_endpoint,
        "messages_payload": messages_payload,
        "api_key": api_key,
        "temp": temp,
        "system_message": system_message,
        "streaming": streaming,
        "minp": minp,
        "maxp": maxp,
        "model": model,
        "topk": topk,
        "topp": topp,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "logit_bias": logit_bias,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "seed": seed,
        "stop": stop,
        "response_format": response_format,
        "n": n,
        "user_identifier": user_identifier,
        "extra_headers": extra_headers,
        "extra_body": extra_body,
        "inference_prefix_cache_intent": inference_prefix_cache_intent,
        "app_config": app_config,
        "credentials_resolved": credentials_resolved,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: _provider_call_credentials,
        "http_client_factory": http_client_factory,
        "http_fetcher": http_fetcher,
    }

    # Never log secrets by default; allow opt-in masked key logging via env
    try:
        import os as _os_keys
        _key_val = call_kwargs.get("api_key")
        if (
            _key_val
            and isinstance(_key_val, str)
            and len(_key_val) > 8
            and _shared_is_truthy(_os_keys.getenv("ALLOW_MASKED_KEY_LOG", ""))
        ):
            logger.debug(
                "Chat API Call - API Key (masked): {}...{}",
                _key_val[:4],
                _key_val[-4:]
            )
    except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as key_log_err:
        logger.debug("Could not log masked API key: {}", exception_summary(key_log_err))

    try:
        logger.debug(
            "Calling adapter-backed chat dispatcher with kwargs: {}",
            {k: (type(v) if k != "api_key" else "key_hidden") for k, v in call_kwargs.items()},
        )
        response = perform_chat_api_call(**call_kwargs)

        call_duration = time.time() - start_time
        log_histogram("chat_api_call_duration", call_duration, labels={"api_endpoint": endpoint_lower})
        log_counter("chat_api_call_success", labels={"api_endpoint": endpoint_lower})

        if isinstance(response, str):
             logger.debug("Debug - Chat API Call - Response summary: {}", response_summary(response))
        elif hasattr(response, '__iter__') and not isinstance(response, (str, bytes, dict)):
             logger.debug("Debug - Chat API Call - Response: Streaming Generator")
        else:
             logger.debug("Debug - Chat API Call - Response summary: {}", response_summary(response))
        if streaming:
            return execute_stream(response)
        return response

    # --- Exception Mapping (copied from your original, ensure it's still relevant) ---
    except (
        ChatAuthenticationError,
        ChatRateLimitError,
        ChatBadRequestError,
        ChatConfigurationError,
        ChatProviderError,
        ChatAPIError,
    ) as e_chat_direct:
        # This catches cases where the handler itself has already processed an error
        # (e.g. non-HTTP error, or it decided to raise a specific Chat*Error type)
        # and raises one of our custom exceptions.
        # Safely access status_code with fallback
        status_code = getattr(e_chat_direct, 'status_code', 500)
        logger.error(
            "Handler for {} directly raised: {}",
            endpoint_lower,
            exception_summary(e_chat_direct),
            exc_info=status_code >= 500)
        raise  # Re-raise the specific error
    except (ValueError, TypeError, KeyError) as e:
        logger.error(
            "Value/Type/Key error during chat API call setup for {}: {}",
            endpoint_lower,
            exception_summary(e),
            exc_info=True,
        )
        error_type = "Configuration/Parameter Error"
        if "Unsupported API endpoint" in str(e):
            raise ChatConfigurationError(provider=endpoint_lower, message=f"Unsupported API endpoint: {endpoint_lower}") from e
        else:
            raise ChatBadRequestError(provider=endpoint_lower, message=f"{error_type} for {endpoint_lower}: {e}") from e
    except (KeyboardInterrupt, SystemExit):
        # Don't catch system-level signals - let them propagate
        raise
    except _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS as e:
        status_code = _get_http_status_from_exception(e)
        if status_code is not None:
            error_text = _get_http_error_text(e)
            log_message_base = f"{endpoint_lower} API call failed with status {status_code}"
            try:
                logger.error("{} details_summary={}", log_message_base, text_summary(error_text), exc_info=False)
            except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as log_e:
                logger.error("Error during logging HTTP error details: {}", exception_summary(log_e))
            sanitized_error = _sanitize_error_for_client(error_text)
            if status_code == 401:
                raise ChatAuthenticationError(provider=endpoint_lower,
                                              message="Authentication failed. Please check your API key.") from e
            if status_code == 429:
                raise ChatRateLimitError(provider=endpoint_lower,
                                         message="Rate limit exceeded. Please try again later.") from e
            if 400 <= status_code < 500:
                raise ChatBadRequestError(provider=endpoint_lower,
                                          message=f"Invalid request (Status {status_code}). {sanitized_error}") from e
            if 500 <= status_code < 600:
                raise ChatProviderError(provider=endpoint_lower,
                                        message=f"Provider error (Status {status_code}). Please try again.",
                                        status_code=status_code) from e
            raise ChatAPIError(provider=endpoint_lower,
                               message=f"Unexpected error (Status {status_code}). {sanitized_error}",
                               status_code=status_code) from e
        if _is_network_exception(e):
            logger.error(
                "Network error connecting to {}: {}",
                endpoint_lower,
                exception_summary(e),
                exc_info=False,
            )
            raise ChatProviderError(provider=endpoint_lower, message="Network error. Please check your connection.", status_code=504) from e
        logger.error(
            "Unexpected internal error in chat_api_call for {}: {}",
            endpoint_lower,
            exception_summary(e),
        )
        raise ChatAPIError(provider=endpoint_lower,
                           message=f"An unexpected internal error occurred in chat_api_call for {endpoint_lower}: {str(e)}",
                           status_code=500) from e


async def chat_api_call_async(
    api_endpoint: str,
    messages_payload: list[dict[str, Any]],
    api_key: Optional[str] = None,
    temp: Optional[float] = None,
    system_message: Optional[str] = None,
    streaming: Optional[bool] = None,
    minp: Optional[float] = None,
    maxp: Optional[float] = None,
    model: Optional[str] = None,
    topk: Optional[int] = None,
    topp: Optional[float] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    logit_bias: Optional[dict[str, float]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, dict[str, Any]]] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[Union[str, list[str]]] = None,
    response_format: Optional[dict[str, str]] = None,
    n: Optional[int] = None,
    user_identifier: Optional[str] = None,
    extra_headers: Optional[dict[str, str]] = None,
    extra_body: Optional[dict[str, Any]] = None,
    inference_prefix_cache_intent: Optional[dict[str, Any]] = None,
    app_config: Optional[dict[str, Any]] = None,
    http_client_factory: Optional[Callable[[int], Any]] = None,
    http_fetcher: Optional[Callable[..., Any]] = None,
):
    """Async dispatcher that forwards to the adapter registry.

    Returns either a regular dict (non-stream) or an async iterator (streaming).
    """
    resolved_endpoint = resolve_provider(model=model, provider=api_endpoint)
    endpoint_lower = resolved_endpoint.lower()
    log_legacy_once(
        "chat_orchestrator.chat_api_call_async",
        "chat_orchestrator.chat_api_call_async is deprecated; use chat_service.perform_chat_api_call_async instead.",
    )

    call_kwargs = {
        "api_endpoint": resolved_endpoint,
        "messages_payload": messages_payload,
        "api_key": api_key,
        "temp": temp,
        "system_message": system_message,
        "streaming": streaming,
        "minp": minp,
        "maxp": maxp,
        "model": model,
        "topk": topk,
        "topp": topp,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
        "logit_bias": logit_bias,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "seed": seed,
        "stop": stop,
        "response_format": response_format,
        "n": n,
        "user_identifier": user_identifier,
        "extra_headers": extra_headers,
        "extra_body": extra_body,
        "inference_prefix_cache_intent": inference_prefix_cache_intent,
        "app_config": app_config,
        "http_client_factory": http_client_factory,
        "http_fetcher": http_fetcher,
    }

    try:
        response = await perform_chat_api_call_async(**call_kwargs)
        if streaming:
            return execute_stream(response)
        return response
    except Exception as e:
        mapped_error = map_stream_error(e)
        if _is_network_exception(e):
            raise ChatProviderError(
                provider=endpoint_lower,
                message=f"Network error: {mapped_error['message']}",
                status_code=504,
            ) from e
        if isinstance(
            e,
            (
                ChatAPIError,
                ChatProviderError,
                ChatBadRequestError,
                ChatAuthenticationError,
                ChatRateLimitError,
                ChatConfigurationError,
            ),
        ):
            raise
        # Surface as provider error for unexpected conditions
        raise ChatProviderError(
            provider=endpoint_lower,
            message=f"Unexpected error: {mapped_error['message']}",
        ) from e


#
# End of chat_orchestrator.py
####################################################################################################

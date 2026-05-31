# chat_orchestrator.py
# Description: Core chat orchestration functions for LLM interactions
"""
This module provides the core chat orchestration functionality, including
the main chat_api_call dispatcher and the chat function for multimodal
interactions with various LLM providers.
"""
#
# Imports
import asyncio
import atexit
import os
import threading
import time
from collections.abc import Awaitable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar, Union

#
# 3rd-party Libraries
from loguru import logger
from loguru import logger as logging

#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ResponseFormat
from tldw_Server_API.app.core.Chat import command_router
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.Chat.chat_dictionary import (
    ChatDictionary,
    parse_user_dict_markdown_file,
    process_user_input,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    perform_chat_api_call,
    perform_chat_api_call_async,
)
from tldw_Server_API.app.core.Chat.orchestrator.provider_resolution import (
    resolve_provider,
)
from tldw_Server_API.app.core.Chat.orchestrator.request_validation import (
    normalize_selected_parts,
    normalize_temperature,
)
from tldw_Server_API.app.core.Chat.orchestrator.error_mapping import (
    map_stream_error,
)
from tldw_Server_API.app.core.Chat.orchestrator.stream_execution import (
    execute_stream,
)
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.exceptions import (
    NetworkError,
    RetryExhaustedError,
    SyncCallInEventLoopError,
)
from tldw_Server_API.app.core.LLM_Calls.deprecation import log_legacy_once
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
    *((_REQUESTS_REQUEST_EXCEPTION,) if _REQUESTS_REQUEST_EXCEPTION else ()),
    *((_HTTPX_REQUEST_ERROR,) if _HTTPX_REQUEST_ERROR else ()),
    *((_HTTPX_HTTP_ERROR,) if _HTTPX_HTTP_ERROR else ()),
)

#
####################################################################################################
#
# Type variables
_T = TypeVar("_T")
#
####################################################################################################
#
# Module-level ThreadPoolExecutor for sync-to-async bridging
# Reusing a single executor avoids resource exhaustion under load
#
_SYNC_EXECUTOR: Optional[ThreadPoolExecutor] = None
_SYNC_EXECUTOR_LOCK = threading.Lock()


def _get_sync_executor() -> ThreadPoolExecutor:
    """Get or create the module-level ThreadPoolExecutor for sync coroutine execution.

    Uses double-checked locking for thread-safe lazy initialization.
    The executor is shared across all calls to avoid creating a new executor
    per-call, which would cause resource exhaustion under load.
    """
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is None:
        with _SYNC_EXECUTOR_LOCK:
            if _SYNC_EXECUTOR is None:
                _SYNC_EXECUTOR = ThreadPoolExecutor(
                    max_workers=4,
                    thread_name_prefix="chat_sync_coro"
                )
    return _SYNC_EXECUTOR


def _shutdown_sync_executor() -> None:
    """Shutdown the module-level executor on interpreter exit.

    Uses wait=True with cancel_futures=True to ensure clean shutdown.
    This allows in-flight tasks to complete while preventing new submissions.
    """
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is not None:
        try:
            # cancel_futures=True (Python 3.9+) ensures pending futures are cancelled
            # wait=True ensures we wait for running tasks to complete
            _SYNC_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            # Python < 3.9 doesn't support cancel_futures
            _SYNC_EXECUTOR.shutdown(wait=True)
        except Exception as shutdown_error:  # noqa: BLE001 - shutdown must not raise during exit
            logger.bind(error_type=type(shutdown_error).__name__).debug(
                "Chat orchestrator sync executor shutdown raised non-fatal error"
            )


# Register cleanup on interpreter shutdown
atexit.register(_shutdown_sync_executor)

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
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(response, "status", None)
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                pass
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if status is not None:
        try:
            return int(status)
        except (TypeError, ValueError):
            return None
    if isinstance(exc, NetworkError):
        import re
        match = re.search(r"HTTP\\s+(\\d{3})", str(exc))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


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
        logging.bind(error_type=type(e).__name__).error("Error calculating token count")
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
    logging.info(f"Chat API Call - Routing to endpoint: {endpoint_lower}")
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
            logging.debug(
                "Chat API Call - API Key (masked): %s...%s",
                _key_val[:4],
                _key_val[-4:]
            )
    except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as key_log_err:
        logging.debug(f"Could not log masked API key: {key_log_err}")

    try:
        logging.debug(
            "Calling adapter-backed chat dispatcher with kwargs: {}",
            {k: (type(v) if k != "api_key" else "key_hidden") for k, v in call_kwargs.items()},
        )
        response = perform_chat_api_call(**call_kwargs)

        call_duration = time.time() - start_time
        log_histogram("chat_api_call_duration", call_duration, labels={"api_endpoint": endpoint_lower})
        log_counter("chat_api_call_success", labels={"api_endpoint": endpoint_lower})

        if isinstance(response, str):
             logging.debug(f"Debug - Chat API Call - Response (first 500 chars): {response[:500]}...")
        elif hasattr(response, '__iter__') and not isinstance(response, (str, bytes, dict)):
             logging.debug("Debug - Chat API Call - Response: Streaming Generator")
        else:
             logging.debug(f"Debug - Chat API Call - Response Type: {type(response)}")
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
        # Escape curly braces in the error message to avoid loguru formatting issues
        error_message = getattr(e_chat_direct, 'message', str(e_chat_direct))
        escaped_message = str(error_message).replace("{", "{{").replace("}", "}}")
        # Safely access status_code with fallback
        status_code = getattr(e_chat_direct, 'status_code', 500)
        logging.error(
            f"Handler for {endpoint_lower} directly raised: {type(e_chat_direct).__name__} - {escaped_message}",
            exc_info=status_code >= 500)
        raise  # Re-raise the specific error
    except (ValueError, TypeError, KeyError) as e:
        logging.error(f"Value/Type/Key error during chat API call setup for {endpoint_lower}: {e}", exc_info=True)
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
                logging.error("%s. Details: %s", log_message_base, error_text[:500], exc_info=False)
            except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as log_e:
                logging.error(f"Error during logging HTTP error details: {log_e}")
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
        mapped_error = map_stream_error(e)
        if _is_network_exception(e):
            logging.error(
                f"Network error connecting to {endpoint_lower}: {mapped_error['message']}",
                exc_info=False,
            )
            raise ChatProviderError(provider=endpoint_lower, message="Network error. Please check your connection.", status_code=504) from e
        logging.exception(
            f"Unexpected internal error in chat_api_call for {endpoint_lower}: {e}")
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


# Default timeout for synchronous coroutine execution (5 minutes)
# Configurable via CHAT_SYNC_CORO_TIMEOUT_SECONDS environment variable
def _get_sync_coro_timeout() -> float:
    """Get configurable timeout for sync coroutine execution."""
    import os
    try:
        env_val = os.getenv("CHAT_SYNC_CORO_TIMEOUT_SECONDS")
        if env_val is not None:
            return max(1.0, float(env_val))
    except (ValueError, TypeError):
        pass
    return 300.0

_SYNC_CORO_TIMEOUT_SECONDS = _get_sync_coro_timeout()


def _async_only_enabled() -> bool:
    """Return True when CHAT_COMMANDS_ASYNC_ONLY is enabled."""
    value = os.getenv("CHAT_COMMANDS_ASYNC_ONLY", "")
    return _shared_is_truthy(value)


def _sync_in_event_loop_strict() -> bool:
    """Return True when CHAT_SYNC_IN_EVENT_LOOP_STRICT is enabled."""
    value = os.getenv("CHAT_SYNC_IN_EVENT_LOOP_STRICT", "")
    return _shared_is_truthy(value)


def _run_coro_sync(coro: Awaitable[_T], timeout: float = _SYNC_CORO_TIMEOUT_SECONDS) -> _T:
    """
    Run an async coroutine from synchronous code in a loop-safe way.

    If no event loop is running on the current thread, this uses asyncio.run
    directly. When a loop is already running, it offloads execution to a worker
    thread that owns its own event loop to avoid nested-loop errors.

    Uses a module-level ThreadPoolExecutor to avoid creating a new executor
    per call, which would cause resource exhaustion under load.

    Args:
        coro: The coroutine to execute
        timeout: Maximum time to wait for completion (seconds). Default is 5 minutes.

    Returns:
        The result of the coroutine

    Raises:
        TimeoutError: If the coroutine doesn't complete within the timeout
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread: safe to use asyncio.run directly.
        return asyncio.run(coro)

    # Running inside an event loop on this thread: offload to a worker thread
    # that owns its own event loop. Use the pooled executor to avoid resource leak.
    executor = _get_sync_executor()
    future = executor.submit(lambda: asyncio.run(coro))
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        logger.error(f"Synchronous coroutine execution timed out after {timeout}s")
        future.cancel()
        raise


def _run_achat_sync(
    message: str,
    history: list[dict[str, Any]],
    media_content: Optional[dict[str, str]],
    selected_parts: list[str],
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
    chatdict_entries: Optional[list[Any]] = None,
    max_tokens: int = 500,
    strategy: str = "sorted_evenly",
    current_image_input: Optional[dict[str, str]] = None,
    image_history_mode: str = "tag_past",
    llm_max_tokens: Optional[int] = None,
    llm_seed: Optional[int] = None,
    llm_stop: Optional[Union[str, list[str]]] = None,
    llm_response_format: Optional[ResponseFormat] = None,
    llm_n: Optional[int] = None,
    llm_user_identifier: Optional[str] = None,
    llm_logprobs: Optional[bool] = None,
    llm_top_logprobs: Optional[int] = None,
    llm_logit_bias: Optional[dict[str, float]] = None,
    llm_presence_penalty: Optional[float] = None,
    llm_frequency_penalty: Optional[float] = None,
    llm_tools: Optional[list[dict[str, Any]]] = None,
    llm_tool_choice: Optional[Union[str, dict[str, Any]]] = None,
) -> Union[str, Any]:
    """Run the async achat() orchestrator from synchronous code.

    This helper is safe to call when no event loop is running on the current
    thread (uses asyncio.run), and will offload to a worker thread when a loop
    is already running to avoid nested event loops or deadlocks.
    """

    return _run_coro_sync(
        achat(
            message=message,
            history=history,
            media_content=media_content,
            selected_parts=selected_parts,
            api_endpoint=api_endpoint,
            api_key=api_key,
            custom_prompt=custom_prompt,
            temperature=temperature,
            system_message=system_message,
            streaming=streaming,
            minp=minp,
            maxp=maxp,
            model=model,
            topp=topp,
            topk=topk,
            chatdict_entries=chatdict_entries,
            max_tokens=max_tokens,
            strategy=strategy,
            current_image_input=current_image_input,
            image_history_mode=image_history_mode,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_stop=llm_stop,
            llm_response_format=llm_response_format,
            llm_n=llm_n,
            llm_user_identifier=llm_user_identifier,
            llm_logprobs=llm_logprobs,
            llm_top_logprobs=llm_top_logprobs,
            llm_logit_bias=llm_logit_bias,
            llm_presence_penalty=llm_presence_penalty,
            llm_frequency_penalty=llm_frequency_penalty,
            llm_tools=llm_tools,
            llm_tool_choice=llm_tool_choice,
        )
    )

#
####################################################################################################
#
# Main Chat Function
#

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
def _chat_sync_impl(
    message: str,
    history: list[dict[str, Any]],
    media_content: Optional[dict[str, str]],
    selected_parts: list[str],
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
    chatdict_entries: Optional[list[Any]] = None, # Should be List[ChatDictionary]
    max_tokens: int = 500,
    strategy: str = "sorted_evenly",
    current_image_input: Optional[dict[str, str]] = None,
    image_history_mode: str = "tag_past",
    llm_max_tokens: Optional[int] = None,
    llm_seed: Optional[int] = None,
    llm_stop: Optional[Union[str, list[str]]] = None,
    llm_response_format: Optional[ResponseFormat] = None,
    llm_n: Optional[int] = None,
    llm_user_identifier: Optional[str] = None,
    llm_logprobs: Optional[bool] = None,
    llm_top_logprobs: Optional[int] = None,
    llm_logit_bias: Optional[dict[str, float]] = None,
    llm_presence_penalty: Optional[float] = None,
    llm_frequency_penalty: Optional[float] = None,
    llm_tools: Optional[list[dict[str, Any]]] = None,
    llm_tool_choice: Optional[Union[str, dict[str, Any]]] = None
) -> Union[str, Any]:  # Any for streaming generator
    """
    Internal synchronous implementation of the chat orchestration logic.

    This contains the original synchronous behavior used prior to the
    async-first refactor and is retained for legacy streaming callers and as
    a reference while the async `achat` path becomes canonical.
    """
    log_counter("chat_attempt_multimodal", labels={"api_endpoint": api_endpoint, "image_mode": image_history_mode})
    start_time = time.time()

    try:
        logging.info(f"Debug - Chat Function - Input Text: '{message}', Image provided: {'Yes' if current_image_input else 'No'}")
        logging.info(f"Debug - Chat Function - History length: {len(history)}, Image History Mode: {image_history_mode}")

        # Ensure selected_parts is a list
        selected_parts = normalize_selected_parts(selected_parts)

        # Parse slash-commands before dictionary processing
        injected_command_system_text: Optional[str] = None
        # Initialize command variables for safe access later (instead of using unsafe locals() checks)
        cmd_name: Optional[str] = None
        cmd_args: Optional[str] = None
        cmd_res: Optional[Any] = None
        if command_router.commands_enabled() and isinstance(message, str):
            parsed = command_router.parse_slash_command(message)
            if parsed:
                cmd_name, cmd_args = parsed
                # Build minimal context; user_identifier may be None
                auth_user_int = None
                try:
                    if llm_user_identifier is not None:
                        auth_user_int = int(llm_user_identifier)  # best-effort parse for RBAC
                except _CHAT_ORCHESTRATOR_COERCE_EXCEPTIONS:
                    auth_user_int = None
                ctx = command_router.CommandContext(user_id=llm_user_identifier or "anonymous", auth_user_id=auth_user_int)
                cmd_res = _run_coro_sync(
                    command_router.async_dispatch_command(ctx, cmd_name, cmd_args)
                )
                injected_text = command_router.build_injection_text(cmd_name, cmd_res.content)
                if cmd_res.ok:
                    injection_mode = command_router.get_injection_mode()
                    # Start with the args-only message (command token removed)
                    base_args = (cmd_args or "").strip()
                    if injection_mode == "preface":
                        prefix = f"{injected_text}\n\n"
                        message = f"{prefix}{base_args}" if base_args else prefix.strip()
                    elif injection_mode == "replace":
                        # Replace the user's message content with the command result
                        # Include the marker for traceability, consistent with preface
                        message = injected_text.strip()
                    else:  # default: system injection
                        message = base_args
                        injected_command_system_text = injected_text
                else:
                    # On error, provide a short system injection so the model has context; strip the command from user text
                    message = (cmd_args or "").strip()
                    injected_command_system_text = injected_text

        # Process message with Chat Dictionary (text only for now)
        processed_text_message = message
        if chatdict_entries and message:
            processed_text_message = process_user_input(
                message, chatdict_entries, max_tokens=max_tokens, strategy=strategy
            )

        # --- Construct messages payload for the LLM API (OpenAI format) ---
        llm_messages_payload: list[dict[str, Any]] = []

        # PHILOSOPHY:
        # `chat()` prepares the `llm_messages_payload` (user/assistant turns with multimodal content)
        # and collects a separate `system_message`.
        # `chat_api_call()` hands both to the adapter registry via `chat_service.perform_chat_api_call`.
        # Each adapter is responsible for system-message placement:
        #   - Providers that require a system message inside `messages` (OpenAI) prepend it.
        #   - Providers that accept a dedicated system field (Anthropic `system_prompt`) map it.
        # This keeps `chat()` provider-agnostic.


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
                            if role == "user":
                                last_user_image_url_from_history = image_url_data
                        elif image_history_mode == "send_last_user_image" and role == "user":
                            last_user_image_url_from_history = image_url_data # Track, add later
                        elif image_history_mode == "tag_past":
                            mime_type_part = "image"
                            if image_url_data.startswith("data:image/") and ";base64," in image_url_data:
                                try:
                                    mime_type_part = image_url_data.split(';base64,')[0].split('/')[-1]
                                except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as e:
                                    logging.debug(f"Failed to extract MIME type from data URI: {e}")
                                    mime_type_part = "image"
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
        current_user_content_parts: list[dict[str, Any]] = []

        # Combine RAG, custom_prompt (if it's for current turn's text), and processed_text_message
        # Deciding where `custom_prompt` goes: if it's a direct instruction for *this* turn,
        # it should be part of the user's text. If it's more like a persona or ongoing rule,
        # it's better in `system_message`. Let's assume it's for this turn.
        final_text_for_current_message = processed_text_message
        if custom_prompt: # Prepend custom_prompt if it exists
            final_text_for_current_message = f"{custom_prompt}\n\n{final_text_for_current_message}"

        final_text_for_current_message = f"{rag_text_prefix}{final_text_for_current_message}".strip()

        # Inject command result as a separate system message when configured
        if injected_command_system_text:
            # Enrich with audit metadata (non-visible) for downstream logging/adapters that preserve message fields
            _cmd_meta = {
                "source": "slash_command",
                "command": cmd_name,
                "args": cmd_args,
                "mode": command_router.get_injection_mode(),
                "result_ok": getattr(cmd_res, 'ok', False) if cmd_res is not None else False,
                "error": (getattr(cmd_res, 'metadata', {}) or {}).get("error") if cmd_res is not None else None,
                "rbac": (getattr(cmd_res, 'metadata', {}) or {}).get("rbac") if cmd_res is not None else None,
            }
            msg_obj = {
                "role": "system",
                "name": "system-command",
                "content": [{"type": "text", "text": injected_command_system_text}],
                "metadata": {"tldw_injection": _cmd_meta},
            }
            llm_messages_payload.append(msg_obj)

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
        temperature_float = normalize_temperature(temperature, default=0.7)

        logging.debug("Debug - Chat Function - Final LLM Payload (structure, image data truncated):")
        for i, msg_p in enumerate(llm_messages_payload):
            content_log = []
            if isinstance(msg_p.get("content"), list):
                for _part_idx, part_c in enumerate(msg_p["content"]):
                    if part_c.get("type") == "text":
                        content_log.append(f"text: '{part_c['text'][:30]}...'")
                    elif part_c.get("type") == "image_url":
                        content_log.append(f"image: '{part_c['image_url']['url'][:40]}...'")
            logging.debug(f"  Msg {i}: Role: {msg_p['role']}, Content: [{', '.join(content_log)}]")

        logging.debug(f"Debug - Chat Function - Temperature: {temperature}")
        # Avoid logging secrets unless explicitly enabled
        try:
            if api_key and _shared_is_truthy(os.getenv("ALLOW_MASKED_KEY_LOG", "")):
                logging.debug("Debug - Chat Function - API Key (masked): %s...%s", api_key[:4], api_key[-4:])
        except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as key_log_err:
            logging.debug(f"Could not log masked API key: {key_log_err}")
        logging.debug(f"Debug - Chat Function - Prompt: {custom_prompt}")

        # --- Call the LLM via the updated chat_api_call ---
        # Preload config once and pass down to provider to avoid repeated loads
        preloaded_cfg = load_and_log_configs()
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
            tool_choice=llm_tool_choice,
            app_config=preloaded_cfg,
        )

        if streaming:
            logging.debug("Chat Function - Response: Streaming Generator")
            return response
        else:
            chat_duration = time.time() - start_time
            log_histogram("chat_duration_multimodal", chat_duration, labels={"api_endpoint": api_endpoint})
            log_counter("chat_success_multimodal", labels={"api_endpoint": api_endpoint})
            logging.debug(f"Chat Function - Response (first 500 chars): {str(response)[:500]}")

            loaded_config_data = preloaded_cfg or load_and_log_configs()
            post_gen_replacement_config = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement')
            if post_gen_replacement_config and isinstance(response, str):
                post_gen_replacement_dict_path = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement_dict')
                if post_gen_replacement_dict_path and os.path.exists(post_gen_replacement_dict_path):
                    try:
                        parsed_entries = parse_user_dict_markdown_file(post_gen_replacement_dict_path)
                        if parsed_entries:
                            post_gen_chat_dict_objects = [
                                ChatDictionary(key=k, content=str(v)) for k, v in parsed_entries.items()
                            ]
                            response = process_user_input(response, post_gen_chat_dict_objects)
                            logging.debug(
                                f"Response after post-gen replacement (first 500 chars): {str(response)[:500]}"
                            )
                        else:
                            logging.debug("Post-gen dictionary parsed but resulted in no ChatDictionary objects.")
                    except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as e_post_gen:
                        logging.error(f"Error during post-generation replacement: {e_post_gen}", exc_info=True)
                else:
                    logging.warning("Post-gen replacement enabled but dict file not found/configured.")
            return response

    except ChatAPIError:
        # Re-raise ChatAPIError subclasses as-is for proper upstream handling
        raise
    except _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS as e:
        log_counter("chat_error_multimodal", labels={"api_endpoint": api_endpoint, "error": str(e)})
        logging.error(f"Error in multimodal chat function: {str(e)}", exc_info=True)
        # Raise a proper exception instead of returning an error string
        raise ChatProviderError(
            message=f"An error occurred in the chat function: {str(e)}",
            status_code=500,
            provider=api_endpoint,
            details=str(e)
        ) from e


def chat(
    message: str,
    history: list[dict[str, Any]],
    media_content: Optional[dict[str, str]],
    selected_parts: list[str],
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
    chatdict_entries: Optional[list[Any]] = None,  # Should be List[ChatDictionary]
    max_tokens: int = 500,
    strategy: str = "sorted_evenly",
    current_image_input: Optional[dict[str, str]] = None,
    image_history_mode: str = "tag_past",
    llm_max_tokens: Optional[int] = None,
    llm_seed: Optional[int] = None,
    llm_stop: Optional[Union[str, list[str]]] = None,
    llm_response_format: Optional[ResponseFormat] = None,
    llm_n: Optional[int] = None,
    llm_user_identifier: Optional[str] = None,
    llm_logprobs: Optional[bool] = None,
    llm_top_logprobs: Optional[int] = None,
    llm_logit_bias: Optional[dict[str, float]] = None,
    llm_presence_penalty: Optional[float] = None,
    llm_frequency_penalty: Optional[float] = None,
    llm_tools: Optional[list[dict[str, Any]]] = None,
    llm_tool_choice: Optional[Union[str, dict[str, Any]]] = None,
) -> Union[str, Any]:
    """
    Public synchronous chat entrypoint.

    For non-streaming calls, this function acts as a sync wrapper around the
    async `achat` orchestrator. When called from a running event loop, it
    offloads work to a worker thread and returns an awaitable future; async
    callers should prefer awaiting `achat(...)` directly.

    For streaming calls, it delegates to the legacy synchronous implementation
    preserved in `_chat_sync_impl` to maintain existing generator semantics for
    legacy consumers.
    """
    if _async_only_enabled():
        raise RuntimeError(
            "CHAT_COMMANDS_ASYNC_ONLY is enabled. "
            "Use await achat(...) instead of chat()."
        )
    if streaming:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "chat(streaming=True) cannot be called from an active event loop. "
                "Use await achat(...) or run streaming chat from a non-async context."
            )
        return _chat_sync_impl(
            message=message,
            history=history,
            media_content=media_content,
            selected_parts=selected_parts,
            api_endpoint=api_endpoint,
            api_key=api_key,
            custom_prompt=custom_prompt,
            temperature=temperature,
            system_message=system_message,
            streaming=streaming,
            minp=minp,
            maxp=maxp,
            model=model,
            topp=topp,
            topk=topk,
            chatdict_entries=chatdict_entries,
            max_tokens=max_tokens,
            strategy=strategy,
            current_image_input=current_image_input,
            image_history_mode=image_history_mode,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_stop=llm_stop,
            llm_response_format=llm_response_format,
            llm_n=llm_n,
            llm_user_identifier=llm_user_identifier,
            llm_logprobs=llm_logprobs,
            llm_top_logprobs=llm_top_logprobs,
            llm_logit_bias=llm_logit_bias,
            llm_presence_penalty=llm_presence_penalty,
            llm_frequency_penalty=llm_frequency_penalty,
            llm_tools=llm_tools,
            llm_tool_choice=llm_tool_choice,
        )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _run_achat_sync(
            message=message,
            history=history,
            media_content=media_content,
            selected_parts=selected_parts,
            api_endpoint=api_endpoint,
            api_key=api_key,
            custom_prompt=custom_prompt,
            temperature=temperature,
            system_message=system_message,
            streaming=streaming,
            minp=minp,
            maxp=maxp,
            model=model,
            topp=topp,
            topk=topk,
            chatdict_entries=chatdict_entries,
            max_tokens=max_tokens,
            strategy=strategy,
            current_image_input=current_image_input,
            image_history_mode=image_history_mode,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_stop=llm_stop,
            llm_response_format=llm_response_format,
            llm_n=llm_n,
            llm_user_identifier=llm_user_identifier,
            llm_logprobs=llm_logprobs,
            llm_top_logprobs=llm_top_logprobs,
            llm_logit_bias=llm_logit_bias,
            llm_presence_penalty=llm_presence_penalty,
            llm_frequency_penalty=llm_frequency_penalty,
            llm_tools=llm_tools,
            llm_tool_choice=llm_tool_choice,
        )

    if _sync_in_event_loop_strict():
        raise SyncCallInEventLoopError(
            "chat() called from a running event loop. "
            "Use await achat(...) or disable CHAT_SYNC_IN_EVENT_LOOP_STRICT."
        )

    logging.warning(
        "chat() called from a running event loop; returning a Future. "
        "Prefer await achat(...) in async contexts."
    )
    executor = _get_sync_executor()
    return loop.run_in_executor(
        executor,
        lambda: _run_achat_sync(
            message=message,
            history=history,
            media_content=media_content,
            selected_parts=selected_parts,
            api_endpoint=api_endpoint,
            api_key=api_key,
            custom_prompt=custom_prompt,
            temperature=temperature,
            system_message=system_message,
            streaming=streaming,
            minp=minp,
            maxp=maxp,
            model=model,
            topp=topp,
            topk=topk,
            chatdict_entries=chatdict_entries,
            max_tokens=max_tokens,
            strategy=strategy,
            current_image_input=current_image_input,
            image_history_mode=image_history_mode,
            llm_max_tokens=llm_max_tokens,
            llm_seed=llm_seed,
            llm_stop=llm_stop,
            llm_response_format=llm_response_format,
            llm_n=llm_n,
            llm_user_identifier=llm_user_identifier,
            llm_logprobs=llm_logprobs,
            llm_top_logprobs=llm_top_logprobs,
            llm_logit_bias=llm_logit_bias,
            llm_presence_penalty=llm_presence_penalty,
            llm_frequency_penalty=llm_frequency_penalty,
            llm_tools=llm_tools,
            llm_tool_choice=llm_tool_choice,
        ),
    )


async def achat(
    message: str,
    history: list[dict[str, Any]],
    media_content: Optional[dict[str, str]],
    selected_parts: list[str],
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
    chatdict_entries: Optional[list[Any]] = None, # Should be List[ChatDictionary]
    max_tokens: int = 500,
    strategy: str = "sorted_evenly",
    current_image_input: Optional[dict[str, str]] = None,
    image_history_mode: str = "tag_past",
    llm_max_tokens: Optional[int] = None,
    llm_seed: Optional[int] = None,
    llm_stop: Optional[Union[str, list[str]]] = None,
    llm_response_format: Optional[ResponseFormat] = None,
    llm_n: Optional[int] = None,
    llm_user_identifier: Optional[str] = None,
    llm_logprobs: Optional[bool] = None,
    llm_top_logprobs: Optional[int] = None,
    llm_logit_bias: Optional[dict[str, float]] = None,
    llm_presence_penalty: Optional[float] = None,
    llm_frequency_penalty: Optional[float] = None,
    llm_tools: Optional[list[dict[str, Any]]] = None,
    llm_tool_choice: Optional[Union[str, dict[str, Any]]] = None,
) -> Union[str, Any]:
    """Async variant of chat() that uses async slash-command dispatch and async provider calls.

    Mirrors the behavior of chat() but awaits command_router.async_dispatch_command
    and chat_api_call_async to ensure proper concurrency semantics.
    """
    log_counter("chat_attempt_multimodal", labels={"api_endpoint": api_endpoint, "image_mode": image_history_mode})
    start_time = time.time()

    try:
        logging.info(f"Debug - Chat Function (async) - Input Text: '{message}', Image provided: {'Yes' if current_image_input else 'No'}")
        logging.info(f"Debug - Chat Function (async) - History length: {len(history)}, Image History Mode: {image_history_mode}")

        selected_parts = normalize_selected_parts(selected_parts)

        injected_command_system_text: Optional[str] = None
        # Initialize command variables for safe access later (instead of using unsafe locals() checks)
        cmd_name: Optional[str] = None
        cmd_args: Optional[str] = None
        cmd_res: Optional[Any] = None
        if command_router.commands_enabled() and isinstance(message, str):
            parsed = command_router.parse_slash_command(message)
            if parsed:
                cmd_name, cmd_args = parsed
                auth_user_int = None
                try:
                    if llm_user_identifier is not None:
                        auth_user_int = int(llm_user_identifier)
                except _CHAT_ORCHESTRATOR_COERCE_EXCEPTIONS:
                    auth_user_int = None
                ctx = command_router.CommandContext(user_id=llm_user_identifier or "anonymous", auth_user_id=auth_user_int)
                cmd_res = await command_router.async_dispatch_command(ctx, cmd_name, cmd_args)
                injected_text = command_router.build_injection_text(cmd_name, cmd_res.content)
                if cmd_res.ok:
                    injection_mode = command_router.get_injection_mode()
                    base_args = (cmd_args or "").strip()
                    if injection_mode == "preface":
                        prefix = f"{injected_text}\n\n"
                        message = f"{prefix}{base_args}" if base_args else prefix.strip()
                    elif injection_mode == "replace":
                        message = injected_text.strip()
                    else:
                        message = base_args
                        injected_command_system_text = injected_text
                else:
                    message = (cmd_args or "").strip()
                    injected_command_system_text = injected_text

        processed_text_message = message
        if chatdict_entries and message:
            processed_text_message = process_user_input(
                message, chatdict_entries, max_tokens=max_tokens, strategy=strategy
            )

        llm_messages_payload: list[dict[str, Any]] = []

        # 2. Process History (now expecting list of OpenAI message dicts)
        last_user_image_url_from_history: Optional[str] = None
        for hist_msg_obj in history:
            role = hist_msg_obj.get("role")
            original_content = hist_msg_obj.get("content")
            processed_hist_content_parts: list[dict[str, Any]] = []

            if isinstance(original_content, str):
                processed_hist_content_parts.append({"type": "text", "text": original_content})
            elif isinstance(original_content, list):
                for part in original_content:
                    if part.get("type") == "text":
                        processed_hist_content_parts.append(part)
                    elif part.get("type") == "image_url":
                        image_url_data = part.get("image_url", {}).get("url", "")
                        if image_history_mode == "send_all":
                            processed_hist_content_parts.append(part)
                            if role == "user":
                                last_user_image_url_from_history = image_url_data
                        elif image_history_mode == "send_last_user_image" and role == "user":
                            # Track only; append later to the last user message
                            last_user_image_url_from_history = image_url_data
                        elif image_history_mode == "tag_past":
                            mime_type_part = "image"
                            if image_url_data.startswith("data:image/") and ";base64," in image_url_data:
                                try:
                                    mime_type_part = image_url_data.split(";base64,")[0].split("/")[-1]
                                except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as e:
                                    logging.debug(f"Failed to extract MIME type from data URI: {e}")
                                    mime_type_part = "image"
                            processed_hist_content_parts.append(
                                {"type": "text", "text": f"<image: prior_history.{mime_type_part}>"}
                            )
                        # "ignore_past": do nothing, image part is skipped

            if processed_hist_content_parts:
                llm_messages_payload.append({"role": role, "content": processed_hist_content_parts})

        # Handle "send_last_user_image" - append it to the last user message in payload if applicable
        if image_history_mode == "send_last_user_image" and last_user_image_url_from_history:
            appended_to_last = False
            for i in range(len(llm_messages_payload) - 1, -1, -1):
                if llm_messages_payload[i]["role"] == "user":
                    if not isinstance(llm_messages_payload[i]["content"], list):
                        llm_messages_payload[i]["content"] = [
                            {"type": "text", "text": str(llm_messages_payload[i]["content"])}
                        ]
                    is_duplicate = any(
                        p.get("type") == "image_url"
                        and p.get("image_url", {}).get("url") == last_user_image_url_from_history
                        for p in llm_messages_payload[i]["content"]
                    )
                    if not is_duplicate:
                        llm_messages_payload[i]["content"].append(
                            {"type": "image_url", "image_url": {"url": last_user_image_url_from_history}}
                        )
                    appended_to_last = True
                    break
            if not appended_to_last:
                logging.debug(
                    "Could not append last_user_image_from_history, no suitable prior user message or already present. "
                    f"Image: {last_user_image_url_from_history[:60]}..."
                )

        # 3. Add RAG Content (prepended to current user's text)
        rag_text_prefix = ""
        if media_content and selected_parts:
            try:
                rag_text_prefix = "\n\n".join(
                    [
                        f"{part.capitalize()}: {media_content.get(part, '')}"
                        for part in selected_parts
                        if media_content.get(part)
                    ]
                ).strip()
                if rag_text_prefix:
                    rag_text_prefix += "\n\n---\n\n"
            except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS:
                rag_text_prefix = ""

        current_user_content_parts: list[dict[str, Any]] = []
        final_text_for_current_message = processed_text_message
        if custom_prompt:
            final_text_for_current_message = f"{custom_prompt}\n\n{final_text_for_current_message}"
        final_text_for_current_message = f"{rag_text_prefix}{final_text_for_current_message}".strip()

        if injected_command_system_text:
            _cmd_meta = {
                "source": "slash_command",
                "command": cmd_name,
                "args": cmd_args,
                "mode": command_router.get_injection_mode(),
                "result_ok": getattr(cmd_res, 'ok', False) if cmd_res is not None else False,
                "error": (getattr(cmd_res, 'metadata', {}) or {}).get("error") if cmd_res is not None else None,
                "rbac": (getattr(cmd_res, 'metadata', {}) or {}).get("rbac") if cmd_res is not None else None,
            }
            msg_obj = {
                "role": "system",
                "name": "system-command",
                "content": [{"type": "text", "text": injected_command_system_text}],
                "metadata": {"tldw_injection": _cmd_meta},
            }
            llm_messages_payload.append(msg_obj)

        if final_text_for_current_message:
            current_user_content_parts.append({"type": "text", "text": final_text_for_current_message})
        if current_image_input and current_image_input.get('base64_data') and current_image_input.get('mime_type'):
            image_url = f"data:{current_image_input['mime_type']};base64,{current_image_input['base64_data']}"
            current_user_content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
        if not current_user_content_parts:
            logging.warning("Current user message has no text or image content parts. Sending a placeholder.")
            current_user_content_parts.append({"type": "text", "text": "(No user input for this turn)"})

        llm_messages_payload.append({"role": "user", "content": current_user_content_parts})

        temperature_float = normalize_temperature(temperature, default=0.7)

        logging.debug("Debug - Async Chat Function - Final LLM Payload prepared")

        preloaded_cfg = load_and_log_configs()
        response = await chat_api_call_async(
            api_endpoint=api_endpoint,
            api_key=api_key,
            messages_payload=llm_messages_payload,
            temp=temperature_float,
            system_message=system_message,
            streaming=streaming,
            minp=minp, maxp=maxp, model=model, topp=topp, topk=topk,
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
            tool_choice=llm_tool_choice,
            app_config=preloaded_cfg,
        )

        if streaming:
            logging.debug("Async Chat Function - Response: Streaming Generator")
            return response
        else:
            chat_duration = time.time() - start_time
            log_histogram("chat_duration_multimodal", chat_duration, labels={"api_endpoint": api_endpoint})
            log_counter("chat_success_multimodal", labels={"api_endpoint": api_endpoint})

            loaded_config_data = preloaded_cfg or load_and_log_configs()
            post_gen_replacement_config = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement')
            if post_gen_replacement_config and isinstance(response, str):
                post_gen_replacement_dict_path = loaded_config_data.get('chat_dictionaries', {}).get('post_gen_replacement_dict')
                if post_gen_replacement_dict_path and os.path.exists(post_gen_replacement_dict_path):
                    try:
                        parsed_entries = parse_user_dict_markdown_file(post_gen_replacement_dict_path)
                        if parsed_entries:
                            post_gen_chat_dict_objects = [
                                ChatDictionary(key=k, content=str(v)) for k, v in parsed_entries.items()
                            ]
                            response = process_user_input(response, post_gen_chat_dict_objects)
                            logging.debug(
                                f"Async response after post-gen replacement (first 500 chars): {str(response)[:500]}"
                            )
                        else:
                            logging.debug("Post-gen dictionary parsed but resulted in no ChatDictionary objects.")
                    except _CHAT_ORCHESTRATOR_NONCRITICAL_EXCEPTIONS as e_post_gen:
                        logging.error(f"Error during post-generation replacement: {e_post_gen}", exc_info=True)
                else:
                    logging.warning("Post-gen replacement enabled but dict file not found/configured.")
            return response

    except ChatAPIError:
        # Re-raise ChatAPIError subclasses as-is for proper upstream handling
        raise
    except _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS as e:
        log_counter("chat_error_multimodal", labels={"api_endpoint": api_endpoint, "error": str(e)})
        logging.error(f"Error in async multimodal chat function: {str(e)}", exc_info=True)
        # Raise a proper exception instead of returning an error string
        raise ChatProviderError(
            message=f"An error occurred in the async chat function: {str(e)}",
            status_code=500,
            provider=api_endpoint,
            details=str(e)
        ) from e

#
# End of chat_orchestrator.py
####################################################################################################

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any, Callable

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
)
from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.config import load_settings
from tldw_Server_API.app.core.exceptions import EgressPolicyError, raise_detached_error
from tldw_Server_API.app.core.http_client import (
    RetryPolicy as _HC_RetryPolicy,
)
from tldw_Server_API.app.core.http_client import (
    create_client as _hc_create_client,
)
from tldw_Server_API.app.core.http_client import (
    fetch as _hc_fetch,
)
from tldw_Server_API.app.core.http_client import (
    stream_response as _hc_stream_response,
)
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    get_http_status_from_exception,
    is_network_error,
)
from tldw_Server_API.app.core.LLM_Calls.local_cache_diagnostics import build_local_cache_diagnostic
from tldw_Server_API.app.core.LLM_Calls.payload_utils import (
    _sanitize_payload_for_logging,
    merge_extra_body,
    merge_extra_headers,
)
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import (
    resolve_trusted_provider_endpoint,
)
from tldw_Server_API.app.core.LLM_Calls.sse import (
    finalize_stream,
    is_done_line,
    normalize_provider_line,
    openai_delta_chunk,
    sse_data,
    sse_done,
)
from tldw_Server_API.app.core.LLM_Calls.streaming import wrap_sync_stream
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope
from tldw_Server_API.app.core.Utils.Utils import logging

from .base import ChatProvider, apply_tool_choice, raise_if_in_band_provider_error

_LOCAL_HTTP_EXCEPTIONS: tuple[type[BaseException], ...] = ()
try:
    import httpx as _local_httpx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _local_httpx = None
else:  # pragma: no cover - imported in runtime when available
    _LOCAL_HTTP_EXCEPTIONS = _LOCAL_HTTP_EXCEPTIONS + (
        _local_httpx.HTTPError,
        _local_httpx.RequestError,
        _local_httpx.HTTPStatusError,
    )

try:
    import requests as _local_requests  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _local_requests = None
else:  # pragma: no cover - imported in runtime when available
    _LOCAL_HTTP_EXCEPTIONS = _LOCAL_HTTP_EXCEPTIONS + (
        _local_requests.exceptions.RequestException,
        _local_requests.exceptions.HTTPError,
    )

_LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
) + _LOCAL_HTTP_EXCEPTIONS

_LOCAL_PROVIDER_UNAVAILABLE_MESSAGE = (
    "The local model provider is currently unavailable."
)


def _bounded_local_provider_name(value: Any) -> str:
    """Return a bounded provider label safe for operational logs and wire types."""

    normalized = "".join(
        character
        for character in str(value or "").strip().lower()
        if character.isalnum() or character in ".-_"
    )[:64]
    return normalized or "local-provider"


def _log_local_provider_failure(
    provider_name: Any,
    exc: BaseException,
    *,
    phase: str,
) -> None:
    """Log only bounded failure metadata, never upstream text or tracebacks."""

    status = get_http_status_from_exception(exc) if isinstance(exc, Exception) else None
    logging.error(
        "{}: Local provider failure phase={} error_type={} upstream_status={}",
        _bounded_local_provider_name(provider_name),
        phase,
        _bounded_local_provider_name(type(exc).__name__),
        status if isinstance(status, int) else "unknown",
        exc_info=False,
    )


def _close_local_provider_resource(
    resource: Any,
    provider_name: Any,
    *,
    phase: str,
) -> None:
    """Close a provider resource without allowing cleanup failures to escape."""

    close = getattr(resource, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # noqa: BLE001 - third-party cleanup must stay terminal
        _log_local_provider_failure(provider_name, exc, phase=phase)


def _local_provider_stream_error_frame(provider_name: Any) -> str:
    """Build the canonical bounded SSE error for local-provider failures."""

    provider = _bounded_local_provider_name(provider_name)
    return sse_data(
        {
            "error": {
                "code": "provider_unavailable",
                "message": _LOCAL_PROVIDER_UNAVAILABLE_MESSAGE,
                "type": f"{provider}_stream_error",
            }
        }
    )


def _is_local_provider_stream_error_line(line: str) -> bool:
    """Detect provider error events without forwarding their untrusted payload."""

    stripped = line.strip()
    if not stripped.lower().startswith("data:") or is_done_line(stripped):
        return False
    try:
        payload = json.loads(stripped.split(":", 1)[1].strip())
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("error") not in (None, "", {}, []):
        return True
    event_type = str(payload.get("type") or "").strip().lower()
    return (
        event_type == "error"
        or event_type.endswith(".error")
        or event_type.endswith("_error")
    )


def _raise_sanitized_local_provider_error(
    provider_name: Any,
    exc: Exception,
    *,
    phase: str,
) -> None:
    """Map an upstream failure to a detached, bounded public exception."""

    provider = _bounded_local_provider_name(provider_name)
    status = get_http_status_from_exception(exc)
    _log_local_provider_failure(provider, exc, phase=phase)
    if isinstance(exc, ChatConfigurationError):
        raise_detached_error(
            ChatConfigurationError(
                provider=provider,
                message="The local model provider configuration is invalid.",
            )
        )
    if isinstance(exc, ChatBadRequestError) or (
        isinstance(status, int) and 400 <= status < 500
    ):
        raise_detached_error(
            ChatBadRequestError(
                provider=provider,
                message="The local model provider rejected the request.",
            )
        )
    public_status = 504 if is_network_error(exc) else 502
    raise_detached_error(
        ChatProviderError(
            provider=provider,
            message=_LOCAL_PROVIDER_UNAVAILABLE_MESSAGE,
            status_code=public_status,
        )
    )


def _select_local_app_config(
    app_config: dict[str, Any] | None,
    *,
    credentials_resolved: bool,
) -> dict[str, Any]:
    """Honor an authoritative runtime snapshot, including an empty one."""
    if credentials_resolved:
        return app_config if isinstance(app_config, dict) else {}
    return app_config or load_settings()


def _extract_text_from_message_content(content: str | list[dict[str, Any]], provider_name: str, msg_index: int) -> str:
    """Extracts and concatenates text parts from a message's content, logging warnings for images."""
    text_parts = []
    has_image = False
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
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


def _chat_with_openai_compatible_local_server(
        api_base_url: str,
        model_name: str | None,
        input_data: list[dict[str, Any]],  # This is messages_payload
        api_key: str | None = None,
        temp: float | None = None,
        system_message: str | None = None, # This will be prepended to messages by this function
        streaming: bool | None = False,
        max_tokens: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        n: int | None = None,
        stop: str | list[str] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        seed: int | None = None,
        response_format: dict[str, str] | None = None, # e.g. {"type": "json_object"}
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        user_identifier: str | None = None, # maps to 'user' in OpenAI spec
        provider_name: str = "Local OpenAI-Compatible Server",
        timeout: int = 120,
        api_retries: int = 1,
        api_retry_delay: int = 1,
        filter_unknown_params: bool = False,
        http_client_factory: Callable[[int], Any] | None = None,
        http_fetcher: Callable[..., Any] | None = None,  # Mirrors signature of _hc_fetch(method=..., url=..., ...)
        http_streamer: Callable[..., Any] | None = None,
        configured_endpoint_scope: ConfiguredEndpointScope | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        app_config: dict[str, Any] | None = None,
        inference_prefix_cache_intent: dict[str, Any] | None = None,
):
    safe_provider_name = _bounded_local_provider_name(provider_name)
    logging.debug("{}: Chat request starting", safe_provider_name)

    headers = {'Content-Type': 'application/json'}
    if api_key: # Some local servers might use a key
        headers['Authorization'] = f'Bearer {api_key}'
    headers = merge_extra_headers(headers, {"extra_headers": extra_headers})

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

    payload: dict[str, Any] = {
        "messages": api_messages,
        "stream": streaming,
    }
    if model_name:
        payload["model"] = model_name
    if temp is not None:
        payload["temperature"] = temp
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k # OpenAI spec doesn't have top_k for chat, but some servers might
    if min_p is not None:
        payload["min_p"] = min_p # Not standard OpenAI, but some servers might support
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if n is not None:
        payload["n"] = n
    if stop is not None:
        payload["stop"] = stop
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if logit_bias is not None:
        payload["logit_bias"] = logit_bias
    if seed is not None:
        payload["seed"] = seed
    if response_format is not None:
        payload["response_format"] = response_format
    if tools is not None:
        payload["tools"] = tools
    apply_tool_choice(payload, tools, tool_choice)
    if logprobs is not None:
        payload["logprobs"] = logprobs
    if top_logprobs is not None: # Can only be used if logprobs is true
        if logprobs:
            payload["top_logprobs"] = top_logprobs
        else:
            logging.warning(f"{provider_name}: top_logprobs provided without logprobs=True. Ignoring top_logprobs.")
    if user_identifier is not None:
        payload["user"] = user_identifier

    if tool_choice is not None and not tools:
        raise ChatBadRequestError(provider=provider_name, message="tool_choice requires tools")

    payload = merge_extra_body(payload, {"extra_body": extra_body})

    # Optionally filter unknown/non-standard keys for strict OpenAI-compatible servers
    if filter_unknown_params:
        allowed_keys = {
            "messages",
            "model",
            "temperature",
            "top_p",
            "max_tokens",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
            "response_format",
            "tools",
            "tool_choice",
            "logprobs",
            "top_logprobs",
            "user",
            "stream",
        }
        payload = {k: v for k, v in payload.items() if k in allowed_keys}

    cache_diagnostic = build_local_cache_diagnostic(
        provider=provider_name,
        request={
            "extra_body": extra_body,
            "inference_prefix_cache_intent": inference_prefix_cache_intent,
        },
        payload=payload,
        app_config=app_config,
    )

    def attach_cache_diagnostics(data: Any) -> Any:
        if isinstance(data, dict) and cache_diagnostic.has_signal:
            data.setdefault("tldw_local_cache_diagnostics", cache_diagnostic.to_metadata())
        return data

    # Construct full API URL for chat completions
    chat_completions_path = "v1/chat/completions" # Standard OpenAI path
    normalized_base = (api_base_url or "").strip()
    if not normalized_base:
        raise ChatConfigurationError(provider=provider_name, message=f"{provider_name} API base URL is required.")
    normalized_base = normalized_base.rstrip("/")
    lower_base = normalized_base.lower()

    if "chat/completions" in lower_base or lower_base.endswith("/completion"):
        full_api_url = normalized_base
    elif lower_base.endswith("/v1"):
        full_api_url = normalized_base + "/chat/completions"
    else:
        full_api_url = normalized_base + "/" + chat_completions_path

    logging.debug(
        "{}: Posting local chat request. Payload keys: {}",
        safe_provider_name,
        list(payload.keys()),
    )
    payload_metadata = _sanitize_payload_for_logging(payload)
    logging.debug("{}: Payload metadata: {}", safe_provider_name, payload_metadata)


    # All requests, including tests, use the checked central transport.
    session_factory = http_client_factory or _hc_create_client

    def create_session() -> Any:
        try:
            return session_factory(timeout=timeout)
        except TypeError:
            return session_factory(timeout)

    if streaming:

        def stream_generator():
            upstream_done = False
            error_pending = False
            response_obj = None
            session = None
            try:
                try:
                    session = create_session()
                    logging.debug("{}: Opening streaming connection", safe_provider_name)
                    stream_impl = http_streamer or _hc_stream_response
                    with stream_impl(
                        method="POST",
                        url=full_api_url,
                        configured_endpoint=configured_endpoint_scope,
                        client=session,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    ) as response:
                        response_obj = response
                        response.raise_for_status()
                        logging.debug("{}: Streaming response received", safe_provider_name)
                        for line in response.iter_lines():
                            if not line:
                                continue
                            decoded = (
                                line.decode("utf-8", errors="replace")
                                if isinstance(line, (bytes, bytearray))
                                else str(line)
                            )
                            if is_done_line(decoded):
                                upstream_done = True
                                break
                            if _is_local_provider_stream_error_line(decoded):
                                error_pending = True
                                break
                            normalized = normalize_provider_line(decoded)
                            if normalized is not None:
                                yield normalized
                except EgressPolicyError:
                    raise
                except Exception as exc:  # noqa: BLE001 - sanitize arbitrary provider failures
                    if not upstream_done:
                        _log_local_provider_failure(
                            safe_provider_name,
                            exc,
                            phase="stream_request",
                        )
                        error_pending = True

                if error_pending:
                    yield _local_provider_stream_error_frame(safe_provider_name)
                # Defer the terminal marker until the response context exits. This
                # prevents cleanup failures from producing DONE -> error -> DONE.
                yield from finalize_stream(response_obj, done_already=False)
            finally:
                if session is not None:
                    _close_local_provider_resource(
                        session,
                        safe_provider_name,
                        phase="stream_client_close",
                    )

        return stream_generator()

    session = None
    try:
        session = create_session()
        attempts = max(1, int(api_retries) + 1)
        base_ms = max(50, int(api_retry_delay * 1000))
        policy = _HC_RetryPolicy(attempts=attempts, backoff_base_ms=base_ms)
        fetch_impl = http_fetcher or _hc_fetch
        response = fetch_impl(
            method="POST",
            url=full_api_url,
            configured_endpoint=configured_endpoint_scope,
            client=session,
            headers=headers,
            json=payload,
            timeout=timeout,
            retry=policy,
        )
        try:
            response.raise_for_status()
            data = response.json()
            raise_if_in_band_provider_error(
                safe_provider_name,
                data,
                phase="nonstream_response",
            )
            logging.debug("{}: Non-streaming request successful", safe_provider_name)
            return attach_cache_diagnostics(data)
        finally:
            _close_local_provider_resource(
                response,
                safe_provider_name,
                phase="nonstream_response_close",
            )
    except EgressPolicyError:
        raise
    except Exception as exc:  # noqa: BLE001 - sanitize arbitrary provider failures
        _raise_sanitized_local_provider_error(
            safe_provider_name,
            exc,
            phase="nonstream_request",
        )
    finally:
        if session is not None:
            _close_local_provider_resource(
                session,
                safe_provider_name,
                phase="nonstream_client_close",
            )


def _local_llm_request(
        input_data: list[dict[str, Any]],
        temp: float | None = None,
        temperature: float | None = None,
        system_message: str | None = None,
        streaming: bool | None = None,
        stream: bool | None = None,
        model: str | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        # Note: custom_prompt_arg is legacy-only; OpenAI-compatible servers expect prompts in messages.
        # It's better handled by the `chat` function by prepending to the user message if needed.
        # For now, we assume it's already part of input_data or handled by system_message.
        custom_prompt_arg: str | None = None, # Mapped from 'prompt'
         # Adding other OpenAI compatible params from your map if this server type is meant to be generic OpenAI
        response_format: dict[str, str] | None = None,
        n: int | None = None,
        user_identifier: str | None = None,
        logit_bias: dict[str, float] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        app_config: dict[str, Any] | None = None,
        http_client_factory: Callable[[int], Any] | None = None,
        http_fetcher: Callable[..., Any] | None = None,
        http_streamer: Callable[..., Any] | None = None,
        configured_endpoint_base_url: str | None = None,
        configured_endpoint_scope: ConfiguredEndpointScope | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        api_key: str | None = None,
        credentials_resolved: bool = False,
        timeout: float | None = None,
):
    if temperature is not None:
        if temp is not None and temp != temperature:
            logging.warning("local_llm: Received both 'temp' and 'temperature'; using 'temp'")
        else:
            temp = temperature
    if stream is not None:
        if streaming is not None and streaming != stream:
            logging.warning("local_llm: Received both 'streaming' and 'stream'; preferring explicit 'stream' value")
        streaming = stream
    if model and (model.lower() == "none" or model.strip() == ""):
        model = None
    loaded_config_data = _select_local_app_config(
        app_config,
        credentials_resolved=credentials_resolved,
    )
    cfg_section = 'local_llm' # Generic section for "local-llm" type
    cfg = loaded_config_data.get(cfg_section, {})

    api_base_url = configured_endpoint_base_url or cfg.get('api_ip', 'http://127.0.0.1:8080')
    current_api_key = api_key or cfg.get('api_key') # Local servers might not need a key

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


    effective_timeout = timeout if timeout is not None else int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str):
        current_streaming = current_streaming.lower() == "true"
    if isinstance(current_logprobs, str):
        current_logprobs = current_logprobs.lower() == "true"
    # Coerce numeric/string config values to correct types
    try:
        if isinstance(current_top_p, str):
            current_top_p = float(current_top_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("local_llm: Failed to coerce top_p='%s' to float; sending as-is", current_top_p)
    try:
        if isinstance(current_top_k, str):
            current_top_k = int(current_top_k)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("local_llm: Failed to coerce top_k='%s' to int; sending as-is", current_top_k)
    try:
        if isinstance(current_min_p, str):
            current_min_p = float(current_min_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("local_llm: Failed to coerce min_p='%s' to float; sending as-is", current_min_p)
    try:
        if isinstance(current_seed, str):
            current_seed = int(current_seed)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("local_llm: Failed to coerce seed='%s' to int; sending as-is", current_seed)
    try:
        if isinstance(current_presence_penalty, str):
            current_presence_penalty = float(current_presence_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("local_llm: Failed to coerce presence_penalty='%s' to float; sending as-is", current_presence_penalty)
    try:
        if isinstance(current_frequency_penalty, str):
            current_frequency_penalty = float(current_frequency_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("local_llm: Failed to coerce frequency_penalty='%s' to float; sending as-is", current_frequency_penalty)
    try:
        if isinstance(current_n, str):
            current_n = int(current_n)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("local_llm: Failed to coerce n='%s' to int; sending as-is", current_n)
    try:
        if isinstance(current_top_logprobs, str):
            current_top_logprobs = int(current_top_logprobs)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("local_llm: Failed to coerce top_logprobs='%s' to int; sending as-is", current_top_logprobs)

    if custom_prompt_arg:
        logging.info(f"{cfg_section}: 'custom_prompt_arg' received. Ensure it's incorporated into 'input_data' or 'system_message' by the calling function if intended for the prompt, as this handler uses OpenAI message format.")

    return _chat_with_openai_compatible_local_server(
        api_base_url=api_base_url,
        model_name=current_model,
        input_data=input_data,
        api_key=current_api_key,
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
        timeout=effective_timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay,
        filter_unknown_params=bool(cfg.get('strict_openai_compat', False)),
        http_client_factory=http_client_factory,
        http_fetcher=http_fetcher,
        http_streamer=http_streamer,
        configured_endpoint_scope=configured_endpoint_scope,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )



def _llama_request(
        input_data: list[dict[str, Any]],
        api_key: str | None = None, # from map
        custom_prompt: str | None = None,  # from map, Mapped from 'prompt'
        temp: float | None = None, # from map, generic name is 'temperature'
        temperature: float | None = None,
        system_prompt: str | None = None,  # from map, Mapped from 'system_message'
        streaming: bool | None = None, # from map
        stream: bool | None = None, # alias from provider map
        model: str | None = None, # from map
        top_k: int | None = None, # from map
        top_p: float | None = None, # from map
        min_p: float | None = None, # from map
        n_predict: int | None = None, # from map, mapped from max_tokens
        seed: int | None = None, # from map
        stop: str | list[str] | None = None, # from map
        response_format: dict[str, str] | None = None, # from map
        logit_bias: dict[str, float] | None = None, # from map
        n: int | None = None, # from map, number of completions to request
        presence_penalty: float | None = None, # from map
        frequency_penalty: float | None = None, # from map
        # api_url is tricky. Your notes say "positional argument".
        # If chat_api_call is the sole entry, this needs to be passed via kwargs if mapped,
        # or loaded from config if not passed. Let's assume it's primarily from config for now.
        api_url: str | None = None, # Used by legacy dispatch when special handling exists
        app_config: dict[str, Any] | None = None,
        http_client_factory: Callable[[int], Any] | None = None,
        http_fetcher: Callable[..., Any] | None = None,
        http_streamer: Callable[..., Any] | None = None,
        configured_endpoint_base_url: str | None = None,
        configured_endpoint_scope: ConfiguredEndpointScope | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        inference_prefix_cache_intent: dict[str, Any] | None = None,
        credentials_resolved: bool = False,
        timeout: float | None = None,
):
    if temperature is not None:
        if temp is not None and temp != temperature:
            logging.warning("Llama.cpp: Received both 'temp' and 'temperature'; using 'temp' value")
        else:
            temp = temperature
    if stream is not None:
        if streaming is not None and streaming != stream:
            logging.warning("Llama.cpp: Received both 'streaming' and 'stream'; preferring explicit 'stream' value")
        streaming = stream
    if model and (model.lower() == "none" or model.strip() == ""):
        model = None
    loaded_config_data = _select_local_app_config(
        app_config,
        credentials_resolved=credentials_resolved,
    )
    cfg = loaded_config_data.get('llama_api', {})

    current_api_base_url = configured_endpoint_base_url or cfg.get('api_ip')
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

    # Handle multiple completions: llama.cpp's OpenAI-compatible server accepts 'n'.
    current_n = n if n is not None else cfg.get('n', cfg.get('n_probs'))


    effective_timeout = timeout if timeout is not None else int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str):
        current_streaming = current_streaming.lower() == "true"
    # Coerce numeric/string config values to correct types
    try:
        if isinstance(current_top_p, str):
            current_top_p = float(current_top_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Llama.cpp: Failed to coerce top_p='%s' to float; sending as-is", current_top_p)
    try:
        if isinstance(current_top_k, str):
            current_top_k = int(current_top_k)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Llama.cpp: Failed to coerce top_k='%s' to int; sending as-is", current_top_k)
    try:
        if isinstance(current_min_p, str):
            current_min_p = float(current_min_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Llama.cpp: Failed to coerce min_p='%s' to float; sending as-is", current_min_p)
    try:
        if isinstance(current_seed, str):
            current_seed = int(current_seed)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Llama.cpp: Failed to coerce seed='%s' to int; sending as-is", current_seed)
    try:
        if isinstance(current_presence_penalty, str):
            current_presence_penalty = float(current_presence_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Llama.cpp: Failed to coerce presence_penalty='%s' to float; sending as-is", current_presence_penalty)
    try:
        if isinstance(current_frequency_penalty, str):
            current_frequency_penalty = float(current_frequency_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Llama.cpp: Failed to coerce frequency_penalty='%s' to float; sending as-is", current_frequency_penalty)
    try:
        if isinstance(current_n, str):
            current_n = int(current_n)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Llama.cpp: Failed to coerce n='%s' to int; sending as-is", current_n)
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
        n=current_n,
        stop=current_stop,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        logit_bias=current_logit_bias,
        seed=current_seed,
        response_format=current_response_format,
        # tools, tool_choice, logprobs, top_logprobs, user_identifier could be added if llama.cpp supports them via OpenAI compat layer
        provider_name="Llama.cpp",
        timeout=effective_timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay,
        filter_unknown_params=bool(cfg.get('strict_openai_compat', False)),
        http_client_factory=http_client_factory,
        http_fetcher=http_fetcher,
        http_streamer=http_streamer,
        configured_endpoint_scope=configured_endpoint_scope,
        extra_headers=extra_headers,
        extra_body=extra_body,
        app_config=loaded_config_data,
        inference_prefix_cache_intent=inference_prefix_cache_intent,
    )



# System prompts not supported through API requests.
# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def _kobold_request(
        input_data: list[dict[str, Any]],
        api_key: str | None = None,
        custom_prompt_input: str | None = None, # Mapped from 'prompt'
        temp: float | None = None, # Mapped from 'temp'
        system_message: str | None = None, # Mapped
        streaming: bool | None = False, # Mapped
        model: str | None = None, # Mapped
        top_k: int | None = None, # Mapped
        top_p: float | None = None, # Mapped
        max_length: int | None = None, # Mapped from 'max_tokens'
        stop_sequence: str | list[str] | None = None, # Mapped from 'stop'
        num_responses: int | None = None, # Mapped from 'n'
        seed: int | None = None, # Mapped from 'seed'
        app_config: dict[str, Any] | None = None,
        http_client_factory: Callable[[int], Any] | None = None,
        http_fetcher: Callable[..., Any] | None = None,
        http_streamer: Callable[..., Any] | None = None,
        configured_endpoint_base_url: str | None = None,
        configured_endpoint_scope: ConfiguredEndpointScope | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        credentials_resolved: bool = False,
        timeout: float | None = None,
):
    if model and (model.lower() == "none" or model.strip() == ""):
        model = None
    logging.debug("KoboldAI (Native): Chat request starting...")
    loaded_config_data = _select_local_app_config(
        app_config,
        credentials_resolved=credentials_resolved,
    )
    cfg = loaded_config_data.get('kobold_api', {})

    current_api_key = api_key or cfg.get('api_key')
    api_url = configured_endpoint_base_url or cfg.get('api_ip')
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

    # Coerce numeric/string config values to correct types
    try:
        if isinstance(current_top_p, str):
            current_top_p = float(current_top_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Kobold: Failed to coerce top_p='%s' to float; sending as-is", current_top_p)
    try:
        if isinstance(current_top_k, str):
            current_top_k = int(current_top_k)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Kobold: Failed to coerce top_k='%s' to int; sending as-is", current_top_k)
    try:
        if isinstance(current_num_responses, str):
            current_num_responses = int(current_num_responses)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Kobold: Failed to coerce num_responses='%s' to int; sending as-is", current_num_responses)
    try:
        if isinstance(current_seed, str):
            current_seed = int(current_seed)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Kobold: Failed to coerce seed='%s' to int; sending as-is", current_seed)

    max_context_length = int(cfg.get('max_context_length', 2048)) # Kobold uses max_context_length for context window
    effective_timeout = timeout if timeout is not None else int(cfg.get('api_timeout', 180))
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
    if current_api_key:
        headers['X-Api-Key'] = current_api_key # Some Kobold forks might use this
    headers = merge_extra_headers(headers, {"extra_headers": extra_headers})

    payload: dict[str, Any] = {
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
    if current_stop_sequence is not None:
        payload['stop_sequence'] = current_stop_sequence # List of strings
    if current_num_responses is not None:
        payload['n'] = current_num_responses # Number of responses
    if current_seed is not None:
        payload['seed'] = current_seed
    payload = merge_extra_body(payload, {"extra_body": extra_body})

    # Kobold specific params (can be added from cfg if needed and supported)
    if cfg.get('rep_pen') is not None:
        payload['rep_pen'] = float(cfg['rep_pen'])
    # Other kobold params: typical_p, tfs, top_a, etc. could be added from cfg

    logging.debug(
        "KoboldAI (Native): Posting request. prompt_length={} chars",
        len(final_prompt_string),
    )
    payload_metadata = _sanitize_payload_for_logging(
        payload,
        message_keys=(),
        text_keys=("prompt",),
    )
    logging.debug(f"KoboldAI (Native) payload metadata: {payload_metadata}")


    try:
        policy = _HC_RetryPolicy(attempts=max(1, int(api_retries) + 1), backoff_base_ms=max(50, int(api_retry_delay * 1000)))
        fetch_impl = http_fetcher or _hc_fetch
        response = fetch_impl(
            method="POST",
            url=api_url,
            configured_endpoint=configured_endpoint_scope,
            headers=headers,
            json=payload,
            retry=policy,
            timeout=effective_timeout,
        )
        try:
            response.raise_for_status()
            response_data = response.json()
            raise_if_in_band_provider_error(
                "kobold",
                response_data,
                phase="nonstream_response",
            )
        finally:
            _close_local_provider_resource(
                response,
                "kobold",
                phase="nonstream_response_close",
            )

        if response_data and 'results' in response_data and len(response_data['results']) > 0:
            # Kobold /generate usually returns a list of results, each with 'text'
            # If n > 1, there might be multiple. For now, taking the first.
            generated_text = response_data['results'][0].get('text', '').strip()
            logging.debug("KoboldAI (Native): Chat request successful.")
            # To make it somewhat OpenAI-like for the dispatcher, wrap in a choices structure.
            # This assumes non-streaming. Streaming would need a generator yielding SSE-like events.
            return {"choices": [{"message": {"role": "assistant", "content": generated_text}, "finish_reason": "stop"}]} # Assuming "stop"
        else:
            raise ChatProviderError(
                provider="kobold",
                message="KoboldAI returned an invalid response.",
            )

    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS as e_http:
        _raise_sanitized_local_provider_error(
            "kobold",
            e_http,
            phase="request",
        )


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
# Oobabooga with OpenAI extension
def _ooba_request(
    input_data: list[dict[str, Any]],
    api_key: str | None = None, # from map
    custom_prompt: str | None = None,  # from map, Mapped from 'prompt'
    temp: float | None = None, # from map, generic name 'temperature'
    temperature: float | None = None,
    system_prompt: str | None = None,  # from map, Mapped from 'system_message'
    streaming: bool | None = None, # from map
    stream: bool | None = None,
    model: str | None = None, # from map
    top_k: int | None = None, # from map
    top_p: float | None = None, # from map (ooba might use 'top_p')
    min_p: float | None = None, # from map
    max_tokens: int | None = None, # from map
    seed: int | None = None, # from map
    stop: str | list[str] | None = None, # from map
    response_format: dict[str, str] | None = None, # from map
    n: int | None = None, # from map
    user_identifier: str | None = None, # from map
    logit_bias: dict[str, float] | None = None, # from map
    presence_penalty: float | None = None, # from map
    frequency_penalty: float | None = None, # from map
    api_url: str | None = None, # Specific, not from generic map unless handled
    app_config: dict[str, Any] | None = None,
    http_client_factory: Callable[[int], Any] | None = None,
    http_fetcher: Callable[..., Any] | None = None,
    http_streamer: Callable[..., Any] | None = None,
    configured_endpoint_base_url: str | None = None,
    configured_endpoint_scope: ConfiguredEndpointScope | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    credentials_resolved: bool = False,
    timeout: float | None = None,
):
    if temperature is not None:
        if temp is not None and temp != temperature:
            logging.warning("Oobabooga: Received both 'temp' and 'temperature'; using 'temp' value")
        else:
            temp = temperature
    if stream is not None:
        if streaming is not None and streaming != stream:
            logging.warning("Oobabooga: Received both 'streaming' and 'stream'; preferring explicit 'stream' value")
        streaming = stream
    if model and (model.lower() == "none" or model.strip() == ""):
        model = None
    loaded_config_data = _select_local_app_config(
        app_config,
        credentials_resolved=credentials_resolved,
    )
    cfg = loaded_config_data.get('ooba_api', {})

    current_api_base_url = configured_endpoint_base_url or cfg.get('api_ip')
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

    effective_timeout = timeout if timeout is not None else int(cfg.get('api_timeout', 180)) # Ooba can be slow
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str):
        current_streaming = current_streaming.lower() == "true"
    # Coerce numeric/string config values to correct types
    try:
        if isinstance(current_top_p, str):
            current_top_p = float(current_top_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Oobabooga: Failed to coerce top_p='%s' to float; sending as-is", current_top_p)
    try:
        if isinstance(current_top_k, str):
            current_top_k = int(current_top_k)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Oobabooga: Failed to coerce top_k='%s' to int; sending as-is", current_top_k)
    try:
        if isinstance(current_min_p, str):
            current_min_p = float(current_min_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Oobabooga: Failed to coerce min_p='%s' to float; sending as-is", current_min_p)
    try:
        if isinstance(current_seed, str):
            current_seed = int(current_seed)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Oobabooga: Failed to coerce seed='%s' to int; sending as-is", current_seed)
    try:
        if isinstance(current_presence_penalty, str):
            current_presence_penalty = float(current_presence_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Oobabooga: Failed to coerce presence_penalty='%s' to float; sending as-is", current_presence_penalty)
    try:
        if isinstance(current_frequency_penalty, str):
            current_frequency_penalty = float(current_frequency_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Oobabooga: Failed to coerce frequency_penalty='%s' to float; sending as-is", current_frequency_penalty)
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
        timeout=effective_timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay,
        filter_unknown_params=bool(cfg.get('strict_openai_compat', False)),
        http_client_factory=http_client_factory,
        http_fetcher=http_fetcher,
        http_streamer=http_streamer,
        configured_endpoint_scope=configured_endpoint_scope,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )


# TabbyAPI (seems OpenAI compatible)
def _tabbyapi_request(
    input_data: list[dict[str, Any]],
    api_key: str | None = None, # from map
    custom_prompt_input: str | None = None, # from map ('prompt')
    temp: float | None = None, # from map (mapped to 'temperature' in generic)
    temperature: float | None = None,
    system_message: str | None = None, # from map
    streaming: bool | None = None, # from map
    stream: bool | None = None,
    model: str | None = None, # from map
    top_k: int | None = None, # from map
    top_p: float | None = None, # from map
    min_p: float | None = None, # from map
    max_tokens: int | None = None, # from map
    seed: int | None = None, # from map
    stop: str | list[str] | None = None, # from map
    app_config: dict[str, Any] | None = None,
    # Additional OpenAI-compatible params (pass-through if supported by server)
    response_format: dict[str, str] | None = None,
    n: int | None = None,
    user_identifier: str | None = None,
    logit_bias: dict[str, float] | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    api_url: str | None = None,
    http_client_factory: Callable[[int], Any] | None = None,
    http_fetcher: Callable[..., Any] | None = None,
    http_streamer: Callable[..., Any] | None = None,
    configured_endpoint_base_url: str | None = None,
    configured_endpoint_scope: ConfiguredEndpointScope | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    credentials_resolved: bool = False,
    timeout: float | None = None,
):
    if temperature is not None:
        if temp is not None and temp != temperature:
            logging.warning("TabbyAPI: Received both 'temp' and 'temperature'; using 'temp' value")
        else:
            temp = temperature
    if stream is not None:
        if streaming is not None and streaming != stream:
            logging.warning("TabbyAPI: Received both 'streaming' and 'stream'; preferring explicit 'stream' value")
        streaming = stream
    if model and (model.lower() == "none" or model.strip() == ""):
        model = None
    loaded_config_data = _select_local_app_config(
        app_config,
        credentials_resolved=credentials_resolved,
    )
    cfg = loaded_config_data.get('tabby_api', {})

    api_base_url = configured_endpoint_base_url or cfg.get('api_ip')
    if not api_base_url:
        raise ChatConfigurationError(provider="tabbyapi", message="TabbyAPI URL (api_ip) is required.")

    current_api_key = api_key or cfg.get('api_key')
    current_model = model or cfg.get('model')
    # Accept both temp/temperature from legacy callers; prefer temp when both present.
    current_temp_val = temp if temp is not None else float(cfg.get('temperature', cfg.get('temp', 0.7)))


    current_streaming = streaming if streaming is not None else cfg.get('streaming', False)
    current_top_k = top_k if top_k is not None else cfg.get('top_k')
    current_top_p = top_p if top_p is not None else cfg.get('top_p')
    current_min_p = min_p if min_p is not None else cfg.get('min_p')
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

    effective_timeout = timeout if timeout is not None else int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str):
        current_streaming = current_streaming.lower() == "true"
    # Coerce numeric/string config values to correct types
    try:
        if isinstance(current_top_p, str):
            current_top_p = float(current_top_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("TabbyAPI: Failed to coerce top_p='%s' to float; sending as-is", current_top_p)
    try:
        if isinstance(current_top_k, str):
            current_top_k = int(current_top_k)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("TabbyAPI: Failed to coerce top_k='%s' to int; sending as-is", current_top_k)
    try:
        if isinstance(current_min_p, str):
            current_min_p = float(current_min_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("TabbyAPI: Failed to coerce min_p='%s' to float; sending as-is", current_min_p)
    try:
        if isinstance(current_seed, str):
            current_seed = int(current_seed)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("TabbyAPI: Failed to coerce seed='%s' to int; sending as-is", current_seed)
    try:
        if isinstance(current_presence_penalty, str):
            current_presence_penalty = float(current_presence_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("TabbyAPI: Failed to coerce presence_penalty='%s' to float; sending as-is", current_presence_penalty)
    try:
        if isinstance(current_frequency_penalty, str):
            current_frequency_penalty = float(current_frequency_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("TabbyAPI: Failed to coerce frequency_penalty='%s' to float; sending as-is", current_frequency_penalty)
    try:
        if isinstance(current_n, str):
            current_n = int(current_n)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("TabbyAPI: Failed to coerce n='%s' to int; sending as-is", current_n)
    try:
        if isinstance(current_top_logprobs, str):
            current_top_logprobs = int(current_top_logprobs)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("TabbyAPI: Failed to coerce top_logprobs='%s' to int; sending as-is", current_top_logprobs)
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
        response_format=current_response_format,
        n=current_n,
        user_identifier=current_user_identifier,
        logit_bias=current_logit_bias,
        presence_penalty=current_presence_penalty,
        frequency_penalty=current_frequency_penalty,
        logprobs=current_logprobs,
        top_logprobs=current_top_logprobs,
        tools=current_tools,
        tool_choice=current_tool_choice,
        provider_name="TabbyAPI",
        timeout=effective_timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay,
        filter_unknown_params=bool(cfg.get('strict_openai_compat', False)),
        http_client_factory=http_client_factory,
        http_fetcher=http_fetcher,
        http_streamer=http_streamer,
        configured_endpoint_scope=configured_endpoint_scope,
        extra_headers=extra_headers,
        extra_body=extra_body,
        # Add other OpenAI params here if TabbyAPI supports them
    )


# vLLM (OpenAI compatible)
def _vllm_request(
    input_data: list[dict[str, Any]],
    api_key: str | None = None, # from map
    custom_prompt_input: str | None = None, # from map ('prompt')
    temp: float | None = None,
    # vLLM's map has 'temp':'temperature', 'system_prompt':'system_message' etc.
    # These are the provider-specific names this function receives.
    temperature: float | None = None, # from map (mapped from generic 'temp')
    system_prompt: str | None = None,   # from map (mapped from generic 'system_message')
    streaming: bool | None = None,   # from map
    stream: bool | None = None,
    model: str | None = None,         # from map
    top_k: int | None = None,         # from map
    top_p: float | None = None,         # from map (mapped from generic 'topp')
    min_p: float | None = None,         # from map (mapped from generic 'minp')
    max_tokens: int | None = None,      # from map
    seed: int | None = None,          # from map
    stop: str | list[str] | None = None, # from map
    response_format: dict[str, str] | None = None, # from map
    n: int | None = None,             # from map
    logit_bias: dict[str, float] | None = None, # from map
    presence_penalty: float | None = None, # from map
    frequency_penalty: float | None = None, # from map
    logprobs: bool | None = None,     # from map
    user_identifier: str | None = None, # from map
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    top_logprobs: int | None = None,
    vllm_api_url: str | None = None, # Specific config, not from generic map typically
    app_config: dict[str, Any] | None = None,
    http_client_factory: Callable[[int], Any] | None = None,
    http_fetcher: Callable[..., Any] | None = None,
    http_streamer: Callable[..., Any] | None = None,
    configured_endpoint_base_url: str | None = None,
    configured_endpoint_scope: ConfiguredEndpointScope | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    inference_prefix_cache_intent: dict[str, Any] | None = None,
    credentials_resolved: bool = False,
    timeout: float | None = None,
                                       # Could be loaded from cfg or passed if chat_api_call handles it
):
    if temp is not None:
        if temperature is not None and temperature != temp:
            logging.warning("vLLM: Received both 'temp' and 'temperature'; using 'temp' value")
        temperature = temp
    if stream is not None:
        if streaming is not None and streaming != stream:
            logging.warning("vLLM: Received both 'streaming' and 'stream'; preferring explicit 'stream' value")
        streaming = stream
    if model and (model.lower() == "none" or model.strip() == ""):
        model = None
    loaded_config_data = _select_local_app_config(
        app_config,
        credentials_resolved=credentials_resolved,
    )
    cfg = loaded_config_data.get('vllm_api', {})

    # vllm_api_url is a specific argument for this function if it's set up in legacy dispatch
    # otherwise, it falls back to config.
    current_api_base_url = configured_endpoint_base_url or cfg.get('api_ip')
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
    current_top_logprobs = top_logprobs if top_logprobs is not None else cfg.get('top_logprobs')
    current_tools = tools if tools is not None else cfg.get('tools')
    current_tool_choice = tool_choice if tool_choice is not None else cfg.get('tool_choice')
    # If vLLM supports top_logprobs, keep it in the signature and pass through.
    current_user_identifier = user_identifier if user_identifier is not None else cfg.get('user_identifier')


    effective_timeout = timeout if timeout is not None else int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str):
        current_streaming = current_streaming.lower() == "true"
    # Coerce numeric/string config values to correct types
    try:
        if isinstance(current_top_p, str):
            current_top_p = float(current_top_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("vLLM: Failed to coerce top_p='%s' to float; sending as-is", current_top_p)
    try:
        if isinstance(current_top_k, str):
            current_top_k = int(current_top_k)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("vLLM: Failed to coerce top_k='%s' to int; sending as-is", current_top_k)
    try:
        if isinstance(current_min_p, str):
            current_min_p = float(current_min_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("vLLM: Failed to coerce min_p='%s' to float; sending as-is", current_min_p)
    try:
        if isinstance(current_seed, str):
            current_seed = int(current_seed)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("vLLM: Failed to coerce seed='%s' to int; sending as-is", current_seed)
    try:
        if isinstance(current_presence_penalty, str):
            current_presence_penalty = float(current_presence_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("vLLM: Failed to coerce presence_penalty='%s' to float; sending as-is", current_presence_penalty)
    try:
        if isinstance(current_frequency_penalty, str):
            current_frequency_penalty = float(current_frequency_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("vLLM: Failed to coerce frequency_penalty='%s' to float; sending as-is", current_frequency_penalty)
    try:
        if isinstance(current_n, str):
            current_n = int(current_n)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("vLLM: Failed to coerce n='%s' to int; sending as-is", current_n)
    try:
        if isinstance(current_top_logprobs, str):
            current_top_logprobs = int(current_top_logprobs)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("vLLM: Failed to coerce top_logprobs='%s' to int; sending as-is", current_top_logprobs)
    if isinstance(current_logprobs, str):
        current_logprobs = current_logprobs.lower() == "true"
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
        top_logprobs=current_top_logprobs,
        tools=current_tools,
        tool_choice=current_tool_choice,
        user_identifier=current_user_identifier,
        provider_name="vLLM",
        timeout=effective_timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay,
        filter_unknown_params=bool(cfg.get('strict_openai_compat', False)),
        http_client_factory=http_client_factory,
        http_fetcher=http_fetcher,
        http_streamer=http_streamer,
        configured_endpoint_scope=configured_endpoint_scope,
        extra_headers=extra_headers,
        extra_body=extra_body,
        app_config=loaded_config_data,
        inference_prefix_cache_intent=inference_prefix_cache_intent,
        # tools, tool_choice for vLLM? If supported, add to map and pass.
    )


# Aphrodite (seems to be an OpenAI compatible engine)
def _aphrodite_request(
    input_data: list[dict[str, Any]],
    api_key: str | None = None, # from map
    custom_prompt: str | None = None,  # from map ('prompt')
    # Aphrodite's map uses 'temp':'temperature', etc.
    temp: float | None = None,
    temperature: float | None = None, # from map (mapped from generic 'temp')
    system_message: str | None = None, # from map
    streaming: bool | None = None,   # from map
    stream: bool | None = None,
    model: str | None = None,         # from map
    top_k: int | None = None,         # from map
    top_p: float | None = None,         # from map (mapped from generic 'topp')
    min_p: float | None = None,         # from map (mapped from generic 'minp')
    max_tokens: int | None = None,      # from map
    seed: int | None = None,          # from map
    stop: str | list[str] | None = None, # from map
    response_format: dict[str, str] | None = None, # from map
    n: int | None = None,             # from map
    logit_bias: dict[str, float] | None = None, # from map
    presence_penalty: float | None = None, # legacy alias
    frequency_penalty: float | None = None, # legacy alias
    logprobs: bool | None = None,     # from map
    user_identifier: str | None = None, # from map
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    top_logprobs: int | None = None,
    api_url: str | None = None,
    app_config: dict[str, Any] | None = None,
    http_client_factory: Callable[[int], Any] | None = None,
    http_fetcher: Callable[..., Any] | None = None,
    http_streamer: Callable[..., Any] | None = None,
    configured_endpoint_base_url: str | None = None,
    configured_endpoint_scope: ConfiguredEndpointScope | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    credentials_resolved: bool = False,
    timeout: float | None = None,
    # top_logprobs, tools, tool_choice not in Aphrodite's map currently
):
    if temp is not None:
        if temperature is not None and temperature != temp:
            logging.warning("Aphrodite: Received both 'temp' and 'temperature'; using 'temp' value")
        temperature = temp
    if stream is not None:
        if streaming is not None and streaming != stream:
            logging.warning("Aphrodite: Received both 'streaming' and 'stream'; preferring explicit 'stream' value")
        streaming = stream
    if model and (model.lower() == "none" or model.strip() == ""):
        model = None
    loaded_config_data = _select_local_app_config(
        app_config,
        credentials_resolved=credentials_resolved,
    )
    cfg = loaded_config_data.get('aphrodite_api', {})

    api_base_url = configured_endpoint_base_url or cfg.get('api_ip')
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

    effective_timeout = timeout if timeout is not None else int(cfg.get('api_timeout', 120))
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str):
        current_streaming = current_streaming.lower() == "true"
    if isinstance(current_logprobs, str):
        current_logprobs = current_logprobs.lower() == "true"
    # Coerce numeric/string config values to correct types
    try:
        if isinstance(current_top_p, str):
            current_top_p = float(current_top_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Aphrodite: Failed to coerce top_p='%s' to float; sending as-is", current_top_p)
    try:
        if isinstance(current_top_k, str):
            current_top_k = int(current_top_k)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Aphrodite: Failed to coerce top_k='%s' to int; sending as-is", current_top_k)
    try:
        if isinstance(current_min_p, str):
            current_min_p = float(current_min_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Aphrodite: Failed to coerce min_p='%s' to float; sending as-is", current_min_p)
    try:
        if isinstance(current_seed, str):
            current_seed = int(current_seed)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Aphrodite: Failed to coerce seed='%s' to int; sending as-is", current_seed)
    try:
        if isinstance(current_presence_penalty, str):
            current_presence_penalty = float(current_presence_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Aphrodite: Failed to coerce presence_penalty='%s' to float; sending as-is", current_presence_penalty)
    try:
        if isinstance(current_frequency_penalty, str):
            current_frequency_penalty = float(current_frequency_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Aphrodite: Failed to coerce frequency_penalty='%s' to float; sending as-is", current_frequency_penalty)
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
        top_logprobs=top_logprobs,
        tools=tools,
        tool_choice=tool_choice,
        user_identifier=current_user_identifier,
        provider_name="Aphrodite Engine",
        timeout=effective_timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay,
        filter_unknown_params=bool(cfg.get('strict_openai_compat', False)),
        http_client_factory=http_client_factory,
        http_fetcher=http_fetcher,
        http_streamer=http_streamer,
        configured_endpoint_scope=configured_endpoint_scope,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )


# Ollama (with OpenAI compatible endpoint)
def _ollama_request(
    input_data: list[dict[str, Any]],
    api_key: str | None = None, # from map, Ollama doesn't use key but map has it
    custom_prompt: str | None = None,  # from map ('prompt')
    # Ollama map: 'temp':'temperature', 'system_message':'system_message', 'topp':'top_p', etc.
    temp: float | None = None,
    temperature: float | None = None,  # from map (mapped from generic 'temp')
    system_message: str | None = None, # from map
    # Back-compat alias if any caller passed 'system'
    system: str | None = None,
    model: str | None = None,          # from map
    streaming: bool | None = None,    # from map
    stream: bool | None = None,
    top_p: float | None = None,          # from map (mapped from generic 'topp')
    top_k: int | None = None,          # from map
    # Ollama specific params from map, ensure they are OpenAI compatible if passed to generic func
    num_predict: int | None = None,      # from map (mapped from generic 'max_tokens')
    # Back-compat alias from some direct callers
    max_tokens: int | None = None,
    seed: int | None = None,             # from map
    stop: str | list[str] | None = None, # from map
    format_str: str | dict[str, Any] | None = None,       # from map (mapped from generic 'response_format', e.g. "json" or {'type': 'json_object'})
    # Back-compat alias if any caller passed 'format'
    format: str | None = None,
                                            # _chat_with_openai_compatible_local_server expects dict {"type": "json_object"}
    presence_penalty: float | None = None, # from map
    frequency_penalty: float | None = None, # from map
    # api_url is specific for Ollama if passed directly, else from config
    api_url: str | None = None,
    user_identifier: str | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    app_config: dict[str, Any] | None = None,
    http_client_factory: Callable[[int], Any] | None = None,
    http_fetcher: Callable[..., Any] | None = None,
    http_streamer: Callable[..., Any] | None = None,
    configured_endpoint_base_url: str | None = None,
    configured_endpoint_scope: ConfiguredEndpointScope | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    credentials_resolved: bool = False,
    timeout: float | None = None,
    # _chat_with_openai_compatible_local_server supports extra OpenAI fields (logit_bias, n, tools, etc.).
    # Add to this signature if Ollama supports them.
):
    if temp is not None:
        if temperature is not None and temperature != temp:
            logging.warning("Ollama: Received both 'temp' and 'temperature'; using 'temp' value")
        temperature = temp
    if stream is not None:
        if streaming is not None and streaming != stream:
            logging.warning("Ollama: Received both 'streaming' and 'stream'; preferring explicit 'stream' value")
        streaming = stream
    # Harmonize system alias
    if (system_message is None) and (system is not None):
        system_message = system
    if model and (model.lower() == "none" or model.strip() == ""):
        model = None
    loaded_config_data = _select_local_app_config(
        app_config,
        credentials_resolved=credentials_resolved,
    )
    cfg = loaded_config_data.get('ollama_api', {})

    current_api_base_url = configured_endpoint_base_url or cfg.get('api_url')
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
    # Support both num_predict (native) and max_tokens (alias) from callers
    if num_predict is not None:
        current_max_tokens = num_predict
    elif max_tokens is not None:
        current_max_tokens = max_tokens
    else:
        current_max_tokens = int(cfg.get('num_predict', cfg.get('max_tokens', 4096))) # Ollama uses 'num_predict'
    current_seed = seed if seed is not None else cfg.get('seed') # Ollama uses 'seed'
    current_stop = stop if stop is not None else cfg.get('stop') # Ollama uses 'stop' (list of strings)
    current_user_identifier = user_identifier if user_identifier is not None else cfg.get('user_identifier', cfg.get('user'))
    current_logprobs = logprobs if logprobs is not None else cfg.get('logprobs')
    current_top_logprobs = top_logprobs if top_logprobs is not None else cfg.get('top_logprobs')
    current_tools = tools if tools is not None else cfg.get('tools')
    current_tool_choice = tool_choice if tool_choice is not None else cfg.get('tool_choice')

    # Handle response_format for Ollama:
    # Ollama's format string ("json") maps to OpenAI's response_format {"type": "json_object"}.
    ollama_response_format_dict: dict[str, str] | None = None
    # Prefer explicit format_str argument, then alias 'format', then config key
    actual_format_value: str | dict[str, Any] | None = (
        format_str if format_str is not None else (format if format is not None else cfg.get('format'))
    )
    if isinstance(actual_format_value, dict):
        # Accept OpenAI-style dict: {'type': 'json_object'}
        fmt_type = str(actual_format_value.get('type', '')).lower()
        if fmt_type == 'json_object':
            ollama_response_format_dict = {"type": "json_object"}
        elif fmt_type:
            logging.warning(f"Ollama: Unsupported response_format dict type '{fmt_type}'. Only 'json_object' is recognized.")
    elif isinstance(actual_format_value, str):
        if actual_format_value.lower() == 'json':
            ollama_response_format_dict = {"type": "json_object"}
        elif actual_format_value:
            logging.warning(f"Ollama: Unsupported format string '{actual_format_value}'. Only 'json' is translated to OpenAI's response_format dict.")


    current_presence_penalty = presence_penalty if presence_penalty is not None else cfg.get('presence_penalty') # Ollama uses 'presence_penalty'
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else cfg.get('frequency_penalty') # Ollama uses 'frequency_penalty'

    # Ollama also supports other native parameters like 'num_ctx', 'tfs_z', 'mirostat', etc.
    # Add them to the signature if full coverage is desired; for now, focus on OpenAI-compatible ones.

    effective_timeout = timeout if timeout is not None else int(cfg.get('api_timeout', 300)) # Ollama can be slow
    api_retries = int(cfg.get('api_retries', 1))
    api_retry_delay = int(cfg.get('api_retry_delay', 1))

    if isinstance(current_streaming, str):
        current_streaming = current_streaming.lower() == "true"
    # Coerce numeric/string config values to correct types for Ollama's JSON schema
    try:
        if isinstance(current_top_p, str):
            current_top_p = float(current_top_p)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Ollama: Failed to coerce top_p='%s' to float; sending as-is", current_top_p)
    try:
        if isinstance(current_top_k, str):
            current_top_k = int(current_top_k)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Ollama: Failed to coerce top_k='%s' to int; sending as-is", current_top_k)
    try:
        if isinstance(current_presence_penalty, str):
            current_presence_penalty = float(current_presence_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Ollama: Failed to coerce presence_penalty='%s' to float; sending as-is", current_presence_penalty)
    try:
        if isinstance(current_frequency_penalty, str):
            current_frequency_penalty = float(current_frequency_penalty)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Ollama: Failed to coerce frequency_penalty='%s' to float; sending as-is", current_frequency_penalty)
    try:
        if isinstance(current_seed, str):
            current_seed = int(current_seed)
    except _LOCAL_ADAPTERS_NONCRITICAL_EXCEPTIONS:
        logging.warning("Ollama: Failed to coerce seed='%s' to int; sending as-is", current_seed)
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
        user_identifier=current_user_identifier,
        logprobs=current_logprobs,
        top_logprobs=current_top_logprobs,
        tools=current_tools,
        tool_choice=current_tool_choice,
        provider_name="Ollama",
        timeout=effective_timeout,
        api_retries=api_retries,
        api_retry_delay=api_retry_delay,
        filter_unknown_params=bool(cfg.get('strict_openai_compat', False)),
        http_client_factory=http_client_factory,
        http_fetcher=http_fetcher,
        http_streamer=http_streamer,
        configured_endpoint_scope=configured_endpoint_scope,
        extra_headers=extra_headers,
        extra_body=extra_body,
    )


class _LocalAdapterBase(ChatProvider):
    """Base adapter for local providers backed by local adapter helpers."""

    supports_streaming = True
    supports_tools = False
    default_timeout_seconds = 120
    max_output_tokens_default: int | None = 4096
    _handler = None
    http_client_factory: Callable[..., Any] | None = None
    http_fetcher: Callable[..., Any] = staticmethod(_hc_fetch)
    http_streamer: Callable[..., Any] = staticmethod(_hc_stream_response)

    _RESERVED_CONTEXT_KEYS = frozenset(
        {
            "base_url",
            "api_base_url",
            "api_url",
            "configured_endpoint_base_url",
            "configured_endpoint_scope",
            "endpoint_provenance",
            "_endpoint_provenance",
            "http_client_factory",
            "http_fetcher",
            "http_streamer",
            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
        }
    )

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": bool(self.supports_streaming),
            "supports_tools": bool(self.supports_tools),
            "default_timeout_seconds": self.default_timeout_seconds,
            "max_output_tokens_default": self.max_output_tokens_default,
        }

    def _sanitize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Discard all request-owned endpoint and transport context."""
        sanitized = dict(request or {})
        for key in tuple(sanitized):
            if key in self._RESERVED_CONTEXT_KEYS or key.endswith("_api_url"):
                sanitized.pop(key, None)
        return sanitized

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        raise NotImplementedError

    def _wrap_non_streaming(self, response: Any) -> Iterable[str]:
        content = extract_response_content(response)
        if content:
            yield openai_delta_chunk(content)
        yield sse_done()

    def _call_handler(
        self,
        request: dict[str, Any],
        *,
        streaming: bool | None,
        timeout: float | None = None,
    ) -> Any:
        raw_request, credentials = self._bind_request_credentials_with_handle(request)
        if credentials is not None:
            endpoint = credentials.trusted_endpoint
            if endpoint is None or not endpoint.scope.matches(endpoint.base_url):
                raise ChatConfigurationError(
                    provider=self.name,
                    message=f"{self.name} endpoint is not configured.",
                )
            raw_request["api_key"] = credentials.api_key
            raw_request["credentials_resolved"] = True
        else:
            endpoint = resolve_trusted_provider_endpoint(self.name)
            if endpoint is None:
                raise ChatConfigurationError(
                    provider=self.name,
                    message=f"{self.name} endpoint is not configured.",
                )

        sanitized = self._sanitize_request(raw_request)
        sanitized = validate_payload(self.name, sanitized)
        args = self._to_handler_args(sanitized, streaming=streaming)
        args.update(
            {
                "configured_endpoint_base_url": endpoint.base_url,
                "configured_endpoint_scope": endpoint.scope,
                "http_client_factory": self.http_client_factory or _hc_create_client,
                "http_fetcher": self.http_fetcher,
                "http_streamer": self.http_streamer,
            }
        )
        if timeout is not None:
            args["timeout"] = timeout
        handler = self._handler
        if handler is None:
            raise RuntimeError(f"{self.name} adapter missing handler")
        return handler(**args)

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return self._call_handler(request, streaming=False, timeout=timeout)

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        result = self._call_handler(request, streaming=True, timeout=timeout)
        if not isinstance(result, (dict, str, bytes, bytearray)) and hasattr(result, "__iter__"):
            return result
        return self._wrap_non_streaming(result)

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item


class LocalLLMAdapter(_LocalAdapterBase):
    name = "local-llm"
    supports_tools = True
    _handler = staticmethod(_local_llm_request)

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if streaming is not None:
            stream_flag = streaming
        return {
            "input_data": request.get("messages") or [],
            "api_key": request.get("api_key"),
            "custom_prompt_arg": request.get("custom_prompt_arg"),
            "temp": request.get("temperature"),
            "system_message": request.get("system_message"),
            "streaming": stream_flag,
            "model": request.get("model"),
            "top_k": request.get("top_k"),
            "top_p": request.get("top_p"),
            "min_p": request.get("min_p"),
            "max_tokens": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "response_format": request.get("response_format"),
            "n": request.get("n"),
            "user_identifier": request.get("user"),
            "logit_bias": request.get("logit_bias"),
            "presence_penalty": request.get("presence_penalty"),
            "frequency_penalty": request.get("frequency_penalty"),
            "logprobs": request.get("logprobs"),
            "top_logprobs": request.get("top_logprobs"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
        }


class LlamaCppAdapter(_LocalAdapterBase):
    name = "llama.cpp"
    supports_tools = False
    _handler = staticmethod(_llama_request)

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if streaming is not None:
            stream_flag = streaming
        return {
            "input_data": request.get("messages") or [],
            "api_key": request.get("api_key"),
            "custom_prompt": request.get("custom_prompt_arg"),
            "temp": request.get("temperature"),
            "system_prompt": request.get("system_message"),
            "streaming": stream_flag,
            "model": request.get("model"),
            "top_k": request.get("top_k"),
            "top_p": request.get("top_p"),
            "min_p": request.get("min_p"),
            "n_predict": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "response_format": request.get("response_format"),
            "logit_bias": request.get("logit_bias"),
            "n": request.get("n"),
            "presence_penalty": request.get("presence_penalty"),
            "frequency_penalty": request.get("frequency_penalty"),
            "api_url": request.get("api_url"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
            "inference_prefix_cache_intent": request.get("inference_prefix_cache_intent"),
        }


class KoboldAdapter(_LocalAdapterBase):
    name = "kobold"
    supports_streaming = False
    supports_tools = False
    _handler = staticmethod(_kobold_request)

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if streaming is not None:
            stream_flag = streaming
        return {
            "input_data": request.get("messages") or [],
            "api_key": request.get("api_key"),
            "custom_prompt_input": request.get("custom_prompt_arg"),
            "temp": request.get("temperature"),
            "system_message": request.get("system_message"),
            "streaming": stream_flag,
            "model": request.get("model"),
            "top_k": request.get("top_k"),
            "top_p": request.get("top_p"),
            "max_length": request.get("max_tokens"),
            "stop_sequence": request.get("stop"),
            "num_responses": request.get("n"),
            "seed": request.get("seed"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
        }


class OobaAdapter(_LocalAdapterBase):
    name = "ooba"
    supports_tools = False
    _handler = staticmethod(_ooba_request)

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if streaming is not None:
            stream_flag = streaming
        return {
            "input_data": request.get("messages") or [],
            "api_key": request.get("api_key"),
            "custom_prompt": request.get("custom_prompt_arg"),
            "temp": request.get("temperature"),
            "system_prompt": request.get("system_message"),
            "streaming": stream_flag,
            "model": request.get("model"),
            "top_k": request.get("top_k"),
            "top_p": request.get("top_p"),
            "min_p": request.get("min_p"),
            "max_tokens": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "response_format": request.get("response_format"),
            "n": request.get("n"),
            "user_identifier": request.get("user"),
            "logit_bias": request.get("logit_bias"),
            "presence_penalty": request.get("presence_penalty"),
            "frequency_penalty": request.get("frequency_penalty"),
            "api_url": request.get("api_url"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
        }


class TabbyAPIAdapter(_LocalAdapterBase):
    name = "tabbyapi"
    supports_tools = True
    _handler = staticmethod(_tabbyapi_request)

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if streaming is not None:
            stream_flag = streaming
        return {
            "input_data": request.get("messages") or [],
            "api_key": request.get("api_key"),
            "custom_prompt_input": request.get("custom_prompt_arg"),
            "temp": request.get("temperature"),
            "system_message": request.get("system_message"),
            "streaming": stream_flag,
            "model": request.get("model"),
            "top_k": request.get("top_k"),
            "top_p": request.get("top_p"),
            "min_p": request.get("min_p"),
            "max_tokens": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "response_format": request.get("response_format"),
            "n": request.get("n"),
            "user_identifier": request.get("user"),
            "logit_bias": request.get("logit_bias"),
            "presence_penalty": request.get("presence_penalty"),
            "frequency_penalty": request.get("frequency_penalty"),
            "logprobs": request.get("logprobs"),
            "top_logprobs": request.get("top_logprobs"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "api_url": request.get("api_url"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
        }


class VLLMAdapter(_LocalAdapterBase):
    name = "vllm"
    supports_tools = True
    _handler = staticmethod(_vllm_request)

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if streaming is not None:
            stream_flag = streaming
        return {
            "input_data": request.get("messages") or [],
            "api_key": request.get("api_key"),
            "custom_prompt_input": request.get("custom_prompt_arg"),
            "temp": request.get("temperature"),
            "system_prompt": request.get("system_message"),
            "streaming": stream_flag,
            "model": request.get("model"),
            "top_k": request.get("top_k"),
            "top_p": request.get("top_p"),
            "min_p": request.get("min_p"),
            "max_tokens": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "response_format": request.get("response_format"),
            "n": request.get("n"),
            "logit_bias": request.get("logit_bias"),
            "presence_penalty": request.get("presence_penalty"),
            "frequency_penalty": request.get("frequency_penalty"),
            "logprobs": request.get("logprobs"),
            "top_logprobs": request.get("top_logprobs"),
            "user_identifier": request.get("user"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "vllm_api_url": request.get("api_url"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
            "inference_prefix_cache_intent": request.get("inference_prefix_cache_intent"),
        }


class OllamaAdapter(_LocalAdapterBase):
    name = "ollama"
    supports_tools = True
    _handler = staticmethod(_ollama_request)

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if streaming is not None:
            stream_flag = streaming
        return {
            "input_data": request.get("messages") or [],
            "api_key": request.get("api_key"),
            "custom_prompt": request.get("custom_prompt_arg"),
            "temp": request.get("temperature"),
            "system_message": request.get("system_message"),
            "streaming": stream_flag,
            "model": request.get("model"),
            "top_p": request.get("top_p"),
            "top_k": request.get("top_k"),
            "num_predict": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "format_str": request.get("response_format"),
            "presence_penalty": request.get("presence_penalty"),
            "frequency_penalty": request.get("frequency_penalty"),
            "user_identifier": request.get("user"),
            "logprobs": request.get("logprobs"),
            "top_logprobs": request.get("top_logprobs"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "api_url": request.get("api_url"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
        }


class AphroditeAdapter(_LocalAdapterBase):
    name = "aphrodite"
    supports_tools = True
    _handler = staticmethod(_aphrodite_request)

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if streaming is not None:
            stream_flag = streaming
        return {
            "input_data": request.get("messages") or [],
            "api_key": request.get("api_key"),
            "custom_prompt": request.get("custom_prompt_arg"),
            "temp": request.get("temperature"),
            "system_message": request.get("system_message"),
            "streaming": stream_flag,
            "model": request.get("model"),
            "top_k": request.get("top_k"),
            "top_p": request.get("top_p"),
            "min_p": request.get("min_p"),
            "max_tokens": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "response_format": request.get("response_format"),
            "n": request.get("n"),
            "logit_bias": request.get("logit_bias"),
            "presence_penalty": request.get("presence_penalty"),
            "frequency_penalty": request.get("frequency_penalty"),
            "logprobs": request.get("logprobs"),
            "top_logprobs": request.get("top_logprobs"),
            "user_identifier": request.get("user"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "api_url": request.get("api_url"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
        }

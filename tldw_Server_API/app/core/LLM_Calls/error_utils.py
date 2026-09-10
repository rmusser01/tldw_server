from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.exceptions import (
    NetworkError,
    RetryExhaustedError,
    raise_detached_error,
)

_ERROR_UTILS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    json.JSONDecodeError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    re.error,
)

_PRIVACY_SAFE_PROVIDER_ERRORS: ContextVar[bool] = ContextVar(
    "privacy_safe_provider_errors",
    default=False,
)


@contextmanager
def provider_error_privacy_scope(enabled: bool) -> Iterator[None]:
    """Temporarily select metadata-only provider error handling for this call."""

    token = _PRIVACY_SAFE_PROVIDER_ERRORS.set(bool(enabled))
    try:
        yield
    finally:
        _PRIVACY_SAFE_PROVIDER_ERRORS.reset(token)


def provider_errors_are_privacy_safe() -> bool:
    """Return whether the current provider call must avoid upstream detail."""

    return _PRIVACY_SAFE_PROVIDER_ERRORS.get()


def _bounded_log_label(value: Any) -> str:
    """Return a bounded label safe for provider-failure metadata."""

    normalized = "".join(
        character
        for character in str(value or "").strip().lower()
        if character.isalnum() or character in ".-_"
    )[:64]
    return normalized or "unknown"


def log_provider_failure(
    provider: Any,
    exc: BaseException,
    *,
    phase: str,
    status_code: int | None = None,
) -> None:
    """Log bounded upstream failure metadata without URLs, bodies, or exception text."""

    if status_code is None and isinstance(exc, Exception):
        try:
            status_code = get_http_status_from_exception(exc)
        except Exception:  # noqa: BLE001 - logging must never replace the provider failure
            status_code = None
    logger.error(
        "{} provider failure phase={} error_type={} upstream_status={}",
        _bounded_log_label(provider),
        _bounded_log_label(phase),
        _bounded_log_label(type(exc).__name__),
        status_code if isinstance(status_code, int) else "unknown",
    )


def build_sanitized_chat_error(
    provider: str,
    *,
    status_code: int | None = None,
    auth_statuses: tuple[int, ...] = (401, 403),
    rate_limit_statuses: tuple[int, ...] = (429,),
    bad_request_statuses: tuple[int, ...] = (400, 404, 422),
    treat_other_4xx_as_bad_request: bool = True,
) -> ChatAPIError:
    """Build one typed public error without reflecting untrusted upstream text."""

    safe_provider = _bounded_log_label(provider)
    if status_code in auth_statuses:
        return ChatAuthenticationError(
            provider=safe_provider,
            status_code=status_code or 401,
        )
    if status_code in rate_limit_statuses:
        return ChatRateLimitError(provider=safe_provider)
    if status_code in bad_request_statuses or (
        treat_other_4xx_as_bad_request
        and status_code is not None
        and 400 <= status_code < 500
    ):
        return ChatBadRequestError(provider=safe_provider)
    if status_code is not None and 500 <= status_code < 600:
        return ChatProviderError(provider=safe_provider, status_code=status_code)
    if status_code is None:
        return ChatProviderError(provider=safe_provider)
    return ChatAPIError(provider=safe_provider, status_code=status_code)


def get_http_status_from_exception(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from common exception shapes."""
    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            status = getattr(response, attr, None)
            if status is not None:
                try:
                    return int(status)
                except (TypeError, ValueError):
                    pass
    for attr in ("status_code", "status"):
        status = getattr(exc, attr, None)
        if status is not None:
            try:
                return int(status)
            except (TypeError, ValueError):
                pass
    if isinstance(exc, NetworkError):
        match = re.search(r"HTTP\\s+(\\d{3})", str(exc))
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def get_http_error_text(exc: Exception) -> str:
    """Return an error detail string from common response/exception shapes."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            text = getattr(response, "text", None)
        except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS as response_exc:
            text = None
            if getattr(response_exc.__class__, "__name__", "") == "ResponseNotRead":
                try:
                    response.read()
                    text = getattr(response, "text", None)
                except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
                    text = None
        if text is None:
            try:
                text = getattr(response, "content", None)
            except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS as response_exc:
                text = None
                if getattr(response_exc.__class__, "__name__", "") == "ResponseNotRead":
                    try:
                        response.read()
                        text = getattr(response, "content", None)
                    except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
                        text = None
            if isinstance(text, (bytes, bytearray)):
                try:
                    text = text.decode("utf-8", errors="replace")
                except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
                    text = None
        if text is not None:
            return str(text)
    response_text = getattr(exc, "response_text", None)
    if response_text:
        return str(response_text)
    return str(exc)


def _redact_sensitive_text(text: str) -> str:
    if not text:
        return text
    try:
        text = re.sub(r"(?i)(authorization\s*:\s*bearer)\s+[^\s,;]+", r"\1 [REDACTED]", text)
        text = re.sub(r"(?i)(bearer)\s+[^\s,;]+", r"\1 [REDACTED]", text)
        text = re.sub(r'(?i)("api[_ -]?key"\s*:\s*)"[^"]+"', r'\1"[REDACTED]"', text)
        text = re.sub(r"(?i)(api[_ -]?key\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]", text)
    except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
        return text
    return text


def _safe_http_error_metadata(body_json: Any = None, body_text: str | None = None) -> dict[str, Any]:
    """Build log-safe metadata for upstream error payloads without body content."""
    metadata: dict[str, Any] = {}
    if isinstance(body_json, dict):
        metadata["body_shape"] = "object"
        err_obj = body_json.get("error")
        if isinstance(err_obj, dict):
            for key in ("type", "code", "param", "status"):
                value = err_obj.get(key)
                if isinstance(value, (str, int, float, bool)):
                    metadata[f"error_{key}"] = _redact_sensitive_text(str(value))[:160]
            message = err_obj.get("message")
            if isinstance(message, str):
                metadata["error_message_chars"] = len(message)
        elif isinstance(err_obj, str):
            metadata["error_chars"] = len(err_obj)
        for key in ("type", "code", "param", "status"):
            value = body_json.get(key)
            if isinstance(value, (str, int, float, bool)) and f"error_{key}" not in metadata:
                metadata[f"error_{key}"] = _redact_sensitive_text(str(value))[:160]
        metadata["top_level_key_count"] = len(body_json)
    elif body_json is not None:
        metadata["body_shape"] = type(body_json).__name__
    elif body_text:
        metadata["body_shape"] = "raw_text"
        metadata["raw_body_chars"] = len(body_text)
    return metadata


def log_http_400_body(provider: str, exc: Exception, parsed_body: Any = None, max_chars: int = 2000) -> None:
    try:
        status = get_http_status_from_exception(exc)
    except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
        status = None
    if status != 400:
        return
    body_json = None
    body_text = None
    if parsed_body is not None:
        body_json = parsed_body
    else:
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                body_json = resp.json()
            except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
                body_json = None
    if body_json is not None:
        try:
            body_text = json.dumps(body_json, ensure_ascii=True)
        except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
            body_text = str(body_json)
    else:
        try:
            body_text = get_http_error_text(exc)
        except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
            body_text = None
    if body_json is None and not body_text:
        return
    metadata = _safe_http_error_metadata(body_json, str(body_text) if body_text else None)
    try:
        metadata_text = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
    except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
        metadata_text = "{}"
    if max_chars is not None and len(metadata_text) > max_chars:
        metadata_text = metadata_text[:max_chars] + "...(truncated)"
    logger.warning(f"{provider or 'unknown'}: upstream 400 response metadata: {metadata_text}")


def log_provider_failure_metadata(
    provider: str,
    exc: Exception,
    *,
    phase: str = "request",
) -> None:
    """Log an upstream failure without response or exception text."""

    try:
        status = get_http_status_from_exception(exc)
    except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
        status = None
    logger.error(
        "{} provider {} failed: status={} error_type={}",
        provider or "unknown",
        phase,
        status,
        type(exc).__name__,
    )


def privacy_safe_chat_error(provider: str, exc: Exception) -> ChatAPIError:
    """Return a typed provider error containing no upstream body or exception text."""

    resolved_provider = str(getattr(exc, "provider", None) or provider or "unknown")
    status_code = get_http_status_from_exception(exc)
    if isinstance(exc, ChatAuthenticationError) or status_code in {401, 403}:
        safe: ChatAPIError = ChatAuthenticationError(
            provider=resolved_provider,
            message="Provider authentication failed.",
        )
    elif isinstance(exc, ChatRateLimitError) or status_code == 429:
        safe = ChatRateLimitError(
            provider=resolved_provider,
            message="Provider rate limit exceeded.",
        )
    elif isinstance(exc, ChatBadRequestError) or status_code in {400, 404, 422}:
        safe = ChatBadRequestError(
            provider=resolved_provider,
            message="Provider rejected the request.",
        )
    elif isinstance(exc, ChatConfigurationError):
        safe = ChatConfigurationError(
            provider=resolved_provider,
            message="Provider configuration is unavailable.",
        )
    elif isinstance(exc, ChatProviderError) or (
        status_code is not None and status_code >= 500
    ):
        safe = ChatProviderError(
            provider=resolved_provider,
            message="Provider request failed.",
            status_code=status_code or 502,
        )
    else:
        safe = ChatAPIError(
            provider=resolved_provider,
            message="Provider request failed.",
            status_code=status_code or 500,
        )

    retry_after = getattr(exc, "retry_after", None)
    if retry_after is None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if isinstance(headers, Mapping):
            retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is not None:
        safe.retry_after = retry_after
    return safe


def raise_chat_error_from_http(
    provider: str,
    exc: Exception,
    *,
    auth_statuses: tuple[int, ...] = (401, 403),
    rate_limit_statuses: tuple[int, ...] = (429,),
    bad_request_statuses: tuple[int, ...] = (400, 404, 422),
    treat_other_4xx_as_bad_request: bool = True,
) -> None:
    """Raise a detached, typed error without reflecting upstream response text."""
    status_code = get_http_status_from_exception(exc)
    response = getattr(exc, "response", None)
    parsed_body = None

    if response is not None:
        try:
            parsed_body = response.json()
        except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
            parsed_body = None
        log_http_400_body(provider, exc, parsed_body)
        metadata = _safe_http_error_metadata(parsed_body)
        try:
            metadata_text = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
        except _ERROR_UTILS_NONCRITICAL_EXCEPTIONS:
            metadata_text = "{}"
        logger.error(
            f"{_bounded_log_label(provider)} HTTP error response "
            f"status={status_code if isinstance(status_code, int) else 'unknown'} "
            f"metadata={metadata_text}"
        )
    else:
        log_provider_failure(
            provider,
            exc,
            phase="http_response",
            status_code=status_code,
        )

    raise_detached_error(
        build_sanitized_chat_error(
            provider,
            status_code=status_code,
            auth_statuses=auth_statuses,
            rate_limit_statuses=rate_limit_statuses,
            bad_request_statuses=bad_request_statuses,
            treat_other_4xx_as_bad_request=treat_other_4xx_as_bad_request,
        )
    )


def is_network_error(exc: Exception) -> bool:
    if isinstance(exc, (NetworkError, RetryExhaustedError)):
        return True
    module = getattr(exc.__class__, "__module__", "")
    name = exc.__class__.__name__
    if module.startswith("requests"):
        return "RequestException" in name or "ConnectionError" in name or "Timeout" in name
    if module.startswith("httpx"):
        return "RequestError" in name or "Connect" in name or "Timeout" in name
    return False


def is_http_status_error(exc: Exception) -> bool:
    module = getattr(exc.__class__, "__module__", "")
    name = exc.__class__.__name__
    if module.startswith("httpx"):
        return name == "HTTPStatusError"
    if module.startswith("requests"):
        return name == "HTTPError"
    return False


def is_chunked_encoding_error(exc: Exception) -> bool:
    module = getattr(exc.__class__, "__module__", "")
    name = exc.__class__.__name__
    return module.startswith("requests") and name == "ChunkedEncodingError"

from __future__ import annotations

"""Derive an HTTP status from a transport exception of unknown provenance.

One job: given an exception raised by *some* HTTP client -- httpx, requests, aiohttp,
or this codebase's own NetworkError -- work out what status the server actually
returned, and what it said.

This lived in core/LLM_Calls/error_utils.py and was copied four ways. Two copies
carried a double-escaped regex that could never match, so an upstream 429 arrived as
ChatProviderError's default 502 with no Retry-After. A third (Embeddings) has no
message-text branch at all and inverts the attribute precedence, checking
exc.status_code before exc.response, so an aiohttp ClientResponseError returns None
there and the right status elsewhere. Three TTS adapters additionally re-implement
is_http_status_error recognising only httpx, while this one also recognises requests.

It lives under core/Utils/ rather than core/LLM_Calls/ because TTS, Local_LLM and
Embeddings all need it, and none of them should depend on the LLM_Calls package to
classify an HTTP error. error_utils re-exports these names, so its existing importers
are unaffected.
"""

import re
from typing import Any

from tldw_Server_API.app.core.exceptions import NetworkError

# Failures while *inspecting* an exception must never mask the exception itself.
_HTTP_STATUS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    TypeError,
    ValueError,
    OSError,
)
_ERROR_UTILS_NONCRITICAL_EXCEPTIONS = _HTTP_STATUS_NONCRITICAL_EXCEPTIONS

__all__ = [
    "get_http_error_text",
    "get_http_status_from_exception",
    "is_chunked_encoding_error",
    "is_http_status_error",
]


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
        match = re.search(r"HTTP\s+(\d{3})", str(exc))
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

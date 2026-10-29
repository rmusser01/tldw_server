"""Provider result and stream-error normalization shared by adapters and chat streaming (TASK-13950.13)."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from tldw_Server_API.app.core.exceptions import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatConfigurationError,
    ChatModelNotAvailableError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.LLM_Calls.sse import SSE_CONTROL_FIELD_PREFIXES, is_done_line

_PROVIDER_RESULT_NONCRITICAL_EXCEPTIONS = (
    ChatAPIError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    json.JSONDecodeError,
)

# Comments and event: lines are always SSE control. id:/retry: are control only
# inside SSE-framed chunks; an unframed provider chunk starting with them is
# assistant text. sse.normalize_provider_line only ever sees framed lines, so it
# drops every SSE_CONTROL_FIELD_PREFIXES entry unconditionally.
_ALWAYS_SSE_CONTROL_PREFIXES = (":", "event:")
_FRAMED_ONLY_SSE_CONTROL_PREFIXES = tuple(
    prefix for prefix in SSE_CONTROL_FIELD_PREFIXES if prefix not in _ALWAYS_SSE_CONTROL_PREFIXES
)

PROVIDER_STREAM_ERROR_MESSAGES = {
    "provider_authentication_failed": "The selected provider credentials could not be authenticated.",
    "invalid_provider_credentials": "The selected provider credentials are invalid.",
    "missing_provider_credentials": "The selected provider credentials are not configured.",
    "credential_store_unavailable": "Provider credential storage is temporarily unavailable.",
    "credential_scope_revoked": "The selected provider credential scope is no longer available.",
    "provider_disabled": "The selected provider is disabled by administrator policy.",
    "model_not_allowed": "The selected model is not allowed for this provider.",
    "provider_configuration_invalid": "The selected provider configuration is invalid.",
    "model_not_available": "The selected model is not available for this provider.",
    "provider_model_inventory_unavailable": (
        "The provider's model list could not be checked right now, so the model could not be confirmed. "
        "Try again shortly."
    ),
    "provider_unavailable": "The chat service provider is currently unavailable.",
    "provider_output_limit": (
        "The model reached its output token limit before producing a final answer. "
        "Increase the output token limit and try again."
    ),
}
_PROVIDER_STREAM_ERROR_STATUS = {
    "model_not_available": 400,
    "provider_authentication_failed": 502,
    "provider_unavailable": 502,
    "provider_output_limit": 502,
    "provider_disabled": 403,
    "model_not_allowed": 403,
}
_PROVIDER_STREAM_ERROR_TYPE_CODES = {
    "chatauthenticationerror": "provider_authentication_failed",
    "chatconfigurationerror": "provider_configuration_invalid",
    "chatprovidererror": "provider_unavailable",
    "chatapierror": "provider_unavailable",
}


@dataclass(frozen=True, slots=True)
class NormalizedProviderStreamError:
    """Bounded public provider error and its explicit replay capability."""

    code: str
    message: str
    status_code: int
    replay_certified: bool = False
    credential_refresh_retry_certified: bool = False


def _provider_stream_code(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if normalized in PROVIDER_STREAM_ERROR_MESSAGES:
        return normalized
    return _PROVIDER_STREAM_ERROR_TYPE_CODES.get(normalized.lower())


def _provider_stream_error_candidate(value: Any) -> tuple[Any, bool]:
    """Return an error candidate and whether an error envelope was present."""
    if isinstance(value, BaseException):
        return value, True
    if isinstance(value, dict):
        if "error" in value and value.get("error") is not None:
            return value.get("error"), True
        if value.get("type") == "error" or value.get("error_code") is not None:
            return value, True
        return None, False
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        text = value
    else:
        return None, False

    for raw_line in text.splitlines() or [text]:
        candidate_text = raw_line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
        if not candidate_text:
            continue
        if not candidate_text.startswith("data:"):
            continue
        candidate_text = candidate_text[len("data:") :].strip()
        try:
            decoded = json.loads(candidate_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        candidate, present = _provider_stream_error_candidate(decoded)
        if present:
            return candidate, True
    return None, False


def _normalized_provider_error_for_code(value: Any) -> Optional[NormalizedProviderStreamError]:
    """Build a canonical error from one explicit internal code or type name."""

    code = _provider_stream_code(value)
    if code is None:
        return None
    return NormalizedProviderStreamError(
        code=code,
        message=PROVIDER_STREAM_ERROR_MESSAGES[code],
        status_code=_PROVIDER_STREAM_ERROR_STATUS.get(code, 503),
    )


def normalize_provider_stream_error(value: Any) -> Optional[NormalizedProviderStreamError]:
    """Normalize raised or in-band provider failures without retaining raw detail."""
    if isinstance(value, NormalizedProviderStreamError):
        return value

    if isinstance(value, bytes):
        explicit_text = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        explicit_text = value
    else:
        explicit_text = None
    if explicit_text is not None:
        explicit_error = _normalized_provider_error_for_code(explicit_text)
        explicit_error = explicit_error or _normalize_unframed_provider_stream_error(
            explicit_text
        )
        if explicit_error is not None:
            return explicit_error

    candidate, present = _provider_stream_error_candidate(value)
    if not present:
        return None

    code: Optional[str] = None
    if isinstance(candidate, ChatAuthenticationError):
        code = "provider_authentication_failed"
    elif isinstance(candidate, ChatConfigurationError):
        code = _provider_stream_code(getattr(candidate, "error_code", None))
        code = code or "provider_configuration_invalid"
    elif isinstance(candidate, BaseException):
        code = _provider_stream_code(getattr(candidate, "code", None))
        code = code or _provider_stream_code(getattr(candidate, "error_code", None))
        detail = getattr(candidate, "detail", None)
        if code is None and isinstance(detail, dict):
            code = _provider_stream_code(detail.get("error_code") or detail.get("code"))
        if code is None:
            code = _provider_stream_code(type(candidate).__name__)
    elif isinstance(candidate, dict):
        code = _provider_stream_code(
            candidate.get("error_code") or candidate.get("code") or candidate.get("type")
        )
    else:
        code = _provider_stream_code(candidate)

    if code not in PROVIDER_STREAM_ERROR_MESSAGES:
        code = "provider_unavailable"
    message = PROVIDER_STREAM_ERROR_MESSAGES[code]
    if code == "model_not_available" and isinstance(
        candidate, (ChatModelNotAvailableError, SanitizedProviderStreamError),
    ):
        # Built from the request's own provider and model, never upstream text: the same answer as non-stream.
        message = str(candidate.message)
    replay_certified = bool(
        isinstance(value, ChatAPIError)
        and getattr(value, "upstream_dispatched", None) is False
        and getattr(value, "output_emitted", None) is False
        and getattr(value, "allow_non_stream_fallback", None) is True
    )
    credential_refresh_retry_certified = bool(
        isinstance(candidate, BaseException)
        and getattr(candidate, "credential_refresh_retry_safe", None) is True
    )
    return NormalizedProviderStreamError(
        code=code,
        message=message,
        status_code=_PROVIDER_STREAM_ERROR_STATUS.get(code, 503),
        replay_certified=replay_certified,
        credential_refresh_retry_certified=credential_refresh_retry_certified,
    )


def _normalize_unframed_provider_stream_error(
    value: str,
) -> Optional[NormalizedProviderStreamError]:
    """Accept only an explicit canonical code from an unframed JSON string.

    A provider may legitimately stream assistant-authored JSON containing an
    ``error`` field.  Arbitrary error envelopes are therefore authoritative
    only when carried by an SSE ``data:`` frame (or as a structured adapter
    object before string conversion).  The raw-string compatibility path is
    intentionally narrower and recognizes only our bounded internal codes.
    """

    try:
        decoded = json.loads(value.strip())
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    candidate, present = _provider_stream_error_candidate(decoded)
    if not present:
        return None
    if isinstance(candidate, dict):
        code = _provider_stream_code(
            candidate.get("error_code") or candidate.get("code") or candidate.get("type")
        )
    else:
        code = _provider_stream_code(candidate)
    return _normalized_provider_error_for_code(code)


def provider_result_contains_error(
    value: Any,
    *,
    legacy_error_prefix: bool = False,
) -> bool:
    """Return whether a supported response structure contains a provider error.

    Only response-bearing fields are traversed. Tool inputs and function
    arguments are deliberately excluded so domain objects containing an
    ``error`` key remain valid tool data.
    """

    seen: set[int] = set()

    def visit(item: Any) -> bool:
        if normalize_provider_stream_error(item) is not None:
            return True
        if isinstance(item, str):
            if not legacy_error_prefix:
                return False
            if item.lstrip().lower().startswith("error:"):
                return True
            for raw_line in item.splitlines():
                line = raw_line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
                if line.startswith("data:") and line[len("data:") :].lstrip().lower().startswith(
                    "error:"
                ):
                    return True
            return False
        if isinstance(item, bytes):
            return False
        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in seen:
                return False
            seen.add(identity)
            return any(visit(nested) for nested in item)
        if not isinstance(item, dict):
            return False

        identity = id(item)
        if identity in seen:
            return False
        seen.add(identity)

        for field in ("choices", "message", "delta", "content"):
            if field in item and visit(item[field]):
                return True

        block_type = item.get("type")
        if block_type in {"text", "output_text"} and "text" in item:
            return visit(item["text"])
        return False

    return visit(value)


def provider_stream_error_payload(value: Any) -> dict[str, dict[str, str]]:
    """Build the canonical SSE/HTTP-safe provider error payload."""
    normalized = _normalized_provider_error_for_code(value)
    normalized = normalized or normalize_provider_stream_error(value)
    if normalized is None:
        normalized = NormalizedProviderStreamError(
            code="provider_unavailable",
            message=PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"],
            status_code=502,
        )
    return {
        "error": {
            "code": normalized.code,
            "type": normalized.code,
            "message": normalized.message,
        }
    }


def sanitized_provider_stream_exception(value: Any) -> SanitizedProviderStreamError:
    """Return a core safe exception for one raised provider failure."""
    normalized = _normalized_provider_error_for_code(value)
    normalized = normalized or normalize_provider_stream_error(value)
    if normalized is None:
        normalized = NormalizedProviderStreamError(
            code="provider_unavailable",
            message=PROVIDER_STREAM_ERROR_MESSAGES["provider_unavailable"],
            status_code=502,
        )
    return SanitizedProviderStreamError(
        code=normalized.code,
        message=normalized.message,
        status_code=normalized.status_code,
        replay_certified=normalized.replay_certified,
        credential_refresh_retry_certified=(
            normalized.credential_refresh_retry_certified
        ),
    )


def _extract_text_from_upstream_sse(chunk_str: str) -> tuple[Optional[str], Optional[dict[str, Any]], bool]:
    """
    Normalize provider-emitted SSE frames to plain text content.

    Accepts a string that may be:
      - a raw text fragment (returns it as text_content)
      - an SSE line like "data: {...}" (extracts JSON and returns delta.content if present)
      - an SSE DONE line "data: [DONE]" (signals completion via is_done=True)

    Returns: (text_content, error_payload, is_done)
      - text_content: extracted textual delta (or original text) or None
      - error_payload: if upstream provided an error object, return it for direct emission
      - is_done: True if upstream indicated [DONE]
    """
    if not chunk_str:
        return None, None, False

    # Normalize common invisible prefixes (BOM, zero-width spaces) and trim whitespace
    s = chunk_str.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
    is_sse_framed = "\n\n" in chunk_str or "\r\n\r\n" in chunk_str or s.startswith((":", "event:", "data:"))

    # Ignore SSE control-only lines from upstream
    if (
        s.startswith(_ALWAYS_SSE_CONTROL_PREFIXES) or (is_sse_framed and s.startswith(_FRAMED_ONLY_SSE_CONTROL_PREFIXES))
    ) and "data:" not in s:
        return None, None, False

    # If any 'data:' line exists, try to parse; some providers send 'event:' + 'data:' pairs or multiple frames
    if s.startswith("data:") or ("\ndata:" in s or s.startswith("event:") or "data:" in s):
        saw_done = False
        first_error = None
        # Process by lines to handle possible multi-line chunks
        for line in s.splitlines():
            ls = line.lstrip("\ufeff\u200b\u200c\u200d\u2060").strip()
            if not ls:
                continue
            if ls.startswith(_ALWAYS_SSE_CONTROL_PREFIXES) or ls.startswith(_FRAMED_ONLY_SSE_CONTROL_PREFIXES):
                # Skip SSE control fields
                continue
            if not ls.startswith("data:"):
                continue
            payload_str = ls[len("data:") :].strip()
            if is_done_line(ls):
                saw_done = True
                continue
            try:
                data = json.loads(payload_str)
            except _PROVIDER_RESULT_NONCRITICAL_EXCEPTIONS:
                # Try next line if present
                continue

            if (
                isinstance(data, dict)
                and data.get("error") is not None
                and first_error is None
            ):
                first_error = provider_stream_error_payload(data)
                continue

            if isinstance(data, dict):
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0] or {}
                    delta = first.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        return str(content), None, False
                    # Fallback to message.content (non-stream case)
                    message = first.get("message") or {}
                    msg_content = message.get("content")
                    if msg_content:
                        return str(msg_content), None, False
        # If no content found but DONE or error encountered, reflect that
        if first_error is not None:
            return None, first_error, False
        if saw_done:
            return None, None, True
        return None, None, False

    # Not an SSE frame; treat as plain text chunk
    return chunk_str, None, False


def extract_response_content(llm_response: Any) -> Optional[str]:
    """
    Extract text content from LLM response.

    Args:
        llm_response: Response from LLM API

    Returns:
        Extracted text content or None
    """
    if isinstance(llm_response, str):
        return llm_response
    elif isinstance(llm_response, dict):
        # OpenAI-style response
        choices = llm_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            return message.get("content")

    return None

"""First-run first-chat verification helpers."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
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
from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

DEFAULT_FIRST_CHAT_PROMPT = "Please reply with a short hello so setup can confirm chat works."
FIRST_CHAT_RESPONSE_TEXT_MAX_LENGTH = 500
_SECRET_LIKE_TEXT_RE = re.compile(
    r"(?i)(?:"
    r"sk-[A-Za-z0-9_-]{3,}|"
    r"xox[baprs]-[A-Za-z0-9-]{6,}|"
    r"gh[pousr]_[A-Za-z0-9_]{6,}|"
    r"github_pat_[A-Za-z0-9_]{6,}|"
    r"hf_[A-Za-z0-9]{6,}|"
    r"gsk_[A-Za-z0-9]{6,}|"
    r"pplx-[A-Za-z0-9_-]{6,}|"
    r"AIza[0-9A-Za-z_-]{6,}|"
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}|"
    r"bearer\s+\S{6,}|"
    r"(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S{3,}"
    r")"
)
_LOCAL_PATH_TEXT_RE = re.compile(
    r"(?:"
    r"(?:/Users|/home|/private|/var|/tmp|/etc)/[^\s,;\"']+|"
    r"[A-Za-z]:\\[^\s,;\"']+"
    r")"
)


@dataclass(frozen=True)
class FirstChatVerificationResult:
    """Sanitized first-chat verification result."""

    status: str
    provider: str
    model: str
    response_id: str | None = None
    response_text: str | None = None
    failure_category: str | None = None
    message: str | None = None


async def _call_chat_completion(*, provider: str, model: str, prompt: str) -> Any:
    """Call the adapter-backed chat completion path; kept patchable for tests."""
    return await perform_chat_api_call_async(
        api_endpoint=provider,
        model=model,
        messages_payload=[{"role": "user", "content": prompt}],
        stream=False,
    )


async def verify_first_chat(
    *,
    provider: str,
    model: str,
    prompt: str = DEFAULT_FIRST_CHAT_PROMPT,
) -> FirstChatVerificationResult:
    """Verify that a configured provider/model can return a first chat response."""
    provider = provider.strip()
    model = model.strip()
    prompt = prompt.strip() or DEFAULT_FIRST_CHAT_PROMPT
    try:
        response = await _call_chat_completion(provider=provider, model=model, prompt=prompt)
    except Exception as exc:  # noqa: BLE001 - setup responses must stay sanitized
        category, message = _classify_failure(exc)
        logger.warning(
            "First-run first-chat verification failed for provider={} model={} category={}: {}",
            provider,
            model,
            category,
            type(exc).__name__,
        )
        return FirstChatVerificationResult(
            status="failed",
            provider=provider,
            model=model,
            failure_category=category,
            message=message,
        )

    response_text = _extract_response_text(response)
    if not response_text:
        return FirstChatVerificationResult(
            status="failed",
            provider=provider,
            model=model,
            failure_category="empty_response",
            message="The provider returned an empty chat response.",
        )

    return FirstChatVerificationResult(
        status="ready",
        provider=provider,
        model=model,
        response_id=_extract_response_id(response),
        response_text=_sanitize_response_text(response_text),
    )


def _extract_response_id(response: Any) -> str | None:
    response_id = _get_value(response, "id")
    return response_id if isinstance(response_id, str) and response_id.strip() else None


def _extract_response_text(response: Any) -> str | None:
    choices = _get_value(response, "choices")
    if not choices:
        return _normalize_content(_get_value(response, "content") or _get_value(response, "text"))
    try:
        first_choice = choices[0]
    except (KeyError, IndexError, TypeError):
        return None

    message = _get_value(first_choice, "message")
    if message is not None:
        content = _get_value(message, "content")
        text = _normalize_content(content)
        if text:
            return text

    return _normalize_content(_get_value(first_choice, "text") or _get_value(first_choice, "content"))


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _normalize_content(content: Any) -> str | None:
    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        text = "".join(parts).strip()
        return text or None
    return None


def _sanitize_response_text(text: str) -> str:
    sanitized = _SECRET_LIKE_TEXT_RE.sub("[redacted-secret]", text)
    sanitized = _LOCAL_PATH_TEXT_RE.sub("[redacted-path]", sanitized).strip()
    if len(sanitized) <= FIRST_CHAT_RESPONSE_TEXT_MAX_LENGTH:
        return sanitized
    return sanitized[: FIRST_CHAT_RESPONSE_TEXT_MAX_LENGTH - 1].rstrip() + "..."


def _classify_failure(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, ChatAuthenticationError):
        return "auth_failed", "Provider authentication failed. Check the saved credentials."
    if isinstance(exc, ChatConfigurationError):
        return "configuration_error", "Provider configuration is incomplete or invalid."
    if isinstance(exc, ChatBadRequestError):
        return "request_invalid", "The provider rejected the first-chat request."
    if isinstance(exc, ChatRateLimitError):
        return "rate_limited", "The provider rate limit was reached. Try again later."
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)):
        return "network_error", "The provider could not be reached."
    if isinstance(exc, ChatProviderError):
        if getattr(exc, "status_code", None) == 504:
            return "network_error", "The provider could not be reached."
        return "provider_error", "The provider returned an error."
    if isinstance(exc, ChatAPIError):
        status_code = getattr(exc, "status_code", None)
        if status_code in {401, 403}:
            return "auth_failed", "Provider authentication failed. Check the saved credentials."
        if status_code == 400:
            return "request_invalid", "The provider rejected the first-chat request."
        if status_code == 429:
            return "rate_limited", "The provider rate limit was reached. Try again later."
        return "provider_error", "The provider returned an error."
    return "provider_error", "First-chat verification failed."

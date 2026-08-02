"""Infrastructure adapter for one bounded prompt-improvement generation call."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import resolve_byok_credentials
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatRateLimitError,
)
from tldw_Server_API.app.core.Chat.chat_route_resolver import (
    ChatRouteResolutionError,
    resolve_chat_route,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    is_model_known_for_provider,
    perform_chat_api_call_async,
    resolve_provider_api_key,
)
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import (
    list_registered_providers,
    provider_requires_api_key,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_improvement import (
    PROMPT_IMPROVEMENT_LIMITS,
)

PROMPT_IMPROVEMENT_MAX_OUTPUT_TOKENS = 1024
PROMPT_IMPROVEMENT_CALL_POLICY = ProviderCallPolicy(
    max_transport_attempts=1,
    allow_streaming=False,
    allow_tools=False,
    allow_stop=False,
    allow_response_format=False,
    candidate_count=1,
    temperature=0.2,
    top_p=0.95,
    privacy_safe_errors=True,
)
_MAX_RETRY_AFTER_SECONDS = 86_400
_REFUSAL_FINISH_REASONS = frozenset(
    {"content_filter", "refusal", "safety", "blocked"}
)

_PUBLIC_MESSAGES = {
    "missing_model": "Select an active chat model and try again.",
    "unsupported_model": "The selected chat model is not available.",
    "provider_not_configured": "The active provider is not configured for this request.",
    "provider_rate_limited": "The active provider is temporarily rate limited.",
    "provider_timeout": "The active provider timed out.",
    "provider_unavailable": "The active provider is temporarily unavailable.",
    "model_refusal": "The active model did not provide an improvement candidate.",
    "invalid_model_output": "The active model returned an unusable response.",
    "internal_error": "The prompt improvement request could not be completed.",
}


@dataclass(frozen=True)
class PromptImprovementDispatchResult:
    """Normalized provider text and the concrete route that produced it."""

    text: str
    provider: str
    model: str
    display_name: str


class PromptImprovementDispatchError(RuntimeError):
    """Sanitized infrastructure failure for endpoint error mapping."""

    def __init__(
        self,
        code: str,
        *,
        internal_detail: object | None = None,
        retryable: bool = False,
        retry_after_seconds: int | None = None,
    ) -> None:
        del internal_detail
        public_message = _PUBLIC_MESSAGES.get(code, _PUBLIC_MESSAGES["internal_error"])
        super().__init__(public_message)
        self.code = code if code in _PUBLIC_MESSAGES else "internal_error"
        self.retryable = bool(retryable)
        self.retry_after_seconds = _bounded_retry_after(retry_after_seconds)


async def dispatch_prompt_improvement(
    *,
    request: Any,
    current_user: Any,
    routing_decision_store: Any,
    selected_model: str,
    provider_hint: str | None,
    messages: list[dict[str, str]],
    request_id: str,
    configured_providers_getter: Callable[[], dict[str, Any]],
) -> PromptImprovementDispatchResult:
    """Resolve the active chat route and perform exactly one generation call."""

    model_snapshot = selected_model.strip() if isinstance(selected_model, str) else ""
    if not model_snapshot:
        raise PromptImprovementDispatchError("missing_model")
    provider_snapshot = provider_hint.strip() if isinstance(provider_hint, str) else None
    if provider_snapshot == "":
        provider_snapshot = None
    if provider_snapshot is None and "/" in model_snapshot:
        inline_provider = model_snapshot.split("/", 1)[0].strip().lower()
        if inline_provider not in set(list_registered_providers()):
            raise PromptImprovementDispatchError("unsupported_model")

    try:
        route_request = ChatCompletionRequest(
            model=model_snapshot,
            api_provider=provider_snapshot,
            messages=[
                {
                    "role": "user",
                    "content": "Resolve the active route for a prompt improvement request.",
                }
            ],
            stream=False,
            tools=None,
        )
    except ValidationError as exc:
        raise PromptImprovementDispatchError(
            "unsupported_model",
            internal_detail=exc,
        ) from exc

    try:
        provider_listing = configured_providers_getter()
        if not isinstance(provider_listing, Mapping):
            raise TypeError("configured provider listing must be a mapping")
        default_provider = str(provider_listing.get("default_provider") or "").strip()
        resolved = await resolve_chat_route(
            route_request,
            request=request,
            sticky_store=routing_decision_store,
            current_user=current_user,
            request_id=request_id,
            configured_providers_getter=lambda: dict(provider_listing),
            surface="prompt_improvement",
            endpoint="POST:/api/v1/prompts/improve",
            scope=None,
            latest_user_turn="",
            requested_capabilities={
                "tools": False,
                "vision": False,
                "json_mode": False,
                "reasoning": False,
            },
            default_provider=default_provider or None,
            metrics_default_provider=default_provider or None,
        )
    except ChatRouteResolutionError as exc:
        code = (
            "missing_model"
            if exc.code == "auto_routing_no_candidates"
            else "provider_unavailable"
        )
        raise PromptImprovementDispatchError(
            code,
            internal_detail=exc,
            retryable=code == "provider_unavailable",
        ) from exc
    except PromptImprovementDispatchError:
        raise
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise PromptImprovementDispatchError(
            "internal_error",
            internal_detail=exc,
        ) from exc

    provider = str(getattr(resolved, "provider", "") or "").strip().lower()
    model = str(getattr(resolved, "model", "") or "").strip()
    if not provider or not model:
        raise PromptImprovementDispatchError("missing_model")
    if (
        len(provider) > PROMPT_IMPROVEMENT_LIMITS.max_provider_chars
        or len(model) > PROMPT_IMPROVEMENT_LIMITS.max_model_chars
    ):
        raise PromptImprovementDispatchError("unsupported_model")
    if model_snapshot.casefold() != "auto":
        availability = is_model_known_for_provider(provider, model)
        if availability is False:
            raise PromptImprovementDispatchError("unsupported_model")

    def fallback_resolver(name: str) -> str | None:
        key, _debug = resolve_provider_api_key(
            name,
            prefer_module_keys_in_tests=True,
        )
        return key

    user_id = _coerce_user_id(current_user)
    try:
        byok = await resolve_byok_credentials(
            provider,
            user_id=user_id,
            request=request,
            fallback_resolver=fallback_resolver,
        )
    except (ChatAuthenticationError, ChatConfigurationError) as exc:
        raise PromptImprovementDispatchError(
            "provider_not_configured",
            internal_detail=exc,
        ) from exc
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise PromptImprovementDispatchError(
            "provider_not_configured",
            internal_detail=exc,
        ) from exc

    api_key = getattr(byok, "api_key", None)
    if provider_requires_api_key(provider) and not (
        isinstance(api_key, str) and api_key.strip()
    ):
        raise PromptImprovementDispatchError("provider_not_configured")

    try:
        try:
            provider_response = await perform_chat_api_call_async(
                api_endpoint=provider,
                messages_payload=messages,
                api_key=api_key,
                model=model,
                max_tokens=PROMPT_IMPROVEMENT_MAX_OUTPUT_TOKENS,
                streaming=False,
                tools=None,
                user_identifier=str(user_id if user_id is not None else "prompt-improvement"),
                app_config=getattr(byok, "app_config", None),
                call_policy=PROMPT_IMPROVEMENT_CALL_POLICY,
            )
        except Exception as exc:  # noqa: BLE001 - translate all provider SDK failures
            raise _map_provider_exception(exc) from exc
    finally:
        try:
            await byok.touch_last_used()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    text, finish_reason = _normalize_provider_response(provider_response)
    if not text.strip():
        if finish_reason in _REFUSAL_FINISH_REASONS:
            raise PromptImprovementDispatchError("model_refusal")
        raise PromptImprovementDispatchError("invalid_model_output")

    return PromptImprovementDispatchResult(
        text=text,
        provider=provider,
        model=model,
        display_name=model,
    )


def _coerce_user_id(current_user: Any) -> int | None:
    value = getattr(current_user, "id_int", None)
    if value is None:
        value = getattr(current_user, "id", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_provider_response(response: Any) -> tuple[str, str | None]:
    """Extract plain assistant text without preserving a provider response body."""

    finish_reason: str | None = None
    content: Any = response
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                finish_reason = _normalized_finish_reason(first.get("finish_reason"))
                message = first.get("message")
                if isinstance(message, Mapping):
                    if message.get("refusal") and not message.get("content"):
                        finish_reason = "refusal"
                    content = message.get("content")
                else:
                    content = first.get("text")
        elif "content" in response:
            content = response.get("content")
        elif "output_text" in response:
            content = response.get("output_text")
    else:
        choices = getattr(response, "choices", None)
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            finish_reason = _normalized_finish_reason(
                getattr(first, "finish_reason", None)
            )
            message = getattr(first, "message", None)
            content = getattr(message, "content", None) if message is not None else None
        elif hasattr(response, "content"):
            content = response.content

    return _normalize_content_text(content), finish_reason


def _normalize_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, (bytes, bytearray)):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        else:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _normalized_finish_reason(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _map_provider_exception(exc: Exception) -> PromptImprovementDispatchError:
    status_code = _exception_status_code(exc)
    retry_after = _exception_retry_after(exc)
    if isinstance(exc, ChatRateLimitError) or status_code == 429:
        return PromptImprovementDispatchError(
            "provider_rate_limited",
            internal_detail=exc,
            retryable=True,
            retry_after_seconds=retry_after,
        )
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or status_code in {408, 504}:
        return PromptImprovementDispatchError(
            "provider_timeout",
            internal_detail=exc,
            retryable=True,
        )
    if isinstance(exc, (ChatAuthenticationError, ChatConfigurationError)) or status_code in {
        401,
        403,
    }:
        return PromptImprovementDispatchError(
            "provider_not_configured",
            internal_detail=exc,
        )
    if isinstance(exc, ChatBadRequestError) or status_code in {400, 404}:
        return PromptImprovementDispatchError(
            "unsupported_model",
            internal_detail=exc,
        )
    return PromptImprovementDispatchError(
        "provider_unavailable",
        internal_detail=exc,
        retryable=True,
    )


def _exception_status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    if value is None:
        value = getattr(getattr(exc, "response", None), "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _exception_retry_after(exc: Exception) -> int | None:
    value = getattr(exc, "retry_after", None)
    if value is None:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if isinstance(headers, Mapping):
            value = headers.get("Retry-After") or headers.get("retry-after")
    return _bounded_retry_after(value)


def _bounded_retry_after(value: Any) -> int | None:
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return min(seconds, _MAX_RETRY_AFTER_SECONDS)


__all__ = [
    "PROMPT_IMPROVEMENT_MAX_OUTPUT_TOKENS",
    "PromptImprovementDispatchError",
    "PromptImprovementDispatchResult",
    "dispatch_prompt_improvement",
]

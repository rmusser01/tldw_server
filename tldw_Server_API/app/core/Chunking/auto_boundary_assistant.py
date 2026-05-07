"""LLM boundary assistant for Auto Chunking plans."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key

_DEFAULT_MAX_EXCERPT_CHARS = 12_000
_DEFAULT_TIMEOUT_SECONDS = 8.0
_DEFAULT_MAX_TOKENS = 600
_MIN_ASSISTANT_MAX_SIZE = 128
_MAX_ASSISTANT_MAX_SIZE = 4_000
_ALLOWED_ASSISTANT_METHODS = {
    "sentences",
    "words",
    "paragraphs",
    "semantic",
    "structure_aware",
    "ebook_chapters",
}
_DERIVED_VIEW_RE = re.compile(r"^[a-z][a-z0-9_:-]{0,63}$")

_SYSTEM_PROMPT = """\
You refine ingestion-time Auto Chunking plans.

Return ONLY one JSON object. Do not include Markdown, comments, or chunk text.

Allowed fields:
- method: one of sentences, words, paragraphs, semantic, structure_aware, ebook_chapters
- max_size: integer from 128 to 4000
- overlap: integer >= 0 and less than max_size
- derived_views: array of short snake_case labels
- rationale: one concise sentence explaining the refinement

Rules:
- Refine only the deterministic plan and bounded excerpt supplied by the server.
- Do not rewrite source content.
- Do not return chunk bodies, file paths to read, URLs to fetch, code, or prompts.
- If the deterministic plan is already appropriate, return the same method, max_size, and overlap with a short rationale.
"""


@dataclass(frozen=True)
class AutoChunkBoundaryAssistantRequest:
    """Input for one bounded boundary-assistant refinement attempt."""

    chunk_options: dict[str, Any]
    chunking_plan: dict[str, Any]
    media_type: str | None = None
    source_name: str | None = None
    extracted_text: str | None = None
    provider: str | None = None
    model: str | None = None
    timeout_sec: float = _DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class AutoChunkBoundaryAssistantResult:
    """Validated assistant result or deterministic fallback marker."""

    used_llm: bool
    chunk_options: dict[str, Any] | None = None
    derived_views: tuple[str, ...] = ()
    rationale: str = ""
    fallback_reason: str | None = None
    provider: str | None = None
    model: str | None = None

    @classmethod
    def success(
        cls,
        *,
        chunk_options: dict[str, Any],
        derived_views: tuple[str, ...] = (),
        rationale: str = "",
        provider: str | None = None,
        model: str | None = None,
    ) -> "AutoChunkBoundaryAssistantResult":
        return cls(
            used_llm=True,
            chunk_options=dict(chunk_options),
            derived_views=tuple(derived_views),
            rationale=str(rationale or "").strip(),
            fallback_reason=None,
            provider=provider,
            model=model,
        )

    @classmethod
    def fallback(
        cls,
        *,
        reason: str,
        rationale: str,
    ) -> "AutoChunkBoundaryAssistantResult":
        return cls(
            used_llm=False,
            chunk_options=None,
            derived_views=(),
            rationale=str(rationale or "").strip(),
            fallback_reason=str(reason or "ai_assist_provider_error"),
        )


class AutoChunkBoundaryAssistant(Protocol):
    """Narrow interface for Auto Chunking boundary refinement."""

    async def refine(self, request: AutoChunkBoundaryAssistantRequest) -> AutoChunkBoundaryAssistantResult:
        """Return validated refinements or a fallback marker."""


@dataclass(frozen=True)
class _Availability:
    available: bool
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    app_config: dict[str, Any] | None = None
    reason: str = ""


class ChatAutoChunkBoundaryAssistant:
    """Boundary assistant backed by the existing async chat service."""

    def __init__(
        self,
        *,
        chat_call: Callable[..., Any] | None = None,
        config_loader: Callable[[], dict[str, Any] | None] | None = None,
        registry_getter: Callable[[], Any] | None = None,
        api_key_resolver: Callable[..., str | None] | None = None,
        provider_requires_key: Callable[[str], bool] | None = None,
        default_provider: str | None = None,
        max_excerpt_chars: int = _DEFAULT_MAX_EXCERPT_CHARS,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._chat_call = chat_call
        self._config_loader = config_loader or ensure_app_config
        self._registry_getter = registry_getter or get_registry
        self._api_key_resolver = api_key_resolver or resolve_provider_api_key_from_config
        self._provider_requires_key = provider_requires_key or provider_requires_api_key
        self._default_provider = default_provider
        self._max_excerpt_chars = max(0, int(max_excerpt_chars))
        self._max_tokens = max(1, int(max_tokens))

    async def refine(self, request: AutoChunkBoundaryAssistantRequest) -> AutoChunkBoundaryAssistantResult:
        try:
            availability = await asyncio.to_thread(self._check_availability, request)
        except Exception as exc:
            logger.debug("Auto Chunking boundary assistant availability check failed: {}", type(exc).__name__)
            return AutoChunkBoundaryAssistantResult.fallback(
                reason="ai_assist_provider_error",
                rationale=f"{type(exc).__name__}: availability check failed.",
            )
        if not availability.available:
            return AutoChunkBoundaryAssistantResult.fallback(
                reason="ai_assist_unavailable",
                rationale=availability.reason or "LLM boundary assistant is unavailable.",
            )

        try:
            raw_response = await asyncio.wait_for(
                self._call_chat(request, availability),
                timeout=max(0.001, float(request.timeout_sec or _DEFAULT_TIMEOUT_SECONDS)),
            )
        except asyncio.TimeoutError:
            return AutoChunkBoundaryAssistantResult.fallback(
                reason="ai_assist_timeout",
                rationale=f"Timed out after {request.timeout_sec:g} seconds.",
            )
        except Exception as exc:
            logger.debug("Auto Chunking boundary assistant provider call failed: {}", type(exc).__name__)
            return AutoChunkBoundaryAssistantResult.fallback(
                reason="ai_assist_provider_error",
                rationale=f"{type(exc).__name__}: provider call failed.",
            )

        response_text = _extract_llm_response_text(raw_response)
        return parse_boundary_assistant_response(
            response_text,
            request=request,
            provider=availability.provider,
            model=availability.model,
        )

    def _check_availability(self, request: AutoChunkBoundaryAssistantRequest) -> _Availability:
        app_config = self._load_config()
        provider = normalize_provider(request.provider or self._default_provider or _configured_default_provider(app_config))
        if not provider:
            return _Availability(False, reason="LLM provider is not configured.")

        registry = self._registry_getter()
        adapter = registry.get_adapter(provider) if registry is not None else None
        if adapter is None:
            return _Availability(False, provider=provider, reason=f"LLM adapter unavailable for provider '{provider}'.")
        provider = _canonical_provider_from_adapter(adapter, fallback=provider)

        model = str(request.model or "").strip() or resolve_provider_model(provider, app_config) or None
        if not model:
            return _Availability(False, provider=provider, reason=f"Model is not configured for provider '{provider}'.")

        api_key = self._resolve_api_key(provider, app_config)
        if self._provider_requires_key(provider) and not api_key:
            return _Availability(
                False,
                provider=provider,
                model=model,
                reason=f"API key is not configured for provider '{provider}'.",
            )

        return _Availability(True, provider=provider, model=model, api_key=api_key, app_config=app_config)

    def _load_config(self) -> dict[str, Any]:
        try:
            loaded = self._config_loader()
        except Exception:
            loaded = {}
        return loaded if isinstance(loaded, dict) else {}

    def _resolve_api_key(self, provider: str, app_config: dict[str, Any]) -> str | None:
        if _callable_accepts_positional_args(self._api_key_resolver, 2):
            return self._api_key_resolver(provider, app_config)
        return self._api_key_resolver(provider)

    async def _call_chat(self, request: AutoChunkBoundaryAssistantRequest, availability: _Availability) -> Any:
        chat_call = self._chat_call
        if chat_call is None:
            from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

            chat_call = perform_chat_api_call_async

        call_kwargs: dict[str, Any] = {
            "api_provider": availability.provider,
            "model": availability.model,
            "api_key": availability.api_key,
            "messages": _build_messages(request, max_excerpt_chars=self._max_excerpt_chars),
            "temperature": 0.1,
            "max_tokens": self._max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "app_config": availability.app_config,
        }
        result = chat_call(**call_kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


def extract_bounded_text_excerpt(text: str | None, *, max_chars: int = _DEFAULT_MAX_EXCERPT_CHARS) -> str:
    """Return a bounded prefix excerpt for assistant context."""
    if not text:
        return ""
    return text[: max(0, int(max_chars))]


def parse_boundary_assistant_response(
    response_text: str | None,
    *,
    request: AutoChunkBoundaryAssistantRequest,
    provider: str | None,
    model: str | None,
) -> AutoChunkBoundaryAssistantResult:
    """Parse and validate strict JSON suggestions from the assistant."""
    text = (response_text or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return AutoChunkBoundaryAssistantResult.fallback(
            reason="ai_assist_invalid_response",
            rationale="Assistant response was not valid JSON.",
        )
    if not isinstance(payload, dict):
        return AutoChunkBoundaryAssistantResult.fallback(
            reason="ai_assist_invalid_response",
            rationale="Assistant response JSON was not an object.",
        )

    try:
        chunk_options, derived_views, rationale = _validate_suggestion(payload, request=request)
    except ValueError as exc:
        return AutoChunkBoundaryAssistantResult.fallback(
            reason="ai_assist_invalid_response",
            rationale=str(exc),
        )

    return AutoChunkBoundaryAssistantResult.success(
        chunk_options=chunk_options,
        derived_views=derived_views,
        rationale=rationale,
        provider=provider,
        model=model,
    )


def append_auto_chunking_fallback(
    chunk_options: dict[str, Any] | None,
    chunking_plan: dict[str, Any],
    reason: str,
    rationale: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return deterministic options and plan with AI-assist fallback metadata."""
    updated_plan = dict(chunking_plan)
    updated_plan["used_llm"] = False
    updated_plan["fallback_reason"] = _append_fallback_reason(updated_plan.get("fallback_reason"), reason)
    updated_plan["rationale"] = _append_rationale(updated_plan.get("rationale"), rationale)
    return (dict(chunk_options) if isinstance(chunk_options, dict) else chunk_options), updated_plan


def apply_auto_chunk_boundary_result(
    chunk_options: dict[str, Any],
    chunking_plan: dict[str, Any],
    result: AutoChunkBoundaryAssistantResult,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a successful assistant result to options and plan metadata."""
    if not result.used_llm or not isinstance(result.chunk_options, dict):
        reason = result.fallback_reason or "ai_assist_provider_error"
        return append_auto_chunking_fallback(chunk_options, chunking_plan, reason, result.rationale)

    refined_options = dict(chunk_options)
    refined_options.update(result.chunk_options)
    updated_plan = dict(chunking_plan)
    updated_plan.update(
        {
            "used_llm": True,
            "method": refined_options.get("method"),
            "max_size": refined_options.get("max_size"),
            "overlap": refined_options.get("overlap"),
            "fallback_reason": None,
            "rationale": result.rationale or "AI assist refined the deterministic Auto Chunking plan.",
        }
    )
    if result.derived_views:
        updated_plan["derived_views"] = list(result.derived_views)
    if result.provider:
        updated_plan["provider"] = result.provider
    if result.model:
        updated_plan["model"] = result.model
    return refined_options, updated_plan


def _validate_suggestion(
    payload: dict[str, Any],
    *,
    request: AutoChunkBoundaryAssistantRequest,
) -> tuple[dict[str, Any], tuple[str, ...], str]:
    base_options = dict(request.chunk_options)
    method = str(payload.get("method") or base_options.get("method") or "").strip()
    if method not in _ALLOWED_ASSISTANT_METHODS:
        raise ValueError(f"Assistant response method '{method}' is not allowed.")
    media_type = str(
        request.media_type
        or (
            request.chunking_plan.get("profile", {}).get("media_type")
            if isinstance(request.chunking_plan.get("profile"), dict)
            else ""
        )
        or ""
    ).strip().lower()
    if method == "ebook_chapters" and media_type != "ebook":
        raise ValueError("Assistant response method 'ebook_chapters' is only allowed for ebook media.")

    max_size = _coerce_int(payload.get("max_size", base_options.get("max_size")), "max_size")
    if max_size < _MIN_ASSISTANT_MAX_SIZE or max_size > _MAX_ASSISTANT_MAX_SIZE:
        raise ValueError(
            f"Assistant response max_size must be between {_MIN_ASSISTANT_MAX_SIZE} and {_MAX_ASSISTANT_MAX_SIZE}."
        )

    overlap = _coerce_int(payload.get("overlap", base_options.get("overlap", 0)), "overlap")
    if overlap < 0 or overlap >= max_size:
        raise ValueError("Assistant response overlap must be non-negative and less than max_size.")

    derived_views = _validate_derived_views(payload.get("derived_views", request.chunking_plan.get("derived_views")))
    rationale = _bounded_string(payload.get("rationale"), max_chars=300)
    if not rationale:
        rationale = "Assistant refined the deterministic Auto Chunking plan."

    refined_options = dict(base_options)
    refined_options.update(
        {
            "method": method,
            "max_size": max_size,
            "overlap": overlap,
        }
    )
    return refined_options, derived_views, rationale


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Assistant response {field_name} must be an integer.") from exc


def _validate_derived_views(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Assistant response derived_views must be a list.")
    views: list[str] = []
    for raw_view in value[:8]:
        if not isinstance(raw_view, str):
            raise ValueError("Assistant response derived_views entries must be strings.")
        view = raw_view.strip()
        if not _DERIVED_VIEW_RE.match(view):
            raise ValueError("Assistant response derived_views entries must be short machine labels.")
        if view not in views:
            views.append(view)
    return tuple(views)


def _bounded_string(value: Any, *, max_chars: int) -> str:
    if value is None:
        return ""
    return str(value).strip()[: max(0, int(max_chars))]


def _append_fallback_reason(existing: Any, reason: str) -> str:
    reasons = [
        item
        for item in str(existing or "").split(";")
        if item and not (item == "ai_assist_unavailable" and reason != "ai_assist_unavailable")
    ]
    if reason not in reasons:
        reasons.append(reason)
    return ";".join(reasons)


def _append_rationale(existing: Any, rationale: str) -> str:
    existing_text = str(existing or "").strip()
    rationale_text = str(rationale or "").strip()
    if not existing_text:
        return rationale_text
    if not rationale_text or rationale_text in existing_text:
        return existing_text
    return f"{existing_text} {rationale_text}"


def _extract_llm_response_text(raw_response: Any) -> str:
    extracted = extract_response_content(raw_response)
    if extracted is not None:
        return extracted
    if isinstance(raw_response, dict):
        for key in ("content", "text"):
            value = raw_response.get(key)
            if isinstance(value, str):
                return value
    return str(raw_response or "")


def _canonical_provider_from_adapter(adapter: Any, *, fallback: str) -> str:
    adapter_name = getattr(adapter, "name", None)
    if isinstance(adapter_name, str) and adapter_name.strip():
        return normalize_provider(adapter_name)
    return fallback


def _callable_accepts_positional_args(func: Callable[..., Any], count: int) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    positional_count = 0
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
    return positional_count >= count


def _configured_default_provider(app_config: dict[str, Any]) -> str | None:
    for section_name in ("llm_api_settings", "API", "Chat-API"):
        section = app_config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in ("default_api", "default_chat_provider"):
            value = section.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    try:
        from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER

        return str(DEFAULT_LLM_PROVIDER or "").strip() or None
    except Exception:
        return None


def _build_messages(
    request: AutoChunkBoundaryAssistantRequest,
    *,
    max_excerpt_chars: int,
) -> list[dict[str, str]]:
    excerpt = extract_bounded_text_excerpt(request.extracted_text, max_chars=max_excerpt_chars)
    payload = {
        "media_type": request.media_type,
        "source_name": request.source_name,
        "deterministic_chunk_options": request.chunk_options,
        "deterministic_chunking_plan": {
            key: request.chunking_plan.get(key)
            for key in (
                "goal",
                "method",
                "max_size",
                "overlap",
                "derived_views",
                "rationale",
                "profile",
            )
        },
        "bounded_excerpt": excerpt,
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

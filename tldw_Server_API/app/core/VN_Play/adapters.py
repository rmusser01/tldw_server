"""VN Play turn adapters."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatConfigurationError,
    ChatRateLimitError,
)

from tldw_Server_API.app.core.VN_Play.models import TurnResult
from tldw_Server_API.app.core.VN_Play.parser import (
    NormalizedTurnResult,
    parse_model_turn,
)


VN_PLAY_SYSTEM_MESSAGE = (
    "You are a VN Play runtime writer. Return only valid JSON with narration, "
    "dialogue, optional choices, optional scene_directives, optional "
    "visual_directives, and summary."
)


class VNPlayModelError(RuntimeError):
    """Raised when the backing chat provider fails."""


class VNGenerationAdapterError(RuntimeError):
    """Raised when a scripted VN generation provider call fails."""

    def __init__(
        self,
        public_error_code: str,
        *,
        debug_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(public_error_code)
        self.public_error_code = public_error_code
        self.debug_metadata = dict(debug_metadata or {})


@dataclass(frozen=True, slots=True)
class VNGenerationCallRequest:
    """Provider call contract for one scripted VN generation attempt."""

    profile_snapshot: Mapping[str, Any]
    messages: Sequence[Mapping[str, Any]]
    output_schema: str
    usage_context: Mapping[str, Any] = field(default_factory=dict)
    system_message: str | None = None


@dataclass(frozen=True, slots=True)
class VNGenerationCallResult:
    """Provider response content plus metadata needed by revision persistence."""

    raw_content: str
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VNGenerationModerationResult:
    """Moderation decision for generated VN output."""

    allowed: bool
    status: str
    public_error_code: str | None = None
    audit_metadata: dict[str, Any] = field(default_factory=dict)
    debug_metadata: dict[str, Any] = field(default_factory=dict)


VN_GENERATION_SYSTEM_MESSAGE = (
    "You are a VN scripted generation writer. Return only valid JSON for the "
    "requested schema. Do not include script labels, next targets, variables, "
    "or control-flow instructions."
)


async def perform_chat_api_call_async(**kwargs: Any) -> Any:
    """Late-bound chat service call so tests can monkeypatch this module safely."""
    from tldw_Server_API.app.core.Chat.chat_service import (
        perform_chat_api_call_async as chat_call,
    )

    return await chat_call(**kwargs)


class DeterministicVNPlayAdapter:
    """Deterministic adapter for mocked UI flows."""

    async def generate_turn(self, context: Any) -> TurnResult:
        input_payload = dict(getattr(context, "input_payload", {}) or {})
        text = str(input_payload.get("input_text") or input_payload.get("choice_id") or "continue")
        return TurnResult(
            narrative_text=f"Echo: {text}",
            dialogue=[{"speaker": "Narrator", "text": f"Echo: {text}"}],
            scene_updates={"mood": "neutral"},
        )


class ChatVNPlayTurnAdapter:
    """Adapter that requests structured VN turn JSON from the chat service."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def generate_turn(self, context: Any) -> NormalizedTurnResult:
        provider = self.provider or _session_setting(context, "provider")
        model = self.model or _session_setting(context, "model")
        messages = _build_messages(context)
        kwargs: dict[str, Any] = {
            "provider": provider,
            "api_endpoint": provider,
            "model": model,
            "messages": messages,
            "system_message": VN_PLAY_SYSTEM_MESSAGE,
            "temp": self.temperature,
            "stream": False,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        try:
            response = await perform_chat_api_call_async(**kwargs)
        except Exception as exc:
            raise VNPlayModelError("vn_play_chat_provider_failed") from exc

        return parse_model_turn(_extract_content(response), mode=_context_mode(context))


class FreeformVNPlayAdapter:
    """Freeform adapter wrapper for future mode-specific prompt shaping."""

    def __init__(self, chat_adapter: ChatVNPlayTurnAdapter | None = None) -> None:
        self.chat_adapter = chat_adapter or ChatVNPlayTurnAdapter()

    async def generate_turn(self, context: Any) -> NormalizedTurnResult:
        return await self.chat_adapter.generate_turn(context)


class StoryVNPlayAdapter:
    """Story/CYOA adapter wrapper for future mode-specific prompt shaping."""

    def __init__(self, chat_adapter: ChatVNPlayTurnAdapter | None = None) -> None:
        self.chat_adapter = chat_adapter or ChatVNPlayTurnAdapter()

    async def generate_turn(self, context: Any) -> NormalizedTurnResult:
        return await self.chat_adapter.generate_turn(context)


class ScriptedVNGenerationAdapter:
    """Adapter for profile-pinned scripted VN generation provider calls."""

    async def generate(self, request: VNGenerationCallRequest) -> VNGenerationCallResult:
        definition = _profile_definition(request.profile_snapshot)
        provider = _first_string(definition, "provider", "api_provider", "api_endpoint")
        model = _first_string(definition, "model", "model_name")
        if not provider or not model:
            raise VNGenerationAdapterError(
                "provider_unavailable",
                debug_metadata={"reason": "profile_snapshot_missing_provider_or_model"},
            )

        kwargs: dict[str, Any] = {
            "provider": provider,
            "api_endpoint": provider,
            "model": model,
            "messages": [dict(message) for message in request.messages],
            "system_message": request.system_message or VN_GENERATION_SYSTEM_MESSAGE,
            "temp": _profile_temperature(definition),
            "stream": False,
            "response_format": {"type": "json_object"},
            "vn_output_schema": request.output_schema,
        }
        max_tokens = _profile_max_tokens(definition)
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        kwargs.update(_usage_context_kwargs(request.usage_context))

        try:
            response = await perform_chat_api_call_async(**kwargs)
        except Exception as exc:
            raise _generation_adapter_error(exc, provider=provider) from exc

        try:
            raw_content = _extract_content(response)
        except VNPlayModelError as exc:
            raise VNGenerationAdapterError(
                "model_error",
                debug_metadata={"provider": provider, "error_type": type(exc).__name__},
            ) from exc

        return VNGenerationCallResult(
            raw_content=raw_content,
            usage_metadata=_extract_usage_metadata(response),
            response_metadata=_extract_response_metadata(response),
        )


class GenerationModerationAdapter:
    """Moderation seam for scripted VN generated output activation."""

    def __init__(self, moderation_service: Any | None = None) -> None:
        self.moderation_service = moderation_service

    async def moderate_output(
        self,
        text: str,
        *,
        profile_snapshot: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> VNGenerationModerationResult:
        definition = _profile_definition(profile_snapshot)
        moderation_required = _moderation_required(definition)
        if not moderation_required:
            return VNGenerationModerationResult(
                allowed=True,
                status="skipped",
                audit_metadata={"moderation_skipped_by_policy": True},
            )

        if self.moderation_service is None:
            return VNGenerationModerationResult(
                allowed=False,
                status="failed",
                public_error_code="moderation_unavailable",
                debug_metadata={"reason": "moderation_service_unavailable"},
            )

        try:
            decision = await _call_moderation_service(
                self.moderation_service,
                text,
                context=dict(context),
            )
        except Exception as exc:
            return VNGenerationModerationResult(
                allowed=False,
                status="failed",
                public_error_code="moderation_unavailable",
                debug_metadata={"error_type": type(exc).__name__},
            )

        allowed = decision.get("allowed")
        if allowed is not True:
            if allowed is not False:
                return VNGenerationModerationResult(
                    allowed=False,
                    status="failed",
                    public_error_code="moderation_unavailable",
                    debug_metadata={
                        "reason": "moderation_result_missing_allowed",
                        "result_type": type(allowed).__name__,
                    },
                )
            return VNGenerationModerationResult(
                allowed=False,
                status="blocked",
                public_error_code=str(decision.get("public_error_code") or "moderation_blocked"),
                debug_metadata={key: value for key, value in decision.items() if key != "allowed"},
            )
        return VNGenerationModerationResult(
            allowed=True,
            status="passed",
            audit_metadata={key: value for key, value in decision.items() if key != "allowed"},
        )


def _build_messages(context: Any) -> list[dict[str, str]]:
    payload = {
        "mode": _context_mode(context),
        "input": dict(getattr(context, "input_payload", {}) or {}),
        "scene_state": _public_attrs(getattr(context, "scene_state", None)),
        "recent_events": _summarize_events(getattr(context, "recent_events", []) or []),
        "session": _session_summary(getattr(context, "session", None)),
        "instructions": {
            "freeform": "Continue the scene with concise narration and dialogue.",
            "story": "Return two to five choices when the session mode is story.",
        },
    }
    return [{"role": "user", "content": json.dumps(payload, sort_keys=True)}]


def _extract_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, Sequence) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                content = first.get("text")
                if isinstance(content, str):
                    return content
        content = response.get("content")
        if isinstance(content, str):
            return content
    raise VNPlayModelError("vn_play_chat_response_missing_content")


def _profile_definition(profile_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    definition = profile_snapshot.get("definition")
    if isinstance(definition, Mapping):
        return dict(definition)
    return dict(profile_snapshot)


def _first_string(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _profile_temperature(definition: Mapping[str, Any]) -> float:
    value = definition.get("temperature")
    if isinstance(value, (int, float)):
        return float(value)
    return 0.7


def _profile_max_tokens(definition: Mapping[str, Any]) -> int | None:
    for key in ("max_output_tokens", "max_tokens"):
        value = definition.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _usage_context_kwargs(usage_context: Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "vn_session_id",
        "script_id",
        "script_version_id",
        "generation_id",
        "generation_request_id",
        "generation_revision_id",
        "generation_profile_key",
        "generation_profile_snapshot_id",
        "generation_point_key",
    }
    return {key: usage_context[key] for key in allowed_keys if key in usage_context}


def _generation_adapter_error(exc: Exception, *, provider: str) -> VNGenerationAdapterError:
    status_code = getattr(exc, "status_code", None)
    exc_name = type(exc).__name__.lower()
    message = str(exc).lower()
    if isinstance(exc, TimeoutError) or status_code in {408, 504} or "timeout" in exc_name or "timeout" in message:
        public_error_code = "model_timeout"
    elif isinstance(exc, (ChatConfigurationError, ChatRateLimitError)) or status_code in {401, 403, 404, 429, 503}:
        public_error_code = "provider_unavailable"
    else:
        public_error_code = "model_error"
    debug_metadata = {
        "provider": provider,
        "error_type": type(exc).__name__,
    }
    if isinstance(exc, ChatAPIError):
        debug_metadata["status_code"] = exc.status_code
    return VNGenerationAdapterError(public_error_code, debug_metadata=debug_metadata)


def _extract_usage_metadata(response: Any) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        return {}
    usage = response.get("usage") or response.get("usage_metadata")
    if not isinstance(usage, Mapping):
        return {}
    return _json_safe_mapping(usage)


def _extract_response_metadata(response: Any) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        return {}
    metadata: dict[str, Any] = {}
    for key in ("id", "model", "created", "system_fingerprint"):
        value = response.get(key)
        if value is not None:
            metadata[key] = value
    return metadata


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False))
    except (TypeError, ValueError):
        return {
            str(key): item
            for key, item in value.items()
            if isinstance(item, (str, int, float, bool)) or item is None
        }


def _moderation_required(definition: Mapping[str, Any]) -> bool:
    if definition.get("moderation_required") is True:
        return True
    hosting = _first_string(
        definition,
        "hosting",
        "deployment",
        "provider_hosting",
        "provider_class",
        "deployment_class",
    )
    hosting = hosting.lower() if hosting is not None else None
    if hosting in {"hosted", "public"}:
        return True
    moderation = definition.get("moderation")
    if isinstance(moderation, Mapping) and "required" in moderation:
        return bool(moderation["required"])
    return False


async def _call_moderation_service(
    moderation_service: Any,
    text: str,
    *,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    if hasattr(moderation_service, "moderate"):
        result = moderation_service.moderate(text, context=context)
    elif callable(moderation_service):
        result = moderation_service(text, context=context)
    else:
        raise TypeError("moderation_service_not_callable")
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, Mapping):
        raise TypeError("moderation_result_must_be_mapping")
    return result


def _context_mode(context: Any) -> str:
    session = getattr(context, "session", None)
    return str(getattr(session, "mode", None) or "freeform")


def _session_setting(context: Any, key: str) -> Any:
    session = getattr(context, "session", None)
    settings = getattr(session, "settings", {}) or {}
    if isinstance(settings, Mapping):
        return settings.get(key)
    return None


def _session_summary(session: Any) -> dict[str, Any]:
    if session is None:
        return {}
    return {
        "id": getattr(session, "id", None),
        "mode": getattr(session, "mode", None),
        "title": getattr(session, "title", None),
        "content_rating": getattr(session, "content_rating", None),
        "scene_version": getattr(session, "scene_version", None),
    }


def _summarize_events(events: Sequence[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for event in events[-12:]:
        if isinstance(event, Mapping):
            summaries.append(
                {
                    "sequence_number": event.get("sequence_number"),
                    "event_type": event.get("event_type"),
                    "event_payload": event.get("event_payload"),
                }
            )
    return summaries


def _public_attrs(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {
        key: getattr(value, key)
        for key in dir(value)
        if not key.startswith("_") and not callable(getattr(value, key))
    }

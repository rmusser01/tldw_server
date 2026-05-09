"""VN Play turn adapters."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

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

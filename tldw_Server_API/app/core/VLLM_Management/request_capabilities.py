"""Helpers for inferring managed vLLM capability requirements from requests."""

from __future__ import annotations

from typing import Any, Iterable


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _iter_message_content_parts(messages: Iterable[Any] | None) -> Iterable[Any]:
    for message in messages or []:
        content = _get_value(message, "content")
        if isinstance(content, list):
            for part in content:
                yield part


def infer_chat_request_capabilities(messages: Iterable[Any] | None) -> tuple[str, ...]:
    """Infer required managed-instance capabilities from chat message content.

    The returned tuple always includes ``chat``. Known non-text content parts add
    their modality-specific requirements so callers can reject incompatible
    managed instances before dispatching provider requests.
    """

    capabilities = {"chat"}
    for part in _iter_message_content_parts(messages):
        part_type = str(_get_value(part, "type") or "").strip().lower()
        if part_type == "image_url":
            capabilities.add("vision")
        elif part_type in {"input_audio", "audio"}:
            capabilities.add("audio")
    return tuple(sorted(capabilities))


__all__ = ["infer_chat_request_capabilities"]

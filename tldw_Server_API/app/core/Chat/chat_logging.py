"""Safe Chat logging summaries.

Helpers in this module intentionally describe shape, counts, and text lengths
without returning user, prompt, tool, image, or assistant content.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

_ALLOWED_MESSAGE_ROLES = {"assistant", "developer", "function", "system", "tool", "user"}
_ALLOWED_CONTENT_PART_TYPES = {"file", "image_url", "input_audio", "refusal", "text"}


def text_summary(value: Any) -> dict[str, Any]:
    """Return a text presence/type/length summary without exposing content."""
    if value is None:
        return {"present": False, "type": "NoneType", "chars": 0}
    if isinstance(value, bytes):
        return {"present": True, "type": "bytes", "chars": len(value)}
    text = value if isinstance(value, str) else str(value)
    return {"present": True, "type": type(value).__name__, "chars": len(text)}


def _text_list_summary(values: Any) -> dict[str, Any]:
    if values is None:
        return {"count": 0, "items": []}
    if isinstance(values, str) or not isinstance(values, Sequence):
        values = [values]
    items = [text_summary(value) for value in values]
    return {"count": len(items), "items": items}


def _safe_bucket(value: Any, allowed_values: set[str], *, fallback: str = "unknown") -> str:
    if value is None:
        return fallback
    text = str(value)
    if text in allowed_values:
        return text
    return "other"


def prompt_template_summary(
    *,
    template_name: str | None,
    system_message: Any,
    payload_system_messages: Any = None,
    request_system_messages: Any = None,
    character_name: str | None = None,
) -> dict[str, Any]:
    """Summarize prompt templating inputs and outputs without prompt text."""
    return {
        "template_name": template_name,
        "system_message": text_summary(system_message),
        "payload_system_messages": _text_list_summary(payload_system_messages),
        "request_system_messages": _text_list_summary(request_system_messages),
        "character_name": character_name,
    }


def _content_summary(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"kind": "text", "chars": len(content)}
    if isinstance(content, list):
        part_kinds: Counter[str] = Counter()
        text_chars = 0
        image_count = 0
        for part in content:
            if isinstance(part, Mapping):
                part_type = _safe_bucket(part.get("type", "unknown"), _ALLOWED_CONTENT_PART_TYPES)
                part_kinds[part_type] += 1
                if part_type == "text":
                    text_value = part.get("text")
                    if isinstance(text_value, str):
                        text_chars += len(text_value)
                    elif text_value is not None:
                        text_chars += len(str(text_value))
                elif part_type == "image_url":
                    image_count += 1
            else:
                part_kinds[type(part).__name__] += 1
        return {
            "kind": "parts",
            "count": len(content),
            "part_kinds": dict(sorted(part_kinds.items())),
            "text_chars": text_chars,
            "image_count": image_count,
        }
    if content is None:
        return {"kind": "none", "chars": 0}
    return {"kind": type(content).__name__, "chars": len(str(content))}


def message_payload_summary(messages: Any) -> dict[str, Any]:
    """Summarize chat message payload shape without message content."""
    if not isinstance(messages, list):
        return {"kind": type(messages).__name__, "count": 0}

    roles: Counter[str] = Counter()
    summaries: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, Mapping):
            role = _safe_bucket(message.get("role", "unknown"), _ALLOWED_MESSAGE_ROLES)
            metadata = message.get("metadata")
            roles[role] += 1
            summaries.append(
                {
                    "role": role,
                    "content": _content_summary(message.get("content")),
                    "has_tool_calls": bool(message.get("tool_calls")),
                    "has_function_call": bool(message.get("function_call")),
                    "has_metadata": isinstance(metadata, Mapping) and bool(metadata),
                    "metadata_key_count": len(metadata) if isinstance(metadata, Mapping) else 0,
                }
            )
        else:
            role = type(message).__name__
            roles[role] += 1
            summaries.append({"role": role, "content": {"kind": role, "chars": len(str(message))}})
    return {"kind": "list", "count": len(messages), "roles": dict(sorted(roles.items())), "messages": summaries}


def tool_payload_summary(value: Any) -> dict[str, Any]:
    """Summarize tool payloads without arguments, outputs, or errors."""
    if isinstance(value, list):
        item_kinds = Counter(type(item).__name__ for item in value)
        return {"kind": "list", "count": len(value), "item_kinds": dict(sorted(item_kinds.items()))}
    if isinstance(value, Mapping):
        return {
            "kind": "dict",
            "key_count": len(value),
            "has_arguments": "arguments" in value,
            "has_output": "output" in value or "result" in value,
            "has_error": "error" in value,
        }
    if value is None:
        return {"kind": "none"}
    return {"kind": type(value).__name__}


def exception_summary(exc: BaseException) -> dict[str, Any]:
    """Summarize exception metadata without rendering the exception message."""
    summary: dict[str, Any] = {"type": type(exc).__name__}
    for attr in ("status_code", "status", "provider"):
        value = getattr(exc, attr, None)
        if value is not None:
            summary[attr] = value
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            status_code = getattr(response, "status", None)
        if status_code is not None:
            summary["response_status_code"] = status_code
    return summary


def response_summary(response: Any) -> dict[str, Any]:
    """Summarize a provider response without generated assistant content."""
    if isinstance(response, str):
        return {"kind": "str", "chars": len(response)}
    if isinstance(response, bytes):
        return {"kind": "bytes", "chars": len(response)}
    if isinstance(response, Mapping):
        summary: dict[str, Any] = {
            "kind": "dict",
            "key_count": len(response),
            "has_choices": "choices" in response,
            "has_usage": "usage" in response,
            "has_error": "error" in response,
        }
        choices = response.get("choices")
        if isinstance(choices, list):
            summary["choices"] = {"count": len(choices)}
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            summary["usage_key_count"] = len(usage)
        return summary
    if hasattr(response, "__aiter__"):
        return {"kind": "async_iterator", "type": type(response).__name__}
    if hasattr(response, "__iter__") and not isinstance(response, (str, bytes, dict)):
        return {"kind": "iterator", "type": type(response).__name__}
    if response is None:
        return {"kind": "none"}
    return {"kind": type(response).__name__}

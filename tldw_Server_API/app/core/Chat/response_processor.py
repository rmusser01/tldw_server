from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

_EXPECTED_CONTENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)
_MISSING = object()


@dataclass
class NonStreamChoice:
    index: int
    choice: dict[str, Any]
    message: dict[str, Any]
    content: Any | None
    content_text: str
    tool_calls: Any | None
    function_call: Any | None


def _safe_getattr(obj: Any, name: str) -> Any | None:
    try:
        return getattr(obj, name, None)
    except _EXPECTED_CONTENT_EXCEPTIONS:
        return None


def _safe_str(content: Any) -> str:
    try:
        return str(content)
    except _EXPECTED_CONTENT_EXCEPTIONS:
        return ""


def extract_text_from_content(content: Any | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif "text" in item and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            else:
                item_type = _safe_getattr(item, "type")
                item_text = _safe_getattr(item, "text")
                if item_type == "text" and isinstance(item_text, str):
                    parts.append(item_text)
        return "\n".join(parts)
    return _safe_str(content)


def collect_non_stream_choices(llm_response: Any) -> list[NonStreamChoice]:
    if not isinstance(llm_response, dict):
        return []
    raw_choices = llm_response.get("choices")
    if not isinstance(raw_choices, list):
        return []
    choices: list[NonStreamChoice] = []
    for index, raw_choice in enumerate(raw_choices):
        if not isinstance(raw_choice, dict):
            continue
        message = raw_choice.get("message")
        if not isinstance(message, dict):
            message = {}
            raw_choice["message"] = message
        content = message.get("content")
        choices.append(
            NonStreamChoice(
                index=index,
                choice=raw_choice,
                message=message,
                content=content,
                content_text=extract_text_from_content(content),
                tool_calls=message.get("tool_calls"),
                function_call=message.get("function_call"),
            )
        )
    return choices


def set_choice_content(choice: NonStreamChoice, content: Any | None) -> None:
    choice.message["content"] = content
    choice.content = content
    choice.content_text = extract_text_from_content(content)


def apply_redaction_to_content(content: Any | None, redact_text: Callable[[str], str]) -> Any | None:
    if isinstance(content, str):
        return redact_text(content)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            redacted_content = dict(content)
            redacted_content["text"] = redact_text(redacted_content["text"])
            return redacted_content
        return content
    if isinstance(content, list):
        redacted_items: list[Any] = []
        for item in content:
            if isinstance(item, dict):
                new_item = dict(item)
                if isinstance(new_item.get("text"), str):
                    new_item["text"] = redact_text(new_item["text"])
                redacted_items.append(new_item)
            elif isinstance(item, str):
                redacted_items.append(redact_text(item))
            else:
                item_type = _safe_getattr(item, "type")
                item_text = _safe_getattr(item, "text")
                if item_type == "text" and isinstance(item_text, str):
                    new_text = redact_text(item_text)
                    model_copy = _safe_getattr(item, "model_copy")
                    if callable(model_copy):
                        try:
                            redacted_items.append(model_copy(update={"text": new_text}))
                            continue
                        except _EXPECTED_CONTENT_EXCEPTIONS:
                            pass
                    copy_method = _safe_getattr(item, "copy")
                    if callable(copy_method):
                        try:
                            redacted_items.append(copy_method(update={"text": new_text}))
                            continue
                        except _EXPECTED_CONTENT_EXCEPTIONS:
                            pass
                    redacted_items.append({"type": "text", "text": new_text})
                else:
                    redacted_items.append(item)
        return redacted_items
    content_text = _safe_str(content)
    if content_text:
        return redact_text(content_text)
    return content


def estimate_completion_tokens_from_choices(choices: list[NonStreamChoice]) -> int:
    return sum(max(0, len(choice.content_text) // 4) for choice in choices)


def primary_choice(choices: list[NonStreamChoice]) -> NonStreamChoice | None:
    return choices[0] if choices else None


def inject_assistant_name_into_choices(choices: list[NonStreamChoice], assistant_name: str | None) -> None:
    if not assistant_name:
        return
    for choice in choices:
        if not choice.message.get("name"):
            choice.message["name"] = assistant_name


def validate_structured_choices(
    *,
    choices: list[NonStreamChoice],
    structured_request_context: Any,
    validate_structured_response: Callable[..., dict[str, Any] | None],
    fallback_content: Any = _MISSING,
) -> dict[str, Any] | None:
    if not choices:
        if fallback_content is _MISSING:
            return None
        return validate_structured_response(
            raw_text=fallback_content,
            structured_request_context=structured_request_context,
        )

    metadata_by_choice: list[tuple[int, dict[str, Any]]] = []
    for choice in choices:
        metadata = validate_structured_response(
            raw_text=choice.content,
            structured_request_context=structured_request_context,
        )
        if metadata is not None:
            metadata_by_choice.append((choice.index, metadata))
    if not metadata_by_choice:
        return None
    if len(choices) == 1 and len(metadata_by_choice) == 1:
        return metadata_by_choice[0][1]
    if len(metadata_by_choice) == 1:
        choice_index, metadata = metadata_by_choice[0]
        return {"choice_index": choice_index, **metadata}
    return {
        "choices": [
            {"choice_index": choice_index, **metadata}
            for choice_index, metadata in metadata_by_choice
        ]
    }

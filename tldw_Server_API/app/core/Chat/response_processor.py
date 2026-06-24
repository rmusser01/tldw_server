from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class NonStreamChoice:
    index: int
    choice: dict[str, Any]
    message: dict[str, Any]
    content: Any | None
    content_text: str
    tool_calls: Any | None
    function_call: Any | None


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
        return "\n".join(parts)
    return str(content)


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
                redacted_items.append(item)
        return redacted_items
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
) -> dict[str, Any] | None:
    metadata_by_choice: list[dict[str, Any]] = []
    for choice in choices:
        metadata = validate_structured_response(
            raw_text=choice.content,
            structured_request_context=structured_request_context,
        )
        if metadata is not None:
            metadata_by_choice.append({"choice_index": choice.index, **metadata})
    if not metadata_by_choice:
        return None
    if len(metadata_by_choice) == 1:
        return metadata_by_choice[0]
    return {"choices": metadata_by_choice}

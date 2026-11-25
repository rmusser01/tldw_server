"""Utilities for normalizing and mutating non-streaming chat responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
from typing import TYPE_CHECKING
import json as _json

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Chat.call_state import NonStreamCall

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
    """Normalized view of a chat completion choice and its assistant message."""

    index: int
    choice: dict[str, Any]
    message: dict[str, Any]
    content: Any | None
    content_text: str
    tool_calls: Any | None
    function_call: Any | None


def _safe_getattr(obj: Any, name: str) -> Any | None:
    """Return an attribute value without letting unusual response objects fail extraction."""

    try:
        return getattr(obj, name, None)
    except _EXPECTED_CONTENT_EXCEPTIONS:
        return None


def _safe_str(content: Any) -> str:
    """Convert content to text while treating provider object failures as empty text."""

    try:
        return str(content)
    except _EXPECTED_CONTENT_EXCEPTIONS:
        return ""


def _extract_dict_text(content: dict[str, Any]) -> str:
    """Extract human-visible text from dict-shaped message content."""

    if content.get("type") == "text" and isinstance(content.get("text"), str):
        return content["text"]
    if isinstance(content.get("text"), str):
        return content["text"]
    return ""


def extract_text_from_content(content: Any | None) -> str:
    """Extract text used for moderation and usage accounting from assistant content."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _extract_dict_text(content)
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
    """Collect assistant message choices from a non-streaming provider response."""

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
            continue
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
    """Update a choice message in place and keep the normalized text cache in sync."""

    choice.message["content"] = content
    choice.content = content
    choice.content_text = extract_text_from_content(content)


def apply_redaction_to_content(content: Any | None, redact_text: Callable[[str], str]) -> Any | None:
    """Apply text redaction while preserving provider content container shapes."""

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
    """Estimate completion token usage from normalized choice text."""

    return sum(max(0, len(choice.content_text) // 4) for choice in choices)


def primary_choice(choices: list[NonStreamChoice]) -> NonStreamChoice | None:
    """Return the choice with OpenAI-compatible index zero, if present."""

    return next((choice for choice in choices if choice.index == 0), None)


def inject_assistant_name_into_choices(choices: list[NonStreamChoice], assistant_name: str | None) -> None:
    """Populate assistant names on choices that do not already provide one."""

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
    """Run structured-response validation for each normalized choice."""

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


def _refresh_primary_choice_state(cst: "NonStreamCall") -> None:
    cst.first_choice = cs.primary_choice(cst.processed_choices)
    if cst.first_choice is not None:
        cst.content_to_save = cst.first_choice.content
        cst.tool_calls_to_save = cst.first_choice.tool_calls
        cst.function_call_to_save = cst.first_choice.function_call
        cst.content_text_for_usage = cst.first_choice.content_text
    else:
        cst.content_text_for_usage = cs._extract_text_from_content(cst.content_to_save)


async def _apply_output_safety_to_current_choices(cst: "NonStreamCall") -> None:

    result = await cs.apply_output_safety_to_choices(
        choices=cst.processed_choices,
        fallback_content=cst.content_to_save,
        fallback_content_text=cst.content_text_for_usage,
        runtime=cs.OutputModerationRuntime(
            request=cst.request,
            client_id=cst.client_id,
            conversation_id=cst.final_conversation_id,
            metrics=cst.metrics,
            audit_service=cst.audit_service,
            audit_context=cst.audit_context,
            moderation_getter=cst.moderation_getter,
            self_monitoring_service=cst.self_monitoring_service,
            topic_monitoring_getter=cs.get_topic_monitoring_service,
            review_item_callback=cs._capture_moderation_review_item_safely_async,
            completion_metric_callback=lambda outcome: cs._emit_chat_run_first_completion_metric(
                cst.metrics,
                context=cst.run_first_metric_context,
                outcome=outcome,
            ),
        ),
    )
    cst.content_to_save = result.content_to_save
    cst.content_text_for_usage = result.content_text_for_usage
    _refresh_primary_choice_state(cst)


async def collect_non_stream_reply(cst: "NonStreamCall") -> None:
    """Normalise the reply and collect its choices."""
    if isinstance(cst.llm_response, str) and cs.should_force_normalize_string_responses():
        cst.llm_response = cs._wrap_raw_string_response(cst.llm_response, cst.model)

    cst.processed_choices: list[cs.NonStreamChoice] = cs.collect_non_stream_choices(cst.llm_response)
    cst.first_choice = cs.primary_choice(cst.processed_choices)
    cst.content_to_save: Any | None = cst.first_choice.content if cst.first_choice else None
    cst.tool_calls_to_save: Any | None = cst.first_choice.tool_calls if cst.first_choice else None
    cst.function_call_to_save: Any | None = cst.first_choice.function_call if cst.first_choice else None
    cst.first_turn_tool_calls: Any | None = cst.tool_calls_to_save
    cst.first_turn_function_call: Any | None = cst.function_call_to_save
    if cst.llm_response and isinstance(cst.llm_response, dict):
        await _completion_pipeline._record_dict_response_usage(
            cst,
            response=cst.llm_response,
            messages=cst.templated_llm_payload,
            content=cst.content_to_save,
            provider_name=cst.selected_provider,
            model_name=cst.model,
            started_at=cst.llm_start_time,
        )
    elif isinstance(cst.llm_response, str):
        cst.content_to_save = cst.llm_response
        await _completion_pipeline._record_dict_response_usage(
            cst,
            response={},
            messages=cst.templated_llm_payload,
            content=cst.content_to_save,
            provider_name=cst.selected_provider,
            model_name=cst.model,
            started_at=cst.llm_start_time,
        )
    elif cst.llm_response is None:
        cs._emit_chat_run_first_completion_metric(
            cst.metrics,
            context=cst.run_first_metric_context,
            outcome="error",
        )
        raise cs.ChatProviderError(provider=cst.provider, message="Provider unavailable or returned no response", status_code=502)

    # Cache content text for moderation/usage when content is non-string
    cst.content_text_for_usage = cs._extract_text_from_content(cst.content_to_save)


async def encode_non_stream_payload(cst: "NonStreamCall") -> None:
    """Run-first metrics, the assistant name, the encoding and the tldw_* metadata."""
    cs._emit_chat_run_first_tool_path_metrics(
        cst.metrics,
        context=cst.run_first_metric_context,
        tool_calls=cst.first_turn_tool_calls,
        function_call=cst.first_turn_function_call,
    )
    cs._emit_chat_run_first_completion_metric(
        cst.metrics,
        context=cst.run_first_metric_context,
        outcome="success",
    )

    if cs.INJECT_ASSISTANT_NAME and isinstance(cst.llm_response, dict):
        try:
            asst_name = cs.sanitize_sender_name(
                cst.character_card_for_context.get("name") if cst.character_card_for_context else None
            )
            cs.inject_assistant_name_into_choices(cs.collect_non_stream_choices(cst.llm_response), asst_name)
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            pass

    # Encode payload (large responses via CPU-bound handler)
    if cst.llm_response and isinstance(cst.llm_response, dict) and len(str(cst.llm_response)) > 10000:
        encoded_json = await cs.process_large_json_async(cst.llm_response)
        cst.encoded_payload = _json.loads(encoded_json)
    else:
        cst.encoded_payload = await cst.current_loop.run_in_executor(None, cs.jsonable_encoder, cst.llm_response)

    if isinstance(cst.encoded_payload, dict):
        if cs.INJECT_ASSISTANT_NAME:
            try:
                asst_name = cs.sanitize_sender_name(
                    cst.character_card_for_context.get("name") if cst.character_card_for_context else None
                )
                cs.inject_assistant_name_into_choices(cs.collect_non_stream_choices(cst.encoded_payload), asst_name)
            except cs._CHAT_NONCRITICAL_EXCEPTIONS:
                pass
        cst.encoded_payload["tldw_conversation_id"] = cst.final_conversation_id
        if cst.user_message_id:
            cst.encoded_payload["tldw_user_message_id"] = cst.user_message_id
        if cst.assistant_message_id:
            cst.encoded_payload["tldw_message_id"] = cst.assistant_message_id
        if cst.system_message_id:
            cst.encoded_payload["tldw_system_message_id"] = cst.system_message_id
        if cst.tool_execution_payload is not None:
            cst.encoded_payload["tldw_tool_results"] = cst.tool_execution_payload
        if cst.tool_auto_continue_meta is not None:
            cst.encoded_payload["tldw_tool_auto_continue"] = cst.tool_auto_continue_meta
        if cst.normalized_continuation_metadata is not None:
            cst.encoded_payload["tldw_continuation"] = cst.normalized_continuation_metadata
        if cst.structured_metadata is not None:
            cst.encoded_payload["tldw_structured"] = cst.structured_metadata
        if cst.prompt_guardrail_decision is not None:
            cst.encoded_payload["tldw_prompt_guardrails"] = (
                cst.prompt_guardrail_decision.to_response_metadata()
            )


# late imports: moved code reads these at call time (TASK-13950.25 R5)
from tldw_Server_API.app.core.Chat import completion_pipeline as _completion_pipeline
from tldw_Server_API.app.core.Chat import chat_service as cs

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.streaming_utils import (
    MAX_TOOL_ARGUMENT_LENGTH,
    MAX_TOOL_CALL_INDEX,
    invoke_stream_close_bounded,
    normalize_provider_stream_error,
    provider_result_contains_error,
)


def _blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if text is not None:
                parts.append(str(text))
    return "".join(parts)


def _system_to_text(system: Any) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        text = _blocks_to_text([b for b in system if isinstance(b, dict)])
        return text if text else None
    return None


def _image_block_to_openai_part(block: dict[str, Any]) -> dict[str, Any] | None:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    src_type = source.get("type")
    if src_type == "base64":
        media_type = source.get("media_type") or "application/octet-stream"
        data = source.get("data") or ""
        url = f"data:{media_type};base64,{data}"
    elif src_type == "url":
        url = source.get("url")
    else:
        return None
    if not isinstance(url, str) or not url:
        return None
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_to_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _blocks_to_text([b for b in content if isinstance(b, dict)])
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=True)
    except Exception:
        return str(content)


def _tool_use_to_openai_call(block: dict[str, Any], tool_index: int) -> dict[str, Any] | None:
    name = block.get("name")
    tool_id = block.get("id") or f"tool_{tool_index}"
    if not isinstance(name, str) or not name:
        return None
    arguments = block.get("input")
    try:
        arguments_json = json.dumps(arguments, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        arguments_json = "{}"
    return {
        "index": tool_index,
        "id": tool_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments_json},
    }


def anthropic_messages_to_openai(
    messages: list[dict[str, Any]],
    system: Any | None,
) -> tuple[list[dict[str, Any]], str | None]:
    system_message = _system_to_text(system)
    openai_messages: list[dict[str, Any]] = []
    tool_result_counter = 0

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in {"user", "assistant"}:
            continue

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            openai_messages.append({"role": role, "content": ""})
            continue

        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_index = 0
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text")
                    if text is not None:
                        text_parts.append(str(text))
                elif block_type == "tool_use":
                    call = _tool_use_to_openai_call(block, tool_index)
                    if call:
                        tool_calls.append(call)
                        tool_index += 1
            message_payload: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts),
            }
            if tool_calls:
                message_payload["tool_calls"] = tool_calls
            openai_messages.append(message_payload)
            continue

        # user role with mixed content
        user_parts: list[dict[str, Any]] = []
        has_image = False

        def _flush_user_parts() -> None:
            nonlocal user_parts, has_image
            if not user_parts:
                return
            if has_image or len(user_parts) > 1:
                openai_messages.append({"role": "user", "content": list(user_parts)})
            else:
                text_part = user_parts[0]
                openai_messages.append({"role": "user", "content": text_part.get("text", "")})
            user_parts = []
            has_image = False

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if text is not None:
                    user_parts.append({"type": "text", "text": str(text)})
            elif block_type == "image":
                part = _image_block_to_openai_part(block)
                if part:
                    user_parts.append(part)
                    has_image = True
            elif block_type == "tool_result":
                _flush_user_parts()
                tool_id = block.get("tool_use_id") or block.get("id")
                if not tool_id:
                    tool_id = f"tool_result_{tool_result_counter}"
                    tool_result_counter += 1
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": _tool_result_to_text(block),
                    }
                )
            else:
                # Ignore unknown user blocks to avoid injecting unsupported content types.
                continue

        _flush_user_parts()

    return openai_messages, system_message


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = tool.get("description")
        input_schema = tool.get("input_schema")
        payload: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or "",
                "parameters": input_schema if isinstance(input_schema, dict) else {},
            },
        }
        converted.append(payload)
    return converted or None


def anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if choice is None:
        return None
    if isinstance(choice, str):
        lowered = choice.lower().strip()
        if lowered in {"auto", "none"}:
            return lowered
        if lowered == "any":
            return "required"
        # Treat other strings as tool name hints
        return {"type": "function", "function": {"name": choice}}
    if isinstance(choice, dict):
        tool_type = choice.get("type")
        if tool_type == "tool":
            name = choice.get("name")
            if isinstance(name, str) and name:
                return {"type": "function", "function": {"name": name}}
        if tool_type == "any":
            return "required"
        if tool_type == "auto":
            return "auto"
    return choice


def _openai_content_to_blocks(content: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
        return blocks
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text")
                if text is not None:
                    blocks.append({"type": "text", "text": str(text)})
            elif ptype == "image_url":
                image_url = part.get("image_url") or {}
                url = image_url.get("url")
                if isinstance(url, str) and url:
                    blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _finish_reason_to_stop_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "refusal",
    }
    return mapping.get(reason)


def openai_response_to_anthropic(response: dict[str, Any], *, model: str | None) -> dict[str, Any]:
    choice = None
    if isinstance(response.get("choices"), list) and response["choices"]:
        choice = response["choices"][0]
    message = (choice or {}).get("message") or {}
    content_blocks = _openai_content_to_blocks(message.get("content"))

    tool_calls = message.get("tool_calls") or []
    legacy_function_call = message.get("function_call")
    if not tool_calls and isinstance(legacy_function_call, dict):
        tool_calls = [{"function": legacy_function_call}]
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") or {}
            name = func.get("name") or ""
            args = func.get("arguments")
            input_obj: Any = {}
            if isinstance(args, str):
                try:
                    input_obj = json.loads(args)
                except Exception:
                    input_obj = args
            elif args is not None:
                input_obj = args
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or f"tool_{len(content_blocks)}",
                    "name": name,
                    "input": input_obj,
                }
            )

    finish_reason = (choice or {}).get("finish_reason")
    usage = response.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    msg_id = response.get("id") or f"msg_{uuid.uuid4().hex}"
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model or response.get("model"),
        "content": content_blocks,
        "stop_reason": _finish_reason_to_stop_reason(finish_reason),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def _sse_event(event_type: str, payload: dict[str, Any]) -> str:
    if "type" not in payload:
        payload = dict(payload)
        payload["type"] = event_type
    return f"event: {event_type}\n" + f"data: {json.dumps(payload, ensure_ascii=True)}\n\n"


def _parse_openai_sse_line(line: str) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.lower() == "data: [done]":
        return {"_done": True}
    if not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:") :].strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except (TypeError, ValueError):
        raise ValueError("Malformed provider SSE data") from None
    if not isinstance(data, dict):
        raise ValueError("Malformed provider SSE data")
    return data


def _openai_error_to_anthropic_event(data: dict[str, Any]) -> str:
    """Translate an OpenAI-compatible error frame into Anthropic SSE."""
    del data
    return _sse_event(
        "error",
        {
            "error": {
                "type": "api_error",
                "message": "The upstream provider returned an error.",
            }
        },
    )


async def _aiter_lines(stream: Any) -> AsyncIterator[str]:
    if hasattr(stream, "__aiter__"):
        async for item in stream:
            if item is None:
                continue
            yield item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
    else:
        for item in stream:
            if item is None:
                continue
            yield item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)


async def _maybe_close_stream(stream: Any) -> None:
    if stream is None:
        return
    close_fn = getattr(stream, "aclose", None)
    close_kind = "aclose"
    if not callable(close_fn):
        close_fn = getattr(stream, "close", None)
        close_kind = "close"
    if not callable(close_fn):
        return
    try:
        await invoke_stream_close_bounded(close_fn)
    except Exception as stream_close_error:
        logger.debug(
            "Anthropic stream {} failed; error_type={}",
            close_kind,
            type(stream_close_error).__name__,
        )


def _extract_choice(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            return choice
    return {}


def _openai_stream_payload_has_structural_error(data: dict[str, Any]) -> bool:
    """Inspect response envelopes without treating assistant text as an error."""

    if normalize_provider_stream_error(data) is not None:
        return True
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if normalize_provider_stream_error(choice) is not None:
            return True
        for field in ("message", "delta"):
            container = choice.get(field)
            if not isinstance(container, dict):
                continue
            if normalize_provider_stream_error(container) is not None:
                return True
            content = container.get("content")
            if isinstance(content, list) and any(
                isinstance(block, dict)
                and normalize_provider_stream_error(block) is not None
                for block in content
            ):
                return True
    return False


async def openai_stream_to_anthropic(
    stream: Any,
    *,
    model: str | None,
) -> AsyncIterator[str]:
    message_id = f"msg_{uuid.uuid4().hex}"
    message_started = False
    text_block_index: int | None = None
    next_block_index = 0
    open_blocks: list[int] = []
    tool_blocks_by_id: dict[str, dict[str, Any]] = {}
    tool_blocks_by_index: dict[int, dict[str, Any]] = {}
    tool_states: list[dict[str, Any]] = []
    tool_stream_invalid = False
    tool_retained_chars = 0
    final_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def retain_tool_chars(length: int) -> bool:
        """Account for request-local buffered tool state within the repo cap."""
        nonlocal tool_retained_chars
        if length < 0 or tool_retained_chars + length > MAX_TOOL_ARGUMENT_LENGTH:
            return False
        tool_retained_chars += length
        return True

    def finalized_tool_events(
        *,
        start_index: int,
        allow_partial: bool = False,
    ) -> list[str] | None:
        """Return valid buffered tool events, or ``None`` for malformed state."""
        if tool_stream_invalid:
            return None
        events: list[str] = []
        for offset, state in enumerate(tool_states):
            tool_id = state.get("id")
            name = state.get("name")
            if state.get("invalid") or state.get("arguments_before_identity"):
                return None
            if not isinstance(tool_id, str) or not tool_id.strip():
                return None
            if not isinstance(name, str) or not name.strip():
                return None
            buffered_input = state.get("buffer")
            if not isinstance(buffered_input, str):
                return None
            if allow_partial:
                partial_json = buffered_input
            else:
                try:
                    tool_input = json.loads(buffered_input or "")
                except (TypeError, ValueError, json.JSONDecodeError):
                    return None
                if not isinstance(tool_input, dict):
                    return None
                partial_json = json.dumps(
                    tool_input,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            block_index = start_index + offset
            events.append(
                _sse_event(
                    "content_block_start",
                    {
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": name,
                            "input": {},
                        },
                    },
                )
            )
            events.append(
                _sse_event(
                    "content_block_delta",
                    {
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": partial_json,
                        },
                    },
                )
            )
            events.append(
                _sse_event(
                    "content_block_stop",
                    {"index": block_index},
                )
            )
        return events

    def valid_usage_count(value: Any) -> bool:
        return type(value) is int and value >= 0

    try:
        async for raw_line in _aiter_lines(stream):
            if provider_result_contains_error(raw_line, legacy_error_prefix=True):
                yield _openai_error_to_anthropic_event({})
                return
            data = _parse_openai_sse_line(raw_line)
            if not data:
                continue
            if data.get("_done"):
                break
            if _openai_stream_payload_has_structural_error(data):
                yield _openai_error_to_anthropic_event(data)
                return

            choices = data.get("choices")
            if (
                not isinstance(choices, list)
                or not choices
                or not isinstance(choices[0], dict)
            ):
                yield _openai_error_to_anthropic_event({})
                return
            choice = _extract_choice(data)
            raw_delta = choice.get("delta")
            if raw_delta is None:
                delta: dict[str, Any] = {}
            elif isinstance(raw_delta, dict):
                delta = raw_delta
            else:
                yield _openai_error_to_anthropic_event({})
                return
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and (
                not isinstance(finish_reason, str) or not finish_reason.strip()
            ):
                yield _openai_error_to_anthropic_event({})
                return

            if not message_started:
                message_started = True
                yield _sse_event(
                    "message_start",
                    {
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "model": model or data.get("model"),
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": dict(final_usage),
                        }
                    },
                )

            if isinstance(delta, dict):
                content = delta.get("content")
                if content is not None:
                    if not isinstance(content, str):
                        yield _openai_error_to_anthropic_event({})
                        return
                    if text_block_index is None:
                        text_block_index = next_block_index
                        next_block_index += 1
                        open_blocks.append(text_block_index)
                        yield _sse_event(
                            "content_block_start",
                            {
                                "index": text_block_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    yield _sse_event(
                        "content_block_delta",
                        {
                            "index": text_block_index,
                            "delta": {"type": "text_delta", "text": content},
                        },
                    )

            tool_calls = delta.get("tool_calls")
            legacy_function_call = delta.get("function_call")
            if not tool_calls and isinstance(legacy_function_call, dict):
                tool_calls = [
                    {
                        "index": 0,
                        "function": legacy_function_call,
                        "_legacy_function_call": True,
                    }
                ]
            elif tool_calls is not None and not isinstance(tool_calls, list):
                tool_stream_invalid = True
            if isinstance(tool_calls, list):
                for tool_delta in tool_calls:
                    if not isinstance(tool_delta, dict):
                        tool_stream_invalid = True
                        continue
                    func = tool_delta.get("function")
                    if not isinstance(func, dict):
                        tool_stream_invalid = True
                        continue
                    name = func.get("name")
                    args = func.get("arguments")
                    tool_id = tool_delta.get("id")
                    tool_index = tool_delta.get("index")
                    is_legacy = tool_delta.get("_legacy_function_call") is True

                    if tool_index is not None and (
                        type(tool_index) is not int
                        or tool_index < 0
                        or tool_index > MAX_TOOL_CALL_INDEX
                    ):
                        tool_stream_invalid = True
                        continue

                    state = None
                    if type(tool_index) is int:
                        state = tool_blocks_by_index.get(tool_index)
                    if state is None and isinstance(tool_id, str) and tool_id:
                        state = tool_blocks_by_id.get(tool_id)

                    if state is None:
                        output_index = next_block_index + len(tool_states)
                        retained_index = (
                            tool_index if type(tool_index) is int else output_index
                        )
                        legacy_id = f"tool_{output_index}" if is_legacy else None
                        if not retain_tool_chars(
                            1
                            + len(str(retained_index))
                            + (len(legacy_id) if legacy_id is not None else 0)
                        ):
                            yield _openai_error_to_anthropic_event({})
                            return
                        state = {
                            "provider_index": tool_index,
                            "name": None,
                            "buffer": "",
                            "id": legacy_id,
                            "legacy": is_legacy,
                            "invalid": False,
                            "arguments_before_identity": False,
                        }
                        tool_states.append(state)
                        if type(tool_index) is int:
                            tool_blocks_by_index[tool_index] = state
                    elif is_legacy != bool(state.get("legacy")):
                        state["invalid"] = True

                    if type(tool_index) is int:
                        provider_index = state.get("provider_index")
                        indexed_state = tool_blocks_by_index.get(tool_index)
                        if provider_index is None and indexed_state in (None, state):
                            state["provider_index"] = tool_index
                            tool_blocks_by_index[tool_index] = state
                        elif provider_index != tool_index or indexed_state not in (
                            None,
                            state,
                        ):
                            state["invalid"] = True

                    if tool_id is not None:
                        existing_state = (
                            tool_blocks_by_id.get(tool_id)
                            if isinstance(tool_id, str)
                            else None
                        )
                        if (
                            not isinstance(tool_id, str)
                            or not tool_id.strip()
                            or state.get("id") not in (None, tool_id)
                            or (
                                existing_state is not None
                                and existing_state is not state
                            )
                        ):
                            state["invalid"] = True
                        else:
                            if state.get("id") is None and not retain_tool_chars(
                                len(tool_id)
                            ):
                                yield _openai_error_to_anthropic_event({})
                                return
                            state["id"] = tool_id
                            tool_blocks_by_id.setdefault(tool_id, state)
                    if name is not None:
                        if (
                            not isinstance(name, str)
                            or not name.strip()
                            or state.get("name") not in (None, name)
                        ):
                            state["invalid"] = True
                        else:
                            if state.get("name") is None and not retain_tool_chars(
                                len(name)
                            ):
                                yield _openai_error_to_anthropic_event({})
                                return
                            state["name"] = name
                    if args is not None:
                        if not isinstance(args, str):
                            state["invalid"] = True
                        elif args:
                            if not state.get("id") or not state.get("name"):
                                state["arguments_before_identity"] = True
                            if not retain_tool_chars(len(args)):
                                yield _openai_error_to_anthropic_event({})
                                return
                            state["buffer"] += args

            usage = data.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                if "prompt_tokens" in usage and not valid_usage_count(prompt_tokens):
                    yield _openai_error_to_anthropic_event({})
                    return
                if "completion_tokens" in usage and not valid_usage_count(
                    completion_tokens
                ):
                    yield _openai_error_to_anthropic_event({})
                    return
                if valid_usage_count(prompt_tokens):
                    final_usage["input_tokens"] = prompt_tokens
                if valid_usage_count(completion_tokens):
                    final_usage["output_tokens"] = completion_tokens
            elif usage is not None:
                yield _openai_error_to_anthropic_event({})
                return

            if finish_reason:
                stop_reason = _finish_reason_to_stop_reason(finish_reason)
                has_tools = bool(tool_states)
                if stop_reason is None or (
                    has_tools
                    and stop_reason not in {"max_tokens", "tool_use"}
                ) or (not has_tools and stop_reason == "tool_use"):
                    yield _openai_error_to_anthropic_event({})
                    return
                tool_events = finalized_tool_events(
                    start_index=next_block_index,
                    allow_partial=stop_reason == "max_tokens",
                )
                if tool_events is None:
                    yield _openai_error_to_anthropic_event({})
                    return
                for idx in list(open_blocks):
                    yield _sse_event(
                        "content_block_stop",
                        {"index": idx},
                    )
                open_blocks.clear()
                for event in tool_events:
                    yield event
                yield _sse_event(
                    "message_delta",
                    {
                        "delta": {
                            "stop_reason": stop_reason,
                            "stop_sequence": None,
                        },
                        "usage": dict(final_usage),
                    },
                )
                yield _sse_event("message_stop", {})
                return

        yield _openai_error_to_anthropic_event({})
    finally:
        await _maybe_close_stream(stream)

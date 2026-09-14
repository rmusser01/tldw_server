from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Iterable
from typing import Any

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
)
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import _safe_cast
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    build_sanitized_chat_error,
    is_http_status_error,
    is_network_error,
    log_provider_failure,
    log_provider_failure_metadata,
    privacy_safe_chat_error,
    provider_errors_are_privacy_safe,
    raise_chat_error_from_http,
)
from tldw_Server_API.app.core.LLM_Calls.payload_utils import (
    _sanitize_payload_for_logging,
    merge_extra_body,
    merge_extra_headers,
)
from tldw_Server_API.app.core.LLM_Calls.sse import (
    finalize_stream,
    openai_delta_chunk,
    sse_data,
    sse_done,
)
from tldw_Server_API.app.core.LLM_Calls.streaming import (
    provider_stream_error_frame,
    wrap_sync_stream,
)
from tldw_Server_API.app.core.Utils.Utils import logging

from .base import ChatProvider, raise_if_in_band_provider_error


def _summarize_response_for_logging(response: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if not isinstance(response, dict):
        return {"type": type(response).__name__}
    summary["keys"] = list(response.keys())
    if "finish_reason" in response:
        summary["finish_reason"] = response.get("finish_reason")
    text = response.get("text")
    if isinstance(text, str):
        summary["text_chars"] = len(text)
    tool_calls = response.get("tool_calls")
    if isinstance(tool_calls, list):
        summary["tool_calls_count"] = len(tool_calls)
    elif tool_calls is not None:
        summary["tool_calls_present"] = True
    meta = response.get("meta")
    if isinstance(meta, dict):
        billed = meta.get("billed_units")
        if isinstance(billed, dict):
            billed_summary = {
                k: billed.get(k)
                for k in ("input_tokens", "output_tokens")
                if billed.get(k) is not None
            }
            if billed_summary:
                summary["billed_units"] = billed_summary
    return summary


def _cohere_request(
    input_data: list[dict[str, Any]],
    model: str | None = None,
    api_key: str | None = None,
    system_prompt: str | None = None,
    temp: float | None = None,
    streaming: bool | None = False,
    topp: float | None = None,
    topk: int | None = None,
    max_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
    seed: int | None = None,
    num_generations: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    custom_prompt_arg: str | None = None,
    app_config: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    credentials_resolved: bool = False,
):
    logging.debug(f"Cohere Chat: Request process starting for model '{model}' (Streaming: {streaming})")
    if credentials_resolved:
        loaded_config_data = app_config if isinstance(app_config, dict) else {}
    else:
        loaded_config_data = app_config or load_and_log_configs() or {}
    if not isinstance(loaded_config_data, dict):
        loaded_config_data = {}
    cohere_config = loaded_config_data.get("cohere_api", loaded_config_data.get("API", {}).get("cohere", {}))

    final_api_key = api_key or cohere_config.get("api_key")
    if not final_api_key:
        raise ChatAuthenticationError(provider="cohere", message="Cohere API key is missing.")
    logging.debug("Cohere: Using configured API key")

    final_model = model or cohere_config.get("model", "command-r")
    resolved_temp_from_cfg = cohere_config.get("temperature")
    current_temp = temp if temp is not None else _safe_cast(resolved_temp_from_cfg, float, None)
    resolved_p_cfg = cohere_config.get("top_p")
    if resolved_p_cfg is None:
        resolved_p_cfg = cohere_config.get("p")
    current_p = topp if topp is not None else _safe_cast(resolved_p_cfg, float, None)
    resolved_k_cfg = cohere_config.get("top_k")
    if resolved_k_cfg is None:
        resolved_k_cfg = cohere_config.get("k")
    current_k = topk if topk is not None else _safe_cast(resolved_k_cfg, int, None)
    current_max_tokens = max_tokens if max_tokens is not None else _safe_cast(cohere_config.get("max_tokens"), int, None)
    current_stop_sequences = stop_sequences if stop_sequences is not None else cohere_config.get("stop_sequences")
    current_seed = seed if seed is not None else _safe_cast(cohere_config.get("seed"), int, None)
    current_frequency_penalty = frequency_penalty if frequency_penalty is not None else _safe_cast(
        cohere_config.get("frequency_penalty"), float, None
    )
    current_presence_penalty = presence_penalty if presence_penalty is not None else _safe_cast(
        cohere_config.get("presence_penalty"), float, None
    )
    current_tools = tools if tools is not None else cohere_config.get("tools")
    current_num_generations = num_generations if num_generations is not None else _safe_cast(
        cohere_config.get("num_generations"), int, None
    )

    api_base_url = (base_url or cohere_config.get("api_base_url", "https://api.cohere.ai")).rstrip("/")
    COHERE_CHAT_URL = f"{api_base_url}/v1/chat"

    configured_timeout = _safe_cast(cohere_config.get("api_timeout"), float, 180.0)
    request_timeout = _safe_cast(timeout, float) if timeout is not None else None
    timeout_seconds = request_timeout if request_timeout is not None else configured_timeout

    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if streaming else "application/json",
    }
    headers = merge_extra_headers(headers, {"extra_headers": extra_headers})

    chat_history_for_cohere: list[dict[str, str]] = []
    current_user_message_str = ""
    preamble_str = system_prompt or ""

    temp_messages = list(input_data)

    if not preamble_str and temp_messages and temp_messages[0]["role"] == "system":
        preamble_str = temp_messages.pop(0)["content"]
        logging.debug(
            f"Cohere: Using system message from input_data as preamble (chars={len(str(preamble_str))})."
        )

    if not temp_messages:
        if custom_prompt_arg:
            current_user_message_str = custom_prompt_arg
            logging.warning("Cohere: No user/assistant messages in input_data, using custom_prompt_arg as user message.")
        else:
            raise ChatBadRequestError(
                provider="cohere",
                message="No user/assistant messages found for Cohere chat after processing system message.",
            )
    elif temp_messages[-1]["role"] == "user":
        last_msg_content = temp_messages[-1]["content"]
        if isinstance(last_msg_content, list):
            current_user_message_str = next(
                (part["text"] for part in last_msg_content if part.get("type") == "text"), ""
            )
        else:
            current_user_message_str = str(last_msg_content)
        chat_history_for_cohere = temp_messages[:-1]
    else:
        current_user_message_str = custom_prompt_arg or "Please respond."
        chat_history_for_cohere = temp_messages
        logging.warning(
            "Cohere: Last message in payload was not 'user'. Using fallback user message (chars=%d).",
            len(current_user_message_str),
        )

    if custom_prompt_arg and current_user_message_str != custom_prompt_arg:
        current_user_message_str += f"\n{custom_prompt_arg}"
        logging.debug("Cohere: Appended custom_prompt_arg to current user message.")

    if not current_user_message_str.strip():
        raise ChatBadRequestError(provider="cohere", message="Current user message for Cohere is empty after processing.")

    transformed_history = []
    for msg in chat_history_for_cohere:
        role = msg.get("role", "").lower()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = next((part["text"] for part in content if part.get("type") == "text"), "")

        if role == "user":
            transformed_history.append({"role": "USER", "message": str(content)})
        elif role == "assistant":
            transformed_history.append({"role": "CHATBOT", "message": str(content)})

    payload: dict[str, Any] = {
        "model": final_model,
        "message": current_user_message_str,
    }
    if transformed_history:
        payload["chat_history"] = transformed_history
    if preamble_str:
        payload["preamble"] = preamble_str
    if current_temp is not None:
        payload["temperature"] = current_temp
    if current_p is not None:
        payload["p"] = current_p
    if current_k is not None:
        payload["k"] = current_k
    if current_max_tokens is not None:
        payload["max_tokens"] = current_max_tokens
    if current_stop_sequences:
        payload["stop_sequences"] = current_stop_sequences
    if current_seed is not None:
        payload["seed"] = current_seed
    if current_frequency_penalty is not None:
        payload["frequency_penalty"] = current_frequency_penalty
    if current_presence_penalty is not None:
        payload["presence_penalty"] = current_presence_penalty
    if current_tools:
        payload["tools"] = current_tools

    if streaming:
        payload["stream"] = True
    else:
        payload["stream"] = False
        if current_num_generations is not None:
            if current_num_generations > 0:
                payload["num_generations"] = current_num_generations
            else:
                logging.warning("Cohere: 'num_generations' must be > 0. Ignoring.")
    payload = merge_extra_body(payload, {"extra_body": extra_body})

    cohere_payload_metadata = _sanitize_payload_for_logging(
        payload,
        message_keys=("chat_history",),
        text_keys=("message", "preamble"),
    )
    logging.debug(f"Cohere request metadata: {cohere_payload_metadata}")
    logging.debug("Cohere request endpoint resolved")

    from tldw_Server_API.app.core.LLM_Calls import chat_calls as _chat_calls
    session = _chat_calls.create_session_with_retries(
        total=_safe_cast(cohere_config.get("api_retries"), int, 1)
    )

    try:
        if streaming:
            response = session.post(COHERE_CHAT_URL, headers=headers, json=payload, stream=True, timeout=timeout_seconds)
            try:
                response.raise_for_status()
            except Exception:
                with contextlib.suppress(Exception):
                    response.close()
                raise
            logging.debug("Cohere: Streaming response connection established.")
            session_handle = session
            response_handle = response

            def stream_generator_cohere_text_chunks(response_iterator):
                stream_properly_closed = False
                try:
                    for line_bytes in response_iterator:
                        if not line_bytes:
                            continue
                        decoded_line = (
                            line_bytes.decode("utf-8", errors="replace")
                            if isinstance(line_bytes, (bytes, bytearray))
                            else str(line_bytes)
                        )
                        decoded_line = decoded_line.strip()
                        if not decoded_line:
                            continue
                        raise_if_in_band_provider_error(
                            "cohere",
                            decoded_line,
                            phase="stream_response",
                        )

                        if decoded_line.startswith("data:"):
                            json_data_str = decoded_line[len("data:") :].strip()
                            if not json_data_str:
                                continue
                            try:
                                cohere_event = json.loads(json_data_str)
                            except json.JSONDecodeError:
                                logging.warning(
                                    "Cohere stream returned malformed JSON chars={}",
                                    len(json_data_str),
                                )
                                continue

                            event_type = cohere_event.get("event_type")
                            if event_type == "text-generation":
                                text_chunk = cohere_event.get("text")
                                if text_chunk:
                                    yield sse_data({
                                        "choices": [{"delta": {"content": str(text_chunk)}}],
                                        "provider_response": cohere_event,
                                    })
                            elif event_type == "stream-end":
                                stream_properly_closed = True
                                yield sse_done()
                                return
                            else:
                                continue
                        else:
                            yield openai_delta_chunk(decoded_line)

                except Exception as e_stream:
                    log_provider_failure(
                        "cohere",
                        e_stream,
                        phase="stream_iteration",
                    )
                    yield provider_stream_error_frame("cohere")
                finally:
                    yield from finalize_stream(response_handle, done_already=stream_properly_closed)
                    with contextlib.suppress(Exception):
                        session_handle.close()

            session = None
            return stream_generator_cohere_text_chunks(response_handle.iter_lines())

        response = session.post(COHERE_CHAT_URL, headers=headers, json=payload, stream=False, timeout=timeout_seconds)
        response.raise_for_status()
        response_data = response.json()
        raise_if_in_band_provider_error(
            "cohere",
            response_data,
            phase="chat_response",
        )
        logging.debug(
            "Cohere non-streaming response metadata: %s",
            _summarize_response_for_logging(response_data),
        )

        chat_id = response_data.get("generation_id", f"chatcmpl-cohere-{time.time_ns()}")
        created_timestamp = int(time.time())
        choices_payload = []
        finish_reason = response_data.get("finish_reason", "stop")

        if response_data.get("text"):
            choices_payload.append(
                {"message": {"role": "assistant", "content": response_data["text"]}, "finish_reason": finish_reason, "index": 0}
            )
        elif response_data.get("tool_calls"):
            openai_like_tool_calls = []
            for tc in response_data.get("tool_calls", []):
                openai_like_tool_calls.append(
                    {
                        "id": f"call_{tc.get('name', 'tool')}_{time.time_ns()}",
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("parameters", {})),
                        },
                    }
                )
            choices_payload.append(
                {
                    "message": {"role": "assistant", "content": None, "tool_calls": openai_like_tool_calls},
                    "finish_reason": "tool_calls",
                    "index": 0,
                }
            )
        else:
            logging.warning(
                "Cohere non-streaming response missing text/tool calls metadata={}",
                _summarize_response_for_logging(response_data),
            )
            choices_payload.append(
                {"message": {"role": "assistant", "content": ""}, "finish_reason": finish_reason, "index": 0}
            )

        usage_data = None
        meta = response_data.get("meta")
        if meta and meta.get("billed_units"):
            billed_units = meta["billed_units"]
            prompt_tokens = billed_units.get("input_tokens")
            completion_tokens = billed_units.get("output_tokens")
            if prompt_tokens is not None and completion_tokens is not None:
                usage_data = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

        openai_compatible_response = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_timestamp,
            "model": final_model,
            "choices": choices_payload,
            "provider_response": response_data,
        }
        if usage_data:
            openai_compatible_response["usage"] = usage_data
        return openai_compatible_response

    except KeyError as exc:
        log_provider_failure(
            "cohere",
            exc,
            phase="payload_or_response_shape",
        )
        raise_detached_error(ChatBadRequestError(provider="cohere"))
    except Exception as e:
        if provider_errors_are_privacy_safe():
            log_provider_failure_metadata("cohere", e)
            raise privacy_safe_chat_error("cohere", e) from None
        if is_http_status_error(e):
            raise_chat_error_from_http("cohere", e)
        if is_network_error(e):
            log_provider_failure("cohere", e, phase="network_request")
            raise_detached_error(
                build_sanitized_chat_error("cohere", status_code=504)
            )
        log_provider_failure("cohere", e, phase="request")
        status_code = (
            getattr(e, "status_code", None) if isinstance(e, ChatAPIError) else None
        )
        raise_detached_error(
            build_sanitized_chat_error("cohere", status_code=status_code)
        )
    finally:
        if session:
            try:
                session.close()
            except Exception as close_error:
                log_provider_failure(
                    "cohere",
                    close_error,
                    phase="client_close",
                )


class CohereAdapter(ChatProvider):
    name = "cohere"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 180,
        }

    def _to_handler_args(
        self,
        request: dict[str, Any],
        *,
        streaming: bool | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if stream_flag is None:
            stream_flag = request.get("streaming")
        if streaming is not None:
            stream_flag = streaming

        return {
            "input_data": request.get("messages") or [],
            "model": request.get("model"),
            "api_key": request.get("api_key"),
            "system_prompt": request.get("system_message"),
            "temp": request.get("temperature"),
            "streaming": bool(stream_flag) if stream_flag is not None else False,
            "topp": request.get("top_p"),
            "topk": request.get("top_k"),
            "max_tokens": request.get("max_tokens"),
            "stop_sequences": request.get("stop"),
            "seed": request.get("seed"),
            "num_generations": request.get("num_generations"),
            "frequency_penalty": request.get("frequency_penalty"),
            "presence_penalty": request.get("presence_penalty"),
            "tools": request.get("tools"),
            "custom_prompt_arg": request.get("custom_prompt_arg"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
            "base_url": request.get("base_url"),
            "timeout": timeout,
        }

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = self._bind_request_credentials(request)
        request = validate_payload(self.name, request or {})
        return _cohere_request(**self._to_handler_args(request, streaming=False, timeout=timeout))

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = self._bind_request_credentials(request)
        request = validate_payload(self.name, request or {})
        return _cohere_request(**self._to_handler_args(request, streaming=True, timeout=timeout))

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

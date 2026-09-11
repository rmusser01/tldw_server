from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import _safe_cast
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.error_utils import (
    build_sanitized_chat_error,
    is_http_status_error,
    is_network_error,
    log_provider_failure,
    raise_chat_error_from_http,
)
from tldw_Server_API.app.core.LLM_Calls.payload_utils import (
    _sanitize_payload_for_logging,
    merge_extra_body,
    merge_extra_headers,
)
from tldw_Server_API.app.core.LLM_Calls.sse import (
    finalize_stream,
    is_done_line,
    normalize_provider_line,
    sse_done,
)
from tldw_Server_API.app.core.LLM_Calls.streaming import provider_stream_error_frame
from tldw_Server_API.app.core.Utils.Utils import logging

from .base import ChatProvider, raise_if_in_band_provider_error


def _zai_request(
    input_data: list[dict[str, Any]],
    model: str | None = None,
    api_key: str | None = None,
    system_message: str | None = None,
    temp: float | None = None,
    maxp: float | None = None,
    streaming: bool | None = False,
    max_tokens: int | None = None,
    n: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    do_sample: bool | None = None,
    request_id: str | None = None,
    custom_prompt_arg: str | None = None,
    app_config: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    base_url: str | None = None,
    timeout: float | None = None,
    credentials_resolved: bool = False,
):
    if credentials_resolved:
        loaded_config_data = app_config if isinstance(app_config, dict) else {}
    else:
        loaded_config_data = app_config or load_and_log_configs() or {}
    if not isinstance(loaded_config_data, dict):
        loaded_config_data = {}
    zai_config = loaded_config_data.get("zai_api", {})

    final_api_key = api_key or zai_config.get("api_key")
    if not final_api_key:
        logging.error("Z.AI: API key is missing.")
        raise ChatConfigurationError(provider="zai", message="Z.AI API Key is required but not found.")

    logging.debug("Z.AI: Using configured API key")

    current_model = model or zai_config.get("model", "glm-4.5-flash")
    current_temp = temp if temp is not None else _safe_cast(zai_config.get("temperature"), float, 0.7)
    current_top_p = maxp if maxp is not None else _safe_cast(zai_config.get("top_p"), float, 0.95)
    current_streaming_cfg = zai_config.get("streaming", False)
    current_streaming = (
        streaming
        if streaming is not None
        else (str(current_streaming_cfg).lower() == "true" if isinstance(current_streaming_cfg, str) else bool(current_streaming_cfg))
    )
    current_max_tokens = max_tokens if max_tokens is not None else _safe_cast(zai_config.get("max_tokens"), int, 4096)

    api_messages = []
    if system_message:
        api_messages.append({"role": "system", "content": system_message})
    api_messages.extend(input_data)

    payload: dict[str, Any] = {
        "model": current_model,
        "messages": api_messages,
        "stream": current_streaming,
    }

    if current_temp is not None:
        payload["temperature"] = current_temp
    if current_top_p is not None:
        payload["top_p"] = current_top_p
    if current_max_tokens is not None:
        payload["max_tokens"] = current_max_tokens
    if n is not None:
        payload["n"] = n
    if do_sample is not None:
        payload["do_sample"] = do_sample
    if tools is not None:
        payload["tools"] = tools
    if request_id is not None:
        payload["request_id"] = request_id

    headers = {
        "Authorization": f"Bearer {final_api_key}",
        "Content-Type": "application/json",
    }
    headers = merge_extra_headers(headers, {"extra_headers": extra_headers})

    api_base_url = base_url or zai_config.get("api_base_url", "https://api.z.ai/api/paas/v4")
    api_url = api_base_url.rstrip("/") + "/chat/completions"

    payload = merge_extra_body(payload, {"extra_body": extra_body})
    payload_metadata = _sanitize_payload_for_logging(payload)
    logging.debug(f"Z.AI request metadata: {payload_metadata}")
    configured_timeout = _safe_cast(zai_config.get("api_timeout"), float, 90.0)
    request_timeout = _safe_cast(timeout, float) if timeout is not None else None
    effective_timeout = request_timeout if request_timeout is not None else configured_timeout

    try:
        if current_streaming:
            logging.debug("Z.AI: Posting request (streaming)")
            from tldw_Server_API.app.core.LLM_Calls import chat_calls as _chat_calls
            session = _chat_calls.create_session_with_retries(
                total=_safe_cast(zai_config.get("api_retries"), int, 1)
            )
            response = None
            try:
                response = session.post(api_url, headers=headers, json=payload, stream=True, timeout=effective_timeout)
                response.raise_for_status()

                def stream_generator():
                    done_sent = False
                    skip_finalize = False
                    try:
                        for raw_line in response.iter_lines(decode_unicode=True):
                            if not raw_line:
                                continue
                            raise_if_in_band_provider_error(
                                "zai",
                                raw_line,
                                phase="stream_response",
                            )
                            if is_done_line(raw_line):
                                done_sent = True
                            normalized = normalize_provider_line(raw_line)
                            if normalized is None:
                                continue
                            yield normalized
                        if not done_sent:
                            done_sent = True
                            yield sse_done()
                    except GeneratorExit:
                        skip_finalize = True
                        try:
                            response.close()
                        finally:
                            with contextlib.suppress(Exception):
                                session.close()
                        raise
                    except Exception as e_stream:
                        log_provider_failure(
                            "zai",
                            e_stream,
                            phase="stream_iteration",
                        )
                        yield provider_stream_error_frame("zai")
                    finally:
                        try:
                            if not skip_finalize:
                                yield from finalize_stream(response, done_already=done_sent)
                        finally:
                            with contextlib.suppress(Exception):
                                session.close()

                return stream_generator()
            except Exception:
                if response is not None:
                    with contextlib.suppress(Exception):
                        response.close()
                with contextlib.suppress(Exception):
                    session.close()
                raise

        logging.debug("Z.AI: Posting request (non-streaming)")
        from tldw_Server_API.app.core.LLM_Calls import chat_calls as _chat_calls
        session = _chat_calls.create_session_with_retries(
            total=_safe_cast(zai_config.get("api_retries"), int, 1)
        )
        try:
            response = session.post(api_url, headers=headers, json=payload, timeout=effective_timeout)
            logging.debug(f"Z.AI: Full API response status: {response.status_code}")
            response.raise_for_status()
            try:
                response_data = response.json()
                raise_if_in_band_provider_error(
                    "zai",
                    response_data,
                    phase="chat_response",
                )
            finally:
                with contextlib.suppress(Exception):
                    response.close()
            logging.debug("Z.AI: Non-streaming request successful.")
            return response_data
        finally:
            with contextlib.suppress(Exception):
                session.close()

    except Exception as e:
        if is_http_status_error(e):
            raise_chat_error_from_http("zai", e)
        if is_network_error(e):
            log_provider_failure("zai", e, phase="network_request")
            raise_detached_error(
                build_sanitized_chat_error("zai", status_code=504)
            )
        log_provider_failure("zai", e, phase="request")
        raise_detached_error(build_sanitized_chat_error("zai"))


class ZaiAdapter(ChatProvider):
    name = "zai"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 90,
            "max_output_tokens_default": 4096,
        }

    def _to_handler_args(self, request: dict[str, Any], *, streaming: bool | None) -> dict[str, Any]:
        stream_flag = request.get("stream")
        if streaming is not None:
            stream_flag = streaming
        return {
            "input_data": request.get("messages") or [],
            "model": request.get("model"),
            "api_key": request.get("api_key"),
            "system_message": request.get("system_message"),
            "temp": request.get("temperature"),
            "maxp": request.get("top_p"),
            "streaming": stream_flag,
            "max_tokens": request.get("max_tokens"),
            "n": request.get("n"),
            "tools": request.get("tools"),
            "do_sample": request.get("do_sample"),
            "request_id": request.get("request_id"),
            "custom_prompt_arg": request.get("custom_prompt_arg"),
            "app_config": request.get("app_config"),
            "credentials_resolved": request.get("credentials_resolved") is True,
            "extra_headers": request.get("extra_headers"),
            "extra_body": request.get("extra_body"),
            "base_url": request.get("base_url"),
        }

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = self._bind_request_credentials(request)
        sanitized = validate_payload(self.name, request or {})
        handler_args = self._to_handler_args(sanitized, streaming=False)
        handler_args["timeout"] = timeout
        return _zai_request(**handler_args)

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = self._bind_request_credentials(request)
        sanitized = validate_payload(self.name, request or {})
        handler_args = self._to_handler_args(sanitized, streaming=True)
        handler_args["timeout"] = timeout
        return _zai_request(**handler_args)

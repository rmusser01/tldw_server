from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.http_client import (
    create_client as _hc_create_client,
)
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.payload_utils import merge_extra_body, merge_extra_headers
from tldw_Server_API.app.core.LLM_Calls.sse import (
    finalize_stream,
    is_done_line,
    normalize_provider_line,
    sse_done,
)
from tldw_Server_API.app.core.LLM_Calls.streaming import wrap_sync_stream

from .base import ChatProvider

# Expose a patchable factory for tests; production uses the centralized client
http_client_factory = _hc_create_client


class HuggingFaceAdapter(ChatProvider):
    name = "huggingface"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": False,
            "default_timeout_seconds": 120,
            "max_output_tokens_default": 2048,
        }

    def _to_handler_args(self, request: dict[str, Any]) -> dict[str, Any]:
        streaming_raw = request.get("stream")
        if streaming_raw is None:
            streaming_raw = request.get("streaming")
        return {
            "input_data": request.get("messages") or [],
            "model": request.get("model"),
            "api_key": request.get("api_key"),
            "system_message": request.get("system_message"),
            "temp": request.get("temperature"),
            "streaming": streaming_raw,
            "top_p": request.get("top_p"),
            "top_k": request.get("top_k"),
            "max_tokens": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "response_format": request.get("response_format"),
            "num_return_sequences": request.get("n"),
            "user": request.get("user"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "logit_bias": request.get("logit_bias"),
            "presence_penalty": request.get("presence_penalty"),
            "frequency_penalty": request.get("frequency_penalty"),
            "logprobs": request.get("logprobs"),
            "top_logprobs": request.get("top_logprobs"),
            "custom_prompt_arg": request.get("custom_prompt_arg"),
            "app_config": request.get("app_config"),
        }

    @staticmethod
    def _mask_headers(headers: dict[str, str]) -> dict[str, str]:
        masked = {}
        for k, v in headers.items():
            masked[k] = "***" if k.lower() == "authorization" else v
        return masked

    def _resolve_url_and_headers(self, request: dict[str, Any]) -> dict[str, Any]:
        cfg = (request.get("app_config") or {}).get("huggingface_api", {})
        override_base = request.get("base_url")
        api_base = cfg.get("api_base_url")  # may be None
        runtime_base_override = cfg.get("_runtime_base_url_override") is True
        use_router = str(
            cfg.get("use_router_url_format", cfg.get("huggingface_use_router_url_format", "false"))
        ).lower() == "true"
        chat_path = cfg.get("api_chat_path") or cfg.get("huggingface_api_chat_path")
        if not chat_path:
            base = (api_base or "").rstrip("/")
            if base.endswith("/v1") or "api-inference.huggingface.co/v1" in base:
                chat_path = "chat/completions"
            else:
                chat_path = "v1/chat/completions"
        model = request.get("model") or cfg.get("model_id") or cfg.get("model")
        if not model:
            model = "unspecified"
        if use_router:
            base = override_base or (
                (api_base if runtime_base_override else None)
                or cfg.get("router_base_url")
                or cfg.get("huggingface_router_base_url")
                or "https://router.huggingface.co/hf-inference"
            )
            base = str(base).rstrip("/")
            url = f"{base}/models/{model.strip('/')}/{chat_path.lstrip('/')}"
        else:
            base = str(override_base or api_base or "https://api-inference.huggingface.co/v1").rstrip("/")
            url = f"{base}/{chat_path.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        api_key = request.get("api_key") or cfg.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return {"url": url, "headers": headers}

    def _resolve_timeout(self, request: dict[str, Any], fallback: float | None) -> float:
        try:
            cfg = (request.get("app_config") or {}).get("huggingface_api", {})
            t = cfg.get("api_timeout")
            if t is not None:
                try:
                    return float(t)
                except (TypeError, ValueError) as timeout_parse_error:
                    logger.debug("HuggingFace adapter timeout value is not numeric", exc_info=timeout_parse_error)
        except Exception as config_error:
            logger.debug("HuggingFace adapter failed to read timeout config", exc_info=config_error)
        if fallback is not None:
            return float(fallback)
        return float(self.capabilities().get("default_timeout_seconds", 120))

    def _build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = request.get("messages") or []
        system_message = request.get("system_message")
        payload_messages: list[dict[str, Any]] = []
        if system_message:
            payload_messages.append({"role": "system", "content": system_message})
        payload_messages.extend(messages)
        payload: dict[str, Any] = {"messages": payload_messages}
        if request.get("model") is not None:
            payload["model"] = request.get("model")
        # Common OpenAI-like knobs (HF may ignore unsupported ones)
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "seed",
            "stop",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "response_format",
        ):
            if request.get(k) is not None:
                payload[k] = request.get(k)
        if request.get("tools") is not None:
            payload["tools"] = request.get("tools")
        if request.get("tool_choice") is not None:
            payload["tool_choice"] = request.get("tool_choice")
        if request.get("logprobs") is not None:
            payload["logprobs"] = request.get("logprobs")
        if request.get("top_logprobs") is not None and request.get("logprobs"):
            payload["top_logprobs"] = request.get("top_logprobs")
        if request.get("user") is not None:
            payload["user"] = request.get("user")
        return payload

    def normalize_error(self, exc: Exception):  # type: ignore[override]
        from tldw_Server_API.app.core.LLM_Calls.error_utils import (
            get_http_error_text,
            get_http_status_from_exception,
            is_http_status_error,
            log_http_400_body,
        )
        if is_http_status_error(exc):
            from tldw_Server_API.app.core.Chat.Chat_Deps import (
                ChatAPIError,
                ChatAuthenticationError,
                ChatBadRequestError,
                ChatProviderError,
                ChatRateLimitError,
            )
            resp = getattr(exc, "response", None)
            status = get_http_status_from_exception(exc)
            # Try parsing HF error wrapper {"error": {"message": "...", "type": "..."}}
            detail = None
            try:
                body = resp.json() if resp is not None else None
            except Exception:
                body = None
            log_http_400_body(self.name, exc, body)
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    msg = (err.get("message") or "").strip()
                    typ = (err.get("type") or "").strip()
                    detail = (f"{typ} {msg}" if typ else msg) or None
            if not detail:
                detail = get_http_error_text(exc)
            if status in (400, 404, 422):
                return ChatBadRequestError(provider=self.name, message=str(detail))
            if status in (401, 403):
                return ChatAuthenticationError(provider=self.name, message=str(detail))
            if status == 429:
                return ChatRateLimitError(provider=self.name, message=str(detail))
            if status and 500 <= status < 600:
                return ChatProviderError(provider=self.name, message=str(detail), status_code=status)
            return ChatAPIError(provider=self.name, message=str(detail), status_code=status or 500)
        return super().normalize_error(exc)

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = validate_payload(self.name, request or {})
        info = self._resolve_url_and_headers(request)
        url = info["url"]
        headers = info["headers"]
        payload = self._build_payload(request)
        payload["stream"] = True
        payload = merge_extra_body(payload, request)
        headers = merge_extra_headers(headers, request)
        try:
            resolved_timeout = self._resolve_timeout(request, timeout)
            logger.debug("HuggingFace headers: {}", self._mask_headers(headers))
            with http_client_factory(timeout=resolved_timeout) as client:
                with client.stream("POST", url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    seen_done = False
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        try:
                            line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                        except Exception:
                            line = str(raw)
                        if is_done_line(line):
                            if not seen_done:
                                seen_done = True
                                yield sse_done()
                            continue
                        normalized = normalize_provider_line(line)
                        if normalized is not None:
                            yield normalized
                    yield from finalize_stream(response=resp, done_already=seen_done)
            return
        except Exception as e:
            raise self.normalize_error(e) from e

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = validate_payload(self.name, request or {})
        info = self._resolve_url_and_headers(request)
        url = info["url"]
        headers = info["headers"]
        payload = self._build_payload(request)
        payload["stream"] = False
        payload = merge_extra_body(payload, request)
        headers = merge_extra_headers(headers, request)
        try:
            resolved_timeout = self._resolve_timeout(request, timeout)
            logger.debug("HuggingFace headers: {}", self._mask_headers(headers))
            with http_client_factory(timeout=resolved_timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            raise self.normalize_error(e) from e

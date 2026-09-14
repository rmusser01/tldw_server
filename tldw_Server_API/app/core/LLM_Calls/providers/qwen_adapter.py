from __future__ import annotations

import asyncio
import os
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

_QWEN_REGION_BASE_URLS: dict[str, str] = {
    "sg": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}
_QWEN_DEFAULT_REGION = "sg"
_QWEN_DEFAULT_BASE_URL = _QWEN_REGION_BASE_URLS[_QWEN_DEFAULT_REGION]


class QwenAdapter(ChatProvider):
    name = "qwen"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 90,
            "max_output_tokens_default": 8192,
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
            # Qwen uses 'maxp' in legacy; map from top_p
            "maxp": request.get("top_p"),
            "streaming": streaming_raw,
            "max_tokens": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "response_format": request.get("response_format"),
            "n": request.get("n"),
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

    def _apply_config_defaults(self, request: dict[str, Any]) -> dict[str, Any]:
        merged = dict(request)
        cfg = (merged.get("app_config") or {}).get("qwen_api", {})
        if merged.get("api_key") is None and cfg.get("api_key") is not None:
            merged["api_key"] = cfg.get("api_key")
        if merged.get("model") is None:
            merged["model"] = cfg.get("model") or "qwen-plus"
        if merged.get("temperature") is None and cfg.get("temperature") is not None:
            merged["temperature"] = cfg.get("temperature")
        if merged.get("top_p") is None and cfg.get("top_p") is not None:
            merged["top_p"] = cfg.get("top_p")
        if merged.get("max_tokens") is None and cfg.get("max_tokens") is not None:
            merged["max_tokens"] = cfg.get("max_tokens")
        if merged.get("seed") is None and cfg.get("seed") is not None:
            merged["seed"] = cfg.get("seed")
        if merged.get("stop") is None and cfg.get("stop") is not None:
            merged["stop"] = cfg.get("stop")
        if merged.get("response_format") is None and cfg.get("response_format") is not None:
            merged["response_format"] = cfg.get("response_format")
        if merged.get("n") is None and cfg.get("n") is not None:
            merged["n"] = cfg.get("n")
        if merged.get("user") is None and cfg.get("user") is not None:
            merged["user"] = cfg.get("user")
        if merged.get("tools") is None and cfg.get("tools") is not None:
            merged["tools"] = cfg.get("tools")
        if merged.get("tool_choice") is None and cfg.get("tool_choice") is not None:
            merged["tool_choice"] = cfg.get("tool_choice")
        if merged.get("logit_bias") is None and cfg.get("logit_bias") is not None:
            merged["logit_bias"] = cfg.get("logit_bias")
        if merged.get("presence_penalty") is None and cfg.get("presence_penalty") is not None:
            merged["presence_penalty"] = cfg.get("presence_penalty")
        if merged.get("frequency_penalty") is None and cfg.get("frequency_penalty") is not None:
            merged["frequency_penalty"] = cfg.get("frequency_penalty")
        if merged.get("logprobs") is None and cfg.get("logprobs") is not None:
            merged["logprobs"] = cfg.get("logprobs")
        if merged.get("top_logprobs") is None and cfg.get("top_logprobs") is not None:
            merged["top_logprobs"] = cfg.get("top_logprobs")
        return merged

    def _base_url(self, cfg: dict[str, Any] | None, request: dict[str, Any] | None = None) -> str:
        # DashScope OpenAI-compatible endpoint
        override = (request or {}).get("base_url")
        if isinstance(override, str) and override.strip():
            return override.strip().rstrip("/")
        credentials_resolved = (request or {}).get("credentials_resolved") is True
        if not credentials_resolved:
            env_base = os.getenv("QWEN_BASE_URL")
            if isinstance(env_base, str) and env_base.strip():
                return env_base.strip().rstrip("/")

        qwen_cfg: dict[str, Any] = {}
        if cfg:
            qwen_cfg = cfg.get("qwen_api") or {}

        api_base = qwen_cfg.get("api_base_url")
        if isinstance(api_base, str) and api_base.strip():
            return api_base.strip().rstrip("/")

        cfg_region = str(qwen_cfg.get("region") or "").strip().lower()
        if credentials_resolved:
            region = cfg_region or _QWEN_DEFAULT_REGION
        else:
            env_region = str(os.getenv("QWEN_REGION") or "").strip().lower()
            region = env_region or cfg_region or _QWEN_DEFAULT_REGION

        return _QWEN_REGION_BASE_URLS.get(region, _QWEN_DEFAULT_BASE_URL).rstrip("/")

    def _resolve_timeout(self, request: dict[str, Any], fallback: float | None) -> float:
        try:
            cfg = request.get("app_config") or {}
            qcfg = cfg.get("qwen_api") or {}
            t = qcfg.get("api_timeout")
            if t is not None:
                try:
                    return float(t)
                except (TypeError, ValueError) as timeout_parse_error:
                    logger.debug("Qwen adapter timeout value is not numeric", exc_info=timeout_parse_error)
        except Exception as config_error:
            logger.debug("Qwen adapter failed to read timeout config", exc_info=config_error)
        if fallback is not None:
            return float(fallback)
        return float(self.capabilities().get("default_timeout_seconds", 90))

    def _headers(self, api_key: str | None) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    def _build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = request.get("messages") or []
        system_message = request.get("system_message")
        payload_messages: list[dict[str, Any]] = []
        if system_message and not any((m.get("role") == "system") for m in messages):
            payload_messages.append({"role": "system", "content": system_message})
        payload_messages.extend(messages)
        payload: dict[str, Any] = {
            "model": request.get("model"),
            "messages": payload_messages,
        }
        # Common OpenAI-compatible fields
        for k in (
            "temperature",
            "top_p",
            "max_tokens",
            "seed",
            "stop",
            "n",
            "user",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
        ):
            if request.get(k) is not None:
                payload[k] = request.get(k)
        if request.get("tools") is not None:
            payload["tools"] = request.get("tools")
        if request.get("tool_choice") is not None:
            payload["tool_choice"] = request.get("tool_choice")
        if request.get("response_format") is not None:
            payload["response_format"] = request.get("response_format")
        return payload

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = self._bind_request_credentials(request)
        request = self._apply_config_defaults(request or {})
        request = validate_payload(self.name, request or {})
        api_key = request.get("api_key")
        if not api_key:
            from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
            raise ChatConfigurationError(provider=self.name, message="Qwen API Key required.")
        cfg = request.get("app_config") or {}
        url = f"{self._base_url(cfg, request)}/chat/completions"
        headers = self._headers(api_key)
        payload = self._build_payload(request)
        payload["stream"] = False
        payload = merge_extra_body(payload, request)
        headers = merge_extra_headers(headers, request)
        try:
            resolved_timeout = self._resolve_timeout(request, timeout)
            with http_client_factory(timeout=resolved_timeout) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                self._raise_if_in_band_provider_error(data, phase="chat_response")
                return data
        except Exception as e:
            self._raise_sanitized_provider_failure(e, phase="chat")

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = self._bind_request_credentials(request)
        request = self._apply_config_defaults(request or {})
        request = validate_payload(self.name, request or {})
        api_key = request.get("api_key")
        if not api_key:
            from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
            raise ChatConfigurationError(provider=self.name, message="Qwen API Key required.")
        cfg = request.get("app_config") or {}
        url = f"{self._base_url(cfg, request)}/chat/completions"
        headers = self._headers(api_key)
        payload = self._build_payload(request)
        payload["stream"] = True
        payload = merge_extra_body(payload, request)
        headers = merge_extra_headers(headers, request)
        try:
            resolved_timeout = self._resolve_timeout(request, timeout)
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
                        self._raise_if_in_band_provider_error(
                            line,
                            phase="stream_response",
                        )
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
            self._raise_sanitized_provider_failure(e, phase="stream")

    def normalize_error(self, exc: Exception):  # type: ignore[override]
        """Delegate to the shared bounded error policy."""
        return super().normalize_error(exc)

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
            body = None
            try:
                body = resp.json()
            except Exception:
                body = None
            log_http_400_body(self.name, exc, body)
            detail = None
            if isinstance(body, dict) and isinstance(body.get("error"), dict):
                eobj = body["error"]
                msg = (eobj.get("message") or "").strip()
                typ = (eobj.get("type") or "").strip()
                detail = (f"{typ} {msg}" if typ else msg) or str(exc)
            else:
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

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

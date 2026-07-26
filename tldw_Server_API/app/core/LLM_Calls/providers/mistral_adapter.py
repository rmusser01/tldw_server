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

# Expose a patchable factory for tests; production uses centralized client
http_client_factory = _hc_create_client


def _stream_debug_enabled(provider: str) -> bool:
    value = (os.getenv("LLM_ADAPTERS_STREAM_DEBUG") or "").strip().lower()
    if not value:
        return False
    if value in {"1", "true", "yes", "on", "all"}:
        return True
    providers = {p.strip() for p in value.split(",") if p.strip()}
    return provider.lower() in providers


class MistralAdapter(ChatProvider):
    name = "mistral"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 60,
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
            "streaming": streaming_raw,
            "topp": request.get("top_p"),
            "max_tokens": request.get("max_tokens"),
            "random_seed": request.get("seed"),
            "top_k": request.get("top_k"),
            "safe_prompt": request.get("safe_prompt"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "response_format": request.get("response_format"),
            "custom_prompt_arg": request.get("custom_prompt_arg"),
            "app_config": request.get("app_config"),
        }

    def _apply_config_defaults(self, request: dict[str, Any]) -> dict[str, Any]:
        merged = dict(request)
        cfg = (merged.get("app_config") or {}).get("mistral_api", {})
        if merged.get("api_key") is None and cfg.get("api_key") is not None:
            merged["api_key"] = cfg.get("api_key")
        if merged.get("model") is None:
            merged["model"] = cfg.get("model") or "mistral-large-latest"
        if merged.get("temperature") is None and cfg.get("temperature") is not None:
            merged["temperature"] = cfg.get("temperature")
        if merged.get("top_p") is None and cfg.get("top_p") is not None:
            merged["top_p"] = cfg.get("top_p")
        if merged.get("max_tokens") is None and cfg.get("max_tokens") is not None:
            merged["max_tokens"] = cfg.get("max_tokens")
        if merged.get("seed") is None and cfg.get("random_seed") is not None:
            merged["seed"] = cfg.get("random_seed")
        if merged.get("top_k") is None and cfg.get("top_k") is not None:
            merged["top_k"] = cfg.get("top_k")
        if merged.get("safe_prompt") is None and cfg.get("safe_prompt") is not None:
            merged["safe_prompt"] = cfg.get("safe_prompt")
        if merged.get("tools") is None and cfg.get("tools") is not None:
            merged["tools"] = cfg.get("tools")
        if merged.get("tool_choice") is None and cfg.get("tool_choice") is not None:
            merged["tool_choice"] = cfg.get("tool_choice")
        if merged.get("response_format") is None and cfg.get("response_format") is not None:
            merged["response_format"] = cfg.get("response_format")
        return merged

    def _base_url(self) -> str:
        return os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1").rstrip("/")

    def _resolve_base_url(self, request: dict[str, Any]) -> str:
        override = (request or {}).get("base_url")
        if isinstance(override, str) and override.strip():
            return override.strip()
        cfg = (request or {}).get("app_config") or {}
        if isinstance(cfg, dict):
            mcfg = cfg.get("mistral_api") or {}
            if isinstance(mcfg, dict):
                base = mcfg.get("api_base_url")
                if isinstance(base, str) and base.strip():
                    return base.strip().rstrip("/")
        if request.get("credentials_resolved") is True:
            return "https://api.mistral.ai/v1"
        return self._base_url()

    def _resolve_timeout(self, request: dict[str, Any], fallback: float | None) -> float:
        cfg = (request or {}).get("app_config") or {}
        if isinstance(cfg, dict):
            mcfg = cfg.get("mistral_api") or {}
            if isinstance(mcfg, dict):
                t = mcfg.get("api_timeout")
                if t is not None:
                    try:
                        return float(t)
                    except (TypeError, ValueError):
                        t = None
        if fallback is not None:
            return float(fallback)
        return float(self.capabilities().get("default_timeout_seconds", 60))

    def _headers(self, api_key: str | None) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    def _build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = request.get("messages") or []
        system_message = request.get("system_message")
        payload_messages: list[dict[str, Any]] = []
        if system_message:
            payload_messages.append({"role": "system", "content": system_message})
        payload_messages.extend(messages)
        payload: dict[str, Any] = {
            "model": request.get("model"),
            "messages": payload_messages,
        }
        if request.get("temperature") is not None:
            payload["temperature"] = request.get("temperature")
        if request.get("top_p") is not None:
            payload["top_p"] = request.get("top_p")
        if request.get("max_tokens") is not None:
            payload["max_tokens"] = request.get("max_tokens")
        if request.get("stop") is not None:
            payload["stop"] = request.get("stop")
        if request.get("tools") is not None:
            payload["tools"] = request.get("tools")
        if request.get("tool_choice") is not None:
            payload["tool_choice"] = request.get("tool_choice")
        if request.get("response_format") is not None:
            payload["response_format"] = request.get("response_format")
        if request.get("seed") is not None:
            payload["seed"] = request.get("seed")
        if request.get("top_k") is not None:
            payload["top_k"] = request.get("top_k")
        if request.get("safe_prompt") is not None:
            payload["safe_prompt"] = request.get("safe_prompt")
        return payload

    @staticmethod
    def _normalize_to_openai_shape(data: dict[str, Any]) -> dict[str, Any]:
        # Mistral speaks OpenAI-compatible shapes for chat/completions; passthrough
        return data

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

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = self._bind_request_credentials(request)
        request = self._apply_config_defaults(request or {})
        request = validate_payload(self.name, request or {})
        api_key = request.get("api_key")
        if not api_key:
            from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
            raise ChatConfigurationError(provider=self.name, message="Mistral API Key required.")
        url = f"{self._resolve_base_url(request)}/chat/completions"
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
                return self._normalize_to_openai_shape(data)
        except Exception as e:
            self._raise_sanitized_provider_failure(e, phase="chat")

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = self._bind_request_credentials(request)
        request = self._apply_config_defaults(request or {})
        request = validate_payload(self.name, request or {})
        api_key = request.get("api_key")
        if not api_key:
            from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
            raise ChatConfigurationError(provider=self.name, message="Mistral API Key required.")
        url = f"{self._resolve_base_url(request)}/chat/completions"
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
                    debug_stream = _stream_debug_enabled(self.name)
                    seen_done = False
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        if debug_stream:
                            logger.debug("{} stream chunk received", self.name)
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

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

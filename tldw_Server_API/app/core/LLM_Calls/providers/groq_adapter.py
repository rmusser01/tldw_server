from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

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

http_client_factory = _hc_create_client


def _prefer_httpx_in_tests() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


class GroqAdapter(ChatProvider):
    name = "groq"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 90,
            "max_output_tokens_default": 4096,
        }

    def _use_native_http(self) -> bool:
        # Always native unless explicitly disabled
        v = (os.getenv("LLM_ADAPTERS_NATIVE_HTTP_GROQ") or "").lower()
        return v not in {"0", "false", "no", "off"}

    def _base_url(self) -> str:
        import os
        # Groq exposes OpenAI-compatible API under /openai/v1
        return os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    def _resolve_base_url(self, request: dict[str, Any]) -> str:
        override = (request or {}).get("base_url")
        if isinstance(override, str) and override.strip():
            return override.strip()
        cfg = (request or {}).get("app_config") or {}
        if isinstance(cfg, dict):
            g = cfg.get("groq_api") or {}
            if isinstance(g, dict):
                base = g.get("api_base_url")
                if isinstance(base, str) and base.strip():
                    return base.strip()
        if request.get("credentials_resolved") is True:
            return "https://api.groq.com/openai/v1"
        return self._base_url()

    def _resolve_timeout(self, request: dict[str, Any], fallback: float | None) -> float:
        cfg = (request or {}).get("app_config") or {}
        if isinstance(cfg, dict):
            g = cfg.get("groq_api") or {}
            if isinstance(g, dict):
                t = g.get("api_timeout")
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
        payload = {
            "model": request.get("model"),
            "messages": payload_messages,
        }
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
        ):
            value = request.get(key)
            if value is not None:
                payload[key] = value
        # Tools and tool_choice gating (consistent with OpenAI-compatible behavior)
        tools = request.get("tools")
        if tools is not None:
            payload["tools"] = tools
        tc = request.get("tool_choice")
        if tc == "none":
            payload["tool_choice"] = "none"
        elif tc is not None and tools:
            payload["tool_choice"] = tc
        if request.get("response_format") is not None:
            payload["response_format"] = request.get("response_format")
        if request.get("seed") is not None:
            payload["seed"] = request.get("seed")
        if request.get("stop") is not None:
            payload["stop"] = request.get("stop")
        return payload

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = self._bind_request_credentials(request)
        request = validate_payload(self.name, request or {})
        if _prefer_httpx_in_tests() or os.getenv("PYTEST_CURRENT_TEST") or self._use_native_http():
            api_key = request.get("api_key")
            headers = self._headers(api_key)
            url = f"{self._resolve_base_url(request).rstrip('/')}/chat/completions"
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

        # Native disabled -> error to avoid legacy recursion
        raise RuntimeError("GroqAdapter native HTTP disabled by configuration")

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = self._bind_request_credentials(request)
        request = validate_payload(self.name, request or {})
        if _prefer_httpx_in_tests() or os.getenv("PYTEST_CURRENT_TEST") or self._use_native_http():
            api_key = request.get("api_key")
            headers = self._headers(api_key)
            url = f"{self._resolve_base_url(request).rstrip('/')}/chat/completions"
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

        # Native disabled -> error to avoid legacy recursion
        raise RuntimeError("GroqAdapter native HTTP disabled by configuration")

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

    def normalize_error(self, exc: Exception):  # type: ignore[override]
        """Delegate to the shared bounded error policy."""
        return super().normalize_error(exc)

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.cache_intents import (
    apply_billing_prompt_cache_intent,
    attach_cache_intent_metadata,
)
from tldw_Server_API.app.core.LLM_Calls.payload_utils import merge_extra_body, merge_extra_headers
from tldw_Server_API.app.core.LLM_Calls.sse import (
    finalize_stream,
    is_done_line,
    normalize_provider_line,
    sse_done,
)
from tldw_Server_API.app.core.LLM_Calls.streaming import wrap_sync_stream

from .base import ChatProvider


def _prefer_httpx_in_tests() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


from tldw_Server_API.app.core.http_client import (
    create_client as _hc_create_client,
)

http_client_factory = _hc_create_client

_OPENROUTER_CONFIG_EXCEPTIONS = (AttributeError, KeyError, TypeError, ValueError)
_OPENROUTER_DECODE_EXCEPTIONS = (TypeError, UnicodeDecodeError, ValueError)


def _build_openrouter_client_exceptions() -> tuple[type[BaseException], ...]:
    excs: list[type[BaseException]] = [
        ConnectionError,
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
    ]
    try:
        import httpx  # type: ignore
        excs.append(httpx.HTTPError)
    except (AttributeError, ImportError):
        pass
    return tuple(excs)


_OPENROUTER_CLIENT_EXCEPTIONS = _build_openrouter_client_exceptions()


class OpenRouterAdapter(ChatProvider):
    name = "openrouter"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 90,
            "max_output_tokens_default": 8192,
        }

    def _use_native_http(self) -> bool:
        # Always native unless explicitly disabled
        v = (os.getenv("LLM_ADAPTERS_NATIVE_HTTP_OPENROUTER") or "").lower()
        return v not in {"0", "false", "no", "off"}

    def _base_url(self) -> str:
        import os
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    def _resolve_base_url(self, request: dict[str, Any]) -> str:
        override = (request or {}).get("base_url")
        if isinstance(override, str) and override.strip():
            return override.strip()
        try:
            cfg = (request or {}).get("app_config") or {}
            or_cfg = cfg.get("openrouter_api") or {}
            base = or_cfg.get("api_base_url")
            if isinstance(base, str) and base.strip():
                return base.strip()
        except _OPENROUTER_CONFIG_EXCEPTIONS:
            pass
        return self._base_url()

    def _resolve_timeout(self, request: dict[str, Any], fallback: float | None) -> float:
        try:
            cfg = (request or {}).get("app_config") or {}
            or_cfg = cfg.get("openrouter_api") or {}
            t = or_cfg.get("api_timeout")
            if t is not None:
                try:
                    return float(t)
                except (TypeError, ValueError):
                    pass
        except _OPENROUTER_CONFIG_EXCEPTIONS:
            pass
        if fallback is not None:
            return float(fallback)
        return float(self.capabilities().get("default_timeout_seconds", 90))

    def _headers(self, api_key: str | None, request: dict[str, Any] | None = None) -> dict[str, str]:
        """Build headers including OpenRouter-specific metadata.

        - Authorization: Bearer <key>
        - HTTP-Referer: site URL (from config or env), defaults to http://localhost
        - X-Title: site name (from config or env), defaults to TLDW-API
        """
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"

        # Preserve provider-specific header quirks used by OpenRouter
        site_url = os.getenv("OPENROUTER_SITE_URL")
        site_name = os.getenv("OPENROUTER_SITE_NAME")
        try:
            cfg = (request or {}).get("app_config") or {}
            or_cfg = cfg.get("openrouter_api") or {}
            site_url = or_cfg.get("site_url") or site_url
            site_name = or_cfg.get("site_name") or site_name
        except _OPENROUTER_CONFIG_EXCEPTIONS:
            # best-effort; fall back to env/defaults
            pass
        # OpenRouter strongly prefers a valid public referer; use their site as a safe default
        h["HTTP-Referer"] = site_url or "https://openrouter.ai"
        h["X-Title"] = site_name or "TLDW-API"
        return h

    def _build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = request.get("messages") or []
        system_message = request.get("system_message")
        payload_messages: list[dict[str, Any]] = []
        if system_message:
            payload_messages.append({"role": "system", "content": system_message})
        payload_messages.extend(messages)
        # Start with required fields
        payload = {
            "model": request.get("model"),
            "messages": payload_messages,
        }
        # Add optional fields only when not None to avoid sending nulls
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "n",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
        ):
            val = request.get(key)
            if val is not None:
                payload[key] = val
        tool_choice = request.get("tool_choice")
        tools = request.get("tools")
        if tool_choice == "none":
            payload["tool_choice"] = "none"
        elif tool_choice is not None and tools:
            payload["tool_choice"] = tool_choice
        if tools is not None and tool_choice != "none":
            payload["tools"] = tools
        rf = request.get("response_format")
        # Forward response_format as-is for parity with other adapters and tests
        # (e.g., JSON mode: {"type": "json_object"}).
        if rf is not None:
            payload["response_format"] = rf
        if request.get("seed") is not None:
            payload["seed"] = request.get("seed")
        if request.get("stop") is not None:
            payload["stop"] = request.get("stop")
        return payload

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = validate_payload(self.name, request or {})
        if _prefer_httpx_in_tests() or os.getenv("PYTEST_CURRENT_TEST") or self._use_native_http():
            api_key = request.get("api_key")
            headers = self._headers(api_key, request)
            url = f"{self._resolve_base_url(request).rstrip('/')}/chat/completions"
            payload = self._build_payload(request)
            payload["stream"] = False
            payload, cache_intent_diagnostic = apply_billing_prompt_cache_intent(self.name, payload, request)
            payload = merge_extra_body(payload, request)
            headers = merge_extra_headers(headers, request)
            try:
                resolved_timeout = self._resolve_timeout(request, timeout)
                with http_client_factory(timeout=resolved_timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    return attach_cache_intent_metadata(resp.json(), cache_intent_diagnostic)
            except _OPENROUTER_CLIENT_EXCEPTIONS as e:
                raise self.normalize_error(e) from e

        # Native disabled -> error to avoid legacy recursion
        raise RuntimeError("OpenRouterAdapter native HTTP disabled by configuration")

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = validate_payload(self.name, request or {})
        if _prefer_httpx_in_tests() or os.getenv("PYTEST_CURRENT_TEST") or self._use_native_http():
            api_key = request.get("api_key")
            headers = self._headers(api_key, request)
            url = f"{self._resolve_base_url(request).rstrip('/')}/chat/completions"
            payload = self._build_payload(request)
            payload["stream"] = True
            payload, _cache_intent_diagnostic = apply_billing_prompt_cache_intent(self.name, payload, request)
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
                            except _OPENROUTER_DECODE_EXCEPTIONS:
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
            except _OPENROUTER_CLIENT_EXCEPTIONS as e:
                raise self.normalize_error(e) from e

        # Native disabled -> error to avoid legacy recursion
        raise RuntimeError("OpenRouterAdapter native HTTP disabled by configuration")

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

    def normalize_error(self, exc: Exception):  # type: ignore[override]
        """Parse OpenRouter error payloads and map to Chat*Error types.

        OpenRouter is OpenAI-compatible; error bodies often match {error: {message, type}}.
        """
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
            except (AttributeError, TypeError, ValueError):
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

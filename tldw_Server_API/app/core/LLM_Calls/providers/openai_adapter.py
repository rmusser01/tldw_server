from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import AsyncIterator, Iterable
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - optional for static analysis
    httpx = None

from tldw_Server_API.app.core.http_client import (
    create_client as _hc_create_client,
)
from tldw_Server_API.app.core.LLM_Calls.capability_registry import normalize_payload, validate_payload
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import _safe_cast
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

from .base import ChatProvider

# Expose a patchable factory for tests; production uses the centralized client
http_client_factory = _hc_create_client

_OPENAI_HTTP_EXCEPTIONS: tuple[type[BaseException], ...] = ()
if httpx is not None:
    _OPENAI_HTTP_EXCEPTIONS = (httpx.HTTPError,)

_OPENAI_ADAPTER_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeError,
    ValueError,
) + _OPENAI_HTTP_EXCEPTIONS

# Reuse the existing, stable implementation to ensure behavior parity during migration
# Do not import legacy handler at module import time to keep tests patchable.
# Resolve the function from the module at call time so monkeypatching
# tldw_Server_API.app.core.LLM_Calls.chat_calls.chat_with_openai works.


class OpenAIAdapter(ChatProvider):
    name = "openai"

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 90,
            "max_output_tokens_default": 4096,
        }

    def _apply_config_defaults(self, request: dict[str, Any]) -> dict[str, Any]:
        cfg = (request or {}).get("app_config") or {}
        oa = cfg.get("openai_api") or {}
        numeric_casts = {
            "temperature": float,
            "top_p": float,
            "max_tokens": int,
            "max_completion_tokens": int,
            "n": int,
            "seed": int,
            "presence_penalty": float,
            "frequency_penalty": float,
        }
        for key in (
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "response_format",
            "stop",
        ):
            if request.get(key) is None and oa.get(key) is not None:
                value = oa.get(key)
                caster = numeric_casts.get(key)
                if caster is not None:
                    value = _safe_cast(value, caster, None)
                    if value is None:
                        continue
                request[key] = value
        return request

    def _to_handler_args(self, request: dict[str, Any]) -> dict[str, Any]:
        """Translate OpenAI-like request dict to chat_with_openai kwargs."""
        messages = request.get("messages") or []
        model = request.get("model")
        api_key = request.get("api_key")
        system_message = request.get("system_message")
        temperature = request.get("temperature")
        top_p = request.get("top_p")
        # Preserve None to allow legacy default-from-config behavior
        streaming_raw = request.get("stream")
        if streaming_raw is None:
            streaming_raw = request.get("streaming")

        args: dict[str, Any] = {
            "input_data": messages,
            "model": model,
            "api_key": api_key,
            "system_message": system_message,
            "temp": temperature,
            "maxp": top_p,
            "streaming": streaming_raw,
            "frequency_penalty": request.get("frequency_penalty"),
            "logit_bias": request.get("logit_bias"),
            "logprobs": request.get("logprobs"),
            "top_logprobs": request.get("top_logprobs"),
            "max_tokens": request.get("max_tokens"),
            "n": request.get("n"),
            "presence_penalty": request.get("presence_penalty"),
            "response_format": request.get("response_format"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "user": request.get("user"),
            "custom_prompt_arg": request.get("custom_prompt_arg"),
            "app_config": request.get("app_config"),
        }
        return args

    def _use_native_http(self) -> bool:
        # Always use native HTTP for OpenAI adapter unless explicitly disabled
        v = (os.getenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI") or "").lower()
        return v not in {"0", "false", "no", "off"}

    def _build_openai_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages") or []
        system_message = request.get("system_message")
        payload_messages: list[dict[str, Any]] = []
        if system_message:
            payload_messages.append({"role": "system", "content": system_message})
        # Assume messages are already OpenAI format
        payload_messages.extend(messages)
        payload: dict[str, Any] = {
            "model": request.get("model"),
            "messages": payload_messages,
        }
        temperature = request.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        top_p = request.get("top_p")
        if top_p is not None:
            payload["top_p"] = top_p
        max_completion = request.get("max_completion_tokens")
        if max_completion is not None:
            payload["max_completion_tokens"] = max_completion
        else:
            max_tokens = request.get("max_tokens")
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
        n = request.get("n")
        if n is not None:
            payload["n"] = n
        presence_penalty = request.get("presence_penalty")
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        frequency_penalty = request.get("frequency_penalty")
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        logit_bias = request.get("logit_bias")
        if logit_bias is not None:
            payload["logit_bias"] = logit_bias
        user = request.get("user")
        if user is not None:
            payload["user"] = user
        # Propagate explicit stream flag for testability and parity with legacy path
        if request.get("stream") is not None:
            payload["stream"] = bool(request.get("stream"))
        # Tools and tool_choice gating to mirror legacy behavior
        tools = request.get("tools")
        if tools is not None:
            payload["tools"] = tools
        tool_choice = request.get("tool_choice")
        if tool_choice == "none":
            payload["tool_choice"] = "none"
        elif tool_choice is not None and tools:
            payload["tool_choice"] = tool_choice
        if request.get("response_format") is not None:
            payload["response_format"] = request.get("response_format")
        if request.get("seed") is not None:
            payload["seed"] = request.get("seed")
        if request.get("stop") is not None:
            payload["stop"] = request.get("stop")
        if request.get("logprobs") is not None:
            payload["logprobs"] = request.get("logprobs")
        if request.get("top_logprobs") is not None and request.get("logprobs"):
            payload["top_logprobs"] = request.get("top_logprobs")
        # gpt-5 models use max_completion_tokens and reject top_p
        model = payload.get("model")
        if isinstance(model, str) and model.lower().startswith("gpt-5"):
            if "max_tokens" in payload and "max_completion_tokens" not in payload:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
            else:
                payload.pop("max_tokens", None)
            payload.pop("top_p", None)
        return payload

    def _openai_base_url(self) -> str:
        import os
        # Match legacy resolution precedence used by chat_calls._resolve_openai_api_base
        env_api_base = (
            os.getenv("OPENAI_API_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("MOCK_OPENAI_BASE_URL")
        )
        return env_api_base or "https://api.openai.com/v1"

    def _resolve_base_url(self, request: dict[str, Any]) -> str:
        """Resolve API base URL: app_config.openai_api.api_base_url -> env -> default."""
        override = (request or {}).get("base_url")
        if isinstance(override, str) and override.strip():
            return override.strip()
        try:
            cfg = (request or {}).get("app_config") or {}
            oa = cfg.get("openai_api") or {}
            base = oa.get("api_base_url") or oa.get("api_base") or oa.get("base_url")
            if isinstance(base, str) and base.strip():
                return base.strip()
        except (AttributeError, LookupError, TypeError):
            pass
        return self._openai_base_url()

    def _resolve_timeout(self, request: dict[str, Any], fallback: float | None) -> float:
        try:
            cfg = (request or {}).get("app_config") or {}
            oa = cfg.get("openai_api") or {}
            t = oa.get("api_timeout")
            if t is not None:
                try:
                    return float(t)
                except (TypeError, ValueError):
                    pass
        except (AttributeError, LookupError, TypeError):
            pass
        if fallback is not None:
            return float(fallback)
        return float(self.capabilities().get("default_timeout_seconds", 60))

    def _openai_headers(self, api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = normalize_payload(self.name, request or {})
        request = self._apply_config_defaults(request)
        request = validate_payload(self.name, request)
        if self._use_native_http():
            api_key = request.get("api_key")
            payload = self._build_openai_payload(request)
            payload["stream"] = False
            url = f"{self._resolve_base_url(request).rstrip('/')}/chat/completions"
            payload, cache_intent_diagnostic = apply_billing_prompt_cache_intent(self.name, payload, request)
            payload = merge_extra_body(payload, request)
            headers = merge_extra_headers(self._openai_headers(api_key), request)
            try:
                resolved_timeout = self._resolve_timeout(request, timeout)
                with http_client_factory(timeout=resolved_timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    return attach_cache_intent_metadata(resp.json(), cache_intent_diagnostic)
            except _OPENAI_ADAPTER_NONCRITICAL_EXCEPTIONS as e:
                raise self.normalize_error(e) from e

        # If disabled explicitly, raise clear error rather than falling back
        raise RuntimeError("OpenAIAdapter native HTTP disabled by configuration")

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = normalize_payload(self.name, request or {})
        request = self._apply_config_defaults(request)
        request = validate_payload(self.name, request)
        if self._use_native_http():
            api_key = request.get("api_key")
            payload = self._build_openai_payload(request)
            payload["stream"] = True
            url = f"{self._resolve_base_url(request).rstrip('/')}/chat/completions"
            payload, _cache_intent_diagnostic = apply_billing_prompt_cache_intent(self.name, payload, request)
            payload = merge_extra_body(payload, request)
            headers = merge_extra_headers(self._openai_headers(api_key), request)
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
                            except (AttributeError, TypeError, UnicodeError):
                                line = str(raw)
                            # Canonicalize provider lines to OpenAI-style SSE
                            if is_done_line(line):
                                if not seen_done:
                                    seen_done = True
                                    yield sse_done()
                                continue
                            normalized = normalize_provider_line(line)
                            if normalized is not None:
                                yield normalized
                        # Ensure a single terminal DONE marker
                        yield from finalize_stream(response=resp, done_already=seen_done)
                return
            except _OPENAI_ADAPTER_NONCRITICAL_EXCEPTIONS as e:
                raise self.normalize_error(e) from e

        # If disabled explicitly, raise clear error rather than falling back
        raise RuntimeError("OpenAIAdapter native HTTP disabled by configuration")

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        gen = self.stream(request, timeout=timeout)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        stop_event = threading.Event()

        def _worker() -> None:
            try:
                for item in gen:
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                try:
                    if hasattr(gen, "close"):
                        gen.close()
                except _OPENAI_ADAPTER_NONCRITICAL_EXCEPTIONS:
                    pass
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop_event.set()

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
                eobj.get("code")
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

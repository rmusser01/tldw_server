from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

from tldw_Server_API.app.core.http_client import (
    create_client as _hc_create_client,
)
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_endpoint_env_keys,
    custom_openai_provider_name,
    custom_openai_section_name,
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
from tldw_Server_API.app.core.testing import is_truthy

from .base import ChatProvider

http_client_factory = _hc_create_client


class CustomOpenAIAdapter(ChatProvider):
    name = "custom-openai-api"
    config_section = "custom_openai_api"
    default_base_url = "http://127.0.0.1:11434/v1"
    default_base_url_env: tuple[str, ...] = custom_openai_endpoint_env_keys(1)

    def capabilities(self) -> dict[str, Any]:
        return {
            "supports_streaming": True,
            "supports_tools": True,
            "default_timeout_seconds": 120,
            "max_output_tokens_default": 4096,
        }

    def _to_handler_args(self, request: dict[str, Any]) -> dict[str, Any]:
        streaming_raw = request.get("stream")
        if streaming_raw is None:
            streaming_raw = request.get("streaming")
        return {
            "input_data": request.get("messages") or [],
            "api_key": request.get("api_key"),
            "custom_prompt_arg": request.get("custom_prompt_arg"),
            "temp": request.get("temperature"),
            "system_message": request.get("system_message"),
            "streaming": streaming_raw,
            "model": request.get("model"),
            # Compatibility knobs
            "maxp": request.get("top_p"),
            "topp": request.get("top_p"),
            "minp": request.get("min_p"),
            "topk": request.get("top_k"),
            "max_tokens": request.get("max_tokens"),
            "seed": request.get("seed"),
            "stop": request.get("stop"),
            "response_format": request.get("response_format"),
            "n": request.get("n"),
            "user_identifier": request.get("user"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
            "logit_bias": request.get("logit_bias"),
            "presence_penalty": request.get("presence_penalty"),
            "frequency_penalty": request.get("frequency_penalty"),
            "logprobs": request.get("logprobs"),
            "top_logprobs": request.get("top_logprobs"),
            "app_config": request.get("app_config"),
        }

    def _use_native_http(self) -> bool:
        import os
        if os.getenv("PYTEST_CURRENT_TEST"):
            return True
        v = (os.getenv("LLM_ADAPTERS_NATIVE_HTTP_CUSTOM_OPENAI") or "").strip().lower()
        if v in {"0", "false", "no", "off"}:
            return False
        if is_truthy(v):
            return True
        return True

    def _headers(self, api_key: str | None) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    def _resolve_base(self, request: dict[str, Any]) -> str:
        """Resolve the endpoint base URL from request, app config, env, or defaults."""
        override = (request or {}).get("base_url")
        if isinstance(override, str) and override.strip():
            return override.strip().rstrip("/")

        cfg = request.get("app_config") or {}
        section = cfg.get(self.config_section) or {}
        base = section.get("api_ip") or section.get("api_base_url")
        if not base:
            for env_key in self.default_base_url_env:
                env_val = os.getenv(env_key)
                if isinstance(env_val, str) and env_val.strip():
                    base = env_val.strip()
                    break
        if not base:
            if self.default_base_url:
                base = self.default_base_url
            else:
                raise RuntimeError(f"{self.name} requires an explicit base URL")
        return str(base).rstrip("/")

    @staticmethod
    def _build_chat_completions_url(base: str) -> str:
        lower = base.lower()
        if lower.endswith("/v1"):
            return f"{base}/chat/completions"
        if lower.endswith("/chat/completions"):
            return base
        return f"{base}/v1/chat/completions"

    def _build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        messages: list[dict[str, Any]] = request.get("messages") or []
        system_message = request.get("system_message")
        payload_messages: list[dict[str, Any]] = []
        if system_message:
            payload_messages.append({"role": "system", "content": system_message})
        payload_messages.extend(messages)
        payload: dict[str, Any] = {"messages": payload_messages, "stream": False}
        if request.get("model") is not None:
            payload["model"] = request.get("model")
        # OpenAI-compatible
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
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

    def _normalize_response(self, data: dict[str, Any]) -> dict[str, Any]:
        # Assume OpenAI-compatible; passthrough
        return data

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = validate_payload(self.name, request or {})
        if self._use_native_http():
            api_key = request.get("api_key")
            headers = self._headers(api_key)
            base = self._resolve_base(request)
            url = self._build_chat_completions_url(base)
            payload = self._build_payload(request)
            payload["stream"] = False
            payload = merge_extra_body(payload, request)
            headers = merge_extra_headers(headers, request)
            try:
                with http_client_factory(timeout=timeout or 120.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    resp.raise_for_status()
                    return self._normalize_response(resp.json())
            except Exception as e:
                raise self.normalize_error(e) from e
        raise RuntimeError("CustomOpenAIAdapter native HTTP disabled by configuration")

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = validate_payload(self.name, request or {})
        if self._use_native_http():
            api_key = request.get("api_key")
            headers = self._headers(api_key)
            base = self._resolve_base(request)
            url = self._build_chat_completions_url(base)
            payload = self._build_payload(request)
            payload["stream"] = True
            payload = merge_extra_body(payload, request)
            headers = merge_extra_headers(headers, request)
            try:
                with http_client_factory(timeout=timeout or 120.0) as client:
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
        raise RuntimeError("CustomOpenAIAdapter native HTTP disabled by configuration")

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

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


class CustomOpenAIAdapter2(CustomOpenAIAdapter):
    name = "custom-openai-api-2"
    config_section = "custom_openai_api_2"
    default_base_url = ""
    default_base_url_env = custom_openai_endpoint_env_keys(2)


def make_custom_openai_adapter_class(number: int) -> type[CustomOpenAIAdapter]:
    """Create or return the adapter class for a custom OpenAI provider slot."""
    if number == 1:
        return CustomOpenAIAdapter
    if number == 2:
        return CustomOpenAIAdapter2
    return type(
        f"CustomOpenAIAdapter{number}",
        (CustomOpenAIAdapter,),
        {
            "name": custom_openai_provider_name(number),
            "config_section": custom_openai_section_name(number),
            "default_base_url": "",
            "default_base_url_env": custom_openai_endpoint_env_keys(number),
            "__module__": __name__,
        },
    )


class NovitaAdapter(CustomOpenAIAdapter):
    name = "novita"
    config_section = "novita_api"
    default_base_url = "https://api.novita.ai/openai"
    default_base_url_env = ("NOVITA_BASE_URL", "NOVITA_API_BASE_URL")


class PoeAdapter(CustomOpenAIAdapter):
    name = "poe"
    config_section = "poe_api"
    default_base_url = "https://api.poe.com/v1"
    default_base_url_env = ("POE_BASE_URL", "POE_API_BASE_URL")


class TogetherAdapter(CustomOpenAIAdapter):
    name = "together"
    config_section = "together_api"
    default_base_url = "https://api.together.xyz/v1"
    default_base_url_env = ("TOGETHER_BASE_URL", "TOGETHER_API_BASE_URL")

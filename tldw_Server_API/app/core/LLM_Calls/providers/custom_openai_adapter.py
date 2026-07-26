from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_endpoint_env_keys,
    custom_openai_provider_name,
    custom_openai_section_name,
)
from tldw_Server_API.app.core.exceptions import EgressPolicyError, raise_detached_error
from tldw_Server_API.app.core.http_client import fetch as _hc_fetch
from tldw_Server_API.app.core.http_client import stream_response as _hc_stream_response
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.payload_utils import merge_extra_body, merge_extra_headers
from tldw_Server_API.app.core.LLM_Calls.provider_config_resolution import (
    TrustedProviderEndpoint,
    resolve_trusted_provider_endpoint,
)
from tldw_Server_API.app.core.LLM_Calls.sse import (
    is_done_line,
    normalize_provider_line,
    sse_done,
)
from tldw_Server_API.app.core.LLM_Calls.streaming import (
    provider_stream_error_frame,
    wrap_sync_stream,
)
from tldw_Server_API.app.core.testing import is_truthy

from .base import ChatProvider


def _provider_response_has_error(value: Any) -> bool:
    """Detect a protocol-owned provider error without retaining its detail."""

    from tldw_Server_API.app.core.Chat.streaming_utils import (
        normalize_provider_stream_error,
    )

    return normalize_provider_stream_error(value) is not None


class CustomOpenAIAdapter(ChatProvider):
    name = "custom-openai-api"
    config_section = "custom_openai_api"
    default_base_url = "http://127.0.0.1:11434/v1"
    default_base_url_env: tuple[str, ...] = custom_openai_endpoint_env_keys(1)
    http_fetcher = staticmethod(_hc_fetch)
    http_streamer = staticmethod(_hc_stream_response)

    _GENERIC_ENDPOINT_KEYS = (
        "base_url",
        "api_base_url",
        "api_base",
        "api_url",
        "api_ip",
    )

    _RESERVED_CONTEXT_KEYS = frozenset(
        {
            "app_config",
            "configured_endpoint",
            "configured_endpoint_base_url",
            "configured_endpoint_scope",
            "endpoint_provenance",
            "http_client_factory",
            "http_fetcher",
            "http_streamer",
            "trusted_base_url_override",
            PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
        }
    )

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
        for key in self._request_endpoint_keys():
            override = (request or {}).get(key)
            if isinstance(override, str) and override.strip():
                return override.strip().rstrip("/")

        cfg = request.get("app_config") or {}
        section = cfg.get(self.config_section) or {}
        base = section.get("api_ip") or section.get("api_base_url")
        if not base and request.get("credentials_resolved") is not True:
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

    def _is_configured_custom(self) -> bool:
        """Return whether this adapter is a configured custom slot, not a public service."""
        from tldw_Server_API.app.core.custom_openai_providers import custom_openai_provider_number

        return custom_openai_provider_number(self.name) is not None

    def _request_endpoint_keys(self) -> tuple[str, ...]:
        """Return supported raw endpoint fields in compatibility precedence order."""
        if not self._is_configured_custom():
            return ("base_url",)
        return tuple(
            dict.fromkeys(
                (*self._GENERIC_ENDPOINT_KEYS, *(key.lower() for key in self.default_base_url_env))
            )
        )

    def _sanitize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Strip request-owned authorization and transport context before validation."""
        sanitized = dict(request or {})
        for key in (*self._RESERVED_CONTEXT_KEYS, *self._request_endpoint_keys()):
            sanitized.pop(key, None)
        sanitized.pop("_endpoint_provenance", None)
        return sanitized

    def _resolve_transport_context(
        self,
        request: dict[str, Any],
        credentials: ProviderCallCredentials | None = None,
    ) -> tuple[str, TrustedProviderEndpoint | None]:
        """Resolve a scoped server endpoint or an explicit ordinary-egress endpoint."""
        if credentials is not None:
            endpoint = credentials.trusted_endpoint
            if self._is_configured_custom():
                if endpoint is None:
                    raise ChatConfigurationError(
                        provider=self.name,
                        message=f"{self.name} endpoint is not configured.",
                    )
                return endpoint.base_url, endpoint
            return self._resolve_base(request), endpoint

        if not self._is_configured_custom():
            return self._resolve_base(request), None

        provenance = request.get("_endpoint_provenance")
        if provenance in {"byok", "request_override"}:
            return self._resolve_base(request), None

        endpoint = resolve_trusted_provider_endpoint(self.name)
        if endpoint is None:
            raise RuntimeError(f"{self.name} requires an explicit configured base URL")
        return endpoint.base_url, endpoint

    def _consume_runtime_credentials(
        self,
        request: dict[str, Any],
    ) -> ProviderCallCredentials | None:
        """Replace loose credential fields with one authentic runtime snapshot."""

        bound, credentials = self._bind_request_credentials_with_handle(request)
        request.clear()
        request.update(bound)
        return credentials

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
        raw_request = dict(request or {})
        credentials = self._consume_runtime_credentials(raw_request)
        base, endpoint = self._resolve_transport_context(raw_request, credentials)
        request = validate_payload(self.name, self._sanitize_request(raw_request))
        if self._use_native_http():
            api_key = request.get("api_key")
            headers = self._headers(api_key)
            url = self._build_chat_completions_url(base)
            payload = self._build_payload(request)
            payload["stream"] = False
            payload = merge_extra_body(payload, request)
            headers = merge_extra_headers(headers, request)
            try:
                redirect_options = (
                    {} if self._is_configured_custom() else {"allow_redirects": False}
                )
                resp = self.http_fetcher(
                    method="POST",
                    url=url,
                    configured_endpoint=endpoint.scope if endpoint else None,
                    headers=headers,
                    json=payload,
                    timeout=timeout or 120.0,
                    **redirect_options,
                )
                try:
                    resp.raise_for_status()
                    data = resp.json()
                    if _provider_response_has_error(data):
                        raise RuntimeError("Provider returned an error response")
                    return self._normalize_response(data)
                finally:
                    resp.close()
            except EgressPolicyError:
                raise
            except Exception as e:
                raise_detached_error(super().normalize_error(e))
        raise RuntimeError("CustomOpenAIAdapter native HTTP disabled by configuration")

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        raw_request = dict(request or {})
        credentials = self._consume_runtime_credentials(raw_request)
        base, endpoint = self._resolve_transport_context(raw_request, credentials)
        request = validate_payload(self.name, self._sanitize_request(raw_request))
        if self._use_native_http():
            api_key = request.get("api_key")
            headers = self._headers(api_key)
            url = self._build_chat_completions_url(base)
            payload = self._build_payload(request)
            payload["stream"] = True
            payload = merge_extra_body(payload, request)
            headers = merge_extra_headers(headers, request)
            try:
                provider_error = False
                stream_completed = False
                with self.http_streamer(
                    method="POST",
                    url=url,
                    configured_endpoint=endpoint.scope if endpoint else None,
                    headers=headers,
                    json=payload,
                    timeout=timeout or 120.0,
                ) as resp:
                    resp.raise_for_status()
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        try:
                            line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                        except Exception:
                            line = str(raw)
                        if is_done_line(line):
                            break
                        if _provider_response_has_error(line):
                            provider_error = True
                            break
                        normalized = normalize_provider_line(line)
                        if normalized is not None:
                            yield normalized
                    stream_completed = True
                if not stream_completed:
                    raise RuntimeError("Provider stream did not complete")
                if provider_error:
                    yield provider_stream_error_frame(self.name)
                yield sse_done()
                return
            except EgressPolicyError:
                raise
            except Exception as e:
                raise_detached_error(super().normalize_error(e))
        raise RuntimeError("CustomOpenAIAdapter native HTTP disabled by configuration")

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

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

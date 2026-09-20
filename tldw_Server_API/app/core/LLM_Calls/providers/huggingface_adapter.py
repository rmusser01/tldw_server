from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_config import is_runtime_base_url_override
from tldw_Server_API.app.core.exceptions import raise_detached_error
from tldw_Server_API.app.core.http_client import (
    create_client as _hc_create_client,
)
from tldw_Server_API.app.core.LLM_Calls.capability_registry import validate_payload
from tldw_Server_API.app.core.LLM_Calls.payload_utils import (
    encode_huggingface_model_path,
    encode_provider_model_path,
    merge_extra_body,
    merge_extra_headers,
)
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
_DEFAULT_API_BASE = "https://api-inference.huggingface.co/v1"
_DEFAULT_ROUTER_BASE = "https://router.huggingface.co/hf-inference"


def _optional_config_text(value: Any) -> str | None:
    """Normalize an optional config value without treating whitespace as configured."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _configuration_error() -> Exception:
    """Build one bounded configuration error without reflecting endpoint data."""
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError

    return ChatConfigurationError(
        provider="huggingface",
        message="Invalid Hugging Face endpoint configuration.",
    )


def _validated_base_url(value: Any) -> str:
    """Return one normalized HTTP(S) base URL or fail closed."""
    normalized = _optional_config_text(value)
    if normalized is None:
        raise _configuration_error()
    try:
        parsed = urlsplit(normalized)
        _ = parsed.port
    except (TypeError, ValueError):
        raise _configuration_error() from None
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "\\" in normalized
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in normalized)
    ):
        raise _configuration_error()
    return normalized.rstrip("/")


def _configured_chat_path(cfg: Mapping[str, Any], selected_base: str) -> str:
    """Resolve and validate the path after the final endpoint has been selected."""
    explicit_path = _optional_config_text(cfg.get("api_chat_path"))
    if explicit_path is None:
        explicit_path = _optional_config_text(cfg.get("huggingface_api_chat_path"))
    if explicit_path is not None:
        try:
            return encode_provider_model_path(explicit_path.strip("/"))
        except ValueError:
            raise _configuration_error() from None

    path_segments = [segment for segment in urlsplit(selected_base).path.split("/") if segment]
    return "chat/completions" if path_segments and path_segments[-1] == "v1" else "v1/chat/completions"


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
        return dict.fromkeys(headers, "***")

    def _resolve_url_and_headers(self, request: dict[str, Any]) -> dict[str, Any]:
        """Resolve a Hugging Face endpoint from trusted server-built request state.

        ``app_config`` is an internal adapter contract. Public request handlers
        must reject it and remove undeclared fields before adapter dispatch.
        """
        app_config = request.get("app_config") or {}
        cfg_value = app_config.get("huggingface_api", {}) if isinstance(app_config, Mapping) else {}
        cfg = cfg_value if isinstance(cfg_value, Mapping) else {}
        override_base = _optional_config_text(request.get("base_url"))
        api_base = _optional_config_text(cfg.get("api_base_url"))
        runtime_base_override = is_runtime_base_url_override(
            cfg.get("_runtime_base_url_override")
        )
        if runtime_base_override and api_base is None:
            raise _configuration_error()
        if runtime_base_override:
            _validated_base_url(api_base)

        use_router_value = _optional_config_text(cfg.get("use_router_url_format"))
        if use_router_value is None:
            use_router_value = _optional_config_text(
                cfg.get("huggingface_use_router_url_format")
            )
        use_router = (use_router_value or "false").casefold() == "true"

        model = request.get("model") or cfg.get("model_id") or cfg.get("model")
        if not model:
            model = "unspecified"
        if use_router:
            router_base = _optional_config_text(cfg.get("router_base_url"))
            if router_base is None:
                router_base = _optional_config_text(cfg.get("huggingface_router_base_url"))
            selected_base = (
                override_base
                or (api_base if runtime_base_override else None)
                or router_base
                or _DEFAULT_ROUTER_BASE
            )
            base = _validated_base_url(selected_base)
            chat_path = _configured_chat_path(cfg, base)
            try:
                model_path = encode_huggingface_model_path(model)
            except ValueError:
                from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError

                raise ChatBadRequestError(
                    provider=self.name,
                    message="Invalid provider model identifier.",
                ) from None
            url = f"{base}/models/{model_path}/{chat_path}"
        else:
            base = _validated_base_url(override_base or api_base or _DEFAULT_API_BASE)
            chat_path = _configured_chat_path(cfg, base)
            url = f"{base}/{chat_path}"
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
                    logger.debug(
                        "HuggingFace adapter timeout value is not numeric error_type={}",
                        type(timeout_parse_error).__name__,
                    )
        except Exception as config_error:
            logger.debug(
                "HuggingFace adapter failed to read timeout config error_type={}",
                type(config_error).__name__,
            )
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
        """Delegate to the shared bounded error policy."""
        return super().normalize_error(exc)

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        request = self._bind_request_credentials(request)
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
            raise_detached_error(self.normalize_error(e))

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(self.chat, request, timeout=timeout)

    async def astream(self, request: dict[str, Any], *, timeout: float | None = None) -> AsyncIterator[str]:
        async for item in wrap_sync_stream(self.stream(request, timeout=timeout)):
            yield item

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        request = self._bind_request_credentials(request)
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
                data = resp.json()
                self._raise_if_in_band_provider_error(data, phase="chat_response")
                return data
        except Exception as e:
            raise_detached_error(self.normalize_error(e))

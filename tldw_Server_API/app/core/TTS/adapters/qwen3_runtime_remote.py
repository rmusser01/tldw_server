"""Remote Qwen3-TTS runtime backed by an OpenAI-compatible HTTP service."""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.exceptions import (
    NetworkError as CoreNetworkError,
)
from tldw_Server_API.app.core.exceptions import (
    RetryExhaustedError,
    raise_detached_error,
)
from tldw_Server_API.app.core.http_client import RetryPolicy, apost, astream_bytes

from ..tts_exceptions import (
    TTSError,
    TTSNetworkError,
    TTSProviderError,
    TTSProviderInitializationError,
    auth_error,
    rate_limit_error,
    timeout_error,
)
from ..tts_resource_manager import get_resource_manager
from .base import AudioFormat, TTSCapabilities, TTSRequest, TTSResponse

if TYPE_CHECKING:
    from .qwen3_tts_adapter import Qwen3TTSAdapter


def _rebuild_typed_tts_error(
    exc: TTSError,
    *,
    provider: str,
    message: str,
) -> TTSError:
    """Recreate a local TTS error category without copying provider-owned state."""

    error_class: type[TTSError] = type(exc)
    if error_class.__module__ != TTSError.__module__:
        error_class = TTSError
    return error_class(
        message,
        provider=provider,
        details={"error_type": error_class.__name__},
    )


class RemoteQwenRuntime:
    """Execute Qwen3-TTS requests against a separately hosted backend."""

    runtime_name = "remote"

    def __init__(self, adapter_or_config: Qwen3TTSAdapter | dict[str, Any]) -> None:
        if hasattr(adapter_or_config, "config"):
            adapter = adapter_or_config
            self._adapter = adapter
            self.config = dict(getattr(adapter, "config", {}) or {})
            self.provider_key = getattr(adapter, "PROVIDER_KEY", "qwen3_tts")
            self.provider_name = getattr(adapter, "provider_name", self.provider_key)
            self.sample_rate = getattr(adapter, "sample_rate", 24000)
            self.supported_languages = set(getattr(adapter, "SUPPORTED_LANGUAGES", {"en"}))
            self.supported_voices: list[str] = []
        else:
            self._adapter = None
            self.config = dict(adapter_or_config or {})
            self.provider_key = "qwen3_tts"
            self.provider_name = "Qwen3TTS"
            self.sample_rate = int(self.config.get("sample_rate") or 24000)
            self.supported_languages = {
                "auto",
                "en",
                "zh",
                "ja",
                "ko",
                "de",
                "fr",
                "ru",
                "pt",
                "es",
                "it",
            }
            self.supported_voices = []

        self.base_url = str(self.config.get("base_url") or "").strip()
        self.api_key = str(self.config.get("api_key") or "").strip() or None
        self.client = None

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def _capability_override(self) -> dict[str, Any]:
        override = self.config.get("capability_override")
        if isinstance(override, dict):
            return dict(override)
        return {}

    def _resolve_supported_voices(self, override: dict[str, Any]) -> list[str]:
        configured = override.get("supported_voices")
        if not isinstance(configured, list):
            return list(self.supported_voices)
        voices: list[str] = []
        for voice in configured:
            if not isinstance(voice, str):
                continue
            normalized = voice.strip()
            if normalized:
                voices.append(normalized)
        return voices

    def _is_httpx_exception(self, exc: Exception) -> bool:
        module = getattr(exc.__class__, "__module__", "")
        return module.startswith("httpx")

    def _is_http_status_error(self, exc: Exception) -> bool:
        if not self._is_httpx_exception(exc):
            return False
        return exc.__class__.__name__ == "HTTPStatusError"

    def _is_timeout_error(self, exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return True
        name = exc.__class__.__name__.lower()
        return "timeout" in name

    def _parse_retry_after(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _normalize_http_status_error(self, exc: Exception) -> TTSError:
        """Map an HTTP status failure without retaining its request or response."""

        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        headers = getattr(response, "headers", {}) if response is not None else {}

        if status_code == 401:
            return auth_error(self.provider_key, "Invalid API key")
        if status_code == 429:
            retry_after = None
            if hasattr(headers, "get"):
                retry_after = self._parse_retry_after(headers.get("retry-after"))
            return rate_limit_error(
                self.provider_key,
                retry_after=retry_after,
            )
        if status_code == 400:
            return TTSProviderError(
                "Invalid request to remote Qwen backend",
                provider=self.provider_key,
                error_code="BAD_REQUEST",
                details={"status": status_code},
            )
        return TTSProviderError(
            "Remote Qwen API request failed",
            provider=self.provider_key,
            error_code=str(status_code),
            details={"status": status_code},
        )

    def _normalize_remote_error(self, exc: Exception) -> TTSError:
        """Create a sanitized domain error from a remote transport failure."""

        if self._is_http_status_error(exc):
            return self._normalize_http_status_error(exc)
        if isinstance(exc, (CoreNetworkError, RetryExhaustedError)) or self._is_httpx_exception(exc):
            if self._is_timeout_error(exc):
                return timeout_error(
                    self.provider_key,
                    timeout_seconds=int(self.config.get("timeout") or 60),
                )
            return TTSNetworkError(
                f"Network request to {self.provider_key} failed",
                provider=self.provider_key,
                error_code="NETWORK_ERROR",
                details={"error_type": type(exc).__name__},
            )
        if isinstance(exc, TTSError):
            return _rebuild_typed_tts_error(
                exc,
                provider=self.provider_key,
                message="Remote Qwen request failed",
            )
        return TTSProviderError(
            "Unexpected error in remote Qwen runtime",
            provider=self.provider_key,
            details={"error_type": type(exc).__name__},
        )

    async def _handle_http_status_error(self, exc: Exception) -> None:
        """Raise the sanitized status error for compatibility with direct callers."""

        safe_error = self._normalize_http_status_error(exc)
        raise_detached_error(safe_error)

    async def _raise_remote_error(self, exc: Exception) -> None:
        """Raise the sanitized remote error for compatibility with direct callers."""

        safe_error = self._normalize_remote_error(exc)
        raise_detached_error(safe_error)

    async def initialize(self) -> bool:
        if not self.base_url:
            raise TTSProviderInitializationError(
                "Qwen3-TTS remote runtime requires base_url",
                provider=self.provider_key,
            )
        resource_manager = await get_resource_manager()
        self.client = await resource_manager.get_http_client(
            provider=f"{self.provider_key}_{self.runtime_name}",
            base_url=self.base_url,
        )
        return True

    async def get_capabilities(self) -> TTSCapabilities:
        override = self._capability_override()
        supported_voices = []
        runtime_supported_voices = self._resolve_supported_voices(override)
        if runtime_supported_voices:
            from .base import VoiceInfo

            supported_voices = [VoiceInfo(id=voice, name=voice) for voice in runtime_supported_voices]
        supports_streaming = self._coerce_bool(override.get("supports_streaming"), default=False)
        supports_voice_cloning = self._coerce_bool(override.get("supports_voice_cloning"), default=False)
        supports_emotion_control = self._coerce_bool(override.get("supports_emotion_control"), default=False)
        supported_modes = override.get("supported_modes")
        if not isinstance(supported_modes, list) or not supported_modes:
            supported_modes = ["custom_voice_preset"]
        supports_uploaded_custom_voices = self._coerce_bool(
            override.get("supports_uploaded_custom_voices"),
            default=False,
        )
        return TTSCapabilities(
            provider_name=self.provider_name,
            supported_languages=set(self.supported_languages),
            supported_voices=supported_voices,
            supported_formats={
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.AAC,
                AudioFormat.WAV,
                AudioFormat.PCM,
            },
            max_text_length=int(self.config.get("max_text_length") or 5000),
            supports_streaming=supports_streaming,
            supports_voice_cloning=supports_voice_cloning,
            supports_emotion_control=supports_emotion_control,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.PCM,
            metadata={
                "runtime": self.runtime_name,
                "supported_modes": supported_modes,
                "supports_uploaded_custom_voices": supports_uploaded_custom_voices,
            },
        )

    def _build_payload(self, request: TTSRequest, resolved_model: str, mode: str) -> dict[str, Any]:
        extras = request.extra_params or {}
        payload: dict[str, Any] = {
            "model": resolved_model,
            "input": request.text,
            "voice": request.voice or "",
            "response_format": request.format.value,
            "speed": request.speed,
            "extra_body": {},
        }

        language = request.language or extras.get("language")
        if isinstance(language, str) and language.strip():
            payload["extra_body"]["language"] = language.strip()

        if mode == "voice_clone":
            ref_text = (
                extras.get("reference_text")
                or extras.get("ref_text")
                or extras.get("voice_reference_text")
            )
            if ref_text:
                payload["extra_body"]["ref_text"] = ref_text
            if request.voice_reference:
                payload["extra_body"]["ref_audio_b64"] = base64.b64encode(request.voice_reference).decode("ascii")
            if extras.get("x_vector_only_mode") is not None:
                payload["extra_body"]["x_vector_only_mode"] = extras.get("x_vector_only_mode")
            if extras.get("voice_clone_prompt") is not None:
                payload["extra_body"]["voice_clone_prompt"] = extras.get("voice_clone_prompt")
        elif mode == "voice_design":
            description = extras.get("description") or extras.get("instruction") or extras.get("instruct")
            if description:
                payload["extra_body"]["description"] = description

        return payload

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_stream_retry_policy(self) -> RetryPolicy:
        return RetryPolicy(attempts=1)

    async def _stream_audio(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncGenerator[bytes, None]:
        safe_error: TTSError | None = None
        try:
            async with contextlib.aclosing(
                astream_bytes(
                    method="POST",
                    url=self.base_url,
                    client=self.client,
                    headers=headers,
                    json=payload,
                    retry=self._build_stream_retry_policy(),
                    chunk_size=1024,
                    sensitive_observability=True,
                )
            ) as stream:
                async for chunk in stream:
                    if chunk:
                        yield chunk
        except Exception as exc:
            safe_error = self._normalize_remote_error(exc)
        if safe_error is not None:
            raise_detached_error(safe_error)

    async def _generate_complete(self, headers: dict[str, str], payload: dict[str, Any]) -> bytes:
        response = await apost(
            url=self.base_url,
            client=self.client,
            headers=headers,
            json=payload,
            sensitive_observability=True,
        )
        response.raise_for_status()
        return response.content

    async def generate(self, request: TTSRequest, resolved_model: str, mode: str) -> TTSResponse:
        payload = self._build_payload(request, resolved_model=resolved_model, mode=mode)
        headers = self._build_headers()

        try:
            if request.stream:
                return TTSResponse(
                    audio_stream=self._stream_audio(headers, payload),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    provider=self.provider_key,
                    model=resolved_model,
                    metadata={"runtime": self.runtime_name},
                )

            audio_data = await self._generate_complete(headers, payload)
            return TTSResponse(
                audio_content=audio_data,
                format=request.format,
                sample_rate=self.sample_rate,
                provider=self.provider_key,
                model=resolved_model,
                metadata={"runtime": self.runtime_name},
            )
        except Exception as exc:
            safe_error = self._normalize_remote_error(exc)
        raise_detached_error(safe_error)

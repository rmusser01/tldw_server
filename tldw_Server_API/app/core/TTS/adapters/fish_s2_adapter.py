"""Fish Audio S2 provider adapter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any, Optional

from loguru import logger

from ..backends.fish_s2_base import FishS2Backend
from ..backends.fish_s2_native_http import FishS2NativeHttpBackend
from ..tts_exceptions import (
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
)
from ..tts_validation import validate_tts_request
from .base import (
    AudioFormat,
    ProviderStatus,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
)


_SUPPORTED_LANGUAGES = {
    "ar",
    "auto",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hi",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sv",
    "th",
    "tr",
    "uk",
    "vi",
    "zh",
}


def _build_backend(config: dict[str, Any]) -> FishS2Backend:
    backend_name = str(config.get("backend", "native_http")).strip().lower()
    if backend_name == "native_http":
        return FishS2NativeHttpBackend(config)
    if backend_name == "local_runtime":
        raise TTSProviderInitializationError(
            "Fish S2 local_runtime backend is not implemented yet",
            provider="fish_s2",
        )
    raise TTSProviderInitializationError(
        f"Unknown Fish S2 backend '{backend_name}'",
        provider="fish_s2",
        details={"backend": backend_name},
    )


class FishS2Adapter(TTSAdapter):
    """Registry-facing adapter for Fish Audio S2."""

    PROVIDER_KEY = "fish_s2"
    SUPPORTED_FORMATS = {AudioFormat.WAV, AudioFormat.MP3, AudioFormat.PCM}

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        extras = cfg.get("extra_params", {}) or {}

        self.backend_name = str(cfg.get("backend", "native_http")).strip().lower()
        self.sample_rate = int(cfg.get("sample_rate", 24000))
        self.max_text_length = int(cfg.get("max_text_length", 5000))
        self.default_chunk_length = extras.get("default_chunk_length")
        self.default_normalize = extras.get("default_normalize")
        self.default_use_memory_cache = extras.get("default_use_memory_cache")
        self._backend: FishS2Backend | None = None

    async def ensure_initialized(self) -> bool:
        """Propagate initialization errors for clearer operator feedback."""
        if self._initialized:
            return True

        async with self._init_lock:
            if self._initialized:
                return True

            self._status = ProviderStatus.INITIALIZING
            success = await self.initialize()
            if success:
                self._capabilities = await self.get_capabilities()
                self._status = ProviderStatus.AVAILABLE
                self._initialized = True
            else:
                self._status = ProviderStatus.ERROR
            return success

    async def initialize(self) -> bool:
        self._backend = _build_backend(self.config)

        is_healthy = await self._backend.health_check()
        if not is_healthy:
            raise TTSProviderInitializationError(
                "Fish S2 backend health check failed",
                provider=self.PROVIDER_KEY,
            )

        logger.info("Fish S2 adapter initialized with backend={}", self.backend_name)
        return True

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="Fish Audio S2",
            supported_languages=_SUPPORTED_LANGUAGES,
            supported_voices=[],
            supported_formats=self.SUPPORTED_FORMATS,
            max_text_length=self.max_text_length,
            supports_streaming=True,
            supports_voice_cloning=True,
            supports_emotion_control=False,
            supports_speech_rate=False,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=250,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "Fish S2 adapter not initialized",
                provider=self.PROVIDER_KEY,
            )
        if self._backend is None:
            raise TTSProviderInitializationError(
                "Fish S2 backend is not available",
                provider=self.PROVIDER_KEY,
            )

        normalized_request = self._normalize_request(request)
        validate_tts_request(normalized_request, provider=self.provider_key)

        reference_id = self._resolve_reference_id(normalized_request)
        backend_extra_params = self._build_backend_extra_params(normalized_request.extra_params)
        result = await self._backend.synthesize(
            text=self.preprocess_text(normalized_request.text),
            response_format=normalized_request.format.value,
            streaming=normalized_request.stream,
            reference_id=reference_id,
            extra_params=backend_extra_params or None,
        )

        if normalized_request.stream:
            return TTSResponse(
                audio_stream=result,  # type: ignore[arg-type]
                format=normalized_request.format,
                sample_rate=self.sample_rate,
                channels=1,
                voice_used=reference_id or normalized_request.voice,
                provider=self.PROVIDER_KEY,
                model=normalized_request.model,
            )

        return TTSResponse(
            audio_data=result,  # type: ignore[arg-type]
            format=normalized_request.format,
            sample_rate=self.sample_rate,
            channels=1,
            voice_used=reference_id or normalized_request.voice,
            provider=self.PROVIDER_KEY,
            model=normalized_request.model,
        )

    async def add_reference(
        self,
        *,
        reference_id: str,
        audio_b64: str,
        reference_text: str,
    ) -> dict[str, Any]:
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "Fish S2 adapter not initialized",
                provider=self.PROVIDER_KEY,
            )
        if self._backend is None:
            raise TTSProviderInitializationError(
                "Fish S2 backend is not available",
                provider=self.PROVIDER_KEY,
            )
        return await self._backend.add_reference(
            reference_id=reference_id,
            audio_b64=audio_b64,
            reference_text=reference_text,
        )

    async def delete_reference(self, *, reference_id: str) -> bool:
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "Fish S2 adapter not initialized",
                provider=self.PROVIDER_KEY,
            )
        if self._backend is None:
            raise TTSProviderInitializationError(
                "Fish S2 backend is not available",
                provider=self.PROVIDER_KEY,
            )
        return await self._backend.delete_reference(reference_id=reference_id)

    def _normalize_request(self, request: TTSRequest) -> TTSRequest:
        extras = dict(request.extra_params or {})
        voice = request.voice

        if isinstance(voice, str) and voice.startswith("fishref:"):
            logical_id = voice.split("fishref:", 1)[-1].strip()
            if logical_id and "reference_id" not in extras:
                extras["reference_id"] = logical_id
            voice = None

        return replace(request, voice=voice, extra_params=extras)

    def _resolve_reference_id(self, request: TTSRequest) -> str | None:
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        explicit_reference_id = extras.get("reference_id")
        if explicit_reference_id:
            return str(explicit_reference_id)

        voice = request.voice or ""
        if isinstance(voice, str) and voice.startswith("fishref:"):
            logical_id = voice.split("fishref:", 1)[-1].strip()
            return logical_id or None
        return None

    def _build_backend_extra_params(self, extra_params: dict[str, Any] | None) -> dict[str, Any]:
        extras = dict(extra_params or {})
        extras.pop("reference_id", None)

        backend_extra_params: dict[str, Any] = {}
        if self.default_chunk_length is not None and "chunk_length" not in extras:
            backend_extra_params["chunk_length"] = self.default_chunk_length
        if self.default_normalize is not None and "normalize" not in extras:
            backend_extra_params["normalize"] = self.default_normalize
        if self.default_use_memory_cache is not None and "use_memory_cache" not in extras:
            backend_extra_params["use_memory_cache"] = self.default_use_memory_cache

        for key in (
            "chunk_length",
            "normalize",
            "seed",
            "top_p",
            "temperature",
            "repetition_penalty",
            "use_memory_cache",
            "references",
        ):
            value = extras.get(key)
            if value is not None:
                backend_extra_params[key] = value

        return backend_extra_params

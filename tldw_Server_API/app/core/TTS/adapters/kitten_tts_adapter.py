"""KittenTTS adapter implementation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any, Optional

import numpy as np
from loguru import logger

from ..streaming_audio_writer import AudioNormalizer, StreamingAudioWriter
from ..tts_exceptions import TTSGenerationError, TTSProviderInitializationError, TTSProviderNotConfiguredError
from ..tts_validation import validate_tts_request
from ..utils import parse_bool
from ..vendors.kittentts_compat import (
    DEFAULT_REPO_ID,
    DEFAULT_VOICE_ALIASES,
    KittenRuntime,
    download_model_assets,
    normalize_repo_id,
    resolve_model_revision,
)
from .base import AudioFormat, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo


class KittenTTSAdapter(TTSAdapter):
    """Adapter for KittenTTS ONNX models."""

    PROVIDER_KEY = "kitten_tts"
    STATIC_CAPABILITY_DISCOVERY = True
    handles_text_chunking = True

    SUPPORTED_FORMATS = {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.PCM}
    SUPPORTED_LANGUAGES = {"en"}
    DEFAULT_SAMPLE_RATE = 24000
    MAX_TEXT_LENGTH = 5000

    _VOICE_GENDERS = {
        "Bella": "female",
        "Jasper": "male",
        "Luna": "female",
        "Bruno": "male",
        "Rosie": "female",
        "Hugo": "male",
        "Kiki": "female",
        "Leo": "male",
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        self.model_name = normalize_repo_id(cfg.get("model") or DEFAULT_REPO_ID)
        extras = cfg.get("extra_params", {}) or {}
        self.model_revision = (
            str(
                cfg.get("model_revision")
                or extras.get("model_revision")
                or os.getenv("KITTEN_TTS_MODEL_REVISION")
                or ""
            ).strip()
            or None
        )
        self.cache_dir = cfg.get("cache_dir") or extras.get("cache_dir")
        self.auto_download = parse_bool(cfg.get("auto_download"), default=True)
        self.clean_text = parse_bool(cfg.get("clean_text"), default=parse_bool(extras.get("clean_text"), default=False))
        self.sample_rate = int(cfg.get("sample_rate", self.DEFAULT_SAMPLE_RATE))
        self._audio_normalizer = AudioNormalizer()
        self._runtime: Optional[KittenRuntime] = None
        self._runtime_assets = None
        self._runtime_lock = asyncio.Lock()
        self._voice_infos = self._build_voice_infos()

    def _build_voice_infos(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id=display_name,
                name=display_name,
                gender=self._VOICE_GENDERS.get(display_name),
                language="en",
                description="KittenTTS bundled English voice",
                styles=["neutral"],
                use_case=["general"],
            )
            for display_name in DEFAULT_VOICE_ALIASES
        ]

    def _resolve_model_name(self, request_model: Optional[str]) -> str:
        return normalize_repo_id(request_model or self.model_name or DEFAULT_REPO_ID)

    async def _load_runtime_for_model(self, model_name: str) -> KittenRuntime:
        async with self._runtime_lock:
            active_repo = getattr(self._runtime_assets, "repo_id", None)
            if self._runtime is not None and active_repo == model_name:
                return self._runtime

            try:
                configured_revision = self.model_revision if model_name == self.model_name else None
                resolved_revision = resolve_model_revision(model_name, configured_revision)
                assets = await asyncio.to_thread(
                    download_model_assets,
                    model_name,
                    cache_dir=self.cache_dir,
                    auto_download=self.auto_download,
                    revision=resolved_revision,
                )
                runtime = await asyncio.to_thread(KittenRuntime, assets)
            except Exception as exc:
                raise TTSProviderInitializationError(
                    "Failed to initialize KittenTTS runtime",
                    provider=self.PROVIDER_KEY,
                    details={"model": model_name, "error": str(exc)},
                ) from exc

            self._runtime_assets = assets
            self._runtime = runtime
            self.model_name = assets.repo_id
            self.model_revision = assets.revision
            self.sample_rate = int(getattr(runtime, "sample_rate", self.sample_rate))
            return runtime

    async def initialize(self) -> bool:
        """Initialize capabilities; load the request-selected model on demand."""
        logger.info("KittenTTS adapter initialized; model assets load on demand")
        return True

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="KittenTTS",
            supported_languages=self.SUPPORTED_LANGUAGES,
            supported_voices=list(self._voice_infos),
            supported_formats=self.SUPPORTED_FORMATS,
            max_text_length=self.MAX_TEXT_LENGTH,
            supports_streaming=True,
            supports_voice_cloning=False,
            supports_emotion_control=False,
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=True,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=1200,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        requested_model = self._resolve_model_name(request.model)
        validate_tts_request(request, provider=self.PROVIDER_KEY)
        if not self._initialized:
            runtime = await self._load_runtime_for_model(requested_model)
            self._initialized = True
            self.sample_rate = int(getattr(runtime, "sample_rate", self.sample_rate))
        else:
            if not await self.ensure_initialized():
                raise TTSProviderNotConfiguredError(
                    "KittenTTS adapter not initialized",
                    provider=self.PROVIDER_KEY,
                )
            runtime = await self._load_runtime_for_model(requested_model)
        clean_text = parse_bool(
            (request.extra_params or {}).get("clean_text"),
            default=self.clean_text,
        )

        try:
            audio = await asyncio.to_thread(
                runtime.generate,
                request.text,
                voice=request.voice,
                speed=float(request.speed),
                clean_text=clean_text,
            )
        except Exception as exc:
            raise TTSGenerationError(
                "KittenTTS generation failed",
                provider=self.PROVIDER_KEY,
                details={"model": requested_model, "error": str(exc)},
            ) from exc

        audio_np = np.asarray(audio)
        if audio_np.size == 0:
            raise TTSGenerationError(
                "KittenTTS returned no audio samples",
                provider=self.PROVIDER_KEY,
                details={"model": requested_model},
            )

        voice_used = request.voice or "Leo"

        if request.stream:
            return TTSResponse(
                audio_stream=self._stream_audio(audio_np, request.format),
                format=request.format,
                sample_rate=self.sample_rate,
                text_processed=request.text,
                voice_used=voice_used,
                provider=self.PROVIDER_KEY,
                model=requested_model,
            )

        audio_bytes = await self.convert_audio_format(
            audio_np,
            source_format=AudioFormat.PCM,
            target_format=request.format,
            sample_rate=self.sample_rate,
        )
        return TTSResponse(
            audio_data=audio_bytes,
            format=request.format,
            sample_rate=self.sample_rate,
            text_processed=request.text,
            voice_used=voice_used,
            provider=self.PROVIDER_KEY,
            model=requested_model,
        )

    async def _stream_audio(self, audio: np.ndarray, target_format: AudioFormat) -> AsyncGenerator[bytes, None]:
        writer = StreamingAudioWriter(
            format=target_format.value,
            sample_rate=self.sample_rate,
            channels=1,
        )
        try:
            normalized = self._audio_normalizer.normalize(audio, target_dtype=np.int16)
            chunk = writer.write_chunk(normalized)
            if chunk:
                yield chunk
            final_chunk = writer.write_chunk(finalize=True)
            if final_chunk:
                yield final_chunk
        finally:
            writer.close()

    async def _cleanup_resources(self) -> None:
        self._runtime = None
        self._runtime_assets = None

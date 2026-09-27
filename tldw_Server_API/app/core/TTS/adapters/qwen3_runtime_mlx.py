from __future__ import annotations

import asyncio
import importlib
import platform
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from ..tts_exceptions import TTSGenerationError, TTSProviderInitializationError, TTSValidationError
from .base import AudioFormat, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo

if TYPE_CHECKING:
    from .qwen3_tts_adapter import Qwen3TTSAdapter


class Qwen3MlxRuntime:
    runtime_name = "mlx"

    def __init__(self, adapter: "Qwen3TTSAdapter") -> None:
        self.adapter = adapter
        self._load_model = None
        self._model = None
        self._model_id: str | None = None
        self._model_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        if platform.system() != "Darwin" or platform.machine().lower() != "arm64":
            raise TTSProviderInitializationError(
                "Qwen3-TTS MLX runtime requires macOS on Apple Silicon",
                provider=self.adapter.PROVIDER_KEY,
            )
        try:
            module = importlib.import_module("mlx_audio.tts.utils")
            self._load_model = getattr(module, "load_model")
        except Exception as exc:
            raise TTSProviderInitializationError(
                "mlx-audio is required for the Qwen3-TTS MLX runtime",
                provider=self.adapter.PROVIDER_KEY,
            ) from exc
        return True

    async def get_capabilities(self) -> TTSCapabilities:
        max_text_length = self.adapter._coerce_int(self.adapter.config.get("max_text_length")) or 5000
        voices = [VoiceInfo(id=speaker, name=speaker) for speaker in self.adapter.CUSTOMVOICE_SPEAKERS]
        return TTSCapabilities(
            provider_name=self.adapter.provider_name,
            supported_languages=set(self.adapter.SUPPORTED_LANGUAGES),
            supported_voices=voices,
            supported_formats={
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.AAC,
                AudioFormat.WAV,
                AudioFormat.PCM,
            },
            max_text_length=max_text_length,
            supports_streaming=True,
            supports_voice_cloning=False,
            supports_emotion_control=False,
            sample_rate=self.adapter.sample_rate,
            default_format=AudioFormat.PCM,
            metadata={
                "runtime": self.runtime_name,
                "supported_modes": ["custom_voice_preset"],
                "supports_uploaded_custom_voices": False,
                "streaming_mode": "buffered_fallback",
            },
        )

    def validate_mode(self, request: TTSRequest, mode: str) -> None:
        if isinstance(request.voice, str) and request.voice.startswith("custom:"):
            raise TTSValidationError(
                "Uploaded custom voices are not supported by the MLX runtime in v1",
                provider=self.adapter.PROVIDER_KEY,
            )
        if mode in {"voice_clone", "voice_design"}:
            raise TTSValidationError(
                f"Mode '{mode}' is not supported by the MLX runtime",
                provider=self.adapter.PROVIDER_KEY,
            )

    def _resolve_backend_model_id(self, resolved_model: str) -> str:
        configured_path = str(self.adapter.config.get("model_path") or "").strip()
        if configured_path:
            return configured_path

        configured_model = str(self.adapter.config.get("mlx_model") or "").strip()
        if configured_model:
            return configured_model

        resolved_key = (resolved_model or "").strip().lower()
        if "1.7b" in resolved_key:
            return "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
        return "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"

    def _resolve_language(self, request: TTSRequest) -> str:
        language = self.adapter._resolve_language(request)
        mapping = {
            "auto": "English",
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "de": "German",
            "fr": "French",
            "ru": "Russian",
            "pt": "Portuguese",
            "es": "Spanish",
            "it": "Italian",
        }
        return mapping.get(language, language)

    def _resolve_speaker(self, request: TTSRequest) -> str:
        speaker = self.adapter._resolve_speaker(request.voice)
        if not speaker or speaker not in self.adapter.CUSTOMVOICE_SPEAKERS:
            raise TTSValidationError(
                "MLX runtime requires a preset Qwen3 speaker",
                provider=self.adapter.PROVIDER_KEY,
            )
        return speaker

    async def _get_model(self, resolved_model: str) -> tuple[object, str]:
        backend_model_id = self._resolve_backend_model_id(resolved_model)
        if self._model is not None and self._model_id == backend_model_id:
            return self._model, backend_model_id
        if self._load_model is None:
            raise TTSProviderInitializationError(
                "mlx-audio backend was not initialized",
                provider=self.adapter.PROVIDER_KEY,
            )
        async with self._model_lock:
            if self._model is None or self._model_id != backend_model_id:
                self._model = await asyncio.to_thread(self._load_model, backend_model_id)
                self._model_id = backend_model_id
        return self._model, backend_model_id

    def _run_generation(
        self,
        model: object,
        *,
        text: str,
        speaker: str,
        language: str,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        if not hasattr(model, "generate"):
            raise TTSGenerationError(
                "MLX runtime model does not expose generate()",
                provider=self.adapter.PROVIDER_KEY,
            )
        results = model.generate(
            text=text,
            voice=speaker,
            language=language,
            speed=speed,
        )
        return self._collect_pcm_audio(results)

    def _collect_pcm_audio(self, results: Iterable[object]) -> tuple[np.ndarray, int]:
        chunks: list[np.ndarray] = []
        sample_rate = self.adapter.sample_rate
        for result in results:
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            chunk = np.asarray(audio)
            if chunk.ndim > 1:
                chunk = chunk.reshape(-1)
            if chunk.size == 0:
                continue
            chunks.append(chunk)
            rate = getattr(result, "sample_rate", None) or getattr(result, "sampling_rate", None)
            if rate:
                sample_rate = int(rate)
        if not chunks:
            raise TTSGenerationError(
                "MLX runtime returned no audio chunks",
                provider=self.adapter.PROVIDER_KEY,
            )
        pcm = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        if pcm.dtype != np.int16:
            pcm = self.adapter._audio_normalizer.normalize(pcm, target_dtype=np.int16)
        return pcm, sample_rate

    async def generate(self, request: TTSRequest, resolved_model: str, mode: str) -> TTSResponse:
        self.validate_mode(request, mode)
        model, backend_model_id = await self._get_model(resolved_model)
        speaker = self._resolve_speaker(request)
        language = self._resolve_language(request)

        try:
            pcm_audio, sample_rate = await asyncio.to_thread(
                self._run_generation,
                model,
                text=request.text,
                speaker=speaker,
                language=language,
                speed=request.speed,
            )
        except Exception as exc:
            raise TTSGenerationError(
                "Qwen3-TTS MLX runtime generation failed",
                provider=self.adapter.PROVIDER_KEY,
                details={"runtime": self.runtime_name, "model": backend_model_id, "error": str(exc)},
            ) from exc

        audio_bytes = await self.adapter.convert_audio_format(
            pcm_audio,
            target_format=request.format,
            sample_rate=sample_rate,
        )

        metadata = {
            "runtime": self.runtime_name,
            "backend_model": backend_model_id,
        }
        if request.stream:
            metadata["streaming_fallback"] = "buffered"
            return TTSResponse(
                audio_stream=self.adapter._chunk_bytes(audio_bytes),
                format=request.format,
                sample_rate=sample_rate,
                provider=self.adapter.PROVIDER_KEY,
                model=resolved_model,
                metadata=metadata,
            )

        return TTSResponse(
            audio_content=audio_bytes,
            format=request.format,
            sample_rate=sample_rate,
            provider=self.adapter.PROVIDER_KEY,
            model=resolved_model,
            metadata=metadata,
        )

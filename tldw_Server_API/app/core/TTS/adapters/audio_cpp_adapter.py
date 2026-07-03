"""audio.cpp TTS provider adapter."""

from __future__ import annotations

import asyncio
import base64
import inspect
import math
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger

from ..tts_exceptions import (
    TTSGenerationError,
    TTSModelNotFoundError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSValidationError,
)
from .audio_cpp_client import AudioCppClient, AudioCppSpeechResult
from .audio_cpp_config import PROVIDER_KEY as AUDIO_CPP_PROVIDER_KEY
from .audio_cpp_config import AudioCppConfig, filter_request_options
from .audio_cpp_sidecar_supervisor import AudioCppSidecarSupervisor
from .base import (
    AudioFormat,
    ProviderStatus,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
    VoiceInfo,
)

_SUPPORTED_LANGUAGES = {"en"}
_SUPPORTED_FORMATS = {
    AudioFormat.WAV,
    AudioFormat.MP3,
    AudioFormat.OPUS,
    AudioFormat.FLAC,
    AudioFormat.AAC,
    AudioFormat.PCM,
}
_NAMESPACED_MODEL_PREFIXES = (
    "audio_cpp:",
    "audio-cpp:",
    "audiocpp:",
    "audio_cpp/",
    "audio-cpp/",
    "audiocpp/",
)
_CONTENT_TYPE_FORMATS = {
    "audio/wav": AudioFormat.WAV,
    "audio/x-wav": AudioFormat.WAV,
    "audio/mpeg": AudioFormat.MP3,
    "audio/mp3": AudioFormat.MP3,
    "audio/opus": AudioFormat.OPUS,
    "audio/flac": AudioFormat.FLAC,
    "audio/aac": AudioFormat.AAC,
    "audio/l16": AudioFormat.PCM,
    "application/octet-stream": AudioFormat.WAV,
}


class AudioCppTTSAdapter(TTSAdapter):
    """Registry-facing adapter for an audiocpp_server speech endpoint."""

    PROVIDER_KEY = AUDIO_CPP_PROVIDER_KEY
    SUPPORTED_FORMATS = _SUPPORTED_FORMATS

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        cfg = config or {}
        self._audio_cpp_config = AudioCppConfig.from_provider_config(cfg, repo_root=Path.cwd())
        self.sample_rate = int(cfg.get("sample_rate") or 24000)
        self.max_text_length = int(cfg.get("max_text_length") or 5000)
        self._client: Any | None = cfg.get("client") or cfg.get("_client")
        self._owns_client = self._client is None
        self._sidecar_supervisor: Any | None = cfg.get("sidecar_supervisor") or cfg.get("_sidecar_supervisor")
        self._owns_sidecar_supervisor = self._sidecar_supervisor is None
        self._available_models: list[str] = []
        self._voices = self._parse_voice_catalog(cfg)

    async def ensure_initialized(self) -> bool:
        """Propagate initialization errors so registry logs keep the root cause."""
        if self._initialized:
            return True

        async with self._init_lock:
            if self._initialized:
                return True

            self._status = ProviderStatus.INITIALIZING
            try:
                success = await self.initialize()
                if success:
                    self._capabilities = await self.get_capabilities()
                    self._status = ProviderStatus.AVAILABLE
                    self._initialized = True
                else:
                    self._status = ProviderStatus.ERROR
                return success
            except Exception:
                self._status = ProviderStatus.ERROR
                raise

    async def initialize(self) -> bool:
        if self._client is None:
            base_url = self._audio_cpp_config.base_url
            if self._audio_cpp_config.managed:
                if self._sidecar_supervisor is None:
                    self._sidecar_supervisor = AudioCppSidecarSupervisor(
                        self.config,
                        repo_root=self._audio_cpp_config.repo_root,
                    )
                    self._owns_sidecar_supervisor = True
                base_url = await self._sidecar_supervisor.ensure_started()
            self._client = AudioCppClient(
                base_url=base_url,
                timeout=float(self._audio_cpp_config.timeout),
                allow_remote_base_url=self._audio_cpp_config.allow_remote_base_url,
            )
            self._owns_client = True

        health = await self._client.health()
        if isinstance(health, dict):
            status = str(health.get("status") or health.get("state") or "ok").strip().lower()
            if status in {"error", "failed", "unhealthy"}:
                raise TTSProviderInitializationError(
                    "audio.cpp server health check failed",
                    provider=self.PROVIDER_KEY,
                    details={"status": status},
                )

        self._available_models = await self._client.list_models()
        self._validate_configured_model_available()
        logger.info("audio.cpp TTS adapter initialized model={}", self._configured_upstream_model())
        return True

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="audio.cpp",
            supported_languages=_SUPPORTED_LANGUAGES,
            supported_voices=list(self._voices.values()),
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
            metadata={
                "incremental_streaming": False,
                "native_streaming": False,
                "managed": self._audio_cpp_config.managed,
                "voice_reference_mode": self._voice_reference_mode_metadata(),
            },
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "audio.cpp adapter not initialized",
                provider=self.PROVIDER_KEY,
            )
        if self._client is None:
            raise TTSProviderInitializationError(
                "audio.cpp client is not available",
                provider=self.PROVIDER_KEY,
            )

        is_valid, error = await self.validate_request(request)
        if not is_valid:
            raise TTSValidationError(
                error or "audio.cpp request is invalid",
                provider=self.PROVIDER_KEY,
            )

        payload, ignored_options, voice_used, staged_reference = await self._build_payload(request)
        try:
            result = await self._client.speech(payload)
            return self._build_response(
                request=request,
                result=result,
                ignored_options=ignored_options,
                voice_used=voice_used,
            )
        finally:
            self._cleanup_reference(staged_reference)

    async def _cleanup_resources(self) -> None:
        client = self._client
        if client is not None and self._owns_client:
            close = getattr(client, "close", None)
            if callable(close):
                maybe_close = close()
                if inspect.isawaitable(maybe_close):
                    await maybe_close
        supervisor = self._sidecar_supervisor
        if supervisor is not None and self._owns_sidecar_supervisor:
            shutdown = getattr(supervisor, "shutdown", None)
            if callable(shutdown):
                maybe_shutdown = shutdown()
                if inspect.isawaitable(maybe_shutdown):
                    await maybe_shutdown

    def _parse_voice_catalog(self, config: dict[str, Any]) -> dict[str, VoiceInfo]:
        extras = config.get("extra_params") if isinstance(config.get("extra_params"), dict) else {}
        voices = extras.get("voices") if isinstance(extras, dict) else None
        if not isinstance(voices, dict):
            return {}

        catalog: dict[str, VoiceInfo] = {}
        for voice_id, raw_mapping in voices.items():
            normalized_id = str(voice_id).strip()
            if not normalized_id or not isinstance(raw_mapping, dict):
                continue
            catalog[normalized_id] = VoiceInfo(
                id=normalized_id,
                name=str(raw_mapping.get("name") or normalized_id),
                language=str(raw_mapping.get("language") or "en"),
                gender=raw_mapping.get("gender"),
                description=raw_mapping.get("description"),
            )
        return catalog

    def _voice_mapping(self, voice_id: str | None) -> dict[str, Any] | None:
        if not voice_id:
            return None
        extras = self.config.get("extra_params") if isinstance(self.config.get("extra_params"), dict) else {}
        voices = extras.get("voices") if isinstance(extras, dict) else None
        if not isinstance(voices, dict):
            return None
        mapping = voices.get(str(voice_id))
        return dict(mapping) if isinstance(mapping, dict) else None

    def _configured_upstream_model(self) -> str:
        server_model = self._audio_cpp_config.server.get("model")
        if isinstance(server_model, dict):
            model_id = str(server_model.get("id") or "").strip()
            if model_id:
                return model_id
        return self._strip_model_namespace(self._audio_cpp_config.model)

    def _resolve_upstream_model(self, request: TTSRequest) -> str:
        return self._strip_model_namespace(request.model or self._audio_cpp_config.model)

    @staticmethod
    def _strip_model_namespace(model: str | None) -> str:
        value = str(model or "").strip()
        lowered = value.lower()
        for prefix in _NAMESPACED_MODEL_PREFIXES:
            if lowered.startswith(prefix):
                return value[len(prefix):].strip() or "pocket-tts"
        return value or "pocket-tts"

    def _validate_configured_model_available(self) -> None:
        if not self._available_models:
            return
        configured_model = self._configured_upstream_model().lower()
        known_models = {
            self._strip_model_namespace(model).lower()
            for model in self._available_models
            if str(model).strip()
        }
        if configured_model not in known_models:
            raise TTSModelNotFoundError(
                "audio.cpp configured model is not listed by the server",
                provider=self.PROVIDER_KEY,
                details={"model": configured_model},
            )

    async def _build_payload(
        self,
        request: TTSRequest,
    ) -> tuple[dict[str, Any], dict[str, str], str | None, Path | None]:
        ignored_options: dict[str, str] = {}
        payload: dict[str, Any] = {
            "model": self._resolve_upstream_model(request),
            "input": self.preprocess_text(request.text),
        }

        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        filtered_options, ignored_extras = filter_request_options(
            extras,
            allowlist=self._audio_cpp_config.request_option_allowlist,
        )
        payload.update(filtered_options)
        ignored_options.update(ignored_extras)

        if request.speed is not None and not math.isclose(float(request.speed), 1.0, abs_tol=1e-12):
            ignored_options["speed"] = "unsupported"

        voice_used = request.voice
        self._apply_voice_mapping(request, payload)
        staged_reference = await self._stage_reference_audio(request.voice_reference)
        if staged_reference is not None:
            payload["voice_ref"] = str(staged_reference)
            voice_used = request.voice or "reference_audio"

        return payload, ignored_options, voice_used, staged_reference

    def _apply_voice_mapping(self, request: TTSRequest, payload: dict[str, Any]) -> None:
        mapping = self._voice_mapping(request.voice)
        if mapping is None:
            return

        request_field = mapping.get("request_field")
        if request_field is None:
            if not request.voice_reference:
                raise TTSValidationError(
                    f"audio.cpp voice '{request.voice}' is catalog metadata only; provide reference audio",
                    provider=self.PROVIDER_KEY,
                    details={"voice": request.voice},
                )
            return

        field_name = str(request_field).strip()
        if not field_name or field_name in {"voice", "voice_id"}:
            raise TTSValidationError(
                "audio.cpp voice mapping uses an unsupported request field",
                provider=self.PROVIDER_KEY,
                details={"voice": request.voice, "request_field": field_name},
            )
        payload[field_name] = mapping.get("upstream_value") or request.voice

    async def _stage_reference_audio(self, voice_reference: Any) -> Path | None:
        if voice_reference is None:
            return None
        if (
            not self._audio_cpp_config.managed
            and self._audio_cpp_config.external_voice_reference_mode != "shared_path"
        ):
            raise TTSValidationError(
                "audio.cpp voice_reference requires managed mode or external shared_path mode",
                provider=self.PROVIDER_KEY,
                details={"external_voice_reference_mode": self._audio_cpp_config.external_voice_reference_mode},
            )

        audio_bytes = self._extract_voice_reference_bytes(voice_reference)
        path = self._audio_cpp_config.build_reference_scratch_path()
        await asyncio.to_thread(self._write_reference_audio_sync, path, audio_bytes)
        return path

    @staticmethod
    def _write_reference_audio_sync(path: Path, audio_bytes: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio_bytes)

    def _extract_voice_reference_bytes(self, voice_reference: Any) -> bytes:
        if isinstance(voice_reference, bytes):
            return voice_reference
        if isinstance(voice_reference, bytearray):
            return bytes(voice_reference)
        if isinstance(voice_reference, str):
            try:
                return base64.b64decode(voice_reference, validate=True)
            except ValueError as exc:
                raise TTSValidationError(
                    "audio.cpp voice_reference must be bytes or base64 audio",
                    provider=self.PROVIDER_KEY,
                    details={"type": "str"},
                ) from exc
        raise TTSValidationError(
            "audio.cpp voice_reference must be bytes or base64 audio",
            provider=self.PROVIDER_KEY,
            details={"type": type(voice_reference).__name__},
        )

    def _cleanup_reference(self, staged_reference: Path | None) -> None:
        if staged_reference is None or self._audio_cpp_config.retain_request_artifacts:
            return
        with suppress(OSError, ValueError):
            staged_reference.unlink(missing_ok=True)

    def _build_response(
        self,
        *,
        request: TTSRequest,
        result: AudioCppSpeechResult,
        ignored_options: dict[str, str],
        voice_used: str | None,
    ) -> TTSResponse:
        if not result.audio_bytes:
            raise TTSGenerationError(
                "audio.cpp server returned empty audio",
                provider=self.PROVIDER_KEY,
                error_code="EMPTY_AUDIO",
            )

        metadata = dict(result.metadata or {})
        content_type = result.content_type or metadata.get("upstream_response_format") or "audio/wav"
        metadata.update(
            {
                "provider": self.PROVIDER_KEY,
                "model": request.model or self._audio_cpp_config.model,
                "managed": self._audio_cpp_config.managed,
                "incremental_streaming": False,
                "voice_reference_mode": self._voice_reference_mode_metadata(),
                "ignored_options": ignored_options,
                "upstream_response_format": content_type,
            }
        )

        return TTSResponse(
            audio_data=result.audio_bytes,
            audio_stream=None,
            format=self._resolve_response_format(content_type, metadata),
            sample_rate=self.sample_rate,
            channels=1,
            voice_used=voice_used,
            provider=self.PROVIDER_KEY,
            model=request.model or self._audio_cpp_config.model,
            metadata=metadata,
        )

    def _resolve_response_format(self, content_type: str, metadata: dict[str, Any]) -> AudioFormat:
        normalized = str(content_type or "").split(";", 1)[0].strip().lower()
        if normalized == "application/json":
            json_format = str(metadata.get("json_format") or "").strip().lower()
            if json_format:
                with suppress(ValueError):
                    return AudioFormat(json_format)
            return AudioFormat.WAV
        return _CONTENT_TYPE_FORMATS.get(normalized, AudioFormat.WAV)

    def _voice_reference_mode_metadata(self) -> str:
        if self._audio_cpp_config.managed:
            return "managed"
        return self._audio_cpp_config.external_voice_reference_mode

from __future__ import annotations

import contextlib
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ..tts_exceptions import (
    TTSGenerationError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSUnsupportedFormatError,
    TTSValidationError,
)
from ..tts_validation import validate_tts_request
from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse
from .omnivoice_sidecar_protocol import build_sidecar_auth_headers
from .omnivoice_sidecar_supervisor import create_sidecar_async_client


class OmniVoiceAdapter(TTSAdapter):
    """Thin OmniVoice adapter backed by the local sidecar supervisor."""

    PROVIDER_KEY = "omnivoice"
    SUPPORTED_FORMATS: frozenset[AudioFormat] = frozenset(
        {
            AudioFormat.MP3,
            AudioFormat.OPUS,
            AudioFormat.AAC,
            AudioFormat.FLAC,
            AudioFormat.WAV,
            AudioFormat.PCM,
        }
    )
    SUPPORTED_LANGUAGES = frozenset({"en"})
    DEFAULT_SAMPLE_RATE = 24000
    MAX_TEXT_LENGTH = 5000
    DEFAULT_TIMEOUT_SECONDS = 30.0
    VALID_MODES = frozenset({"auto", "clone"})

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        extras = cfg.get("extra_params", {}) or {}

        def _cfg_value(key: str, default: Any = None) -> Any:
            value = cfg.get(key)
            if value is None:
                value = extras.get(key)
            return default if value is None else value

        self.sample_rate = int(_cfg_value("sample_rate", self.DEFAULT_SAMPLE_RATE))
        self.timeout = float(_cfg_value("timeout", self.DEFAULT_TIMEOUT_SECONDS))
        temp_dir = _cfg_value("temp_dir")
        self.temp_dir = Path(temp_dir).expanduser() if temp_dir else None
        self._supervisor = _cfg_value("_supervisor")

    def set_supervisor(self, supervisor: Any) -> None:
        self._supervisor = supervisor

    async def initialize(self) -> bool:
        if self.sample_rate <= 0:
            raise TTSProviderInitializationError(
                "OmniVoice sample_rate must be a positive integer",
                provider=self.PROVIDER_KEY,
            )
        if self.timeout <= 0:
            raise TTSProviderInitializationError(
                "OmniVoice timeout must be positive",
                provider=self.PROVIDER_KEY,
            )

        if self._supervisor is None:
            self._status = ProviderStatus.NOT_CONFIGURED
            self._initialized = False
            logger.info("OmniVoice adapter not configured: sidecar supervisor is not attached")
            return False

        self._capabilities = await self.get_capabilities()
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        logger.info(
            "OmniVoice adapter initialized (sample_rate={}, timeout={})",
            self.sample_rate,
            self.timeout,
        )
        return True

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="OmniVoice",
            supported_languages=set(self.SUPPORTED_LANGUAGES),
            supported_voices=[],
            supported_formats=set(self.SUPPORTED_FORMATS),
            max_text_length=self.MAX_TEXT_LENGTH,
            supports_streaming=False,
            supports_voice_cloning=True,
            supports_emotion_control=False,
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=None,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "OmniVoice adapter not initialized",
                provider=self.PROVIDER_KEY,
            )

        if request.stream:
            raise TTSValidationError(
                "OmniVoice does not support streaming in v1",
                provider=self.PROVIDER_KEY,
            )

        if request.format not in self.SUPPORTED_FORMATS:
            raise TTSUnsupportedFormatError(
                f"Format {request.format.value} not supported by OmniVoice",
                provider=self.PROVIDER_KEY,
            )

        try:
            validate_tts_request(request, provider=self.PROVIDER_KEY)
        except TTSValidationError:
            raise
        except Exception as exc:
            raise TTSValidationError(
                f"Validation failed for OmniVoice request: {exc}",
                provider=self.PROVIDER_KEY,
            ) from exc

        supervisor = self._supervisor
        if supervisor is None:
            raise TTSProviderNotConfiguredError(
                "OmniVoice sidecar supervisor is not attached",
                provider=self.PROVIDER_KEY,
            )

        mode = self._resolve_mode(request)
        sample_rate = self._resolve_sample_rate(request)
        reference_audio_path = self._materialize_reference_audio(request) if request.voice_reference else None
        payload = self._build_sidecar_payload(
            request,
            mode=mode,
            sample_rate=sample_rate,
            reference_audio_path=reference_audio_path,
        )

        try:
            base_url = await supervisor.ensure_started()
            token = getattr(supervisor, "sidecar_token", None)
            headers = build_sidecar_auth_headers(token) if token else {}

            async with create_sidecar_async_client(timeout=self.timeout) as client:
                response = await client.post(
                    f"{str(base_url).rstrip('/')}/v1/synthesize",
                    json=payload,
                    headers=headers,
                )

            if response.status_code != 200:
                raise TTSGenerationError(
                    "OmniVoice sidecar returned an error",
                    provider=self.PROVIDER_KEY,
                    details={
                        "status_code": response.status_code,
                        "response_text": response.text,
                    },
                )

            sidecar_audio_format = (
                response.headers.get("X-OmniVoice-Audio-Format", "wav").strip().lower()
            )
            if sidecar_audio_format not in {"wav", "pcm"}:
                raise TTSGenerationError(
                    "OmniVoice sidecar returned an unsupported audio format",
                    provider=self.PROVIDER_KEY,
                    details={"audio_format": sidecar_audio_format},
                )

            audio_bytes = response.content
            if not audio_bytes:
                raise TTSGenerationError(
                    "OmniVoice sidecar returned empty audio",
                    provider=self.PROVIDER_KEY,
                )

            response_audio, response_format = self._normalize_sidecar_audio(
                audio_bytes,
                sidecar_audio_format=sidecar_audio_format,
                requested_format=request.format,
                sample_rate=sample_rate,
            )

            return TTSResponse(
                audio_data=response_audio,
                format=response_format,
                sample_rate=sample_rate,
                channels=int(response.headers.get("X-OmniVoice-Channels", "1") or "1"),
                text_processed=request.text,
                voice_used=request.voice,
                provider=self.PROVIDER_KEY,
                model=request.model or self.PROVIDER_KEY,
                metadata={
                    "transport": "sidecar",
                    "sidecar_mode": mode,
                    "sidecar_audio_format": sidecar_audio_format,
                    "reference_audio_path": str(reference_audio_path) if reference_audio_path else None,
                },
            )
        finally:
            if reference_audio_path is not None:
                with contextlib.suppress(OSError):
                    reference_audio_path.unlink()

    def _resolve_mode(self, request: TTSRequest) -> str:
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        explicit_mode = extras.get("omnivoice_mode", extras.get("mode"))
        if isinstance(explicit_mode, str):
            normalized = explicit_mode.strip().lower()
            if normalized in self.VALID_MODES:
                return normalized

        voice = (request.voice or "").strip().lower()
        if request.voice_reference is not None:
            return "clone"
        if voice == "clone" or voice.startswith("custom:"):
            return "clone"
        return "auto"

    def _resolve_sample_rate(self, request: TTSRequest) -> int:
        if request.target_sample_rate and int(request.target_sample_rate) > 0:
            return int(request.target_sample_rate)
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        for key in ("target_sample_rate", "sample_rate"):
            value = extras.get(key)
            if value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return self.sample_rate

    def _build_sidecar_payload(
        self,
        request: TTSRequest,
        *,
        mode: str,
        sample_rate: int,
        reference_audio_path: Optional[Path],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "text": self.preprocess_text(request.text),
            "mode": mode,
            "sample_rate": sample_rate,
        }
        if mode != "clone":
            voice = (request.voice or "").strip() or "auto"
            if voice and not voice.startswith("custom:") and voice.lower() != "clone":
                payload["voice"] = voice
        else:
            extras = request.extra_params if isinstance(request.extra_params, dict) else {}
            reference_text = (
                extras.get("reference_text")
                or extras.get("ref_text")
                or extras.get("voice_reference_text")
            )
            if reference_audio_path is not None:
                payload["reference_audio_path"] = str(reference_audio_path)
            if isinstance(reference_text, str) and reference_text.strip():
                payload["reference_text"] = reference_text.strip()
        return payload

    def _materialize_reference_audio(self, request: TTSRequest) -> Optional[Path]:
        if request.voice_reference is None:
            return None

        self._ensure_temp_dir()
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="omnivoice_ref_",
            dir=str(self.temp_dir) if self.temp_dir else None,
            delete=False,
        )
        temp_path = Path(temp_file.name)
        with temp_file:
            temp_file.write(request.voice_reference)
        return temp_path

    def _ensure_temp_dir(self) -> None:
        if self.temp_dir is None:
            return
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_sidecar_audio(
        self,
        audio_bytes: bytes,
        *,
        sidecar_audio_format: str,
        requested_format: AudioFormat,
        sample_rate: int,
    ) -> tuple[bytes, AudioFormat]:
        if requested_format == AudioFormat.PCM:
            if sidecar_audio_format == "pcm":
                return audio_bytes, AudioFormat.PCM
            return self._wav_to_pcm(audio_bytes), AudioFormat.PCM

        if requested_format == AudioFormat.WAV:
            if sidecar_audio_format == "wav":
                return audio_bytes, AudioFormat.WAV
            return self._pcm_to_wav(audio_bytes, sample_rate=sample_rate), AudioFormat.WAV

        if sidecar_audio_format == "wav":
            return audio_bytes, AudioFormat.WAV
        return self._pcm_to_wav(audio_bytes, sample_rate=sample_rate), AudioFormat.WAV

    def _wav_to_pcm(self, audio_bytes: bytes) -> bytes:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes())

    def _pcm_to_wav(self, audio_bytes: bytes, *, sample_rate: int) -> bytes:
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        return buffer.getvalue()

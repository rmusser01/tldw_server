"""OmniVoice TTS adapter backed by the local authenticated sidecar.

The adapter validates public TTS requests, materializes optional clone-mode
reference audio, delegates synthesis to the sidecar supervisor, and normalizes
the returned audio format before handing responses back to the TTS service.
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from ..audio_converter import AudioConverter
from ..tts_exceptions import (
    TTSGenerationError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSUnsupportedFormatError,
    TTSValidationError,
)
from ..tts_validation import validate_tts_request
from ..utils import parse_bool
from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse
from .omnivoice_sidecar_protocol import build_sidecar_auth_headers
from .omnivoice_sidecar_supervisor import create_sidecar_async_client

GENERATION_PARAM_TYPES = {
    "num_step": int,
    "guidance_scale": float,
    "denoise": bool,
    "t_shift": float,
    "position_temperature": float,
    "class_temperature": float,
    "layer_penalty_factor": float,
    "duration": float,
    "speed": float,
    "postprocess_output": bool,
    "preprocess_prompt": bool,
    "audio_chunk_duration": float,
    "audio_chunk_threshold": float,
}
INSTRUCT_KEYS = ("instruct", "voice_design", "voice_description")
LANGUAGE_KEYS = ("language_id", "language")
REFERENCE_TEXT_KEYS = ("reference_text", "ref_text", "voice_reference_text")
UNSUPPORTED_OMNIVOICE_KEYS = {"omnivoice_temperature", "omnivoice_top_p", "omnivoice_seed"}
AVAILABILITY_ERROR_CODES = {"MODEL_NOT_AVAILABLE", "RUNTIME_IMPORT_FAILED", "MODEL_LOAD_FAILED"}
VALIDATION_ERROR_CODES = {
    "INVALID_REFERENCE_AUDIO",
    "INVALID_GENERATION_PARAMETER",
    "REFERENCE_PATH_NOT_ALLOWED",
}


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
    VALID_MODES = frozenset({"auto", "design", "clone"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
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
        scratch_dir = _cfg_value("scratch_dir")
        self.scratch_dir = Path(scratch_dir).expanduser() if scratch_dir else None
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

        reference_audio_path = await self._materialize_reference_audio(request) if request.voice_reference else None
        mode = self._resolve_mode(request, reference_audio_path=reference_audio_path)
        requested_sample_rate = self._resolve_sample_rate(request)
        payload = self._build_sidecar_payload(
            request,
            mode=mode,
            sample_rate=requested_sample_rate,
            reference_audio_path=reference_audio_path,
        )

        try:
            base_url = await supervisor.ensure_started()
            token = getattr(supervisor, "sidecar_token", None)
            headers = build_sidecar_auth_headers(token) if token else {}
            channels = 1
            get_http_client = getattr(supervisor, "get_http_client", None)
            if callable(get_http_client):
                client = await get_http_client()
                response = await client.post(
                    f"{str(base_url).rstrip('/')}/v1/synthesize",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            else:
                async with create_sidecar_async_client(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{str(base_url).rstrip('/')}/v1/synthesize",
                        json=payload,
                        headers=headers,
                    )

            if response.status_code != 200:
                self._raise_for_sidecar_error(response)

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
            channels = int(response.headers.get("X-OmniVoice-Channels", "1") or "1")
            native_sample_rate = self._parse_sample_rate_header(
                response.headers.get("X-OmniVoice-Sample-Rate")
            )
            if not audio_bytes:
                raise TTSGenerationError(
                    "OmniVoice sidecar returned empty audio",
                    provider=self.PROVIDER_KEY,
                )

            response_audio, response_format = await self._normalize_sidecar_audio(
                audio_bytes,
                sidecar_audio_format=sidecar_audio_format,
                requested_format=request.format,
                sample_rate=native_sample_rate,
                channels=channels,
            )

            return TTSResponse(
                audio_data=response_audio,
                format=response_format,
                sample_rate=native_sample_rate,
                channels=channels,
                text_processed=request.text,
                voice_used=request.voice,
                provider=self.PROVIDER_KEY,
                model=request.model or self.PROVIDER_KEY,
                metadata={
                    "transport": "sidecar",
                    "sidecar_mode": mode,
                    "sidecar_audio_format": sidecar_audio_format,
                    "sidecar_native_sample_rate": native_sample_rate,
                    "requested_sample_rate": requested_sample_rate,
                    "used_reference_audio": reference_audio_path is not None,
                },
            )
        finally:
            if reference_audio_path is not None:
                with contextlib.suppress(OSError):
                    reference_audio_path.unlink()

    def _resolve_mode(self, request: TTSRequest, *, reference_audio_path: Optional[Path] = None) -> str:
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        explicit_mode = extras.get("omnivoice_mode", extras.get("mode"))
        normalized_mode = None
        if isinstance(explicit_mode, str):
            normalized = explicit_mode.strip().lower()
            if normalized in self.VALID_MODES:
                normalized_mode = normalized
            elif normalized:
                raise TTSValidationError(
                    "OmniVoice mode must be one of auto, design, clone",
                    provider=self.PROVIDER_KEY,
                )
        elif explicit_mode is not None:
            raise TTSValidationError("OmniVoice mode must be a string", provider=self.PROVIDER_KEY)

        voice = (request.voice or "").strip().lower()
        instruct = self._resolve_instruct(extras)
        clone_requested = bool(reference_audio_path) or request.voice_reference is not None
        clone_requested = clone_requested or voice == "clone" or voice.startswith("custom:")
        design_requested = instruct is not None
        if clone_requested and design_requested:
            raise TTSValidationError(
                "OmniVoice design instruct cannot be combined with clone reference audio",
                provider=self.PROVIDER_KEY,
            )
        inferred = "clone" if clone_requested else "design" if design_requested else "auto"
        if normalized_mode is not None:
            if normalized_mode == "auto" and inferred != "auto":
                raise TTSValidationError(
                    "OmniVoice mode=auto conflicts with design or clone inputs",
                    provider=self.PROVIDER_KEY,
                )
            if normalized_mode == "design" and clone_requested:
                raise TTSValidationError("OmniVoice mode=design conflicts with clone inputs", provider=self.PROVIDER_KEY)
            if normalized_mode == "design" and instruct is None:
                raise TTSValidationError("OmniVoice mode=design requires instruct", provider=self.PROVIDER_KEY)
            if normalized_mode == "clone" and design_requested:
                raise TTSValidationError("OmniVoice mode=clone conflicts with instruct", provider=self.PROVIDER_KEY)
            if normalized_mode == "clone" and not clone_requested:
                raise TTSValidationError(
                    "OmniVoice mode=clone requires reference audio",
                    provider=self.PROVIDER_KEY,
                )
            return normalized_mode
        return inferred

    def _resolve_instruct(self, extras: dict[str, Any]) -> Optional[str]:
        values: list[tuple[str, str]] = []
        for key in INSTRUCT_KEYS:
            value = extras.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TTSValidationError(f"OmniVoice {key} must be a string", provider=self.PROVIDER_KEY)
            stripped = value.strip()
            if stripped:
                values.append((key, stripped))
        unique = {value for _, value in values}
        if len(unique) > 1:
            raise TTSValidationError(
                "Conflicting OmniVoice instruct aliases provided",
                provider=self.PROVIDER_KEY,
                details={"aliases": [key for key, _ in values]},
            )
        return values[0][1] if values else None

    def _resolve_language_id(self, request: TTSRequest, extras: dict[str, Any]) -> Optional[str]:
        values: list[tuple[str, str]] = []
        for key in LANGUAGE_KEYS:
            value = extras.get(key)
            if value is None:
                continue
            stripped = str(value).strip()
            if stripped:
                values.append((key, stripped))
        request_language = getattr(request, "language", None)
        if request_language is not None:
            stripped = str(request_language).strip()
            if stripped and stripped.lower() != "en":
                values.append(("request.language", stripped))
        unique = {value.lower() for _, value in values}
        if len(unique) > 1:
            raise TTSValidationError(
                "Conflicting OmniVoice language aliases provided",
                provider=self.PROVIDER_KEY,
                details={"aliases": [key for key, _ in values]},
            )
        return values[0][1] if values else None

    def _resolve_generation(self, extras: dict[str, Any]) -> dict[str, Any]:
        generation: dict[str, Any] = {}
        for key, value in extras.items():
            if key in GENERATION_PARAM_TYPES:
                generation[key] = self._coerce_generation_value(key, value, GENERATION_PARAM_TYPES[key])
                continue
            if (
                key.startswith("omnivoice_")
                and key not in {"omnivoice_mode"}
            ) or key in UNSUPPORTED_OMNIVOICE_KEYS:
                raise TTSValidationError(
                    f"Unknown or unsupported OmniVoice generation parameter: {key}",
                    provider=self.PROVIDER_KEY,
                )
        return generation

    def _coerce_generation_value(self, key: str, value: Any, target_type: type) -> Any:
        try:
            if target_type is bool:
                return parse_bool(value, default=False)
            if target_type is int:
                return int(value)
            if target_type is float:
                return float(value)
        except (TypeError, ValueError) as exc:
            raise TTSValidationError(
                f"OmniVoice generation parameter {key} has invalid type",
                provider=self.PROVIDER_KEY,
            ) from exc
        return value

    def _resolve_reference_text(self, extras: dict[str, Any]) -> Optional[str]:
        for key in REFERENCE_TEXT_KEYS:
            value = extras.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

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
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        payload: dict[str, Any] = {
            "text": self.preprocess_text(request.text),
            "mode": mode,
            "requested_sample_rate": sample_rate,
            "generation": self._resolve_generation(extras),
        }
        instruct = self._resolve_instruct(extras)
        language_id = self._resolve_language_id(request, extras)
        if mode == "design" and instruct:
            payload["instruct"] = instruct
        if language_id:
            payload["language_id"] = language_id
        if mode != "clone":
            voice = (request.voice or "").strip() or "auto"
            if voice and not voice.startswith("custom:") and voice.lower() != "clone":
                payload["voice"] = voice
        else:
            reference_text = self._resolve_reference_text(extras)
            if reference_audio_path is not None:
                payload["reference_audio_path"] = str(reference_audio_path)
            if reference_text:
                payload["reference_text"] = reference_text
        return payload

    async def _materialize_reference_audio(self, request: TTSRequest) -> Optional[Path]:
        if request.voice_reference is None:
            return None
        return await asyncio.to_thread(self._materialize_reference_audio_sync, request.voice_reference)

    def _materialize_reference_audio_sync(self, voice_reference: bytes) -> Path:
        target_dir = self.scratch_dir or self.temp_dir
        if target_dir is not None:
            target_dir.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="omnivoice_ref_",
            dir=str(target_dir) if target_dir else None,
            delete=False,
        )
        temp_path = Path(temp_file.name)
        with temp_file:
            temp_file.write(voice_reference)
        return temp_path

    def _write_temp_audio_file_sync(self, audio_data: bytes, *, suffix: str) -> Path:
        self._ensure_temp_dir()
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            dir=str(self.temp_dir) if self.temp_dir else None,
            delete=False,
        ) as handle:
            handle.write(audio_data)
            return Path(handle.name)

    def _reserve_temp_audio_path_sync(self, *, suffix: str) -> Path:
        self._ensure_temp_dir()
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            dir=str(self.temp_dir) if self.temp_dir else None,
            delete=False,
        ) as handle:
            return Path(handle.name)

    @staticmethod
    def _remove_temp_path_sync(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.unlink()

    @staticmethod
    def _sanitize_sidecar_error_text(response_text: str | None) -> str:
        if str(response_text or "").strip():
            return "OmniVoice sidecar reported an internal error; see server logs."
        return "OmniVoice sidecar returned an empty error response."

    @staticmethod
    def _sanitize_structured_sidecar_message(message: str | None) -> str:
        sanitized = str(message or "").strip()
        if not sanitized:
            return "OmniVoice sidecar returned an empty error response."
        sanitized = re.sub(r"[\x00-\x1f\x7f]+", " ", sanitized)
        sanitized = re.sub(r"\b(?:https?|file)://[^\s<>'\"]+", "[redacted-url]", sanitized)
        sanitized = re.sub(r"\b[A-Za-z]:\\[^\s:;,)\]}]+(?:\\[^\s:;,)\]}]+)*", "[redacted-path]", sanitized)
        sanitized = re.sub(r"(?<!\w)~(?:/[^\s:;,)\]}]+)+", "[redacted-path]", sanitized)
        sanitized = re.sub(r"(?<!\w)/(?:[^\s/:;,)\]}]+/)+[^\s:;,)\]}]+", "[redacted-path]", sanitized)
        sanitized = re.sub(
            r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]+",
            lambda match: f"{match.group(1)}=[redacted-secret]",
            sanitized,
        )
        sanitized = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [redacted-token]", sanitized)
        sanitized = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[redacted-token]", sanitized)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        return sanitized[:500] if sanitized else "OmniVoice sidecar returned an empty error response."

    def _parse_sample_rate_header(self, value: str | None) -> int:
        try:
            sample_rate = int(value or self.DEFAULT_SAMPLE_RATE)
        except (TypeError, ValueError):
            sample_rate = self.DEFAULT_SAMPLE_RATE
        return sample_rate if sample_rate > 0 else self.DEFAULT_SAMPLE_RATE

    def _raise_for_sidecar_error(self, response: Any) -> None:
        content_type = response.headers.get("content-type", "")
        payload: dict[str, Any] = {}
        if content_type.lower().startswith("application/json"):
            with contextlib.suppress(ValueError, TypeError):
                parsed = response.json()
                if isinstance(parsed, dict):
                    payload = parsed
        error = payload.get("error") if isinstance(payload, dict) else None
        code = None
        message = None
        retryable = False
        if isinstance(error, dict):
            raw_code = error.get("code")
            code = str(raw_code).strip() if raw_code is not None else None
            raw_message = error.get("message")
            message = str(raw_message).strip() if raw_message is not None else None
            retryable = bool(error.get("retryable", False))
        details = {
            "status_code": response.status_code,
            "response_text": self._sanitize_sidecar_error_text(getattr(response, "text", None)),
        }
        if code:
            details["sidecar_error_code"] = code
            details["retryable"] = retryable
        if message:
            details["sidecar_error_message"] = self._sanitize_structured_sidecar_message(message)
        logger.warning(
            "OmniVoice sidecar returned status {} with sanitized error code {}",
            response.status_code,
            code or "unknown",
        )
        if code in AVAILABILITY_ERROR_CODES:
            raise TTSProviderNotConfiguredError(
                "OmniVoice sidecar runtime is not available",
                provider=self.PROVIDER_KEY,
                error_code=code,
                details=details,
            )
        if code in VALIDATION_ERROR_CODES:
            raise TTSValidationError(
                "OmniVoice sidecar rejected the request",
                provider=self.PROVIDER_KEY,
                error_code=code,
                details=details,
            )
        raise TTSGenerationError(
            "OmniVoice sidecar returned an error",
            provider=self.PROVIDER_KEY,
            error_code=code,
            details=details,
        )

    def _ensure_temp_dir(self) -> None:
        if self.temp_dir is None:
            return
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def _normalize_sidecar_audio(
        self,
        audio_bytes: bytes,
        *,
        sidecar_audio_format: str,
        requested_format: AudioFormat,
        sample_rate: int,
        channels: int,
    ) -> tuple[bytes, AudioFormat]:
        if requested_format == AudioFormat.PCM:
            if sidecar_audio_format == "pcm":
                return audio_bytes, AudioFormat.PCM
            return self._wav_to_pcm(audio_bytes), AudioFormat.PCM

        wav_audio = (
            audio_bytes
            if sidecar_audio_format == "wav"
            else self._pcm_to_wav(audio_bytes, sample_rate=sample_rate, channels=channels)
        )
        if requested_format == AudioFormat.WAV:
            return wav_audio, AudioFormat.WAV

        return await self._convert_wav_to_requested_format(
            wav_audio,
            requested_format=requested_format,
            sample_rate=sample_rate,
            channels=channels,
        )

    async def _convert_wav_to_requested_format(
        self,
        audio_bytes: bytes,
        *,
        requested_format: AudioFormat,
        sample_rate: int,
        channels: int,
    ) -> tuple[bytes, AudioFormat]:
        input_path = None
        output_path = None
        try:
            input_path = await asyncio.to_thread(
                self._write_temp_audio_file_sync,
                audio_bytes,
                suffix=".wav",
            )
            output_path = await asyncio.to_thread(
                self._reserve_temp_audio_path_sync,
                suffix=f".{requested_format.value}",
            )
            converted = await AudioConverter.convert_format(
                input_path,
                output_path,
                requested_format.value,
                sample_rate=sample_rate,
                channels=channels,
            )
            if not converted or output_path is None or not output_path.exists():
                raise TTSGenerationError(
                    "OmniVoice audio conversion failed",
                    provider=self.PROVIDER_KEY,
                    details={"target_format": requested_format.value},
                )

            converted_bytes = await asyncio.to_thread(output_path.read_bytes)
            return converted_bytes, requested_format
        finally:
            if input_path is not None:
                await asyncio.to_thread(self._remove_temp_path_sync, input_path)
            if output_path is not None:
                await asyncio.to_thread(self._remove_temp_path_sync, output_path)

    def _wav_to_pcm(self, audio_bytes: bytes) -> bytes:
        with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes())

    def _pcm_to_wav(self, audio_bytes: bytes, *, sample_rate: int, channels: int) -> bytes:
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(max(1, channels))
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        return buffer.getvalue()

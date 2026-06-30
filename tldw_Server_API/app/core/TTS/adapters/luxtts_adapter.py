# luxtts_adapter.py
# Description: LuxTTS adapter implementation (ZipVoice-based voice cloning)
#
# Imports
import asyncio
import base64
import importlib
import sys
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Optional

#
# Third-party Imports
import numpy as np
from loguru import logger

from ..streaming_audio_writer import AudioNormalizer, StreamingAudioWriter
from ..tts_exceptions import (
    TTSGenerationError,
    TTSInvalidVoiceReferenceError,
    TTSModelLoadError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSUnsupportedFormatError,
    TTSValidationError,
)
from ..tts_validation import validate_tts_request
from ..utils import parse_bool

#
# Local Imports
from .base import (
    AudioFormat,
    ProviderStatus,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
)

#
#######################################################################################################################
#
# LuxTTS Adapter


def _safe_exception_label(exc: BaseException) -> str:
    """Return a non-sensitive exception identifier for logs."""
    return type(exc).__name__


class LuxTTSAdapter(TTSAdapter):
    """Adapter for LuxTTS (ZipVoice-based voice cloning)."""

    PROVIDER_KEY = "lux_tts"
    SUPPORTED_FORMATS: set[AudioFormat] = {
        AudioFormat.MP3,
        AudioFormat.WAV,
        AudioFormat.FLAC,
        AudioFormat.OPUS,
        AudioFormat.AAC,
        AudioFormat.PCM,
    }
    SUPPORTED_LANGUAGES = {"en"}
    DEFAULT_SAMPLE_RATE = 48000
    DEFAULT_PROMPT_SAMPLE_RATE = 24000
    MAX_TEXT_LENGTH = 5000

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        cfg_extras = cfg.get("extra_params") if isinstance(cfg.get("extra_params"), dict) else {}

        self.model_path = (
            cfg.get("lux_tts_model")
            or cfg.get("model")
            or cfg.get("model_path")
            or "YatharthS/LuxTTS"
        )
        self.module_path = Path(
            cfg.get("lux_tts_module_path")
            or cfg.get("module_path")
            or "LuxTTS"
        ).expanduser()

        self.device_pref = str(
            cfg.get("lux_tts_device")
            or cfg.get("device")
            or "auto"
        ).lower()
        self.device = self._resolve_device(self.device_pref)
        self.threads = self._coerce_int(
            cfg.get("lux_tts_threads") or cfg.get("threads"),
            default=4,
        )

        self.sample_rate = self._coerce_int(
            cfg.get("lux_tts_sample_rate") or cfg.get("sample_rate"),
            default=self.DEFAULT_SAMPLE_RATE,
        )
        self.reference_sample_rate = self._coerce_int(
            cfg.get("lux_tts_reference_sample_rate") or cfg.get("reference_sample_rate"),
            default=self.DEFAULT_PROMPT_SAMPLE_RATE,
        )

        self.prompt_duration = self._coerce_float(
            cfg.get("lux_tts_prompt_duration")
            or cfg.get("prompt_duration")
            or cfg.get("reference_duration")
            or cfg_extras.get("prompt_duration")
            or cfg_extras.get("reference_duration")
            or cfg_extras.get("duration"),
            default=5.0,
        )
        self.prompt_rms = self._coerce_float(
            cfg.get("lux_tts_prompt_rms")
            or cfg.get("prompt_rms")
            or cfg.get("rms")
            or cfg_extras.get("prompt_rms")
            or cfg_extras.get("rms"),
            default=0.001,
        )
        self.num_steps = self._coerce_int(
            cfg.get("lux_tts_num_steps")
            or cfg.get("num_steps")
            or cfg_extras.get("num_steps"),
            default=4,
        )
        self.guidance_scale = self._coerce_float(
            cfg.get("lux_tts_guidance_scale")
            or cfg.get("guidance_scale")
            or cfg.get("cfg_scale")
            or cfg_extras.get("guidance_scale")
            or cfg_extras.get("cfg_scale"),
            default=3.0,
        )
        self.t_shift = self._coerce_float(
            cfg.get("lux_tts_t_shift")
            or cfg.get("t_shift")
            or cfg_extras.get("t_shift"),
            default=0.5,
        )
        self.return_smooth = parse_bool(
            self._first_non_null(
                cfg.get("lux_tts_return_smooth"),
                cfg.get("return_smooth"),
                cfg_extras.get("return_smooth"),
            ),
            default=False,
        )
        self.validate_reference = parse_bool(
            self._first_non_null(
                cfg.get("lux_tts_validate_reference"),
                cfg.get("validate_reference"),
                cfg_extras.get("validate_reference"),
            ),
            default=True,
        )
        self.convert_reference = parse_bool(
            self._first_non_null(
                cfg.get("lux_tts_convert_reference"),
                cfg.get("convert_reference"),
                cfg_extras.get("convert_reference"),
            ),
            default=True,
        )
        self.stream_chunk_samples = self._coerce_int(
            cfg.get("lux_tts_stream_chunk_samples")
            or cfg.get("stream_chunk_samples")
            or cfg_extras.get("stream_chunk_samples"),
            default=8192,
        )

        self._engine: Optional[Any] = None
        self._engine_lock = asyncio.Lock()
        self._audio_normalizer = AudioNormalizer()

    async def initialize(self) -> bool:
        """Initialize LuxTTS (load models)."""
        if self._initialized:
            return True

        LuxTTS = self._load_engine_class()
        try:
            engine = await asyncio.to_thread(
                LuxTTS,
                self.model_path,
                device=self.device,
                threads=self.threads,
            )
        except Exception as exc:
            self._status = ProviderStatus.ERROR
            raise TTSProviderInitializationError(
                "Failed to initialize LuxTTS",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

        self._engine = engine
        try:
            from ..tts_resource_manager import get_resource_manager
            resource_manager = await get_resource_manager()
            register_result = resource_manager.register_model(
                provider=self.PROVIDER_KEY,
                model_instance=self._engine,
                cleanup_callback=self._cleanup_resources,
                model_key=str(self.model_path),
            )
            if asyncio.iscoroutine(register_result):
                await register_result
        except Exception as registration_error:
            logger.debug(
                "LuxTTS provider registration failed; continuing; exception_type={}",
                _safe_exception_label(registration_error),
            )
        self._capabilities = await self.get_capabilities()
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        logger.info(
            'LuxTTS adapter initialized (device={}, model={})',
            self.device,
            self.model_path,
        )
        return True

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="LuxTTS",
            supported_languages=self.SUPPORTED_LANGUAGES,
            supported_voices=[],
            supported_formats=self.SUPPORTED_FORMATS,
            max_text_length=self.MAX_TEXT_LENGTH,
            supports_streaming=True,
            supports_voice_cloning=True,
            supports_emotion_control=False,
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=1500,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "LuxTTS adapter not initialized",
                provider=self.PROVIDER_KEY,
            )

        if request.format not in self.SUPPORTED_FORMATS:
            raise TTSUnsupportedFormatError(
                f"Format {request.format.value} not supported by LuxTTS",
                provider=self.PROVIDER_KEY,
            )

        try:
            validate_tts_request(request, provider=self.PROVIDER_KEY, config=self.config)
        except TTSValidationError:
            raise
        except Exception as exc:
            raise TTSValidationError(
                f"Validation failed for LuxTTS request: {exc}",
                provider=self.PROVIDER_KEY,
            ) from exc

        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        voice_bytes = self._extract_voice_reference(request.voice_reference)
        voice_bytes = await self._prepare_voice_reference(voice_bytes, extras)
        voice_path = self._write_temp_audio(voice_bytes)

        prompt_duration = self._coerce_float(
            extras.get("prompt_duration")
            or extras.get("reference_duration")
            or extras.get("duration"),
            default=self.prompt_duration,
        )
        prompt_rms = self._coerce_float(
            extras.get("prompt_rms")
            or extras.get("rms"),
            default=self.prompt_rms,
        )
        num_steps = self._coerce_int(extras.get("num_steps"), default=self.num_steps)
        guidance_scale = self._coerce_float(
            extras.get("guidance_scale") or extras.get("cfg_scale"),
            default=self.guidance_scale,
        )
        t_shift = self._coerce_float(extras.get("t_shift"), default=self.t_shift)
        return_smooth = parse_bool(extras.get("return_smooth"), default=self.return_smooth)

        try:
            async with self._engine_lock:
                encode_dict = await asyncio.to_thread(
                    self._engine.encode_prompt,  # type: ignore[union-attr]
                    voice_path,
                    duration=prompt_duration,
                    rms=prompt_rms,
                )
                wav = await asyncio.to_thread(
                    self._engine.generate_speech,  # type: ignore[union-attr]
                    request.text,
                    encode_dict,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    t_shift=t_shift,
                    speed=request.speed,
                    return_smooth=return_smooth,
                )

            audio_np = self._coerce_audio_array(wav)
            audio_i16 = self._audio_normalizer.normalize(audio_np, target_dtype=np.int16)
            sample_rate = self._resolve_output_sample_rate(return_smooth, extras)

            if request.stream:
                return TTSResponse(
                    audio_stream=self._stream_audio(audio_i16, request, sample_rate),
                    format=request.format,
                    sample_rate=sample_rate,
                    channels=1,
                    text_processed=request.text,
                    voice_used=request.voice or "clone",
                    provider=self.PROVIDER_KEY,
                    model=request.model or "lux_tts",
                    metadata={
                        "prompt_duration": prompt_duration,
                        "prompt_rms": prompt_rms,
                        "num_steps": num_steps,
                        "guidance_scale": guidance_scale,
                        "t_shift": t_shift,
                        "return_smooth": return_smooth,
                    },
                )

            audio_bytes = await self.convert_audio_format(
                audio_i16,
                source_format=AudioFormat.PCM,
                target_format=request.format,
                sample_rate=sample_rate,
            )
            return TTSResponse(
                audio_data=audio_bytes,
                format=request.format,
                sample_rate=sample_rate,
                channels=1,
                text_processed=request.text,
                voice_used=request.voice or "clone",
                provider=self.PROVIDER_KEY,
                model=request.model or "lux_tts",
                metadata={
                    "prompt_duration": prompt_duration,
                    "prompt_rms": prompt_rms,
                    "num_steps": num_steps,
                    "guidance_scale": guidance_scale,
                    "t_shift": t_shift,
                    "return_smooth": return_smooth,
                },
            )
        except Exception as exc:
            logger.error(
                "LuxTTS generation failed; exception_type={}",
                _safe_exception_label(exc),
            )
            raise TTSGenerationError(
                "LuxTTS generation failed",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc
        finally:
            Path(voice_path).unlink(missing_ok=True)

    async def _cleanup_resources(self) -> None:
        self._engine = None

    def _load_engine_class(self):
        try:
            from zipvoice.luxvoice import LuxTTS  # type: ignore
            return LuxTTS
        except ImportError as exc:
            module_path = self.module_path
            module_dir = module_path if module_path.is_dir() else module_path.parent
            if module_dir.exists():
                module_path_str = str(module_dir)
                if module_path_str not in sys.path:
                    sys.path.insert(0, module_path_str)
                try:
                    module = importlib.import_module("zipvoice.luxvoice")
                    engine_cls = getattr(module, "LuxTTS", None)
                    if engine_cls is None:
                        raise AttributeError("LuxTTS not found in zipvoice.luxvoice")
                    return engine_cls
                except Exception as inner_exc:
                    raise TTSModelLoadError(
                        "LuxTTS module could not be imported",
                        provider=self.PROVIDER_KEY,
                        details={"error": str(inner_exc)},
                    ) from inner_exc
            raise TTSModelLoadError(
                "LuxTTS package not installed",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc), "suggestion": "Install LuxTTS (pip install -r requirements.txt)"},
            ) from exc

    def _resolve_device(self, device_pref: str) -> str:
        if device_pref in {"cpu", "cuda"}:
            if device_pref == "cuda" and not self._cuda_available():
                return "cpu"
            return device_pref
        if device_pref in {"auto", "gpu", "cuda_auto"}:
            return "cuda" if self._cuda_available() else "cpu"
        if device_pref in {"mps", "metal"}:
            return "cpu"
        return "cuda" if self._cuda_available() else "cpu"

    def _cuda_available(self) -> bool:
        try:
            import torch  # type: ignore
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _extract_voice_reference(self, voice_reference: Any) -> bytes:
        if voice_reference is None:
            raise TTSInvalidVoiceReferenceError(
                "LuxTTS requires voice_reference audio bytes",
                provider=self.PROVIDER_KEY,
            )

        if isinstance(voice_reference, (bytes, bytearray)):
            return bytes(voice_reference)

        if isinstance(voice_reference, str):
            try:
                # Support data URL or raw base64
                if "," in voice_reference:
                    voice_reference = voice_reference.split(",", 1)[1]
                return base64.b64decode(voice_reference)
            except Exception as exc:
                raise TTSInvalidVoiceReferenceError(
                    "LuxTTS voice_reference is not valid base64",
                    provider=self.PROVIDER_KEY,
                    details={"error": str(exc)},
                ) from exc

        raise TTSInvalidVoiceReferenceError(
            "LuxTTS voice_reference must be bytes or base64 string",
            provider=self.PROVIDER_KEY,
            details={"type": type(voice_reference).__name__},
        )

    async def _prepare_voice_reference(self, voice_bytes: bytes, extras: dict[str, Any]) -> bytes:
        validate_ref = self._resolve_bool_setting(
            extras,
            ("validate_reference", "validate_voice_reference"),
            default=self.validate_reference,
        )
        convert_ref = self._resolve_bool_setting(
            extras,
            ("convert_reference", "convert_voice_reference"),
            default=self.convert_reference,
        )

        if not (validate_ref or convert_ref):
            return voice_bytes

        from tldw_Server_API.app.core.TTS.audio_utils import AudioProcessor

        processor = AudioProcessor()
        if validate_ref:
            is_valid, error_msg, _ = processor.validate_audio(
                voice_bytes,
                provider=self.PROVIDER_KEY,
                check_duration=True,
                check_quality=False,
            )
            if not is_valid:
                raise TTSInvalidVoiceReferenceError(
                    error_msg or "LuxTTS voice reference validation failed",
                    provider=self.PROVIDER_KEY,
                )

        if convert_ref:
            voice_bytes = await processor.convert_audio_async(
                voice_bytes,
                target_format="wav",
                target_sample_rate=self.reference_sample_rate,
                provider=self.PROVIDER_KEY,
            )

        return voice_bytes

    def _write_temp_audio(self, audio_bytes: bytes) -> str:
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                prefix="lux_tts_voice_",
            ) as tmp:
                tmp.write(audio_bytes)
                return tmp.name
        except Exception as exc:
            raise TTSInvalidVoiceReferenceError(
                "Failed to prepare LuxTTS voice reference file",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

    def _resolve_output_sample_rate(self, return_smooth: bool, extras: dict[str, Any]) -> int:
        for key in ("sample_rate", "output_sample_rate"):
            if key in extras:
                coerced = self._coerce_int(extras.get(key), default=None)
                if coerced:
                    return coerced
        engine = self._engine
        vocos = getattr(engine, "vocos", None) if engine else None
        if vocos is not None:
            try:
                return self.DEFAULT_SAMPLE_RATE if vocos.return_48k else self.DEFAULT_PROMPT_SAMPLE_RATE
            except Exception as sample_rate_error:
                logger.debug(
                    "LuxTTS sample-rate probe failed; using fallback; exception_type={}",
                    _safe_exception_label(sample_rate_error),
                )
        if return_smooth:
            return self.DEFAULT_PROMPT_SAMPLE_RATE
        return int(self.sample_rate)

    def _stream_audio(
        self,
        audio_i16: np.ndarray,
        request: TTSRequest,
        sample_rate: int,
    ) -> AsyncGenerator[bytes, None]:
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=sample_rate,
            channels=1,
        )

        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        chunk_samples = self._resolve_stream_chunk_samples(extras, sample_rate)

        async def stream() -> AsyncGenerator[bytes, None]:
            try:
                for start in range(0, len(audio_i16), chunk_samples):
                    chunk = audio_i16[start:start + chunk_samples]
                    data = writer.write_chunk(chunk)
                    if data:
                        yield data
                final_bytes = writer.write_chunk(finalize=True)
                if final_bytes:
                    yield final_bytes
            except Exception as exc:
                logger.error(
                    "LuxTTS streaming failed; exception_type={}",
                    _safe_exception_label(exc),
                )
                raise TTSGenerationError(
                    "LuxTTS streaming failed",
                    provider=self.PROVIDER_KEY,
                    details={"error": str(exc)},
                ) from exc
            finally:
                writer.close()

        return stream()

    def _resolve_stream_chunk_samples(self, extras: dict[str, Any], sample_rate: int) -> int:
        if "stream_chunk_samples" in extras:
            return self._coerce_int(extras.get("stream_chunk_samples"), default=self.stream_chunk_samples)
        if "stream_chunk_ms" in extras or "stream_chunk_size_ms" in extras:
            raw = extras.get("stream_chunk_ms") if "stream_chunk_ms" in extras else extras.get("stream_chunk_size_ms")
            ms = self._coerce_float(raw, default=None)
            if ms and ms > 0:
                return max(1, int(sample_rate * (ms / 1000.0)))
        return int(self.stream_chunk_samples)

    def _coerce_audio_array(self, wav: Any) -> np.ndarray:
        try:
            if hasattr(wav, "detach"):
                wav = wav.detach()
            if hasattr(wav, "cpu"):
                wav = wav.cpu()
            if hasattr(wav, "numpy"):
                wav = wav.numpy()
        except Exception as tensor_convert_error:
            logger.debug(
                "LuxTTS tensor conversion failed; falling back to numpy coercion; exception_type={}",
                _safe_exception_label(tensor_convert_error),
            )
        audio_np = np.asarray(wav, dtype=np.float32)
        if audio_np.ndim > 1:
            audio_np = np.reshape(audio_np, -1)
        return audio_np

    @staticmethod
    def _coerce_int(value: Any, default: Optional[int]) -> int:
        try:
            if value is None:
                return int(default) if default is not None else 0
            return int(value)
        except (TypeError, ValueError):
            return int(default) if default is not None else 0

    @staticmethod
    def _coerce_float(value: Any, default: Optional[float]) -> float:
        try:
            if value is None:
                return float(default) if default is not None else 0.0
            return float(value)
        except (TypeError, ValueError):
            return float(default) if default is not None else 0.0

    @staticmethod
    def _resolve_bool_setting(
        extras: dict[str, Any],
        extra_keys: tuple[str, ...],
        default: bool,
    ) -> bool:
        for key in extra_keys:
            if key in extras:
                return parse_bool(extras.get(key), default=default)
        return default

    @staticmethod
    def _first_non_null(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

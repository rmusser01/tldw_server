# pocket_tts_adapter.py
# Description: PocketTTS ONNX TTS adapter implementation
#
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
    TTSModelNotFoundError,
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
# PocketTTS ONNX Adapter


def _safe_exception_label(exc: BaseException) -> str:
    """Return a non-sensitive exception identifier for logs."""
    return type(exc).__name__


class PocketTTSOnnxAdapter(TTSAdapter):
    """Adapter for PocketTTS ONNX (voice-cloning, streaming-capable)."""

    PROVIDER_KEY = "pocket_tts"
    SUPPORTED_FORMATS: set[AudioFormat] = {
        AudioFormat.MP3,
        AudioFormat.WAV,
        AudioFormat.OPUS,
        AudioFormat.FLAC,
        AudioFormat.PCM,
        AudioFormat.AAC,
    }
    SUPPORTED_LANGUAGES = {"en"}
    DEFAULT_SAMPLE_RATE = 24000
    MAX_TEXT_LENGTH = 5000
    VALID_PRECISIONS = {"int8", "fp32"}

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        extras = cfg.get("extra_params", {}) or {}

        self.models_dir = Path(
            cfg.get("model_path")
            or cfg.get("pocket_tts_model_path")
            or "models/pocket_tts_onnx/onnx"
        ).expanduser()
        self.tokenizer_path = Path(
            cfg.get("tokenizer_path")
            or cfg.get("pocket_tts_tokenizer_path")
            or "models/pocket_tts_onnx/tokenizer.model"
        ).expanduser()
        self.module_path = Path(
            cfg.get("module_path")
            or cfg.get("pocket_tts_module_path")
            or self.models_dir.parent
        ).expanduser()

        self.precision = str(
            cfg.get("precision")
            or cfg.get("pocket_tts_precision")
            or "int8"
        ).lower()
        self.device = str(
            cfg.get("device")
            or cfg.get("pocket_tts_device")
            or "auto"
        ).lower()

        def _float_setting(key: str, default: float) -> float:
            raw = cfg.get(key)
            if raw is None:
                raw = extras.get(key)
            try:
                return float(raw)
            except (TypeError, ValueError):
                return float(default)

        def _int_setting(key: str, default: int) -> int:
            raw = cfg.get(key)
            if raw is None:
                raw = extras.get(key)
            try:
                return int(raw)
            except (TypeError, ValueError):
                return int(default)

        self.temperature = _float_setting("temperature", 0.7)
        self.lsd_steps = _int_setting("lsd_steps", 10)
        self.max_frames = _int_setting("max_frames", 500)

        self.stream_first_chunk_frames = _int_setting("stream_first_chunk_frames", 2)
        self.stream_target_buffer_sec = _float_setting("stream_target_buffer_sec", 0.2)
        self.stream_max_chunk_frames = _int_setting("stream_max_chunk_frames", 15)

        self.sample_rate = int(cfg.get("sample_rate", self.DEFAULT_SAMPLE_RATE))
        self._audio_normalizer = AudioNormalizer()
        self._engine: Optional[Any] = None
        self._engine_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Load PocketTTS ONNX models and tokenizer."""
        if self._initialized:
            return True

        if self.precision not in self.VALID_PRECISIONS:
            raise TTSProviderInitializationError(
                f"Invalid precision '{self.precision}' for PocketTTS",
                provider=self.PROVIDER_KEY,
                details={"valid_precisions": sorted(self.VALID_PRECISIONS)},
            )

        if self.device not in {"auto", "cpu", "cuda"}:
            raise TTSProviderInitializationError(
                f"Invalid device '{self.device}' for PocketTTS",
                provider=self.PROVIDER_KEY,
                details={"valid_devices": ["auto", "cpu", "cuda"]},
            )

        self._validate_assets()

        PocketTTSOnnx = self._load_engine_class()

        try:
            engine = await asyncio.to_thread(
                PocketTTSOnnx,
                models_dir=str(self.models_dir),
                tokenizer_path=str(self.tokenizer_path),
                precision=self.precision,
                device=self.device,
                temperature=self.temperature,
                lsd_steps=self.lsd_steps,
            )
        except Exception as exc:
            raise TTSProviderInitializationError(
                "Failed to initialize PocketTTS ONNX engine",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

        self._engine = engine
        try:
            from ..tts_resource_manager import get_resource_manager
            resource_manager = await get_resource_manager()
            register_result = resource_manager.register_model(
                provider=self.PROVIDER_KEY,
                model_instance=engine,
                cleanup_callback=self._cleanup_resources,
                model_key=f"{self.models_dir}:{self.precision}",
            )
            if asyncio.iscoroutine(register_result):
                await register_result
        except Exception as registration_error:
            logger.debug(
                "PocketTTS provider registration failed; continuing; exception_type={}",
                _safe_exception_label(registration_error),
            )
        try:
            self.sample_rate = int(getattr(engine, "SAMPLE_RATE", self.sample_rate))
        except Exception:
            self.sample_rate = self.DEFAULT_SAMPLE_RATE

        self._capabilities = await self.get_capabilities()
        self._status = ProviderStatus.AVAILABLE
        self._initialized = True
        logger.info(
            'PocketTTS adapter initialized (models_dir={}, precision={}, device={})',
            self.models_dir,
            self.precision,
            self.device,
        )
        return True

    def _load_engine_class(self):
        try:
            from pocket_tts_onnx import PocketTTSOnnx  # type: ignore
            return PocketTTSOnnx
        except ImportError as exc:
            module_path = self.module_path
            module_dir = module_path if module_path.is_dir() else module_path.parent
            if module_dir.exists():
                module_path_str = str(module_dir)
                if module_path_str not in sys.path:
                    sys.path.insert(0, module_path_str)
                try:
                    module = importlib.import_module("pocket_tts_onnx")
                    engine_cls = getattr(module, "PocketTTSOnnx", None)
                    if engine_cls is None:
                        raise AttributeError("PocketTTSOnnx not found in module")
                    return engine_cls
                except Exception as inner_exc:
                    raise TTSModelLoadError(
                        "PocketTTS ONNX module could not be imported",
                        provider=self.PROVIDER_KEY,
                        details={
                            "error": str(inner_exc),
                            "module_path": module_path_str,
                            "suggestion": "python Helper_Scripts/TTS_Installers/install_tts_pocket_tts_onnx.py",
                        },
                    ) from inner_exc

            raise TTSModelLoadError(
                "PocketTTS ONNX module not found",
                provider=self.PROVIDER_KEY,
                details={
                    "error": str(exc),
                    "module_path": str(module_dir),
                    "suggestion": "python Helper_Scripts/TTS_Installers/install_tts_pocket_tts_onnx.py",
                },
            ) from exc

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="PocketTTS",
            supported_languages=self.SUPPORTED_LANGUAGES,
            supported_voices=[],
            supported_formats=self.SUPPORTED_FORMATS,
            max_text_length=self.MAX_TEXT_LENGTH,
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
            latency_ms=800,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "PocketTTS adapter not initialized",
                provider=self.PROVIDER_KEY,
            )

        if request.format not in self.SUPPORTED_FORMATS:
            raise TTSUnsupportedFormatError(
                f"Format {request.format.value} not supported by PocketTTS",
                provider=self.PROVIDER_KEY,
            )

        if not self._engine:
            raise TTSProviderNotConfiguredError(
                "PocketTTS engine not available",
                provider=self.PROVIDER_KEY,
            )

        try:
            validate_tts_request(request, provider=self.PROVIDER_KEY)
        except TTSValidationError:
            raise
        except Exception as exc:
            raise TTSValidationError(
                f"Validation failed for PocketTTS request: {exc}",
                provider=self.PROVIDER_KEY,
            ) from exc

        extras = request.extra_params or {}
        if not isinstance(extras, dict):
            extras = {}
        voice_bytes = self._extract_voice_reference(request.voice_reference)
        voice_bytes = await self._prepare_voice_reference(voice_bytes, extras)

        voice_path = self._write_temp_audio(voice_bytes)
        try:
            if request.stream:
                audio_stream = self._stream_audio(request, voice_path, extras)
                return TTSResponse(
                    audio_stream=audio_stream,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    text_processed=request.text,
                    voice_used=request.voice,
                    provider=self.PROVIDER_KEY,
                    model=request.model or "pocket-tts-onnx",
                )

            async with self._engine_lock:
                audio = await asyncio.to_thread(
                    self._engine.generate,  # type: ignore[union-attr]
                    request.text,
                    voice_path,
                    self._resolve_max_frames(extras),
                )

            audio_np = np.asarray(audio)
            if audio_np.ndim > 1:
                audio_np = audio_np.reshape(-1)
            audio_i16 = self._audio_normalizer.normalize(audio_np, target_dtype=np.int16)

            audio_bytes = await self.convert_audio_format(
                audio_i16,
                source_format=AudioFormat.PCM,
                target_format=request.format,
                sample_rate=self.sample_rate,
            )
            return TTSResponse(
                audio_data=audio_bytes,
                format=request.format,
                sample_rate=self.sample_rate,
                channels=1,
                text_processed=request.text,
                voice_used=request.voice,
                provider=self.PROVIDER_KEY,
                model=request.model or "pocket-tts-onnx",
            )
        except Exception as exc:
            logger.error(
                "PocketTTS generation failed; exception_type={}",
                _safe_exception_label(exc),
            )
            raise TTSGenerationError(
                "PocketTTS generation failed",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc
        finally:
            Path(voice_path).unlink(missing_ok=True)

    async def _cleanup_resources(self) -> None:
        self._engine = None

    def _validate_assets(self) -> None:
        if not self.models_dir.exists():
            raise TTSModelNotFoundError(
                f"PocketTTS models directory not found at {self.models_dir}",
                provider=self.PROVIDER_KEY,
                details={"models_dir": str(self.models_dir)},
            )

        if not self.tokenizer_path.exists():
            raise TTSModelNotFoundError(
                f"PocketTTS tokenizer not found at {self.tokenizer_path}",
                provider=self.PROVIDER_KEY,
                details={"tokenizer_path": str(self.tokenizer_path)},
            )

        suffix = "_int8" if self.precision == "int8" else ""
        required_files = [
            f"flow_lm_main{suffix}.onnx",
            f"flow_lm_flow{suffix}.onnx",
            f"mimi_decoder{suffix}.onnx",
            "mimi_encoder.onnx",
            "text_conditioner.onnx",
        ]
        missing = [name for name in required_files if not (self.models_dir / name).exists()]
        if missing:
            raise TTSModelNotFoundError(
                "PocketTTS ONNX assets missing",
                provider=self.PROVIDER_KEY,
                details={"missing": missing, "models_dir": str(self.models_dir)},
            )

    def _extract_voice_reference(self, voice_reference: Any) -> bytes:
        if voice_reference is None:
            raise TTSInvalidVoiceReferenceError(
                "PocketTTS requires voice_reference audio bytes",
                provider=self.PROVIDER_KEY,
            )

        if isinstance(voice_reference, (bytes, bytearray)):
            return bytes(voice_reference)

        if isinstance(voice_reference, str):
            try:
                return base64.b64decode(voice_reference)
            except Exception as exc:
                raise TTSInvalidVoiceReferenceError(
                    "PocketTTS voice_reference is not valid base64",
                    provider=self.PROVIDER_KEY,
                    details={"error": str(exc)},
                ) from exc

        raise TTSInvalidVoiceReferenceError(
            "PocketTTS voice_reference must be bytes or base64 string",
            provider=self.PROVIDER_KEY,
            details={"type": type(voice_reference).__name__},
        )

    async def _prepare_voice_reference(self, voice_bytes: bytes, extras: dict[str, Any]) -> bytes:
        validate_ref = self._resolve_bool_setting(
            extras,
            ("validate_reference", "validate_voice_reference"),
            ("validate_reference", "validate_voice_reference"),
            default=True,
        )
        convert_ref = self._resolve_bool_setting(
            extras,
            ("convert_reference", "convert_voice_reference"),
            ("convert_reference", "convert_voice_reference"),
            default=True,
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
                    error_msg or "PocketTTS voice reference validation failed",
                    provider=self.PROVIDER_KEY,
                )

        if convert_ref:
            voice_bytes = await processor.convert_audio_async(
                voice_bytes,
                target_format="wav",
                target_sample_rate=self.sample_rate,
                provider=self.PROVIDER_KEY,
            )

        return voice_bytes

    def _write_temp_audio(self, audio_bytes: bytes) -> str:
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                prefix="pocket_tts_voice_",
            ) as tmp:
                tmp.write(audio_bytes)
                return tmp.name
        except Exception as exc:
            raise TTSInvalidVoiceReferenceError(
                "Failed to prepare PocketTTS voice reference file",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

    def _resolve_max_frames(self, extras: dict[str, Any]) -> int:
        max_frames = extras.get("max_frames")
        if max_frames is None:
            return self.max_frames
        try:
            return int(max_frames)
        except (TypeError, ValueError):
            return self.max_frames

    def _stream_audio(
        self,
        request: TTSRequest,
        voice_path: str,
        extras: dict[str, Any],
    ) -> AsyncGenerator[bytes, None]:
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=self.sample_rate,
            channels=1,
        )
        normalizer = self._audio_normalizer

        first_chunk_frames = self._coerce_int(
            extras.get("stream_first_chunk_frames")
            if "stream_first_chunk_frames" in extras
            else extras.get("first_chunk_frames"),
            self.stream_first_chunk_frames,
        )
        target_buffer_sec = self._coerce_float(
            extras.get("stream_target_buffer_sec")
            if "stream_target_buffer_sec" in extras
            else extras.get("target_buffer_sec"),
            self.stream_target_buffer_sec,
        )
        max_chunk_frames = self._coerce_int(
            extras.get("stream_max_chunk_frames")
            if "stream_max_chunk_frames" in extras
            else extras.get("max_chunk_frames"),
            self.stream_max_chunk_frames,
        )
        max_frames = self._resolve_max_frames(extras)

        async def stream() -> AsyncGenerator[bytes, None]:
            sentinel = object()
            loop = asyncio.get_running_loop()
            try:
                async with self._engine_lock:
                    generator = self._engine.stream(  # type: ignore[union-attr]
                        request.text,
                        voice_path,
                        max_frames=max_frames,
                        first_chunk_frames=first_chunk_frames,
                        target_buffer_sec=target_buffer_sec,
                        max_chunk_frames=max_chunk_frames,
                    )
                    while True:
                        chunk = await loop.run_in_executor(
                            None, lambda: next(generator, sentinel)
                        )
                        if chunk is sentinel:
                            break
                        data = self._convert_stream_chunk(chunk, normalizer, writer)
                        if data:
                            yield data

                final_bytes = writer.write_chunk(finalize=True)
                if final_bytes:
                    yield final_bytes
            except Exception as exc:
                logger.error(
                    "PocketTTS streaming failed; exception_type={}",
                    _safe_exception_label(exc),
                )
                raise TTSGenerationError(
                    "PocketTTS streaming failed",
                    provider=self.PROVIDER_KEY,
                    details={"error": str(exc)},
                ) from exc
            finally:
                writer.close()
                Path(voice_path).unlink(missing_ok=True)

        return stream()

    def _convert_stream_chunk(
        self,
        chunk: Any,
        normalizer: AudioNormalizer,
        writer: StreamingAudioWriter,
    ) -> bytes:
        if chunk is None:
            return b""
        try:
            audio_np = np.asarray(chunk).astype(np.float32)
        except Exception as exc:
            logger.warning(
                "PocketTTS stream chunk conversion error; exception_type={}",
                _safe_exception_label(exc),
            )
            return b""

        audio_np = np.squeeze(audio_np)
        if audio_np.size == 0:
            return b""

        audio_i16 = normalizer.normalize(audio_np, target_dtype=np.int16)
        return writer.write_chunk(audio_i16)

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_bool_setting(
        self,
        extras: dict[str, Any],
        extra_keys: tuple[str, ...],
        config_keys: tuple[str, ...],
        default: bool,
    ) -> bool:
        for key in extra_keys:
            if key in extras:
                return parse_bool(extras.get(key), default=default)
        for key in config_keys:
            if key in self.config:
                return parse_bool(self.config.get(key), default=default)
        return default

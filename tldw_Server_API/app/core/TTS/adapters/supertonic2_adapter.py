# supertonic2_adapter.py
# Description: Supertonic2 ONNX TTS adapter implementation
#
import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Optional

#
# Third-party Imports
import numpy as np
from loguru import logger

from ..streaming_audio_writer import AudioNormalizer
from ..tts_exceptions import (
    TTSGenerationError,
    TTSModelLoadError,
    TTSModelNotFoundError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSUnsupportedFormatError,
    TTSValidationError,
    TTSVoiceNotFoundError,
)
from ..tts_validation import validate_tts_request

#
# Local Imports
from .base import (
    AudioFormat,
    ProviderStatus,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
    VoiceInfo,
)

#
#######################################################################################################################
#
# Supertonic2 Adapter Implementation


class Supertonic2OnnxAdapter(TTSAdapter):
    """Adapter for the Supertonic2 ONNX TTS engine."""

    PROVIDER_KEY = "supertonic2"
    SUPPORTED_FORMATS = {AudioFormat.MP3, AudioFormat.WAV}
    SUPPORTED_LANGUAGES = {"en", "ko", "es", "pt", "fr"}
    MAX_TEXT_LENGTH = 15000
    DEFAULT_SAMPLE_RATE = 24000

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}
        extras = cfg.get("extra_params", {}) or {}

        self.onnx_dir = Path(cfg.get("model_path", "models/supertonic2/onnx")).expanduser()
        self.voice_styles_dir = Path(
            extras.get("voice_styles_dir", "models/supertonic2/voice_styles")
        ).expanduser()
        self.default_voice = extras.get("default_voice", "supertonic2_m1")
        self.voice_files: dict[str, str] = extras.get("voice_files", {}) or {}
        self.default_total_step = int(extras.get("default_total_step", 5))
        self.default_speed = float(extras.get("default_speed", 1.05))
        self.n_test = int(extras.get("n_test", 1))
        self.device = cfg.get("device", "cpu")
        self.use_gpu = str(self.device).lower() == "cuda"

        self.sample_rate = int(cfg.get("sample_rate", self.DEFAULT_SAMPLE_RATE))
        stream_chunk = extras.get("stream_chunk_size") or cfg.get("stream_chunk_size")
        try:
            if stream_chunk is None:
                perf_cfg = cfg.get("performance", {}) or {}
                stream_chunk = perf_cfg.get("stream_chunk_size")
            self.stream_chunk_size = int(stream_chunk) if stream_chunk else 8192
        except Exception:
            self.stream_chunk_size = 8192

        self._engine: Optional[Any] = None
        self._load_voice_style = None
        self._voice_to_path: dict[str, Path] = {}
        self._voice_infos: list[VoiceInfo] = []
        self._audio_normalizer = AudioNormalizer()
        self._engine_lock = asyncio.Lock()

    async def ensure_initialized(self) -> bool:
        """
        Ensure the provider is initialized, propagating TTSErrors so callers see
        clear misconfiguration messages (e.g., missing model or voice files).
        """
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

    # ---------------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------------
    async def initialize(self) -> bool:
        """Load the Supertonic2 ONNX engine and discover voice styles."""
        if self._initialized:
            return True

        if not self.onnx_dir.exists():
            raise TTSModelNotFoundError(
                f"Supertonic2 ONNX directory not found at {self.onnx_dir}",
                provider=self.PROVIDER_KEY,
                details={"onnx_dir": str(self.onnx_dir)},
            )

        if not self.voice_styles_dir.exists():
            raise TTSModelNotFoundError(
                f"Supertonic2 voice styles directory not found at {self.voice_styles_dir}",
                provider=self.PROVIDER_KEY,
                details={"voice_styles_dir": str(self.voice_styles_dir)},
            )

        try:
            from tldw_Server_API.app.core.TTS.vendors import supertonic2 as vendor
        except ImportError as exc:
            raise TTSModelLoadError(
                "Supertonic2 vendor package not available",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

        try:
            engine = await asyncio.to_thread(
                vendor.load_text_to_speech, str(self.onnx_dir), self.use_gpu
            )
        except FileNotFoundError as exc:
            raise TTSModelNotFoundError(
                f"Supertonic2 model assets missing under {self.onnx_dir}",
                provider=self.PROVIDER_KEY,
                details={"onnx_dir": str(self.onnx_dir)},
            ) from exc
        except Exception as exc:
            raise TTSProviderInitializationError(
                "Failed to initialize Supertonic2 engine",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

        self._engine = engine
        self._load_voice_style = vendor.load_voice_style
        try:
            from ..tts_resource_manager import get_resource_manager
            resource_manager = await get_resource_manager()
            register_result = resource_manager.register_model(
                provider=self.PROVIDER_KEY,
                model_instance=self._engine,
                cleanup_callback=lambda: setattr(self, "_engine", None),
                model_key=str(self.onnx_dir),
            )
            if asyncio.iscoroutine(register_result):
                await register_result
        except Exception as registration_error:
            logger.debug(
                "Supertonic2 provider registration failed; continuing; exception_type={}",
                type(registration_error).__name__,
            )

        try:
            self.sample_rate = int(getattr(self._engine, "sample_rate", self.sample_rate))
        except Exception:
            logger.debug(
                'Supertonic2: unable to read sample_rate from engine, using default {}',
                self.sample_rate,
            )

        self._voice_to_path, self._voice_infos = self._load_voice_mappings()

        self._capabilities = await self.get_capabilities()
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        logger.info(
            'Supertonic2 adapter initialized (onnx_dir={}, voices={}, sample_rate={})',
            self.onnx_dir,
            len(self._voice_infos),
            self.sample_rate,
        )
        return True

    def _load_voice_mappings(self) -> tuple[dict[str, Path], list[VoiceInfo]]:
        """Build mapping of voice IDs to style files and VoiceInfo entries."""
        voice_map: dict[str, Path] = {}
        voices: list[VoiceInfo] = []
        missing_default = False

        for voice_id, filename in self.voice_files.items():
            path = (self.voice_styles_dir / filename).expanduser()
            if not path.exists():
                if voice_id == self.default_voice:
                    missing_default = True
                logger.warning(
                    'Supertonic2 voice style missing: voice_id={} path={}', voice_id, path
                )
                continue
            voice_map[voice_id] = path
            gender = None
            lowered = voice_id.lower()
            if "m" in lowered:
                gender = "male"
            elif "f" in lowered:
                gender = "female"
            voices.append(
                VoiceInfo(
                    id=voice_id,
                    name=voice_id.replace("_", " ").title(),
                    gender=gender,
                    language="en",
                    description="Supertonic2 voice style",
                    styles=["neutral"],
                    use_case=["general"],
                )
            )

        if missing_default or self.default_voice not in voice_map:
            raise TTSModelNotFoundError(
                f"Default Supertonic2 voice style '{self.default_voice}' not found",
                provider=self.PROVIDER_KEY,
                details={
                    "voice_styles_dir": str(self.voice_styles_dir),
                    "default_voice": self.default_voice,
                },
            )

        return voice_map, voices

    # ---------------------------------------------------------------------------------
    # Capabilities
    # ---------------------------------------------------------------------------------
    async def get_capabilities(self) -> TTSCapabilities:
        """Return declared Supertonic2 capabilities."""
        voices = self._voice_infos or []
        return TTSCapabilities(
            provider_name="Supertonic2",
            supported_languages=self.SUPPORTED_LANGUAGES,
            supported_voices=voices,
            supported_formats=self.SUPPORTED_FORMATS,
            max_text_length=self.MAX_TEXT_LENGTH,
            supports_streaming=True,
            supports_voice_cloning=False,
            supports_emotion_control=False,
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=3500,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV,
        )

    # ---------------------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------------------
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate audio using Supertonic2 (non-streaming and pseudo-streaming)."""
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "Supertonic2 adapter not initialized",
                provider=self.PROVIDER_KEY,
            )

        if request.format not in self.SUPPORTED_FORMATS:
            raise TTSUnsupportedFormatError(
                f"Format {request.format.value} not supported by Supertonic2",
                provider=self.PROVIDER_KEY,
            )

        try:
            validate_tts_request(request, provider=self.PROVIDER_KEY)
        except TTSValidationError:
            raise
        except Exception as exc:
            raise TTSValidationError(
                f"Validation failed for Supertonic2 request: {exc}",
                provider=self.PROVIDER_KEY,
            ) from exc

        voice_id = request.voice or self.default_voice
        style_path = self._voice_to_path.get(voice_id)
        if not style_path:
            raise TTSVoiceNotFoundError(
                f"Voice '{voice_id}' not found for Supertonic2",
                provider=self.PROVIDER_KEY,
                details={"voice": voice_id},
            )

        extras = request.extra_params or {}
        total_step = int(extras.get("total_step", self.default_total_step))
        speed = float(request.speed if request.speed is not None else self.default_speed)
        language = (request.language or "en").lower()

        try:
            style = await asyncio.to_thread(self._load_voice_style, [str(style_path)], False)
        except Exception as exc:
            raise TTSModelLoadError(
                f"Failed to load Supertonic2 voice style {style_path}",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

        try:
            async with self._engine_lock:
                wav, duration = await asyncio.to_thread(
                    self._engine, request.text, language, style, total_step, speed
                )
        except Exception as exc:
            raise TTSGenerationError(
                "Supertonic2 generation failed",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

        audio_array = self._prepare_audio_array(wav, duration)
        if audio_array.dtype != np.int16:
            audio_array = self._audio_normalizer.normalize(audio_array, target_dtype=np.int16)

        audio_bytes = await self.convert_audio_format(
            audio_array,
            source_format=AudioFormat.PCM,
            target_format=request.format,
            sample_rate=self.sample_rate,
        )

        if request.stream:
            audio_stream = self._build_stream(audio_bytes)
            return TTSResponse(
                audio_stream=audio_stream,
                format=request.format,
                sample_rate=self.sample_rate,
                channels=1,
                text_processed=request.text,
                voice_used=voice_id,
                provider=self.PROVIDER_KEY,
                model=request.model or "tts-supertonic2-1",
            )

        return TTSResponse(
            audio_data=audio_bytes,
            format=request.format,
            sample_rate=self.sample_rate,
            channels=1,
            text_processed=request.text,
            voice_used=voice_id,
            provider=self.PROVIDER_KEY,
            model=request.model or "tts-supertonic2-1",
        )

    def _prepare_audio_array(self, wav: Any, duration: Any) -> np.ndarray:
        """Trim and normalize engine output to a 1-D numpy array."""
        try:
            arr = np.asarray(wav)
        except Exception as exc:
            raise TTSGenerationError(
                "Supertonic2 returned non-array audio data",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

        if arr.ndim == 2:
            if arr.shape[0] != 1:
                raise TTSGenerationError(
                    f"Expected batch size 1 from Supertonic2, got shape {arr.shape}",
                    provider=self.PROVIDER_KEY,
                )
            arr = arr[0]
        elif arr.ndim != 1:
            raise TTSGenerationError(
                f"Unexpected Supertonic2 audio shape {arr.shape}",
                provider=self.PROVIDER_KEY,
            )

        try:
            dur_val = float(duration[0]) if isinstance(duration, (list, tuple, np.ndarray)) else float(duration)
            end_idx = int(self.sample_rate * dur_val)
            if end_idx > 0:
                arr = arr[:end_idx]
        except Exception as trim_error:
            logger.debug(
                "Supertonic2 audio trim by duration failed; using untrimmed audio; exception_type={}",
                type(trim_error).__name__,
            )
        return arr

    def _build_stream(self, audio_bytes: bytes) -> AsyncGenerator[bytes, None]:
        """Create pseudo-streaming generator from encoded audio bytes."""
        chunk_size = max(1024, int(self.stream_chunk_size or 8192))

        async def _byte_stream():
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                if chunk:
                    yield chunk

        return _byte_stream()

    async def close(self):
        """Clean up resources and reset state."""
        try:
            await super().close()
        finally:
            self._engine = None
            self._load_voice_style = None
            self._voice_infos = []
            self._voice_to_path = {}

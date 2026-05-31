"""
chatterbox_adapter.py
Description: Chatterbox TTS adapter implementation (Resemble AI)
Updated to use upstream chatterbox package (v0.1.4):
- Imports from chatterbox.tts and chatterbox.mtl_tts
- Supports multilingual via language_id
- Uses generate(...) waveform (no native streaming) and progressively streams encoded chunks
- Disables upstream watermarking by replacing watermarker with a no-op
"""

# Imports
import asyncio
import contextlib
import importlib
import os
import sys
from collections.abc import AsyncGenerator
from typing import Any, Optional

import numpy as np

# Third-party Imports
from loguru import logger

from ..tts_exceptions import (
    TTSModelLoadError,
)
from ..utils import parse_bool

# Local Imports
from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo

_CHATTERBOX_IMPORT_EXCEPTIONS = (ImportError, ModuleNotFoundError)
_CHATTERBOX_RUNTIME_EXCEPTIONS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_CHATTERBOX_NUMERIC_EXCEPTIONS = (TypeError, ValueError)

#######################################################################################################################
# No-op watermarker to ensure no watermark is applied

torch: Any | None = None
_TORCH_IMPORT_ERROR: Exception | None = None
_TORCH_IMPORT_ATTEMPTED: bool = False


def _is_test_runtime() -> bool:
    test_flags = {"1", "true", "yes", "y", "on"}
    if str(os.getenv("PYTEST_CURRENT_TEST", "")).strip():
        return True
    if str(os.getenv("MINIMAL_TEST_APP", "")).strip().lower() in test_flags:
        return True
    if str(os.getenv("TLDW_TEST_MODE", "")).strip().lower() in test_flags:
        return True
    return any("pytest" in str(arg or "") for arg in sys.argv)


def _get_torch(*, allow_import: bool) -> Any | None:
    global torch, _TORCH_IMPORT_ERROR, _TORCH_IMPORT_ATTEMPTED

    if torch is not None:
        return torch
    if _TORCH_IMPORT_ERROR is not None:
        return None
    if not allow_import:
        return None
    if _TORCH_IMPORT_ATTEMPTED:
        return None

    _TORCH_IMPORT_ATTEMPTED = True
    try:
        torch = importlib.import_module("torch")
        _TORCH_IMPORT_ERROR = None
    except Exception as exc:
        _TORCH_IMPORT_ERROR = exc
        torch = None
    return torch


def _torch_cuda_available(*, allow_import: bool = False) -> bool:
    torch_mod = _get_torch(allow_import=allow_import and not _is_test_runtime())
    if torch_mod is None:
        return False
    try:
        return bool(torch_mod.cuda.is_available())
    except Exception:
        return False


def _torch_mps_available(*, allow_import: bool = False) -> bool:
    torch_mod = _get_torch(allow_import=allow_import and not _is_test_runtime())
    if torch_mod is None:
        return False
    try:
        return bool(hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available())
    except Exception:
        return False

class _NoopWatermarker:
    def apply_watermark(self, wav: np.ndarray, sample_rate: int = 24000):
        return wav


#######################################################################################################################
# Chatterbox TTS Adapter Implementation

class ChatterboxAdapter(TTSAdapter):
    """
    Adapter for Chatterbox TTS from Resemble AI.
    Updated to upstream API with emotion exaggeration and multilingual support.
    """

    # Emotion labels maintained for UI compatibility; mapped to `exaggeration` scalar
    EMOTIONS = {
        "neutral", "happy", "sad", "angry", "surprised",
        "fearful", "disgusted", "excited", "calm", "confused"
    }

    # Optional character voices (for UI/tests; not used by upstream directly)
    CHARACTER_VOICES = {
        "narrator": "narrator",
        "hero": "hero",
        "villain": "villain",
        "sidekick": "sidekick",
        "sage": "sage",
        "comic_relief": "comic_relief",
    }

    # Voice presets (cosmetic metadata)
    VOICE_PRESETS = {
        "default": VoiceInfo(
            id="default",
            name="Default",
            gender="neutral",
            description="Default Chatterbox voice",
            styles=["neutral", "conversational"]
        ),
        "energetic": VoiceInfo(
            id="energetic",
            name="Energetic",
            gender="neutral",
            description="High energy voice",
            styles=["excited", "happy"]
        ),
        "calm": VoiceInfo(
            id="calm",
            name="Calm",
            gender="neutral",
            description="Calm and soothing voice",
            styles=["calm", "neutral"]
        ),
        "professional": VoiceInfo(
            id="professional",
            name="Professional",
            gender="neutral",
            description="Professional business voice",
            styles=["neutral", "confident"]
        )
    }

    # Multilingual language codes supported upstream
    MULTILINGUAL_LANGS: set[str] = {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
        "sw", "tr", "zh"
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)

        # Device selection: prefer explicit config; otherwise CUDA if available, else CPU.
        preferred = self.config.get("chatterbox_device") or self.config.get("device")
        if preferred:
            pref = str(preferred).lower()
            if pref == "cuda":
                self.device = "cuda" if _torch_cuda_available(allow_import=True) else "cpu"
            elif pref == "mps":
                if _torch_mps_available(allow_import=True):
                    self.device = "mps"
                else:
                    self.device = "cuda" if _torch_cuda_available(allow_import=True) else "cpu"
            elif pref == "cpu":
                self.device = "cpu"
            else:
                # Unknown preference; fall back to CUDA/CPU
                self.device = "cuda" if _torch_cuda_available(allow_import=True) else "cpu"
        else:
            self.device = "cuda" if _torch_cuda_available(allow_import=True) else "cpu"

        # Provider settings
        self.use_multilingual = self.config.get("chatterbox_use_multilingual", self.config.get("use_multilingual", False))
        self.disable_watermark = self.config.get("chatterbox_disable_watermark", self.config.get("disable_watermark", True))

        # Default sampling/expression parameters
        self.default_exaggeration = float(self.config.get("chatterbox_default_exaggeration", 0.5))
        self.default_cfg_weight = float(self.config.get("chatterbox_cfg_weight", 0.5))
        self.default_temperature = float(self.config.get("chatterbox_temperature", 0.8))
        self.default_repetition_penalty = float(self.config.get("chatterbox_repetition_penalty", 1.2))
        self.default_min_p = float(self.config.get("chatterbox_min_p", 0.05))
        self.default_top_p = float(self.config.get("chatterbox_top_p", 1.0))

        # Model instances (lazy-loaded based on language)
        self.model_en = None  # ChatterboxTTS
        self.model_multi = None  # ChatterboxMultilingualTTS

        # Audio parameters (sample rate will be taken from model)
        self.sample_rate = 24000

        # Target latency hint (progressive encoding)
        self.target_latency_ms = 200

        # Auto-download toggle: config override > env overrides > default True
        cfg_auto = self.config.get("chatterbox_auto_download") or self.config.get("auto_download")
        env_auto = os.getenv("CHATTERBOX_AUTO_DOWNLOAD") or os.getenv("TTS_AUTO_DOWNLOAD")
        self.auto_download = parse_bool(cfg_auto, default=parse_bool(env_auto, default=True))

    async def initialize(self) -> bool:
        """Initialize the Chatterbox TTS adapter (lazy model load)."""
        if _get_torch(allow_import=True) is None:
            logger.warning(
                f"{self.provider_name}: torch unavailable; disabling provider. error={_TORCH_IMPORT_ERROR}"
            )
            self._status = ProviderStatus.NOT_CONFIGURED
            return False
        try:
            # Verify the upstream package is available
            try:
                import chatterbox  # noqa: F401
            except _CHATTERBOX_IMPORT_EXCEPTIONS as e:
                suggestion = (
                    "pip install chatterbox-tts\n"
                    "or install from source: git clone https://github.com/resemble-ai/chatterbox && pip install -e ."
                )
                logger.error(f"{self.provider_name}: chatterbox package not installed")
                raise TTSModelLoadError(
                    "Failed to import chatterbox package",
                    provider=self.provider_name,
                    details={"error": str(e), "suggestion": suggestion}
                ) from e

            # Defer heavy model weights loading until first request
            self._status = ProviderStatus.AVAILABLE
            return True
        except (TTSModelLoadError, *_CHATTERBOX_RUNTIME_EXCEPTIONS) as e:
            logger.error(f"{self.provider_name}: Initialization failed ({type(e).__name__})")
            self._status = ProviderStatus.ERROR
            return False

    async def get_capabilities(self) -> TTSCapabilities:
        """Get Chatterbox TTS capabilities"""
        langs = self.MULTILINGUAL_LANGS if self.use_multilingual else {"en"}
        return TTSCapabilities(
            provider_name="Chatterbox",
            supported_languages=langs,
            supported_voices=list(self.VOICE_PRESETS.values()),
            supported_formats={
                AudioFormat.WAV,
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.FLAC,
                AudioFormat.PCM
            },
            max_text_length=10000,
            supports_streaming=True,
            supports_voice_cloning=True,
            supports_emotion_control=True,  # via `exaggeration`
            supports_speech_rate=False,  # not supported upstream
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=self.target_latency_ms,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Chatterbox TTS"""
        if not await self.ensure_initialized():
            raise ValueError(f"{self.provider_name} not initialized")

        # Validate request against adapter capabilities
        is_valid, error = await self.validate_request(request)
        if not is_valid:
            raise ValueError(error)

        # Determine model (multilingual or english)
        language_id = (request.language or "en").lower()
        model = await self._get_model(language_id)
        self.sample_rate = int(getattr(model, 'sr', 24000))

        # Handle voice cloning if reference provided
        voice_reference_path = None
        if request.voice_reference:
            voice_reference_path = await self._prepare_voice_reference(request.voice_reference)

        # Compute exaggeration from emotion + intensity
        exaggeration = self._map_emotion_to_exaggeration(
            request.emotion,
            request.emotion_intensity
        )

        logger.info(
            f"{self.provider_name}: Generating speech (lang={language_id}, voice={request.voice or 'default'}, fmt={request.format.value})"
        )

        try:
            if request.stream:
                return TTSResponse(
                    audio_stream=self._stream_audio_chatterbox(
                        request,
                        language_id,
                        voice_reference_path,
                        exaggeration
                    ),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=request.voice or "default",
                    provider=self.provider_name,
                    metadata={
                        "language": language_id,
                        "exaggeration": exaggeration,
                        "watermarked": False
                    }
                )
            else:
                audio_data = await self._generate_complete_chatterbox(
                    request,
                    language_id,
                    voice_reference_path,
                    exaggeration
                )
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=request.voice or "default",
                    provider=self.provider_name,
                    metadata={
                        "language": language_id,
                        "exaggeration": exaggeration,
                        "watermarked": False
                    }
                )
        finally:
            # Clean up temp voice reference
            if voice_reference_path:
                try:
                    from pathlib import Path
                    Path(voice_reference_path).unlink(missing_ok=True)
                except (OSError, TypeError, ValueError):
                    pass

    async def _stream_audio_chatterbox(
        self,
        request: TTSRequest,
        language_id: str,
        voice_reference_path: Optional[str],
        exaggeration: float,
    ) -> AsyncGenerator[bytes, None]:
        """Generate waveform with upstream model, progressively encode and stream bytes."""
        model = await self._get_model(language_id)

        try:
            # Prepare kwargs for upstream generate
            gen_kwargs: dict[str, Any] = {
                "audio_prompt_path": voice_reference_path,
                "exaggeration": exaggeration,
                "cfg_weight": request.extra_params.get("cfg_weight", self.default_cfg_weight),
                "temperature": request.extra_params.get("temperature", self.default_temperature),
                "repetition_penalty": request.extra_params.get("repetition_penalty", self.default_repetition_penalty),
                "min_p": request.extra_params.get("min_p", self.default_min_p),
                "top_p": request.extra_params.get("top_p", self.default_top_p),
            }

            # Generate full waveform tensor (1, N)
            if self.use_multilingual and language_id != "en":
                audio_tensor = model.generate(
                    self.preprocess_text(request.text),
                    language_id=language_id,
                    **gen_kwargs,
                )
            else:
                audio_tensor = model.generate(
                    self.preprocess_text(request.text),
                    **gen_kwargs,
                )
            # Stream using shared waveform streamer
            from tldw_Server_API.app.core.TTS.waveform_streamer import stream_encoded_waveform
            async for chunk in stream_encoded_waveform(
                audio_tensor,
                format=request.format.value,
                sample_rate=self.sample_rate,
                channels=1,
                chunk_duration_sec=0.2,
            ):
                if chunk:
                    yield chunk

        finally:
            pass

    async def _generate_complete_chatterbox(
        self,
        request: TTSRequest,
        language_id: str,
        voice_reference_path: Optional[str],
        exaggeration: float,
    ) -> bytes:
        """Generate complete audio by aggregating streamed chunks."""
        out = bytearray()
        async for chunk in self._stream_audio_chatterbox(
            request, language_id, voice_reference_path, exaggeration
        ):
            out += chunk
        return bytes(out)

    def _map_emotion_to_exaggeration(self, emotion: Optional[str], intensity: float) -> float:
        """Map emotion label + intensity to upstream `exaggeration` scalar [0.0, 1.0]."""
        base_map = {
            None: self.default_exaggeration,
            "neutral": 0.5,
            "calm": 0.3,
            "sad": 0.4,
            "happy": 0.7,
            "excited": 0.7,
            "angry": 0.8,
            "surprised": 0.6,
            "fearful": 0.6,
            "disgusted": 0.6,
            "confused": 0.5,
        }
        base = base_map.get((emotion or "").lower(), self.default_exaggeration)
        # Scale around base with intensity [0.0..2.0]; clamp to [0.0..1.0]
        try:
            e = float(base) * float(max(0.0, min(2.0, intensity)))
        except _CHATTERBOX_NUMERIC_EXCEPTIONS:
            e = base
        return max(0.0, min(1.0, e))

    async def _prepare_voice_reference(self, voice_reference: bytes) -> Optional[str]:
        """
        Prepare voice reference audio for Chatterbox.

        Args:
            voice_reference: Voice reference audio bytes

        Returns:
            Path to temporary voice reference file or None if processing fails
        """
        try:
            import tempfile

            from tldw_Server_API.app.core.TTS.audio_utils import process_voice_reference_async

            # Process voice reference for Chatterbox requirements
            processed_audio, error = await process_voice_reference_async(
                voice_reference,
                provider='chatterbox',
                validate=True,
                convert=True
            )

            if error:
                logger.error("Voice reference processing failed")
                return None

            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False,
                prefix='chatterbox_voice_'
            ) as tmp_file:
                tmp_file.write(processed_audio)
                tmp_path = tmp_file.name

            logger.info("Voice reference prepared")
            return tmp_path

        except _CHATTERBOX_RUNTIME_EXCEPTIONS as e:
            logger.error(f"Failed to prepare voice reference ({type(e).__name__})")
            return None

    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to Chatterbox voice"""
        v = (voice_id or "").lower()
        # Default should map to narrator for friendlier baseline
        if v == "default":
            return "narrator"
        # Character and preset checks
        if v in self.CHARACTER_VOICES:
            return self.CHARACTER_VOICES[v]
        if v in self.VOICE_PRESETS:
            return v

        # Common mappings + synonyms used in tests
        voice_mappings = {
            "assistant": "sidekick",
            "friendly": "energetic",
            "soothing": "calm",
            "business": "professional",
            "neutral": "narrator",
            "evil": "villain",
            "wise": "sage",
            "funny": "comic_relief",
        }

        return voice_mappings.get(v, "narrator")

    def preprocess_text(self, text: str, **kwargs) -> str:
        """Preprocess text for Chatterbox"""
        # Basic preprocessing
        text = super().preprocess_text(text)

        # Chatterbox-specific preprocessing
        # Remove excessive punctuation that might affect emotion
        import re
        text = re.sub(r'[!]{2,}', '!', text)  # Multiple exclamations to one
        text = re.sub(r'[?]{2,}', '?', text)  # Multiple questions to one
        text = re.sub(r'\.{4,}', '...', text)  # Normalize ellipsis

        return text

    async def close(self):
        """Clean up resources"""
        self.model_en = None
        self.model_multi = None
        # Clear GPU cache if CUDA is available
        torch_mod = _get_torch(allow_import=False)
        if _torch_cuda_available(allow_import=False) and torch_mod is not None:
            torch_mod.cuda.empty_cache()
        await super().close()

    async def _cleanup_resources(self):
        """Adapter-specific cleanup invoked by base.close()."""
        # Clear commonly used attributes to satisfy tests and free memory
        for attr in ("model", "vocoder", "tokenizer", "processor"):
            if hasattr(self, attr):
                with contextlib.suppress(AttributeError, RuntimeError, TypeError, ValueError):
                    setattr(self, attr, None)
        # Ensure our lazy models are cleared as well
        self.model_en = None
        self.model_multi = None

    async def _get_model(self, language_id: str):
        """Get or load the appropriate upstream model for the language."""
        if self.use_multilingual and language_id != "en":
            if self.model_multi is None:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                logger.info(f"{self.provider_name}: Loading multilingual model on {self.device}")
                # If auto-download disabled, hint upstream to work offline
                if not self.auto_download:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                self.model_multi = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                # Disable watermark if configured
                if self.disable_watermark and hasattr(self.model_multi, 'watermarker'):
                    self.model_multi.watermarker = _NoopWatermarker()
                self.sample_rate = int(getattr(self.model_multi, 'sr', 24000))
                # Register model with resource manager (best-effort)
                try:
                    from ..tts_resource_manager import get_resource_manager
                    resource_manager = await get_resource_manager()
                    register_result = resource_manager.register_model(
                        provider=self.provider_name.lower(),
                        model_instance=self.model_multi,
                        cleanup_callback=self._cleanup_resources,
                        model_key=f"multi:{self.device}",
                    )
                    if asyncio.iscoroutine(register_result):
                        await register_result
                except _CHATTERBOX_RUNTIME_EXCEPTIONS:
                    pass
            return self.model_multi
        else:
            if self.model_en is None:
                from chatterbox.tts import ChatterboxTTS
                logger.info(f"{self.provider_name}: Loading English model on {self.device}")
                if not self.auto_download:
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                self.model_en = ChatterboxTTS.from_pretrained(device=self.device)
                if self.disable_watermark and hasattr(self.model_en, 'watermarker'):
                    self.model_en.watermarker = _NoopWatermarker()
                self.sample_rate = int(getattr(self.model_en, 'sr', 24000))
                # Register model with resource manager (best-effort)
                try:
                    from ..tts_resource_manager import get_resource_manager
                    resource_manager = await get_resource_manager()
                    register_result = resource_manager.register_model(
                        provider=self.provider_name.lower(),
                        model_instance=self.model_en,
                        cleanup_callback=self._cleanup_resources,
                        model_key=f"en:{self.device}",
                    )
                    if asyncio.iscoroutine(register_result):
                        await register_result
                except _CHATTERBOX_RUNTIME_EXCEPTIONS:
                    pass
            return self.model_en

#
# End of chatterbox_adapter.py
#######################################################################################################################

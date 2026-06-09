"""
chatterbox_adapter.py
Description: Chatterbox TTS adapter implementation (Resemble AI)
Updated to use upstream chatterbox package (v0.1.7 family API):
- Imports from chatterbox.tts and chatterbox.mtl_tts
- Supports Turbo via chatterbox.tts_turbo
- Supports multilingual via language_id
- Uses generate(...) waveform (no native streaming) and progressively streams encoded chunks
- Disables upstream watermarking by replacing watermarker with a no-op
"""

# Imports
import asyncio
import contextlib
import hashlib
import importlib
import inspect
import os
import sys
from collections import OrderedDict
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

# Third-party Imports
from loguru import logger

from ..tts_exceptions import (
    TTSModelLoadError,
)
from ..utils import parse_bool
from ..chatterbox_catalog import (
    CHATTERBOX_LANGUAGE_CODES,
    ChatterboxModelFamily,
    resolve_chatterbox_model_family,
)

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
_BF16_MODE_OFF = "off"
_BF16_MODE_ON = "on"
_BF16_MODE_AUTO = "auto"
_UNSUPPORTED_CONDITIONALS_CACHE = object()

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


def _select_default_device(*, allow_import: bool = True) -> str:
    """Select the best locally available Chatterbox runtime device."""
    if _torch_cuda_available(allow_import=allow_import):
        return "cuda"
    if _torch_mps_available(allow_import=allow_import):
        return "mps"
    return "cpu"


def _coerce_positive_int(value: Any, *, default: int) -> int:
    """Return a positive integer config value or a conservative default."""
    try:
        parsed = int(value)
    except _CHATTERBOX_NUMERIC_EXCEPTIONS:
        return default
    return parsed if parsed > 0 else default


def _coerce_float(value: Any, *, default: float) -> float:
    """Return a floating-point config value or a conservative default."""
    try:
        return float(value)
    except _CHATTERBOX_NUMERIC_EXCEPTIONS:
        return default


def _normalize_bf16_mode(value: Any) -> str:
    """Normalize BF16 config/env input into off, on, or auto."""
    if isinstance(value, str):
        token = value.strip().lower()
        if token == _BF16_MODE_AUTO:
            return _BF16_MODE_AUTO
        return _BF16_MODE_ON if parse_bool(token, default=False) else _BF16_MODE_OFF
    return _BF16_MODE_ON if parse_bool(value, default=False) else _BF16_MODE_OFF


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
    MULTILINGUAL_LANGS: set[str] = CHATTERBOX_LANGUAGE_CODES

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
                    self.device = _select_default_device(allow_import=True)
            elif pref in {"auto", "default"}:
                self.device = _select_default_device(allow_import=True)
            elif pref == "cpu":
                self.device = "cpu"
            else:
                # Unknown preference; fall back to CUDA/CPU
                self.device = _select_default_device(allow_import=True)
        else:
            self.device = _select_default_device(allow_import=True)

        # Provider settings
        self.use_multilingual = parse_bool(
            self.config.get("chatterbox_use_multilingual", self.config.get("use_multilingual")),
            default=False,
        )
        self.disable_watermark = parse_bool(
            self.config.get("chatterbox_disable_watermark", self.config.get("disable_watermark")),
            default=True,
        )
        bf16_config = self.config.get("chatterbox_use_bf16")
        if bf16_config is None:
            bf16_config = self.config.get("use_bf16")
        if bf16_config is None:
            bf16_config = os.getenv("TTS_BF16")
        self.bf16_mode = _normalize_bf16_mode(bf16_config)
        self.model_variant = self.config.get("chatterbox_variant", self.config.get("variant"))
        self.model_path = self.config.get("chatterbox_model_path", self.config.get("model_path"))
        self.multilingual_model_path = self.config.get(
            "chatterbox_multilingual_model_path",
            self.config.get("multilingual_model_path"),
        )
        self.turbo_model_path = self.config.get(
            "chatterbox_turbo_model_path",
            self.config.get("turbo_model_path"),
        )
        self.vc_model_path = self.config.get(
            "chatterbox_vc_model_path",
            self.config.get("vc_model_path"),
        )

        # Default sampling/expression parameters
        self.default_exaggeration = _coerce_float(
            self.config.get("chatterbox_default_exaggeration", self.config.get("default_exaggeration")),
            default=0.5,
        )
        self.default_cfg_weight = _coerce_float(
            self.config.get("chatterbox_cfg_weight", self.config.get("cfg_weight")),
            default=0.5,
        )
        self.default_temperature = _coerce_float(
            self.config.get("chatterbox_temperature", self.config.get("temperature")),
            default=0.8,
        )
        self.default_repetition_penalty = _coerce_float(
            self.config.get("chatterbox_repetition_penalty", self.config.get("repetition_penalty")),
            default=1.2,
        )
        self.default_min_p = _coerce_float(
            self.config.get("chatterbox_min_p", self.config.get("min_p")),
            default=0.05,
        )
        self.default_top_p = _coerce_float(
            self.config.get("chatterbox_top_p", self.config.get("top_p")),
            default=1.0,
        )
        try:
            self.conditionals_cache_size = max(
                0,
                int(
                    self.config.get(
                        "chatterbox_conditionals_cache_size",
                        self.config.get("conditionals_cache_size", 16),
                    )
                ),
            )
        except _CHATTERBOX_NUMERIC_EXCEPTIONS:
            self.conditionals_cache_size = 16

        # Model instances (lazy-loaded based on language)
        self.model_en = None  # ChatterboxTTS
        self.model_multi = None  # ChatterboxMultilingualTTS
        self.model_turbo = None  # ChatterboxTurboTTS
        self.model_vc = None  # ChatterboxVC
        self._conditionals_cache: OrderedDict[tuple[str, str, float], Any] = OrderedDict()

        # Audio parameters (sample rate will be taken from model)
        self.sample_rate = 24000

        # Target latency hint (progressive encoding)
        self.target_latency_ms = _coerce_positive_int(
            self.config.get(
                "chatterbox_target_latency_ms",
                self.config.get("target_latency_ms"),
            ),
            default=200,
        )

        # Auto-download toggle: config override > env overrides > default True
        cfg_auto = self.config.get("chatterbox_auto_download")
        if cfg_auto is None:
            cfg_auto = self.config.get("auto_download")
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
        family = resolve_chatterbox_model_family(
            config_variant=self.model_variant,
            use_multilingual=self.use_multilingual,
        )
        langs = self.MULTILINGUAL_LANGS if self.use_multilingual or family is ChatterboxModelFamily.MULTILINGUAL else {"en"}
        standard_generation_parameters = [
            "exaggeration",
            "cfg_weight",
            "temperature",
            "repetition_penalty",
            "min_p",
            "top_p",
            "top_k",
            "seed",
            "speed_factor",
        ]
        metadata = {
            "model_families": {
                "standard": {
                    "model_ids": ["chatterbox", "chatterbox-emotion"],
                    "languages": ["en"],
                    "supports_emotion_control": True,
                    "supports_voice_cloning": True,
                },
                "multilingual": {
                    "model_ids": ["chatterbox-multilingual"],
                    "languages": sorted(self.MULTILINGUAL_LANGS),
                    "supports_emotion_control": True,
                    "supports_voice_cloning": True,
                },
                "turbo": {
                    "model_ids": ["chatterbox-turbo"],
                    "languages": ["en"],
                    "supports_emotion_control": False,
                    "supports_voice_cloning": True,
                    "supports_paralinguistic_tags": True,
                    "paralinguistic_tags": ["[laugh]", "[cough]", "[chuckle]"],
                },
            },
            "supported_model_ids": [
                "chatterbox",
                "chatterbox-emotion",
                "chatterbox-multilingual",
                "chatterbox-turbo",
            ],
            "generation_parameters": {
                "standard": standard_generation_parameters,
                "multilingual": standard_generation_parameters,
                "turbo": [
                    "temperature",
                    "repetition_penalty",
                    "top_p",
                    "top_k",
                    "norm_loudness",
                    "speed_factor",
                ],
            },
            "speed_factor": {
                "request_fields": ["extra_params.speed_factor", "speed"],
                "requires_runtime_support": True,
            },
            "chunking": {
                "request_fields": ["extra_params.split_text", "extra_params.chunk_size"],
                "service_modes": ["non_streaming"],
            },
            "bf16": {
                "config_keys": ["chatterbox_use_bf16", "use_bf16"],
                "environment_variable": "TTS_BF16",
                "modes": [_BF16_MODE_OFF, _BF16_MODE_ON, _BF16_MODE_AUTO],
                "default": _BF16_MODE_OFF,
            },
            "voice_conversion": {
                "endpoint": "/api/v1/audio/voice-conversion",
                "model_id": "chatterbox-vc",
                "supports_streaming": True,
                "target_voice_optional": True,
            },
        }
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
            default_format=AudioFormat.WAV,
            metadata=metadata,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Chatterbox TTS"""
        if not await self.ensure_initialized():
            raise ValueError(f"{self.provider_name} not initialized")

        # Validate request against adapter capabilities
        is_valid, error = await self.validate_request(request)
        if not is_valid:
            raise ValueError(error)

        # Determine model family (standard, multilingual, or turbo)
        language_id = (request.language or "en").lower()
        family = self._resolve_request_family(request, language_id)
        model = await self._get_model(language_id, family=family)
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
        metadata = self._build_generation_metadata(
            request,
            language_id=language_id,
            family=family,
            exaggeration=exaggeration,
        )

        try:
            if request.stream:
                audio_stream = self._stream_audio_chatterbox(
                    request,
                    language_id,
                    voice_reference_path,
                    exaggeration,
                    family,
                )
                if voice_reference_path:
                    audio_stream = self._stream_with_voice_reference_cleanup(
                        audio_stream,
                        voice_reference_path,
                    )
                    voice_reference_path = None
                return TTSResponse(
                    audio_stream=audio_stream,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=request.voice or "default",
                    provider=self.provider_name,
                    metadata=metadata
                )
            else:
                audio_data = await self._generate_complete_chatterbox(
                    request,
                    language_id,
                    voice_reference_path,
                    exaggeration,
                    family
                )
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=request.voice or "default",
                    provider=self.provider_name,
                    metadata=metadata
                )
        finally:
            # Clean up temp voice reference
            if voice_reference_path:
                self._cleanup_voice_reference_path(voice_reference_path)

    @staticmethod
    def _cleanup_voice_reference_path(voice_reference_path: str) -> None:
        """Remove a temporary Chatterbox voice-reference file."""
        with contextlib.suppress(OSError, TypeError, ValueError):
            Path(voice_reference_path).unlink(missing_ok=True)

    async def _stream_with_voice_reference_cleanup(
        self,
        stream: AsyncGenerator[bytes, None],
        voice_reference_path: str,
    ) -> AsyncGenerator[bytes, None]:
        """Yield a Chatterbox stream and remove its temp voice reference afterwards."""
        try:
            async for chunk in stream:
                yield chunk
        finally:
            self._cleanup_voice_reference_path(voice_reference_path)

    def _stream_chunk_duration_sec(self) -> float:
        """Return the configured progressive stream chunk duration in seconds."""
        return _coerce_positive_int(self.target_latency_ms, default=200) / 1000.0

    def _resolve_request_family(self, request: TTSRequest, language_id: str) -> ChatterboxModelFamily:
        """Resolve the upstream Chatterbox family for one request."""
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        config_variant = (
            extras.get("chatterbox_variant")
            or extras.get("model_family")
            or extras.get("variant")
            or self.model_variant
        )
        use_multilingual = parse_bool(
            extras.get("use_multilingual", self.use_multilingual),
            default=self.use_multilingual,
        )
        model_hint = getattr(request, "model", None) or extras.get("model")
        return resolve_chatterbox_model_family(
            model_hint,
            language=language_id,
            config_variant=config_variant,
            use_multilingual=use_multilingual,
        )

    def _configured_model_path(self, family: ChatterboxModelFamily) -> Optional[str]:
        """Return the configured model path for a TTS family."""
        if family is ChatterboxModelFamily.MULTILINGUAL:
            return self.multilingual_model_path
        if family is ChatterboxModelFamily.TURBO:
            return self.turbo_model_path
        return self.model_path

    def _configured_vc_model_path(self) -> Optional[str]:
        """Return the configured model path for voice conversion."""
        return self.vc_model_path

    @staticmethod
    def _resolve_local_model_path(raw_path: Any) -> Optional[str]:
        """Return an existing local model path, leaving repo IDs on the pretrained loader."""
        if raw_path is None:
            return None
        value = str(raw_path).strip()
        if not value:
            return None
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            path = Path(value).expanduser()
            if path.exists():
                return str(path)
        return None

    def _load_chatterbox_runtime(self, runtime_cls: Any, *, model_path: Optional[str]) -> Any:
        """Load one Chatterbox runtime from a local path or upstream pretrained defaults."""
        local_model_path = self._resolve_local_model_path(model_path)
        if local_model_path:
            from_local = getattr(runtime_cls, "from_local", None)
            if callable(from_local):
                logger.info(f"{self.provider_name}: Loading Chatterbox model from local path")
                return from_local(local_model_path, device=self.device)
            raise TTSModelLoadError(
                "Configured Chatterbox local model path requires a runtime with from_local() support",
                provider=self.provider_name,
                details={"model_path": local_model_path},
            )

        if not self.auto_download:
            raise TTSModelLoadError(
                "Chatterbox auto-download is disabled; configure a local model path",
                provider=self.provider_name,
                details={"model_path": model_path},
            )
        return runtime_cls.from_pretrained(device=self.device)

    def _resolve_seed(self, request: TTSRequest) -> Optional[int]:
        """Return a normalized request seed from typed or extra parameters."""
        seed = getattr(request, "seed", None)
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        if seed is None:
            seed = extras.get("seed")
        if seed is None:
            seed = extras.get("generation_seed")
        try:
            return int(seed) if seed is not None else None
        except _CHATTERBOX_NUMERIC_EXCEPTIONS:
            return None

    def _resolve_speed_factor(self, request: TTSRequest) -> Optional[float]:
        """Return an upstream Chatterbox speed factor when explicitly requested."""
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        explicit_speed_factor = extras.get("speed_factor")
        if explicit_speed_factor is not None:
            try:
                speed_factor = float(explicit_speed_factor)
            except _CHATTERBOX_NUMERIC_EXCEPTIONS:
                return None
            return speed_factor if speed_factor > 0 else None

        request_speed = getattr(request, "speed", None)
        try:
            speed_factor = float(request_speed)
        except _CHATTERBOX_NUMERIC_EXCEPTIONS:
            return None
        if abs(speed_factor - 1.0) <= 1e-12 or speed_factor <= 0:
            return None
        return speed_factor

    def _resolve_turbo_ignored_controls(
        self,
        request: TTSRequest,
        *,
        exaggeration: float,
    ) -> list[str]:
        """List request controls intentionally ignored by Chatterbox Turbo."""
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        ignored: list[str] = []

        if "cfg_weight" in extras:
            ignored.append("cfg_weight")
        if request.emotion:
            ignored.extend(["emotion", "emotion_intensity"])
        if request.emotion or "exaggeration" in extras or exaggeration != self.default_exaggeration:
            ignored.append("exaggeration")
        if "min_p" in extras:
            ignored.append("min_p")

        return ignored

    def _build_generation_metadata(
        self,
        request: TTSRequest,
        *,
        language_id: str,
        family: ChatterboxModelFamily,
        exaggeration: float,
    ) -> dict[str, Any]:
        """Build response metadata for one Chatterbox generation request."""
        metadata: dict[str, Any] = {
            "language": language_id,
            "exaggeration": exaggeration,
            "model_family": family.value,
            "model": request.model,
            "seed": self._resolve_seed(request),
            "watermarked": not self.disable_watermark,
        }
        if family is ChatterboxModelFamily.TURBO:
            metadata["ignored_controls"] = self._resolve_turbo_ignored_controls(
                request,
                exaggeration=exaggeration,
            )
        return metadata

    def _build_generation_kwargs(
        self,
        request: TTSRequest,
        *,
        voice_reference_path: Optional[str],
        exaggeration: float,
        family: ChatterboxModelFamily,
    ) -> dict[str, Any]:
        """Build candidate kwargs for upstream Chatterbox generate calls."""
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        speed_factor = self._resolve_speed_factor(request)
        if family is ChatterboxModelFamily.TURBO:
            gen_kwargs: dict[str, Any] = {
                "temperature": extras.get("temperature", self.default_temperature),
                "repetition_penalty": extras.get("repetition_penalty", self.default_repetition_penalty),
                "top_p": extras.get("top_p", self.default_top_p),
            }
            if voice_reference_path:
                gen_kwargs["audio_prompt_path"] = voice_reference_path
            if speed_factor is not None:
                gen_kwargs["speed_factor"] = speed_factor
            if "top_k" in extras:
                gen_kwargs["top_k"] = extras["top_k"]
            if "norm_loudness" in extras:
                gen_kwargs["norm_loudness"] = extras["norm_loudness"]
            return {key: value for key, value in gen_kwargs.items() if value is not None}

        gen_kwargs: dict[str, Any] = {
            "exaggeration": exaggeration,
            "cfg_weight": extras.get("cfg_weight", self.default_cfg_weight),
            "temperature": extras.get("temperature", self.default_temperature),
            "repetition_penalty": extras.get("repetition_penalty", self.default_repetition_penalty),
            "min_p": extras.get("min_p", self.default_min_p),
            "top_p": extras.get("top_p", self.default_top_p),
        }
        if voice_reference_path:
            gen_kwargs["audio_prompt_path"] = voice_reference_path
        if speed_factor is not None:
            gen_kwargs["speed_factor"] = speed_factor
        if "top_k" in extras:
            gen_kwargs["top_k"] = extras["top_k"]
        seed = self._resolve_seed(request)
        if seed is not None:
            gen_kwargs["seed"] = seed

        return {key: value for key, value in gen_kwargs.items() if value is not None}

    def _filter_generation_kwargs(self, model: Any, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop kwargs unsupported by the concrete upstream generate signature."""
        try:
            signature = inspect.signature(model.generate)
        except (TypeError, ValueError):
            return gen_kwargs

        parameters = signature.parameters
        supports_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
        )
        if supports_kwargs:
            return gen_kwargs
        return {key: value for key, value in gen_kwargs.items() if key in parameters}

    @staticmethod
    def _hash_voice_reference_file(voice_reference_path: str) -> str:
        """Return a SHA256 digest for one voice reference file."""
        digest = hashlib.sha256()
        with Path(voice_reference_path).open("rb") as audio_file:
            for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    async def _voice_conditionals_cache_key(
        self,
        voice_reference_path: str,
        *,
        family: ChatterboxModelFamily,
        exaggeration: float,
    ) -> Optional[tuple[str, str, float]]:
        """Build a stable in-process cache key for one reference audio file."""
        try:
            digest = await asyncio.to_thread(self._hash_voice_reference_file, voice_reference_path)
            normalized_exaggeration = round(float(exaggeration), 4)
        except _CHATTERBOX_RUNTIME_EXCEPTIONS:
            return None
        return (family.value, digest, normalized_exaggeration)

    @classmethod
    def _conditionals_to_cpu_for_cache(cls, conditionals: Any) -> Any:
        """Return a cacheable copy of conditionals moved away from accelerator memory."""
        if conditionals is None or isinstance(conditionals, (str, bytes, int, float, bool)):
            return conditionals
        if isinstance(conditionals, dict):
            cached: dict[Any, Any] = {}
            for key, value in conditionals.items():
                cached_value = cls._conditionals_to_cpu_for_cache(value)
                if cached_value is _UNSUPPORTED_CONDITIONALS_CACHE:
                    return _UNSUPPORTED_CONDITIONALS_CACHE
                cached[key] = cached_value
            return cached
        if isinstance(conditionals, list):
            cached_list = []
            for value in conditionals:
                cached_value = cls._conditionals_to_cpu_for_cache(value)
                if cached_value is _UNSUPPORTED_CONDITIONALS_CACHE:
                    return _UNSUPPORTED_CONDITIONALS_CACHE
                cached_list.append(cached_value)
            return cached_list
        if isinstance(conditionals, tuple):
            cached_tuple = []
            for value in conditionals:
                cached_value = cls._conditionals_to_cpu_for_cache(value)
                if cached_value is _UNSUPPORTED_CONDITIONALS_CACHE:
                    return _UNSUPPORTED_CONDITIONALS_CACHE
                cached_tuple.append(cached_value)
            return tuple(cached_tuple)

        detached = conditionals
        detach = getattr(detached, "detach", None)
        if callable(detach):
            detached = detach()

        cpu = getattr(detached, "cpu", None)
        if callable(cpu):
            return cpu()

        to_device = getattr(detached, "to", None)
        if callable(to_device):
            return to_device("cpu")

        return _UNSUPPORTED_CONDITIONALS_CACHE

    def _conditionals_for_cache(self, conditionals: Any) -> Any:
        """Normalize conditionals before retaining them in the adapter LRU cache."""
        try:
            cacheable_conditionals = self._conditionals_to_cpu_for_cache(conditionals)
        except _CHATTERBOX_RUNTIME_EXCEPTIONS as exc:
            logger.debug(
                f"{self.provider_name}: voice conditionals cache normalization skipped ({type(exc).__name__})"
            )
            return None
        if cacheable_conditionals is _UNSUPPORTED_CONDITIONALS_CACHE:
            return None
        return cacheable_conditionals

    def _assign_conditionals(self, model: Any, conditionals: Any) -> None:
        """Assign cached Chatterbox conditionals, moving them to the adapter device when possible."""
        to_device = getattr(conditionals, "to", None)
        if callable(to_device):
            with contextlib.suppress(*_CHATTERBOX_RUNTIME_EXCEPTIONS):
                conditionals = to_device(self.device)
        setattr(model, "conds", conditionals)

    async def _prepare_cached_conditionals(
        self,
        model: Any,
        *,
        voice_reference_path: Optional[str],
        family: ChatterboxModelFamily,
        exaggeration: float,
    ) -> bool:
        """Prepare or reuse Chatterbox voice conditionals for a reference prompt."""
        if not voice_reference_path:
            return False

        prepare_conditionals = getattr(model, "prepare_conditionals", None)
        if not callable(prepare_conditionals):
            return False

        cache_key = await self._voice_conditionals_cache_key(
            voice_reference_path,
            family=family,
            exaggeration=exaggeration,
        )
        if cache_key is not None and cache_key in self._conditionals_cache:
            conditionals = self._conditionals_cache[cache_key]
            self._conditionals_cache.move_to_end(cache_key)
            self._assign_conditionals(model, conditionals)
            return True

        kwargs: dict[str, Any] = {}
        try:
            signature = inspect.signature(prepare_conditionals)
            parameters = signature.parameters
            supports_kwargs = any(
                param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
            )
            if supports_kwargs or "exaggeration" in parameters:
                kwargs["exaggeration"] = exaggeration
        except (TypeError, ValueError):
            kwargs["exaggeration"] = exaggeration

        try:
            maybe_result = prepare_conditionals(voice_reference_path, **kwargs)
            if inspect.isawaitable(maybe_result):
                await maybe_result
        except _CHATTERBOX_RUNTIME_EXCEPTIONS as exc:
            logger.debug(
                f"{self.provider_name}: voice conditionals preparation unavailable ({type(exc).__name__}); "
                "falling back to audio_prompt_path"
            )
            return False

        conditionals = getattr(model, "conds", None)
        if cache_key is not None and conditionals is not None and self.conditionals_cache_size > 0:
            cacheable_conditionals = self._conditionals_for_cache(conditionals)
            if cacheable_conditionals is not None:
                self._conditionals_cache[cache_key] = cacheable_conditionals
                self._conditionals_cache.move_to_end(cache_key)
                while len(self._conditionals_cache) > self.conditionals_cache_size:
                    self._conditionals_cache.popitem(last=False)
        return True

    def _apply_generation_seed(self, seed: Optional[int]) -> None:
        """Best-effort deterministic seed for upstream Chatterbox sampling."""
        if seed is None:
            return
        torch_mod = _get_torch(allow_import=True)
        if torch_mod is None:
            return
        with contextlib.suppress(*_CHATTERBOX_RUNTIME_EXCEPTIONS):
            torch_mod.manual_seed(seed)
        with contextlib.suppress(*_CHATTERBOX_RUNTIME_EXCEPTIONS):
            if hasattr(torch_mod, "cuda"):
                torch_mod.cuda.manual_seed_all(seed)

    def _bf16_autocast_device_type(self) -> str:
        """Return the torch autocast device token for the selected runtime device."""
        device = str(self.device or "cpu").split(":", 1)[0].lower()
        return device if device in {"cuda", "cpu", "mps"} else "cpu"

    def _should_use_bf16(self) -> bool:
        """Return whether the current configuration should use BF16 for TTS generation."""
        if self.bf16_mode == _BF16_MODE_OFF:
            return False

        torch_mod = _get_torch(allow_import=True)
        if torch_mod is None or not hasattr(torch_mod, "bfloat16"):
            return False

        if self.bf16_mode == _BF16_MODE_ON:
            return True

        if self.bf16_mode != _BF16_MODE_AUTO:
            return False

        if self._bf16_autocast_device_type() != "cuda":
            return False

        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is None:
            return False
        try:
            if not bool(cuda_mod.is_available()):
                return False
        except _CHATTERBOX_RUNTIME_EXCEPTIONS:
            return False

        is_supported = getattr(cuda_mod, "is_bf16_supported", None)
        if callable(is_supported):
            try:
                return bool(is_supported())
            except _CHATTERBOX_RUNTIME_EXCEPTIONS:
                return False
        return True

    def _prepare_bf16_runtime(self, model: Any) -> None:
        """Best-effort T3 BF16 preparation for Chatterbox TTS models."""
        if not self._should_use_bf16():
            return
        if getattr(model, "_tldw_bf16_prepared", None) is True:
            return

        torch_mod = _get_torch(allow_import=True)
        dtype = getattr(torch_mod, "bfloat16", None) if torch_mod is not None else None
        if dtype is None:
            return

        t3_module = getattr(model, "t3", None)
        to_dtype = getattr(t3_module, "to", None)
        if not callable(to_dtype):
            return

        try:
            converted = to_dtype(dtype=dtype)
        except _CHATTERBOX_RUNTIME_EXCEPTIONS as exc:
            logger.debug(
                f"{self.provider_name}: BF16 T3 preparation skipped ({type(exc).__name__})"
            )
            return

        if converted is not None:
            with contextlib.suppress(*_CHATTERBOX_RUNTIME_EXCEPTIONS):
                setattr(model, "t3", converted)
        with contextlib.suppress(*_CHATTERBOX_RUNTIME_EXCEPTIONS):
            setattr(model, "_tldw_bf16_prepared", True)

    def _bf16_autocast_context(self):
        """Return a torch autocast context for BF16 generation, or a no-op context."""
        if not self._should_use_bf16():
            return contextlib.nullcontext()

        torch_mod = _get_torch(allow_import=True)
        autocast = getattr(torch_mod, "autocast", None) if torch_mod is not None else None
        dtype = getattr(torch_mod, "bfloat16", None) if torch_mod is not None else None
        if not callable(autocast) or dtype is None:
            return contextlib.nullcontext()

        try:
            return autocast(device_type=self._bf16_autocast_device_type(), dtype=dtype)
        except _CHATTERBOX_RUNTIME_EXCEPTIONS as exc:
            logger.debug(
                f"{self.provider_name}: BF16 autocast unavailable ({type(exc).__name__}); using default precision"
            )
            return contextlib.nullcontext()

    async def _stream_audio_chatterbox(
        self,
        request: TTSRequest,
        language_id: str,
        voice_reference_path: Optional[str],
        exaggeration: float,
        family: ChatterboxModelFamily,
    ) -> AsyncGenerator[bytes, None]:
        """Generate waveform with upstream model, progressively encode and stream bytes."""
        model = await self._get_model(language_id, family=family)

        try:
            prepared_conditionals = await self._prepare_cached_conditionals(
                model,
                voice_reference_path=voice_reference_path,
                family=family,
                exaggeration=exaggeration,
            )
            # Prepare kwargs for upstream generate
            gen_kwargs = self._build_generation_kwargs(
                request,
                voice_reference_path=None if prepared_conditionals else voice_reference_path,
                exaggeration=exaggeration,
                family=family,
            )
            self._apply_generation_seed(self._resolve_seed(request))
            filtered_kwargs = self._filter_generation_kwargs(model, gen_kwargs)
            self._prepare_bf16_runtime(model)

            # Generate full waveform tensor (1, N)
            with self._bf16_autocast_context():
                if family is ChatterboxModelFamily.MULTILINGUAL:
                    audio_tensor = model.generate(
                        self.preprocess_text(request.text),
                        language_id=language_id,
                        **filtered_kwargs,
                    )
                else:
                    audio_tensor = model.generate(
                        self.preprocess_text(request.text),
                        **filtered_kwargs,
                    )
            # Stream using shared waveform streamer
            from tldw_Server_API.app.core.TTS.waveform_streamer import stream_encoded_waveform
            async for chunk in stream_encoded_waveform(
                audio_tensor,
                format=request.format.value,
                sample_rate=self.sample_rate,
                channels=1,
                chunk_duration_sec=self._stream_chunk_duration_sec(),
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
        family: ChatterboxModelFamily,
    ) -> bytes:
        """Generate complete audio by aggregating streamed chunks."""
        out = bytearray()
        async for chunk in self._stream_audio_chatterbox(
            request, language_id, voice_reference_path, exaggeration, family
        ):
            out += chunk
        return bytes(out)

    async def convert_voice(
        self,
        *,
        source_audio_path: str,
        target_voice_path: Optional[str],
        format: AudioFormat = AudioFormat.WAV,
        stream: bool = True,
    ) -> TTSResponse:
        """Convert source speech into the target voice using ChatterboxVC."""
        model = await self._get_vc_model()
        self.sample_rate = int(getattr(model, "sr", 24000))
        metadata = {
            "mode": "voice_conversion",
            "model_family": "voice_conversion",
            "target_voice_path_provided": bool(target_voice_path),
            "watermarked": not self.disable_watermark,
        }

        if stream:
            return TTSResponse(
                audio_stream=self._stream_voice_conversion_chatterbox(
                    source_audio_path=source_audio_path,
                    target_voice_path=target_voice_path,
                    format=format,
                ),
                format=format,
                sample_rate=self.sample_rate,
                channels=1,
                voice_used="target_reference" if target_voice_path else "default",
                provider=self.provider_name,
                model="chatterbox-vc",
                metadata=metadata,
            )

        audio_data = await self._generate_complete_voice_conversion(
            source_audio_path=source_audio_path,
            target_voice_path=target_voice_path,
            format=format,
        )
        return TTSResponse(
            audio_data=audio_data,
            format=format,
            sample_rate=self.sample_rate,
            channels=1,
            voice_used="target_reference" if target_voice_path else "default",
            provider=self.provider_name,
            model="chatterbox-vc",
            metadata=metadata,
        )

    async def _stream_voice_conversion_chatterbox(
        self,
        *,
        source_audio_path: str,
        target_voice_path: Optional[str],
        format: AudioFormat,
    ) -> AsyncGenerator[bytes, None]:
        """Generate a VC waveform and stream encoded audio bytes."""
        model = await self._get_vc_model()
        self.sample_rate = int(getattr(model, "sr", 24000))
        audio_tensor = model.generate(
            audio=source_audio_path,
            target_voice_path=target_voice_path,
        )

        from tldw_Server_API.app.core.TTS.waveform_streamer import stream_encoded_waveform

        async for chunk in stream_encoded_waveform(
            audio_tensor,
            format=format.value,
            sample_rate=self.sample_rate,
            channels=1,
            chunk_duration_sec=self._stream_chunk_duration_sec(),
        ):
            if chunk:
                yield chunk

    async def _generate_complete_voice_conversion(
        self,
        *,
        source_audio_path: str,
        target_voice_path: Optional[str],
        format: AudioFormat,
    ) -> bytes:
        """Generate complete VC audio by aggregating encoded stream chunks."""
        out = bytearray()
        async for chunk in self._stream_voice_conversion_chatterbox(
            source_audio_path=source_audio_path,
            target_voice_path=target_voice_path,
            format=format,
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
        self.model_turbo = None
        self.model_vc = None
        self._conditionals_cache.clear()
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
        self.model_turbo = None
        self.model_vc = None
        self._conditionals_cache.clear()

    async def _get_model(
        self,
        language_id: str,
        family: Optional[Union[str, ChatterboxModelFamily]] = None,
    ):
        """Get or load the appropriate upstream model for the language."""
        if family is None:
            family = resolve_chatterbox_model_family(
                language=language_id,
                config_variant=self.model_variant,
                use_multilingual=self.use_multilingual,
            )
        family = ChatterboxModelFamily(family)

        if family is ChatterboxModelFamily.MULTILINGUAL:
            if self.model_multi is None:
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                logger.info(f"{self.provider_name}: Loading multilingual model on {self.device}")
                self.model_multi = self._load_chatterbox_runtime(
                    ChatterboxMultilingualTTS,
                    model_path=self._configured_model_path(ChatterboxModelFamily.MULTILINGUAL),
                )
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
        elif family is ChatterboxModelFamily.TURBO:
            if self.model_turbo is None:
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                logger.info(f"{self.provider_name}: Loading Turbo model on {self.device}")
                self.model_turbo = self._load_chatterbox_runtime(
                    ChatterboxTurboTTS,
                    model_path=self._configured_model_path(ChatterboxModelFamily.TURBO),
                )
                if self.disable_watermark and hasattr(self.model_turbo, 'watermarker'):
                    self.model_turbo.watermarker = _NoopWatermarker()
                self.sample_rate = int(getattr(self.model_turbo, 'sr', 24000))
                try:
                    from ..tts_resource_manager import get_resource_manager
                    resource_manager = await get_resource_manager()
                    register_result = resource_manager.register_model(
                        provider=self.provider_name.lower(),
                        model_instance=self.model_turbo,
                        cleanup_callback=self._cleanup_resources,
                        model_key=f"turbo:{self.device}",
                    )
                    if asyncio.iscoroutine(register_result):
                        await register_result
                except _CHATTERBOX_RUNTIME_EXCEPTIONS:
                    pass
            return self.model_turbo
        else:
            if self.model_en is None:
                from chatterbox.tts import ChatterboxTTS
                logger.info(f"{self.provider_name}: Loading English model on {self.device}")
                self.model_en = self._load_chatterbox_runtime(
                    ChatterboxTTS,
                    model_path=self._configured_model_path(ChatterboxModelFamily.STANDARD),
                )
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

    async def _get_vc_model(self):
        """Get or load the upstream Chatterbox voice conversion model."""
        if self.model_vc is None:
            from chatterbox.vc import ChatterboxVC

            logger.info(f"{self.provider_name}: Loading voice conversion model on {self.device}")
            self.model_vc = self._load_chatterbox_runtime(
                ChatterboxVC,
                model_path=self._configured_vc_model_path(),
            )
            if self.disable_watermark and hasattr(self.model_vc, 'watermarker'):
                self.model_vc.watermarker = _NoopWatermarker()
            self.sample_rate = int(getattr(self.model_vc, 'sr', 24000))
            try:
                from ..tts_resource_manager import get_resource_manager
                resource_manager = await get_resource_manager()
                register_result = resource_manager.register_model(
                    provider=self.provider_name.lower(),
                    model_instance=self.model_vc,
                    cleanup_callback=self._cleanup_resources,
                    model_key=f"vc:{self.device}",
                )
                if asyncio.iscoroutine(register_result):
                    await register_result
            except _CHATTERBOX_RUNTIME_EXCEPTIONS:
                pass
        return self.model_vc

#
# End of chatterbox_adapter.py
#######################################################################################################################

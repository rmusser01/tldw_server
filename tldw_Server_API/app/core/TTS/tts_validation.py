# tts_validation.py
# Description: Input validation and sanitization for TTS requests
#
# Imports
import base64
import html
import math
import re
import unicodedata
from typing import Any, Optional, Union

#
# Third-party Imports
from loguru import logger

from .adapters.base import AudioFormat, TTSRequest
from .chatterbox_catalog import (
    CHATTERBOX_LANGUAGE_CODES,
    ChatterboxModelFamily,
    resolve_chatterbox_model_family,
)

#
# Local Imports
from .tts_exceptions import (
    TTSInvalidInputError,
    TTSInvalidVoiceReferenceError,
    TTSTextTooLongError,
    TTSUnsupportedFormatError,
    TTSUnsupportedLanguageError,
    TTSValidationError,
    TTSVoiceNotFoundError,
)
from .utils import parse_bool, resolve_qwen3_runtime_name
from .voice_manager import PROVIDER_REQUIREMENTS

#
#######################################################################################################################
#
# Provider Limits

OMNIVOICE_GENERATION_PARAM_RANGES = {
    "num_step": (int, 1, 128),
    "guidance_scale": (float, 0.0, 30.0),
    "denoise": (bool, None, None),
    "t_shift": (float, None, None),
    "position_temperature": (float, 0.0, 10.0),
    "class_temperature": (float, 0.0, 10.0),
    "layer_penalty_factor": (float, 0.0, 10.0),
    "duration": (float, 0.0, None),
    "speed": (float, 0.0, 4.0),
    "postprocess_output": (bool, None, None),
    "preprocess_prompt": (bool, None, None),
    "audio_chunk_duration": (float, 0.0, None),
    "audio_chunk_threshold": (float, 0.0, None),
}
OMNIVOICE_INSTRUCT_KEYS = ("instruct", "voice_design", "voice_description")
OMNIVOICE_LANGUAGE_KEYS = ("language_id", "language")
OMNIVOICE_SUPPORTED_NON_GENERATION_KEYS = {
    "mode",
    "omnivoice_mode",
    "reference_text",
    "ref_text",
    "voice_reference_text",
    "target_sample_rate",
    "sample_rate",
    "reference_duration_min",
    "request_id",
    "correlation_id",
    *OMNIVOICE_INSTRUCT_KEYS,
    *OMNIVOICE_LANGUAGE_KEYS,
}
OMNIVOICE_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
OMNIVOICE_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


class ProviderLimits:
    """Provider-specific limits and constraints"""

    LIMITS = {
        "openai": {
            "max_text_length": 4096,
            "languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"],
            "valid_voices": {"alloy", "echo", "fable", "onyx", "nova", "shimmer"},
            "valid_formats": {"mp3", "opus", "aac", "flac", "wav", "pcm"},
            "min_speed": 0.25,
            "max_speed": 4.0
        },
        "elevenlabs": {
            "max_text_length": 5000,
            "valid_formats": {"mp3", "pcm", "ulaw"},
            "min_stability": 0.0,
            "max_stability": 1.0,
            "min_similarity": 0.0,
            "max_similarity": 1.0
        },
        "kokoro": {
            "max_text_length": 1000000,
            "languages": ["en"],
            "valid_formats": {"wav", "mp3"},
            "min_speed": 0.5,
            "max_speed": 2.0
        },
        "higgs": {
            "max_text_length": 8000,
            "valid_formats": {"wav", "mp3", "opus"},
            "min_speed": 0.5,
            "max_speed": 2.0
        },
        "dia": {
            "max_text_length": 10000,
            "valid_formats": {"wav", "mp3"},
            "min_speed": 0.5,
            "max_speed": 2.0,
            "max_speakers": 4
        },
        "chatterbox": {
            "max_text_length": 10000,
            "valid_formats": {"wav", "mp3", "opus", "flac", "pcm"},
            "min_speed": 0.5,
            "max_speed": 2.0
        },
        "vibevoice": {
            "max_text_length": 10000,
            "valid_formats": {"wav", "mp3"},
            "min_speed": 0.5,
            "max_speed": 2.0,
            "max_speakers": 4
        },
        "vibevoice_realtime": {
            "max_text_length": 8192,
            "languages": ["en"],
            "valid_formats": {"pcm", "wav", "mp3", "opus", "flac"},
            "min_speed": 0.25,
            "max_speed": 4.0
        },
        "index_tts": {
            "max_text_length": 4000,
            "languages": ["en", "zh"],
            "valid_formats": {"mp3", "wav"},
            "min_speed": 0.5,
            "max_speed": 2.0
        },
        "supertonic": {
            "max_text_length": 15000,
            "languages": ["en"],
            "valid_formats": {"mp3", "wav"},
            "min_speed": 0.9,
            "max_speed": 1.5
        },
        "supertonic2": {
            "max_text_length": 15000,
            "languages": ["en", "ko", "es", "pt", "fr"],
            "valid_formats": {"mp3", "wav"},
            "min_speed": 0.9,
            "max_speed": 1.5
        },
        "pocket_tts": {
            "max_text_length": 5000,
            "languages": ["en"],
            "valid_formats": {"mp3", "wav", "opus", "flac", "pcm", "aac"},
            "min_speed": 0.25,
            "max_speed": 4.0
        },
        "pocket_tts_cpp": {
            "max_text_length": 5000,
            "languages": ["en"],
            "valid_formats": {"mp3", "wav", "opus", "flac", "pcm", "aac"},
            "min_speed": 0.25,
            "max_speed": 4.0
        },
        "kitten_tts": {
            "max_text_length": 5000,
            "languages": ["en"],
            "valid_formats": {"mp3", "wav", "pcm"},
            "min_speed": 0.25,
            "max_speed": 4.0,
        },
        "lux_tts": {
            "max_text_length": 5000,
            "languages": ["en"],
            "valid_formats": {"mp3", "wav", "flac", "opus", "aac", "pcm"},
            "min_speed": 0.25,
            "max_speed": 4.0
        },
        "qwen3_tts": {
            "max_text_length": 5000,
            "languages": ["auto", "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
            "valid_formats": {"mp3", "opus", "aac", "wav", "pcm"},
            "min_speed": 0.25,
            "max_speed": 4.0
        },
        "omnivoice": {
            "max_text_length": 5000,
            "languages": ["en"],
            "valid_formats": {"mp3", "opus", "aac", "flac", "wav", "pcm"},
            "min_speed": 0.25,
            "max_speed": 4.0,
        }
    }

    @classmethod
    def get_limits(cls, provider: str) -> dict[str, Any]:
        """Get limits for a specific provider"""
        # Return default limits if provider not found
        default_limits = {
            "max_text_length": 5000,
            "languages": ["en"],
            "valid_formats": {"mp3", "wav"},
            "min_speed": 0.5,
            "max_speed": 2.0
        }
        return cls.LIMITS.get(provider, default_limits)

    @classmethod
    def get_max_text_length(cls, provider: str) -> int:
        """Get maximum text length for provider"""
        limits = cls.get_limits(provider)
        return limits.get("max_text_length", 5000)  # Default 5000

    @classmethod
    def is_valid_voice(cls, provider: str, voice: str) -> bool:
        """Check if voice is valid for provider"""
        limits = cls.get_limits(provider)
        valid_voices = limits.get("valid_voices")
        if valid_voices is None:
            return True  # No restriction
        return voice in valid_voices

    @classmethod
    def is_valid_format(cls, provider: str, format: str) -> bool:
        """Check if format is valid for provider"""
        limits = cls.get_limits(provider)
        valid_formats = limits.get("valid_formats", {"mp3", "wav"})
        return format.lower() in valid_formats


#
# Input Validation and Sanitization

class TTSInputValidator:
    """
    Comprehensive input validator for TTS requests.
    Handles text sanitization, format validation, and security checks.
    """

    # Security patterns to detect potential injection attacks
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'vbscript:',                 # VBScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'expression\s*\(',           # CSS expressions
        r'@import',                   # CSS imports
        r'\\x[0-9a-fA-F]{2}',        # Hex escapes
        r'\\u[0-9a-fA-F]{4}',        # Unicode escapes
        r'&#[0-9]+;',                 # HTML numeric entities
        r'&#x[0-9a-fA-F]+;',         # HTML hex entities
        # SQL injection patterns
        r"'\s*(OR|AND)\s+'?\d+'?\s*=\s*'?\d+'?",  # SQL injection
        r';\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE)\s+', # SQL commands
        r'UNION\s+SELECT',            # Union select
        r"--\s*$",                    # SQL comments
        # Command injection patterns
        r';\s*rm\s+-rf',              # Unix file deletion
        r'\|\s*cat\s+/etc/',          # Unix file reading
        r'`[^`]+`',                   # Command substitution
        r'\$\([^)]+\)',               # Command substitution
        r'&\s*del\s+',                # Windows file deletion
        r'whoami',                    # System info command
        r'curl\s+evil',               # Malicious downloads
        # Path traversal
        r'\.\./\.\.',                 # Path traversal
        r'\.\.\\\.\.\\',              # Windows path traversal
    ]

    # Compiled regex patterns for performance
    DANGEROUS_REGEX = [re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in DANGEROUS_PATTERNS]

    # Maximum text length per provider (characters).
    # For unknown providers, default to a more permissive 5000 characters to
    # avoid surprising rejections when the underlying engine can handle more.
    MAX_TEXT_LENGTHS = {
        "openai": 4096,
        "elevenlabs": 5000,
        "kokoro": 1000000,
        "higgs": 50000,
        "dia": 30000,
        "chatterbox": 10000,
        "vibevoice": 15000,
        "vibevoice_realtime": 8192,
        "neutts": 5000,
        "index_tts": 4000,
        "supertonic": 15000,
        "supertonic2": 15000,
        "pocket_tts": 5000,
        "pocket_tts_cpp": 5000,
        "kitten_tts": 5000,
        "echo_tts": 768,
        "lux_tts": 5000,
        "qwen3_tts": 5000,
        "omnivoice": 5000,
        "default": 5000,
    }

    # Maximum UTF-8 byte length per provider (excluding BOS token).
    MAX_TEXT_BYTES = {
        "echo_tts": 767,
    }

    # Providers that require a voice reference audio input
    REQUIRES_VOICE_REFERENCE = {
        "echo_tts",
        "lux_tts",
    }

    # Supported languages by provider
    SUPPORTED_LANGUAGES = {
        "openai": {"en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"},
        "elevenlabs": {"en", "es", "fr", "de", "it", "pt", "pl", "hi", "ar", "zh", "ja", "ko"},
        "kokoro": {"en"},
        "higgs": {"en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"},
        "dia": {"en"},
        "chatterbox": {"en"},
        "vibevoice": {"en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"},
        "vibevoice_realtime": {"en"},
        "neutts": {"en", "en-us", "en-gb"},
        "index_tts": {"en", "zh"},
        "supertonic": {"en"},
        "supertonic2": {"en", "ko", "es", "pt", "fr"},
        "pocket_tts": {"en"},
        "pocket_tts_cpp": {"en"},
        "kitten_tts": {"en"},
        "echo_tts": {"en"},
        "lux_tts": {"en"},
        "qwen3_tts": {"auto", "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"},
    }

    # Supported audio formats by provider
    SUPPORTED_FORMATS = {
        "openai": {AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.FLAC, AudioFormat.WAV, AudioFormat.PCM},
        "elevenlabs": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OPUS},
        "kokoro": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OPUS},
        "higgs": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.FLAC},
        "dia": {AudioFormat.MP3, AudioFormat.WAV},
        "chatterbox": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OPUS, AudioFormat.FLAC, AudioFormat.PCM},
        "vibevoice": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.FLAC, AudioFormat.OPUS},
        "vibevoice_realtime": {AudioFormat.PCM, AudioFormat.WAV, AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.FLAC},
        "neutts": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OPUS, AudioFormat.FLAC, AudioFormat.PCM},
        "index_tts": {AudioFormat.MP3, AudioFormat.WAV},
        "supertonic": {AudioFormat.MP3, AudioFormat.WAV},
        "supertonic2": {AudioFormat.MP3, AudioFormat.WAV},
        "pocket_tts": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OPUS, AudioFormat.FLAC, AudioFormat.PCM, AudioFormat.AAC},
        "pocket_tts_cpp": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.OPUS, AudioFormat.FLAC, AudioFormat.PCM, AudioFormat.AAC},
        "kitten_tts": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.PCM},
        "echo_tts": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.FLAC, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.PCM},
        "lux_tts": {AudioFormat.MP3, AudioFormat.WAV, AudioFormat.FLAC, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.PCM},
        "qwen3_tts": {AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.WAV, AudioFormat.PCM},
        "omnivoice": {AudioFormat.MP3, AudioFormat.OPUS, AudioFormat.AAC, AudioFormat.FLAC, AudioFormat.WAV, AudioFormat.PCM},
    }

    # Voice reference file validation
    VOICE_REF_MAX_SIZE = 50 * 1024 * 1024  # 50MB
    VOICE_REF_MAX_DURATION = 300  # 5 minutes
    VOICE_REF_ALLOWED_FORMATS = {".mp3", ".wav", ".flac", ".opus", ".m4a", ".ogg"}
    VOICE_REF_ALLOWED_MIME_TYPES = {
        "audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac",
        "audio/opus", "audio/ogg", "audio/mp4", "audio/x-m4a"
    }
    EMO_REF_MAX_SIZE = 20 * 1024 * 1024  # 20MB limit for emotion references
    VOICE_CLONE_PROMPT_MAX_KB_DEFAULT = 256
    QWEN3_CUSTOMVOICE_SPEAKERS = {
        "vivian",
        "serena",
        "uncle_fu",
        "dylan",
        "eric",
        "ryan",
        "aiden",
        "ono_anna",
        "sohee",
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize the validator with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.strict_mode = self.config.get("strict_validation", True)
        self.max_text_length_override = self.config.get("max_text_length")
        logger.debug(f"TTSInputValidator initialized (strict_mode={self.strict_mode})")

    def _get_provider_setting(self, provider: Optional[str], key: str) -> Optional[Any]:
        if not provider or not isinstance(self.config, dict):
            return None
        providers_cfg = self.config.get("providers")
        if isinstance(providers_cfg, dict):
            provider_cfg = providers_cfg.get(provider)
            if isinstance(provider_cfg, dict) and key in provider_cfg:
                return provider_cfg.get(key)
        # Legacy/flat config key fallback
        legacy_key = f"{provider}_{key}"
        if legacy_key in self.config:
            return self.config.get(legacy_key)
        return None

    def _coerce_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return None
        return coerced

    def _coerce_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _resolve_qwen3_runtime(self, provider: Optional[str]) -> str:
        configured = self._get_provider_setting(provider, "runtime")
        return resolve_qwen3_runtime_name(configured)

    def _decode_base64_payload(self, payload: str) -> bytes:
        """Decode base64 payload with optional data URL prefix."""
        if "," in payload:
            payload = payload.split(",", 1)[1]
        try:
            return base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise TTSInvalidInputError("voice_clone_prompt must be valid base64") from exc

    def _validate_voice_clone_prompt(self, payload: Any, provider: Optional[str]) -> None:
        if payload is None:
            return
        max_kb = self._coerce_int(self._get_provider_setting(provider, "voice_clone_prompt_max_kb"))
        if max_kb is None or max_kb <= 0:
            max_kb = self.VOICE_CLONE_PROMPT_MAX_KB_DEFAULT
        max_bytes = max_kb * 1024

        decoded: Optional[bytes] = None
        if isinstance(payload, (bytes, bytearray)):
            decoded = bytes(payload)
        elif isinstance(payload, str):
            decoded = self._decode_base64_payload(payload)
        elif isinstance(payload, dict):
            fmt = payload.get("format")
            data_b64 = payload.get("data_b64")
            if fmt and fmt != "qwen3_tts_prompt_v1":
                raise TTSInvalidInputError("voice_clone_prompt format must be 'qwen3_tts_prompt_v1'")
            if not isinstance(data_b64, str) or not data_b64.strip():
                raise TTSInvalidInputError("voice_clone_prompt must include data_b64")
            decoded = self._decode_base64_payload(data_b64)
        else:
            raise TTSInvalidInputError(
                "voice_clone_prompt must be base64 string or {format,data_b64} object"
            )

        if decoded is None:
            raise TTSInvalidInputError("voice_clone_prompt payload could not be decoded")
        if max_bytes and len(decoded) > max_bytes:
            raise TTSInvalidInputError(
                f"voice_clone_prompt too large: {len(decoded)} bytes (max {max_bytes})"
            )

    def _normalize_qwen3_speaker(self, speaker: str) -> str:
        normalized = speaker.strip().lower()
        normalized = normalized.replace(" ", "_").replace("-", "_")
        normalized = re.sub(r"_+", "_", normalized)
        return normalized

    def sanitize_text(self, text: str, provider: Optional[str] = None) -> str:
        """
        Sanitize input text for TTS generation.

        Args:
            text: Input text to sanitize
            provider: TTS provider name for provider-specific rules

        Returns:
            Sanitized text

        Raises:
            TTSInvalidInputError: If text contains dangerous content
        """
        if not text or not text.strip():
            raise TTSInvalidInputError("Text cannot be empty or whitespace only")

        original_text = text

        # 1. Normalize Unicode characters
        text = unicodedata.normalize('NFKC', text)

        # 2. Check for dangerous patterns and remove them
        for pattern in self.DANGEROUS_REGEX:
            if pattern.search(text):
                logger.warning(f"Dangerous pattern detected and removed: {pattern.pattern[:50]}")
                # Always remove dangerous patterns for security
                text = pattern.sub('', text)

                # In strict mode, also raise an error
                if self.strict_mode:
                    raise TTSInvalidInputError(
                        "Text contains potentially dangerous content",
                        details={"pattern": pattern.pattern[:50]}
                    )

        # 3. Remove HTML tags - TTS doesn't need HTML
        # Strip all HTML tags since they shouldn't be spoken
        text = re.sub(r'<[^>]+>', '', text)
        # Also remove any remaining HTML entities
        text = html.unescape(text)

        # 4. Remove or replace problematic characters
        text = self._clean_control_characters(text)

        # 5. Provider-specific sanitization
        if provider:
            text = self._provider_specific_sanitization(text, provider)

        # 6. Final validation
        if len(text.strip()) == 0:
            raise TTSInvalidInputError("Text became empty after sanitization")

        logger.debug(f"Text sanitized: {len(original_text)} -> {len(text)} chars")
        return text.strip()

    def validate_text_length(self, text: str, provider: Optional[str] = None, max_length: Optional[int] = None):
        """Public method to validate text length"""
        if max_length:
            # Override max length for this validation
            old_max = self.max_text_length_override
            self.max_text_length_override = max_length
            try:
                return self._validate_text(text, provider)
            finally:
                self.max_text_length_override = old_max
        else:
            return self._validate_text(text, provider)

    def validate_language(self, language: Optional[str], provider: Optional[Union[str, list[str]]] = None):
        """Public method to validate language"""
        # None language is valid (will use default)
        if language is None:
            return

        # Handle test case where supported languages are passed directly
        if isinstance(provider, list):
            supported_languages = provider
            if language not in supported_languages:
                raise TTSUnsupportedLanguageError(
                    f"Language '{language}' not supported. Supported: {supported_languages}",
                    details={"requested_language": language, "supported_languages": supported_languages}
                )
            return
        return self._validate_language(language, provider)

    def validate_format(self, format: AudioFormat, provider: Optional[Union[str, set[AudioFormat]]] = None):
        """Public method to validate format"""
        # Handle test case where supported formats are passed directly
        if isinstance(provider, set):
            supported_formats = provider
            if format not in supported_formats:
                raise TTSUnsupportedFormatError(
                    f"Audio format '{format.value}' not supported. Supported: {[f.value for f in supported_formats]}",
                    details={"requested_format": format.value, "supported_formats": [f.value for f in supported_formats]}
                )
            return
        return self._validate_format(format, provider)

    def validate_parameters(self, request: TTSRequest):
        """Public method to validate parameters"""
        return self._validate_parameters(request)

    def validate_voice_reference(self, voice_ref_data: bytes):
        """Public method to validate voice reference"""
        return self._validate_voice_reference(voice_ref_data)

    def validate_request(self, request: TTSRequest, provider: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """
        Validate a complete TTS request.

        Args:
            request: TTS request to validate
            provider: TTS provider name

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Validate text
            self._validate_text(request.text, provider)

            # Validate format
            if request.format:
                self._validate_format(request.format, provider)

            # Validate language (allow extra_params.language override)
            language = request.language
            extras = request.extra_params or {}
            if not language and isinstance(extras, dict):
                extra_language = extras.get("language")
                if isinstance(extra_language, str) and extra_language.strip():
                    language = extra_language.strip()
            if language:
                if provider == "chatterbox":
                    self._validate_chatterbox_language(language, request)
                else:
                    self._validate_language(language, provider)

            # Validate voice
            if request.voice:
                self._validate_voice(request.voice, provider)

            min_duration = None
            max_duration = None
            if isinstance(request.extra_params, dict):
                min_duration = self._coerce_float(request.extra_params.get("reference_duration_min"))

            # Provider-specific validations for Qwen3-TTS
            skip_voice_reference_validation = False

            if provider == "qwen3_tts":
                model_name = (getattr(request, "model", None) or "").strip().lower()
                extras = request.extra_params or {}
                runtime_name = self._resolve_qwen3_runtime(provider)
                if runtime_name == "mlx":
                    if request.voice and isinstance(request.voice, str) and request.voice.startswith("custom:"):
                        raise TTSInvalidInputError(
                            "Uploaded custom voices are not supported by the MLX runtime in v1",
                            provider=provider,
                        )
                    if "voicedesign" in model_name:
                        raise TTSInvalidInputError(
                            "Mode 'voice_design' is not supported by the MLX runtime",
                            provider=provider,
                        )
                    clone_requested = bool(
                        request.voice_reference
                        or (isinstance(extras, dict) and (
                            extras.get("reference_text")
                            or extras.get("ref_text")
                            or extras.get("voice_reference_text")
                            or extras.get("x_vector_only_mode")
                            or extras.get("voice_clone_prompt")
                        ))
                        or model_name.endswith("base")
                        or model_name.endswith("-base")
                    )
                    if clone_requested:
                        raise TTSInvalidInputError(
                            "Mode 'voice_clone' is not supported by the MLX runtime",
                            provider=provider,
                        )
                if model_name == "auto" or "customvoice" in model_name:
                    if request.voice and isinstance(request.voice, str) and not request.voice.startswith("custom:"):
                        normalized = self._normalize_qwen3_speaker(request.voice)
                        if normalized not in self.QWEN3_CUSTOMVOICE_SPEAKERS:
                            raise TTSVoiceNotFoundError(
                                f"Invalid Qwen3 CustomVoice speaker: {request.voice}",
                                provider=provider,
                            )
                if model_name.endswith("base") or model_name.endswith("-base"):
                    if not request.voice_reference:
                        raise TTSInvalidVoiceReferenceError(
                            "Voice reference is required for Qwen3 Base models",
                            provider=provider,
                        )
                    x_vector_only = False
                    if isinstance(extras, dict):
                        x_vector_only = parse_bool(extras.get("x_vector_only_mode"), default=False)
                    ref_text = None
                    if isinstance(extras, dict):
                        ref_text = (
                            extras.get("reference_text")
                            or extras.get("ref_text")
                            or extras.get("voice_reference_text")
                        )
                    if not x_vector_only and not (isinstance(ref_text, str) and ref_text.strip()):
                        raise TTSInvalidInputError(
                            "reference_text is required for Qwen3 Base models unless x_vector_only_mode is true",
                            provider=provider,
                        )
                    if min_duration is None:
                        min_duration = 3.0
            elif provider == "pocket_tts_cpp":
                voice = (request.voice or "").strip()
                if not request.voice_reference and not voice.startswith("custom:"):
                    raise TTSInvalidVoiceReferenceError(
                        "PocketTTS.cpp requires a direct voice_reference or a stored custom: voice",
                        provider=provider,
                    )
                duration_limits = PROVIDER_REQUIREMENTS.get("pocket_tts_cpp", {}).get("duration", {})
                if min_duration is None:
                    min_duration = self._coerce_float(duration_limits.get("min"))
                max_duration = self._coerce_float(duration_limits.get("max"))
                skip_voice_reference_validation = bool(
                    voice.startswith("custom:")
                    and isinstance(extras, dict)
                    and extras.get("pocket_tts_cpp_voice_path")
                )
            elif provider == "omnivoice":
                voice = (request.voice or "").strip()
                is_clone_voice = voice.lower() == "clone"
                is_custom_voice = voice.startswith("custom:")
                ref_text = None
                if isinstance(extras, dict):
                    ref_text = (
                        extras.get("reference_text")
                        or extras.get("ref_text")
                        or extras.get("voice_reference_text")
                    )
                clone_requested = bool(request.voice_reference) or is_clone_voice or is_custom_voice
                if is_clone_voice and not request.voice_reference:
                    raise TTSInvalidVoiceReferenceError(
                        "OmniVoice clone requests require voice_reference",
                        provider=provider,
                    )
                if is_custom_voice and not request.voice_reference:
                    raise TTSInvalidVoiceReferenceError(
                        "OmniVoice custom: voices require a resolved voice_reference before provider validation",
                        provider=provider,
                    )
                if clone_requested and not (isinstance(ref_text, str) and ref_text.strip()):
                    raise TTSInvalidInputError(
                        "OmniVoice cloning requires reference_text",
                        provider=provider,
                    )
                duration_limits = PROVIDER_REQUIREMENTS.get("omnivoice", {}).get("duration", {})
                if min_duration is None:
                    min_duration = self._coerce_float(duration_limits.get("min"))
                max_duration = self._coerce_float(duration_limits.get("max"))

            # Validate parameters (provider-aware)
            self._validate_parameters(request, provider)

            # Validate voice reference if required
            if provider in self.REQUIRES_VOICE_REFERENCE and not request.voice_reference:
                raise TTSInvalidVoiceReferenceError(
                    "Voice reference is required for this provider",
                    provider=provider,
                )

            # Validate voice reference if provided
            if request.voice_reference and not skip_voice_reference_validation:
                self._validate_voice_reference(
                    request.voice_reference,
                    min_duration=min_duration,
                    max_duration=max_duration,
                )

            return True, None

        except TTSValidationError as e:
            return False, str(e)
        except Exception as e:
            logger.error(f"Unexpected validation error; exception_type={type(e).__name__}")
            return False, f"Validation failed: {str(e)}"

    def _validate_text(self, text: str, provider: Optional[str] = None):
        """Validate text content"""
        if not text or not text.strip():
            raise TTSInvalidInputError("Text cannot be empty")

        # Check length limits (characters)
        max_length = self.max_text_length_override or self.MAX_TEXT_LENGTHS.get(provider, self.MAX_TEXT_LENGTHS["default"])
        provider_max_length = self._coerce_int(self._get_provider_setting(provider, "max_text_length"))
        if provider_max_length and provider_max_length > 0:
            max_length = provider_max_length

        if len(text) > max_length:
            raise TTSTextTooLongError(
                f"Text length ({len(text)}) exceeds maximum of {max_length} characters",
                provider=provider,
                details={"text_length": len(text), "max_length": max_length}
            )

        # Provider-specific UTF-8 byte cap (exclude BOS)
        if provider in self.MAX_TEXT_BYTES:
            byte_len = len(text.encode("utf-8"))
            max_bytes = self.MAX_TEXT_BYTES[provider]
            if byte_len > max_bytes:
                raise TTSTextTooLongError(
                    f"Text byte length ({byte_len}) exceeds maximum of {max_bytes} bytes",
                    provider=provider,
                    details={"text_byte_length": byte_len, "max_bytes": max_bytes}
                )

        # Check for excessive repetition (potential abuse)
        if self._has_excessive_repetition(text):
            raise TTSInvalidInputError(
                "Text contains excessive repetition",
                provider=provider
            )

    def _validate_format(self, format: AudioFormat, provider: Optional[str] = None):
        """Validate audio format"""
        if provider and provider in self.SUPPORTED_FORMATS and format not in self.SUPPORTED_FORMATS[provider]:
            supported = [fmt.value for fmt in self.SUPPORTED_FORMATS[provider]]
            raise TTSUnsupportedFormatError(
                f"Format '{format.value}' not supported by {provider}. Supported: {supported}",
                provider=provider,
                details={"requested_format": format.value, "supported_formats": supported}
            )

    def _validate_language(self, language: str, provider: Optional[str] = None):
        """Validate language code"""
        if provider == "omnivoice":
            return
        if provider and provider in self.SUPPORTED_LANGUAGES:
            if language not in self.SUPPORTED_LANGUAGES[provider]:
                supported = list(self.SUPPORTED_LANGUAGES[provider])
                raise TTSUnsupportedLanguageError(
                    f"Language '{language}' not supported by {provider}. Supported: {supported}",
                    provider=provider,
                    details={"requested_language": language, "supported_languages": supported}
                )

    def _validate_chatterbox_language(self, language: str, request: TTSRequest) -> None:
        """Validate Chatterbox language support by selected model family."""
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        model_hint = getattr(request, "model", None) or extras.get("model")
        config_variant = (
            extras.get("chatterbox_variant")
            or extras.get("model_family")
            or extras.get("variant")
            or self._get_provider_setting("chatterbox", "variant")
        )
        use_multilingual = parse_bool(
            extras.get("use_multilingual", self._get_provider_setting("chatterbox", "use_multilingual")),
            default=False,
        )
        family = resolve_chatterbox_model_family(
            model_hint,
            language=language,
            config_variant=config_variant,
            use_multilingual=use_multilingual,
        )
        supported = CHATTERBOX_LANGUAGE_CODES if family is ChatterboxModelFamily.MULTILINGUAL else {"en"}
        normalized_language = (language or "").strip().casefold()
        if normalized_language not in supported:
            supported_list = sorted(supported)
            raise TTSUnsupportedLanguageError(
                f"Language '{language}' not supported by chatterbox {family.value}. Supported: {supported_list}",
                provider="chatterbox",
                details={
                    "requested_language": language,
                    "supported_languages": supported_list,
                    "model_family": family.value,
                }
            )

    def _validate_voice(self, voice: str, provider: Optional[str] = None):
        """Validate voice selection"""
        normalized_provider = (provider or "").lower()

        if voice and str(voice).startswith("custom:"):
            custom_id = str(voice).split(":", 1)[1]
            if not custom_id or not re.match(r'^[a-zA-Z0-9_-]+$', custom_id):
                raise TTSVoiceNotFoundError(
                    f"Invalid custom voice name format: {voice}",
                    provider=provider
                )
            if len(custom_id) > 200:
                raise TTSVoiceNotFoundError(
                    "Custom voice name too long",
                    provider=provider
                )
            return

        # For unknown/third-party providers, avoid over-constraining opaque
        # voice identifiers. Adapters are expected to perform any provider-
        # specific validation. Here we only enforce non-emptiness and a
        # generous upper bound on length.
        if normalized_provider and normalized_provider not in self.SUPPORTED_LANGUAGES and normalized_provider not in self.SUPPORTED_FORMATS:
            if not voice or not str(voice).strip():
                raise TTSVoiceNotFoundError(
                    "Voice name cannot be empty",
                    provider=provider,
                )
            if len(voice) > 200:
                raise TTSVoiceNotFoundError(
                    "Voice name too long",
                    provider=provider,
                )
            return

        # Basic voice name validation for providers we know about
        if not re.match(r'^[a-zA-Z0-9_-]+$', voice):
            raise TTSVoiceNotFoundError(
                f"Invalid voice name format: {voice}",
                provider=provider
            )

        # Length check
        if len(voice) > 100:
            raise TTSVoiceNotFoundError(
                "Voice name too long",
                provider=provider
            )

    def _validate_parameters(self, request: TTSRequest, provider: Optional[str] = None):
        """Validate TTS parameters"""
        raw_speed = getattr(request, "_original_speed", request.speed)
        # Provider-aware speed validation; fall back to generic defaults when
        # no provider hint is available.
        try:
            if provider:
                limits = ProviderLimits.get_limits(provider)
                min_speed = float(limits.get("min_speed", 0.1))
                max_speed = float(limits.get("max_speed", 3.0))
            else:
                min_speed, max_speed = 0.1, 3.0
        except Exception:
            min_speed, max_speed = 0.1, 3.0
        if raw_speed < min_speed or raw_speed > max_speed:
            raise TTSInvalidInputError(
                f"Speed must be between {min_speed} and {max_speed}, got {raw_speed}",
                details={"min_speed": min_speed, "max_speed": max_speed}
            )

        # Pitch validation
        raw_pitch = getattr(request, "_original_pitch", request.pitch)
        if raw_pitch < -20.0 or raw_pitch > 20.0:
            raise TTSInvalidInputError(
                f"Pitch must be between -20.0 and 20.0, got {raw_pitch}"
            )

        # Volume validation
        raw_volume = getattr(request, "_original_volume", request.volume)
        if raw_volume < 0.0 or raw_volume > 2.0:
            raise TTSInvalidInputError(
                f"Volume must be between 0.0 and 2.0, got {raw_volume}"
            )

        # Emotion intensity validation
        if request.emotion_intensity < 0.0 or request.emotion_intensity > 2.0:
            raise TTSInvalidInputError(
                f"Emotion intensity must be between 0.0 and 2.0, got {request.emotion_intensity}"
            )

        extras = request.extra_params or {}
        if extras:
            emo_alpha = extras.get("emo_alpha")
            if emo_alpha is not None:
                try:
                    emo_alpha = float(emo_alpha)
                except Exception as exc:
                    raise TTSInvalidInputError(f"emo_alpha must be numeric, got {emo_alpha!r}") from exc
                if emo_alpha < 0.0 or emo_alpha > 1.0:
                    raise TTSInvalidInputError("emo_alpha must be between 0.0 and 1.0")

            emo_vector = extras.get("emo_vector")
            if emo_vector is not None:
                if not isinstance(emo_vector, (list, tuple)):
                    raise TTSInvalidInputError("emo_vector must be a list or tuple of floats")
                if len(emo_vector) not in (0, 8):
                    raise TTSInvalidInputError("emo_vector must contain 8 values (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm)")
                for value in emo_vector:
                    if not isinstance(value, (int, float)):
                        raise TTSInvalidInputError("emo_vector entries must be numeric")

            emo_audio_reference = extras.get("emo_audio_reference")
            if emo_audio_reference is not None:
                if isinstance(emo_audio_reference, str):
                    try:
                        emo_audio_bytes = base64.b64decode(emo_audio_reference, validate=True)
                    except Exception as exc:
                        raise TTSInvalidInputError("emo_audio_reference must be valid base64 audio") from exc
                elif isinstance(emo_audio_reference, (bytes, bytearray)):
                    emo_audio_bytes = bytes(emo_audio_reference)
                else:
                    raise TTSInvalidInputError("emo_audio_reference must be a base64 string or bytes")

                if len(emo_audio_bytes) > self.EMO_REF_MAX_SIZE:
                    raise TTSInvalidInputError(
                        f"Emotion reference audio too large: {len(emo_audio_bytes)} bytes (max {self.EMO_REF_MAX_SIZE})"
                    )

            interval_silence = extras.get("interval_silence")
            if interval_silence is not None:
                try:
                    interval_value = int(interval_silence)
                except Exception as exc:
                    raise TTSInvalidInputError("interval_silence must be an integer millisecond value") from exc
                if interval_value < 0 or interval_value > 5000:
                    raise TTSInvalidInputError("interval_silence must be between 0 and 5000 milliseconds")

            max_tokens = extras.get("max_text_tokens_per_segment")
            if max_tokens is not None:
                try:
                    max_tokens_value = int(max_tokens)
                except Exception as exc:
                    raise TTSInvalidInputError("max_text_tokens_per_segment must be an integer") from exc
                if max_tokens_value <= 0:
                    raise TTSInvalidInputError("max_text_tokens_per_segment must be greater than zero")

            voice_clone_prompt = extras.get("voice_clone_prompt")
            if voice_clone_prompt is not None:
                self._validate_voice_clone_prompt(voice_clone_prompt, provider)

            if provider == "omnivoice":
                reference_text = (
                    extras.get("reference_text")
                    or extras.get("ref_text")
                    or extras.get("voice_reference_text")
                )
                if reference_text is not None:
                    if not isinstance(reference_text, str) or not reference_text.strip():
                        raise TTSInvalidInputError("reference_text must be a non-empty string")

                mode = extras.get("omnivoice_mode", extras.get("mode"))
                if mode is not None:
                    if not isinstance(mode, str):
                        raise TTSInvalidInputError("OmniVoice mode must be a string")
                    if mode.strip().lower() not in {"auto", "design", "clone"}:
                        raise TTSInvalidInputError("OmniVoice mode must be 'auto', 'design', or 'clone'")
                self._validate_omnivoice_extra_params(request, extras)

    def _validate_omnivoice_extra_params(self, request: TTSRequest, extras: dict[str, Any]) -> None:
        instruct_values: list[str] = []
        for key in OMNIVOICE_INSTRUCT_KEYS:
            value = extras.get(key)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TTSInvalidInputError(f"OmniVoice {key} must be a string")
            stripped = value.strip()
            if stripped:
                instruct_values.append(stripped)
        if len(set(instruct_values)) > 1:
            raise TTSInvalidInputError("Conflicting OmniVoice instruct aliases provided")
        design_requested = bool(instruct_values)

        language_values: list[str] = []
        for key in OMNIVOICE_LANGUAGE_KEYS:
            value = extras.get(key)
            if value is None:
                continue
            stripped = str(value).strip()
            if stripped:
                language_values.append(stripped.lower())
        request_language = getattr(request, "language", None)
        if request_language is not None:
            stripped = str(request_language).strip()
            if stripped and stripped.lower() != "en":
                language_values.append(stripped.lower())
        if len(set(language_values)) > 1:
            raise TTSInvalidInputError("Conflicting OmniVoice language aliases provided")

        mode = extras.get("omnivoice_mode", extras.get("mode"))
        normalized_mode = None
        if mode is not None:
            if not isinstance(mode, str):
                raise TTSInvalidInputError("OmniVoice mode must be a string")
            normalized_mode = mode.strip().lower()
            if normalized_mode not in {"auto", "design", "clone"}:
                raise TTSInvalidInputError("OmniVoice mode must be 'auto', 'design', or 'clone'")
        voice = (request.voice or "").strip().lower()
        clone_requested = bool(request.voice_reference) or voice == "clone" or voice.startswith("custom:")
        if normalized_mode == "auto" and (design_requested or clone_requested):
            raise TTSInvalidInputError("OmniVoice mode=auto conflicts with design or clone inputs")
        if normalized_mode == "design" and not design_requested:
            raise TTSInvalidInputError("OmniVoice mode=design requires instruct")
        if normalized_mode == "design" and clone_requested:
            raise TTSInvalidInputError("OmniVoice mode=design conflicts with clone inputs")
        if normalized_mode == "clone" and design_requested:
            raise TTSInvalidInputError("OmniVoice mode=clone conflicts with instruct")
        if normalized_mode == "clone" and not clone_requested:
            raise TTSInvalidInputError("OmniVoice mode=clone requires reference audio")

        for key, value in extras.items():
            if key in OMNIVOICE_GENERATION_PARAM_RANGES:
                expected_type, min_value, max_value = OMNIVOICE_GENERATION_PARAM_RANGES[key]
                self._validate_omnivoice_generation_value(
                    key,
                    value,
                    expected_type,
                    min_value,
                    max_value,
                )
                continue
            if key in OMNIVOICE_SUPPORTED_NON_GENERATION_KEYS:
                continue
            if key.startswith("omnivoice_"):
                raise TTSInvalidInputError(f"Unknown OmniVoice generation parameter: {key}")

    def _validate_omnivoice_generation_value(
        self,
        key: str,
        value: Any,
        expected_type: type,
        min_value: Optional[float],
        max_value: Optional[float],
    ) -> None:
        try:
            if expected_type is bool:
                self._validate_omnivoice_bool_generation_value(key, value)
                return
            if expected_type is int:
                parsed = self._coerce_omnivoice_int_generation_value(key, value)
            else:
                parsed = self._coerce_omnivoice_float_generation_value(key, value)
        except Exception as exc:
            raise TTSInvalidInputError(f"OmniVoice generation parameter {key} has invalid type") from exc
        if min_value is not None:
            if key in {"duration", "speed", "audio_chunk_duration", "audio_chunk_threshold"}:
                if parsed <= min_value:
                    raise TTSInvalidInputError(f"OmniVoice generation parameter {key} must be greater than {min_value}")
            elif parsed < min_value:
                raise TTSInvalidInputError(f"OmniVoice generation parameter {key} must be at least {min_value}")
        if max_value is not None and parsed > max_value:
            raise TTSInvalidInputError(f"OmniVoice generation parameter {key} must be at most {max_value}")

    def _validate_omnivoice_bool_generation_value(self, key: str, value: Any) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in OMNIVOICE_TRUE_VALUES or normalized in OMNIVOICE_FALSE_VALUES:
                return
        raise TTSInvalidInputError(f"OmniVoice generation parameter {key} must be a boolean")

    def _coerce_omnivoice_int_generation_value(self, key: str, value: Any) -> int:
        if isinstance(value, bool):
            raise TTSInvalidInputError(f"OmniVoice generation parameter {key} must be an integer")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if re.fullmatch(r"[+-]?\d+", stripped):
                return int(stripped)
        raise TTSInvalidInputError(f"OmniVoice generation parameter {key} must be an integer")

    def _coerce_omnivoice_float_generation_value(self, key: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TTSInvalidInputError(f"OmniVoice generation parameter {key} must be a finite number")
        try:
            parsed = float(value)
        except Exception as exc:
            raise TTSInvalidInputError(f"OmniVoice generation parameter {key} must be a finite number") from exc
        if not math.isfinite(parsed):
            raise TTSInvalidInputError(f"OmniVoice generation parameter {key} must be a finite number")
        return parsed

    def _validate_voice_reference(
        self,
        voice_ref_data: bytes,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ):
        """Validate voice reference audio for cloning"""
        if len(voice_ref_data) == 0:
            raise TTSInvalidVoiceReferenceError("Voice reference data is empty")

        if len(voice_ref_data) > self.VOICE_REF_MAX_SIZE:
            raise TTSInvalidVoiceReferenceError(
                f"Voice reference file too large: {len(voice_ref_data)} bytes (max: {self.VOICE_REF_MAX_SIZE})",
                details={"file_size": len(voice_ref_data), "max_size": self.VOICE_REF_MAX_SIZE}
            )

        # Basic file type validation (check magic bytes)
        if not self._is_valid_audio_file(voice_ref_data):
            raise TTSInvalidVoiceReferenceError(
                "Voice reference file is not a valid audio format"
            )

        if (min_duration is not None and min_duration > 0) or (max_duration is not None and max_duration > 0):
            try:
                import io

                import soundfile as sf
            except Exception as exc:
                raise TTSInvalidVoiceReferenceError(
                    "Reference duration validation requires soundfile",
                    details={"error": str(exc)},
                ) from exc
            try:
                with sf.SoundFile(io.BytesIO(voice_ref_data)) as info:
                    duration = float(len(info)) / float(info.samplerate or 1)
            except Exception as exc:
                raise TTSInvalidVoiceReferenceError(
                    "Unable to read voice reference audio for duration validation",
                    details={"error": str(exc)},
                ) from exc
            if min_duration is not None and duration < min_duration:
                raise TTSInvalidVoiceReferenceError(
                    f"Voice reference audio too short: {duration:.2f}s (min {min_duration}s)",
                    details={"duration_seconds": duration, "min_seconds": min_duration},
                )
            if max_duration is not None and duration > max_duration:
                raise TTSInvalidVoiceReferenceError(
                    f"Voice reference audio too long: {duration:.2f}s (max {max_duration}s)",
                    details={"duration_seconds": duration, "max_seconds": max_duration},
                )

    def _sanitize_html(self, text: str) -> str:
        """Sanitize HTML content while preserving safe tags"""
        # For now, just escape everything - can be enhanced with a proper HTML sanitizer
        return html.escape(text, quote=True)

    def _clean_control_characters(self, text: str) -> str:
        """Remove or replace control characters"""
        # Remove most control characters but keep common whitespace
        cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

        # Replace multiple spaces/tabs with single space but preserve newlines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)

        # Replace multiple newlines with double newline
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        return cleaned

    def _provider_specific_sanitization(self, text: str, provider: str) -> str:
        """Apply provider-specific text sanitization"""
        if provider == "openai":
            # OpenAI specific rules
            return text
        elif provider == "elevenlabs":
            # ElevenLabs specific rules
            return text
        elif provider in ["kokoro", "kitten_tts", "higgs", "dia", "chatterbox", "vibevoice", "pocket_tts", "pocket_tts_cpp", "lux_tts"]:
            # Local model specific rules - more conservative
            # Remove URLs and email addresses
            text = re.sub(r'https?://\S+', '[URL]', text)
            text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
            return text

        return text

    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character or word repetition"""
        # Check for repeated characters (like "aaaaaaa")
        if re.search(r'(.)\1{10,}', text):
            return True

        # Check for repeated words
        words = text.lower().split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
                # If any word appears more than 30% of the time, it's excessive
                if word_counts[word] > len(words) * 0.3:
                    return True

        return False

    def _is_valid_audio_file(self, data: bytes) -> bool:
        """Check if data starts with valid audio file magic bytes"""
        if len(data) < 4:
            return False

        # Check common audio file signatures
        signatures = [
            b'ID3',      # MP3 with ID3
            b'\xff\xfb', # MP3
            b'\xff\xf3', # MP3
            b'\xff\xf2', # MP3
            b'RIFF',     # WAV
            b'fLaC',     # FLAC
            b'OggS',     # OGG/OPUS
            b'FORM',     # AIFF
        ]

        for sig in signatures:
            if data.startswith(sig):
                return True

        # Check for MP4/M4A (more complex)
        return bool(len(data) >= 8 and data[4:8] == b'ftyp')


# Convenience validation functions
def validate_text_input(text: str, provider: Optional[str] = None, config: Optional[dict[str, Any]] = None) -> str:
    """
    Validate and sanitize text input for TTS.

    Args:
        text: Input text
        provider: TTS provider name
        config: Validation configuration

    Returns:
        Sanitized text

    Raises:
        TTSInvalidInputError: If text is invalid
    """
    validator = TTSInputValidator(config)
    return validator.sanitize_text(text, provider)


def validate_tts_request(request: TTSRequest, provider: Optional[str] = None, config: Optional[dict[str, Any]] = None) -> None:
    """
    Validate complete TTS request.

    Args:
        request: TTS request to validate
        provider: TTS provider name
        config: Validation configuration

    Raises:
        TTSValidationError: If request is invalid
    """
    validator = TTSInputValidator(config)
    is_valid, error_message = validator.validate_request(request, provider)

    if not is_valid:
        raise TTSValidationError(error_message, provider=provider)


def validate_voice_reference(voice_ref_data: bytes, config: Optional[dict[str, Any]] = None) -> None:
    """
    Validate voice reference audio data.

    Args:
        voice_ref_data: Voice reference audio bytes
        config: Validation configuration

    Raises:
        TTSInvalidVoiceReferenceError: If voice reference is invalid
    """
    validator = TTSInputValidator(config)
    validator._validate_voice_reference(voice_ref_data)

#
# End of tts_validation.py
#######################################################################################################################

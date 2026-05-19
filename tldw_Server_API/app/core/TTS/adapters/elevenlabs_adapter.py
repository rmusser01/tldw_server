# elevenlabs_adapter.py
# Description: ElevenLabs TTS adapter implementation
#
# Imports
import os
from collections.abc import AsyncGenerator
from typing import Any, Optional

#
# Third-party Imports
from loguru import logger

try:
    import httpx
except ImportError:  # pragma: no cover - optional import guard
    httpx = None

#
# Local Imports
import contextlib

from tldw_Server_API.app.core.http_client import afetch, astream_bytes

from ..tts_exceptions import (
    TTSAuthenticationError,
    TTSError,
    TTSGenerationError,
    TTSProviderError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSTimeoutError,
    TTSValidationError,
    rate_limit_error,
)
from ..tts_resource_manager import get_resource_manager
from ..tts_validation import validate_tts_request
from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo

#
#######################################################################################################################
#
# ElevenLabs TTS Adapter Implementation

def _is_httpx_exception(exc: Exception) -> bool:
    module = getattr(exc.__class__, "__module__", "")
    return module.startswith("httpx")


def _is_http_status_error(exc: Exception) -> bool:
    if not _is_httpx_exception(exc):
        return False
    return exc.__class__.__name__ == "HTTPStatusError"


def _safe_exception_label(exc: BaseException) -> str:
    """Return a non-sensitive exception identifier for logs."""
    return type(exc).__name__


_ELEVENLABS_HTTP_EXCEPTIONS: tuple[type[BaseException], ...] = ()
if httpx is not None:
    _ELEVENLABS_HTTP_EXCEPTIONS = (httpx.HTTPError,)

_ELEVENLABS_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeError,
    ValueError,
) + _ELEVENLABS_HTTP_EXCEPTIONS

class ElevenLabsAdapter(TTSAdapter):
    """Adapter for ElevenLabs TTS API"""

    # ElevenLabs API endpoints
    BASE_URL = "https://api.elevenlabs.io/v1"

    # Default voice IDs (can be customized)
    DEFAULT_VOICES = {
        "rachel": VoiceInfo(
            id="21m00Tcm4TlvDq8ikWAM",
            name="Rachel",
            gender="female",
            language="en",
            description="American female voice"
        ),
        "drew": VoiceInfo(
            id="29vD33N1CtxCmqQRPOHJ",
            name="Drew",
            gender="male",
            language="en",
            description="American male voice"
        ),
        "clyde": VoiceInfo(
            id="2EiwWnXFnvU5JabPnv8n",
            name="Clyde",
            gender="male",
            language="en",
            description="American male voice"
        ),
        "paul": VoiceInfo(
            id="5Q0t7uMcjvnagumLfvZi",
            name="Paul",
            gender="male",
            language="en",
            description="American male voice"
        ),
        "domi": VoiceInfo(
            id="AZnzlk1XvdvUeBnXmlld",
            name="Domi",
            gender="female",
            language="en",
            description="American female voice"
        ),
        "dave": VoiceInfo(
            id="CYw3kZ02Hs0563khs1Fj",
            name="Dave",
            gender="male",
            language="en-gb",
            description="British male voice"
        ),
        "fin": VoiceInfo(
            id="D38z5RcWu1voky8WS1ja",
            name="Fin",
            gender="male",
            language="en",
            description="Irish male voice"
        ),
        "bella": VoiceInfo(
            id="EXAVITQu4vr4xnSDxMaL",
            name="Bella",
            gender="female",
            language="en",
            description="American female voice"
        ),
        "antoni": VoiceInfo(
            id="ErXwobaYiN019PkySvjV",
            name="Antoni",
            gender="male",
            language="en",
            description="American male voice"
        ),
        "thomas": VoiceInfo(
            id="GBv7mTt0atIp3Br8iCZE",
            name="Thomas",
            gender="male",
            language="en",
            description="American male voice"
        )
    }

    # Model configurations
    MODELS = {
        "eleven_monolingual_v1": {
            "model_id": "eleven_monolingual_v1",
            "languages": ["en"],
            "description": "English only, fastest"
        },
        "eleven_multilingual_v1": {
            "model_id": "eleven_multilingual_v1",
            "languages": ["en", "de", "pl", "es", "it", "fr", "pt", "hi", "ar"],
            "description": "V1 with multiple languages"
        },
        "eleven_multilingual_v2": {
            "model_id": "eleven_multilingual_v2",
            "languages": ["en", "de", "pl", "es", "it", "fr", "pt", "hi", "ar", "ja", "ko", "zh", "nl", "tr", "sv", "id", "fil", "ms", "ro", "uk", "el"],
            "description": "V2 with extensive language support"
        },
        "eleven_turbo_v2": {
            "model_id": "eleven_turbo_v2",
            "languages": ["en"],
            "description": "Turbo model for low latency English"
        }
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)

        # API configuration
        self.api_key = self.config.get("elevenlabs_api_key") or os.getenv("ELEVENLABS_API_KEY")
        # Normalize placeholder/empty values to None so unit tests can detect "not configured"
        if isinstance(self.api_key, str):
            _raw = self.api_key.strip()
            placeholder_tokens = {
                "<elevenlabs_api_key>",
                "<eleven_labs_api_key>",
                "your-elevenlabs-api-key",
                "your_elevenlabs_api_key",
                "",
                "none",
                "null",
            }
            if _raw.lower() in placeholder_tokens:
                self.api_key = None
        self.base_url = self.config.get("elevenlabs_base_url", self.BASE_URL)

        # Model selection
        self.default_model = self.config.get("elevenlabs_model", "eleven_monolingual_v1")

        # Voice settings
        self.stability = self.config.get("elevenlabs_stability", 0.5)
        self.similarity_boost = self.config.get("elevenlabs_similarity_boost", 0.5)
        self.style = self.config.get("elevenlabs_style", 0.0)
        self.use_speaker_boost = self.config.get("elevenlabs_speaker_boost", True)

        # HTTP client
        self.client: Optional[Any] = None

        # Cache for user voices
        self._user_voices: list[VoiceInfo] = []

    async def initialize(self) -> bool:
        """Initialize the ElevenLabs adapter"""
        try:
            if not self.api_key:
                error_msg = f"{self.provider_name}: No API key provided"
                logger.warning(error_msg)
                self._status = ProviderStatus.NOT_CONFIGURED
                raise TTSProviderNotConfiguredError(error_msg, provider=self.provider_name)

            # Get HTTP client from resource manager
            resource_manager = await get_resource_manager()
            self.client = await resource_manager.get_http_client(
                provider=self.provider_name.lower(),
                base_url=self.base_url
            )

            # Test API connection and fetch user voices
            await self._fetch_user_voices()

            # Mark initialized and cache capabilities
            self._capabilities = await self.get_capabilities()
            self._initialized = True
            self._status = ProviderStatus.AVAILABLE
            logger.info(f"{self.provider_name}: Initialized successfully")
            return True

        except TTSProviderNotConfiguredError:
            return False
        except _ELEVENLABS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(
                f"{self.provider_name}: Initialization failed; "
                f"exception_type={_safe_exception_label(e)}"
            )
            self._status = ProviderStatus.ERROR
            raise TTSProviderInitializationError(
                f"Failed to initialize {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e)}
            ) from e

    async def _fetch_user_voices(self):
        """Fetch available voices from ElevenLabs API"""
        try:
            headers = {"xi-api-key": self.api_key}
            response = await afetch(
                method="GET",
                url=f"{self.base_url}/voices",
                client=self.client,
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                self._user_voices = []

                for voice in data.get("voices", []):
                    voice_info = VoiceInfo(
                        id=voice["voice_id"],
                        name=voice["name"],
                        gender=voice.get("labels", {}).get("gender", "unknown"),
                        language=voice.get("labels", {}).get("language", "en"),
                        description=voice.get("labels", {}).get("description", ""),
                        use_case=voice.get("labels", {}).get("use_case", [])
                    )
                    self._user_voices.append(voice_info)

                logger.info(f"{self.provider_name}: Loaded {len(self._user_voices)} voices")
            else:
                logger.warning(f"{self.provider_name}: Failed to fetch voices: {response.status_code}")

        except _ELEVENLABS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(
                f"{self.provider_name}: Error fetching voices; "
                f"exception_type={_safe_exception_label(e)}"
            )

    async def get_capabilities(self) -> TTSCapabilities:
        """Get ElevenLabs TTS capabilities"""
        # Determine supported languages based on selected model
        model_config = self.MODELS.get(self.default_model, self.MODELS["eleven_monolingual_v1"])
        supported_languages = set(model_config["languages"])

        # Combine default and user voices
        all_voices = list(self.DEFAULT_VOICES.values()) + self._user_voices

        return TTSCapabilities(
            provider_name="ElevenLabs",
            supported_languages=supported_languages,
            supported_voices=all_voices,
            supported_formats={
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.PCM,
                AudioFormat.WAV,
            },
            max_text_length=5000,  # ElevenLabs character limit
            supports_streaming=True,
            supports_voice_cloning=True,  # Pro feature
            supports_emotion_control=True,  # Via voice settings
            supports_speech_rate=False,  # Not directly supported
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=300,  # Typical latency
            sample_rate=44100,  # Default output sample rate
            default_format=AudioFormat.MP3
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using ElevenLabs TTS"""
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                f"{self.provider_name} not initialized",
                provider=self.provider_name
            )

        # Validate request using new validation system
        try:
            validate_tts_request(request, provider=self.provider_key)
        except Exception as e:
            logger.error(
                f"{self.provider_name} request validation failed; "
                f"exception_type={_safe_exception_label(e)}"
            )
            raise

        # Prepare voice ID
        voice_id = self._get_voice_id(request.voice or "rachel")

        # Select model
        model_id = self._select_model(request)

        logger.info(
            f"{self.provider_name}: Generating speech with voice={voice_id}, "
            f"model={model_id}, format={request.format.value}"
        )

        try:
            if request.stream:
                return TTSResponse(
                    audio_stream=self._stream_audio_elevenlabs(
                        text=request.text,
                        voice_id=voice_id,
                        model_id=model_id,
                        request=request,
                    ),
                    format=request.format,
                    sample_rate=self._infer_sample_rate(request.format),
                    channels=1,
                    voice_used=request.voice or "rachel",
                    provider=self.provider_name
                )
            else:
                # Generate complete audio (use non-stream endpoint for efficiency)
                audio_data = await self._generate_complete_elevenlabs(
                    text=request.text,
                    voice_id=voice_id,
                    model_id=model_id,
                    request=request
                )
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self._infer_sample_rate(request.format),
                    channels=1,
                    voice_used=request.voice or "rachel",
                    provider=self.provider_name
                )

        except (TTSProviderNotConfiguredError, TTSAuthenticationError, TTSRateLimitError, TTSQuotaExceededError, TTSValidationError):
            raise
        except _ELEVENLABS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(
                f"{self.provider_name} generation error; "
                f"exception_type={_safe_exception_label(e)}"
            )
            raise TTSGenerationError(
                f"Failed to generate speech with {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e), "error_type": type(e).__name__}
            ) from e

    async def _stream_audio_elevenlabs(
        self,
        text: str,
        voice_id: str,
        model_id: str,
        request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from ElevenLabs API"""
        url = f"{self.base_url}/text-to-speech/{voice_id}/stream"

        headers = {
            "Accept": self._get_accept_header(request.format),
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        # Prepare voice settings
        voice_settings = {
            "stability": request.extra_params.get("stability", self.stability),
            "similarity_boost": request.extra_params.get("similarity_boost", self.similarity_boost),
            "style": request.extra_params.get("style", self.style),
            "use_speaker_boost": request.extra_params.get("speaker_boost", self.use_speaker_boost)
        }

        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }

        try:
            chunk_count = 0
            async for chunk in astream_bytes(
                method="POST",
                url=url,
                client=self.client,
                headers=headers,
                json=payload,
                chunk_size=1024,
            ):
                if chunk:
                    chunk_count += 1
                    yield chunk

                    if chunk_count % 10 == 0:
                        logger.debug(f"{self.provider_name}: Streamed {chunk_count} chunks")

            logger.info(f"{self.provider_name}: Successfully streamed {chunk_count} chunks")

        except Exception as e:
            if _is_http_status_error(e):
                response = getattr(e, "response", None)
                status = getattr(response, "status_code", None)
                logger.error(f"{self.provider_name} HTTP error: {status} - response body redacted")
                self._raise_mapped_http_error(e)
            logger.error(
                f"{self.provider_name} streaming error; "
                f"exception_type={_safe_exception_label(e)}"
            )
            raise

    async def _generate_complete_elevenlabs(
        self,
        text: str,
        voice_id: str,
        model_id: str,
        request: TTSRequest
    ) -> bytes:
        """Generate complete audio from ElevenLabs (non-stream endpoint)."""
        url = f"{self.base_url}/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Accept": self._get_accept_header(request.format)
        }
        voice_settings = {
            "stability": request.extra_params.get("stability", self.stability),
            "similarity_boost": request.extra_params.get("similarity_boost", self.similarity_boost),
            "style": request.extra_params.get("style", self.style),
            "use_speaker_boost": request.extra_params.get("speaker_boost", self.use_speaker_boost)
        }
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings
        }
        try:
            response = await afetch(
                method="POST",
                url=url,
                client=self.client,
                headers=headers,
                json=payload,
            )
            try:
                response.raise_for_status()
            except Exception as e:
                if _is_http_status_error(e):
                    self._raise_mapped_http_error(e)
                raise
            return response.content or b""
        except TTSError:
            # Propagate mapped TTS exceptions without wrapping/logging
            raise
        except Exception as e:
            if _is_http_status_error(e):
                self._raise_mapped_http_error(e)
            logger.error(
                f"{self.provider_name} non-stream error; "
                f"exception_type={_safe_exception_label(e)}"
            )
            raise

    def _get_voice_id(self, voice_name: str) -> str:
        """Get ElevenLabs voice ID from voice name"""
        # Check if it's already a voice ID (alphanumeric, typically >=20 chars)
        if voice_name and voice_name.isalnum() and len(voice_name) >= 20:
            return voice_name

        # Check default voices
        voice_lower = voice_name.lower()
        if voice_lower in self.DEFAULT_VOICES:
            return self.DEFAULT_VOICES[voice_lower].id

        # Check user voices
        for voice in self._user_voices:
            if voice.name.lower() == voice_lower:
                return voice.id

        # Default to Rachel
        logger.warning(f"{self.provider_name}: Voice not found, using default")
        return self.DEFAULT_VOICES["rachel"].id

    def _select_model(self, request: TTSRequest) -> str:
        """Select appropriate ElevenLabs model"""
        # Check if model specified in extra params
        if "model" in request.extra_params:
            return request.extra_params["model"]

        # Select based on language
        if request.language and request.language != "en":
            # Use multilingual model for non-English
            return "eleven_multilingual_v2"

        # Use configured default
        return self.default_model

    def _get_accept_header(self, format: AudioFormat) -> str:
        """Get Accept header for requested format"""
        format_map = {
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.WAV: "audio/wav",
            AudioFormat.PCM: "audio/pcm",
            AudioFormat.OPUS: "audio/opus",
        }
        # Default to mpeg for unknown or unsupported formats
        return format_map.get(format, "audio/mpeg")

    def _infer_sample_rate(self, format: AudioFormat) -> int:
        """Best-effort sample rate heuristic by format."""
        if format == AudioFormat.OPUS:
            return 48000
        # Default to 44100 for mp3/wav/pcm/ulaw
        return 44100

    def _raise_mapped_http_error(self, e: Exception) -> None:
        """Map HTTP errors to TTS exceptions for consistent handling upstream."""
        response = getattr(e, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
        try:
            text = response.text if response is not None else None
        except (AttributeError, RuntimeError, TypeError, ValueError):
            text = None
        if status in (401, 403):
            raise TTSAuthenticationError(f"{self.provider_name} authentication failed", provider=self.provider_name, details={"status": status, "body": text})
        if status == 429:
            raise TTSRateLimitError(f"{self.provider_name} rate limit exceeded", provider=self.provider_name, details={"status": status})
        if status in (408, 504):
            raise TTSTimeoutError(f"{self.provider_name} timeout", provider=self.provider_name, details={"status": status})
        if status and 500 <= status < 600:
            raise TTSProviderError(f"{self.provider_name} upstream error", provider=self.provider_name, details={"status": status})
        # Generic mapping for other 4xx
        raise TTSProviderError(f"{self.provider_name} request failed ({status})", provider=self.provider_name, details={"status": status, "body": text})

    async def _cleanup_resources(self):
        """Clean up ElevenLabs adapter resources"""
        if self.client:
            try:
                await self.client.aclose()
                self.client = None
                logger.debug(f"{self.provider_name}: HTTP client closed")
            except (AttributeError, OSError, RuntimeError) as e:
                logger.warning(
                    f"{self.provider_name}: Error closing HTTP client; "
                    f"exception_type={_safe_exception_label(e)}"
                )

    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to ElevenLabs voice"""
        # Common mappings
        voice_mappings = {
            "female": "rachel",
            "male": "drew",
            "british": "dave",
            "irish": "fin",
            "young_female": "bella",
            "young_male": "antoni"
        }

        return voice_mappings.get((voice_id or "").lower(), voice_id)

    def preprocess_text(self, text: str, **kwargs) -> str:
        """Preprocess text for ElevenLabs"""
        # ElevenLabs handles most text normalization internally
        # Just ensure text isn't too long
        if len(text) > 5000:
            logger.warning(f"{self.provider_name}: Text truncated to 5000 characters")
            text = text[:5000]

        return text.strip()

# Backward-compat alias expected by some tests
class ElevenLabsTTSAdapter(ElevenLabsAdapter):
    """Compatibility wrapper with extended ElevenLabs interface for TTS_NEW tests.

    - Accepts generic config keys (api_key, base_url, timeout)
    - Exposes convenience attributes/methods expected by the new tests
    - Performs lightweight validation raising exceptions on invalid input
    """

    PROVIDER_KEY = "elevenlabs"

    def __init__(self, config: Optional[dict[str, Any]] = None):
        cfg = config.copy() if isinstance(config, dict) else {}
        # Map generic keys to adapter-specific keys used by the base class
        mapped_cfg: dict[str, Any] = {}
        if "api_key" in cfg:
            mapped_cfg["elevenlabs_api_key"] = cfg.get("api_key")
        if "base_url" in cfg:
            mapped_cfg["elevenlabs_base_url"] = cfg.get("base_url")
        if "timeout" in cfg:
            mapped_cfg["timeout"] = cfg.get("timeout")

        # If API key isn't provided via config or env, tests expect an error at construction time
        temp_key = mapped_cfg.get("elevenlabs_api_key") or os.getenv("ELEVENLABS_API_KEY")
        if not temp_key:
            raise TTSProviderNotConfiguredError("ElevenLabs API key not configured", provider="elevenlabs")

        super().__init__(mapped_cfg)
        # Record generic properties used by tests
        self._provider_simple = "elevenlabs"
        self._timeout = cfg.get("timeout")

    # --- Simple attributes/properties expected by tests ---
    @property
    def provider(self) -> str:
        return self._provider_simple

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def supported_models(self) -> list[str]:
        return list(self.MODELS.keys())

    # --- Convenience API ---
    async def fetch_voices(self) -> list[dict[str, Any]]:
        """Return available voices as a list of dicts from the public API."""
        if not self.client:
            from tldw_Server_API.app.core.http_client import create_async_client
            self.client = create_async_client()
        headers = {"xi-api-key": self.api_key}
        resp = await afetch(
            method="GET",
            url=f"{self.base_url}/voices",
            client=self.client,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        return data.get("voices", [])

    async def get_voice_info(self, voice_id: str) -> dict[str, Any]:
        if not self.client:
            from tldw_Server_API.app.core.http_client import create_async_client
            self.client = create_async_client()
        headers = {"xi-api-key": self.api_key}
        resp = await afetch(
            method="GET",
            url=f"{self.base_url}/voices/{voice_id}",
            client=self.client,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json() or {}

    async def clone_voice(self, name: str, samples: list[bytes]) -> str:
        if not self.client:
            from tldw_Server_API.app.core.http_client import create_async_client
            self.client = create_async_client()
        headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {"name": name, "samples": [s.decode("latin1") if isinstance(s, (bytes, bytearray)) else s for s in samples]}
        resp = await afetch(
            method="POST",
            url=f"{self.base_url}/voices/add",
            client=self.client,
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        return data.get("voice_id") or data.get("id") or ""

    async def get_usage(self) -> dict[str, Any]:
        if not self.client:
            from tldw_Server_API.app.core.http_client import create_async_client
            self.client = create_async_client()
        headers = {"xi-api-key": self.api_key}
        resp = await afetch(
            method="GET",
            url=f"{self.base_url}/user",
            client=self.client,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        count = int(data.get("character_count", 0))
        limit = int(data.get("character_limit", 0))
        remaining = max(0, limit - count)
        return {
            "character_count": count,
            "character_limit": limit,
            "remaining": remaining,
            **{k: v for k, v in data.items() if k not in {"character_count", "character_limit"}},
        }

    async def validate_request(self, request: TTSRequest) -> None:
        """Raise on invalid requests (new-test-friendly behavior)."""
        # Basic text validation
        if not request.text or not str(request.text).strip():
            from ..tts_exceptions import TTSInvalidInputError
            raise TTSInvalidInputError("Text cannot be empty", provider=self._provider_simple)
        if len(request.text) > 5000:
            from ..tts_exceptions import TTSTextTooLongError
            raise TTSTextTooLongError("Text exceeds maximum for ElevenLabs", provider=self._provider_simple)

        # Model validation when provided
        if request.model and request.model not in self.MODELS:
            from ..tts_exceptions import TTSValidationError
            raise TTSValidationError("Invalid model for ElevenLabs", provider=self._provider_simple, details={"model": request.model})

        # Voice settings bounds (when provided)
        vs = request.voice_settings
        if vs is not None:
            def _in01(x: Optional[float]) -> bool:
                return x is None or (0.0 <= float(x) <= 1.0)
            if not _in01(vs.stability) or not _in01(vs.similarity_boost):
                from ..tts_exceptions import TTSValidationError
                raise TTSValidationError("Voice settings out of range", provider=self._provider_simple)

    async def generate(self, request: TTSRequest) -> TTSResponse:
        # Default to non-streaming for adapter.generate() to match unit tests
        # (streaming is exercised via generate_stream())
        request.stream = False
        # Map request fields into extra_params expected by base adapter
        if request.model:
            request.extra_params["model"] = request.model
        if request.voice_settings:
            if request.voice_settings.stability is not None:
                request.extra_params["stability"] = request.voice_settings.stability
            if request.voice_settings.similarity_boost is not None:
                request.extra_params["similarity_boost"] = request.voice_settings.similarity_boost
            if request.voice_settings.style is not None:
                request.extra_params["style"] = request.voice_settings.style
            if request.voice_settings.use_speaker_boost is not None:
                request.extra_params["speaker_boost"] = request.voice_settings.use_speaker_boost

        # Perform new-style validation (raises on error)
        await self.validate_request(request)

        # Delegate to base class for actual API call
        response = await super().generate(request)
        # Ensure test-expected fields
        response.provider = self._provider_simple
        response.model = request.extra_params.get("model") or self._select_model(request)
        # Mark turbo flag for turbo model
        if response.model and str(response.model).startswith("eleven_turbo"):
            response.metadata["turbo"] = True
        return response

    async def generate_stream(self, request: TTSRequest) -> AsyncGenerator[bytes, None]:
        # Ensure initialization client exists if needed
        if not self.client:
            # Use centralized client (still httpx.AsyncClient) for policy defaults
            from tldw_Server_API.app.core.http_client import create_async_client
            self.client = create_async_client()

        # Prepare voice/model
        voice_id = self._get_voice_id(request.voice or "rachel")
        if request.model:
            request.extra_params["model"] = request.model
        model_id = self._select_model(request)

        url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
        headers = {
            "Accept": self._get_accept_header(request.format),
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        voice_settings = {
            "stability": request.extra_params.get("stability", self.stability),
            "similarity_boost": request.extra_params.get("similarity_boost", self.similarity_boost),
            "style": request.extra_params.get("style", self.style),
            "use_speaker_boost": request.extra_params.get("speaker_boost", self.use_speaker_boost),
        }
        payload = {"text": request.text, "model_id": model_id, "voice_settings": voice_settings}

        try:
            async for chunk in astream_bytes(
                method="POST",
                url=url,
                client=self.client,
                headers=headers,
                json=payload,
            ):
                if chunk:
                    yield chunk
        except _ELEVENLABS_NONCRITICAL_EXCEPTIONS as e:
            if _is_http_status_error(e):
                self._raise_mapped_http_error(e)
            raise

    # Override error mapping to align with tests (invalid voice -> validation error, 429 cases)
    def _raise_mapped_http_error(self, e: Exception) -> None:
        response = getattr(e, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
        try:
            data = response.json() if response is not None else {}
        except (AttributeError, TypeError, ValueError):
            data = {}
        detail = data.get("detail", {}) if isinstance(data, dict) else {}
        code = detail.get("status") or detail.get("code")

        if status in (401, 403):
            raise TTSAuthenticationError("elevenlabs authentication failed", provider=self._provider_simple)
        if status == 429:
            # Distinguish quota vs. rate limit when possible
            if code == "quota_exceeded":
                raise TTSQuotaExceededError("elevenlabs quota exceeded", provider=self._provider_simple)
            retry = None
            try:
                retry = int((response.headers or {}).get("retry-after", "0")) if response is not None else None
            except (TypeError, ValueError):
                retry = None
            err = rate_limit_error(self._provider_simple, retry_after=retry)
            # Expose retry_after directly for tests
            with contextlib.suppress(AttributeError, TypeError):
                err.retry_after = retry
            raise err
        if status and 400 <= status < 500 and code == "invalid_voice_id":
            from ..tts_exceptions import TTSValidationError
            message = None
            try:
                message = (detail.get("message") if isinstance(detail, dict) else None) or "Invalid voice id"
            except (AttributeError, TypeError):
                message = "Invalid voice id"
            raise TTSValidationError(message, provider=self._provider_simple, details={"status": status})

        # Fallback to base behavior
        return super()._raise_mapped_http_error(e)

#
# End of elevenlabs_adapter.py
#######################################################################################################################

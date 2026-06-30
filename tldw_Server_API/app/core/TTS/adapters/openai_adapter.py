# openai_adapter.py
# Description: OpenAI TTS API adapter implementation
#
# Imports
import asyncio
import os
from collections.abc import AsyncGenerator
from typing import Any, Optional

#
from loguru import logger

from tldw_Server_API.app.core.exceptions import NetworkError as CoreNetworkError
from tldw_Server_API.app.core.exceptions import RetryExhaustedError

#
# Local Imports
from tldw_Server_API.app.core.http_client import apost, astream_bytes

from ..tts_exceptions import (
    TTSAuthenticationError,
    TTSGenerationError,
    TTSNetworkError,
    TTSProviderError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSRateLimitError,
    TTSTimeoutError,
    TTSValidationError,
    auth_error,
    network_error,
    rate_limit_error,
    timeout_error,
)
from ..tts_resource_manager import get_resource_manager
from ..tts_validation import validate_tts_request
from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo

#
#######################################################################################################################
#
# OpenAI TTS Adapter Implementation

def _is_httpx_exception(exc: Exception) -> bool:
    module = getattr(exc.__class__, "__module__", "")
    return module.startswith("httpx")


def _is_http_status_error(exc: Exception) -> bool:
    if not _is_httpx_exception(exc):
        return False
    return exc.__class__.__name__ == "HTTPStatusError"


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    name = exc.__class__.__name__.lower()
    return "timeout" in name


def _safe_exception_label(exc: BaseException) -> str:
    """Return a non-sensitive exception identifier for logs."""
    return type(exc).__name__


class OpenAIAdapter(TTSAdapter):
    """Adapter for OpenAI's TTS API.
    Note: This class implements all abstract methods so it can be instantiated
    directly by tests that import OpenAIAdapter (legacy name).
    """

    # OpenAI voice definitions
    VOICES = {
        "alloy": VoiceInfo(
            id="alloy",
            name="Alloy",
            gender="neutral",
            description="Neutral and balanced voice"
        ),
        "echo": VoiceInfo(
            id="echo",
            name="Echo",
            gender="male",
            description="Male voice with clarity"
        ),
        "fable": VoiceInfo(
            id="fable",
            name="Fable",
            gender="neutral",
            description="Expressive and dynamic voice"
        ),
        "onyx": VoiceInfo(
            id="onyx",
            name="Onyx",
            gender="male",
            description="Deep and authoritative male voice"
        ),
        "nova": VoiceInfo(
            id="nova",
            name="Nova",
            gender="female",
            description="Warm and friendly female voice"
        ),
        "shimmer": VoiceInfo(
            id="shimmer",
            name="Shimmer",
            gender="female",
            description="Soft and pleasant female voice"
        )
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        # Normalize placeholder/empty keys often present in test envs to None
        if isinstance(self.api_key, str):
            _raw = self.api_key.strip()
            placeholder_tokens = {
                "<openai_api_key>",
                "your-openai-api-key",
                "your_openai_api_key",
                "sk-mock-key-12345",
                "",
                "none",
                "null",
            }
            if _raw.lower() in placeholder_tokens:
                self.api_key = None
        self.base_url = self.config.get("openai_base_url", "https://api.openai.com/v1/audio/speech")
        # Support both legacy and new config keys for model selection
        self.model = (
            self.config.get("openai_tts_model")
            or self.config.get("openai_model")
            or "tts-1"
        )  # e.g., "tts-1" or "tts-1-hd"
        self.client: Optional[Any] = None
        # Optional: perform a lightweight API-key verification call during
        # initialize() when enabled via configuration.
        self._verify_api_key_on_init: bool = bool(self.config.get("verify_api_key_on_init"))

        if not self.api_key:
            logger.warning(f"{self.provider_name}: API key not configured")
            self._status = ProviderStatus.NOT_CONFIGURED

    async def initialize(self) -> bool:
        """Initialize the OpenAI adapter"""
        try:
            if not self.api_key:
                error_msg = f"{self.provider_name}: Cannot initialize without API key"
                logger.error(error_msg)
                self._status = ProviderStatus.NOT_CONFIGURED
                raise TTSProviderNotConfiguredError(error_msg, provider=self.provider_name)

            # Get HTTP client from resource manager. By default we avoid
            # making a network call here so that initialization does not
            # depend on external API availability. When explicitly enabled
            # via configuration, a lightweight API-key verification call
            # can be performed below.
            resource_manager = await get_resource_manager()
            self.client = await resource_manager.get_http_client(
                provider=self.provider_name.lower(),
                base_url=self.base_url
            )

            # Prepare auth headers for subsequent requests (and optional verify).
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Optional: best-effort API key verification on init.
            # This is disabled by default to keep startup fast and resilient
            # to transient network issues. When enabled, only clear auth
            # failures are treated as fatal; other errors are logged and
            # deferred to the first real request.
            if self._verify_api_key_on_init and self.client is not None:
                try:
                    payload = {
                        "model": self.model,
                        "input": "test",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "speed": 1.0,
                    }
                    # Reuse the same error-mapping logic as normal requests.
                    try:
                        await self._generate_complete(headers, payload)
                    except Exception as e:
                        if not _is_http_status_error(e):
                            raise
                        # Map HTTP errors to TTS exceptions so that clear
                        # auth failures are treated as fatal below while
                        # other conditions remain non-fatal.
                        await self._handle_http_status_error(e)
                    logger.info(f"{self.provider_name}: API key verified during initialization")
                except TTSAuthenticationError as auth_exc:
                    logger.error(
                        f"{self.provider_name}: API key verification failed during initialization; "
                        f"exception_type={_safe_exception_label(auth_exc)}"
                    )
                    self._status = ProviderStatus.ERROR
                    raise TTSProviderInitializationError(
                        f"Failed to initialize {self.provider_name}: authentication failed",
                        provider=self.provider_name,
                        details={"error": str(auth_exc), "error_type": type(auth_exc).__name__},
                    ) from auth_exc
                except (TTSRateLimitError, TTSNetworkError, TTSTimeoutError, TTSProviderError) as non_fatal:
                    logger.warning(
                        f"{self.provider_name}: API key verification during initialization did not succeed "
                        f"(exception_type={_safe_exception_label(non_fatal)}). Continuing initialization; "
                        "the first real request will surface any persistent issues."
                    )
                except Exception as exc:
                    logger.warning(
                        f"{self.provider_name}: Unexpected error during API key verification on init; "
                        f"exception_type={_safe_exception_label(exc)}. "
                        "Continuing initialization."
                    )

            # Mark initialized and cache capabilities for direct initialize() calls
            self._capabilities = await self.get_capabilities()
            self._initialized = True
            self._status = ProviderStatus.AVAILABLE
            logger.info(f"{self.provider_name}: Initialized successfully")
            return True

        except TTSProviderNotConfiguredError:
            return False
        except Exception as e:
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

    async def get_capabilities(self) -> TTSCapabilities:
        """Get OpenAI TTS capabilities"""
        return TTSCapabilities(
            provider_name="OpenAI",
            supported_languages={"en"},  # OpenAI TTS primarily supports English
            supported_voices=list(self.VOICES.values()),
            supported_formats={
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.AAC,
                AudioFormat.FLAC,
                AudioFormat.WAV,
                AudioFormat.PCM
            },
            max_text_length=4096,
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
            latency_ms=200,  # Approximate
            sample_rate=24000,
            default_format=AudioFormat.MP3
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using OpenAI TTS API"""
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

        # Map voice if needed
        voice = self.map_voice(request.voice or "alloy")

        # Prepare request payload
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "input": self.preprocess_text(request.text),
            "voice": voice,
            "response_format": request.format.value,
            "speed": request.speed
        }

        logger.info(
            f"{self.provider_name}: Generating speech with model={self.model}, "
            f"voice={voice}, format={request.format.value}"
        )

        try:
            if request.stream:
                # Return streaming response using POST to align with tests
                return TTSResponse(
                    audio_stream=self._stream_audio(headers, payload),
                    format=request.format,
                    sample_rate=24000,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )
            else:
                # Return complete audio
                audio_data = await self._generate_complete(headers, payload)
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=24000,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )

        except Exception as e:
            await self._raise_normalized_request_error(e)

    async def _handle_http_status_error(self, e: Exception) -> None:
        """Normalize HTTP status errors into TTS-specific exceptions."""
        response = getattr(e, "response", None)
        status_code = getattr(response, "status_code", None)
        headers = getattr(response, "headers", {}) if response is not None else {}
        error_msg = ""
        if response is not None:
            try:
                if hasattr(response, "aread"):
                    error_content = await response.aread()
                else:
                    error_content = response.read()
                error_msg = error_content.decode()
            except Exception:
                error_msg = ""
        logger.error(
            f"{self.provider_name} API error: {status_code}; "
            f"exception_type={_safe_exception_label(e)}; response body redacted"
        )

        if status_code == 401:
            # Standardize message and provider fields
            raise auth_error(self.provider_name, "Invalid API key")
        elif status_code == 429:
            # Try to extract retry-after header
            retry_after = headers.get("retry-after") if isinstance(headers, dict) else None
            raise rate_limit_error(
                self.provider_name,
                retry_after=int(retry_after) if retry_after else None,
            )
        elif status_code == 400:
            raise TTSProviderError(
                f"Invalid request to OpenAI: {error_msg}",
                provider=self.provider_name,
                error_code="BAD_REQUEST",
            )
        else:
            raise TTSProviderError(
                f"OpenAI API error: {error_msg}",
                provider=self.provider_name,
                error_code=str(status_code),
            )

    async def _raise_normalized_request_error(self, e: Exception) -> None:
        if _is_http_status_error(e):
            await self._handle_http_status_error(e)
        if isinstance(e, (CoreNetworkError, RetryExhaustedError)) or _is_httpx_exception(e):
            logger.error(
                f"{self.provider_name} network/timeout error; "
                f"exception_type={_safe_exception_label(e)}"
            )
            reason = str(e) or e.__class__.__name__
            if _is_timeout_error(e) or "timeout" in reason.lower():
                raise timeout_error(self.provider_name, timeout_seconds=60.0) from e
            raise network_error(self.provider_name, e) from e
        if not isinstance(
            e,
            (
                TTSProviderError,
                TTSAuthenticationError,
                TTSRateLimitError,
                TTSNetworkError,
                TTSTimeoutError,
            ),
        ):
            logger.error(
                f"{self.provider_name} unexpected error; "
                f"exception_type={_safe_exception_label(e)}"
            )
            raise TTSProviderError(
                f"Unexpected error in {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e), "error_type": type(e).__name__}
            ) from e
        raise e

    async def _stream_audio(
        self,
        headers: dict[str, str],
        payload: dict[str, Any]
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from OpenAI API with egress policy enforcement."""
        try:
            logger.debug(f"{self.provider_name}: _stream_audio calling apost url={self.base_url}")
            response = await apost(
                url=self.base_url,
                client=self.client,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            total_bytes = 0
            try:
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    yield chunk
                logger.debug(f"{self.provider_name}: Streamed {total_bytes} bytes")
            finally:
                try:
                    if hasattr(response, "aclose"):
                        await response.aclose()  # type: ignore[func-returns-value]
                except Exception as stream_close_error:
                    logger.debug(
                        "OpenAI TTS response close after stream failed; "
                        f"exception_type={_safe_exception_label(stream_close_error)}"
                    )
        except Exception as e:
            logger.error(
                f"{self.provider_name} streaming error; "
                f"exception_type={_safe_exception_label(e)}"
            )
            await self._raise_normalized_request_error(e)

    async def _generate_complete(
        self,
        headers: dict[str, str],
        payload: dict[str, Any]
    ) -> bytes:
        """Generate complete audio from OpenAI API"""
        logger.debug(f"{self.provider_name}: _generate_complete calling apost url={self.base_url}")
        response = await apost(
            url=self.base_url,
            client=self.client,
            headers=headers,
            json=payload,
        )
        logger.debug(f"{self.provider_name}: _generate_complete received response status={getattr(response, 'status_code', 'n/a')}")
        response.raise_for_status()
        return response.content

    async def _cleanup_resources(self):
        """Clean up OpenAI adapter resources"""
        # Note: HTTP clients are now managed by the resource manager
        # No need to manually close them as they use connection pooling
        try:
            # Clear reference to client
            self.client = None
            logger.debug(f"{self.provider_name}: Resources cleaned up")
        except Exception as e:
            logger.warning(
                f"{self.provider_name}: Error during cleanup; "
                f"exception_type={_safe_exception_label(e)}"
            )

    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to OpenAI voice"""
        # Check if it's already a valid OpenAI voice
        if voice_id in self.VOICES:
            return voice_id

        # Try common mappings
        voice_mappings = {
            "male": "onyx",
            "female": "nova",
            "neutral": "alloy",
            "deep": "onyx",
            "soft": "shimmer",
            "expressive": "fable",
            "narrator": "fable",
        }

        return voice_mappings.get(voice_id.lower(), "alloy")

# Backward-compat alias expected by some tests
class OpenAITTSAdapter(OpenAIAdapter):
    """Compatibility wrapper with extended OpenAI interface for TTS_NEW tests.

    - Accepts generic config keys (api_key, base_url, timeout)
    - Exposes convenience attributes/methods expected by the new tests
    - Performs raise-style validation for invalid inputs
    """

    PROVIDER_KEY = "openai"
    SUPPORTED_MODELS = ["tts-1", "tts-1-hd"]

    def __init__(self, config: Optional[dict[str, Any]] = None):
        cfg = config.copy() if isinstance(config, dict) else {}
        mapped_cfg: dict[str, Any] = {}
        if "api_key" in cfg:
            mapped_cfg["openai_api_key"] = cfg.get("api_key")
        if "openai_api_key" in cfg and "openai_api_key" not in mapped_cfg:
            mapped_cfg["openai_api_key"] = cfg.get("openai_api_key")
        if "base_url" in cfg:
            # The base class expects the speech endpoint URL
            base = cfg.get("base_url")
            if base and base.endswith("/v1"):
                base = base + "/audio/speech"
            mapped_cfg["openai_base_url"] = base or cfg.get("openai_base_url")
        elif "openai_base_url" in cfg:
            mapped_cfg["openai_base_url"] = cfg.get("openai_base_url")
        if "timeout" in cfg:
            mapped_cfg["timeout"] = cfg.get("timeout")

        # If API key not present, tests expect construction to fail
        temp_key = mapped_cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not temp_key:
            raise TTSProviderNotConfiguredError("OpenAI API key not configured", provider="openai")

        super().__init__(mapped_cfg)
        self._provider_simple = "openai"
        self._timeout = cfg.get("timeout")

    # --- Simple attributes/properties expected by tests ---
    @property
    def provider(self) -> str:
        return self._provider_simple

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def supported_models(self) -> list:
        return list(self.SUPPORTED_MODELS)

    @property
    def supported_voices(self) -> list:
        return list(self.VOICES.keys())

    # --- Convenience API ---
    async def validate_request(self, request: TTSRequest) -> None:
        """Raise on invalid requests (new-test-friendly behavior)."""
        # Text presence and length
        if not request.text or not str(request.text).strip():
            from ..tts_exceptions import TTSInvalidInputError
            raise TTSInvalidInputError("Text cannot be empty", provider=self._provider_simple)
        if len(request.text) > 4096:
            from ..tts_exceptions import TTSTextTooLongError
            raise TTSTextTooLongError("Text exceeds maximum for OpenAI", provider=self._provider_simple)

        # Model validity (when provided)
        if request.model and request.model not in self.SUPPORTED_MODELS:
            raise TTSValidationError("Invalid model for OpenAI", provider=self._provider_simple, details={"model": request.model})

        # Voice validity
        voice = (request.voice or "alloy").lower()
        if voice not in self.VOICES:
            raise TTSValidationError("Invalid voice for OpenAI", provider=self._provider_simple, details={"voice": request.voice})

        # Speed bounds
        if request.speed is not None and not (0.25 <= float(request.speed) <= 4.0):
            raise TTSValidationError("Speed out of range", provider=self._provider_simple, details={"speed": request.speed})

    async def generate(self, request: TTSRequest) -> TTSResponse:
        # Perform new-style validation (raises on error)
        await self.validate_request(request)

        # Map model hint onto base adapter configuration for this call
        # The base adapter already honors request.speed/format/voice mapping
        if request.model:
            # Prefer request-specific model by temporarily overriding self.model
            old_model = getattr(self, "model", None)
            self.model = request.model
        else:
            old_model = None

        try:
            resp = await super().generate(request)
        except (TTSAuthenticationError, TTSRateLimitError, TTSNetworkError, TTSTimeoutError):
            # Preserve typed provider failures so callers and health checks can
            # classify auth/network/rate-limit issues accurately.
            raise
        except TTSProviderError as e:
            # Normalize to generation error for tests
            raise TTSGenerationError(str(e), provider=self._provider_simple, details={"error_type": type(e).__name__}) from e
        except Exception as e:
            if _is_httpx_exception(e):
                raise TTSGenerationError(str(e), provider=self._provider_simple, details={"error_type": type(e).__name__}) from e
            raise
        finally:
            if old_model is not None:
                self.model = old_model

        # Ensure test-expected fields/metadata
        resp.provider = self._provider_simple
        resp.model = request.model or getattr(self, "model", None)
        if request.text:
            resp.metadata["characters"] = len(request.text)
        # audio_content is synchronized in TTSResponse __post_init__
        return resp

    async def generate_stream(self, request: TTSRequest) -> AsyncGenerator[bytes, None]:
        # Basic validation; allow streaming in tests
        await self.validate_request(request)
        model = request.model or getattr(self, "model", "tts-1")
        voice = self.map_voice(request.voice or "alloy")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "input": self.preprocess_text(request.text),
            "voice": voice,
            "response_format": request.format.value,
            "speed": request.speed
        }

        client = self.client
        try:
            async for chunk in astream_bytes(
                method="POST",
                url=self.base_url,
                headers=headers,
                json=payload,
                client=client,
            ):
                if chunk:
                    yield chunk
        except Exception as e:
            if isinstance(e, (CoreNetworkError, RetryExhaustedError)) or _is_httpx_exception(e):
                # Wrap network/API issues as generation errors per tests
                raise TTSGenerationError(str(e), provider=self._provider_simple) from e
            raise

    def get_info(self) -> dict[str, Any]:
        return {
            "provider": self._provider_simple,
            "models": list(self.SUPPORTED_MODELS),
            "voices": list(self.VOICES.keys()),
            "max_characters": 4096,
        }

#
# End of openai_adapter.py
#######################################################################################################################

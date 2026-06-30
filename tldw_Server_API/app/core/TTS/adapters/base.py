# base.py
# Description: Base adapter classes and interfaces for TTS providers
#
# Imports
import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

#
# Third-party Imports
import numpy as np
from loguru import logger

#
# Local Imports
#
#######################################################################################################################
#
# Enums and Data Classes

class AudioFormat(Enum):
    """Supported audio output formats"""
    MP3 = "mp3"
    WAV = "wav"
    OPUS = "opus"
    FLAC = "flac"
    AAC = "aac"
    PCM = "pcm"
    OGG = "ogg"
    WEBM = "webm"
    ULAW = "ulaw"  # μ-law encoding used by telephony systems

class ProviderStatus(Enum):
    """Provider availability status"""
    AVAILABLE = "available"
    INITIALIZING = "initializing"
    ERROR = "error"
    DISABLED = "disabled"
    NOT_CONFIGURED = "not_configured"

@dataclass
class VoiceInfo:
    """Information about an available voice"""
    id: str
    name: str
    gender: Optional[str] = None
    language: str = "en"
    accent: Optional[str] = None
    age: Optional[str] = None
    description: Optional[str] = None
    preview_url: Optional[str] = None
    styles: list[str] = field(default_factory=list)
    use_case: list[str] = field(default_factory=list)

@dataclass
class TTSCapabilities:
    """Capabilities of a TTS provider"""
    provider_name: str
    supported_languages: set[str]
    supported_voices: list[VoiceInfo]
    supported_formats: set[AudioFormat]
    max_text_length: int
    supports_streaming: bool
    supports_voice_cloning: bool = False
    supports_emotion_control: bool = False
    supports_speech_rate: bool = True
    supports_pitch_control: bool = False
    supports_volume_control: bool = False
    supports_ssml: bool = False
    supports_phonemes: bool = False
    supports_multi_speaker: bool = False
    supports_background_audio: bool = False
    latency_ms: Optional[int] = None  # Average latency in milliseconds
    sample_rate: int = 24000
    default_format: AudioFormat = AudioFormat.MP3
    metadata: dict[str, Any] = field(default_factory=dict)

# Backward-compatibility for tests expecting VoiceSettings
@dataclass
class VoiceSettings:
    """Voice settings and metadata (adapter-agnostic).

    Supports both simple metadata (id/name) and provider-specific tuning
    such as ElevenLabs' voice settings.
    """
    # Optional metadata
    id: Optional[str] = None
    name: Optional[str] = None
    language: str = "en"
    gender: Optional[str] = None
    # Style can be numeric intensity for some providers
    style: Optional[float] = None
    # ElevenLabs tuning fields (optional)
    stability: Optional[float] = None
    similarity_boost: Optional[float] = None
    use_speaker_boost: Optional[bool] = None

@dataclass
class TTSRequest:
    """Unified TTS request format"""
    text: str
    voice: Optional[str] = None
    language: Optional[str] = "en"
    format: AudioFormat = AudioFormat.MP3
    target_sample_rate: Optional[int] = None
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    emotion: Optional[str] = None
    emotion_intensity: float = 1.0
    style: Optional[str] = None
    voice_reference: Optional[bytes] = None  # For voice cloning
    ssml: bool = False
    stream: bool = True
    # Optional provider/model hints for adapter selection
    provider: Optional[str] = None
    model: Optional[str] = None
    # Multi-speaker dialogue support
    speakers: Optional[dict[str, str]] = None  # Speaker ID to voice mapping

    # Advanced generation parameters (especially for VibeVoice)
    seed: Optional[int] = None  # For reproducible generation
    cfg_scale: Optional[float] = None  # 1.0-2.0, controls generation guidance
    diffusion_steps: Optional[int] = None  # 5-100, quality vs speed
    temperature: Optional[float] = None  # 0.1-2.0, sampling randomness
    top_p: Optional[float] = None  # 0.1-1.0, nucleus sampling
    attention_type: Optional[str] = None  # auto, sdpa, flash_attention_2, eager

    # Structured voice settings (e.g., ElevenLabs)
    voice_settings: Optional[VoiceSettings] = None
    # Additional provider-specific parameters
    extra_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Clamp speed into a broadly accepted range
        try:
            self._original_speed = self.speed
            if self.speed is not None:
                # Project-wide contract (see tests): 0.25 <= speed <= 4.0
                if self.speed < 0.25:
                    self.speed = 0.25
                elif self.speed > 4.0:
                    self.speed = 4.0
        except Exception as speed_validation_error:
            logger.debug(
                "Voice settings speed normalization failed; exception_type={}",
                type(speed_validation_error).__name__,
            )
        try:
            self._original_pitch = self.pitch
        except Exception:
            self._original_pitch = None
        try:
            self._original_volume = self.volume
        except Exception:
            self._original_volume = None
        # Coerce provider/model to lowercase strings if present
        try:
            if isinstance(self.provider, str):
                self.provider = self.provider.lower()
            if isinstance(self.model, str):
                self.model = self.model.lower()
        except Exception as casing_error:
            logger.debug(
                "TTS provider/model lowercase normalization failed; exception_type={}",
                type(casing_error).__name__,
            )
        # If voice_settings arrives as a plain dict (from round-trip), coerce to VoiceSettings
        try:
            if isinstance(self.voice_settings, dict):
                self.voice_settings = VoiceSettings(**self.voice_settings)
        except Exception as settings_coercion_error:
            # If coercion fails, leave as-is; validation layer may catch it later
            logger.debug(
                "Voice settings coercion from dict failed; exception_type={}",
                type(settings_coercion_error).__name__,
            )

    def dict(self) -> dict[str, Any]:
        """Return a dictionary representation (compat with tests)."""
        from dataclasses import asdict
        return asdict(self)

@dataclass
class TTSResponse:
    """Response from TTS generation"""
    # Support both legacy 'audio_data' and newer 'audio_content' naming.
    audio_data: Optional[bytes] = None
    audio_content: Optional[bytes] = None
    audio_stream: Optional[AsyncGenerator[bytes, None]] = None
    format: AudioFormat = AudioFormat.MP3
    sample_rate: int = 24000
    channels: int = 1
    # Support both 'duration' and 'duration_seconds' naming
    duration: Optional[float] = None
    duration_seconds: Optional[float] = None
    text_processed: Optional[str] = None
    voice_used: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Keep audio_data and audio_content in sync for compatibility
        if self.audio_content is None and self.audio_data is not None:
            self.audio_content = self.audio_data
        elif self.audio_data is None and self.audio_content is not None:
            self.audio_data = self.audio_content
        # Keep duration aliases in sync
        if self.duration_seconds is None and self.duration is not None:
            self.duration_seconds = self.duration
        elif self.duration is None and self.duration_seconds is not None:
            self.duration = self.duration_seconds

    def dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class TTSAdapter(ABC):
    """
    Abstract base class for TTS provider adapters.
    All TTS providers must implement this interface.
    """
    # Adapters can opt out of service-level chunking if they handle it internally.
    handles_text_chunking: bool = False

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize the adapter with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config or {}
        self._status = ProviderStatus.NOT_CONFIGURED
        self._capabilities: Optional[TTSCapabilities] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def status(self) -> ProviderStatus:
        """Get current provider status"""
        return self._status

    @property
    def capabilities(self) -> Optional[TTSCapabilities]:
        """Get provider capabilities"""
        return self._capabilities

    @property
    def provider_name(self) -> str:
        """Get provider name"""
        return self.__class__.__name__.replace('Adapter', '')

    @property
    def provider_key(self) -> str:
        """Get canonical provider key for config/metrics/validation."""
        provider_key = getattr(self, "PROVIDER_KEY", None)
        if isinstance(provider_key, str) and provider_key.strip():
            return provider_key.strip().lower()
        name = self.provider_name
        if isinstance(name, str) and name:
            return name.lower()
        return self.__class__.__name__.lower()

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the TTS provider (load models, connect to API, etc).

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """
        Generate speech from text.

        Args:
            request: TTSRequest object with generation parameters

        Returns:
            TTSResponse object with audio data or stream
        """
        pass

    @abstractmethod
    async def get_capabilities(self) -> TTSCapabilities:
        """
        Get the capabilities of this provider.

        Returns:
            TTSCapabilities object describing what this provider supports
        """
        pass

    async def validate_request(self, request: TTSRequest) -> tuple[bool, Optional[str]]:
        """
        Validate if the request can be handled by this provider.

        Args:
            request: TTSRequest to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._initialized:
            return False, f"{self.provider_name} not initialized"

        if not self._capabilities:
            return False, f"{self.provider_name} capabilities not available"

        # Check text length
        if len(request.text) > self._capabilities.max_text_length:
            return False, f"Text exceeds maximum length of {self._capabilities.max_text_length}"

        # Check format support
        if request.format not in self._capabilities.supported_formats:
            return False, f"Format {request.format.value} not supported"

        # Check language support
        if request.language and request.language not in self._capabilities.supported_languages:
            return False, f"Language {request.language} not supported"

        # Check streaming support
        if request.stream and not self._capabilities.supports_streaming:
            return False, "Streaming not supported"

        # Check voice cloning
        if request.voice_reference and not self._capabilities.supports_voice_cloning:
            return False, "Voice cloning not supported"

        # Check emotion control
        if request.emotion and not self._capabilities.supports_emotion_control:
            return False, "Emotion control not supported"

        return True, None

    async def ensure_initialized(self) -> bool:
        """
        Ensure the provider is initialized (thread-safe).

        Returns:
            bool: True if initialized successfully
        """
        if self._initialized:
            return True

        async with self._init_lock:
            if self._initialized:
                return True

            try:
                self._status = ProviderStatus.INITIALIZING
                success = await self.initialize()
                if success:
                    self._capabilities = await self.get_capabilities()
                    self._status = ProviderStatus.AVAILABLE
                    self._initialized = True
                else:
                    self._status = ProviderStatus.ERROR
                return success
            except Exception as e:
                logger.error(
                    f"{self.provider_name} initialization failed; "
                    f"exception_type={type(e).__name__}"
                )
                self._status = ProviderStatus.ERROR
                return False

    async def convert_audio_format(
        self,
        audio_data: np.ndarray,
        source_format: AudioFormat,
        target_format: AudioFormat,
        sample_rate: int = 24000
    ) -> bytes:
        """
        Convert audio between formats.

        Args:
            audio_data: Audio data as numpy array
            source_format: Source audio format
            target_format: Target audio format
            sample_rate: Sample rate of the audio

        Returns:
            Converted audio as bytes
        """
        # This will use the existing StreamingAudioWriter
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import AudioNormalizer, StreamingAudioWriter

        normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=target_format.value,
            sample_rate=sample_rate,
            channels=1
        )

        try:
            # Normalize to int16 if needed
            if audio_data.dtype != np.int16:
                audio_data = normalizer.normalize(audio_data, target_dtype=np.int16)

            # Write and finalize
            chunk_bytes = writer.write_chunk(audio_data) or b""
            final_bytes = writer.write_chunk(finalize=True) or b""

            if target_format == AudioFormat.PCM:
                # PCM returns raw payload during the initial write
                return chunk_bytes

            return chunk_bytes + final_bytes
        finally:
            writer.close()

    async def close(self):
        """Clean up resources"""
        try:
            # Perform adapter-specific cleanup
            await self._cleanup_resources()
            self._initialized = False
            self._status = ProviderStatus.DISABLED
            logger.info(f"{self.provider_name} adapter closed")
        except Exception as e:
            logger.error(
                f"Error closing {self.provider_name} adapter; "
                f"exception_type={type(e).__name__}"
            )

    async def _cleanup_resources(self):  # noqa: B027
        """Override this method for adapter-specific cleanup"""
        pass

    def map_voice(self, voice_id: str) -> str:
        """
        Map a generic voice ID to provider-specific voice.

        Args:
            voice_id: Generic voice identifier

        Returns:
            Provider-specific voice identifier
        """
        # Default implementation returns the same ID
        # Subclasses can override for custom mapping
        return voice_id

    def preprocess_text(self, text: str, **kwargs) -> str:
        """
        Preprocess text before sending to TTS engine.

        Args:
            text: Input text
            **kwargs: Additional preprocessing options

        Returns:
            Preprocessed text
        """
        # Default implementation does minimal preprocessing
        # Subclasses can override for provider-specific needs
        return text.strip()

    def parse_dialogue(self, text: str) -> list[tuple[str, str]]:
        """
        Parse multi-speaker dialogue from text.

        Args:
            text: Text potentially containing speaker markers

        Returns:
            List of (speaker, text) tuples
        """
        # Simple implementation for dialogue parsing
        # Format: "Speaker1: Hello there! Speaker2: Hi!"
        import re

        pattern = r'([A-Za-z0-9]+):\s*([^:]+?)(?=(?:[A-Za-z0-9]+:|$))'
        matches = re.findall(pattern, text)
        # Strip whitespace from dialogue text
        matches = [(speaker, dialogue.strip()) for speaker, dialogue in matches]

        if matches:
            return matches
        else:
            return [("default", text)]

    async def stream_audio(
        self,
        audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks with any necessary processing.

        Args:
            audio_chunks: Async generator of audio chunks

        Yields:
            Processed audio chunks
        """
        async for chunk in audio_chunks:
            if chunk:
                yield chunk

#
# End of base.py
#######################################################################################################################

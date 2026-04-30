# dia_adapter.py
# Description: Dia TTS adapter implementation for ultra-realistic dialogue generation
#
# Imports
import asyncio
import contextlib
import importlib
import os
import re
import sys
from collections.abc import AsyncGenerator
from typing import Any, Optional

import numpy as np

#
# Third-party Imports
from loguru import logger

from ..tts_exceptions import (
    TTSGPUError,
    TTSModelLoadError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
)
from ..tts_resource_manager import get_resource_manager
from ..tts_validation import validate_tts_request
from ..utils import parse_bool

#
# Local Imports
from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo

#
#######################################################################################################################
#
# Dia TTS Adapter Implementation

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


def _safe_exception_label(exc: BaseException | None) -> str:
    if exc is None:
        return "unknown"
    return type(exc).__name__


class DiaAdapter(TTSAdapter):
    """Adapter for Dia TTS model (dialogue generation specialist)"""

    # Dia special tags for nonverbal audio
    NONVERBAL_TAGS = {
        "(laughs)", "(coughs)", "(gasps)", "(sighs)", "(clears throat)",
        "(sniffles)", "(yawns)", "(groans)", "(whispers)", "(shouts)",
        "(mumbles)", "(stutters)", "(pauses)", "(hesitates)", "(breathes)"
    }

    # Default speaker voices (Dia generates dynamic voices)
    DEFAULT_SPEAKERS = {
        "speaker1": VoiceInfo(
            id="speaker1",
            name="Speaker 1",
            gender="neutral",
            description="Primary dialogue speaker"
        ),
        "speaker2": VoiceInfo(
            id="speaker2",
            name="Speaker 2",
            gender="neutral",
            description="Secondary dialogue speaker"
        ),
        "speaker3": VoiceInfo(
            id="speaker3",
            name="Speaker 3",
            gender="neutral",
            description="Tertiary dialogue speaker"
        ),
        "narrator": VoiceInfo(
            id="narrator",
            name="Narrator",
            gender="neutral",
            description="Narration voice"
        )
    }
    # Expose presets alias for tests and UI parity
    VOICE_PRESETS = DEFAULT_SPEAKERS

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)

        # Model configuration
        self.model_path = self.config.get("dia_model_path", "nari-labs/dia")
        self.model_revision = self.config.get("dia_model_revision") or os.getenv("DIA_MODEL_REVISION")
        # Device selection: prefer explicit; fallback to CUDA if available else CPU
        preferred = self.config.get("dia_device")
        if preferred:
            pref = str(preferred).lower()
            if pref == "cuda":
                self.device = "cuda" if _torch_cuda_available(allow_import=True) else "cpu"
            elif pref == "cpu":
                self.device = "cpu"
            elif pref == "mps":
                self.device = "mps" if _torch_mps_available(allow_import=True) else ("cuda" if _torch_cuda_available(allow_import=True) else "cpu")
            else:
                self.device = "cuda" if _torch_cuda_available(allow_import=True) else "cpu"
        else:
            self.device = "cuda" if _torch_cuda_available(allow_import=True) else "cpu"

        # Auto-download toggle: config override > env overrides > default True
        cfg_auto = self.config.get("dia_auto_download")
        env_auto = os.getenv("DIA_AUTO_DOWNLOAD") or os.getenv("TTS_AUTO_DOWNLOAD")
        self.auto_download = parse_bool(cfg_auto, default=parse_bool(env_auto, default=True))

        # Audio configuration
        self.sample_rate = self.config.get("dia_sample_rate", 24000)

        # Model instances
        self.model = None
        self.processor = None

        # Dialogue processing
        self.auto_detect_speakers = self.config.get("dia_auto_detect_speakers", True)
        self.max_speakers = self.config.get("dia_max_speakers", 5)

        # Performance settings
        self.use_safetensors = self.config.get("dia_use_safetensors", True)
        self.use_bf16 = self.config.get("dia_use_bf16", True) and self.device == "cuda"

    async def initialize(self) -> bool:
        """Initialize the Dia TTS model"""
        if _get_torch(allow_import=True) is None:
            logger.warning(
                f"{self.provider_name}: torch unavailable; disabling provider. "
                f"exception_type={_safe_exception_label(_TORCH_IMPORT_ERROR)}"
            )
            self._status = ProviderStatus.NOT_CONFIGURED
            return False
        try:
            logger.info(f"{self.provider_name}: Loading Dia TTS model (1.6B parameters)...")

            # Get resource manager for memory monitoring
            resource_manager = await get_resource_manager()

            # Load model and processor (callable for testing patching)
            await self._load_dia_model()

            # Register model with resource manager
            if self.model:
                register_result = resource_manager.register_model(
                    provider=self.provider_name.lower(),
                    model_instance=self.model,
                    cleanup_callback=self._cleanup_resources,
                    model_key=str(self.model_path),
                )
                if asyncio.iscoroutine(register_result):
                    await register_result

            logger.info(
                f"{self.provider_name}: Initialized successfully "
                f"(Device: {self.device}, BF16: {self.use_bf16})"
            )
            self._status = ProviderStatus.AVAILABLE
            return True

        except TTSModelLoadError:
            raise
        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                raise TTSGPUError(
                    f"GPU error initializing {self.provider_name}",
                    provider=self.provider_name,
                    details={"error": str(e), "device": self.device}
                ) from e
            raise
        except Exception as e:
            logger.error(
                f"{self.provider_name}: Initialization failed; "
                f"exception_type={_safe_exception_label(e)}"
            )
            self._status = ProviderStatus.ERROR
            raise TTSProviderInitializationError(
                f"Failed to initialize {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e), "model_path": self.model_path}
            ) from e

    async def _load_dia_model(self) -> bool:
        """Load Dia processor and model. Split out for easier testing/mocking."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except ImportError as e:
            logger.error(f"{self.provider_name}: transformers library not installed")
            self._status = ProviderStatus.NOT_CONFIGURED
            raise TTSModelLoadError(
                "Failed to import required dependencies",
                provider=self.provider_name,
                details={"error": str(e), "suggestion": "pip install transformers torch"}
            ) from e

        # Load processor
        logger.info(f"{self.provider_name}: Loading processor from {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            trust_remote_code=True,
            local_files_only=(not self.auto_download)
        )  # nosec B615

        # Load model with appropriate dtype
        logger.info(f"{self.provider_name}: Loading model from {self.model_path}")
        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": self.use_safetensors,
        }
        torch_mod = _get_torch(allow_import=True)
        if torch_mod is None:
            raise TTSModelLoadError(
                "Failed to import required dependencies",
                provider=self.provider_name,
                details={"error": str(_TORCH_IMPORT_ERROR), "suggestion": "pip install torch"},
            )
        if self.use_bf16:
            model_kwargs["torch_dtype"] = torch_mod.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            revision=self.model_revision,
            local_files_only=(not self.auto_download),
            **model_kwargs,
        )  # nosec B615
        self.model = self.model.to(self.device)
        self.model.eval()
        return True

    async def get_capabilities(self) -> TTSCapabilities:
        """Get Dia TTS capabilities"""
        return TTSCapabilities(
            provider_name="Dia",
            supported_languages={"en"},  # Currently English only
            supported_voices=list(self.DEFAULT_SPEAKERS.values()),
            supported_formats={
                AudioFormat.WAV,
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.FLAC,
                AudioFormat.PCM
            },
            max_text_length=30000,
            supports_streaming=True,
            supports_voice_cloning=True,  # Via audio prompts
            supports_emotion_control=False,  # Emotion through nonverbal tags
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=True,  # Core feature - multi-speaker dialogue
            supports_background_audio=False,
            latency_ms=500 if self.device == "cuda" else 3000,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Dia TTS"""
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

        # Process text for dialogue
        dialogue_parts = self._process_dialogue(request.text, request.speakers)

        logger.info(
            f"{self.provider_name}: Generating dialogue with {len(dialogue_parts)} parts, "
            f"format={request.format.value}"
        )

        try:
            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio_dia(dialogue_parts, request),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    provider=self.provider_name,
                    metadata={"dialogue_parts": len(dialogue_parts)}
                )
            else:
                # Generate complete audio
                audio_data = await self._generate_complete_dia(dialogue_parts, request)
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    provider=self.provider_name,
                    metadata={"dialogue_parts": len(dialogue_parts)}
                )

        except Exception as e:
            logger.error(
                f"{self.provider_name} generation error; "
                f"exception_type={_safe_exception_label(e)}"
            )
            raise

    async def _stream_audio_dia(
        self,
        dialogue_parts: list[dict[str, Any]],
        request: TTSRequest
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from Dia model"""
        if not self.model or not self.processor:
            raise ValueError("Dia model not initialized")

        # Import StreamingAudioWriter for format conversion
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import AudioNormalizer, StreamingAudioWriter

        normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=self.sample_rate,
            channels=1
        )

        try:
            # Prepare input for Dia
            input_text = self._format_dia_input(dialogue_parts, request)

            # Process with Dia
            inputs = self.processor(
                text=input_text,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            # Set generation parameters
            extras = request.extra_params or {}
            if not isinstance(extras, dict):
                extras = {}
            gen_kwargs = {
                "max_new_tokens": int(extras.get("max_new_tokens", 2000)),
                "temperature": extras.get("temperature", 0.8),
                "do_sample": True,
                "top_p": extras.get("top_p", 0.95),
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }
            if extras.get("min_new_tokens") is not None:
                with contextlib.suppress(Exception):
                    gen_kwargs["min_new_tokens"] = int(extras.get("min_new_tokens"))

            # Add seed for consistent voice if specified
            if request.seed:
                gen_kwargs["seed"] = request.seed
                torch_mod = _get_torch(allow_import=True)
                if torch_mod is not None:
                    torch_mod.manual_seed(request.seed)

            # Generate audio
            torch_mod = _get_torch(allow_import=True)
            if torch_mod is None:
                raise TTSProviderNotConfiguredError(
                    "Dia generation requires torch, which is unavailable",
                    provider=self.provider_name,
                )
            with torch_mod.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            # Decode to audio waveform
            audio_array = self._decode_dia_output(outputs[0])

            # Process audio in chunks for streaming
            chunk_size = int(self.sample_rate * 0.5)  # 0.5 second chunks
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]

                if len(chunk) > 0:
                    # Normalize to int16
                    normalized_chunk = normalizer.normalize(chunk, target_dtype=np.int16)

                    # Encode to target format
                    encoded_bytes = writer.write_chunk(normalized_chunk)
                    if encoded_bytes:
                        yield encoded_bytes

            # Finalize stream
            final_bytes = writer.write_chunk(finalize=True)
            if final_bytes:
                yield final_bytes

            logger.info(f"{self.provider_name}: Successfully generated dialogue audio")

        except Exception as e:
            logger.error(
                f"{self.provider_name} streaming error; "
                f"exception_type={_safe_exception_label(e)}"
            )
            raise
        finally:
            writer.close()

    async def _generate_complete_dia(
        self,
        dialogue_parts: list[dict[str, Any]],
        request: TTSRequest
    ) -> bytes:
        """Generate complete audio from Dia"""
        all_audio = b""
        async for chunk in self._stream_audio_dia(dialogue_parts, request):
            all_audio += chunk
        return all_audio

    def _process_dialogue(
        self,
        text: str,
        speakers: Optional[dict[str, str]] = None
    ) -> list[dict[str, Any]]:
        """
        Process text into dialogue parts with speaker assignments.
        Returns list of dicts with speaker, text, and any nonverbal cues.
        """
        parts = []

        # Check if text already has speaker markers
        if self._has_speaker_markers(text):
            # Parse existing dialogue format
            dialogue_pieces = self.parse_dialogue(text)
            for speaker, content in dialogue_pieces:
                # Extract nonverbal cues
                content, nonverbal = self._extract_nonverbal(content)
                parts.append({
                    "speaker": speaker,
                    "text": content.strip(),
                    "nonverbal": nonverbal,
                    "voice": speakers.get(speaker, speaker) if speakers else speaker
                })
        else:
            # Treat as single speaker or auto-detect
            if self.auto_detect_speakers:
                # Simple heuristic: split on quotation marks
                segments = self._split_on_quotes(text)
                for i, segment in enumerate(segments):
                    if segment.strip():
                        speaker = f"speaker{(i % self.max_speakers) + 1}"
                        segment, nonverbal = self._extract_nonverbal(segment)
                        parts.append({
                            "speaker": speaker,
                            "text": segment.strip(),
                            "nonverbal": nonverbal,
                            "voice": speakers.get(speaker, speaker) if speakers else speaker
                        })
            else:
                # Single speaker
                text, nonverbal = self._extract_nonverbal(text)
                parts.append({
                    "speaker": "speaker1",
                    "text": text.strip(),
                    "nonverbal": nonverbal,
                    "voice": speakers.get("speaker1", "speaker1") if speakers else "speaker1"
                })

        return parts

    def _has_speaker_markers(self, text: str) -> bool:
        """Check if text has speaker markers like 'Name:'"""
        pattern = r'^[A-Za-z0-9]+:\s*'
        return bool(re.search(pattern, text, re.MULTILINE))

    def _split_on_quotes(self, text: str) -> list[str]:
        """Split text on quotation marks for dialogue detection"""
        # Split on quotes but keep the quotes
        parts = re.split(r'(["\'].*?["\'])', text)
        return [p for p in parts if p.strip()]

    def _extract_nonverbal(self, text: str) -> tuple[str, list[str]]:
        """Extract nonverbal cues from text"""
        nonverbal = []
        for tag in self.NONVERBAL_TAGS:
            if tag in text:
                nonverbal.append(tag)
                # Keep the tags in the text for Dia to process
        return text, nonverbal

    def _format_dia_input(
        self,
        dialogue_parts: list[dict[str, Any]],
        request: TTSRequest
    ) -> str:
        """
        Format input for Dia model.
        Dia expects specific formatting for multi-speaker dialogue.
        """
        formatted_parts = []

        for part in dialogue_parts:
            # Format: [Speaker]: text (nonverbal)
            speaker_tag = f"[{part['voice']}]"
            text = part['text']

            # Add nonverbal cues
            for cue in part.get('nonverbal', []):
                if cue not in text:
                    text += f" {cue}"

            formatted_parts.append(f"{speaker_tag}: {text}")

        # Join all parts
        full_text = "\n".join(formatted_parts)

        # Add speed modifier if needed
        if request.speed != 1.0:
            full_text = f"<speed:{request.speed}>{full_text}</speed>"

        # Add voice reference prompt if provided
        if request.voice_reference:
            full_text = f"<voice_prompt>{full_text}</voice_prompt>"

        return full_text

    def _decode_dia_output(self, tokens: Any) -> np.ndarray:
        """
        Decode Dia output tokens to audio waveform.
        This is a placeholder - actual Dia uses custom decoding.
        """
        # In production, this would use Dia's audio decoder
        # For now, return dummy audio
        num_samples = len(tokens) * 150  # Approximation
        return np.random.randn(num_samples).astype(np.float32) * 0.1

    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to Dia speaker"""
        if voice_id in self.DEFAULT_SPEAKERS:
            return voice_id

        # Map common names to speakers
        voice_mappings = {
            "alice": "speaker1",
            "bob": "speaker2",
            "charlie": "speaker3",
            "narrator": "narrator",
            "main": "speaker1",
            "secondary": "speaker2"
        }

        return voice_mappings.get(voice_id.lower(), "speaker1")

    async def close(self):
        """Clean up resources"""
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        self.model = None
        self.processor = None

        # Clear GPU cache if CUDA is available
        torch_mod = _get_torch(allow_import=False)
        if _torch_cuda_available(allow_import=False) and torch_mod is not None:
            torch_mod.cuda.empty_cache()

        await super().close()

#
# End of dia_adapter.py
#######################################################################################################################

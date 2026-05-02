# higgs_adapter.py
# Description: Higgs Audio V2 TTS adapter implementation
#
# Imports
import asyncio
import contextlib
import importlib
import os
import sys
from collections.abc import AsyncGenerator
from typing import Any, Optional

import numpy as np

#
# Third-party Imports
from loguru import logger

from ..tts_exceptions import (
    TTSGPUError,
    TTSInsufficientMemoryError,
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
# Higgs Audio V2 TTS Adapter Implementation

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


def _safe_log_error_type(error: object) -> str:
    return type(error).__name__


class HiggsAdapter(TTSAdapter):
    """Adapter for Higgs Audio V2 TTS model from Boson AI"""

    # Supported languages (50+ languages)
    SUPPORTED_LANGUAGES = {
        "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "it", "pt",
        "nl", "pl", "tr", "ar", "hi", "id", "ms", "th", "vi", "sv",
        "da", "no", "fi", "cs", "el", "he", "hu", "ro", "sk", "uk",
        "bg", "hr", "sr", "sl", "lt", "lv", "et", "sq", "mk", "ca",
        "eu", "gl", "is", "ga", "cy", "mt", "lb", "yi", "ur", "fa"
    }

    # Voice presets (can be extended with voice cloning)
    VOICE_PRESETS = {
        "narrator": VoiceInfo(
            id="narrator",
            name="Narrator",
            gender="neutral",
            description="Professional narration voice",
            use_case=["audiobook", "documentation"]
        ),
        "conversational": VoiceInfo(
            id="conversational",
            name="Conversational",
            gender="neutral",
            description="Natural conversational voice",
            use_case=["dialogue", "chat"]
        ),
        "expressive": VoiceInfo(
            id="expressive",
            name="Expressive",
            gender="neutral",
            description="Emotionally expressive voice",
            use_case=["storytelling", "drama"]
        ),
        "melodic": VoiceInfo(
            id="melodic",
            name="Melodic",
            gender="neutral",
            description="Musical and melodic voice",
            use_case=["singing", "humming"]
        )
    }

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)

        # Model configuration
        self.model_path = self.config.get(
            "higgs_model_path",
            "bosonai/higgs-audio-v2-generation-3B-base"
        )
        self.tokenizer_path = self.config.get(
            "higgs_tokenizer_path",
            "bosonai/higgs-audio-v2-tokenizer"
        )
        preferred_device = str(self.config.get("higgs_device", "cuda")).lower()
        cuda_available = _torch_cuda_available(allow_import=True)

        if preferred_device == "cuda":
            self.device = "cuda" if cuda_available else "cpu"
            if not cuda_available:
                logger.warning(
                    f"{self.provider_name}: CUDA requested but unavailable; falling back to CPU"
                )
        elif preferred_device == "cpu":
            self.device = "cpu"
        else:
            # Preserve any custom device strings (e.g., 'mps'); default to CPU when unsupported
            self.device = preferred_device if preferred_device else ("cuda" if cuda_available else "cpu")

        # Auto-download toggle: config override > env overrides > default True
        cfg_auto = self.config.get("higgs_auto_download")
        env_auto = os.getenv("HIGGS_AUTO_DOWNLOAD") or os.getenv("TTS_AUTO_DOWNLOAD")
        self.auto_download = parse_bool(cfg_auto, default=parse_bool(env_auto, default=True))

        # Audio configuration (24kHz for Higgs V2)
        self.sample_rate = 24000
        self.frame_rate = 25  # 25 frames per second tokenizer

        # Model instances
        self.model = None
        self.tokenizer = None
        self.audio_tokenizer = None

        # Performance settings
        self.use_fp16 = self.config.get("higgs_use_fp16", True) and self.device == "cuda"
        self.batch_size = self.config.get("higgs_batch_size", 1)

    async def initialize(self) -> bool:
        """Initialize the Higgs Audio V2 model"""
        if _get_torch(allow_import=True) is None:
            logger.warning(
                f"{self.provider_name}: torch unavailable; disabling provider. error={_TORCH_IMPORT_ERROR}"
            )
            self._status = ProviderStatus.NOT_CONFIGURED
            return False
        try:
            logger.info(f"{self.provider_name}: Loading Higgs Audio V2 model...")

            # Get resource manager for memory monitoring
            resource_manager = await get_resource_manager()

            # Check memory before loading model (support sync or async monitors in tests)
            _crit = resource_manager.memory_monitor.is_memory_critical()
            if asyncio.iscoroutine(_crit):
                _crit = await _crit
            if _crit:
                raise TTSInsufficientMemoryError(
                    "Insufficient memory to load Higgs model",
                    provider=self.provider_name,
                    details=resource_manager.memory_monitor.get_memory_usage()
                )

            # Check if boson_multimodal library is available
            try:
                from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
            except ImportError:
                logger.error(f"{self.provider_name}: boson_multimodal library not installed")
                logger.info("Install Higgs Audio dependencies from: https://github.com/boson-ai/higgs-audio")
                # Gracefully indicate not configured rather than raising for integration environments
                self._status = ProviderStatus.NOT_CONFIGURED
                return False

            # Initialize HiggsAudioServeEngine
            # If auto-download is disabled and a remote path is configured, abort with guidance
            if not self.auto_download:
                is_remote = (
                    isinstance(self.model_path, str)
                    and (self.model_path.startswith("http://") or self.model_path.startswith("https://")
                         or ("/" in self.model_path and not os.path.exists(self.model_path)))
                )
                if is_remote:
                    raise TTSModelLoadError(
                        "Auto-download disabled and remote model path configured",
                        provider=self.provider_name,
                        details={
                            "model_path": self.model_path,
                            "suggestion": "Set higgs_auto_download=true or provide a local model path"
                        }
                    )
            logger.info(f"{self.provider_name}: Initializing HiggsAudioServeEngine...")
            # Note: HiggsAudioServeEngine expects (model_name_or_path, audio_tokenizer_name_or_path, tokenizer_name_or_path=None, device=...)
            # Use explicit keyword names from the official API to avoid signature mismatch
            self.serve_engine = HiggsAudioServeEngine(
                model_name_or_path=self.model_path,
                audio_tokenizer_name_or_path=self.tokenizer_path,
                device=self.device
            )

            # Nudge audio tokenizer to CPU on Apple Silicon (MPS) for stability
            # Official engine binds tokenizer to the same device; embeddings/quant ops can struggle on MPS.
            if self.device == "mps":
                try:
                    from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
                        load_higgs_audio_tokenizer as _load_higgs_tokenizer,
                    )
                    self.serve_engine.audio_tokenizer = _load_higgs_tokenizer(
                        self.tokenizer_path, device="cpu"
                    )
                    # Refresh dependent fields
                    self.serve_engine.audio_tokenizer_tps = self.serve_engine.audio_tokenizer.tps
                    self.serve_engine.samples_per_token = int(
                        self.serve_engine.audio_tokenizer.sampling_rate // self.serve_engine.audio_tokenizer_tps
                    )
                    self.serve_engine.hamming_window_len = (
                        2 * self.serve_engine.audio_num_codebooks * self.serve_engine.samples_per_token
                    )
                    logger.info(
                        f"{self.provider_name}: Audio tokenizer moved to CPU for MPS stability"
                    )
                except Exception as _mps_e:
                    logger.warning(
                        f"{self.provider_name}: Failed to move tokenizer to CPU on MPS: {_mps_e}"
                    )

            # Register model with resource manager
            if self.serve_engine:
                register_result = resource_manager.register_model(
                    provider=self.provider_name.lower(),
                    model_instance=self.serve_engine,
                    cleanup_callback=self._cleanup_resources,
                    model_key=str(self.model_path),
                )
                if asyncio.iscoroutine(register_result):
                    await register_result

            logger.info(
                f"{self.provider_name}: Initialized successfully "
                f"(Device: {self.device})"
            )
            self._status = ProviderStatus.AVAILABLE
            return True

        except (TTSInsufficientMemoryError, TTSModelLoadError):
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
                f"{self.provider_name}: Initialization failed "
                f"(error_type={_safe_log_error_type(e)})"
            )
            self._status = ProviderStatus.ERROR
            raise TTSProviderInitializationError(
                f"Failed to initialize {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e), "model_path": self.model_path}
            ) from e

    async def get_capabilities(self) -> TTSCapabilities:
        """Get Higgs Audio V2 capabilities"""
        return TTSCapabilities(
            provider_name="Higgs",
            supported_languages=self.SUPPORTED_LANGUAGES,
            supported_voices=list(self.VOICE_PRESETS.values()),
            supported_formats={
                AudioFormat.WAV,
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.FLAC,
                AudioFormat.PCM
            },
            max_text_length=50000,  # Higgs can handle very long texts
            supports_streaming=True,
            supports_voice_cloning=True,  # 3-10 seconds of audio needed
            supports_emotion_control=True,  # Advanced emotion control
            supports_speech_rate=True,
            supports_pitch_control=True,
            supports_volume_control=True,
            supports_ssml=False,
            supports_phonemes=True,
            supports_multi_speaker=True,  # Natural multi-speaker dialogues
            supports_background_audio=True,  # Can generate speech with background music
            latency_ms=200 if self.device == "cuda" else 2000,
            sample_rate=24000,
            default_format=AudioFormat.WAV
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using Higgs Audio V2"""
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
                f"{self.provider_name} request validation failed "
                f"(error_type={_safe_log_error_type(e)})"
            )
            raise

        # Prepare generation parameters
        voice = request.voice or "conversational"

        # Handle voice cloning if reference provided
        voice_reference_path = None
        if request.voice_reference:
            voice_reference_path = await self._prepare_voice_reference(request.voice_reference)
            # Use "cloned" as voice identifier when using reference
            voice = "cloned" if voice_reference_path else voice

        logger.info(
            f"{self.provider_name}: Generating speech with voice={voice}, "
            f"language={request.language}, format={request.format.value}"
        )

        try:
            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio_higgs(request, voice_reference_path),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )
            else:
                # Generate complete audio
                audio_data = await self._generate_complete_higgs(request, voice_reference_path)
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name
                )

        except Exception as e:
            logger.error(
                f"{self.provider_name} generation error "
                f"(error_type={_safe_log_error_type(e)})"
            )
            raise

    async def _stream_audio_higgs(
        self,
        request: TTSRequest,
        voice_reference_path: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from Higgs model"""
        if not hasattr(self, 'serve_engine') or not self.serve_engine:
            raise ValueError("Higgs serve engine not initialized")

        # Import required modules (torchaudio optional; avoid hard dependency)
        from boson_multimodal.serve.serve_engine import HiggsAudioResponse

        from tldw_Server_API.app.core.TTS.streaming_audio_writer import AudioNormalizer, StreamingAudioWriter

        normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=self.sample_rate,
            channels=1
        )

        try:
            # Prepare ChatML format for Higgs
            chat_ml_sample = self._prepare_higgs_chat_ml(request, voice_reference_path)

            # Generate with HiggsAudioServeEngine
            logger.info(f"{self.provider_name}: Starting generation...")
            extras = request.extra_params or {}
            if not isinstance(extras, dict):
                extras = {}
            gen_kwargs = {
                "chat_ml_sample": chat_ml_sample,
                "max_new_tokens": int(extras.get("max_new_tokens", 1024)),
                "temperature": extras.get("temperature", 1.0),
                "top_p": extras.get("top_p", 0.95),
                "top_k": extras.get("top_k", 50),
                "stop_strings": ["<|end_of_text|>", "<|eot_id|>"],
            }
            if extras.get("min_new_tokens") is not None:
                with contextlib.suppress(Exception):
                    gen_kwargs["min_new_tokens"] = int(extras.get("min_new_tokens"))
            output: HiggsAudioResponse = self.serve_engine.generate(**gen_kwargs)

            # Convert numpy audio to tensor and back to numpy for processing
            audio_array = output.audio

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

            logger.info(f"{self.provider_name}: Successfully generated audio")
            logger.debug(f"Generated text: {output.generated_text}")

        except Exception as e:
            logger.error(
                f"{self.provider_name} streaming error "
                f"(error_type={_safe_log_error_type(e)})"
            )
            raise
        finally:
            writer.close()
            # Clean up voice reference file if used
            if voice_reference_path:
                try:
                    from pathlib import Path
                    Path(voice_reference_path).unlink(missing_ok=True)
                    logger.debug(f"Cleaned up voice reference: {voice_reference_path}")
                except Exception as e:
                    logger.warning(
                        "Failed to clean up voice reference "
                        f"(error_type={_safe_log_error_type(e)})"
                    )

    async def _generate_complete_higgs(
        self,
        request: TTSRequest,
        voice_reference_path: Optional[str] = None
    ) -> bytes:
        """Generate complete audio from Higgs"""
        # Collect all streamed chunks
        all_audio = b""
        async for chunk in self._stream_audio_higgs(request, voice_reference_path):
            all_audio += chunk
        return all_audio

    def _prepare_higgs_chat_ml(
        self,
        request: TTSRequest,
        voice_reference_path: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Prepare ChatML format input for Higgs.
        Higgs uses a specific format for generation.
        """
        # We construct HuggingFace-like ChatML using the official boson_multimodal dataclasses
        # while keeping a dict payload to satisfy existing unit tests.
        try:
            from boson_multimodal.data_types import AudioContent, Message
            use_dataclasses = True
        except Exception:
            # Fallback to plain dicts if library is not available (e.g., unit tests without deps)
            Message = None       # type: ignore
            AudioContent = None  # type: ignore
            use_dataclasses = False

        messages: list[Any] = []

        # Optional system prompt
        system_prompt = request.extra_params.get("system_prompt")
        if system_prompt:
            if use_dataclasses:
                messages.append(Message(role="system", content=system_prompt))
            else:
                messages.append({"role": "system", "content": system_prompt})

        # If we have a voice reference, follow the official pattern:
        #   user (reference text) -> assistant (audio content) -> user (generation text)
        if voice_reference_path:
            ref_text = request.extra_params.get("reference_text", "Here is a reference voice sample.")
            if use_dataclasses:
                messages.append(Message(role="user", content=ref_text))
                messages.append(Message(role="assistant", content=AudioContent(audio_url=voice_reference_path)))
            else:
                messages.append({"role": "user", "content": ref_text})
                messages.append({
                    "role": "assistant",
                    "content": {"type": "audio", "audio_url": voice_reference_path}
                })

        # Build final user content
        user_content = request.text

        # Language instruction
        if request.language and request.language != "en":
            user_content = f"Please generate speech in {request.language}: {user_content}"

        # Emotion/style instructions
        if request.emotion:
            intensity_desc = (
                "strongly" if request.emotion_intensity > 1.5
                else "moderately" if request.emotion_intensity > 0.5
                else "slightly"
            )
            user_content = f"Say this {intensity_desc} {request.emotion}: {user_content}"

        if request.style:
            user_content = f"In a {request.style} style: {user_content}"

        # Multi-speaker hint (official model also supports [SPEAKERi] tags; we keep a generic hint here)
        if request.speakers:
            user_content = f"Generate a dialogue with multiple speakers: {user_content}"

        if use_dataclasses:
            messages.append(Message(role="user", content=user_content))
        else:
            messages.append({"role": "user", "content": user_content})

        # Return a dict payload compatible with both unit tests and the serve engine
        payload: dict[str, Any] = {
            "messages": messages,
            # The following fields are not used by the official serve engine, but are
            # kept to satisfy existing unit tests and potential higher-level consumers.
            "voice": request.voice or "narrator",
            "speed": request.speed,
            "seed": request.seed,
        }

        if voice_reference_path:
            # Preserve legacy key for unit tests while embedding reference audio above
            payload["reference_audio_path"] = voice_reference_path
            payload["voice"] = "cloned"
            logger.info(f"Added voice reference to Higgs ChatML: {voice_reference_path}")

        return payload

    async def _prepare_voice_reference(self, voice_reference: bytes) -> Optional[str]:
        """
        Prepare voice reference audio for Higgs.
        Higgs needs 3-10 seconds of audio at 24kHz.

        Args:
            voice_reference: Voice reference audio bytes

        Returns:
            Path to temporary voice reference file or None if processing fails
        """
        try:
            import tempfile

            from tldw_Server_API.app.core.TTS.audio_utils import process_voice_reference_async

            # Process voice reference for Higgs requirements
            processed_audio, error = await process_voice_reference_async(
                voice_reference,
                provider='higgs',
                validate=True,
                convert=True
            )

            if error:
                logger.error(
                    "Voice reference processing failed "
                    f"(error_type={_safe_log_error_type(error)})"
                )
                return None

            # Save to temporary file
            with tempfile.NamedTemporaryFile(
                suffix='.wav',
                delete=False,
                prefix='higgs_voice_'
            ) as tmp_file:
                tmp_file.write(processed_audio)
                tmp_path = tmp_file.name

            logger.info(f"Voice reference prepared for Higgs: {tmp_path}")
            return tmp_path

        except Exception as e:
            logger.error(
                "Failed to prepare voice reference "
                f"(error_type={_safe_log_error_type(e)})"
            )
            return None

    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to Higgs voice preset"""
        if voice_id in self.VOICE_PRESETS:
            return voice_id

        # Map common voice types
        voice_mappings = {
            "default": "conversational",
            "assistant": "conversational",
            "narrator": "narrator",
            "expressive": "expressive",
            "emotional": "expressive",
            "singing": "melodic",
            "musical": "melodic"
        }

        return voice_mappings.get(voice_id.lower(), "conversational")

    async def close(self):
        """Clean up resources"""
        if hasattr(self, 'serve_engine') and self.serve_engine:
            del self.serve_engine
            self.serve_engine = None

        # Clear GPU cache if using CUDA
        torch_mod = _get_torch(allow_import=False)
        if self.device == "cuda" and _torch_cuda_available(allow_import=False) and torch_mod is not None:
            torch_mod.cuda.empty_cache()

        await super().close()

#
# End of higgs_adapter.py
#######################################################################################################################

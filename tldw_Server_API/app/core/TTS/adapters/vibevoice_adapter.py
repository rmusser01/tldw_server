# vibevoice_adapter.py
# Description: VibeVoice TTS adapter implementation - emotion and tone-aware synthesis
#
# Imports
import asyncio
import gc
import importlib
import importlib.util
import os
import re
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Optional

import numpy as np

#
# Third-party Imports
from loguru import logger

from ..tts_exceptions import (
    TTSGenerationError,
    TTSGPUError,
    TTSInsufficientMemoryError,
    TTSModelLoadError,
    TTSModelNotFoundError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
)
from ..tts_resource_manager import get_resource_manager
from ..tts_validation import validate_tts_request
from ..utils import parse_bool

#
# Local Imports
from .base import AudioFormat, ProviderStatus, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse, VoiceInfo

_VIBEVOICE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    EOFError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)

#
#######################################################################################################################
#
# VibeVoice TTS Adapter Implementation

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
        return bool(
            hasattr(torch_mod.backends, "mps")
            and torch_mod.backends.mps.is_available()
            and torch_mod.backends.mps.is_built()
        )
    except Exception:
        return False

class VibeVoiceAdapter(TTSAdapter):
    """
    Adapter for Microsoft VibeVoice - Expressive long-form multi-speaker conversational audio.
    Generates up to 90 minutes of speech with 4 distinct speakers.
    Features spontaneous background music and emergent singing capabilities.
    """

    # Model variants available
    MODEL_VARIANTS = {
        "1.5B": {
            "path": "microsoft/VibeVoice-1.5B",
            "context": 64000,  # 64K context (~90 min generation)
            "frame_rate": 7.5   # Hz
        },
        "7B": {
            # Official 7B repository
            "path": "vibevoice/VibeVoice-7B",
            "context": 32000,  # 32K context (~45 min generation)
            "frame_rate": 7.5   # Hz
        },
        "7B-Q8": {
            # Community 8-bit quantized 7B variant (reduced VRAM usage)
            "path": "FabioSarracino/VibeVoice-Large-Q8",
            "context": 32000,
            "frame_rate": 7.5
        }
    }

    # Voice presets loaded from files
    VOICE_PRESETS = {}

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)

        # Model variant selection (1.5B or 7B)
        self.variant = self.config.get("vibevoice_variant", "1.5B")
        if self.variant not in self.MODEL_VARIANTS:
            logger.warning(f"Unknown VibeVoice variant {self.variant}, defaulting to 1.5B")
            self.variant = "1.5B"

        # Model configuration
        variant_config = self.MODEL_VARIANTS[self.variant]
        self.model_path = self.config.get("vibevoice_model_path", variant_config["path"])
        self.model_revision = self.config.get("vibevoice_model_revision") or os.getenv("VIBEVOICE_MODEL_REVISION")
        self.context_length = variant_config["context"]
        self.frame_rate = variant_config["frame_rate"]

        # Device configuration with MPS support
        if self.config.get("vibevoice_device"):
            self.device = self.config.get("vibevoice_device")
        elif _torch_cuda_available(allow_import=True):
            self.device = "cuda"
        elif _torch_mps_available(allow_import=True):
            self.device = "mps"
            logger.info("Using MPS (Apple Silicon) for VibeVoice")
        else:
            self.device = "cpu"

        # Audio configuration (VibeVoice typically uses 24 kHz)
        self.sample_rate = self.config.get("vibevoice_sample_rate", 24000)

        # Model instances
        self.model = None
        self.processor = None

        # Speaker settings (VibeVoice supports up to 4 speakers)
        self.max_speakers = 4
        self.default_speaker = self.config.get("vibevoice_default_speaker", 1)

        # Context awareness
        self.enable_context = self.config.get("vibevoice_context", True)
        self.context_window = self.config.get("vibevoice_context_window", 512)

        # VibeVoice specific features
        self.enable_background_music = self.config.get("vibevoice_background_music", False)
        self.enable_singing = self.config.get("vibevoice_enable_singing", False)

        # Performance settings
        self.use_fp16 = self.config.get("vibevoice_use_fp16", True) and self.device in ["cuda", "mps"]
        self.batch_size = self.config.get("vibevoice_batch_size", 1)

        # Memory optimization settings (4-bit quantization is effectively CUDA-only)
        requested_quant = parse_bool(self.config.get("vibevoice_use_quantization", False), default=False)
        self.use_quantization = requested_quant and self.device == "cuda"
        # If using the pre-quantized 7B-Q8 variant (or Q8 repo), avoid stacking 4-bit quantization
        try:
            if self.variant.upper() == "7B-Q8" or "vibevoice-large-q8" in str(self.model_path).lower():
                if self.use_quantization:
                    logger.info("VibeVoice: Disabling additional 4-bit quantization for 7B-Q8 model")
                self.use_quantization = False
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS:
            pass
        self.auto_cleanup = self.config.get("vibevoice_auto_cleanup", True)
        # Auto-download behavior: config override > env overrides > default False
        cfg_auto = self.config.get("vibevoice_auto_download")
        env_auto = os.getenv("VIBEVOICE_AUTO_DOWNLOAD") or os.getenv("TTS_AUTO_DOWNLOAD")
        # Default to False to prevent background model downloads unless explicitly allowed
        self.auto_download = parse_bool(cfg_auto, default=parse_bool(env_auto, default=False))

        # Advanced attention settings
        self.enable_sage = self.config.get("vibevoice_enable_sage", False)
        self.attention_fallback_chain = [
            "flash_attention_2",
            "sage" if self.enable_sage else None,
            "sdpa",
            "eager"
        ]
        self.attention_fallback_chain = [a for a in self.attention_fallback_chain if a]  # Remove None

        # Default generation parameters
        self.default_cfg_scale = self.config.get("vibevoice_cfg_scale", 1.3)
        self.default_diffusion_steps = self.config.get("vibevoice_diffusion_steps", 20)
        self.default_temperature = self.config.get("vibevoice_temperature", 1.0)
        self.default_top_p = self.config.get("vibevoice_top_p", 0.95)
        self.default_top_k = self.config.get("vibevoice_top_k", 50)  # New parameter
        self.default_attention_type = self.config.get("vibevoice_attention_type", "auto")

        # Streaming optimization
        self.stream_chunk_size = self.config.get("vibevoice_stream_chunk_size", 0.25)  # seconds
        self.stream_buffer_size = self.config.get("vibevoice_stream_buffer_size", 4096)

        # Setup paths
        self.model_dir = Path(self.config.get("vibevoice_model_dir", "./models/vibevoice"))
        self.cache_dir = Path(self.config.get("vibevoice_cache_dir", "./cache/vibevoice"))

        # Voice samples folder for 1-shot cloning (like VibeVoice demo)
        self.voices_dir = Path(self.config.get("vibevoice_voices_dir", "./voices"))
        self.available_voices = {}  # Maps voice names to file paths
        # Optional default mapping for speakers to voices (ids or file paths)
        self.default_speakers_to_voices = self.config.get("vibevoice_speakers_to_voices")

        # Cancellation support
        self._generation_cancelled = False
        self._current_generation_task = None

        # Memory tracking
        self._memory_stats = {
            "peak_vram_gb": 0,
            "current_vram_gb": 0,
            "quantization_savings_gb": 0
        }

    def _load_voice_files(self):
        """Load voice samples from voices folder for voice cloning"""
        self.available_voices.clear()
        self.VOICE_PRESETS.clear()

        # Supported audio formats
        audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]

        if self.voices_dir.exists():
            logger.info(f"Loading voices from {self.voices_dir}")

            # Scan for voice files
            for ext in audio_extensions:
                for voice_file in self.voices_dir.glob(f"*{ext}"):
                    voice_name = voice_file.stem  # Filename without extension
                    self.available_voices[voice_name] = str(voice_file)

                    # Try to extract gender from filename (e.g., "en-Alice_woman")
                    gender = "neutral"
                    if "woman" in voice_name.lower() or "female" in voice_name.lower():
                        gender = "female"
                    elif "man" in voice_name.lower() or "male" in voice_name.lower():
                        gender = "male"

                    # Add to voice presets
                    self.VOICE_PRESETS[voice_name] = VoiceInfo(
                        id=voice_name,
                        name=voice_name.replace("_", " ").replace("-", " ").title(),
                        gender=gender,
                        description=f"Voice loaded from {voice_file.name}",
                        styles=["file-based"]
                    )
                    logger.info(f"Loaded voice: {voice_name} from {voice_file}")

            if not self.available_voices:
                logger.warning(f"No voice files found in {self.voices_dir}")
                # Add default speaker options when no voice files are found
                for i in range(1, 5):
                    speaker_id = f"speaker_{i}"
                    self.VOICE_PRESETS[speaker_id] = VoiceInfo(
                        id=speaker_id,
                        name=f"Speaker {i}",
                        gender="neutral",
                        description=f"Default speaker {i}",
                        styles=["default"]
                    )
        else:
            logger.info(f"Voices directory {self.voices_dir} not found, using speaker numbers only")
            # Add default speaker options
            for i in range(1, 5):
                speaker_id = f"speaker_{i}"
                self.VOICE_PRESETS[speaker_id] = VoiceInfo(
                    id=speaker_id,
                    name=f"Speaker {i}",
                    gender="neutral",
                    description=f"Default speaker {i}",
                    styles=["default"]
                )

    async def _load_user_voices(self, user_id: int):
        """Load user-specific uploaded voices"""
        try:
            from ...voice_manager import get_voice_manager

            voice_manager = get_voice_manager()
            user_voices = await voice_manager.list_user_voices(user_id)

            for voice_info in user_voices:
                if voice_info.provider == "vibevoice":
                    # Get full path to voice file
                    voices_path = voice_manager.get_user_voices_path(user_id)
                    voice_file_path = voices_path / voice_info.file_path

                    if voice_file_path.exists():
                        # Register with custom: prefix
                        custom_id = f"custom:{voice_info.voice_id}"
                        self.available_voices[custom_id] = str(voice_file_path)

                        # Add to voice presets
                        self.VOICE_PRESETS[custom_id] = VoiceInfo(
                            id=custom_id,
                            name=voice_info.name,
                            gender="neutral",
                            description=voice_info.description or f"Custom voice: {voice_info.name}",
                            styles=["custom", "uploaded"]
                        )

                        logger.info(f"Loaded user voice: {voice_info.name} ({custom_id})")

            logger.info(f"Loaded {len(user_voices)} user voices for VibeVoice")

        except ImportError:
            logger.debug("Voice manager not available, skipping user voice loading")
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error loading user voices: {e}")

    async def initialize(self, user_id: Optional[int] = None) -> bool:
        """Initialize the VibeVoice TTS model"""
        if _get_torch(allow_import=True) is None:
            logger.warning(
                f"{self.provider_name}: torch unavailable; disabling provider. error={_TORCH_IMPORT_ERROR}"
            )
            self._status = ProviderStatus.NOT_CONFIGURED
            return False
        try:
            logger.info(f"{self.provider_name}: Initializing VibeVoice TTS (variant: {self.variant}, model: {self.model_path})...")

            # Load voice files from voices folder
            self._load_voice_files()

            # Load user-specific voices if user_id provided
            if user_id:
                await self._load_user_voices(user_id)

            # Get resource manager for memory monitoring
            resource_manager = await get_resource_manager()

            # Check memory before loading model
            if resource_manager.memory_monitor.is_memory_critical():
                raise TTSInsufficientMemoryError(
                    "Insufficient memory to load VibeVoice model",
                    provider=self.provider_name,
                    details=resource_manager.memory_monitor.get_memory_usage()
                )

            # Check for model files
            if not self._check_model_files():
                if not self.auto_download:
                    raise TTSModelLoadError(
                        "VibeVoice models not found locally and auto-download is disabled",
                        provider=self.provider_name,
                        details={
                            "model_path": self.model_path,
                            "suggestion": "Enable vibevoice_auto_download or preinstall models into model_dir"
                        }
                    )
                logger.info(f"{self.provider_name}: Downloading VibeVoice models...")
                await self._download_models()

            # Load the model components
            logger.info(f"{self.provider_name}: Loading VibeVoice components...")

            # Import VibeVoice
            try:
                from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

                # Load processor
                logger.info(f"Loading VibeVoice processor from {self.model_path}")
                self.processor = VibeVoiceProcessor.from_pretrained(self.model_path)
                torch_mod = _get_torch(allow_import=True)
                if torch_mod is None:
                    raise TTSProviderNotConfiguredError(
                        "VibeVoice requires torch, which is unavailable",
                        provider=self.provider_name,
                    )

                # Determine dtype and attention implementation
                if self.device == "cuda":
                    load_dtype = torch_mod.bfloat16 if self.use_fp16 else torch_mod.float32
                elif self.device == "mps":
                    load_dtype = torch_mod.float16 if self.use_fp16 else torch_mod.float32
                else:
                    load_dtype = torch_mod.float32

                # Determine attention implementation with fallback chain
                attn_impl = self._get_best_attention_implementation()

                # Load main model with optional quantization
                logger.info(f"Loading VibeVoice model with dtype={load_dtype}, device={self.device}, quantization={self.use_quantization}")

                if self.use_quantization:
                    # Load with 4-bit quantization for memory savings
                    try:
                        from transformers import BitsAndBytesConfig
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=load_dtype,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.model_path,
                            quantization_config=quantization_config,
                            device_map="auto" if self.device != "cpu" else None,
                            attn_implementation=attn_impl
                        )
                        # Calculate memory savings
                        self._calculate_quantization_savings()
                        logger.info(f"Loaded model with 4-bit quantization, estimated savings: {self._memory_stats['quantization_savings_gb']:.2f} GB")
                    except ImportError:
                        logger.warning("BitsAndBytes not available, falling back to standard loading. Install with: pip install bitsandbytes")
                        self.use_quantization = False
                        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            self.model_path,
                            torch_dtype=load_dtype,
                            device_map="auto" if self.device != "cpu" else None,
                            attn_implementation=attn_impl
                        )
                else:
                    self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        self.model_path,
                        torch_dtype=load_dtype,
                        device_map="auto" if self.device != "cpu" else None,
                        attn_implementation=attn_impl
                    )

                # Set model to eval and configure inference
                self.model.eval()
                self.model.set_ddpm_inference_steps(num_steps=self.default_diffusion_steps)

                # Track memory usage
                self._update_memory_stats()

            except ImportError as e:
                error_msg = (
                    f"{self.provider_name}: Required libraries not installed. "
                    f"To use VibeVoice, follow these steps:\n"
                    f"1. Ensure VibeVoice is installed (official repo):\n"
                    f"   git clone https://github.com/microsoft/VibeVoice.git libs/VibeVoice\n"
                    f"   cd libs/VibeVoice && pip install -e .\n"
                    f"2. The {self.variant} model will auto-download on first use from:\n"
                    f"   {self.model_path}"
                )
                logger.error(f"{self.provider_name}: Required libraries not installed: {e}")
                logger.error(error_msg)
                self._status = ProviderStatus.NOT_CONFIGURED
                raise TTSModelLoadError(
                    "Failed to import required libraries",
                    provider=self.provider_name,
                    details={"error": str(e), "suggestion": error_msg}
                ) from e

            # Set to evaluation mode
            if self.model:
                self.model.eval()

                # Register model with resource manager
                register_result = resource_manager.register_model(
                    provider=self.provider_name.lower(),
                    model_instance=self.model,
                    cleanup_callback=self._cleanup_resources,
                    model_key=f"{self.variant}:{self.model_path}",
                )
                if asyncio.iscoroutine(register_result):
                    await register_result

            # Warm up the model
            await self._warmup_model()

            logger.info(
                f"{self.provider_name}: Initialized successfully "
                f"(Device: {self.device}, FP16: {self.use_fp16}, Context: {self.enable_context})"
            )
            self._status = ProviderStatus.AVAILABLE
            return True

        except (TTSInsufficientMemoryError, TTSModelLoadError) as e:
            logger.error(f"{self.provider_name}: Initialization failed due to model/memory error: {e}")
            self._status = ProviderStatus.ERROR
            return False
        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                raise TTSGPUError(
                    f"GPU error initializing {self.provider_name}",
                    provider=self.provider_name,
                    details={"error": str(e), "device": self.device}
                ) from e
            raise
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name}: Initialization failed: {e}")
            self._status = ProviderStatus.ERROR
            raise TTSProviderInitializationError(
                f"Failed to initialize {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e), "model_path": self.model_path}
            ) from e

    def _check_model_files(self) -> bool:
        """Check if model files exist locally."""
        required_files = [
            "config.json",
            "model.safetensors",
            "tokenizer_config.json",
            "vibe_encoder.pt"
        ]

        # If model_path is a local directory, validate there first
        try_path = None
        try:
            mp = Path(self.model_path)
            if mp.exists() and mp.is_dir():
                try_path = mp
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS:
            try_path = None

        search_dir = try_path or self.model_dir
        if not search_dir.exists():
            return False

        for file in required_files:
            if not (search_dir / file).exists():
                return False

        # Align model_path to local directory if we validated against model_dir
        if not try_path:
            self.model_path = str(self.model_dir)
        return True

    async def _download_models(self):
        """Download VibeVoice models if not present with progress tracking"""
        # If model_path is already a local directory, don't attempt network download
        try:
            mp = Path(self.model_path)
            if mp.exists() and mp.is_dir():
                logger.info(f"{self.provider_name}: Using existing local model at {mp}")
                return True
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS:
            pass

        if not self.auto_download:
            logger.info(f"{self.provider_name}: Auto-download disabled, skipping model download")
            return False

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download
            from tqdm import tqdm

            logger.info(f"{self.provider_name}: Downloading {self.variant} model from {self.model_path}...")

            class DownloadProgress:
                def __init__(self):
                    self.pbar = None
                def __call__(self, num_bytes):
                    if self.pbar is None:
                        self.pbar = tqdm(unit='B', unit_scale=True, desc="Downloading VibeVoice")
                    self.pbar.update(num_bytes)
            progress = DownloadProgress()

            local_dir = snapshot_download(
                repo_id=self.model_path,
                revision=self.model_revision,
                local_dir=str(self.model_dir),
                cache_dir=str(self.cache_dir),
                resume_download=True,
                # No symlinks parameter needed in recent huggingface_hub
            )  # nosec B615
            if progress.pbar:
                progress.pbar.close()
            logger.info(f"{self.provider_name}: Model download complete at {local_dir}")
            self.model_path = local_dir
            return True
        except ImportError:
            logger.error(f"{self.provider_name}: huggingface_hub not installed. Run: pip install huggingface-hub")
            return False
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name}: Model download failed: {e}")
            return False

    async def _warmup_model(self):
        """Warm up the model with a test generation"""
        if not self.model:
            return

        try:
            logger.debug(f"{self.provider_name}: Warming up model...")
            # Optional, guarded micro-forward to catch lazy init issues
            if self.config.get("vibevoice_enable_warmup_forward", False) and self.processor:
                tiny_inputs = self.processor(
                    text=["Speaker 1: warmup.\n Speech output:\n"],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                torch_mod = _get_torch(allow_import=True)
                if torch_mod is None:
                    logger.debug(f"{self.provider_name}: skipping warmup forward because torch is unavailable")
                    return
                for k, v in tiny_inputs.items():
                    if torch_mod.is_tensor(v):
                        tiny_inputs[k] = v.to(self.device)
                with torch_mod.no_grad():
                    _ = self.model.generate(
                        **tiny_inputs,
                        max_new_tokens=32,
                        cfg_scale=1.0,
                        tokenizer=self.processor.tokenizer,
                        generation_config={"do_sample": False},
                        show_progress_bar=False,
                        verbose=False,
                    )
            logger.debug(f"{self.provider_name}: Model warmup complete")
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"{self.provider_name}: Warmup failed: {e}")

    def _build_voice_samples(
        self,
        formatted_text: str,
        voice_reference_path: Optional[str],
        primary_voice: str,
        speakers_to_voices: Optional[dict[str, str]] = None,
    ) -> list[str]:
        """Build ordered voice sample list aligned to speakers in formatted_text.

        speakers_to_voices: mapping of speaker id (str or int) to either a voice id in available_voices
        or a file path. Speaker ids may be 0- or 1-based; mapping will be normalized.
        """
        # Detect speakers in text
        speakers = re.findall(r"^\s*Speaker\s+(\d+)\s*:\s*", formatted_text, flags=re.IGNORECASE | re.MULTILINE)
        unique_speakers = sorted({int(s) for s in speakers}) if speakers else [1]
        min_spk = min(unique_speakers) if unique_speakers else 1
        num_speakers = len(unique_speakers) if unique_speakers else 1

        # Collect available files from voices dir
        available_voice_files: list[str] = []
        if self.voices_dir.exists():
            for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"):
                for p in self.voices_dir.glob(f"*{ext}"):
                    available_voice_files.append(str(p))

        # Normalize provided mapping
        mapping: dict[int, str] = {}
        if speakers_to_voices:
            for k, v in speakers_to_voices.items():
                try:
                    spk = int(k)
                except _VIBEVOICE_NONCRITICAL_EXCEPTIONS:
                    continue
                # Normalize to 0-based index used by processor enumeration
                idx = spk - min_spk if spk >= min_spk else spk
                mapping[idx] = v

        # Build voice samples list
        voice_samples: list[str] = [None] * num_speakers  # type: ignore
        # Fill from mapping first
        for idx, val in mapping.items():
            if 0 <= idx < num_speakers:
                # Resolve via available_voices id if present, else treat as path
                if val in self.available_voices:
                    voice_samples[idx] = self.available_voices[val]
                else:
                    voice_samples[idx] = val
        # Ensure first speaker has a voice
        if voice_samples and voice_samples[0] is None:
            if voice_reference_path:
                voice_samples[0] = voice_reference_path
            elif primary_voice in self.available_voices:
                voice_samples[0] = self.available_voices[primary_voice]
            elif available_voice_files:
                voice_samples[0] = available_voice_files[0]
        # Fill remaining speakers from available files if unset
        for idx in range(1, num_speakers):
            if voice_samples[idx] is None:
                pick = available_voice_files[idx] if len(available_voice_files) > idx else None
                if pick is None:
                    # Not enough samples; disable cloning
                    return []
                voice_samples[idx] = pick

        # Replace any remaining None with first available, else return empty to disable
        if any(v is None for v in voice_samples):
            return []
        return voice_samples  # type: ignore

    async def get_capabilities(self) -> TTSCapabilities:
        """Get VibeVoice TTS capabilities"""
        # Variant-specific max generation time

        return TTSCapabilities(
            provider_name=f"VibeVoice-{self.variant}",
            supported_languages={"en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"},
            supported_voices=list(self.VOICE_PRESETS.values()),
            supported_formats={
                AudioFormat.WAV,
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.FLAC,
                AudioFormat.PCM,
                AudioFormat.OGG
            },
            max_text_length=self.context_length,  # Use variant-specific context length
            supports_streaming=True,
            supports_voice_cloning=True,  # Supports cloning via voice reference folder
            supports_emotion_control=False,  # Does not have direct emotion control
            supports_speech_rate=True,
            supports_pitch_control=False,
            supports_volume_control=True,
            supports_ssml=False,  # Uses its own markup
            supports_phonemes=False,
            supports_multi_speaker=True,  # Up to 4 speakers
            supports_background_audio=True,  # Spontaneous music generation
            latency_ms=150 if self.device == "cuda" else 800,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech using VibeVoice TTS"""
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                f"{self.provider_name} not initialized",
                provider=self.provider_name
            )

        # Validate request using new validation system
        try:
            validate_tts_request(request, provider=self.provider_key)
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name} request validation failed: {e}")
            raise

        # Check if a different model variant was requested
        requested_model = getattr(request, "model", None) or request.extra_params.get("model")
        if requested_model and requested_model in self.MODEL_VARIANTS and requested_model != self.variant:
            logger.info(f"Switching VibeVoice model from {self.variant} to {requested_model}")
            self.variant = requested_model
            # Reload model with new variant
            await self._reload_model_for_variant(requested_model)

        # Extract generation parameters
        cfg_scale = request.cfg_scale or request.extra_params.get("cfg_scale", self.default_cfg_scale)
        diffusion_steps = request.diffusion_steps or request.extra_params.get("diffusion_steps", self.default_diffusion_steps)
        temperature = request.temperature or request.extra_params.get("temperature", self.default_temperature)
        top_p = request.top_p or request.extra_params.get("top_p", self.default_top_p)
        top_k = request.extra_params.get("top_k", self.default_top_k)  # New parameter
        seed = request.seed or request.extra_params.get("seed")
        attention_type = request.attention_type or request.extra_params.get("attention_type", self.default_attention_type)

        # Validate parameters with extended ranges from ComfyUI
        cfg_scale = max(1.0, min(2.0, cfg_scale))
        diffusion_steps = max(5, min(100, diffusion_steps))
        temperature = max(0.1, min(2.0, temperature))
        top_p = max(0.1, min(1.0, top_p))
        top_k = max(0, min(100, top_k)) if top_k else 0  # 0 means disabled

        # Parse text for multi-speaker support
        text, speaker_mapping = self._parse_multi_speaker_text(request.text)

        # Process voice and speaker settings
        voice = request.voice or "speaker_1"
        speaker_id = request.extra_params.get("speaker_id", self.default_speaker)
        # Ensure speaker_id is within valid range (1-4)
        speaker_id = max(1, min(4, speaker_id))

        # If multi-speaker text detected, use speaker mapping
        voice_references = {}
        if speaker_mapping:
            for speaker_num, _speaker_text in speaker_mapping.items():
                # Map speaker to voice or generate synthetic
                speaker_voice = f"speaker_{speaker_num}"
                if speaker_voice in self.available_voices:
                    voice_references[speaker_num] = self.available_voices[speaker_voice]
                else:
                    # Generate synthetic voice for this speaker
                    voice_references[speaker_num] = await self._generate_synthetic_voice(speaker_num)

        # Handle voice cloning - check custom voices
        voice_reference_path = None

        # Check if voice starts with "custom:" prefix (uploaded voice)
        if voice.startswith("custom:"):
            if voice in self.available_voices:
                voice_reference_path = self.available_voices[voice]
                logger.info(f"Using uploaded voice '{voice}' from {voice_reference_path}")
            else:
                logger.warning(f"Custom voice '{voice}' not found, using default")
                voice = "speaker_1"
        # Check if voice is loaded from voices folder
        elif voice in self.available_voices:
            voice_reference_path = self.available_voices[voice]
            logger.info(f"Using voice '{voice}' from {voice_reference_path}")
        # Otherwise check if voice reference bytes provided
        elif request.voice_reference:
            voice_reference_path = await self._prepare_voice_reference(request.voice_reference)
            # Use "cloned" as voice identifier when using reference
            voice = "cloned" if voice_reference_path else voice
        # If no voice reference, generate synthetic voice
        elif not voice_reference_path and voice.startswith("speaker_"):
            voice_reference_path = await self._generate_synthetic_voice(speaker_id)

        # Process multi-speaker if needed
        request.speakers if hasattr(request, 'speakers') else None

        logger.info(
            f"{self.provider_name}: Generating speech with voice={voice}, "
            f"speaker_id={speaker_id}, format={request.format.value}, "
            f"cfg={cfg_scale}, steps={diffusion_steps}, temp={temperature}, top_p={top_p}"
        )

        try:
            # Reset cancellation flag for new generation
            self._generation_cancelled = False

            # Prepare generation config with all parameters
            gen_config = {
                "cfg_scale": cfg_scale,
                "diffusion_steps": diffusion_steps,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "seed": seed,
                "attention_type": attention_type,
                "speaker_mapping": speaker_mapping,
                "voice_references": voice_references,
                "speakers_to_voices": request.extra_params.get("speakers_to_voices") if hasattr(request, 'extra_params') else None,
            }

            if request.stream:
                # Return streaming response
                return TTSResponse(
                    audio_stream=self._stream_audio_vibevoice(request, voice, speaker_id, voice_reference_path, gen_config),
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name,
                    metadata={
                        "speaker_id": speaker_id,
                        "context_enabled": self.enable_context,
                        "background_music": self.enable_background_music,
                        "singing_enabled": self.enable_singing,
                        "cfg_scale": cfg_scale,
                        "diffusion_steps": diffusion_steps,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "seed": seed,
                        "memory_usage": self.get_memory_usage() if self.device != "cpu" else None
                    }
                )
            else:
                # Generate complete audio
                audio_data = await self._generate_complete_vibevoice(request, voice, speaker_id, voice_reference_path, gen_config)
                return TTSResponse(
                    audio_data=audio_data,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    voice_used=voice,
                    provider=self.provider_name,
                    metadata={
                        "speaker_id": speaker_id,
                        "context_enabled": self.enable_context,
                        "background_music": self.enable_background_music,
                        "singing_enabled": self.enable_singing,
                        "cfg_scale": cfg_scale,
                        "diffusion_steps": diffusion_steps,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "seed": seed,
                        "memory_usage": self.get_memory_usage() if self.device != "cpu" else None
                    }
                )

        except (TTSProviderNotConfiguredError, TTSModelLoadError):
            raise
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name} generation error: {e}")
            raise TTSGenerationError(
                f"Failed to generate speech with {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e), "error_type": type(e).__name__}
            ) from e
        finally:
            # Auto cleanup if enabled
            if self.auto_cleanup:
                await self.cleanup_after_generation()


    async def _stream_audio_vibevoice(
        self,
        request: TTSRequest,
        voice: str,
        speaker_id: int,
        voice_reference_path: Optional[str] = None,
        gen_config: Optional[dict[str, Any]] = None
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio from VibeVoice model"""
        if not self.model or not self.processor:
            raise TTSModelNotFoundError(
                "VibeVoice model not initialized",
                provider=self.provider_name
            )

        # Import StreamingAudioWriter
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import AudioNormalizer, StreamingAudioWriter

        normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=self.sample_rate,
            channels=1
        )

        try:
            # Prepare input with speaker settings
            input_data = self._prepare_vibevoice_input(request, voice, speaker_id)

            # Prepare input data for generation
            # VibeVoice uses voice samples directly, not embeddings

            # Merge default mapping from config with request-provided mapping
            merged_mapping = None
            if isinstance(self.default_speakers_to_voices, dict):
                merged_mapping = dict(self.default_speakers_to_voices)
            req_mapping = gen_config.get("speakers_to_voices") if isinstance(gen_config, dict) else None
            if isinstance(req_mapping, dict):
                merged_mapping = {**(merged_mapping or {}), **req_mapping}

            # Prepare voice samples list ordered by speaker index expected by VibeVoice
            voice_samples = self._build_voice_samples(
                input_data["text"],
                voice_reference_path,
                voice,
                merged_mapping,
            )

            # Prepare inputs for the model
            inputs = self.processor(
                text=[input_data["text"]],  # Wrap in list for batch
                voice_samples=[voice_samples] if voice_samples else None,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True
            )
            torch_mod = _get_torch(allow_import=True)
            if torch_mod is None:
                raise TTSProviderNotConfiguredError(
                    "VibeVoice generation requires torch, which is unavailable",
                    provider=self.provider_name,
                )

            # Move tensors to device
            for k, v in inputs.items():
                if torch_mod.is_tensor(v):
                    inputs[k] = v.to(self.device)

            # Prepare generation config from gen_config parameter
            cfg_scale = gen_config.get("cfg_scale", self.default_cfg_scale)
            temperature = gen_config.get("temperature", self.default_temperature)
            top_p = gen_config.get("top_p", self.default_top_p)
            top_k = gen_config.get("top_k", self.default_top_k)
            seed = gen_config.get("seed")

            # Set seed for reproducibility
            if seed is not None:
                torch_mod.manual_seed(seed)
                if self.device == "cuda" and _torch_cuda_available(allow_import=False):
                    torch_mod.cuda.manual_seed(seed)

            # Create generation config
            generation_config = {
                'do_sample': temperature > 0.0,
                'temperature': temperature if temperature > 0.0 else 1.0,
                'top_p': top_p,
            }

            # Add top_k if specified
            if top_k > 0:
                generation_config['top_k'] = top_k

            # Generate with VibeVoice model
            with torch_mod.no_grad():
                # Check for cancellation before generation
                self._check_cancellation()

                outputs = self.model.generate(
                    **inputs,
                    # Bound generation to prevent runaway
                    max_new_tokens=self.context_length,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config=generation_config,
                    is_prefill=bool(voice_samples),
                    refresh_negative=True,
                    show_progress_bar=False,
                    verbose=False
                )

                # Get the generated audio
                if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                    audio_array = outputs.speech_outputs[0].cpu().numpy()

                    # Stream the audio in chunks with configurable size
                    chunk_size = int(self.sample_rate * self.stream_chunk_size)
                    for i in range(0, len(audio_array), chunk_size):
                        # Check for cancellation during streaming
                        self._check_cancellation()

                        chunk = audio_array[i:i + chunk_size]

                        if len(chunk) > 0:
                            # Normalize to int16
                            normalized_chunk = normalizer.normalize(chunk, target_dtype=np.int16)

                            # Encode to target format
                            encoded_bytes = writer.write_chunk(normalized_chunk)
                            if encoded_bytes:
                                yield encoded_bytes
                else:
                    logger.error("No audio output generated from VibeVoice")
                    raise TTSGenerationError(
                        "Failed to generate audio",
                        provider=self.provider_name
                    )

            # Finalize stream
            final_bytes = writer.write_chunk(finalize=True)
            if final_bytes:
                yield final_bytes

            logger.info(f"{self.provider_name}: Successfully generated audio with speaker={speaker_id}")

        except TTSModelNotFoundError:
            raise
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"{self.provider_name} streaming error: {e}")
            raise TTSGenerationError(
                f"Streaming error in {self.provider_name}",
                provider=self.provider_name,
                details={"error": str(e)}
            ) from e
        finally:
            writer.close()
            # Clean up voice reference file if used (but not custom voices from Voices folder)
            if voice_reference_path and voice_reference_path not in self.available_voices.values():
                try:
                    from pathlib import Path
                    Path(voice_reference_path).unlink(missing_ok=True)
                    logger.debug(f"Cleaned up temporary voice reference: {voice_reference_path}")
                except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
                    logger.warning(f"Failed to clean up voice reference: {e}")

    async def _generate_complete_vibevoice(
        self,
        request: TTSRequest,
        voice: str,
        speaker_id: int,
        voice_reference_path: Optional[str] = None,
        gen_config: Optional[dict[str, Any]] = None
    ) -> bytes:
        """Generate complete audio from VibeVoice"""
        all_audio = b""
        async for chunk in self._stream_audio_vibevoice(request, voice, speaker_id, voice_reference_path):
            all_audio += chunk
        return all_audio

    def _prepare_vibevoice_input(
        self,
        request: TTSRequest,
        voice: str,
        speaker_id: int
    ) -> dict[str, Any]:
        """Prepare input for VibeVoice with speaker settings"""
        # Ensure text is formatted as expected by VibeVoiceProcessor
        text = self.preprocess_text(request.text)

        return {
            "text": text,
            "voice": voice,
            "speaker_id": speaker_id,
            "speed": request.speed,
            "pitch": request.pitch,
            "volume": request.volume,
            "background_music": self.enable_background_music,
            "singing": self.enable_singing
        }


    async def _prepare_voice_reference(self, voice_reference: bytes) -> Optional[str]:
        """
        Prepare voice reference audio for VibeVoice with zero-shot cloning support.
        VibeVoice supports single-sample voice cloning (3-10 seconds recommended).

        Args:
            voice_reference: Voice reference audio bytes

        Returns:
            Path to temporary voice reference file or None if processing fails
        """
        try:
            import tempfile

            import librosa
            import soundfile as sf

            from tldw_Server_API.app.core.TTS.audio_utils import process_voice_reference_async

            # Process voice reference for VibeVoice requirements
            processed_audio, error = await process_voice_reference_async(
                voice_reference,
                provider='vibevoice',
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
                prefix='vibevoice_voice_'
            ) as tmp_file:
                tmp_file.write(processed_audio)
                tmp_path = tmp_file.name

            # Validate duration (3-10 seconds recommended)
            try:
                # Load audio to check duration
                audio_data, sr = sf.read(tmp_path)
                duration = len(audio_data) / sr

                if duration < 3.0:
                    logger.warning(f"Voice reference is {duration:.1f}s, recommended minimum is 3s for better quality")
                elif duration > 10.0:
                    logger.info(f"Voice reference is {duration:.1f}s, truncating to 10s for optimal performance")
                    # Truncate to 10 seconds
                    max_samples = int(10.0 * sr)
                    audio_data = audio_data[:max_samples]
                    sf.write(tmp_path, audio_data, sr)
                else:
                    logger.info(f"Voice reference duration: {duration:.1f}s (optimal)")

                # Resample to 24kHz if needed (VibeVoice standard)
                if sr != 24000:
                    logger.info(f"Resampling voice reference from {sr}Hz to 24000Hz")
                    audio_resampled = librosa.resample(audio_data, orig_sr=sr, target_sr=24000)
                    sf.write(tmp_path, audio_resampled, 24000)

            except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Could not validate voice reference duration: {e}")

            logger.info("Voice reference prepared for VibeVoice")
            return tmp_path

        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error("Failed to prepare voice reference ({})", type(e).__name__)
            return None

    def map_voice(self, voice_id: str) -> str:
        """Map generic voice ID to VibeVoice voice"""
        # Handle custom: prefix
        if voice_id.startswith("custom:"):
            if voice_id in self.available_voices:
                return voice_id
            else:
                logger.warning(f"Custom voice {voice_id} not found")
                return "speaker_1"

        # Check available voices first
        if voice_id in self.available_voices:
            return voice_id

        # Then check presets
        if voice_id in self.VOICE_PRESETS:
            return voice_id

        # Try to map speaker numbers
        if voice_id.isdigit():
            speaker_num = int(voice_id)
            if 1 <= speaker_num <= 4:
                return f"speaker_{speaker_num}"

        # Default to speaker 1
        logger.warning(f"Voice {voice_id} not found, using default speaker_1")
        return "speaker_1"

    def _parse_multi_speaker_text(self, text: str) -> tuple[str, Optional[dict[int, str]]]:
        """Parse text for multi-speaker markers and return cleaned text with speaker mapping."""
        # Pattern to match various speaker formats
        # Supports: [1]:, [Speaker1]:, Speaker 1:, etc.
        patterns = [
            r'\[(\d+)\]:\s*',  # [1]: text
            r'\[Speaker\s*(\d+)\]:\s*',  # [Speaker1]: text
            r'Speaker\s*(\d+):\s*',  # Speaker 1: text
        ]

        speaker_mapping = {}
        cleaned_text = text

        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                speaker_num = int(match.group(1))
                # Convert to 0-based indexing for internal use
                speaker_idx = speaker_num - 1
                if 0 <= speaker_idx < 4:  # VibeVoice supports up to 4 speakers
                    speaker_mapping[speaker_idx] = f"speaker_{speaker_num}"
                    # Mark the text segment for this speaker
                    cleaned_text = cleaned_text.replace(match.group(0), f"[{speaker_idx}] ")

        # If no speaker markers found, return original text
        if not speaker_mapping:
            return text, None

        return cleaned_text, speaker_mapping

    def preprocess_text(self, text: str, **kwargs) -> str:
        """Preprocess and format text for VibeVoice."""
        # Basic cleanup
        text = super().preprocess_text(text)
        # If already in "Speaker N:" format, leave as is
        if re.search(r"^\s*Speaker\s+\d+\s*:", text, flags=re.IGNORECASE | re.MULTILINE):
            return text
        # Attempt to convert bracketed markers like [1]: to Speaker 1:
        # Replace occurrences at line starts
        text_conv = re.sub(r"^\s*\[(?:Speaker\s*)?(\d+)\]\s*:\s*",
                           lambda m: f"Speaker {m.group(1)}: ",
                           text,
                           flags=re.IGNORECASE | re.MULTILINE)
        if re.search(r"^\s*Speaker\s+\d+\s*:", text_conv, flags=re.IGNORECASE | re.MULTILINE):
            return text_conv
        # As a safe fallback, wrap entire text as a single-speaker script
        return f"Speaker 1: {text.strip()}"

    async def _generate_synthetic_voice(self, speaker_id: int) -> Optional[str]:
        """Generate a synthetic voice sample for a speaker when no reference is available."""
        try:
            logger.info(f"Generating synthetic voice for speaker {speaker_id}")

            # Create a deterministic synthetic voice based on speaker ID
            # This would typically use the model's voice generation capabilities
            # For now, return None to use default voice
            # In a real implementation, this would generate a voice sample

            # Placeholder for synthetic voice generation
            # Could use techniques like:
            # - Random noise with speaker-specific seed
            # - Pre-generated synthetic samples
            # - Model's built-in voice synthesis

            return None

        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to generate synthetic voice: {e}")
            return None

    async def _reload_model_for_variant(self, variant: str):
        """Reload the model with a different variant (1.5B or 7B)"""
        if variant not in self.MODEL_VARIANTS:
            raise ValueError(f"Invalid variant: {variant}. Must be one of {list(self.MODEL_VARIANTS.keys())}")

        # Clean up existing model
        await self._cleanup_resources()

        # Update configuration for new variant
        variant_config = self.MODEL_VARIANTS[variant]
        self.model_path = variant_config["path"]
        self.context_length = variant_config["context"]
        self.frame_rate = variant_config["frame_rate"]

        # Reinitialize with new variant
        logger.info(f"Reloading VibeVoice with variant: {variant}")
        await self.initialize()

    def _get_best_attention_implementation(self) -> str:
        """Get the best available attention implementation based on hardware and config."""
        if self.default_attention_type != "auto":
            return self.default_attention_type

        for attn_type in self.attention_fallback_chain:
            if self._is_attention_available(attn_type):
                logger.info(f"Using attention implementation: {attn_type}")
                return attn_type

        # Default fallback
        logger.info("Using default eager attention")
        return "eager"

    def _is_attention_available(self, attn_type: str) -> bool:
        """Check if a specific attention implementation is available."""
        if attn_type == "flash_attention_2":
            # Check for flash attention support
            if self.device == "cuda" and importlib.util.find_spec("flash_attn") is not None:
                return True
        elif attn_type == "sage":
            # Check for SageAttention
            if self.enable_sage:
                if importlib.util.find_spec("sageattention") is not None:
                    return True
                logger.debug("SageAttention not available")
        elif attn_type == "sdpa":
            # SDPA is generally available in newer PyTorch
            torch_mod = _get_torch(allow_import=False)
            if torch_mod is None:
                return False
            return hasattr(torch_mod.nn.functional, "scaled_dot_product_attention")
        elif attn_type == "eager":
            # Always available
            return True
        return False

    def _calculate_quantization_savings(self):
        """Calculate estimated memory savings from quantization."""
        # Estimate based on model variant
        if self.variant == "1.5B":
            original_size_gb = 3.0  # Approximate size in FP16
            quantized_size_gb = 1.1  # 4-bit quantized size
        elif self.variant == "7B":
            original_size_gb = 14.0  # Approximate size in FP16
            quantized_size_gb = 5.0  # 4-bit quantized size
        else:
            original_size_gb = 3.0
            quantized_size_gb = 1.1

        self._memory_stats["quantization_savings_gb"] = original_size_gb - quantized_size_gb
        self._memory_stats["quantization_savings_percent"] = (
            (original_size_gb - quantized_size_gb) / original_size_gb * 100
        )

    def _update_memory_stats(self):
        """Update memory usage statistics."""
        try:
            torch_mod = _get_torch(allow_import=False)
            if self.device == "cuda" and _torch_cuda_available(allow_import=False) and torch_mod is not None:
                # Get CUDA memory stats
                allocated_gb = torch_mod.cuda.memory_allocated() / 1024**3
                reserved_gb = torch_mod.cuda.memory_reserved() / 1024**3

                self._memory_stats["current_vram_gb"] = allocated_gb
                self._memory_stats["reserved_vram_gb"] = reserved_gb

                # Update peak if necessary
                if allocated_gb > self._memory_stats["peak_vram_gb"]:
                    self._memory_stats["peak_vram_gb"] = allocated_gb

                logger.debug(f"VRAM usage: {allocated_gb:.2f}GB allocated, {reserved_gb:.2f}GB reserved")
            elif self.device == "mps" and _torch_mps_available(allow_import=False) and torch_mod is not None:
                # MPS memory tracking is limited
                try:
                    allocated_gb = torch_mod.mps.current_allocated_memory() / 1024**3
                    self._memory_stats["current_vram_gb"] = allocated_gb

                    if allocated_gb > self._memory_stats["peak_vram_gb"]:
                        self._memory_stats["peak_vram_gb"] = allocated_gb
                except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"MPS memory tracking not available: error={e}")
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Could not update memory stats: {e}")

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics."""
        self._update_memory_stats()
        return self._memory_stats.copy()

    async def _cleanup_resources(self):
        """Clean up VibeVoice adapter resources with enhanced memory management."""
        try:
            # Cancel any ongoing generation
            if self._current_generation_task:
                self._generation_cancelled = True
                if not self._current_generation_task.done():
                    self._current_generation_task.cancel()
                self._current_generation_task = None

            if self.model:
                del self.model
            if self.processor:
                del self.processor

            self.model = None
            self.processor = None

            # Force garbage collection
            gc.collect()

            # Clear GPU cache if using CUDA
            torch_mod = _get_torch(allow_import=False)
            if self.device == "cuda" and _torch_cuda_available(allow_import=False) and torch_mod is not None:
                torch_mod.cuda.empty_cache()
                torch_mod.cuda.synchronize()
            elif self.device == "mps" and _torch_mps_available(allow_import=False) and torch_mod is not None:
                # Clear MPS cache for Apple Silicon
                torch_mod.mps.empty_cache()
                torch_mod.mps.synchronize()

            # Update memory stats after cleanup
            self._update_memory_stats()

            logger.debug(f"{self.provider_name}: Resources cleaned up")
        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"{self.provider_name}: Error during cleanup ({{}})", type(e).__name__)

    async def cleanup_after_generation(self):
        """Clean up after each generation to free memory."""
        try:
            # Clear any temporary tensors
            gc.collect()

            # Clear device cache without unloading model if auto_cleanup is enabled
            if self.auto_cleanup:
                torch_mod = _get_torch(allow_import=False)
                if self.device == "cuda" and _torch_cuda_available(allow_import=False) and torch_mod is not None:
                    torch_mod.cuda.empty_cache()
                elif self.device == "mps" and _torch_mps_available(allow_import=False) and torch_mod is not None:
                    torch_mod.mps.empty_cache()

                # Update memory stats
                self._update_memory_stats()
                logger.debug(f"Post-generation cleanup: {self._memory_stats['current_vram_gb']:.2f}GB in use")

        except _VIBEVOICE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("Post-generation cleanup error ({})", type(e).__name__)

    def cancel_generation(self):
        """Cancel the current generation task."""
        self._generation_cancelled = True
        logger.info(f"{self.provider_name}: Generation cancelled by user")

    def _check_cancellation(self):
        """Check if generation has been cancelled."""
        if self._generation_cancelled:
            self._generation_cancelled = False  # Reset for next generation
            raise TTSGenerationError(
                "Generation cancelled by user",
                provider=self.provider_name,
                details={"reason": "user_cancellation"}
            )

#
# End of vibevoice_adapter.py
#######################################################################################################################

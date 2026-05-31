# echo_tts_adapter.py
# Description: Echo-TTS adapter implementation (CUDA-only, voice reference required)
#
# Imports
import asyncio
import hashlib
import importlib
import re
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

#
# Third-party Imports
import numpy as np
from loguru import logger

from ..streaming_audio_writer import AudioNormalizer, StreamingAudioWriter
from ..tts_exceptions import (
    TTSGenerationError,
    TTSInvalidVoiceReferenceError,
    TTSModelLoadError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSUnsupportedFormatError,
    TTSValidationError,
)
from ..tts_resource_manager import get_resource_manager
from ..tts_validation import TTSInputValidator, validate_tts_request
from ..utils import parse_bool

#
# Local Imports
from .base import (
    AudioFormat,
    ProviderStatus,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
)

#
#######################################################################################################################
#
# Echo-TTS Adapter


def _safe_exception_label(exc: BaseException) -> str:
    """Return a non-sensitive exception identifier for logs."""
    return type(exc).__name__


class EchoTTSAdapter(TTSAdapter):
    """Adapter for Echo-TTS (CUDA-only, speaker reference required)."""

    PROVIDER_KEY = "echo_tts"
    handles_text_chunking = True
    SUPPORTED_FORMATS: set[AudioFormat] = {
        AudioFormat.MP3,
        AudioFormat.WAV,
        AudioFormat.FLAC,
        AudioFormat.OPUS,
        AudioFormat.AAC,
        AudioFormat.PCM,
    }
    SUPPORTED_LANGUAGES = {"en"}
    DEFAULT_SAMPLE_RATE = 44100
    MAX_TEXT_LENGTH = 768
    MAX_TEXT_BYTES = 767
    DEFAULT_BLOCK_SIZE = 160

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)
        cfg = config or {}

        extra_cfg = cfg.get("extra_params") if isinstance(cfg.get("extra_params"), dict) else {}

        self.model_repo = (
            cfg.get("echo_tts_model")
            or cfg.get("model")
            or extra_cfg.get("model")
            or "jordand/echo-tts-base"
        )
        self.fish_ae_repo = (
            cfg.get("echo_tts_fish_ae_repo")
            or cfg.get("fish_ae_repo")
            or extra_cfg.get("fish_ae_repo")
            or "jordand/fish-s1-dac-min"
        )
        self.pca_state_file = (
            cfg.get("echo_tts_pca_state_file")
            or cfg.get("pca_state_file")
            or extra_cfg.get("pca_state_file")
            or "pca_state.safetensors"
        )
        self.module_path = Path(
            cfg.get("echo_tts_module_path")
            or cfg.get("module_path")
            or "../echo-tts"
        ).expanduser()

        self.device_pref = str(
            cfg.get("echo_tts_device")
            or cfg.get("device")
            or extra_cfg.get("device")
            or "auto"
        ).lower()
        self.sample_rate = int(
            cfg.get("echo_tts_sample_rate")
            or cfg.get("sample_rate")
            or extra_cfg.get("sample_rate")
            or self.DEFAULT_SAMPLE_RATE
        )

        self.cache_size = int(
            cfg.get("echo_tts_cache_size")
            or cfg.get("cache_size")
            or extra_cfg.get("cache_size")
            or 8
        )
        self.cache_ttl_sec = float(
            cfg.get("echo_tts_cache_ttl_sec")
            or cfg.get("cache_ttl_sec")
            or extra_cfg.get("cache_ttl_sec")
            or 3600
        )
        self.cache_on_device = parse_bool(
            cfg.get("echo_tts_cache_on_device")
            or cfg.get("cache_on_device")
            or extra_cfg.get("cache_on_device"),
            default=False,
        )

        self.max_reference_seconds = int(
            cfg.get("echo_tts_max_reference_seconds")
            or cfg.get("max_reference_seconds")
            or extra_cfg.get("max_reference_seconds")
            or 300
        )

        self._model = None
        self._fish_ae = None
        self._pca_state = None
        self._echo_modules_loaded = False
        self._model_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()
        self._speaker_cache: OrderedDict[str, tuple[float, Any, Any]] = OrderedDict()

        self._echo_inference = None
        self._echo_blockwise = None

    async def initialize(self) -> bool:
        """Initialize Echo-TTS adapter (verify dependencies and CUDA availability)."""
        try:
            torch = self._import_torch()
            self._resolve_device(torch)
            self._load_echo_modules()
            self._status = ProviderStatus.AVAILABLE
            return True
        except Exception as exc:
            logger.error(
                "{}: initialization failed; exception_type={}",
                self.provider_name,
                _safe_exception_label(exc),
            )
            self._status = ProviderStatus.ERROR
            if isinstance(exc, TTSProviderInitializationError):
                raise
            return False

    async def get_capabilities(self) -> TTSCapabilities:
        supports_streaming = True
        if not self._echo_modules_loaded:
            # Best effort: capabilities should still list streaming support if module exists.
            try:
                self._load_echo_modules()
            except (ImportError, ModuleNotFoundError, OSError, RuntimeError):
                supports_streaming = False
        return TTSCapabilities(
            provider_name="EchoTTS",
            supported_languages=self.SUPPORTED_LANGUAGES,
            supported_voices=[],
            supported_formats=self.SUPPORTED_FORMATS,
            max_text_length=self.MAX_TEXT_LENGTH,
            supports_streaming=supports_streaming,
            supports_voice_cloning=True,
            supports_emotion_control=False,
            supports_speech_rate=False,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=False,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=1500,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.WAV,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "Echo-TTS adapter not initialized",
                provider=self.PROVIDER_KEY,
            )

        if request.format not in self.SUPPORTED_FORMATS:
            raise TTSUnsupportedFormatError(
                f"Format {request.format.value} not supported by Echo-TTS",
                provider=self.PROVIDER_KEY,
            )

        extras = request.extra_params or {}
        if not isinstance(extras, dict):
            extras = {}

        max_chars, max_bytes = self._resolve_chunk_limits(extras)
        chunk_flag = self._resolve_chunking_flag(extras)
        text_exceeds = self._text_exceeds_limits(request.text, max_chars, max_bytes)

        if chunk_flag is False and text_exceeds:
            # Force validation error for oversized text when chunking explicitly disabled.
            validate_tts_request(request, provider=self.PROVIDER_KEY)

        chunking_enabled = text_exceeds if chunk_flag is None else bool(chunk_flag)
        text_chunks = (
            self._split_text_chunks(request.text, max_chars=max_chars, max_bytes=max_bytes)
            if chunking_enabled
            else [request.text]
        )

        if not text_chunks:
            raise TTSValidationError(
                "Validation failed for Echo-TTS request: text cannot be empty",
                provider=self.PROVIDER_KEY,
            )

        try:
            if len(text_chunks) > 1:
                validation_request = replace(request, text=text_chunks[0])
                validate_tts_request(validation_request, provider=self.PROVIDER_KEY)
                self._validate_text_chunks(
                    text_chunks,
                    max_chars=max_chars,
                    max_bytes=max_bytes,
                )
            else:
                validate_tts_request(request, provider=self.PROVIDER_KEY)
        except TTSValidationError:
            raise
        except Exception as exc:
            raise TTSValidationError(
                f"Validation failed for Echo-TTS request: {exc}",
                provider=self.PROVIDER_KEY,
            ) from exc

        if not request.voice_reference:
            raise TTSInvalidVoiceReferenceError(
                "Echo-TTS requires voice_reference audio bytes",
                provider=self.PROVIDER_KEY,
            )

        await self._ensure_models_loaded()

        voice_bytes = self._extract_voice_reference(request.voice_reference)
        voice_bytes = await self._prepare_voice_reference(voice_bytes, extras)

        cache_key = self._cache_key(voice_bytes)
        speaker_latent, speaker_mask = await self._get_cached_speaker_latent(cache_key)
        if speaker_latent is None or speaker_mask is None:
            speaker_latent, speaker_mask = await self._compute_speaker_latent(voice_bytes)
            await self._store_speaker_latent(cache_key, speaker_latent, speaker_mask)

        self._import_torch()
        inference = self._echo_inference
        if inference is None:
            raise TTSModelLoadError(
                "Echo-TTS inference module not loaded",
                provider=self.PROVIDER_KEY,
            )

        try:
            speaker_latent = speaker_latent.to(self._model.device)
            speaker_mask = speaker_mask.to(self._model.device)

            if len(text_chunks) > 1:
                if request.stream:
                    audio_stream = self._stream_chunked_audio(
                        request=request,
                        text_chunks=text_chunks,
                        extras=extras,
                        inference=inference,
                        speaker_latent=speaker_latent,
                        speaker_mask=speaker_mask,
                    )
                    return TTSResponse(
                        audio_stream=audio_stream,
                        format=request.format,
                        sample_rate=self.sample_rate,
                        channels=1,
                        text_processed=request.text,
                        voice_used=request.voice,
                        provider=self.PROVIDER_KEY,
                        model=request.model or self.model_repo,
                    )

                audio_bytes = await self._generate_chunked_audio(
                    request=request,
                    text_chunks=text_chunks,
                    extras=extras,
                    inference=inference,
                    speaker_latent=speaker_latent,
                    speaker_mask=speaker_mask,
                )
                return TTSResponse(
                    audio_data=audio_bytes,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    text_processed=request.text,
                    voice_used=request.voice,
                    provider=self.PROVIDER_KEY,
                    model=request.model or self.model_repo,
                )

            text_input_ids, text_mask, normalized_text_value = self._prepare_text_inputs(
                request,
                extras,
                inference,
            )
            if request.stream:
                audio_stream = self._stream_audio(
                    request=request,
                    extras=extras,
                    inference=inference,
                    text_input_ids=text_input_ids,
                    text_mask=text_mask,
                    speaker_latent=speaker_latent,
                    speaker_mask=speaker_mask,
                )
                return TTSResponse(
                    audio_stream=audio_stream,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    channels=1,
                    text_processed=normalized_text_value,
                    voice_used=request.voice,
                    provider=self.PROVIDER_KEY,
                    model=request.model or self.model_repo,
                )

            latent_out = self._run_full_generation(
                inference=inference,
                text_input_ids=text_input_ids,
                text_mask=text_mask,
                speaker_latent=speaker_latent,
                speaker_mask=speaker_mask,
                extras=extras,
            )
            audio_bytes = await self._latent_to_audio_bytes(
                inference=inference,
                latent_out=latent_out,
                request_format=request.format,
            )
            return TTSResponse(
                audio_data=audio_bytes,
                format=request.format,
                sample_rate=self.sample_rate,
                channels=1,
                text_processed=normalized_text_value,
                voice_used=request.voice,
                provider=self.PROVIDER_KEY,
                model=request.model or self.model_repo,
            )
        except Exception as exc:
            logger.error(
                "Echo-TTS generation failed; exception_type={}",
                _safe_exception_label(exc),
            )
            raise TTSGenerationError(
                "Echo-TTS generation failed",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

    async def generate_stream(self, request: TTSRequest):
        """Legacy streaming entrypoint used by older tests."""
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "Echo-TTS adapter not initialized",
                provider=self.PROVIDER_KEY,
            )

        request.stream = True
        response = await self.generate(request)
        if response.audio_stream is None:
            raise TTSGenerationError(
                "Echo-TTS streaming did not return an audio stream",
                provider=self.PROVIDER_KEY,
            )
        async for chunk in response.audio_stream:
            yield chunk

    async def _cleanup_resources(self) -> None:
        self._model = None
        self._fish_ae = None
        self._pca_state = None
        self._speaker_cache.clear()

    def _import_torch(self):
        try:
            import torch  # type: ignore
            return torch
        except Exception as exc:
            raise TTSModelLoadError(
                "PyTorch is required for Echo-TTS",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

    def _resolve_device(self, torch_module) -> None:
        if self.device_pref == "auto":
            device = "cuda"
        elif self.device_pref == "cuda":
            device = self.device_pref
        elif self.device_pref == "cpu":
            raise TTSProviderInitializationError(
                "Echo-TTS is CUDA-only; device 'cpu' is not supported",
                provider=self.PROVIDER_KEY,
                details={"valid_devices": ["auto", "cuda"]},
            )
        else:
            raise TTSProviderInitializationError(
                f"Unsupported device '{self.device_pref}' for Echo-TTS",
                provider=self.PROVIDER_KEY,
                details={"valid_devices": ["auto", "cuda"]},
            )

        if device == "cuda" and not torch_module.cuda.is_available():
            raise TTSProviderInitializationError(
                "CUDA is required for Echo-TTS but is not available",
                provider=self.PROVIDER_KEY,
            )

        self.device = device

    def _load_echo_modules(self) -> None:
        if self._echo_modules_loaded:
            return
        try:
            inference = importlib.import_module("inference")
            blockwise = importlib.import_module("inference_blockwise")
        except (ImportError, ModuleNotFoundError):
            module_dir = self.module_path
            if not module_dir.exists():
                raise TTSModelLoadError(
                    "Echo-TTS module path not found",
                    provider=self.PROVIDER_KEY,
                    details={"module_path": str(module_dir)},
                ) from None
            module_path_str = str(module_dir)
            if module_path_str not in sys.path:
                sys.path.insert(0, module_path_str)
            inference = importlib.import_module("inference")
            blockwise = importlib.import_module("inference_blockwise")

        self._echo_inference = inference
        self._echo_blockwise = blockwise
        self._echo_modules_loaded = True

    async def _ensure_models_loaded(self) -> None:
        if self._model is not None and self._fish_ae is not None and self._pca_state is not None:
            return
        async with self._model_lock:
            if self._model is not None and self._fish_ae is not None and self._pca_state is not None:
                return

            self._load_echo_modules()
            inference = self._echo_inference
            if inference is None:
                raise TTSModelLoadError(
                    "Echo-TTS inference module not loaded",
                    provider=self.PROVIDER_KEY,
                )

            torch = self._import_torch()

            model_dtype = torch.bfloat16
            try:
                if parse_bool(self.config.get("echo_tts_use_fp16"), default=False):
                    model_dtype = torch.float16
                elif parse_bool(self.config.get("echo_tts_use_fp32"), default=False):
                    model_dtype = torch.float32
            except (TypeError, ValueError):
                model_dtype = torch.bfloat16

            self._model = await asyncio.to_thread(
                inference.load_model_from_hf,
                repo_id=self.model_repo,
                device=self.device,
                dtype=model_dtype,
                compile=False,
                delete_blockwise_modules=True,
            )
            self._fish_ae = await asyncio.to_thread(
                inference.load_fish_ae_from_hf,
                repo_id=self.fish_ae_repo,
                device=self.device,
                dtype=torch.float32,
                compile=False,
            )
            self._pca_state = await asyncio.to_thread(
                inference.load_pca_state_from_hf,
                repo_id=self.model_repo,
                device=self.device,
                filename=self.pca_state_file,
            )

            # Register model for memory monitoring (best-effort)
            try:
                resource_manager = await get_resource_manager()
                register_result = resource_manager.register_model(
                    provider=self.PROVIDER_KEY,
                    model_instance=self._model,
                    cleanup_callback=self._cleanup_resources,
                    model_key=str(self.model_repo),
                )
                if asyncio.iscoroutine(register_result):
                    await register_result
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass

    def _cache_key(self, voice_bytes: bytes) -> str:
        digest = hashlib.sha256(voice_bytes).hexdigest()
        return f"{digest}:{self.model_repo}:{self.fish_ae_repo}:{self.device}:{self.sample_rate}"

    async def _get_cached_speaker_latent(self, cache_key: str) -> tuple[Optional[Any], Optional[Any]]:
        if self.cache_size <= 0:
            return None, None
        async with self._cache_lock:
            self._prune_cache_locked()
            entry = self._speaker_cache.get(cache_key)
            if not entry:
                return None, None
            timestamp, latent, mask = entry
            self._speaker_cache.move_to_end(cache_key)
            if not self.cache_on_device:
                latent = latent.to(self.device)
                mask = mask.to(self.device)
            return latent, mask

    async def _store_speaker_latent(self, cache_key: str, latent: Any, mask: Any) -> None:
        if self.cache_size <= 0:
            return
        try:
            resource_manager = await get_resource_manager()
            if resource_manager.memory_monitor.is_memory_critical():
                async with self._cache_lock:
                    self._speaker_cache.clear()
                return
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass

        if not self.cache_on_device:
            latent = latent.detach().cpu()
            mask = mask.detach().cpu()
        else:
            latent = latent.detach()
            mask = mask.detach()

        async with self._cache_lock:
            self._speaker_cache[cache_key] = (time.time(), latent, mask)
            self._speaker_cache.move_to_end(cache_key)
            self._prune_cache_locked()

    def _prune_cache_locked(self) -> None:
        if self.cache_ttl_sec > 0:
            now = time.time()
            expired = [key for key, (ts, _, _) in self._speaker_cache.items() if now - ts > self.cache_ttl_sec]
            for key in expired:
                self._speaker_cache.pop(key, None)
        while len(self._speaker_cache) > self.cache_size:
            self._speaker_cache.popitem(last=False)

    async def _compute_speaker_latent(self, voice_bytes: bytes):
        inference = self._echo_inference
        if inference is None:
            raise TTSModelLoadError(
                "Echo-TTS inference module not loaded",
                provider=self.PROVIDER_KEY,
            )
        voice_path = self._write_temp_audio(voice_bytes)
        try:
            speaker_audio = await asyncio.to_thread(
                inference.load_audio,
                voice_path,
                self.max_reference_seconds,
            )
        finally:
            Path(voice_path).unlink(missing_ok=True)

        speaker_audio = speaker_audio.to(device=self._model.device, dtype=self._fish_ae.dtype)
        speaker_latent, speaker_mask = inference.get_speaker_latent_and_mask(
            self._fish_ae,
            self._pca_state,
            speaker_audio,
        )
        return speaker_latent, speaker_mask

    def _prepare_text_inputs(
        self,
        request: TTSRequest,
        extras: dict[str, Any],
        inference: Any,
    ) -> tuple[Any, Any, str]:
        normalize_text = self._coerce_bool(extras.get("normalize_text"), True)
        text_input_ids, text_mask, normalized_text = inference.get_text_input_ids_and_mask(
            [request.text],
            max_length=min(self.MAX_TEXT_LENGTH, 768),
            device=self._model.device,
            normalize=normalize_text,
            return_normalized_text=True,
            pad_to_max=False,
        )
        normalized_text_value = normalized_text[0] if normalized_text else request.text
        return text_input_ids, text_mask, normalized_text_value

    def _run_full_generation(
        self,
        *,
        inference: Any,
        text_input_ids: Any,
        text_mask: Any,
        speaker_latent: Any,
        speaker_mask: Any,
        extras: dict[str, Any],
    ):
        seq_len = self._resolve_sequence_length(extras)
        rng_seed = self._coerce_int(extras.get("rng_seed"), 0)
        cfg_scale_text = self._coerce_float(extras.get("cfg_scale_text"), 3.0)
        cfg_scale_speaker = self._coerce_float(extras.get("cfg_scale_speaker"), 8.0)
        cfg_min_t = self._coerce_float(extras.get("cfg_min_t"), 0.5)
        cfg_max_t = self._coerce_float(extras.get("cfg_max_t"), 1.0)
        truncation_factor = self._coerce_optional_float(extras.get("truncation_factor"))
        rescale_k = self._coerce_optional_float(extras.get("rescale_k"))
        rescale_sigma = self._coerce_optional_float(extras.get("rescale_sigma"))
        speaker_kv_scale = self._coerce_optional_float(extras.get("speaker_kv_scale"))
        speaker_kv_min_t = self._coerce_optional_float(extras.get("speaker_kv_min_t"))
        speaker_kv_max_layers = self._coerce_optional_int(extras.get("speaker_kv_max_layers"))
        num_steps = self._coerce_int(extras.get("num_steps"), 40)

        return inference.sample_euler_cfg_independent_guidances(
            model=self._model,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            text_input_ids=text_input_ids,
            text_mask=text_mask,
            rng_seed=rng_seed,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_max_layers=speaker_kv_max_layers,
            speaker_kv_min_t=speaker_kv_min_t,
            sequence_length=seq_len,
        )

    async def _latent_to_audio_bytes(
        self,
        *,
        inference: Any,
        latent_out: Any,
        request_format: AudioFormat,
    ) -> bytes:
        audio_np = await self._latent_to_audio_np(
            inference=inference,
            latent_out=latent_out,
        )
        return await self.convert_audio_format(
            audio_np,
            source_format=AudioFormat.PCM,
            target_format=request_format,
            sample_rate=self.sample_rate,
        )

    async def _latent_to_audio_np(self, *, inference: Any, latent_out: Any) -> np.ndarray:
        audio_out = inference.ae_decode(self._fish_ae, self._pca_state, latent_out)
        audio_out = inference.crop_audio_to_flattening_point(audio_out, latent_out[0])
        audio_np = np.asarray(audio_out[0].detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        return audio_np

    def _stream_audio(
        self,
        *,
        request: TTSRequest,
        extras: dict[str, Any],
        inference: Any,
        text_input_ids: Any,
        text_mask: Any,
        speaker_latent: Any,
        speaker_mask: Any,
    ):
        blockwise = self._echo_blockwise
        if blockwise is None:
            raise TTSGenerationError(
                "Echo-TTS blockwise module not available for streaming",
                provider=self.PROVIDER_KEY,
            )

        torch = self._import_torch()
        audio_normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=self.sample_rate,
            channels=1,
        )

        seq_len = self._resolve_sequence_length(extras)
        block_size = self.DEFAULT_BLOCK_SIZE
        if block_size <= 0:
            block_size = 160
        block_sizes = []
        remaining = seq_len
        while remaining > 0:
            chunk = min(block_size, remaining)
            block_sizes.append(chunk)
            remaining -= chunk

        rng_seed = self._coerce_int(extras.get("rng_seed"), 0)
        cfg_scale_text = self._coerce_float(extras.get("cfg_scale_text"), 3.0)
        cfg_scale_speaker = self._coerce_float(extras.get("cfg_scale_speaker"), 8.0)
        cfg_min_t = self._coerce_float(extras.get("cfg_min_t"), 0.5)
        cfg_max_t = self._coerce_float(extras.get("cfg_max_t"), 1.0)
        truncation_factor = self._coerce_optional_float(extras.get("truncation_factor"))
        rescale_k = self._coerce_optional_float(extras.get("rescale_k"))
        rescale_sigma = self._coerce_optional_float(extras.get("rescale_sigma"))
        speaker_kv_scale = self._coerce_optional_float(extras.get("speaker_kv_scale"))
        speaker_kv_min_t = self._coerce_optional_float(extras.get("speaker_kv_min_t"))
        speaker_kv_max_layers = self._coerce_optional_int(extras.get("speaker_kv_max_layers"))
        num_steps = self._coerce_int(extras.get("num_steps"), 40)

        async def stream():
            init_scale = 0.999
            device = self._model.device
            dtype = self._model.dtype
            batch_size = text_input_ids.shape[0]

            rng = torch.Generator(device=device).manual_seed(rng_seed)
            t_schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=device) * init_scale

            text_mask_uncond = torch.zeros_like(text_mask)
            speaker_mask_uncond = torch.zeros_like(speaker_mask)

            kv_text_cond = self._model.get_kv_cache_text(text_input_ids, text_mask)
            kv_speaker_cond = self._model.get_kv_cache_speaker(speaker_latent.to(dtype))

            kv_text_full = blockwise._concat_kv_caches(kv_text_cond, kv_text_cond, kv_text_cond)
            kv_speaker_full = blockwise._concat_kv_caches(kv_speaker_cond, kv_speaker_cond, kv_speaker_cond)

            full_text_mask = torch.cat([text_mask, text_mask_uncond, text_mask], dim=0)
            full_speaker_mask = torch.cat([speaker_mask, speaker_mask, speaker_mask_uncond], dim=0)

            prefix_latent = torch.zeros(
                (batch_size, sum(block_sizes), 80),
                device=device,
                dtype=torch.float32,
            )

            start_pos = 0
            for current_block_size in block_sizes:
                if speaker_kv_scale is not None:
                    blockwise._multiply_kv_cache(kv_speaker_cond, speaker_kv_scale, speaker_kv_max_layers)
                    kv_speaker_full = blockwise._concat_kv_caches(
                        kv_speaker_cond, kv_speaker_cond, kv_speaker_cond
                    )

                full_prefix_latent = torch.cat([prefix_latent, prefix_latent, prefix_latent], dim=0)
                kv_latent_full = self._model.get_kv_cache_latent(full_prefix_latent.to(dtype))
                kv_latent_cond = [(k[:batch_size], v[:batch_size]) for k, v in kv_latent_full]

                x_t = torch.randn(
                    (batch_size, current_block_size, 80),
                    device=device,
                    dtype=torch.float32,
                    generator=rng,
                )
                if truncation_factor is not None:
                    x_t = x_t * truncation_factor

                for i in range(num_steps):
                    t, t_next = t_schedule[i], t_schedule[i + 1]
                    has_cfg = ((t >= cfg_min_t) * (t <= cfg_max_t)).item()

                    if has_cfg:
                        v_cond, v_uncond_text, v_uncond_speaker = self._model(
                            x=torch.cat([x_t, x_t, x_t], dim=0).to(dtype),
                            t=(torch.ones((batch_size * 3,), device=device) * t).to(dtype),
                            text_mask=full_text_mask,
                            speaker_mask=full_speaker_mask,
                            start_pos=start_pos,
                            kv_cache_text=kv_text_full,
                            kv_cache_speaker=kv_speaker_full,
                            kv_cache_latent=kv_latent_full,
                        ).float().chunk(3, dim=0)
                        v_pred = (
                            v_cond
                            + cfg_scale_text * (v_cond - v_uncond_text)
                            + cfg_scale_speaker * (v_cond - v_uncond_speaker)
                        )
                    else:
                        v_pred = self._model(
                            x=x_t.to(dtype),
                            t=(torch.ones((batch_size,), device=device) * t).to(dtype),
                            text_mask=text_mask,
                            speaker_mask=speaker_mask,
                            start_pos=start_pos,
                            kv_cache_text=kv_text_cond,
                            kv_cache_speaker=kv_speaker_cond,
                            kv_cache_latent=kv_latent_cond,
                        ).float()

                    if rescale_k is not None and rescale_sigma is not None:
                        v_pred = blockwise._temporal_score_rescale(v_pred, x_t, t, rescale_k, rescale_sigma)

                    if (
                        speaker_kv_scale is not None
                        and speaker_kv_min_t is not None
                        and t_next < speaker_kv_min_t
                        and t >= speaker_kv_min_t
                    ):
                        blockwise._multiply_kv_cache(kv_speaker_cond, 1.0 / speaker_kv_scale, speaker_kv_max_layers)
                        kv_speaker_full = blockwise._concat_kv_caches(
                            kv_speaker_cond, kv_speaker_cond, kv_speaker_cond
                        )

                    x_t = x_t + v_pred * (t_next - t)

                prefix_latent[:, start_pos:start_pos + current_block_size] = x_t

                audio_chunk = inference.ae_decode(self._fish_ae, self._pca_state, x_t)
                audio_np = audio_chunk[0].detach().cpu().numpy().astype(np.float32)
                audio_i16 = audio_normalizer.normalize(audio_np, target_dtype=np.int16)
                data = writer.write_chunk(audio_i16)
                if data:
                    yield data

                start_pos += current_block_size

            final_bytes = writer.write_chunk(finalize=True)
            if final_bytes:
                yield final_bytes

        return stream()

    async def _generate_chunked_audio(
        self,
        *,
        request: TTSRequest,
        text_chunks: list[str],
        extras: dict[str, Any],
        inference: Any,
        speaker_latent: Any,
        speaker_mask: Any,
    ) -> bytes:
        interval_silence_ms = self._coerce_int(extras.get("interval_silence"), 0)
        silence_samples = max(0, int(self.sample_rate * interval_silence_ms / 1000.0))
        silence = np.zeros(silence_samples, dtype=np.float32) if silence_samples > 0 else None

        audio_segments: list[np.ndarray] = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_request = replace(request, text=chunk_text)
            text_input_ids, text_mask, _ = self._prepare_text_inputs(
                chunk_request,
                extras,
                inference,
            )
            latent_out = self._run_full_generation(
                inference=inference,
                text_input_ids=text_input_ids,
                text_mask=text_mask,
                speaker_latent=speaker_latent,
                speaker_mask=speaker_mask,
                extras=extras,
            )
            audio_np = await self._latent_to_audio_np(
                inference=inference,
                latent_out=latent_out,
            )
            if audio_np.size > 0:
                audio_segments.append(audio_np)
            if silence is not None and idx < len(text_chunks) - 1:
                audio_segments.append(silence)

        if not audio_segments:
            raise TTSGenerationError(
                "Echo-TTS generation produced no audio",
                provider=self.PROVIDER_KEY,
            )

        full_audio = np.concatenate(audio_segments).astype(np.float32)
        return await self.convert_audio_format(
            full_audio,
            source_format=AudioFormat.PCM,
            target_format=request.format,
            sample_rate=self.sample_rate,
        )

    def _stream_chunked_audio(
        self,
        *,
        request: TTSRequest,
        text_chunks: list[str],
        extras: dict[str, Any],
        inference: Any,
        speaker_latent: Any,
        speaker_mask: Any,
    ):
        interval_silence_ms = self._coerce_int(extras.get("interval_silence"), 0)
        silence_samples = max(0, int(self.sample_rate * interval_silence_ms / 1000.0))
        silence_i16 = np.zeros(silence_samples, dtype=np.int16) if silence_samples > 0 else None

        audio_normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=self.sample_rate,
            channels=1,
        )

        async def stream():
            try:
                for idx, chunk_text in enumerate(text_chunks):
                    chunk_request = replace(request, text=chunk_text)
                    text_input_ids, text_mask, _ = self._prepare_text_inputs(
                        chunk_request,
                        extras,
                        inference,
                    )
                    latent_out = self._run_full_generation(
                        inference=inference,
                        text_input_ids=text_input_ids,
                        text_mask=text_mask,
                        speaker_latent=speaker_latent,
                        speaker_mask=speaker_mask,
                        extras=extras,
                    )
                    audio_np = await self._latent_to_audio_np(
                        inference=inference,
                        latent_out=latent_out,
                    )
                    if audio_np.size > 0:
                        audio_i16 = audio_normalizer.normalize(audio_np, target_dtype=np.int16)
                        data = writer.write_chunk(audio_i16)
                        if data:
                            yield data
                    if silence_i16 is not None and idx < len(text_chunks) - 1:
                        data = writer.write_chunk(silence_i16)
                        if data:
                            yield data

                final_bytes = writer.write_chunk(finalize=True)
                if final_bytes:
                    yield final_bytes
            finally:
                writer.close()

        return stream()

    def _extract_voice_reference(self, voice_reference: Any) -> bytes:
        if voice_reference is None:
            raise TTSInvalidVoiceReferenceError(
                "Echo-TTS requires voice_reference audio bytes",
                provider=self.PROVIDER_KEY,
            )
        if isinstance(voice_reference, (bytes, bytearray)):
            return bytes(voice_reference)
        if isinstance(voice_reference, str):
            try:
                from tldw_Server_API.app.core.TTS.audio_utils import AudioProcessor
                return AudioProcessor().decode_base64_audio(voice_reference)
            except Exception as exc:
                raise TTSInvalidVoiceReferenceError(
                    "Echo-TTS voice_reference is not valid base64",
                    provider=self.PROVIDER_KEY,
                    details={"error": str(exc)},
                ) from exc
        raise TTSInvalidVoiceReferenceError(
            "Echo-TTS voice_reference must be bytes or base64 string",
            provider=self.PROVIDER_KEY,
            details={"type": type(voice_reference).__name__},
        )

    async def _prepare_voice_reference(self, voice_bytes: bytes, extras: dict[str, Any]) -> bytes:
        validate_ref = self._coerce_bool(extras.get("validate_reference"), True)
        convert_ref = self._coerce_bool(extras.get("convert_reference"), True)

        from tldw_Server_API.app.core.TTS.audio_utils import AudioProcessor

        processor = AudioProcessor()
        if validate_ref:
            is_valid, error_msg, _ = processor.validate_audio(
                voice_bytes,
                provider=self.PROVIDER_KEY,
                check_duration=True,
                check_quality=False,
            )
            if not is_valid:
                raise TTSInvalidVoiceReferenceError(
                    error_msg or "Echo-TTS voice reference validation failed",
                    provider=self.PROVIDER_KEY,
                )

        if convert_ref:
            voice_bytes = await processor.convert_audio_async(
                voice_bytes,
                target_format="wav",
                target_sample_rate=self.sample_rate,
                provider=self.PROVIDER_KEY,
            )

        return voice_bytes

    def _write_temp_audio(self, audio_bytes: bytes) -> str:
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                prefix="echo_tts_voice_",
            ) as tmp:
                tmp.write(audio_bytes)
                return tmp.name
        except Exception as exc:
            raise TTSInvalidVoiceReferenceError(
                "Failed to prepare Echo-TTS voice reference file",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc

    def _resolve_sequence_length(self, extras: dict[str, Any]) -> int:
        return self._coerce_int(extras.get("sequence_length"), 640)

    def _resolve_chunking_flag(self, extras: dict[str, Any]) -> Optional[bool]:
        for key in ("chunk_text", "enable_chunking", "chunking"):
            if key in extras:
                return self._coerce_bool(extras.get(key), default=False)
        cfg = self.config or {}
        cfg_extras = cfg.get("extra_params") if isinstance(cfg.get("extra_params"), dict) else {}
        for key in ("echo_tts_chunk_text", "chunk_text", "enable_chunking"):
            if key in cfg:
                return self._coerce_bool(cfg.get(key), default=False)
        for key in ("chunk_text", "enable_chunking", "chunking"):
            if key in cfg_extras:
                return self._coerce_bool(cfg_extras.get(key), default=False)
        return None

    def _resolve_chunk_limits(self, extras: dict[str, Any]) -> tuple[int, int]:
        cfg = self.config or {}
        cfg_extras = cfg.get("extra_params") if isinstance(cfg.get("extra_params"), dict) else {}
        max_chars = self._coerce_int(
            extras.get("chunk_max_chars")
            or extras.get("max_chunk_chars")
            or cfg.get("echo_tts_chunk_max_chars")
            or cfg.get("chunk_max_chars")
            or cfg_extras.get("chunk_max_chars"),
            self.MAX_TEXT_LENGTH,
        )
        max_bytes = self._coerce_int(
            extras.get("chunk_max_bytes")
            or extras.get("max_chunk_bytes")
            or cfg.get("echo_tts_chunk_max_bytes")
            or cfg.get("chunk_max_bytes")
            or cfg_extras.get("chunk_max_bytes"),
            self.MAX_TEXT_BYTES,
        )
        max_chars = max(1, min(max_chars, self.MAX_TEXT_LENGTH))
        max_bytes = max(1, min(max_bytes, self.MAX_TEXT_BYTES))
        return max_chars, max_bytes

    def _text_exceeds_limits(self, text: str, max_chars: int, max_bytes: int) -> bool:
        if len(text) > max_chars:
            return True
        try:
            if len(text.encode("utf-8")) > max_bytes:
                return True
        except UnicodeError:
            return True
        return False

    def _fits_text_limits(self, text: str, max_chars: int, max_bytes: int) -> bool:
        if len(text) > max_chars:
            return False
        try:
            return len(text.encode("utf-8")) <= max_bytes
        except UnicodeError:
            return False

    def _split_text_chunks(self, text: str, *, max_chars: int, max_bytes: int) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        if self._fits_text_limits(text, max_chars, max_bytes):
            return [text]

        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            if not sentence:
                continue
            candidate = sentence if not current else f"{current} {sentence}"
            if self._fits_text_limits(candidate, max_chars, max_bytes):
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
                current = ""

            if self._fits_text_limits(sentence, max_chars, max_bytes):
                current = sentence
            else:
                chunks.extend(self._split_text_by_words(sentence, max_chars=max_chars, max_bytes=max_bytes))

        if current:
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_text_by_words(self, text: str, *, max_chars: int, max_bytes: int) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunks: list[str] = []
        current = ""

        for word in words:
            candidate = word if not current else f"{current} {word}"
            if self._fits_text_limits(candidate, max_chars, max_bytes):
                current = candidate
                continue

            if current:
                chunks.append(current.strip())
                current = ""

            if self._fits_text_limits(word, max_chars, max_bytes):
                current = word
            else:
                chunks.extend(self._split_text_by_bytes(word, max_bytes=max_bytes, max_chars=max_chars))

        if current:
            chunks.append(current.strip())
        return chunks

    def _split_text_by_bytes(self, text: str, *, max_bytes: int, max_chars: Optional[int] = None) -> list[str]:
        if max_bytes <= 0:
            max_bytes = self.MAX_TEXT_BYTES
        if max_chars is None or max_chars <= 0:
            max_chars = self.MAX_TEXT_LENGTH
        chunks: list[str] = []
        current_chars: list[str] = []
        current_bytes = 0

        for ch in text:
            try:
                ch_bytes = len(ch.encode("utf-8"))
            except UnicodeError:
                ch_bytes = 1
            if current_chars and (current_bytes + ch_bytes > max_bytes or len(current_chars) + 1 > max_chars):
                chunks.append("".join(current_chars))
                current_chars = []
                current_bytes = 0
            current_chars.append(ch)
            current_bytes += ch_bytes

        if current_chars:
            chunks.append("".join(current_chars))
        return chunks

    def _validate_text_chunks(self, chunks: list[str], *, max_chars: int, max_bytes: int) -> None:
        validator = TTSInputValidator()
        for chunk in chunks:
            validator.validate_text_length(chunk, provider=self.PROVIDER_KEY, max_length=max_chars)
            if len(chunk.encode("utf-8")) > max_bytes:
                raise TTSValidationError(
                    f"Text byte length exceeds chunk limit of {max_bytes} bytes",
                    provider=self.PROVIDER_KEY,
                )

    def _coerce_bool(self, value: Any, default: bool) -> bool:
        try:
            return parse_bool(value, default=default)
        except (TypeError, ValueError):
            return default

    def _coerce_int(self, value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _coerce_optional_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _coerce_optional_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

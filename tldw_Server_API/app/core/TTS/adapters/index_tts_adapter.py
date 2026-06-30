# index_tts_adapter.py
# Description: Adapter for IndexTTS2 local text-to-speech engine
#
# Imports
import asyncio
import base64
import contextlib
import tempfile
import wave
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Optional

#
# Third-party Imports
import numpy as np
from loguru import logger

from ..audio_converter import AudioConverter
from ..streaming_audio_writer import AudioNormalizer, StreamingAudioWriter
from ..tts_exceptions import (
    TTSGenerationError,
    TTSModelLoadError,
    TTSModelNotFoundError,
    TTSProviderInitializationError,
    TTSProviderNotConfiguredError,
    TTSUnsupportedFormatError,
    TTSValidationError,
    TTSVoiceCloningError,
)
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
# IndexTTS2 Adapter Implementation


def _safe_exception_label(exc: BaseException) -> str:
    """Return a non-sensitive exception identifier for logs."""
    return type(exc).__name__


class IndexTTS2Adapter(TTSAdapter):
    """Adapter integrating the IndexTTS2 autoregressive TTS engine."""

    PROVIDER_KEY = "index_tts"
    DEFAULT_SAMPLE_RATE = 22050  # Native IndexTTS2 output sample rate (Hz)
    STREAM_SAMPLE_RATE = 22050
    SUPPORTED_FORMATS = {AudioFormat.MP3, AudioFormat.WAV}

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config=config)

        # Resolve configuration with sensible defaults
        cfg = config or {}
        self.model_dir = Path(
            cfg.get("index_tts_model_dir")
            or cfg.get("model_dir")
            or cfg.get("index_tts_model_path")
            or "checkpoints"
        )
        self.cfg_path = Path(
            cfg.get("index_tts_cfg_path")
            or cfg.get("cfg_path")
            or self.model_dir / "config.yaml"
        )
        self.device = cfg.get("index_tts_device") or cfg.get("device")
        self.use_fp16 = parse_bool(cfg.get("index_tts_use_fp16", cfg.get("use_fp16", False)), default=False)
        self.use_cuda_kernel = parse_bool(
            cfg.get("index_tts_use_cuda_kernel", cfg.get("use_cuda_kernel", False)), default=False
        )
        self.use_deepspeed = parse_bool(
            cfg.get("index_tts_use_deepspeed", cfg.get("use_deepspeed", False)), default=False
        )
        self.quick_streaming_tokens = int(
            cfg.get("index_tts_quick_streaming_tokens", 0)
        )
        self.interval_silence = int(cfg.get("index_tts_interval_silence", 200))
        self.verbose = parse_bool(cfg.get("index_tts_verbose", False), default=False)
        self.max_text_tokens_per_segment = int(
            cfg.get("index_tts_max_text_tokens_per_segment", 120)
        )
        self.more_segment_before = int(cfg.get("index_tts_more_segment_before", 0))
        self.output_sample_rate = int(cfg.get("sample_rate", self.DEFAULT_SAMPLE_RATE))

        # Engine instance populated during initialization
        self._engine = None

        if not self.cfg_path.exists() or not self.model_dir.exists():
            logger.warning(
                'IndexTTS2 configuration missing: cfg={} model_dir={}',
                self.cfg_path,
                self.model_dir,
            )
            self._status = ProviderStatus.NOT_CONFIGURED

    # ---------------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------------
    async def initialize(self) -> bool:
        """Load the IndexTTS2 engine and warm-up metadata."""
        if self._initialized:
            return True

        if not self.cfg_path.exists():
            raise TTSModelNotFoundError(
                f"IndexTTS2 config not found at {self.cfg_path}",
                provider=self.provider_name,
                details={"cfg_path": str(self.cfg_path)},
            )

        if not self.model_dir.exists():
            raise TTSModelNotFoundError(
                f"IndexTTS2 model directory not found at {self.model_dir}",
                provider=self.provider_name,
                details={"model_dir": str(self.model_dir)},
            )

        try:
            self._status = ProviderStatus.INITIALIZING
            engine = await asyncio.to_thread(self._create_engine)
            self._engine = engine
            try:
                from ..tts_resource_manager import get_resource_manager
                resource_manager = await get_resource_manager()
                register_result = resource_manager.register_model(
                    provider=self.provider_name.lower(),
                    model_instance=engine,
                    cleanup_callback=lambda: setattr(self, "_engine", None),
                    model_key=str(self.model_dir),
                )
                if asyncio.iscoroutine(register_result):
                    await register_result
            except Exception as registration_error:
                logger.debug(
                    "IndexTTS provider registration failed; continuing; exception_type={}",
                    _safe_exception_label(registration_error),
                )

            # Cache capabilities for later use
            self._capabilities = await self.get_capabilities()
            self._initialized = True
            self._status = ProviderStatus.AVAILABLE
            logger.info("IndexTTS2 adapter initialized (device={})", self.device or "auto")
            return True

        except TTSModelNotFoundError:
            self._status = ProviderStatus.ERROR
            raise
        except ImportError as exc:
            self._status = ProviderStatus.ERROR
            raise TTSModelLoadError(
                "IndexTTS2 dependencies missing",
                provider=self.provider_name,
                details={"error": str(exc)},
            ) from exc
        except Exception as exc:
            self._status = ProviderStatus.ERROR
            raise TTSProviderInitializationError(
                "Failed to initialize IndexTTS2",
                provider=self.provider_name,
                details={"error": str(exc)},
            ) from exc

    def _create_engine(self):
        """Instantiate the IndexTTS2 engine (runs in executor)."""
        from indextts.infer_v2 import IndexTTS2  # Local import to avoid hard dependency at module load

        kwargs: dict[str, Any] = {
            "cfg_path": str(self.cfg_path),
            "model_dir": str(self.model_dir),
            "use_fp16": self.use_fp16,
            "use_deepspeed": self.use_deepspeed,
        }
        # Only pass optional flags when explicitly configured
        if self.device:
            kwargs["device"] = self.device
        if self.use_cuda_kernel is not None:
            kwargs["use_cuda_kernel"] = self.use_cuda_kernel

        logger.info(
            'Loading IndexTTS2 engine cfg={} model_dir={} device={} fp16={} deepspeed={} cuda_kernel={}',
            kwargs.get("cfg_path"),
            kwargs.get("model_dir"),
            kwargs.get("device"),
            kwargs.get("use_fp16"),
            kwargs.get("use_deepspeed"),
            kwargs.get("use_cuda_kernel"),
        )
        return IndexTTS2(**kwargs)

    # ---------------------------------------------------------------------------------
    # Capabilities
    # ---------------------------------------------------------------------------------
    async def get_capabilities(self) -> TTSCapabilities:
        """Return declared IndexTTS2 capabilities."""
        voices = []  # IndexTTS2 relies on voice cloning from reference audio
        return TTSCapabilities(
            provider_name="IndexTTS2",
            supported_languages={"en", "zh"},  # Primary datasets; extend as docs mature
            supported_voices=voices,
            supported_formats=self.SUPPORTED_FORMATS,
            max_text_length=4000,
            supports_streaming=True,
            supports_voice_cloning=True,
            supports_emotion_control=True,
            supports_speech_rate=False,
            supports_pitch_control=False,
            supports_volume_control=False,
            supports_ssml=False,
            supports_phonemes=True,
            supports_multi_speaker=False,
            supports_background_audio=False,
            latency_ms=1500,
            sample_rate=self.output_sample_rate,
            default_format=AudioFormat.MP3,
        )

    # ---------------------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------------------
    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate audio using IndexTTS2 (streaming or non-streaming)."""
        if not await self.ensure_initialized():
            raise TTSProviderNotConfiguredError(
                "IndexTTS2 adapter not initialized",
                provider=self.provider_name,
            )

        if request.format not in self.SUPPORTED_FORMATS:
            raise TTSUnsupportedFormatError(
                f"Format {request.format.value} not supported by IndexTTS2",
                provider=self.provider_name,
            )

        if not request.voice_reference:
            raise TTSValidationError(
                "IndexTTS2 requires voice_reference audio for cloning",
                provider=self.provider_name,
            )

        extras = dict(request.extra_params or {})

        temp_paths: list[str] = []
        try:
            (
                speaker_path,
                emo_path,
                infer_kwargs,
                generation_kwargs,
                temp_paths,
            ) = await self._prepare_generation_inputs(request, extras)

            metadata = {
                "model_dir": str(self.model_dir),
                "cfg_path": str(self.cfg_path),
            }

            if request.stream:
                stream = self._stream_audio_index_tts(
                    request,
                    speaker_path,
                    infer_kwargs,
                    generation_kwargs,
                    temp_paths,
                )
                return TTSResponse(
                    audio_stream=stream,
                    format=request.format,
                    sample_rate=self.output_sample_rate,
                    channels=1,
                    text_processed=request.text,
                    voice_used=request.voice,
                    provider=self.provider_name,
                    metadata=metadata,
                )

            result = await asyncio.to_thread(
                self._engine.infer,
                speaker_path,
                request.text,
                None,
                **infer_kwargs,
                **generation_kwargs,
            )

            audio_bytes, duration, sample_rate = await self._normalize_audio(
                result,
                request.format,
            )

            return TTSResponse(
                audio_data=audio_bytes,
                format=request.format,
                sample_rate=sample_rate,
                channels=1,
                duration_seconds=duration,
                text_processed=request.text,
                voice_used=request.voice,
                provider=self.provider_name,
                metadata=metadata,
            )

        except (TTSValidationError, TTSUnsupportedFormatError):
            raise
        except TTSVoiceCloningError:
            raise
        except Exception as exc:
            logger.error(
                "IndexTTS2 generation failed; exception_type={}",
                _safe_exception_label(exc),
            )
            raise TTSGenerationError(
                "IndexTTS2 generation failed",
                provider=self.provider_name,
                details={"error": str(exc)},
            ) from exc
        finally:
            if not request.stream:
                self._cleanup_temp_paths(temp_paths)

    # ---------------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------------
    async def _write_temp_audio(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """Persist raw audio bytes to a temporary file."""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                return tmp.name
        except Exception as exc:
            raise TTSVoiceCloningError(
                "Failed to prepare voice reference audio",
                provider=self.provider_name,
                details={"error": str(exc)},
            ) from exc

    async def _prepare_emotion_reference(self, extras: dict[str, Any]) -> Optional[str]:
        """Materialize optional emotion reference audio."""
        emo_audio_reference = extras.get("emo_audio_reference")
        emo_audio_path = extras.get("emo_audio_path")

        if emo_audio_path:
            return str(emo_audio_path)

        if not emo_audio_reference:
            return None

        try:
            if isinstance(emo_audio_reference, str):
                emo_bytes = base64.b64decode(emo_audio_reference)
            elif isinstance(emo_audio_reference, bytes):
                emo_bytes = emo_audio_reference
            else:
                raise ValueError("Unsupported emo_audio_reference type")
        except Exception as exc:
            raise TTSValidationError(
                f"Invalid emo_audio_reference payload: {exc}",
                provider=self.provider_name,
            ) from exc

        return await self._write_temp_audio(emo_bytes, suffix=".wav")

    async def _prepare_generation_inputs(
        self,
        request: TTSRequest,
        extras: dict[str, Any],
    ) -> tuple[str, Optional[str], dict[str, Any], dict[str, Any], list[str]]:
        """Prepare temp files and inference kwargs for generation."""
        temp_paths: list[str] = []
        speaker_path = await self._write_temp_audio(request.voice_reference, suffix=".wav")
        temp_paths.append(speaker_path)

        emo_path = await self._prepare_emotion_reference(extras)
        if emo_path:
            temp_paths.append(emo_path)

        infer_kwargs, generation_kwargs = self._build_infer_kwargs(
            extras,
            emo_path,
            stream=request.stream,
        )

        return speaker_path, emo_path, infer_kwargs, generation_kwargs, temp_paths

    def _cleanup_temp_paths(self, paths: list[str]) -> None:
        """Remove temporary files created during inference."""
        for path in paths:
            with contextlib.suppress(Exception):
                Path(path).unlink(missing_ok=True)

    def _stream_audio_index_tts(
        self,
        request: TTSRequest,
        speaker_path: str,
        infer_kwargs: dict[str, Any],
        generation_kwargs: dict[str, Any],
        temp_paths: list[str],
    ) -> AsyncGenerator[bytes, None]:
        """Return async generator that streams IndexTTS2 audio."""
        stream_sample_rate = self.output_sample_rate or self.STREAM_SAMPLE_RATE
        if stream_sample_rate <= 0:
            stream_sample_rate = self.STREAM_SAMPLE_RATE

        writer = StreamingAudioWriter(
            format=request.format.value,
            sample_rate=stream_sample_rate,
            channels=1,
        )
        normalizer = AudioNormalizer()

        async def stream() -> AsyncGenerator[bytes, None]:
            sentinel = object()
            loop = asyncio.get_running_loop()
            iterator = await asyncio.to_thread(
                self._engine.infer,
                speaker_path,
                request.text,
                None,
                **infer_kwargs,
                **generation_kwargs,
            )
            try:
                while True:
                    chunk = await loop.run_in_executor(
                        None, lambda: next(iterator, sentinel)
                    )
                    if chunk is sentinel:
                        break
                    data = self._convert_stream_chunk(
                        chunk,
                        normalizer,
                        writer,
                        stream_sample_rate,
                    )
                    if data:
                        yield data

                final_bytes = writer.write_chunk(finalize=True)
                if final_bytes:
                    yield final_bytes
            except Exception as exc:
                logger.error(
                    "IndexTTS2 streaming failed; exception_type={}",
                    _safe_exception_label(exc),
                )
                raise TTSGenerationError(
                    "IndexTTS2 streaming failed",
                    provider=self.provider_name,
                    details={"error": str(exc)},
                ) from exc
            finally:
                writer.close()
                self._cleanup_temp_paths(temp_paths)

        return stream()

    def _convert_stream_chunk(
        self,
        chunk: Any,
        normalizer: AudioNormalizer,
        writer: StreamingAudioWriter,
        target_sample_rate: int,
    ) -> bytes:
        """Convert a raw IndexTTS2 tensor chunk into encoded bytes."""
        if chunk is None:
            return b""

        try:
            np_chunk = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
        except Exception as exc:
            logger.warning(
                "IndexTTS2 streaming chunk conversion error; exception_type={}",
                _safe_exception_label(exc),
            )
            return b""

        np_chunk = np.squeeze(np_chunk)
        if np_chunk.size == 0:
            return b""

        audio_float = np_chunk.astype(np.float32) / 32767.0

        if target_sample_rate != self.STREAM_SAMPLE_RATE:
            try:
                import torch  # type: ignore
                import torchaudio  # type: ignore

                tensor = torch.from_numpy(audio_float).unsqueeze(0)
                resampled = torchaudio.functional.resample(
                    tensor,
                    self.STREAM_SAMPLE_RATE,
                    target_sample_rate,
                )
                audio_float = resampled.squeeze(0).cpu().numpy()
            except Exception as exc:
                logger.warning(
                    'IndexTTS2 streaming using numpy resample fallback; exception_type={}',
                    _safe_exception_label(exc),
                )
                try:
                    orig_len = audio_float.shape[0]
                    target_len = max(
                        int(round(orig_len * target_sample_rate / self.STREAM_SAMPLE_RATE)),
                        1,
                    )
                    orig_idx = np.linspace(0.0, 1.0, orig_len, dtype=np.float32)
                    target_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
                    audio_float = np.interp(target_idx, orig_idx, audio_float).astype(np.float32)
                except Exception as interp_exc:
                    logger.error(
                        'IndexTTS2 streaming interpolation failed; exception_type={}',
                        _safe_exception_label(interp_exc),
                    )
                    # Fall back to native sample rate
                    target_sample_rate = self.STREAM_SAMPLE_RATE

        audio_int16 = normalizer.normalize(audio_float, target_dtype=np.int16)
        return writer.write_chunk(audio_int16)

    def _build_infer_kwargs(
        self,
        extras: dict[str, Any],
        emo_path: Optional[str],
        stream: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Prepare keyword arguments for IndexTTS2.infer."""
        infer_kwargs: dict[str, Any] = {
            "emo_audio_prompt": emo_path,
            "emo_alpha": float(extras.get("emo_alpha", 1.0)),
            "emo_vector": extras.get("emo_vector"),
            "use_emo_text": bool(extras.get("use_emo_text", False)),
            "emo_text": extras.get("emo_text"),
            "use_random": bool(extras.get("use_random", False)),
            "interval_silence": int(extras.get("interval_silence", self.interval_silence)),
            "verbose": bool(extras.get("verbose", self.verbose)),
            "max_text_tokens_per_segment": int(
                extras.get("max_text_tokens_per_segment", self.max_text_tokens_per_segment)
            ),
            "stream_return": stream,
            "more_segment_before": int(
                extras.get("more_segment_before", self.more_segment_before)
            ),
        }

        if self.quick_streaming_tokens:
            infer_kwargs["quick_streaming_tokens"] = self.quick_streaming_tokens

        generation_kwargs = extras.get("generation") or {}
        if not isinstance(generation_kwargs, dict):
            generation_kwargs = {}

        return infer_kwargs, generation_kwargs

    async def _normalize_audio(
        self,
        result: Any,
        target_format: AudioFormat,
    ) -> tuple[bytes, float, int]:
        """
        Convert IndexTTS2 inference results into requested audio format.

        Returns:
            Tuple of (audio bytes, duration_seconds, sample_rate)
        """
        sampling_rate: Optional[int] = None
        audio_array: Optional[np.ndarray] = None

        if isinstance(result, tuple) and len(result) == 2:
            sampling_rate, audio_data = result
            audio_array = np.asarray(audio_data, dtype=np.int16)
        elif isinstance(result, str):
            # Engine already saved to disk; read and resample/convert if needed
            path = Path(result)
            if not path.exists():
                raise TTSGenerationError(
                    "IndexTTS2 returned invalid output path",
                    provider=self.provider_name,
                )
            data = path.read_bytes()
            # Assume WAV file when engine returns a path; hand off to converter below
            return await self._convert_bytes_via_ffmpeg(
                data,
                original_sample_rate=None,
                target_format=target_format,
            )
        else:
            raise TTSGenerationError(
                "IndexTTS2 returned unexpected payload",
                provider=self.provider_name,
                details={"type": str(type(result))},
            )

        if audio_array is None or sampling_rate is None:
            raise TTSGenerationError(
                "IndexTTS2 produced no audio data",
                provider=self.provider_name,
            )

        # Ensure two-dimensional shape (samples, channels)
        if audio_array.ndim == 1:
            audio_array = audio_array.reshape(-1, 1)
        elif audio_array.ndim > 2:
            audio_array = audio_array.reshape(audio_array.shape[0], -1)

        duration = len(audio_array) / float(sampling_rate)

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "index_tts_raw.wav"
            self._write_wave_file(raw_path, audio_array, sampling_rate)

            # Resample to adapter sample rate if needed
            resampled_path = raw_path
            resampled_rate = sampling_rate
            if sampling_rate != self.output_sample_rate:
                resampled_path = Path(tmpdir) / "index_tts_resampled.wav"
                success = await AudioConverter.convert_to_wav(
                    raw_path,
                    resampled_path,
                    sample_rate=self.output_sample_rate,
                    channels=1,
                )
                if success:
                    resampled_rate = self.output_sample_rate
                else:
                    resampled_path = raw_path
                    resampled_rate = sampling_rate

            final_bytes = await self._encode_audio(resampled_path, resampled_rate, target_format)
            return final_bytes, duration, resampled_rate

    async def _convert_bytes_via_ffmpeg(
        self,
        audio_bytes: bytes,
        original_sample_rate: Optional[int],
        target_format: AudioFormat,
    ) -> tuple[bytes, float, int]:
        """Handle conversion when IndexTTS2 writes directly to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "index_tts_input.wav"
            input_path.write_bytes(audio_bytes)
            duration, sample_rate = await self._probe_duration(input_path, original_sample_rate)
            final_bytes = await self._encode_audio(input_path, sample_rate, target_format)
            return final_bytes, duration, sample_rate

    async def _encode_audio(
        self,
        wav_path: Path,
        sample_rate: int,
        target_format: AudioFormat,
    ) -> bytes:
        """Encode audio file into requested format."""
        if target_format == AudioFormat.WAV:
            return wav_path.read_bytes()

        output_path = wav_path.parent / "index_tts_output"
        success = await AudioConverter.convert_format(
            wav_path,
            output_path,
            target_format.value,
            sample_rate=sample_rate,
            channels=1,
        )
        if success:
            target_path = output_path.with_suffix(f".{target_format.value}")
            data = target_path.read_bytes()
            with contextlib.suppress(Exception):
                target_path.unlink(missing_ok=True)
            return data

        # Fall back to WAV bytes if conversion fails
        logger.warning("Falling back to WAV output for IndexTTS2 (format conversion failed)")
        return wav_path.read_bytes()

    async def _probe_duration(self, wav_path: Path, sample_rate_hint: Optional[int]) -> tuple[float, int]:
        """Estimate audio duration for metadata."""
        if not wav_path.exists():
            return 0.0, sample_rate_hint or self.output_sample_rate

        try:
            with wave.open(str(wav_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
                return duration, rate
        except Exception:
            logger.debug("Failed to inspect WAV header for duration; using hints")
            return 0.0, sample_rate_hint or self.output_sample_rate

    def _write_wave_file(self, path: Path, audio: np.ndarray, sample_rate: int) -> None:
        """Persist numpy int16 audio array to WAV."""
        mono = np.asarray(audio, dtype=np.int16)
        if mono.ndim == 2:
            mono = mono.mean(axis=1) if mono.shape[1] > 1 else mono[:, 0]

        mono = np.asarray(mono, dtype=np.int16)

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(mono.tobytes())

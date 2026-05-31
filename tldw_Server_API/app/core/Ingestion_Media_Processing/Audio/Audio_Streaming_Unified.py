# Audio_Streaming_Unified.py
#########################################
# Unified Real-time Streaming Transcription for All Nemo Models
# This module provides WebSocket-based real-time transcription using all Nemo models
# with support for Parakeet (all variants) and Canary (multilingual).
#
####################
# Function List
#
# 1. BaseStreamingTranscriber - Abstract base class for streaming transcription
# 2. ParakeetStreamingTranscriber - Parakeet-specific implementation
# 3. CanaryStreamingTranscriber - Canary-specific implementation
# 4. UnifiedStreamingTranscriber - Factory and unified interface
# 5. handle_unified_websocket - Unified WebSocket handler
#
####################

import asyncio
import base64
import copy
import importlib
import json
import math
import os
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4

import numpy as np
from fastapi import WebSocketDisconnect
from loguru import logger

from tldw_Server_API.app.core.Audio.streaming_exceptions import QuotaExceeded
from tldw_Server_API.app.core.Streaming.streams import WebSocketStream
from tldw_Server_API.app.core.testing import is_truthy

from .Audio_Streaming_Parakeet import AudioBuffer, StreamingConfig
from .ws_control_protocol import WSControlProtocolConfig, WSControlSession

# Import existing implementations
from .Audio_Transcription_Nemo import (
    load_canary_model,
    load_parakeet_model,
    transcribe_with_canary,
    transcribe_with_parakeet,
)
try:  # pragma: no cover - optional on non-macOS or minimal test envs
    from .Audio_Transcription_Parakeet_MLX import (
        create_parakeet_mlx_streaming_session,
        transcribe_with_parakeet_mlx,
    )
except ImportError as exc:  # pragma: no cover - provide safe fallbacks
    logger.debug("Parakeet MLX import unavailable; using no-op fallbacks: {}", exc)

    def create_parakeet_mlx_streaming_session(*_args: Any, **_kwargs: Any) -> None:  # type: ignore[override]
        """Return no streaming session when the MLX backend is unavailable."""
        return None

    def transcribe_with_parakeet_mlx(*_args: Any, **_kwargs: Any) -> str:  # type: ignore[override]
        """Return an empty transcription when the MLX backend is unavailable."""
        return ""

from .model_utils import normalize_model_and_variant

_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    NameError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    WebSocketDisconnect,
)

# Expose config loader for tests to monkeypatch at module scope
try:  # pragma: no cover - import may fail in minimal test environments
    from tldw_Server_API.app.core.config import load_comprehensive_config as load_comprehensive_config  # type: ignore
except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:  # pragma: no cover
    def load_comprehensive_config():  # type: ignore
        return None


def _safe_temp_subdir(raw: Optional[str]) -> Optional[Path]:
    """Return a sanitized temp-only subdirectory path derived from untrusted input."""
    if not raw:
        return None
    try:
        raw_str = str(raw).strip()
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        return None
    if not raw_str:
        return None
    name = Path(raw_str).name
    if name in {"", ".", ".."}:
        return None
    safe = "".join(
        ch if (ch.isascii() and (ch.isalnum() or ch in "._-")) else "_"
        for ch in name
    ).strip("._-")
    if not safe:
        return None
    if len(safe) > 200:
        safe = safe[:200].rstrip("._-")
    if not safe:
        return None
    base = Path(tempfile.gettempdir()) / "tldw_diarization"
    return base / safe


def _get_ws_control_protocol_config() -> WSControlProtocolConfig:
    """Load bounded WS control settings from the canonical STT config surface."""
    try:
        from tldw_Server_API.app.core.config import get_stt_config

        stt_cfg = get_stt_config() or {}
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        stt_cfg = {}

    def _float_setting(key: str, default: float) -> float:
        try:
            value = float(stt_cfg.get(key, default))
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
            return default
        return value if value >= 0 else default

    return WSControlProtocolConfig(
        ws_control_v2_enabled=bool(stt_cfg.get("ws_control_v2_enabled", False)),
        paused_audio_queue_cap_seconds=_float_setting("paused_audio_queue_cap_seconds", 2.0),
        overflow_warning_interval_seconds=_float_setting("overflow_warning_interval_seconds", 5.0),
    )


def _estimate_audio_seconds(audio_bytes: bytes, sample_rate: int) -> float:
    try:
        from tldw_Server_API.app.core.Usage.audio_quota import bytes_to_seconds as _bytes_to_seconds

        return float(_bytes_to_seconds(len(audio_bytes), int(sample_rate or 16000)))
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        return float(len(audio_bytes)) / float(4 * max(1, int(sample_rate or 16000)))


def _drop_oldest_buffered_audio(
    paused_audio_chunks: "deque[tuple[bytes, float]]",
    dropped_seconds: float,
) -> None:
    remaining = max(0.0, float(dropped_seconds))
    bytes_per_sample = 4  # float32 PCM
    while remaining > 0 and paused_audio_chunks:
        chunk_bytes, chunk_seconds = paused_audio_chunks[0]
        if chunk_seconds <= remaining:
            paused_audio_chunks.popleft()
            remaining -= chunk_seconds
            continue
        if chunk_seconds <= 0:
            paused_audio_chunks.popleft()
            continue
        chunk_len = len(chunk_bytes)
        if chunk_len <= 0:
            paused_audio_chunks.popleft()
            continue

        trim_ratio = min(1.0, remaining / chunk_seconds)
        target_trim_bytes = max(float(bytes_per_sample), float(chunk_len) * trim_ratio)
        trim_bytes = min(
            chunk_len,
            int(math.ceil(target_trim_bytes / float(bytes_per_sample))) * bytes_per_sample,
        )
        if trim_bytes >= chunk_len:
            paused_audio_chunks.popleft()
            remaining -= chunk_seconds
            continue

        dropped_chunk_seconds = chunk_seconds * (trim_bytes / float(chunk_len))
        paused_audio_chunks[0] = (
            chunk_bytes[trim_bytes:],
            max(0.0, chunk_seconds - dropped_chunk_seconds),
        )
        remaining = max(0.0, remaining - dropped_chunk_seconds)


_VALID_VAD_STATUSES = {"enabled", "disabled", "fail_open"}
_VALID_DIARIZATION_STATUSES = {"enabled", "disabled", "unavailable"}
_VALID_DIARIZATION_DETAIL_CODES = {
    "init_unavailable",
    "init_failed",
    "persist_degraded",
    "persist_disabled",
    "finalize_failed",
}
_DIARIZATION_SUMMARY_MAX_LENGTH = 160


def _normalize_stream_diagnostic_status(value: Any, *, allowed: set[str], default: str) -> str:
    try:
        normalized = str(value or "").strip().lower()
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        normalized = ""
    return normalized if normalized in allowed else default


def _sanitize_diarization_details(details: Any) -> dict[str, Any] | None:
    if not isinstance(details, dict):
        return None
    code = _normalize_stream_diagnostic_status(
        details.get("code"),
        allowed=_VALID_DIARIZATION_DETAIL_CODES,
        default="",
    )
    if not code:
        return None
    payload: dict[str, Any] = {"code": code}
    summary = details.get("summary")
    if summary is not None:
        try:
            summary_text = str(summary).strip()
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
            summary_text = ""
        if summary_text:
            payload["summary"] = summary_text[:_DIARIZATION_SUMMARY_MAX_LENGTH]
    return payload


def _build_transcript_diagnostics(
    *,
    auto_commit: bool,
    vad_status: Any,
    diarization_status: Any,
    diarization_details: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "auto_commit": bool(auto_commit),
        "vad_status": _normalize_stream_diagnostic_status(
            vad_status,
            allowed=_VALID_VAD_STATUSES,
            default="disabled",
        ),
        "diarization_status": _normalize_stream_diagnostic_status(
            diarization_status,
            allowed=_VALID_DIARIZATION_STATUSES,
            default="unavailable",
        ),
    }
    sanitized_details = _sanitize_diarization_details(diarization_details)
    if sanitized_details is not None:
        payload["diarization_details"] = sanitized_details
    return payload

# Expose get_whisper_model at module scope so tests can monkeypatch it.
# Keep this lazy: importing Audio_Transcription_Lib at module import time can
# force heavy dependency imports (e.g., torch/faster-whisper) in test contexts.
get_whisper_model = None  # type: ignore[assignment]
_WHISPER_COMPUTE_TYPE_OVERRIDE = ""  # type: ignore[assignment]


def _resample_audio_if_needed(audio, sample_rate, target_sr=16000):  # type: ignore
    """Resample audio while avoiding heavy dependency imports during tests."""
    if sample_rate == target_sr:
        return np.asarray(audio, dtype=np.float32)

    lib_mod = sys.modules.get(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib"
    )
    if lib_mod is not None:
        impl = getattr(lib_mod, "_resample_audio_if_needed", None)
        if callable(impl):
            try:
                return impl(audio, sample_rate, target_sr=target_sr)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                pass

    try:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return arr
        ratio = float(target_sr) / float(sample_rate)
        new_len = max(1, round(arr.shape[0] * ratio))
        x_old = np.linspace(0.0, 1.0, num=arr.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
        return np.interp(x_new, x_old, arr).astype(np.float32, copy=False)
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        return audio

# Optional torch/torchaudio/Nemo modules for Parakeet RNNT streaming.
# Keep these lazy so importing this module in unit tests never forces native
# torch/faster-whisper loads.
torch = None  # type: ignore[assignment]
torchaudio = None  # type: ignore[assignment]
nemo_asr = None  # type: ignore[assignment]
RNNTDecodingConfig = None  # type: ignore[assignment]
batched_hyps_to_hypotheses = None  # type: ignore[assignment]
ContextSize = StreamingBatchedAudioBuffer = None  # type: ignore[assignment]
_TORCH_IMPORT_ATTEMPTED = False
_TORCHAUDIO_IMPORT_ATTEMPTED = False
_NEMO_RNNT_IMPORT_ATTEMPTED = False


def _get_torch_module():
    global torch, _TORCH_IMPORT_ATTEMPTED
    if torch is not None:
        return torch
    if _TORCH_IMPORT_ATTEMPTED:
        return None
    _TORCH_IMPORT_ATTEMPTED = True
    try:
        torch = importlib.import_module("torch")  # type: ignore[assignment]
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        torch = None  # type: ignore[assignment]
    return torch


def _get_torchaudio_module():
    global torchaudio, _TORCHAUDIO_IMPORT_ATTEMPTED
    if torchaudio is not None:
        return torchaudio
    if _TORCHAUDIO_IMPORT_ATTEMPTED:
        return None
    _TORCHAUDIO_IMPORT_ATTEMPTED = True
    try:
        torchaudio = importlib.import_module("torchaudio")  # type: ignore[assignment]
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        torchaudio = None  # type: ignore[assignment]
    return torchaudio


def _ensure_nemo_rnnt_deps() -> None:
    global nemo_asr
    global RNNTDecodingConfig
    global batched_hyps_to_hypotheses
    global ContextSize
    global StreamingBatchedAudioBuffer
    global _NEMO_RNNT_IMPORT_ATTEMPTED

    if (
        nemo_asr is not None
        and RNNTDecodingConfig is not None
        and batched_hyps_to_hypotheses is not None
        and ContextSize is not None
        and StreamingBatchedAudioBuffer is not None
    ):
        return
    if _NEMO_RNNT_IMPORT_ATTEMPTED:
        return
    _NEMO_RNNT_IMPORT_ATTEMPTED = True

    try:
        nemo_asr = importlib.import_module("nemo.collections.asr")  # type: ignore[assignment]
        _rnnt = importlib.import_module(
            "nemo.collections.asr.parts.submodules.rnnt_decoding"
        )
        _rnnt_utils = importlib.import_module(
            "nemo.collections.asr.parts.utils.rnnt_utils"
        )
        _stream_utils = importlib.import_module(
            "nemo.collections.asr.parts.utils.streaming_utils"
        )
        RNNTDecodingConfig = getattr(_rnnt, "RNNTDecodingConfig", None)  # type: ignore[assignment]
        batched_hyps_to_hypotheses = getattr(
            _rnnt_utils, "batched_hyps_to_hypotheses", None
        )  # type: ignore[assignment]
        ContextSize = getattr(_stream_utils, "ContextSize", None)  # type: ignore[assignment]
        StreamingBatchedAudioBuffer = getattr(
            _stream_utils, "StreamingBatchedAudioBuffer", None
        )  # type: ignore[assignment]
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        nemo_asr = None  # type: ignore[assignment]
        RNNTDecodingConfig = None  # type: ignore[assignment]
        batched_hyps_to_hypotheses = None  # type: ignore[assignment]
        ContextSize = StreamingBatchedAudioBuffer = None  # type: ignore[assignment]


def _is_transcription_error_message(text: str) -> bool:
    """Resolve STT sentinel checks without importing heavy modules."""
    lib_mod = sys.modules.get(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib"
    )
    if lib_mod is None:
        return False
    _impl = getattr(lib_mod, "is_transcription_error_message", None)
    if not callable(_impl):
        return False
    try:
        return bool(_impl(text))
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        return False

import contextlib

from .Audio_Streaming_Insights import LiveInsightSettings, LiveMeetingInsights

try:
    from .Diarization_Lib import DiarizationError, DiarizationService
except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - optional dependency probing
    DiarizationService = None  # type: ignore

    class DiarizationError(Exception):  # type: ignore
        """Fallback diarization error when service is unavailable."""
        pass

# Optional: Parakeet Core adapter
try:  # pragma: no cover - optional integration path
    from .Parakeet_Core_Streaming.config import StreamingConfig as _CoreConfig
    from .Parakeet_Core_Streaming.transcriber import ParakeetCoreTranscriber as _CoreTranscriber

    class _ParakeetCoreAdapter:
        """Adapter exposing the same interface expected by handle_unified_websocket.

        Delegates to the self-contained Parakeet core streaming transcriber so this
        unified path can benefit from the improved buffering/metadata and variant handling.
        """

        def __init__(self, config: 'UnifiedStreamingConfig') -> None:
            """
            Initialize the adapter with the provided unified streaming configuration.

            Stores the given UnifiedStreamingConfig and prepares an internal placeholder for the underlying core transcriber (left as None until initialization).

            Parameters:
                config (UnifiedStreamingConfig): Configuration that drives model selection, runtime options, and behavior for the underlying transcriber.
            """
            self._uconf = config
            self._core = None  # type: ignore

        def initialize(self) -> None:
            # Map Unified config to core config
            """
            Initialize the core Parakeet transcriber adapter by mapping the unified streaming configuration to the core transcriber config and creating the core transcriber instance.

            Creates a _CoreConfig from the adapter's unified config, instantiates _CoreTranscriber with that config, and validates that the chosen model variant is available. Raises a RuntimeError with message "parakeet_variant_unavailable: <variant>" when the core transcriber does not expose the expected decode function, to trigger higher-level fallback logic.
            """
            c = _CoreConfig(
                model=self._uconf.model,
                model_variant=self._uconf.model_variant,
                sample_rate=self._uconf.sample_rate,
                chunk_duration=self._uconf.chunk_duration,
                overlap_duration=self._uconf.overlap_duration,
                max_buffer_duration=self._uconf.max_buffer_duration,
                enable_partial=self._uconf.enable_partial,
                partial_interval=self._uconf.partial_interval,
                min_partial_duration=self._uconf.min_partial_duration,
                language=self._uconf.language,
            )
            self._core = _CoreTranscriber(config=c)
            # Ensure chosen variant is available; if not, raise to trigger fallback logic
            try:
                # Prefer explicit check against the core helper so tests can monkeypatch it
                from .Parakeet_Core_Streaming import transcriber as _core_tx  # type: ignore
                _fn = getattr(_core_tx, "_variant_decode_fn", None)
                if callable(_fn):
                    available = _fn(c.model, c.model_variant)
                    if available is None:
                        raise RuntimeError(f"parakeet_variant_unavailable: {c.model_variant}")
                # Fallback: inspect instantiated core transcriber
                if getattr(self._core, "decode_fn", None) is None:
                    raise RuntimeError(f"parakeet_variant_unavailable: {c.model_variant}")
            except RuntimeError:
                # Re-raise variant unavailability to be caught by higher-level handler
                raise
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                # If validation fails unexpectedly, defer to runtime; do not block initialization
                # The streaming path will still handle decode failures gracefully.
                pass
            try:
                decode_fn = getattr(self._core, 'decode_fn', None)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                decode_fn = None
            if decode_fn is None:
                raise RuntimeError(f"parakeet_variant_unavailable: {self._uconf.model_variant}")
        async def process_audio_chunk(self, audio_data: bytes):  # -> Optional[Dict[str, Any]]
            """
            Forward an incoming audio chunk to the underlying core transcriber and return any resulting transcription data.

            Parameters:
                audio_data (bytes): Raw audio bytes received from the client.

            Returns:
                dict: A transcription result object (partial or final) when available, or `None` if no transcription is produced.
            """
            return await self._core.process_audio_chunk(audio_data)  # type: ignore

        def get_full_transcript(self) -> str:
            """
            Retrieve the concatenated transcript produced by the underlying transcriber.

            Returns:
                full_transcript (str): The combined transcript text of all finalized segments in chronological order.
            """
            return self._core.get_full_transcript()  # type: ignore

        def reset(self) -> None:
            """
            Reset the adapter and its underlying core transcriber state.

            This clears any internal buffers and runtime state held by the adapter by delegating reset to the wrapped core transcriber.
            """
            self._core.reset()  # type: ignore

        def cleanup(self) -> None:
            """
            Reset the underlying core adapter/transcriber and ignore any errors raised.

            This calls the core object's `reset` method if present; exceptions from that call are suppressed to ensure cleanup does not raise.
            """
            try:
                self._core.reset()  # type: ignore
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                pass

except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - keep unified path working when core module absent
    _ParakeetCoreAdapter = None  # type: ignore


@dataclass
class UnifiedStreamingConfig(StreamingConfig):
    """Extended configuration for unified streaming."""
    model: str = 'parakeet'  # 'parakeet', 'canary', 'whisper', or 'qwen3-asr'
    model_variant: str = 'standard'  # For Parakeet: 'standard', 'onnx', 'mlx'
    language: Optional[str] = None  # Language code for transcription
    auto_detect_language: bool = False  # Auto-detect language
    enable_vad: bool = True  # Voice Activity Detection default on for lower latency
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 250  # Silence window before considering EOS
    vad_turn_stop_secs: float = 0.2  # Wall clock silence duration to finalize a turn
    vad_min_utterance_secs: float = 0.4  # Guard minimum speech duration before auto-finalizing
    min_partial_duration: float = 0.5
    # Whisper-specific options
    whisper_model_size: str = 'distil-large-v3'  # Whisper model size
    beam_size: int = 5  # Beam search size
    vad_filter: bool = False  # Use VAD filter for Whisper
    task: str = 'transcribe'  # 'transcribe' or 'translate'
    # Diarization-specific options
    diarization_enabled: bool = False
    diarization_store_audio: bool = False
    diarization_storage_dir: Optional[str] = None
    diarization_num_speakers: Optional[int] = None
    # Parakeet RNNT streaming
    parakeet_use_rnnt_streamer: bool = True
    parakeet_rnnt_model_name: str = "nvidia/parakeet-tdt-0.6b-v3"
    parakeet_rnnt_device: Optional[str] = None
    parakeet_rnnt_left_context_s: float = 10.0
    parakeet_rnnt_max_buffer_s: float = 40.0


class SileroTurnDetector:
    """
    Lightweight Silero VAD gate that marks end-of-speech to trigger an auto-commit.

    This helper intentionally fails open: if Silero VAD cannot be loaded, it will
    disable auto-commit and emit a single warning.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        enabled: bool,
        vad_threshold: float,
        min_silence_ms: int,
        turn_stop_secs: float,
        min_utterance_secs: float = 0.4,
    ) -> None:
        self.available = False
        self.unavailable_reason: Optional[str] = None
        self._iterator = None
        self._armed = False
        self._speech_started_at: Optional[float] = None
        self._last_speech_at: Optional[float] = None
        self._last_trigger_at: Optional[float] = None
        self._sample_rate = int(sample_rate or 16000)
        self._turn_stop_secs = max(0.05, float(turn_stop_secs))
        self._min_utterance_secs = max(0.0, float(min_utterance_secs))
        self._vad_threshold = float(vad_threshold)
        self._backend: str = "silero_hub"
        self._onnx_session = None
        self._onnx_input_name: Optional[str] = None

        if not enabled:
            self.unavailable_reason = "disabled"
            return

        # Resolve backend from config ([Diarization].vad_backend); default to silero_hub
        backend = "silero_hub"
        onnx_model_path: Optional[str] = None
        try:
            cfg = load_comprehensive_config()
            if cfg and cfg.has_section("Diarization"):
                try:
                    raw_backend = cfg.get("Diarization", "vad_backend", fallback=backend)
                    backend = str(raw_backend or "").strip().lower() or backend
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    backend = "silero_hub"
                try:
                    onnx_model_path = cfg.get("Diarization", "onnx_model_path", fallback=None)
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    onnx_model_path = None
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
            backend = "silero_hub"

        self._backend = backend

        # Backend: ONNX Silero via onnxruntime (no torch.hub)
        if self._backend == "onnx_silero":
            try:
                import onnxruntime  # type: ignore
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as err:  # pragma: no cover - optional dependency
                self.unavailable_reason = f"onnxruntime_not_available: {err}"
                logger.warning(f"ONNX Silero VAD unavailable (onnxruntime missing); continuing without auto-commit: {err}")
                return

            model_path_str = onnx_model_path or "models/silero_vad/silero_vad_v6.onnx"
            model_path = Path(model_path_str).expanduser()
            if not model_path.is_absolute():
                model_path = (Path.cwd() / model_path).resolve()
            if not model_path.is_file():
                self.unavailable_reason = f"onnx_model_missing: {model_path}"
                logger.warning(f"ONNX Silero VAD model not found at {model_path}; continuing without auto-commit")
                return

            try:
                session = onnxruntime.InferenceSession(str(model_path), providers=onnxruntime.get_available_providers())
                inputs = session.get_inputs()
                if not inputs:
                    raise RuntimeError("ONNX Silero VAD session has no inputs")
                self._onnx_session = session
                self._onnx_input_name = inputs[0].name
                self.available = True
                logger.info(f"Streaming ONNX Silero VAD initialized from {model_path}")
                return
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as err:  # pragma: no cover - defensive
                self.unavailable_reason = f"onnx_session_error: {err}"
                logger.warning(f"ONNX Silero VAD failed to initialize; continuing without auto-commit: {err}")
                self.available = False
                self._onnx_session = None
                return

        # Backend: original Silero via torch.hub
        try:
            try:
                from .VAD_Lib import (
                    _lazy_import_silero_vad,  # type: ignore
                    get_silero_vad_unavailable_reason,
                )
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                _lazy_import_silero_vad = None  # type: ignore
                get_silero_vad_unavailable_reason = lambda: None  # type: ignore

            if _lazy_import_silero_vad is None:
                self.unavailable_reason = "silero_vad_not_available"
                logger.warning("Silero VAD unavailable; continuing without auto-commit")
                return

            model, utils = _lazy_import_silero_vad()
            VADIterator = None
            if utils and len(utils) > 3:
                VADIterator = utils[3]

            if not model or VADIterator is None:
                self.unavailable_reason = (
                    get_silero_vad_unavailable_reason() or "silero_vad_not_available"
                )
                logger.warning(
                    f"Silero VAD unavailable ({self.unavailable_reason}); continuing without auto-commit"
                )
                return

            self._iterator = VADIterator(
                model=model,
                threshold=self._vad_threshold,
                sampling_rate=self._sample_rate,
                min_silence_duration_ms=int(min_silence_ms),
                speech_pad_ms=30,
            )
            self.available = True
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as err:  # pragma: no cover - defensive; exercised in fail-open tests
            self.unavailable_reason = str(err)
            logger.warning(f"Silero VAD failed to initialize; continuing without auto-commit: {err}")
            self.available = False
            self._iterator = None

    @property
    def last_trigger_at(self) -> Optional[float]:
        """Return the last auto-commit trigger timestamp if one was raised."""
        return self._last_trigger_at

    def _saw_speech(self, vad_result: Any) -> bool:
        """Best-effort speech detection from Silero iterator outputs."""
        if vad_result is None:
            return False
        try:
            if isinstance(vad_result, dict):
                if vad_result.get("speech_timestamps"):
                    return True
                if vad_result.get("start") is not None or vad_result.get("end") is not None:
                    return True
                probs = vad_result.get("speech_probs") or vad_result.get("probs")
                if probs:
                    try:
                        return max(float(p) for p in probs) >= self._vad_threshold
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        return True
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
            return False
        return False

    def observe(self, audio_bytes: bytes) -> bool:
        """
        Feed an audio chunk into the detector.

        Returns True exactly once per speech turn when silence >= turn_stop_secs
        occurs after a detected speech span (minimum utterance guard applied).
        """
        if not self.available:
            return False
        if not audio_bytes:
            return False

        # Backend: ONNX Silero (simple chunk-level gate using max probability)
        if self._backend == "onnx_silero":
            if self._onnx_session is None or self._onnx_input_name is None:
                return False
            try:
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                if audio_np.size == 0:
                    return False

                # Prepare input tensor based on model input rank
                input_meta = self._onnx_session.get_inputs()[0]
                shape = input_meta.shape
                if len(shape) == 3:
                    audio_in = audio_np.reshape(1, 1, -1)
                elif len(shape) == 2:
                    audio_in = audio_np.reshape(1, -1)
                else:
                    audio_in = audio_np.reshape(1, -1)

                outputs = self._onnx_session.run(None, {self._onnx_input_name: audio_in})
                if not outputs:
                    return False
                probs = np.asarray(outputs[0]).reshape(-1)
                if probs.size == 0:
                    return False
                speech_detected = bool(float(probs.max()) >= self._vad_threshold)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as err:
                logger.warning(f"ONNX Silero VAD failed during observe; disabling auto-commit: {err}")
                self.available = False
                self.unavailable_reason = str(err)
                return False
        else:
            # Backend: classic Silero iterator
            if not self._iterator:
                return False
            try:
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                if audio_np.size == 0:
                    return False
                audio_in = audio_np
                # Prefer torch tensor input when available (Silero expects torch tensors)
                torch_mod = _get_torch_module()
                if torch_mod is not None:
                    try:
                        audio_in = torch_mod.from_numpy(audio_np)  # type: ignore
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        audio_in = audio_np
                vad_result = self._iterator(audio_in, return_seconds=False)
                speech_detected = self._saw_speech(vad_result)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as err:
                logger.warning(f"Silero VAD failed during observe; disabling auto-commit: {err}")
                self.available = False
                self.unavailable_reason = str(err)
                try:
                    if hasattr(self._iterator, "reset_states"):
                        self._iterator.reset_states()
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    pass
                return False

        now = time.time()

        if speech_detected:
            self._speech_started_at = self._speech_started_at or now
            self._last_speech_at = now
            self._armed = True
            return False

        # No speech detected in this chunk
        if not self._armed:
            return False

        # Require a minimum utterance duration before triggering EOS
        if self._speech_started_at and (now - self._speech_started_at) < self._min_utterance_secs:
            return False

        if self._last_speech_at and (now - self._last_speech_at) >= self._turn_stop_secs:
            self._armed = False
            self._speech_started_at = None
            self._last_speech_at = None
            self._last_trigger_at = now
            try:
                if hasattr(self._iterator, "reset_states"):
                    self._iterator.reset_states()
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                pass
            return True

        return False


class StreamingDiarizer:
    """Best-effort wrapper to reuse the offline DiarizationService during streaming."""

    def __init__(
        self,
        sample_rate: int,
        *,
        store_audio: bool = False,
        storage_dir: Optional[str] = None,
        num_speakers: Optional[int] = None,
    ) -> None:
        """
        Initialize the streaming diarizer used to collect audio and produce speaker-aligned segments.

        Parameters:
            sample_rate (int): Audio sample rate in Hz used for buffering and diarization. Defaults to 16000 when falsy.
            store_audio (bool): If True, persist the combined audio to disk when finalizing diarization.
            storage_dir (Optional[str]): Untrusted directory name hint for persisted audio. Sanitized to a safe subdirectory under system temp; absolute paths and path traversal attempts are constrained to basename only.
            num_speakers (Optional[int]): Optional hint for the expected number of speakers; passed to the underlying diarization service when available.
        """
        self.sample_rate = int(sample_rate or 16000)
        self.store_audio = bool(store_audio)
        self.storage_dir = _safe_temp_subdir(storage_dir)
        self.num_speakers = num_speakers
        self._audio_chunks: list[np.ndarray] = []
        self._transcript_segments: list[dict[str, Any]] = []
        self._mapping: dict[int, dict[str, Any]] = {}
        self._last_result: dict[str, Any] = {}
        self._dirty = False
        self._persist_path: Optional[Path] = None
        self._persist_method: Optional[str] = None  # 'soundfile' | 'scipy' | 'wave' | None
        self._lock = asyncio.Lock()
        self._service = None
        self.available = False
        self._service_checked = False
        self._service_error: Optional[str] = None

    async def label_segment(self, audio_np: np.ndarray, segment_meta: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Append audio/transcript and return the speaker for the latest segment."""
        if not await self._ensure_service():
            return None
        async with self._lock:
            self._audio_chunks.append(np.array(audio_np, copy=True))
            self._transcript_segments.append({
                "start": float(segment_meta.get("segment_start") or segment_meta.get("chunk_start") or 0.0),
                "end": float(segment_meta.get("segment_end") or segment_meta.get("chunk_end") or 0.0),
                "text": segment_meta.get("text", ""),
                "segment_id": int(segment_meta.get("segment_id", 0)),
            })
            self._dirty = True
            mapping = await self._ensure_mapping()
            segment_id = int(segment_meta.get("segment_id", 0))
            return mapping.get(segment_id)

    async def finalize(self) -> tuple[dict[int, dict[str, Any]], Optional[str], Optional[list[dict[str, Any]]]]:
        """Ensure latest mapping is available and optionally persist audio."""
        if not await self._ensure_service():
            return {}, None, None
        async with self._lock:
            self._dirty = True
            mapping = await self._ensure_mapping()
            audio_path = None
            if self.store_audio and self._audio_chunks:
                audio_path = await self._persist_audio()
            speakers = self._last_result.get("speakers")
            return mapping, audio_path, speakers

    async def reset(self) -> None:
        """
        Clear all buffered audio, transcripts, and diarization state, and reset persistence metadata.

        This empties the internal audio chunk and transcript segment buffers, clears any computed speaker mapping, resets the last result and dirty flag, and clears persistence path and method so the diarizer returns to an initial state.
        """
        async with self._lock:
            self._audio_chunks.clear()
            self._transcript_segments.clear()
            self._mapping.clear()
            self._last_result = {}
            self._dirty = False
            self._persist_path = None
            self._persist_method = None

    async def close(self) -> None:
        """
        Close the diarizer, clear buffered audio and transcripts, and release any held resources.
        """
        await self.reset()

    async def ensure_ready(self) -> bool:
        """Public helper to eagerly initialize the diarization backend."""
        return await self._ensure_service()

    async def _ensure_service(self) -> bool:
        if self._service_checked:
            return self.available and self._service is not None
        self._service_checked = True
        if DiarizationService is None:
            logger.debug("Streaming diarizer: DiarizationService import unavailable.")
            self.available = False
            self._service = None
            return False
        loop = asyncio.get_running_loop()
        try:
            service = await loop.run_in_executor(None, DiarizationService)
            is_available = bool(getattr(service, "is_available", True))
            if not is_available:
                logger.warning("Streaming diarizer dependencies missing; disabling diarization.")
                self.available = False
                self._service = None
                return False
            self._service = service
            self.available = True
            return True
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(f"Streaming diarizer unavailable: {exc}")
            self.available = False
            self._service = None
            self._service_error = str(exc)
            return False

    async def _ensure_mapping(self) -> dict[int, dict[str, Any]]:
        if not await self._ensure_service():
            return {}
        if not self._dirty:
            return self._mapping
        loop = asyncio.get_running_loop()
        mapping = await loop.run_in_executor(None, self._run_alignment_sync)
        if mapping is not None:
            self._mapping = mapping
            self._dirty = False
        return self._mapping

    def _run_alignment_sync(self) -> Optional[dict[int, dict[str, Any]]]:
        if not self._service or not self._audio_chunks:
            return {}
        combined = self._combined_audio()
        if combined.size == 0:
            return {}
        tmp_path = None
        try:
            tmp_path = self._write_temp_wav(combined)
            transcripts = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg.get("text", ""),
                    "segment_id": seg["segment_id"],
                }
                for seg in self._transcript_segments
            ]
            result = self._service.diarize(
                str(tmp_path),
                transcription_segments=transcripts,
                num_speakers=self.num_speakers,
            )
            self._last_result = result or {}
            segments = self._last_result.get("segments", []) or transcripts
            mapping: dict[int, dict[str, Any]] = {}
            for seg in segments:
                seg_id = seg.get("segment_id") or seg.get("id")
                if seg_id is None:
                    continue
                mapping[int(seg_id)] = {
                    "speaker_id": seg.get("speaker_id"),
                    "speaker_label": seg.get("speaker_label"),
            }
            return mapping
        except DiarizationError as err:
            logger.error(f"Streaming diarizer failed: {err}")
            self.available = False
            return {}
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as exc:
            logger.exception("Streaming diarizer unexpected error: {}", exc)
            return {}
        finally:
            if tmp_path:
                with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                    Path(tmp_path).unlink(missing_ok=True)

    def _combined_audio(self) -> np.ndarray:
        if not self._audio_chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._audio_chunks) == 1:
            return np.array(self._audio_chunks[0], copy=True)
        return np.concatenate(self._audio_chunks)

    def _write_temp_wav(self, audio_np: np.ndarray) -> Path:
        """
        Write a temporary WAV file from a mono audio NumPy array and return its filesystem path.

        Parameters:
            audio_np (np.ndarray): 1-D NumPy array of audio samples, expected in float32 range [-1.0, 1.0]; the method will convert to an appropriate PCM format if needed.

        Returns:
            Path: Filesystem path to the created temporary WAV file.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        # Prefer soundfile; fall back to scipy.io.wavfile or wave if unavailable
        try:
            import soundfile as sf  # type: ignore
            sf.write(str(tmp_path), audio_np, self.sample_rate)
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as sf_err:
            logger.warning(f"soundfile unavailable for temp WAV write: {sf_err}; falling back")
            # Try scipy
            try:
                from scipy.io import wavfile  # type: ignore
                # Convert float32 [-1,1] to int16 for scipy
                pcm16 = np.clip(audio_np, -1.0, 1.0)
                pcm16 = (pcm16 * 32767.0).astype(np.int16)
                wavfile.write(str(tmp_path), self.sample_rate, pcm16)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as scipy_err:
                logger.warning(f"scipy.io.wavfile write failed: {scipy_err}; using wave module")
                # Fallback to wave module (int16 PCM)
                import wave
                pcm16 = np.clip(audio_np, -1.0, 1.0)
                pcm16 = (pcm16 * 32767.0).astype(np.int16)
                with wave.open(str(tmp_path), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(pcm16.tobytes())
        return tmp_path

    async def _persist_audio(self) -> Optional[str]:
        """
        Run the synchronous audio persistence routine in a thread executor and return the persisted file path if any.

        Returns:
            persisted_path (Optional[str]): Path to the persisted audio file if persistence succeeded, `None` if no file was written.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._persist_audio_sync)

    def _persist_audio_sync(self) -> Optional[str]:
        """
        Persist the currently buffered audio to a WAV file using the best available backend.

        Attempts to write the concatenated buffered audio to disk (preferred backends tried in order: soundfile, scipy.io.wavfile, wave). On success sets self._persist_path to the file path and self._persist_method to the backend used and returns the file path as a string. If no audio is buffered or all backends fail, sets self._persist_method to None and returns `None`.

        Returns:
            str or None: Absolute path to the persisted WAV file on success, `None` if no audio was written.
        """
        audio_np = self._combined_audio()
        if audio_np.size == 0:
            return None
        out_dir = self.storage_dir or Path(tempfile.gettempdir())
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self._persist_path:
            filename = f"stream_{uuid4().hex}.wav"
            self._persist_path = out_dir / filename
        # Prefer soundfile; fall back to scipy.io.wavfile or wave if unavailable
        try:
            try:
                import soundfile as sf  # type: ignore
                sf.write(str(self._persist_path), audio_np, self.sample_rate)
                self._persist_method = "soundfile"
                return str(self._persist_path)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as sf_err:
                logger.warning(f"soundfile unavailable for persistence: {sf_err}; falling back")
                try:
                    from scipy.io import wavfile  # type: ignore
                    pcm16 = np.clip(audio_np, -1.0, 1.0)
                    pcm16 = (pcm16 * 32767.0).astype(np.int16)
                    wavfile.write(str(self._persist_path), self.sample_rate, pcm16)
                    self._persist_method = "scipy"
                    return str(self._persist_path)
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as scipy_err:
                    logger.warning(f"scipy.io.wavfile persistence failed: {scipy_err}; using wave module")
                    try:
                        import wave
                        pcm16 = np.clip(audio_np, -1.0, 1.0)
                        pcm16 = (pcm16 * 32767.0).astype(np.int16)
                        with wave.open(str(self._persist_path), 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(self.sample_rate)
                            wf.writeframes(pcm16.tobytes())
                        self._persist_method = "wave"
                        return str(self._persist_path)
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as wave_err:
                        logger.warning(f"wave module persistence failed: {wave_err}")
                        self._persist_method = None
                        # Disable further attempts to persist for this session
                        self.store_audio = False
                        return None
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as persist_err:
            logger.error(f"Audio persistence failed: {persist_err}")
            self._persist_method = None
            # Disable further attempts to persist for this session
            self.store_audio = False
            return None

    @property
    def persistence_method(self) -> Optional[str]:
        """
        Name of the audio persistence backend selected for writing WAV files.

        Returns:
            persistence_backend (Optional[str]): The backend identifier (`'soundfile'`, `'scipy'`, or `'wave'`) if a persistence writer was selected, or `None` if no persistence backend is available.
        """
        return self._persist_method


_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS = _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS + (
    DiarizationError,
    QuotaExceeded,
)


class BaseStreamingTranscriber(ABC):
    """
    Abstract base class for streaming transcribers.

    Defines the common interface for all streaming transcription implementations.
    """

    def __init__(self, config: UnifiedStreamingConfig):
        """Initialize base transcriber."""
        self.config = config
        self.buffer = AudioBuffer(
            sample_rate=config.sample_rate,
            max_duration=config.max_buffer_duration
        )
        self.model = None
        self.is_running = False
        self.transcription_history = []
        self.last_partial_time = 0
        self.segment_index = 0
        self.total_processed_seconds = 0.0

    @abstractmethod
    def initialize(self):
        """Load and initialize the model."""
        pass

    @abstractmethod
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[dict[str, Any]]:
        """Process a chunk of audio data."""
        pass

    def get_full_transcript(self) -> str:
        """Get the complete transcript so far."""
        return " ".join(self.transcription_history)

    def reset(self):
        """Reset the transcriber state."""
        self.buffer.clear()
        self.transcription_history.clear()
        self.last_partial_time = 0
        self.segment_index = 0
        self.total_processed_seconds = 0.0

    def cleanup(self):
        """Clean up resources."""
        self.model = None
        self.reset()

    def _prepare_partial_metadata(self, buffer_duration: float) -> dict[str, float]:
        """Attach common metadata for partial updates."""
        buffer_duration = float(buffer_duration)
        start = float(self.total_processed_seconds)
        return {
            "segment_id": self.segment_index + 1,
            "segment_start": start,
            "segment_end": start + buffer_duration,
            "buffer_duration": buffer_duration,
            "cumulative_audio": float(self.total_processed_seconds),
        }

    def _prepare_final_metadata(self, chunk_duration: float) -> dict[str, float]:
        """Attach metadata for finalized segments and advance the timeline cursor."""
        chunk_duration = float(chunk_duration)
        if chunk_duration < 0:
            chunk_duration = 0.0
        overlap_cfg = max(float(self.config.overlap_duration or 0.0), 0.0)
        overlap_used = 0.0 if self.segment_index == 0 else min(overlap_cfg, chunk_duration)
        new_audio_duration = chunk_duration - overlap_used
        if self.segment_index == 0:
            new_audio_duration = chunk_duration
        if new_audio_duration < 0:
            new_audio_duration = 0.0

        segment_start = float(self.total_processed_seconds)
        segment_end = segment_start + new_audio_duration
        chunk_start = max(segment_start - overlap_used, 0.0)
        chunk_end = chunk_start + chunk_duration

        self.total_processed_seconds = segment_end
        self.segment_index += 1

        return {
            "segment_id": self.segment_index,
            "segment_start": segment_start,
            "segment_end": segment_end,
            "chunk_duration": chunk_duration,
            "overlap": overlap_used,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "new_audio_duration": new_audio_duration,
            "cumulative_audio": float(self.total_processed_seconds),
        }


def _strip_parakeet_eou_token(text: str) -> str:
    """
    Remove the literal `<EOU>` token emitted by Parakeet-Realtime-EOU style
    models from user-visible text.

    This is a small, easily testable helper used by the RNNT streaming path
    so that clients do not see end-of-utterance markers in transcripts while
    still benefitting from the model's segmentation behaviour.
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
            return ""
    return text.replace("<EOU>", "").strip()


class _ParakeetRNNTStreamer:
    """
    Lightweight adapter around Nemo RNNT streaming models (Parakeet TDT and
    related checkpoints, including Parakeet-Realtime-EOU).

    The implementation is generic over the underlying RNNT checkpoint name.
    For models such as `nvidia/parakeet_realtime_eou_120m-v1` that emit a
    special `<EOU>` token at the end of each utterance, the `push()` method
    strips the literal token from user-visible text while preserving normal
    streaming behaviour.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str],
        left_context_s: float,
        chunk_s: float,
        right_context_s: float,
        max_buffer_s: float,
        batch_size: int = 1,
    ) -> None:
        _ensure_nemo_rnnt_deps()
        torch_mod = _get_torch_module()
        if nemo_asr is None or RNNTDecodingConfig is None or batched_hyps_to_hypotheses is None or ContextSize is None:
            raise RuntimeError("Nemo RNNT streaming requires nemo_toolkit[asr] and its dependencies")
        if torch_mod is None:
            raise RuntimeError("PyTorch is required for Parakeet RNNT streaming")

        self._torch = torch_mod
        self.device = (
            torch_mod.device(device)
            if device
            else torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
        )
        # Avoid mutating global grad/precision state here; the streaming hot
        # path runs under torch.inference_mode() and model parameters are
        # frozen below. We only tune thread count for performance.
        with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
            torch_mod.set_num_threads(max(1, torch_mod.get_num_threads() or 1))

        self.model = (
            nemo_asr.models.EncDecRNNTModel.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        for p in self.model.parameters():
            with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                p.requires_grad_(False)

        try:
            if hasattr(self.model, "preprocessor") and hasattr(self.model.preprocessor, "featurizer"):
                self.model.preprocessor.featurizer.dither = 0.0
                self.model.preprocessor.featurizer.pad_to = 0
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
            pass

        dec_cfg = RNNTDecodingConfig(
            strategy="greedy_batch",
            fused_batch_size=-1,
            compute_timestamps=False,
        )
        try:
            dec_cfg.greedy.loop_labels = True
            dec_cfg.greedy.preserve_alignments = False
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
            pass
        self.model.change_decoding_strategy(dec_cfg)
        self._decoding_computer = self.model.decoding.decoding.decoding_computer

        mcfg = copy.deepcopy(getattr(self.model, "_cfg", getattr(self.model, "cfg", None)))
        if mcfg is None or not hasattr(mcfg, "preprocessor"):
            raise RuntimeError("Unable to access Parakeet RNNT model config")

        self.sample_rate: int = int(getattr(mcfg.preprocessor, "sample_rate", 16000))
        window_stride: float = float(getattr(mcfg.preprocessor, "window_stride", 0.01) or 0.01)
        self.frames_per_second: float = 1.0 / window_stride

        if not hasattr(self.model, "encoder") or not hasattr(self.model.encoder, "subsampling_factor"):
            raise RuntimeError("Parakeet RNNT encoder must expose subsampling_factor for streaming alignment")
        self.subsampling: int = int(self.model.encoder.subsampling_factor)

        feat_f2a = self._floor_multiple(int(self.sample_rate * window_stride), self.subsampling)
        self.enc_f2a = feat_f2a * self.subsampling

        self.ctx_enc = ContextSize(
            left=int(left_context_s * self.frames_per_second / self.subsampling),
            chunk=int(chunk_s * self.frames_per_second / self.subsampling),
            right=int(right_context_s * self.frames_per_second / self.subsampling),
        )
        self.ctx_samp = ContextSize(
            left=self.ctx_enc.left * self.subsampling * feat_f2a,
            chunk=self.ctx_enc.chunk * self.subsampling * feat_f2a,
            right=self.ctx_enc.right * self.subsampling * feat_f2a,
        )

        self.max_samples = int(max_buffer_s * self.sample_rate)
        self.batch_size = batch_size

        self._stream_np: Optional[np.ndarray] = None
        self._buf: Optional[StreamingBatchedAudioBuffer] = None
        self._prev_state = None
        self._cur_hyps = None
        self._l = 0
        self._r = 0
        self._resampler_cache = {}

    @staticmethod
    def _floor_multiple(a: int, b: int) -> int:
        return (a // b) * b

    @staticmethod
    def _to_mono(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 2:
            x = x.mean(axis=-1 if x.shape[-1] in (1, 2) else 1)
        return x.astype(np.float32, copy=False)

    def _resample_if_needed(self, x: np.ndarray, in_sr: int) -> np.ndarray:
        if in_sr == self.sample_rate:
            return x.astype(np.float32, copy=False)
        in_sr = int(in_sr)
        cache_key = (in_sr, self.sample_rate)
        torchaudio_mod = _get_torchaudio_module()
        if torchaudio_mod is not None:
            if cache_key not in self._resampler_cache:
                try:
                    self._resampler_cache[cache_key] = torchaudio_mod.transforms.Resample(
                        orig_freq=in_sr, new_freq=self.sample_rate
                    )
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    self._resampler_cache[cache_key] = None
            resampler = self._resampler_cache.get(cache_key)
            if resampler is not None:
                try:
                    y = resampler(self._torch.from_numpy(x))
                    return y.numpy().astype(np.float32, copy=False)
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    pass
        try:
            return _resample_audio_if_needed(x, in_sr, target_sr=self.sample_rate)
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
            # Naive linear fallback
            ratio = float(self.sample_rate) / float(in_sr)
            new_len = max(1, round(len(x) * ratio))
            x_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            return np.interp(x_new, x_old, x).astype(np.float32, copy=False)

    def reset(self) -> None:
        self._stream_np = None
        self._buf = None
        self._prev_state = None
        self._cur_hyps = None
        self._l = 0
        self._r = 0

    def push(self, audio_np: np.ndarray, input_sr: int) -> str:
        if audio_np is None or audio_np.size == 0:
            return ""
        torch_mod = self._torch
        y = self._to_mono(audio_np)
        y = self._resample_if_needed(y, input_sr)

        if self._stream_np is None or self._stream_np.size == 0:
            self._stream_np = y
        else:
            self._stream_np = np.concatenate([self._stream_np, y])
            if self._stream_np.size > self.max_samples:
                drop = self._stream_np.size - self.max_samples
                self._stream_np = self._stream_np[-self.max_samples:]
                self._l = max(0, self._l - drop)
                self._r = max(self.ctx_samp.chunk + self.ctx_samp.right, self._r - drop)

        if self._buf is None:
            self._buf = StreamingBatchedAudioBuffer(
                batch_size=self.batch_size,
                context_samples=self.ctx_samp,
                dtype=torch_mod.float32,
                device=self.device,
            )
            self._l = 0
            self._r = self.ctx_samp.chunk + self.ctx_samp.right

        a = (
            torch_mod.from_numpy(self._stream_np)
            .unsqueeze(0)
            .to(torch_mod.float32)
            .to(self.device)
        )

        with torch_mod.inference_mode():
            while self._l < a.shape[1]:
                if a.shape[1] < self._r:
                    break
                clen = int(self._r - self._l)
                if clen <= 0:
                    break

                is_last_chunk = False
                is_last_b = torch_mod.tensor(
                    [False], dtype=torch_mod.bool, device=self.device
                )
                clen_b = torch_mod.tensor(
                    [clen], dtype=torch_mod.long, device=self.device
                )

                self._buf.add_audio_batch_(
                    a[:, self._l:self._r],
                    audio_lengths=clen_b,
                    is_last_chunk=is_last_chunk,
                    is_last_chunk_batch=is_last_b,
                )

                enc, _ = self.model(
                    input_signal=self._buf.samples,
                    input_signal_length=self._buf.context_size_batch.total(),
                )
                enc = enc.transpose(1, 2)  # [B, T, C]

                enc_ctx = self._buf.context_size.subsample(factor=self.enc_f2a)
                enc_ctx_b = self._buf.context_size_batch.subsample(factor=self.enc_f2a)

                enc = enc[:, enc_ctx.left:]

                hyps, _, self._prev_state = self._decoding_computer(
                    x=enc, out_len=enc_ctx_b.chunk, prev_batched_state=self._prev_state
                )

                if self._cur_hyps is None:
                    self._cur_hyps = hyps
                else:
                    self._cur_hyps.merge_(hyps)

                self._l = self._r
                self._r = self._r + self.ctx_samp.chunk

        outs = (
            batched_hyps_to_hypotheses(self._cur_hyps, None, batch_size=self.batch_size)
            if self._cur_hyps is not None
            else []
        )
        for h in outs:
            with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                h.text = self.model.tokenizer.ids_to_text(h.y_sequence.tolist())
        if not outs:
            return ""

        text = outs[0].text
        # For Parakeet-Realtime-EOU style models, drop the literal "<EOU>"
        # marker so clients see clean text while still benefiting from the
        # model's end-of-utterance detection for segmentation.
        return _strip_parakeet_eou_token(text)


class ParakeetStreamingTranscriber(BaseStreamingTranscriber):
    """
    Parakeet-specific streaming transcriber.

    Supports all Parakeet variants: standard, ONNX, MLX.
    """

    def initialize(self):
        """Load the Parakeet model based on configuration."""
        variant = self.config.model_variant
        logger.info(f"Loading Parakeet model (variant: {variant})")
        self._rnnt_streamer: Optional[_ParakeetRNNTStreamer] = None
        self._rnnt_last_partial: str = ""
        self._rnnt_last_final: str = ""
        self._mlx_stream_session: Any = None
        self._mlx_last_partial: str = ""
        self._mlx_last_final: str = ""

        if variant == 'mlx':
            # MLX model is loaded on-demand in transcribe function
            # First check if MLX dependencies are available
            try:
                importlib.import_module(
                    "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX"
                )
                logger.info("Using Parakeet MLX variant (lazy loading)")
                self.model = "mlx"  # Placeholder to indicate MLX is ready
                try:
                    from tldw_Server_API.app.core.config import get_stt_config

                    stt_cfg = get_stt_config() or {}
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    stt_cfg = {}

                def _cfg_int(key: str, default: int) -> int:
                    """Return an integer config value with safe fallback coercion."""
                    try:
                        value = stt_cfg.get(key, default)
                        return int(value)
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        return default

                def _cfg_float(key: str, default: float) -> float:
                    """Return a float config value with safe fallback coercion."""
                    try:
                        value = stt_cfg.get(key, default)
                        return float(value)
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        return default

                def _cfg_str(key: str) -> Optional[str]:
                    """Return a non-empty string config value or ``None``."""
                    value = stt_cfg.get(key)
                    if value is None:
                        return None
                    value_str = str(value).strip()
                    return value_str if value_str else None

                def _cfg_bool(key: str, *, default: bool = False) -> bool:
                    """Return a boolean config value with string/number normalization."""
                    value = stt_cfg.get(key)
                    if value is None:
                        return default
                    if isinstance(value, bool):
                        return value
                    value_str = str(value).strip().lower()
                    if value_str in {"1", "true", "yes", "on"}:
                        return True
                    if value_str in {"0", "false", "no", "off"}:
                        return False
                    return default

                self._mlx_stream_session = create_parakeet_mlx_streaming_session(
                    model_path=_cfg_str("mlx_model_id"),
                    cache_dir=_cfg_str("mlx_cache_dir"),
                    context_size=(
                        _cfg_int("mlx_stream_context_left", 256),
                        _cfg_int("mlx_stream_context_right", 256),
                    ),
                    depth=max(_cfg_int("mlx_stream_depth", 1), 1),
                    keep_original_attention=_cfg_bool(
                        "mlx_stream_keep_original_attention",
                        default=False,
                    ),
                    decoding_mode=_cfg_str("mlx_decoding_mode"),
                    beam_size=_cfg_int("mlx_beam_size", 0) or None,
                    length_penalty=_cfg_float("mlx_length_penalty", 0.0) if stt_cfg.get("mlx_length_penalty") is not None else None,
                    patience=_cfg_float("mlx_patience", 0.0) if stt_cfg.get("mlx_patience") is not None else None,
                    duration_reward=_cfg_float("mlx_duration_reward", 0.0) if stt_cfg.get("mlx_duration_reward") is not None else None,
                    sentence_max_words=_cfg_int("mlx_sentence_max_words", 0) or None,
                    sentence_silence_gap=_cfg_float("mlx_sentence_silence_gap", 0.0) if stt_cfg.get("mlx_sentence_silence_gap") is not None else None,
                    sentence_max_duration=_cfg_float("mlx_sentence_max_duration", 0.0) if stt_cfg.get("mlx_sentence_max_duration") is not None else None,
                )
                if self._mlx_stream_session is not None:
                    logger.info("Initialized Parakeet MLX native streaming session")
                else:
                    logger.info("Parakeet MLX native streaming unavailable; using legacy chunk decode path")
                return  # Success
            except ImportError as e:
                logger.error(f"Failed to import Parakeet MLX: {e}")
                raise RuntimeError(
                    "Parakeet MLX dependencies not available. Install with: pip install mlx mlx-lm"
                ) from e
        else:
            # Load standard or ONNX variant (requires Nemo)
            try:
                self.model = load_parakeet_model(variant)
                if self.model is None:
                    raise RuntimeError(f"Failed to load Parakeet {variant} model")
                logger.info(f"Loaded Parakeet {variant} model")

                # RNNT streaming backend is optional and only enabled when the
                # UnifiedStreamingConfig fields are present. When a legacy
                # Parakeet StreamingConfig is provided (used by older tests),
                # these attributes may be missing; in that case we fall back to
                # the chunked implementation without raising.
                use_rnnt = bool(getattr(self.config, "parakeet_use_rnnt_streamer", False))
                if use_rnnt and _ParakeetRNNTStreamer is not None:
                    try:
                        model_name = getattr(
                            self.config,
                            "parakeet_rnnt_model_name",
                            "nvidia/parakeet-tdt-0.6b-v3",
                        )
                        device = getattr(self.config, "parakeet_rnnt_device", None)
                        left_context_s = float(
                            getattr(self.config, "parakeet_rnnt_left_context_s", 10.0)
                        )
                        max_buffer_s = float(
                            getattr(self.config, "parakeet_rnnt_max_buffer_s", 40.0)
                        )
                        chunk_s = max(
                            float(getattr(self.config, "chunk_duration", 0.0) or 0.0),
                            0.1,
                        )
                        right_context_s = max(
                            float(getattr(self.config, "overlap_duration", 0.0) or 0.0),
                            0.0,
                        )

                        self._rnnt_streamer = _ParakeetRNNTStreamer(
                            model_name=model_name,
                            device=device,
                            left_context_s=left_context_s,
                            chunk_s=chunk_s,
                            right_context_s=right_context_s,
                            max_buffer_s=max_buffer_s,
                            batch_size=1,
                        )
                        logger.info("Initialized Parakeet RNNT streaming backend")
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as rnnt_err:
                        logger.warning(
                            f"Parakeet RNNT streaming unavailable, using legacy chunking: {rnnt_err}"
                        )
            except ImportError as e:
                if "nemo" in str(e).lower():
                    logger.warning(
                        "Nemo toolkit not installed, attempting to fallback to MLX variant"
                    )
                    # Try to fallback to MLX variant
                    try:
                        importlib.import_module(
                            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX"
                        )
                        logger.info(
                            "Falling back to Parakeet MLX variant due to missing Nemo"
                        )
                        self.config.model_variant = "mlx"
                        self.model = "mlx"  # Placeholder to indicate MLX is ready
                        return  # Success with fallback
                    except ImportError:
                        logger.error(
                            "MLX fallback failed - MLX dependencies not available"
                        )
                raise RuntimeError(
                    f"Nemo toolkit not installed for {variant} variant and MLX fallback unavailable. "
                    f"Install Nemo with: pip install nemo_toolkit[asr] "
                    f"OR install MLX with: pip install mlx mlx-lm"
                ) from e

    def reset(self):
        """Reset transcriber buffers and Parakeet RNNT streaming state."""
        super().reset()
        streamer = getattr(self, "_rnnt_streamer", None)
        if streamer is not None:
            try:
                streamer.reset()
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                # Fail open: RNNT will rebuild state on next push if needed.
                pass
        # Clear RNNT-specific tracking so partial/final comparisons start fresh
        self._rnnt_last_partial = ""
        self._rnnt_last_final = ""
        mlx_session = getattr(self, "_mlx_stream_session", None)
        if mlx_session is not None and hasattr(mlx_session, "close"):
            try:
                mlx_session.close()
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                pass
        self._mlx_stream_session = None
        self._mlx_last_partial = ""
        self._mlx_last_final = ""

    async def process_audio_chunk(self, audio_data: bytes) -> Optional[dict[str, Any]]:
        """
        Process audio chunk with Parakeet.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Transcription result or None
        """
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.float32)

        # Add to buffer
        self.buffer.add(audio_np)

        current_time = time.time()
        buffer_duration = self.buffer.get_duration()

        # RNNT streaming path (standard/ONNX) avoids temp WAV I/O
        if self._rnnt_streamer is not None:
            try:
                full_text = self._rnnt_streamer.push(audio_np, self.config.sample_rate)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as rnnt_err:
                logger.warning(f"Parakeet RNNT streaming failed, falling back to legacy chunking: {rnnt_err}")
                self._rnnt_streamer = None
                full_text = ""

            if full_text:
                try:
                    from .Audio_Custom_Vocabulary import postprocess_text_if_enabled
                    full_text = postprocess_text_if_enabled(full_text)
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    pass

            if (
                self.config.enable_partial
                and current_time - self.last_partial_time > self.config.partial_interval
                and buffer_duration > max(self.config.min_partial_duration, 0.1)
                and full_text
                and full_text != self._rnnt_last_partial
            ):
                self._rnnt_last_partial = full_text
                self.last_partial_time = current_time
                metadata = self._prepare_partial_metadata(buffer_duration)
                result = {
                    "type": "partial",
                    "text": full_text,
                    "timestamp": current_time,
                    "is_final": False,
                    "model": f"parakeet-{self.config.model_variant}"
                }
                result.update(metadata)
                return result

            if buffer_duration >= self.config.chunk_duration:
                audio_chunk = self.buffer.get_audio(self.config.chunk_duration)
                if audio_chunk is not None:
                    self.buffer.consume(
                        self.config.chunk_duration,
                        self.config.overlap_duration
                    )
                    if full_text and full_text != self._rnnt_last_final:
                        self._rnnt_last_final = full_text
                        chunk_duration = float(len(audio_chunk)) / float(self.config.sample_rate or 1)
                        metadata = self._prepare_final_metadata(chunk_duration)
                        self.transcription_history.append(full_text)
                        result = {
                            "type": "final",
                            "text": full_text,
                            "timestamp": current_time,
                            "is_final": True,
                            "model": f"parakeet-{self.config.model_variant}"
                        }
                        result.update(metadata)
                        result["_audio_chunk"] = np.array(audio_chunk, copy=True)
                        return result

            return None

        if self.config.model_variant == "mlx" and self._mlx_stream_session is not None:
            try:
                artifact = self._mlx_stream_session.add_audio(audio_np, sample_rate=self.config.sample_rate)
                full_text = str((artifact or {}).get("text", "")).strip()
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as mlx_stream_err:
                logger.warning(f"Parakeet MLX native streaming failed, falling back to legacy chunking: {mlx_stream_err}")
                try:
                    self._mlx_stream_session.close()
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    pass
                self._mlx_stream_session = None
                full_text = ""

            if full_text:
                try:
                    from .Audio_Custom_Vocabulary import postprocess_text_if_enabled

                    full_text = postprocess_text_if_enabled(full_text)
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    pass

            if (
                self.config.enable_partial
                and current_time - self.last_partial_time > self.config.partial_interval
                and buffer_duration > max(self.config.min_partial_duration, 0.1)
                and full_text
                and full_text != self._mlx_last_partial
            ):
                self._mlx_last_partial = full_text
                self.last_partial_time = current_time
                metadata = self._prepare_partial_metadata(buffer_duration)
                result = {
                    "type": "partial",
                    "text": full_text,
                    "timestamp": current_time,
                    "is_final": False,
                    "model": f"parakeet-{self.config.model_variant}",
                }
                result.update(metadata)
                return result

            if buffer_duration >= self.config.chunk_duration:
                audio_chunk = self.buffer.get_audio(self.config.chunk_duration)
                if audio_chunk is not None:
                    self.buffer.consume(
                        self.config.chunk_duration,
                        self.config.overlap_duration
                    )
                    if full_text and full_text != self._mlx_last_final:
                        self._mlx_last_final = full_text
                        self.transcription_history.append(full_text)
                        chunk_duration = float(len(audio_chunk)) / float(self.config.sample_rate or 1)
                        metadata = self._prepare_final_metadata(chunk_duration)
                        result = {
                            "type": "final",
                            "text": full_text,
                            "timestamp": current_time,
                            "is_final": True,
                            "model": f"parakeet-{self.config.model_variant}",
                        }
                        result.update(metadata)
                        result["_audio_chunk"] = np.array(audio_chunk, copy=True)
                        return result

            return None

        # Check if we should send a partial result
        if (self.config.enable_partial and
            current_time - self.last_partial_time > self.config.partial_interval and
            buffer_duration > 0.5):

            # Get audio for partial transcription
            audio_for_partial = self.buffer.get_audio()
            if audio_for_partial is not None and len(audio_for_partial) > 0:
                # Transcribe partial audio
                if self.config.model_variant == 'mlx':
                    text = transcribe_with_parakeet_mlx(
                        audio_for_partial,
                        sample_rate=self.config.sample_rate
                    )
                else:
                    # Use standard/ONNX implementation
                    text = transcribe_with_parakeet(
                        audio_for_partial,
                        self.config.sample_rate,
                        self.config.model_variant
                    )

                self.last_partial_time = current_time

                if text:
                    # Apply custom vocabulary post-replacements if enabled
                    try:
                        from .Audio_Custom_Vocabulary import postprocess_text_if_enabled
                        text = postprocess_text_if_enabled(text)
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        pass
                    metadata = self._prepare_partial_metadata(buffer_duration)
                    result = {
                        "type": "partial",
                        "text": text,
                        "timestamp": current_time,
                        "is_final": False,
                        "model": f"parakeet-{self.config.model_variant}"
                    }
                    result.update(metadata)
                    return result

        # Check if we have enough audio for a final chunk
        if buffer_duration >= self.config.chunk_duration:
            # Get chunk for transcription
            audio_chunk = self.buffer.get_audio(self.config.chunk_duration)

            if audio_chunk is not None:
                # Transcribe the chunk
                if self.config.model_variant == 'mlx':
                    text = transcribe_with_parakeet_mlx(
                        audio_chunk,
                        sample_rate=self.config.sample_rate
                    )
                else:
                    text = transcribe_with_parakeet(
                        audio_chunk,
                        self.config.sample_rate,
                        self.config.model_variant
                    )

                # Consume the buffer, keeping overlap
                self.buffer.consume(
                    self.config.chunk_duration,
                    self.config.overlap_duration
                )

                if text:
                    # Apply custom vocabulary post-replacements if enabled
                    try:
                        from .Audio_Custom_Vocabulary import postprocess_text_if_enabled
                        text = postprocess_text_if_enabled(text)
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        pass
                    self.transcription_history.append(text)
                    chunk_duration = float(len(audio_chunk)) / float(self.config.sample_rate or 1)
                    metadata = self._prepare_final_metadata(chunk_duration)
                    result = {
                        "type": "final",
                        "text": text,
                        "timestamp": current_time,
                        "is_final": True,
                        "model": f"parakeet-{self.config.model_variant}"
                    }
                    result.update(metadata)
                    result["_audio_chunk"] = np.array(audio_chunk, copy=True)
                    return result

        return None


class CanaryStreamingTranscriber(BaseStreamingTranscriber):
    """
    Canary-specific streaming transcriber.

    Supports multilingual transcription with language detection.
    """

    def initialize(self):
        """Load the Canary model."""
        logger.info("Loading Canary multilingual model")
        self.model = load_canary_model()
        if self.model is None:
            raise RuntimeError("Failed to load Canary model")
        logger.info("Loaded Canary model")

        # Set default language/task if not specified. For Canary we support the
        # same "transcribe" vs "translate" semantics as the offline helper.
        if not self.config.language:
            self.config.language = 'en'  # Default to English
        if not getattr(self.config, "task", None):
            self.config.task = "transcribe"

    async def process_audio_chunk(self, audio_data: bytes) -> Optional[dict[str, Any]]:
        """
        Process audio chunk with Canary.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Transcription result or None
        """
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.float32)

        # Add to buffer
        self.buffer.add(audio_np)

        current_time = time.time()
        buffer_duration = self.buffer.get_duration()

        # Check if we should send a partial result
        if (self.config.enable_partial and
            current_time - self.last_partial_time > self.config.partial_interval and
            buffer_duration > 0.5):

            # Get audio for partial transcription
            audio_for_partial = self.buffer.get_audio()
            if audio_for_partial is not None and len(audio_for_partial) > 0:
                # Transcribe partial audio. When task='translate', we request
                # English output to mirror the HTTP translation endpoint.
                text = transcribe_with_canary(
                    audio_for_partial,
                    self.config.sample_rate,
                    self.config.language,
                    task=self.config.task,
                    target_language="en" if self.config.task == "translate" else None,
                )

                self.last_partial_time = current_time

                if text:
                    # Apply custom vocabulary post-replacements if enabled
                    try:
                        from .Audio_Custom_Vocabulary import postprocess_text_if_enabled
                        text = postprocess_text_if_enabled(text)
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        pass
                    # Detect language if auto-detection is enabled
                    detected_language = self.config.language
                    if self.config.auto_detect_language:
                        # Simple heuristic - could be improved with actual language detection
                        detected_language = self._detect_language(text)

                    metadata = self._prepare_partial_metadata(buffer_duration)
                    result = {
                        "type": "partial",
                        "text": text,
                        "timestamp": current_time,
                        "is_final": False,
                        "language": detected_language,
                        "model": "canary-1b"
                    }
                    result.update(metadata)
                    return result

        # Check if we have enough audio for a final chunk
        if buffer_duration >= self.config.chunk_duration:
            # Get chunk for transcription
            audio_chunk = self.buffer.get_audio(self.config.chunk_duration)

            if audio_chunk is not None:
                # Transcribe the chunk (same task/target semantics as partials)
                text = transcribe_with_canary(
                    audio_chunk,
                    self.config.sample_rate,
                    self.config.language,
                    task=self.config.task,
                    target_language="en" if self.config.task == "translate" else None,
                )

                # Consume the buffer, keeping overlap
                self.buffer.consume(
                    self.config.chunk_duration,
                    self.config.overlap_duration
                )

                if text:
                    # Apply custom vocabulary post-replacements if enabled
                    try:
                        from .Audio_Custom_Vocabulary import postprocess_text_if_enabled
                        text = postprocess_text_if_enabled(text)
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        pass
                    # Detect language if auto-detection is enabled
                    detected_language = self.config.language
                    if self.config.auto_detect_language:
                        detected_language = self._detect_language(text)

                    self.transcription_history.append(text)
                    chunk_duration = float(len(audio_chunk)) / float(self.config.sample_rate or 1)
                    metadata = self._prepare_final_metadata(chunk_duration)
                    result = {
                        "type": "final",
                        "text": text,
                        "timestamp": current_time,
                        "is_final": True,
                        "language": detected_language,
                        "model": "canary-1b"
                    }
                    result.update(metadata)
                    result["_audio_chunk"] = np.array(audio_chunk, copy=True)
                    return result

        return None

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection heuristic.

        In production, this should use a proper language detection library.
        """
        # This is a placeholder - in reality, you'd use langdetect or similar
        # For now, just return the configured language
        return self.config.language


class WhisperStreamingTranscriber(BaseStreamingTranscriber):
    """
    Whisper-specific streaming transcriber using faster-whisper.

    Optimized for accuracy with configurable model sizes and features.
    """

    def initialize(self):
        """Load the Whisper model based on configuration."""
        logger.info(f"WhisperStreamingTranscriber.initialize() called with config: "
                   f"whisper_model_size={self.config.whisper_model_size}, "
                   f"language={self.config.language}, task={self.config.task}")

        try:
            # Prefer a module-level override (tests may monkeypatch this symbol on
            # Audio_Streaming_Unified) and fall back to the library import.
            get_whisper_model = globals().get('get_whisper_model')  # type: ignore[assignment]
            if get_whisper_model is None:
                logger.debug("Importing get_whisper_model from Audio_Transcription_Lib")
                from .Audio_Transcription_Lib import get_whisper_model  # type: ignore[no-redef]
                logger.debug("Successfully imported get_whisper_model")

            # Determine device and compute type. Handle partial torch stubs that
            # may not expose a cuda namespace in unit tests.
            device = 'cpu'
            try:
                torch_mod = globals().get("torch") or sys.modules.get("torch")
                cuda_mod = getattr(torch_mod, "cuda", None)
                is_available = getattr(cuda_mod, "is_available", None)
                if callable(is_available) and is_available():
                    device = 'cuda'
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                device = 'cpu'

            # Compute type is primarily determined by device but may be
            # overridden via [STT-Settings].whisper_compute_type when set to a
            # non-empty value other than "auto", mirroring the offline
            # speech_to_text path.
            _CT_OVERRIDE = globals().get("_WHISPER_COMPUTE_TYPE_OVERRIDE", "")  # type: ignore

            if _CT_OVERRIDE and str(_CT_OVERRIDE).strip().lower() != "auto":
                compute_type = str(_CT_OVERRIDE).strip()
            else:
                compute_type = 'float16' if device == 'cuda' else 'int8'

            logger.info(f"Loading Whisper model: {self.config.whisper_model_size} on {device} with compute_type: {compute_type}")

            # Load the model using existing function
            self.model = get_whisper_model(self.config.whisper_model_size, device)  # type: ignore[misc]

            if self.model is None:
                raise RuntimeError(f"Failed to load Whisper model: {self.config.whisper_model_size}")

            logger.info(f"Successfully loaded Whisper model: {self.config.whisper_model_size}, model object: {type(self.model)}")

            # Set transcription options
            self.transcribe_options = {
                'beam_size': self.config.beam_size,
                'best_of': self.config.beam_size,
                'vad_filter': self.config.vad_filter,
                'task': self.config.task
            }

            if self.config.language and not self.config.auto_detect_language:
                self.transcribe_options['language'] = self.config.language
            # Inject custom vocabulary initial prompt if configured
            try:
                from .Audio_Custom_Vocabulary import initial_prompt_if_enabled
                _init_prompt = initial_prompt_if_enabled()
                if _init_prompt:
                    self.transcribe_options['initial_prompt'] = _init_prompt
                    logger.info("Applied custom vocabulary initial_prompt for Whisper streaming")
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as _cv_err:
                logger.debug(f"Whisper streaming initial_prompt skipped: {_cv_err}")

            # Whisper works better with longer audio chunks
            self.min_chunk_duration = 1.0  # Minimum 1 second of audio
            self.optimal_chunk_duration = 5.0  # Optimal chunk size for Whisper

        except ImportError as e:
            logger.error(f"Failed to import Whisper dependencies: {e}")
            raise RuntimeError("Whisper dependencies not available") from e
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise

    async def process_audio_chunk(self, audio_data: bytes) -> Optional[dict[str, Any]]:
        """
        Process audio chunk with Whisper.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Transcription result or None
        """
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.float32)

        # Add to buffer
        self.buffer.add(audio_np)

        current_time = time.time()
        buffer_duration = self.buffer.get_duration()

        # Check if we should send a partial result
        # Whisper needs more audio for good results, so we wait for more data
        if (self.config.enable_partial and
            current_time - self.last_partial_time > self.config.partial_interval and
            buffer_duration >= self.min_chunk_duration):

            # Get audio for partial transcription
            audio_for_partial = self.buffer.get_audio()
            if audio_for_partial is not None and len(audio_for_partial) > 0:
                # Transcribe partial audio
                text = self._transcribe_audio(audio_for_partial)

                self.last_partial_time = current_time

                if text:
                    metadata = self._prepare_partial_metadata(buffer_duration)
                    result = {
                        "type": "partial",
                        "text": text,
                        "timestamp": current_time,
                        "is_final": False,
                        "model": f"whisper-{self.config.whisper_model_size}"
                    }
                    result.update(metadata)
                    return result

        # Check if we have enough audio for a final chunk
        # Use optimal chunk duration for better accuracy
        if buffer_duration >= self.optimal_chunk_duration:
            # Get chunk for transcription
            audio_chunk = self.buffer.get_audio(self.optimal_chunk_duration)

            if audio_chunk is not None:
                # Transcribe the chunk
                text = self._transcribe_audio(audio_chunk)

                # Consume the buffer, keeping overlap for context
                self.buffer.consume(
                    self.optimal_chunk_duration,
                    self.config.overlap_duration
                )

                if text:
                    self.transcription_history.append(text)
                    chunk_duration = float(len(audio_chunk)) / float(self.config.sample_rate or 1)
                    metadata = self._prepare_final_metadata(chunk_duration)
                    result = {
                        "type": "final",
                        "text": text,
                        "timestamp": current_time,
                        "is_final": True,
                        "model": f"whisper-{self.config.whisper_model_size}",
                        "language": self.config.language if self.config.language else "auto"
                    }
                    result.update(metadata)
                    result["_audio_chunk"] = np.array(audio_chunk, copy=True)
                    return result

        return None

    def _transcribe_audio(self, audio_np: np.ndarray) -> str:
        """
        Transcribe audio using Whisper model.

        Args:
            audio_np: Audio data as numpy array

        Returns:
            Transcribed text
        """
        try:
            # Resample to Whisper expected rate when necessary and avoid disk I/O
            audio_for_model = _resample_audio_if_needed(audio_np, self.config.sample_rate, target_sr=16000)

            segments_raw, info = self.model.transcribe(
                audio_for_model,
                **self.transcribe_options
            )

            text_parts = []
            for segment in segments_raw:
                text_parts.append(segment.text.strip())

            text = " ".join(text_parts)
            try:
                from .Audio_Custom_Vocabulary import postprocess_text_if_enabled
                text = postprocess_text_if_enabled(text)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                pass

            if self.config.auto_detect_language and hasattr(info, 'language'):
                logger.debug(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")

            return text

        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error during Whisper transcription: {e}")
            return ""


class Qwen3ASRStreamingTranscriber(BaseStreamingTranscriber):
    """
    Qwen3-ASR streaming transcriber using vLLM HTTP backend.

    Buffers audio chunks and sends them to an external vLLM server
    for transcription when chunk_duration is reached. Supports 30 languages
    with automatic language detection.
    """

    def __init__(self, config: UnifiedStreamingConfig):
        """Initialize Qwen3-ASR streaming transcriber."""
        super().__init__(config)
        self._vllm_base_url: str = ""
        self._httpx_client = None
        self._accumulated_audio: list[np.ndarray] = []

    def initialize(self):
        """Initialize the Qwen3-ASR transcriber with vLLM backend."""
        logger.info("Qwen3ASRStreamingTranscriber.initialize() called")

        # Get vLLM base URL from STT config
        try:
            from tldw_Server_API.app.core.config import get_stt_config
            stt_cfg = get_stt_config() or {}
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
            stt_cfg = {}

        self._vllm_base_url = str(stt_cfg.get("qwen3_asr_vllm_base_url", "")).strip()

        if not self._vllm_base_url:
            raise RuntimeError(
                "Qwen3-ASR streaming requires qwen3_asr_vllm_base_url to be configured. "
                "Set [STT-Settings].qwen3_asr_vllm_base_url in config.txt."
            )

        # Verify httpx is available
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise RuntimeError(
                "httpx is required for Qwen3-ASR streaming. Install with: pip install httpx"
            ) from None

        logger.info(f"Qwen3-ASR streaming initialized with vLLM at {self._vllm_base_url}")
        self.model = "qwen3-asr"  # Placeholder to indicate model is ready

    def reset(self):
        """Reset transcriber state and clear accumulated audio."""
        super().reset()
        self._accumulated_audio.clear()

    def cleanup(self):
        """Clean up resources."""
        self._accumulated_audio.clear()
        self._httpx_client = None
        super().cleanup()

    async def process_audio_chunk(self, audio_data: bytes) -> Optional[dict[str, Any]]:
        """
        Process audio chunk with Qwen3-ASR via vLLM HTTP.

        Buffers audio locally and sends to vLLM when chunk_duration is reached.

        Args:
            audio_data: Raw audio bytes (float32)

        Returns:
            Transcription result dict or None if not enough audio yet
        """
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_data, dtype=np.float32)

        # Add to our buffer and accumulated audio
        self.buffer.add(audio_np)
        self._accumulated_audio.append(audio_np.copy())

        current_time = time.time()
        buffer_duration = self.buffer.get_duration()

        # Check for partial results (send accumulated audio for transcription)
        if (
            self.config.enable_partial
            and current_time - self.last_partial_time > self.config.partial_interval
            and buffer_duration >= self.config.min_partial_duration
        ):
            # Combine accumulated audio
            if self._accumulated_audio:
                combined_audio = np.concatenate(self._accumulated_audio)
                text = await self._transcribe_via_vllm(combined_audio)

                self.last_partial_time = current_time

                if text:
                    metadata = self._prepare_partial_metadata(buffer_duration)
                    result = {
                        "type": "partial",
                        "text": text,
                        "timestamp": current_time,
                        "is_final": False,
                        "model": "qwen3-asr",
                    }
                    result.update(metadata)
                    return result

        # Check if we have enough audio for a final chunk
        if buffer_duration >= self.config.chunk_duration:
            # Get chunk from buffer
            audio_chunk = self.buffer.get_audio(self.config.chunk_duration)

            if audio_chunk is not None:
                # Transcribe using vLLM
                text = await self._transcribe_via_vllm(audio_chunk)

                # Consume the buffer, keeping overlap
                self.buffer.consume(
                    self.config.chunk_duration,
                    self.config.overlap_duration,
                )

                # Clear accumulated audio after successful final transcription
                # (keep overlap worth of audio for context)
                overlap_samples = int(self.config.overlap_duration * self.config.sample_rate)
                if overlap_samples > 0 and len(audio_chunk) > overlap_samples:
                    self._accumulated_audio = [audio_chunk[-overlap_samples:].copy()]
                else:
                    self._accumulated_audio.clear()

                if text:
                    self.transcription_history.append(text)
                    chunk_duration = float(len(audio_chunk)) / float(self.config.sample_rate or 1)
                    metadata = self._prepare_final_metadata(chunk_duration)
                    result = {
                        "type": "final",
                        "text": text,
                        "timestamp": current_time,
                        "is_final": True,
                        "model": "qwen3-asr",
                        "language": self.config.language if self.config.language else "auto",
                    }
                    result.update(metadata)
                    result["_audio_chunk"] = np.array(audio_chunk, copy=True)
                    return result

        return None

    async def _transcribe_via_vllm(self, audio_np: np.ndarray) -> str:
        """
        Transcribe audio by sending to vLLM HTTP endpoint.

        Args:
            audio_np: Audio data as numpy array (float32)

        Returns:
            Transcribed text
        """
        if not self._vllm_base_url:
            logger.error("vLLM base URL not configured")
            return ""

        try:
            import asyncio
            import tempfile

            # Write audio to temporary WAV file for HTTP upload
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            # Write WAV using soundfile or fallback
            try:
                import soundfile as sf
                sf.write(str(tmp_path), audio_np, self.config.sample_rate)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                # Fallback to wave module
                import wave
                pcm16 = np.clip(audio_np, -1.0, 1.0)
                pcm16 = (pcm16 * 32767.0).astype(np.int16)
                with wave.open(str(tmp_path), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.config.sample_rate)
                    wf.writeframes(pcm16.tobytes())

            url = f"{self._vllm_base_url.rstrip('/')}/v1/audio/transcriptions"

            # Use asyncio.to_thread for sync httpx client to avoid blocking
            def _do_http_request():
                with open(tmp_path, "rb") as f:
                    files = {"file": (tmp_path.name, f, "audio/wav")}
                    data = {"model": "qwen3-asr"}
                    if self.config.language:
                        data["language"] = self.config.language

                    with self._httpx.Client(timeout=60.0) as client:
                        response = client.post(url, files=files, data=data)
                        response.raise_for_status()
                        return response.json()

            result = await asyncio.to_thread(_do_http_request)

            # Clean up temp file
            with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                tmp_path.unlink(missing_ok=True)

            text = str(result.get("text", "")).strip()

            # Apply custom vocabulary post-processing if enabled
            try:
                from .Audio_Custom_Vocabulary import postprocess_text_if_enabled
                text = postprocess_text_if_enabled(text)
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                pass

            return text

        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as exc:
            logger.error(f"Qwen3-ASR vLLM transcription failed: {exc}")
            return ""


class UnifiedStreamingTranscriber:
    """
    Factory and unified interface for streaming transcribers.

    Automatically selects the appropriate transcriber based on configuration.
    """

    def __init__(self, config: UnifiedStreamingConfig):
        """Initialize unified transcriber."""
        self.config = config
        self.transcriber = None

    def initialize(self):
        """
        Selects and initializes the model-specific streaming transcriber based on this instance's configuration.

        For the Parakeet model, prefers the Parakeet Core adapter when available and falls back to the legacy Parakeet transcriber otherwise. After selection, the chosen transcriber is instantiated, its initialize method is called, and the transcriber instance is stored on self.transcriber.
        """
        model_lower = self.config.model.lower()

        if model_lower == 'canary':
            self.transcriber = CanaryStreamingTranscriber(self.config)
        elif model_lower == 'whisper':
            self.transcriber = WhisperStreamingTranscriber(self.config)
        elif model_lower in ('qwen3-asr', 'qwen3_asr', 'qwen3asr'):
            self.transcriber = Qwen3ASRStreamingTranscriber(self.config)
        else:  # Parakeet (default)
            # Prefer the core adapter when available; fall back to legacy transcriber
            if _ParakeetCoreAdapter is not None:
                self.transcriber = _ParakeetCoreAdapter(self.config)  # type: ignore
            else:
                self.transcriber = ParakeetStreamingTranscriber(self.config)

        self.transcriber.initialize()
        logger.info(f"Initialized {self.config.model} transcriber")

    async def process_audio_chunk(self, audio_data: bytes) -> Optional[dict[str, Any]]:
        """Process audio chunk with selected transcriber."""
        if not self.transcriber:
            raise RuntimeError("Transcriber not initialized")
        return await self.transcriber.process_audio_chunk(audio_data)

    def get_full_transcript(self) -> str:
        """Get complete transcript."""
        if not self.transcriber:
            return ""
        return self.transcriber.get_full_transcript()

    def reset(self):
        """Reset transcriber state."""
        if self.transcriber:
            self.transcriber.reset()

    def cleanup(self):
        """Clean up resources."""
        if self.transcriber:
            self.transcriber.cleanup()
        self.transcriber = None

# Explicit export list to aid test imports and static analyzers
__all__ = [
    'UnifiedStreamingConfig',
    'BaseStreamingTranscriber',
    'WhisperStreamingTranscriber',
    'CanaryStreamingTranscriber',
    'ParakeetStreamingTranscriber',
    'Qwen3ASRStreamingTranscriber',
    'UnifiedStreamingTranscriber',
    'SileroTurnDetector',
]


def _clamp_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    """Clamp a float-like value to the provided bounds with a safe fallback."""
    try:
        return max(min_value, min(max_value, float(value)))
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        return default


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    """Clamp an int-like value to the provided bounds with a safe fallback."""
    try:
        return int(max(min_value, min(max_value, int(value))))
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        return default


async def handle_unified_websocket(
    websocket,
    config: Optional[UnifiedStreamingConfig] = None,
    on_audio_seconds: Optional[Callable[[float, int], Awaitable[None]]] = None,
    on_heartbeat: Optional[Callable[[], Awaitable[None]]] = None,
    on_stream_config_resolved: Optional[Callable[[dict[str, Any], UnifiedStreamingConfig], Awaitable[None]]] = None,
    on_transcript_result: Optional[Callable[[dict[str, Any], str], Awaitable[None]]] = None,
    on_full_transcript: Optional[Callable[[str, bool], Awaitable[None]]] = None,
):
    """
    Handle a WebSocket connection to perform unified real-time transcription across Parakeet, Canary, Whisper, and Qwen3-ASR models.

    This handler waits for an optional client configuration message, initializes a model-specific transcriber (with runtime fallback logic), and processes incoming base64-encoded audio messages into partial and final transcription results sent to the client as JSON frames. It optionally integrates streaming diarization and live insights, emits structured status/warning/error frames, supports a commit/reset/stop control messages, and ensures proper cleanup of transcriber, diarizer, and insights resources on exit.

    Parameters:
        websocket: The WebSocket connection object used to receive client messages and send JSON frames.
        config (Optional[UnifiedStreamingConfig]): Optional initial streaming configuration; updated if a client config message is received.
        on_audio_seconds (Optional[Callable[[float, int], Awaitable[None]]]): Optional callback invoked before processing each audio chunk with two arguments: computed audio duration in seconds and the sample rate in Hz.
        on_heartbeat (Optional[Callable[[], Awaitable[None]]]): Optional callback invoked on each received audio message to refresh external TTLs (e.g., Redis-based heartbeats).
        on_stream_config_resolved (Optional[Callable[[dict[str, Any], UnifiedStreamingConfig], Awaitable[None]]]): Optional callback invoked after config handling resolves, with raw config payload and resolved config object.
        on_transcript_result (Optional[Callable[[dict[str, Any], str], Awaitable[None]]]): Optional callback invoked for each emitted partial/final transcript result, with the outgoing result payload and current full transcript snapshot.
        on_full_transcript (Optional[Callable[[str, bool], Awaitable[None]]]): Optional callback invoked whenever the handler emits a full transcript (manual/auto commit).
    """
    logger.info("=== handle_unified_websocket STARTED ===")

    # Wrap the WebSocket with standardized lifecycle (ping/error/done) and metrics.
    # Optional idle timeout for WS via env (tests/ops); default None leaves it off here
    try:
        _raw_idle = os.getenv("AUDIO_WS_IDLE_TIMEOUT_S") or os.getenv("STREAM_IDLE_TIMEOUT_S")
        _idle_timeout = float(_raw_idle) if _raw_idle else None
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
        _idle_timeout = None

    stream = WebSocketStream(
        websocket,
        heartbeat_interval_s=None,  # use env default
        compat_error_type=is_truthy(os.getenv("AUDIO_WS_COMPAT_ERROR_TYPE", "1")),
        close_on_done=True,
        idle_timeout_s=_idle_timeout,
        labels={"component": "audio", "endpoint": "audio_unified_ws"},
    )
    await stream.start()
    # Ensure downstream helpers using the raw websocket route sends through the stream where possible
    # Do not monkeypatch websocket.send_json; endpoints may rely on specific semantics

    if not config:
        config = UnifiedStreamingConfig()
        logger.info("Created default config")
    else:
        logger.info(f"Received config from caller: model={config.model}, variant={config.model_variant}")

    logger.info(f"Initial config: model={config.model}, variant={config.model_variant}")
    transcriber = None  # Initialize transcriber after config is set
    insights_settings: Optional[LiveInsightSettings] = None
    insights_engine: Optional[LiveMeetingInsights] = None
    diarizer: Optional[StreamingDiarizer] = None
    turn_detector: Optional[SileroTurnDetector] = None
    vad_status = "disabled"
    diarization_status = "disabled"
    diarization_details: dict[str, Any] | None = None
    control_session = WSControlSession(_get_ws_control_protocol_config())
    paused_audio_chunks: deque[tuple[bytes, float]] = deque()
    vad_warning_sent = False
    transcript_dirty = False
    full_transcript_emitted = False

    try:
        # Always wait for configuration message from client
        config_received = False
        config_payload: dict[str, Any] = {}
        try:
            logger.info("Waiting for configuration message from client...")
            config_message = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)  # Increased timeout
            # Do not log raw payload contents (may include base64 audio); log metadata only
            logger.info(f"Received message (length={len(config_message)})")
            config_data = json.loads(config_message)
            if isinstance(config_data, dict):
                config_payload = config_data
            logger.info(f"Parsed config data type: {config_data.get('type')}")

            if config_data.get("type") == "config":
                # Update configuration
                old_variant = config.model_variant
                raw_model = config_data.get("model", "parakeet")
                variant_override = config_data.get("variant") or config_data.get("model_variant")
                new_model, new_variant = normalize_model_and_variant(
                    raw_model,
                    current_model=config.model,
                    current_variant=config.model_variant,
                    variant_override=variant_override,
                )
                config.model = new_model
                config.model_variant = new_variant
                config.language = config_data.get("language", "en")
                config.sample_rate = config_data.get("sample_rate", 16000)
                config.auto_detect_language = config_data.get("auto_detect_language", False)
                config.chunk_duration = config_data.get("chunk_duration", 2.0)
                config.enable_partial = config_data.get("enable_partial", True)
                raw_vad = config_data.get("enable_vad", config.enable_vad)
                if isinstance(raw_vad, str):
                    raw_vad = is_truthy(raw_vad)
                elif raw_vad is not None and not isinstance(raw_vad, (bool, int)):
                    logger.debug(
                        f"Unexpected type for enable_vad in config: {type(raw_vad).__name__}; "
                        f"coercing to bool(raw_vad)={bool(raw_vad)}"
                    )
                config.enable_vad = bool(raw_vad)
                config.vad_threshold = _clamp_float(
                    config_data.get("vad_threshold", config.vad_threshold),
                    default=config.vad_threshold,
                    min_value=0.1,
                    max_value=0.9,
                )
                config.vad_min_silence_ms = _clamp_int(
                    config_data.get("min_silence_ms", config.vad_min_silence_ms),
                    default=config.vad_min_silence_ms,
                    min_value=150,
                    max_value=1500,
                )
                config.vad_turn_stop_secs = _clamp_float(
                    config_data.get("turn_stop_secs", config.vad_turn_stop_secs),
                    default=config.vad_turn_stop_secs,
                    min_value=0.1,
                    max_value=0.75,
                )
                if "min_utterance_secs" in config_data:
                    config.vad_min_utterance_secs = _clamp_float(
                        config_data.get("min_utterance_secs", config.vad_min_utterance_secs),
                        default=config.vad_min_utterance_secs,
                        min_value=0.25,
                        max_value=3.0,
                    )
                # High-level task hint used by Whisper and Canary; defaults to
                # "transcribe" when not provided.
                try:
                    raw_task = str(config_data.get("task", config.task or "transcribe")).strip().lower()
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    raw_task = "transcribe"
                config.task = raw_task if raw_task in {"transcribe", "translate"} else "transcribe"
                config.parakeet_use_rnnt_streamer = config_data.get("parakeet_use_rnnt_streamer", config.parakeet_use_rnnt_streamer)
                config.parakeet_rnnt_model_name = config_data.get("parakeet_rnnt_model_name", config.parakeet_rnnt_model_name)
                config.parakeet_rnnt_device = config_data.get("parakeet_rnnt_device", config.parakeet_rnnt_device)
                config.parakeet_rnnt_left_context_s = float(config_data.get("parakeet_rnnt_left_context_s", config.parakeet_rnnt_left_context_s))
                config.parakeet_rnnt_max_buffer_s = float(config_data.get("parakeet_rnnt_max_buffer_s", config.parakeet_rnnt_max_buffer_s))
                # Optional partial emission tuning
                try:
                    if "min_partial_duration" in config_data:
                        config.min_partial_duration = max(0.0, float(config_data.get("min_partial_duration")))
                except (TypeError, ValueError):
                    logger.warning("Invalid min_partial_duration in config; keeping previous value")

                # Whisper-specific configuration
                if config.model.lower() == "whisper":
                    config.whisper_model_size = config_data.get("whisper_model_size", "distil-large-v3")
                    config.beam_size = config_data.get("beam_size", 5)
                    config.vad_filter = config_data.get("vad_filter", False)
                    config.task = config_data.get("task", "transcribe")

                insights_payload = config_data.get("insights") or config_data.get("meeting_insights")
                if insights_payload is not None:
                    try:
                        insights_settings = LiveInsightSettings.from_client_payload(insights_payload)
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as insight_err:
                        logger.error(f"Failed to parse live insights config: {insight_err}")
                        insights_settings = LiveInsightSettings(enabled=False)
                elif config_data.get("insights_enabled") is True and insights_settings is None:
                    insights_settings = LiveInsightSettings(enabled=True)
                elif config_data.get("insights_enabled") is False:
                    insights_settings = LiveInsightSettings(enabled=False)

                diarization_payload = config_data.get("diarization")
                if diarization_payload is not None:
                    enabled_field = diarization_payload.get("enabled")
                    config.diarization_enabled = bool(enabled_field) if enabled_field is not None else True
                    if "store_audio" in diarization_payload:
                        config.diarization_store_audio = bool(diarization_payload.get("store_audio"))
                    storage_dir = diarization_payload.get("storage_dir")
                    if storage_dir:
                        safe_dir = _safe_temp_subdir(storage_dir)
                        if safe_dir:
                            config.diarization_storage_dir = str(safe_dir)
                        else:
                            logger.warning("Ignoring diarization.storage_dir from client; using temp directory only")
                            config.diarization_storage_dir = None
                    if "num_speakers" in diarization_payload:
                        try:
                            config.diarization_num_speakers = int(diarization_payload.get("num_speakers") or 0) or None
                        except (TypeError, ValueError):
                            logger.warning("Invalid diarization.num_speakers value; ignoring.")
                elif "diarization_enabled" in config_data:
                    config.diarization_enabled = bool(config_data.get("diarization_enabled"))
                if "diarization_store_audio" in config_data:
                    config.diarization_store_audio = bool(config_data.get("diarization_store_audio"))

                logger.info(f"Config updated: model={config.model}, variant changed from {old_variant} to {config.model_variant}, "
                           f"sample_rate={config.sample_rate}, chunk_duration={config.chunk_duration}")
                config_received = True

                # Prepare acknowledgment (do not send to keep protocol noise-free for tests)
                status_msg = {
                    "type": "status",
                    "state": "configured",
                    "model": config.model
                }

                if config.model.lower() == "parakeet":
                    status_msg["variant"] = config.model_variant
                elif config.model.lower() == "canary":
                    status_msg["language"] = config.language
                elif config.model.lower() == "whisper":
                    status_msg["whisper_model"] = config.whisper_model_size
                    status_msg["task"] = config.task
                    status_msg["language"] = config.language if config.language else "auto"

                # Intentionally not sending status frame; log only
                logger.info(f"Config acknowledged (not sent to client): {status_msg}")
            else:
                # Do not log full payload to avoid dumping base64 audio
                msg_type = config_data.get('type')
                data_len = len(config_data.get('data', '')) if isinstance(config_data.get('data'), str) else 0
                logger.warning(f"Received non-config message type: {msg_type} (payload length ~{data_len})")
        except asyncio.TimeoutError:
            logger.warning(f"Config message timeout after 15s. Using default configuration: model={config.model}, variant={config.model_variant}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse config message as JSON: {e}")
            logger.warning("Using default configuration due to JSON parse error")
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as e:
            logger.exception("Unexpected error receiving config message: {}", e)
            logger.warning("Using default configuration due to error")

        if not config_received:
            logger.warning(f"No valid config received. Proceeding with: model={config.model}, variant={config.model_variant}")

        if on_stream_config_resolved is not None:
            try:
                await on_stream_config_resolved(config_payload, config)
            except Exception as cb_exc:  # noqa: BLE001 - callback failures must not break streaming
                logger.debug(f"on_stream_config_resolved callback failed: {cb_exc}")

        protocol_decision = control_session.apply_config(config_payload if config_received else None)
        for event in protocol_decision.events:
            await stream.send_json(event)

        # Create transcriber with config
        if transcriber is None:
            logger.info(f"Creating UnifiedStreamingTranscriber for model: {config.model}")
            transcriber = UnifiedStreamingTranscriber(config)

        try:
            logger.info(f"Initializing transcriber for model: {config.model}")
            logger.info(f"Configuration details: model_variant={config.model_variant}, "
                       f"whisper_model_size={getattr(config, 'whisper_model_size', 'N/A')}, "
                       f"sample_rate={config.sample_rate}, language={config.language}")
            transcriber.initialize()
            logger.info(f"Transcriber initialized successfully for model: {config.model}")
            if config.enable_vad:
                vad_status = "fail_open"
                turn_detector = SileroTurnDetector(
                    sample_rate=config.sample_rate,
                    enabled=True,
                    vad_threshold=config.vad_threshold,
                    min_silence_ms=config.vad_min_silence_ms,
                    turn_stop_secs=config.vad_turn_stop_secs,
                    min_utterance_secs=config.vad_min_utterance_secs,
                )
                if not turn_detector.available and not vad_warning_sent:
                    vad_warning_sent = True
                    logger.warning(
                        f"Silero VAD unavailable ({turn_detector.unavailable_reason}); continuing without auto-commit"
                    )
                    turn_detector = None
                else:
                    vad_status = "enabled"
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as e:
            error_msg = f"Failed to initialize {config.model} model: {str(e)}"
            safe_model_error = "Streaming model initialization failed"
            logger.exception(error_msg)
            # Emit structured warning about model/variant unavailability before fallback attempts
            with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                await stream.send_json({
                    "type": "warning",
                    "state": "model_unavailable",
                    "error_type": "model_unavailable",
                    "message": "Requested model/variant unavailable; attempting fallback if enabled",
                    "details": {
                        "model": config.model,
                        "variant": getattr(config, 'model_variant', None),
                        "error": safe_model_error,
                    },
                })

            # Check if fallback to Whisper is enabled in config (module-level alias for test monkeypatching).
            # Default behavior is fail-fast unless explicitly enabled.
            comprehensive_config = load_comprehensive_config()
            fallback_enabled = False
            try:
                if comprehensive_config is not None and hasattr(comprehensive_config, "has_section"):
                    if comprehensive_config.has_section('STT-Settings'):
                        fallback_value = comprehensive_config.get(
                            'STT-Settings',
                            'streaming_fallback_to_whisper',
                            fallback='false',
                        )
                        fallback_enabled = is_truthy(str(fallback_value))
                    else:
                        logger.info(
                            "No [STT-Settings] section found in config; "
                            "defaulting streaming_fallback_to_whisper=false. "
                            "Set [STT-Settings].streaming_fallback_to_whisper=true to opt in."
                        )
                elif isinstance(comprehensive_config, dict):
                    stt_section = comprehensive_config.get("STT-Settings", {}) or {}
                    fallback_value = stt_section.get("streaming_fallback_to_whisper", False)
                    fallback_enabled = is_truthy(str(fallback_value))
                logger.info(f"Streaming fallback to Whisper enabled: {fallback_enabled}")
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as config_error:
                logger.warning(
                    "Could not read streaming_fallback_to_whisper from config; "
                    "defaulting to fail-fast (fallback disabled). "
                    f"Error: {config_error}. To opt in, set [STT-Settings].streaming_fallback_to_whisper=true."
                )
                fallback_enabled = False

            # Try to fall back to Whisper if enabled and not already using Whisper
            if fallback_enabled and config.model.lower() != 'whisper':
                logger.info("Fallback to Whisper is enabled in config. Attempting to fall back...")
                try:
                    original_model = config.model
                    config.model = 'whisper'
                    config.whisper_model_size = 'distil-large-v3'
                    transcriber = UnifiedStreamingTranscriber(config)
                    transcriber.initialize()
                    logger.info("Successfully fell back to Whisper model")

                    # Notify client about fallback
                    await stream.send_json({
                        "type": "warning",
                        "message": f"{original_model} model unavailable, using Whisper instead",
                        "fallback": True,
                        "original_model": original_model,
                        "active_model": "whisper"
                    })
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as fallback_error:
                    safe_fallback_error = "Streaming fallback initialization failed"
                    logger.error(f"Fallback to Whisper also failed: {fallback_error}")
                    # Send standardized error and close with mapped code (1011)
                    await stream.error(
                        "provider_error",
                        "No transcription models available. Please install required dependencies.",
                        data={
                            "model": config.model,
                            "variant": getattr(config, 'model_variant', None),
                            "original_error": safe_model_error,
                            "fallback_error": safe_fallback_error,
                            "suggestion": "Install nemo_toolkit[asr] for Parakeet/Canary or ensure faster-whisper is installed",
                        },
                    )
                    return
            else:
                # Fallback disabled or already using Whisper: emit explicit model_unavailable error
                suggestion = ""
                if config.model.lower() in ['parakeet', 'canary']:
                    suggestion = "Install nemo_toolkit[asr]: pip install nemo_toolkit[asr]"
                elif config.model.lower() == 'whisper':
                    suggestion = "Ensure faster-whisper is installed: pip install faster-whisper"

                # Standardized error with compatibility field 'error_type' via compat_error_type=True
                await stream.error(
                    "model_unavailable",
                    "Requested model/variant unavailable and fallback disabled",
                    data={
                        "model": config.model,
                        "variant": getattr(config, 'model_variant', None),
                        "error": safe_model_error,
                        "fallback_enabled": fallback_enabled,
                        "suggestion": suggestion,
                    },
                )
                return

        if diarizer is None and config.diarization_enabled:
            try:
                diarizer = StreamingDiarizer(
                    sample_rate=config.sample_rate,
                    store_audio=config.diarization_store_audio,
                    storage_dir=config.diarization_storage_dir,
                    num_speakers=config.diarization_num_speakers,
                )
                ready = await diarizer.ensure_ready()
                if not ready:
                    diarization_status = "unavailable"
                    diarization_details = {
                        "code": "init_unavailable",
                        "summary": "Diarization disabled: dependencies missing or initialization failed",
                    }
                    logger.warning("Streaming diarizer unavailable during initialization; disabling diarization.")
                    await stream.send_json({
                        "type": "warning",
                        "state": "diarization_unavailable",
                        "message": "Diarization disabled: dependencies missing or initialization failed",
                        "details": getattr(diarizer, "_service_error", None),
                    })
                    diarizer = None
                else:
                    diarization_status = "enabled"
                    diarization_details = None
                    await stream.send_json({
                        "type": "status",
                        "state": "diarization_enabled",
                        "diarization": {
                            "store_audio": config.diarization_store_audio,
                            "storage_dir": config.diarization_storage_dir,
                            "num_speakers": config.diarization_num_speakers,
                        },
                    })
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as diar_err:
                diarization_status = "unavailable"
                diarization_details = {
                    "code": "init_failed",
                    "summary": "Diarization disabled: initialization failed",
                }
                logger.exception("Failed to initialize streaming diarizer: {}", diar_err)
                await stream.send_json({
                    "type": "warning",
                    "state": "diarization_unavailable",
                    "message": "Diarization disabled: initialization failed",
                    "details": "Diarization initialization failed",
                })
                diarizer = None

        if insights_engine is None and insights_settings and insights_settings.enabled:
            try:
                insights_engine = LiveMeetingInsights(websocket, insights_settings)
                logger.info(
                    f"Live insights enabled (provider={insights_engine.provider}, model={insights_engine.model})"
                )
                await stream.send_json({
                    "type": "status",
                    "state": "insights_enabled",
                    "insights": insights_engine.describe()
                })
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as insight_err:
                logger.exception("Failed to initialize live insights engine: {}", insight_err)
                await stream.send_json({
                    "type": "warning",
                    "state": "insights_unavailable",
                    "message": "Live insights disabled: initialization failed",
                    "details": "Live insights initialization failed"
                })
                insights_engine = None
        elif insights_settings and not insights_settings.enabled:
            logger.info("Live insights explicitly disabled for this session.")

        # Do not send a ready status frame to minimize protocol chatter

        async def _emit_full_transcript(commit_received_at: Optional[float], *, auto_commit: bool = False) -> None:
            """
            Emit the full transcript and related artifacts, mirroring the manual commit path.

            Args:
                commit_received_at: Timestamp when the commit (manual or auto) was triggered.
                auto_commit: Whether the emission was triggered by VAD turn detection.
            """
            nonlocal transcript_dirty, full_transcript_emitted
            if transcriber is None:
                return
            full_transcript = transcriber.get_full_transcript()
            _final_emit_at = time.time()
            _eos_detected_at = _final_emit_at
            try:
                _commit_ts = float(commit_received_at) if commit_received_at is not None else None
                if _commit_ts and _commit_ts > 0:
                    _eos_detected_at = _commit_ts
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                _eos_detected_at = _final_emit_at
            transcript_diarization_status = diarization_status
            transcript_diarization_details = copy.deepcopy(diarization_details)
            diarization_followup_frames: list[dict[str, Any]] = []
            # Record STT finalization latency metric (commit → final emit)
            try:
                from tldw_Server_API.app.core.Metrics import get_metrics_registry
                reg = get_metrics_registry()
                # Determine model/variant labels when available
                _model = getattr(config, "model", None) or "parakeet"
                _variant = getattr(config, "model_variant", None) or "standard"
                reg.observe(
                    "stt_final_latency_seconds",
                    max(0.0, _final_emit_at - _eos_detected_at),
                    labels={"model": str(_model), "variant": str(_variant), "endpoint": "audio_unified_ws"},
                )
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                pass
            if insights_engine:
                try:
                    await insights_engine.on_commit(full_transcript)
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as insight_err:
                    logger.exception("Live insights final summary failed: {}", insight_err)
            if diarizer:
                try:
                    mapping, audio_path, speakers = await diarizer.finalize()
                    if mapping or audio_path or speakers:
                        speaker_map = [
                            {
                                "segment_id": seg_id,
                                "speaker_id": info.get("speaker_id"),
                                "speaker_label": info.get("speaker_label"),
                            }
                            for seg_id, info in sorted(mapping.items())
                        ]
                        diarization_followup_frames.append({
                            "type": "diarization_summary",
                            "speaker_map": speaker_map,
                            "audio_path": audio_path,
                            "speakers": speakers,
                            "persistence_method": getattr(diarizer, "persistence_method", None),
                        })
                    # Emit structured warning when persistence requested but unavailable
                    try:
                        if (
                            config.diarization_store_audio
                            and (audio_path is None or not audio_path)
                        ):
                            diarization_followup_frames.append({
                                "type": "warning",
                                "warning_type": "audio_persistence_unavailable",
                                "message": "Audio persistence was requested but is unavailable; continuing without persisted WAV",
                            })
                        # Emit detailed status for persistence state
                        if config.diarization_store_audio:
                            _method = getattr(diarizer, "persistence_method", None)
                            if audio_path and _method and _method != "soundfile":
                                transcript_diarization_details = {
                                    "code": "persist_degraded",
                                    "summary": "Audio persisted using degraded fallback method",
                                }
                                diarization_followup_frames.append({
                                    "type": "status",
                                    "state": "diarization_persist_degraded",
                                    "persistence_method": _method,
                                })
                            elif (not audio_path) or (_method is None):
                                transcript_diarization_details = {
                                    "code": "persist_disabled",
                                    "summary": "Audio persistence unavailable for this session",
                                }
                                diarization_followup_frames.append({
                                    "type": "status",
                                    "state": "diarization_persist_disabled",
                                    "persistence_method": _method,
                                })
                    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        pass
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as diar_err:
                    transcript_diarization_status = "unavailable"
                    transcript_diarization_details = {
                        "code": "finalize_failed",
                        "summary": "Diarization finalize failed",
                    }
                    logger.exception("Diarization finalize failed: {}", diar_err)

            payload = {
                "type": "full_transcript",
                "text": full_transcript,
                "timestamp": _final_emit_at,
                # EOS/turn-end anchor clients can thread into downstream TTS.
                "voice_to_voice_start": _eos_detected_at,
            }
            payload.update(
                _build_transcript_diagnostics(
                    auto_commit=auto_commit,
                    vad_status=vad_status,
                    diarization_status=transcript_diarization_status,
                    diarization_details=transcript_diarization_details,
                )
            )
            if on_full_transcript is not None:
                try:
                    await on_full_transcript(full_transcript, auto_commit)
                except Exception as cb_exc:  # noqa: BLE001 - callback failures must not break streaming
                    logger.debug(f"on_full_transcript callback failed: {cb_exc}")
            await stream.send_json(payload)
            full_transcript_emitted = True
            transcript_dirty = False
            for frame in diarization_followup_frames:
                await stream.send_json(frame)

        async def _process_audio_chunk(audio_bytes: bytes) -> bool:
            nonlocal transcript_dirty
            if transcriber is None:
                return True

            result = await transcriber.process_audio_chunk(audio_bytes)
            transcript_dirty = True
            if not result:
                return True

            # Detect STT error sentinels so they do not leak as user text.
            text_field = result.get("text")
            if isinstance(text_field, str) and _is_transcription_error_message(text_field):
                logger.error(f"Unified streaming STT error sentinel: {text_field}")
                await stream.error(
                    "provider_error",
                    "Transcription error from STT provider",
                    data={
                        "model": getattr(config, "model", None),
                        "variant": getattr(config, "model_variant", None),
                        "language": getattr(config, "language", None),
                        "raw_error": "Transcription provider returned an error",
                    },
                )
                return False

            audio_np = result.pop("_audio_chunk", None)
            if audio_np is not None and diarizer:
                try:
                    speaker_info = await diarizer.label_segment(
                        audio_np,
                        {
                            "segment_id": result.get("segment_id"),
                            "segment_start": result.get("segment_start"),
                            "segment_end": result.get("segment_end"),
                            "chunk_start": result.get("chunk_start"),
                            "chunk_end": result.get("chunk_end"),
                            "text": result.get("text"),
                        },
                    )
                    if speaker_info:
                        if speaker_info.get("speaker_id") is not None:
                            result.setdefault("speaker_id", speaker_info["speaker_id"])
                        if speaker_info.get("speaker_label"):
                            result.setdefault("speaker_label", speaker_info["speaker_label"])
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as diar_err:
                    logger.exception("Diarization update failed: {}", diar_err)

            if on_transcript_result is not None:
                try:
                    full_snapshot = transcriber.get_full_transcript() if transcriber else ""
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    full_snapshot = ""
                try:
                    await on_transcript_result(result, full_snapshot)
                except Exception as cb_exc:  # noqa: BLE001 - callback failures must not break streaming
                    logger.debug(f"on_transcript_result callback failed: {cb_exc}")

            await stream.send_json(result)
            if insights_engine and result.get("is_final"):
                try:
                    await insights_engine.on_transcript(result)
                except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as insight_err:
                    logger.exception("Live insights failed to ingest segment: {}", insight_err)
            return True

        # Process messages
        while True:
            try:
                message = await websocket.receive_text()
                with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                    stream.mark_activity()
                data = json.loads(message)

                if data.get("type") == "audio":
                    # Decode audio data
                    audio_base64 = data.get("data", "")
                    audio_bytes = base64.b64decode(audio_base64)
                    # Optional heartbeat to refresh stream TTL when using Redis counters
                    if on_heartbeat is not None:
                        with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                            await on_heartbeat()

                    if control_session.state == "paused":
                        buffered_seconds = _estimate_audio_seconds(audio_bytes, int(config.sample_rate or 16000))
                        paused_audio_chunks.append((audio_bytes, buffered_seconds))
                        paused_audio_decision = control_session.buffer_paused_audio(
                            buffered_seconds,
                            now=time.time(),
                        )
                        if paused_audio_decision.dropped_seconds > 0:
                            _drop_oldest_buffered_audio(paused_audio_chunks, paused_audio_decision.dropped_seconds)
                        for event in paused_audio_decision.events:
                            await stream.send_json(event)
                        continue

                    # Bill usage seconds only for audio that will be processed (not paused/dropped)
                    if on_audio_seconds is not None:
                        seconds = _estimate_audio_seconds(audio_bytes, int(config.sample_rate or 16000))
                        await on_audio_seconds(seconds, int(config.sample_rate or 16000))

                    auto_commit_triggered = False
                    if turn_detector:
                        auto_commit_triggered = turn_detector.observe(audio_bytes)
                        if not turn_detector.available and not vad_warning_sent:
                            vad_warning_sent = True
                            vad_status = "fail_open"
                            logger.warning(
                                f"Silero VAD disabled mid-stream ({turn_detector.unavailable_reason}); continuing without auto-commit"
                            )
                            turn_detector = None
                    if not await _process_audio_chunk(audio_bytes):
                        return
                    if auto_commit_triggered:
                        await _emit_full_transcript(
                            commit_received_at=getattr(turn_detector, "last_trigger_at", None),
                            auto_commit=True,
                        )

                elif data.get("type") in {"control", "commit", "reset", "stop"}:
                    decision = control_session.handle_frame(data)
                    if decision.error:
                        await stream.send_json(decision.error)
                        continue

                    for event in decision.events:
                        await stream.send_json(event)

                    if decision.intent == "resume":
                        queued_audio = list(paused_audio_chunks)
                        paused_audio_chunks.clear()
                        control_session.release_paused_audio()
                        for buffered_audio, _buffered_seconds in queued_audio:
                            if not await _process_audio_chunk(buffered_audio):
                                return

                    if decision.should_reset:
                        paused_audio_chunks.clear()
                        transcriber.reset()
                        transcript_dirty = False
                        full_transcript_emitted = False
                        if insights_engine:
                            try:
                                await insights_engine.reset()
                            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as insight_err:
                                logger.exception("Live insights reset failed: {}", insight_err)
                        if diarizer:
                            try:
                                await diarizer.reset()
                            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as diar_err:
                                logger.exception("Diarization reset failed: {}", diar_err)

                    if decision.should_emit_full_transcript:
                        if transcript_dirty or not full_transcript_emitted:
                            await _emit_full_transcript(time.time(), auto_commit=False)

                    if decision.should_close:
                        paused_audio_chunks.clear()
                        with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                            await stream.done()
                        break

            except json.JSONDecodeError:
                await stream.error("validation_error", "Invalid JSON message")
            except QuotaExceeded as qe:
                # Emit a single standardized error and close via the stream abstraction.
                _quota = getattr(qe, "quota", "daily_minutes")
                with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                    await stream.error("quota_exceeded", "Streaming transcription quota exceeded", data={"quota": _quota})
                return
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as e:
                if isinstance(e, WebSocketDisconnect):
                    # Let disconnect bubble to the outer handler for graceful shutdown
                    raise
                logger.error(f"Error processing message: {e}")
                await stream.error("internal_error", "Streaming audio processing failed")

    except asyncio.TimeoutError:
        await stream.error("idle_timeout", "Configuration timeout")
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"WebSocket handler error: {e}")
        try:
            await stream.error("internal_error", "Streaming server error")
        except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as send_err:
            logger.debug(f"Failed to send error frame on websocket: error={send_err}")
    finally:
        # Clean up
        if transcriber:
            transcriber.cleanup()
        if insights_engine:
            try:
                await insights_engine.close()
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as insight_err:
                logger.error(f"Failed to close live insights engine: {insight_err}")
        if diarizer:
            try:
                await diarizer.close()
            except _AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS as diar_err:
                logger.error(f"Failed to close diarizer: {diar_err}")
        with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
            await stream.stop()


# Export main components
__all__ = [
    'UnifiedStreamingConfig',
    'BaseStreamingTranscriber',
    'ParakeetStreamingTranscriber',
    'CanaryStreamingTranscriber',
    'WhisperStreamingTranscriber',
    'Qwen3ASRStreamingTranscriber',
    'UnifiedStreamingTranscriber',
    'SileroTurnDetector',
    'QuotaExceeded',
    'handle_unified_websocket'
]

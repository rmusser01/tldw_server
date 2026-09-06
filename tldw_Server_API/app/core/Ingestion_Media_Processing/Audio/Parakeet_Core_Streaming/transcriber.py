"""
Parakeet Core Streaming Transcriber
----------------------------------

Self-contained streaming transcriber that:
- Buffers float32 mono audio
- Emits partial results on a cadence
- Emits final results when `chunk_duration` is reached
- Computes robust segment metadata (segment_id, starts/ends, overlap)

The transcriber accepts an optional `decode_fn(audio_np, sample_rate) -> str`
so it can wrap any Parakeet decode implementation. If omitted, it will
attempt to use the local Nemo-based `transcribe_with_parakeet` function.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from loguru import logger

from .buffer import AudioBuffer
from .config import StreamingConfig

DecodeFn = Callable[[np.ndarray, int], str]
_PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ConnectionError,
    ImportError,
    LookupError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeError,
    ValueError,
    binascii.Error,
)


def _variant_decode_fn(model: str, variant: str) -> DecodeFn | None:
    """Build a decode function for the requested model/variant.

    This core supports only Parakeet variants. For:
    - standard: Nemo Parakeet TDT (greedy)
    - onnx:     ONNX session with mel features + tokenizer (chunk merge inside impl)
    - mlx:      Apple Silicon MLX path; handles ndarray or temp wav as needed
    """
    if str(model or "").lower() != "parakeet":
        return None
    v = str(variant or "standard").lower()

    if v == "onnx":
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
                transcribe_with_parakeet_onnx as _tx_onnx,
            )

            def _fn(audio_np: np.ndarray, sr: int) -> str:
                """
                Transcribe a NumPy audio buffer using the ONNX Parakeet backend.

                Parameters:
                    audio_np (np.ndarray): Mono float32 audio samples.
                    sr (int): Sample rate in Hertz.

                Returns:
                    str: Transcribed text.
                """
                text = _tx_onnx(audio_np, sample_rate=sr)
                # The file-oriented backend uses this status for empty recognition.
                # It must not become spoken words in streaming frames or history.
                return "" if text == "[No speech detected]" else text

            return _fn
        except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("Failed to import ONNX backend: {}", e)
            return None
    elif v == "mlx":
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
                create_parakeet_mlx_streaming_session as _create_mlx_stream,
            )
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
                transcribe_with_parakeet_mlx as _tx_mlx,
            )

            session = None
            try:
                session = _create_mlx_stream()
            except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS as session_err:
                logger.debug("MLX streaming session unavailable; using stateless fallback decode: {}", session_err)
            last_audio = np.array([], dtype=np.float32)
            last_text = ""

            def _close_session() -> None:
                nonlocal session, last_audio, last_text
                if session is not None:
                    with contextlib.suppress(_PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS):
                        session.close()
                session = None
                last_audio = np.array([], dtype=np.float32)
                last_text = ""

            def _reset_session() -> None:
                nonlocal session
                _close_session()
                with contextlib.suppress(_PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS):
                    session = _create_mlx_stream()

            def _fn(audio_np: np.ndarray, sr: int) -> str:
                """
                Transcribe the given audio using the Parakeet MLX backend.

                Returns:
                    The transcription text produced by the MLX model.
                """
                nonlocal session, last_audio, last_text
                if session is None:
                    return _tx_mlx(audio_np, sample_rate=sr)

                try:
                    audio_arr = np.asarray(audio_np, dtype=np.float32).flatten()
                except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS:
                    return _tx_mlx(audio_np, sample_rate=sr)

                if audio_arr.size == 0:
                    return last_text

                if (
                    last_audio.size > 0
                    and audio_arr.size >= last_audio.size
                    and np.array_equal(audio_arr[: last_audio.size], last_audio)
                ):
                    delta = audio_arr[last_audio.size :]
                else:
                    # Audio buffer was rewound/trimmed (final chunk consume path).
                    # Reset session and continue from current window.
                    _reset_session()
                    if session is None:
                        return _tx_mlx(audio_arr, sample_rate=sr)
                    delta = audio_arr

                last_audio = audio_arr
                if delta.size == 0:
                    return last_text

                artifact = session.add_audio(delta, sample_rate=sr)
                last_text = str((artifact or {}).get("text", "")).strip()
                return last_text

            _fn.close = _close_session
            _fn.reset_session = _reset_session
            return _fn
        except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("Failed to import MLX backend: {}", e)
            return None
    else:
        # standard (Nemo Parakeet TDT, greedy)
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                transcribe_with_parakeet as _tx_nemo,
            )

            def _fn(audio_np: np.ndarray, sr: int) -> str:
                """
                Transcribe the given audio array using the Nemo Parakeet "standard" variant.

                Parameters:
                    audio_np (np.ndarray): Mono float32 audio samples to transcribe.
                    sr (int): Sample rate of the audio in Hertz.

                Returns:
                    transcription (str): Decoded text produced from the audio.
                """
                return _tx_nemo(audio_np, sample_rate=sr, variant="standard")

            return _fn
        except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS as e:
            logger.debug("Failed to import standard Parakeet backend: {}", e)
            return None


@dataclass
class ParakeetCoreTranscriber:
    config: StreamingConfig
    decode_fn: DecodeFn | None = None

    def __post_init__(self) -> None:
        """
        Initialize internal buffers and runtime state for the transcriber.

        Sets up the audio buffer using configured sample rate and maximum buffer duration, initializes timing, segment and history counters, and selects a decode function based on the configured model/variant when no decode function was provided.
        """
        self.buffer = AudioBuffer(sample_rate=self.config.sample_rate, max_duration=self.config.max_buffer_duration)
        self._last_partial_time = 0.0
        self._total_processed_seconds = 0.0
        self._segment_index = 0
        self._history: list[str] = []
        if self.decode_fn is None:
            self.decode_fn = _variant_decode_fn(self.config.model, self.config.model_variant)

    # --- Public API ---
    def reset(self) -> None:
        """
        Reset the transcriber to its initial empty state.

        Clears the audio buffer and transcript history, and resets partial-timing, total-processed time, and segment index counters to zero.
        """
        self.buffer.clear()
        self._history.clear()
        self._last_partial_time = 0.0
        self._total_processed_seconds = 0.0
        self._segment_index = 0
        reset_hook = getattr(self.decode_fn, "reset_session", None)
        if callable(reset_hook):
            with contextlib.suppress(_PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS):
                reset_hook()

    def close(self) -> None:
        """
        Release decoder-associated resources, if the active decode function exposes a close hook.

        This is primarily used by long-lived streaming backends (for example MLX)
        that hold session-level resources beyond a single decode call.
        """
        close_hook = getattr(self.decode_fn, "close", None)
        if callable(close_hook):
            with contextlib.suppress(_PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS):
                close_hook()

    def get_full_transcript(self) -> str:
        """
        Get the entire accumulated transcript as a single string.

        Returns:
            A string containing all history entries joined by single spaces; empty string if there are no entries.
        """
        return " ".join(self._history)

    async def process_audio_chunk(self, audio: bytes | str | np.ndarray) -> dict[str, Any] | None:
        """
        Process an incoming audio chunk and emit a partial or final transcription frame when available.

        Accepts audio as raw float32 bytes, a base64 string encoding float32 bytes, or a numpy float32 array. Depending on streaming configuration and buffered audio duration, this may produce:
        - a partial frame (when partials are enabled and the partial cadence has elapsed), or
        - a final frame (when buffered audio reaches the configured chunk duration).

        Returned frame structure (when produced):
        - "type": "partial" or "final"
        - "text": transcribed text
        - "timestamp": UNIX time of emission
        - "is_final": boolean indicating finality
        - additional metadata from internal metadata helpers (segment and timing fields)
        - for final frames only: "_audio_chunk": numpy array copy of the audio used for the final transcript

        Returns:
            dict: A partial or final frame as described above, or `None` if no frame is emitted for this chunk.
        """
        audio_np = self._coerce_to_np(audio)
        if audio_np is None or audio_np.size == 0:
            return None
        self.buffer.add(audio_np)

        now = time.time()
        buf_dur = self.buffer.get_duration()

        # Partials
        if (
            self.config.enable_partial
            and (now - self._last_partial_time) >= float(self.config.partial_interval)
            and buf_dur > float(self.config.min_partial_duration)
        ):
            audio_for_partial = self.buffer.get_audio()
            if audio_for_partial is not None and audio_for_partial.size > 0:
                # Offload potentially blocking decode to a worker thread
                text = await self._decode_async(audio_for_partial)
                # Apply custom vocabulary post-replacements if enabled
                if text:
                    try:  # lazy import to keep core lightweight
                        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Custom_Vocabulary import (
                            postprocess_text_if_enabled as _cv_post,
                        )

                        text = _cv_post(text) or text
                    except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug("Custom vocabulary post-processing failed: {}", e)
                self._last_partial_time = now
                if text:
                    frame = {
                        "type": "partial",
                        "text": text,
                        "timestamp": now,
                        "is_final": False,
                    }
                    frame.update(self._prepare_partial_metadata(buf_dur))
                    return frame

        # Finals
        if buf_dur >= float(self.config.chunk_duration):
            audio_chunk = self.buffer.get_audio(self.config.chunk_duration)
            if audio_chunk is not None and audio_chunk.size > 0:
                # Offload potentially blocking decode to a worker thread
                text = await self._decode_async(audio_chunk)
                # Apply custom vocabulary post-replacements if enabled
                if text:
                    try:  # lazy import to keep core lightweight
                        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Custom_Vocabulary import (
                            postprocess_text_if_enabled as _cv_post,
                        )

                        text = _cv_post(text) or text
                    except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug("Custom vocabulary post-processing failed: {}", e)
                self.buffer.consume(self.config.chunk_duration, self.config.overlap_duration)
                if text:
                    self._history.append(text)
                    chunk_seconds = float(len(audio_chunk)) / float(self.config.sample_rate or 1)
                    frame = {
                        "type": "final",
                        "text": text,
                        "timestamp": now,
                        "is_final": True,
                    }
                    frame.update(self._prepare_final_metadata(chunk_seconds))
                    # include optional audio reference for downstream (e.g., diarization)
                    frame["_audio_chunk"] = np.array(audio_chunk, copy=True)
                    return frame

        return None

    async def flush(self) -> dict[str, Any] | None:
        """
        Emit any remaining buffered audio as a final transcription frame, clear the buffer, and append the transcript to history.

        If there is buffered audio, decode it to text, apply optional post-processing, append the text to the transcriber history, and return a final frame containing transcription, timestamp, is_final=True, and the segment/chunk metadata produced by _prepare_final_metadata. The buffer is cleared regardless of whether decoding yields text.

        Returns:
            dict: A final frame with keys including `"type"`, `"text"`, `"timestamp"`, `"is_final"` and segment/chunk metadata (e.g., `segment_id`, `segment_start`, `segment_end`, `chunk_duration`, `overlap`, `chunk_start`, `chunk_end`, `new_audio_duration`, `cumulative_audio`) when a transcript was produced.
            None: If there is no buffered audio or decoding produced no text.
        """
        if self.buffer.get_duration() <= 0:
            return None
        audio_np = self.buffer.get_audio()
        self.buffer.clear()
        if audio_np is None or audio_np.size == 0:
            return None
        # Offload potentially blocking decode to a worker thread
        text = await self._decode_async(audio_np)
        # Apply custom vocabulary post-replacements if enabled
        if text:
            try:  # lazy import to keep core lightweight
                from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Custom_Vocabulary import (
                    postprocess_text_if_enabled as _cv_post,
                )

                text = _cv_post(text) or text
            except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug("Custom vocabulary post-processing failed: {}", e)
        if text:
            self._history.append(text)
            now = time.time()
            chunk_seconds = float(len(audio_np)) / float(self.config.sample_rate or 1)
            frame = {
                "type": "final",
                "text": text,
                "timestamp": now,
                "is_final": True,
            }
            frame.update(self._prepare_final_metadata(chunk_seconds))
            # include optional audio reference for downstream (e.g., diarization)
            frame["_audio_chunk"] = np.array(audio_np, copy=True)
            return frame
        return None

    # --- Internals ---
    def _coerce_to_np(self, audio: bytes | str | np.ndarray) -> np.ndarray | None:
        """
        Convert an audio input into a mono float32 NumPy array suitable for decoding.

        Parameters:
            audio (bytes | str | numpy.ndarray): Input audio as one of:
                - a NumPy array (any dtype/shape), which will be converted to float32 and averaged to mono if multi-channel;
                - a base64-encoded string containing raw float32 samples;
                - raw bytes or bytearray containing either ASCII/base64 text (will be decoded) or raw float32 bytes.

        Returns:
            numpy.ndarray | None: A 1-D NumPy array of dtype float32 with mono audio samples, or `None` if the input type or conversion fails. The function aligns raw byte input to 4-byte (float32) boundaries and averages channels to produce mono.
        """
        if isinstance(audio, np.ndarray):
            arr = audio
        elif isinstance(audio, str):
            # Treat strings as base64-encoded float32 bytes; be strict and validate the alphabet.
            try:
                # Remove all ASCII whitespace to tolerate wrapped base64 input
                s = "".join(audio.split())
                # Normalize padding for providers that omit '='
                pad_len = (-len(s)) % 4
                if pad_len:
                    s = s + ("=" * pad_len)
                raw = base64.b64decode(s, validate=True)
            except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS:
                return None
            # align to float32 sample size
            if (len(raw) % 4) != 0:
                raw = raw[: len(raw) - (len(raw) % 4)]
            arr = np.frombuffer(raw, dtype=np.float32)
        elif isinstance(audio, (bytes, bytearray)):
            raw = bytes(audio)
            # Robust auto-detection: try strict base64 decode; if it fails, treat as raw float32 bytes
            try:
                # Remove ASCII whitespace before validating; keep original for fallback
                candidate = b"".join(raw.split())
                pad_len = (-len(candidate)) % 4
                if pad_len:
                    candidate = candidate + (b"=" * pad_len)
                decoded = base64.b64decode(candidate, validate=True)
                raw = decoded
            except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS:
                # Not valid base64: proceed with raw bytes
                pass
            if (len(raw) % 4) != 0:
                raw = raw[: len(raw) - (len(raw) % 4)]
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            return None

        # ensure mono float32
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1).astype(np.float32)
        return arr

    def _decode(self, audio_np: np.ndarray) -> str:
        # (Re)select decoder if needed (model/variant may have been updated externally)
        """
        Attempt to decode a mono float32 audio numpy array into text using the configured decode function.

        Parameters:
            audio_np (np.ndarray): Mono float32 audio samples to decode.

        Returns:
            str: Decoded transcription as a string, or an empty string if no decoder is available or decoding fails.
        """
        if not self.decode_fn:
            self.decode_fn = _variant_decode_fn(self.config.model, self.config.model_variant)
        if not self.decode_fn:
            return ""
        try:
            return str(self.decode_fn(audio_np, int(self.config.sample_rate)))
        except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS as e:
            logger.opt(exception=True).warning("Decode failed: {}", e)
            return ""

    async def _decode_async(self, audio_np: np.ndarray) -> str:
        """
        Decode the given mono float32 audio array to text using the configured decode function in a worker thread.

        If no decode function is set, attempts to select one based on the current model/variant; on any error or if no decoder is available, returns an empty string. Parameters:
            audio_np (np.ndarray): Mono float32 audio samples to decode.

        Returns:
            str: Decoded transcript, or an empty string on failure or when no decoder is available.
        """
        if not self.decode_fn:
            self.decode_fn = _variant_decode_fn(self.config.model, self.config.model_variant)
        if not self.decode_fn:
            return ""
        try:
            return str(await asyncio.to_thread(self.decode_fn, audio_np, int(self.config.sample_rate)))
        except _PARAKEET_TRANSCRIBER_NONCRITICAL_EXCEPTIONS as e:
            logger.opt(exception=True).warning("Decode failed (async): {}", e)
            return ""

    def _prepare_partial_metadata(self, buffer_duration: float) -> dict[str, float]:
        """
        Compute metadata for a partial transcription frame based on the current buffered audio.

        Parameters:
            buffer_duration (float): Duration of audio currently buffered, in seconds.

        Returns:
            dict: Mapping with the following keys:
                - segment_id: The 1-based index of the upcoming segment.
                - segment_start: Timestamp (seconds) where this partial segment begins relative to the audio stream.
                - segment_end: Timestamp (seconds) where this partial segment ends (segment_start + buffer_duration).
                - buffer_duration: The provided buffer duration in seconds.
                - cumulative_audio: Total processed audio duration (seconds) before this partial.
        """
        start = float(self._total_processed_seconds)
        return {
            "segment_id": self._segment_index + 1,
            "segment_start": start,
            "segment_end": start + float(buffer_duration),
            "buffer_duration": float(buffer_duration),
            "cumulative_audio": float(self._total_processed_seconds),
        }

    def _prepare_final_metadata(self, chunk_duration: float) -> dict[str, float]:
        """
        Compute metadata for a finalized audio segment including timing, overlap, and cumulative totals.

        Parameters:
            chunk_duration (float): Duration in seconds of the chunk being finalized.

        Returns:
            Dict[str, float]: Metadata for the finalized segment containing:
                - segment_id: Sequential segment index (1-based after update).
                - segment_start: Start time (seconds) of the new audio portion before overlap.
                - segment_end: End time (seconds) of the new audio portion before overlap.
                - chunk_duration: The input chunk duration (seconds).
                - overlap: Amount of overlap (seconds) applied to this chunk.
                - chunk_start: Start time (seconds) of the full chunk including overlap.
                - chunk_end: End time (seconds) of the full chunk including overlap.
                - new_audio_duration: Duration (seconds) of audio from this chunk that advances the processed timeline.
                - cumulative_audio: Total processed audio duration (seconds) after this chunk.
        """
        chunk_duration = max(float(chunk_duration), 0.0)
        overlap_cfg = max(float(self.config.overlap_duration or 0.0), 0.0)
        overlap_used = 0.0 if self._segment_index == 0 else min(overlap_cfg, chunk_duration)

        new_audio_duration = chunk_duration - overlap_used
        if self._segment_index == 0:
            new_audio_duration = chunk_duration
        if new_audio_duration < 0:
            new_audio_duration = 0.0

        segment_start = float(self._total_processed_seconds)
        segment_end = segment_start + new_audio_duration
        chunk_start = max(segment_start - overlap_used, 0.0)
        chunk_end = chunk_start + chunk_duration

        self._total_processed_seconds = segment_end
        self._segment_index += 1

        return {
            "segment_id": self._segment_index,
            "segment_start": segment_start,
            "segment_end": segment_end,
            "chunk_duration": chunk_duration,
            "overlap": overlap_used,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "new_audio_duration": new_audio_duration,
            "cumulative_audio": float(self._total_processed_seconds),
        }


__all__ = ["DecodeFn", "ParakeetCoreTranscriber"]

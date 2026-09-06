"""Bounded whole-turn recognition for Persona's revisable speech transcript."""

import asyncio
import threading
import time
from time import monotonic
from typing import Any

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.Chat.streaming_utils import (
    await_bounded_owned_operation,
    create_bounded_stream_task,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
    WhisperStreamingTranscriber,
)
from tldw_Server_API.app.core.Persona.live_voice_runtime import PersonaVoiceInputLimitError


class PersonaWhisperTranscriber(WhisperStreamingTranscriber):
    """Revise one spoken turn without joining separately decoded overlaps."""

    DECODE_TIMEOUT_SECONDS = 30.0

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._generation = 0
        self._closed = False
        self._decode_pending = False
        self._decode_task: asyncio.Future[Any] | None = None
        self._next_decode_at = 0.0
        self._decoded_samples = 0
        self._auto_commit_samples: int | None = None
        self._result: dict[str, Any] | None = None
        self._error: Exception | None = None

    @property
    def recognition_pending(self) -> bool:
        """Whether this runtime still owns inference, including late cleanup."""
        return self._decode_pending

    @property
    def auto_commit_pending(self) -> bool:
        """Whether VAD has frozen a turn that still needs commitment."""
        return self._auto_commit_samples is not None

    def request_auto_commit(self) -> None:
        """Freeze the first VAD boundary while subsequent audio stays buffered."""
        if self._auto_commit_samples is None:
            self._auto_commit_samples = sum(len(chunk) for chunk in self.buffer.data)

    def auto_commit_ready(self, result: dict[str, Any] | None) -> bool:
        """Commit only a completed snapshot covering the frozen boundary."""
        return (
            self._auto_commit_samples is not None
            and result is not None
            and round(result["buffer_duration"] * self.config.sample_rate) == self._auto_commit_samples
        )

    def audio_after_auto_commit(self) -> bytes:
        """Return audio to replay once into the next turn and its VAD detector."""
        audio = self.buffer.get_audio()
        if self._auto_commit_samples is None or audio is None:
            return b""
        return audio[self._auto_commit_samples :].tobytes()

    def reset(self) -> None:
        """Discard the turn, retaining any worker until it really exits."""
        super().reset()
        self._generation += 1
        self._decoded_samples = 0
        self._auto_commit_samples = None
        self._result = None
        self._error = None

    def cleanup(self) -> None:
        """Retire this runtime without releasing a model used by native code."""
        self._closed = True
        self.reset()
        if not self._decode_pending:
            super().cleanup()

    def _finish_decode(self) -> None:
        self._decode_pending = False
        self._next_decode_at = monotonic() + self.config.partial_interval
        if self._closed:
            super().cleanup()

    async def _decode(self, audio: Any, generation: int, duration: float) -> None:
        released = threading.Event()
        cleanup_claimed = threading.Event()

        def recognize() -> str:
            try:
                return self._transcribe_audio(audio)
            finally:
                released.set()

        try:
            text = await await_bounded_owned_operation(
                asyncio.to_thread(recognize),
                timeout_seconds=self.DECODE_TIMEOUT_SECONDS,
                timeout_message="Speech recognition timed out. Stop voice and try again.",
                on_abandoned=self._finish_decode,
                released_event=released,
                cleanup_claimed=cleanup_claimed,
            )
            if generation == self._generation and not self._closed:
                self.transcription_history[:] = [text] if text else []
                self.last_partial_time = time.time()
                self._result = {
                    "type": "partial",
                    "text": text,
                    "timestamp": self.last_partial_time,
                    "is_final": False,
                    "model": f"whisper-{self.config.whisper_model_size}",
                    **self._prepare_partial_metadata(duration),
                }
        except Exception as exc:  # noqa: BLE001 - retain failures for the socket's existing STT error path
            logger.warning("Persona speech decoding failed (type={})", type(exc).__name__)
            if generation == self._generation and not self._closed:
                self._error = exc
        finally:
            if not cleanup_claimed.is_set():
                self._finish_decode()

    async def process_audio_chunk(self, audio_data: bytes) -> dict[str, Any] | None:
        """Ingest bounded audio and collect a completed snapshot without waiting."""
        if self._closed:
            raise RuntimeError("Speech recognition is stopped. Prepare voice again.")
        if self._error is not None:
            raise self._error
        audio = np.frombuffer(audio_data, dtype=np.float32)
        buffered_samples = sum(len(chunk) for chunk in self.buffer.data)
        max_samples = int(self.config.sample_rate * self.config.max_buffer_duration)
        if buffered_samples + len(audio) > max_samples:
            raise PersonaVoiceInputLimitError(
                f"Keep spoken turns within {self.config.max_buffer_duration:g} seconds and start voice again."
            )
        if len(audio):
            self.buffer.add(audio)
        samples = self._auto_commit_samples if self.auto_commit_pending else buffered_samples + len(audio)
        duration = samples / self.config.sample_rate
        if (
            not self._decode_pending
            and (duration >= self.min_chunk_duration or self.auto_commit_pending)
            and samples > self._decoded_samples
            and monotonic() >= self._next_decode_at
        ):
            self._decode_task = create_bounded_stream_task(
                self._decode(self.buffer.get_audio()[:samples], self._generation, duration)
            )
            self._decode_pending = True
            self._decoded_samples = samples
        result, self._result = self._result, None
        return result

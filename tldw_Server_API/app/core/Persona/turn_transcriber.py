"""Bounded whole-turn recognition for Persona's revisable speech transcript."""

import asyncio
import threading
import time
import traceback
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from loguru import logger

from tldw_Server_API.app.core.Chat.streaming_utils import (
    await_bounded_owned_operation,
    create_bounded_stream_task,
)
from tldw_Server_API.app.core.exceptions import PersonaVoiceInputLimitError, PersonaVoiceRecognitionError
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
    BaseStreamingTranscriber,
)


class PersonaTurnTranscriber(BaseStreamingTranscriber):
    """Own one bounded, revisable speech turn and at most one background decode.

    Args:
        config: Streaming configuration defining float32 sample rate, buffer bound
            and partial cadence. A backend subclass must initialize its model and
            implement _transcribe_audio plus recognition_model_name.

    Use from one event loop after backend initialization. Ingestion never waits
    for native decoding; later calls collect revisions or raise retained failures.
    Reset invalidates publication while retaining native work. Cleanup retires the
    instance permanently and defers model release until owned work actually exits.
    """

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
        """Return whether native inference or its retained cleanup is still owned."""
        return self._decode_pending

    @property
    def auto_commit_pending(self) -> bool:
        """Return whether VAD has frozen a boundary still awaiting commitment/reset."""
        return self._auto_commit_samples is not None

    def request_auto_commit(self) -> None:
        """Freeze the first VAD boundary until reset; repeated requests do nothing.

        Later audio remains buffered under the same total bound, but is excluded
        from boundary recognition and returned by audio_after_auto_commit.
        """
        if self._auto_commit_samples is None:
            self._auto_commit_samples = sum(len(chunk) for chunk in self.buffer.data)

    def auto_commit_ready(self, result: dict[str, Any] | None) -> bool:
        """Check whether a collected result exactly covers the frozen VAD boundary.

        Args:
            result: Optional recognition frame from process_audio_chunk, including
                its buffer_duration in seconds.

        Returns:
            True only when the result covers the first requested boundary.
        """
        return (
            self._auto_commit_samples is not None
            and result is not None
            and round(result["buffer_duration"] * self.config.sample_rate) == self._auto_commit_samples
        )

    def audio_after_auto_commit(self) -> bytes:
        """Return buffered float32 audio after the frozen boundary, or empty bytes.

        The caller retrieves this before reset, then replays it once into the next
        turn and its VAD detector. Reading does not consume or clear the carry.
        """
        audio = self.buffer.get_audio()
        if self._auto_commit_samples is None or audio is None:
            return b""
        return audio[self._auto_commit_samples :].tobytes()

    def reset(self) -> None:
        """Clear audio, transcript, failure and VAD boundary; invalidate late output.

        Native inference continues under ownership. A fresh turn waits for that
        worker and the completion-based cadence before scheduling its next decode.
        Reset does not reopen an instance retired by cleanup.
        """
        super().reset()
        self._generation += 1
        self._decoded_samples = 0
        self._auto_commit_samples = None
        self._result = None
        self._error = None

    def cleanup(self) -> None:
        """Permanently stop ingestion and invalidate all pending publication.

        Returns immediately. Model release occurs now if idle, or in the owned
        worker completion callback; recognition_pending remains true until then.
        """
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
                    "model": self.recognition_model_name,
                    **self._prepare_partial_metadata(duration),
                }
        except Exception as exc:  # noqa: BLE001 - retain failures for the socket's existing STT error path
            # Preserve stack locations, never native messages, source lines or locals.
            frames = [
                {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(exc.__traceback__, limit=20)
            ]
            logger.bind(
                recognition_model=self.recognition_model_name,
                recognition_generation=generation,
                recognition_traceback=frames,
            ).warning("Persona speech decoding failed (type={})", type(exc).__name__)
            if generation == self._generation and not self._closed:
                self._error = exc
        finally:
            if not cleanup_claimed.is_set():
                self._finish_decode()

    async def process_audio_chunk(self, audio_data: bytes) -> dict[str, Any] | None:
        """Ingest audio and collect a completed revision without waiting for inference.

        Args:
            audio_data: Mono float32 PCM at config.sample_rate; empty bytes poll
                completed work without adding audio.

        Returns:
            A partial frame with text (possibly empty), timestamp, is_final=False,
            model and buffer_duration, or None if no revision is ready. New text
            replaces the whole prior hypothesis rather than appending fragments.

        Raises:
            PersonaVoiceRecognitionError: This instance is stopped or the decoder
                returned a known failure status.
            PersonaVoiceInputLimitError: Audio exceeds the configured turn bound.
            Exception: A retained backend/timeout failure or bounded-task admission
                failure. Callers must retire the instance through the STT error path.
        """
        if self._closed:
            raise PersonaVoiceRecognitionError("stopped")
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

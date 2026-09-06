"""Bounded whole-turn recognition for Persona's revisable speech transcript."""

import time
from typing import Any

import numpy as np

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
    WhisperStreamingTranscriber,
)
from tldw_Server_API.app.core.Persona.live_voice_runtime import PersonaVoiceInputLimitError


class PersonaWhisperTranscriber(WhisperStreamingTranscriber):
    """Revise one spoken turn without joining separately decoded overlaps."""

    async def process_audio_chunk(self, audio_data: bytes) -> dict[str, Any] | None:
        """Return a current transcript snapshot, rejecting audio buffer overflow."""
        audio = np.frombuffer(audio_data, dtype=np.float32)
        buffered_samples = sum(len(chunk) for chunk in self.buffer.data)
        max_samples = int(self.config.sample_rate * self.config.max_buffer_duration)
        if buffered_samples + len(audio) > max_samples:
            raise PersonaVoiceInputLimitError(
                f"Keep spoken turns within {self.config.max_buffer_duration:g} seconds and start voice again."
            )
        self.buffer.add(audio)
        now = time.time()
        duration = self.buffer.get_duration()
        if duration < self.min_chunk_duration or now - self.last_partial_time <= self.config.partial_interval:
            return None

        text = self._transcribe_audio(self.buffer.get_audio())
        self.last_partial_time = now
        # History is the latest snapshot, including a correction to no speech.
        self.transcription_history[:] = [text] if text else []
        return {
            "type": "partial",
            "text": text,
            "timestamp": now,
            "is_final": False,
            "model": f"whisper-{self.config.whisper_model_size}",
            **self._prepare_partial_metadata(duration),
        }

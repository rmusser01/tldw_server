"""Whisper backend for Persona's bounded whole-turn speech runtime."""

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
    WhisperStreamingTranscriber,
)
from tldw_Server_API.app.core.Persona.turn_transcriber import PersonaTurnTranscriber


class PersonaWhisperTranscriber(PersonaTurnTranscriber, WhisperStreamingTranscriber):
    """Retain the existing Whisper loader and filtered recognition."""

    @property
    def recognition_model_name(self) -> str:
        """Identify the selected Whisper model in recognition frames."""
        return f"whisper-{self.config.whisper_model_size}"

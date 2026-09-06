"""Parakeet ONNX backend for Persona's bounded whole-turn speech runtime."""

import numpy as np

from tldw_Server_API.app.core.Persona.turn_transcriber import PersonaTurnTranscriber


class PersonaParakeetOnnxTranscriber(PersonaTurnTranscriber):
    """Revise full turns with the configured ONNX decoder, without chunk joins."""

    recognition_model_name = "parakeet-onnx"

    def initialize(self) -> None:
        """Select the existing ONNX decoder; its loader retains the model cache."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.transcriber import (
            _variant_decode_fn,
        )

        self.model = _variant_decode_fn("parakeet", "onnx")
        if self.model is None:
            raise RuntimeError("Parakeet ONNX speech recognition is unavailable.")
        self.min_chunk_duration = 1.0

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Custom_Vocabulary import (
            postprocess_text_if_enabled,
        )

        text = self.model(audio, self.config.sample_rate)
        return (postprocess_text_if_enabled(text) or text) if text else ""

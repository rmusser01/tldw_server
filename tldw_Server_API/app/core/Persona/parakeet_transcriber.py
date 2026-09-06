"""Parakeet ONNX backend for Persona's bounded whole-turn speech runtime."""

import numpy as np

from tldw_Server_API.app.core.exceptions import PersonaVoiceRecognitionError
from tldw_Server_API.app.core.Persona.turn_transcriber import PersonaTurnTranscriber

# Exact statuses emitted by the legacy file decoder; ordinary speech is untouched.
_ONNX_ERROR_STATUSES = frozenset(
    {
        "[Error: Failed to load ONNX model]",
        "[Error: Failed to load audio]",
        "[Error: Invalid audio data type]",
        "[Error: Feature extraction failed]",
        "[Error: No output from model]",
        "[Error: Parakeet ONNX transcription failed]",
    }
)


class PersonaParakeetOnnxTranscriber(PersonaTurnTranscriber):
    """Revise full turns using the existing cached ONNX decoder and vocabulary.

    Initialize before ingestion. Model loading is lazy and can fail on the first
    decode; preparation alone does not establish inference availability. Inherits
    bounded background ownership, revision and retirement from PersonaTurnTranscriber.
    """

    recognition_model_name = "parakeet-onnx"

    def initialize(self) -> None:
        """Select the decoder without loading model weights or starting inference.

        Returns:
            None. Installs the decoder and the one-second partial threshold.

        Raises:
            PersonaVoiceRecognitionError: The ONNX backend is unavailable.
        """
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.transcriber import (
            _variant_decode_fn,
        )

        self.model = _variant_decode_fn("parakeet", "onnx")
        if self.model is None:
            raise PersonaVoiceRecognitionError("unavailable")
        self.min_chunk_duration = 1.0

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Custom_Vocabulary import (
            postprocess_text_if_enabled,
        )

        text = self.model(audio, self.config.sample_rate)
        if text in _ONNX_ERROR_STATUSES:
            raise PersonaVoiceRecognitionError("failed")
        return (postprocess_text_if_enabled(text) or text) if text else ""

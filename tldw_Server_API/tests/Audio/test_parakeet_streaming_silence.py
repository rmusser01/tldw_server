"""Parakeet's backend silence status is not recognized speech."""

import numpy as np
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_ONNX as onnx
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.config import StreamingConfig
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.transcriber import (
    ParakeetCoreTranscriber,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["partial", "final", "flush"])
async def test_onnx_silence_never_enters_streaming_transcript(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(onnx, "transcribe_with_parakeet_onnx", lambda *args, **kwargs: "[No speech detected]")
    transcriber = ParakeetCoreTranscriber(
        config=StreamingConfig(
            model_variant="onnx",
            sample_rate=16000,
            enable_partial=mode == "partial",
            partial_interval=0,
            min_partial_duration=0.3,
            chunk_duration=1 if mode == "final" else 5,
        )
    )
    try:
        result = await transcriber.process_audio_chunk(np.zeros(16000, dtype=np.float32))
        if mode == "flush":
            result = await transcriber.flush()
        assert result is None
        assert transcriber.get_full_transcript() == ""
    finally:
        transcriber.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["No speech detected", "The Blue Notebook is ready.", "ready ready"])
async def test_onnx_streaming_preserves_real_words(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(onnx, "transcribe_with_parakeet_onnx", lambda *args, **kwargs: text)
    transcriber = ParakeetCoreTranscriber(
        config=StreamingConfig(
            model_variant="onnx",
            sample_rate=16000,
            enable_partial=False,
            chunk_duration=1,
        )
    )
    try:
        result = await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32))
        assert result is not None
        assert result["text"] == text
        assert transcriber.get_full_transcript() == text
    finally:
        transcriber.close()

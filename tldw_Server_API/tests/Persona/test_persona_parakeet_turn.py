"""A later whole-turn recognition replaces mistaken chunk text in Persona."""

import asyncio

import numpy as np
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_ONNX as onnx
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming import transcriber as core
from tldw_Server_API.app.core.Persona import live_stt

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_corrected_turn_replaces_early_chunk_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    text = ["Right."]
    monkeypatch.setattr(core.time, "time", lambda: now[0])
    monkeypatch.setattr(onnx, "transcribe_with_parakeet_onnx", lambda *args, **kwargs: text[0])
    transcriber = live_stt.create_persona_live_stt_transcriber(voice_runtime={"stt_model": "parakeet-onnx"})
    transcriber.initialize()

    async def feed(samples: int):
        result = await transcriber.process_audio_chunk(np.ones(samples, dtype=np.float32).tobytes())
        if getattr(transcriber, "recognition_pending", False):
            async with asyncio.timeout(2):
                while transcriber.recognition_pending:
                    await asyncio.sleep(0.005)
            result = await transcriber.process_audio_chunk(b"") or result
        return result

    try:
        await feed(80000)
        now[0] += 0.1
        await feed(4096)  # The legacy adapter finalizes a mistaken prefix here.
        text[0] = "Say the blue notebook is ready."
        now[0] += 0.5
        await asyncio.sleep(0.36)
        result = await feed(16000)
        assert live_stt.persona_live_transcript_snapshot(transcriber=transcriber, result=result) == text[0]
    finally:
        transcriber.cleanup()

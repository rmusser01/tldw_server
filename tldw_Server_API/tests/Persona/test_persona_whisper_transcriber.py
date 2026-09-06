"""Persona speech snapshots retain one bounded turn across decoder updates."""

from types import SimpleNamespace

import numpy as np
import pytest

from tldw_Server_API.app.api.v1.endpoints import persona
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Streaming_Unified as streaming


@pytest.fixture
def recognition(monkeypatch):
    now = [1000.0]
    calls = []
    phrase = ["Reply with the blue notebook is ready."]

    class Model:
        def transcribe(self, audio, **kwargs):
            calls.append(audio.copy())
            text = phrase[0] if len(audio) >= 96000 else "Reply with"
            return [SimpleNamespace(text=text)], SimpleNamespace()

    monkeypatch.setattr(streaming, "get_whisper_model", lambda *args: Model(), raising=False)
    monkeypatch.setattr(streaming.time, "time", lambda: now[0])
    transcriber = persona._create_persona_live_stt_transcriber(
        voice_runtime={"stt_model": "tiny.en", "enable_vad": False},
    )
    transcriber.initialize()
    yield transcriber, now, calls, phrase
    transcriber.cleanup()


@pytest.mark.asyncio
async def test_whisper_revises_whole_turn_across_five_second_boundary(recognition):
    transcriber, now, calls, phrase = recognition
    await transcriber.process_audio_chunk(np.zeros(80000, dtype=np.float32).tobytes())
    now[0] += 0.1  # Between partial updates: generic streaming finalizes here.
    await transcriber.process_audio_chunk(np.zeros(4096, dtype=np.float32).tobytes())
    now[0] += 0.5
    result = await transcriber.process_audio_chunk(np.zeros(16000, dtype=np.float32).tobytes())
    assert persona._persona_live_transcript_snapshot(transcriber=transcriber, result=result) == phrase[0]
    assert len(calls[-1]) == 100096


@pytest.mark.asyncio
async def test_whisper_preserves_repetition_and_empty_corrections(recognition):
    transcriber, now, _, phrase = recognition
    phrase[0] = "ready ready"
    result = await transcriber.process_audio_chunk(np.zeros(96000, dtype=np.float32).tobytes())
    assert result["text"] == "ready ready"
    phrase[0] = ""
    now[0] += 0.5
    result = await transcriber.process_audio_chunk(np.zeros(4096, dtype=np.float32).tobytes())
    assert result is not None
    assert persona._persona_live_transcript_snapshot(transcriber=transcriber, result=result) == ""


@pytest.mark.asyncio
async def test_whisper_rejects_buffer_overflow_without_discarding_earlier_audio(recognition):
    transcriber, _, calls, _ = recognition
    await transcriber.process_audio_chunk(np.zeros(16000 * 30, dtype=np.float32).tobytes())
    previous_calls = len(calls)
    with pytest.raises(ValueError, match="30 seconds"):
        await transcriber.process_audio_chunk(np.zeros(4096, dtype=np.float32).tobytes())
    assert len(calls) == previous_calls


@pytest.mark.asyncio
async def test_whisper_reset_starts_fresh_audio_and_transcript(recognition):
    transcriber, now, calls, _ = recognition
    await transcriber.process_audio_chunk(np.zeros(96000, dtype=np.float32).tobytes())
    transcriber.reset()
    assert transcriber.get_full_transcript() == ""
    now[0] += 0.5
    result = await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
    assert result["text"] == "Reply with"
    assert np.array_equal(calls[-1], np.ones(16000, dtype=np.float32))

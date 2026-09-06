"""Persona speech snapshots retain one bounded turn across decoder updates."""

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Streaming_Unified as streaming
from tldw_Server_API.app.core.Persona import live_stt, turn_transcriber

pytestmark = pytest.mark.unit


@pytest.fixture(params=["tiny.en", "parakeet-onnx"])
def recognition(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Any:
    now = [1000.0]
    calls = []
    phrase = ["Reply with the blue notebook is ready."]

    class Model:
        def transcribe(self, audio: Any, **kwargs: Any) -> Any:
            calls.append(audio.copy())
            text = phrase[0] if len(audio) >= 96000 else "Reply with"
            return [SimpleNamespace(text=text)], SimpleNamespace()

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_ONNX as onnx

    def decode_onnx(audio: Any, **kwargs: Any) -> str:
        calls.append(audio.copy())
        return phrase[0] if len(audio) >= 96000 else "Reply with"

    monkeypatch.setattr(onnx, "transcribe_with_parakeet_onnx", decode_onnx)
    monkeypatch.setattr(streaming, "get_whisper_model", lambda *args: Model(), raising=False)
    monkeypatch.setattr(streaming.time, "time", lambda: now[0])
    monkeypatch.setattr(turn_transcriber, "monotonic", lambda: now[0])
    transcriber = live_stt.create_persona_live_stt_transcriber(
        voice_runtime={"stt_model": request.param, "enable_vad": False},
    )
    transcriber.initialize()
    yield transcriber, now, calls, phrase
    transcriber.cleanup()


async def collect(transcriber: Any) -> dict[str, Any]:
    """Collect a completed snapshot while continuing the audio frame loop."""
    async with asyncio.timeout(2):
        while True:
            result = await transcriber.process_audio_chunk(b"")
            if result is not None:
                return result
            await asyncio.sleep(0.005)


async def settled(transcriber: Any) -> None:
    async with asyncio.timeout(2):
        while transcriber.recognition_pending:
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
@pytest.mark.parametrize("samples", [8000, 20000])
async def test_auto_commit_decodes_exact_boundary_and_retains_later_audio(
    recognition: Any, monkeypatch: pytest.MonkeyPatch, samples: int
) -> None:
    transcriber, now, calls, _ = recognition
    await transcriber.process_audio_chunk(np.zeros(samples, dtype=np.float32).tobytes())
    finalize = getattr(transcriber, "request_auto_commit", None)
    assert callable(finalize), "VAD must retain its boundary until recognition finishes"
    finalize()
    assert not transcriber.auto_commit_ready(None)
    await transcriber.process_audio_chunk(np.ones(4096, dtype=np.float32).tobytes())
    result = await collect(transcriber)
    assert transcriber.auto_commit_ready(result)
    assert len(calls[-1]) == samples
    assert not np.any(calls[-1])  # Speech after the boundary is not merged into this turn.
    remaining = transcriber.audio_after_auto_commit()
    assert np.array_equal(np.frombuffer(remaining, dtype=np.float32), np.ones(4096, dtype=np.float32))
    transcriber.reset()
    assert not transcriber.auto_commit_pending
    now[0] += 0.5
    await transcriber.process_audio_chunk(remaining + np.ones(12000, dtype=np.float32).tobytes())
    await collect(transcriber)
    assert np.all(calls[-1] == 1)


@pytest.mark.asyncio
async def test_recognition_revises_whole_turn_across_five_second_boundary(recognition: Any) -> None:
    transcriber, now, calls, phrase = recognition
    await transcriber.process_audio_chunk(np.zeros(80000, dtype=np.float32).tobytes())
    await collect(transcriber)
    now[0] += 0.1  # Between partial updates: generic streaming finalizes here.
    await transcriber.process_audio_chunk(np.zeros(4096, dtype=np.float32).tobytes())
    now[0] += 0.5
    await transcriber.process_audio_chunk(np.zeros(16000, dtype=np.float32).tobytes())
    result = await collect(transcriber)
    assert live_stt.persona_live_transcript_snapshot(transcriber=transcriber, result=result) == phrase[0]
    assert len(calls[-1]) == 100096


@pytest.mark.asyncio
async def test_recognition_preserves_repetition_and_empty_corrections(recognition: Any) -> None:
    transcriber, now, _, phrase = recognition
    phrase[0] = "ready ready"
    await transcriber.process_audio_chunk(np.zeros(96000, dtype=np.float32).tobytes())
    result = await collect(transcriber)
    assert result["text"] == "ready ready"
    phrase[0] = ""
    now[0] += 0.5
    await transcriber.process_audio_chunk(np.zeros(4096, dtype=np.float32).tobytes())
    result = await collect(transcriber)
    assert result is not None
    assert live_stt.persona_live_transcript_snapshot(transcriber=transcriber, result=result) == ""


@pytest.mark.asyncio
async def test_recognition_rejects_buffer_overflow_without_discarding_earlier_audio(recognition: Any) -> None:
    transcriber, _, calls, _ = recognition
    await transcriber.process_audio_chunk(np.zeros(16000 * 30, dtype=np.float32).tobytes())
    await collect(transcriber)
    previous_calls = len(calls)
    with pytest.raises(ValueError, match="30 seconds"):
        await transcriber.process_audio_chunk(np.zeros(4096, dtype=np.float32).tobytes())
    assert len(calls) == previous_calls


@pytest.mark.asyncio
async def test_recognition_reset_starts_fresh_audio_and_transcript(recognition: Any) -> None:
    transcriber, now, calls, _ = recognition
    await transcriber.process_audio_chunk(np.zeros(96000, dtype=np.float32).tobytes())
    await collect(transcriber)
    transcriber.reset()
    assert transcriber.get_full_transcript() == ""
    now[0] += 0.5
    await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
    result = await collect(transcriber)
    assert result["text"] == "Reply with"
    assert np.array_equal(calls[-1], np.ones(16000, dtype=np.float32))


@pytest.mark.asyncio
async def test_slow_decode_does_not_hold_audio_ingestion(recognition: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    transcriber, _, _, _ = recognition
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_decode(audio: Any) -> str:
        started.set()
        release.wait(2)
        finished.set()
        return "notebook"

    monkeypatch.setattr(transcriber, "_transcribe_audio", slow_decode)
    # A bounded fallback prevents the synchronous baseline from hanging pytest.
    timer = threading.Timer(1, release.set)
    timer.start()
    try:
        result = await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
        assert not finished.is_set(), "Audio ingestion waited for the whole decoder"
        assert result is None
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.005)
        assert started.is_set()
        await transcriber.process_audio_chunk(np.ones(4096, dtype=np.float32).tobytes())
        assert transcriber.buffer.get_duration() > 1
    finally:
        release.set()
        timer.cancel()
        for _ in range(100):
            if finished.is_set():
                break
            await asyncio.sleep(0.005)
        await settled(transcriber)


@pytest.mark.asyncio
async def test_busy_decode_coalesces_audio_and_uses_completion_cadence(
    recognition: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcriber, now, _, _ = recognition
    started, release = threading.Event(), threading.Event()
    calls = []

    def decode(audio: Any) -> str:
        calls.append(len(audio))
        started.set()
        release.wait(2)
        return "ready"

    monkeypatch.setattr(transcriber, "_transcribe_audio", decode)
    try:
        await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
        assert await asyncio.to_thread(started.wait, 1)
        for _ in range(8):
            now[0] += 0.25
            await transcriber.process_audio_chunk(np.ones(4096, dtype=np.float32).tobytes())
        assert calls == [16000]
        release.set()
        assert (await collect(transcriber))["text"] == "ready"
        await transcriber.process_audio_chunk(b"")
        now[0] += 0.34
        await transcriber.process_audio_chunk(b"")
        assert calls == [16000]
        now[0] += 0.02
        assert (await collect(transcriber))["text"] == "ready"
        assert calls == [16000, 16000 + 8 * 4096]
        now[0] += 1
        await transcriber.process_audio_chunk(b"")
        assert not transcriber.recognition_pending  # Unchanged audio is not retried.
    finally:
        release.set()
        await settled(transcriber)


@pytest.mark.asyncio
@pytest.mark.parametrize("retire", [False, True])
async def test_reset_discards_late_decode_and_cleanup_retains_model(
    recognition: Any, monkeypatch: pytest.MonkeyPatch, retire: bool
) -> None:
    transcriber, now, _, _ = recognition
    started, release = threading.Event(), threading.Event()
    model = transcriber.model

    def decode(audio: Any) -> str:
        started.set()
        release.wait(2)
        assert transcriber.model is model
        return "stale words"

    monkeypatch.setattr(transcriber, "_transcribe_audio", decode)
    try:
        await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
        assert await asyncio.to_thread(started.wait, 1)
        transcriber.cleanup() if retire else transcriber.reset()
        assert transcriber.model is model
        assert transcriber.recognition_pending
        if not retire:
            await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
        release.set()
        await settled(transcriber)
        assert transcriber.get_full_transcript() == ""
        if retire:
            assert transcriber.model is None
            with pytest.raises(RuntimeError, match="stopped"):
                await transcriber.process_audio_chunk(b"")
        else:
            assert await transcriber.process_audio_chunk(b"") is None
            monkeypatch.setattr(transcriber, "_transcribe_audio", lambda audio: "fresh words")
            now[0] += 0.5
            assert (await collect(transcriber))["text"] == "fresh words"
    finally:
        release.set()
        await settled(transcriber)


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", ["STREAM_TASK_MAX_ACTIVE", "STREAM_CLEANUP_TASK_MAX_ACTIVE"])
async def test_decode_capacity_rejects_without_starting_model(
    recognition: Any, monkeypatch: pytest.MonkeyPatch, capacity: str
) -> None:
    from tldw_Server_API.app.core.Chat import streaming_utils

    transcriber, _, calls, _ = recognition
    monkeypatch.setattr(streaming_utils, capacity, 0)
    with pytest.raises(streaming_utils.StreamTaskCapacityError):
        await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
        await collect(transcriber)
    await settled(transcriber)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_abandoned_decode_keeps_cleanup_owned(
    recognition: Any, monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    transcriber, _, _, _ = recognition
    started, release = threading.Event(), threading.Event()
    model = transcriber.model

    def decode(audio: Any) -> str:
        started.set()
        release.wait(2)
        assert transcriber.model is model
        return "late words"

    monkeypatch.setattr(transcriber, "_transcribe_audio", decode)
    if not cancel:
        monkeypatch.setattr(transcriber, "DECODE_TIMEOUT_SECONDS", 0.05)
    try:
        await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
        assert await asyncio.to_thread(started.wait, 1)
        if cancel:
            transcriber._decode_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await transcriber._decode_task
        else:
            with pytest.raises(TimeoutError, match="timed out"):
                await collect(transcriber)
        transcriber.cleanup()
        assert transcriber.recognition_pending
        assert transcriber.model is model
        release.set()
        await settled(transcriber)
        assert transcriber.model is None
        assert transcriber.get_full_transcript() == ""
    finally:
        release.set()
        await settled(transcriber)

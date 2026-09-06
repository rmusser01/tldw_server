"""A later whole-turn recognition replaces mistaken chunk text in Persona."""

import asyncio
from typing import Any

import numpy as np
import pytest

from tldw_Server_API.app.core.exceptions import PersonaVoiceRecognitionError
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

    async def feed(samples: int) -> dict[str, Any] | None:
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        "[Error: Failed to load ONNX model]",
        "[Error: Failed to load audio]",
        "[Error: Invalid audio data type]",
        "[Error: Feature extraction failed]",
        "[Error: No output from model]",
        "[Error: Parakeet ONNX transcription failed]",
    ],
)
async def test_backend_error_status_never_becomes_a_transcript(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    monkeypatch.setattr(onnx, "transcribe_with_parakeet_onnx", lambda *args, **kwargs: status)
    transcriber = live_stt.create_persona_live_stt_transcriber(voice_runtime={"stt_model": "parakeet-onnx"})
    transcriber.initialize()
    try:
        await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
        async with asyncio.timeout(2):
            while transcriber.recognition_pending:
                await asyncio.sleep(0.005)
        with pytest.raises(PersonaVoiceRecognitionError, match="Speech recognition failed") as error:
            await transcriber.process_audio_chunk(b"")
        assert error.value.code == "failed"
        assert transcriber.get_full_transcript() == ""
    finally:
        transcriber.cleanup()


@pytest.mark.parametrize("text", ["Error: Feature extraction failed", "[a spoken aside]", "ready ready"])
def test_ordinary_words_are_not_error_statuses(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    monkeypatch.setattr(onnx, "transcribe_with_parakeet_onnx", lambda *args, **kwargs: text)
    transcriber = live_stt.create_persona_live_stt_transcriber(voice_runtime={"stt_model": "parakeet-onnx"})
    transcriber.initialize()
    try:
        assert transcriber._transcribe_audio(np.ones(16000, dtype=np.float32)) == text
    finally:
        transcriber.cleanup()


def test_unavailable_onnx_decoder_has_typed_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "_variant_decode_fn", lambda *args: None)
    transcriber = live_stt.create_persona_live_stt_transcriber(voice_runtime={"stt_model": "parakeet-onnx"})
    with pytest.raises(PersonaVoiceRecognitionError) as error:
        transcriber.initialize()
    assert getattr(error.value, "code", None) == "unavailable"


@pytest.mark.asyncio
async def test_stopped_recognition_has_typed_failure() -> None:
    transcriber = live_stt.create_persona_live_stt_transcriber(voice_runtime={"stt_model": "parakeet-onnx"})
    transcriber.cleanup()
    with pytest.raises(PersonaVoiceRecognitionError) as error:
        await transcriber.process_audio_chunk(b"")
    assert getattr(error.value, "code", None) == "stopped"


@pytest.mark.asyncio
async def test_decode_failure_logs_correlated_stack_without_private_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from loguru import logger

    transcriber = live_stt.create_persona_live_stt_transcriber(voice_runtime={"stt_model": "parakeet-onnx"})
    transcriber.initialize()
    records = []
    private_value = "private-speech-and-credential-value"

    def fail_decode(audio: np.ndarray) -> str:
        raise RuntimeError(private_value)

    monkeypatch.setattr(transcriber, "_transcribe_audio", fail_decode)
    sink = logger.add(lambda message: records.append(message.record))
    try:
        with logger.contextualize(persona_session_id="session-safe", client_message_id="turn-safe"):
            await transcriber.process_audio_chunk(np.ones(16000, dtype=np.float32).tobytes())
        async with asyncio.timeout(2):
            while transcriber.recognition_pending:
                await asyncio.sleep(0.005)
        record = next(record for record in records if "Persona speech decoding failed" in record["message"])
        assert record["extra"]["persona_session_id"] == "session-safe"
        assert record["extra"]["client_message_id"] == "turn-safe"
        assert any(frame["function"] == "fail_decode" for frame in record["extra"]["recognition_traceback"])
        assert private_value not in str(records)
        assert record["exception"] is None
    finally:
        logger.remove(sink)
        transcriber.cleanup()

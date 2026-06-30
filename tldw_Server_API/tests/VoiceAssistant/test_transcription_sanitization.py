from __future__ import annotations

import sys
from types import ModuleType

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant


pytestmark = pytest.mark.unit


class WebSocketRecorder:
    def __init__(self) -> None:
        self.sent_json: list[dict] = []

    async def send_json(self, payload: dict) -> None:
        self.sent_json.append(payload)


@pytest.mark.asyncio
async def test_transcribe_audio_sanitizes_backend_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged_errors = []

    class LoggerStub:
        def error(self, message, *args, **kwargs):
            logged_errors.append((message, args, kwargs))

    def fake_transcribe_audio(*args, **kwargs):
        raise RuntimeError("stt backend exploded /private/audio.wav")

    transcription_module = ModuleType(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib"
    )
    transcription_module.transcribe_audio = fake_transcribe_audio

    monkeypatch.setattr(voice_assistant, "logger", LoggerStub())
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
        transcription_module,
    )

    with pytest.raises(RuntimeError, match="stt backend exploded"):
        await voice_assistant._transcribe_audio(b"\x00\x00\x00\x00", {})

    assert logged_errors == [("Audio transcription failed", (), {})]


@pytest.mark.asyncio
async def test_process_audio_command_sanitizes_transcription_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged_errors = []

    class LoggerStub:
        def error(self, message, *args, **kwargs):
            logged_errors.append((message, args, kwargs))

    async def fake_transcribe_audio(audio_bytes, config):
        raise RuntimeError("stt pipeline exploded /private/audio.wav")

    monkeypatch.setattr(voice_assistant, "logger", LoggerStub())
    monkeypatch.setattr(voice_assistant, "_transcribe_audio", fake_transcribe_audio)

    websocket = WebSocketRecorder()
    await voice_assistant._process_audio_command(
        websocket=websocket,
        audio_buffer=[b"audio"],
        user_id=1,
        session_id="session-1",
        config={},
        router_instance=object(),
    )

    assert logged_errors == [("Transcription failed", (), {})]

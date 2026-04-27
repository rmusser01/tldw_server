from __future__ import annotations

import sys
from types import ModuleType

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant


pytestmark = pytest.mark.unit


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

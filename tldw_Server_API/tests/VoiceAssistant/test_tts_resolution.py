from __future__ import annotations

import base64
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import voice_assistant


pytestmark = pytest.mark.unit


class _RecordingTTSService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_speech(self, request, provider=None, fallback=None):
        self.calls.append(
            {
                "request": request,
                "provider": provider,
                "fallback": fallback,
            }
        )
        yield b"audio-chunk"


class _FailingTTSService:
    async def generate_speech(self, request, provider=None, fallback=None):
        raise RuntimeError("tts stream backend exploded /private/tts-stream")
        yield b"unreachable"


class _WebSocketRecorder:
    def __init__(self) -> None:
        self.sent_json: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent_json.append(payload)


@pytest.mark.asyncio
async def test_generate_tts_audio_maps_openai_provider_to_valid_model(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _RecordingTTSService()

    async def _fake_get_tts_service_v2():
        return service

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_service_v2",
        _fake_get_tts_service_v2,
    )

    audio_bytes, mime_type = await voice_assistant._generate_tts_audio(
        text="hello world",
        provider="openai",
        voice="alloy",
        response_format="mp3",
    )

    assert audio_bytes == b"audio-chunk"
    assert mime_type == "audio/mpeg"
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["provider"] == "openai"
    assert call["request"].model == "tts-1"
    assert call["request"].voice == "alloy"


@pytest.mark.asyncio
async def test_stream_tts_response_uses_explicit_tts_model_in_config(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _RecordingTTSService()

    async def _fake_get_tts_service_v2():
        return service

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_service_v2",
        _fake_get_tts_service_v2,
    )

    websocket = _WebSocketRecorder()
    await voice_assistant._stream_tts_response(
        websocket=websocket,
        text="hello world",
        config={
            "tts_provider": "openai",
            "tts_model": "tts-1-hd",
            "tts_voice": "alloy",
            "tts_format": "mp3",
        },
    )

    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["provider"] == "openai"
    assert call["request"].model == "tts-1-hd"
    assert call["request"].voice == "alloy"
    assert websocket.sent_json[0]["data"] == base64.b64encode(b"audio-chunk").decode("ascii")


@pytest.mark.asyncio
async def test_stream_tts_response_sanitizes_backend_failure_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logged_errors = []

    class LoggerStub:
        def error(self, message, *args, **kwargs):
            logged_errors.append((message, args, kwargs))

    async def _fake_get_tts_service_v2():
        return _FailingTTSService()

    monkeypatch.setattr(voice_assistant, "logger", LoggerStub())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_service_v2",
        _fake_get_tts_service_v2,
    )

    websocket = _WebSocketRecorder()
    await voice_assistant._stream_tts_response(
        websocket=websocket,
        text="hello world",
        config={"tts_provider": "openai"},
    )

    assert logged_errors == [("TTS streaming failed", (), {})]

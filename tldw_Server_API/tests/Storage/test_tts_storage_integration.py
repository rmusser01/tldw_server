"""
Tests for TTS storage integration (return_download_link behavior).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoint
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


class _DummyTTSService:
    def generate_speech(self, *args, **kwargs):
        async def _gen():
            yield b"hello "
            yield b"world"
        return _gen()


def _make_request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/audio/speech",
        "headers": [],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    }
    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    return Request(scope, _receive)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_return_download_link_persists_tts_audio(monkeypatch):
    """Non-streaming requests with return_download_link register a file and set header."""
    monkeypatch.setattr(audio_endpoint, "_sanitize_speech_request", lambda *args, **kwargs: None)

    async def _resolve_tts_byok(*args, **kwargs):
        return (1, {}, None)

    monkeypatch.setattr(audio_endpoint, "_resolve_tts_byok", _resolve_tts_byok)

    save_mock = AsyncMock(return_value={"id": 123})
    monkeypatch.setattr(audio_endpoint, "save_and_register_tts_audio", save_mock)

    req = OpenAISpeechRequest(
        input="hello world",
        model="kokoro",
        stream=False,
        response_format="mp3",
        return_download_link=True,
    )
    request = _make_request()
    usage_log = SimpleNamespace(log_event=lambda *args, **kwargs: None)
    user = SimpleNamespace(id=1)

    resp = await audio_endpoint.create_speech(
        req,
        request,
        tts_service=_DummyTTSService(),
        current_user=user,
        usage_log=usage_log,
    )

    assert resp.headers.get("X-Download-Path") == "/api/v1/storage/files/123/download"
    assert resp.headers.get("X-Generated-File-Id") == "123"
    assert save_mock.await_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_return_download_link_rejected_for_streaming(monkeypatch):
    """Streaming requests cannot request return_download_link."""
    monkeypatch.setattr(audio_endpoint, "_sanitize_speech_request", lambda *args, **kwargs: None)

    async def _resolve_tts_byok(*args, **kwargs):
        return (1, {}, None)

    monkeypatch.setattr(audio_endpoint, "_resolve_tts_byok", _resolve_tts_byok)

    req = OpenAISpeechRequest(
        input="hello world",
        model="kokoro",
        stream=True,
        response_format="mp3",
        return_download_link=True,
    )
    request = _make_request()
    usage_log = SimpleNamespace(log_event=lambda *args, **kwargs: None)
    user = SimpleNamespace(id=1)

    with pytest.raises(HTTPException) as exc:
        await audio_endpoint.create_speech(
            req,
            request,
            tts_service=_DummyTTSService(),
            current_user=user,
            usage_log=usage_log,
        )

    assert exc.value.status_code == 400


@pytest.mark.unit
def test_tts_headers_via_http(monkeypatch):
    """HTTP call returns storage download headers when requested."""
    monkeypatch.setattr(audio_endpoint, "_sanitize_speech_request", lambda *args, **kwargs: None)

    async def _resolve_tts_byok(*args, **kwargs):
        return (1, {}, None)

    monkeypatch.setattr(audio_endpoint, "_resolve_tts_byok", _resolve_tts_byok)

    save_mock = AsyncMock(return_value={"id": 456})
    monkeypatch.setattr(audio_endpoint, "save_and_register_tts_audio", save_mock)

    usage_log = SimpleNamespace(log_event=lambda *args, **kwargs: None)

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[audio_endpoint.get_tts_service] = lambda: _DummyTTSService()
    app.dependency_overrides[audio_endpoint.get_usage_event_logger] = lambda: usage_log
    app.dependency_overrides[audio_endpoint.check_rate_limit] = lambda: None

    try:
        settings = get_settings()
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "hello world",
                    "voice": "af_heart",
                    "response_format": "mp3",
                    "stream": False,
                    "return_download_link": True,
                },
                headers={"X-API-KEY": settings.SINGLE_USER_API_KEY},
            )
        assert resp.status_code == 200
        assert resp.headers.get("X-Download-Path") == "/api/v1/storage/files/456/download"
        assert resp.headers.get("X-Generated-File-Id") == "456"
    finally:
        app.dependency_overrides.pop(get_request_user, None)
        app.dependency_overrides.pop(audio_endpoint.get_tts_service, None)
        app.dependency_overrides.pop(audio_endpoint.get_usage_event_logger, None)
        app.dependency_overrides.pop(audio_endpoint.check_rate_limit, None)

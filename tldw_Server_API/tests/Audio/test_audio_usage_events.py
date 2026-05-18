import pytest
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.audio.audio_tts as audio_tts
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


pytestmark = pytest.mark.unit


class _DummyLogger:
    def __init__(self):
        self.events = []

    def log_event(self, name, resource_id=None, tags=None, metadata=None):
        self.events.append((name, resource_id, tags, metadata))


class _FakeTTSService:
    def generate_speech(
        self,
        request_data,
        provider=None,
        fallback=True,
        provider_overrides=None,
        voice_to_voice_start=None,
        voice_to_voice_route="audio.speech",
        user_id=None,
        metadata_only=False,
        request_id=None,
    ):
        async def _gen():
            yield b"audio-bytes"

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


@pytest.mark.asyncio
async def test_tts_usage_event_logged(monkeypatch):
    dummy = _DummyLogger()

    async def _resolve_tts_byok(*args, **kwargs):
        _ = args, kwargs
        return (1, {}, None)

    def _shim_attr(name: str):
        if name == "_sanitize_speech_request":
            return lambda *args, **kwargs: "openai"
        if name == "_resolve_tts_byok":
            return _resolve_tts_byok
        raise NameError(name)

    monkeypatch.setattr(audio_tts, "_audio_shim_attr", _shim_attr, raising=True)

    response = await audio_tts.create_speech(
        OpenAISpeechRequest(
            model="tts-1",
            input="hello",
            voice="alloy",
            response_format="mp3",
            stream=False,
        ),
        _make_request(),
        tts_service=_FakeTTSService(),
        current_user=User(id=1, username="tester", email=None, is_active=True),
        media_db=None,
        usage_log=dummy,
    )

    assert response.status_code == 200
    assert response.body == b"audio-bytes"
    assert any(e[0] == "audio.tts" for e in dummy.events)

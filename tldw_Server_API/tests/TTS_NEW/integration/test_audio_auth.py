"""
Minimal auth tests for /api/v1/audio/speech using dependency overrides.
Validates 401 without auth and 200 with user + stubbed TTS service.
"""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints

pytestmark = [pytest.mark.integration]


def _api_key_headers() -> dict[str, str]:
    return {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}


def test_speech_requires_auth_401(bypass_api_limits):
    with bypass_api_limits(app), TestClient(app) as client:
        payload = {
            "model": "kokoro",
            "input": "Hello",
            "voice": "af_bella",
            "response_format": "mp3",
            "stream": False,
        }
        resp = client.post("/api/v1/audio/speech", json=payload)
        assert resp.status_code == 401


def test_speech_ok_with_override(monkeypatch, bypass_api_limits):
    with bypass_api_limits(app), TestClient(app) as client:
        async def _override_user():
            return User(id=1, username="tester", email="t@example.com", is_active=True)

        app.dependency_overrides[get_request_user] = _override_user

        class _StubTTS:
            def generate_speech(self, *args, **kwargs):
                async def _gen():
                    yield b"abc"
                return _gen()

        async def _override_tts_service():
            return _StubTTS()

        app.dependency_overrides[audio_endpoints.get_tts_service] = _override_tts_service

        try:
            payload = {
                "model": "kokoro",
                "input": "Hello",
                "voice": "af_bella",
                "response_format": "mp3",
                "stream": False,
            }
            resp = client.post("/api/v1/audio/speech", json=payload, headers=_api_key_headers())
            assert resp.status_code == 200
            assert resp.headers.get("content-type", "").startswith("audio/mpeg")
            assert resp.content == b"abc"
        finally:
            app.dependency_overrides.pop(get_request_user, None)
            app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)

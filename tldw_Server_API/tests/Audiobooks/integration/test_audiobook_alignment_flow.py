import base64
import copy
import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import get_usage_event_logger
from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints
from tldw_Server_API.app.api.v1.endpoints.audio.audiobooks import router as audiobooks_router
from tldw_Server_API.app.core.AuthNZ import llm_provider_overrides
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

pytestmark = pytest.mark.integration


class _NullUsageLogger:
    def log_event(self, *_args, **_kwargs):
        return None


@pytest.fixture()
def client_audio_audiobooks():
    """Build a standalone audio app with a healthy empty override snapshot."""
    with llm_provider_overrides._OVERRIDE_LOCK:
        original_overrides = copy.deepcopy(llm_provider_overrides._OVERRIDE_CACHE)
        original_healthy = llm_provider_overrides._OVERRIDE_CACHE_HEALTHY
        original_ttl_enabled = (
            not llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS
        )
    llm_provider_overrides.set_llm_provider_overrides_cache_for_tests({})
    app = FastAPI()
    app.include_router(audio_endpoints.router, prefix="/api/v1/audio")
    app.include_router(audiobooks_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    async def _no_rate_limit():
        return None

    async def _usage_logger_override():
        return _NullUsageLogger()

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[audio_endpoints.check_rate_limit] = _no_rate_limit
    app.dependency_overrides[get_usage_event_logger] = _usage_logger_override
    try:
        with TestClient(app) as client:
            client.headers.update({"X-API-KEY": get_settings().SINGLE_USER_API_KEY})
            yield client
    finally:
        app.dependency_overrides.clear()
        llm_provider_overrides.set_llm_provider_overrides_cache_for_tests(
            original_overrides,
            healthy=original_healthy,
            ttl_enabled=original_ttl_enabled,
        )


def test_audio_speech_alignment_to_subtitles(client_audio_audiobooks):
    alignment_payload = {
        "engine": "kokoro",
        "sample_rate": 24000,
        "words": [
            {"word": "Hello", "start_ms": 0, "end_ms": 400},
            {"word": "world.", "start_ms": 450, "end_ms": 900},
        ],
    }

    async def mock_stream(request_obj, *args, **kwargs):
        request_obj._tts_metadata = {"alignment": alignment_payload}
        yield b"audio_data"

    with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
        mock_generate_speech.side_effect = mock_stream

        response = client_audio_audiobooks.post(
            "/api/v1/audio/speech",
            json={
                "input": "Hello world.",
                "voice": "af_heart",
                "model": "kokoro",
                "response_format": "wav",
                "stream": False,
            },
        )

        assert response.status_code == 200
        encoded = response.headers.get("X-TTS-Alignment")
        assert encoded
        alignment = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))

    subtitle_response = client_audio_audiobooks.post(
        "/api/v1/audiobooks/subtitles",
        json={
            "format": "srt",
            "mode": "sentence",
            "variant": "wide",
            "alignment": alignment,
        },
    )

    assert subtitle_response.status_code == 200
    assert "Hello world." in subtitle_response.text
    assert "00:00:00,000 --> 00:00:00,900" in subtitle_response.text

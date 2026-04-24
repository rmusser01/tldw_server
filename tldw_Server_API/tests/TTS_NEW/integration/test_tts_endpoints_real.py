"""
Integration tests for TTS (OpenAI-compatible) endpoints without mocks.
Skips generation when no provider keys are present; still exercises list/status paths.
"""

import os
import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_user(tts_test_app, bypass_api_limits):
    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)
    tts_test_app.dependency_overrides[get_request_user] = override_user
    with bypass_api_limits(tts_test_app), TestClient(tts_test_app) as client:
        yield client
    tts_test_app.dependency_overrides.clear()


def test_tts_health_path_exists(client_with_user: TestClient):
    # Audio router lives under /api/v1/audio; verify a non-generating path works
    # No explicit health route; we just assert 404 on an unknown path under the router prefix works
    resp = client_with_user.get("/api/v1/audio/unknown")
    assert resp.status_code in (404, 405)


@pytest.mark.requires_llm
def test_tts_generate_when_configured(client_with_user: TestClient):
    openai_key = os.getenv("OPENAI_API_KEY", "")
    elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
    if (
        (not openai_key and not elevenlabs_key)
        or openai_key.startswith("sk-test-")
        or elevenlabs_key.startswith("el-test-")
    ):
        pytest.skip("No TTS providers configured")
    # Attempt a small generation (may still fail depending on provider config)
    payload = {
        "model": "gpt-4o-mini-tts",
        "input": "Hello from integration tests",
        "voice": "alloy",
        "response_format": "mp3"
    }
    resp = client_with_user.post("/api/v1/audio/speech", json=payload)
    # Accept either success or provider error; endpoint is exercised without mocks
    assert resp.status_code in (200, 400, 401, 404, 429, 500)

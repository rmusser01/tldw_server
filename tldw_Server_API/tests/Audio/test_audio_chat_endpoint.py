import base64
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router

# Non-sensitive test API key used only in tests.
TEST_API_KEY = "test-api-key-1234567890"


def _encode_silence_base64(duration_sec: float = 0.1, sr: int = 16000) -> str:
    buf = io.BytesIO()
    data = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(buf, data, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    # Stub speech chat service to avoid heavy STT/LLM/TTS
    from tldw_Server_API.app.core.Streaming import speech_chat_service
    from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
        SpeechChatRequest,
        SpeechChatResponse,
        SpeechChatTiming,
    )

    async def _fake_run_speech_chat_turn(
        *,
        request_data: SpeechChatRequest,
        request=None,
        current_user,
        chat_db,
        tts_service,
    ) -> SpeechChatResponse:
        return SpeechChatResponse(
            session_id=request_data.session_id or "conv-ep-1",
            user_transcript="stub transcript",
            assistant_text="stub reply",
            output_audio=_encode_silence_base64(),
            output_audio_mime_type="audio/mpeg",
            timing=SpeechChatTiming(stt_ms=1.0, llm_ms=2.0, tts_ms=3.0),
            token_usage=None,
            metadata={"from_test": True},
            action_result=None,
        )

    # Bypass AuthNZ for these endpoint-level tests by overriding the
    # request-user dependency with a lightweight single-user stub.
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User

    async def _fake_get_request_user() -> User:
        # Minimal single-user stub; auth details are exercised elsewhere.
        return User(id=1, username="single_user")

    monkeypatch.setattr(
        speech_chat_service,
        "run_speech_chat_turn",
        _fake_run_speech_chat_turn,
    )

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _fake_get_request_user
    app.include_router(audio_router, prefix="/api/v1/audio")
    with TestClient(app) as c:
        yield c


def test_audio_chat_endpoint_success(client: TestClient):
    payload = {
        "session_id": None,
        "input_audio": _encode_silence_base64(),
        "input_audio_format": "wav",
        "llm_config": {"model": "gpt-4o-mini", "api_provider": "openai"},
    }
    r = client.post(
        "/api/v1/audio/chat",
        json=payload,
        headers={"X-API-KEY": TEST_API_KEY},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == "conv-ep-1"
    assert body["user_transcript"] == "stub transcript"
    assert body["assistant_text"] == "stub reply"
    assert body["output_audio"]
    assert body["output_audio_mime_type"].startswith("audio/")
    assert "action_result" in body
    assert body["action_result"] is None


def test_audio_chat_endpoint_concurrency_limit(monkeypatch, client: TestClient):
    # Patch can_start_stream to force a quota failure
    from tldw_Server_API.app.api.v1.endpoints import audio as audio_module

    async def _deny(_user_id: int):
        return False, "Concurrent streams limit reached (1)"

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.audio_quota.can_start_stream", _deny)

    payload = {
        "session_id": None,
        "input_audio": _encode_silence_base64(),
        "input_audio_format": "wav",
        "llm_config": {"model": "gpt-4o-mini", "api_provider": "openai"},
    }
    r = client.post("/api/v1/audio/chat", json=payload, headers={"X-API-KEY": TEST_API_KEY})
    assert r.status_code == 429
    assert "Concurrent" in r.text

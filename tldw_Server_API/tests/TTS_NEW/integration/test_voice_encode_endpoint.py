import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.TTS.voice_manager import VoiceEncodeResult


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()
    app = FastAPI()
    app.include_router(audio_router, prefix="/api/v1/audio")
    with TestClient(app) as c:
        yield c


def test_voice_encode_endpoint_returns_result(client, monkeypatch):
    called = {"user_id": None}

    class _FakeVoiceManager:
        async def encode_voice_reference(self, user_id, voice_id, provider, reference_text=None, force=False):
            called["user_id"] = user_id
            return VoiceEncodeResult(
                voice_id=voice_id,
                provider=provider,
                cached=False,
                ref_codes_len=3,
                reference_text=reference_text,
            )

    def _fake_get_voice_manager():
        return _FakeVoiceManager()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        _fake_get_voice_manager,
        raising=True,
    )

    payload = {
        "voice_id": "voice-123",
        "provider": "neutts",
        "reference_text": "Hello there",
        "force": True,
    }
    r = client.post(
        "/api/v1/audio/voices/encode",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["voice_id"] == "voice-123"
    assert body["provider"] == "neutts"
    assert body["ref_codes_len"] == 3
    assert str(called["user_id"]) == "1"


def test_omnivoice_voice_encode_endpoint_returns_metadata_only_result(client, monkeypatch):
    class _FakeVoiceManager:
        async def encode_voice_reference(self, user_id, voice_id, provider, reference_text=None, force=False):
            return VoiceEncodeResult(
                voice_id=voice_id,
                provider=provider,
                cached=False,
                ref_codes_len=None,
                reference_text=reference_text,
            )

    def _fake_get_voice_manager():
        return _FakeVoiceManager()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        _fake_get_voice_manager,
        raising=True,
    )

    payload = {
        "voice_id": "voice-omnivoice",
        "provider": "omnivoice",
        "reference_text": "Stored transcript",
        "force": True,
    }
    r = client.post(
        "/api/v1/audio/voices/encode",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["voice_id"] == "voice-omnivoice"
    assert body["provider"] == "omnivoice"
    assert body["ref_codes_len"] is None
    assert body["reference_text"] == "Stored transcript"

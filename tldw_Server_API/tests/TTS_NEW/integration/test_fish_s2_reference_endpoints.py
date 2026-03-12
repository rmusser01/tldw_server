import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio import audio_voices
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()
    app = FastAPI()
    app.include_router(audio_voices.router, prefix="/api/v1/audio")
    with TestClient(app) as c:
        yield c, app


def test_create_fish_s2_reference_from_existing_voice(client):
    client_obj, app = client
    called = {}

    class _FakeTTSService:
        async def create_fish_s2_reference(self, **kwargs):
            called.update(kwargs)
            return {
                "reference_id": kwargs["voice_id"],
                "voice_id": kwargs["voice_id"],
                "remote_reference_id": "tldw_u1_voice-1",
                "reference_text": kwargs["reference_text"],
                "cached": False,
            }

    app.dependency_overrides[audio_voices.get_tts_service] = lambda: _FakeTTSService()
    try:
        response = client_obj.post(
            "/api/v1/audio/providers/fish_s2/references",
            data={"voice_id": "voice-1", "reference_text": "stored text"},
            headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        )
    finally:
        app.dependency_overrides.pop(audio_voices.get_tts_service, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["reference_id"] == "voice-1"
    assert body["remote_reference_id"] == "tldw_u1_voice-1"
    assert str(called["user_id"]) == "1"
    assert called["voice_id"] == "voice-1"


def test_create_fish_s2_reference_requires_upload_fields_when_voice_missing(client):
    client_obj, app = client
    app.dependency_overrides[audio_voices.get_tts_service] = lambda: object()
    try:
        response = client_obj.post(
            "/api/v1/audio/providers/fish_s2/references",
            data={},
            headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        )
    finally:
        app.dependency_overrides.pop(audio_voices.get_tts_service, None)

    assert response.status_code == 400


def test_list_fish_s2_references(client):
    client_obj, app = client

    class _FakeTTSService:
        async def list_fish_s2_references(self, **kwargs):
            return [
                {
                    "reference_id": "voice-1",
                    "voice_id": "voice-1",
                    "name": "Voice One",
                    "reference_text": "stored text",
                    "remote_reference_id": "tldw_u1_voice-1",
                }
            ]

    app.dependency_overrides[audio_voices.get_tts_service] = lambda: _FakeTTSService()
    try:
        response = client_obj.get(
            "/api/v1/audio/providers/fish_s2/references",
            headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        )
    finally:
        app.dependency_overrides.pop(audio_voices.get_tts_service, None)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 1
    assert body["references"][0]["reference_id"] == "voice-1"


def test_delete_fish_s2_reference(client):
    client_obj, app = client
    called = {}

    class _FakeTTSService:
        async def delete_fish_s2_reference(self, **kwargs):
            called.update(kwargs)
            return {"reference_id": kwargs["reference_id"], "deleted": True}

    app.dependency_overrides[audio_voices.get_tts_service] = lambda: _FakeTTSService()
    try:
        response = client_obj.delete(
            "/api/v1/audio/providers/fish_s2/references/voice-1",
            headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        )
    finally:
        app.dependency_overrides.pop(audio_voices.get_tts_service, None)

    assert response.status_code == 200, response.text
    assert response.json() == {"reference_id": "voice-1", "deleted": True}
    assert str(called["user_id"]) == "1"
    assert called["reference_id"] == "voice-1"

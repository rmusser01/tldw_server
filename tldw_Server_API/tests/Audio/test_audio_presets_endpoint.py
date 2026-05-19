from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


@dataclass
class PresetClient:
    client: TestClient
    db: MediaDatabase
    current_user_id: list[int]

    def as_user(self, user_id: int) -> None:
        self.current_user_id[0] = user_id


@pytest.fixture
def preset_client(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")

    db = MediaDatabase(db_path=str(tmp_path / "Media_DB_v2.db"), client_id="audio-preset-tests")
    current_user_id = [1]

    async def _user():
        user_id = current_user_id[0]
        return User(id=user_id, username=f"user-{user_id}", email=None, is_active=True)

    app = FastAPI()
    app.include_router(audio_router, prefix="/api/v1/audio")
    app.dependency_overrides[get_media_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = _user

    with TestClient(app) as client:
        yield PresetClient(client=client, db=db, current_user_id=current_user_id)

    db.close_connection()


def _tts_payload(**overrides):
    payload = {
        "kind": "tts",
        "name": "OpenAI Alloy MP3",
        "description": "Fast OpenAI smoke-test voice",
        "favorite": True,
        "is_default": True,
        "config": {
            "provider": "openai",
            "model": "tts-1",
            "voice": "alloy",
            "response_format": "mp3",
            "speed": 1.0,
        },
        "capability_assumptions": {
            "provider": "openai",
            "model": "tts-1",
            "availability": {"value": "ready", "source": "health"},
        },
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_audio_presets_crud_default_and_soft_delete(preset_client: PresetClient):
    client = preset_client.client

    created = client.post("/api/v1/audio/presets", json=_tts_payload())
    assert created.status_code == 201
    first = created.json()
    first_id = first["id"]
    assert first["owner_user_id"] == "1"
    assert first["kind"] == "tts"
    assert first["name"] == "OpenAI Alloy MP3"
    assert first["favorite"] is True
    assert first["is_default"] is True
    assert first["config"]["voice"] == "alloy"

    second = client.post(
        "/api/v1/audio/presets",
        json=_tts_payload(
            name="OpenAI Nova WAV",
            config={
                "provider": "openai",
                "model": "tts-1",
                "voice": "nova",
                "response_format": "wav",
            },
        ),
    )
    assert second.status_code == 201
    second_id = second.json()["id"]

    listed = client.get("/api/v1/audio/presets?kind=tts")
    assert listed.status_code == 200
    items = listed.json()["items"]
    by_id = {item["id"]: item for item in items}
    assert list(by_id) == [second_id, first_id]
    assert by_id[second_id]["is_default"] is True
    assert by_id[first_id]["is_default"] is False

    updated = client.patch(
        f"/api/v1/audio/presets/{first_id}",
        json={
            "name": "OpenAI Alloy Fast",
            "favorite": False,
            "config": {
                "provider": "openai",
                "model": "tts-1",
                "voice": "alloy",
                "response_format": "mp3",
                "speed": 1.15,
            },
        },
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "OpenAI Alloy Fast"
    assert updated.json()["favorite"] is False
    assert updated.json()["config"]["speed"] == 1.15

    validation = client.post(f"/api/v1/audio/presets/{first_id}/validate")
    assert validation.status_code == 200
    assert validation.json()["valid"] is True
    assert validation.json()["warnings"] == []
    assert validation.json()["preset"]["id"] == first_id

    deleted = client.delete(f"/api/v1/audio/presets/{first_id}")
    assert deleted.status_code == 204

    after_delete = client.get("/api/v1/audio/presets?kind=tts")
    assert after_delete.status_code == 200
    assert [item["id"] for item in after_delete.json()["items"]] == [second_id]


@pytest.mark.unit
def test_audio_presets_are_owner_scoped(preset_client: PresetClient):
    client = preset_client.client

    created = client.post("/api/v1/audio/presets", json=_tts_payload(name="User 1 preset"))
    assert created.status_code == 201
    preset_id = created.json()["id"]

    preset_client.as_user(2)

    listed = client.get("/api/v1/audio/presets?kind=tts")
    assert listed.status_code == 200
    assert listed.json()["items"] == []

    update = client.patch(f"/api/v1/audio/presets/{preset_id}", json={"name": "Stolen"})
    assert update.status_code == 404

    delete = client.delete(f"/api/v1/audio/presets/{preset_id}")
    assert delete.status_code == 404

    preset_client.as_user(1)
    owner_list = client.get("/api/v1/audio/presets?kind=tts")
    assert [item["id"] for item in owner_list.json()["items"]] == [preset_id]


@pytest.mark.unit
def test_browser_tts_preset_is_marked_non_portable_and_requires_revalidation(
    preset_client: PresetClient,
):
    client = preset_client.client

    response = client.post(
        "/api/v1/audio/presets",
        json=_tts_payload(
            name="Browser fallback",
            config={
                "provider": "browser",
                "voice": "System Default",
                "rate": 1,
            },
        ),
    )
    assert response.status_code == 201
    preset = response.json()
    assert preset["config"]["browser_local"] is True
    assert preset["config"]["requires_browser_revalidation"] is True

    validation = client.post(f"/api/v1/audio/presets/{preset['id']}/validate")
    assert validation.status_code == 200
    payload = validation.json()
    assert payload["valid"] is True
    assert payload["warnings"] == [
        {
            "code": "browser_tts_revalidation_required",
            "message": "Browser TTS presets depend on the current browser and must be revalidated before use.",
            "field": "config.provider",
        }
    ]


@pytest.mark.unit
def test_audio_presets_reject_unknown_kind_and_secret_config_keys(
    preset_client: PresetClient,
):
    client = preset_client.client

    bad_kind = client.post(
        "/api/v1/audio/presets",
        json=_tts_payload(kind="voice"),
    )
    assert bad_kind.status_code == 422

    secret_config = client.post(
        "/api/v1/audio/presets",
        json=_tts_payload(config={"provider": "openai", "model": "tts-1", "api_key": "sk-test"}),
    )
    assert secret_config.status_code == 422

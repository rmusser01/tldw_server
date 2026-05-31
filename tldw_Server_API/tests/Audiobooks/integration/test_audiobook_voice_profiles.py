import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audiobooks import router as audiobooks_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

pytestmark = pytest.mark.integration


@pytest.fixture()
def client_voice_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    app = FastAPI()
    app.include_router(audiobooks_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_voice_profile_crud(client_voice_profiles):
    create_payload = {
        "name": "Narrator + Dialog",
        "default_voice": "af_heart",
        "default_speed": 1.0,
        "chapter_overrides": [{"chapter_id": "ch_005", "voice": "am_adam", "speed": 0.98}],
    }

    create_resp = client_voice_profiles.post("/api/v1/audiobooks/voices/profiles", json=create_payload)
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["profile_id"].startswith("vp_")
    assert created["name"] == create_payload["name"]
    assert created["default_voice"] == create_payload["default_voice"]

    list_resp = client_voice_profiles.get("/api/v1/audiobooks/voices/profiles")
    assert list_resp.status_code == 200
    payload = list_resp.json()
    listed = payload["profiles"]
    assert len(listed) == 1
    assert listed[0]["profile_id"] == created["profile_id"]
    assert listed[0]["chapter_overrides"] == create_payload["chapter_overrides"]
    assert payload["total"] == 1
    assert payload["limit"] == 100
    assert payload["offset"] == 0
    assert payload["has_more"] is False
    assert payload["next_offset"] is None
    assert payload["pagination"] == {
        "mode": "offset",
        "total": 1,
        "limit": 100,
        "offset": 0,
        "has_more": False,
        "next_offset": None,
    }

    delete_resp = client_voice_profiles.delete(f"/api/v1/audiobooks/voices/profiles/{created['profile_id']}")
    assert delete_resp.status_code == 200
    deleted = delete_resp.json()
    assert deleted["profile_id"] == created["profile_id"]
    assert deleted["deleted"] is True

    list_resp = client_voice_profiles.get("/api/v1/audiobooks/voices/profiles")
    assert list_resp.status_code == 200
    assert list_resp.json() == {
        "profiles": [],
        "total": 0,
        "limit": 100,
        "offset": 0,
        "has_more": False,
        "next_offset": None,
        "pagination": {
            "mode": "offset",
            "total": 0,
            "limit": 100,
            "offset": 0,
            "has_more": False,
            "next_offset": None,
        },
    }

"""Integration tests for Audio Studio project endpoints."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio_studio import router as audio_studio_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_audio_studio(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))

    app = FastAPI()
    app.include_router(audio_studio_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_workflows_endpoint_lists_first_class_workflows(client_audio_studio: TestClient) -> None:
    response = client_audio_studio.get("/api/v1/audio-studio/workflows")

    assert response.status_code == 200
    assert response.json() == {
        "workflows": [
            {"id": "narration", "label": "Narration"},
            {"id": "podcast", "label": "Podcast"},
            {"id": "briefing", "label": "Briefing"},
            {"id": "music", "label": "Music"},
        ]
    }


def test_create_project_supports_first_class_workflows(client_audio_studio: TestClient) -> None:
    for workflow in ("narration", "podcast", "briefing", "music"):
        response = client_audio_studio.post(
            "/api/v1/audio-studio/projects",
            json={"title": f"{workflow} project", "workflow": workflow},
        )
        assert response.status_code == 200
        assert response.json()["workflow"] == workflow


def test_project_crud_and_resource_upserts(client_audio_studio: TestClient) -> None:
    create_response = client_audio_studio.post(
        "/api/v1/audio-studio/projects",
        json={"title": "Narration", "workflow": "narration", "settings": {"format": "mp3"}},
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["project_id"].startswith("ast_")
    assert created["current_revision_id"].startswith("rev_")

    list_response = client_audio_studio.get("/api/v1/audio-studio/projects")
    assert list_response.status_code == 200
    assert [row["project_id"] for row in list_response.json()["projects"]] == [created["project_id"]]

    get_response = client_audio_studio.get(f"/api/v1/audio-studio/projects/{created['project_id']}")
    assert get_response.status_code == 200
    assert get_response.json()["title"] == "Narration"

    patch_response = client_audio_studio.patch(
        f"/api/v1/audio-studio/projects/{created['project_id']}",
        json={"title": "Renamed", "base_revision_id": created["current_revision_id"]},
    )
    assert patch_response.status_code == 200
    patched = patch_response.json()
    assert patched["title"] == "Renamed"
    assert patched["current_revision_id"] != created["current_revision_id"]

    stale_section_response = client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/sections/sec_001",
        json={"title": "Intro", "body_text": "Hello", "base_revision_id": created["current_revision_id"]},
    )
    assert stale_section_response.status_code == 409

    section_response = client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/sections/sec_001",
        json={"title": "Intro", "body_text": "Hello", "base_revision_id": patched["current_revision_id"]},
    )
    assert section_response.status_code == 200
    section = section_response.json()
    assert section["section_id"] == "sec_001"

    track_response = client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/tracks/trk_001",
        json={
            "name": "Narration",
            "kind": "speech",
            "base_revision_id": section["current_revision_id"],
        },
    )
    assert track_response.status_code == 200
    track = track_response.json()
    assert track["track_id"] == "trk_001"

    clip_response = client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/clips/clip_001",
        json={
            "section_id": "sec_001",
            "track_id": "trk_001",
            "title": "Intro Clip",
            "clip_type": "speech",
            "base_revision_id": track["current_revision_id"],
        },
    )
    assert clip_response.status_code == 200
    clip = clip_response.json()
    assert clip["clip_id"] == "clip_001"

    delete_response = client_audio_studio.delete(f"/api/v1/audio-studio/projects/{created['project_id']}")
    assert delete_response.status_code == 200
    assert delete_response.json()["archived"] is True

    assert client_audio_studio.get(f"/api/v1/audio-studio/projects/{created['project_id']}").status_code == 404


def test_project_lookup_does_not_leak_across_users(client_audio_studio: TestClient) -> None:
    create_response = client_audio_studio.post(
        "/api/v1/audio-studio/projects",
        json={"title": "Private", "workflow": "briefing"},
    )
    assert create_response.status_code == 200
    project_id = create_response.json()["project_id"]

    async def override_other_user():
        return User(id=2, username="other", email="o@e.com", is_active=True, is_admin=False)

    client_audio_studio.app.dependency_overrides[get_request_user] = override_other_user

    assert client_audio_studio.get(f"/api/v1/audio-studio/projects/{project_id}").status_code == 404
    assert client_audio_studio.patch(
        f"/api/v1/audio-studio/projects/{project_id}",
        json={"title": "Nope", "base_revision_id": "rev_missing"},
    ).status_code == 404
    assert client_audio_studio.delete(f"/api/v1/audio-studio/projects/{project_id}").status_code == 404

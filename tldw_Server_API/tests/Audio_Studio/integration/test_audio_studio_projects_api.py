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

    missing_body_response = client_audio_studio.request(
        "DELETE",
        f"/api/v1/audio-studio/projects/{created['project_id']}",
    )
    assert missing_body_response.status_code == 422

    missing_revision_response = client_audio_studio.request(
        "DELETE",
        f"/api/v1/audio-studio/projects/{created['project_id']}",
        json={},
    )
    assert missing_revision_response.status_code == 422

    stale_delete_response = client_audio_studio.request(
        "DELETE",
        f"/api/v1/audio-studio/projects/{created['project_id']}",
        json={"base_revision_id": created["current_revision_id"]},
    )
    assert stale_delete_response.status_code == 409

    delete_response = client_audio_studio.request(
        "DELETE",
        f"/api/v1/audio-studio/projects/{created['project_id']}",
        json={"base_revision_id": clip["current_revision_id"]},
    )
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
    assert client_audio_studio.request(
        "DELETE",
        f"/api/v1/audio-studio/projects/{project_id}",
        json={"base_revision_id": "rev_missing"},
    ).status_code == 404


def test_project_update_with_stale_base_does_not_change_state(client_audio_studio: TestClient) -> None:
    create_response = client_audio_studio.post(
        "/api/v1/audio-studio/projects",
        json={"title": "Initial", "workflow": "narration"},
    )
    assert create_response.status_code == 200
    created = create_response.json()

    first_update = client_audio_studio.patch(
        f"/api/v1/audio-studio/projects/{created['project_id']}",
        json={"title": "First Update", "base_revision_id": created["current_revision_id"]},
    )
    assert first_update.status_code == 200

    stale_update = client_audio_studio.patch(
        f"/api/v1/audio-studio/projects/{created['project_id']}",
        json={"title": "Stale Update", "base_revision_id": created["current_revision_id"]},
    )
    assert stale_update.status_code == 409

    get_response = client_audio_studio.get(f"/api/v1/audio-studio/projects/{created['project_id']}")
    assert get_response.status_code == 200
    current = get_response.json()
    assert current["title"] == "First Update"
    assert current["current_revision_id"] == first_update.json()["current_revision_id"]


def test_project_description_can_be_cleared_intentionally(client_audio_studio: TestClient) -> None:
    create_response = client_audio_studio.post(
        "/api/v1/audio-studio/projects",
        json={"title": "Described", "workflow": "briefing", "description": "Summary"},
    )
    assert create_response.status_code == 200
    created = create_response.json()

    empty_description = client_audio_studio.patch(
        f"/api/v1/audio-studio/projects/{created['project_id']}",
        json={"description": "", "base_revision_id": created["current_revision_id"]},
    )
    assert empty_description.status_code == 200
    assert empty_description.json()["description"] == ""

    omitted_description = client_audio_studio.patch(
        f"/api/v1/audio-studio/projects/{created['project_id']}",
        json={
            "title": "Still Empty",
            "base_revision_id": empty_description.json()["current_revision_id"],
        },
    )
    assert omitted_description.status_code == 200
    assert omitted_description.json()["description"] == ""

    null_description = client_audio_studio.patch(
        f"/api/v1/audio-studio/projects/{created['project_id']}",
        json={
            "description": None,
            "base_revision_id": omitted_description.json()["current_revision_id"],
        },
    )
    assert null_description.status_code == 200
    assert null_description.json()["description"] is None


def test_audio_studio_path_ids_reject_invalid_or_oversized_values(client_audio_studio: TestClient) -> None:
    create_response = client_audio_studio.post(
        "/api/v1/audio-studio/projects",
        json={"title": "Paths", "workflow": "music"},
    )
    assert create_response.status_code == 200
    created = create_response.json()
    base_revision_id = created["current_revision_id"]
    oversized_id = "x" * 121

    assert client_audio_studio.get("/api/v1/audio-studio/projects/bad%20id").status_code == 422
    assert client_audio_studio.get(f"/api/v1/audio-studio/projects/{oversized_id}").status_code == 422

    section_payload = {"title": "Intro", "body_text": "Hello", "base_revision_id": base_revision_id}
    assert client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/sections/bad%20id",
        json=section_payload,
    ).status_code == 422
    assert client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/sections/{oversized_id}",
        json=section_payload,
    ).status_code == 422

    track_payload = {"name": "Narration", "kind": "speech", "base_revision_id": base_revision_id}
    assert client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/tracks/bad%20id",
        json=track_payload,
    ).status_code == 422

    clip_payload = {
        "track_id": "trk_001",
        "title": "Clip",
        "clip_type": "speech",
        "base_revision_id": base_revision_id,
    }
    assert client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/clips/bad%20id",
        json=clip_payload,
    ).status_code == 422


def test_clip_upsert_rejects_missing_references(client_audio_studio: TestClient) -> None:
    create_response = client_audio_studio.post(
        "/api/v1/audio-studio/projects",
        json={"title": "Refs", "workflow": "narration"},
    )
    assert create_response.status_code == 200
    created = create_response.json()

    missing_track = client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/clips/clip_missing_track",
        json={
            "track_id": "trk_missing",
            "title": "Missing track",
            "clip_type": "speech",
            "base_revision_id": created["current_revision_id"],
        },
    )
    assert missing_track.status_code == 400
    assert missing_track.json()["detail"] == "audio_studio_track_not_found"

    track_response = client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/tracks/trk_001",
        json={
            "name": "Narration",
            "kind": "speech",
            "base_revision_id": created["current_revision_id"],
        },
    )
    assert track_response.status_code == 200
    track = track_response.json()

    missing_section = client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/clips/clip_missing_section",
        json={
            "section_id": "sec_missing",
            "track_id": "trk_001",
            "title": "Missing section",
            "clip_type": "speech",
            "base_revision_id": track["current_revision_id"],
        },
    )
    assert missing_section.status_code == 400
    assert missing_section.json()["detail"] == "audio_studio_section_not_found"

    section_response = client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/sections/sec_001",
        json={
            "title": "Intro",
            "body_text": "Hello",
            "base_revision_id": track["current_revision_id"],
        },
    )
    assert section_response.status_code == 200
    section = section_response.json()

    missing_artifact = client_audio_studio.put(
        f"/api/v1/audio-studio/projects/{created['project_id']}/clips/clip_missing_artifact",
        json={
            "section_id": "sec_001",
            "track_id": "trk_001",
            "title": "Missing artifact",
            "clip_type": "speech",
            "artifact_id": "art_missing",
            "base_revision_id": section["current_revision_id"],
        },
    )
    assert missing_artifact.status_code == 400
    assert missing_artifact.json()["detail"] == "audio_studio_artifact_not_found"

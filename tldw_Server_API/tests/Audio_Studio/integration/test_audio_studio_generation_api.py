"""Integration tests for Audio Studio generation endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio_studio import router as audio_studio_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_audio_studio_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    app = FastAPI()
    app.include_router(audio_studio_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def _create_project_and_section(client: TestClient) -> tuple[dict, dict]:
    create_response = client.post(
        "/api/v1/audio-studio/projects",
        json={"title": "Narration", "workflow": "narration"},
    )
    assert create_response.status_code == 200
    project = create_response.json()

    section_response = client.put(
        f"/api/v1/audio-studio/projects/{project['project_id']}/sections/sec_001",
        json={
            "title": "Intro",
            "body_text": "Hello world",
            "base_revision_id": project["current_revision_id"],
        },
    )
    assert section_response.status_code == 200
    return project, section_response.json()


def test_generation_endpoint_creates_idempotent_secret_free_job(
    client_audio_studio_generation: TestClient,
) -> None:
    project, section = _create_project_and_section(client_audio_studio_generation)
    payload = {
        "kind": "speech",
        "provider": "tts",
        "target_resource_kind": "section",
        "target_resource_id": "sec_001",
        "target_revision_id": section["current_revision_id"],
        "idempotency_key": "client-generation-key-123456",
        "options": {"voice": "af_heart", "format": "mp3"},
    }

    response = client_audio_studio_generation.post(
        f"/api/v1/audio-studio/projects/{project['project_id']}/generations",
        json=payload,
    )
    duplicate = client_audio_studio_generation.post(
        f"/api/v1/audio-studio/projects/{project['project_id']}/generations",
        json=payload,
    )

    assert response.status_code == 202
    assert duplicate.status_code == 202
    accepted = response.json()
    assert duplicate.json()["job_id"] == accepted["job_id"]
    assert accepted["status"] == "queued"
    assert accepted["provider"] == "tts"
    assert accepted["kind"] == "speech"

    job_response = client_audio_studio_generation.get(
        f"/api/v1/audio-studio/projects/{project['project_id']}/generations/{accepted['job_id']}",
    )
    assert job_response.status_code == 200
    assert job_response.json()["job_id"] == accepted["job_id"]

    db = CollectionsDatabase.for_user(user_id=1)
    project_row = db.get_audio_studio_project_by_project_id(project["project_id"])
    generation_row = db.get_audio_studio_generation_job(
        project_row_id=project_row.id,
        job_id=accepted["job_id"],
    )
    assert "api_key" not in generation_row.request_json
    assert "secret" not in generation_row.request_json.lower()


def test_generation_endpoint_rejects_stale_revision_before_job_creation(
    client_audio_studio_generation: TestClient,
) -> None:
    project, _section = _create_project_and_section(client_audio_studio_generation)

    response = client_audio_studio_generation.post(
        f"/api/v1/audio-studio/projects/{project['project_id']}/generations",
        json={
            "kind": "speech",
            "provider": "tts",
            "target_resource_kind": "section",
            "target_resource_id": "sec_001",
            "target_revision_id": "rev_stale",
            "idempotency_key": "client-generation-key-stale",
            "options": {"voice": "af_heart"},
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "stale_target_revision"


def test_provider_and_artifact_listing_do_not_expose_secrets(
    client_audio_studio_generation: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_BASE_URL", "https://ace.example.test")
    monkeypatch.setenv("AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", "https://ace.example.test")
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_API_KEY", "secret-provider-key")
    project, _section = _create_project_and_section(client_audio_studio_generation)

    providers = client_audio_studio_generation.get("/api/v1/audio-studio/providers")
    artifacts = client_audio_studio_generation.get(
        f"/api/v1/audio-studio/projects/{project['project_id']}/artifacts",
    )

    assert providers.status_code == 200
    assert artifacts.status_code == 200
    assert [row["provider_id"] for row in providers.json()["providers"]] == ["tts", "ace_step"]
    assert "secret-provider-key" not in json.dumps(providers.json())
    assert artifacts.json() == {"artifacts": [], "limit": 100, "offset": 0}

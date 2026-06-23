"""Integration tests for Audio Studio render/export endpoints."""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio_studio import router as audio_studio_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_audio_studio_render_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    app = FastAPI()
    app.include_router(audio_studio_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client, tmp_path
    app.dependency_overrides.clear()


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24_000)
        wav_file.writeframes(b"\x01\x00" * 1200)


def _create_project_and_artifact(client: TestClient, tmp_path: Path) -> tuple[dict, dict]:
    create_response = client.post(
        "/api/v1/audio-studio/projects",
        json={"title": "Narration", "workflow": "narration"},
    )
    assert create_response.status_code == 200
    project = create_response.json()
    db = CollectionsDatabase.for_user(user_id=1)
    project_row = db.get_audio_studio_project_by_project_id(project["project_id"])
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path)
    content = wav_path.read_bytes()
    artifact = db.create_audio_studio_artifact(
        project_row_id=project_row.id,
        artifact_id="art_clip",
        artifact_type="clip_audio",
        provider="tts",
        output_id=None,
        storage_path=str(wav_path),
        mime_type="audio/wav",
        size_bytes=len(content),
        source_resource_kind="clip",
        source_resource_id="clip_001",
        source_revision_id=project["current_revision_id"],
        content_hash=hashlib.sha256(content).hexdigest(),
        metadata_json=json.dumps({"duration_ms": 50}),
    )
    return project, {"artifact_id": artifact.artifact_id, "content_hash": artifact.content_hash}


def test_render_endpoint_creates_distinct_idempotent_job_with_manifest(
    client_audio_studio_render_export,
) -> None:
    client, tmp_path = client_audio_studio_render_export
    project, artifact = _create_project_and_artifact(client, tmp_path)
    payload = {
        "render_type": "preview_mix",
        "target_resource_kind": "render",
        "target_resource_id": "render_001",
        "target_revision_id": project["current_revision_id"],
        "idempotency_key": "client-render-key-123456",
        "options": {
            "output_format": "wav",
            "artifact_refs": [
                {
                    "artifact_id": artifact["artifact_id"],
                    "source_revision_id": project["current_revision_id"],
                }
            ],
        },
    }

    response = client.post(
        f"/api/v1/audio-studio/projects/{project['project_id']}/renders",
        json=payload,
    )
    duplicate = client.post(
        f"/api/v1/audio-studio/projects/{project['project_id']}/renders",
        json=payload,
    )

    assert response.status_code == 202
    assert duplicate.status_code == 202
    accepted = response.json()
    assert duplicate.json()["job_id"] == accepted["job_id"]
    assert accepted["job_type"] == "audio_studio_render"
    assert accepted["render_id"] == "render_001"
    assert accepted["manifest"]["source_artifacts"][0]["content_hash"] == artifact["content_hash"]

    job_response = client.get(
        f"/api/v1/audio-studio/projects/{project['project_id']}/renders/{accepted['job_id']}",
    )
    assert job_response.status_code == 200
    assert job_response.json()["job_id"] == accepted["job_id"]


def test_export_endpoint_creates_distinct_idempotent_job_with_source_hashes(
    client_audio_studio_render_export,
) -> None:
    client, tmp_path = client_audio_studio_render_export
    project, artifact = _create_project_and_artifact(client, tmp_path)
    payload = {
        "export_type": "zip_package",
        "target_resource_kind": "export",
        "target_resource_id": "export_001",
        "target_revision_id": project["current_revision_id"],
        "idempotency_key": "client-export-key-123456",
        "options": {
            "artifact_refs": [
                {
                    "artifact_id": artifact["artifact_id"],
                    "source_revision_id": project["current_revision_id"],
                }
            ],
        },
    }

    response = client.post(
        f"/api/v1/audio-studio/projects/{project['project_id']}/exports",
        json=payload,
    )
    duplicate = client.post(
        f"/api/v1/audio-studio/projects/{project['project_id']}/exports",
        json=payload,
    )

    assert response.status_code == 202
    assert duplicate.status_code == 202
    accepted = response.json()
    assert duplicate.json()["job_id"] == accepted["job_id"]
    assert accepted["job_type"] == "audio_studio_export"
    assert accepted["export_id"] == "export_001"
    assert accepted["manifest"]["source_artifacts"][0]["content_hash"] == artifact["content_hash"]

    job_response = client.get(
        f"/api/v1/audio-studio/projects/{project['project_id']}/exports/{accepted['job_id']}",
    )
    assert job_response.status_code == 200
    assert job_response.json()["job_id"] == accepted["job_id"]

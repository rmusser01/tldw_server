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


def _b06_queued_speech(client):
    project, section = _create_project_and_section(client)
    path = f"/api/v1/audio-studio/projects/{project['project_id']}"
    accepted = client.post(path + "/generations", json={
        "kind": "speech", "provider": "tts", "target_resource_kind": "section",
        "target_resource_id": "sec_001", "target_revision_id": section["current_revision_id"],
        "idempotency_key": "b06-real-queued-generation", "options": {"model": "pocket-tts", "voice": "reference", "format": "wav"},
    })
    assert accepted.status_code == 202  # nosec B101 - Real Jobs admission, not a fake worker result.
    return path, accepted.json()["job_id"]


def test_b06_generation_status_uses_actual_jobs_state(client_audio_studio_generation):
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    client = client_audio_studio_generation
    path, job_id = _b06_queued_speech(client)
    jm = JobManager()
    row = jm.get_job_by_uuid(job_id)
    assert jm.cancel_job(row["id"], expected_uuid=job_id, expected_domain="audio_studio", expected_job_type="audio_studio_generate")  # nosec B101 - Guarded real Jobs cancellation.
    response = client.get(path + "/generations/" + job_id)
    assert response.status_code == 200  # nosec B101 - Owned status endpoint.
    assert response.json()["status"] == "cancelled"  # nosec B101 - A stale domain index must not claim queued after actual cancellation.


def test_b06_project_jobs_reopen_without_cross_project_or_owner_rows(client_audio_studio_generation):
    client = client_audio_studio_generation
    path, job_id = _b06_queued_speech(client)
    _other_path, other_job_id = _b06_queued_speech(client)
    response = client.get(path + "/jobs")
    assert response.status_code == 200  # nosec B101 - Project job discovery survives page reload.
    assert [row["job_id"] for row in response.json()["jobs"]] == [job_id]  # nosec B101 - No other project job or payload.
    assert other_job_id not in response.text  # nosec B101 - Exact foreign UUID exclusion.
    async def ordinary_other():
        return User(id=2, username="other", email="other@example.com", is_active=True, is_admin=False)
    client.app.dependency_overrides[get_request_user] = ordinary_other
    assert client.get(path + "/jobs").status_code == 404  # nosec B101 - Preserve owner admission before Jobs lookup.


def test_b06_ordinary_owner_can_cancel_queued_job_without_admin_access(client_audio_studio_generation):
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    client = client_audio_studio_generation
    path, job_id = _b06_queued_speech(client)
    async def ordinary_owner():
        return User(id=1, username="ordinary", email="ordinary@example.com", is_active=True, is_admin=False)
    client.app.dependency_overrides[get_request_user] = ordinary_owner
    response = client.post(path + "/jobs/" + job_id + "/cancel")
    assert response.status_code == 200  # nosec B101 - Owner cancellation is not an admin route.
    assert response.json()["status"] == "cancelled"  # nosec B101 - Real queued work is terminally cancelled before provider execution.
    assert JobManager().get_job_by_uuid(job_id)["status"] == "cancelled"  # nosec B101 - Actual manager state, not UI-only cancellation.


def test_b06_cancel_race_cannot_revoke_running_worker_lease(client_audio_studio_generation, monkeypatch):
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    client = client_audio_studio_generation
    path, job_id = _b06_queued_speech(client)
    cancel = JobManager.cancel_job
    acquired = []
    def race(self, job_row_id, **kwargs):
        job = self.acquire_next_job(domain="audio_studio", queue="default", lease_seconds=60, worker_id="b06-racing-worker", owner_user_id="1")
        acquired.append(job)
        return cancel(self, job_row_id, **kwargs)
    monkeypatch.setattr(JobManager, "cancel_job", race)
    response = client.post(path + "/jobs/" + job_id + "/cancel")
    assert response.status_code == 409  # nosec B101 - Queued-only cancellation must refuse the acquisition race.
    actual = JobManager().get_job_by_uuid(job_id)
    assert (actual["status"], actual["worker_id"], actual["lease_id"]) == ("processing", acquired[0]["worker_id"], acquired[0]["lease_id"])  # nosec B101 - Preserve exact running incarnation/lease.


def test_b06_foreign_owner_cannot_cancel_or_read_generation(client_audio_studio_generation):
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    client = client_audio_studio_generation
    path, job_id = _b06_queued_speech(client)
    async def other_owner():
        return User(id=2, username="other", email="other@example.com", is_active=True, is_admin=False)
    client.app.dependency_overrides[get_request_user] = other_owner
    assert client.post(path + "/jobs/" + job_id + "/cancel").status_code == 404  # nosec B101 - Known UUID cannot bypass project ownership.
    assert client.get(path + "/generations/" + job_id).status_code == 404  # nosec B101 - No foreign generation state disclosure.
    assert JobManager().get_job_by_uuid(job_id)["status"] == "queued"  # nosec B101 - Refused cancellation leaves the actual row intact.


def test_b06_jobs_cursor_keeps_all_rows_without_duplicates(client_audio_studio_generation):
    client = client_audio_studio_generation
    project, section = _create_project_and_section(client)
    path = "/api/v1/audio-studio/projects/" + project["project_id"]
    expected = []
    for number in range(3):
        response = client.post(path + "/generations", json={"kind": "speech", "provider": "tts", "target_resource_kind": "section", "target_resource_id": "sec_001", "target_revision_id": section["current_revision_id"], "idempotency_key": f"b06-cursor-generation-{number}", "options": {}})
        assert response.status_code == 202  # nosec B101 - Real distinct accepted jobs, with actual timestamp ties when present.
        expected.append(response.json()["job_id"])
    params = {"limit": 1}
    seen = []
    for _ in range(4):
        response = client.get(path + "/jobs", params=params)
        assert response.status_code == 200  # nosec B101 - Both cursor timestamp/id accepted through public validation.
        page = response.json()
        seen.extend(row["job_id"] for row in page["jobs"])
        if page["next_before_id"] is None:
            break
        params.update(before_id=page["next_before_id"], created_before=page["next_created_before"])
    assert len(seen) == len(set(seen)) == len(expected) and set(seen) == set(expected)  # nosec B101 - Stable complete boundary, including equal-created-time id ordering.

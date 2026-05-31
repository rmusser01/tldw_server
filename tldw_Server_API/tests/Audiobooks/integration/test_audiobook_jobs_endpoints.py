import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.endpoints.audio.audiobooks import router as audiobooks_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

pytestmark = pytest.mark.integration


@pytest.fixture()
def client_audiobooks_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))

    app = FastAPI()
    app.include_router(audiobooks_router, prefix="/api/v1")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture()
def job_payload():
    return {
        "project_title": "Example Book",
        "source": {"input_type": "txt", "raw_text": "Hello world."},
        "chapters": [
            {"chapter_id": "ch_001", "include": True, "voice": "af_heart", "speed": 1.0}
        ],
        "output": {"merge": True, "per_chapter": True, "formats": ["mp3"]},
        "subtitles": {"formats": ["srt"], "mode": "sentence", "variant": "wide"},
        "queue": {"priority": 3, "batch_group": "batch_01"},
    }


class _FakeAudiobookCollectionsDB:
    def __init__(self):
        self.user_id = "1"
        self.project = SimpleNamespace(
            id=10,
            project_id="abk_test",
            settings_json=json.dumps({"project_id": "abk_test"}),
        )
        self.chapter_calls: list[tuple[int, int]] = []
        self.artifact_calls: list[tuple[int, int]] = []

    def get_audiobook_project(self, project_id: int):
        if project_id == self.project.id:
            return self.project
        raise KeyError("audiobook_project_not_found")

    def get_audiobook_project_by_project_id(self, project_id: str):
        if project_id == self.project.project_id:
            return self.project
        raise KeyError("audiobook_project_not_found")

    def list_audiobook_chapters(self, *, project_id: int, limit: int, offset: int):
        self.chapter_calls.append((limit, offset))
        rows = [
            SimpleNamespace(
                id=101 + index,
                project_id=project_id,
                chapter_index=index,
                title=f"Chapter {index + 1}",
                start_offset=index * 100,
                end_offset=(index + 1) * 100,
                voice_profile_id=None,
                speed=None,
                metadata_json=json.dumps({"chapter_id": f"ch_{index + 1:03d}"}),
            )
            for index in range(3)
        ]
        return rows[offset : offset + limit]

    def list_audiobook_artifacts(self, *, project_id: int, limit: int, offset: int):
        self.artifact_calls.append((limit, offset))
        rows = [
            SimpleNamespace(
                id=201 + index,
                project_id=project_id,
                artifact_type="audio" if index % 2 == 0 else "subtitle",
                format="mp3" if index % 2 == 0 else "srt",
                output_id=301 + index,
                metadata_json=json.dumps({"scope": "chapter", "chapter_id": f"ch_{index + 1:03d}"}),
            )
            for index in range(3)
        ]
        return rows[offset : offset + limit]


@pytest.fixture()
def client_audiobook_project_lists():
    app = FastAPI()
    app.include_router(audiobooks_router, prefix="/api/v1")
    fake_db = _FakeAudiobookCollectionsDB()

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    async def override_collections_db():
        return fake_db

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_collections_db_for_user] = override_collections_db
    app.state.fake_collections_db = fake_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_create_job_status_and_artifacts(client_audiobooks_jobs, job_payload):
    create_resp = client_audiobooks_jobs.post("/api/v1/audiobooks/jobs", json=job_payload)
    assert create_resp.status_code == 200
    data = create_resp.json()
    assert data["status"] == "queued"
    assert isinstance(data["job_id"], int)
    assert data["project_id"].startswith("abk_")

    from tldw_Server_API.app.core.Jobs.manager import JobManager

    jm = JobManager()
    job = jm.get_job(int(data["job_id"]))
    assert job is not None
    assert job.get("batch_group") == "batch_01"
    progress_payload = {
        "stage": "audiobook_tts",
        "chapter_index": 1,
        "chapters_total": 3,
        "item_index": 0,
        "items_total": 1,
    }
    jm.update_job_progress(
        int(data["job_id"]),
        progress_percent=33,
        progress_message=json.dumps(progress_payload),
    )

    status_resp = client_audiobooks_jobs.get(f"/api/v1/audiobooks/jobs/{data['job_id']}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == data["job_id"]
    assert status_data["project_id"] == data["project_id"]
    assert status_data["status"] in {"queued", "processing", "completed", "failed", "canceled"}
    progress = status_data.get("progress") or {}
    assert progress.get("chapter_index") == 1
    assert progress.get("chapters_total") == 3
    assert progress.get("item_index") == 0
    assert progress.get("items_total") == 1

    artifacts_resp = client_audiobooks_jobs.get(f"/api/v1/audiobooks/jobs/{data['job_id']}/artifacts")
    assert artifacts_resp.status_code == 200
    artifacts = artifacts_resp.json()
    assert artifacts["project_id"] == data["project_id"]
    assert isinstance(artifacts["artifacts"], list)


def test_job_access_denied_for_other_user(client_audiobooks_jobs, job_payload):
    create_resp = client_audiobooks_jobs.post("/api/v1/audiobooks/jobs", json=job_payload)
    assert create_resp.status_code == 200
    data = create_resp.json()

    async def override_other_user():
        return User(id=2, username="other", email="o@e.com", is_active=True, is_admin=False)

    client_audiobooks_jobs.app.dependency_overrides[get_request_user] = override_other_user

    status_resp = client_audiobooks_jobs.get(f"/api/v1/audiobooks/jobs/{data['job_id']}")
    assert status_resp.status_code == 404

    artifacts_resp = client_audiobooks_jobs.get(f"/api/v1/audiobooks/jobs/{data['job_id']}/artifacts")
    assert artifacts_resp.status_code == 404


def test_project_chapters_include_canonical_offset_pagination(client_audiobook_project_lists) -> None:
    """Project chapter lists expose canonical metadata while preserving chapters."""
    response = client_audiobook_project_lists.get(
        "/api/v1/audiobooks/projects/abk_test/chapters",
        params={"limit": 1, "offset": 0},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [chapter["title"] for chapter in payload["chapters"]] == ["Chapter 1"]
    assert payload["project_id"] == "abk_test"
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert payload["has_more"] is True
    assert payload["next_offset"] == 1
    assert payload["pagination"] == {
        "mode": "offset",
        "total": None,
        "limit": 1,
        "offset": 0,
        "has_more": True,
        "next_offset": 1,
    }
    assert client_audiobook_project_lists.app.state.fake_collections_db.chapter_calls == [(2, 0)]


def test_project_artifacts_include_canonical_offset_pagination(client_audiobook_project_lists) -> None:
    """Project artifact lists expose canonical metadata while preserving artifacts."""
    response = client_audiobook_project_lists.get(
        "/api/v1/audiobooks/projects/abk_test/artifacts",
        params={"limit": 1, "offset": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [artifact["output_id"] for artifact in payload["artifacts"]] == [302]
    assert payload["project_id"] == "abk_test"
    assert payload["limit"] == 1
    assert payload["offset"] == 1
    assert payload["has_more"] is True
    assert payload["next_offset"] == 2
    assert payload["pagination"] == {
        "mode": "offset",
        "total": None,
        "limit": 1,
        "offset": 1,
        "has_more": True,
        "next_offset": 2,
    }
    assert client_audiobook_project_lists.app.state.fake_collections_db.artifact_calls == [(2, 1)]

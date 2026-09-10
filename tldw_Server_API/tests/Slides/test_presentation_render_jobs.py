import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps import get_slides_db_for_user
from tldw_Server_API.app.api.v1.endpoints.slides import router as slides_router
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Slides.presentation_rendering import PresentationRenderResult
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase
from tldw_Server_API.app.services import presentation_render_jobs_worker


def _user_overrides(app: FastAPI, tmp_path: Path) -> None:
    async def _override_user():
        return User(id=1, username="tester", email=None, is_active=True, is_admin=True)

    async def _override_principal(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",
            jti=None,
            roles=["admin"],
            permissions=["media.create", "media.read", "media.update", "media.delete"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    async def _override_db():
        db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
        try:
            yield db
        finally:
            db.close_connection()

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal
    app.dependency_overrides[get_slides_db_for_user] = _override_db


class _FakeCollectionsDB:
    def __init__(self) -> None:
        self.rows = []
        self.list_calls: list[dict[str, object]] = []

    def list_output_artifacts(self, **kwargs):
        self.list_calls.append(dict(kwargs))
        return list(self.rows), len(self.rows)


@pytest.fixture()
def render_jobs_client(tmp_path, monkeypatch):
    user_db_base = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
    jobs_db_path = tmp_path / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    app = FastAPI()
    app.include_router(slides_router, prefix="/api/v1", tags=["slides"])
    _user_overrides(app, tmp_path)
    fake_collections = _FakeCollectionsDB()

    async def _override_collections_db():
        return fake_collections

    app.dependency_overrides[get_collections_db_for_user] = _override_collections_db

    with TestClient(app) as client:
        yield client, jobs_db_path, fake_collections

    app.dependency_overrides.clear()


def _create_presentation(client: TestClient) -> tuple[str, str]:
    response = client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "Deck",
            "theme": "black",
            "slides": [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "Deck",
                    "content": "",
                    "speaker_notes": "Intro narration",
                    "metadata": {},
                }
            ],
            "studio_data": {"origin": "blank"},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"], response.headers["ETag"]


def test_submit_render_job_snapshots_current_presentation_version(render_jobs_client):
    client, jobs_db_path, _fake_collections = render_jobs_client
    presentation_id, etag = _create_presentation(client)

    response = client.post(
        f"/api/v1/slides/presentations/{presentation_id}/render-jobs",
        json={"format": "mp4"},
        headers={"If-Match": etag},
    )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["job_type"] == "presentation_render"
    assert body["presentation_version"] == 1

    job = JobManager(Path(jobs_db_path)).get_job(int(body["job_id"]))
    assert job is not None
    assert job["payload"]["presentation_id"] == presentation_id
    assert job["payload"]["presentation_version"] == 1
    assert job["payload"]["format"] == "mp4"


def test_render_job_endpoints_offload_blocking_calls_and_use_configured_queue(render_jobs_client, monkeypatch):
    client, jobs_db_path, fake_collections = render_jobs_client
    presentation_id, etag = _create_presentation(client)
    monkeypatch.setenv("PRESENTATION_RENDER_JOBS_QUEUE", "presentation-low")
    offloaded_calls: list[str] = []

    async def _fake_to_thread(func, *args, **kwargs):
        offloaded_calls.append(getattr(func, "__name__", repr(func)))
        return func(*args, **kwargs)

    monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.slides.asyncio.to_thread", _fake_to_thread)
    fake_collections.rows = [
        type(
            "OutputRow",
            (),
            {
                "id": 7,
                "format": "mp4",
                "type": "presentation_render",
                "title": "Deck Render",
                "metadata_json": json.dumps(
                    {
                        "origin": "presentation_studio",
                        "presentation_id": presentation_id,
                        "presentation_version": 1,
                    }
                ),
            },
        )()
    ]

    create_response = client.post(
        f"/api/v1/slides/presentations/{presentation_id}/render-jobs",
        json={"format": "mp4"},
        headers={"If-Match": etag},
    )

    assert create_response.status_code == 202, create_response.text
    job_id = int(create_response.json()["job_id"])
    job = JobManager(Path(jobs_db_path)).get_job(job_id)
    assert job is not None
    assert job["queue"] == "low"

    status_response = client.get(f"/api/v1/slides/render-jobs/{job_id}")
    assert status_response.status_code == 200, status_response.text

    artifacts_response = client.get(f"/api/v1/slides/presentations/{presentation_id}/render-artifacts")
    assert artifacts_response.status_code == 200, artifacts_response.text

    assert "create_job" in offloaded_calls
    assert "get_job" in offloaded_calls
    assert "list_output_artifacts" in offloaded_calls
    assert fake_collections.list_calls
    assert fake_collections.list_calls[-1]["metadata_presentation_id"] == presentation_id


def test_submit_render_job_rejects_when_rendering_disabled(render_jobs_client, monkeypatch):
    client, _jobs_db_path, _fake_collections = render_jobs_client
    presentation_id, etag = _create_presentation(client)
    monkeypatch.setenv("PRESENTATION_RENDER_ENABLED", "false")

    response = client.post(
        f"/api/v1/slides/presentations/{presentation_id}/render-jobs",
        json={"format": "mp4"},
        headers={"If-Match": etag},
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == "presentation_render_unavailable"


def test_list_render_artifacts_returns_outputs_for_presentation(render_jobs_client):
    client, _jobs_db_path, fake_collections = render_jobs_client
    presentation_id, _etag = _create_presentation(client)
    row = type(
        "OutputRow",
        (),
        {
            "id": 7,
            "format": "mp4",
            "type": "presentation_render",
            "title": "Deck Render",
            "metadata_json": json.dumps(
                {
                    "origin": "presentation_studio",
                    "presentation_id": presentation_id,
                    "presentation_version": 1,
                }
            ),
        },
    )()
    fake_collections.rows = [row]

    response = client.get(f"/api/v1/slides/presentations/{presentation_id}/render-artifacts")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["presentation_id"] == presentation_id
    assert payload["artifacts"][0]["output_id"] == row.id
    assert payload["artifacts"][0]["format"] == "mp4"
    assert payload["artifacts"][0]["download_url"] == f"/api/v1/outputs/{row.id}/download"


def test_collections_output_listing_filters_by_presentation_id_metadata(tmp_path, monkeypatch):
    user_db_base = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
    with CollectionsDatabase.for_user(user_id="1") as collections_db:
        match = collections_db.create_output_artifact(
            job_id=1,
            type_="presentation_render",
            title="Deck A",
            format_="mp4",
            storage_path="deck-a.mp4",
            metadata_json=json.dumps(
                {
                    "origin": "presentation_studio",
                    "presentation_id": "deck-a",
                    "presentation_version": 1,
                }
            ),
        )
        collections_db.create_output_artifact(
            job_id=2,
            type_="presentation_render",
            title="Deck B",
            format_="mp4",
            storage_path="deck-b.mp4",
            metadata_json=json.dumps(
                {
                    "origin": "presentation_studio",
                    "presentation_id": "deck-b",
                    "presentation_version": 1,
                }
            ),
        )

        rows, total = collections_db.list_output_artifacts(
            limit=50,
            offset=0,
            type_="presentation_render",
            metadata_origin="presentation_studio",
            metadata_presentation_id="deck-a",
        )

    assert total == 1
    assert [row.id for row in rows] == [match.id]


def test_render_worker_helpers_have_docstrings():
    helpers = [
        presentation_render_jobs_worker._jobs_manager,
        presentation_render_jobs_worker._coerce_int,
        presentation_render_jobs_worker._build_worker_config,
        presentation_render_jobs_worker._resolve_queue_name,
        presentation_render_jobs_worker._normalize_payload,
        presentation_render_jobs_worker._resolve_user_id,
        presentation_render_jobs_worker._create_slides_db,
    ]

    for helper in helpers:
        assert helper.__doc__ is not None
        assert helper.__doc__.strip()


@pytest.mark.asyncio
async def test_render_worker_persists_output_artifact(monkeypatch, tmp_path):
    user_db_base = tmp_path / "user_dbs"
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
    jobs_db_path = tmp_path / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))

    slides_db_path = DatabasePaths.get_slides_db_path(1)
    slides_db = SlidesDatabase(db_path=str(slides_db_path), client_id="1")
    try:
        presentation = slides_db.create_presentation(
            presentation_id=None,
            title="Deck",
            description=None,
            theme="black",
            marp_theme=None,
            settings=None,
            studio_data=json.dumps({"origin": "blank"}),
            template_id=None,
            slides=json.dumps(
                [
                    {
                        "order": 0,
                        "layout": "title",
                        "title": "Deck",
                        "content": "",
                        "speaker_notes": "Intro",
                        "metadata": {},
                    }
                ]
            ),
            slides_text="Deck\nIntro",
            source_type="manual",
            source_ref=None,
            source_query=None,
            custom_css=None,
        )
    finally:
        slides_db.close_connection()

    jm = JobManager(Path(jobs_db_path))
    created = jm.create_job(
        domain="presentation_render",
        queue="default",
        job_type="presentation_render",
        payload={
            "user_id": 1,
            "presentation_id": presentation.id,
            "presentation_version": 1,
            "format": "mp4",
        },
        owner_user_id="1",
    )
    job = jm.get_job(int(created["id"]))
    assert job is not None

    output_file = tmp_path / "rendered.mp4"
    output_file.write_bytes(b"video-bytes")
    render_kwargs: dict[str, object] = {}

    def _fake_render_presentation_video(**kwargs):
        render_kwargs.update(kwargs)
        return PresentationRenderResult(
            output_format="mp4",
            storage_path="rendered.mp4",
            output_path=output_file,
            byte_size=len(b"video-bytes"),
        )

    monkeypatch.setattr(
        presentation_render_jobs_worker,
        "render_presentation_video",
        _fake_render_presentation_video,
    )

    recorded: dict[str, object] = {}

    class _FakeCollections:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def create_output_artifact(self, **kwargs):
            recorded.update(kwargs)
            return SimpleNamespace(id=99)

    monkeypatch.setattr(
        presentation_render_jobs_worker.CollectionsDatabase,
        "for_user",
        lambda user_id: _FakeCollections(),
    )

    result = await presentation_render_jobs_worker.process_presentation_render_job(
        job,
        job_manager=jm,
        worker_id="test-worker",
    )

    assert result["output_id"] == 99
    assert result["presentation_id"] == presentation.id
    assert result["presentation_version"] == 1
    assert recorded["type_"] == "presentation_render"
    metadata = json.loads(str(recorded["metadata_json"]))
    assert metadata["presentation_id"] == presentation.id
    assert metadata["presentation_version"] == 1
    assert render_kwargs["user_id"] == 1


@pytest.mark.asyncio
async def test_render_worker_rejects_forged_standalone_job_before_snapshot_or_render(
    monkeypatch,
):
    events: list[str] = []

    class _StandaloneSlidesDB:
        def get_presentation_kind(self, presentation_id: str):
            assert presentation_id == "html-deck"
            events.append("kind")
            return SimpleNamespace(content_kind="standalone_html")

        def close_connection(self):
            events.append("close")

    def _snapshot_must_not_load(*_args, **_kwargs):
        events.append("snapshot")
        raise AssertionError("standalone payload must not reach snapshot loading")

    def _render_must_not_start(*_args, **_kwargs):
        events.append("render")
        raise AssertionError("standalone payload must not reach Playwright or ffmpeg")

    monkeypatch.setattr(
        presentation_render_jobs_worker,
        "_create_slides_db",
        lambda _user_id: _StandaloneSlidesDB(),
    )
    monkeypatch.setattr(
        presentation_render_jobs_worker,
        "load_presentation_render_snapshot",
        _snapshot_must_not_load,
    )
    monkeypatch.setattr(
        presentation_render_jobs_worker,
        "render_presentation_video",
        _render_must_not_start,
    )
    job_manager = SimpleNamespace(update_job_progress=lambda *_args, **_kwargs: None)
    forged_job = {
        "id": 77,
        "job_type": "presentation_render",
        "owner_user_id": "1",
        "payload": {
            "user_id": 1,
            "presentation_id": "html-deck",
            "presentation_version": 9,
            "format": "mp4",
        },
    }

    with pytest.raises(
        presentation_render_jobs_worker.PresentationRenderJobError,
        match="operation_not_supported_for_content_kind",
    ):
        await presentation_render_jobs_worker.process_presentation_render_job(
            forged_job,
            job_manager=job_manager,
            worker_id="forged-job-test",
        )

    assert events == ["kind", "close"]

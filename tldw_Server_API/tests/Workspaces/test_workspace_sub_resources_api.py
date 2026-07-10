"""Tests for workspace sub-resource API endpoints."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError


@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.upsert_workspace("ws-1", "Test WS")
    return d


async def _allow_rate_limit() -> None:
    return None


async def _reviewer() -> SimpleNamespace:
    return SimpleNamespace(id="reviewer-7")


@pytest.fixture
def workspace_source_client(db: CharactersRAGDB) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    app.dependency_overrides[get_request_user] = _reviewer
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: None
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _source_payload(source_id: str, **overrides: object) -> dict[str, object]:
    return {
        "id": source_id,
        "media_id": 1,
        "title": f"Source {source_id}",
        "source_type": "pdf",
        **overrides,
    }


class TestSourceEndpoints:
    def test_add_and_list_sources(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "Video",
            "source_type": "video",
        })
        sources = db.list_workspace_sources("ws-1")
        assert len(sources) == 1
        assert sources[0]["title"] == "Video"

    def test_update_source_returns_409_on_stale_version(self, db):
        db.add_workspace_source("ws-1", {
            "id": "src-1", "media_id": 1, "title": "X",
            "source_type": "video",
        })
        db.update_workspace_source("ws-1", "src-1", {"title": "Y"}, expected_version=1)
        with pytest.raises(ConflictError):
            db.update_workspace_source("ws-1", "src-1", {"title": "Z"}, expected_version=1)


@pytest.mark.integration
def test_create_source_defaults_review_state_to_unset(
    workspace_source_client: TestClient,
) -> None:
    response = workspace_source_client.post(
        "/api/v1/workspaces/ws-1/sources",
        json=_source_payload("src-default"),
    )

    assert response.status_code == 201, response.text
    source = response.json()
    assert source["review_state"] == "unset"
    assert source["review_state_updated_at"]
    assert source["reviewed_at"] is None
    assert source["reviewed_by_user_id"] is None


@pytest.mark.integration
def test_create_source_accepts_needs_review(
    workspace_source_client: TestClient,
) -> None:
    response = workspace_source_client.post(
        "/api/v1/workspaces/ws-1/sources",
        json=_source_payload("src-needs-review", review_state="needs_review"),
    )

    assert response.status_code == 201, response.text
    assert response.json()["review_state"] == "needs_review"


@pytest.mark.integration
def test_create_source_rejects_reviewed_state(
    workspace_source_client: TestClient,
) -> None:
    response = workspace_source_client.post(
        "/api/v1/workspaces/ws-1/sources",
        json=_source_payload("src-invalid-create", review_state="reviewed"),
    )

    assert response.status_code == 422, response.text


@pytest.mark.integration
def test_update_source_to_reviewed_records_authenticated_user(
    workspace_source_client: TestClient,
) -> None:
    created = workspace_source_client.post(
        "/api/v1/workspaces/ws-1/sources",
        json=_source_payload("src-reviewed"),
    ).json()

    response = workspace_source_client.put(
        "/api/v1/workspaces/ws-1/sources/src-reviewed",
        json={"review_state": "reviewed", "version": created["version"]},
    )

    assert response.status_code == 200, response.text
    source = response.json()
    assert source["review_state"] == "reviewed"
    assert source["review_state_updated_at"]
    assert source["reviewed_at"]
    assert source["reviewed_by_user_id"] == "reviewer-7"


@pytest.mark.integration
def test_update_source_to_needs_review_clears_reviewed_fields(
    workspace_source_client: TestClient,
) -> None:
    created = workspace_source_client.post(
        "/api/v1/workspaces/ws-1/sources",
        json=_source_payload("src-reset"),
    ).json()
    reviewed = workspace_source_client.put(
        "/api/v1/workspaces/ws-1/sources/src-reset",
        json={"review_state": "reviewed", "version": created["version"]},
    ).json()

    response = workspace_source_client.put(
        "/api/v1/workspaces/ws-1/sources/src-reset",
        json={"review_state": "needs_review", "version": reviewed["version"]},
    )

    assert response.status_code == 200, response.text
    source = response.json()
    assert source["review_state"] == "needs_review"
    assert source["review_state_updated_at"]
    assert source["reviewed_at"] is None
    assert source["reviewed_by_user_id"] is None


@pytest.mark.integration
def test_update_source_rejects_invalid_review_state(
    workspace_source_client: TestClient,
) -> None:
    created = workspace_source_client.post(
        "/api/v1/workspaces/ws-1/sources",
        json=_source_payload("src-invalid-update"),
    ).json()

    response = workspace_source_client.put(
        "/api/v1/workspaces/ws-1/sources/src-invalid-update",
        json={"review_state": "approved", "version": created["version"]},
    )

    assert response.status_code == 422, response.text


@pytest.mark.integration
def test_batch_review_state_update_returns_updated_sources(
    workspace_source_client: TestClient,
) -> None:
    for source_id in ("src-batch-1", "src-batch-2"):
        response = workspace_source_client.post(
            "/api/v1/workspaces/ws-1/sources",
            json=_source_payload(source_id),
        )
        assert response.status_code == 201, response.text

    response = workspace_source_client.put(
        "/api/v1/workspaces/ws-1/sources/review-state",
        json={
            "source_ids": ["src-batch-2", "src-batch-1"],
            "review_state": "reviewed",
        },
    )

    assert response.status_code == 200, response.text
    sources = response.json()
    assert [source["id"] for source in sources] == ["src-batch-2", "src-batch-1"]
    assert all(source["review_state"] == "reviewed" for source in sources)
    assert all(source["reviewed_by_user_id"] == "reviewer-7" for source in sources)


@pytest.mark.integration
def test_batch_review_state_update_is_atomic_when_source_is_missing(
    workspace_source_client: TestClient,
    db: CharactersRAGDB,
) -> None:
    for source_id in ("src-atomic-1", "src-atomic-2"):
        response = workspace_source_client.post(
            "/api/v1/workspaces/ws-1/sources",
            json=_source_payload(source_id),
        )
        assert response.status_code == 201, response.text

    response = workspace_source_client.put(
        "/api/v1/workspaces/ws-1/sources/review-state",
        json={
            "source_ids": ["src-atomic-1", "src-missing"],
            "review_state": "reviewed",
        },
    )

    assert response.status_code == 409, response.text
    sources = {source["id"]: source for source in db.list_workspace_sources("ws-1")}
    assert sources["src-atomic-1"]["review_state"] == "unset"
    assert sources["src-atomic-2"]["review_state"] == "unset"


class TestArtifactEndpoints:
    def test_add_and_list_artifacts(self, db):
        db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "Summary",
        })
        arts = db.list_workspace_artifacts("ws-1")
        assert len(arts) == 1

    def test_update_artifact_returns_409_on_stale_version(self, db):
        db.add_workspace_artifact("ws-1", {
            "id": "art-1", "artifact_type": "summary", "title": "X",
        })
        db.update_workspace_artifact("ws-1", "art-1", {"title": "Y"}, expected_version=1)
        with pytest.raises(ConflictError):
            db.update_workspace_artifact("ws-1", "art-1", {"title": "Z"}, expected_version=1)


class TestNoteEndpoints:
    def test_add_and_list_notes(self, db):
        db.add_workspace_note("ws-1", {"title": "N1", "content": "body"})
        notes = db.list_workspace_notes("ws-1")
        assert len(notes) == 1
        assert notes[0]["title"] == "N1"

    def test_soft_delete_note_hides_from_list(self, db):
        note = db.add_workspace_note("ws-1", {"title": "N1", "content": ""})
        db.delete_workspace_note("ws-1", note["id"])
        assert db.list_workspace_notes("ws-1") == []

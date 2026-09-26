"""Tests for workspace sub-resource API endpoints."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
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
def test_update_source_treats_null_review_state_as_omitted(
    workspace_source_client: TestClient,
) -> None:
    created = workspace_source_client.post(
        "/api/v1/workspaces/ws-1/sources",
        json=_source_payload("src-null-review-state"),
    ).json()

    response = workspace_source_client.put(
        "/api/v1/workspaces/ws-1/sources/src-null-review-state",
        json={"title": "Renamed", "review_state": None, "version": created["version"]},
    )

    assert response.status_code == 200, response.text
    source = response.json()
    assert source["title"] == "Renamed"
    assert source["review_state"] == "unset"
    assert source["version"] == created["version"] + 1


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


@pytest.mark.integration
@pytest.mark.parametrize("source_id", ["", "   "], ids=["empty", "blank"])
def test_batch_review_state_update_rejects_blank_source_ids(
    workspace_source_client: TestClient,
    source_id: str,
) -> None:
    response = workspace_source_client.put(
        "/api/v1/workspaces/ws-1/sources/review-state",
        json={"source_ids": [source_id], "review_state": "needs_review"},
    )

    assert response.status_code == 422, response.text


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


@pytest.fixture
def workspace_notes_client(db: CharactersRAGDB) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(kind="user", user_id=7)
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=7)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    for rate_limit in (
        WORKSPACES_READ_RATE_LIMIT,
        WORKSPACES_WRITE_RATE_LIMIT,
        WORKSPACES_DELETE_RATE_LIMIT,
    ):
        app.dependency_overrides[rate_limit] = _allow_rate_limit
    with TestClient(app) as client:
        yield client


@pytest.mark.integration
@pytest.mark.parametrize("resource", ["sources", "artifacts", "notes?limit=1&offset=1"])
@pytest.mark.parametrize("expected_user_id", ["8", "", "not-a-user"])
def test_bundle_reads_reject_mismatched_expected_user_before_db_access(
    workspace_notes_client: TestClient,
    resource: str,
    expected_user_id: str,
) -> None:
    db_accesses = []

    def unexpected_db_access() -> CharactersRAGDB:
        db_accesses.append(True)
        raise HTTPException(status_code=500, detail="Bundle DB acquired before scope check")

    workspace_notes_client.app.dependency_overrides[get_chacha_db_for_user] = unexpected_db_access
    response = workspace_notes_client.get(
        f"/api/v1/workspaces/ws-1/{resource}",
        headers={"X-TLDW-Expected-User-ID": expected_user_id},
    )
    assert response.status_code == 412, response.text
    assert db_accesses == []
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"]["code"] == "request_config_scope_changed"


@pytest.mark.integration
@pytest.mark.parametrize("resource", ["sources", "artifacts"])
@pytest.mark.parametrize("expected_user_id", [None, "7", " 7 "], ids=["absent", "matching", "padded"])
def test_bundle_reads_preserve_content_with_optional_expected_user(
    workspace_notes_client: TestClient,
    db: CharactersRAGDB,
    resource: str,
    expected_user_id: str | None,
) -> None:
    db.add_workspace_source("ws-1", _source_payload("src-1"))
    db.add_workspace_artifact("ws-1", {
        "id": "art-1", "artifact_type": "summary", "title": "Owned artifact",
    })
    headers = {} if expected_user_id is None else {"X-TLDW-Expected-User-ID": expected_user_id}
    response = workspace_notes_client.get(f"/api/v1/workspaces/ws-1/{resource}", headers=headers)
    assert response.status_code == 200, response.text
    assert [(row["id"], row["workspace_id"], row["title"]) for row in response.json()] == (
        [("src-1", "ws-1", "Source src-1")]
        if resource == "sources" else [("art-1", "ws-1", "Owned artifact")]
    )


@pytest.mark.integration
@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "DELETE"])
def test_notes_reject_mismatched_expected_user_before_db_access(
    workspace_notes_client: TestClient,
    method: str,
) -> None:
    db_accesses = []

    def unexpected_db_access() -> CharactersRAGDB:
        db_accesses.append(True)
        raise HTTPException(status_code=500, detail="Notes DB acquired before scope check")

    workspace_notes_client.app.dependency_overrides[get_chacha_db_for_user] = unexpected_db_access
    path = "/api/v1/workspaces/ws-1/notes"
    if method in {"PUT", "DELETE"}:
        path += "/1"
    response = workspace_notes_client.request(
        method,
        path,
        headers={"X-TLDW-Expected-User-ID": "8"},
        json={"title": "Blocked", "content": "Blocked", "version": 1},
    )

    assert response.status_code == 412, response.text
    assert db_accesses == []
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"]["code"] == "request_config_scope_changed"


@pytest.mark.integration
@pytest.mark.parametrize("expected_user_id", [None, "7"], ids=["absent", "matching"])
def test_notes_preserve_crud_with_optional_expected_user(
    workspace_notes_client: TestClient,
    db: CharactersRAGDB,
    expected_user_id: str | None,
) -> None:
    headers = {} if expected_user_id is None else {"X-TLDW-Expected-User-ID": expected_user_id}
    path = "/api/v1/workspaces/ws-1/notes"
    created = workspace_notes_client.post(
        path, headers=headers, json={"title": "Original", "content": "Note body"}
    )
    assert created.status_code == 201, created.text
    note = created.json()
    assert note["title"] == "Original"
    assert note["content"] == "Note body"

    listed = workspace_notes_client.get(path, headers=headers)
    assert listed.status_code == 200, listed.text
    assert listed.json() == [note]

    note_path = f"{path}/{note['id']}"
    updated = workspace_notes_client.put(
        note_path, headers=headers, json={"title": "Updated", "version": note["version"]}
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["title"] == "Updated"
    assert updated.json()["version"] == note["version"] + 1
    assert db.list_workspace_notes("ws-1")[0]["title"] == "Updated"

    stale = workspace_notes_client.put(
        note_path, headers=headers, json={"title": "Stale", "version": note["version"]}
    )
    assert stale.status_code == 409, stale.text

    deleted = workspace_notes_client.delete(note_path, headers=headers)
    assert deleted.status_code == 204, deleted.text
    assert deleted.content == b""
    assert db.list_workspace_notes("ws-1") == []


@pytest.mark.integration
@pytest.mark.parametrize("path", ["/api/v1/workspaces/", "/api/v1/workspaces/ws-1/context"])
def test_workspace_directory_rejects_changed_account_before_db_access(
    workspace_notes_client: TestClient,
    path: str,
) -> None:
    db_accesses = []

    def unexpected_db_access() -> CharactersRAGDB:
        db_accesses.append(True)
        raise HTTPException(status_code=500, detail="DB acquired before scope check")

    workspace_notes_client.app.dependency_overrides[get_chacha_db_for_user] = unexpected_db_access
    response = workspace_notes_client.get(path, headers={"X-TLDW-Expected-User-ID": "8"})
    assert response.status_code == 412, response.text
    assert db_accesses == []
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.integration
@pytest.mark.parametrize("method", ["GET", "PATCH"])
def test_workspace_metadata_rejects_mismatched_expected_user_before_db_access(
    workspace_notes_client: TestClient,
    method: str,
) -> None:
    db_accesses = []

    def unexpected_db_access() -> CharactersRAGDB:
        db_accesses.append(True)
        raise HTTPException(status_code=500, detail="Workspace DB acquired before scope check")

    workspace_notes_client.app.dependency_overrides[get_chacha_db_for_user] = unexpected_db_access
    response = workspace_notes_client.request(
        method,
        "/api/v1/workspaces/ws-1",
        headers={"X-TLDW-Expected-User-ID": "8"},
        json={"name": "Blocked", "version": 1},
    )

    assert response.status_code == 412, response.text
    assert db_accesses == []
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"]["code"] == "request_config_scope_changed"


@pytest.mark.integration
@pytest.mark.parametrize("expected_user_id", [None, "7"], ids=["absent", "matching"])
def test_workspace_metadata_preserves_read_patch_and_conflict_with_optional_expected_user(
    workspace_notes_client: TestClient,
    db: CharactersRAGDB,
    expected_user_id: str | None,
) -> None:
    headers = {} if expected_user_id is None else {"X-TLDW-Expected-User-ID": expected_user_id}
    path = "/api/v1/workspaces/ws-1"
    fetched = workspace_notes_client.get(path, headers=headers)
    assert fetched.status_code == 200, fetched.text
    workspace = fetched.json()
    assert workspace["id"] == "ws-1"
    assert workspace["name"] == "Test WS"

    updated = workspace_notes_client.patch(
        path, headers=headers, json={"name": "Renamed", "version": workspace["version"]}
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["name"] == "Renamed"
    assert updated.json()["version"] == workspace["version"] + 1

    stale = workspace_notes_client.patch(
        path, headers=headers, json={"name": "Stale", "version": workspace["version"]}
    )
    assert stale.status_code == 409, stale.text
    stored = db.get_workspace("ws-1")
    assert stored["name"] == "Renamed"
    assert stored["version"] == updated.json()["version"]

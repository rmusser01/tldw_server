"""Tests for workspace sub-resource API endpoints."""

import sqlite3
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.utils.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.workspace_artifact_catalog import list_workspace_artifact_revisions
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError


@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.upsert_workspace("ws-1", "Test WS")
    return d


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def catalog_db(request: pytest.FixtureRequest) -> Iterator[CharactersRAGDB]:
    if request.param == "sqlite":
        yield request.getfixturevalue("db")
        return
    backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    database = CharactersRAGDB(":memory:", client_id="user-1", backend=backend)
    try:
        database.upsert_workspace("ws-1", "Workspace")
        yield database
    finally:
        database.close_all_connections()
        backend.get_pool().close_all()


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
    def test_revision_inventory_avoids_reading_artifact_bodies(self, db: CharactersRAGDB) -> None:
        db.add_workspace_artifact("ws-1", {"id": "a", "artifact_type": "brief", "title": "Findings"})
        connection = db.get_connection()

        def forbid_payload(action, table, column, _database, _origin):
            if action == sqlite3.SQLITE_READ and table == "workspace_artifacts" and column in {
                "content", "review_metadata_json", "export_refs_json", "source_lineage_json",
            }:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(forbid_payload)
        try:
            assert [row["id"] for row in list_workspace_artifact_revisions(db, "ws-1")] == ["a"]
        finally:
            connection.set_authorizer(None)

    def test_catalog_defers_only_compatible_terminal_section_jobs(self, catalog_db: CharactersRAGDB) -> None:
        for artifact_id, extra, client in [
            ("complete", {}, {}),
            ("completed", {"status": "completed", "artifact_type": "compare_sources"}, {"id": "completed"}),
            ("failure", {"status": "failed", "artifact_type": "report"}, {"removedFromStudio": False}),
            ("working", {"status": "generating"}, {}),
            ("other_type", {"artifact_type": "infographic"}, {}),
            ("other_producer", {"producer_metadata": {"origin": "acp"}}, {}),
            ("alias", {}, {"id": "different"}),
            ("bad-alias", {}, {"id": "x/y"}),
            ("null-alias", {}, {"id": None}),
            ("hidden", {}, {"removedFromStudio": True}),
            ("archived", {"review_state": "archived"}, {}),
            ("future", {}, {"schemaVersion": 2}),
            ("malformed", {}, {"schemaVersion": True}),
        ]:
            catalog_db.add_workspace_artifact("ws-1", {
                "id": artifact_id, "title": artifact_id, "artifact_type": "summary", "status": "complete",
                "producer_metadata": {"origin": "research_workspace_output_job"},
                "version_metadata": {"workspaceClient": {"schemaVersion": 1, **client}}, **extra,
            })
        rows = list_workspace_artifact_revisions(catalog_db, "ws-1")
        assert {row["id"]: row["display_id"] for row in rows if row["defer_detail"]} == {
            "complete": "complete", "completed": "completed", "failure": "failure", "alias": "different",
        }

    @pytest.mark.parametrize("origin", ["research_pack_import", "workspace_snapshot_import"])
    def test_imported_section_identity_tracks_logical_and_server_versions(
        self, catalog_db: CharactersRAGDB, origin: str,
    ) -> None:
        producer = {"origin": "research_workspace_output_job", "settings": {"model": "luna", "provider": "openai"}}
        catalog_db.add_workspace_artifact("ws-1", {
            "id": "portable", "artifact_type": "summary", "title": "Reviewed", "status": "completed",
            "producer_metadata": {"origin": origin, "importedProducer": producer},
            "version_metadata": {"workspaceClient": {"schemaVersion": 1, "id": "display", "version": 311}},
        })
        row, = list_workspace_artifact_revisions(catalog_db, "ws-1")
        assert (row["defer_detail"], row["display_id"], row["display_version"], row["version"]) == (
            True, "display", 311, 1,
        )
        assert (row["imported"], row["model"], row["provider"]) == (True, "luna", "openai")

    def test_catalog_retains_section_with_null_character_in_evidence(self, catalog_db: CharactersRAGDB) -> None:
        catalog_db.add_workspace_artifact("ws-1", {
            "id": "legacy", "title": "Legacy", "artifact_type": "report", "status": "completed",
            "producer_metadata": {"origin": "research_workspace_output_job"},
            "version_metadata": {"workspaceClient": {"schemaVersion": 1, "data": {"evidence": "x\x00y"}}},
        })
        row, = list_workspace_artifact_revisions(catalog_db, "ws-1")
        assert row["id"] == "legacy"

    @pytest.mark.integration
    def test_artifact_revisions_and_detail_match_canonical_workspace(
        self, db: CharactersRAGDB, workspace_source_client: TestClient,
    ) -> None:
        base = "/api/v1/workspaces/ws-1"
        art = db.add_workspace_artifact("ws-1", {
            "id": "report", "artifact_type": "brief", "title": "Findings",
            "content": "Retained section. " * 10_000,
            "review_metadata": {"claims": [{"quote": "Exact passage", "page": 42}]},
        })
        db.update_workspace_artifact("ws-1", "report", {"review_state": "accepted"}, expected_version=1)
        db.append_workspace_artifact_export_ref("ws-1", "report", {"format": "md"})
        db.upsert_workspace("ws-other", "Other workspace")
        db.add_workspace_artifact("ws-other", {"id": "foreign", "artifact_type": "brief", "title": "Foreign"})

        revisions = workspace_source_client.get(f"{base}/artifact-revisions")
        assert revisions.status_code == 200, revisions.text
        assert revisions.json() == [{
            "id": "report", "workspace_id": "ws-1", "version": 2,
            "created_at": art["created_at"], "title": "Findings", "artifact_type": "brief",
            "review_state": "accepted", "status": "pending", "defer_detail": False, "display_id": "report",
            "display_version": 2, "imported": False, "model": None, "provider": None,
        }]
        detail = workspace_source_client.get(f"{base}/artifacts/report")
        assert detail.status_code == 200, detail.text
        assert detail.json()["content"] == art["content"]
        assert detail.json()["review_metadata"] == art["review_metadata"]
        assert [detail.json()] == workspace_source_client.get(f"{base}/artifacts").json()
        assert workspace_source_client.get(f"{base}/artifacts/foreign").status_code == 404
        db.delete_workspace_artifact("ws-1", "report")
        assert workspace_source_client.get(f"{base}/artifact-revisions").json() == []
        assert workspace_source_client.get(f"{base}/artifacts/report").status_code == 404

    @pytest.mark.integration
    @pytest.mark.parametrize("suffix", ["artifact-revisions", "artifacts/report"])
    def test_artifact_reads_require_visible_workspace(
        self, db: CharactersRAGDB, workspace_source_client: TestClient, suffix: str,
    ) -> None:
        db.add_workspace_artifact("ws-1", {"id": "report", "artifact_type": "brief", "title": "Findings"})
        db.delete_workspace("ws-1", expected_version=1)
        response = workspace_source_client.get(f"/api/v1/workspaces/ws-1/{suffix}")
        assert response.status_code == 404
        assert response.json()["detail"] == "Workspace not found"

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

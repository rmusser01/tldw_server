from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import flashcards as flashcards_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager


pytestmark = pytest.mark.integration


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "study-pack-endpoints.db"), client_id="study-pack-endpoints-tests")
    chacha.upsert_workspace("ws-1", "Workspace 1")
    try:
        yield chacha
    finally:
        chacha.close_connection()


@pytest.fixture
def jobs_db_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "study-pack-jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(path))
    return path


@pytest.fixture
def client(db: CharactersRAGDB, jobs_db_path: Path):
    app = FastAPI()
    app.include_router(flashcards_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db

    async def _override_user() -> User:
        return User(id=1, username="tester", email="t@example.com", is_active=True, roles=["admin"], is_admin=True)

    async def _override_principal() -> AuthPrincipal:
        return AuthPrincipal(kind="user", user_id=1, roles=["admin"], permissions=["*"], is_admin=True)

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def _jobs_manager(jobs_db_path: Path) -> JobManager:
    return JobManager(db_path=jobs_db_path)


def _complete_job(jm: JobManager, job_id: int) -> dict[str, object]:
    acquired = jm.acquire_next_job(domain="study_packs", queue="default", lease_seconds=30, worker_id="study-pack-test")
    assert acquired is not None  # nosec B101
    assert int(acquired["id"]) == job_id  # nosec B101
    return acquired


def _seed_pack(db: CharactersRAGDB, *, title: str = "Networks 101") -> tuple[int, int]:
    deck_id = db.add_deck(f"{title} Deck", workspace_id="ws-1")
    pack_id = db.create_study_pack(
        title=title,
        workspace_id="ws-1",
        deck_id=deck_id,
        source_bundle_json={
            "items": [
                {
                    "source_type": "note",
                    "source_id": "note-1",
                    "label": "Seed note",
                    "locator": {"note_id": "note-1"},
                }
            ]
        },
        generation_options_json={"deck_mode": "new"},
    )
    return pack_id, deck_id


def test_create_study_pack_job_enqueues_user_scoped_job(
    client: TestClient,
    jobs_db_path: Path,
):
    response = client.post(
        "/api/v1/flashcards/study-packs/jobs",
        json={
            "title": "Biology",
            "workspace_id": "ws-1",
            "source_items": [{"source_type": "note", "source_id": "note-1"}],
        },
    )

    assert response.status_code == 202  # nosec B101
    body = response.json()
    assert body["job"]["status"] in {"queued", "running"}  # nosec B101

    job = _jobs_manager(jobs_db_path).get_job(int(body["job"]["id"]))

    assert job is not None  # nosec B101
    assert job["domain"] == "study_packs"  # nosec B101
    assert job["job_type"] == "study_pack_generate"  # nosec B101
    assert str(job["owner_user_id"]) == "1"  # nosec B101
    assert job["payload"]["title"] == "Biology"  # nosec B101


def test_get_study_pack_job_status_returns_completed_pack_result(
    client: TestClient,
    db: CharactersRAGDB,
    jobs_db_path: Path,
):
    pack_id, deck_id = _seed_pack(db, title="TCP Fundamentals")
    jm = _jobs_manager(jobs_db_path)
    job = jm.create_job(
        domain="study_packs",
        queue="default",
        job_type="study_pack_generate",
        payload={
            "title": "TCP Fundamentals",
            "workspace_id": "ws-1",
            "source_items": [{"source_type": "note", "source_id": "note-1"}],
        },
        owner_user_id="1",
        priority=5,
        max_retries=2,
    )
    acquired = _complete_job(jm, int(job["id"]))
    jm.complete_job(
        int(job["id"]),
        result={"pack_id": pack_id, "deck_id": deck_id},
        worker_id="study-pack-test",
        lease_id=str(acquired["lease_id"]),
    )

    response = client.get(f"/api/v1/flashcards/study-packs/jobs/{job['id']}")

    assert response.status_code == 200  # nosec B101
    body = response.json()
    assert body["job"]["status"] == "completed"  # nosec B101
    assert int(body["study_pack"]["id"]) == pack_id  # nosec B101
    assert int(body["study_pack"]["deck_id"]) == deck_id  # nosec B101


def test_get_study_pack_detail_returns_persisted_pack(
    client: TestClient,
    db: CharactersRAGDB,
):
    pack_id, deck_id = _seed_pack(db, title="Routing Basics")

    response = client.get(f"/api/v1/flashcards/study-packs/{pack_id}")

    assert response.status_code == 200  # nosec B101
    body = response.json()
    assert int(body["id"]) == pack_id  # nosec B101
    assert int(body["deck_id"]) == deck_id  # nosec B101
    assert body["title"] == "Routing Basics"  # nosec B101


def test_regenerate_study_pack_job_uses_stored_source_bundle(
    client: TestClient,
    db: CharactersRAGDB,
    jobs_db_path: Path,
):
    pack_id, _deck_id = _seed_pack(db, title="OSI Model")

    response = client.post(f"/api/v1/flashcards/study-packs/{pack_id}/regenerate")

    assert response.status_code == 202  # nosec B101
    body = response.json()
    assert body["job"]["status"] in {"queued", "running"}  # nosec B101

    job = _jobs_manager(jobs_db_path).get_job(int(body["job"]["id"]))

    assert job is not None  # nosec B101
    assert int(job["payload"]["regenerate_from_pack_id"]) == pack_id  # nosec B101
    assert int(job["payload"]["expected_version"]) == 1  # nosec B101
    assert job["payload"]["title"] == "OSI Model"  # nosec B101
    assert job["payload"]["source_items"] == [{"source_type": "note", "source_id": "note-1"}]  # nosec B101


def test_failed_study_pack_jobs_return_diagnostics_without_partial_pack(
    client: TestClient,
    jobs_db_path: Path,
):
    jm = _jobs_manager(jobs_db_path)
    job = jm.create_job(
        domain="study_packs",
        queue="default",
        job_type="study_pack_generate",
        payload={
            "title": "Failure Case",
            "workspace_id": "ws-1",
            "source_items": [{"source_type": "note", "source_id": "note-1"}],
        },
        owner_user_id="1",
        priority=5,
        max_retries=2,
    )
    acquired = _complete_job(jm, int(job["id"]))
    jm.fail_job(
        int(job["id"]),
        error="llm exploded",
        retryable=False,
        worker_id="study-pack-test",
        lease_id=str(acquired["lease_id"]),
    )

    response = client.get(f"/api/v1/flashcards/study-packs/jobs/{job['id']}")

    assert response.status_code == 200  # nosec B101
    body = response.json()
    assert body["job"]["status"] == "failed"  # nosec B101
    assert body["error"] == "Study pack generation failed."  # nosec B101
    assert body["study_pack"] is None  # nosec B101


def test_admin_job_status_reads_completed_pack_from_owner_database(
    jobs_db_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    admin_db = CharactersRAGDB(str(tmp_path / "study-pack-admin.db"), client_id="study-pack-admin-tests")
    owner_db = CharactersRAGDB(str(tmp_path / "study-pack-owner.db"), client_id="study-pack-owner-tests")
    admin_db.upsert_workspace("ws-1", "Workspace 1")
    owner_db.upsert_workspace("ws-1", "Workspace 1")

    async def _override_user() -> User:
        return User(id=1, username="admin", email="admin@example.com", is_active=True, roles=["admin"], is_admin=True)

    async def _override_principal() -> AuthPrincipal:
        return AuthPrincipal(kind="user", user_id=1, roles=["admin"], permissions=["*"], is_admin=True)

    async def _owner_db_lookup(owner_user_id: int):
        assert owner_user_id == 2  # nosec B101
        return owner_db

    monkeypatch.setattr(flashcards_endpoints, "get_chacha_db_for_owner", _owner_db_lookup, raising=False)

    app = FastAPI()
    app.include_router(flashcards_endpoints.router, prefix="/api/v1")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: admin_db
    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal

    try:
        pack_id, deck_id = _seed_pack(owner_db, title="Owner Networks 101")
        jm = _jobs_manager(jobs_db_path)
        job = jm.create_job(
            domain="study_packs",
            queue="default",
            job_type="study_pack_generate",
            payload={
                "title": "Owner Networks 101",
                "workspace_id": "ws-1",
                "source_items": [{"source_type": "note", "source_id": "note-1"}],
            },
            owner_user_id="2",
            priority=5,
            max_retries=2,
        )
        acquired = _complete_job(jm, int(job["id"]))
        jm.complete_job(
            int(job["id"]),
            result={"pack_id": pack_id, "deck_id": deck_id},
            worker_id="study-pack-test",
            lease_id=str(acquired["lease_id"]),
        )

        with TestClient(app) as test_client:
            response = test_client.get(f"/api/v1/flashcards/study-packs/jobs/{job['id']}")

        assert response.status_code == 200  # nosec B101
        body = response.json()
        assert body["job"]["status"] == "completed"  # nosec B101
        assert int(body["study_pack"]["id"]) == pack_id  # nosec B101
        assert int(body["study_pack"]["deck_id"]) == deck_id  # nosec B101
    finally:
        app.dependency_overrides.clear()
        admin_db.close_connection()
        owner_db.close_connection()

"""Exercise native Study publication and replay through the Workspace API."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import flashcards, quizzes, workspaces
from tldw_Server_API.app.api.v1.utils.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

ROOT = "/api/v1/workspaces/ws-1/study-outputs"
pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgres"])
def db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[CharactersRAGDB]:
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgres"
        else None
    )
    repository = CharactersRAGDB(str(tmp_path / "study.db"), client_id="1", backend=backend)
    for workspace_id in ("ws-1", "ws-2"):
        repository.workspace_store.upsert_workspace(workspace_id, workspace_id)
        repository.workspace_store.add_workspace_source(
            workspace_id,
            {"id": "source-1", "media_id": 12, "title": "Source", "source_type": "pdf"},
        )
    yield repository
    repository.close_all_connections()


@pytest.mark.parametrize(
    "drop_table",
    [
        pytest.param("DROP TABLE document_chat_links", id="study-preview"),
        pytest.param("DROP TABLE research_study_outputs", id="document-workspace"),
    ],
)
def test_upgrade_installs_both_tables_from_the_previous_version(db: CharactersRAGDB, drop_table: str) -> None:
    version_query = "SELECT version FROM db_schema_version WHERE schema_name = ?"
    version = db.execute_query(version_query, (db._SCHEMA_NAME,)).fetchone()["version"]
    with db.transaction() as conn:
        conn.execute(drop_table)
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (version - 1, db._SCHEMA_NAME),
        )
    db.close_connection()

    upgraded = CharactersRAGDB(str(db.db_path), client_id="1", backend=db.backend)
    try:
        upgraded.execute_query("SELECT 1 FROM document_chat_links").fetchall()
        upgraded.execute_query("SELECT 1 FROM research_study_outputs").fetchall()
        assert upgraded.workspace_store.get_workspace("ws-1") is not None
        assert upgraded.execute_query(version_query, (db._SCHEMA_NAME,)).fetchone()["version"] == version
    finally:
        upgraded.close_connection()


async def _allow_rate_limit() -> None:
    pass


@pytest.fixture
def client(db: CharactersRAGDB) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(workspaces.router, prefix="/api/v1/workspaces")
    app.include_router(flashcards.router, prefix="/api/v1")
    app.include_router(quizzes.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    with TestClient(app, raise_server_exceptions=False) as http:
        yield http


def _request(kind: str = "flashcards") -> dict:
    item = (
        {"front": "What is relevant?", "back": "A retained example."}
        if kind == "flashcards"
        else {
            "question": "Which option is in the source?",
            "options": ["First", "Second"],
            "correct_index": 1,
            "explanation": "The second option matches.",
        }
    )
    return {
        "operation_id": "op-1",
        "title": "Example output",
        "output_kind": kind,
        "source_snapshot": [
            {
                "source_id": "source-1",
                "catalog_item_id": "12",
                "association_version": 1,
                "source_type": "pdf",
                "ready_modes": ["fts"],
            },
        ],
        "processing_route_fingerprint": "a" * 64,
        "settings_fingerprint": "b" * 64,
        "provider": "mock",
        "model": "sample",
        "items": [{**item, "citations": ["source-1"]}],
    }


@pytest.mark.parametrize("kind", ["flashcards", "quiz"])
def test_save_replay_and_reopen_native_study(client: TestClient, db: CharactersRAGDB, kind: str) -> None:
    assert client.get(ROOT).json() == {
        "contract": "research-study.v1",
        "items": [],
        "limit": 20,
        "offset": 0,
        "total": 0,
    }
    payload = _request(kind)
    created = client.post(ROOT, json=payload)
    assert created.status_code == 200, created.text
    receipt = created.json()
    owner_id = receipt["owner_id"]
    assert receipt["workspace_id"] == "ws-1"
    assert receipt["generation"]["operation_id"] == "op-1"
    assert receipt["generation"]["source_snapshot"] == payload["source_snapshot"]
    assert "items" not in receipt and "request_fingerprint" not in receipt

    if kind == "flashcards":
        parent = client.get(f"/api/v1/flashcards/decks/{owner_id}").json()
        cards = db.flashcard_store.list_flashcards(deck_id=owner_id)
        assert cards[0]["front"] == payload["items"][0]["front"]
        assert '"source_id": "source-1"' in cards[0]["extra"]
    else:
        parent = client.get(f"/api/v1/quizzes/{owner_id}").json()
        questions = client.get(f"/api/v1/quizzes/{owner_id}/questions", params={"include_answers": True}).json()
        assert questions["items"][0]["correct_answer"] == 1
        assert questions["items"][0]["source_citations"][0]["media_id"] == 12
    assert parent["workspace_id"] == "ws-1"
    assert parent["version"] == receipt["version"]
    assert client.get(f"{ROOT}/{kind}/{owner_id}").json() == receipt

    db.workspace_store.update_workspace_source("ws-1", "source-1", {"selected": False}, expected_version=1)
    assert client.post(ROOT, json=payload).json() == receipt
    assert client.post(ROOT, json={**payload, "title": "Different payload"}).status_code == 409
    assert client.get(ROOT).json()["items"] == [receipt]
    assert client.get(ROOT, params={"limit": 1, "offset": 1}).json()["total"] == 1
    assert client.get(f"{ROOT.replace('ws-1', 'ws-2')}/{kind}/{owner_id}").status_code == 404


@pytest.mark.parametrize("kind", ["flashcards", "quiz"])
def test_concurrent_identical_requests_commit_one_native_owner(client: TestClient, kind: str) -> None:
    payload = _request(kind)
    with ThreadPoolExecutor(max_workers=2) as executor:
        requests = [executor.submit(client.post, ROOT, json=payload) for _ in range(2)]
        responses = [request.result(timeout=30) for request in requests]
    assert [response.status_code for response in responses] == [200, 200]
    assert responses[0].json() == responses[1].json()
    assert client.get(ROOT).json()["total"] == 1


@pytest.mark.parametrize("kind", ["flashcards", "quiz"])
def test_child_write_failure_rolls_back_parent_and_leaves_retryable_key(
    client: TestClient, db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    payload = _request(kind)
    payload["items"] *= 2
    parent = db.flashcard_store if kind == "flashcards" else db
    method = "add_flashcard" if kind == "flashcards" else "create_question"
    original = getattr(parent, method)
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second child failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(parent, method, fail_second)
        assert client.post(ROOT, json=payload).status_code == 500
    assert not db.flashcard_store.list_decks(workspace_id="ws-1")
    assert db.list_quizzes(workspace_id="ws-1")["count"] == 0
    assert client.get(ROOT).json()["total"] == 0
    assert client.post(ROOT, json=payload).status_code == 200


@pytest.mark.parametrize("kind", ["flashcards", "quiz"])
def test_moved_or_deleted_owner_cannot_be_recreated(client: TestClient, db: CharactersRAGDB, kind: str) -> None:
    payload = _request(kind)
    owner_id = client.post(ROOT, json=payload).json()["owner_id"]
    if kind == "flashcards":
        assert db.flashcard_store.update_deck(owner_id, workspace_id="ws-2")
    else:
        assert db.update_quiz(owner_id, {"workspace_id": "ws-2"})
    assert client.post(ROOT, json=payload).status_code == 409
    assert client.get(f"{ROOT}/{kind}/{owner_id}").status_code == 404
    assert client.get(ROOT).json()["total"] == 0
    if kind == "flashcards":
        db.flashcard_store.soft_delete_deck_by_id(owner_id)
    else:
        assert db.delete_quiz(owner_id)
    assert client.post(ROOT, json=payload).status_code == 409
    assert client.post(ROOT, json={**payload, "operation_id": "op-2"}).json()["owner_id"] != owner_id


def test_new_save_checks_source_version_media_and_citations(client: TestClient) -> None:
    expected = _request()
    for changed in ({"association_version": 2}, {"source_id": "other"}, {"catalog_item_id": "13"}):
        payload = _request()
        payload["source_snapshot"][0].update(changed)
        payload["items"][0]["citations"] = [payload["source_snapshot"][0]["source_id"]]
        assert client.post(ROOT, json=payload).status_code == 409
    expected["items"][0]["citations"] = ["other"]
    assert client.post(ROOT, json=expected).status_code == 422
    assert client.get(ROOT).json()["total"] == 0


def test_existing_sqlite_study_data_survives_the_receipt_migration(tmp_path, monkeypatch) -> None:
    path = tmp_path / "v85.sqlite"
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 85)
        before = CharactersRAGDB(path, client_id="1")
        before.workspace_store.upsert_workspace("ws", "Workspace")
        old_deck = before.flashcard_store.add_deck("Existing deck", workspace_id="ws")
        before.close_all_connections()

    after = CharactersRAGDB(path, client_id="1")
    assert after.flashcard_store.get_deck(old_deck)["name"] == "Existing deck"
    # Read through the normal store API after real v85-to-v86 initialization.
    from tldw_Server_API.app.core.Research_Workspace.study_publication import ResearchStudyPublisher

    assert ResearchStudyPublisher(after).page("ws", limit=1, offset=0).total == 0
    after.close_all_connections()

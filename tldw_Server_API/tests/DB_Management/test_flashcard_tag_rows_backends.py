"""Tag mutations preserve real SQLite and PostgreSQL row/transaction contracts."""

import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import flashcards
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def tags_db(request, tmp_path):
    """Use the official per-test PostgreSQL fixture and a real SQLite control."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "tags.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _card(db, *, linked=False):
    card = db.add_flashcard({"front": "Marker?", "back": "Amber"})
    if linked:
        # Establish real existing links through the normal first-update path.
        assert db.update_flashcard(card, {}, expected_version=1, tags=["Old", "Keep"])
    return card


def _tags(db, card):
    return sorted(row["keyword"] for row in db.get_keywords_for_flashcard(card))


def test_http_create_with_tags_populates_actual_keyword_links(tags_db):
    app = FastAPI()
    app.include_router(flashcards.router)
    app.dependency_overrides[flashcards.get_chacha_db_for_user] = lambda: tags_db
    with TestClient(app, raise_server_exceptions=False) as client:
        result = client.post("/flashcards", json={"front": "Marker?", "back": "Amber", "tags": ["Citrine"]})
    assert result.status_code == 200, result.text
    card = result.json()["uuid"]
    assert _tags(tags_db, card) == ["Citrine"]
    assert json.loads(tags_db.get_flashcard(card)["tags_json"]) == ["Citrine"]


def test_http_patch_replaces_existing_keyword_links(tags_db):
    card = _card(tags_db, linked=True)
    before = tags_db.get_flashcard(card)
    app = FastAPI()
    app.include_router(flashcards.router)
    app.dependency_overrides[flashcards.get_chacha_db_for_user] = lambda: tags_db
    with TestClient(app, raise_server_exceptions=False) as client:
        result = client.patch(f"/flashcards/{card}", json={"expected_version": before["version"], "tags": ["New"]})
    assert result.status_code == 200, result.text
    assert _tags(tags_db, card) == ["New"]
    assert json.loads(tags_db.get_flashcard(card)["tags_json"]) == ["New"]
    assert result.json()["version"] == before["version"] + 1


@pytest.mark.parametrize("linked", [False, True], ids=["first-tags", "replace-tags"])
def test_http_tag_replacement_updates_links_and_json_mirror(tags_db, linked):
    db = tags_db
    card = _card(db, linked=linked)
    before = db.get_flashcard(card)
    app = FastAPI()
    app.include_router(flashcards.router)
    app.dependency_overrides[flashcards.get_chacha_db_for_user] = lambda: db
    with TestClient(app, raise_server_exceptions=False) as client:
        result = client.put(f"/flashcards/{card}/tags", json={"tags": [" Keep ", "New", " "]})
    assert result.status_code == 200, result.text
    assert result.json()["version"] == before["version"] + 1
    assert _tags(db, card) == ["Keep", "New"]
    assert json.loads(db.get_flashcard(card)["tags_json"]) == ["Keep", "New"]
    assert db.get_flashcard(card)["front"] == before["front"]
    assert db.get_flashcard(card)["back"] == before["back"]


@pytest.mark.parametrize("operation", ["setter", "update"])
def test_existing_tag_links_can_be_cleared(tags_db, operation):
    db = tags_db
    card = _card(db, linked=True)
    before = db.get_flashcard(card)
    if operation == "setter":
        assert db.set_flashcard_tags(card, [])
    else:
        assert db.update_flashcard(card, {}, expected_version=before["version"], tags=[])
    assert _tags(db, card) == []
    assert json.loads(db.get_flashcard(card)["tags_json"]) == []
    assert db.get_flashcard(card)["version"] == before["version"] + 1


@pytest.mark.parametrize("operation", ["setter", "update"])
def test_tag_mutation_remains_owned_by_outer_transaction(tags_db, operation):
    db = tags_db
    card = _card(db, linked=True)
    before = db.get_flashcard(card)
    with pytest.raises(RuntimeError, match="caller rollback"):
        with db.transaction():
            if operation == "setter":
                assert db.set_flashcard_tags(card, ["Changed"])
            else:
                assert db.update_flashcard(card, {}, expected_version=before["version"], tags=["Changed"])
            assert _tags(db, card) == ["Changed"]
            raise RuntimeError("caller rollback")
    assert db.get_flashcard(card) == before
    assert _tags(db, card) == ["Keep", "Old"]


@pytest.mark.parametrize("deleted", [False, True], ids=["missing", "deleted"])
def test_tag_setter_rejects_missing_or_deleted_card(tags_db, deleted):
    db = tags_db
    card = _card(db) if deleted else str(uuid.uuid4())
    if deleted:
        assert db.soft_delete_flashcard(card, expected_version=1)
    with pytest.raises(CharactersRAGDBError, match="Flashcard not found or deleted"):
        db.set_flashcard_tags(card, ["Unwanted"])
    assert db.get_keyword_by_text("Unwanted") is None

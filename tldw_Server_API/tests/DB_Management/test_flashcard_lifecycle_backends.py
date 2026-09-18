"""Flashcard delete/reset contracts work on real SQLite and PostgreSQL rows."""

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import flashcards
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def lifecycle_db(request, tmp_path):
    """Reuse the official per-test PG fixture, with a real SQLite control."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "lifecycle.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


def _client(db):
    app = FastAPI()
    app.include_router(flashcards.router)
    app.dependency_overrides[flashcards.get_chacha_db_for_user] = lambda: db
    return TestClient(app, raise_server_exceptions=False)


def _saved_card(db):
    deck = db.add_deck("Citrine")
    card_uuid = db.add_flashcard({"deck_id": deck, "front": "Marker?", "back": "Amber", "notes": "Keep notes"})
    with db.transaction() as conn:
        conn.execute(
            """UPDATE flashcards
                  SET ef = 1.8, interval_days = 12, repetitions = 7, lapses = 2,
                      queue_state = 'review', step_index = 1, suspended_reason = 'manual',
                      scheduler_state_json = ?, due_at = '2026-09-20T00:00:00Z',
                      last_reviewed_at = '2026-09-15T00:00:00Z'
                WHERE uuid = ?""",
            ('{"stability":12.5}', card_uuid),
        )
    return db.get_flashcard(card_uuid)


def _request(client, operation, card_uuid, version):
    if operation == "delete":
        return client.delete(f"/flashcards/{card_uuid}", params={"expected_version": version})
    return client.post(f"/flashcards/{card_uuid}/reset-scheduling", json={"expected_version": version})


def test_delete_http_is_versioned_and_idempotent(lifecycle_db):
    db = lifecycle_db
    card = _saved_card(db)
    with _client(db) as client:
        deleted = _request(client, "delete", card["uuid"], card["version"])
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"deleted": True}
        assert db.get_flashcard(card["uuid"]) is None
        row = dict(
            db.execute_query("SELECT * FROM flashcards WHERE uuid = ?", (card["uuid"],), read_only=True).fetchone()
        )
        assert row["deleted"] == 1
        assert row["version"] == card["version"] + 1
        assert row["front"] == card["front"]
        repeated = _request(client, "delete", card["uuid"], card["version"])
        assert repeated.status_code == 200, repeated.text
        assert repeated.json() == {"deleted": True}
        repeated_row = dict(
            db.execute_query("SELECT * FROM flashcards WHERE uuid = ?", (card["uuid"],), read_only=True).fetchone()
        )
        assert repeated_row == row


@pytest.mark.parametrize("via_http", [True, False], ids=["http-version", "db-omitted-version"])
def test_reset_restores_scheduling_defaults_and_preserves_content(lifecycle_db, via_http):
    db = lifecycle_db
    card = _saved_card(db)
    with _client(db) as client:
        if via_http:
            reset = client.post(
                f"/flashcards/{card['uuid']}/reset-scheduling",
                json={"expected_version": card["version"]},
            )
            assert reset.status_code == 200, reset.text
            result = reset.json()
        else:
            assert db.reset_flashcard_scheduling(card["uuid"])
            result = db.get_flashcard(card["uuid"])
        assert result["version"] == card["version"] + 1
        assert result["ef"] == pytest.approx(2.5)
        for field in ("interval_days", "repetitions", "lapses"):
            assert result[field] == 0
        for field in ("last_reviewed_at", "step_index", "suspended_reason"):
            assert result[field] is None
        assert result["queue_state"] == "new"
        assert result["due_at"] is not None
        for field in ("uuid", "deck_id", "front", "back", "notes"):
            assert result[field] == card[field]
        assert db.get_flashcard(card["uuid"])["scheduler_state_json"] == "{}"


def test_reset_http_still_requires_version_without_mutation(lifecycle_db):
    card = _saved_card(lifecycle_db)
    with _client(lifecycle_db) as client:
        response = client.post(f"/flashcards/{card['uuid']}/reset-scheduling", json={})
        assert response.status_code == 422, response.text
    assert lifecycle_db.get_flashcard(card["uuid"]) == card


@pytest.mark.parametrize("operation", ["delete", "reset"])
def test_missing_lifecycle_target_returns_conflict(lifecycle_db, operation):
    with _client(lifecycle_db) as client:
        response = _request(client, operation, str(uuid.uuid4()), 1)
        assert response.status_code == 409, response.text
        assert "Flashcard not found" in response.json()["detail"]


@pytest.mark.parametrize("operation", ["delete", "reset"])
def test_stale_lifecycle_version_returns_conflict_without_mutation(lifecycle_db, operation):
    card = _saved_card(lifecycle_db)
    with _client(lifecycle_db) as client:
        response = _request(client, operation, card["uuid"], card["version"] + 1)
        assert response.status_code == 409, response.text
        assert "Version mismatch" in response.json()["detail"]
    assert lifecycle_db.get_flashcard(card["uuid"]) == card


@pytest.mark.parametrize("operation", ["delete", "reset"])
def test_lifecycle_mutation_remains_owned_by_outer_transaction(lifecycle_db, operation):
    db = lifecycle_db
    card = _saved_card(db)
    method = db.soft_delete_flashcard if operation == "delete" else db.reset_flashcard_scheduling
    with pytest.raises(RuntimeError, match="force caller rollback"):
        with db.transaction():
            assert method(card["uuid"], expected_version=card["version"])
            changed = db.get_flashcard(card["uuid"])
            if operation == "delete":
                assert changed is None
            else:
                assert changed["version"] == card["version"] + 1
            raise RuntimeError("force caller rollback")
    assert db.get_flashcard(card["uuid"]) == card

"""Character API dates identify SQLite UTC instants without rewriting storage."""

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import characters_endpoint as characters
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def character_db(request, tmp_path):
    """Reuse the standard official database fixture for both supported stores."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "character-times.db", client_id="324", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.fixture
def character_client(character_db):
    """Exercise production routing/serialization with only the owned DB injected."""
    app = FastAPI()
    app.include_router(characters.router, prefix="/characters")
    app.dependency_overrides[characters.get_chacha_db_for_user] = lambda: character_db
    with TestClient(app) as client:
        yield client


def _instant(value):
    """Parse a serialized timestamp without supplying an implicit timezone."""
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _readbacks(client, character_id, name):
    """Read one owned Character through each public listing/detail path."""
    requests = [
        (f"/characters/{character_id}", None),
        ("/characters/", None),
        ("/characters/query", {"query": name}),
        ("/characters/search/", {"query": name}),
    ]
    results = []
    for path, params in requests:
        response = client.get(path, params=params)
        assert response.status_code == 200, response.text
        body = response.json()
        rows = body if isinstance(body, list) else body.get("items", [body])
        results.append(next(row for row in rows if row["id"] == character_id))
    return results


def test_character_readbacks_preserve_the_seed_or_created_instant(character_db, character_client):
    db = character_db
    if db.backend_type == BackendType.SQLITE:
        stored = db.get_character_card_by_name("Default Assistant")
    else:
        character_id = db.add_character_card({"name": "Timestamp reader"})
        stored = db.get_character_card_by_id(character_id)
    before = dict(stored)
    expected = _instant(stored["created_at"])
    if db.backend_type == BackendType.SQLITE and expected.tzinfo is None:
        expected = expected.replace(tzinfo=timezone.utc)

    for result in _readbacks(character_client, stored["id"], stored["name"]):
        for field in ("created_at", "updated_at", "last_modified"):
            actual = _instant(result[field])
            assert actual.utcoffset() is not None, (field, result[field])
            assert actual == expected
    assert db.get_character_card_by_id(stored["id"]) == before


@pytest.mark.parametrize("value", [
    "2026-09-19 23:55:19",
    "2026-09-19T23:55:19.123456",
    "2026-09-19T23:55:19.123456Z",
    "2026-09-19T16:55:19.123456-07:00",
])
def test_sqlite_legacy_and_offset_dates_are_explicit_without_storage_changes(tmp_path, value):
    db = CharactersRAGDB(tmp_path / "legacy-times.db", client_id="324")
    try:
        with db.transaction() as conn:
            conn.execute(
                "UPDATE character_cards SET created_at = ?, last_modified = ? WHERE id = ?",
                (value, value, 1),
            )
        before = db.get_character_card_by_id(1)
        app = FastAPI()
        app.include_router(characters.router, prefix="/characters")
        app.dependency_overrides[characters.get_chacha_db_for_user] = lambda: db
        with TestClient(app) as client:
            result = client.get("/characters/1").json()
        expected = _instant(value)
        if expected.tzinfo is None:
            expected = expected.replace(tzinfo=timezone.utc)
        assert _instant(result["created_at"]) == expected
        assert _instant(result["updated_at"]) == expected
        assert db.get_character_card_by_id(1) == before
    finally:
        db.close_all_connections()

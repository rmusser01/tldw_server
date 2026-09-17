"""UAT214: actual character WorldBook reads use the portable read boundary."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import characters_endpoint as characters
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendType, CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def books(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "book-reads.db", client_id="2", backend=backend)
    service = WorldBookService(db)
    character = db.add_character_card({"name": "WorldBook reader"})
    try:
        yield db, service, character
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _seed(db, character):
    # Real table writes isolate the reader contract from unrelated legacy writer APIs.
    with db.transaction() as conn:
        for book_id, name, enabled, deleted, attached_enabled, priority in [
            (101, "Alpha", True, False, True, 2),
            (102, "Beta", True, False, True, 2),
            (103, "Top", True, False, True, 9),
            (104, "Disabled book", False, False, True, 8),
            (105, "Disabled attachment", True, False, False, 7),
            (106, "Deleted book", True, True, True, 10),
        ]:
            conn.execute("INSERT INTO world_books (id, name, enabled, deleted) VALUES (?, ?, ?, ?)",
                         (book_id, name, enabled, deleted))
            conn.execute("INSERT INTO character_world_books (character_id, world_book_id, enabled, priority) "
                         "VALUES (?, ?, ?, ?)", (character, book_id, attached_enabled, priority))
        for enabled in (True, False):
            conn.execute("INSERT INTO world_book_entries (world_book_id, keywords, content, enabled) "
                         "VALUES (?, ?, ?, ?)", (101, '["lore"]', "Fixture lore", enabled))


def test_empty_character_world_books(books):
    _db, service, character = books
    assert service.get_character_world_books(character) == []
    assert service.get_entry_counts_for_world_books([]) == {}


@pytest.mark.parametrize("enabled_only,expected", [
    (True, [103, 101, 102]), (False, [103, 104, 105, 101, 102])
])
def test_populated_character_books_keep_enabled_deleted_and_order(books, enabled_only, expected):
    db, service, character = books
    _seed(db, character)
    rows = service.get_character_world_books(character, enabled_only=enabled_only)
    assert [row["id"] for row in rows] == expected
    assert rows[0]["attachment_priority"] == 9
    assert bool(rows[0]["attachment_enabled"]) is True


def test_entry_counts_normalize_subset_and_keep_zero_and_disabled_entries(books):
    db, service, character = books
    _seed(db, character)
    assert service.get_entry_counts_for_world_books([101, "101", 102, 999, -1, "bad"]) == {
        101: 2, 102: 0, 999: 0
    }
    assert service.get_entry_counts_for_world_books() == {101: 2}


@pytest.mark.parametrize("populated", [False, True])
def test_actual_character_world_books_route_returns_complete_rows(books, populated):
    db, _service, character = books
    if populated:
        _seed(db, character)
    app = FastAPI()
    app.include_router(characters.router, prefix="/api/v1/characters")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/api/v1/characters/{character}/world-books")
    assert response.status_code == 200, response.text
    rows = response.json()
    assert [row["world_book_id"] for row in rows] == ([103, 101, 102] if populated else [])
    if populated:
        assert [row["entry_count"] for row in rows] == [0, 2, 0]


def test_readers_preserve_callers_pending_writes_and_rollback(books):
    db, service, character = books
    _seed(db, character)
    with pytest.raises(RuntimeError, match="caller rollback"):
        with db.transaction() as conn:
            conn.execute("UPDATE world_books SET name = ? WHERE id = ?", ("Pending name", 103))
            conn.execute("INSERT INTO world_book_entries (world_book_id, keywords, content) VALUES (?, ?, ?)",
                         (102, '[]', "Pending entry"))
            assert service.get_character_world_books(character)[0]["name"] == "Pending name"
            assert service.get_entry_counts_for_world_books([102]) == {102: 1}
            raise RuntimeError("caller rollback")
    assert service.get_character_world_books(character)[0]["name"] == "Top"
    assert service.get_entry_counts_for_world_books([102]) == {102: 0}


def test_standalone_world_book_reads_finish_owned_postgres_transaction(books):
    db, service, character = books
    _seed(db, character)
    assert len(service.get_character_world_books(character)) == 3
    if db.backend_type == BackendType.POSTGRESQL:
        assert db.get_connection()._connection.info.transaction_status.name == "IDLE"
    assert service.get_entry_counts_for_world_books([101]) == {101: 2}
    if db.backend_type == BackendType.POSTGRESQL:
        assert db.get_connection()._connection.info.transaction_status.name == "IDLE"


@pytest.mark.parametrize("include_disabled,expected", [
    (False, [101, 102, 105, 103]), (True, [101, 102, 105, 104, 103])
])
def test_world_book_catalogue_filters_deleted_and_disabled_in_name_order(books, include_disabled, expected):
    db, service, character = books
    assert service.list_world_books(include_disabled=include_disabled) == []
    _seed(db, character)
    assert [book["id"] for book in service.list_world_books(include_disabled=include_disabled)] == expected
    if db.backend_type == BackendType.POSTGRESQL:
        assert db.get_connection()._connection.info.transaction_status.name == "IDLE"


@pytest.mark.parametrize("populated", [False, True])
def test_actual_world_book_catalogue_route_returns_rows_and_counts(books, populated):
    db, _service, character = books
    if populated:
        _seed(db, character)
    app = FastAPI()
    app.include_router(characters.router, prefix="/api/v1/characters")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/characters/world-books", params={"include_disabled": True})
    assert response.status_code == 200, response.text
    data = response.json()
    assert [book["id"] for book in data["world_books"]] == ([101, 102, 105, 104, 103] if populated else [])
    assert data["total"] == (5 if populated else 0)
    assert data["enabled_count"] == (4 if populated else 0)
    assert data["disabled_count"] == (1 if populated else 0)
    if populated:
        assert [book["entry_count"] for book in data["world_books"]] == [2, 0, 0, 0, 0]


def test_catalogue_keeps_callers_uncommitted_write_inside_rollback(books):
    db, service, character = books
    _seed(db, character)
    with pytest.raises(RuntimeError, match="caller rollback"):
        with db.transaction() as conn:
            conn.execute("UPDATE world_books SET name = ? WHERE id = ?", ("A pending name", 103))
            pending = {book["id"]: book["name"] for book in service.list_world_books()}
            assert pending[103] == "A pending name"
            raise RuntimeError("caller rollback")
    assert [book["name"] for book in service.list_world_books()] == [
        "Alpha", "Beta", "Disabled attachment", "Top"
    ]

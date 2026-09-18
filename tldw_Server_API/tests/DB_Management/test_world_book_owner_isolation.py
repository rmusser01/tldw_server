"""World Book operations must remain private across actual database owners."""

import sqlite3
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def books(request, tmp_path):
    """Use separate SQLite files or the official shared PostgreSQL fixture."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    alice = CharactersRAGDB(tmp_path / "alice.db", client_id="1", backend=backend)
    bob = CharactersRAGDB(tmp_path / "bob.db", client_id="2", backend=backend)
    try:
        first = WorldBookService(alice)
        book = first.create_world_book("Alice private world", description="Private description")
        entry = first.add_entry(book, ["private"], "Alice private lore")
        alice.close_connection()
        second = WorldBookService(bob)
        character = bob.add_character_card({"name": "Bob character"})
        bob.close_connection()
        yield SimpleNamespace(alice=alice, bob=bob, first=first, second=second,
                              book=book, entry=entry, character=character)
    finally:
        alice.close_all_connections()
        bob.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.mark.parametrize("operation", ["list", "book", "name", "entries", "entry", "search"])
def test_foreign_world_book_content_is_not_visible(books, operation):
    service = books.second
    if operation == "list":
        assert service.list_world_books(include_disabled=True) == []
    elif operation == "book":
        assert service.get_world_book(books.book) is None
    elif operation == "name":
        assert service.get_world_book(name="Alice private world") is None
    elif operation == "entries":
        assert service.get_entries(books.book) == []
    elif operation == "entry":
        assert service.get_entry(books.entry) is None
    else:
        assert service.search_entries(query="private") == []


@pytest.mark.parametrize("operation", ["book-update", "book-delete", "book-purge", "entry-update", "entry-delete", "attach"])
def test_foreign_world_book_mutations_cannot_change_canonical_data(books, operation):
    before_book = books.first.get_world_book(books.book)
    entry_fields = ("id", "world_book_id", "keywords", "content", "priority", "enabled")
    before_entry = {key: books.first.get_entry(books.entry).get(key) for key in entry_fields}
    books.alice.close_connection()
    service = books.second
    if operation == "book-update":
        result = service.update_world_book(books.book, description="Unauthorized replacement")
    elif operation == "book-delete":
        result = service.delete_world_book(books.book)
    elif operation == "book-purge":
        result = service.delete_world_book(books.book, hard_delete=True)
    elif operation == "entry-update":
        result = service.update_entry(books.entry, content="Unauthorized replacement")
    elif operation == "entry-delete":
        result = service.delete_entry(books.entry)
    else:
        result = service.attach_to_character(books.book, books.character)
    books.bob.close_connection()
    fresh = WorldBookService(books.alice)
    assert fresh.get_world_book(books.book) == before_book
    assert {key: fresh.get_entry(books.entry).get(key) for key in entry_fields} == before_entry
    assert not result


def test_world_book_names_are_unique_per_owner(books):
    other = books.second.create_world_book("Alice private world")
    assert books.second.get_world_book(other)["description"] is None
    assert books.first.get_world_book(books.book)["description"] == "Private description"


@pytest.mark.parametrize("operation", ["search", "statistics", "book-statistics", "bulk-update", "toggle", "clone"])
def test_owned_world_book_secondary_operations_work_on_both_backends(books, operation):
    service = books.second
    own = service.create_world_book("Bob's owned book")
    entry = service.add_entry(own, ["owned"], "Bob's owned content")
    if operation == "search":
        assert [row["id"] for row in service.search_entries(query="owned")] == [entry]
    elif operation == "statistics":
        stats = service.get_statistics()
        assert stats["total_world_books"] == 1
        assert stats["total_entries"] == 1
    elif operation == "book-statistics":
        assert service.get_statistics(own)["total_entries"] == 1
    elif operation == "bulk-update":
        assert service.bulk_update_entries(own, [entry], enabled=False, priority=7) == 1
        assert service.get_entry(entry)["enabled"] is False
        assert service.get_entry(entry)["priority"] == 7
    elif operation == "toggle":
        assert service.toggle_entry_enabled(entry) is True
        assert service.get_entry(entry)["enabled"] is False
    else:
        clone = service.clone_world_book(own, "Bob's copy")
        assert [row["content"] for row in service.get_entries(clone)] == ["Bob's owned content"]


def test_foreign_books_do_not_contribute_to_counts_or_statistics(books):
    assert books.second.get_entry_counts_for_world_books() == {}
    assert books.second.get_statistics()["total_entries"] == 0
    assert books.second.get_statistics()["total_world_books"] == 0


def test_foreign_book_cannot_accept_new_entries(books):
    with pytest.raises((InputError, CharactersRAGDBError, sqlite3.IntegrityError)):
        books.second.add_entry(books.book, ["injected"], "Unauthorized content")
    # SQLite rejects the absent parent via its foreign key. PostgreSQL rejects
    # the existing foreign parent before executing the insert.
    books.bob.close_connection()
    assert [entry["content"] for entry in WorldBookService(books.alice).get_entries(books.book)] == ["Alice private lore"]


def test_foreign_entry_bulk_update_cannot_change_the_parent(books):
    assert books.second.bulk_update_entries(books.book, [books.entry], priority=99) == 0
    books.bob.close_connection()
    assert WorldBookService(books.alice).get_entry(books.entry)["priority"] == 0


@pytest.mark.parametrize("literal", ["%", "_", "\\", "!"])
def test_world_book_search_treats_pattern_characters_literally(books, literal):
    book = books.second.create_world_book("Literal search")
    match = books.second.add_entry(book, ["match"], f"Literal {literal} text")
    books.second.add_entry(book, ["other"], "Unrelated content")
    assert [row["id"] for row in books.second.search_entries(query=literal)] == [match]


@pytest.mark.parametrize("operation", ["bulk", "toggle", "clone"])
def test_secondary_mutations_remain_inside_the_callers_rollback(books, operation):
    service = books.second
    own = service.create_world_book("Rollback owned book")
    entry = service.add_entry(own, ["rollback"], "Keep this content")
    with pytest.raises(RuntimeError, match="caller rollback"):
        with books.bob.transaction():
            if operation == "bulk":
                assert service.bulk_update_entries(own, [entry], enabled=False) == 1
            elif operation == "toggle":
                assert service.toggle_entry_enabled(entry)
            else:
                service.clone_world_book(own, "Rolled back clone")
            raise RuntimeError("caller rollback")
    fresh = WorldBookService(books.bob)
    assert fresh.get_entry(entry)["enabled"] is True
    assert fresh.get_world_book(name="Rolled back clone") is None


@pytest.mark.postgres
def test_legacy_shared_books_remain_unassigned_until_ownership_is_established(pg_database_config, tmp_path):
    """A later account must never acquire old ownerless rows by opening the app."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "legacy.db", client_id="2", backend=backend)
    try:
        original = WorldBookService(db)
        book = original.create_world_book("Legacy unassigned book")
        entry = original.add_entry(book, ["retained"], "Retained legacy content")
        with db.transaction() as conn:
            conn.execute("ALTER TABLE world_books DROP COLUMN client_id")
            conn.execute("ALTER TABLE world_books ADD CONSTRAINT world_books_name_key UNIQUE (name)")
        migrated = WorldBookService(db)
        assert migrated.list_world_books() == []
        assert migrated.get_entry(entry) is None
        with db.transaction() as conn:
            retained = conn.execute("SELECT content FROM world_book_entries WHERE id = ?", (entry,)).fetchone()
            assert retained["content"] == "Retained legacy content"
            assert conn.execute("SELECT client_id FROM world_books WHERE id = ?", (book,)).fetchone()["client_id"] is None
            # Simulate a maintenance assignment after the real owner is known.
            conn.execute("UPDATE world_books SET client_id = ? WHERE id = ?", ("2", book))
        assert WorldBookService(db).get_entry(entry)["content"] == "Retained legacy content"
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()

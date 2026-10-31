"""Real SQLite and PostgreSQL World Book lifecycle contracts."""

import pytest

from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDB,
    ConflictError,
)

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def world_book_db(request, tmp_path):
    """Use SQLite or the official PostgreSQL fixture through the same service."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "world-book-lifecycle.db", client_id="210", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _seed_entry(db, service):
    """Seed a book and entry through the existing database abstraction."""
    world_book_id = service.create_world_book("Seeded lifecycle book")
    with db.transaction() as conn:
        query = """
            INSERT INTO world_book_entries (world_book_id, keywords, content, priority)
            VALUES (?, ?, ?, ?)
        """
        if db.backend_type == BackendType.POSTGRESQL:
            query += " RETURNING id"
        cursor = conn.execute(query, (world_book_id, '["seed"]', "seed content", 2))
        if db.backend_type == BackendType.POSTGRESQL:
            entry_id = cursor.fetchone()["id"]
        else:
            entry_id = cursor.lastrowid
    return world_book_id, int(entry_id)


def test_world_book_lifecycle_is_portable_and_invalidates_cached_reads(world_book_db):
    db = world_book_db
    service = WorldBookService(db)
    character_id = db.add_character_card({"name": "Lifecycle character"})

    world_book_id = service.create_world_book("Lifecycle book", description="before")
    assert service.get_world_book(world_book_id)["description"] == "before"
    assert [book["id"] for book in service.list_world_books()] == [world_book_id]

    assert service.update_world_book(world_book_id, description="after", expected_version=1)
    assert service.get_world_book(world_book_id)["description"] == "after"
    with pytest.raises(ConflictError, match="Version mismatch"):
        service.update_world_book(world_book_id, description="conflict", expected_version=1)

    entry_id = service.add_entry(world_book_id, ["first"], "entry before", priority=2)
    assert service.get_entry(entry_id)["content"] == "entry before"
    assert [entry["id"] for entry in service.get_entries(world_book_id)] == [entry_id]

    assert service.update_entry(entry_id, keywords=["updated"], content="entry after", priority=8)
    assert service.get_entry(entry_id)["content"] == "entry after"
    assert service.get_entries(world_book_id)[0]["keywords"] == ["updated"]

    assert service.attach_to_character(world_book_id, character_id, enabled=True, priority=1)["success"]
    assert service.attach_to_character(world_book_id, character_id, enabled=False, priority=9)["success"]
    attached = service.get_character_world_books(character_id, enabled_only=False)
    assert [(row["id"], bool(row["attachment_enabled"]), row["attachment_priority"]) for row in attached] == [
        (world_book_id, False, 9)
    ]
    assert service.detach_from_character(world_book_id, character_id)["success"]
    assert service.get_character_world_books(character_id, enabled_only=False) == []

    assert service.delete_entry(entry_id)
    assert service.get_entry(entry_id) is None
    assert service.get_entries(world_book_id) == []

    assert service.delete_world_book(world_book_id)
    assert service.get_world_book(world_book_id) is None
    assert service.list_world_books() == []

    purge_id = service.create_world_book("Permanent lifecycle book")
    purge_entry_id = service.add_entry(purge_id, ["purge"], "remove")
    assert service.delete_world_book(purge_id, hard_delete=True)
    assert service.get_world_book(purge_id) is None
    assert service.get_entry(purge_entry_id) is None


def test_world_book_entry_writes_remain_inside_caller_rollback(world_book_db):
    db = world_book_db
    service = WorldBookService(db)
    world_book_id = service.create_world_book("Rollback lifecycle book")

    with pytest.raises(RuntimeError, match="rollback lifecycle entry"):
        with db.transaction():
            entry_id = service.add_entry(world_book_id, ["temporary"], "temporary content")
            assert service.get_entry(entry_id)["content"] == "temporary content"
            raise RuntimeError("rollback lifecycle entry")

    assert service.get_entries(world_book_id) == []


@pytest.mark.parametrize("operation", ["update-entry", "delete-entry", "detach", "delete-book"])
def test_world_book_lifecycle_mutations_remain_inside_caller_rollback(world_book_db, operation):
    db = world_book_db
    service = WorldBookService(db)
    world_book_id, entry_id = _seed_entry(db, service)
    character_id = db.add_character_card({"name": f"Rollback {operation}"})
    assert service.attach_to_character(world_book_id, character_id)["success"]

    with pytest.raises(RuntimeError, match="rollback lifecycle mutation"):
        with db.transaction():
            if operation == "update-entry":
                assert service.update_entry(entry_id, content="pending")
            elif operation == "delete-entry":
                assert service.delete_entry(entry_id)
            elif operation == "detach":
                assert service.detach_from_character(world_book_id, character_id)["success"]
            else:
                assert service.delete_world_book(world_book_id)
            raise RuntimeError("rollback lifecycle mutation")

    if operation == "update-entry":
        assert service.get_entry(entry_id)["content"] == "seed content"
    elif operation == "delete-entry":
        assert service.get_entry(entry_id) is not None
    elif operation == "detach":
        assert service.get_character_world_books(character_id)
    else:
        assert service.get_world_book(world_book_id) is not None

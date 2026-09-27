"""World-book schema initialization respects real backend transaction ownership."""

import pytest

from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def world_book_db(request, tmp_path):
    """Use the official temporary PostgreSQL DB or a temporary SQLite DB."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(str(tmp_path / "world-books.db"), client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


def test_world_book_initialization_is_repeatable_and_committed(world_book_db):
    WorldBookService(world_book_db)
    WorldBookService(world_book_db)
    # A later rollback must not remove successfully initialized tables.
    world_book_db.get_connection().rollback()
    assert world_book_db.execute_query("SELECT COUNT(*) AS cnt FROM world_books").fetchone()["cnt"] == 0
    assert world_book_db.execute_query("SELECT COUNT(*) AS cnt FROM world_book_entries").fetchone()["cnt"] == 0
    assert world_book_db.execute_query("SELECT COUNT(*) AS cnt FROM character_world_books").fetchone()["cnt"] == 0


def test_world_book_initialization_does_not_commit_callers_transaction(world_book_db):
    with pytest.raises(RuntimeError, match="Rollback caller"):
        with world_book_db.transaction():
            conversation_id = world_book_db.add_conversation({"title": "Uncommitted caller"})
            WorldBookService(world_book_db)
            raise RuntimeError("Rollback caller")
    assert world_book_db.get_conversation_by_id(conversation_id) is None


def test_world_book_initialization_allows_callers_commit(world_book_db):
    with world_book_db.transaction():
        conversation_id = world_book_db.add_conversation({"title": "Committed caller"})
        WorldBookService(world_book_db)
    assert world_book_db.get_conversation_by_id(conversation_id)["title"] == "Committed caller"

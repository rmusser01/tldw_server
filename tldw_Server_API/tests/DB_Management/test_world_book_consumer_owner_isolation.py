"""World Book consumers must enforce the same owner boundary as the service."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
    _load_world_books_for_participants,
    _materialize_world_books_by_id,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import WorldBooksRetriever

pytestmark = pytest.mark.integration

FOREIGN_CONTENT = "Alice private consumer lore"
OWN_CONTENT = "Bob authorized consumer lore"


def _seed_consumer_books(tmp_path, backend):
    """Keep SQLite IDs distinct while sharing the official PostgreSQL backend."""
    alice = CharactersRAGDB(tmp_path / "consumer-alice.db", client_id="1", backend=backend)
    bob = CharactersRAGDB(tmp_path / "consumer-bob.db", client_id="2", backend=backend)
    try:
        alice_service = WorldBookService(alice)
        # SQLite uses independent sequences. Reserve its first IDs so Bob's
        # positive control cannot accidentally occupy Alice's addressed IDs.
        padding = alice_service.create_world_book("Sequence padding")
        alice_service.add_entry(padding, ["padding"], "Unrelated padding")
        foreign_book = alice_service.create_world_book("Alice private book")
        foreign_entry = alice_service.add_entry(foreign_book, ["private"], FOREIGN_CONTENT)
        alice.close_connection()

        bob_service = WorldBookService(bob)
        own_book = bob_service.create_world_book("Bob authorized book")
        own_entry = bob_service.add_entry(own_book, ["authorized"], OWN_CONTENT)
        character = bob.add_character_card({"name": "Bob consumer character"})
        assert bob_service.attach_to_character(own_book, character)
        if backend is not None:
            # Model a persisted cross-owner association allowed before upgrade;
            # the current public service correctly refuses new associations.
            with bob.transaction() as conn:
                conn.execute(
                    "INSERT INTO character_world_books (character_id, world_book_id) VALUES (?, ?)",
                    (character, foreign_book),
                )
        bob.close_connection()
        yield SimpleNamespace(
            alice=alice, bob=bob, character=character,
            own_book=own_book, own_entry=own_entry,
            foreign_book=foreign_book, foreign_entry=foreign_entry,
        )
    finally:
        alice.close_all_connections()
        bob.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def consumer_books(request, tmp_path):
    """Use per-user SQLite files or the official shared PostgreSQL fixture."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    yield from _seed_consumer_books(tmp_path, backend)


@pytest.fixture
def legacy_consumer_books(pg_database_config, tmp_path):
    """Retain a legacy attachment while leaving the book's owner unassigned."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    fixture = _seed_consumer_books(tmp_path, backend)
    books = next(fixture)
    try:
        with books.alice.transaction() as conn:
            conn.execute("UPDATE world_books SET client_id = NULL WHERE id = ?", (books.foreign_book,))
        books.alice.close_connection()
        yield books
    finally:
        fixture.close()


@pytest.mark.parametrize("operation", ["retrieve", "metadata"])
async def test_rag_world_books_read_only_current_owner(consumer_books, operation):
    books = consumer_books
    retriever = WorldBooksRetriever(None, db_adapter=books.bob)
    if operation == "retrieve":
        own = await retriever.retrieve(OWN_CONTENT)
        assert [document.metadata["entry_id"] for document in own] == [books.own_entry]
        assert await retriever.retrieve(FOREIGN_CONTENT) == []
    else:
        own = await retriever.get_metadata(f"world_book_entry_{books.own_entry}")
        assert own["content"] == OWN_CONTENT
        assert await retriever.get_metadata(f"world_book_entry_{books.foreign_entry}") == {}


def _assert_materialization_is_private(books, operation):
    """Exercise real materialization SQL with both authorized and hidden lore."""
    with books.bob.transaction() as conn:
        if operation == "participants":
            materialized = _load_world_books_for_participants(
                conn, [books.character], owner_user_id="2",
            )[books.character]
            assert [book["id"] for book in materialized] == [books.own_book]
            assert materialized[0]["entries"][0]["content"] == OWN_CONTENT
        else:
            own = _materialize_world_books_by_id(
                conn, [books.own_book], owner_user_id="2",
                participant_character_ids=[books.character],
            )
            assert own[books.own_book]["entries"][0]["content"] == OWN_CONTENT
            with pytest.raises(InputError, match="not found"):
                _materialize_world_books_by_id(
                    conn, [books.foreign_book], owner_user_id="2",
                    participant_character_ids=[books.character],
                )


@pytest.mark.parametrize("operation", ["participants", "explicit"])
def test_character_materialization_rejects_foreign_books(consumer_books, operation):
    _assert_materialization_is_private(consumer_books, operation)


@pytest.mark.postgres
@pytest.mark.parametrize("operation", ["retrieve", "metadata", "participants", "explicit"])
async def test_unassigned_legacy_books_remain_hidden_from_consumers(legacy_consumer_books, operation):
    books = legacy_consumer_books
    # The service already denies these rows. Consumers must agree even while
    # pre-upgrade attachment records remain in the database.
    assert WorldBookService(books.bob).get_world_book(books.foreign_book) is None
    books.bob.close_connection()
    if operation == "retrieve":
        assert await WorldBooksRetriever(None, db_adapter=books.bob).retrieve(FOREIGN_CONTENT) == []
    elif operation == "metadata":
        assert await WorldBooksRetriever(None, db_adapter=books.bob).get_metadata(
            f"world_book_entry_{books.foreign_entry}",
        ) == {}
    else:
        _assert_materialization_is_private(books, operation)


@pytest.mark.parametrize("operation", ["count", "ids", "preview", "unique-name"])
def test_chatbook_world_book_queries_stay_within_current_owner(consumer_books, operation):
    books = consumer_books
    if operation == "count":
        assert books.bob.count_chatbook_scope_category("world_books") == 1
    elif operation == "ids":
        assert books.bob.list_chatbook_scope_ids("world_books") == [str(books.own_book)]
    else:
        # These two read methods need only the real account DB. Avoid unrelated
        # export directories, Jobs migrations and adapters in the constructor.
        service = object.__new__(ChatbookService)
        service.db = books.bob
        service.user_id = "2"
        service.user_id_int = 2
        if operation == "preview":
            assert service.preview_export(["world_books"])["world_books"] == 1
        else:
            WorldBookService(books.alice).create_world_book("Import name (1)")
            books.alice.close_connection()
            own = WorldBookService(books.bob)
            own.create_world_book("Owned name (1)")
            assert service._generate_unique_name("Owned name", "world_book") == "Owned name (2)"
            assert service._generate_unique_name("Import name", "world_book") == "Import name (1)"

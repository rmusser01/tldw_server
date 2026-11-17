"""World-book schema initialization respects real backend transaction ownership."""

import pytest
from psycopg import sql

from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.tests.DB_Management.test_postgres_nonowner_runtime import migrated_runtime as migrated_runtime

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


@pytest.fixture
def migrated_world_books(migrated_runtime):
    """Reuse the official owner/ordinary-role fixture with actual canonical books."""
    f = migrated_runtime
    owner_backend = DatabaseBackendFactory.create_backend(f.owner.config)
    owner_db = None
    try:
        owner_db = CharactersRAGDB(":memory:", client_id="101", backend=owner_backend)
        books = WorldBookService(owner_db)
        book = books.create_world_book("X01 retained world", description="Retained private lore")
        entry = books.add_entry(book, ["Rowan"], "Rowan Observatory opened in 1988.")
        with owner_backend.transaction() as conn:
            conn.execute(
                sql.SQL(
                    "GRANT SELECT,INSERT,UPDATE,DELETE ON world_books,world_book_entries,character_world_books TO {}"
                ).format(sql.Identifier(f.role))
            )
            conn.execute(
                sql.SQL(
                    "GRANT USAGE,SELECT ON SEQUENCE world_books_id_seq,world_book_entries_id_seq,character_world_books_id_seq TO {}"
                ).format(sql.Identifier(f.role))
            )
        yield f, owner_backend, book, entry
    finally:
        try:
            if owner_db is not None:
                owner_db.close_connection()
        finally:
            owner_backend.get_pool().close_all()


def test_nonowner_world_book_service_preserves_content_and_tenant_boundaries(migrated_world_books):
    """An ordinary serving role can reopen migrated books without owner DDL."""
    f, _owner_backend, book, entry = migrated_world_books
    db = CharactersRAGDB(":memory:", client_id="101", backend=f.backend)
    other_db = None
    try:
        try:
            service = WorldBookService(db)
        except (DatabaseError, CharactersRAGDBError) as error:
            raise AssertionError({"phase": "ordinary-world-book-constructor", "error": type(error).__name__}) from None
        assert service.get_world_book(book)["description"] == "Retained private lore"  # nosec B101 -- genuine migrated data.
        assert service.get_entry(entry)["content"] == "Rowan Observatory opened in 1988."  # nosec B101 -- exact retained lore.
        new = service.create_world_book("X01 new world")
        assert service.get_world_book(new)["name"] == "X01 new world"  # nosec B101 -- real ordinary-role write.
        other_db = CharactersRAGDB(":memory:", client_id="202", backend=f.backend)
        other = WorldBookService(other_db)
        assert other.get_world_book(book) is None and other.get_entry(entry) is None  # nosec B101 -- no foreign lore visibility.
    finally:
        db.close_all_connections()
        if other_db is not None:
            other_db.close_all_connections()


@pytest.mark.parametrize(
    "drift",
    [
        "DROP TABLE public.world_book_entries",
        "ALTER TABLE public.world_book_entries DROP COLUMN metadata",
        "DROP INDEX public.idx_world_books_owner_name",
        "ALTER TABLE public.world_book_entries DROP CONSTRAINT world_book_entries_pkey",
        "ALTER TABLE public.character_world_books DROP CONSTRAINT character_world_books_character_id_world_book_id_key",
        "ALTER TABLE public.world_book_entries DROP CONSTRAINT world_book_entries_world_book_id_fkey",
        "ALTER TABLE public.character_world_books DROP CONSTRAINT character_world_books_character_id_fkey; "
        "ALTER TABLE public.character_world_books ADD FOREIGN KEY(character_id) REFERENCES character_cards(id) ON DELETE RESTRICT",
        "ALTER TABLE public.world_books ADD UNIQUE(name)",
    ],
    ids=[
        "missing-table",
        "missing-column",
        "owner-unique",
        "primary-key",
        "attachment-unique",
        "missing-link",
        "wrong-delete",
        "global-name",
    ],
)
def test_nonowner_world_book_refuses_drift_without_repair(migrated_world_books, drift):
    """Ordinary construction cannot repair a partial or incompatible owner schema."""
    f, owner_backend, _book, _entry = migrated_world_books
    for statement in drift.split(";"):
        owner_backend.execute(statement)
    queries = (
        "SELECT table_name,column_name FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name,ordinal_position",
        "SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename,indexname",
        "SELECT conrelid,conname,pg_get_constraintdef(oid) AS definition FROM pg_constraint WHERE connamespace='public'::regnamespace ORDER BY conrelid,conname",
    )
    before = [owner_backend.execute(query).rows for query in queries]
    db = CharactersRAGDB(":memory:", client_id="101", backend=f.backend)
    try:
        with pytest.raises((DatabaseError, CharactersRAGDBError)):
            WorldBookService(db)
        assert [owner_backend.execute(query).rows for query in queries] == before  # nosec B101 -- refusal cannot publish owner repairs.
    finally:
        db.close_all_connections()


def test_nonowner_world_book_initialization_preserves_callers_rollback(migrated_world_books):
    """Read-only readiness must not commit the caller's pending business writes."""
    f, _owner_backend, _book, _entry = migrated_world_books
    db = CharactersRAGDB(":memory:", client_id="101", backend=f.backend)
    try:
        service = WorldBookService(db)
        with pytest.raises(RuntimeError, match="Rollback caller"):
            with db.transaction():
                book = service.create_world_book("X01 rolled back world")
                WorldBookService(db)
                raise RuntimeError("Rollback caller")
        assert service.get_world_book(book) is None  # nosec B101 -- caller transaction remains the only commit authority.
    finally:
        db.close_all_connections()

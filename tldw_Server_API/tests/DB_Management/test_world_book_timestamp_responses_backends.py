"""World Book responses identify timestamp instants at the service/API boundary."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.api.v1.endpoints import characters_endpoint as characters
from tldw_Server_API.app.api.v1.schemas.world_book_schemas import (
    CharacterWorldBookAttachment,
    WorldBookCreate,
    WorldBookUpdate,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def world_book_db(request, tmp_path):
    """Use the standard temporary backend fixtures without a custom database setup."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "world-book-timestamps.db", client_id="260", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.fixture
def pg_world_book_db(pg_database_config, tmp_path):
    """Provide the official PostgreSQL fixture for session-timezone controls."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "world-book-timestamps.db", client_id="260", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def _assert_explicit_instant(value: datetime) -> None:
    assert value.tzinfo is not None
    assert value.utcoffset() is not None


def _assert_same_instant(*values: datetime) -> None:
    instants = [value.astimezone(timezone.utc) for value in values]
    assert len(set(instants)) == 1


def test_world_book_endpoint_readbacks_have_explicit_timestamps(world_book_db):
    db = world_book_db
    character_id = db.add_character_card({"name": "Timestamp reader"})

    created = asyncio.run(
        characters.create_world_book(WorldBookCreate(name="Timestamp contract"), db)
    )
    detail = asyncio.run(characters.get_world_book(created.id, db))
    listed = asyncio.run(characters.list_world_books(include_disabled=True, db=db))
    updated = asyncio.run(
        characters.update_world_book(
            created.id,
            WorldBookUpdate(description="updated"),
            expected_version=None,
            db=db,
        )
    )
    attached = asyncio.run(
        characters.attach_world_book_to_character(
            character_id,
            CharacterWorldBookAttachment(world_book_id=created.id),
            db,
        )
    )
    character_books = asyncio.run(characters.get_character_world_books(character_id, db=db))

    catalogue_book = next(book for book in listed.world_books if book.id == created.id)
    for response in [created, detail, catalogue_book, updated, attached, character_books[0]]:
        _assert_explicit_instant(response.created_at)
        _assert_explicit_instant(response.last_modified)
    _assert_same_instant(
        created.created_at,
        detail.created_at,
        catalogue_book.created_at,
        updated.created_at,
        attached.created_at,
        character_books[0].created_at,
    )


def test_world_book_update_keeps_conflicts_and_caller_rollback_consistent(world_book_db):
    db = world_book_db
    service = WorldBookService(db)
    world_book_id = service.create_world_book("Update transaction contract", description="original")
    assert service.get_world_book(world_book_id)["description"] == "original"

    with pytest.raises(RuntimeError, match="rollback update"):
        with db.transaction():
            assert service.update_world_book(
                world_book_id,
                description="pending",
                expected_version=1,
            )
            assert service.get_world_book(world_book_id)["description"] == "pending"
            raise RuntimeError("rollback update")

    assert service.get_world_book(world_book_id)["description"] == "original"
    with pytest.raises(ConflictError, match="Version mismatch"):
        service.update_world_book(
            world_book_id,
            description="conflict",
            expected_version=2,
        )


def test_world_book_attach_is_idempotent_validation_safe_and_rollbackable(world_book_db):
    db = world_book_db
    service = WorldBookService(db)
    character_id = db.add_character_card({"name": "Attachment transaction reader"})
    world_book_id = service.create_world_book("Attachment transaction contract")

    assert service.attach_to_character(world_book_id, character_id, enabled=True, priority=1)["success"]
    assert service.attach_to_character(world_book_id, character_id, enabled=False, priority=8)["success"]
    attached = service.get_character_world_books(character_id, enabled_only=False)
    assert len(attached) == 1
    assert bool(attached[0]["attachment_enabled"]) is False
    assert attached[0]["attachment_priority"] == 8
    assert service.attach_to_character(world_book_id, character_id + 100_000)["success"] is False

    rollback_book_id = service.create_world_book("Attachment caller rollback")
    with pytest.raises(RuntimeError, match="rollback attach"):
        with db.transaction():
            assert service.attach_to_character(rollback_book_id, character_id)["success"]
            assert {
                row["id"] for row in service.get_character_world_books(character_id, enabled_only=False)
            } == {world_book_id, rollback_book_id}
            raise RuntimeError("rollback attach")

    assert {
        row["id"] for row in service.get_character_world_books(character_id, enabled_only=False)
    } == {world_book_id}


@pytest.mark.parametrize(
    ("session_timezone", "wall_time", "expected_offset"),
    [
        ("America/Los_Angeles", "2026-01-15 04:30:00", timedelta(hours=-8)),
        ("America/Los_Angeles", "2026-07-15 04:30:00", timedelta(hours=-7)),
        ("Australia/Eucla", "2026-07-15 04:30:00", timedelta(hours=8, minutes=45)),
    ],
)
def test_postgres_world_book_reads_project_naive_rows_in_the_stable_session_timezone(
    pg_world_book_db,
    session_timezone,
    wall_time,
    expected_offset,
):
    db = pg_world_book_db
    service = WorldBookService(db)
    world_book_id = service.create_world_book(f"Timezone {session_timezone} {wall_time}")
    raw = db._get_thread_connection()
    raw.execute("SELECT set_config('TimeZone', %s, false)", (session_timezone,))
    raw.execute(
        "UPDATE world_books SET created_at = %s, last_modified = %s WHERE id = %s",
        (wall_time, wall_time, world_book_id),
    )
    raw.commit()

    response = asyncio.run(characters.get_world_book(world_book_id, db))

    assert response.created_at.utcoffset() == expected_offset
    assert response.last_modified.utcoffset() == expected_offset
    expected = datetime.fromisoformat(wall_time).replace(tzinfo=response.created_at.tzinfo)
    assert response.created_at.astimezone(timezone.utc) == expected.astimezone(timezone.utc)
    assert response.last_modified.astimezone(timezone.utc) == expected.astimezone(timezone.utc)

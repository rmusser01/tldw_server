"""Storage contract for client-id chat creation (D7 P3).

A create request that names its own conversation id stores a fingerprint of
that request on the conversation row, in the same INSERT. The conversation
primary key is what keeps a client id to one chat; these tests pin both halves
on SQLite and PostgreSQL.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.integration

COLUMN = "create_request_fingerprint"
FINGERPRINT = "a" * 64
CHAT_ID = "3f1c9a52-6a2b-4c1e-9d7f-0b8e2a4c6d10"


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def db_factory(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Callable[[], CharactersRAGDB]]:
    """Open real databases through their initializer and close every handle."""
    config = request.getfixturevalue("pg_database_config") if request.param == "postgres" else None
    opened: list[CharactersRAGDB] = []

    def create() -> CharactersRAGDB:
        backend = DatabaseBackendFactory.create_backend(config) if config else None
        database = CharactersRAGDB(tmp_path / "fingerprint.db", client_id="1", backend=backend)
        opened.append(database)
        return database

    try:
        yield create
    finally:
        for database in opened:
            database.close_all_connections()


def test_fingerprint_is_stored_with_the_conversation_insert(db_factory: Callable[[], CharactersRAGDB]) -> None:
    db = db_factory()
    created = db.add_conversation(
        {"id": CHAT_ID, "title": "Client named", "client_id": "1"},
        create_request_fingerprint=FINGERPRINT,
    )
    assert created == CHAT_ID
    assert db.get_conversation_by_id(CHAT_ID)[COLUMN] == FINGERPRINT


def test_ordinary_insert_stores_no_fingerprint(db_factory: Callable[[], CharactersRAGDB]) -> None:
    db = db_factory()
    cid = db.add_conversation({"title": "Server named", "client_id": "1"})
    assert db.get_conversation_by_id(cid)[COLUMN] is None


def test_fingerprint_cannot_be_smuggled_through_conversation_data(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """Only the explicit keyword sets it, so copied or imported rows never carry one."""
    db = db_factory()
    cid = db.add_conversation({"title": "Imported", "client_id": "1", COLUMN: FINGERPRINT})
    assert db.get_conversation_by_id(cid)[COLUMN] is None


@pytest.mark.parametrize("bad", ["", "A" * 64, "a" * 63, "a" * 65, "g" * 64, 64])
def test_malformed_fingerprint_is_rejected_before_insert(
    db_factory: Callable[[], CharactersRAGDB], bad: object
) -> None:
    db = db_factory()
    with pytest.raises(InputError):
        db.add_conversation({"id": CHAT_ID, "title": "Bad", "client_id": "1"}, create_request_fingerprint=bad)
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True) is None


@pytest.mark.parametrize(
    "bad",
    ["A" * 64, "a" * 63, "a" * 65, "z" * 64, "a" * 64 + "\x00" + "x" * 8],
    ids=["uppercase", "short", "long", "not-hex", "nul-suffix"],
)
def test_column_rejects_malformed_fingerprints_written_directly(
    db_factory: Callable[[], CharactersRAGDB], bad: str
) -> None:
    """The schema enforces the digest shape even for writes that bypass the store."""
    db = db_factory()
    if bad.count("\x00") and db.backend_type.value == "postgresql":
        pytest.skip("PostgreSQL text cannot hold NUL at all")
    cid = db.add_conversation({"title": "Direct", "client_id": "1"})
    with pytest.raises((sqlite3.IntegrityError, DatabaseError, CharactersRAGDBError)):
        with db.transaction() as conn:
            conn.execute("UPDATE conversations SET create_request_fingerprint = ? WHERE id = ?", (bad, cid))
    assert db.get_conversation_by_id(cid)[COLUMN] is None


def test_duplicate_id_is_refused_by_the_primary_key(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """The database, not an application read, keeps one conversation per id."""
    db = db_factory()
    db.add_conversation({"id": CHAT_ID, "title": "First", "client_id": "1"}, create_request_fingerprint=FINGERPRINT)
    with pytest.raises(ConflictError) as raised:
        db.add_conversation(
            {"id": CHAT_ID, "title": "Second", "client_id": "1"}, create_request_fingerprint="b" * 64
        )
    assert raised.value.entity == "conversations"
    assert raised.value.entity_id == CHAT_ID
    row = db.get_conversation_by_id(CHAT_ID)
    assert (row["title"], row[COLUMN]) == ("First", FINGERPRINT)


def test_upgrade_adds_an_empty_fingerprint_to_existing_chats(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chats created before the column existed never look like client-id creates."""
    with monkeypatch.context() as previous:
        previous.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 74)
        previous.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 78)
        old = db_factory()
        cid = old.add_conversation({"title": "Before upgrade", "client_id": "1"})
        assert COLUMN not in old.get_conversation_by_id(cid)
        old.close_all_connections()
    upgraded = db_factory()
    row = upgraded.get_conversation_by_id(cid)
    assert row["title"] == "Before upgrade"
    assert row[COLUMN] is None
    version = upgraded.execute_query(
        "SELECT version FROM db_schema_version WHERE schema_name = ?", (upgraded._SCHEMA_NAME,), read_only=True
    ).fetchone()["version"]
    expected = (
        CharactersRAGDB._POSTGRES_SCHEMA_VERSION
        if upgraded.backend_type.value == "postgresql"
        else CharactersRAGDB._CURRENT_SCHEMA_VERSION
    )
    assert version == expected

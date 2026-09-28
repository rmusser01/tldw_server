"""
Regression tests for Character Chat services when operating against a PostgreSQL backend.
"""

import contextlib
from typing import Any, Dict, List, Optional

import pytest

from tldw_Server_API.app.core.Character_Chat.chat_dictionary import (
    ChatDictionaryEntry,
    ChatDictionaryService,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError, ConflictError


class _RecordingCursor:
    """Simple cursor stub that mimics the attributes used by the services."""

    def __init__(self, sql: str, row: Optional[Dict[str, Any]] = None):
        self.sql = sql
        self.rowcount = 0
        self.lastrowid = None
        self._row = row
        self._fetched = False

    def fetchall(self):

        if self._row is None:
            return []
        if not self._fetched:
            self._fetched = True
            return [self._row]
        return []

    def fetchone(self):

        if self._fetched:
            return None
        self._fetched = True
        return self._row


class RecordingConnection:
    """Connection stub that records executed SQL."""

    def __init__(self):

        self.executed_sql = []
        self.committed = False
        self._next_id = 1

    def execute(self, sql, params=None):

        self.executed_sql.append(sql)
        row = None
        if "returning id" in sql.lower():
            row = {"id": self._next_id}
            self._next_id += 1
        return _RecordingCursor(sql, row=row)

    def commit(self):

        self.committed = True

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc, tb):

        return False


class UniqueViolationConnection(RecordingConnection):
    """Connection stub that simulates a unique constraint violation."""

    def execute(self, sql, params=None):

        self.executed_sql.append(sql)
        if "INSERT INTO CHAT_DICTIONARIES" in sql.upper():
            raise CharactersRAGDBError("duplicate key value violates unique constraint")
        return _RecordingCursor(sql)


class StubDB:
    """Minimal DB stub that hands out predetermined connection objects."""

    def __init__(self, connections):

        self.backend_type = BackendType.POSTGRESQL
        self.client_id = "owner-1"  # world books are owner-scoped (a9f0c1a1b5)
        self._connections = list(connections)

    def get_connection(self):

        if not self._connections:
            raise AssertionError("No stub connections remaining for test")
        return self._connections.pop(0)

    @contextlib.contextmanager
    def transaction(self):
        # WorldBookService writes through db.transaction() (9cc0b15671).
        conn = self.get_connection()
        yield conn
        conn.commit()


def _gather_sql(connection: RecordingConnection) -> str:
    return " ".join(sql.lower() for sql in connection.executed_sql)


@pytest.mark.unit
def test_chat_dictionary_init_uses_postgres_friendly_schema():
    init_conn = RecordingConnection()
    db = StubDB([init_conn])

    ChatDictionaryService(db)

    executed = _gather_sql(init_conn)
    assert "autoincrement" not in executed
    assert "serial" in executed


@pytest.mark.unit
def test_world_book_init_uses_postgres_friendly_schema():
    init_conn = RecordingConnection()
    db = StubDB([init_conn])

    WorldBookService(db)

    executed = _gather_sql(init_conn)
    assert "autoincrement" not in executed
    assert "serial" in executed


@pytest.mark.unit
def test_chat_dictionary_unique_violation_raises_conflict():
    init_conn = RecordingConnection()
    failing_conn = UniqueViolationConnection()
    db = StubDB([init_conn, failing_conn])

    service = ChatDictionaryService(db)

    with pytest.raises(ConflictError):
        service.create_dictionary(name="Duplicate")

    assert not failing_conn.committed


@pytest.mark.unit
def test_integer_probability_treated_as_percentage():
    entry = ChatDictionaryEntry("trigger", "value", probability=1)
    assert entry.probability == pytest.approx(0.01)

    entry_high = ChatDictionaryEntry("trigger", "value", probability=75)
    assert entry_high.probability == pytest.approx(0.75)


@pytest.mark.unit
def test_create_dictionary_uses_returning_and_row_id():
    init_conn = RecordingConnection()
    insert_conn = RecordingConnection()
    db = StubDB([init_conn, insert_conn])

    service = ChatDictionaryService(db)
    new_id = service.create_dictionary(name="Lore Dict")

    assert new_id == 1
    executed = _gather_sql(insert_conn)
    assert "returning id" in executed
    assert insert_conn.committed


@pytest.mark.unit
def test_create_world_book_uses_returning_and_row_id():
    init_conn = RecordingConnection()
    insert_conn = RecordingConnection()
    db = StubDB([init_conn, insert_conn])

    service = WorldBookService(db)
    new_id = service.create_world_book(name="Lore Book")

    assert new_id == 1
    executed = _gather_sql(insert_conn)
    assert "returning id" in executed
    assert insert_conn.committed


@pytest.mark.unit
def test_dictionary_entry_insert_uses_returning_clause():
    init_conn = RecordingConnection()
    dict_insert_conn = RecordingConnection()
    entry_conn = RecordingConnection()
    db = StubDB([init_conn, dict_insert_conn, entry_conn])

    service = ChatDictionaryService(db)
    dictionary_id = service.create_dictionary(name="Lore Dict")
    entry_id = service.add_entry(dictionary_id, pattern="foo", content="bar")

    assert dictionary_id == 1
    assert entry_id == 1
    executed = _gather_sql(entry_conn)
    assert "returning id" in executed


@pytest.mark.unit
def test_world_book_entry_insert_uses_returning_clause():
    init_conn = RecordingConnection()
    book_insert_conn = RecordingConnection()
    entry_conn = RecordingConnection()
    db = StubDB([init_conn, book_insert_conn, entry_conn])

    service = WorldBookService(db)
    world_book_id = service.create_world_book(name="Lore Book")
    # PostgreSQL entry writes first confirm the owner's book exists
    # (a9f0c1a1b5); that read is not what this test pins.
    service.get_world_book = lambda book_id: {"id": book_id}
    entry_id = service.add_world_book_entry(
        world_book_id,
        keywords=["hero"],
        content="Hero lore",
        priority=10,
    )

    assert world_book_id == 1
    assert entry_id == 1
    executed = _gather_sql(entry_conn)
    assert "returning id" in executed

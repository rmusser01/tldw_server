"""
Recording-stub tests for Character Chat service PostgreSQL branches.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Character_Chat.chat_dictionary import (
    ChatDictionaryEntry,
    ChatDictionaryService,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError, ConflictError, InputError


class _RecordingCursor:
    """Simple cursor stub that mimics the attributes used by the services."""

    def __init__(self, sql: str, row: dict[str, Any] | None = None):
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

    def __init__(self, *, selected_row=None, fail_on=None):
        self.executed_sql = []
        self.executed_params = []
        self.committed = False
        self.rolled_back = False
        self._next_id = 1
        self._selected_row = selected_row
        self._fail_on = fail_on
        self.transaction_depth = 0
        self._connection = self
        self.info = SimpleNamespace(transaction_status=SimpleNamespace(name="IDLE"))
        self._backend = SimpleNamespace(_tx_depth=lambda conn: conn.transaction_depth)

    def execute(self, sql, params=None):

        self.executed_sql.append(sql)
        self.executed_params.append(params)
        if self._fail_on and self._fail_on in sql.lower():
            raise CharactersRAGDBError("injected statement failure")
        row = self._selected_row if sql.strip().lower().startswith("select") else None
        if "returning id" in sql.lower():
            row = {"id": self._next_id}
            self._next_id += 1
        return _RecordingCursor(sql, row=row)

    def commit(self):

        self.committed = True

    def rollback(self):
        self.rolled_back = True

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
        self.client_id = "world-book-owner"
        self._connections = list(connections)
        self._state = SimpleNamespace(tx_depth=0)
        self.query_calls = []

    def get_connection(self):

        if not self._connections:
            raise AssertionError("No stub connections remaining for test")
        return self._connections.pop(0)

    def _connection_state(self):
        return self._state

    def execute_query(self, sql, params=None, *, read_only=False):
        self.query_calls.append((sql, params, read_only))
        return self.get_connection().execute(sql, params)

    @contextmanager
    def transaction(self):
        """Record the commit/rollback boundary used by service-owned writes."""
        conn = self.get_connection()
        self._state.tx_depth += 1
        conn.transaction_depth += 1
        conn.info.transaction_status.name = "INTRANS"
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            self._state.tx_depth -= 1
            conn.transaction_depth -= 1
            conn.info.transaction_status.name = "IDLE"


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

    assert init_conn.committed
    assert not init_conn.rolled_back


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

    assert "client_id" in executed
    assert insert_conn.executed_params == [("Lore Book", None, 3, 500, False, True, db.client_id)]
    assert db.client_id not in executed
    assert not insert_conn.rolled_back


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
    lookup_conn = RecordingConnection(selected_row={"id": 1, "name": "Lore Book"})
    entry_conn = RecordingConnection()
    # The read checks transaction state, then executes on that same connection.
    db = StubDB([init_conn, book_insert_conn, lookup_conn, lookup_conn, entry_conn])

    service = WorldBookService(db)
    world_book_id = service.create_world_book(name="Lore Book")
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
    assert "client_id = ?" in _gather_sql(lookup_conn)
    assert lookup_conn.executed_params == [(world_book_id, False, db.client_id)]
    assert db.query_calls == [(lookup_conn.executed_sql[0], (world_book_id, False, db.client_id), True)]
    assert entry_conn.executed_params == [(world_book_id, '["hero"]', "Hero lore", 10, True, False, False, True, "{}")]
    assert entry_conn.committed
    assert not entry_conn.rolled_back


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["initialize", "create"])
def test_world_book_transaction_rolls_back_failed_statement(operation):
    failing_conn = RecordingConnection(
        fail_on="create table" if operation == "initialize" else "insert into world_books",
    )
    connections = [failing_conn] if operation == "initialize" else [RecordingConnection(), failing_conn]
    db = StubDB(connections)
    with pytest.raises(CharactersRAGDBError, match="injected statement failure"):
        service = WorldBookService(db)
        if operation == "create":
            service.create_world_book(name="Uncommitted book")
    assert failing_conn.rolled_back
    assert not failing_conn.committed
    assert db._connection_state().tx_depth == 0


@pytest.mark.unit
def test_world_book_entry_rejects_missing_owner_before_insert():
    lookup_conn = RecordingConnection()
    entry_conn = RecordingConnection()
    db = StubDB([RecordingConnection(), lookup_conn, lookup_conn, entry_conn])
    service = WorldBookService(db)

    with pytest.raises(InputError, match="World book not found"):
        service.add_world_book_entry(99, keywords=["private"], content="Private lore")

    assert "client_id = ?" in _gather_sql(lookup_conn)
    assert lookup_conn.executed_params == [(99, False, db.client_id)]
    assert db.query_calls == [(lookup_conn.executed_sql[0], (99, False, db.client_id), True)]
    assert entry_conn.executed_sql == []
    assert not entry_conn.committed

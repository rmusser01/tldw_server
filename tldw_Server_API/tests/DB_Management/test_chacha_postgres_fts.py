from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.character_store import CharacterStore
from tldw_Server_API.app.core.DB_Management.chacha.keyword_store import KeywordStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class _CursorStub:
    """Minimal stub matching the subset of cursor API used by CharactersRAGDB."""

    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self._rows = list(rows)

    def fetchall(self) -> List[Dict[str, Any]]:

        return list(self._rows)


def _make_postgres_db() -> CharactersRAGDB:

    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db.client_id = "pg-test"
    db._local = SimpleNamespace(backend_ref=None)
    db._uses_shared_content_backend = False
    db._backend = MagicMock()
    db._backend.backend_type = BackendType.POSTGRESQL
    db._CHARACTER_CARD_JSON_FIELDS = []
    db.character_store = CharacterStore(db)
    db.keyword_store = KeywordStore(db)

    class _TxConn:
        _connection = None

    @contextmanager
    def fake_transaction() -> Iterable[_TxConn]:
        yield _TxConn()

    db.transaction = fake_transaction  # type: ignore[assignment]
    return db


@pytest.fixture()
def postgres_db() -> CharactersRAGDB:
    """Return a Postgres-backed CharactersRAGDB stub."""
    return _make_postgres_db()


def test_rebuild_full_text_indexes_postgres_calls_backend():

    db = _make_postgres_db()
    db._FTS_CONFIG = [
        ("character_cards_fts", "character_cards", ["name"]),
        ("messages_fts", "messages", ["content"]),
    ]

    db.rebuild_full_text_indexes()

    expected_calls = [
        call(
            table_name="character_cards_fts",
            source_table="character_cards",
            columns=["name"],
            connection=None,
        ),
        call(
            table_name="messages_fts",
            source_table="messages",
            columns=["content"],
            connection=None,
        ),
    ]

    assert db.backend.create_fts_table.call_args_list == expected_calls


def test_rebuild_full_text_indexes_sqlite_executes_rebuild():

    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = SimpleNamespace(backend_ref=None)
    db._uses_shared_content_backend = False
    db._backend = MagicMock()
    db._backend.backend_type = BackendType.SQLITE
    db._FTS_CONFIG = [
        ("keywords_fts", "keywords", []),
        ("notes_fts", "notes", []),
    ]

    executed: List[str] = []

    class _Conn:
        def execute(self, sql: str, *_args: Any) -> None:
            executed.append(sql)

    @contextmanager
    def fake_transaction() -> Iterable[_Conn]:
        yield _Conn()

    db.transaction = fake_transaction  # type: ignore[assignment]

    db.rebuild_full_text_indexes()

    assert executed == [
        "INSERT INTO keywords_fts(keywords_fts) VALUES('rebuild')",
        "INSERT INTO notes_fts(notes_fts) VALUES('rebuild')",
    ]


def test_search_character_cards_postgres_uses_tsquery(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_postgres_db()
    db.client_id = "character-owner"
    rows = [{"id": 1, "rank": 0.42}]
    db.execute_query = MagicMock(return_value=_CursorStub(rows))
    db._deserialize_row_fields = lambda row, _fields: row  # type: ignore[assignment]

    result = db.search_character_cards("dragon rider", limit=5)

    assert result == rows
    assert db.execute_query.call_count == 1
    sql, params = db.execute_query.call_args[0]
    assert "ts_rank" in sql and "to_tsquery('english', ?)" in sql
    assert "cc.client_id = ?" in sql
    assert params == ("dragon & rider", "dragon & rider", "character-owner", 5)


def test_list_flashcards_postgres_translates_fts(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_postgres_db()
    db._get_current_utc_timestamp_iso = lambda: "2025-01-01T00:00:00Z"  # type: ignore[assignment]
    db.execute_query = MagicMock(return_value=_CursorStub([]))

    result = db.list_flashcards(q="alchemy", limit=10, offset=0)

    assert result == []
    sql, params = db.execute_query.call_args[0]
    assert "flashcards_fts_tsv @@ to_tsquery('english', ?)" in sql
    assert "alchemy" in params


@pytest.mark.unit
def test_get_flashcards_by_uuids_postgres_uses_boolean_deleted(
    postgres_db: CharactersRAGDB,
) -> None:
    """Verify Postgres uses boolean deleted clause for UUID lookups."""
    db = postgres_db
    db.execute_query = MagicMock(return_value=_CursorStub([]))

    result = db.get_flashcards_by_uuids(["uuid-1", "uuid-2"])

    assert result == []
    sql, params = db.execute_query.call_args[0]
    assert "f.deleted = FALSE" in sql
    assert "f.client_id = ?" in sql
    assert params == ("uuid-1", "uuid-2", "pg-test")


def test_manage_link_postgres_uses_on_conflict():

    db = _make_postgres_db()
    db.client_id = "pg-test"
    db._get_current_utc_timestamp_iso = lambda: "2025-01-01T00:00:00Z"  # type: ignore[assignment]

    class _Cursor:
        def __init__(self, rowcount: int = 0):
            self.rowcount = rowcount

    class _Conn:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def execute(self, sql: str, params: Any = None) -> _Cursor:
            self.calls.append(sql)
            return _Cursor(rowcount=1)

    conn = _Conn()

    @contextmanager
    def fake_transaction():
        yield conn

    db.transaction = fake_transaction  # type: ignore[assignment]

    assert db._manage_link("flashcard_keywords", "card_id", 1, "keyword_id", 2, "link") is True

    insert_calls = [sql.lower() for sql in conn.calls if sql.lower().startswith("insert into flashcard_keywords")]
    assert insert_calls, "expected insert into flashcard_keywords to be executed"
    assert all("on conflict" in sql for sql in insert_calls)
    assert all("insert or ignore" not in sql for sql in insert_calls)


def test_set_flashcard_tags_postgres_uses_on_conflict():

    db = _make_postgres_db()
    db.client_id = "pg-test"
    db._get_current_utc_timestamp_iso = lambda: "2025-01-01T00:00:00Z"  # type: ignore[assignment]

    class _Cursor:
        def __init__(self, *, rows: Optional[List[Any]] = None, rowcount: int = 0):
            self._rows = rows or []
            self.rowcount = rowcount

        def fetchone(self):

            return self._rows[0] if self._rows else None

        def fetchall(self):

            return list(self._rows)

    class _Conn:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def execute(self, sql: str, params: Any = None) -> _Cursor:
            self.calls.append(sql)
            sql_upper = sql.strip().upper()
            if sql_upper.startswith("SELECT ID FROM FLASHCARDS"):
                return _Cursor(rows=[{"id": 1}])
            if sql_upper.startswith("SELECT KEYWORD_ID FROM FLASHCARD_KEYWORDS"):
                return _Cursor(rows=[])
            if sql_upper.startswith("SELECT ID, DELETED FROM CHACHA_KEYWORDS"):
                assert params == ("pg-test", "Alpha")
                return _Cursor(rows=[{"id": 7, "deleted": False}])
            if "INSERT INTO FLASHCARD_KEYWORDS" in sql_upper:
                return _Cursor(rowcount=1)
            if sql_upper.startswith("UPDATE FLASHCARDS SET"):
                return _Cursor(rowcount=1)
            return _Cursor()

    conn = _Conn()

    @contextmanager
    def fake_transaction():
        yield conn

    db.transaction = fake_transaction  # type: ignore[assignment]

    assert db.set_flashcard_tags("uuid-123", ["Alpha"]) is True

    insert_calls = [sql.lower() for sql in conn.calls if sql.lower().startswith("insert into flashcard_keywords")]
    assert insert_calls, "expected flashcard_keywords insert when adding tags"
    assert all("on conflict (card_id, keyword_id) do nothing" in sql for sql in insert_calls)
    assert all("insert or ignore" not in sql for sql in insert_calls)


def test_search_keywords_postgres_uses_tsquery(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_postgres_db()
    rows = [{"id": 1, "keyword": "fruit", "rank": 0.88}]
    db.execute_query = MagicMock(return_value=_CursorStub(rows))

    result = db.search_keywords("fruit", limit=5)

    assert result == rows
    assert db.execute_query.call_count == 1
    sql, params = db.execute_query.call_args[0]
    assert "keywords_fts_tsv" in sql and "to_tsquery('english', ?)" in sql
    assert "k.client_id = ?" in sql
    assert params == ("fruit", "pg-test", "fruit", 5)


@pytest.mark.parametrize(
    "mode", ["fts", "fallback", "keyword_fts", "keyword_only", "keyword_count_fts", "keyword_count_only"]
)
def test_postgres_notes_search_uses_selected_owner_predicates(monkeypatch, mode):
    from tldw_Server_API.app.core.DB_Management.chacha.note_store import NoteStore

    db = _make_postgres_db()
    store = NoteStore(db)
    statements = []

    def selected_owner(owner_client_id, alias=""):
        # Distinct bindings prove the returned predicate and parameters travel
        # together; production policy still selects the unchanged single owner.
        return f" AND {alias}.client_id = ?", (f"selected-{alias}",)

    def execute(query, params):
        statements.append((query, params))
        return SimpleNamespace(fetchall=lambda: [{"id": "owned-note"}], fetchone=lambda: {"cnt": 1})

    monkeypatch.setattr(db, "_selected_owner_filter", selected_owner)
    monkeypatch.setattr(db, "execute_query", execute)
    monkeypatch.setattr(db, "_map_table_for_backend", lambda name: name)
    if mode == "fallback":
        monkeypatch.setattr(
            "tldw_Server_API.app.core.DB_Management.chacha.note_store.FTSQueryTranslator.normalize_query",
            lambda *args: "",
        )
    if mode in ("fts", "fallback"):
        assert store.search_notes("needle", limit=3, offset=2) == [{"id": "owned-note"}]
    elif mode.startswith("keyword_count"):
        assert store.count_notes_matching_keywords("needle" if mode.endswith("fts") else None, ["tag"]) == 1
    else:
        assert store.search_notes_with_keywords(
            "needle" if mode.endswith("fts") else None, ["tag"], limit=3, offset=2
        ) == [{"id": "owned-note"}]
    query, params = statements[-1]
    assert "n.client_id = ?" in query
    assert "selected-n" in params
    assert db.client_id not in params
    if mode.startswith("keyword"):
        assert "k.client_id = ?" in query
        assert "selected-k" in params


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (
            "UPDATE notes SET title = ? WHERE id = ? AND deleted = 0",
            "UPDATE notes SET title = %s WHERE id = %s AND deleted = FALSE",
        ),
        (
            "UPDATE notes SET deleted = 1 WHERE id = ? AND deleted = 0",
            "UPDATE notes SET deleted = TRUE WHERE id = %s AND deleted = FALSE",
        ),
        (
            "UPDATE notes SET deleted = 0 WHERE id = ? AND deleted = 1",
            "UPDATE notes SET deleted = FALSE WHERE id = %s AND deleted = TRUE",
        ),
        (
            "SELECT k.* FROM keywords k WHERE k.id = ? AND k.deleted = 0",
            "SELECT k.* FROM keywords k WHERE k.id = %s AND k.deleted = FALSE",
        ),
    ],
)
@pytest.mark.parametrize("path", ["backend", "chacha-transaction"])
def test_notes_boolean_literals_reach_postgres_driver_as_booleans(query, expected, path):
    """Both execution routes must normalize repository SQL before psycopg sees it."""
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
    from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import PostgreSQLBackend
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendConnectionWrapper

    backend = PostgreSQLBackend(DatabaseConfig(backend_type=BackendType.POSTGRESQL))
    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.description = None
    cursor.statusmessage = "UPDATE 1" if query.startswith("UPDATE") else "SELECT 0"
    cursor.rowcount = 1
    params = ("title", "note") if "title = ?" in query else ("note",)
    if path == "chacha-transaction":
        db = _make_postgres_db()
        db._backend = backend
        BackendConnectionWrapper(db, connection, backend).execute(query, params)
    else:
        backend.execute(query, params, connection=connection)

    cursor.execute.assert_called_once_with(expected, params)


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["search", "metadata"])
@pytest.mark.parametrize("path", ["backend", "chacha-transaction", "chacha-adapter"])
def test_chat_helpers_reach_postgres_driver_with_boolean_visibility_filters(
    monkeypatch: pytest.MonkeyPatch, operation: str, path: str,
) -> None:
    """Keep helper SQL portable without sending integer boolean comparisons to psycopg."""
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
    from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import PostgreSQLBackend
    from tldw_Server_API.app.core.DB_Management.chacha.chat_history_queries import (
        get_chat_history_metadata,
        search_chat_history,
    )
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendConnectionWrapper

    backend = PostgreSQLBackend(DatabaseConfig(backend_type=BackendType.POSTGRESQL))
    db = _make_postgres_db()
    db._backend = backend
    connection = MagicMock()
    cursor = connection.cursor.return_value
    cursor.description = [("id",)]
    cursor.statusmessage = "SELECT 1"
    cursor.rowcount = 1
    cursor.fetchall.return_value = [{"id": "message-1"}]
    wrapped_connection = BackendConnectionWrapper(db, connection, backend)
    monkeypatch.setattr(db, "get_connection", lambda: wrapped_connection)

    def execute(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if path == "backend":
            return backend.execute(query, params, connection=connection).rows
        if path == "chacha-transaction":
            return wrapped_connection.execute(query, params).fetchall()
        return db.execute_query(query, params).fetchall()

    if operation == "search":
        result = search_chat_history(execute, "needle'", db_adapter=db, limit=3)
        assert result == [{"id": "message-1"}]
        expected_params = ("pg-test", "%needle'%", "knowledge_qa", 3)
    else:
        result = get_chat_history_metadata(execute, "message-1", db_adapter=db)
        assert result == {"id": "message-1"}
        expected_params = ("message-1", "pg-test")

    cursor.execute.assert_called_once()
    sql, params = cursor.execute.call_args.args
    assert "m.deleted = FALSE" in sql
    assert "conv.deleted = FALSE" in sql
    assert "conv.client_id = %s" in sql
    assert "deleted = 0" not in sql
    assert params == expected_params


@pytest.mark.parametrize("operation", ["single", "foreign-note", "batch", "reverse", "foreign-keyword"])
def test_postgres_note_keyword_queries_filter_both_endpoints(postgres_db, operation):
    """Execute the PostgreSQL-selected predicates against malformed link rows.

    SQLite runs this portable SQL without requiring a live PostgreSQL service;
    the database facade retains its PostgreSQL owner selection and table mapping.
    """
    import sqlite3
    from contextlib import closing

    from tldw_Server_API.app.core.DB_Management.chacha.note_store import NoteStore

    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        connection.executescript(
            """
            CREATE TABLE notes(id TEXT, client_id TEXT, deleted INTEGER, last_modified TEXT);
            CREATE TABLE chacha_keywords(id INTEGER, keyword TEXT, client_id TEXT, deleted INTEGER);
            CREATE TABLE note_keywords(note_id TEXT, keyword_id INTEGER);
            INSERT INTO notes VALUES ('own', 'pg-test', 0, ''), ('foreign', 'other', 0, '');
            INSERT INTO chacha_keywords VALUES (1, 'Own', 'pg-test', 0), (2, 'Foreign', 'other', 0);
            INSERT INTO note_keywords VALUES ('own', 1), ('own', 2), ('foreign', 1), ('foreign', 2);
            """
        )
        postgres_db.execute_query = lambda query, params, **kwargs: connection.execute(query, params)
        store = NoteStore(postgres_db)
        if operation == "single":
            assert [row["id"] for row in store.get_keywords_for_note("own")] == [1]
        elif operation == "foreign-note":
            assert store.get_keywords_for_note("foreign") == []
        elif operation == "batch":
            result = store.get_keywords_for_notes(["own", "foreign"])
            assert {note: [row["id"] for row in rows] for note, rows in result.items()} == {"own": [1], "foreign": []}
        elif operation == "reverse":
            assert [row["id"] for row in store.get_notes_for_keyword(1)] == ["own"]
        else:
            assert store.get_notes_for_keyword(2) == []

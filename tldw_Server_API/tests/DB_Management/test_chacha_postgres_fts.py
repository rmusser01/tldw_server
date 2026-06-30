from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence
from unittest.mock import MagicMock, call

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.character_store import CharacterStore
from tldw_Server_API.app.core.DB_Management.chacha.keyword_store import KeywordStore


class _CursorStub:
    """Minimal stub matching the subset of cursor API used by CharactersRAGDB."""

    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self._rows = list(rows)

    def fetchall(self) -> List[Dict[str, Any]]:

        return list(self._rows)


def _make_postgres_db() -> CharactersRAGDB:

    db = CharactersRAGDB.__new__(CharactersRAGDB)
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
    rows = [{"id": 1, "rank": 0.42}]
    db.execute_query = MagicMock(return_value=_CursorStub(rows))
    db._deserialize_row_fields = lambda row, _fields: row  # type: ignore[assignment]

    result = db.search_character_cards("dragon rider", limit=5)

    assert result == rows
    assert db.execute_query.call_count == 1
    sql, params = db.execute_query.call_args[0]
    assert "ts_rank" in sql and "to_tsquery('english', ?)" in sql
    assert params == ("dragon & rider", "dragon & rider", 5)


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
    assert params == ("uuid-1", "uuid-2")


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
    db.get_keyword_by_text = lambda _text: None  # type: ignore[assignment]
    db.add_keyword = lambda _text: 7  # type: ignore[assignment]

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
                return _Cursor(rows=[(1,)])
            if sql_upper.startswith("SELECT KEYWORD_ID FROM FLASHCARD_KEYWORDS"):
                return _Cursor(rows=[])
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
    assert params == ("fruit", "fruit", 5)

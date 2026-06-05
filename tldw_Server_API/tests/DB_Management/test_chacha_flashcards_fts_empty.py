from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.fts_translator import FTSQueryTranslator


def _make_sqlite_db() -> CharactersRAGDB:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = SimpleNamespace(backend_type=BackendType.SQLITE)
    db._uses_shared_content_backend = False
    db._local = SimpleNamespace(backend_ref=None)
    db.execute_query = MagicMock()
    db._get_current_utc_timestamp_iso = lambda: "2025-01-01T00:00:00Z"  # type: ignore[assignment]
    return db


def test_list_flashcards_sqlite_empty_fts_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_sqlite_db()
    monkeypatch.setattr(FTSQueryTranslator, "normalize_query", lambda _q, _backend: "")

    result = db.list_flashcards(q="!!!", limit=10, offset=0)

    assert result == []
    db.execute_query.assert_not_called()


def test_count_flashcards_sqlite_empty_fts_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _make_sqlite_db()
    monkeypatch.setattr(FTSQueryTranslator, "normalize_query", lambda _q, _backend: "")

    result = db.count_flashcards(q="!!!")

    assert result == 0
    db.execute_query.assert_not_called()

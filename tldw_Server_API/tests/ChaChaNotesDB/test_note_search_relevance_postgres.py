"""Notes search relevance, sort and snippets on PostgreSQL (NL-07, #3101).

The same contract as ``test_note_search_relevance.py``: relevance (ts_rank) by
default, an explicit sort ordered in SQL (including ``SELECT DISTINCT`` tag
searches, where PostgreSQL needs the sort key in the select list), and a
``ts_headline`` excerpt split into match segments. Another owner's notes never
appear and never get a snippet.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration

_FILLER = "Notes about flour, water and salt. " * 12


def _add(db: CharactersRAGDB, title: str, content: str, modified: str) -> str:
    note_id = db.add_note(title=title, content=content)
    db.execute_query(
        "UPDATE notes SET last_modified = ?, created_at = ? WHERE id = ?",
        (modified, modified, note_id),
        commit=True,
    )
    return note_id


def _tag(db: CharactersRAGDB, note_id: str, keyword: str) -> None:
    existing = db.get_keyword_by_text(keyword)
    keyword_id = existing["id"] if existing else db.add_keyword(keyword)
    db.link_note_to_keyword(note_id, keyword_id)


def _titles(rows: list[dict]) -> list[str]:
    return [row["title"] for row in rows]


def test_postgres_search_relevance_sort_and_snippets(pg_database_config: DatabaseConfig) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="930101", backend=backend)
    other = CharactersRAGDB(db_path=":memory:", client_id="930102", backend=backend)
    try:
        loaf = _add(
            db,
            "Sourdough country loaf",
            "Feed the sourdough starter, then mix the sourdough levain.",
            "2026-01-01T00:00:00+00:00",
        )
        bread = _add(db, "bread log", f"{_FILLER}Tried a sourdough crumb test. {_FILLER}", "2026-02-01T00:00:00+00:00")
        apple = _add(
            db, "Apple pie", f"{_FILLER}A friend mentioned sourdough once. {_FILLER}", "2026-03-01T00:00:00+00:00"
        )
        for note_id in (loaf, bread, apple):
            _tag(db, note_id, "baking")
        foreign = _add(other, "Another owner's sourdough", "sourdough sourdough", "2026-05-01T00:00:00+00:00")
        _tag(other, foreign, "baking")

        store = db.note_store
        relevance = store.search_notes("sourdough", limit=10)
        assert _titles(relevance)[0] == "Sourdough country loaf"
        assert set(_titles(relevance)) == {"Sourdough country loaf", "bread log", "Apple pie"}

        assert _titles(store.search_notes("sourdough", limit=10, sort_by="last_modified", sort_order="desc")) == [
            "Apple pie",
            "bread log",
            "Sourdough country loaf",
        ]
        assert _titles(store.search_notes("sourdough", limit=10, sort_by="title", sort_order="asc")) == [
            "Apple pie",
            "bread log",
            "Sourdough country loaf",
        ]
        # SELECT DISTINCT tag searches, with and without a query.
        assert _titles(
            store.search_notes_with_keywords("sourdough", ["baking"], limit=10, sort_by="title", sort_order="desc")
        ) == ["Sourdough country loaf", "bread log", "Apple pie"]
        assert _titles(
            store.search_notes_with_keywords(None, ["baking"], limit=10, sort_by="title", sort_order="asc")
        ) == ["Apple pie", "bread log", "Sourdough country loaf"]
        assert _titles(store.search_notes_with_keywords("sourdough", ["baking"], limit=10))[0] == (
            "Sourdough country loaf"
        )

        snippets = store.note_search_snippets("sourdough", [loaf, bread, apple, foreign])
        assert set(snippets) == {loaf, bread, apple}
        apple_segments = snippets[apple]
        assert any(segment["match"] and segment["text"].lower() == "sourdough" for segment in apple_segments)
        assert "sourdough" in "".join(segment["text"] for segment in apple_segments)

        # Note text holding the marker characters is stripped of them before marking.
        forged = _add(
            db, "Markers", "Forged \ue000fake\ue001 words, then the sourdough starter.", "2026-06-01T00:00:00+00:00"
        )
        forged_segments = store.note_search_snippets("sourdough", [forged])[forged]
        assert [segment["text"] for segment in forged_segments if segment["match"]] == ["sourdough"]
        forged_text = "".join(segment["text"] for segment in forged_segments)
        assert "\ue000" not in forged_text and "\ue001" not in forged_text
        assert "Forged fake words" in forged_text
    finally:
        db.close_all_connections()
        other.close_all_connections()

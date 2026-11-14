"""Exact, all/any tag filtering for notes on PostgreSQL (NL-N1, #3101)."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def _tag(db: CharactersRAGDB, note_id: str, keyword: str) -> None:
    existing = db.get_keyword_by_text(keyword)
    keyword_id = existing["id"] if existing else db.add_keyword(keyword)
    db.link_note_to_keyword(note_id, keyword_id)


def _titles(db: CharactersRAGDB, tokens: list[str], match: str, query: str | None = None) -> list[str]:
    rows = db.search_notes_with_keywords(query, tokens, limit=50, offset=0, match=match)
    return sorted(row["title"] for row in rows)


def test_postgres_tag_filter_is_exact_and_supports_all_and_any(pg_database_config: DatabaseConfig) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="930001", backend=backend)
    other = CharactersRAGDB(db_path=":memory:", client_id="930002", backend=backend)
    try:
        for title, tags in {
            "evaluation only": ["evaluation"],
            "eval only": ["Eval"],
            "rag only": ["rag"],
            "rag and baking": ["rag", "baking"],
        }.items():
            note_id = db.add_note(title=title, content=f"Body of {title}")
            for tag in tags:
                _tag(db, note_id, tag)
        # Another owner's note with both tags must never count for this owner.
        foreign = other.add_note(title="foreign rag and baking", content="Body")
        _tag(other, foreign, "rag")
        _tag(other, foreign, "baking")

        assert _titles(db, ["eval"], "all") == ["eval only"]
        assert _titles(db, ["rag", "baking"], "all") == ["rag and baking"]
        assert _titles(db, ["rag", "baking"], "any") == ["rag and baking", "rag only"]
        assert _titles(db, ["rag", "baking"], "all", query="Body") == ["rag and baking"]
        assert db.count_notes_matching_keywords(None, ["rag", "baking"], match="all") == 1
        assert db.count_notes_matching_keywords(None, ["rag", "baking"], match="any") == 2
        assert db.count_notes_matching_keywords(None, ["eval"], match="all") == 1
    finally:
        db.close_all_connections()
        other.close_all_connections()


def test_postgres_tag_filter_matches_by_keyword_identity(pg_database_config: DatabaseConfig) -> None:
    """LOWER() misses ß/SS; the filter compares normalize_keyword_identity instead."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="930003", backend=backend)
    other = CharactersRAGDB(db_path=":memory:", client_id="930004", backend=backend)
    try:
        for title, tags in {
            "eclair": ["Éclair"],
            "street": ["Straße"],
            "both": ["Éclair", "Straße"],
        }.items():
            note_id = db.add_note(title=title, content=f"Body of {title}")
            for tag in tags:
                _tag(db, note_id, tag)
        foreign = other.add_note(title="foreign street", content="Body")
        _tag(other, foreign, "Straße")

        assert _titles(db, ["ÉCLAIR"], "all") == ["both", "eclair"]
        assert _titles(db, ["STRASSE"], "all") == ["both", "street"]
        assert _titles(db, ["éclair", "strasse"], "all") == ["both"]
        assert _titles(db, ["éclair", "strasse"], "any") == ["both", "eclair", "street"]
        assert db.count_notes_matching_keywords(None, ["STRASSE"], match="all") == 2
        assert db.count_notes_matching_keywords(None, ["éclair", "strasse"], match="any") == 3
    finally:
        db.close_all_connections()
        other.close_all_connections()

"""Note tag filters match by keyword identity on SQLite (TASK-13950.5 follow-up).

SQLite's LOWER() and NOCASE fold ASCII only, so a filter that compared Python-lowered
tokens with ``LOWER(k.keyword)`` never matched a stored "Éclair" for "éclair", and
"STRASSE" never matched "Straße". Filters resolve tokens through
``normalize_keyword_identity`` (NFC plus casefold) instead, as Media does.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Sync_Contracts.notes_organization import new_organization_sync_id

pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(str(tmp_path / "tag_identity.sqlite"), client_id="tag-identity")
    yield database
    database.close_connection()


def _tag(db: CharactersRAGDB, note_id: str, keyword: str) -> None:
    existing = db.get_keyword_by_text(keyword)
    keyword_id = existing["id"] if existing else db.add_keyword(keyword)
    db.link_note_to_keyword(note_id, keyword_id)


def _titles(db: CharactersRAGDB, tokens: list[str], match: str) -> list[str]:
    rows = db.search_notes_with_keywords(None, tokens, limit=50, offset=0, match=match)
    return sorted(row["title"] for row in rows)


@pytest.fixture()
def tagged(db: CharactersRAGDB) -> CharactersRAGDB:
    for title, tags in {
        "eclair": ["Éclair"],
        "street": ["Straße"],
        "both": ["Éclair", "Straße"],
        "plain": ["plain"],
    }.items():
        note_id = db.add_note(title=title, content=f"Body of {title}")
        for tag in tags:
            _tag(db, note_id, tag)
    return db


@pytest.mark.parametrize("token", ["éclair", "ÉCLAIR", "Éclair"])
def test_non_ascii_capitalised_tag_matches_any_casing(tagged: CharactersRAGDB, token: str) -> None:
    assert _titles(tagged, [token], "all") == ["both", "eclair"]
    assert tagged.count_notes_matching_keywords(None, [token], match="all") == 2


def test_sharp_s_tag_matches_its_casefold(tagged: CharactersRAGDB) -> None:
    assert _titles(tagged, ["STRASSE"], "all") == ["both", "street"]
    assert tagged.count_notes_matching_keywords(None, ["strasse"], match="any") == 2


def test_all_and_any_combine_non_ascii_tags(tagged: CharactersRAGDB) -> None:
    assert _titles(tagged, ["éclair", "STRASSE"], "all") == ["both"]
    assert _titles(tagged, ["éclair", "STRASSE"], "any") == ["both", "eclair", "street"]
    assert tagged.count_notes_matching_keywords(None, ["éclair", "STRASSE"], match="all") == 1
    assert tagged.count_notes_matching_keywords(None, ["éclair", "STRASSE"], match="any") == 3


def test_tokens_with_the_same_identity_count_once_under_all(tagged: CharactersRAGDB) -> None:
    assert _titles(tagged, ["Éclair", "éclair"], "all") == ["both", "eclair"]


def test_unknown_token_matches_nothing_under_all_and_is_ignored_under_any(tagged: CharactersRAGDB) -> None:
    assert _titles(tagged, ["éclair", "missing"], "all") == []
    assert _titles(tagged, ["éclair", "missing"], "any") == ["both", "eclair"]
    assert _titles(tagged, ["missing"], "any") == []


def test_all_matches_a_legacy_duplicate_keyword_row(db: CharactersRAGDB) -> None:
    """A note tagged through an older identity variant still has the tag."""
    note_id = db.add_note(title="legacy", content="Body")
    _tag(db, note_id, "Éclair")
    with db.transaction() as conn:
        # Rows from before identity-based uniqueness: NOCASE lets "éclair" sit beside "Éclair".
        cursor = conn.execute(
            "INSERT INTO keywords (sync_id, keyword, client_id) VALUES (?, ?, ?)",
            (new_organization_sync_id(), "éclair", db.client_id),
        )
        legacy_id = cursor.lastrowid
    other_id = db.add_note(title="legacy only", content="Body")
    db.link_note_to_keyword(other_id, legacy_id)
    _tag(db, other_id, "plain")

    assert _titles(db, ["ÉCLAIR"], "all") == ["legacy", "legacy only"]
    assert _titles(db, ["ÉCLAIR", "plain"], "all") == ["legacy only"]

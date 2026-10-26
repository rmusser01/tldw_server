"""Case-insensitive Prompts keyword identity (TASK-13950.5, slice 2).

The Prompts DB is SQLite-only (no Postgres backend), so these cases run on
SQLite. ``PromptKeywordsTable.keyword`` is ``UNIQUE COLLATE NOCASE``, which
folds ASCII only; lookups fall back to ``normalize_keyword_identity`` so
"Éclair"/"éclair", "Straße"/"STRASSE" and NFC forms are one keyword. Display
casing is the first-seen casing. Known limit: non-ASCII duplicates that already
exist are not merged; lookups resolve them to the earliest row.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase

pytestmark = pytest.mark.unit

VARIANTS = [
    pytest.param("Foo", "foo", id="ascii"),
    pytest.param("Straße", "STRASSE", id="sharp-s"),
    pytest.param("Éclair", "éclair", id="accented-capital"),
    pytest.param("éclair", "éclair", id="nfc-composition"),
]


@pytest.fixture
def db(tmp_path):
    database = PromptsDatabase(tmp_path / "prompts.db", client_id="913950")
    yield database
    database.close_connection()


def _prompt_keywords(db: PromptsDatabase, name: str) -> list[str]:
    prompt = db.get_prompt_by_name(name)
    assert prompt is not None
    return db.fetch_keywords_for_prompt(prompt["id"])


@pytest.mark.parametrize(("stored", "probe"), VARIANTS)
def test_adding_a_case_variant_keeps_one_keyword_found_in_either_case(db, stored: str, probe: str) -> None:
    keyword_id, _ = db.add_keyword(stored)

    assert db.add_keyword(probe)[0] == keyword_id
    assert db.fetch_all_keywords() == [stored]
    for text in (stored, probe):
        found = db.get_active_keyword_by_text(text)
        assert found is not None and (found["id"], found["keyword"]) == (keyword_id, stored), text


@pytest.mark.parametrize(("stored", "probe"), VARIANTS)
def test_linking_case_variants_to_a_prompt_with_the_keyword_keeps_one_link(db, stored: str, probe: str) -> None:
    db.add_prompt("P", "author", "details", keywords=[stored])
    prompt = db.get_prompt_by_name("P")
    assert prompt is not None

    db.update_keywords_for_prompt(prompt["id"], [probe, stored])

    assert _prompt_keywords(db, "P") == [stored]
    results, total = db.search_prompts(f"keyword:{probe}")
    assert total == 1 and [row["name"] for row in results] == ["P"]


def test_new_keywords_keep_the_first_seen_casing(db) -> None:
    db.add_prompt("P", "author", "details", keywords=["foo", "Foo", "BAR", "bar"])

    assert _prompt_keywords(db, "P") == ["BAR", "foo"]


@pytest.mark.parametrize(("stored", "probe"), VARIANTS)
def test_soft_delete_in_another_case_then_readd_restores_the_same_keyword(db, stored: str, probe: str) -> None:
    keyword_id, _ = db.add_keyword(stored)

    assert db.soft_delete_keyword(probe) is True
    assert db.get_active_keyword_by_text(stored) is None
    assert db.add_keyword(probe)[0] == keyword_id
    assert db.fetch_all_keywords() == [stored]


def _insert_keyword(db: PromptsDatabase, keyword: str, uuid: str) -> int:
    """Seed a row the store itself would not create (legacy data)."""
    cursor = db.execute_query(
        "INSERT INTO PromptKeywordsTable (keyword, uuid, last_modified, version, client_id, deleted) "
        "VALUES (?, ?, '2026-10-01T00:00:00Z', 1, '913950', 0)",
        (keyword, uuid),
        commit=True,
    )
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def test_an_active_identity_match_wins_over_a_deleted_sql_match(db) -> None:
    deleted_id, _ = db.add_keyword("strasse")
    assert db.soft_delete_keyword("strasse") is True
    active_id = _insert_keyword(db, "Straße", "uuid-strasse-active")

    assert db.add_keyword("STRASSE")[0] == active_id != deleted_id
    found = db.get_active_keyword_by_text("STRASSE")
    assert found is not None and found["id"] == active_id
    assert db.fetch_all_keywords() == ["Straße"]


def test_keyword_search_finds_prompts_on_every_existing_identity_duplicate(db) -> None:
    db.add_prompt("P1", "author", "details", keywords=["Éclair"])
    db.add_prompt("P2", "author", "details")
    duplicate_id = _insert_keyword(db, "éclair", "uuid-eclair-duplicate")
    prompt = db.get_prompt_by_name("P2")
    assert prompt is not None
    db.execute_query(
        "INSERT INTO PromptKeywordLinks (prompt_id, keyword_id) VALUES (?, ?)",
        (prompt["id"], duplicate_id),
        commit=True,
    )

    results, total = db.search_prompts("keyword:ÉCLAIR")

    assert total == 2 and sorted(row["name"] for row in results) == ["P1", "P2"]

"""Notes search keeps relevance, honours an explicit sort and shows where a note matched (NL-07, #3101).

The search endpoint used to accept ``sort_by``/``sort_order`` and ignore them,
and the WebUI re-sorted each page by date, so the best title match for
"sourdough" rendered last and body-only hits showed only the first line.
``search_notes`` and ``search_notes_with_keywords`` now order by relevance
(bm25) unless a sort is asked for, and ``note_search_snippets`` returns a short,
highlighted excerpt of the body for the notes on a page.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_store import (
    NOTE_SEARCH_SNIPPET_MAX_CHARS,
    note_search_snippet_segments,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError

pytestmark = pytest.mark.unit

_FILLER = "Notes about flour, water and salt. " * 12


@pytest.fixture()
def db(tmp_path) -> Iterator[CharactersRAGDB]:
    database = CharactersRAGDB(str(tmp_path / "note_search_relevance.sqlite"), client_id="search-relevance")
    yield database
    database.close_all_connections()


def _add(db: CharactersRAGDB, title: str, content: str, *, modified: str, created: str | None = None) -> str:
    note_id = db.add_note(title=title, content=content)
    db.execute_query(
        "UPDATE notes SET last_modified = ?, created_at = ? WHERE id = ?",
        (modified, created or modified, note_id),
        commit=True,
    )
    return note_id


def _tag(db: CharactersRAGDB, note_id: str, keyword: str) -> None:
    existing = db.get_keyword_by_text(keyword)
    keyword_id = existing["id"] if existing else db.add_keyword(keyword)
    db.link_note_to_keyword(note_id, keyword_id)


@pytest.fixture()
def library(db: CharactersRAGDB) -> dict[str, str]:
    """Three hits whose relevance order is the reverse of their date order."""
    ids = {
        "loaf": _add(
            db,
            "Sourdough country loaf",
            "Feed the sourdough starter, then mix the sourdough levain.",
            modified="2026-01-01T00:00:00+00:00",
            created="2026-01-01T00:00:00+00:00",
        ),
        "bread": _add(
            db,
            "bread log",
            f"{_FILLER}Tried a sourdough crumb test. {_FILLER}",
            modified="2026-02-01T00:00:00+00:00",
            created="2025-06-01T00:00:00+00:00",
        ),
        "apple": _add(
            db,
            "Apple pie",
            f"{_FILLER} A friend mentioned sourdough once. {_FILLER}",
            modified="2026-03-01T00:00:00+00:00",
            created="2025-01-01T00:00:00+00:00",
        ),
        "other": _add(db, "Unrelated", "Nothing to see.", modified="2026-04-01T00:00:00+00:00"),
    }
    for key in ("loaf", "bread", "apple"):
        _tag(db, ids[key], "baking")
    return ids


def _titles(rows: list[dict]) -> list[str]:
    return [row["title"] for row in rows]


def test_default_order_is_relevance(db: CharactersRAGDB, library: dict[str, str]) -> None:
    rows = db.search_notes("sourdough", limit=10)
    assert _titles(rows)[0] == "Sourdough country loaf"
    assert set(_titles(rows)) == {"Sourdough country loaf", "bread log", "Apple pie"}


@pytest.mark.parametrize(
    ("sort_by", "sort_order", "expected"),
    [
        ("last_modified", "desc", ["Apple pie", "bread log", "Sourdough country loaf"]),
        ("last_modified", "asc", ["Sourdough country loaf", "bread log", "Apple pie"]),
        ("created_at", "desc", ["Sourdough country loaf", "bread log", "Apple pie"]),
        ("title", "asc", ["Apple pie", "bread log", "Sourdough country loaf"]),
        ("title", "desc", ["Sourdough country loaf", "bread log", "Apple pie"]),
    ],
)
def test_explicit_sort_orders_the_whole_result_in_sql(
    db: CharactersRAGDB, library: dict[str, str], sort_by: str, sort_order: str, expected: list[str]
) -> None:
    rows = db.note_store.search_notes("sourdough", limit=10, sort_by=sort_by, sort_order=sort_order)
    assert _titles(rows) == expected
    # Paging follows the same order, so page 2 continues page 1.
    first = db.note_store.search_notes("sourdough", limit=2, offset=0, sort_by=sort_by, sort_order=sort_order)
    second = db.note_store.search_notes("sourdough", limit=2, offset=2, sort_by=sort_by, sort_order=sort_order)
    assert _titles(first) + _titles(second) == expected


def test_relevance_sort_is_the_same_as_the_default(db: CharactersRAGDB, library: dict[str, str]) -> None:
    default = _titles(db.note_store.search_notes("sourdough", limit=10))
    assert _titles(db.note_store.search_notes("sourdough", limit=10, sort_by="relevance")) == default


def test_unsupported_sort_is_rejected(db: CharactersRAGDB, library: dict[str, str]) -> None:
    with pytest.raises(InputError):
        db.note_store.search_notes("sourdough", sort_by="rank; DROP TABLE notes", sort_order="desc")


def test_tag_search_with_a_query_keeps_relevance_and_honours_sort(
    db: CharactersRAGDB, library: dict[str, str]
) -> None:
    store = db.note_store
    relevance = store.search_notes_with_keywords("sourdough", ["baking"], limit=10)
    assert _titles(relevance)[0] == "Sourdough country loaf"
    by_title = store.search_notes_with_keywords(
        "sourdough", ["baking"], limit=10, sort_by="title", sort_order="asc"
    )
    assert _titles(by_title) == ["Apple pie", "bread log", "Sourdough country loaf"]


def test_tag_only_search_honours_sort(db: CharactersRAGDB, library: dict[str, str]) -> None:
    store = db.note_store
    assert _titles(store.search_notes_with_keywords(None, ["baking"], limit=10)) == [
        "Apple pie",
        "bread log",
        "Sourdough country loaf",
    ]
    assert _titles(
        store.search_notes_with_keywords(None, ["baking"], limit=10, sort_by="title", sort_order="desc")
    ) == ["Sourdough country loaf", "bread log", "Apple pie"]


def test_snippets_highlight_the_body_match(db: CharactersRAGDB, library: dict[str, str]) -> None:
    snippets = db.note_store.note_search_snippets("sourdough", list(library.values()))
    assert set(snippets) == {library["loaf"], library["bread"], library["apple"]}
    apple = snippets[library["apple"]]
    assert {"text": "sourdough", "match": True} in apple
    text = "".join(segment["text"] for segment in apple)
    assert "A friend mentioned sourdough once." in text
    # The excerpt is short: a window around the hit, not the whole body.
    assert len(text) <= NOTE_SEARCH_SNIPPET_MAX_CHARS
    assert text.startswith("…")


def test_snippets_skip_title_only_hits_and_other_notes(db: CharactersRAGDB) -> None:
    title_only = _add(db, "Sourdough plan", "Shopping list for later.", modified="2026-01-01T00:00:00+00:00")
    body = _add(db, "Plan", "Sourdough first.", modified="2026-01-02T00:00:00+00:00")
    deleted = _add(db, "Gone", "sourdough leftovers", modified="2026-01-03T00:00:00+00:00")
    db.soft_delete_note(deleted, expected_version=db.get_note_by_id(deleted)["version"])
    unrelated = _add(db, "Other", "sourdough again", modified="2026-01-04T00:00:00+00:00")

    snippets = db.note_store.note_search_snippets("sourdough", [title_only, body, deleted])

    assert set(snippets) == {body}
    assert snippets[body] == [{"text": "Sourdough", "match": True}, {"text": " first.", "match": False}]
    assert unrelated not in snippets


def test_snippets_need_a_term_and_ids(db: CharactersRAGDB, library: dict[str, str]) -> None:
    assert db.note_store.note_search_snippets("   ", list(library.values())) == {}
    assert db.note_store.note_search_snippets("sourdough", []) == {}


def test_snippet_segments_parse_markers_and_flatten_whitespace() -> None:
    marked = "Feed\n\nthe \ue000sourdough\ue001  starter \ue000Sourdough\ue001"
    assert note_search_snippet_segments(marked) == [
        {"text": "Feed the ", "match": False},
        {"text": "sourdough", "match": True},
        {"text": " starter ", "match": False},
        {"text": "Sourdough", "match": True},
    ]


def test_snippet_segments_without_a_match_are_dropped() -> None:
    assert note_search_snippet_segments("Shopping list for later.") is None
    assert note_search_snippet_segments("") is None
    assert note_search_snippet_segments(None) is None


def test_snippet_segments_are_bounded() -> None:
    marked = "\ue000hit\ue001 " + "word " * 400
    segments = note_search_snippet_segments(marked)
    assert segments is not None
    text = "".join(segment["text"] for segment in segments)
    assert len(text) <= NOTE_SEARCH_SNIPPET_MAX_CHARS
    assert text.endswith("…")
    assert segments[0] == {"text": "hit", "match": True}


def _recorded_order_clauses(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch, search) -> list[str]:
    """Run ``search`` and return the ORDER BY clause of each notes query it sent."""
    sent: list[str] = []
    original = db.execute_query

    def record(query, *args, **kwargs):
        flat = " ".join(str(query).split())
        if "ORDER BY" in flat and "LIMIT" in flat:
            sent.append(flat[flat.rindex("ORDER BY") : flat.rindex("LIMIT")].strip())
        return original(query, *args, **kwargs)

    monkeypatch.setattr(db, "execute_query", record)
    search()
    return sent


@pytest.mark.parametrize(
    "search",
    [
        lambda db: db.search_notes("sourdough", limit=2, offset=1),
        lambda db: db.search_notes_with_keywords("sourdough", ["baking"], limit=2, offset=1),
        lambda db: db.search_notes_with_keywords(None, ["baking"], limit=2, offset=1),
    ],
    ids=["text", "text-and-tags", "tags-only"],
)
def test_relevance_order_breaks_ties_by_id(
    db: CharactersRAGDB, library: dict[str, str], monkeypatch: pytest.MonkeyPatch, search
) -> None:
    """Notes that tie on score and date must page in one stable order."""
    clauses = _recorded_order_clauses(db, monkeypatch, lambda: search(db))
    assert clauses
    assert all(clause.endswith("id ASC") for clause in clauses), clauses


def test_tied_notes_page_without_overlap(db: CharactersRAGDB) -> None:
    ids = {
        _add(db, "Same", "a sourdough note", modified="2026-01-01T00:00:00+00:00") for _ in range(4)
    }
    seen = [db.search_notes("sourdough", limit=1, offset=offset)[0]["id"] for offset in range(4)]
    assert set(seen) == ids
    assert seen == sorted(seen)


def test_note_text_cannot_forge_a_highlight(db: CharactersRAGDB) -> None:
    """Note text holding the private marker characters shows as text, never as a match."""
    note_id = _add(
        db,
        "Markers",
        "Forged fake words, then the sourdough starter.",
        modified="2026-01-01T00:00:00+00:00",
    )

    segments = db.note_store.note_search_snippets("sourdough", [note_id])[note_id]

    assert [segment["text"] for segment in segments if segment["match"]] == ["sourdough"]
    text = "".join(segment["text"] for segment in segments)
    assert "" not in text and "" not in text
    assert "Forged fake words" in text

"""An include list decides WHICH notes come back, not what order they come back in.

Both include-list paths -- _retrieve_allowed_notes_via_sql and its ChaChaNotes
sibling -- used to stamp score=1.0 on every row and return them in last_modified
order, so a caller asking for relevance got recency wearing a perfect-relevance score.

Cross-source rank fusion (ADR-049) stopped those rows crowding other sources out of a
multi-source result set, which was the acute failure. It does nothing for a
single-source notes query, which is what these tests cover.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    NotesDBRetriever,
    RetrievalConfig,
)

_ROWS = [
    {"id": "1", "title": "Weekly review", "content": "nothing to see"},
    {"id": "2", "title": "Alpha launch", "content": "the alpha plan"},
    {"id": "3", "title": "Budget", "content": "mentions alpha once"},
]


def _retriever(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]]):
    retriever = NotesDBRetriever.__new__(NotesDBRetriever)
    retriever.config = RetrievalConfig(max_results=10)
    retriever.chacha_db = None
    monkeypatch.setattr(
        retriever, "_execute_query", lambda *_args, **_kwargs: list(rows), raising=False
    )
    return retriever


@pytest.mark.unit
@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"title": "Alpha", "content": "alpha"}, 1.5),
        ({"title": "Alpha", "content": "none"}, 1.0),
        ({"title": "Other", "content": "alpha"}, 0.5),
        ({"title": "Other", "content": "none"}, 0.0),
    ],
)
def test_text_match_score(row: dict[str, Any], expected: float) -> None:
    assert NotesDBRetriever._text_match_score("alpha", row) == expected


@pytest.mark.unit
def test_an_empty_query_scores_every_row_equally() -> None:
    """There is no relevance to measure, so claiming a difference would be worse."""
    assert NotesDBRetriever._text_match_score("", {"title": "a", "content": "b"}) == 1.0
    assert NotesDBRetriever._text_match_score("   ", {"title": "a", "content": "b"}) == 1.0


@pytest.mark.unit
def test_included_notes_come_back_in_relevance_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The database returns last_modified order; relevance reorders it."""
    retriever = _retriever(monkeypatch, _ROWS)

    documents = retriever._retrieve_allowed_notes_via_sql(
        ["1", "2", "3"], None, "alpha"
    )

    assert [document.metadata["note_id"] for document in documents] == ["2", "3", "1"]
    assert [document.score for document in documents] == [1.5, 0.5, 0.0]


@pytest.mark.unit
def test_a_note_that_does_not_match_is_still_returned_but_scored_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The include list is the filter -- no text match is REQUIRED, only scored."""
    retriever = _retriever(monkeypatch, _ROWS)

    documents = retriever._retrieve_allowed_notes_via_sql(
        ["1", "2", "3"], None, "alpha"
    )

    assert len(documents) == 3
    assert documents[-1].metadata["note_id"] == "1"
    assert documents[-1].score == 0.0


@pytest.mark.unit
def test_the_chacha_include_path_scores_the_same_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both include-list paths must agree; only one of them is used per deployment."""
    retriever = NotesDBRetriever.__new__(NotesDBRetriever)
    retriever.config = RetrievalConfig(max_results=10)
    by_id = {row["id"]: row for row in _ROWS}
    retriever.chacha_db = type(
        "_Db", (), {"get_note_by_id": staticmethod(lambda note_id: by_id.get(note_id))}
    )()

    documents = retriever._retrieve_allowed_notes_via_chacha(["1", "2", "3"], None, "alpha")

    assert [document.metadata["note_id"] for document in documents] == ["2", "3", "1"]
    assert [document.score for document in documents] == [1.5, 0.5, 0.0]

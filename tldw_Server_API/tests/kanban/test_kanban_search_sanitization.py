from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints.kanban import kanban_search


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.warning_calls = []

    def warning(self, *args, **kwargs):
        self.warning_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "vector backend exploded",
    "/private/kanban-vector.db",
)


def _patch_logger(monkeypatch: pytest.MonkeyPatch) -> _LoggerStub:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(kanban_search, "logger", logger_stub, raising=True)
    return logger_stub


def _assert_sanitized_warning(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.warning_calls
    messages = [args[0] for args, _kwargs in logger_stub.warning_calls if args]
    assert expected_message in messages
    rendered_calls = repr(logger_stub.warning_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


class _FtsFallbackDb:
    def __init__(self):
        self.search_calls = 0

    def search_cards(self, **_kwargs):
        self.search_calls += 1
        return ([{"id": 1}], 1)


class _FailingVectorSearch:
    available = True

    def search(self, **_kwargs):
        raise RuntimeError("vector backend exploded /private/kanban-vector.db")


def test_vector_search_fallback_sanitizes_failure_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_logger(monkeypatch)
    db = _FtsFallbackDb()

    cards, total, mode = kanban_search._perform_vector_search(
        db=db,
        vector_search=_FailingVectorSearch(),
        query="auth",
        board_id=None,
        label_ids=None,
        priority=None,
        include_archived=False,
        limit=10,
        offset=0,
    )

    assert cards == [{"id": 1}]
    assert total == 1
    assert mode == "fts"
    assert db.search_calls == 1
    _assert_sanitized_warning(logger_stub, "Vector search failed, falling back to FTS")


class _HybridVectorOnlyFetchDb(_FtsFallbackDb):
    def get_cards_by_ids(self, **_kwargs):
        raise RuntimeError("vector backend exploded /private/kanban-vector.db")


class _VectorOnlyResultSearch:
    available = True

    def search(self, **_kwargs):
        return [{"card_id": 2, "relevance_score": 0.9}]


def test_hybrid_search_vector_only_fetch_sanitizes_failure_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_logger(monkeypatch)

    cards, total, mode = kanban_search._perform_hybrid_search(
        db=_HybridVectorOnlyFetchDb(),
        vector_search=_VectorOnlyResultSearch(),
        query="auth",
        board_id=None,
        label_ids=None,
        priority=None,
        include_archived=False,
        limit=10,
        offset=0,
    )

    assert [card["id"] for card in cards] == [1]
    assert total == 1
    assert mode == "hybrid"
    _assert_sanitized_warning(logger_stub, "Failed to fetch vector-only cards in hybrid search")


def test_hybrid_search_fallback_sanitizes_failure_log(monkeypatch: pytest.MonkeyPatch):
    logger_stub = _patch_logger(monkeypatch)

    cards, total, mode = kanban_search._perform_hybrid_search(
        db=_FtsFallbackDb(),
        vector_search=_FailingVectorSearch(),
        query="auth",
        board_id=None,
        label_ids=None,
        priority=None,
        include_archived=False,
        limit=10,
        offset=0,
    )

    assert [card["id"] for card in cards] == [1]
    assert total == 1
    assert mode == "fts"
    _assert_sanitized_warning(logger_stub, "Hybrid search failed, using FTS only")

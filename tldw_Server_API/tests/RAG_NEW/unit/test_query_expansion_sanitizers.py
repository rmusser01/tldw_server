"""Sanitizer coverage for query expansion fallback logs."""

import pytest

from tldw_Server_API.app.core.RAG.rag_service import query_expansion


pytestmark = pytest.mark.unit

_SENSITIVE_SUBSTRINGS = ("/tmp/source", "token=secret")


class _RecordingLogger:
    def __init__(self) -> None:
        self.debug_records: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debug_records.append((str(message), args, dict(kwargs)))


def _failing_expander(label: str):
    async def fail(_query: str):
        raise RuntimeError(f"{label} failed for /tmp/source?token=secret")

    return fail


def _assert_debug_records_are_sanitized(
    logger_stub: _RecordingLogger,
    expected_messages: list[str],
) -> None:
    assert [record[0] for record in logger_stub.debug_records] == expected_messages

    for _message, args, kwargs in logger_stub.debug_records:
        assert args == ()
        assert kwargs == {}
        assert "exc_info" not in kwargs

    serialized_records = repr(logger_stub.debug_records)
    for sensitive in _SENSITIVE_SUBSTRINGS:
        assert sensitive not in serialized_records


@pytest.mark.asyncio
async def test_multi_strategy_expansion_sanitizes_all_fallback_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    query = "private query"

    monkeypatch.setattr(query_expansion, "logger", logger_stub)
    monkeypatch.setattr(
        query_expansion,
        "expand_acronyms",
        _failing_expander("acronym expansion"),
    )
    monkeypatch.setattr(
        query_expansion,
        "expand_synonyms",
        _failing_expander("synonym expansion"),
    )
    monkeypatch.setattr(
        query_expansion,
        "domain_specific_expansion",
        _failing_expander("domain expansion"),
    )

    result = await query_expansion.multi_strategy_expansion(
        query,
        strategies=["acronym", "synonym", "domain"],
    )

    assert result == query
    _assert_debug_records_are_sanitized(
        logger_stub,
        [
            "Acronym expansion failed; continuing to next strategy",
            "Synonym expansion failed; continuing to next strategy",
            "Domain expansion failed; continuing",
        ],
    )

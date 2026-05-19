from __future__ import annotations

import sqlite3
from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.feedback_system as feedback_system
from tldw_Server_API.app.core.RAG.rag_service.feedback_system import (
    FeedbackAnalyzer,
    FeedbackEntry,
    FeedbackStore,
    FeedbackSystem,
    FeedbackType,
)


pytestmark = pytest.mark.unit


_SECRET_PATH = "/private/rag-feedback/tenant-token.db"
_SECRET_TOKEN = "feedback-secret-token"
_SECRET_USER_ID = f"user-{_SECRET_TOKEN}"
_SECRET_QUERY = f"reset auth {_SECRET_TOKEN}"
_SECRET_DOC_ID = f"doc-{_SECRET_TOKEN}"
_SENSITIVE_MARKERS = (
    "backend exploded",
    _SECRET_PATH,
    "tenant-token.db",
    _SECRET_TOKEN,
    _SECRET_USER_ID,
    _SECRET_QUERY,
    _SECRET_DOC_ID,
)


def _capture_log_records(level: str = "ERROR") -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []

    def _sink(message: Any) -> None:
        records.append(
            {
                "message": str(message.record.get("message") or ""),
                "extra": dict(message.record.get("extra") or {}),
                "exception": message.record.get("exception"),
            }
        )

    sink_id = feedback_system.logger.add(_sink, level=level)
    return records, sink_id


def _assert_sanitized_records(
    records: list[dict[str, Any]],
    expected_messages: list[str],
) -> None:
    assert [record["message"] for record in records] == expected_messages
    rendered_logs = "\n".join(str(record) for record in records)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_logs
    assert all("exc_info" not in record["extra"] for record in records)
    assert all(record["exception"] is None for record in records)


def _secret_exception() -> RuntimeError:
    return RuntimeError(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")


class _MalformedFeedbackStore:
    db_path = ":memory:"

    def get_document_feedback(self, _document_id: str) -> list[dict[str, Any]]:
        return [
            {"feedback_type": "relevance", "value": f"bad relevance {_SECRET_TOKEN}"},
            {"feedback_type": "helpful", "value": f"bad helpful {_SECRET_TOKEN}"},
            {"feedback_type": "dwell_time", "value": f"bad dwell {_SECRET_TOKEN}"},
        ]

    def get_feedback_for_query(self, _query: str) -> list[dict[str, Any]]:
        return [
            {
                "document_id": _SECRET_DOC_ID,
                "feedback_type": "relevance",
                "value": f"bad relevance {_SECRET_TOKEN}",
            },
            {
                "document_id": _SECRET_DOC_ID,
                "feedback_type": "helpful",
                "value": f"bad helpful {_SECRET_TOKEN}",
            },
            {
                "document_id": _SECRET_DOC_ID,
                "feedback_type": "dwell_time",
                "value": f"bad dwell {_SECRET_TOKEN}",
            },
        ]


def test_feedback_store_add_feedback_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FeedbackStore.__new__(FeedbackStore)
    store.db_path = _SECRET_PATH
    entry = FeedbackEntry(
        id="fb-1",
        query=_SECRET_QUERY,
        document_id=_SECRET_DOC_ID,
        user_id=_SECRET_USER_ID,
        feedback_type=FeedbackType.HELPFUL,
        value=True,
    )

    def _raise_connect(_db_path: str) -> None:
        raise sqlite3.Error(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")

    monkeypatch.setattr(feedback_system.sqlite3, "connect", _raise_connect)
    records, sink_id = _capture_log_records()

    try:
        result = store.add_feedback(entry)
    finally:
        feedback_system.logger.remove(sink_id)

    assert result is False
    _assert_sanitized_records(records, ["Failed to store feedback"])


def test_feedback_analyzer_document_parse_logs_are_sanitized() -> None:
    analyzer = FeedbackAnalyzer(_MalformedFeedbackStore())  # type: ignore[arg-type]
    records, sink_id = _capture_log_records("DEBUG")

    try:
        result = analyzer.calculate_document_score(_SECRET_DOC_ID)
    finally:
        feedback_system.logger.remove(sink_id)

    assert result == 0.5
    _assert_sanitized_records(
        records,
        [
            "Failed to parse relevance feedback value",
            "Failed to parse helpful feedback value",
            "Failed to parse dwell_time feedback value",
        ],
    )


def test_feedback_analyzer_query_parse_logs_are_sanitized() -> None:
    analyzer = FeedbackAnalyzer(_MalformedFeedbackStore())  # type: ignore[arg-type]
    records, sink_id = _capture_log_records("DEBUG")

    try:
        perf = analyzer.get_query_performance(_SECRET_QUERY)
    finally:
        feedback_system.logger.remove(sink_id)

    assert perf.avg_relevance == 0.0
    assert perf.helpful_count == 0
    assert perf.unhelpful_count == 0
    assert perf.avg_dwell_time == 0.0
    _assert_sanitized_records(
        records,
        [
            "Failed to parse relevance score for query document",
            "Failed to parse helpful feedback value",
            "Failed to parse dwell_time feedback value",
        ],
    )


def test_feedback_analyzer_reranking_parse_log_is_sanitized() -> None:
    analyzer = FeedbackAnalyzer(_MalformedFeedbackStore())  # type: ignore[arg-type]
    records, sink_id = _capture_log_records("DEBUG")

    try:
        weights = analyzer.get_reranking_weights(_SECRET_QUERY, [_SECRET_DOC_ID])
    finally:
        feedback_system.logger.remove(sink_id)

    assert weights == {_SECRET_DOC_ID: 1.0}
    _assert_sanitized_records(
        records,
        [
            "Failed to parse relevance feedback value",
            "Failed to parse helpful feedback value",
            "Failed to parse dwell_time feedback value",
            "Failed to parse relevance score for query document",
        ],
    )


@pytest.mark.asyncio
async def test_feedback_system_submit_feedback_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system = FeedbackSystem.__new__(FeedbackSystem)
    system.store = object()
    system.analyzer = object()
    system.active_sessions = {}

    def _raise_generate_feedback_id(*_args: Any, **_kwargs: Any) -> str:
        raise _secret_exception()

    monkeypatch.setattr(system, "generate_feedback_id", _raise_generate_feedback_id)
    records, sink_id = _capture_log_records()

    try:
        result = await system.submit_feedback(
            query=_SECRET_QUERY,
            document_id=_SECRET_DOC_ID,
            user_id=_SECRET_USER_ID,
            feedback_type=FeedbackType.HELPFUL,
            value=True,
        )
    finally:
        feedback_system.logger.remove(sink_id)

    assert result is False
    _assert_sanitized_records(records, ["Failed to submit feedback"])

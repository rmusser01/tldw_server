from __future__ import annotations

from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.analytics_system as analytics_system
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.RAG.rag_service.analytics_system import AnalyticsStore


pytestmark = pytest.mark.unit


_SECRET_PATH = "/private/analytics-system/tenant-token.db"
_SECRET_TOKEN = "analytics-secret-token"
_SENSITIVE_MARKERS = (
    "backend exploded",
    _SECRET_PATH,
    "tenant-token.db",
    _SECRET_TOKEN,
)


class _FailingAnalyticsDb:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __getattr__(self, name: str) -> Any:
        def _raise(*args: Any) -> None:
            self.calls.append((name, args))
            raise self.exc

        return _raise


def _analytics_store_with_failing_db(exc: Exception) -> tuple[AnalyticsStore, _FailingAnalyticsDb]:
    store = AnalyticsStore.__new__(AnalyticsStore)
    db = _FailingAnalyticsDb(exc)
    store.db = db
    return store, db


def _capture_analytics_error_logs() -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = analytics_system.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="ERROR",
    )
    return messages, sink_id


@pytest.mark.asyncio
@pytest.mark.parametrize("exc_type", [BackendDatabaseError, RuntimeError])
@pytest.mark.parametrize(
    ("store_method", "args", "fallback", "expected_log"),
    [
        (
            "record_search",
            ({"query": "reset auth", "results_count": 1},),
            False,
            "Failed to record search analytics",
        ),
        (
            "record_feedback",
            ({"feedback_type": "helpful", "helpful": True},),
            False,
            "Failed to record feedback",
        ),
        (
            "record_event",
            ({"event_type": "search", "metrics": {"latency_ms": 10}},),
            False,
            "Failed to record analytics event",
        ),
        (
            "record_document_performance",
            ({"document_id": "doc-1", "clicks": 1},),
            False,
            "Failed to record document performance",
        ),
        (
            "record_error",
            ({"error_type": "backend", "count": 1},),
            False,
            "Failed to record error",
        ),
        (
            "record_feature_usage",
            ({"feature": "hybrid_search", "count": 1},),
            False,
            "Failed to record feature usage",
        ),
        (
            "get_analytics_summary",
            (7,),
            {},
            "Failed to get analytics summary",
        ),
        (
            "cleanup_old_data",
            (90,),
            0,
            "Failed to cleanup old data",
        ),
    ],
)
async def test_analytics_store_fail_open_logs_are_sanitized(
    exc_type: type[Exception],
    store_method: str,
    args: tuple[Any, ...],
    fallback: Any,
    expected_log: str,
) -> None:
    exc = exc_type(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")
    store, db = _analytics_store_with_failing_db(exc)
    messages, sink_id = _capture_analytics_error_logs()

    try:
        result = await getattr(store, store_method)(*args)
    finally:
        analytics_system.logger.remove(sink_id)

    assert result == fallback
    assert len(db.calls) == 1
    assert db.calls[0][0] == store_method
    assert messages == [expected_log]
    rendered_logs = "\n".join(messages)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_logs

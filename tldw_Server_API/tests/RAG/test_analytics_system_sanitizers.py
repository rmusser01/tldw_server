from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.analytics_system as analytics_system
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.RAG.rag_service.analytics_system import (
    AnalyticsStore,
    UnifiedFeedbackSystem,
    UserFeedbackStore,
)


pytestmark = pytest.mark.unit


_SECRET_PATH = "/private/analytics-system/tenant-token.db"
_SECRET_TOKEN = "analytics-secret-token"
_SECRET_FEEDBACK_ID = f"fb_secret_{_SECRET_TOKEN}"
_SECRET_USER_ID = f"user-{_SECRET_TOKEN}"
_SENSITIVE_MARKERS = (
    "backend exploded",
    _SECRET_PATH,
    "tenant-token.db",
    _SECRET_TOKEN,
    _SECRET_FEEDBACK_ID,
    _SECRET_USER_ID,
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

    sink_id = analytics_system.logger.add(_sink, level=level)
    return records, sink_id


class _FailingFeedbackConnection:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def execute(self, *_args: Any, **_kwargs: Any) -> Any:
        raise self.exc


class _FailingFeedbackTransaction:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def __enter__(self) -> _FailingFeedbackConnection:
        return _FailingFeedbackConnection(self.exc)

    def __exit__(self, *_args: Any) -> bool:
        return False


class _FailingFeedbackDb:
    backend_type = BackendType.SQLITE

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def transaction(self) -> _FailingFeedbackTransaction:
        return _FailingFeedbackTransaction(self.exc)

    def execute_query(self, *_args: Any, **_kwargs: Any) -> Any:
        raise self.exc


def _user_feedback_store_with_failing_db(exc: Exception) -> UserFeedbackStore:
    store = UserFeedbackStore.__new__(UserFeedbackStore)
    store.db = _FailingFeedbackDb(exc)
    return store


def _assert_sanitized_records(
    records: list[dict[str, Any]],
    expected_message: str,
) -> None:
    assert [record["message"] for record in records] == [expected_message]
    rendered_logs = "\n".join(str(record) for record in records)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_logs
    assert all("exc_info" not in record["extra"] for record in records)
    assert all(record["exception"] is None for record in records)


def _secret_exception() -> RuntimeError:
    return RuntimeError(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")


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


def test_user_feedback_store_schema_initialization_log_is_sanitized() -> None:
    exc = _secret_exception()
    db = _FailingFeedbackDb(exc)
    records, sink_id = _capture_log_records()

    try:
        UserFeedbackStore(db)
    finally:
        analytics_system.logger.remove(sink_id)

    _assert_sanitized_records(records, "Failed to initialize feedback schema")


@pytest.mark.asyncio
async def test_user_feedback_store_add_feedback_log_is_sanitized_and_reraises() -> None:
    exc = _secret_exception()
    store = _user_feedback_store_with_failing_db(exc)
    records, sink_id = _capture_log_records()

    try:
        with pytest.raises(RuntimeError) as raised:
            await store.add_feedback(
                conversation_id="conv-1",
                query="reset auth",
                document_ids=["doc-1"],
                chunk_ids=["chunk-1"],
            )
    finally:
        analytics_system.logger.remove(sink_id)

    assert raised.value is exc
    _assert_sanitized_records(records, "Failed to add feedback")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "args", "expected_result", "expected_message"),
    [
        (
            "merge_feedback_update",
            (_SECRET_FEEDBACK_ID, ["missing_details"], "note"),
            None,
            "Failed to merge feedback update",
        ),
        (
            "get_conversation_feedback",
            ("conv-1",),
            [],
            "Failed to get conversation feedback",
        ),
        (
            "get_feedback_by_id",
            (_SECRET_FEEDBACK_ID,),
            None,
            "Failed to get feedback by id",
        ),
    ],
)
async def test_user_feedback_store_fallback_logs_are_sanitized(
    method_name: str,
    args: tuple[Any, ...],
    expected_result: Any,
    expected_message: str,
) -> None:
    store = _user_feedback_store_with_failing_db(_secret_exception())
    records, sink_id = _capture_log_records()

    try:
        result = await getattr(store, method_name)(*args)
    finally:
        analytics_system.logger.remove(sink_id)

    assert result == expected_result
    _assert_sanitized_records(records, expected_message)


@pytest.mark.asyncio
async def test_user_feedback_store_delete_feedback_log_is_sanitized_and_reraises() -> None:
    exc = _secret_exception()
    store = _user_feedback_store_with_failing_db(exc)
    records, sink_id = _capture_log_records()

    try:
        with pytest.raises(RuntimeError) as raised:
            await store.delete_feedback(_SECRET_FEEDBACK_ID)
    finally:
        analytics_system.logger.remove(sink_id)

    assert raised.value is exc
    _assert_sanitized_records(records, "Failed to delete feedback")


class _FailingImplicitAnalytics:
    async def record_event(self, *_args: Any, **_kwargs: Any) -> None:
        raise _secret_exception()


def _install_personalization_module(monkeypatch: pytest.MonkeyPatch, cls: type[Any]) -> None:
    module = ModuleType("user_personalization_store")
    module.UserPersonalizationStore = cls
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.RAG.rag_service.user_personalization_store",
        module,
    )


@pytest.mark.asyncio
async def test_unified_feedback_implicit_personalization_skip_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _InvalidPersonalizationStore:
        def __init__(self, _user_id: str) -> None:
            raise ValueError(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")

    _install_personalization_module(monkeypatch, _InvalidPersonalizationStore)
    system = UnifiedFeedbackSystem.__new__(UnifiedFeedbackSystem)
    system.enable_analytics = False
    system.analytics = None
    records, sink_id = _capture_log_records("DEBUG")

    try:
        result = await system.record_implicit_interaction(
            user_id=_SECRET_USER_ID,
            query="reset auth",
            doc_id="doc-1",
            event_type="click",
        )
    finally:
        analytics_system.logger.remove(sink_id)

    assert result is None
    _assert_sanitized_records(records, "Personalization store update skipped")


@pytest.mark.asyncio
async def test_unified_feedback_implicit_personalization_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenPersonalizationStore:
        def __init__(self, _user_id: str) -> None:
            pass

        def record_event(self, **_kwargs: Any) -> None:
            raise _secret_exception()

    _install_personalization_module(monkeypatch, _BrokenPersonalizationStore)
    system = UnifiedFeedbackSystem.__new__(UnifiedFeedbackSystem)
    system.enable_analytics = False
    system.analytics = None
    records, sink_id = _capture_log_records("DEBUG")

    try:
        result = await system.record_implicit_interaction(
            user_id=_SECRET_USER_ID,
            query="reset auth",
            doc_id="doc-1",
            event_type="copy",
        )
    finally:
        analytics_system.logger.remove(sink_id)

    assert result is None
    _assert_sanitized_records(records, "Personalization store update failed")


@pytest.mark.asyncio
async def test_unified_feedback_implicit_outer_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WorkingPersonalizationStore:
        def __init__(self, _user_id: str) -> None:
            pass

        def record_event(self, **_kwargs: Any) -> None:
            pass

    _install_personalization_module(monkeypatch, _WorkingPersonalizationStore)
    system = UnifiedFeedbackSystem.__new__(UnifiedFeedbackSystem)
    system.enable_analytics = True
    system.analytics = _FailingImplicitAnalytics()
    records, sink_id = _capture_log_records("DEBUG")

    try:
        result = await system.record_implicit_interaction(
            user_id=_SECRET_USER_ID,
            query="reset auth",
            doc_id="doc-1",
            event_type="expand",
            session_id=f"session-{_SECRET_TOKEN}",
        )
    finally:
        analytics_system.logger.remove(sink_id)

    assert result is None
    _assert_sanitized_records(records, "Implicit interaction recording failed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("store_factory", "expected_message"),
    [
        (
            lambda: type(
                "_InvalidBoostStore",
                (),
                {
                    "__init__": lambda self, _user_id: (_ for _ in ()).throw(
                        ValueError(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")
                    )
                },
            ),
            "Feedback boost skipped",
        ),
        (
            lambda: type(
                "_FailingBoostStore",
                (),
                {
                    "__init__": lambda self, _user_id: None,
                    "boost_documents": lambda self, _documents, corpus=None: (_ for _ in ()).throw(
                        RuntimeError(f"backend exploded at {_SECRET_PATH}?token={_SECRET_TOKEN}")
                    ),
                },
            ),
            "Feedback boost failed",
        ),
    ],
)
async def test_apply_feedback_boost_logs_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    store_factory: Any,
    expected_message: str,
) -> None:
    _install_personalization_module(monkeypatch, store_factory())
    documents = [SimpleNamespace(id=f"doc-{_SECRET_TOKEN}")]
    context = SimpleNamespace(
        config={
            "feedback": {"apply_boost": True},
            "user_id": _SECRET_USER_ID,
            "index_namespace": f"corpus-{_SECRET_TOKEN}",
        },
        documents=documents,
    )
    records, sink_id = _capture_log_records("DEBUG")

    try:
        result = await analytics_system.apply_feedback_boost(context)
    finally:
        analytics_system.logger.remove(sink_id)

    assert result is context
    assert context.documents is documents
    _assert_sanitized_records(records, expected_message)

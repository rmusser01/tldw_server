"""Sanitizer coverage for ChromaDB optimizer fallback logs."""

import pytest

from tldw_Server_API.app.core.RAG.rag_service import chromadb_optimizer
from tldw_Server_API.app.core.RAG.rag_service.chromadb_optimizer import (
    ChromaDBOptimizationConfig,
    ChromaDBOptimizer,
    OptimizedChromaStore,
)


pytestmark = pytest.mark.unit

_EMPTY_SEARCH_RESULT = {
    "ids": [[]],
    "distances": [[]],
    "documents": [[]],
    "metadatas": [[]],
}
_SENSITIVE_SUBSTRINGS = ("/tmp/source", "token=secret", "chromadb failed")


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("debug", str(message), args, dict(kwargs)))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("error", str(message), args, dict(kwargs)))

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("info", str(message), args, dict(kwargs)))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("warning", str(message), args, dict(kwargs)))


class _FailingQueryCollection:
    name = "private_collection"

    def query(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("chromadb failed /tmp/source token=secret")


class _FailingAddCollection:
    def add(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("chromadb failed /tmp/source token=secret")


class _FailingMetadataCollection:
    def get(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("chromadb failed /tmp/source token=secret")


class _FailingGetCollectionClient:
    def get_collection(self, _collection_name: str) -> None:
        raise RuntimeError("chromadb failed /tmp/source token=secret")

    def create_collection(self, *_args: object, **_kwargs: object) -> object:
        return object()


class _PersistentClientFactory:
    def __call__(self, *_args: object, **_kwargs: object) -> _FailingGetCollectionClient:
        return _FailingGetCollectionClient()


class _CountingCollection:
    def count(self) -> int:
        return 0


class _FailingBatchOptimizer:
    async def batch_add_optimized(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError("chromadb failed /tmp/source token=secret")


def _assert_records_are_sanitized(
    logger_stub: _RecordingLogger,
    expected_records: list[tuple[str, str]],
) -> None:
    recorded_messages = [
        (level, message)
        for level, message, _args, _kwargs in logger_stub.records
    ]
    assert recorded_messages == expected_records

    for _level, _message, args, kwargs in logger_stub.records:
        assert args == ()
        assert "exc_info" not in kwargs

    serialized_records = repr(logger_stub.records)
    for sensitive in _SENSITIVE_SUBSTRINGS:
        assert sensitive not in serialized_records


@pytest.mark.asyncio
async def test_search_with_cache_failure_returns_empty_result_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_optimizer, "logger", logger_stub)
    monkeypatch.setattr(chromadb_optimizer, "CHROMADB_AVAILABLE", True)

    optimizer = ChromaDBOptimizer(ChromaDBOptimizationConfig(enable_result_cache=False))

    try:
        result = await optimizer.search_with_cache(
            _FailingQueryCollection(),
            query_text="private query",
        )
    finally:
        optimizer.executor.shutdown(wait=True)

    assert result == _EMPTY_SEARCH_RESULT
    _assert_records_are_sanitized(
        logger_stub,
        [
            ("info", "Initialized ChromaDB optimizer"),
            ("error", "ChromaDB search failed"),
        ],
    )


def test_client_get_collection_fallback_sanitizes_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_optimizer, "logger", logger_stub)
    monkeypatch.setattr(chromadb_optimizer, "CHROMADB_AVAILABLE", True)
    monkeypatch.setattr(
        chromadb_optimizer,
        "chromadb",
        type("FakeChromaModule", (), {"PersistentClient": _PersistentClientFactory()})(),
    )
    monkeypatch.setattr(chromadb_optimizer, "Settings", lambda **_kwargs: object())

    client = OptimizedChromaStore("/tmp/source", "private_collection")
    client.optimizer.executor.shutdown(wait=True)

    assert client.collection is not None
    _assert_records_are_sanitized(
        logger_stub,
        [
            ("info", "Initialized ChromaDB optimizer"),
            ("debug", "Chroma get_collection failed, creating new collection"),
        ],
    )


@pytest.mark.asyncio
async def test_client_add_documents_failure_returns_false_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_optimizer, "logger", logger_stub)

    client = OptimizedChromaStore.__new__(OptimizedChromaStore)
    client.config = ChromaDBOptimizationConfig()
    client.collection = _CountingCollection()
    client.optimizer = _FailingBatchOptimizer()

    result = await client.add_documents(
        documents=["doc"],
        embeddings=[[0.1]],
        metadatas=[{"source": "private"}],
        ids=["doc-1"],
    )

    assert result is False
    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to add documents"),
        ],
    )


@pytest.mark.asyncio
async def test_batch_add_failure_reraises_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_optimizer, "logger", logger_stub)
    monkeypatch.setattr(chromadb_optimizer, "CHROMADB_AVAILABLE", True)

    optimizer = ChromaDBOptimizer(ChromaDBOptimizationConfig(enable_result_cache=False))

    try:
        with pytest.raises(RuntimeError, match="chromadb failed /tmp/source token=secret"):
            await optimizer.batch_add_optimized(
                _FailingAddCollection(),
                documents=["doc"],
                embeddings=[[0.1]],
                metadatas=[{"source": "private"}],
                ids=["doc-1"],
            )
    finally:
        optimizer.executor.shutdown(wait=True)

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("info", "Initialized ChromaDB optimizer"),
            ("error", "Failed to add batch 1"),
        ],
    )


@pytest.mark.asyncio
async def test_optimize_metadata_indexing_failure_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_optimizer, "logger", logger_stub)
    monkeypatch.setattr(chromadb_optimizer, "CHROMADB_AVAILABLE", True)

    optimizer = ChromaDBOptimizer(ChromaDBOptimizationConfig(enable_result_cache=False))

    try:
        await optimizer.optimize_metadata_indexing(_FailingMetadataCollection())
    finally:
        optimizer.executor.shutdown(wait=True)

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("info", "Initialized ChromaDB optimizer"),
            ("warning", "Could not optimize metadata indexing"),
        ],
    )

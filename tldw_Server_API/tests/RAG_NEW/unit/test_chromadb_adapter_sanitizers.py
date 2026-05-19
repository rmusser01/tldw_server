"""Sanitizer coverage for ChromaDB vector store adapter fallback logs."""

import contextlib

import pytest

from tldw_Server_API.app.core.RAG.rag_service.vector_stores import chromadb_adapter
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.base import (
    VectorStoreConfig,
    VectorStoreType,
)
from tldw_Server_API.app.core.RAG.rag_service.vector_stores.chromadb_adapter import (
    ChromaDBAdapter,
)


pytestmark = pytest.mark.unit

_PRIVATE_COLLECTION = "private_collection"
_PRIVATE_EXCEPTION = "chromadb failed /tmp/source token=secret"
_SENSITIVE_SUBSTRINGS = (_PRIVATE_COLLECTION, "/tmp/source", "token=secret", "chromadb failed")


class _NoopTracingManager:
    @contextlib.contextmanager
    def span(self, *_args: object, **_kwargs: object):
        yield None


@pytest.fixture(autouse=True)
def _disable_tracing_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        chromadb_adapter,
        "get_tracing_manager",
        lambda: _NoopTracingManager(),
    )


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("info", str(message), args, dict(kwargs)))

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("error", str(message), args, dict(kwargs)))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("warning", str(message), args, dict(kwargs)))

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("debug", str(message), args, dict(kwargs)))


class _NamedCollection:
    def __init__(self, name: str) -> None:
        self.name = name


class _FailingQueryCollection:
    def query(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(_PRIVATE_EXCEPTION)


class _FailingStatsClient:
    def get_collection(self, *, name: str) -> None:
        assert name == _PRIVATE_COLLECTION
        raise RuntimeError(_PRIVATE_EXCEPTION)


class _FailingCreateCollection:
    def modify(self, *, metadata: dict[str, object]) -> None:
        assert metadata["embedding_dimension"] == 2
        raise RuntimeError(_PRIVATE_EXCEPTION)


class _FailingDeleteCollectionClient:
    def delete_collection(self, *, name: str) -> None:
        assert name == _PRIVATE_COLLECTION
        raise RuntimeError(_PRIVATE_EXCEPTION)


class _FailingListCollectionsClient:
    def list_collections(self) -> None:
        raise RuntimeError(_PRIVATE_EXCEPTION)


class _FailingDeleteVectorsCollection:
    def get(self, *_args: object, **_kwargs: object) -> dict[str, list[str]]:
        return {"ids": ["vec-1"]}

    def delete(self, *, ids: list[str]) -> None:
        assert ids == ["vec-1"]
        raise RuntimeError(_PRIVATE_EXCEPTION)


class _FailingDeleteVectorsClient:
    def get_collection(self, *, name: str) -> _FailingDeleteVectorsCollection:
        assert name == _PRIVATE_COLLECTION
        return _FailingDeleteVectorsCollection()


class _PrivateFailingEmbeddings:
    def tolist(self) -> list[list[float]]:
        raise RuntimeError(_PRIVATE_EXCEPTION)

    def __getitem__(self, _index: int) -> list[float]:
        raise RuntimeError(_PRIVATE_EXCEPTION)


class _FailingEmbeddingsListCollection:
    def count(self) -> int:
        return 1

    def get(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "ids": ["vec-1"],
            "embeddings": _PrivateFailingEmbeddings(),
            "documents": ["private document"],
            "metadatas": [{"source": "/tmp/source", "token": "secret"}],
        }


class _FailingEmbeddingsListManager:
    def get_or_create_collection(self, collection_name: str) -> _FailingEmbeddingsListCollection:
        assert collection_name == _PRIVATE_COLLECTION
        return _FailingEmbeddingsListCollection()


class _PrivateFailingEmbeddingShape:
    def __len__(self) -> int:
        raise RuntimeError(_PRIVATE_EXCEPTION)


class _FailingEmbeddingShapeStatsCollection:
    metadata: dict[str, object] = {}

    def count(self) -> int:
        return 1

    def get(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {"embeddings": [_PrivateFailingEmbeddingShape()]}


class _FailingEmbeddingShapeStatsClient:
    def get_collection(self, *, name: str) -> _FailingEmbeddingShapeStatsCollection:
        assert name == _PRIVATE_COLLECTION
        return _FailingEmbeddingShapeStatsCollection()


class _FailingEmbeddingShapeStatsManager:
    client = _FailingEmbeddingShapeStatsClient()


class _ListedFailingSearchClient:
    def list_collections(self) -> list[_NamedCollection]:
        return [_NamedCollection(_PRIVATE_COLLECTION)]


class _FailingSearchManager:
    client = _ListedFailingSearchClient()

    def get_or_create_collection(self, collection_name: str) -> _FailingQueryCollection:
        assert collection_name == _PRIVATE_COLLECTION
        return _FailingQueryCollection()


class _FailingCreateManager:
    def get_or_create_collection(self, collection_name: str) -> _FailingCreateCollection:
        assert collection_name == _PRIVATE_COLLECTION
        return _FailingCreateCollection()


class _FailingDeleteCollectionManager:
    client = _FailingDeleteCollectionClient()


class _FailingListCollectionsManager:
    client = _FailingListCollectionsClient()


class _FailingUpsertManager:
    def store_in_chroma(self, **kwargs: object) -> None:
        assert kwargs["collection_name"] == _PRIVATE_COLLECTION
        raise RuntimeError(_PRIVATE_EXCEPTION)


class _FailingDeleteVectorsManager:
    client = _FailingDeleteVectorsClient()


class _FailingStatsManager:
    client = _FailingStatsClient()


class _FailingDeleteByFilterManager:
    def get_or_create_collection(self, collection_name: str) -> None:
        assert collection_name == _PRIVATE_COLLECTION
        raise RuntimeError(_PRIVATE_EXCEPTION)


def _adapter_with_manager(manager: object) -> ChromaDBAdapter:
    adapter = ChromaDBAdapter(
        VectorStoreConfig(
            store_type=VectorStoreType.CHROMADB,
            connection_params={"embedding_config": {}},
            embedding_dim=2,
            user_id="user-1",
        )
    )
    adapter.manager = manager  # type: ignore[assignment]
    adapter._initialized = True
    return adapter


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
async def test_initialize_failure_reraises_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)

    class _FailingChromaDBManager:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError(_PRIVATE_EXCEPTION)

    monkeypatch.setattr(chromadb_adapter, "ChromaDBManager", _FailingChromaDBManager)
    adapter = ChromaDBAdapter(
        VectorStoreConfig(
            store_type=VectorStoreType.CHROMADB,
            connection_params={"embedding_config": {}},
            embedding_dim=2,
            user_id="user-1",
        )
    )

    with pytest.raises(RuntimeError, match=_PRIVATE_EXCEPTION):
        await adapter.initialize()

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to initialize ChromaDB adapter"),
        ],
    )


@pytest.mark.asyncio
async def test_create_collection_failure_reraises_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingCreateManager())

    with pytest.raises(RuntimeError, match=_PRIVATE_EXCEPTION):
        await adapter.create_collection(_PRIVATE_COLLECTION)

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to create ChromaDB collection"),
        ],
    )


@pytest.mark.asyncio
async def test_delete_collection_failure_reraises_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingDeleteCollectionManager())

    with pytest.raises(RuntimeError, match=_PRIVATE_EXCEPTION):
        await adapter.delete_collection(_PRIVATE_COLLECTION)

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to delete ChromaDB collection"),
        ],
    )


@pytest.mark.asyncio
async def test_list_collections_failure_reraises_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingListCollectionsManager())

    with pytest.raises(RuntimeError, match=_PRIVATE_EXCEPTION):
        await adapter.list_collections()

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to list ChromaDB collections"),
        ],
    )


@pytest.mark.asyncio
async def test_upsert_vectors_failure_reraises_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingUpsertManager())

    with pytest.raises(RuntimeError, match=_PRIVATE_EXCEPTION):
        await adapter.upsert_vectors(
            _PRIVATE_COLLECTION,
            ["vec-1"],
            [[0.1, 0.2]],
            ["private document"],
            [{"source": "/tmp/source", "token": "secret"}],
        )

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to upsert ChromaDB vectors"),
        ],
    )


@pytest.mark.asyncio
async def test_delete_vectors_failure_reraises_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingDeleteVectorsManager())

    with pytest.raises(RuntimeError, match=_PRIVATE_EXCEPTION):
        await adapter.delete_vectors(_PRIVATE_COLLECTION, ["vec-1"])

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to delete ChromaDB vectors"),
        ],
    )


@pytest.mark.asyncio
async def test_list_vectors_embedding_conversion_fallback_sanitizes_debug_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingEmbeddingsListManager())

    result = await adapter.list_vectors_with_embeddings_paginated(
        _PRIVATE_COLLECTION,
        limit=1,
        offset=0,
    )

    assert result == {
        "items": [
            {
                "id": "vec-1",
                "vector": [],
                "content": "private document",
                "metadata": {"source": "/tmp/source", "token": "secret"},
            },
        ],
        "total": 1,
    }
    _assert_records_are_sanitized(
        logger_stub,
        [
            ("debug", "Chroma adapter failed to convert embeddings to list"),
        ],
    )


@pytest.mark.asyncio
async def test_search_failure_reraises_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingSearchManager())

    with pytest.raises(RuntimeError, match=_PRIVATE_EXCEPTION):
        await adapter.search(_PRIVATE_COLLECTION, [0.1, 0.2], k=1)

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to search ChromaDB collection"),
        ],
    )


@pytest.mark.asyncio
async def test_multi_search_collection_failure_returns_empty_and_sanitizes_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingSearchManager())

    result = await adapter.multi_search([_PRIVATE_COLLECTION], [0.1, 0.2], k=1)

    assert result == []
    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to search ChromaDB collection"),
            ("warning", "Failed to search ChromaDB collection during multi-search"),
        ],
    )


@pytest.mark.asyncio
async def test_multi_search_list_collections_failure_reraises_and_sanitizes_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingListCollectionsManager())

    with pytest.raises(RuntimeError, match=_PRIVATE_EXCEPTION):
        await adapter.multi_search([_PRIVATE_COLLECTION], [0.1, 0.2], k=1)

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to list ChromaDB collections"),
            ("error", "Failed to perform ChromaDB multi-search"),
        ],
    )


@pytest.mark.asyncio
async def test_get_collection_stats_failure_reraises_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingStatsManager())

    with pytest.raises(RuntimeError, match=_PRIVATE_EXCEPTION):
        await adapter.get_collection_stats(_PRIVATE_COLLECTION)

    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to get ChromaDB collection stats"),
        ],
    )


@pytest.mark.asyncio
async def test_get_collection_stats_embedding_shape_fallback_sanitizes_debug_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingEmbeddingShapeStatsManager())

    result = await adapter.get_collection_stats(_PRIVATE_COLLECTION)

    assert result == {
        "name": _PRIVATE_COLLECTION,
        "count": 1,
        "dimension": 2,
        "metadata": {},
        "distance_metric": "cosine",
    }
    _assert_records_are_sanitized(
        logger_stub,
        [
            ("debug", "Chroma adapter failed to inspect embedding shape"),
        ],
    )


@pytest.mark.asyncio
async def test_delete_by_filter_failure_returns_zero_and_sanitizes_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _RecordingLogger()
    monkeypatch.setattr(chromadb_adapter, "logger", logger_stub)
    adapter = _adapter_with_manager(_FailingDeleteByFilterManager())

    result = await adapter.delete_by_filter(_PRIVATE_COLLECTION, {"source": "private"})

    assert result == 0
    _assert_records_are_sanitized(
        logger_stub,
        [
            ("error", "Failed to delete ChromaDB vectors by filter"),
        ],
    )

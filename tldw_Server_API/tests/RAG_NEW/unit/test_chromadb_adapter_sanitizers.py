"""Sanitizer coverage for ChromaDB vector store adapter fallback logs."""

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


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("error", str(message), args, dict(kwargs)))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.records.append(("warning", str(message), args, dict(kwargs)))


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


class _ListedFailingSearchClient:
    def list_collections(self) -> list[_NamedCollection]:
        return [_NamedCollection(_PRIVATE_COLLECTION)]


class _FailingSearchManager:
    client = _ListedFailingSearchClient()

    def get_or_create_collection(self, collection_name: str) -> _FailingQueryCollection:
        assert collection_name == _PRIVATE_COLLECTION
        return _FailingQueryCollection()


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

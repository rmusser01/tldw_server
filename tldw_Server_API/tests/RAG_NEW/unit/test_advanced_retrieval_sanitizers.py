"""Sanitizer coverage for advanced retrieval fallback logs."""

import pytest

from tldw_Server_API.app.core.RAG.rag_service import advanced_retrieval as ar
from tldw_Server_API.app.core.RAG.rag_service.types import Document


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message):
        self.warnings.append(str(message))


class _QueryEmbeddingFailureService:
    async def create_embedding(self, *, text, user_id=None):
        raise RuntimeError("query embedding failed for /private/rag-query.db?token=secret")


class _SpanEmbeddingFailureService:
    async def create_embedding(self, *, text, user_id=None):
        return [1.0, 0.0]

    async def create_embeddings_batch(self, batch, *, user_id=None):
        raise RuntimeError("span embedding failed for /private/rag-span.db?token=secret")


def _docs() -> list[Document]:
    return [
        Document(id="doc-1", content="alpha beta gamma", metadata={}, score=0.5),
        Document(id="doc-2", content="delta epsilon zeta", metadata={}, score=0.4),
    ]


def _assert_log_is_sanitized(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.warnings == [expected_message]
    joined = "\n".join(logger_stub.warnings)
    assert "/private/" not in joined
    assert "secret" not in joined
    assert "rag-query.db" not in joined
    assert "rag-span.db" not in joined


@pytest.mark.asyncio
async def test_query_embedding_fallback_warning_omits_backend_exception(monkeypatch):
    logger_stub = _LoggerStub()
    documents = _docs()

    monkeypatch.setattr(ar, "logger", logger_stub)
    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _QueryEmbeddingFailureService())

    result = await ar.apply_multi_vector_passages("private query", documents)

    assert result is documents
    _assert_log_is_sanitized(
        logger_stub,
        "Query embedding failed; skipping multi-vector passages",
    )


@pytest.mark.asyncio
async def test_span_embedding_fallback_warning_omits_backend_exception(monkeypatch):
    logger_stub = _LoggerStub()
    documents = _docs()

    monkeypatch.setattr(ar, "logger", logger_stub)
    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _SpanEmbeddingFailureService())

    result = await ar.apply_multi_vector_passages("alpha", documents)

    assert result is documents
    _assert_log_is_sanitized(
        logger_stub,
        "Span embeddings failed; skipping multi-vector passages",
    )

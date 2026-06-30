"""Sanitizer coverage for enhanced chunking fallback logs."""

import pytest

from tldw_Server_API.app.core.RAG.rag_service import enhanced_chunking_integration as eci
from tldw_Server_API.app.core.RAG.rag_service.types import (
    DataSource,
    Document,
    RAGPipelineContext,
)


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(str(message))

    def info(self, message: str) -> None:
        self.infos.append(str(message))


class _FailingChunker:
    def __init__(self, _config) -> None:
        pass

    def chunk_text_with_metadata(self, **_kwargs):
        raise RuntimeError(
            "chunker failed for /private/rag/chunker.db?token=secret-token"
        )


@pytest.mark.asyncio
async def test_per_document_chunking_fallback_log_omits_raw_doc_id_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(eci, "logger", logger_stub)
    monkeypatch.setattr(eci, "Chunker", _FailingChunker)

    document = Document(
        id="/private/rag/documents/source.txt?token=secret-token",
        content="Sensitive source content should fall back unchanged.",
        metadata={"title": "Sensitive Source"},
        source=DataSource.MEDIA_DB,
        score=0.84,
    )
    context = RAGPipelineContext(
        query="private query",
        original_query="private query",
        documents=[document],
        metadata={},
        config={},
    )

    result = await eci.enhanced_chunk_documents(context)

    assert result is context
    assert result.documents == [document]
    assert result.documents[0] is document
    assert result.metadata["enhanced_chunking_applied"] is True
    assert result.metadata["total_chunks_created"] == 0
    assert result.metadata["chunk_type_distribution"] == {}
    assert logger_stub.errors == ["Failed to chunk document: RuntimeError"]

    serialized_logs = repr(logger_stub.errors)
    assert "/private/" not in serialized_logs
    assert "secret-token" not in serialized_logs
    assert "source.txt" not in serialized_logs
    assert "chunker failed" not in serialized_logs
    assert "chunker.db" not in serialized_logs

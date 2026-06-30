from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import tldw_Server_API.app.core.LLM_Calls as llm_calls
from tldw_Server_API.app.core.RAG.rag_service import hyde
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline


class _FakeLogger:
    def __init__(self):
        self.debugs = []
        self.warnings = []

    def debug(self, message):
        self.debugs.append(message)

    def warning(self, message):
        self.warnings.append(message)


@pytest.mark.unit
def test_generate_with_llm_sanitizes_generation_failure(monkeypatch):
    secret = "sk-secret-hyde-generation"

    def fake_analyze(**kwargs):
        raise RuntimeError(secret)

    fake_sgl = SimpleNamespace(analyze=fake_analyze)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(hyde, "logger", fake_logger)
    monkeypatch.setattr(llm_calls, "Summarization_General_Lib", fake_sgl)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib",
        fake_sgl,
    )

    assert hyde._generate_with_llm("prompt", "openai", "gpt-4o-mini") == ""
    assert fake_logger.warnings == ["HyDE LLM generation failed"]
    assert secret not in "\n".join(fake_logger.warnings + fake_logger.debugs)


@pytest.mark.unit
def test_generate_with_llm_sanitizes_utility_unavailable(monkeypatch):
    secret = "sk-secret-hyde-utility"
    real_import = __import__("builtins").__import__

    def fake_import(name, *args, **kwargs):
        if name == "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib":
            raise ImportError(secret)
        return real_import(name, *args, **kwargs)

    fake_logger = _FakeLogger()
    monkeypatch.setattr(hyde, "logger", fake_logger)
    monkeypatch.setattr("builtins.__import__", fake_import)

    assert hyde._generate_with_llm("prompt", "openai", "gpt-4o-mini") is None
    assert fake_logger.debugs == ["HyDE LLM utility unavailable"]
    assert secret not in "\n".join(fake_logger.warnings + fake_logger.debugs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_sanitizes_embedding_failure(monkeypatch):
    secret = "sk-secret-hyde-embedding"
    real_import = __import__("builtins").__import__

    def fake_import(name, *args, **kwargs):
        if name == "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create":
            raise RuntimeError(secret)
        return real_import(name, *args, **kwargs)

    fake_logger = _FakeLogger()
    monkeypatch.setattr(hyde, "logger", fake_logger)
    monkeypatch.setattr("builtins.__import__", fake_import)

    assert await hyde.embed_text("text") is None
    assert fake_logger.warnings == ["HyDE embedding failed"]
    assert secret not in "\n".join(fake_logger.warnings + fake_logger.debugs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unified_pipeline_with_hyde_merges_results():
    """Ensure HyDE path runs and merged results are returned when enabled."""
    base_docs = [
        Document(id="base1", content="Base A", metadata={}, source=DataSource.MEDIA_DB, score=0.2),
        Document(id="base2", content="Base B", metadata={}, source=DataSource.MEDIA_DB, score=0.1),
    ]
    hyde_docs = [
        Document(id="hyde1", content="HyDE A", metadata={}, source=DataSource.MEDIA_DB, score=0.9),
        Document(id="hyde2", content="HyDE B", metadata={}, source=DataSource.MEDIA_DB, score=0.8),
    ]

    # Fake retriever that returns baseline docs and offers a MEDIA_DB retriever with retrieve_hybrid
    class _FakeMediaRetriever:
        async def retrieve_hybrid(self, *args, **kwargs):
            # Ensure HyDE vector was provided via kwargs for vector search
            assert "query_vector" in kwargs
            return hyde_docs

    class _FakeMultiRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {DataSource.MEDIA_DB: _FakeMediaRetriever()}
        async def retrieve(self, *args, **kwargs):
            return base_docs

    with patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever", _FakeMultiRetriever), \
         patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.generate_hypothetical_answer", return_value="Hypo answer"), \
         patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.hyde_embed_text", new=AsyncMock(return_value=[0.1, 0.2, 0.3])):
        result = await unified_rag_pipeline(
            query="test hyde",
            sources=["media_db"],
            top_k=10,
            enable_hyde=True,
            adaptive_hybrid_weights=False,
        )

        # Response shape
        from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
        assert isinstance(result, UnifiedRAGResponse)
        # HyDE metadata present
        assert result.metadata.get("hyde_applied") is True
        assert result.metadata.get("hyde_merged_count") == len(hyde_docs)
        # Documents include both baseline and hyde docs (dedup by id)
        ids = {d["id"] for d in result.documents}
        for d in base_docs + hyde_docs:
            assert d.id in ids

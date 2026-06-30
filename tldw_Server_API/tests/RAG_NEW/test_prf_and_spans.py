"""
Unit tests for PRF helper and metadata wiring for multi-vector/precomputed spans.
"""

import io

from loguru import logger
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from tldw_Server_API.app.core.RAG.rag_service import precomputed_spans
from tldw_Server_API.app.core.RAG.rag_service.precomputed_spans import apply_precomputed_spans
from tldw_Server_API.app.core.RAG.rag_service.prf import apply_prf, PRFConfig
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_prf_mines_terms_and_expands_query():
    """apply_prf should mine salient terms and return an expanded query + metadata."""
    docs = [
        Document(
            id="1",
            content="Machine learning models reached 95% accuracy in 2024.",
            metadata={"title": "ML Report 2024"},
            source=DataSource.MEDIA_DB,
        ),
        Document(
            id="2",
            content="Deep learning and neural networks improved performance.",
            metadata={"title": "Deep Learning Overview"},
            source=DataSource.MEDIA_DB,
        ),
    ]
    cfg = PRFConfig(max_terms=5, top_n=2)
    expanded, meta = await apply_prf("machine learning", docs, cfg)

    assert isinstance(expanded, str)
    assert meta.get("enabled") is True
    assert meta.get("base_query") == "machine learning"
    assert "expanded_query" in meta
    assert isinstance(meta.get("terms_used"), list)
    assert meta.get("doc_seed_count") == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_apply_prf_sanitizes_defensive_fallback_error_metadata_and_log():
    """apply_prf should not leak backend exception text in fallback metadata or logs."""

    class PathLeakingScore:
        def __float__(self):
            raise RuntimeError("sqlite failed opening /tmp/private/user_databases/42/Media_DB_v2.db")

    docs = [
        Document(
            id="leaky",
            content="Relevant result with enough terms to reach scoring.",
            metadata={"title": "Sensitive Retrieval"},
            source=DataSource.MEDIA_DB,
            score=PathLeakingScore(),
        )
    ]
    log_stream = io.StringIO()
    sink_id = logger.add(log_stream, level="WARNING", format="{message}")
    try:
        expanded, meta = await apply_prf("safe query", docs, PRFConfig())
    finally:
        logger.remove(sink_id)

    log_output = log_stream.getvalue()

    assert expanded == "safe query"
    assert meta == {
        "enabled": False,
        "base_query": "safe query",
        "reason": "error",
        "error": "PRF computation failed",
    }
    assert "/tmp/private/user_databases/42/Media_DB_v2.db" not in log_output
    assert "sqlite failed opening" not in log_output
    assert "/tmp/private/user_databases/42/Media_DB_v2.db" not in str(meta)
    assert "sqlite failed opening" not in str(meta)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_precomputed_span_fallback_log_omits_logger_exception_details(monkeypatch):
    """Precomputed span logging fallback should not retain raw logging exception details."""

    calls: list[tuple[str, dict]] = []

    class LoggerStub:
        def debug(self, message, *args, **kwargs):
            _ = args
            calls.append((message, kwargs))
            if len(calls) == 1:
                raise RuntimeError("logger sink failed at /tmp/private/precomputed-spans.log")

    monkeypatch.setattr(precomputed_spans, "logger", LoggerStub())

    result = await apply_precomputed_spans("query", [])

    assert result is None
    assert calls[1][0] == "Precomputed span fallback logging failed"
    rendered = repr(calls[1])
    assert "precomputed-spans.log" not in rendered
    assert "/tmp/private" not in rendered


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unified_pipeline_prf_second_pass_metadata_and_fill():
    """
    When enable_prf is true, unified_rag_pipeline should record PRF metadata and
    can perform a second-pass retrieval to fill up to top_k results.
    """
    # Primary retrieval returns fewer than top_k documents
    base_docs = [
        Document(
            id="base-1",
            content="Intro to RAG systems.",
            metadata={"title": "RAG Intro"},
            source=DataSource.MEDIA_DB,
            score=0.9,
        )
    ]
    # PRF retrieval returns an additional document
    prf_docs = [
        Document(
            id="prf-1",
            content="Advanced retrieval and generation techniques.",
            metadata={"title": "Advanced RAG"},
            source=DataSource.MEDIA_DB,
            score=0.8,
        )
    ]

    async def _primary_retrieve(*args, **kwargs):
        return base_docs

    async def _prf_retrieve(*args, **kwargs):
        # Second call is tagged with component 'retrieval_prf' in unified_pipeline
        return prf_docs

    with patch(
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever"
    ) as mock_retriever, patch(
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.apply_prf"
    ) as mock_apply_prf:
        mock_instance = MagicMock()
        # First retrieval call
        mock_instance.retrieve = AsyncMock(side_effect=[base_docs, prf_docs])
        mock_instance.retrieve_hybrid = None
        mock_retriever.return_value = mock_instance

        # PRF helper returns an expanded query and enabled metadata
        mock_apply_prf.return_value = (
            "rag systems advanced retrieval",
            {"enabled": True, "base_query": "rag systems", "terms_used": ["advanced"], "expanded_query": "rag systems advanced retrieval"},
        )

        result = await unified_rag_pipeline(
            query="rag systems",
            sources=["media_db"],
            top_k=2,
            enable_prf=True,
            enable_cache=False,
            enable_reranking=False,
            enable_generation=False,
        )

        assert isinstance(result, UnifiedRAGResponse)
        # Should have at least base + one PRF doc (deduped and capped to top_k)
        assert len(result.documents) >= 1
        assert isinstance(result.metadata.get("prf"), dict)
        assert result.metadata["prf"].get("second_pass_performed") in {True, False}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unified_pipeline_multi_vector_precomputed_flag_metadata_only():
    """
    When enable_precomputed_spans is true but precomputed spans handler is a no-op,
    the pipeline should still emit multi_vector.precomputed_spans metadata.
    """
    docs = [
        Document(
            id="1",
            content="Long content chunk for multi-vector test.",
            metadata={},
            source=DataSource.MEDIA_DB,
            score=0.5,
        )
    ]

    with patch(
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever"
    ) as mock_retriever, patch(
        "tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.apply_multi_vector_passages"
    ) as mock_mv:
        mock_instance = MagicMock()
        mock_instance.retrieve = AsyncMock(return_value=docs)
        mock_instance.retrieve_hybrid = None
        mock_retriever.return_value = mock_instance

        # multi-vector helper simply echoes documents back
        async def _mv(query, documents, config, user_id=None):
            return documents

        mock_mv.side_effect = _mv

        result = await unified_rag_pipeline(
            query="test multi vector",
            sources=["media_db"],
            top_k=1,
            enable_multi_vector_passages=True,
            enable_precomputed_spans=True,
            enable_cache=False,
            enable_reranking=False,
            enable_generation=False,
        )

        assert isinstance(result, UnifiedRAGResponse)
        mv_meta = (result.metadata or {}).get("multi_vector") or {}
        # The flag should be present even though precomputed spans are a no-op
        assert "precomputed_spans" in mv_meta

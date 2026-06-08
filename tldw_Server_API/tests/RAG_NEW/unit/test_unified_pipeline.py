"""
Unit tests for the unified RAG pipeline.

Tests the single unified pipeline function with all feature combinations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    unified_rag_pipeline,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, SearchResult, DataSource


@pytest.mark.unit
class TestUnifiedPipeline:
    """Test the unified RAG pipeline function."""

    @pytest.mark.asyncio
    async def test_unified_pipeline_minimal(self):
        """Test unified pipeline with minimal parameters."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id="1", content="Test content", metadata={}, source=DataSource.MEDIA_DB, score=0.9)
            ])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                mock_generator_instance = MagicMock()
                mock_generator_instance.generate = AsyncMock(return_value={
                    "answer": "Generated answer",
                    "confidence": 0.85
                })
                mock_generator.return_value = mock_generator_instance

                result = await unified_rag_pipeline(
                    query="What is RAG?",
                    top_k=5
                )

                assert result is not None
                # Normalize access across dict or Pydantic object
                answer = getattr(result, 'generated_answer', None) if not isinstance(result, dict) else result.get('generated_answer') or result.get('answer')
                docs = getattr(result, 'documents', None) if not isinstance(result, dict) else result.get('documents', [])
                # Pydantic response holds list of dicts; when dict, may hold Document objects
                if docs and not isinstance(docs[0], dict):
                    first_id = getattr(docs[0], 'id', None)
                else:
                    first_id = docs[0].get('id') if docs else None
                assert answer == "Generated answer"
                assert first_id is not None

    @pytest.mark.asyncio
    async def test_unified_pipeline_with_cache(self, mock_semantic_cache):
        """Test unified pipeline with caching enabled."""
        # Test cache hit
        cached_result = {
            "answer": "Cached answer",
            "documents": [
                Document(id="cached_1", content="Cached content", metadata={})
            ]
        }
        mock_semantic_cache.get.return_value = cached_result

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache', return_value=mock_semantic_cache):
            result = await unified_rag_pipeline(
                query="cached query",
                enable_cache=True,
                cache_ttl=3600,
                adaptive_cache=False,
            )

            answer = (
                getattr(result, 'generated_answer', None)
                if not isinstance(result, dict)
                else result.get('generated_answer') or result.get('answer')
            )
            docs = (
                getattr(result, 'documents', None)
                if not isinstance(result, dict)
                else result.get('documents', [])
            )
            first_id = None
            if docs:
                first = docs[0]
                first_id = getattr(first, 'id', None) if not isinstance(first, dict) else first.get('id')
            assert answer == "Cached answer"
            assert first_id == "cached_1"
            mock_semantic_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_search_classification_bypasses_retrieval(self):
        """Classification skip_search should bypass retrieval and still allow generation."""
        from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification

        classification = QueryClassification(
            skip_search=True,
            search_local_db=False,
            search_web=False,
            search_academic=False,
            search_discussions=False,
            standalone_query="hello",
            detected_intent="conversational",
            confidence=0.95,
            reasoning="Greeting",
        )

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.classify_and_reformulate',
            AsyncMock(return_value=classification),
        ):
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                    mock_generator_instance = MagicMock()
                    mock_generator_instance.generate = AsyncMock(return_value={"answer": "Hello there"})
                    mock_generator.return_value = mock_generator_instance

                    result = await unified_rag_pipeline(
                        query="hello",
                        enable_query_classification=True,
                        enable_cache=False,
                        enable_reranking=False,
                        enable_generation=True,
                    )

                    mock_retriever.assert_not_called()
                    answer = (
                        getattr(result, 'generated_answer', None)
                        if not isinstance(result, dict)
                        else result.get('generated_answer') or result.get('answer')
                    )
                    metadata = (
                        getattr(result, 'metadata', None)
                        if not isinstance(result, dict)
                        else result.get('metadata', {})
                    ) or {}
                    assert answer == "Hello there"
                    assert metadata.get("classification_skip_search") is True
                    assert metadata.get("retrieval_bypassed", {}).get("reason") == "classification_skip_search"

    @pytest.mark.asyncio
    async def test_research_loop_documents_are_not_overwritten_by_retrieval(self):
        """Successful research-loop output should bypass standard retrieval."""
        from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification
        from tldw_Server_API.app.core.RAG.rag_service.research_agent import ResearchOutput

        classification = QueryClassification(
            skip_search=False,
            search_local_db=True,
            search_web=True,
            search_academic=False,
            search_discussions=False,
            standalone_query="latest python release notes",
            detected_intent="factual",
            confidence=0.8,
            reasoning="Needs search",
        )
        research_output = ResearchOutput(
            query="latest python release notes",
            standalone_query="latest python release notes",
            all_results=[
                {
                    "id": "research-doc-1",
                    "title": "Python Release Notes",
                    "url": "https://example.com/python-release-notes",
                    "content": "Python 3.x includes important changes.",
                    "source": "web",
                    "score": 0.92,
                }
            ],
            total_iterations=1,
            total_results=1,
            total_duration_sec=0.12,
            final_reasoning="Sufficient evidence collected",
            completed=True,
        )

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.classify_and_reformulate',
            AsyncMock(return_value=classification),
        ):
            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.research_loop',
                AsyncMock(return_value=research_output),
            ) as mock_research_loop:
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                    with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                        mock_generator_instance = MagicMock()
                        mock_generator_instance.generate = AsyncMock(return_value={"answer": "Research-backed answer"})
                        mock_generator.return_value = mock_generator_instance

                        result = await unified_rag_pipeline(
                            query="latest python release notes",
                            enable_query_classification=True,
                            enable_research_loop=True,
                            search_depth_mode="balanced",
                            enable_cache=False,
                            enable_reranking=False,
                            enable_generation=True,
                        )

                        mock_research_loop.assert_called_once()
                        mock_retriever.assert_not_called()
                        docs = (
                            getattr(result, 'documents', None)
                            if not isinstance(result, dict)
                            else result.get('documents', [])
                        ) or []
                        metadata = (
                            getattr(result, 'metadata', None)
                            if not isinstance(result, dict)
                            else result.get('metadata', {})
                        ) or {}
                        assert docs
                        assert docs[0].get("id") == "research-doc-1"
                        assert metadata.get("retrieval_bypassed", {}).get("reason") == "research_loop"
                        assert metadata.get("research", {}).get("total_results") == 1

    @pytest.mark.asyncio
    async def test_classification_local_disabled_prefetches_web_and_bypasses_retriever(self):
        """When local DB route is disabled, classifier can trigger external prefetch without local retrieval."""
        from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification

        classification = QueryClassification(
            skip_search=False,
            search_local_db=False,
            search_web=True,
            search_academic=False,
            search_discussions=False,
            standalone_query="latest python release notes",
            detected_intent="factual",
            confidence=0.8,
            reasoning="Need web route only",
        )

        fake_processed = {
            "results": [
                {
                    "title": "Python release notes",
                    "url": "https://example.com/python-release-notes",
                    "content": "Release notes content",
                }
            ]
        }

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.classify_and_reformulate',
            AsyncMock(return_value=classification),
        ):
            with patch(
                'tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch',
                Mock(return_value={"results": []}),
            ):
                with patch(
                    'tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.process_web_search_results',
                    Mock(return_value=fake_processed),
                ):
                    with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                        result = await unified_rag_pipeline(
                            query="latest python release notes",
                            enable_query_classification=True,
                            enable_generation=False,
                            enable_cache=False,
                            enable_reranking=False,
                            search_mode="hybrid",
                            top_k=5,
                        )

                        mock_retriever.assert_not_called()
                        docs = (
                            getattr(result, 'documents', None)
                            if not isinstance(result, dict)
                            else result.get('documents', [])
                        ) or []
                        metadata = (
                            getattr(result, 'metadata', None)
                            if not isinstance(result, dict)
                            else result.get('metadata', {})
                        ) or {}
                        assert len(docs) == 1
                        assert docs[0].get("id") == "https://example.com/python-release-notes"
                        assert metadata.get("retrieval_bypassed", {}).get("reason") == "classification_external_prefetch"
                        assert metadata.get("classification_external_prefetch", {}).get("document_count") == 1

    @pytest.mark.asyncio
    async def test_research_loop_override_passes_iteration_and_registry_controls(self):
        """Research loop should receive iteration override and registry controls from request params."""
        from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification
        from tldw_Server_API.app.core.RAG.rag_service.research_agent import ResearchOutput

        classification = QueryClassification(
            skip_search=False,
            search_local_db=True,
            search_web=True,
            search_academic=True,
            search_discussions=True,
            standalone_query="compare retrieval strategies",
            detected_intent="comparative",
            confidence=0.8,
            reasoning="Needs broad research",
        )
        research_output = ResearchOutput(
            query="compare retrieval strategies",
            standalone_query="compare retrieval strategies",
            all_results=[],
            total_iterations=1,
            total_results=0,
            total_duration_sec=0.01,
            final_reasoning="Done",
            completed=True,
        )

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.classify_and_reformulate',
            AsyncMock(return_value=classification),
        ):
            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.research_loop',
                AsyncMock(return_value=research_output),
            ) as mock_research_loop:
                result = await unified_rag_pipeline(
                    query="compare retrieval strategies",
                    enable_query_classification=True,
                    enable_research_loop=True,
                    search_depth_mode="balanced",
                    research_max_iterations_balanced=3,
                    discussion_platforms=["reddit"],
                    search_url_scraping=False,
                    enable_generation=False,
                    enable_cache=False,
                    enable_reranking=False,
                )

                kwargs = mock_research_loop.call_args.kwargs
                assert kwargs["max_iterations"] == 3
                assert kwargs["discussion_platforms"] == ["reddit"]
                assert kwargs["enable_url_scraping"] is False
                registry = kwargs.get("registry")
                assert registry is not None
                assert registry.get("discussion_search") is not None
                assert registry.get("scrape_url") is None

                metadata = (
                    getattr(result, 'metadata', None)
                    if not isinstance(result, dict)
                    else result.get('metadata', {})
                ) or {}
                assert metadata.get("research", {}).get("max_iterations_requested") == 3
                assert metadata.get("research", {}).get("url_scraping_enabled") is False

    @pytest.mark.asyncio
    async def test_unified_pipeline_with_expansion(self):
        """Test unified pipeline with query expansion."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.multi_strategy_expansion') as mock_expand:
            mock_expand.return_value = "API Application Programming Interface"

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[
                    Document(id="streaming_doc", content="Context for the complete answer.", metadata={}, source=DataSource.MEDIA_DB, score=0.9)
                ])
                mock_retriever.return_value = mock_retriever_instance

                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                    mock_generator_instance = MagicMock()
                    mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                    mock_generator.return_value = mock_generator_instance

                    result = await unified_rag_pipeline(
                        query="API",
                        enable_expansion=True,
                        expansion_strategies=["acronym", "synonym"]
                    )

                    mock_expand.assert_called_once_with(
                        "API",
                        strategies=["acronym", "synonym"]
                    )
                    # Expanded queries should be recorded
                    expanded = getattr(result, 'expanded_queries', None) if not isinstance(result, dict) else result.get('expanded_queries', [])
                    assert any("Application Programming Interface" in q for q in (expanded or []))

    @pytest.mark.asyncio
    async def test_unified_pipeline_with_reranking(self, sample_documents):
        """Test unified pipeline with reranking enabled."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            # Return documents in one order
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id=doc.id, content=doc.content, metadata=doc.metadata, source=DataSource.MEDIA_DB, score=0.8)
                for doc in sample_documents
            ])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker') as mock_reranker_factory:
                mock_reranker = MagicMock()
                # Return documents in different order
                mock_reranker.rerank = AsyncMock(return_value=[
                    sample_documents[2],
                    sample_documents[0],
                    sample_documents[1]
                ])
                mock_reranker_factory.return_value = mock_reranker

                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                    mock_generator_instance = MagicMock()
                    mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                    mock_generator.return_value = mock_generator_instance

                    result = await unified_rag_pipeline(
                        query="test",
                        enable_reranking=True,
                        reranking_strategy="cross_encoder",
                        rerank_top_k=3
                    )

                    mock_reranker.rerank.assert_called_once()
                    # Documents should be reordered
                    docs = getattr(result, 'documents', None) if not isinstance(result, dict) else result.get('documents', [])
                    assert len(docs) <= 3

    @pytest.mark.asyncio
    async def test_unified_pipeline_with_filters(self):
        """Test unified pipeline with various filters."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SecurityFilter') as mock_security:
            from types import SimpleNamespace
            mock_filter = MagicMock()
            async def _filter_by_sensitivity(docs, max_level=None):
                return [d for d in docs if d.metadata.get("sensitive") != True]
            mock_filter.filter_by_sensitivity = AsyncMock(side_effect=_filter_by_sensitivity)
            mock_security.return_value = mock_filter
            # Ensure SensitivityLevel is present
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SensitivityLevel', SimpleNamespace(PUBLIC=1, INTERNAL=2, CONFIDENTIAL=3, RESTRICTED=4)):

                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                    mock_retriever_instance = MagicMock()
                    mock_retriever_instance.retrieve = AsyncMock(return_value=[
                        Document(id="1", content="Public", metadata={"sensitive": False}, source=DataSource.MEDIA_DB, score=0.9),
                        Document(id="2", content="Secret", metadata={"sensitive": True}, source=DataSource.MEDIA_DB, score=0.85)
                    ])
                    mock_retriever.return_value = mock_retriever_instance

                    with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                        mock_generator_instance = MagicMock()
                        mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                        mock_generator.return_value = mock_generator_instance

                        result = await unified_rag_pipeline(
                            query="test",
                            enable_security_filter=True,
                            sensitivity_level="public"
                        )

                        # Only non-sensitive document should be in results
                        docs = getattr(result, 'documents', None) if not isinstance(result, dict) else result.get('documents', [])
                        assert len(docs) == 1
                        first_id = docs[0]['id'] if docs and isinstance(docs[0], dict) else getattr(docs[0], 'id', None)
                        assert first_id == "1"

    @pytest.mark.asyncio
    async def test_unified_pipeline_security_filter_uses_filter_documents_fallback(self):
        """Security filtering should adapt to the current synchronous filter_documents API."""
        from types import SimpleNamespace

        class _FakePIIMatch:
            def to_dict(self):
                return {"type": "person_name"}

        class _FakePIIDetector:
            def detect_pii(self, _text):
                return [_FakePIIMatch()]

        class _FakeSecurityFilter:
            def __init__(self):
                self.pii_detector = _FakePIIDetector()

            def filter_documents(self, documents, user_id="anonymous", max_sensitivity=None, mask_pii=False):
                assert user_id == "anonymous"
                assert max_sensitivity == 1
                filtered = []
                for doc in documents:
                    if doc.get("metadata", {}).get("sensitive"):
                        continue
                    updated = dict(doc)
                    if mask_pii:
                        updated["content"] = updated["content"].replace("Alice", "[REDACTED]")
                        updated["pii_masked"] = True
                    filtered.append(updated)
                return filtered

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SecurityFilter', _FakeSecurityFilter):
            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SensitivityLevel',
                SimpleNamespace(PUBLIC=1, INTERNAL=2, CONFIDENTIAL=3, RESTRICTED=4),
            ):
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                    mock_retriever_instance = MagicMock()
                    mock_retriever_instance.retrieve = AsyncMock(return_value=[
                        Document(id="1", content="Alice can view this", metadata={"sensitive": False}, source=DataSource.MEDIA_DB, score=0.9),
                        Document(id="2", content="Sensitive", metadata={"sensitive": True}, source=DataSource.MEDIA_DB, score=0.7),
                    ])
                    mock_retriever.return_value = mock_retriever_instance

                    result = await unified_rag_pipeline(
                        query="security test",
                        enable_security_filter=True,
                        detect_pii=True,
                        redact_pii=True,
                        sensitivity_level="public",
                        enable_generation=False,
                        enable_cache=False,
                        enable_reranking=False,
                    )

                    docs = getattr(result, 'documents', None) if not isinstance(result, dict) else result.get('documents', [])
                    errors = getattr(result, 'errors', None) if not isinstance(result, dict) else result.get('errors', [])
                    security_report = getattr(result, 'security_report', None) if not isinstance(result, dict) else result.get('security_report', {})

                    assert len(docs) == 1
                    content = docs[0].get("content") if isinstance(docs[0], dict) else getattr(docs[0], "content", "")
                    assert "[REDACTED]" in content
                    assert not any("Security filter failed" in err for err in (errors or []))
                    assert security_report == {"pii_detected": [[{"type": "person_name"}], [{"type": "person_name"}]]}

    @pytest.mark.asyncio
    async def test_unified_pipeline_security_filter_rejects_invalid_sensitivity_level(self):
        """Security filtering should surface invalid sensitivity levels as validation errors."""
        from types import SimpleNamespace

        class _FakeSecurityFilter:
            def filter_by_sensitivity(self, documents, max_level=None):
                return documents

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SecurityFilter', _FakeSecurityFilter):
            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SensitivityLevel',
                SimpleNamespace(PUBLIC=1, INTERNAL=2, CONFIDENTIAL=3, RESTRICTED=4),
            ):
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                    mock_retriever_instance = MagicMock()
                    mock_retriever_instance.retrieve = AsyncMock(return_value=[
                        Document(id="1", content="Visible", metadata={}, source=DataSource.MEDIA_DB, score=0.9),
                    ])
                    mock_retriever.return_value = mock_retriever_instance

                    result = await unified_rag_pipeline(
                        query="security test",
                        enable_security_filter=True,
                        sensitivity_level="topsecret",  # type: ignore[arg-type]
                        enable_generation=False,
                        enable_cache=False,
                        enable_reranking=False,
                    )

                    errors = getattr(result, 'errors', None) if not isinstance(result, dict) else result.get('errors', [])

                    assert any("Invalid sensitivity_level 'topsecret'" in err for err in (errors or []))

    @pytest.mark.asyncio
    async def test_unified_pipeline_security_filter_awaits_async_detector_fallback(self):
        """Security filtering should await async pii_detector.detect_pii fallbacks."""
        from types import SimpleNamespace

        class _AsyncPIIDetector:
            async def detect_pii(self, _text):
                class _FakePIIMatch:
                    def to_dict(self):
                        return {"type": "email"}

                await asyncio.sleep(0)
                return [_FakePIIMatch()]

        class _FakeSecurityFilter:
            def __init__(self):
                self.pii_detector = _AsyncPIIDetector()

            def filter_by_sensitivity(self, documents, max_level=None):
                return documents

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SecurityFilter', _FakeSecurityFilter):
            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SensitivityLevel',
                SimpleNamespace(PUBLIC=1, INTERNAL=2, CONFIDENTIAL=3, RESTRICTED=4),
            ):
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                    mock_retriever_instance = MagicMock()
                    mock_retriever_instance.retrieve = AsyncMock(return_value=[
                        Document(id="1", content="alice@example.com", metadata={}, source=DataSource.MEDIA_DB, score=0.9),
                    ])
                    mock_retriever.return_value = mock_retriever_instance

                    result = await unified_rag_pipeline(
                        query="security test",
                        enable_security_filter=True,
                        detect_pii=True,
                        enable_generation=False,
                        enable_cache=False,
                        enable_reranking=False,
                    )

                    errors = getattr(result, 'errors', None) if not isinstance(result, dict) else result.get('errors', [])
                    security_report = getattr(result, 'security_report', None) if not isinstance(result, dict) else result.get('security_report', {})

                    assert not any("Security filter failed" in err for err in (errors or []))
                    assert security_report == {"pii_detected": [[{"type": "email"}]]}

    @pytest.mark.asyncio
    async def test_unified_pipeline_security_filter_drops_documents_when_filter_returns_none(self):
        """Security filtering should fail closed when filter_by_sensitivity returns no iterable payload."""
        from types import SimpleNamespace

        class _FakeSecurityFilter:
            def filter_by_sensitivity(self, documents, max_level=None):
                return None

            def redact_pii(self, text):
                return text.replace("Alice", "[REDACTED]")

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SecurityFilter', _FakeSecurityFilter):
            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SensitivityLevel',
                SimpleNamespace(PUBLIC=1, INTERNAL=2, CONFIDENTIAL=3, RESTRICTED=4),
            ):
                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                    mock_retriever_instance = MagicMock()
                    mock_retriever_instance.retrieve = AsyncMock(return_value=[
                        Document(id="1", content="Alice public note", metadata={}, source=DataSource.MEDIA_DB, score=0.9),
                    ])
                    mock_retriever.return_value = mock_retriever_instance

                    result = await unified_rag_pipeline(
                        query="security test",
                        enable_security_filter=True,
                        redact_pii=True,
                        enable_generation=False,
                        enable_cache=False,
                        enable_reranking=False,
                    )

                    docs = getattr(result, 'documents', None) if not isinstance(result, dict) else result.get('documents', [])

                    assert docs == []

    @pytest.mark.asyncio
    async def test_unified_pipeline_with_citations(self, sample_documents):
        """Test unified pipeline with citation generation."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id=doc.id, content=doc.content, metadata=doc.metadata, source=DataSource.MEDIA_DB, score=0.9)
                for doc in sample_documents
            ])
            mock_retriever.return_value = mock_retriever_instance

            # Patch the actual generator used by the pipeline
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.CitationGenerator') as mock_citation:
                from types import SimpleNamespace
                mock_citation_instance = MagicMock()
                # Pipeline expects attributes: academic_citations, chunk_citations, inline_markers, citation_map
                dual_result = SimpleNamespace(
                    academic_citations=["[1] Document 1 - Author (2024)"],
                    chunk_citations=[],
                    inline_markers={"[1]": "1"},
                    citation_map={"1": ["1"]}
                )
                mock_citation_instance.generate_citations = AsyncMock(return_value=dual_result)
                mock_citation.return_value = mock_citation_instance

                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                    mock_generator_instance = MagicMock()
                    mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer with citations"})
                    mock_generator.return_value = mock_generator_instance

                    result = await unified_rag_pipeline(
                        query="test",
                        enable_citations=True,
                        citation_style="academic"
                    )

                    citations = getattr(result, 'citations', None) if not isinstance(result, dict) else result.get('citations', {})
                    assert citations is not None
                    mock_citation_instance.generate_citations.assert_called_once()

    @pytest.mark.asyncio
    async def test_unified_pipeline_chunk_type_filter_citations(self):
        """Test chunk_type filtering and chunk-level citation locations."""
        code_doc = Document(
            id="code-1",
            content="def add(a, b): return a + b",
            metadata={"chunk_type": "code", "title": "Code Doc"},
            source=DataSource.MEDIA_DB,
            score=0.92,
            page_number=2,
            section_title="Code Section",
            source_document_id="doc-123",
        )
        text_doc = Document(
            id="text-1",
            content="Plain text content.",
            metadata={"chunk_type": "text", "title": "Text Doc"},
            source=DataSource.MEDIA_DB,
            score=0.75,
            page_number=1,
            section_title="Intro Section",
            source_document_id="doc-123",
        )

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[code_doc, text_doc])
            mock_retriever.return_value = mock_retriever_instance

            result = await unified_rag_pipeline(
                query="code example",
                chunk_type_filter=["code"],
                enable_citations=True,
                enable_chunk_citations=True,
                enable_generation=False,
                enable_cache=False,
                enable_reranking=False,
            )

            docs = getattr(result, 'documents', None) if not isinstance(result, dict) else result.get('documents', [])
            metadata = getattr(result, 'metadata', None) if not isinstance(result, dict) else result.get('metadata', {})
            chunk_citations = getattr(result, 'chunk_citations', None) if not isinstance(result, dict) else result.get('chunk_citations', [])

            assert len(docs) == 1
            assert docs[0].get("id") == "code-1"
            assert docs[0].get("metadata", {}).get("chunk_type") == "code"
            assert metadata.get("chunk_type_filter_before") == 2
            assert metadata.get("chunk_type_filter_after") == 1
            assert chunk_citations
            location = chunk_citations[0].get("location", "")
            assert "Section: Code Section" in location
            assert "Page 2" in location

    @pytest.mark.asyncio
    async def test_unified_pipeline_table_processing_uses_process_document_tables_fallback(self):
        """Table processing should adapt to the current process_document_tables API."""
        class _FakeTableProcessor:
            def process_document_tables(self, text, serialize_method=None):
                assert serialize_method == "hybrid"
                return (f"{text}\n[Table serialized]", [{"rows": 1, "method": serialize_method}])

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.TableProcessor', _FakeTableProcessor):
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[
                    Document(id="1", content="| h |\n| - |\n| v |", metadata={}, source=DataSource.MEDIA_DB, score=0.9)
                ])
                mock_retriever.return_value = mock_retriever_instance

                result = await unified_rag_pipeline(
                    query="table test",
                    enable_table_processing=True,
                    table_method="hybrid",
                    enable_generation=False,
                    enable_cache=False,
                    enable_reranking=False,
                )

                docs = getattr(result, 'documents', None) if not isinstance(result, dict) else result.get('documents', [])
                errors = getattr(result, 'errors', None) if not isinstance(result, dict) else result.get('errors', [])
                first_doc = docs[0]
                content = first_doc.get("content") if isinstance(first_doc, dict) else getattr(first_doc, "content", "")
                metadata = first_doc.get("metadata") if isinstance(first_doc, dict) else getattr(first_doc, "metadata", {})

                assert "[Table serialized]" in content
                assert metadata.get("table_metadata") == [{"rows": 1, "method": "hybrid"}]
                assert not any("Table processing" in err for err in (errors or []))

    @pytest.mark.asyncio
    async def test_unified_pipeline_highlighting_uses_context_helper(self):
        """Result highlighting should use the quick-wins context helper without pipeline errors."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(
                    id="highlight-doc",
                    content="Evidence handling should stay grounded in the selected source.",
                    metadata={},
                    source=DataSource.MEDIA_DB,
                    score=0.91,
                )
            ])
            mock_retriever.return_value = mock_retriever_instance

            result = await unified_rag_pipeline(
                query="evidence handling selected source",
                highlight_results=True,
                highlight_query_terms=True,
                enable_generation=False,
                enable_cache=False,
                enable_reranking=False,
            )

            docs = getattr(result, 'documents', None) if not isinstance(result, dict) else result.get('documents', [])
            errors = getattr(result, 'errors', None) if not isinstance(result, dict) else result.get('errors', [])
            first_doc = docs[0]
            metadata = first_doc.get("metadata") if isinstance(first_doc, dict) else getattr(first_doc, "metadata", {})

            assert metadata.get("highlighted")
            assert metadata.get("match_count", 0) > 0
            assert not any("highlight_results() takes 1 positional argument" in err for err in (errors or []))

    @pytest.mark.asyncio
    async def test_unified_pipeline_all_features(self):
        """Test unified pipeline with all features enabled."""
        result = await unified_rag_pipeline(
            query="What is RAG?",
            # Retrieval settings
            top_k=20,
            enable_hybrid_search=True,
            bm25_weight=0.3,
            vector_weight=0.7,

            # Expansion settings
            enable_expansion=True,
            expansion_strategies=["synonym", "acronym", "entity"],

            # Cache settings
            enable_cache=True,
            cache_ttl=7200,

            # Reranking settings
            enable_reranking=True,
            reranking_strategy="cross_encoder",
            rerank_top_k=5,

            # Filter settings
            enable_security_filter=True,
            user_clearance_level="confidential",
            enable_date_filter=True,
            date_range={"start": "2024-01-01", "end": "2024-12-31"},

            # Generation settings
            temperature=0.5,
            max_tokens=1000,
            enable_streaming=False,

            # Citation settings
            enable_citations=True,
            citation_style="numeric",

            # Analytics
            enable_analytics=True,
            track_performance=True,

            # Advanced features
            enable_spell_check=True,
            enable_result_highlighting=True,
            enable_cost_tracking=True,
            enable_feedback=True
        )

        # Basic assertions - with all mocking this should at least not error
        qv = getattr(result, 'query', None) if not isinstance(result, dict) else result.get('query')
        assert qv == "What is RAG?"

    @pytest.mark.asyncio
    async def test_unified_pipeline_error_handling(self):
        """Test unified pipeline error handling."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever.side_effect = Exception("Retrieval failed")

            result = await unified_rag_pipeline(
                query="test",
                fallback_on_error=True
            )

            # Should return a fallback result instead of raising
            assert result is not None
            # Accept Pydantic or dict
            if isinstance(result, dict):
                assert "error" in result or result.get("generated_answer") is not None or result.get("answer") is not None
            else:
                # Pydantic success path may carry errors list
                assert hasattr(result, 'errors')

    @pytest.mark.asyncio
    async def test_unified_pipeline_with_metadata(self):
        """Test unified pipeline with custom metadata."""
        custom_metadata = {
            "user_id": "user123",
            "session_id": "session456",
            "request_id": "req789",
            "custom_field": "custom_value"
        }

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
                mock_generator_instance = MagicMock()
                mock_generator_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_generator.return_value = mock_generator_instance

                result = await unified_rag_pipeline(
                    query="test",
                    metadata=custom_metadata
                )

                md = getattr(result, 'metadata', None) if not isinstance(result, dict) else result.get('metadata', {})
                for key, value in custom_metadata.items():
                    assert md.get(key) == value



@pytest.mark.unit
class TestUnifiedPipelineParams:
    """Basic parameter validation through unified entry point."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        result = await unified_rag_pipeline(query="   ")
        errs = getattr(result, 'errors', None) if not isinstance(result, dict) else result.get('errors', [])
        assert errs and len(errs) > 0




@pytest.mark.unit
class TestStreamingSupport:
    """Test streaming support in unified pipeline."""

    @pytest.mark.asyncio
    async def test_streaming_disabled(self):
        """Test pipeline with streaming disabled."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
            mock_generator_instance = MagicMock()
            mock_generator_instance.generate = AsyncMock(return_value={
                "answer": "Complete answer",
                "streaming": False
            })
            mock_generator.return_value = mock_generator_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[])
                mock_retriever.return_value = mock_retriever_instance

                result = await unified_rag_pipeline(
                    query="test",
                    enable_streaming=False
                )

                # Normalize
                ans = getattr(result, 'generated_answer', None) if not isinstance(result, dict) else result.get('generated_answer') or result.get('answer')
                assert ans is None
                md = getattr(result, 'metadata', None) if not isinstance(result, dict) else result.get('metadata', {})
                assert md.get("answer_generation_skipped") == "no_documents"
                mock_generator_instance.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_streaming_enabled(self):
        """Test pipeline with streaming enabled."""
        async def mock_stream_generator():
            """Mock streaming generator."""
            chunks = ["This ", "is ", "a ", "streamed ", "response."]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_generator:
            mock_generator_instance = MagicMock()
            mock_generator_instance.generate_stream = mock_stream_generator
            mock_generator.return_value = mock_generator_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[])
                mock_retriever.return_value = mock_retriever_instance

                result = await unified_rag_pipeline(
                    query="test",
                    enable_streaming=True
                )

                # With streaming, result might be an async generator
                if hasattr(result, '__aiter__'):
                    chunks = []
                    async for chunk in result:
                        chunks.append(chunk)

                    assert len(chunks) > 0
                    full_response = "".join(chunks)
                    assert full_response == "This is a streamed response."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

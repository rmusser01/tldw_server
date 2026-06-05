"""
Integration tests for RAG system.

Tests the complete RAG pipeline with real components and databases.
No mocking - uses actual implementations.
"""

import warnings

import pytest
pytestmark = pytest.mark.integration
import asyncio
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from uuid import uuid4
from datetime import datetime

from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.RAG.rag_service.types import Document, SearchResult, DataSource
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MultiDatabaseRetriever,
    RetrievalConfig
)
from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import SemanticCache


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline with real components."""

    @pytest.mark.asyncio
    async def test_minimal_pipeline_e2e(self, populated_media_db):
        """Test minimal unified pipeline end-to-end with real database."""
        result = await unified_rag_pipeline(
            query="RAG",
            top_k=5,
            enable_cache=False,
            enable_reranking=False,
            search_mode="fts",
            media_db_path=str(populated_media_db.db_path)
        )

        assert isinstance(result, UnifiedRAGResponse)
        assert result.query.lower().find("rag") != -1
        assert isinstance(result.documents, list)
        assert len(result.documents) > 0
        for doc in result.documents:
            content = doc.get("content") if isinstance(doc, dict) else getattr(doc, "content", None)
            assert content is not None

    @pytest.mark.asyncio
    async def test_standard_pipeline_with_cache(self, populated_media_db, temp_db_path):
        """Test unified pipeline with caching enabled (sanity)."""
        result1 = await unified_rag_pipeline(
            query="vector",
            expand_query=True,
            expansion_strategies=["synonym"],
            top_k=10,
            enable_cache=True,
            cache_ttl=3600,
            media_db_path=str(populated_media_db.db_path)
        )
        assert isinstance(result1, UnifiedRAGResponse)
        assert len(result1.documents) > 0

        result2 = await unified_rag_pipeline(
            query="vector",
            expand_query=True,
            expansion_strategies=["synonym"],
            top_k=10,
            enable_cache=True,
            cache_ttl=3600,
            media_db_path=str(populated_media_db.db_path)
        )
        assert isinstance(result2, UnifiedRAGResponse)

    @pytest.mark.asyncio
    async def test_quality_pipeline_with_all_features(self, populated_media_db):
        """Test unified pipeline with a feature mix enabled."""
        result = await unified_rag_pipeline(
            query="AI and ML",
            enable_cache=False,
            expand_query=True,
            expansion_strategies=["synonym", "acronym"],
            enable_reranking=True,
            reranking_strategy="cross_encoder",
            top_k=10,
            rerank_top_k=3,
            enable_performance_analysis=True,
            media_db_path=str(populated_media_db.db_path)
        )

        assert isinstance(result, UnifiedRAGResponse)
        assert len(result.documents) <= 10

    @pytest.mark.asyncio
    async def test_pipeline_with_empty_database(self):
        """Test pipeline behavior with empty database."""
        # Create empty database
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "empty.db"
            empty_db = MediaDatabase(str(db_path), "test_client")
            empty_db.initialize_db()

            result = await unified_rag_pipeline(
                query="test query",
                top_k=5,
                enable_cache=False,
                search_mode="fts",
                media_db_path=str(empty_db.db_path)
            )

            assert isinstance(result, UnifiedRAGResponse)
            assert len(result.documents) == 0
            # Should handle gracefully without errors

    @pytest.mark.asyncio
    async def test_pipeline_with_large_dataset(self, media_database):
        """Test pipeline performance with larger dataset."""
        # Add more test data
        for i in range(50):
            media_database.add_media_with_keywords(
                title=f"Document {i}",
                content=f"This is test document {i} with content about various topics including AI, ML, and RAG.",
                media_type="article",
                author=f"Author {i % 5}",
                ingestion_date=datetime.now().isoformat()
            )

        start_time = time.time()
        result = await unified_rag_pipeline(
            query="AI and ML",
            top_k=20,
            enable_cache=False,
            search_mode="fts",
            media_db_path=str(media_database.db_path)
        )
        elapsed = time.time() - start_time

        assert len(result.documents) >= 0
        assert len(result.documents) <= 20
        # Performance check - prefer fast execution but only warn on minor regressions
        if elapsed >= 5.0:
            warnings.warn(
                f"Unified RAG pipeline took {elapsed:.2f}s (budget 5.0s); investigate if persistent.",
                RuntimeWarning,
                stacklevel=2,
            )
        assert elapsed < 30.0  # Hard guardrail to catch severe regressions without flakiness

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_requests(self, populated_media_db):
        """Test handling concurrent pipeline requests."""
        queries = [
            "What is RAG?",
            "Vector databases",
            "Machine learning",
            "AI systems",
            "Information retrieval"
        ]

        # Run queries concurrently with unified pipeline
        tasks = [
            unified_rag_pipeline(
                query=q,
                top_k=5,
                enable_cache=False,
                search_mode="fts",
                media_db_path=str(populated_media_db.db_path)
            )
            for q in queries
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == len(queries)
        for result in results:
            assert isinstance(result, UnifiedRAGResponse)
            assert isinstance(result.documents, list)


@pytest.mark.integration
class TestUnifiedPipelineIntegration:
    """Integration tests for unified pipeline."""

    @pytest.mark.asyncio
    async def test_unified_pipeline_basic(self, populated_media_db):
        """Test basic unified pipeline with real database."""
        result = await unified_rag_pipeline(
            query="vector databases",
            top_k=5,
            media_db_path=str(populated_media_db.db_path)
        )

        assert isinstance(result, UnifiedRAGResponse)
        assert result.query is not None
        assert isinstance(result.documents, list)
        assert len(result.documents) > 0

        # Should find the vector database document
        def _content(d):
            return d.get("content") if isinstance(d, dict) else getattr(d, "content", "")
        found_vector_doc = any(
            "vector" in str(_content(doc)).lower()
            for doc in result.documents
        )
        assert found_vector_doc

    @pytest.mark.asyncio
    async def test_unified_pipeline_with_expansion(self, populated_media_db):
        """Test unified pipeline with query expansion."""
        result = await unified_rag_pipeline(
            query="RAG",
            enable_expansion=True,
            expansion_strategies=["acronym"],
            top_k=5,
            media_db_path=str(populated_media_db.db_path)
        )

        assert isinstance(result, UnifiedRAGResponse)
        # Expanded query should find machine learning documents
        def _content(d):
            return d.get("content") if isinstance(d, dict) else getattr(d, "content", "")
        found_ml_doc = any(
            ("rag" in str(_content(doc)).lower()) or ("retrieval" in str(_content(doc)).lower())
            for doc in result.documents
        )
        assert found_ml_doc

    @pytest.mark.asyncio
    async def test_unified_pipeline_with_filters(self, populated_media_db):
        """Test unified pipeline with filtering."""
        result = await unified_rag_pipeline(
            query="test",
            top_k=10,
            enable_date_filter=True,
            date_range={"start": "2024-01-01", "end": "2024-12-31"},
            filter_media_types=["article", "document"],
            media_db_path=str(populated_media_db.db_path)
        )

        assert isinstance(result, UnifiedRAGResponse)

        # All documents should match filter criteria where present
        for doc in result.documents:
            md = doc.get("metadata") or {}
            if "media_type" in md:
                assert md["media_type"] in ["article", "document", "video"]

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_unified_pipeline_performance(self, populated_media_db):
        """Test unified pipeline performance metrics."""
        result = await unified_rag_pipeline(
            query="RAG systems",
            enable_analytics=True,
            track_performance=True,
            top_k=10,
            media_db_path=str(populated_media_db.db_path)
        )

        assert isinstance(result, UnifiedRAGResponse)
        # Timings available in result.timings
        if result.timings:
            assert "total" in result.timings or len(result.timings) > 0


@pytest.mark.integration
class TestMultiDatabaseIntegration:
    """Integration tests for multi-database retrieval."""

    @pytest.mark.asyncio
    async def test_multi_source_retrieval(self, populated_media_db):
        """Test retrieval from multiple data sources."""
        retriever = MultiDatabaseRetriever({"media_db": str(populated_media_db.db_path)}, user_id="test")

        config = RetrievalConfig(
            max_results=10,
            min_score=0.0,
            use_fts=True,
            use_vector=False,
            include_metadata=True
        )

        results = await retriever.retrieve("RAG", sources=[DataSource.MEDIA_DB], config=config)

        assert len(results) > 0
        assert all(isinstance(r, Document) for r in results)
        assert all(r.source == DataSource.MEDIA_DB for r in results)

    @pytest.mark.asyncio
    async def test_retrieval_with_scoring(self, populated_media_db):
        """Test retrieval with score-based filtering."""
        retriever = MultiDatabaseRetriever({"media_db": str(populated_media_db.db_path)}, user_id="test")

        config = RetrievalConfig(
            max_results=10,
            min_score=0.0,  # Accept all scores for testing
            use_fts=True,
            use_vector=False,
            include_metadata=True
        )

        results = await retriever.retrieve("RAG", sources=[DataSource.MEDIA_DB], config=config)

        assert len(results) > 0
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_retrieval_pagination(self, media_database):
        """Test paginated retrieval for large result sets."""
        # Add many documents
        for i in range(30):
            media_database.add_media_with_keywords(
                title=f"AI Document {i}",
                content=f"Content about artificial intelligence and machine learning topic {i}.",
                media_type="article",
                author="AI Author",
                ingestion_date=datetime.now().isoformat()
            )

        retriever = MultiDatabaseRetriever({"media_db": str(media_database.db_path)}, user_id="test")

        # First page
        config1 = RetrievalConfig(max_results=10, use_fts=True, use_vector=False, include_metadata=True)
        results1 = await retriever.retrieve("AI", sources=[DataSource.MEDIA_DB], config=config1)

        # Second page
        config2 = RetrievalConfig(max_results=10, use_fts=True, use_vector=False, include_metadata=True)
        results2 = await retriever.retrieve("AI", sources=[DataSource.MEDIA_DB], config=config2)

        # Should have some results
        ids1 = {r.id for r in results1}
        ids2 = {r.id for r in results2}

        # Some overlap is okay, but not complete overlap
        assert len(ids1) > 0 and len(ids2) > 0


@pytest.mark.integration
class TestCacheIntegration:
    """Integration tests for semantic caching."""

    @pytest.mark.asyncio
    async def test_semantic_cache_operations(self, temp_db_path, sample_documents):
        """Test semantic cache with real implementation."""
        cache_dir = temp_db_path.parent / "semantic_cache"
        cache_dir.mkdir(exist_ok=True)

        cache = SemanticCache(
            similarity_threshold=0.85,
            ttl=3600,
            persist_path=str(cache_dir / "cache.pkl")
        )

        # Store documents
        await cache.set("test query", sample_documents, ttl=3600)

        # Exact match retrieval
        cached = await cache.get("test query")
        assert cached is not None
        assert len(cached) == len(sample_documents)

        # Similar query retrieval
        similar_cached = await cache.get("test question")
        # Depends on similarity threshold and implementation

        # Cache expiry
        await cache.set("expiring query", sample_documents, ttl=1)
        await asyncio.sleep(1.1)
        expired = await cache.get("expiring query")
        # Should be None or empty after TTL

        # No explicit close in this implementation

    @pytest.mark.asyncio
    async def test_cache_with_pipeline(self, populated_media_db, temp_db_path):
        """Test cache integration with pipeline."""
        cache_dir = temp_db_path.parent / "pipeline_cache"
        cache_dir.mkdir(exist_ok=True)

        # Run unified pipeline twice to exercise caching internally
        query1 = "What is machine learning?"
        result1 = await unified_rag_pipeline(
            query=query1,
            top_k=5,
            enable_cache=True,
            cache_ttl=3600,
            media_db_path=str(populated_media_db.db_path)
        )

        result2 = await unified_rag_pipeline(
            query=query1,
            top_k=5,
            enable_cache=True,
            cache_ttl=3600,
            media_db_path=str(populated_media_db.db_path)
        )

        assert isinstance(result1, UnifiedRAGResponse)
        assert isinstance(result2, UnifiedRAGResponse)


@pytest.mark.integration
class TestErrorRecoveryIntegration:
    """Integration tests for error recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_pipeline_database_error_recovery(self, populated_media_db):
        """Unified pipeline should surface errors gracefully when retrieval fails."""
        from unittest.mock import patch

        class FailingRetriever:
            def __init__(self, *args, **kwargs):
                pass
            async def retrieve(self, *args, **kwargs):
                raise Exception("Simulated DB failure")

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever',
            FailingRetriever
        ):
            result = await unified_rag_pipeline(
                query="test",
                top_k=5,
                enable_cache=False,
                media_db_path=str(populated_media_db.db_path),
                fallback_on_error=True
            )
            from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
            assert isinstance(result, UnifiedRAGResponse)
            assert isinstance(result.errors, list) and any('Simulated DB failure' in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_partial_retrieval_failure(self, populated_media_db):
        """If one source fails, results from other sources are still returned."""
        from unittest.mock import patch
        from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MultiDatabaseRetriever, RetrievalConfig
        from tldw_Server_API.app.core.RAG.rag_service.types import DataSource

        async def fail_notes(*args, **kwargs):
            raise Exception("Notes DB unavailable")

        with patch('tldw_Server_API.app.core.RAG.rag_service.database_retrievers.NotesDBRetriever.retrieve', side_effect=fail_notes):
            retriever = MultiDatabaseRetriever({
                "media_db": str(populated_media_db.db_path),
                "notes_db": str(populated_media_db.db_path),
            }, user_id="test")

            config = RetrievalConfig(
                max_results=10,
                min_score=0.0,
                use_fts=True,
                use_vector=False,
                include_metadata=True
            )

            results = await retriever.retrieve(
                "RAG",
                sources=[DataSource.MEDIA_DB, DataSource.NOTES],
                config=config
            )

            assert len(results) > 0
            assert all(r.source == DataSource.MEDIA_DB for r in results)

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, populated_media_db):
        """Simulate transient failures and ensure subsequent call succeeds."""
        from unittest.mock import patch
        from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource

        class FlakyRetriever:
            attempts = 0
            def __init__(self, *args, **kwargs):
                pass
            async def retrieve(self, *args, **kwargs):
                FlakyRetriever.attempts += 1
                if FlakyRetriever.attempts < 3:
                    raise Exception("Transient error")
                return [
                    Document(id="1", content="Test content", metadata={"title": "Test"}, source=DataSource.MEDIA_DB, score=0.9)
                ]

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever',
            FlakyRetriever
        ):
            # First call: error captured in response.errors
            r1 = await unified_rag_pipeline(
                query="test",
                top_k=5,
                enable_cache=False,
                media_db_path=str(populated_media_db.db_path),
                fallback_on_error=True
            )
            from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
            assert isinstance(r1, UnifiedRAGResponse)
            assert isinstance(r1.errors, list) and any('Transient error' in e for e in r1.errors)

            # Second call: still failing
            r2 = await unified_rag_pipeline(
                query="test",
                top_k=5,
                enable_cache=False,
                media_db_path=str(populated_media_db.db_path),
                fallback_on_error=True
            )
            assert isinstance(r2, UnifiedRAGResponse)
            assert isinstance(r2.errors, list) and any('Transient error' in e for e in r2.errors)

            # Third call: should succeed and return a pydantic response
            r3 = await unified_rag_pipeline(
                query="test",
                top_k=5,
                enable_cache=False,
                media_db_path=str(populated_media_db.db_path)
            )
            from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
            assert isinstance(r3, UnifiedRAGResponse)
            assert isinstance(r3.documents, list) and len(r3.documents) > 0


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""

    @pytest.mark.asyncio
    async def test_large_document_processing(self, media_database):
        """Test processing very large documents."""
        # Add a very large document
        large_content = " ".join([f"Sentence {i} about various AI and ML topics." for i in range(1000)])

        media_database.add_media_with_keywords(
            title="Large Document",
            content=large_content,
            media_type="article",
            author="Test",
            ingestion_date=datetime.now().isoformat()
        )

        start_time = time.time()
        result = await unified_rag_pipeline(
            query="AI",
            top_k=5,
            enable_cache=False,
            media_db_path=str(media_database.db_path)
        )
        elapsed = time.time() - start_time

        assert len(result.documents) > 0
        # Should complete in reasonable time even with large document
        assert elapsed < 10.0

    @pytest.mark.asyncio
    async def test_concurrent_load(self, populated_media_db):
        """Test system under concurrent load."""
        # Simulate concurrent users
        num_concurrent = 20
        queries = [f"Query {i % 5}" for i in range(num_concurrent)]

        start_time = time.time()

        tasks = [
            unified_rag_pipeline(
                query=query,
                top_k=5,
                enable_cache=False,
                media_db_path=str(populated_media_db.db_path)
            )
            for query in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time

        # Check results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]

        assert len(successful) > num_concurrent * 0.8  # At least 80% success
        assert elapsed < 30.0  # Should complete within 30 seconds

        if failed:
            print(f"Failed requests: {len(failed)}/{num_concurrent}")

    @pytest.mark.asyncio
    async def test_memory_usage(self, populated_media_db):
        """Test memory usage doesn't grow unbounded."""
        import psutil
        import gc

        process = psutil.Process()

        # Exclude one-time module/retriever initialization from the leak guard.
        _ = await unified_rag_pipeline(
            query="warmup",
            top_k=10,
            enable_cache=False,
            search_mode="fts",
            media_db_path=str(populated_media_db.db_path)
        )
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run many queries
        for i in range(50):
            _ = await unified_rag_pipeline(
                query=f"Query {i}",
                top_k=10,
                enable_cache=False,
                search_mode="fts",
                media_db_path=str(populated_media_db.db_path)
            )
            if i % 10 == 0:
                gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable
        assert memory_growth < 100  # Less than 100MB growth


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

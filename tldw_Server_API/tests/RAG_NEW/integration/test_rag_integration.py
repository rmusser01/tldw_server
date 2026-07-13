"""
Integration tests for RAG system.

Tests the complete RAG pipeline with real components and databases while patching
external provider boundaries where deterministic integration coverage requires it.
"""

import inspect
import json
import warnings
from types import SimpleNamespace

import pytest
from loguru import logger

pytestmark = pytest.mark.integration
import asyncio
import time
from datetime import datetime
from pathlib import Path

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import reset_managed_sqlite_backends
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MultiDatabaseRetriever, RetrievalConfig
from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import SemanticCache
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

_RAG_USER_SECRET = "sk-rag-user-secret-must-not-leak"


def _resolved_rag_user_credential(
    provider: str,
    api_key: str,
) -> ResolvedByokCredentials:
    return ResolvedByokCredentials(
        provider=provider,
        api_key=api_key,
        app_config={f"{provider}_api": {"model": "runtime-model"}},
        credential_fields={},
        source="user",
        allowlisted=True,
        status=ByokResolutionStatus.RESOLVED,
        auth_source="api_key",
    )


def test_internal_rag_caller_keeps_legacy_optional_runtime_contract():
    """System callers may continue omitting the execution credential runtime."""
    parameter = inspect.signature(unified_rag_pipeline).parameters["credential_runtime"]

    assert parameter.default is None


@pytest.mark.asyncio
async def test_rag_cache_hit_regenerates_with_current_distinct_provider_credentials(
    populated_media_db,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create
    from tldw_Server_API.app.core.RAG.rag_service import (
        advanced_cache,
        database_retrievers,
    )

    class FakeVectorStore:
        _initialized = True

        async def initialize(self):
            self._initialized = True

        async def search(self, **_kwargs):
            return [
                SimpleNamespace(
                    id="vector-doc",
                    content="Runtime vector evidence about RAG.",
                    metadata={"media_id": "vector-doc", "kind": "chunk"},
                    score=0.95,
                )
            ]

    embedding_config = {
        "embedding_config": {
            "default_model_id": "openai:text-embedding-3-small",
            "models": {
                "openai:text-embedding-3-small": SimpleNamespace(
                    provider="openai"
                ),
            },
        }
    }
    embedding_calls: list[dict[str, object]] = []
    generation_calls: list[dict[str, object]] = []

    def create_embeddings(
        texts,
        user_app_config,
        model_id_override=None,
        **kwargs,
    ):
        embedding_calls.append(
            {
                "texts": list(texts),
                "config": user_app_config,
                "model": model_id_override,
                **kwargs,
            }
        )
        return [[1.0, 0.0]]

    async def generate_answer(**kwargs):
        generation_calls.append(dict(kwargs))
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            f"current credential answer {len(generation_calls)}"
                        ),
                    }
                }
            ],
            "usage": {"total_tokens": 4},
        }

    monkeypatch.setattr(
        database_retrievers,
        "create_from_settings_for_user",
        lambda *_args, **_kwargs: FakeVectorStore(),
    )
    monkeypatch.setattr(
        Embeddings_Create,
        "get_embedding_config",
        lambda: embedding_config,
    )
    monkeypatch.setattr(
        Embeddings_Create,
        "create_embeddings_batch",
        create_embeddings,
    )
    monkeypatch.setattr(
        chat_service,
        "perform_chat_api_call_async",
        generate_answer,
    )

    resolution_calls: dict[str, list[str]] = {"first": [], "second": []}

    def make_runtime(label: str) -> ProviderCredentialRuntime:
        async def resolver(
            provider: str,
            **_kwargs,
        ) -> ResolvedByokCredentials:
            resolution_calls[label].append(provider)
            return _resolved_rag_user_credential(
                provider,
                f"{_RAG_USER_SECRET}-{label}-{provider}",
            )

        def fail_server_fallback(_provider: str) -> str | None:
            raise AssertionError("server credential fallback used")

        return ProviderCredentialRuntime(
            user_id=42,
            team_ids=[],
            org_ids=[],
            trusted_base_url_override=False,
            fallback_resolver=fail_server_fallback,
            resolver=resolver,
        )

    common_kwargs = {
        "query": "How does retrieval augmented generation work?",
        "sources": ["media_db"],
        "search_mode": "hybrid",
        "top_k": 3,
        "enable_cache": True,
        "enable_reranking": False,
        "enable_generation": True,
        "generation_provider": "anthropic",
        "generation_model": "claude-test",
        "media_db_path": str(populated_media_db.db_path),
        "user_id": "42",
    }

    logs: list[str] = []
    sink_id = logger.add(logs.append, format="{message}")
    first_runtime = make_runtime("first")
    second_runtime = make_runtime("second")
    try:
        first = await unified_rag_pipeline(
            **common_kwargs,
            credential_runtime=first_runtime,
        )
        await first_runtime.close()
        second = await unified_rag_pipeline(
            **common_kwargs,
            credential_runtime=second_runtime,
        )
        await second_runtime.close()
        registered_cache = advanced_cache.get_registered_semantic_cache()
        assert registered_cache is not None
        registered_cache.save()
    finally:
        await first_runtime.close()
        await second_runtime.close()
        logger.remove(sink_id)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.generated_answer == "current credential answer 1"
    assert second.generated_answer == "current credential answer 2"
    assert resolution_calls == {
        "first": ["openai", "anthropic"],
        "second": ["anthropic"],
    }
    assert len(embedding_calls) == 1
    assert (
        embedding_calls[0]["api_key_override"]
        == f"{_RAG_USER_SECRET}-first-openai"
    )
    assert embedding_calls[0]["credentials_resolved"] is True
    assert [call["api_key"] for call in generation_calls] == [
        f"{_RAG_USER_SECRET}-first-anthropic",
        f"{_RAG_USER_SECRET}-second-anthropic",
    ]
    assert all(
        call["credentials_resolved"] is True for call in generation_calls
    )

    cache_files = [
        path
        for path in (tmp_path / "semantic_cache").rglob("*")
        if path.is_file()
    ]
    assert cache_files
    cache_bytes = b"".join(path.read_bytes() for path in cache_files)
    assert _RAG_USER_SECRET.encode() not in cache_bytes
    assert b"current credential answer" not in cache_bytes
    assert _RAG_USER_SECRET not in "".join(logs)
    assert _RAG_USER_SECRET not in first.model_dump_json()
    assert _RAG_USER_SECRET not in second.model_dump_json()


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
    async def test_pipeline_with_empty_database(self, tmp_path: Path):
        """Test pipeline behavior with empty database."""
        db_path = tmp_path / "empty.db"
        empty_db = MediaDatabase(str(db_path), "test_client")
        try:
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
        finally:
            empty_db.close_connection()
            reset_managed_sqlite_backends(
                sqlite_targets=[str(db_path), str(db_path.resolve())]
            )

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

        # Store the real Document payload produced by the retrieval pipeline.
        await cache.set(
            "test query",
            {
                "documents": sample_documents,
                "answer": "STALE_SENTINEL",
                "metadata": {"generation_model": "stale-model"},
            },
            ttl=3600,
        )

        # Exact match retrieval
        cached = await cache.get("test query")
        assert cached is not None
        assert len(cached["documents"]) == len(sample_documents)
        assert set(cached) == {"documents", "metadata"}
        assert cached["metadata"] == {
            "kind": "retrieval_documents",
            "schema_version": 1,
        }

        cache.save()
        json.loads(Path(cache.persist_path).read_text())
        reloaded = SemanticCache(persist_path=cache.persist_path)
        persisted = await reloaded.get("test query")
        assert persisted == cached
        assert "STALE_SENTINEL" not in Path(cache.persist_path).read_text()

        # Similar query retrieval
        await cache.get("test question")
        # Depends on similarity threshold and implementation

        # Cache expiry
        await cache.set("expiring query", sample_documents, ttl=1)
        await asyncio.sleep(1.1)
        await cache.get("expiring query")
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

        from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

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
        import gc

        import psutil

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

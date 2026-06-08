"""
Unit tests for the unified RAG pipeline - THE ONLY PIPELINE IN USE.

Focuses exclusively on testing the unified_rag_pipeline function
and its actual dependencies.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio
import sqlite3
from typing import Dict, List, Any
from datetime import datetime
import types

import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.core.RAG.rag_service.types import Document, SearchResult, DataSource


@pytest.mark.unit
class TestUnifiedPipelineCore:
    """Core tests for the unified pipeline - the main entry point."""

    @pytest.mark.asyncio
    async def test_minimal_query_execution(self):
        """Test the most basic query execution with minimal parameters."""
        # This is what most users will actually use
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id="1", content="RAG is a retrieval technique", metadata={}, source=DataSource.MEDIA_DB, score=0.9)
            ])
            mock_retriever.return_value = mock_retriever_instance

            # Mock answer generation since it requires LLM
            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={
                    "answer": "RAG combines retrieval with generation.",
                    "confidence": 0.85
                })
                mock_gen.return_value = mock_gen_instance

                # This is the actual function users call
                result = await unified_rag_pipeline(
                    query="What is RAG?",
                    top_k=5
                )

                assert result is not None
                assert result.query == "What is RAG?"
                assert result.generated_answer is not None
                assert isinstance(result.documents, list)
                assert len(result.documents) > 0

    @pytest.mark.asyncio
    async def test_common_user_parameters(self):
        """Test with parameters commonly used by users."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_gen.return_value = mock_gen_instance

                result = await unified_rag_pipeline(
                    query="How does machine learning work?",
                    top_k=10,
                    temperature=0.7,
                    max_tokens=500,
                    enable_cache=True,
                    enable_reranking=True,
                    rerank_top_k=5
                )

                assert result is not None
                assert result.query == "How does machine learning work?"
                # Should have attempted retrieval
                mock_retriever_instance.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_with_media_database(self, mock_media_database):
        """Test with actual media database parameter."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id="1", content="Test content", metadata={}, source=DataSource.MEDIA_DB, score=0.8)
            ])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_gen.return_value = mock_gen_instance

                result = await unified_rag_pipeline(
                    query="test query",
                    top_k=5,
                    media_db=mock_media_database
                )

                # Should pass media_db to retriever
                mock_retriever.assert_called_once()
                call_kwargs = mock_retriever.call_args[1]
                assert call_kwargs.get('media_db') == mock_media_database

    @pytest.mark.asyncio
    async def test_error_handling_with_fallback(self):
        """Test error handling returns graceful fallback."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            # Simulate retrieval failure
            mock_retriever.side_effect = Exception("Database connection failed")

            result = await unified_rag_pipeline(
                query="test query",
                fallback_on_error=True
            )

            # Should return a result even with error
            assert result is not None
            # Should indicate error or provide fallback answer
            assert len(getattr(result, 'errors', []) or []) > 0 or (result.generated_answer is not None)

    @pytest.mark.asyncio
    async def test_empty_retrieval_results(self):
        """Test behavior when no documents are retrieved."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            # No documents found
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={
                    "answer": "I couldn't find relevant information to answer your question.",
                    "confidence": 0.2
                })
                mock_gen.return_value = mock_gen_instance

                result = await unified_rag_pipeline(
                    query="obscure query with no matches",
                    top_k=10
                )

                assert result is not None
                assert result.generated_answer is None
                assert result.metadata.get("answer_generation_skipped") == "no_documents"
                assert len(result.documents) == 0
                mock_gen_instance.generate.assert_not_awaited()


@pytest.mark.unit
class TestUnifiedPipelineFeatures:
    """Test specific features users actually use."""

    @pytest.mark.asyncio
    async def test_query_expansion_feature(self):
        """Test query expansion when explicitly enabled."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.multi_strategy_expansion') as mock_expand:
            mock_expand.return_value = "API Application Programming Interface"

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
                mock_retriever_instance = MagicMock()
                mock_retriever_instance.retrieve = AsyncMock(return_value=[])
                mock_retriever.return_value = mock_retriever_instance

                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                    mock_gen_instance = MagicMock()
                    mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer about API"})
                    mock_gen.return_value = mock_gen_instance

                    result = await unified_rag_pipeline(
                        query="API",
                        enable_expansion=True,
                        expansion_strategies=["acronym"]
                    )

                    # Should have expanded the query
                    mock_expand.assert_called_once_with("API", strategies=["acronym"])

    @pytest.mark.asyncio
    async def test_caching_feature(self, mock_semantic_cache):
        """Test caching when enabled by user."""
        # Setup cache hit scenario
        cached_result = {
            "answer": "Cached answer",
            "documents": [
                Document(id="cached_1", content="Cached content", metadata={})
            ],
            "cached": True
        }
        mock_semantic_cache.get.return_value = cached_result

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache', return_value=mock_semantic_cache):
            result = await unified_rag_pipeline(
                query="cached query",
                enable_cache=True,
                cache_ttl=3600,
                adaptive_cache=False,
            )

            # Should return cached result
            assert result.cache_hit is True
            assert result.generated_answer == "Cached answer"
            assert len(result.documents) == 1
            assert result.documents[0]["content"] == "Cached content"
            mock_semantic_cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_adaptive_cache_selected_when_enabled(self):
        """Adaptive cache should be used when enabled and available."""
        cached_result = {
            "answer": "Adaptive cached answer",
            "documents": [Document(id="cached_1", content="Cached content", metadata={})],
            "cached": True,
        }
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_result

        seen = {}

        def _fake_shared_cache(cache_cls, **_kwargs):

            seen["cache_cls"] = cache_cls
            return mock_cache

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache', side_effect=_fake_shared_cache), \
                patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AdaptiveCache') as mock_adaptive, \
                patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            result = await unified_rag_pipeline(
                query="cached query",
                enable_cache=True,
                adaptive_cache=True,
                enable_generation=False,
            )

        assert seen.get("cache_cls") is mock_adaptive
        mock_retriever.assert_not_called()
        assert result.cache_hit is True
        assert result.generated_answer == "Adaptive cached answer"

    @pytest.mark.asyncio
    async def test_cache_hit_with_legacy_list_payload(self, mock_semantic_cache):
        """Ensure legacy cache entries storing raw document lists still work."""
        legacy_doc = Document(
            id="legacy_1",
            content="Legacy cached content",
            metadata={"source": "media_db"},
            source=DataSource.MEDIA_DB,
            score=0.95,
        )
        mock_semantic_cache.get.return_value = [legacy_doc]
        mock_semantic_cache.find_similar.return_value = None

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache', return_value=mock_semantic_cache), \
                patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            result = await unified_rag_pipeline(
                query="legacy cache hit",
                enable_cache=True,
                enable_generation=False,
                adaptive_cache=False,
            )

        mock_retriever.assert_not_called()
        assert result.cache_hit is True
        assert result.documents
        assert result.documents[0]["content"] == "Legacy cached content"

    @pytest.mark.asyncio
    async def test_cache_storage_persists_documents_and_answer(self):
        """Verify cache set receives structured payload with documents and answer."""

        class RecordingCache:
            def __init__(self, *_, **__):
                self.set_calls = []

            def get(self, _query):

                return None

            def find_similar(self, _query):

                return None

            def set(self, query, value, ttl=None):

                self.set_calls.append((query, value, ttl))

        recording_cache = RecordingCache()
        retrieved_doc = Document(
            id="doc-cache",
            content="Content to cache",
            metadata={"source": "media_db"},
            source=DataSource.MEDIA_DB,
            score=0.88,
        )

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache', return_value=recording_cache), \
             patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_instance = MagicMock()
            mock_instance.retrieve = AsyncMock(return_value=[retrieved_doc])
            mock_retriever.return_value = mock_instance

            await unified_rag_pipeline(
                query="store cache payload",
                enable_cache=True,
                enable_generation=False,
                adaptive_cache=False,
            )

        assert recording_cache.set_calls, "Cache set should have been invoked"
        stored_query, payload, ttl = recording_cache.set_calls[0]
        assert stored_query == "store cache payload"
        assert ttl == 3600
        assert isinstance(payload, dict)
        assert payload.get("cached") is True
        assert payload.get("documents")
        assert payload["documents"][0].id == "doc-cache"

    @pytest.mark.asyncio
    async def test_claim_retrieval_uses_request_scoped_chacha_db(self):
        """Claims pipeline must reuse the caller's ChaCha DB instance."""

        class NullCache:
            def __init__(self, *_, **__):
                pass

            def get(self, _query):

                return None

            def find_similar(self, _query):

                return None

            def set(self, _query, _value, _ttl=None):

                return None

        base_doc = Document(
            id="media-claim",
            content="Evidence content",
            metadata={"media_id": 42, "source": "media_db"},
            source=DataSource.MEDIA_DB,
            score=0.9,
        )
        chacha_db_instance = object()

        class StubMediaRetriever:
            def __init__(self):
                self.retrieve_calls = []
                self.hybrid_calls = []

            async def retrieve(self, query, **kwargs):
                self.retrieve_calls.append((query, kwargs))
                return [base_doc]

            async def retrieve_hybrid(self, query, alpha=0.7, **kwargs):
                self.hybrid_calls.append((query, alpha, kwargs))
                return [base_doc]

        class StubMultiDatabaseRetriever:
            instances: List["StubMultiDatabaseRetriever"] = []

            def __init__(self, db_paths, user_id="0", *, media_db=None, chacha_db=None):

                self.db_paths = db_paths
                self.user_id = user_id
                self.media_db = media_db
                self.chacha_db = chacha_db
                self.retrievers = {DataSource.MEDIA_DB: StubMediaRetriever()}
                self.retrieve_invocations: List[Any] = []
                StubMultiDatabaseRetriever.instances.append(self)

            async def retrieve(self, query, **kwargs):
                self.retrieve_invocations.append((query, kwargs))
                return [base_doc]

            async def retrieve_hybrid(self, query, alpha=0.7, **kwargs):
                return await self.retrievers[DataSource.MEDIA_DB].retrieve_hybrid(query, alpha, **kwargs)

            def close(self):

                return None

        class StubClaimsEngine:
            def __init__(self, _analyze):
                self.run_calls: List[Dict[str, Any]] = []

            async def run(self, **kwargs):
                self.run_calls.append(kwargs)
                retrieve_fn = kwargs.get("retrieve_fn")
                if retrieve_fn:
                    await retrieve_fn("claim text")
                return {"claims": [], "summary": {}}

        StubMultiDatabaseRetriever.instances = []
        dummy_sgl = types.SimpleNamespace(analyze=lambda *_, **__: {})

        with (
            patch.dict('sys.modules', {
                'tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib': dummy_sgl,
            }),
            patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache',
                  return_value=NullCache()),
            patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever',
                  StubMultiDatabaseRetriever),
            patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.ClaimsEngine', StubClaimsEngine),
            patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_answer_gen,
        ):
            answer_gen_instance = MagicMock()
            answer_gen_instance.generate = AsyncMock(return_value={"answer": "Generated answer"})
            mock_answer_gen.return_value = answer_gen_instance

            result = await unified_rag_pipeline(
                query="Explain the claim",
                enable_claims=True,
                enable_cache=False,
                enable_generation=True,
                chacha_db=chacha_db_instance,
                media_db_path=None,
                notes_db_path=None,
                character_db_path=None,
            )

        assert result.generated_answer == "Generated answer"
        instances = StubMultiDatabaseRetriever.instances
        assert len(instances) >= 2
        assert all(inst.chacha_db is chacha_db_instance for inst in instances)
        per_claim_instance = instances[-1]
        media_retriever = per_claim_instance.retrievers[DataSource.MEDIA_DB]
        assert media_retriever.hybrid_calls or media_retriever.retrieve_calls

    @pytest.mark.asyncio
    async def test_preextracted_claims_use_managed_media_database(self):
        """Pre-extracted claims verification must use the managed DB helper."""

        class NullCache:
            def __init__(self, *_, **__):
                pass

            def get(self, _query):
                return None

            def find_similar(self, _query):
                return None

            def set(self, _query, _value, _ttl=None):
                return None

        base_doc = Document(
            id="claim-doc",
            content="Stored claim evidence",
            metadata={"media_id": 7, "source": "media_db"},
            source=DataSource.MEDIA_DB,
            score=0.92,
        )
        managed_calls: List[Dict[str, Any]] = []
        verify_calls: List[Dict[str, Any]] = []

        class StubRetriever:
            async def retrieve(self, query, **kwargs):
                return [base_doc]

        class StubManagedDb:
            def execute_query(self, sql, params):
                assert "SELECT claim_text FROM Claims" in sql
                assert params == (7, 25)
                return types.SimpleNamespace(fetchall=lambda: [("stored claim text",)])

        class StubVerification:
            def __init__(self, claim):
                self.claim = claim
                self.label = "supported"
                self.confidence = 0.91
                self.evidence = [
                    types.SimpleNamespace(
                        doc_id="claim-doc",
                        snippet="Stored claim evidence",
                        score=0.88,
                    )
                ]
                self.citations = ["claim-doc"]
                self.rationale = "Matched stored claim."

        class StubClaimsEngine:
            def __init__(self, _analyze):
                self.verifier = types.SimpleNamespace(verify=self._verify)

            async def _verify(self, **kwargs):
                verify_calls.append(kwargs)
                return StubVerification(kwargs["claim"])

            async def run(self, **kwargs):
                raise AssertionError("engine.run should not execute when pre-extracted claims exist")

        class StubClaim:
            def __init__(self, id, text, span=None):
                self.id = id
                self.text = text
                self.span = span

        def _legacy_media_db(*args, **kwargs):
            raise AssertionError("legacy raw MediaDatabase should not be used")

        class _ManagedDbContext:
            def __init__(self, **kwargs):
                managed_calls.append(kwargs)

            def __enter__(self):
                return StubManagedDb()

            def __exit__(self, exc_type, exc, tb):
                return False

        def _managed_media_database(*, client_id, db_path=None, initialize=True, suppress_init_exceptions=(), suppress_close_exceptions=(), **kwargs):
            return _ManagedDbContext(
                client_id=client_id,
                db_path=db_path,
                initialize=initialize,
                suppress_init_exceptions=suppress_init_exceptions,
                suppress_close_exceptions=suppress_close_exceptions,
                extra=kwargs,
            )

        dummy_sgl = types.SimpleNamespace(analyze=lambda *_, **__: {})
        dummy_config = types.SimpleNamespace(settings={"SERVER_CLIENT_ID": "SERVER-TEST"})
        dummy_claims_engine_module = types.SimpleNamespace(Claim=StubClaim)

        with (
            patch.dict(
                "sys.modules",
                {
                    "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib": dummy_sgl,
                    "tldw_Server_API.app.core.config": dummy_config,
                    "tldw_Server_API.app.core.Claims_Extraction.claims_engine": dummy_claims_engine_module,
                },
            ),
            patch.object(up, "managed_media_database", _managed_media_database, create=True),
            patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache", return_value=NullCache()),
            patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever") as mock_retriever,
            patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.ClaimsEngine", StubClaimsEngine),
            patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator") as mock_answer_gen,
        ):
            mock_retriever.return_value = StubRetriever()
            answer_gen_instance = MagicMock()
            answer_gen_instance.generate = AsyncMock(return_value={"answer": "Generated answer"})
            mock_answer_gen.return_value = answer_gen_instance

            result = await unified_rag_pipeline(
                query="Verify stored claim",
                enable_claims=True,
                enable_cache=False,
                enable_generation=True,
                media_db_path="/tmp/media.db",
                claims_max=25,
            )

        assert managed_calls == [
            {
                "client_id": "SERVER-TEST",
                "db_path": "/tmp/media.db",
                "initialize": False,
                "suppress_init_exceptions": (),
                "suppress_close_exceptions": (
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    sqlite3.Error,
                ),
                "extra": {},
            }
        ]
        assert len(verify_calls) == 1
        assert verify_calls[0]["claim"].text == "stored claim text"
        assert result.metadata["claims"][0]["text"] == "stored claim text"
        assert result.metadata["factuality"]["supported"] == 1

    @pytest.mark.asyncio
    async def test_reranking_feature(self):
        """Test reranking when enabled."""
        initial_docs = [
            Document(id="1", content="Less relevant", metadata={"initial_score": 0.7}),
            Document(id="2", content="Most relevant", metadata={"initial_score": 0.8}),
            Document(id="3", content="Somewhat relevant", metadata={"initial_score": 0.75})
        ]

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id=doc.id, content=doc.content, metadata=doc.metadata, source=DataSource.MEDIA_DB, score=doc.metadata["initial_score"])
                for doc in initial_docs
            ])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker') as mock_reranker_factory:
                mock_reranker = MagicMock()
                # Reranker changes order
                mock_reranker.rerank = AsyncMock(return_value=[
                    initial_docs[1],  # Most relevant now first
                    initial_docs[2],  # Somewhat relevant second
                ])
                mock_reranker_factory.return_value = mock_reranker

                with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                    mock_gen_instance = MagicMock()
                    mock_gen_instance.generate = AsyncMock(return_value={"answer": "Reranked answer"})
                    mock_gen.return_value = mock_gen_instance

                    result = await unified_rag_pipeline(
                        query="test",
                        enable_reranking=True,
                        rerank_top_k=2
                    )

                    # Should have reranked
                    mock_reranker.rerank.assert_called_once()
                    # Should only return top 2 after reranking
                    assert len(result.documents) == 2
                    # Most relevant should be first
                    assert result.documents[0]["id"] == "2"

    @pytest.mark.asyncio
    async def test_filtering_features(self):
        """Test document filtering options."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id="1", content="Recent document", metadata={"date": "2024-06-01", "media_type": "article"}, source=DataSource.MEDIA_DB, score=0.9),
                Document(id="2", content="Old document", metadata={"date": "2023-01-01", "media_type": "video"}, source=DataSource.MEDIA_DB, score=0.85)
            ])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Filtered answer"})
                mock_gen.return_value = mock_gen_instance

                result = await unified_rag_pipeline(
                    query="test",
                    enable_date_filter=True,
                    date_range={"start": "2024-01-01", "end": "2024-12-31"},
                    filter_media_types=["article"]
                )

                # Retriever should be called with filters
                mock_retriever_instance.retrieve.assert_called_once()


@pytest.mark.unit
class TestUnifiedPipelineRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_chatbot_query(self):
        """Test typical chatbot query pattern."""
        # Simulating a chatbot asking about a topic
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id="1", content="Python is a high-level programming language known for its simplicity.", metadata={"source": "tutorial", "author": "Expert"}, source=DataSource.MEDIA_DB, score=0.95)
            ])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={
                    "answer": "Python is a high-level programming language known for its simplicity and readability.",
                    "confidence": 0.9
                })
                mock_gen.return_value = mock_gen_instance

                result = await unified_rag_pipeline(
                    query="Tell me about Python programming",
                    top_k=5,
                    temperature=0.7,
                    max_tokens=200
                )

                assert result is not None
                assert result.generated_answer is not None
                assert "Python" in (result.generated_answer or "")
                assert result.documents[0]["content"] is not None

    @pytest.mark.asyncio
    async def test_research_query(self):
        """Test research/analysis query pattern."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            # Multiple relevant documents for research
            mock_retriever_instance.retrieve = AsyncMock(return_value=[
                Document(id=f"doc_{i}", content=f"Research finding {i}", metadata={"citation": f"Source {i}"}, source=DataSource.MEDIA_DB, score=0.9 - i*0.05)
                for i in range(5)
            ])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={
                    "answer": "Based on multiple sources...",
                    "citations": ["Source 1", "Source 2", "Source 3"]
                })
                mock_gen.return_value = mock_gen_instance

                result = await unified_rag_pipeline(
                    query="What are the latest findings on climate change?",
                    top_k=20,  # Want more sources for research
                    enable_citations=True,
                    temperature=0.3  # Lower temperature for factual accuracy
                )

                assert result is not None
                assert len(result.documents) > 1  # Multiple sources

    @pytest.mark.asyncio
    async def test_api_endpoint_usage(self):
        """Test usage pattern from API endpoint."""
        # Simulate parameters coming from API request
        api_params = {
            "query": "How to implement RAG?",
            "top_k": 10,
            "temperature": 0.5,
            "enable_cache": True,
            "metadata": {
                "user_id": "user123",
                "session_id": "session456",
                "request_id": "req789"
            }
        }

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "RAG implementation guide..."})
                mock_gen.return_value = mock_gen_instance

                result = await unified_rag_pipeline(**api_params)

                assert result is not None
                assert result.metadata["user_id"] == "user123"

    @pytest.mark.asyncio
    async def test_streaming_response(self):
        """Test streaming response for real-time applications."""
        async def mock_stream():
            """Simulate streaming response."""
            chunks = ["RAG ", "is ", "a ", "powerful ", "technique."]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate_stream = mock_stream
                mock_gen.return_value = mock_gen_instance

                result = await unified_rag_pipeline(
                    query="What is RAG?",
                    enable_streaming=True
                )

                # Result should be streamable (only if streaming is supported)
                if hasattr(result, '__aiter__'):
                    chunks = []
                    async for chunk in result:
                        chunks.append(chunk)
                    assert len(chunks) == 5
                    assert "".join(chunks) == "RAG is a powerful technique."


@pytest.mark.unit
class TestUnifiedPipelineValidation:
    """Test input validation and parameter handling."""

    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty or whitespace queries."""
        for invalid_query in ["", "   ", "\n\t"]:
            result = await unified_rag_pipeline(
                query=invalid_query,
                top_k=5
            )

            # Should handle gracefully
            assert result is not None
            # Should indicate invalid query
            assert (result.generated_answer or "").lower().find("invalid") >= 0 or len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_parameter_bounds(self):
        """Test parameter boundary conditions."""
        # Test with extreme but valid values
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_gen.return_value = mock_gen_instance

                # Very high top_k
                result = await unified_rag_pipeline(
                    query="test",
                    top_k=1000
                )
                assert result is not None

                # Very low temperature
                result = await unified_rag_pipeline(
                    query="test",
                    temperature=0.0
                )
                assert result is not None

                # Very high temperature
                result = await unified_rag_pipeline(
                    query="test",
                    temperature=2.0
                )
                assert result is not None

    @pytest.mark.asyncio
    async def test_conflicting_parameters(self):
        """Test handling of conflicting parameters."""
        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
            mock_retriever_instance = MagicMock()
            mock_retriever_instance.retrieve = AsyncMock(return_value=[])
            mock_retriever.return_value = mock_retriever_instance

            with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator') as mock_gen:
                mock_gen_instance = MagicMock()
                mock_gen_instance.generate = AsyncMock(return_value={"answer": "Answer"})
                mock_gen.return_value = mock_gen_instance

                # Rerank_top_k > top_k (conflicting)
                result = await unified_rag_pipeline(
                    query="test",
                    top_k=5,
                    enable_reranking=True,
                    rerank_top_k=10  # Higher than top_k
                )

                # Should handle gracefully, likely cap rerank_top_k to top_k
                assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for the unified RAG pipeline - THE ONLY PIPELINE IN USE.

Focuses exclusively on testing the unified_rag_pipeline function
and its actual dependencies.
"""

import asyncio
import copy
import hashlib
import json
import sqlite3
import time
import types
from contextlib import nullcontext
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up
from tldw_Server_API.app.core.DB_Management.scope_context import (
    ScopeContext,
    get_scope,
    scoped_context,
)
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import SemanticCache
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline


class _MemoryRetrievalCache:
    """Small namespace-local cache used to exercise the shared-cache boundary."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, query: str) -> Any:
        return copy.deepcopy(self.values.get(query))

    def find_similar(self, _query: str) -> None:
        return None

    def set(self, query: str, value: Any, ttl: int | None = None) -> None:
        del ttl
        self.values[query] = copy.deepcopy(value)


def _memory_shared_cache_factory() -> tuple[
    dict[str, _MemoryRetrievalCache],
    Callable[..., _MemoryRetrievalCache],
]:
    """Return a cache factory that shares values only within one namespace."""
    caches: dict[str, _MemoryRetrievalCache] = {}

    def factory(*, namespace: str, **_kwargs: Any) -> _MemoryRetrievalCache:
        return caches.setdefault(namespace, _MemoryRetrievalCache())

    return caches, factory


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
                    top_k=5,
                    enable_cache=False,
                    enable_reranking=False,
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

                await unified_rag_pipeline(
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

                    await unified_rag_pipeline(
                        query="API",
                        enable_expansion=True,
                        expansion_strategies=["acronym"]
                    )

                    # Should have expanded the query
                    mock_expand.assert_called_once_with("API", strategies=["acronym"])

    @pytest.mark.asyncio
    async def test_caching_feature(self, tmp_path):
        """Persisted legacy cache documents are reused while the answer regenerates."""
        query = "cached query"
        cache_path = tmp_path / "semantic_cache.json"
        now = time.time()
        key = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()
        cache_path.write_text(
            json.dumps(
                {
                    "cache": {
                        key: {
                            "value": {
                                "answer": "STALE_SENTINEL",
                                "generated_answer": "STALE_SENTINEL",
                                "documents": [
                                    {"id": "cached_1", "content": "Cached content"}
                                ],
                            },
                            "query": query,
                            "timestamp": now,
                            "ttl": 3600,
                            "access_count": 0,
                            "last_access": now,
                        }
                    },
                    "stats": {},
                    "config": {},
                }
            )
        )
        persisted_cache = SemanticCache(persist_path=str(cache_path))

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            return_value=persisted_cache,
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.AnswerGenerator'
        ) as mock_generator, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever:
            generator = MagicMock()
            generator.generate = AsyncMock(return_value={"answer": "Fresh answer"})
            mock_generator.return_value = generator
            result = await unified_rag_pipeline(
                query=query,
                enable_cache=True,
                cache_ttl=3600,
                adaptive_cache=False,
                enable_reranking=False,
            )

        assert result.cache_hit is True
        assert result.generated_answer == "Fresh answer"
        assert "STALE_SENTINEL" not in repr(result)
        assert len(result.documents) == 1
        assert result.documents[0]["content"] == "Cached content"
        assert result.metadata["retrieval_cache_hit"] is True
        assert result.metadata["generation_executed"] is True
        generator.generate.assert_awaited_once()
        assert generator.generate.await_args.kwargs["context"] == "Cached content"
        mock_retriever.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_namespace_keeps_trusted_users_isolated(self):
        async def capture_namespace(trusted_user_id):
            captured = []

            class NullCache:
                def get(self, _query):
                    return None

                def find_similar(self, _query):
                    return None

                def set(self, _query, _value, ttl=None):
                    return None

            def fake_shared_cache(**kwargs):
                captured.append(kwargs["namespace"])
                return NullCache()

            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
                side_effect=fake_shared_cache,
            ), patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
            ) as mock_retriever:
                retriever = MagicMock()
                retriever.retrieve = AsyncMock(return_value=[
                    Document(id="scope-doc", content="scope evidence", metadata={})
                ])
                mock_retriever.return_value = retriever
                with scoped_context(user_id=trusted_user_id):
                    await unified_rag_pipeline(
                        query="same query",
                        user_id="public-value",
                        workspace_id="same-workspace",
                        index_namespace="same-corpus",
                        enable_cache=True,
                        enable_generation=False,
                        enable_reranking=False,
                        credential_runtime=types.SimpleNamespace(_user_id=trusted_user_id),
                    )

            assert len(captured) == 1
            return captured[0]

        first = await capture_namespace(101)
        second = await capture_namespace(202)

        assert first != second

    @pytest.mark.asyncio
    async def test_cache_namespace_fingerprints_retrieval_scope_and_ownerless_db_paths(self):
        async def capture_namespace(**overrides):
            captured = []

            class NullCache:
                def get(self, _query):
                    return None

                def find_similar(self, _query):
                    return None

                def set(self, _query, _value, ttl=None):
                    return None

            def fake_shared_cache(**kwargs):
                captured.append(kwargs["namespace"])
                return NullCache()

            pipeline_kwargs = {
                "query": "scope query",
                "sources": ["media_db"],
                "search_mode": "fts",
                "top_k": 3,
                "min_score": 0.1,
                "index_namespace": "index-a",
                "workspace_id": "workspace-a",
                "enable_cache": True,
                "enable_generation": False,
                "enable_reranking": False,
                "credential_runtime": types.SimpleNamespace(_user_id=7),
            }
            pipeline_kwargs.update(overrides)

            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
                side_effect=fake_shared_cache,
            ), patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
            ) as mock_retriever:
                retriever = MagicMock()
                retriever.retrieve = AsyncMock(return_value=[
                    Document(id="scope-doc", content="scope evidence", metadata={})
                ])
                mock_retriever.return_value = retriever
                runtime_user_id = getattr(
                    pipeline_kwargs.get("credential_runtime"),
                    "_user_id",
                    None,
                )
                scope_manager = (
                    scoped_context(user_id=runtime_user_id)
                    if runtime_user_id is not None
                    else nullcontext()
                )
                with scope_manager:
                    await unified_rag_pipeline(**pipeline_kwargs)

            assert len(captured) == 1
            return captured[0]

        baseline = await capture_namespace()
        variants = [
            await capture_namespace(sources=["notes"]),
            await capture_namespace(search_mode="hybrid"),
            await capture_namespace(top_k=4),
            await capture_namespace(min_score=0.2),
            await capture_namespace(fts_level="chunk"),
            await capture_namespace(
                enable_date_filter=True,
                date_range={"start": "2024-01-01", "end": "2024-01-31"},
            ),
            await capture_namespace(enable_text_late_chunking=True),
            await capture_namespace(
                enable_text_late_chunking=True,
                chunk_method="sentences",
            ),
            await capture_namespace(
                enable_text_late_chunking=True,
                chunk_size=640,
            ),
            await capture_namespace(
                enable_text_late_chunking=True,
                chunk_overlap=64,
            ),
            await capture_namespace(
                enable_text_late_chunking=True,
                chunk_language="fr",
            ),
            await capture_namespace(index_namespace="index-b"),
        ]
        collection_a = RetrievalPlan(
            query="scope query",
            sources=("media_db",),
            search_mode="fts",
            top_k=3,
            min_score=0.1,
            index_namespace=None,
            collection_names={"media_db": "collection-a"},
        )
        collection_b = RetrievalPlan(
            query="scope query",
            sources=("media_db",),
            search_mode="fts",
            top_k=3,
            min_score=0.1,
            index_namespace=None,
            collection_names={"media_db": "collection-b"},
        )
        variants.extend([
            await capture_namespace(index_namespace=None, retrieval_plan=collection_a),
            await capture_namespace(index_namespace=None, retrieval_plan=collection_b),
        ])

        ownerless_a = await capture_namespace(
            credential_runtime=types.SimpleNamespace(_user_id=None),
            user_id=None,
            media_db_path="/srv/tenant-a/media.db",
        )
        ownerless_b = await capture_namespace(
            credential_runtime=types.SimpleNamespace(_user_id=None),
            user_id=None,
            media_db_path="/srv/tenant-b/media.db",
        )
        ownerless_server_a = await capture_namespace(
            credential_runtime=types.SimpleNamespace(_user_id=None),
            user_id="public-a",
            media_db_path=None,
        )
        ownerless_server_b = await capture_namespace(
            credential_runtime=types.SimpleNamespace(_user_id=None),
            user_id="public-b",
            media_db_path=None,
        )
        assert ownerless_server_a.partition("|")[0] == ownerless_server_b.partition("|")[0], (
            "ownerless runtimes must retain server owner identity",
            ownerless_server_a.partition("|")[0],
            ownerless_server_b.partition("|")[0],
        )
        checks = {
            "retrieval_fields_change_identity": all(
                namespace != baseline for namespace in variants
            ),
            "retrieval_variants_are_distinct": len(set(variants)) == len(variants),
            "ownerless_db_paths_are_distinct": ownerless_a != ownerless_b,
        }
        assert checks == dict.fromkeys(checks, True)

    @pytest.mark.asyncio
    async def test_cache_namespace_tracks_current_content_authorization_scope(self):
        async def capture_namespace(
            scope_kwargs: dict[str, Any],
            *,
            runtime: Any = None,
        ) -> str:
            captured: list[str] = []

            class NullCache:
                def get(self, _query: str) -> None:
                    return None

                def find_similar(self, _query: str) -> None:
                    return None

                def set(self, _query: str, _value: Any, ttl: int | None = None) -> None:
                    del ttl

            def fake_shared_cache(**kwargs: Any) -> NullCache:
                captured.append(kwargs["namespace"])
                return NullCache()

            with patch.object(up, "get_shared_cache", side_effect=fake_shared_cache), patch.object(
                up,
                "MultiDatabaseRetriever",
            ) as mock_retriever:
                retriever = MagicMock()
                retriever.retrieve = AsyncMock(
                    return_value=[Document(id="scope-doc", content="scope evidence", metadata={})]
                )
                mock_retriever.return_value = retriever
                with scoped_context(user_id=7, **scope_kwargs):
                    await unified_rag_pipeline(
                        query="authorization namespace query",
                        workspace_id="same-workspace",
                        enable_cache=True,
                        enable_generation=False,
                        enable_reranking=False,
                        adaptive_cache=False,
                        credential_runtime=runtime or types.SimpleNamespace(_user_id=7),
                    )

            assert len(captured) == 1
            return captured[0]

        baseline_scope = {
            "org_ids": [2, 1],
            "team_ids": [20, 10],
            "active_org_id": 1,
            "active_team_id": 10,
            "is_admin": False,
            "session_role": "content_reader",
        }
        baseline = await capture_namespace(baseline_scope)
        equivalent = await capture_namespace(
            {
                **baseline_scope,
                "org_ids": [1, 2, 1],
                "team_ids": [10, 20, 10],
            }
        )
        private_runtime_fields = await capture_namespace(
            baseline_scope,
            runtime=types.SimpleNamespace(
                _user_id=7,
                _team_ids=[999],
                _org_ids=[998],
                _active_team_id=999,
                _active_org_id=998,
                _is_admin=True,
            ),
        )
        variants = [
            await capture_namespace({**baseline_scope, "team_ids": [10]}),
            await capture_namespace({**baseline_scope, "org_ids": [1]}),
            await capture_namespace({**baseline_scope, "active_org_id": 2}),
            await capture_namespace({**baseline_scope, "active_team_id": 20}),
            await capture_namespace({**baseline_scope, "is_admin": True}),
            await capture_namespace({**baseline_scope, "session_role": "content_writer"}),
        ]

        assert equivalent == baseline
        assert private_runtime_fields == baseline
        assert all(namespace != baseline for namespace in variants)
        assert len(set(variants)) == len(variants)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("ambient_scope", "runtime_user_id", "expected_code"),
        [
            (None, 42, "missing"),
            (types.SimpleNamespace(user_id=42), 42, "malformed"),
            (
                ScopeContext(
                    user_id="42",  # type: ignore[arg-type]
                    org_ids=[],
                    team_ids=[],
                    active_org_id=None,
                    active_team_id=None,
                ),
                42,
                "malformed",
            ),
            (
                ScopeContext(
                    user_id=41,
                    org_ids=[],
                    team_ids=[],
                    active_org_id=None,
                    active_team_id=None,
                ),
                42,
                "user_mismatch",
            ),
            (
                ScopeContext(
                    user_id=42,
                    org_ids=[],
                    team_ids=[],
                    active_org_id=None,
                    active_team_id=None,
                ),
                42.0,
                "user_mismatch",
            ),
        ],
    )
    async def test_authenticated_cache_bypasses_without_matching_ambient_scope(
        self,
        ambient_scope: Any,
        runtime_user_id: Any,
        expected_code: str,
    ) -> None:
        cache = MagicMock()
        cache.get.return_value = None
        cache.find_similar.return_value = None

        with patch.object(
            up,
            "get_scope",
            return_value=ambient_scope,
        ) as mock_get_scope, patch.object(
            up,
            "get_shared_cache",
            return_value=cache,
        ) as mock_shared_cache, patch.object(
            up,
            "MultiDatabaseRetriever",
        ) as mock_retriever:
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(
                return_value=[Document(id="fresh", content="current RLS evidence", metadata={})]
            )
            mock_retriever.return_value = retriever
            result = await unified_rag_pipeline(
                query="authorization cache bypass",
                enable_cache=True,
                enable_generation=False,
                enable_reranking=False,
                adaptive_cache=False,
                credential_runtime=types.SimpleNamespace(_user_id=runtime_user_id),
            )

        mock_get_scope.assert_called_once_with()
        mock_shared_cache.assert_not_called()
        cache.get.assert_not_called()
        cache.set.assert_not_called()
        retriever.retrieve.assert_awaited_once()
        assert result.cache_hit is False
        assert result.metadata["cache_bypassed"] == {
            "reason": "content_authorization_scope_unavailable",
            "code": expected_code,
        }
        assert "42" not in json.dumps(result.metadata["cache_bypassed"])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("privileged_scope", "privileged_document_id"),
        [
            ({"team_ids": [12], "active_team_id": 12}, "team-document"),
            ({"is_admin": True}, "admin-document"),
        ],
    )
    async def test_shared_cache_rechecks_current_rls_after_authorization_change(
        self,
        privileged_scope: dict[str, Any],
        privileged_document_id: str,
    ) -> None:
        _caches, shared_cache_factory = _memory_shared_cache_factory()

        async def scope_aware_retrieve(*_args: Any, **_kwargs: Any) -> list[Document]:
            scope = get_scope()
            assert isinstance(scope, ScopeContext)
            if scope.is_admin:
                document_id = "admin-document"
            elif scope.active_team_id is not None:
                document_id = "team-document"
            else:
                document_id = "personal-document"
            return [Document(id=document_id, content=document_id, metadata={})]

        with patch.object(up, "get_shared_cache", side_effect=shared_cache_factory), patch.object(
            up,
            "MultiDatabaseRetriever",
        ) as mock_retriever:
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(side_effect=scope_aware_retrieve)
            mock_retriever.return_value = retriever
            common = {
                "query": "same protected query",
                "enable_cache": True,
                "enable_generation": False,
                "enable_reranking": False,
                "adaptive_cache": False,
                "credential_runtime": types.SimpleNamespace(_user_id=55),
            }
            with scoped_context(user_id=55, **privileged_scope):
                privileged = await unified_rag_pipeline(**common)
            with scoped_context(user_id=55):
                current = await unified_rag_pipeline(**common)
            with scoped_context(user_id=55):
                current_hit = await unified_rag_pipeline(**common)

        assert privileged.documents[0]["id"] == privileged_document_id
        assert current.documents[0]["id"] == "personal-document"
        assert current_hit.documents[0]["id"] == "personal-document"
        assert privileged.cache_hit is False
        assert current.cache_hit is False
        assert current_hit.cache_hit is True
        assert retriever.retrieve.await_count == 2

    @pytest.mark.concurrent
    @pytest.mark.asyncio
    async def test_concurrent_authorization_scopes_preserve_their_cached_documents(self) -> None:
        _caches, shared_cache_factory = _memory_shared_cache_factory()
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        gate_initial_retrievals = True

        async def scope_aware_retrieve(*_args: Any, **_kwargs: Any) -> list[Document]:
            scope = get_scope()
            assert isinstance(scope, ScopeContext)
            team_id = scope.active_team_id
            assert team_id in {10, 20}
            if gate_initial_retrievals:
                own_event, peer_event = (
                    (first_started, second_started)
                    if team_id == 10
                    else (second_started, first_started)
                )
                own_event.set()
                await peer_event.wait()
            document_id = f"team-{team_id}-document"
            return [Document(id=document_id, content=document_id, metadata={})]

        async def run_for_team(team_id: int) -> Any:
            with scoped_context(
                user_id=91,
                team_ids=[team_id],
                active_team_id=team_id,
            ):
                return await unified_rag_pipeline(
                    query="concurrent protected query",
                    enable_cache=True,
                    enable_generation=False,
                    enable_reranking=False,
                    adaptive_cache=False,
                    credential_runtime=types.SimpleNamespace(
                        _user_id=91,
                        _team_ids=[999],
                    ),
                )

        with patch.object(up, "get_shared_cache", side_effect=shared_cache_factory), patch.object(
            up,
            "MultiDatabaseRetriever",
        ) as mock_retriever:
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(side_effect=scope_aware_retrieve)
            mock_retriever.return_value = retriever

            first, second = await asyncio.gather(run_for_team(10), run_for_team(20))
            gate_initial_retrievals = False
            first_hit = await run_for_team(10)
            second_hit = await run_for_team(20)

        assert first.documents[0]["id"] == "team-10-document"
        assert second.documents[0]["id"] == "team-20-document"
        assert first_hit.documents[0]["id"] == "team-10-document"
        assert second_hit.documents[0]["id"] == "team-20-document"
        assert first_hit.cache_hit is True
        assert second_hit.cache_hit is True
        assert retriever.retrieve.await_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mode_name", "mode_kwargs"),
        [
            ("query_expansion", {"expand_query": True}),
            ("hyde", {"enable_hyde": True}),
            ("prf", {"enable_prf": True}),
            ("query_decomposition", {"enable_query_decomposition": True}),
            ("gap_analysis", {"enable_gap_analysis": True}),
        ],
    )
    async def test_secondary_retrieval_modes_bypass_cache(
        self,
        mode_name,
        mode_kwargs,
    ):
        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache'
        ) as mock_shared_cache, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.multi_strategy_expansion',
            None,
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.QueryRewriter',
            None,
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.generate_hypothetical_answer',
            None,
        ), patch(
            'tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib.analyze',
            return_value="[]",
        ):
            cache = MagicMock()
            mock_shared_cache.return_value = cache
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(return_value=[
                Document(id="fresh", content="fresh evidence", metadata={})
            ])
            mock_retriever.return_value = retriever
            result = await unified_rag_pipeline(
                query="cache bypass query",
                top_k=1,
                enable_cache=True,
                enable_generation=False,
                enable_reranking=False,
                adaptive_cache=False,
                **mode_kwargs,
            )

        mock_shared_cache.assert_not_called()
        cache.get.assert_not_called()
        cache.set.assert_not_called()
        retriever.retrieve.assert_awaited_once()
        assert result.cache_hit is False
        assert result.metadata["cache_bypassed"] == {
            "reason": "secondary_retrieval_mode",
            "modes": [mode_name],
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("source", "scope_kwargs"),
        [
            ("media_db", {"include_media_ids": [7]}),
            ("notes", {"include_note_ids": ["note-7"]}),
        ],
    )
    async def test_explicit_include_ids_bypass_cache(
        self,
        source,
        scope_kwargs,
    ):
        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache'
        ) as mock_shared_cache, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever:
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(return_value=[
                Document(id="scoped", content="scoped evidence", metadata={})
            ])
            mock_retriever.return_value = retriever
            result = await unified_rag_pipeline(
                query="explicit scope query",
                sources=[source],
                top_k=1,
                enable_cache=True,
                enable_generation=False,
                enable_reranking=False,
                adaptive_cache=False,
                **scope_kwargs,
            )

        mock_shared_cache.assert_not_called()
        retriever.retrieve.assert_awaited_once()
        assert result.cache_hit is False
        assert result.metadata["cache_bypassed"] == {
            "reason": "explicit_source_selection",
        }

    @pytest.mark.asyncio
    async def test_auto_temporal_windows_bypass_cache_across_repeated_requests(self):
        cache = MagicMock()
        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            return_value=cache,
        ) as mock_shared_cache, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever:
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(side_effect=lambda *_args, **_kwargs: [
                Document(id="temporal", content="recent evidence", metadata={})
            ])
            mock_retriever.return_value = retriever
            common = {
                "query": "recent evidence",
                "top_k": 1,
                "auto_temporal_filters": True,
                "enable_cache": True,
                "enable_generation": False,
                "enable_reranking": False,
                "adaptive_cache": False,
            }
            first = await unified_rag_pipeline(**common)
            second = await unified_rag_pipeline(**common)

        mock_shared_cache.assert_not_called()
        cache.get.assert_not_called()
        cache.set.assert_not_called()
        assert retriever.retrieve.await_count == 2
        for result in (first, second):
            assert result.cache_hit is False
            assert result.metadata["cache_bypassed"] == {
                "reason": "auto_temporal_window",
            }

    @pytest.mark.asyncio
    async def test_research_loop_evidence_is_not_cached_as_base_retrieval(self):
        from tldw_Server_API.app.core.RAG.rag_service.query_classifier import (
            QueryClassification,
        )
        from tldw_Server_API.app.core.RAG.rag_service.research_agent import (
            ResearchOutput,
        )

        classification = QueryClassification(
            skip_search=False,
            search_local_db=True,
            search_web=True,
            search_academic=False,
            search_discussions=False,
            standalone_query="research query",
            detected_intent="factual",
            confidence=0.8,
            reasoning="Needs research",
        )
        research_output = ResearchOutput(
            query="research query",
            standalone_query="research query",
            all_results=[
                {
                    "id": "research-doc",
                    "content": "external research evidence",
                    "source": "web",
                    "score": 0.9,
                }
            ],
            total_iterations=1,
            total_results=1,
            total_duration_sec=0.1,
            final_reasoning="complete",
            completed=True,
        )
        cache = MagicMock()

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            return_value=cache,
        ) as mock_shared_cache, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.classify_and_reformulate',
            AsyncMock(return_value=classification),
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.research_loop',
            AsyncMock(return_value=research_output),
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_default_registry',
            None,
        ):
            result = await unified_rag_pipeline(
                query="research query",
                enable_query_classification=True,
                enable_research_loop=True,
                enable_cache=True,
                enable_generation=False,
                enable_reranking=False,
                adaptive_cache=False,
            )

        mock_retriever.assert_not_called()
        mock_shared_cache.assert_not_called()
        cache.set.assert_not_called()
        assert result.documents[0]["id"] == "research-doc"
        assert result.metadata["retrieval_bypassed"]["reason"] == "research_loop"

    @pytest.mark.asyncio
    async def test_classification_external_prefetch_is_not_cached_as_base_retrieval(self):
        from tldw_Server_API.app.core.RAG.rag_service.query_classifier import (
            QueryClassification,
        )

        classification = QueryClassification(
            skip_search=False,
            search_local_db=False,
            search_web=True,
            search_academic=False,
            search_discussions=False,
            standalone_query="external query",
            detected_intent="factual",
            confidence=0.8,
            reasoning="External only",
        )
        cache = MagicMock()

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            return_value=cache,
        ) as mock_shared_cache, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.classify_and_reformulate',
            AsyncMock(return_value=classification),
        ), patch(
            'tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.perform_websearch',
            return_value={"results": []},
        ), patch(
            'tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs.process_web_search_results',
            return_value={
                "results": [
                    {
                        "title": "External result",
                        "url": "https://example.com/external",
                        "content": "external evidence",
                    }
                ]
            },
        ):
            result = await unified_rag_pipeline(
                query="external query",
                enable_query_classification=True,
                enable_cache=True,
                enable_generation=False,
                enable_reranking=False,
                adaptive_cache=False,
                top_k=1,
            )

        mock_retriever.assert_not_called()
        mock_shared_cache.assert_not_called()
        cache.set.assert_not_called()
        assert result.documents[0]["id"] == "https://example.com/external"
        assert result.metadata["retrieval_bypassed"]["reason"] == (
            "classification_external_prefetch"
        )

    @pytest.mark.asyncio
    async def test_error_fallback_docs_are_not_cached_as_base_retrieval(self):
        from tldw_Server_API.app.core.RAG.rag_service import database_retrievers

        cache = MagicMock()
        cache.get.return_value = None
        cache.find_similar.return_value = None
        fallback_retriever = MagicMock()
        fallback_retriever.retrieve = AsyncMock(return_value=[
            Document(
                id="fallback-doc",
                content="fallback evidence",
                metadata={},
                source=DataSource.MEDIA_DB,
            )
        ])

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            return_value=cache,
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.execute_retrieval_phase',
            AsyncMock(side_effect=RuntimeError("base retrieval failed")),
        ) as mock_execute, patch.object(
            database_retrievers,
            "MediaDBRetriever",
            return_value=fallback_retriever,
        ):
            result = await unified_rag_pipeline(
                query="fallback cache query",
                sources=["media_db"],
                media_db_path="media.db",
                search_mode="hybrid",
                adaptive_hybrid_weights=False,
                enable_cache=True,
                enable_generation=False,
                enable_reranking=False,
                adaptive_cache=False,
            )

        mock_execute.assert_awaited_once()
        fallback_retriever.retrieve.assert_awaited_once()
        cache.set.assert_not_called()
        assert result.documents[0]["id"] == "fallback-doc"
        assert result.metadata["fallbacks"]["media_db_fts_on_error"] is True
        assert any("base retrieval failed" in error for error in result.errors)

    @pytest.mark.asyncio
    async def test_cache_namespace_uses_post_routing_retrieval_values(self):
        captured = []

        class NullCache:
            def get(self, _query):
                return None

            def find_similar(self, _query):
                return None

            def set(self, _query, _value, ttl=None):
                return None

        class RoutedQuery:
            def route_query(self, _query):
                return {"retrieval_strategy": "precise", "top_k": 7}

        def fake_shared_cache(**kwargs):
            captured.append(kwargs["namespace"])
            return NullCache()

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            side_effect=fake_shared_cache,
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.QueryRouter',
            return_value=RoutedQuery(),
        ):
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(return_value=[
                Document(id="scope-doc", content="scope evidence", metadata={})
            ])
            mock_retriever.return_value = retriever
            common = {
                "query": "routed cache query",
                "top_k": 3,
                "enable_cache": True,
                "enable_generation": False,
                "enable_reranking": False,
                "credential_runtime": types.SimpleNamespace(_user_id=7),
            }
            with scoped_context(user_id=7):
                await unified_rag_pipeline(**common)
                await unified_rag_pipeline(**common, enable_intent_routing=True)

        assert len(captured) == 2
        assert captured[0] != captured[1]

    @pytest.mark.asyncio
    async def test_scoped_cache_setup_failure_does_not_retry_unscoped(self):
        secret = "sk-cache-constructor-sentinel"

        class FailingScopedCache:
            created_without_scope = False

            def __init__(self, *, namespace=None, **_kwargs):
                if namespace is not None:
                    raise TypeError(secret)
                type(self).created_without_scope = True

            def get(self, _query):
                return {
                    "documents": [
                        {"id": "unsafe", "content": "unscoped cached evidence"}
                    ]
                }

        messages = []
        sink_id = logger.add(messages.append)
        try:
            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache',
                FailingScopedCache,
            ), patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
                None,
            ), patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
            ) as mock_retriever:
                retriever = MagicMock()
                retriever.retrieve = AsyncMock(return_value=[
                    Document(id="fresh", content="fresh evidence", metadata={})
                ])
                mock_retriever.return_value = retriever
                with scoped_context(user_id=42):
                    result = await unified_rag_pipeline(
                        query="scoped setup failure",
                        enable_cache=True,
                        enable_generation=False,
                        enable_reranking=False,
                        adaptive_cache=False,
                        credential_runtime=types.SimpleNamespace(_user_id=42),
                    )
        finally:
            logger.remove(sink_id)

        assert FailingScopedCache.created_without_scope is False
        assert result.cache_hit is False
        assert result.documents[0]["content"] == "fresh evidence"
        assert secret not in "".join(messages)

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
        assert result.generated_answer is None
        assert "Adaptive cached answer" not in repr(result)
        assert result.metadata["retrieval_cache_hit"] is True
        assert result.metadata["generation_executed"] is False

    @pytest.mark.asyncio
    async def test_direct_cache_payload_uses_retrieval_only_document_sanitizer(self):
        cached_result = {
            "answer": "STALE_SENTINEL",
            "documents": [
                {
                    "id": "cached-doc",
                    "content": "cached evidence",
                    "score": float("inf"),
                    "metadata": {
                        "answer": "STALE_SENTINEL",
                        "generation_provider": "STALE_SENTINEL",
                        "nested": {
                            "generated_answer": "STALE_SENTINEL",
                            "generation_model": "STALE_SENTINEL",
                            "generation_prompt": "STALE_SENTINEL",
                            "verification_report": {"answer": "STALE_SENTINEL"},
                            "safe": "kept",
                            "nan": float("nan"),
                        },
                    },
                }
            ],
        }
        mock_cache = MagicMock()
        mock_cache.get.return_value = cached_result

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            return_value=mock_cache,
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever:
            result = await unified_rag_pipeline(
                query="direct cache query",
                enable_cache=True,
                enable_generation=False,
                enable_reranking=False,
                adaptive_cache=False,
            )

        mock_retriever.assert_not_called()
        assert result.cache_hit is True
        assert "STALE_SENTINEL" not in repr(result)
        assert result.documents[0]["score"] == 0.0
        assert result.documents[0]["metadata"]["nested"] == {"safe": "kept"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "semantic_match",
        [
            ("too", "many", "fields", 0.5),
            ("", 0.5),
            ("x" * 4097, 0.5),
            ("similar query", True),
            ("similar query", float("nan")),
            ("similar query", -0.1),
            ("similar query", 1.1),
            ("similar query", "0.9"),
        ],
    )
    async def test_malformed_semantic_cache_match_is_a_miss(
        self,
        semantic_match,
    ):
        get_queries = []

        class MalformedMatchCache:
            def get(self, query):
                get_queries.append(query)
                if query != "original query":
                    return {
                        "documents": [
                            {"id": "unsafe", "content": "unsafe cached evidence"}
                        ]
                    }
                return None

            def find_similar(self, _query):
                return semantic_match

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            return_value=MalformedMatchCache(),
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever:
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(return_value=[
                Document(id="fresh", content="fresh evidence", metadata={})
            ])
            mock_retriever.return_value = retriever
            result = await unified_rag_pipeline(
                query="original query",
                enable_cache=True,
                enable_generation=False,
                enable_reranking=False,
                adaptive_cache=False,
            )

        assert get_queries == ["original query"]
        retriever.retrieve.assert_awaited_once()
        assert result.cache_hit is False
        assert "cached_query" not in result.metadata
        assert "cache_similarity" not in result.metadata

    @pytest.mark.asyncio
    async def test_sync_cache_wrappers_returning_awaitables_are_awaited(self):
        calls = []

        class WrappedCache:
            def get(self, query):
                async def resolve():
                    calls.append(("get", query))
                    if query == "similar query":
                        return {
                            "documents": [
                                {"id": "wrapped-doc", "content": "wrapped evidence"}
                            ],
                            "answer": "STALE_SENTINEL",
                        }
                    return None

                return resolve()

            def find_similar(self, query):
                async def resolve():
                    calls.append(("find", query))
                    return "key", "similar query", 0.95

                return resolve()

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache',
            return_value=WrappedCache(),
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            None,
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever:
            result = await unified_rag_pipeline(
                query="wrapped query",
                enable_cache=True,
                enable_generation=False,
                enable_reranking=False,
                adaptive_cache=False,
            )

        assert calls == [
            ("get", "wrapped query"),
            ("find", "wrapped query"),
            ("get", "similar query"),
        ]
        assert result.cache_hit is True
        assert result.documents[0]["content"] == "wrapped evidence"
        assert "STALE_SENTINEL" not in repr(result)
        assert result.metadata["cache_similarity"] == 0.95
        assert "cached_query" not in result.metadata
        mock_retriever.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_cache_wrapper_preserves_cancellation(self):
        class CancellingCache:
            def get(self, _query):
                async def cancel():
                    raise asyncio.CancelledError

                return cancel()

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache',
            return_value=CancellingCache(),
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            None,
        ):
            with pytest.raises(asyncio.CancelledError):
                await unified_rag_pipeline(
                    query="cancel cache lookup",
                    enable_cache=True,
                    enable_generation=False,
                    enable_reranking=False,
                    adaptive_cache=False,
                )

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
        assert result.metadata["retrieval_cache_hit"] is True
        assert result.metadata["generation_executed"] is False

    @pytest.mark.asyncio
    async def test_cache_storage_persists_retrieval_only_payload(self):
        """Verify cache set receives documents and fixed retrieval metadata only."""

        class RecordingCache:
            def __init__(self, *_, **__):
                self.set_calls = []

            def get(self, _query):

                return None

            def find_similar(self, _query):

                return None

            def set(self, query, value, ttl=None):
                async def record():
                    self.set_calls.append((query, value, ttl))

                return record()

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
        assert set(payload) == {"documents", "metadata"}
        assert payload["metadata"] == {
            "kind": "retrieval_documents",
            "schema_version": 1,
        }
        assert payload.get("documents")
        assert payload["documents"][0]["id"] == "doc-cache"
        assert "answer" not in payload
        assert "generated_answer" not in payload

    @pytest.mark.asyncio
    async def test_cache_stores_pre_transformation_retrieval_snapshot(self):
        class MemoryCache:
            def __init__(self):
                self.payload = None

            def get(self, _query):
                return copy.deepcopy(self.payload)

            def find_similar(self, _query):
                return None

            def set(self, _query, value, ttl=None):
                self.payload = copy.deepcopy(value)

        cache = MemoryCache()

        def downweight(documents, strength):
            for document in documents:
                document.score *= strength
            return {"affected": len(documents), "total": len(documents)}

        with patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
            return_value=cache,
        ), patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
        ) as mock_retriever, patch(
            'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.downweight_injection_docs',
            side_effect=downweight,
        ):
            retriever = MagicMock()
            retriever.retrieve = AsyncMock(return_value=[
                Document(
                    id="injection-doc",
                    content="potential injection evidence",
                    metadata={},
                    score=1.0,
                )
            ])
            mock_retriever.return_value = retriever
            common = {
                "query": "snapshot query",
                "enable_cache": True,
                "enable_generation": False,
                "enable_reranking": False,
                "enable_injection_filter": True,
                "injection_filter_strength": 0.5,
                "adaptive_cache": False,
            }
            miss = await unified_rag_pipeline(**common)
            hit = await unified_rag_pipeline(**common)

        retriever.retrieve.assert_awaited_once()
        assert miss.cache_hit is False
        assert hit.cache_hit is True
        assert miss.documents[0]["score"] == 0.5
        assert hit.documents[0]["score"] == 0.5
        assert cache.payload["documents"][0]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_cache_storage_error_log_omits_exception_details(self):
        secret = "sk-cache-storage-sentinel"

        class ExplodingCache:
            def __init__(self, **_kwargs):
                pass

            def get(self, _query):
                return None

            def find_similar(self, _query):
                return None

            def set(self, _query, _value, ttl=None):
                raise RuntimeError(secret)

        messages = []
        sink_id = logger.add(messages.append)
        try:
            with patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.SemanticCache',
                ExplodingCache,
            ), patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.get_shared_cache',
                None,
            ), patch(
                'tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever'
            ) as mock_retriever:
                retriever = MagicMock()
                retriever.retrieve = AsyncMock(return_value=[
                    Document(id="doc", content="retrieved evidence", metadata={})
                ])
                mock_retriever.return_value = retriever
                result = await unified_rag_pipeline(
                    query="storage failure",
                    enable_cache=True,
                    enable_generation=False,
                    enable_reranking=False,
                    adaptive_cache=False,
                )
        finally:
            logger.remove(sink_id)

        assert result.documents
        assert secret not in "".join(messages)

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
            instances: list["StubMultiDatabaseRetriever"] = []

            def __init__(self, db_paths, user_id="0", *, media_db=None, chacha_db=None):

                self.db_paths = db_paths
                self.user_id = user_id
                self.media_db = media_db
                self.chacha_db = chacha_db
                self.retrievers = {DataSource.MEDIA_DB: StubMediaRetriever()}
                self.retrieve_invocations: list[Any] = []
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
                self.run_calls: list[dict[str, Any]] = []

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
        managed_calls: list[dict[str, Any]] = []
        verify_calls: list[dict[str, Any]] = []

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

                await unified_rag_pipeline(
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

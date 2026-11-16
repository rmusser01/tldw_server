"""Hybrid search keeps bounded, scoped evidence from both requested indexes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline as pipeline
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MediaDBRetriever,
    MultiDatabaseRetriever,
    RetrievalConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


def documents(prefix, count=8):
    return [Document(id=f"{prefix}-{rank}", content=f"{prefix} evidence {rank}", metadata={}) for rank in range(count)]


def retriever(limit=8):
    instance = MediaDBRetriever.__new__(MediaDBRetriever)
    instance.config = RetrievalConfig(max_results=limit)
    return instance


@pytest.mark.parametrize("alpha", [0.1, 0.7, 0.99])
def test_disjoint_chunk_ids_keep_top_result_from_both_nonzero_lanes(alpha):
    result = retriever()._reciprocal_rank_fusion(documents("fts"), documents("vector"), alpha)
    ids = [doc.id for doc in result]
    assert len(ids) == len(set(ids)) == 8
    assert {"fts-0", "vector-0"} <= set(ids)
    assert [doc.score for doc in result] == sorted((doc.score for doc in result), reverse=True)


@pytest.mark.parametrize("alpha,prefix", [(0, "fts"), (1, "vector")])
def test_explicit_pure_lane_excludes_zero_weight_evidence(alpha, prefix):
    result = retriever()._reciprocal_rank_fusion(documents("fts", 2), documents("vector", 2), alpha)
    assert [doc.id for doc in result] == [f"{prefix}-0", f"{prefix}-1"]


def test_single_slot_keeps_best_weighted_result_and_shared_ids_are_not_duplicated():
    assert retriever(1)._reciprocal_rank_fusion(documents("fts"), documents("vector"), 0.7)[0].id == "vector-0"
    shared = documents("same", 2)
    assert [doc.id for doc in retriever(2)._reciprocal_rank_fusion(shared, shared, 0.7)] == ["same-0", "same-1"]


@pytest.mark.asyncio
async def test_primary_multi_retriever_forwards_weight_and_selected_media_scope():
    media = retriever()
    media.retrieve_hybrid = AsyncMock(return_value=documents("selected", 1))
    multi = MultiDatabaseRetriever({})
    multi.retrievers = {DataSource.MEDIA_DB: media}
    config = RetrievalConfig(max_results=8)
    config.hybrid_alpha = 0.2
    await multi.retrieve("proposal", sources=[DataSource.MEDIA_DB], config=config, allowed_media_ids=[71])
    assert media.retrieve_hybrid.await_args.kwargs.get("alpha") == 0.2
    assert media.retrieve_hybrid.await_args.kwargs["allowed_media_ids"] == [71]


@pytest.mark.asyncio
async def test_pipeline_preserves_requested_weight_in_primary_config_and_cache(monkeypatch):
    captured, namespaces = [], []

    async def retrieve(*args, **kwargs):
        captured.append(getattr(kwargs["config"], "hybrid_alpha", None))
        return documents("selected", 1)

    class Cache:
        def get(self, query):
            return None

        def find_similar(self, query):
            return None

        def set(self, query, value, ttl=None):
            pass

    def cache_factory(**kwargs):
        namespaces.append(kwargs["namespace"])
        return Cache()

    monkeypatch.setattr(pipeline, "MultiDatabaseRetriever", lambda *args, **kwargs: SimpleNamespace(retrieve=retrieve))
    monkeypatch.setattr(pipeline, "get_shared_cache", cache_factory)
    for alpha in (0.2, 0.8):
        await pipeline.unified_rag_pipeline(
            query="What does the selected source propose?",
            sources=["media_db"],
            search_mode="hybrid",
            hybrid_alpha=alpha,
            adaptive_hybrid_weights=False,
            enable_generation=False,
            enable_reranking=False,
            enable_cache=True,
            adaptive_cache=False,
            enable_pre_retrieval_clarification=False,
        )
    assert captured == [0.2, 0.8]
    assert len(namespaces) == 2 and namespaces[0] != namespaces[1]


@pytest.mark.asyncio
async def test_real_natural_fts_evidence_survives_disjoint_vector_results(media_database):
    text = "The regional offices should be consolidated into one public service."
    media_id, _, _ = media_database.add_media_with_keywords(
        title="Regional proposal",
        media_type="document",
        content=text,
        keywords=[],
        chunks=[{"text": text, "start_char": 0, "end_char": len(text), "chunk_type": "text"}],
    )
    media = MediaDBRetriever(
        str(media_database.db_path),
        media_db=media_database,
        config=RetrievalConfig(max_results=3, min_score=0.2, fts_level="chunk"),
    )
    media._retrieve_vector = AsyncMock(return_value=documents("indexed", 3))
    result = await media.retrieve_hybrid(
        "Does this document propose consolidating regional offices?", alpha=0.7, allowed_media_ids=[media_id]
    )
    assert any(doc.content == text for doc in result)
    assert media._retrieve_vector.await_args.kwargs["allowed_media_ids"] == [media_id]


@given(
    limit=st.integers(min_value=0, max_value=12),
    fts_count=st.integers(min_value=0, max_value=12),
    vector_count=st.integers(min_value=0, max_value=12),
    alpha=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_fusion_respects_limit_unique_identity_and_available_nonzero_lanes(limit, fts_count, vector_count, alpha):
    result = retriever(limit)._reciprocal_rank_fusion(
        documents("fts", fts_count), documents("vector", vector_count), alpha
    )
    ids = [doc.id for doc in result]
    assert len(ids) == len(set(ids)) <= limit
    if limit >= 2 and fts_count and vector_count and 0 < alpha < 1:
        assert {"fts-0", "vector-0"} <= set(ids)
    if alpha == 0:
        assert all(value.startswith("fts-") for value in ids)
    if alpha == 1:
        assert all(value.startswith("vector-") for value in ids)


def test_equal_weight_ties_have_repeatable_rank_order():
    expected = ["fts-0", "vector-0", "fts-1", "vector-1"]
    assert [
        doc.id for doc in retriever(4)._reciprocal_rank_fusion(documents("fts"), documents("vector"), 0.5)
    ] == expected

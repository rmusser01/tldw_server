from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import (
    ResolvedRAGRequest,
)
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan


def _resolved_request() -> ResolvedRAGRequest:
    return ResolvedRAGRequest(
        query="What changed?",
        strategy="standard",
        payload={
            "query": "What changed?",
            "strategy": "standard",
            "sources": ["media_db"],
            "search_mode": "hybrid",
            "top_k": 3,
            "min_score": 0.0,
            "enable_generation": True,
            "include_sources": True,
            "include_metadata": True,
        },
        index_namespace=None,
        rag_profile=None,
        user_id="1",
        feedback_user_id="1",
    )


def _retrieval_plan() -> RetrievalPlan:
    return RetrievalPlan(
        query="What changed?",
        sources=("media_db",),
        search_mode="hybrid",
        top_k=3,
        min_score=0.0,
        index_namespace=None,
        collection_names={"media_db": "user_1_media_embeddings"},
    )


@pytest.mark.asyncio
async def test_unified_pipeline_reuses_resolved_request_and_plan(monkeypatch):
    resolved = _resolved_request()
    plan = _retrieval_plan()
    seen = {}

    async def fake_retrieval_phase(**kwargs):
        seen["retrieval_resolved"] = kwargs["resolved_request"]
        seen["retrieval_plan"] = kwargs["retrieval_plan"]
        return SimpleNamespace(
            documents=[{"id": "doc-1", "content": "retrieved context"}],
            sources=[{"id": "doc-1"}],
            metadata={"retrieval": "ok"},
        )

    async def fake_generation_phase(**kwargs):
        seen["generation_resolved"] = kwargs["resolved_request"]
        seen["generation_plan"] = kwargs["retrieval_plan"]
        return {
            "answer": "generated answer",
            "sources": [],
            "metadata": {"generation": "ok"},
        }

    def fake_coordinate(result, resolved_request, *, retrieval_plan=None, coordinator=None):
        seen["coordinator_resolved"] = resolved_request
        seen["coordinator_plan"] = retrieval_plan
        return result

    monkeypatch.setattr(unified_pipeline, "execute_retrieval_phase", fake_retrieval_phase)
    monkeypatch.setattr(unified_pipeline, "execute_generation_phase", fake_generation_phase)
    monkeypatch.setattr(unified_pipeline, "coordinate_standard_result_evidence", fake_coordinate)

    result = await unified_pipeline.unified_rag_pipeline(
        query=resolved.query,
        sources=list(plan.sources),
        top_k=plan.top_k,
        search_mode=plan.search_mode,
        enable_generation=True,
        resolved_request=resolved,
        retrieval_plan=plan,
    )

    assert result.generated_answer == "generated answer"
    assert seen["retrieval_resolved"] is resolved
    assert seen["retrieval_plan"].query == plan.query
    assert seen["retrieval_plan"].top_k == plan.top_k
    assert seen["retrieval_plan"].collection_names == plan.collection_names
    assert seen["generation_resolved"] is resolved
    assert seen["generation_plan"] is seen["retrieval_plan"]
    assert seen["coordinator_resolved"] is resolved
    assert seen["coordinator_plan"] is seen["retrieval_plan"]


@pytest.mark.asyncio
async def test_unified_pipeline_builds_single_legacy_resolved_request(monkeypatch):
    seen = {}

    async def fake_retrieval_phase(**kwargs):
        seen["retrieval_resolved"] = kwargs["resolved_request"]
        seen["retrieval_plan"] = kwargs["retrieval_plan"]
        return SimpleNamespace(documents=[], sources=[], metadata={})

    async def fake_generation_phase(**kwargs):
        seen["generation_resolved"] = kwargs["resolved_request"]
        seen["generation_plan"] = kwargs["retrieval_plan"]
        return {"answer": "legacy answer", "sources": [], "metadata": {}}

    def fake_coordinate(result, resolved_request, *, retrieval_plan=None, coordinator=None):
        seen["coordinator_resolved"] = resolved_request
        seen["coordinator_plan"] = retrieval_plan
        return result

    monkeypatch.setattr(unified_pipeline, "execute_retrieval_phase", fake_retrieval_phase)
    monkeypatch.setattr(unified_pipeline, "execute_generation_phase", fake_generation_phase)
    monkeypatch.setattr(unified_pipeline, "coordinate_standard_result_evidence", fake_coordinate)

    result = await unified_pipeline.unified_rag_pipeline(
        query="legacy query",
        top_k=5,
        search_mode="fts",
        enable_generation=True,
    )

    assert result.generated_answer == "legacy answer"
    assert seen["retrieval_resolved"] is seen["generation_resolved"]
    assert seen["retrieval_plan"] is seen["generation_plan"]
    assert seen["coordinator_resolved"] is seen["retrieval_resolved"]
    assert seen["coordinator_plan"] is seen["retrieval_plan"]
    assert seen["retrieval_resolved"].query == "legacy query"
    assert seen["retrieval_plan"].top_k == 5


@pytest.mark.asyncio
async def test_unified_pipeline_coordinates_retrieval_only_result(monkeypatch):
    resolved = _resolved_request()
    resolved.payload["enable_generation"] = False
    plan = _retrieval_plan()
    seen = {}

    async def fake_retrieval_phase(**kwargs):
        seen["retrieval_resolved"] = kwargs["resolved_request"]
        seen["retrieval_plan"] = kwargs["retrieval_plan"]
        return SimpleNamespace(
            documents=[{"id": "doc-1", "content": "retrieved only"}],
            metadata={"retrieval": "ok"},
        )

    async def fake_generation_phase(**kwargs):
        raise AssertionError("generation phase should not run")

    real_build_retrieval_only_result = unified_pipeline.build_retrieval_only_result

    def fake_build_retrieval_only_result(**kwargs):
        seen["retrieval_only_resolved"] = kwargs["resolved_request"]
        seen["retrieval_only_plan"] = kwargs["retrieval_plan"]
        return real_build_retrieval_only_result(**kwargs)

    def fake_coordinate(result, resolved_request, *, retrieval_plan=None, coordinator=None):
        seen["coordinator_resolved"] = resolved_request
        seen["coordinator_plan"] = retrieval_plan
        seen["coordinator_result"] = result
        return result

    monkeypatch.setattr(unified_pipeline, "execute_retrieval_phase", fake_retrieval_phase)
    monkeypatch.setattr(unified_pipeline, "execute_generation_phase", fake_generation_phase)
    monkeypatch.setattr(
        unified_pipeline,
        "build_retrieval_only_result",
        fake_build_retrieval_only_result,
    )
    monkeypatch.setattr(unified_pipeline, "coordinate_standard_result_evidence", fake_coordinate)

    result = await unified_pipeline.unified_rag_pipeline(
        query=resolved.query,
        sources=list(plan.sources),
        top_k=plan.top_k,
        search_mode=plan.search_mode,
        enable_reranking=False,
        resolved_request=resolved,
        retrieval_plan=plan,
    )

    assert result.query == resolved.query
    assert result.documents[0]["id"] == "doc-1"
    assert seen["retrieval_resolved"] is resolved
    assert seen["retrieval_plan"].query == plan.query
    assert seen["retrieval_plan"].top_k == plan.top_k
    assert seen["retrieval_plan"].collection_names == plan.collection_names
    assert seen["retrieval_only_resolved"] is resolved
    assert seen["retrieval_only_plan"] is seen["retrieval_plan"]
    assert seen["coordinator_resolved"] is resolved
    assert seen["coordinator_plan"] is seen["retrieval_plan"]
    assert seen["coordinator_result"].metadata["retrieval_plan"]["top_k"] == plan.top_k


@pytest.mark.asyncio
async def test_unified_pipeline_coordinates_cache_hit_generation_enabled(monkeypatch):
    resolved = _resolved_request()
    plan = _retrieval_plan()

    class FakeCache:
        def get(self, query):  # noqa: ANN001
            return {
                "answer": "cached answer",
                "documents": [{"id": "cached-doc", "content": "cached evidence"}],
                "cached": True,
            }

    monkeypatch.setattr(unified_pipeline, "SemanticCache", lambda *args, **kwargs: FakeCache())
    monkeypatch.setattr(unified_pipeline, "AdaptiveCache", None)
    monkeypatch.setattr(unified_pipeline, "get_shared_cache", None)

    result = await unified_pipeline.unified_rag_pipeline(
        query=resolved.query,
        sources=list(plan.sources),
        top_k=plan.top_k,
        search_mode=plan.search_mode,
        enable_generation=True,
        enable_cache=True,
        adaptive_cache=False,
        enable_reranking=False,
        resolved_request=resolved,
        retrieval_plan=plan,
    )

    assert result.cache_hit is True
    assert result.generated_answer == "cached answer"
    assert result.metadata["retrieval_plan"]["top_k"] == plan.top_k


@pytest.mark.asyncio
async def test_unified_pipeline_threads_effective_query_to_generation(monkeypatch):
    resolved = _resolved_request()
    plan = _retrieval_plan()
    seen = {}

    async def fake_classify_and_reformulate(**kwargs):
        seen["classifier_query"] = kwargs["query"]
        return SimpleNamespace(
            standalone_query="effective reformulated query",
            skip_search=False,
            search_local_db=True,
            search_web=False,
            search_academic=False,
            search_discussions=False,
            detected_intent="question",
            confidence=0.9,
            reasoning="test reformulation",
        )

    async def fake_retrieval_phase(**kwargs):
        seen["retrieval_resolved"] = kwargs["resolved_request"]
        seen["retrieval_plan"] = kwargs["retrieval_plan"]
        return SimpleNamespace(
            documents=[SimpleNamespace(id="doc-1", content="effective query evidence")],
            metadata={"retrieval": "ok"},
        )

    async def fake_generation_phase(**kwargs):
        seen["generation_resolved"] = kwargs["resolved_request"]
        seen["generation_plan"] = kwargs["retrieval_plan"]
        return {
            "answer": "effective answer",
            "sources": [],
            "metadata": {},
        }

    class FakeAnswerGenerator:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        unified_pipeline,
        "classify_and_reformulate",
        fake_classify_and_reformulate,
    )
    monkeypatch.setattr(unified_pipeline, "execute_retrieval_phase", fake_retrieval_phase)
    monkeypatch.setattr(unified_pipeline, "execute_generation_phase", fake_generation_phase)
    monkeypatch.setattr(unified_pipeline, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(
        unified_pipeline,
        "coordinate_standard_result_evidence",
        lambda result, resolved_request, *, retrieval_plan=None, coordinator=None: result,
    )

    result = await unified_pipeline.unified_rag_pipeline(
        query=resolved.query,
        sources=list(plan.sources),
        top_k=plan.top_k,
        search_mode=plan.search_mode,
        enable_generation=True,
        enable_query_classification=True,
        enable_reranking=False,
        resolved_request=resolved,
        retrieval_plan=plan,
    )

    assert result.generated_answer == "effective answer"
    assert seen["classifier_query"] == "What changed?"
    assert seen["retrieval_resolved"] is resolved
    assert seen["generation_resolved"] is resolved
    assert seen["generation_resolved"].query == "effective reformulated query"
    assert seen["generation_resolved"].payload["query"] == "effective reformulated query"
    assert seen["retrieval_plan"].query == "effective reformulated query"
    assert seen["generation_plan"].query == "effective reformulated query"

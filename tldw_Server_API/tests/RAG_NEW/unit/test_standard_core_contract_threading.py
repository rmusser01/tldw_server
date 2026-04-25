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
            documents=[],
            sources=[],
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

    def fake_coordinate(result, resolved_request, *, coordinator=None):
        seen["coordinator_resolved"] = resolved_request
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

    assert result["answer"] == "generated answer"
    assert seen["retrieval_resolved"] is resolved
    assert seen["retrieval_plan"] is plan
    assert seen["generation_resolved"] is resolved
    assert seen["generation_plan"] is plan
    assert seen["coordinator_resolved"] is resolved


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

    monkeypatch.setattr(unified_pipeline, "execute_retrieval_phase", fake_retrieval_phase)
    monkeypatch.setattr(unified_pipeline, "execute_generation_phase", fake_generation_phase)
    monkeypatch.setattr(
        unified_pipeline,
        "coordinate_standard_result_evidence",
        lambda result, resolved_request, *, coordinator=None: result,
    )

    result = await unified_pipeline.unified_rag_pipeline(
        query="legacy query",
        top_k=5,
        search_mode="fts",
        enable_generation=True,
    )

    assert result["answer"] == "legacy answer"
    assert seen["retrieval_resolved"] is seen["generation_resolved"]
    assert seen["retrieval_plan"] is seen["generation_plan"]
    assert seen["retrieval_resolved"].query == "legacy query"
    assert seen["retrieval_plan"].top_k == 5

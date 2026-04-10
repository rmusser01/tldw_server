import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.request_bundle import (
    build_request_bundle,
)
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan


pytestmark = pytest.mark.unit


def test_build_request_bundle_resolves_shared_payload_and_retrieval_plan(
) -> None:
    request = UnifiedRAGRequest(
        query="bundle",
        corpus="tenant-a",
        user_id="single_user",
        feedback_user_id="single_user",
        search_mode="vector",
        top_k=9,
        min_score=0.2,
    )

    def _pipeline_kwargs_builder(*, resolved_request, retrieval_plan) -> dict[str, object]:
        return {
            "query": resolved_request.query,
            "sources": list(retrieval_plan.sources),
            "index_namespace": retrieval_plan.index_namespace,
            "user_id": resolved_request.user_id,
            "feedback_user_id": resolved_request.feedback_user_id,
        }

    bundle = build_request_bundle(
        request=request,
        current_user=None,
        pipeline_kwargs_builder=_pipeline_kwargs_builder,
        resolve_request_kwargs={"single_user_id_resolver": lambda: 7},
    )

    assert bundle.resolved_request.index_namespace == "tenant-a"
    assert bundle.retrieval_plan.search_mode == "vector"
    assert bundle.retrieval_plan.top_k == 9
    assert bundle.retrieval_plan.min_score == 0.2
    assert bundle.pipeline_kwargs["query"] == "bundle"
    assert bundle.pipeline_kwargs["sources"] == ["media_db"]
    assert bundle.pipeline_kwargs["index_namespace"] == "tenant-a"
    assert bundle.pipeline_kwargs["user_id"] == "7"
    assert bundle.pipeline_kwargs["feedback_user_id"] == "7"


def test_build_request_bundle_injects_core_contracts_into_pipeline_kwargs() -> None:
    resolved = ResolvedRAGRequest(
        query="bundle seam",
        strategy="standard",
        payload={"query": "bundle seam", "sources": ["notes"]},
        index_namespace="tenant-z",
        rag_profile=None,
        user_id="12",
        feedback_user_id="12",
    )
    plan = RetrievalPlan(
        query="bundle seam",
        sources=("notes",),
        search_mode="vector",
        top_k=3,
        min_score=0.1,
        index_namespace="tenant-z",
    )

    bundle = build_request_bundle(
        request={"query": "bundle seam"},
        current_user=None,
        resolve_request_fn=lambda *args, **kwargs: resolved,  # noqa: ARG005
        build_retrieval_plan_fn=lambda _resolved: plan,
        pipeline_kwargs_builder=lambda **kwargs: {"query": kwargs["resolved_request"].query},
    )

    assert bundle.pipeline_kwargs["query"] == "bundle seam"
    assert bundle.pipeline_kwargs["resolved_request"] is resolved
    assert bundle.pipeline_kwargs["retrieval_plan"] is plan


def test_build_request_bundle_overwrites_builder_mismatched_core_contracts() -> None:
    canonical_resolved = ResolvedRAGRequest(
        query="canonical",
        strategy="standard",
        payload={"query": "canonical", "sources": ["notes"]},
        index_namespace="tenant-canonical",
        rag_profile=None,
        user_id="21",
        feedback_user_id="21",
    )
    canonical_plan = RetrievalPlan(
        query="canonical",
        sources=("notes",),
        search_mode="vector",
        top_k=4,
        min_score=0.2,
        index_namespace="tenant-canonical",
    )
    mismatched_resolved = ResolvedRAGRequest(
        query="mismatched",
        strategy="standard",
        payload={"query": "mismatched", "sources": ["media_db"]},
        index_namespace="tenant-wrong",
        rag_profile=None,
        user_id="999",
        feedback_user_id="999",
    )
    mismatched_plan = RetrievalPlan(
        query="mismatched",
        sources=("media_db",),
        search_mode="fts",
        top_k=99,
        min_score=0.0,
        index_namespace="tenant-wrong",
    )

    bundle = build_request_bundle(
        request={"query": "canonical"},
        current_user=None,
        resolve_request_fn=lambda *args, **kwargs: canonical_resolved,  # noqa: ARG005
        build_retrieval_plan_fn=lambda _resolved: canonical_plan,
        pipeline_kwargs_builder=lambda **kwargs: {
            "query": kwargs["resolved_request"].query,
            "resolved_request": mismatched_resolved,
            "retrieval_plan": mismatched_plan,
        },
    )

    assert bundle.pipeline_kwargs["query"] == "canonical"
    assert bundle.pipeline_kwargs["resolved_request"] is canonical_resolved
    assert bundle.pipeline_kwargs["retrieval_plan"] is canonical_plan

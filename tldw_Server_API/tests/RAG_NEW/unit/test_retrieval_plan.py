import pytest

from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import build_retrieval_plan


pytestmark = pytest.mark.unit


def test_build_retrieval_plan_centralizes_namespace_and_sources() -> None:
    resolved = ResolvedRAGRequest(
        query="retrieval plan",
        strategy="standard",
        payload={
            "query": "retrieval plan",
            "sources": ["media_db", "notes"],
            "search_mode": "hybrid",
            "top_k": 5,
            "min_score": 0.25,
        },
        index_namespace="tenant-a",
        rag_profile=None,
        user_id="9",
        feedback_user_id="9",
    )

    plan = build_retrieval_plan(resolved)

    assert plan.index_namespace == "tenant-a"
    assert plan.sources == ("media_db", "notes")
    assert plan.collection_names["media_db"] == "user_9_media_embeddings"
    assert plan.collection_names["notes"] == "user_9_notes_embeddings"
    assert plan.search_mode == "hybrid"
    assert plan.top_k == 5
    assert plan.min_score == 0.25


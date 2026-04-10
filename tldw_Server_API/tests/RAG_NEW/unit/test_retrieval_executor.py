import pytest

from tldw_Server_API.app.core.RAG.rag_service.evidence_models import RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_executor import execute_retrieval_phase
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_execute_retrieval_phase_returns_retrieved_evidence_from_plan():
    plan = RetrievalPlan(
        query="core executor",
        sources=("media_db",),
        search_mode="hybrid",
        top_k=3,
        min_score=0.0,
        index_namespace="tenant-a",
        collection_names={"media_db": "user_7_media_embeddings"},
    )
    resolved = ResolvedRAGRequest(
        query="core executor",
        strategy="standard",
        payload={"search_mode": "hybrid", "top_k": 3},
        index_namespace="tenant-a",
        rag_profile=None,
        user_id="7",
        feedback_user_id="7",
    )

    class StubRetriever:
        async def retrieve(self, *args, **kwargs):
            return [
                Document(
                    id="doc-1",
                    content="retrieved",
                    metadata={},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    evidence = await execute_retrieval_phase(
        resolved_request=resolved,
        retrieval_plan=plan,
        retriever=StubRetriever(),
    )

    assert isinstance(evidence, RetrievedEvidence)
    assert evidence.documents[0].id == "doc-1"
    assert evidence.metadata["retrieval_plan"]["index_namespace"] == "tenant-a"

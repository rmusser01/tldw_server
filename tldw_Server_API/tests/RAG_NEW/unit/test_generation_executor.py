import pytest

from tldw_Server_API.app.core.RAG.rag_service.evidence_models import DerivedEvidence, RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.generation_executor import execute_generation_phase
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.result_model import RAGResult


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_execute_generation_phase_builds_rag_result_from_derived_evidence():
    resolved = ResolvedRAGRequest(
        query="summarize",
        strategy="standard",
        payload={"enable_generation": True, "generation_prompt": "concise"},
        index_namespace="tenant-a",
        rag_profile=None,
        user_id="7",
        feedback_user_id="7",
    )
    derived = DerivedEvidence(
        retrieved=RetrievedEvidence(documents=[], metadata={"verification_report": {"ok": True}}),
        documents=[{"id": "doc-1", "content": "evidence"}],
        metadata={"chunk_citations": [{"id": "doc-1"}]},
        citations=[{"id": "doc-1"}],
        verification_report={"ok": True},
    )

    async def fake_generate_answer(**kwargs):
        assert kwargs["context"] == "writer context"
        return {
            "answer": "short answer",
            "provider": "stub-provider",
            "model": "stub-model",
            "tokens_used": 17,
            "generation_time": 0.25,
            "metadata": {"nested": "value"},
        }

    result = await execute_generation_phase(
        resolved_request=resolved,
        retrieval_plan=RetrievalPlan(
            query="summarize",
            sources=("media_db",),
            search_mode="hybrid",
            top_k=5,
            min_score=0.0,
            index_namespace="tenant-a",
        ),
        derived_evidence=derived,
        generate_answer_fn=fake_generate_answer,
        generation_context="writer context",
    )

    assert isinstance(result, RAGResult)
    assert result.generated_answer == "short answer"
    assert result.chunk_citations == [{"id": "doc-1"}]
    assert result.verification_report == {"ok": True}
    assert result.metadata["provider"] == "stub-provider"
    assert result.metadata["model"] == "stub-model"
    assert result.metadata["tokens_used"] == 17
    assert result.metadata["generation_time"] == 0.25
    assert result.metadata["nested"] == "value"

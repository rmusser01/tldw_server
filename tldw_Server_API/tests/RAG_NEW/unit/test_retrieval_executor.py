import pytest

from tldw_Server_API.app.core.RAG.rag_service.evidence_models import RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_executor import execute_retrieval_phase
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_execute_retrieval_phase_uses_retrieve_from_plan_when_available():
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
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def retrieve_from_plan(self, retrieval_plan, **kwargs):
            self.calls.append({"retrieval_plan": retrieval_plan, **kwargs})
            return [
                Document(
                    id="doc-1",
                    content="retrieved",
                    metadata={},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ]

    retriever = StubRetriever()
    evidence = await execute_retrieval_phase(resolved_request=resolved, retrieval_plan=plan, retriever=retriever)

    assert isinstance(evidence, RetrievedEvidence)
    assert evidence.documents[0].id == "doc-1"
    assert evidence.metadata["retrieval_plan"]["index_namespace"] == "tenant-a"
    assert retriever.calls[0]["retrieval_plan"] is plan
    assert retriever.calls[0]["config"] is None
    assert retriever.calls[0]["allowed_media_ids"] is None
    assert retriever.calls[0]["allowed_note_ids"] is None


@pytest.mark.asyncio
async def test_execute_retrieval_phase_forwards_legacy_kwargs_to_non_multi_database_retriever():
    plan = RetrievalPlan(
        query="legacy executor",
        sources=("notes", "media_db"),
        search_mode="vector",
        top_k=5,
        min_score=0.25,
        index_namespace="tenant-b",
        collection_names={
            "media_db": "user_7_media_embeddings",
            "notes": "user_7_notes_embeddings",
        },
    )
    resolved = ResolvedRAGRequest(
        query="legacy executor",
        strategy="standard",
        payload={"search_mode": "vector", "top_k": 5},
        index_namespace="tenant-b",
        rag_profile=None,
        user_id="7",
        feedback_user_id="7",
    )

    class StubRetriever:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        async def retrieve(self, *args, **kwargs):
            self.calls.append({"args": args, "kwargs": kwargs})
            return [
                Document(
                    id="doc-legacy",
                    content="retrieved",
                    metadata={},
                    source=DataSource.NOTES,
                    score=0.8,
                )
            ]

    retriever = StubRetriever()
    evidence = await execute_retrieval_phase(
        resolved_request=resolved,
        retrieval_plan=plan,
        retriever=retriever,
        retrieval_config={"max_results": 5},
        allowed_media_ids=[1, 2],
        allowed_note_ids=["note-1"],
    )

    assert isinstance(evidence, RetrievedEvidence)
    call = retriever.calls[0]["kwargs"]
    assert call["query"] == "legacy executor"
    assert call["sources"] == [DataSource.NOTES, DataSource.MEDIA_DB]
    assert call["index_namespace"] == "tenant-b"
    assert call["config"] == {"max_results": 5}
    assert call["retrieval_plan"] is plan
    assert call["allowed_media_ids"] == [1, 2]
    assert call["allowed_note_ids"] == ["note-1"]
    assert evidence.documents[0].id == "doc-legacy"

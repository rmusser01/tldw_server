from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.RAG.rag_service.evidence_models import (
    DerivedEvidence,
    RetrievedEvidence,
)
from tldw_Server_API.app.core.RAG.rag_service.post_retrieval_coordinator import (
    PostRetrievalCoordinator,
    coordinate_standard_result_evidence,
)
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult


pytestmark = pytest.mark.unit


def test_derive_evidence_preserves_retrieved_documents_and_tracks_derived_docs() -> None:
    retrieved_documents = [
        SimpleNamespace(id="retrieved-1", content="Authoritative primary evidence"),
        SimpleNamespace(id="retrieved-2", content="Authoritative corroborating evidence"),
    ]
    derived_document = {
        "id": "derived-1",
        "content": "Derived synthesis over the retrieved set",
        "metadata": {"kind": "synthesis"},
    }
    retrieved = RetrievedEvidence(
        documents=retrieved_documents,
        metadata={
            "route": {"search_mode": "hybrid", "sources": ["media_db"]},
            "chunk_citations": [{"type": "chunk", "id": "chunk-1"}],
            "verification_report": {"status": "verified"},
        },
    )

    coordinator = PostRetrievalCoordinator()

    coordinated = coordinator.derive_evidence(
        None,
        retrieved,
        enable_citations=True,
        enable_verification=True,
        derived_documents=[derived_document],
        derived_from_document_ids=["retrieved-1", "retrieved-2"],
    )

    assert coordinated.retrieved is retrieved
    assert coordinated.documents == [*retrieved_documents, derived_document]
    assert coordinated.metadata == retrieved.metadata
    assert coordinated.citations == [{"type": "chunk", "id": "chunk-1"}]
    assert coordinated.verification_report == {"status": "verified"}
    assert coordinated.derived_from_document_ids == ("retrieved-1", "retrieved-2")


def test_coordinate_standard_result_evidence_routes_through_post_retrieval_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved_document = SimpleNamespace(id="retrieved-1", content="Primary evidence")
    derived_document = {"id": "derived-1", "content": "Derived evidence"}
    resolved_request = SimpleNamespace(query="standard evidence coordination", strategy="standard")
    retrieved = RetrievedEvidence(
        documents=[retrieved_document],
        metadata={
            "chunk_citations": [{"type": "chunk", "id": "chunk-1"}],
            "verification_report": {"status": "pre-existing"},
        },
    )

    class StubCoordinator:
        def derive_evidence(self, request_contract, retrieved_evidence, **kwargs):
            assert request_contract is resolved_request
            assert retrieved_evidence == retrieved
            assert kwargs["enable_citations"] is True
            assert kwargs["enable_verification"] is True
            return DerivedEvidence(
                retrieved=retrieved_evidence,
                documents=[retrieved_document, derived_document],
                metadata={
                    **retrieved_evidence.metadata,
                    "verification_report": {"status": "verified"},
                },
                citations=[{"type": "chunk", "id": "chunk-1"}],
                verification_report={"status": "verified"},
                derived_from_document_ids=("retrieved-1",),
            )

    result = UnifiedSearchResult(
        documents=[retrieved_document],
        query="standard evidence coordination",
        metadata={
            "chunk_citations": [{"type": "chunk", "id": "chunk-1"}],
            "verification_report": {"status": "pre-existing"},
        },
    )

    coordinated = coordinate_standard_result_evidence(
        result,
        resolved_request,
        coordinator=StubCoordinator(),
    )

    assert coordinated.documents == [retrieved_document, derived_document]
    assert coordinated.metadata["chunk_citations"] == [{"type": "chunk", "id": "chunk-1"}]
    assert coordinated.metadata["verification_report"] == {"status": "verified"}
    assert coordinated.metadata["derived_from_document_ids"] == ["retrieved-1"]

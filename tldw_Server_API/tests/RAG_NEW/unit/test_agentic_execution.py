import pytest

from tldw_Server_API.app.core.RAG.rag_service.agentic_execution import build_agentic_derived_evidence
from tldw_Server_API.app.core.RAG.rag_service.evidence_models import DerivedEvidence, RetrievedEvidence


def test_build_agentic_derived_evidence_tracks_actual_lineage_only():
    retrieved = RetrievedEvidence(
        documents=[
            {"id": "doc-1", "content": "one"},
            {"id": "doc-2", "content": "two"},
        ],
        metadata={},
    )

    derived = build_agentic_derived_evidence(
        retrieved_evidence=retrieved,
        synthetic_chunk={"id": "synthetic", "content": "merged"},
        derived_from_document_ids=("doc-2",),
        coarse_docs_window=[{"id": "doc-1"}, {"id": "doc-2"}],
    )

    assert isinstance(derived, DerivedEvidence)
    assert derived.derived_from_document_ids == ("doc-2",)
    assert derived.metadata["coarse_docs"] == [{"id": "doc-1"}, {"id": "doc-2"}]

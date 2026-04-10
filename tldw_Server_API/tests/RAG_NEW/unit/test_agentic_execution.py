import pytest

import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as agentic_chunker
from tldw_Server_API.app.core.RAG.rag_service.agentic_execution import build_agentic_derived_evidence
from tldw_Server_API.app.core.RAG.rag_service.evidence_models import DerivedEvidence, RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


pytestmark = pytest.mark.unit


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


def test_agentic_toolbox_open_section_prefers_db_structure_lookup(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakeDb:
        def lookup_section_by_heading(self, media_id: int, heading: str):
            captured["media_id"] = media_id
            captured["heading"] = heading
            return (7, 21, "Results")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.rag_enable_structure_index",
        lambda default=True: True,
    )
    monkeypatch.setattr(agentic_chunker, "_get_media_db_for_structure", lambda: FakeDb())

    doc = Document(
        id="doc-1",
        content="# Intro\nalpha\n\n# Results\nbeta\n",
        metadata={"title": "Paper", "media_id": 42},
        source=DataSource.MEDIA_DB,
    )
    toolbox = agentic_chunker.AgenticToolbox([doc], agentic_chunker.AgenticConfig(enable_section_index=True))

    assert toolbox.open_section(doc, "Results") == (7, 21)
    assert captured == {"media_id": 42, "heading": "Results"}

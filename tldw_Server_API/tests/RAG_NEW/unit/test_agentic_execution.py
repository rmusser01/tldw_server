import pytest

import tldw_Server_API.app.core.RAG.rag_service.agentic_execution as agentic_execution
from tldw_Server_API.app.core.RAG.rag_service.agentic_execution import (
    AgenticConfig,
    AgenticToolbox,
    build_agentic_derived_evidence,
    build_agentic_execution_context,
)
from tldw_Server_API.app.core.RAG.rag_service.evidence_models import DerivedEvidence, RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
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
    monkeypatch.setattr(agentic_execution, "_get_media_db_for_structure", lambda: FakeDb())

    doc = Document(
        id="doc-1",
        content="# Intro\nalpha\n\n# Results\nbeta\n",
        metadata={"title": "Paper", "media_id": 42},
        source=DataSource.MEDIA_DB,
    )
    toolbox = AgenticToolbox([doc], AgenticConfig(enable_section_index=True))

    assert toolbox.open_section(doc, "Results") == (7, 21)
    assert captured == {"media_id": 42, "heading": "Results"}


def test_build_agentic_execution_context_derives_effective_payload_and_config() -> None:
    resolved_request = ResolvedRAGRequest(
        query="agentic config from canonical contracts",
        strategy="agentic",
        payload={
            "query": "agentic config from canonical contracts",
            "strategy": "agentic",
            "sources": ["notes"],
            "search_mode": "vector",
            "top_k": 5,
            "min_score": 0.15,
            "index_namespace": "tenant-a",
            "agentic_max_tool_calls": 6,
            "agentic_enable_tools": True,
            "agentic_coverage_target": 0.92,
            "agentic_enable_metrics": False,
            "agentic_debug_trace": False,
            "debug_mode": True,
        },
        index_namespace="tenant-a",
        rag_profile="balanced",
        user_id="1",
        feedback_user_id="1",
    )
    retrieval_plan = RetrievalPlan(
        query=resolved_request.query,
        sources=("notes",),
        search_mode="vector",
        top_k=5,
        min_score=0.15,
        index_namespace="tenant-a",
    )

    effective_payload, agentic_config = build_agentic_execution_context(
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
        payload_override={
            **resolved_request.payload,
            "agentic_top_k_docs": 4,
            "agentic_window_chars": 1400,
        },
    )

    assert effective_payload["sources"] == ["notes"]
    assert effective_payload["search_mode"] == "vector"
    assert effective_payload["top_k"] == 5
    assert effective_payload["min_score"] == 0.15
    assert effective_payload["index_namespace"] == "tenant-a"
    assert agentic_config.top_k_docs == 4
    assert agentic_config.window_chars == 1400
    assert agentic_config.max_tool_calls == 6
    assert agentic_config.enable_tools is True
    assert agentic_config.coverage_target == 0.92
    assert agentic_config.enable_metrics is False
    assert agentic_config.debug_trace is True

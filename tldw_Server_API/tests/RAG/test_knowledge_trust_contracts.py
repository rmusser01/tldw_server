import pytest

from tldw_Server_API.app.core.RAG.rag_service.trust_contracts import (
    classify_knowledge_answer_trust,
)
from tldw_Server_API.app.core.RAG.rag_service.response_mapping import (
    rag_result_to_response,
)
from tldw_Server_API.app.core.RAG.rag_service.result_model import RAGResult

pytestmark = pytest.mark.unit


def test_uncited_answer_is_degraded():
    trust = classify_knowledge_answer_trust(
        answer="Unsupported answer.",
        documents=[
            {
                "id": "doc-1",
                "content": "Evidence",
                "metadata": {"source_status": "searched"},
            }
        ],
        citations=[],
        web_fallback_used=False,
    )

    assert trust["state"] == "uncited_degraded_answer"
    assert trust["reason_codes"] == ["missing_citations"]
    assert trust["evidence_origin"] == "local_library"


def test_cited_answer_requires_inspectable_evidence():
    trust = classify_knowledge_answer_trust(
        answer="Grounded answer [1].",
        documents=[
            {
                "id": "doc-1",
                "content": "",
                "metadata": {
                    "source_status": "unavailable",
                    "unavailable_reason": "deleted_or_unavailable",
                },
            }
        ],
        citations=[{"index": 1, "document_id": "doc-1"}],
        web_fallback_used=False,
    )

    assert trust["state"] == "no_answer_insufficient_evidence"
    assert trust["reason_codes"] == ["missing_inspectable_evidence"]
    assert trust["evidence_origin"] == "local_library"


def test_cited_answer_requires_citations_to_map_to_returned_sources():
    trust = classify_knowledge_answer_trust(
        answer="Grounded answer [1].",
        documents=[
            {
                "id": "doc-1",
                "content": "Inspectable evidence.",
                "metadata": {"source_status": "searched"},
            }
        ],
        citations=[{"index": 1, "document_id": "missing-doc"}],
        web_fallback_used=False,
    )

    assert trust["state"] == "uncited_degraded_answer"
    assert trust["reason_codes"] == ["citation_source_not_returned"]


def test_chunk_citation_ids_map_to_returned_sources():
    trust = classify_knowledge_answer_trust(
        answer="Grounded answer [1].",
        documents=[
            {
                "id": "late_chunk:1:0",
                "content": "Inspectable evidence.",
                "metadata": {
                    "chunk_id": "late_chunk:1:0",
                    "media_id": "1",
                    "source_status": "searched",
                },
            }
        ],
        citations=[
            {
                "type": "academic",
                "formatted": "(n.d.). *Grounded source*.",
            },
            {
                "index": 1,
                "chunk_id": "late_chunk:1:0",
                "source_document_id": "1",
            }
        ],
        web_fallback_used=False,
    )

    assert trust["state"] == "cited_answer"
    assert trust["reason_codes"] == []


def test_inline_citation_markers_map_by_returned_source_index():
    trust = classify_knowledge_answer_trust(
        answer="Grounded answer [1].",
        documents=[
            {
                "id": "doc-1",
                "content": "Inspectable evidence.",
                "metadata": {"source_status": "searched"},
            }
        ],
        citations=[],
        web_fallback_used=False,
    )

    assert trust["state"] == "cited_answer"
    assert trust["reason_codes"] == []


def test_web_fallback_origin_is_preserved():
    trust = classify_knowledge_answer_trust(
        answer="Grounded answer [1].",
        documents=[
            {
                "id": "web-1",
                "content": "Web evidence.",
                "metadata": {
                    "source_status": "searched",
                    "evidence_origin": "web_fallback",
                },
            }
        ],
        citations=[{"index": 1, "document_id": "web-1"}],
        web_fallback_used=True,
    )

    assert trust["state"] == "cited_answer"
    assert trust["reason_codes"] == []
    assert trust["evidence_origin"] == "web_fallback"


def test_mixed_origin_when_local_and_web_evidence_are_returned():
    trust = classify_knowledge_answer_trust(
        answer="Grounded answer [1] [2].",
        documents=[
            {
                "id": "local-1",
                "content": "Local evidence.",
                "metadata": {
                    "source_status": "searched",
                    "evidence_origin": "local_library",
                },
            },
            {
                "id": "web-1",
                "content": "Web evidence.",
                "metadata": {
                    "source_status": "searched",
                    "evidence_origin": "web_fallback",
                },
            },
        ],
        citations=[
            {"index": 1, "document_id": "local-1"},
            {"index": 2, "document_id": "web-1"},
        ],
        web_fallback_used=True,
    )

    assert trust["state"] == "cited_answer"
    assert trust["evidence_origin"] == "mixed"


def test_no_documents_without_web_fallback_is_no_results():
    trust = classify_knowledge_answer_trust(
        answer=None,
        documents=[],
        citations=[],
        web_fallback_used=False,
    )

    assert trust["state"] == "no_results"
    assert trust["reason_codes"] == ["no_evidence"]
    assert trust["evidence_origin"] == "local_library"


def test_response_mapping_attaches_knowledge_trust_metadata():
    response = rag_result_to_response(
        RAGResult(
            query="What changed?",
            documents=[
                {
                    "id": "web-1",
                    "content": "Inspectable web evidence.",
                    "metadata": {
                        "source_status": "searched",
                        "evidence_origin": "web_fallback",
                    },
                }
            ],
            citations=[{"index": 1, "document_id": "web-1"}],
            generated_answer="The source explains the change [1].",
            metadata={"web_fallback": {"triggered": True}},
        )
    )

    assert response.metadata["knowledge_trust"] == {
        "state": "cited_answer",
        "reason_codes": [],
        "evidence_origin": "web_fallback",
    }

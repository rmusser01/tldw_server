import pytest

from tldw_Server_API.app.core.RAG.rag_service.response_mapping import (
    rag_result_from_unified_search_result,
    rag_result_to_response,
)
from tldw_Server_API.app.core.RAG.rag_service.result_model import RAGResult
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult


pytestmark = pytest.mark.unit


def test_rag_result_mapping_exposes_declared_response_fields() -> None:
    research_summary = {"total_iterations": 2, "completed": True}
    suggestions = ["Follow up one", "Follow up two"]
    images = [{"title": "img", "url": "https://example.com/image"}]
    videos = [{"title": "vid", "url": "https://example.com/video"}]

    unified = UnifiedSearchResult(
        documents=[
            Document(
                id="doc-1",
                content="evidence",
                metadata={"title": "Doc 1"},
                source=DataSource.WEB_CONTENT,
                score=0.91,
            )
        ],
        query="mapping contract",
        expanded_queries=["mapping contract"],
        metadata={
            "research": research_summary,
            "suggestions": suggestions,
            "images": images,
            "videos": videos,
            "chunk_citations": [{"chunk_id": "c1"}],
            "query_classification": {"intent": "research"},
            "reformulated_query": "mapping contract details",
            "retrieval_metrics": {"mrr": 1.0},
            "faithfulness": {"score": 0.9},
            "verification_report": {"supported": 2, "unsupported": 0},
        },
        timings={"retrieval": 0.05},
        generated_answer="Answer text",
    )

    mapped = rag_result_from_unified_search_result(unified)
    response = rag_result_to_response(mapped)

    assert mapped.chunk_citations == [{"chunk_id": "c1"}]
    assert mapped.query_classification == {"intent": "research"}
    assert mapped.reformulated_query == "mapping contract details"
    assert mapped.research_summary == research_summary
    assert mapped.retrieval_metrics == {"mrr": 1.0}
    assert mapped.faithfulness == {"score": 0.9}
    assert mapped.verification_report == {"supported": 2, "unsupported": 0}
    assert response.query == "mapping contract"
    assert response.documents[0]["id"] == "doc-1"
    assert response.chunk_citations == [{"chunk_id": "c1"}]
    assert response.research_summary == research_summary
    assert response.suggestions == suggestions
    assert response.images == images
    assert response.videos == videos
    assert response.query_classification == {"intent": "research"}
    assert response.reformulated_query == "mapping contract details"
    assert response.retrieval_metrics == {"mrr": 1.0}
    assert response.faithfulness == {"score": 0.9}
    assert response.verification_report == {"supported": 2, "unsupported": 0}


def test_rag_result_mapping_sets_optional_response_fields_to_none_when_metadata_missing() -> None:
    unified = UnifiedSearchResult(documents=[], query="empty metadata")
    mapped = rag_result_from_unified_search_result(unified)
    response = rag_result_to_response(mapped)

    assert response.research_summary is None
    assert response.suggestions is None
    assert response.images is None
    assert response.videos is None


def test_rag_result_mapping_preserves_top_level_generation_fields() -> None:
    result = RAGResult(
        documents=[
            Document(
                id="doc-1",
                content="evidence",
                metadata={"title": "Doc 1"},
                source=DataSource.WEB_CONTENT,
                score=0.91,
            )
        ],
        query="executor output",
        metadata={"model": "stub"},
        chunk_citations=[{"id": "doc-1"}],
        verification_report={"ok": True},
        generated_answer="Generated answer",
    )

    mapped = rag_result_from_unified_search_result(result)
    response = rag_result_to_response(mapped)

    assert mapped.chunk_citations == [{"id": "doc-1"}]
    assert mapped.verification_report == {"ok": True}
    assert response.chunk_citations == [{"id": "doc-1"}]
    assert response.verification_report == {"ok": True}


def test_rag_result_mapping_preserves_dict_shaped_answer_field() -> None:
    mapped = rag_result_from_unified_search_result(
        {
            "documents": [],
            "query": "structured answer",
            "answer": {"summary": "structured", "confidence": 0.9},
            "metadata": {},
        }
    )

    assert mapped.generated_answer == {"summary": "structured", "confidence": 0.9}

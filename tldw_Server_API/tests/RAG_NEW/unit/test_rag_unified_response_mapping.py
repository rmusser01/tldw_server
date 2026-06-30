import pytest

from tldw_Server_API.app.core.RAG.rag_service.response_mapping import (
    rag_result_from_unified_search_result,
    rag_result_to_response,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult


pytestmark = pytest.mark.unit


def test_convert_unified_result_maps_round2_search_agent_response_fields():
    research_summary = {
        "total_iterations": 2,
        "total_results": 3,
        "completed": True,
        "url_dedup": {"urls_seen": 2, "duplicates_merged": 1},
    }
    suggestions = ["What are the trade-offs?", "How do I benchmark this?"]
    images = [
        {
            "title": "Architecture diagram",
            "url": "https://example.com/diagram",
            "thumbnail_url": "https://example.com/diagram-thumb.jpg",
        }
    ]
    videos = [
        {
            "title": "Deep dive video",
            "url": "https://youtube.com/watch?v=abcdefghijk",
            "thumbnail_url": "https://img.youtube.com/vi/abcdefghijk/mqdefault.jpg",
        }
    ]

    result = UnifiedSearchResult(
        documents=[
            Document(
                id="doc-1",
                content="Evidence snippet",
                metadata={"title": "Doc 1", "url": "https://example.com/doc-1"},
                source=DataSource.WEB_CONTENT,
                score=0.91,
            )
        ],
        query="round2 endpoint mapping",
        expanded_queries=["round2 endpoint mapping"],
        metadata={
            "research": research_summary,
            "suggestions": suggestions,
            "images": images,
            "videos": videos,
        },
        timings={"retrieval": 0.05},
        generated_answer="Answer text",
    )

    rag_result = rag_result_from_unified_search_result(result)
    converted = rag_result_to_response(rag_result)

    assert converted.research_summary == research_summary
    assert converted.suggestions == suggestions
    assert converted.images == images
    assert converted.videos == videos
    assert converted.metadata.get("research") == research_summary
    assert converted.metadata.get("suggestions") == suggestions
    assert converted.metadata.get("images") == images
    assert converted.metadata.get("videos") == videos


def test_convert_unified_result_sets_round2_fields_to_none_when_not_present():
    result = UnifiedSearchResult(documents=[], query="empty metadata case")

    converted = rag_result_to_response(rag_result_from_unified_search_result(result))

    assert converted.research_summary is None
    assert converted.suggestions is None
    assert converted.images is None
    assert converted.videos is None


def test_convert_dict_shaped_result_preserves_declared_response_fields():
    result = {
        "documents": [
            {
                "id": "doc-1",
                "content": "Dict-backed evidence",
                "metadata": {"title": "Doc 1"},
                "score": 0.61,
            }
        ],
        "query": "dict fallback result",
        "expanded_queries": ["dict fallback result"],
        "metadata": {
            "chunk_citations": [{"type": "chunk", "id": "chunk-1"}],
            "verification_report": {"status": "verified"},
        },
        "timings": {"retrieval": 0.03},
        "citations": [{"type": "web", "id": "cite-1"}],
        "feedback_id": "fb-1",
        "generated_answer": "Fallback answer",
        "cache_hit": True,
        "errors": ["soft warning"],
        "security_report": {"status": "safe"},
        "total_time": 0.42,
    }

    converted = rag_result_to_response(rag_result_from_unified_search_result(result))

    assert converted.query == "dict fallback result"
    assert converted.documents[0]["id"] == "doc-1"
    assert converted.generated_answer == "Fallback answer"
    assert converted.errors == ["soft warning"]
    assert converted.verification_report == {"status": "verified"}
    assert converted.chunk_citations == [{"type": "chunk", "id": "chunk-1"}]


def test_endpoint_module_no_longer_exports_response_mapping_wrapper():
    import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep

    assert not hasattr(rag_ep, "convert_result_to_response")

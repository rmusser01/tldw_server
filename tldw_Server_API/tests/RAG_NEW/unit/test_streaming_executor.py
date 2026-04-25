from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import stream_rag_events
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult


def _resolved_request(strategy: str = "standard") -> ResolvedRAGRequest:
    return ResolvedRAGRequest(
        query="stream query",
        strategy=strategy,
        payload={
            "query": "stream query",
            "strategy": strategy,
            "sources": ["media_db"],
            "search_mode": "hybrid",
            "top_k": 3,
            "min_score": 0.0,
            "enable_generation": True,
            "generation_provider": "test-provider",
            "generation_model": "test-model",
        },
        index_namespace=None,
        rag_profile=None,
        user_id="1",
        feedback_user_id="1",
    )


def _retrieval_plan() -> RetrievalPlan:
    return RetrievalPlan(
        query="stream query",
        sources=("media_db",),
        search_mode="hybrid",
        top_k=3,
        min_score=0.0,
        index_namespace=None,
        collection_names={"media_db": "user_1_media_embeddings"},
    )


@pytest.mark.asyncio
async def test_stream_rag_events_wraps_standard_stream_result_in_order():
    async def fake_standard_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        return UnifiedSearchResult(
            documents=[
                Document(
                    id="doc-1",
                    content="Context content",
                    metadata={"title": "Doc"},
                    source=DataSource.MEDIA_DB,
                    score=0.9,
                )
            ],
            query=str(kwargs.get("query", "")),
        )

    async def fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        async def _gen():
            yield "answer text"

        context.stream_generator = _gen()
        context.metadata = {"streaming": True}
        return context

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=_resolved_request("standard"),
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=fake_standard_pipeline,
            extra_context={"generate_streaming_response": fake_generate_streaming_response},
        )
    ]

    assert [event["type"] for event in events] == ["contexts", "reasoning", "delta"]  # nosec B101
    assert events[0]["contexts"][0]["id"] == "doc-1"  # nosec B101
    assert events[1]["plan"] == [  # nosec B101
        "Gather top-k contexts",
        "Rerank using strategy=flashrank",
        "Ground claims from sources",
        "Synthesize final answer",
    ]
    assert events[2]["text"] == "answer text"  # nosec B101


@pytest.mark.asyncio
async def test_stream_rag_events_emits_structured_error():
    async def failing_standard_pipeline(**kwargs: Any) -> UnifiedSearchResult:  # noqa: ARG001
        raise RuntimeError("stream failed")

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=_resolved_request("standard"),
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=failing_standard_pipeline,
        )
    ]

    assert [event["type"] for event in events] == ["error"]  # nosec B101
    assert events[0]["message"] == "stream failed"  # nosec B101

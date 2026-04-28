from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.streaming_executor as streaming_executor
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import (
    _PUBLIC_STREAM_ERROR_MESSAGE,
    stream_rag_events,
)
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


async def _fake_generate_streaming_response(context: Any, **kwargs: Any) -> Any:  # noqa: ARG001
    async def _gen():
        yield "answer text"

    context.stream_generator = _gen()
    context.metadata = {"streaming": True}
    return context


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

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=_resolved_request("standard"),
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=fake_standard_pipeline,
            extra_context={"generate_streaming_response": _fake_generate_streaming_response},
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
    async def empty_standard_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        return UnifiedSearchResult(documents=[], query=str(kwargs.get("query", "")))

    async def failing_generate_streaming_response(context: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("stream failed")

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=_resolved_request("standard"),
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=empty_standard_pipeline,
            extra_context={"generate_streaming_response": failing_generate_streaming_response},
        )
    ]

    assert [event["type"] for event in events] == ["contexts", "reasoning", "error"]  # nosec B101
    assert events[2] == {"type": "error", "message": _PUBLIC_STREAM_ERROR_MESSAGE}  # nosec B101


@pytest.mark.asyncio
async def test_stream_rag_events_continues_when_standard_prefetch_fails():
    async def failing_standard_pipeline(**kwargs: Any) -> UnifiedSearchResult:  # noqa: ARG001
        raise RuntimeError("retrieval backend unavailable")

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=_resolved_request("standard"),
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=failing_standard_pipeline,
            extra_context={"generate_streaming_response": _fake_generate_streaming_response},
        )
    ]

    assert [event["type"] for event in events] == ["contexts", "reasoning", "delta"]  # nosec B101
    assert events[0]["contexts"] == []  # nosec B101
    assert events[-1]["text"] == "answer text"  # nosec B101


@pytest.mark.asyncio
async def test_stream_rag_events_continues_with_empty_contexts_when_agentic_prefetch_fails():
    async def standard_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        return UnifiedSearchResult(
            documents=[
                Document(
                    id="standard-doc",
                    content="Standard context",
                    metadata={"title": "Standard"},
                    source=DataSource.MEDIA_DB,
                    score=0.7,
                )
            ],
            query=str(kwargs.get("query", "")),
        )

    async def failing_agentic_pipeline(**kwargs: Any) -> UnifiedSearchResult:  # noqa: ARG001
        raise RuntimeError("agentic backend unavailable")

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=_resolved_request("agentic"),
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=standard_pipeline,
            agentic_pipeline=failing_agentic_pipeline,
            extra_context={"generate_streaming_response": _fake_generate_streaming_response},
        )
    ]

    assert [event["type"] for event in events] == ["contexts", "reasoning", "delta"]  # nosec B101
    assert events[0]["contexts"] == []  # nosec B101
    assert events[-1]["text"] == "answer text"  # nosec B101


@pytest.mark.asyncio
async def test_stream_rag_events_skips_standard_prefetch_for_agentic_strategy():
    calls: list[str] = []

    async def standard_pipeline(**kwargs: Any) -> UnifiedSearchResult:  # noqa: ARG001
        calls.append("standard")
        return UnifiedSearchResult(documents=[], query="standard")

    async def agentic_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        calls.append("agentic")
        return UnifiedSearchResult(
            documents=[
                Document(
                    id="agentic-doc",
                    content="Agentic context",
                    metadata={"title": "Agentic"},
                    source=DataSource.MEDIA_DB,
                    score=0.8,
                )
            ],
            query=str(kwargs.get("query", "")),
            metadata={"agentic_metrics": {"steps": 1}},
        )

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=_resolved_request("agentic"),
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=standard_pipeline,
            agentic_pipeline=agentic_pipeline,
            extra_context={"generate_streaming_response": _fake_generate_streaming_response},
        )
    ]

    assert calls == ["agentic"]  # nosec B101
    assert [event["type"] for event in events] == ["plan", "contexts", "reasoning", "delta"]  # nosec B101
    assert events[1]["contexts"][0]["id"] == "agentic-doc"  # nosec B101


def test_generation_config_logs_config_load_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, object]] = []

    class FakeLogger:
        def opt(self, **kwargs: object) -> "FakeLogger":
            events.append(("opt", kwargs))
            return self

        def warning(self, message: str) -> None:
            events.append(("warning", message))

    def fail_load_and_log_configs() -> dict[str, Any]:
        raise RuntimeError("config unavailable")

    import tldw_Server_API.app.core.config as core_config

    monkeypatch.setattr(core_config, "load_and_log_configs", fail_load_and_log_configs)
    monkeypatch.setattr(streaming_executor, "logger", FakeLogger())

    config = streaming_executor._generation_config(payload={}, request_defaults={})

    assert config["provider"] == "openai"  # nosec B101
    assert config["model"] == "gpt-4o-mini"  # nosec B101
    assert any(name == "opt" for name, _ in events)  # nosec B101
    assert ("warning", "RAG streaming config load failed; using request/env defaults") in events  # nosec B101

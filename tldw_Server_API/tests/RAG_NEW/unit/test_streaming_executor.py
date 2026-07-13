import asyncio
from typing import Any
from types import SimpleNamespace

import pytest

import tldw_Server_API.app.core.RAG.rag_service.streaming_executor as streaming_executor
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
from tldw_Server_API.app.core.RAG.rag_service.generation import GenerationConfig, LLMGenerator
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


class _StreamingRuntime:
    def __init__(self) -> None:
        self.handle = SimpleNamespace(
            provider="test-provider",
            api_key="runtime-stream-key",
            app_config={"TestProvider": {"api_timeout": 9}},
            credentials_resolved=True,
        )
        self.marked: list[Any] = []

    async def resolve(self, provider: str) -> Any:
        return self.handle

    async def mark_used(self, handle: Any) -> None:
        self.marked.append(handle)


async def _consume_tracked_stream(
    monkeypatch: pytest.MonkeyPatch,
    upstream: Any,
    runtime: _StreamingRuntime,
) -> list[Any]:
    async def fake_chat_call(**kwargs: Any) -> Any:
        return upstream

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)
    tracked = await LLMGenerator(
        GenerationConfig(provider="test-provider", model="test-model", streaming=True)
    )._call_llm("prompt", credential_runtime=runtime)
    return [chunk async for chunk in tracked]


@pytest.mark.parametrize(
    "control_chunks",
    [
        [": keepalive\n\n", "event: message\n", "id: 7\n", "retry: 1000\n"],
        ["keepalive", "event: ping"],
        ["data: ping\n\n", "data: pong\n\n", "data: heartbeat\n\n", "data: keepalive\n\n"],
    ],
)
@pytest.mark.asyncio
async def test_stream_controls_then_failure_remain_unmarked(
    monkeypatch: pytest.MonkeyPatch,
    control_chunks: list[str],
) -> None:
    runtime = _StreamingRuntime()

    async def upstream() -> Any:
        for chunk in control_chunks:
            yield chunk
        raise ChatAuthenticationError("private upstream", provider="test-provider")

    with pytest.raises(ChatAuthenticationError):
        await _consume_tracked_stream(monkeypatch, upstream(), runtime)

    assert runtime.marked == []  # nosec B101


@pytest.mark.asyncio
async def test_stream_preoutput_failure_remains_unmarked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _StreamingRuntime()

    async def upstream() -> Any:
        if False:
            yield "unreachable"
        raise ChatAuthenticationError("private upstream", provider="test-provider")

    with pytest.raises(ChatAuthenticationError):
        await _consume_tracked_stream(monkeypatch, upstream(), runtime)

    assert runtime.marked == []  # nosec B101


@pytest.mark.asyncio
async def test_stream_controls_then_cancellation_remain_unmarked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _StreamingRuntime()

    async def upstream() -> Any:
        yield ": keepalive\n\n"
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await _consume_tracked_stream(monkeypatch, upstream(), runtime)

    assert runtime.marked == []  # nosec B101


@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.parametrize("terminal_kind", ["failure", "cancellation"])
@pytest.mark.parametrize(
    ("content", "expected_mark_count"),
    [
        ("ping", 0),
        ("pong", 0),
        ("heartbeat", 0),
        ("keepalive", 0),
        ("", 0),
        ("answer", 1),
    ],
)
@pytest.mark.asyncio
async def test_openai_delta_tracking_classifies_content_before_interruption(
    monkeypatch: pytest.MonkeyPatch,
    stream_kind: str,
    terminal_kind: str,
    content: str,
    expected_mark_count: int,
) -> None:
    runtime = _StreamingRuntime()
    chunk = {"choices": [{"delta": {"content": content}}]}

    def interrupt() -> None:
        if terminal_kind == "failure":
            raise ChatAuthenticationError("private upstream", provider="test-provider")
        raise asyncio.CancelledError

    if stream_kind == "async":
        async def async_upstream() -> Any:
            yield chunk
            interrupt()

        upstream: Any = async_upstream()
    else:
        def sync_upstream() -> Any:
            yield chunk
            interrupt()

        upstream = sync_upstream()

    expected_error = (
        ChatAuthenticationError if terminal_kind == "failure" else asyncio.CancelledError
    )
    with pytest.raises(expected_error):
        await _consume_tracked_stream(monkeypatch, upstream, runtime)

    assert runtime.marked == [runtime.handle] * expected_mark_count  # nosec B101


@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.asyncio
async def test_clean_empty_stream_marks_once(
    monkeypatch: pytest.MonkeyPatch,
    stream_kind: str,
) -> None:
    runtime = _StreamingRuntime()

    if stream_kind == "async":
        async def empty_async() -> Any:
            if False:
                yield "unreachable"

        upstream: Any = empty_async()
    else:
        upstream = iter(())

    chunks = await _consume_tracked_stream(monkeypatch, upstream, runtime)

    assert chunks == []  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101


@pytest.mark.asyncio
async def test_mixed_sse_controls_then_data_marks_once_before_later_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _StreamingRuntime()

    async def upstream() -> Any:
        yield "event: message\n"
        yield 'data: {"choices":[{"delta":{"content":"answer"}}]}'
        raise ChatAuthenticationError("private upstream", provider="test-provider")

    with pytest.raises(ChatAuthenticationError):
        await _consume_tracked_stream(monkeypatch, upstream(), runtime)

    assert runtime.marked == [runtime.handle]  # nosec B101


@pytest.mark.asyncio
async def test_plain_data_sse_content_marks_once_before_later_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _StreamingRuntime()

    async def upstream() -> Any:
        yield "data: legitimate provider text\n\n"
        raise ChatAuthenticationError("private upstream", provider="test-provider")

    with pytest.raises(ChatAuthenticationError):
        await _consume_tracked_stream(monkeypatch, upstream(), runtime)

    assert runtime.marked == [runtime.handle]  # nosec B101


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
async def test_streaming_generation_marks_partial_output_and_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class RecordingRuntime:
        def __init__(self) -> None:
            self.handle = SimpleNamespace(
                provider="test-provider",
                api_key="runtime-stream-key",
                app_config={"TestProvider": {"api_timeout": 9}},
                credentials_resolved=True,
            )
            self.resolved: list[str] = []
            self.marked: list[Any] = []

        async def resolve(self, provider: str) -> Any:
            self.resolved.append(provider)
            return self.handle

        async def mark_used(self, handle: Any) -> None:
            self.marked.append(handle)

    runtime = RecordingRuntime()

    async def empty_standard_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        return UnifiedSearchResult(documents=[], query=str(kwargs.get("query", "")))

    async def fake_chat_call(**kwargs: Any) -> Any:
        captured.update(kwargs)

        async def upstream():
            yield {"choices": [{"delta": {"content": "partial answer"}}]}
            raise ChatAuthenticationError("raw provider secret", provider="test-provider")

        return upstream()

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)
    resolved_request = _resolved_request("standard")
    resolved_request.payload.update(
        {"claims_top_k": 3, "claims_max": 10, "claims_concurrency": 4}
    )

    events = [
        event
        async for event in stream_rag_events(
            resolved_request=resolved_request,
            retrieval_plan=_retrieval_plan(),
            standard_pipeline=empty_standard_pipeline,
            extra_context={"credential_runtime": runtime},
        )
    ]

    assert [event["type"] for event in events] == [  # nosec B101
        "contexts",
        "reasoning",
        "delta",
        "error",
    ]
    assert events[2]["text"] == "partial answer"  # nosec B101
    assert events[3]["code"] == "provider_authentication_failed"  # nosec B101
    assert "raw provider secret" not in str(events)  # nosec B101
    assert runtime.resolved == ["test-provider"]  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101
    assert captured["api_key"] == "runtime-stream-key"  # nosec B101
    assert captured["credentials_resolved"] is True  # nosec B101


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

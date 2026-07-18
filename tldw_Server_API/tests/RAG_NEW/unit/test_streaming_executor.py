import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.streaming_executor as streaming_executor
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
from tldw_Server_API.app.core.RAG.rag_service.generation import (
    GenerationConfig,
    LLMGenerator,
    StreamingGenerator,
)
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import (
    _PUBLIC_STREAM_ERROR_MESSAGE,
    stream_rag_events,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult
from tldw_Server_API.tests.provider_credential_test_helpers import (
    issue_provider_call_credentials_async,
)


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
        self.handle: Any = None
        self.marked: list[Any] = []

    async def resolve(self, provider: str, *, model: str | None = None) -> Any:
        if self.handle is None:
            self.handle = await issue_provider_call_credentials_async(
                provider,
                api_key="runtime-stream-key",
                app_config={"TestProvider": {"api_timeout": 9}},
                model=model,
            )
        return self.handle

    async def mark_used(self, handle: Any) -> None:
        self.marked.append(handle)


class _CloseTrackingAsyncStream:
    def __init__(self, label: str = "answer") -> None:
        self.label = label
        self.close_calls = 0
        self.next_calls = 0
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()

    def __aiter__(self) -> "_CloseTrackingAsyncStream":
        return self

    async def __anext__(self) -> str:
        self.next_calls += 1
        if self.next_calls == 1:
            return self.label
        self.waiting.set()
        await self.release.wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_calls += 1
        self.release.set()


class _CloseTrackingSyncStream:
    def __init__(self) -> None:
        self.close_calls = 0
        self.next_calls = 0

    def __iter__(self) -> "_CloseTrackingSyncStream":
        return self

    def __next__(self) -> str:
        self.next_calls += 1
        if self.next_calls == 1:
            return "answer"
        raise StopIteration

    def close(self) -> None:
        self.close_calls += 1


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


@pytest.mark.parametrize("stream_kind", ["async", "sync"])
@pytest.mark.asyncio
async def test_runtime_tracked_stream_closes_raw_iterator_on_early_close(
    monkeypatch: pytest.MonkeyPatch,
    stream_kind: str,
) -> None:
    raw: Any = (
        _CloseTrackingAsyncStream()
        if stream_kind == "async"
        else _CloseTrackingSyncStream()
    )

    async def fake_chat_call(**_kwargs: Any) -> Any:
        return raw

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)
    tracked = await LLMGenerator(
        GenerationConfig(provider="test-provider", model="test-model", streaming=True)
    )._call_llm("prompt", credential_runtime=_StreamingRuntime())

    assert await tracked.__anext__() == "answer"  # nosec B101
    await tracked.aclose()

    assert raw.close_calls == 1  # nosec B101


@pytest.mark.parametrize("stream_kind", ["async", "sync"])
@pytest.mark.asyncio
async def test_runtime_stream_acquisition_cancellation_closes_late_raw_stream(
    monkeypatch: pytest.MonkeyPatch,
    stream_kind: str,
) -> None:
    raw: Any = (
        _CloseTrackingAsyncStream()
        if stream_kind == "async"
        else _CloseTrackingSyncStream()
    )
    acquisition_started = asyncio.Event()
    release_acquisition = asyncio.Event()

    async def fake_chat_call(**_kwargs: Any) -> Any:
        acquisition_started.set()
        await release_acquisition.wait()
        return raw

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    runtime = _StreamingRuntime()
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)
    task = asyncio.create_task(
        LLMGenerator(
            GenerationConfig(
                provider="test-provider",
                model="test-model",
                streaming=True,
            )
        )._call_llm("prompt", credential_runtime=runtime)
    )

    await acquisition_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()  # nosec B101 - acquisition remains owned until it exits
    release_acquisition.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert raw.close_calls == 1  # nosec B101
    assert runtime.marked == []  # nosec B101


@pytest.mark.parametrize("runtime_bound", [False, True])
@pytest.mark.asyncio
async def test_streaming_generator_closes_raw_iterator_on_early_close(
    monkeypatch: pytest.MonkeyPatch,
    runtime_bound: bool,
) -> None:
    raw = _CloseTrackingAsyncStream()

    async def fake_chat_call(**_kwargs: Any) -> Any:
        return raw

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)
    generator = StreamingGenerator(
        GenerationConfig(provider="test-provider", model="test-model")
    )
    kwargs = {"credential_runtime": _StreamingRuntime()} if runtime_bound else {}
    stream = generator.generate_stream(SimpleNamespace(documents=[]), "query", **kwargs)

    assert await stream.__anext__() == "answer"  # nosec B101
    await stream.aclose()

    assert raw.close_calls == 1  # nosec B101
    assert generator.config.streaming is False  # nosec B101


@pytest.mark.asyncio
async def test_streaming_generator_cancellation_closes_raw_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _CloseTrackingAsyncStream()

    async def fake_chat_call(**_kwargs: Any) -> Any:
        return raw

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)
    stream = StreamingGenerator(
        GenerationConfig(provider="test-provider", model="test-model")
    ).generate_stream(
        SimpleNamespace(documents=[]),
        "query",
        credential_runtime=_StreamingRuntime(),
    )

    assert await stream.__anext__() == "answer"  # nosec B101
    pending = asyncio.create_task(stream.__anext__())
    await raw.waiting.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert raw.close_calls == 1  # nosec B101


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_streaming_generators_close_only_their_raw_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def build(label: str) -> tuple[Any, _CloseTrackingAsyncStream]:
        raw = _CloseTrackingAsyncStream(label)
        generator = StreamingGenerator(
            GenerationConfig(provider="test-provider", model="test-model")
        )

        async def fake_call(_prompt: str, **_kwargs: Any) -> Any:
            return raw

        monkeypatch.setattr(generator, "_call_llm", fake_call)
        stream = generator.generate_stream(SimpleNamespace(documents=[]), "query")
        assert await stream.__anext__() == label  # nosec B101
        return stream, raw

    (stream_a, raw_a), (stream_b, raw_b) = await asyncio.gather(
        build("answer-a"),
        build("answer-b"),
    )

    await stream_a.aclose()
    assert (raw_a.close_calls, raw_b.close_calls) == (1, 0)  # nosec B101

    await stream_b.aclose()
    assert (raw_a.close_calls, raw_b.close_calls) == (1, 1)  # nosec B101


@pytest.mark.parametrize(
    "control_chunks",
    [
        [": keepalive\n\n", "event: message\n", "id: 7\n", "retry: 1000\n"],
        ["keepalive", "event: ping"],
        ["data: ping\n\n", "data: pong\n\n", "data: heartbeat\n\n", "data: keepalive\n\n"],
    ],
)
@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.asyncio
async def test_stream_controls_then_failure_remain_unmarked(
    monkeypatch: pytest.MonkeyPatch,
    control_chunks: list[str],
    stream_kind: str,
) -> None:
    runtime = _StreamingRuntime()

    if stream_kind == "async":
        async def async_upstream() -> Any:
            for chunk in control_chunks:
                yield chunk
            raise ChatAuthenticationError("private upstream", provider="test-provider")

        upstream: Any = async_upstream()
    else:
        def sync_upstream() -> Any:
            yield from control_chunks
            raise ChatAuthenticationError("private upstream", provider="test-provider")

        upstream = sync_upstream()

    with pytest.raises(ChatAuthenticationError):
        await _consume_tracked_stream(monkeypatch, upstream, runtime)

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


@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.parametrize(
    "control_chunk",
    ["keepalive", ": keepalive\n\n", "data: ping\n\n"],
)
@pytest.mark.asyncio
async def test_stream_controls_then_cancellation_remain_unmarked(
    monkeypatch: pytest.MonkeyPatch,
    stream_kind: str,
    control_chunk: str,
) -> None:
    runtime = _StreamingRuntime()

    if stream_kind == "async":
        async def async_upstream() -> Any:
            yield control_chunk
            raise asyncio.CancelledError

        upstream: Any = async_upstream()
    else:
        def sync_upstream() -> Any:
            yield control_chunk
            raise asyncio.CancelledError

        upstream = sync_upstream()

    with pytest.raises(asyncio.CancelledError):
        await _consume_tracked_stream(monkeypatch, upstream, runtime)

    assert runtime.marked == []  # nosec B101


@pytest.mark.parametrize("stream_kind", ["sync", "async"])
@pytest.mark.parametrize("terminal_kind", ["failure", "cancellation"])
@pytest.mark.parametrize(
    ("content", "expected_mark_count"),
    [
        ("ping", 1),
        ("pong", 1),
        ("heartbeat", 1),
        ("keepalive", 1),
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

    assert [event["type"] for event in events] == [  # nosec B101
        "contexts",
        "reasoning",
        "delta",
        "complete",
    ]
    assert events[0]["contexts"][0]["id"] == "doc-1"  # nosec B101
    assert events[1]["plan"] == [  # nosec B101
        "Gather top-k contexts",
        "Rerank using strategy=flashrank",
        "Ground claims from sources",
        "Synthesize final answer",
    ]
    assert events[2]["text"] == "answer text"  # nosec B101


@pytest.mark.asyncio
async def test_stream_rag_events_closes_generation_stream_when_consumer_stops() -> None:
    lifecycle: list[str] = []

    async def empty_standard_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        return UnifiedSearchResult(documents=[], query=str(kwargs.get("query", "")))

    async def capture_generation(context: Any, **_kwargs: Any) -> Any:
        async def generation_stream():
            try:
                yield "answer"
                await asyncio.Event().wait()
            finally:
                lifecycle.append("generation_stream_close")

        context.stream_generator = generation_stream()
        return context

    stream = stream_rag_events(
        resolved_request=_resolved_request("standard"),
        retrieval_plan=_retrieval_plan(),
        standard_pipeline=empty_standard_pipeline,
        extra_context={"generate_streaming_response": capture_generation},
    )
    try:
        while True:
            event = await stream.__anext__()
            if event["type"] == "delta":
                break
    finally:
        await stream.aclose()

    assert lifecycle == ["generation_stream_close"]


@pytest.mark.asyncio
async def test_stream_rag_events_emits_structured_error():
    sentinel = "sk-secret-must-not-appear-in-stream-logs"

    async def empty_standard_pipeline(**kwargs: Any) -> UnifiedSearchResult:
        return UnifiedSearchResult(documents=[], query=str(kwargs.get("query", "")))

    async def failing_generate_streaming_response(context: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError(sentinel)

    logs: list[str] = []
    sink_id = streaming_executor.logger.add(logs.append, format="{message}")
    try:
        events = [
            event
            async for event in stream_rag_events(
                resolved_request=_resolved_request("standard"),
                retrieval_plan=_retrieval_plan(),
                standard_pipeline=empty_standard_pipeline,
                extra_context={"generate_streaming_response": failing_generate_streaming_response},
            )
        ]
    finally:
        streaming_executor.logger.remove(sink_id)

    assert [event["type"] for event in events] == ["contexts", "reasoning", "error"]  # nosec B101
    assert events[2] == {  # nosec B101
        "schema_version": 1,
        "type": "error",
        "code": "stream_internal_error",
        "upstream_dispatched": True,
        "output_emitted": False,
        "allow_non_stream_fallback": False,
        "message": _PUBLIC_STREAM_ERROR_MESSAGE,
    }
    assert sentinel not in "".join(logs)  # nosec B101


@pytest.mark.asyncio
async def test_streaming_generation_marks_partial_output_and_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class RecordingRuntime:
        def __init__(self) -> None:
            self.handle: Any = None
            self.resolved: list[str] = []
            self.marked: list[Any] = []

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            self.resolved.append(provider)
            if self.handle is None:
                self.handle = await issue_provider_call_credentials_async(
                    provider,
                    api_key="runtime-stream-key",
                    app_config={"TestProvider": {"api_timeout": 9}},
                    model=model,
                )
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
    assert events[3]["status_code"] == 502  # nosec B101
    assert events[3]["schema_version"] == 1  # nosec B101
    assert events[3]["upstream_dispatched"] is True  # nosec B101
    assert events[3]["output_emitted"] is True  # nosec B101
    assert events[3]["allow_non_stream_fallback"] is False  # nosec B101
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

    assert [event["type"] for event in events] == [  # nosec B101
        "contexts",
        "reasoning",
        "delta",
        "complete",
    ]
    assert events[0]["contexts"] == []  # nosec B101
    assert events[-2]["text"] == "answer text"  # nosec B101


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

    assert [event["type"] for event in events] == [  # nosec B101
        "contexts",
        "reasoning",
        "delta",
        "complete",
    ]
    assert events[0]["contexts"] == []  # nosec B101
    assert events[-2]["text"] == "answer text"  # nosec B101


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
    assert [event["type"] for event in events] == [  # nosec B101
        "plan",
        "contexts",
        "reasoning",
        "delta",
        "complete",
    ]
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

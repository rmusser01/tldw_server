"""Behavioral contracts for bounded, recoverable MCP result context."""

from __future__ import annotations

import asyncio
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_llm_caller import LLMResponse

pytestmark = pytest.mark.unit


def context(mode="excerpt", worker=None, **kwargs):
    """Build the real policy with small budgets for boundary coverage."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_result_context import (
        ToolResultContext,
        ToolResultPolicy,
    )

    return ToolResultContext(
        ToolResultPolicy(mode=mode, min_input_bytes=1024, max_output_bytes=768, **kwargs),
        worker=worker,
    )


async def prepare(ctx, text, **kwargs):
    return await ctx.prepare(
        tool_name="search",
        arguments={"query": "needle"},
        text=text,
        question="Where is needle?",
        cancel_event=asyncio.Event(),
        **kwargs,
    )


class SelectingWorker:
    """Observe the real extraction request and select a known evidence segment."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    async def call(self, messages, tools):
        self.requests.append((messages, tools))
        if self.error:
            raise self.error
        if self.response is not None:
            return self.response
        payload = json.loads(messages[-1]["content"])
        segment = next(s for s in payload["segments"] if "needle" in s["text"])
        return LLMResponse(text=json.dumps({"segment_ids": [segment["id"]]}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,text,is_error",
    [
        ("off", "x" * 9000, False),
        ("excerpt", "short", False),
        ("worker", "failed " * 2000, True),
    ],
    ids=["disabled", "small", "error"],
)
async def test_ineligible_results_preserve_exact_text(mode, text, is_error):
    worker = SelectingWorker(error=AssertionError("ineligible result reached worker"))
    result = await prepare(context(mode, worker), text, is_error=is_error)
    assert result.output == text


@pytest.mark.asyncio
async def test_excerpt_finds_evidence_and_recovers_omitted_source():
    text = "unrelated " * 250 + "needle is at platform nine. " + "more " * 300
    ctx = context()
    result = await prepare(ctx, text)
    assert "needle is at platform nine" in result.output
    assert len(result.output.encode()) <= 768
    source_id = result.metadata["source_id"]
    for start, end in result.metadata["ranges"]:
        assert text[start:end] in result.output
    assert ctx.source_call(source_id) == ("search", {"query": "needle"})
    reread = ctx.read(source_id, offset=0, limit=40)
    assert text[:40] in reread.output


@pytest.mark.asyncio
async def test_worker_selects_exact_source_and_has_no_tools_or_history():
    worker = SelectingWorker()
    text = "unrelated " * 200 + "needle is blue. " + "tail " * 300
    result = await prepare(context("worker", worker), text)
    assert result.metadata["outcome"] == "worker"
    assert "needle is blue" in result.output
    assert worker.requests[0][1] == []
    assert len(worker.requests[0][0]) == 2
    assert result.metadata["worker_usage"] is None
    assert "unrelated " * 100 not in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"segment_ids": []}',
        '{"segment_ids": [999]}',
        '{"segment_ids": [-1]}',
        '{"segment_ids": [true]}',
        '{"segment_ids": [0, 0]}',
        '{"segment_ids": ["0"]}',
        '{"segment_ids": [0], "invented": "fact"}',
    ],
)
async def test_unverifiable_worker_selection_falls_back(response):
    worker = SelectingWorker(LLMResponse(text=response))
    result = await prepare(context("worker", worker), "x " * 900 + "needle verified")
    assert result.metadata["outcome"] == "worker_invalid"
    assert "needle verified" in result.output
    assert len(result.output.encode()) <= 768


@pytest.mark.asyncio
async def test_worker_failure_does_not_lose_source_or_expose_exception():
    worker = SelectingWorker(error=RuntimeError("private provider credential"))
    result = await prepare(context("worker", worker), "data " * 400 + "needle")
    assert result.metadata["outcome"] == "worker_error"
    assert "private provider credential" not in str(result)
    assert "needle" in result.output


@pytest.mark.asyncio
async def test_worker_timeout_cancels_the_pending_request():
    cancelled = asyncio.Event()

    class SlowWorker:
        async def call(self, messages, tools):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    result = await prepare(context("worker", SlowWorker(), worker_timeout_seconds=0.01), "needle " * 400)
    assert result.metadata["outcome"] == "worker_timeout"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_cancellation_propagates_and_cancels_worker():
    started, cancelled, stop = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class SlowWorker:
        async def call(self, messages, tools):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    ctx = context("worker", SlowWorker())
    job = asyncio.create_task(
        ctx.prepare(
            tool_name="search",
            arguments={},
            text="needle " * 400,
            question="needle",
            cancel_event=stop,
        )
    )
    await started.wait()
    stop.set()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_storage_limit_preserves_original_and_advertises_no_source():
    text = "a" * 3000
    result = await prepare(context(max_retained_bytes=2000), text)
    assert result.output == text
    assert result.metadata["outcome"] == "storage_limit"
    assert "source_id" not in result.metadata


@pytest.mark.asyncio
async def test_worker_input_limit_uses_excerpt_without_worker():
    worker = SelectingWorker(error=AssertionError("oversized request reached worker"))
    result = await prepare(context("worker", worker, max_worker_input_bytes=1024), "needle " * 500)
    assert result.metadata["outcome"] == "worker_input_limit"
    assert len(result.output.encode()) <= 768
    assert result.metadata["worker_called"] is False


@pytest.mark.asyncio
async def test_references_are_run_local_and_read_arguments_are_strict():
    ctx = context()
    result = await prepare(ctx, "abc " * 1000)
    source_id = result.metadata["source_id"]
    with pytest.raises(ValueError, match="Unknown"):
        context().read(source_id)
    for kwargs in ({"offset": -1}, {"offset": True}, {"limit": 0}, {"limit": "10"}):
        with pytest.raises(ValueError):
            ctx.read(source_id, **kwargs)


@pytest.mark.asyncio
@settings(max_examples=30, deadline=None)
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1025, max_size=4000))
async def test_unicode_excerpts_obey_byte_budget_and_preserve_offsets(text):
    result = await prepare(context(), text)
    assert len(result.output.encode("utf-8")) <= 768
    for start, end in result.metadata["ranges"]:
        assert text[start:end] in result.output


@pytest.mark.asyncio
async def test_worker_usage_is_preserved_when_explicitly_reported():
    response = LLMResponse(text='{"segment_ids": [0]}', usage={"input_tokens": 100, "cost_usd": 0.001})
    result = await prepare(context("worker", SelectingWorker(response)), "needle " * 400)
    assert result.metadata["worker_usage"] == {"input_tokens": 100, "cost_usd": 0.001}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        LLMResponse(text="[" * 1500 + "]" * 1500),
        LLMResponse(text="\ud800"),
        LLMResponse(text={"segment_ids": [0]}),
        None,
    ],
    ids=["deep-json", "surrogate", "non-text", "no-response"],
)
async def test_malformed_worker_response_cannot_escape_fallback(response):
    class MalformedWorker:
        async def call(self, messages, tools):
            return response

    result = await prepare(context("worker", MalformedWorker()), "source " * 500 + "needle")
    assert result.metadata["outcome"] == "worker_invalid"
    assert "needle" in result.output


@pytest.mark.asyncio
async def test_non_utf8_source_passes_through_without_unavailable_reference():
    text = "source " * 500 + "\ud800"
    result = await prepare(context(), text)
    assert result.output == text
    assert result.metadata["outcome"] == "unsupported_text"
    assert "source_id" not in result.metadata


@pytest.mark.asyncio
async def test_invalid_usage_is_discarded_without_losing_worker_selection():
    response = LLMResponse(
        text='{"segment_ids": [0]}',
        usage={
            "cost_usd": float("nan"),
            "input_tokens": 10**1000,
            "private": "credential",
            "output_tokens": -1,
        },
    )
    result = await prepare(context("worker", SelectingWorker(response)), "needle " * 400)
    assert result.metadata["worker_usage"] is None
    assert result.metadata["outcome"] == "worker"


@pytest.mark.asyncio
async def test_slow_worker_cleanup_does_not_defeat_deadline():
    cleanup_started, cleanup_finished = asyncio.Event(), asyncio.Event()

    class SlowCleanupWorker:
        async def call(self, messages, tools):
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleanup_finished.set()

    # An outer deadline makes the existing unbounded gather fail promptly.
    result = await asyncio.wait_for(
        prepare(context("worker", SlowCleanupWorker(), worker_timeout_seconds=0.01), "needle " * 400),
        timeout=0.5,
    )
    assert result.metadata["outcome"] == "worker_timeout"
    assert cleanup_started.is_set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_explicit_clear_removes_text_and_releases_retention_capacity():
    ctx = context(max_retained_bytes=3000)
    result = await prepare(ctx, "abc " * 500)
    ctx.clear()
    with pytest.raises(ValueError, match="Unknown"):
        ctx.read(result.metadata["source_id"])
    next_result = await prepare(ctx, "def " * 500)
    assert next_result.metadata["outcome"] == "excerpt"


@pytest.mark.asyncio
async def test_invalid_question_encoding_uses_excerpt_without_calling_worker():
    worker = SelectingWorker(error=AssertionError("invalid request reached worker"))
    ctx = context("worker", worker)
    result = await ctx.prepare(
        tool_name="search",
        arguments={},
        text="verified source " * 200,
        question="\ud800",
        cancel_event=asyncio.Event(),
    )
    assert result.metadata["outcome"] == "worker_input_invalid"
    assert result.metadata["worker_called"] is False
    assert "verified source" in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_text", [None, {}])
async def test_invalid_worker_text_keeps_reported_billing_usage(invalid_text):
    response = LLMResponse(text=invalid_text, usage={"input_tokens": 100, "output_tokens": 15})
    result = await prepare(context("worker", SelectingWorker(response)), "needle " * 400)
    assert result.metadata["outcome"] == "worker_invalid"
    assert result.metadata["worker_usage"] == {"input_tokens": 100, "output_tokens": 15}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["excerpt", "worker"])
async def test_large_result_ranking_observes_scheduled_cancellation(mode):
    stop = asyncio.Event()
    asyncio.get_running_loop().call_soon(stop.set)
    worker = SelectingWorker(error=AssertionError("oversized input must not reach the worker"))
    ctx = context(mode, worker, max_worker_input_bytes=1024)
    with pytest.raises(asyncio.CancelledError):
        await ctx.prepare(
            tool_name="search",
            arguments={},
            text="x" * (1024 * 1024),
            question=" ".join(f"word{index}" for index in range(200)),
            cancel_event=stop,
        )


@pytest.mark.asyncio
async def test_large_result_ranking_allows_other_event_loop_work():
    heartbeat = asyncio.Event()

    async def tick():
        await asyncio.sleep(0)
        heartbeat.set()

    task = asyncio.create_task(tick())
    try:
        result = await prepare(context(), "x" * (1024 * 1024) + "needle is blue.")
        assert heartbeat.is_set()
        assert "needle is blue." in result.output
        assert len(result.output.encode("utf-8")) <= 768
    finally:
        await task

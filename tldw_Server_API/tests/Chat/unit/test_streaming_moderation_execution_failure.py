"""A moderation execution failure is terminal on every stream exit path."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Chat.streaming_utils import StreamingResponseHandler
from tldw_Server_API.app.core.exception_types import MODERATION_EXECUTION_REASONS, ModerationExecutionError

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class Transform:
    def __init__(self, *, reason=None, fail_at="chunk", asynchronous=False):
        self.reason = reason
        self.fail_at = fail_at
        self.asynchronous = asynchronous
        self.calls = 0
        self.flushes = 0

    def __call__(self, text):
        self.calls += 1
        if self.reason and self.fail_at == "chunk" and self.calls == 2:
            raise ModerationExecutionError(self.reason)
        if self.asynchronous:
            return self.value(text.upper())
        return text.upper()

    async def value(self, value):
        await asyncio.sleep(0)
        return value

    def flush(self):
        self.flushes += 1
        if self.reason and self.fail_at == "flush":
            raise ModerationExecutionError(self.reason)
        return self.value("TAIL") if self.asynchronous else "TAIL"


async def source(structured=False, failure=False, closed=None):
    try:
        for text in ["safe", "private"]:
            yield {"choices": [{"delta": {"content": text}}]} if structured else text
        if failure:
            raise RuntimeError("provider failed")
        if structured:
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    finally:
        if closed is not None:
            closed.append(True)


def payloads(frames):
    return [
        json.loads(line[6:])
        for frame in frames
        for line in frame.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize("reason", sorted(MODERATION_EXECUTION_REASONS))
@pytest.mark.parametrize("structured", [False, True])
async def test_failure_after_safe_prefix_never_flushes_or_saves_tail(reason, structured):
    transform = Transform(reason=reason)
    handler = StreamingResponseHandler("1", "test", text_transform=transform)
    save, partial, finalize = AsyncMock(), AsyncMock(), AsyncMock()
    closed = []
    frames = [
        frame
        async for frame in handler.safe_stream_generator(
            source(structured, closed=closed), save, finalize_callback=finalize, partial_save_callback=partial
        )
    ]
    results = payloads(frames)
    assert any(item.get("error", {}).get("type") == "moderation_unavailable" for item in results)
    assert "SAFE" in "".join(frames)
    assert "private" not in "".join(frames).lower()
    assert "TAIL" not in "".join(frames)
    assert transform.flushes == 0
    assert not any(choice.get("finish_reason") == "stop" for item in results for choice in item.get("choices", []))
    assert closed == [True]
    save.assert_not_called()
    partial.assert_not_called()
    finalize.assert_awaited_once_with(success=False, cancelled=False, error=True)


@pytest.mark.parametrize("failure", [False, True])
async def test_awaitable_transform_is_awaited_in_every_path(failure):
    handler = StreamingResponseHandler("1", "test", text_transform=Transform(asynchronous=True))
    save = AsyncMock()
    frames = [frame async for frame in handler.safe_stream_generator(source(failure=failure), save)]
    assert "coroutine" not in "".join(frames)
    assert "SAFE" in "".join(frames)
    assert "TAIL" in "".join(frames)


async def test_sync_transform_remains_compatible():
    handler = StreamingResponseHandler("1", "test", text_transform=Transform())
    save = AsyncMock()
    frames = [frame async for frame in handler.safe_stream_generator(source(), save)]
    assert "SAFE" in "".join(frames)
    assert save.await_args.args[0] == "SAFEPRIVATETAIL"


@pytest.mark.parametrize("site", ["flush", "before_success", "save", "partial"])
async def test_late_failure_prevents_success(site):
    transform = Transform(reason="worker_crashed" if site == "flush" else None, fail_at="flush")
    handler = StreamingResponseHandler("1", "test", text_transform=transform)
    save = AsyncMock(side_effect=ModerationExecutionError("worker_crashed") if site == "save" else None)
    before = AsyncMock(side_effect=ModerationExecutionError("worker_crashed") if site == "before_success" else None)
    partial = AsyncMock(side_effect=ModerationExecutionError("worker_crashed")) if site == "partial" else None
    frames = [
        frame
        async for frame in handler.safe_stream_generator(
            source(failure=site == "partial"), save, before_success_callback=before, partial_save_callback=partial
        )
    ]
    assert any(item.get("error", {}).get("type") == "moderation_unavailable" for item in payloads(frames))
    assert not any(
        choice.get("finish_reason") == "stop" for item in payloads(frames) for choice in item.get("choices", [])
    )


async def test_cancel_partial_settlement_awaits_async_tail():
    transform = Transform(asynchronous=True)
    handler = StreamingResponseHandler("1", "test", text_transform=transform)
    partial = AsyncMock(return_value="partial-id")

    async def interrupted():
        yield "safe"
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        async for _frame in handler.safe_stream_generator(interrupted(), partial_save_callback=partial):
            pass
    await asyncio.sleep(0.05)
    assert partial.await_args.args[0] == "SAFETAIL"


async def test_cancel_flush_failure_cannot_start_partial_save():
    handler = StreamingResponseHandler("1", "test", text_transform=Transform(reason="worker_crashed", fail_at="flush"))
    partial = AsyncMock()

    async def interrupted():
        yield "safe"
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        async for _frame in handler.safe_stream_generator(interrupted(), partial_save_callback=partial):
            pass
    partial.assert_not_called()
    assert handler.error_occurred


async def test_before_success_failure_never_flushes_tail():
    transform = Transform()
    handler = StreamingResponseHandler("1", "test", text_transform=transform)
    before = AsyncMock(side_effect=ModerationExecutionError("worker_crashed"))
    frames = [frame async for frame in handler.safe_stream_generator(source(), before_success_callback=before)]
    assert transform.flushes == 0
    assert "TAIL" not in "".join(frames)


async def test_provider_is_closed_before_terminal_error_is_delivered():
    closed = []
    handler = StreamingResponseHandler("1", "test", text_transform=Transform(reason="worker_crashed"))
    stream = handler.safe_stream_generator(source(closed=closed))
    try:
        async for frame in stream:
            if '"moderation_unavailable"' in frame:
                assert closed == [True]
                break
    finally:
        await stream.aclose()


async def test_error_path_flush_failure_closes_provider_before_delivery():
    closed = []

    class Source:
        def __aiter__(self):
            raise RuntimeError("provider setup failed")

        async def aclose(self):
            closed.append(True)

    handler = StreamingResponseHandler("1", "test", text_transform=Transform(reason="worker_crashed", fail_at="flush"))
    stream = handler.safe_stream_generator(Source())
    try:
        async for frame in stream:
            if '"moderation_unavailable"' in frame:
                assert closed == [True]
                break
        else:
            pytest.fail("missing terminal moderation failure")
    finally:
        await stream.aclose()


async def test_cancel_during_async_flush_still_closes_provider():
    closed = []

    class Source:
        emitted = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.emitted:
                self.emitted = True
                return "safe"
            raise asyncio.CancelledError

        async def aclose(self):
            closed.append(True)

    class CancelFlush(Transform):
        async def flush(self):
            raise asyncio.CancelledError

    handler = StreamingResponseHandler("1", "test", text_transform=CancelFlush())
    partial = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        async for _frame in handler.safe_stream_generator(Source(), partial_save_callback=partial):
            pass
    assert closed == [True]
    partial.assert_not_called()

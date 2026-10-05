"""Streaming settlement records generation metadata and keeps partial replies (D7 P2)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest

from tldw_Server_API.app.core.Chat.streaming_utils import (
    StopStreamWithError,
    StreamingResponseHandler,
    await_stream_operation_bounded,
    create_streaming_response_with_timeout,
)

pytestmark = pytest.mark.unit


def _sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload) + "\n\n"


def _content(text: str, finish_reason: str | None = None) -> str:
    choice: dict[str, Any] = {"index": 0, "delta": {"content": text}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    return _sse({"choices": [choice]})


class Recorder:
    """Capture save calls the way chat_service callbacks receive them.

    With ``gated=True`` the partial write blocks until ``release`` is set, so a
    test can observe the stream while the write is still in flight.
    """

    def __init__(self, gated: bool = False) -> None:
        self.saves: list[dict[str, Any]] = []
        self.partials: list[dict[str, Any]] = []
        self.partial_started = asyncio.Event()
        self.partial_done = asyncio.Event()
        self.release = asyncio.Event()
        if not gated:
            self.release.set()

    async def save(self, text, tool_calls=None, function_call=None, *, generation=None):
        self.saves.append({"text": text, "generation": generation})
        return "saved-complete"

    async def partial(self, text, *, generation):
        self.partial_started.set()
        await self.release.wait()
        self.partials.append({"text": text, "generation": generation})
        self.partial_done.set()
        return "saved-partial"


async def _drain(handler: StreamingResponseHandler, stream, recorder: Recorder) -> list[str]:
    return [
        frame
        async for frame in handler.safe_stream_generator(
            stream,
            save_callback=recorder.save,
            partial_save_callback=recorder.partial,
        )
    ]


async def _partial_settled(recorder: Recorder) -> None:
    """Let the detached write that a cancelled stream leaves behind finish."""
    recorder.release.set()
    await asyncio.wait_for(recorder.partial_done.wait(), timeout=10)


async def _no_partial_settles(recorder: Recorder) -> None:
    """Give a wrongly started detached write time to show up."""
    await asyncio.sleep(0.05)
    assert recorder.partials == []


async def test_complete_reply_passes_finish_reason_and_provider_usage_to_save():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Hello ")
        yield _content("world", "stop")
        yield _sse({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}})
        yield "data: [DONE]\n\n"

    await _drain(handler, provider(), recorder)

    assert recorder.partials == []
    assert recorder.saves == [
        {
            "text": "Hello world",
            "generation": {
                "generation_status": "complete",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        }
    ]


async def test_length_finish_reason_settles_as_length():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Truncated", "length")
        yield "data: [DONE]\n\n"

    await _drain(handler, provider(), recorder)

    assert recorder.saves[0]["generation"] == {"generation_status": "length", "finish_reason": "length"}


async def test_stop_during_the_complete_save_does_not_also_save_a_partial():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def save(text, tool_calls=None, function_call=None, *, generation=None):
        # A stop request (or idle timeout) lands while the complete reply is being written.
        handler.request_stop()
        return await recorder.save(text, tool_calls, function_call, generation=generation)

    async def provider() -> AsyncIterator[str]:
        yield _content("Finished", "stop")
        yield "data: [DONE]\n\n"

    async for _ in handler.safe_stream_generator(
        provider(),
        save_callback=save,
        partial_save_callback=recorder.partial,
    ):
        pass

    assert [entry["text"] for entry in recorder.saves] == ["Finished"]
    await _no_partial_settles(recorder)


async def test_legacy_single_argument_save_callback_still_works():
    handler = StreamingResponseHandler("conv", "model")
    saved: list[str] = []

    async def save(text):
        saved.append(text)

    async def provider() -> AsyncIterator[str]:
        yield _content("plain", "stop")

    async for _ in handler.safe_stream_generator(provider(), save):
        pass

    assert saved == ["plain"]


async def test_stop_signal_settles_partial_as_stopped():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Partial ")
        yield _content("reply")
        handler.request_stop()
        yield _content(" never")

    frames = await _drain(handler, provider(), recorder)

    await _partial_settled(recorder)
    assert recorder.saves == []
    assert recorder.partials == [{"text": "Partial reply", "generation": {"generation_status": "stopped"}}]
    assert not any("[DONE]" in frame for frame in frames)


async def test_client_disconnect_settles_partial_as_interrupted():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Half ")
        yield _content("done")
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _drain(handler, provider(), recorder)

    await _partial_settled(recorder)
    assert recorder.saves == []
    assert recorder.partials == [{"text": "Half done", "generation": {"generation_status": "interrupted"}}]


async def test_disconnect_after_length_finish_keeps_length_status():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Cut off", "length")
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _drain(handler, provider(), recorder)

    await _partial_settled(recorder)
    assert recorder.partials[0]["generation"] == {"generation_status": "length", "finish_reason": "length"}


async def test_upstream_error_after_output_settles_partial_and_reports_its_id():
    handler = StreamingResponseHandler("conv", "model")
    handler.history_persistence_ack = True
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Some ")
        yield _content("text")
        raise RuntimeError("upstream reset")

    frames = await _drain(handler, provider(), recorder)

    assert recorder.saves == []
    assert recorder.partials == [{"text": "Some text", "generation": {"generation_status": "interrupted"}}]
    stream_end = next(frame for frame in frames if frame.startswith("event: stream_end"))
    assert '"tldw_message_id": "saved-partial"' in stream_end
    assert '"success": false' in stream_end


async def test_provider_error_frame_after_output_settles_partial():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Before the ")
        yield _sse({"error": {"message": "overloaded", "code": "provider_unavailable"}})

    await _drain(handler, provider(), recorder)

    assert recorder.partials == [{"text": "Before the ", "generation": {"generation_status": "interrupted"}}]


async def test_upstream_error_before_output_saves_nothing():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        raise RuntimeError("connect failed")
        yield  # pragma: no cover - makes this an async generator

    await _drain(handler, provider(), recorder)

    assert recorder.saves == []
    await _no_partial_settles(recorder)


async def test_disconnect_before_output_saves_nothing():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        raise asyncio.CancelledError()
        yield  # pragma: no cover - makes this an async generator

    with pytest.raises(asyncio.CancelledError):
        await _drain(handler, provider(), recorder)

    await _no_partial_settles(recorder)


async def test_whitespace_only_partial_is_not_saved():
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("   ")
        raise RuntimeError("upstream reset")

    await _drain(handler, provider(), recorder)

    await _no_partial_settles(recorder)


async def test_policy_stop_never_saves_a_partial():
    def blocking_transform(text: str) -> str:
        if "forbidden" in text:
            raise StopStreamWithError("blocked", error_type="output_moderation_block")
        return text

    handler = StreamingResponseHandler("conv", "model", text_transform=blocking_transform)
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Allowed ")
        yield _content("forbidden")
        yield _content(" more")

    await _drain(handler, provider(), recorder)

    assert recorder.saves == []
    await _no_partial_settles(recorder)


async def test_disconnect_after_policy_stop_never_saves_a_partial():
    def blocking_transform(text: str) -> str:
        if "forbidden" in text:
            raise StopStreamWithError("blocked", error_type="output_moderation_block")
        return text

    handler = StreamingResponseHandler("conv", "model", text_transform=blocking_transform)
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Allowed ")
        yield _content("forbidden")

    stream_gen = handler.safe_stream_generator(
        provider(),
        save_callback=recorder.save,
        partial_save_callback=recorder.partial,
    )
    async for frame in stream_gen:
        if "output_moderation_block" in frame:
            break
    # The client goes away while the block is being reported.
    await stream_gen.aclose()

    assert handler.is_cancelled is True
    await _no_partial_settles(recorder)


async def test_partial_includes_text_held_back_by_the_transform():
    class HoldbackTransform:
        """Emit nothing until flushed, like the moderation holdback buffer."""

        def __init__(self) -> None:
            self.held = ""

        def __call__(self, text: str) -> str:
            self.held += text
            return ""

        def flush(self) -> str:
            out, self.held = self.held, ""
            return out

    handler = StreamingResponseHandler("conv", "model", text_transform=HoldbackTransform())
    recorder = Recorder()

    async def provider() -> AsyncIterator[str]:
        yield _content("Held ")
        yield _content("back")
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _drain(handler, provider(), recorder)

    await _partial_settled(recorder)
    assert recorder.partials == [{"text": "Held back", "generation": {"generation_status": "interrupted"}}]


async def test_cancelled_stream_finishes_without_waiting_for_the_partial_write():
    """A cancelled stream must end at once; the endpoint reuses or closes it right after.

    The write is detached instead of awaited, so neither the 50 ms cleanup
    deadlines nor anyio's repeated cancellation can abort it, and the generator
    is never left "already running" while the write is in flight.
    """
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder(gated=True)

    async def provider() -> AsyncIterator[str]:
        yield _content("Slow ")
        yield _content("save")
        await asyncio.Event().wait()

    stream_gen = handler.safe_stream_generator(
        provider(),
        save_callback=recorder.save,
        partial_save_callback=recorder.partial,
    )
    frames: list[str] = []
    while not any("save" in frame for frame in frames):
        frames.append(await stream_gen.__anext__())

    pending_chunk = asyncio.ensure_future(stream_gen.__anext__())
    await asyncio.sleep(0)
    pending_chunk.cancel()
    done, _pending = await asyncio.wait({pending_chunk}, timeout=10)

    # The stream ended although the write has not finished.
    assert pending_chunk in done and pending_chunk.cancelled()
    await asyncio.wait_for(recorder.partial_started.wait(), timeout=10)
    assert recorder.partials == []
    # No "asynchronous generator is already running": the generator is finished.
    await stream_gen.aclose()

    await _partial_settled(recorder)
    assert recorder.partials == [{"text": "Slow save", "generation": {"generation_status": "interrupted"}}]


async def test_closing_a_suspended_stream_does_not_wait_for_the_partial_write():
    """Closing the generator (GeneratorExit) runs under a 50 ms budget; the write outlives it."""
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder(gated=True)

    async def provider() -> AsyncIterator[str]:
        yield _content("Slow ")
        yield _content("save")
        await asyncio.Event().wait()

    stream_gen = handler.safe_stream_generator(
        provider(),
        save_callback=recorder.save,
        partial_save_callback=recorder.partial,
    )
    frames: list[str] = []
    while not any("save" in frame for frame in frames):
        frames.append(await stream_gen.__anext__())

    await asyncio.wait_for(stream_gen.aclose(), timeout=10)

    await asyncio.wait_for(recorder.partial_started.wait(), timeout=10)
    assert recorder.partials == []
    await _partial_settled(recorder)
    assert recorder.partials == [{"text": "Slow save", "generation": {"generation_status": "interrupted"}}]


async def test_partial_is_handed_off_before_cancel_cleanup_that_misses_its_deadline():
    """Cleanup after a disconnect can be cancelled by its deadline; the partial must already be on its way."""
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder()

    async def stuck_finalize(**_kwargs) -> None:
        await asyncio.Event().wait()

    async def provider() -> AsyncIterator[str]:
        yield _content("Kept ")
        yield _content("anyway")
        await asyncio.Event().wait()

    stream_gen = handler.safe_stream_generator(
        provider(),
        save_callback=recorder.save,
        finalize_callback=stuck_finalize,
        partial_save_callback=recorder.partial,
    )
    frames: list[str] = []
    while not any("anyway" in frame for frame in frames):
        frames.append(await stream_gen.__anext__())

    with pytest.raises((asyncio.TimeoutError, asyncio.CancelledError)):
        await await_stream_operation_bounded(stream_gen.aclose(), cleanup=True)

    await _partial_settled(recorder)
    assert recorder.partials == [{"text": "Kept anyway", "generation": {"generation_status": "interrupted"}}]


async def test_partial_write_survives_cancelling_the_stream_that_awaits_it():
    """After a provider failure the stream awaits the write; a late disconnect must not abort it."""
    handler = StreamingResponseHandler("conv", "model")
    recorder = Recorder(gated=True)

    async def provider() -> AsyncIterator[str]:
        yield _content("Some ")
        yield _content("text")
        raise RuntimeError("upstream reset")

    consumer = asyncio.ensure_future(_drain(handler, provider(), recorder))
    await asyncio.wait_for(recorder.partial_started.wait(), timeout=10)
    assert not consumer.done()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert recorder.partials == []
    await _partial_settled(recorder)
    assert recorder.partials == [{"text": "Some text", "generation": {"generation_status": "interrupted"}}]


async def test_partial_saved_when_anyio_scope_cancels_the_response_task():
    """Starlette cancels the response task through a level-triggered anyio scope."""
    recorder = Recorder(gated=True)
    delivered = asyncio.Event()

    async def provider() -> AsyncIterator[str]:
        yield _content("Hello ")
        yield _content("world")
        await asyncio.Event().wait()

    async def stream_response() -> None:
        async for frame in create_streaming_response_with_timeout(
            provider(),
            "conv",
            "model",
            save_callback=recorder.save,
            partial_save_callback=recorder.partial,
            heartbeat_interval=0,
        ):
            if "world" in frame:
                delivered.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(stream_response)
        await delivered.wait()
        task_group.cancel_scope.cancel()

    # The response task is gone; the write it left behind is still in flight.
    await asyncio.wait_for(recorder.partial_started.wait(), timeout=10)
    assert recorder.partials == []
    await _partial_settled(recorder)
    assert recorder.saves == []
    assert recorder.partials == [{"text": "Hello world", "generation": {"generation_status": "interrupted"}}]

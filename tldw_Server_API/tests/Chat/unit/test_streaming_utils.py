# test_streaming_utils.py
# Unit tests for streaming response utilities

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.core.Chat import bounded_daemon as bounded_daemon_module
from tldw_Server_API.app.core.Chat import streaming_utils
from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
from tldw_Server_API.app.core.Chat.streaming_utils import (
    HEARTBEAT_INTERVAL,
    STREAMING_IDLE_TIMEOUT,
    StreamingResponseHandler,
    create_streaming_response_with_timeout,
)


async def _collect_handler_wire(handler, stream) -> str:
    return "".join(
        [message async for message in handler.safe_stream_generator(stream)]
    )


async def _collect_async_items(stream) -> list[object]:
    return [item async for item in stream]


class TestStreamingResponseHandler:
    """Test StreamingResponseHandler class."""

    def test_initialization(self):

        """Test handler initialization."""
        handler = StreamingResponseHandler(
            conversation_id="conv_123",
            model_name="gpt-4",
            idle_timeout=600,
            heartbeat_interval=60
        )

        assert handler.conversation_id == "conv_123"
        assert handler.model_name == "gpt-4"
        assert handler.idle_timeout == 600
        assert handler.heartbeat_interval == 60
        assert handler.is_cancelled is False
        assert handler.error_occurred is False
        assert handler.full_response == []

    async def test_zero_budget_helpers_do_not_schedule_coroutine_bodies(self) -> None:
        streaming_body_ran = False
        endpoint_body_ran = False

        async def streaming_body():
            nonlocal streaming_body_ran
            streaming_body_ran = True

        async def endpoint_body():
            nonlocal endpoint_body_ran
            endpoint_body_ran = True

        with pytest.raises(asyncio.TimeoutError):
            await streaming_utils.await_stream_operation_bounded(
                streaming_body(),
                timeout=0,
            )
        with pytest.raises(asyncio.TimeoutError):
            await chat_endpoint._await_provider_stream_operation(
                endpoint_body(),
                timeout=0,
            )
        await asyncio.sleep(0)

        assert streaming_body_ran is False
        assert endpoint_body_ran is False

    def test_update_activity(self):

        """Test activity timestamp update."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")
        initial_time = handler.last_activity

        time.sleep(0.01)  # Small delay
        handler.update_activity()

        assert handler.last_activity > initial_time

    def test_is_timed_out(self):

        """Test timeout detection."""
        handler = StreamingResponseHandler(
            "conv_123", "gpt-4", idle_timeout=1
        )

        # Should not be timed out initially
        assert handler.is_timed_out() is False

        # Simulate timeout
        handler.last_activity = time.time() - 2
        assert handler.is_timed_out() is True

    def test_cancel(self):

        """Test stream cancellation."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        assert handler.is_cancelled is False
        handler.cancel()
        assert handler.is_cancelled is True

    def test_parse_save_callback_result_accepts_loop_events_alias(self):
        handler = StreamingResponseHandler("conv_123", "gpt-4")
        message_id, events = handler._parse_save_callback_result(
            {
                "saved_message_id": "m-1",
                "loop_events": [{"event": "run_started", "data": {"run_id": "run_1", "seq": 1}}],
            }
        )

        assert message_id == "m-1"
        assert len(events) == 1
        assert events[0]["event"] == "run_started"

    @pytest.mark.asyncio
    async def test_heartbeat_generator(self):
        """Test heartbeat message generation."""
        handler = StreamingResponseHandler(
            "conv_123", "gpt-4",
            idle_timeout=10,
            heartbeat_interval=0.1  # Short interval for testing
        )

        heartbeats = []
        async for message in handler.heartbeat_generator():
            heartbeats.append(message)
            if len(heartbeats) >= 2:
                handler.cancel()  # Stop after 2 heartbeats
                break

        assert len(heartbeats) >= 2
        for hb in heartbeats:
            assert ": heartbeat" in hb
            # Check for ISO timestamp format (can be Z or +00:00)
            assert "\n\n" in hb  # Just check it ends with double newline

    @pytest.mark.asyncio
    async def test_heartbeat_timeout_detection(self):
        """Test heartbeat detects timeout."""
        handler = StreamingResponseHandler(
            "conv_123", "gpt-4",
            idle_timeout=0.1,  # Very short timeout
            heartbeat_interval=0.05
        )

        # Force timeout
        handler.last_activity = time.time() - 1

        messages = []
        async for message in handler.heartbeat_generator():
            messages.append(message)
            break  # Get first message

        assert len(messages) == 1
        payload = json.loads(messages[0].removeprefix("data:").strip())
        assert payload["error"] == {
            "code": "provider_unavailable",
            "type": "provider_unavailable",
            "message": "The chat service provider is currently unavailable.",
        }
        assert handler.is_cancelled is True


@pytest.mark.asyncio
class TestSafeStreamGenerator:
    """Test safe stream generation with error handling."""

    async def test_async_stream_processing(self):
        """Test processing of async stream."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        async def mock_stream():
            yield "Hello"
            yield " "
            yield "World"

        messages = []
        async for message in handler.safe_stream_generator(mock_stream()):
            messages.append(message)

        # Check stream_start event
        assert "event: stream_start" in messages[0]
        assert "conv_123" in messages[0]

        # Check content chunks (should be 3 content messages + 1 finish_reason message)
        content_messages = [m for m in messages if "choices" in m and "delta" in m and "data: " in m]
        # Filter out the finish_reason message
        actual_content_messages = [m for m in content_messages if '"finish_reason"' not in m]
        assert len(actual_content_messages) == 3

        # Check completion
        assert any("[DONE]" in m for m in messages)

        # Check full response collected
        assert "".join(handler.full_response) == "Hello World"

    async def test_sync_stream_processing(self):
        """Test processing of sync stream."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        def mock_stream():

            yield "Sync"
            yield " "
            yield "Stream"

        messages = []
        async for message in handler.safe_stream_generator(mock_stream()):
            messages.append(message)

        # Check content was processed
        assert "".join(handler.full_response) == "Sync Stream"

    async def test_sync_stream_offload_keeps_loop_responsive(self):
        """Sync streams should not block the event loop when offloaded."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        def blocking_stream():

            yield "Start"
            time.sleep(0.15)
            yield "End"

        tick_count = 0

        async def ticker():
            nonlocal tick_count
            start = asyncio.get_running_loop().time()
            while asyncio.get_running_loop().time() - start < 0.12:
                tick_count += 1
                await asyncio.sleep(0.01)

        async def consume():
            async for _ in handler.safe_stream_generator(blocking_stream()):
                pass

        await asyncio.gather(consume(), ticker())

        assert tick_count > 1
        assert "".join(handler.full_response) == "StartEnd"

    @pytest.mark.parametrize("stream_kind", ["sync", "async"])
    @pytest.mark.parametrize(
        "provider_chunk",
        [
            "provider_unavailable",
            "data: provider_unavailable\n\n",
        ],
        ids=["plain", "sse"],
    )
    async def test_allowlisted_error_code_text_remains_provider_output(
        self,
        stream_kind: str,
        provider_chunk: str,
    ) -> None:
        """A model may legitimately emit text equal to an internal error code."""
        handler = StreamingResponseHandler(
            f"conv_code_text_{stream_kind}",
            "gpt-4",
        )

        if stream_kind == "async":
            async def provider_stream():
                yield provider_chunk
        else:
            def provider_stream():
                yield provider_chunk

        messages = [message async for message in handler.safe_stream_generator(provider_stream())]
        wire = "".join(messages)

        assert "provider_unavailable" in wire
        assert '"error"' not in wire
        assert handler.error_occurred is False

    async def test_concurrent_code_text_and_real_error_envelope_remain_isolated(self) -> None:
        legitimate_ready = asyncio.Event()
        error_ready = asyncio.Event()
        release = asyncio.Event()
        upstream_sentinel = "provider-error-body-secret-/srv/upstream"

        async def legitimate_stream():
            legitimate_ready.set()
            await release.wait()
            yield "provider_unavailable"

        async def error_stream():
            error_ready.set()
            await release.wait()
            yield (
                'data: {"error":{"code":"provider_authentication_failed",'
                f'"message":"{upstream_sentinel}"}}}}\n\n'
            )

        legitimate_handler = StreamingResponseHandler("conv_legitimate_code", "gpt-4")
        error_handler = StreamingResponseHandler("conv_real_error", "gpt-4")

        async def collect(handler, stream):
            return "".join(
                [message async for message in handler.safe_stream_generator(stream)]
            )

        legitimate_task = asyncio.create_task(
            collect(legitimate_handler, legitimate_stream())
        )
        error_task = asyncio.create_task(collect(error_handler, error_stream()))
        await asyncio.wait_for(legitimate_ready.wait(), timeout=1.0)
        await asyncio.wait_for(error_ready.wait(), timeout=1.0)
        release.set()
        legitimate_wire, error_wire = await asyncio.gather(legitimate_task, error_task)

        assert "provider_unavailable" in legitimate_wire
        assert '"error"' not in legitimate_wire
        assert legitimate_handler.error_occurred is False
        assert '"code": "provider_authentication_failed"' in error_wire
        assert upstream_sentinel not in error_wire
        assert error_handler.error_occurred is True

    @pytest.mark.parametrize("stream_kind", ["sync", "async"])
    @pytest.mark.parametrize("framed", [False, True], ids=["raw", "sse"])
    async def test_malformed_error_like_assistant_json_is_preserved(
        self,
        stream_kind: str,
        framed: bool,
    ) -> None:
        assistant_fragment = '{"error": "assistant-authored unfinished JSON'
        provider_chunk = (
            f"data: {assistant_fragment}\n\n" if framed else assistant_fragment
        )
        handler = StreamingResponseHandler(
            f"conv_malformed_assistant_{stream_kind}_{framed}",
            "gpt-4",
        )

        if stream_kind == "async":
            async def provider_stream():
                yield provider_chunk
        else:
            def provider_stream():
                yield provider_chunk

        wire = "".join(
            [message async for message in handler.safe_stream_generator(provider_stream())]
        )

        assert "assistant-authored unfinished JSON" in wire
        assert '"code": "provider_unavailable"' not in wire
        assert handler.error_occurred is False

    async def test_concurrent_malformed_assistant_json_and_real_error_remain_isolated(
        self,
    ) -> None:
        ready = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        sentinel = "malformed-concurrent-secret-/srv/provider"

        async def assistant_stream():
            ready[0].set()
            await release.wait()
            yield 'data: {"error": "assistant-authored unfinished JSON\n\n'

        async def error_stream():
            ready[1].set()
            await release.wait()
            yield (
                'data: {"error":{"code":"provider_authentication_failed",'
                f'"message":"{sentinel}"}}}}\n\n'
            )

        async def collect(conversation_id: str, stream: AsyncIterator[str]) -> tuple[str, bool]:
            handler = StreamingResponseHandler(conversation_id, "gpt-4")
            wire = "".join(
                [message async for message in handler.safe_stream_generator(stream)]
            )
            return wire, handler.error_occurred

        assistant_task = asyncio.create_task(
            collect("conv_malformed_assistant", assistant_stream())
        )
        error_task = asyncio.create_task(collect("conv_malformed_error", error_stream()))
        await asyncio.gather(*(asyncio.wait_for(event.wait(), 1.0) for event in ready))
        release.set()
        (assistant_wire, assistant_failed), (error_wire, error_failed) = await asyncio.gather(
            assistant_task,
            error_task,
        )

        assert "assistant-authored unfinished JSON" in assistant_wire
        assert '"code": "provider_unavailable"' not in assistant_wire
        assert assistant_failed is False
        assert '"code": "provider_authentication_failed"' in error_wire
        assert sentinel not in assistant_wire + error_wire
        assert error_failed is True

    @pytest.mark.parametrize("stream_kind", ["sync", "async"])
    async def test_unframed_valid_assistant_error_json_is_preserved(
        self,
        stream_kind: str,
    ) -> None:
        assistant_json = '{"error":"this is requested model content"}'
        handler = StreamingResponseHandler(
            f"conv_valid_assistant_error_json_{stream_kind}",
            "gpt-4",
        )

        if stream_kind == "async":
            async def provider_stream():
                yield assistant_json
        else:
            def provider_stream():
                yield assistant_json

        wire = "".join(
            [message async for message in handler.safe_stream_generator(provider_stream())]
        )

        assert "this is requested model content" in wire
        assert '"code": "provider_unavailable"' not in wire
        assert handler.full_response == [assistant_json]
        assert handler.error_occurred is False

    async def test_concurrent_valid_assistant_error_json_and_framed_error_are_isolated(
        self,
    ) -> None:
        ready = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        sentinel = "valid-json-concurrent-secret-/srv/provider"

        async def assistant_stream():
            ready[0].set()
            await release.wait()
            yield '{"error":"this is requested model content"}'

        async def failed_stream():
            ready[1].set()
            await release.wait()
            yield (
                'data: {"error":{"code":"provider_authentication_failed",'
                f'"message":"{sentinel}"}}}}\n\n'
            )

        async def collect(conversation_id: str, stream: AsyncIterator[str]) -> tuple[str, bool]:
            handler = StreamingResponseHandler(conversation_id, "gpt-4")
            wire = "".join(
                [message async for message in handler.safe_stream_generator(stream)]
            )
            return wire, handler.error_occurred

        assistant_task = asyncio.create_task(
            collect("conv_valid_assistant_error_json", assistant_stream())
        )
        failed_task = asyncio.create_task(
            collect("conv_framed_provider_error", failed_stream())
        )
        try:
            await asyncio.gather(*(asyncio.wait_for(event.wait(), 1.0) for event in ready))
            release.set()
            (assistant_wire, assistant_failed), (failed_wire, failed) = await asyncio.gather(
                assistant_task,
                failed_task,
            )
        finally:
            release.set()

        assert "this is requested model content" in assistant_wire
        assert '"code": "provider_unavailable"' not in assistant_wire
        assert assistant_failed is False
        assert '"code": "provider_authentication_failed"' in failed_wire
        assert sentinel not in assistant_wire + failed_wire
        assert failed is True

    async def test_endpoint_wrappers_preserve_raw_valid_json_during_framed_error_overlap(
        self,
    ) -> None:
        ready = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        sentinel = "endpoint-wrapper-overlap-secret-/srv/provider"
        assistant_state: dict[str, object] = {}
        error_state: dict[str, object] = {}

        async def assistant_adapter():
            ready[0].set()
            await release.wait()
            yield '{"error":"this is requested model content"}'

        async def failing_adapter():
            ready[1].set()
            await release.wait()
            yield (
                'data: {"error":{"code":"provider_authentication_failed",'
                f'"message":"{sentinel}"}}}}\n\n'
            )

        assistant_stream = chat_endpoint._sanitize_provider_stream_call(
            assistant_adapter,
            assistant_state,
        )()
        error_stream = chat_endpoint._sanitize_provider_stream_call(
            failing_adapter,
            error_state,
        )()
        assistant_handler = StreamingResponseHandler("endpoint-raw-json", "gpt-4")
        error_handler = StreamingResponseHandler("endpoint-framed-error", "gpt-4")
        assistant_task = asyncio.create_task(
            _collect_handler_wire(assistant_handler, assistant_stream)
        )
        error_task = asyncio.create_task(
            _collect_handler_wire(error_handler, error_stream)
        )
        try:
            await asyncio.gather(*(asyncio.wait_for(event.wait(), 1.0) for event in ready))
            release.set()
            assistant_wire, error_wire = await asyncio.gather(
                assistant_task,
                error_task,
            )
        finally:
            release.set()

        assert "this is requested model content" in assistant_wire
        assert '"code": "provider_unavailable"' not in assistant_wire
        assert assistant_handler.error_occurred is False
        assert assistant_state.get("code") is None
        assert '"code": "provider_authentication_failed"' in error_wire
        assert sentinel not in assistant_wire + error_wire
        assert error_handler.error_occurred is True

    async def test_endpoint_wrappers_isolate_error_code_text_from_framed_error(
        self,
    ) -> None:
        """Bare model text must not acquire another stream's error provenance."""

        ready = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        sentinel = "endpoint-code-overlap-secret-/srv/provider"
        assistant_state: dict[str, object] = {}
        error_state: dict[str, object] = {}

        async def assistant_adapter():
            ready[0].set()
            await release.wait()
            yield "provider_unavailable"

        async def failing_adapter():
            ready[1].set()
            await release.wait()
            yield (
                'data: {"error":{"code":"provider_authentication_failed",'
                f'"message":"{sentinel}"}}}}\n\n'
            )

        assistant_stream = chat_endpoint._sanitize_provider_stream_call(
            assistant_adapter,
            assistant_state,
        )()
        error_stream = chat_endpoint._sanitize_provider_stream_call(
            failing_adapter,
            error_state,
        )()
        assistant_handler = StreamingResponseHandler("endpoint-code-text", "gpt-4")
        error_handler = StreamingResponseHandler("endpoint-code-error", "gpt-4")
        assistant_task = asyncio.create_task(
            _collect_handler_wire(assistant_handler, assistant_stream)
        )
        error_task = asyncio.create_task(
            _collect_handler_wire(error_handler, error_stream)
        )
        try:
            await asyncio.gather(*(asyncio.wait_for(event.wait(), 1.0) for event in ready))
            release.set()
            assistant_wire, error_wire = await asyncio.gather(
                assistant_task,
                error_task,
            )
        finally:
            release.set()

        assert "provider_unavailable" in assistant_wire
        assert '"error"' not in assistant_wire
        assert assistant_state.get("code") is None
        assert assistant_handler.error_occurred is False
        assert '"code": "provider_authentication_failed"' in error_wire
        assert error_state.get("code") == "provider_authentication_failed"
        assert sentinel not in assistant_wire + error_wire
        assert error_handler.error_occurred is True

    @pytest.mark.parametrize("stream_kind", ["sync", "async"])
    @pytest.mark.parametrize("framed", [False, True], ids=["raw-dict", "sse"])
    @pytest.mark.parametrize(
        "metadata",
        [
            {"code": "ok"},
            {"error": None},
        ],
        ids=["success-code", "null-error"],
    )
    async def test_success_metadata_is_never_normalized_as_provider_error(
        self,
        stream_kind: str,
        framed: bool,
        metadata: dict[str, object],
    ) -> None:
        payload = {
            **metadata,
            "choices": [{"delta": {"content": "healthy metadata output"}}],
        }
        chunk = f"data: {json.dumps(payload)}\n\n" if framed else payload
        handler = StreamingResponseHandler(
            f"conv_success_metadata_{stream_kind}_{framed}",
            "gpt-4",
        )

        if stream_kind == "async":
            async def provider_stream():
                yield chunk
                yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        else:
            def provider_stream():
                yield chunk
                yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        wire = "".join(
            [message async for message in handler.safe_stream_generator(provider_stream())]
        )

        assert "healthy metadata output" in wire
        assert '"code": "provider_unavailable"' not in wire
        assert handler.error_occurred is False

    async def test_concurrent_success_metadata_and_error_envelope_are_isolated(self) -> None:
        ready = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        sentinel = "metadata-error-secret-/srv/provider"

        async def healthy_stream():
            ready[0].set()
            await release.wait()
            yield (
                'data: {"error":null,"code":"ok",'
                '"choices":[{"delta":{"content":"healthy concurrent output"}}]}\n\n'
            )
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

        async def failed_stream():
            ready[1].set()
            await release.wait()
            yield (
                'data: {"error":{"code":"provider_authentication_failed",'
                f'"message":"{sentinel}"}}}}\n\n'
            )

        async def collect(conversation_id: str, stream: AsyncIterator[str]) -> tuple[str, bool]:
            handler = StreamingResponseHandler(conversation_id, "gpt-4")
            wire = "".join(
                [message async for message in handler.safe_stream_generator(stream)]
            )
            return wire, handler.error_occurred

        healthy_task = asyncio.create_task(collect("conv_metadata_ok", healthy_stream()))
        failed_task = asyncio.create_task(collect("conv_metadata_error", failed_stream()))
        await asyncio.gather(*(asyncio.wait_for(event.wait(), 1.0) for event in ready))
        release.set()
        (healthy_wire, healthy_failed), (failed_wire, failed) = await asyncio.gather(
            healthy_task,
            failed_task,
        )

        assert "healthy concurrent output" in healthy_wire
        assert '"code": "provider_unavailable"' not in healthy_wire
        assert healthy_failed is False
        assert '"code": "provider_authentication_failed"' in failed_wire
        assert sentinel not in failed_wire
        assert failed is True

    async def test_first_output_callback_error_never_logs_raw_detail(self) -> None:
        sentinel = "first-output-callback-secret-/srv/runtime"
        handler = StreamingResponseHandler("conv_first_output_callback", "gpt-4")

        def failing_callback() -> None:
            raise RuntimeError(sentinel)

        async def provider_stream():
            yield "safe output"

        logs: list[str] = []
        sink_id = logger.add(logs.append, format="{message}")
        try:
            wire = "".join(
                [
                    message
                    async for message in handler.safe_stream_generator(
                        provider_stream(),
                        on_first_output=failing_callback,
                    )
                ]
            )
        finally:
            logger.remove(sink_id)

        assert "safe output" in wire
        assert sentinel not in wire
        assert sentinel not in "".join(logs)

    async def test_concurrent_before_success_failure_is_bounded_and_isolated(self) -> None:
        sentinel = "before-success-secret-/srv/runtime"
        ready = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()

        async def provider_stream(index: int, content: str):
            ready[index].set()
            await release.wait()
            yield content

        def failing_before_success() -> None:
            raise RuntimeError(sentinel)

        async def collect(handler: StreamingResponseHandler, stream, callback=None) -> str:
            return "".join(
                [
                    message
                    async for message in handler.safe_stream_generator(
                        stream,
                        before_success_callback=callback,
                    )
                ]
            )

        failed_handler = StreamingResponseHandler("conv_before_success_failed", "gpt-4")
        healthy_handler = StreamingResponseHandler("conv_before_success_healthy", "gpt-4")
        logs: list[str] = []
        sink_id = logger.add(logs.append, format="{message}")
        try:
            failed_task = asyncio.create_task(
                collect(
                    failed_handler,
                    provider_stream(0, "bounded output"),
                    failing_before_success,
                )
            )
            healthy_task = asyncio.create_task(
                collect(healthy_handler, provider_stream(1, "healthy output"))
            )
            await asyncio.gather(*(asyncio.wait_for(event.wait(), 1.0) for event in ready))
            release.set()
            failed_wire, healthy_wire = await asyncio.gather(failed_task, healthy_task)
        finally:
            release.set()
            logger.remove(sink_id)

        assert sentinel not in failed_wire
        assert sentinel not in "".join(logs)
        assert '"code": "provider_unavailable"' in failed_wire
        assert failed_handler.error_occurred is True
        assert "healthy output" in healthy_wire
        assert healthy_handler.error_occurred is False

    async def test_post_output_noncooperative_adapter_cleanup_is_bounded_and_isolated(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            streaming_utils,
            "STREAM_TASK_CANCEL_DRAIN_SECONDS",
            0.01,
            raising=False,
        )
        monkeypatch.setattr(
            streaming_utils,
            "STREAM_CLEANUP_TIMEOUT_SECONDS",
            0.01,
            raising=False,
        )
        release = asyncio.Event()
        blocked = asyncio.Event()
        healthy_done = asyncio.Event()
        ticked = asyncio.Event()

        class ResistantAfterOutput:
            def __init__(self) -> None:
                self.first = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.first:
                    self.first = False
                    return 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                blocked.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await release.wait()
                    raise StopAsyncIteration from None

            async def aclose(self) -> None:
                if not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        await release.wait()

        async def healthy_adapter():
            yield 'data: {"choices":[{"delta":{"content":"healthy"}}]}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            healthy_done.set()
            yield "data: [DONE]\n\n"

        async def ticker() -> None:
            await asyncio.sleep(0)
            ticked.set()

        bad_state: dict[str, object] = {}
        bad = chat_endpoint._sanitize_provider_stream_call(
            lambda: ResistantAfterOutput(),
            bad_state,
        )()
        bad_gen = create_streaming_response_with_timeout(
            bad,
            "bad",
            "model",
            idle_timeout=0.01,
            heartbeat_interval=0.01,
        )
        good_gen = create_streaming_response_with_timeout(
            healthy_adapter(),
            "good",
            "model",
            idle_timeout=1,
            heartbeat_interval=0,
        )

        async def collect(gen) -> str:
            return "".join([chunk async for chunk in gen])

        bad_task = asyncio.create_task(collect(bad_gen))
        good_task = asyncio.create_task(collect(good_gen))
        tick_task = asyncio.create_task(ticker())
        await asyncio.wait_for(blocked.wait(), 1.0)
        try:
            good_wire = await asyncio.wait_for(good_task, timeout=1.0)
            await asyncio.wait_for(tick_task, timeout=1.0)
            bad_wire = await asyncio.wait_for(
                asyncio.shield(bad_task),
                timeout=1.0,
            )
            assert not release.is_set()
        finally:
            release.set()
            if not bad_task.done():
                await asyncio.wait_for(bad_task, 1.0)

        assert ticked.is_set() and healthy_done.is_set()
        assert "healthy" in good_wire and "ok" in bad_wire

    async def test_regular_def_async_stream_close_invocation_never_blocks_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(streaming_utils, "STREAM_CLEANUP_TIMEOUT_SECONDS", 0.01)
        close_started = threading.Event()
        release_close = threading.Event()
        close_finished = threading.Event()
        heartbeat = asyncio.Event()

        class RegularBlockingCloseStream:
            def __init__(self) -> None:
                self.sent = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.sent:
                    raise StopAsyncIteration
                self.sent = True
                return "safe regular-close output"

            def aclose(self):
                close_started.set()
                release_close.wait()
                close_finished.set()

        handler = StreamingResponseHandler("regular-close", "gpt-4")
        task = asyncio.create_task(
            _collect_handler_wire(handler, RegularBlockingCloseStream())
        )
        try:
            assert await asyncio.to_thread(close_started.wait, 1.0)
            asyncio.get_running_loop().call_soon(heartbeat.set)
            await asyncio.wait_for(heartbeat.wait(), timeout=1.0)
            wire = await asyncio.wait_for(task, timeout=1.0)
            assert not release_close.is_set()
        finally:
            release_close.set()
            assert await asyncio.to_thread(close_finished.wait, 1.0)
            if not task.done():
                await asyncio.wait_for(task, timeout=1.0)

        assert "safe regular-close output" in wire

    async def test_native_async_generator_close_does_not_nested_acquire_daemon(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pool = BoundedDaemonPool(capacity=1)
        monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
        permit_held = threading.Event()
        release_permit = threading.Event()

        async def native_stream():
            yield "safe output"

        stream = native_stream()
        worker = pool.start(
            lambda: (permit_held.set(), release_permit.wait()),
            name="native-close-capacity-holder",
        )
        try:
            assert await asyncio.to_thread(permit_held.wait, 1.0)
            await streaming_utils.invoke_stream_close_bounded(
                stream.aclose,
                timeout=1.0,
            )
            assert not release_permit.is_set()
        finally:
            release_permit.set()
            worker.join(timeout=1.0)

        assert pool.active_count == 0

    async def test_sync_bridge_admission_failure_uses_reserved_cleanup_capacity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        admission_rejected = threading.Event()
        holder_started = threading.Event()
        release_holder = threading.Event()
        raw_next_called = threading.Event()
        raw_closed = threading.Event()

        class ObservingPool(BoundedDaemonPool):
            def start(self, target, *, name, released_event=None):
                try:
                    return super().start(
                        target,
                        name=name,
                        released_event=released_event,
                    )
                except bounded_daemon_module.DaemonCapacityError:
                    admission_rejected.set()
                    raise

        class RawSyncIterator:
            def __iter__(self):
                return self

            def __next__(self):
                raw_next_called.set()
                raise StopIteration

            def close(self):
                raw_closed.set()

        pool = ObservingPool(capacity=1)
        monkeypatch.setattr(bounded_daemon_module, "STREAM_DAEMON_POOL", pool)
        monkeypatch.setattr(
            streaming_utils,
            "STREAM_CLEANUP_TIMEOUT_SECONDS",
            1.0,
        )
        holder = pool.start(
            lambda: (holder_started.set(), release_holder.wait()),
            name="bridge-admission-holder",
        )
        handler = StreamingResponseHandler("bridge-admission", "gpt-4")
        collect_task = asyncio.create_task(
            _collect_handler_wire(handler, RawSyncIterator())
        )
        try:
            assert await asyncio.to_thread(holder_started.wait, 1.0)
            assert await asyncio.to_thread(admission_rejected.wait, 1.0)
            assert not raw_next_called.is_set()
            assert await asyncio.to_thread(raw_closed.wait, 1.0)
            assert not release_holder.is_set()
            release_holder.set()
            wire = await asyncio.wait_for(collect_task, timeout=1.0)
        finally:
            release_holder.set()
            holder.join(timeout=1.0)
            if not collect_task.done():
                await asyncio.wait_for(collect_task, timeout=1.0)

        assert '"code": "provider_unavailable"' in wire
        assert not raw_next_called.is_set()
        assert pool.active_count == 0

    async def test_endpoint_sanitizer_regular_def_aclose_is_off_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            chat_endpoint,
            "PROVIDER_STREAM_PRIME_CLEANUP_TIMEOUT_SECONDS",
            0.01,
        )
        close_started = threading.Event()
        release_close = threading.Event()
        heartbeat = asyncio.Event()

        class EmptyBlockingCloseStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            def aclose(self):
                close_started.set()
                release_close.wait()

        wrapped = chat_endpoint._sanitize_provider_stream_call(
            EmptyBlockingCloseStream,
            {},
        )()
        task = asyncio.create_task(_collect_async_items(wrapped))
        try:
            assert await asyncio.to_thread(close_started.wait, 1.0)
            asyncio.get_running_loop().call_soon(heartbeat.set)
            await asyncio.wait_for(heartbeat.wait(), timeout=1.0)
            assert await asyncio.wait_for(task, timeout=1.0) == []
            assert not release_close.is_set()
        finally:
            release_close.set()
            if not task.done():
                await asyncio.wait_for(task, timeout=1.0)

    async def test_endpoint_sync_close_handoff_releases_capacity_before_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pool = BoundedDaemonPool(capacity=1)
        monkeypatch.setattr(
            bounded_daemon_module,
            "STREAM_CLEANUP_DAEMON_POOL",
            pool,
        )
        close_called = threading.Event()
        retry_called = threading.Event()

        await chat_endpoint._call_sync_stream_close(close_called.set, timeout=1.0)
        retry_worker = pool.start(retry_called.set, name="close-handoff-retry")
        retry_worker.join(timeout=1.0)

        assert close_called.is_set()
        assert retry_called.is_set()
        assert pool.active_count == 0

    async def test_failing_regular_close_releases_capacity_before_error_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target_finished = threading.Event()
        allow_release = threading.Event()
        healthy_called = threading.Event()

        class GatedReleasePool(BoundedDaemonPool):
            def start(self, target, *, name, released_event=None):
                def gated_target():
                    target()
                    target_finished.set()
                    allow_release.wait()

                return super().start(
                    gated_target,
                    name=name,
                    released_event=released_event,
                )

        pool = GatedReleasePool(capacity=1)
        monkeypatch.setattr(
            bounded_daemon_module,
            "STREAM_CLEANUP_DAEMON_POOL",
            pool,
        )

        def failing_close():
            raise RuntimeError("private close failure")

        failing_task = asyncio.create_task(
            streaming_utils.invoke_stream_close_bounded(
                failing_close,
                timeout=1.0,
            )
        )
        try:
            assert await asyncio.to_thread(target_finished.wait, 1.0)
            await asyncio.sleep(0)
            assert not failing_task.done()
            allow_release.set()
            with pytest.raises(RuntimeError, match="private close failure"):
                await asyncio.wait_for(failing_task, timeout=1.0)

            await streaming_utils.invoke_stream_close_bounded(
                healthy_called.set,
                timeout=1.0,
            )
        finally:
            allow_release.set()
            if not failing_task.done():
                with pytest.raises(RuntimeError):
                    await asyncio.wait_for(failing_task, timeout=1.0)

        assert healthy_called.is_set()
        assert pool.active_count == 0

    async def test_owned_regular_close_awaits_late_awaitable_before_returning(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pool = BoundedDaemonPool(capacity=1)
        monkeypatch.setattr(
            bounded_daemon_module,
            "STREAM_CLEANUP_DAEMON_POOL",
            pool,
        )
        actual_closed = asyncio.Event()

        async def actual_close() -> str:
            actual_closed.set()
            return "closed"

        def delayed_close():
            time.sleep(0.03)
            return actual_close()

        result = await asyncio.wait_for(
            streaming_utils.invoke_owned_stream_close(
                delayed_close,
                timeout=0.005,
            ),
            timeout=1.0,
        )

        assert result == "closed"
        assert actual_closed.is_set()
        assert pool.active_count == 0

    async def test_cancelled_owned_close_finishes_late_returned_awaitable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cancellation cannot release ownership before a late close completes."""

        pool = BoundedDaemonPool(capacity=1)
        monkeypatch.setattr(
            bounded_daemon_module,
            "STREAM_CLEANUP_DAEMON_POOL",
            pool,
        )
        sync_close_started = threading.Event()
        release_sync_close = threading.Event()
        async_close_started = asyncio.Event()
        release_async_close = asyncio.Event()
        async_close_finished = asyncio.Event()

        async def actual_close() -> None:
            async_close_started.set()
            await release_async_close.wait()
            async_close_finished.set()

        def delayed_close():
            sync_close_started.set()
            release_sync_close.wait(timeout=2.0)
            return actual_close()

        caller = asyncio.create_task(
            streaming_utils.invoke_owned_stream_close(
                delayed_close,
                timeout=0.005,
            )
        )
        try:
            while not sync_close_started.is_set():
                await asyncio.sleep(0)
            caller.cancel()
            release_sync_close.set()

            await asyncio.wait_for(async_close_started.wait(), timeout=1.0)
            await asyncio.sleep(0)
            assert caller.done() is False
            assert async_close_finished.is_set() is False

            release_async_close.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, timeout=1.0)
        finally:
            release_sync_close.set()
            release_async_close.set()

        assert async_close_finished.is_set()
        assert pool.active_count == 0

    async def test_owned_close_propagates_child_self_cancellation(self) -> None:
        """A close operation cancelling itself must not enter an infinite drain."""

        async def self_cancelling_close() -> None:
            current = asyncio.current_task()
            assert current is not None
            current.cancel()
            await asyncio.sleep(0)

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(
                streaming_utils.invoke_owned_stream_close(
                    self_cancelling_close,
                    timeout=0.1,
                ),
                timeout=1.0,
            )

    async def test_raw_sync_stream_blocking_close_never_blocks_loop_or_leaks(
        self,
    ) -> None:
        sentinel = "raw-sync-close-secret-/srv/provider"
        next_started = threading.Event()
        release_next = threading.Event()
        close_started = threading.Event()
        release_close = threading.Event()
        close_finished = threading.Event()
        loop_responsive = asyncio.Event()

        class BlockingCloseIterator:
            def __init__(self) -> None:
                self.yielded = False
                self.close_calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                if not self.yielded:
                    self.yielded = True
                    return "safe output"
                next_started.set()
                release_next.wait()
                raise StopIteration

            def close(self) -> None:
                self.close_calls += 1
                close_started.set()
                release_close.wait()
                close_finished.set()
                raise RuntimeError(sentinel)

        iterator = BlockingCloseIterator()
        handler = StreamingResponseHandler("conv_blocking_raw_close", "gpt-4")
        body = handler.safe_stream_generator(iterator)
        while True:
            chunk = await body.__anext__()
            if "safe output" in chunk:
                break
        assert await asyncio.to_thread(next_started.wait, 1.0)

        logs: list[str] = []
        sink_id = logger.add(logs.append, format="{message}")
        close_task = asyncio.create_task(body.aclose())
        try:
            release_next.set()
            assert await asyncio.to_thread(close_started.wait, 1.0)
            asyncio.get_running_loop().call_soon(loop_responsive.set)
            await asyncio.wait_for(loop_responsive.wait(), timeout=1.0)
            await asyncio.wait_for(close_task, timeout=1.0)
            assert not release_close.is_set()
            assert not close_finished.is_set()
        finally:
            release_next.set()
            release_close.set()
            assert await asyncio.to_thread(close_finished.wait, 1.0)
            if not close_task.done():
                await asyncio.wait_for(close_task, timeout=1.0)
            logger.remove(sink_id)

        assert iterator.close_calls == 1
        assert sentinel not in "".join(logs)

    @pytest.mark.parametrize("failure_site", ["next", "close"])
    async def test_sync_bridge_sanitizes_arbitrary_daemon_failures_once(
        self,
        failure_site: str,
    ) -> None:
        sentinel = f"sync-bridge-{failure_site}-secret-/srv/provider"
        close_called = threading.Event()
        daemon_failures: list[threading.ExceptHookArgs] = []
        finalized: list[dict[str, bool]] = []

        class ArbitraryMetadataFailure(Exception):
            pass

        class ArbitraryAdapterFailure(Exception):
            @property
            def code(self) -> str:
                raise ArbitraryMetadataFailure(sentinel)

        class FailingIterator:
            def __init__(self) -> None:
                self._finished = False

            def __iter__(self):
                return self

            def __next__(self):
                if failure_site == "next":
                    raise ArbitraryAdapterFailure(sentinel)
                if self._finished:
                    raise StopIteration
                self._finished = True
                return "safe output"

            def close(self) -> None:
                close_called.set()
                if failure_site == "close":
                    raise ArbitraryAdapterFailure(sentinel)

        async def finalize(**outcome: bool) -> None:
            finalized.append(outcome)

        original_excepthook = threading.excepthook
        logs: list[str] = []
        sink_id = logger.add(logs.append, format="{message}")
        threading.excepthook = daemon_failures.append
        try:
            wire = "".join(
                [
                    chunk
                    async for chunk in create_streaming_response_with_timeout(
                        FailingIterator(),
                        f"sync-bridge-{failure_site}",
                        "model",
                        finalize_callback=finalize,
                        heartbeat_interval=0,
                    )
                ]
            )
        finally:
            threading.excepthook = original_excepthook
            logger.remove(sink_id)

        assert close_called.is_set()
        assert daemon_failures == []
        assert wire.count('"code": "provider_unavailable"') == 1
        assert sentinel not in wire
        assert sentinel not in "".join(logs)
        assert finalized == [
            {"success": False, "cancelled": False, "error": True}
        ]

    async def test_sync_bridge_completion_waits_for_close_handoff(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        close_started = threading.Event()
        release_close = threading.Event()
        close_finished = threading.Event()
        monkeypatch.setattr(
            streaming_utils,
            "STREAM_CLEANUP_TIMEOUT_SECONDS",
            0.01,
        )

        class BlockingCloseIterator:
            def __iter__(self):
                return self

            def __next__(self):
                raise StopIteration

            def close(self) -> None:
                close_started.set()
                release_close.wait(timeout=2.0)
                close_finished.set()

        collect_task = asyncio.create_task(
            _collect_async_items(
                streaming_utils._async_iter_sync_stream(BlockingCloseIterator())
            )
        )
        try:
            assert await asyncio.to_thread(close_started.wait, 1.0)
            await asyncio.sleep(0.03)
            assert collect_task.done() is False
        finally:
            release_close.set()

        assert await asyncio.wait_for(collect_task, timeout=1.0) == []
        assert close_finished.is_set()

    async def test_bytes_stream_processing(self):
        """Test processing of byte stream."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        async def mock_stream():
            yield b"Byte"
            yield b" "
            yield b"Stream"

        messages = []
        async for message in handler.safe_stream_generator(mock_stream()):
            messages.append(message)

        assert "".join(handler.full_response) == "Byte Stream"

    async def test_stream_cancellation(self):
        """Test stream cancellation handling."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        async def mock_stream():
            yield "Start"
            handler.cancel()
            yield "Should not appear"

        messages = []
        async for message in handler.safe_stream_generator(mock_stream()):
            messages.append(message)

        # Should stop after cancellation
        assert "".join(handler.full_response) == "Start"
        assert handler.is_cancelled is True

    async def test_stream_error_handling(self):
        """Test error handling during streaming."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        async def mock_stream():
            yield "Before error"
            raise ValueError("Stream error")

        messages = []
        async for message in handler.safe_stream_generator(mock_stream()):
            messages.append(message)

        # Should have error message
        error_messages = [m for m in messages if "error" in m]
        assert len(error_messages) > 0
        assert handler.error_occurred is True

    @pytest.mark.parametrize(
        "provider_chunk",
        [
            "secret output",
            'data: {"choices":[{"delta":{"content":"secret output"}}]}\n\n',
        ],
    )
    async def test_text_transform_error_stops_without_emitting_original_content(
        self,
        provider_chunk: str,
    ) -> None:
        """Text-transform failures should fail closed instead of leaking raw chunks."""
        sentinel = "transform-secret-/srv/moderation"

        def failing_transform(_text: str) -> str:
            raise RuntimeError(sentinel)

        handler = StreamingResponseHandler(
            "conv_transform_fail",
            "gpt-4",
            text_transform=failing_transform,
        )

        async def mock_stream() -> AsyncIterator[str]:
            yield provider_chunk

        messages = []
        logs: list[str] = []
        sink_id = logger.add(logs.append, format="{message}")
        try:
            async for message in handler.safe_stream_generator(mock_stream()):
                messages.append(message)
        finally:
            logger.remove(sink_id)

        combined_messages = "".join(messages)
        assert "secret output" not in combined_messages
        assert sentinel not in combined_messages
        assert sentinel not in "".join(logs)
        assert handler.full_response == []
        assert any('"error"' in message for message in messages)
        assert handler.error_occurred is True

    @pytest.mark.parametrize("upstream_fails", [False, True])
    async def test_text_transform_flush_error_never_logs_raw_detail(
        self,
        upstream_fails: bool,
    ) -> None:
        sentinel = "flush-secret-/srv/provider-response"

        class FailingFlushTransform:
            def __call__(self, text: str) -> str:
                return text

            def flush(self) -> str:
                raise RuntimeError(sentinel)

        handler = StreamingResponseHandler(
            "conv_transform_flush_fail",
            "gpt-4",
            text_transform=FailingFlushTransform(),
        )

        async def mock_stream() -> AsyncIterator[str]:
            yield "safe output"
            if upstream_fails:
                raise RuntimeError("bounded-upstream-failure")

        messages: list[str] = []
        logs: list[str] = []
        sink_id = logger.add(logs.append, format="{message}")
        try:
            async for message in handler.safe_stream_generator(mock_stream()):
                messages.append(message)
        finally:
            logger.remove(sink_id)

        assert sentinel not in "".join(messages)
        assert sentinel not in "".join(logs)

    async def test_save_callback_execution(self):
        """Test save callback is executed."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")
        save_called = False
        saved_content = None

        async def save_callback(content):
            nonlocal save_called, saved_content
            save_called = True
            saved_content = content

        async def mock_stream():
            yield "Test content"

        messages = []
        async for message in handler.safe_stream_generator(mock_stream(), save_callback):
            messages.append(message)

        assert save_called is True
        assert saved_content == "Test content"

    async def test_save_callback_error(self):
        """Test handling of save callback errors."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        async def failing_callback(_content):
            raise Exception("Save failed")

        async def mock_stream():
            yield "Content"

        with patch('tldw_Server_API.app.core.Chat.streaming_utils.logger') as mock_logger:
            messages = []
            async for message in handler.safe_stream_generator(mock_stream(), failing_callback):
                messages.append(message)

            # Should log error but not crash
            mock_logger.error.assert_called()
            assert "Content" in "".join(handler.full_response)

    async def test_cancelled_error_handling(self):
        """Test handling of asyncio.CancelledError."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        async def mock_stream():
            yield "Start"
            raise asyncio.CancelledError()

        messages = []
        with pytest.raises(asyncio.CancelledError):
            async for message in handler.safe_stream_generator(mock_stream()):
                messages.append(message)

        assert handler.is_cancelled is True
        # Should have logged disconnection
        second_handler = StreamingResponseHandler("conv_456", "gpt-4")
        with patch('tldw_Server_API.app.core.Chat.streaming_utils.logger') as mock_logger:
            with pytest.raises(asyncio.CancelledError):
                async for _ in second_handler.safe_stream_generator(mock_stream()):
                    pass
            mock_logger.info.assert_called()

    async def test_finalize_callback_invoked_on_cancel(self):
        handler = StreamingResponseHandler("conv_finalize_cancel", "gpt-4")
        finalize_callback = AsyncMock()

        async def mock_stream():
            yield "Start"
            handler.cancel()
            yield "After cancel"

        async for _ in handler.safe_stream_generator(
            mock_stream(),
            finalize_callback=finalize_callback,
        ):
            pass

        finalize_callback.assert_awaited_once()
        kwargs = finalize_callback.await_args.kwargs
        assert kwargs["success"] is False
        assert kwargs["cancelled"] is True
        assert kwargs["error"] is False

    async def test_finalize_callback_invoked_on_error(self):
        handler = StreamingResponseHandler("conv_finalize_error", "gpt-4")
        finalize_callback = AsyncMock()

        async def mock_stream():
            yield "Before error"
            raise RuntimeError("boom")

        async for _ in handler.safe_stream_generator(
            mock_stream(),
            finalize_callback=finalize_callback,
        ):
            pass

        finalize_callback.assert_awaited_once()
        kwargs = finalize_callback.await_args.kwargs
        assert kwargs["success"] is False
        assert kwargs["cancelled"] is False
        assert kwargs["error"] is True

    async def test_stream_metadata(self):
        """Test stream metadata messages."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")
        handler.system_message_id = "sys_123"
        handler.continuation_metadata = {
            "applied": True,
            "mode": "branch",
            "from_message_id": "anchor-123",
        }

        async def mock_stream():
            yield "Content"

        async def save_callback(_content):
            return "msg_456"

        messages = []
        async for message in handler.safe_stream_generator(mock_stream(), save_callback):
            messages.append(message)

        # Check stream_start event
        start_msgs = [m for m in messages if "stream_start" in m]
        assert len(start_msgs) == 1
        assert "conv_123" in start_msgs[0]
        start_lines = [line for line in start_msgs[0].splitlines() if line.startswith("data: ")]
        assert len(start_lines) == 1
        start_payload = json.loads(start_lines[0][6:])
        assert start_payload.get("tldw_system_message_id") == "sys_123"
        assert start_payload.get("tldw_conversation_id") == "conv_123"
        assert start_payload.get("tldw_continuation", {}).get("mode") == "branch"

        # Check stream_end event
        end_msgs = [m for m in messages if "stream_end" in m]
        assert len(end_msgs) == 1
        assert "success" in end_msgs[0]
        end_lines = [line for line in end_msgs[0].splitlines() if line.startswith("data: ")]
        assert len(end_lines) == 1
        end_payload = json.loads(end_lines[0][6:])
        assert end_payload.get("tldw_message_id") == "msg_456"
        assert end_payload.get("tldw_system_message_id") == "sys_123"
        assert end_payload.get("tldw_conversation_id") == "conv_123"
        assert end_payload.get("tldw_continuation", {}).get("from_message_id") == "anchor-123"

    async def test_save_callback_can_emit_additional_events(self):
        handler = StreamingResponseHandler("conv_events", "gpt-4")
        tool_result_payload = {
            "tool_results": [
                {
                    "tool_call_id": "c1",
                    "name": "notes.search",
                    "ok": True,
                    "content": "{\"ok\":true}",
                }
            ]
        }

        async def mock_stream():
            yield "Content"

        async def save_callback(_content, _tool_calls=None, _function_call=None):
            return {
                "saved_message_id": "msg_stream_1",
                "events": [
                    {
                        "event": "tool_results",
                        "data": tool_result_payload,
                    }
                ],
            }

        messages = []
        async for message in handler.safe_stream_generator(mock_stream(), save_callback):
            messages.append(message)

        tool_events = [m for m in messages if m.startswith("event: tool_results")]
        assert len(tool_events) == 1
        tool_lines = [line for line in tool_events[0].splitlines() if line.startswith("data: ")]
        assert len(tool_lines) == 1
        payload = json.loads(tool_lines[0][6:])
        assert payload["tool_results"][0]["tool_call_id"] == "c1"
        assert payload["tldw_message_id"] == "msg_stream_1"
        assert payload["tldw_conversation_id"] == "conv_events"

        tool_idx = next(i for i, msg in enumerate(messages) if msg.startswith("event: tool_results"))
        finish_idx = next(i for i, msg in enumerate(messages) if '"finish_reason": "stop"' in msg)
        end_idx = next(i for i, msg in enumerate(messages) if msg.startswith("event: stream_end"))
        done_idx = next(i for i, msg in enumerate(messages) if "data: [DONE]" in msg)
        assert tool_idx < finish_idx < end_idx < done_idx

    @pytest.mark.asyncio
    async def test_sync_stream_closed_on_cancel(self):
        """Ensure underlying sync generator is explicitly closed on cancel."""
        handler = StreamingResponseHandler("conv_close_sync", "gpt-4")

        closed_flag = {"closed": False}

        def provider_stream():

            try:
                yield "A"
                yield "B"
            finally:
                closed_flag["closed"] = True

        async def drive():
            agen = handler.safe_stream_generator(provider_stream())
            # Consume first yield (stream_start)
            await agen.__anext__()
            # Consume first chunk
            await agen.__anext__()
            # Cancel and close early
            handler.cancel()
            await agen.aclose()

        await drive()
        assert closed_flag["closed"] is True

    async def test_done_message_format(self):
        """Test OpenAI-compatible [DONE] message."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        async def mock_stream():
            yield "Test"

        messages = []
        async for message in handler.safe_stream_generator(mock_stream()):
            messages.append(message)

        # Find done message
        done_msgs = [m for m in messages if "[DONE]" in m]
        assert len(done_msgs) == 1

        # Find completion chunk
        completion_msgs = [m for m in messages if "finish_reason" in m]
        assert len(completion_msgs) == 1

        # Parse and verify format
        for msg in completion_msgs:
            if msg.startswith("data: ") and "[DONE]" not in msg:
                data = json.loads(msg[6:msg.index("\n")])
                assert "id" in data
                assert "object" in data
                assert data["object"] == "chat.completion.chunk"
                assert "choices" in data
                assert data["choices"][0]["finish_reason"] == "stop"

    async def test_post_done_ignored_content_does_not_refresh_activity(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        handler = StreamingResponseHandler("conv-post-done", "gpt-4")
        activity_updates: list[None] = []
        monkeypatch.setattr(
            handler,
            "update_activity",
            lambda: activity_updates.append(None),
        )

        async def provider_stream():
            yield "data: [DONE]\n\n"
            yield 'data: {"choices":[{"delta":{"content":"ignored-junk"}}]}\n\n'

        wire = await _collect_handler_wire(handler, provider_stream())

        assert "ignored-junk" not in wire
        assert len(activity_updates) == 2


@pytest.mark.asyncio
class TestCreateStreamingResponseWithTimeout:
    """Test the main streaming response creation function."""

    async def test_basic_streaming(self):
        """Test basic streaming functionality."""
        # Use the StreamingResponseHandler directly for cleaner testing
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        async def mock_stream():
            yield "Hello"
            yield " World"

        save_called = False
        saved_content = None

        async def save_callback(content):
            nonlocal save_called, saved_content
            save_called = True
            saved_content = content

        messages = []
        async for message in handler.safe_stream_generator(mock_stream(), save_callback):
            messages.append(message)

        assert len(messages) > 0
        assert save_called is True
        assert saved_content == "Hello World"

    @pytest.mark.asyncio
    async def test_create_streaming_response_attaches_continuation_metadata(self):
        async def mock_stream():
            yield "Hello"

        generator = create_streaming_response_with_timeout(
            stream=mock_stream(),
            conversation_id="conv_meta",
            model_name="gpt-4",
            continuation_metadata={
                "applied": True,
                "mode": "append",
                "from_message_id": "anchor-meta",
            },
        )

        first = await generator.__anext__()
        assert "event: stream_start" in first
        data_line = next(line for line in first.splitlines() if line.startswith("data: "))
        payload = json.loads(data_line[6:])
        assert payload["tldw_continuation"]["mode"] == "append"

        try:
            await generator.aclose()
        except Exception:
            _ = None

    @pytest.mark.skip(reason="Complex async coordination test - may be flaky")
    async def test_heartbeat_integration(self):
        """Test heartbeat integration with streaming."""
        async def slow_stream():
            yield "Start"
            await asyncio.sleep(0.2)
            yield "End"

        messages = []
        generator = create_streaming_response_with_timeout(
            stream=slow_stream(),
            conversation_id="conv_123",
            model_name="gpt-4",
            heartbeat_interval=0.1
        )

        async for message in generator:
            messages.append(message)
            if "[DONE]" in message or "stream_end" in message:
                break

        # Should have heartbeat messages
        heartbeat_msgs = [m for m in messages if "heartbeat" in m]
        assert len(heartbeat_msgs) >= 1

    @pytest.mark.asyncio
    async def test_async_generator_close_no_runtime_error(self):
        """Closing the async generator should not raise RuntimeError on GeneratorExit."""
        handler = StreamingResponseHandler("conv_close", "gpt-4")

        async def mock_stream():
            yield "Hello"
            await asyncio.sleep(0)
            yield "World"

        agen = handler.safe_stream_generator(mock_stream())

        # Prime the generator (consume stream_start)
        _ = await agen.__anext__()

        # Closing should not raise
        try:
            await agen.aclose()
        except Exception as e:
            pytest.fail(f"aclose() raised an exception: {e}")

        assert handler.is_cancelled is True

    @pytest.mark.asyncio
    async def test_stream_start_emitted_once_and_close_early(self):
        """Ensure stream_start is emitted only once and early close is clean."""
        async def slow_stream():
            yield "chunk1"
            await asyncio.sleep(0.05)
            yield "chunk2"

        gen = create_streaming_response_with_timeout(
            stream=slow_stream(),
            conversation_id="conv_start_once",
            model_name="gpt-4",
            idle_timeout=5,
            heartbeat_interval=0.2,
        )

        # Get first message and ensure it's stream_start
        first = await gen.__anext__()
        assert "event: stream_start" in first

        # Pull a couple more messages, then close early
        messages = []
        for _ in range(3):
            try:
                messages.append(await gen.__anext__())
            except StopAsyncIteration:
                break

        # Close early; should not raise
        try:
            await gen.aclose()
        except Exception as e:
            pytest.fail(f"Early aclose() raised an exception: {e}")

        # Verify only one stream_start was seen in all collected messages
        start_count = sum(1 for m in [first] + messages if "event: stream_start" in m)
        assert start_count == 1


class TestConstants:
    """Test module constants."""

    def test_default_timeout(self):

        """Test default timeout value."""
        assert STREAMING_IDLE_TIMEOUT == 300  # 5 minutes

    def test_default_heartbeat(self):

        """Test default heartbeat interval."""
        # Legacy heartbeat is disabled via config (0) to avoid duplicate heartbeats
        # when unified streaming is enabled. Expect 0 in test configuration.
        assert HEARTBEAT_INTERVAL == 0


class TestStreamingResponseHandlerIntegration:
    """Integration tests for complete streaming workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_success(self):
        """Test complete successful streaming workflow."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")
        save_result = None

        async def save_callback(content):
            nonlocal save_result
            save_result = content

        async def mock_llm_stream():
            responses = ["I", " am", " an", " AI", " assistant"]
            for r in responses:
                yield r
                await asyncio.sleep(0.01)

        all_messages = []
        async for message in handler.safe_stream_generator(mock_llm_stream(), save_callback):
            all_messages.append(message)

        # Verify complete flow
        assert len(all_messages) > 0
        assert handler.error_occurred is False
        assert handler.is_cancelled is False
        assert save_result == "I am an AI assistant"
        assert "".join(handler.full_response) == save_result

    @pytest.mark.asyncio
    async def test_full_workflow_with_error(self):
        """Test complete workflow with error handling."""
        handler = StreamingResponseHandler("conv_123", "gpt-4")

        async def mock_failing_stream():
            yield "Partial"
            yield " response"
            raise ConnectionError("LLM connection lost")

        messages = []
        async for message in handler.safe_stream_generator(mock_failing_stream()):
            messages.append(message)

        assert handler.error_occurred is True
        assert "".join(handler.full_response) == "Partial response"

        # Should have error in messages
        error_found = any("error" in m for m in messages)
        assert error_found is True


@pytest.mark.asyncio
class TestSSENormalization:
    """Tests that upstream provider SSE frames are normalized to plain text chunks."""

    @pytest.mark.parametrize("raw_text", ["id: assistant literal", "retry: assistant literal"])
    async def test_raw_non_sse_control_prefix_is_assistant_content(self, raw_text):
        handler = StreamingResponseHandler("conv_raw_control_prefix", "gpt-4")

        async def provider_stream():
            yield raw_text
            yield "data: [DONE]\n\n"

        messages = [message async for message in handler.safe_stream_generator(provider_stream())]
        content_messages = [
            json.loads(message[6 : message.index("\n")])
            for message in messages
            if message.startswith("data: ") and '"choices"' in message and '"content"' in message
        ]

        assert [message["choices"][0]["delta"]["content"] for message in content_messages] == [raw_text]
        assert handler.full_response == [raw_text]

    @pytest.mark.parametrize("control_line", ["id: stream-7", "retry: 1500"])
    async def test_sse_control_only_line_is_not_assistant_content(self, control_line):
        handler = StreamingResponseHandler("conv_sse_control", "gpt-4")

        async def provider_stream():
            yield f"{control_line}\n\n"
            yield "data: [DONE]\n\n"

        messages = [message async for message in handler.safe_stream_generator(provider_stream())]
        content_messages = [
            message
            for message in messages
            if message.startswith("data: ") and '"choices"' in message and '"content"' in message
        ]

        assert content_messages == []
        assert handler.full_response == []

    async def test_sse_event_id_retry_fields_preserve_data_processing(self):
        handler = StreamingResponseHandler("conv_sse_controls_and_data", "gpt-4")
        payload = {"choices": [{"delta": {"content": "Hello"}}]}

        async def provider_stream():
            yield ("event: chunk\n" "id: stream-8\n" "retry: 2000\n" f"data: {json.dumps(payload)}\n\n")
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        messages = [message async for message in handler.safe_stream_generator(provider_stream())]
        content_messages = [
            json.loads(message[6 : message.index("\n")])
            for message in messages
            if message.startswith("data: ") and '"choices"' in message and '"content"' in message
        ]

        assert [message["choices"][0]["delta"]["content"] for message in content_messages] == ["Hello"]
        assert handler.full_response == ["Hello"]

    async def test_openai_sse_done_sentinel_is_valid_without_finish_reason(self):
        handler = StreamingResponseHandler("conv_sse_done_only", "gpt-4")

        async def provider_stream():
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            yield "data: [DONE]\n\n"

        wire = await _collect_handler_wire(handler, provider_stream())

        assert "Hello" in wire
        assert '"code": "provider_unavailable"' not in wire
        assert '"success": true' in wire
        assert handler.error_occurred is False

    async def test_openai_sse_exhaustion_without_terminal_signal_fails_closed(self):
        handler = StreamingResponseHandler("conv_sse_truncated", "gpt-4")

        async def provider_stream():
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

        wire = await _collect_handler_wire(handler, provider_stream())

        assert '"code": "provider_unavailable"' in wire
        assert '"success": false' in wire
        assert handler.error_occurred is True

    async def test_openai_like_sse_is_normalized(self):
        handler = StreamingResponseHandler("conv_sse", "gpt-4")

        # Upstream emits OpenAI-style SSE frames (each with trailing blank line)
        chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
        chunk2 = {"choices": [{"delta": {"content": " world"}}]}

        async def provider_stream():
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        messages = []
        async for message in handler.safe_stream_generator(provider_stream()):
            messages.append(message)

        # Extract only content chunks we emit to client
        content_lines = [m for m in messages if m.startswith("data: ") and '"choices"' in m and '"delta"' in m and '"content"' in m]
        parsed = []
        for m in content_lines:
            data = json.loads(m[6:m.index("\n")])
            assert data.get("tldw_conversation_id") == "conv_sse"
            content = data["choices"][0]["delta"].get("content")
            if content:
                parsed.append(content)

        assert parsed == ["Hello", " world"]
        assert any("data: [DONE]" in m for m in messages)
        assert "".join(handler.full_response) == "Hello world"

    async def test_multiline_sse_chunk_with_event_and_multiple_data_lines(self):
        handler = StreamingResponseHandler("conv_sse2", "gpt-4")

        # Single upstream chunk that contains an event plus two data lines and a DONE
        part_a = {"choices": [{"delta": {"content": "Part"}}]}
        part_b = {"choices": [{"delta": {"content": " A"}}]}
        multi = (
            "event: chunk\n"
            f"data: {json.dumps(part_a)}\n"
            f"data: {json.dumps(part_b)}\n"
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
            "data: [DONE]\n\n"
        )

        async def provider_stream():
            yield multi

        messages = []
        async for message in handler.safe_stream_generator(provider_stream()):
            messages.append(message)

        content_lines = [m for m in messages if m.startswith("data: ") and '"choices"' in m and '"delta"' in m and '"content"' in m]
        parsed = []
        for m in content_lines:
            data = json.loads(m[6:m.index("\n")])
            content = data["choices"][0]["delta"].get("content")
            if content:
                parsed.append(content)

        assert parsed == ["Part", " A"]
        assert any("data: [DONE]" in m for m in messages)
        assert "".join(handler.full_response) == "Part A"

    async def test_upstream_error_is_forwarded(self):
        handler = StreamingResponseHandler("conv_err", "gpt-4")

        async def provider_stream():
            yield "data: {\"error\": {\"message\": \"oops\", \"type\": \"provider\"}}\n\n"

        messages = []
        async for message in handler.safe_stream_generator(provider_stream()):
            messages.append(message)

        # We should have an error message surfaced to the client
        error_msgs = [m for m in messages if m.startswith("data: ") and '"error"' in m]
        assert len(error_msgs) >= 1
        assert handler.error_occurred is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

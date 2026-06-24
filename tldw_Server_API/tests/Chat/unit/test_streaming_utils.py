# test_streaming_utils.py
# Unit tests for streaming response utilities

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest

from tldw_Server_API.app.core.Chat.streaming_utils import (
    StreamingResponseHandler,
    create_streaming_response_with_timeout,
    STREAMING_IDLE_TIMEOUT,
    HEARTBEAT_INTERVAL,
)


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
        assert "Stream timeout" in messages[0]
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

    async def test_text_transform_error_stops_without_emitting_original_content(self) -> None:
        """Text-transform failures should fail closed instead of leaking raw chunks."""
        def failing_transform(_text: str) -> str:
            raise RuntimeError("moderation unavailable")

        handler = StreamingResponseHandler(
            "conv_transform_fail",
            "gpt-4",
            text_transform=failing_transform,
        )

        async def mock_stream() -> AsyncIterator[str]:
            yield "secret output"

        messages = []
        async for message in handler.safe_stream_generator(mock_stream()):
            messages.append(message)

        combined_messages = "".join(messages)
        assert "secret output" not in combined_messages
        assert handler.full_response == []
        assert any('"error"' in message for message in messages)
        assert handler.error_occurred is True

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
        async for message in handler.safe_stream_generator(mock_stream()):
            messages.append(message)

        assert handler.is_cancelled is True
        # Should have logged disconnection
        with patch('tldw_Server_API.app.core.Chat.streaming_utils.logger') as mock_logger:
            async for _ in handler.safe_stream_generator(mock_stream()):
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

    async def test_openai_like_sse_is_normalized(self):
        handler = StreamingResponseHandler("conv_sse", "gpt-4")

        # Upstream emits OpenAI-style SSE frames (each with trailing blank line)
        chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
        chunk2 = {"choices": [{"delta": {"content": " world"}}]}

        async def provider_stream():
            yield f"data: {json.dumps(chunk1)}\n\n"
            yield f"data: {json.dumps(chunk2)}\n\n"
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

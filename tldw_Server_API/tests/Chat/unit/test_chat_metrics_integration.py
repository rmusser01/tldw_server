# test_chat_metrics_integration.py
# Tests for the chat metrics integration

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
pytestmark = pytest.mark.unit

from tldw_Server_API.app.core.Chat.chat_metrics import (
    ChatMetricsCollector,
    get_chat_metrics,
    ChatMetricLabels
)


class TestChatMetricsCollector:
    """Test ChatMetricsCollector functionality."""

    def test_initialization(self):

        """Test metrics collector initialization."""
        collector = ChatMetricsCollector()

        assert collector.telemetry is not None
        assert collector.meter is not None
        assert collector.tracer is not None
        assert collector.metrics is not None
        assert collector.active_requests == 0
        assert collector.active_streams == 0
        assert collector.active_transactions == 0

    @pytest.mark.asyncio
    async def test_track_request_success(self):
        """Test successful request tracking."""
        collector = ChatMetricsCollector()

        async with collector.track_request(
            provider="openai",
            model="gpt-4",
            streaming=False,
            client_id="test_client"
        ) as span:
            assert collector.active_requests == 1
            span.set_attribute("test", "value")

        assert collector.active_requests == 0

    @pytest.mark.asyncio
    async def test_track_request_error(self):
        """Test request tracking with error."""
        collector = ChatMetricsCollector()

        with pytest.raises(ValueError):
            async with collector.track_request(
                provider="openai",
                model="gpt-4",
                streaming=False
            ) as span:
                assert collector.active_requests == 1
                raise ValueError("Test error")

        assert collector.active_requests == 0

    @pytest.mark.asyncio
    async def test_track_request_clamps_after_external_reset(self):
        """Active request count should never go negative after a reset."""
        collector = ChatMetricsCollector()

        async with collector.track_request(
            provider="openai",
            model="gpt-4",
            streaming=False,
            client_id="test_client"
        ):
            assert collector.active_requests == 1
            collector.reset_active_metrics()
            assert collector.active_requests == 0

        assert collector.active_requests == 0

    @pytest.mark.asyncio
    async def test_track_streaming(self):
        """Test streaming response tracking."""
        collector = ChatMetricsCollector()

        async with collector.track_streaming("conv_123") as tracker:
            assert collector.active_streams == 1

            # Track some chunks and heartbeats
            tracker.add_chunk()
            tracker.add_chunk()
            tracker.add_heartbeat()
            tracker.add_chunk()

        assert collector.active_streams == 0

    @pytest.mark.asyncio
    async def test_track_streaming_uses_start_span_without_context_attach(self):
        """Streaming spans should not rely on current-context attach/detach."""
        collector = ChatMetricsCollector()
        mock_span = MagicMock()
        mock_tracer = MagicMock()
        mock_tracer.start_span.return_value = mock_span
        mock_tracer.start_as_current_span = MagicMock()
        collector.tracer = mock_tracer

        async with collector.track_streaming("conv_otel_regression") as tracker:
            tracker.add_chunk()
            tracker.add_heartbeat()

        mock_tracer.start_span.assert_called_once_with(
            "streaming_response",
            attributes={"conversation_id": "conv_otel_regression"},
        )
        mock_tracer.start_as_current_span.assert_not_called()
        mock_span.end.assert_called_once()

    @pytest.mark.asyncio
    async def test_track_database_operation(self):
        """Test database operation tracking."""
        collector = ChatMetricsCollector()

        async with collector.track_database_operation("save_message") as span:
            # Simulate database work
            await asyncio.sleep(0.01)
            span.set_attribute("rows_affected", 1)

    def test_track_validation_failure(self):

        """Test validation failure tracking."""
        collector = ChatMetricsCollector()

        with patch('tldw_Server_API.app.core.Chat.chat_metrics.logger') as mock_logger:
            collector.track_validation_failure(
                validation_type="request_size",
                error_message="Request too large"
            )

            mock_logger.warning.assert_called_once()

    def test_track_auth_failure(self):

        """Test authentication failure tracking."""
        collector = ChatMetricsCollector()

        with patch('tldw_Server_API.app.core.Chat.chat_metrics.logger') as mock_logger:
            collector.track_auth_failure("Invalid API key")

            mock_logger.warning.assert_called_once()

    def test_track_rate_limit(self):

        """Test rate limit tracking."""
        collector = ChatMetricsCollector()

        with patch('tldw_Server_API.app.core.Chat.chat_metrics.logger') as mock_logger:
            collector.track_rate_limit("client_123")

            mock_logger.info.assert_called_once()

    def test_track_tokens(self):

        """Test token usage tracking."""
        collector = ChatMetricsCollector()

        with patch('tldw_Server_API.app.core.Chat.chat_metrics.logger') as mock_logger:
            collector.track_tokens(
                prompt_tokens=100,
                completion_tokens=50,
                model="gpt-4",
                provider="openai"
            )

            # Should log debug with cost estimate
            mock_logger.debug.assert_called()

    def test_track_character_access(self):

        """Test character access tracking."""
        collector = ChatMetricsCollector()

        # Track cache hit
        collector.track_character_access("char_123", cache_hit=True)

        # Track cache miss
        collector.track_character_access("char_456", cache_hit=False)

    def test_track_conversation(self):

        """Test conversation tracking."""
        collector = ChatMetricsCollector()

        # Track new conversation
        collector.track_conversation("conv_new", is_new=True)

        # Track resumed conversation
        collector.track_conversation("conv_existing", is_new=False)

    def test_track_message_saved(self):

        """Test message save tracking."""
        collector = ChatMetricsCollector()

        collector.track_message_saved("conv_123", "user")
        collector.track_message_saved("conv_123", "assistant")

    def test_track_image_processing(self):

        """Test image processing tracking."""
        collector = ChatMetricsCollector()

        collector.track_image_processing(
            size_bytes=1024 * 50,  # 50KB
            validation_time=0.05
        )

    def test_track_transaction(self):

        """Test transaction tracking."""
        collector = ChatMetricsCollector()

        # Successful transaction with retries
        collector.track_transaction(success=True, retries=2)

        # Failed transaction
        collector.track_transaction(success=False, retries=0)

    def test_track_llm_call(self):

        """Test LLM call tracking."""
        collector = ChatMetricsCollector()

        # Successful call
        collector.track_llm_call(
            provider="openai",
            model="gpt-4",
            latency=1.5,
            success=True
        )

        # Failed call
        collector.track_llm_call(
            provider="anthropic",
            model="claude-3",
            latency=0.5,
            success=False,
            error_type="RateLimitError"
        )

    def test_chat_metrics_records_run_first_variant_labels(self):
        collector = ChatMetricsCollector()
        collector.metrics.run_first_rollout = MagicMock()

        collector.track_run_first_rollout(
            presentation_variant="chat_phase2b_v1",
            cohort="default_on",
            provider="openai",
            model="gpt-4o-mini",
            streaming=False,
            eligible=True,
            ineligible_reason=None,
        )

        collector.metrics.run_first_rollout.add.assert_called_once()
        _, labels = collector.metrics.run_first_rollout.add.call_args.args
        assert labels["presentation_variant"] == "chat_phase2b_v1"
        assert labels["cohort"] == "default_on"
        assert labels["provider"] == "openai"
        assert labels["model"] == "set"
        assert labels["streaming"] == "false"
        assert labels["eligible"] == "true"
        assert labels["ineligible_reason"] == "none"

    def test_chat_metrics_records_out_of_cohort_run_first_labels(self):
        collector = ChatMetricsCollector()
        collector.metrics.run_first_rollout = MagicMock()

        collector.track_run_first_rollout(
            presentation_variant="chat_phase2b_v1",
            cohort="out_of_cohort",
            provider="openai",
            model="gpt-4o-mini",
            streaming=True,
            eligible=False,
            ineligible_reason="provider_not_in_rollout_allowlist",
        )

        collector.metrics.run_first_rollout.add.assert_called_once()
        _, labels = collector.metrics.run_first_rollout.add.call_args.args
        assert labels["presentation_variant"] == "chat_phase2b_v1"
        assert labels["cohort"] == "out_of_cohort"
        assert labels["model"] == "set"
        assert labels["streaming"] == "true"
        assert labels["eligible"] == "false"
        assert labels["ineligible_reason"] == "provider_not_in_rollout_allowlist"

    @pytest.mark.parametrize(
        ("raw_reason", "expected"),
        [
            ("no_tools", "no_tools"),
            ("no_effective_tools", "no_effective_tools"),
            ("provider_not_in_rollout_allowlist", "provider_not_in_rollout_allowlist"),
            ("rollout_off", "rollout_off"),
            ("run_missing", "run_missing"),
            ("run_missing_after_filtering", "run_missing_after_filtering"),
            (None, "none"),
            ("unexpected_reason", "other"),
        ],
    )
    def test_chat_metrics_normalizes_ineligible_reason_to_bounded_values(self, raw_reason, expected):
        collector = ChatMetricsCollector()
        collector.metrics.run_first_rollout = MagicMock()

        collector.track_run_first_rollout(
            presentation_variant="chat_phase2b_v1",
            cohort="default_on",
            provider="openai",
            model="gpt-4o-mini",
            streaming=False,
            eligible=False,
            ineligible_reason=raw_reason,
        )

        collector.metrics.run_first_rollout.add.assert_called_once()
        _, labels = collector.metrics.run_first_rollout.add.call_args.args
        assert labels[ChatMetricLabels.INELIGIBLE_REASON.value] == expected

    def test_chat_metrics_run_first_rollout_fails_open_when_counter_add_raises(self):
        collector = ChatMetricsCollector()
        collector.metrics.run_first_rollout = MagicMock()
        collector.metrics.run_first_rollout.add = MagicMock(side_effect=RuntimeError("metrics down"))

        with patch('tldw_Server_API.app.core.Chat.chat_metrics.logger') as mock_logger:
            collector.track_run_first_rollout(
                presentation_variant="chat_phase2b_v1",
                cohort="default_on",
                provider="openai",
                model="gpt-4o-mini",
                streaming=False,
                eligible=True,
                ineligible_reason=None,
            )

        collector.metrics.run_first_rollout.add.assert_called_once()
        mock_logger.debug.assert_called_once()
    def test_chat_metrics_records_run_first_tool_path_and_completion_labels(self):
        collector = ChatMetricsCollector()
        collector.metrics.run_first_first_tool = MagicMock()
        collector.metrics.run_first_fallback_after_run = MagicMock()
        collector.metrics.run_first_completion_proxy = MagicMock()

        collector.track_run_first_first_tool(
            presentation_variant="chat_phase2b_v1",
            cohort="default_on",
            provider="openai",
            model="gpt-4o-mini",
            streaming=True,
            eligible=True,
            first_tool="run",
        )
        collector.track_run_first_fallback_after_run(
            presentation_variant="chat_phase2b_v1",
            cohort="default_on",
            provider="openai",
            model="gpt-4o-mini",
            streaming=True,
            eligible=True,
            fallback_tool="notes.search",
        )
        collector.track_run_first_completion_proxy(
            presentation_variant="chat_phase2b_v1",
            cohort="default_on",
            provider="openai",
            model="gpt-4o-mini",
            streaming=True,
            eligible=True,
            outcome="success",
        )

        collector.metrics.run_first_first_tool.add.assert_called_once()
        _, first_tool_labels = collector.metrics.run_first_first_tool.add.call_args.args
        assert first_tool_labels["first_tool"] == "run"
        assert first_tool_labels["streaming"] == "true"

        collector.metrics.run_first_fallback_after_run.add.assert_called_once()
        _, fallback_labels = collector.metrics.run_first_fallback_after_run.add.call_args.args
        assert fallback_labels["fallback_tool"] == "other"

        collector.metrics.run_first_completion_proxy.add.assert_called_once()
        _, completion_labels = collector.metrics.run_first_completion_proxy.add.call_args.args
        assert completion_labels["outcome"] == "success"

    def test_get_active_metrics(self):

        """Test getting active operation counts."""
        collector = ChatMetricsCollector()

        collector.active_requests = 3
        collector.active_streams = 2
        collector.active_transactions = 1

        metrics = collector.get_active_metrics()

        assert metrics == {
            "active_requests": 3,
            "active_streams": 2,
            "active_transactions": 1
        }

    def test_cost_estimation(self):

        """Test token cost estimation."""
        collector = ChatMetricsCollector()

        # Test with known model
        with patch('tldw_Server_API.app.core.Chat.chat_metrics.logger') as mock_logger:
            collector.track_tokens(
                prompt_tokens=1000,
                completion_tokens=500,
                model="gpt-4",
                provider="openai"
            )

            # Check that cost was calculated and logged
            call_args = mock_logger.debug.call_args[0][0]
            assert "Estimated cost: $" in call_args
            assert "0.06" in call_args  # Expected cost for gpt-4

        # Test with unknown model (no cost estimation)
        collector.track_tokens(
            prompt_tokens=1000,
            completion_tokens=500,
            model="unknown-model",
            provider="unknown"
        )


class TestGetChatMetrics:
    """Test the global metrics instance."""

    def test_singleton_pattern(self):

        """Test that get_chat_metrics returns singleton."""
        metrics1 = get_chat_metrics()
        metrics2 = get_chat_metrics()

        assert metrics1 is metrics2

    def test_instance_type(self):

        """Test that instance is correct type."""
        metrics = get_chat_metrics()
        assert isinstance(metrics, ChatMetricsCollector)


class TestStreamTracker:
    """Test the StreamTracker helper class."""

    @pytest.mark.asyncio
    async def test_stream_tracker_functionality(self):
        """Test StreamTracker methods."""
        collector = ChatMetricsCollector()

        async with collector.track_streaming("conv_123") as tracker:
            # Test chunk tracking
            tracker.add_chunk()
            tracker.add_chunk()

            # Test heartbeat tracking
            tracker.add_heartbeat()

            # Test timeout tracking
            tracker.timeout()


class TestMetricsIntegration:
    """Test metrics integration with chat endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_in_request_context(self):
        """Test metrics tracking in request context."""
        collector = get_chat_metrics()

        # Simulate a complete request flow
        async with collector.track_request(
            provider="openai",
            model="gpt-4",
            streaming=False,
            client_id="test"
        ):
            # Track validation
            validation_start = time.time()
            await asyncio.sleep(0.01)  # Simulate validation
            validation_time = time.time() - validation_start

            # Track character loading
            collector.track_character_access("char_123", cache_hit=False)

            # Track conversation
            collector.track_conversation("conv_123", is_new=True)

            # Track database operations
            async with collector.track_database_operation("save_message"):
                await asyncio.sleep(0.01)  # Simulate DB work
                collector.track_message_saved("conv_123", "user")

            # Track LLM call
            llm_start = time.time()
            await asyncio.sleep(0.05)  # Simulate LLM call
            llm_latency = time.time() - llm_start
            collector.track_llm_call("openai", "gpt-4", llm_latency, success=True)

            # Track tokens
            collector.track_tokens(
                prompt_tokens=500,
                completion_tokens=200,
                model="gpt-4",
                provider="openai"
            )

            # Track assistant message save
            async with collector.track_database_operation("save_message"):
                await asyncio.sleep(0.01)
                collector.track_message_saved("conv_123", "assistant")

    @pytest.mark.asyncio
    async def test_streaming_metrics_integration(self):
        """Test metrics in streaming context."""
        collector = get_chat_metrics()

        async with collector.track_request(
            provider="anthropic",
            model="claude-3",
            streaming=True,
            client_id="test"
        ):
            # Track streaming
            async with collector.track_streaming("conv_123") as tracker:
                # Simulate streaming chunks
                for i in range(10):
                    tracker.add_chunk()
                    await asyncio.sleep(0.001)

                    # Add heartbeat every 3 chunks
                    if i % 3 == 0:
                        tracker.add_heartbeat()

            # Track message save after streaming
            async with collector.track_database_operation("save_message"):
                collector.track_message_saved("conv_123", "assistant")

    @pytest.mark.asyncio
    async def test_error_metrics_integration(self):
        """Test error tracking in metrics."""
        collector = get_chat_metrics()

        # Test validation failure
        collector.track_validation_failure(
            validation_type="content_size",
            error_message="Content exceeds maximum size"
        )

        # Test auth failure
        collector.track_auth_failure("Invalid bearer token")

        # Test rate limit
        collector.track_rate_limit("aggressive_client")

        # Test failed LLM call
        collector.track_llm_call(
            provider="openai",
            model="gpt-4",
            latency=0.1,
            success=False,
            error_type="NetworkError"
        )

        # Test failed transaction
        collector.track_transaction(success=False, retries=3)

    @pytest.mark.asyncio
    async def test_image_processing_metrics(self):
        """Test image processing metrics."""
        collector = get_chat_metrics()

        # Track multiple images
        images = [
            (1024 * 100, 0.05),  # 100KB, 50ms
            (1024 * 200, 0.08),  # 200KB, 80ms
            (1024 * 50, 0.03),   # 50KB, 30ms
        ]

        for size, time_taken in images:
            collector.track_image_processing(size, time_taken)

    @pytest.mark.asyncio
    async def test_transaction_retry_metrics(self):
        """Test transaction retry metrics."""
        collector = get_chat_metrics()

        # Simulate transaction with retries
        async with collector.track_database_operation("transaction"):
            # First attempt fails
            collector.track_transaction(success=False, retries=0)

            # Retry succeeds
            await asyncio.sleep(0.01)
            collector.track_transaction(success=True, retries=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# chat_metrics.py
# Chat-specific metrics integration for telemetry

import statistics
import time
import threading
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from loguru import logger

from ..Metrics.telemetry import get_telemetry_manager

RUN_FIRST_INELIGIBLE_REASONS = frozenset(
    {
        "no_effective_tools",
        "no_tools",
        "provider_not_in_rollout_allowlist",
        "rollout_off",
        "run_missing",
        "run_missing_after_filtering",
    }
)

_ENDPOINT_METRIC_SAMPLE_SIZE = 256
_endpoint_metric_lock = threading.Lock()
_endpoint_metric_samples = defaultdict(lambda: deque(maxlen=_ENDPOINT_METRIC_SAMPLE_SIZE))
_endpoint_metric_timestamps: dict[str, float] = {}


def _record_endpoint_metric(metric_name: str, value: float) -> None:
    """Store sampled metric values for /api/v1/metrics/chat summary payloads."""
    with _endpoint_metric_lock:
        _endpoint_metric_samples[metric_name].append(float(value))
        _endpoint_metric_timestamps[metric_name] = time.time()


def get_endpoint_metrics_snapshot() -> dict[str, dict[str, float]]:
    """Return summary stats for sampled chat endpoint metrics."""
    with _endpoint_metric_lock:
        copied_samples = {
            metric_name: list(samples)
            for metric_name, samples in _endpoint_metric_samples.items()
        }
        copied_timestamps = dict(_endpoint_metric_timestamps)

    snapshot: dict[str, dict[str, float]] = {}
    for metric_name, numeric_values in copied_samples.items():
        if not numeric_values:
            continue
        snapshot[metric_name] = {
            "count": len(numeric_values),
            "sum": float(sum(numeric_values)),
            "mean": float(statistics.mean(numeric_values)),
            "median": float(statistics.median(numeric_values)),
            "min": float(min(numeric_values)),
            "max": float(max(numeric_values)),
            "stddev": float(statistics.stdev(numeric_values)) if len(numeric_values) > 1 else 0.0,
            "latest": float(numeric_values[-1]),
            "latest_timestamp": float(copied_timestamps.get(metric_name, 0.0)),
        }
    return snapshot


def reset_endpoint_metrics_snapshot() -> None:
    """Clear in-process sampled metrics exposed by /api/v1/metrics/chat."""
    with _endpoint_metric_lock:
        _endpoint_metric_samples.clear()
        _endpoint_metric_timestamps.clear()


class ChatMetricLabels(Enum):
    """Standard labels for chat metrics."""
    PROVIDER = "provider"
    MODEL = "model"
    CHARACTER = "character"
    STREAMING = "streaming"
    ERROR_TYPE = "error_type"
    STATUS = "status"
    TRANSACTION = "transaction"
    CLIENT_ID = "client_id"
    CONVERSATION_ID = "conversation_id"
    MESSAGE_TYPE = "message_type"
    VALIDATION_TYPE = "validation_type"
    RETRY_COUNT = "retry_count"
    USER = "user_id"
    CATEGORY = "category"
    ACTION = "action"
    PRESENTATION_VARIANT = "presentation_variant"
    COHORT = "cohort"
    ELIGIBLE = "eligible"
    INELIGIBLE_REASON = "ineligible_reason"
    FIRST_TOOL = "first_tool"
    FALLBACK_TOOL = "fallback_tool"
    OUTCOME = "outcome"


@dataclass
class ChatMetrics:
    """Container for all chat-related metrics."""

    # Request metrics
    requests_total: Any
    request_duration: Any
    request_size_bytes: Any
    response_size_bytes: Any

    # Streaming metrics
    streaming_duration: Any
    streaming_chunks: Any
    streaming_heartbeats: Any
    streaming_timeouts: Any

    # Token metrics
    tokens_prompt: Any
    tokens_completion: Any
    tokens_total: Any

    # Character metrics
    character_loads: Any
    character_cache_hits: Any
    character_cache_misses: Any

    # Conversation metrics
    conversations_created: Any
    conversations_resumed: Any
    messages_saved: Any

    # Validation metrics
    validation_failures: Any
    validation_duration: Any

    # Image metrics
    images_processed: Any
    image_validation_duration: Any
    image_size_bytes: Any

    # Database metrics
    db_transactions: Any
    db_retries: Any
    db_rollbacks: Any
    db_operation_duration: Any

    # Error metrics
    errors_total: Any
    rate_limits: Any
    auth_failures: Any

    # LLM metrics
    llm_requests: Any
    llm_errors: Any
    llm_latency: Any
    llm_cost_estimate: Any

    # Moderation metrics
    moderation_input_flags: Any
    moderation_output_redacts: Any
    moderation_output_blocks: Any
    moderation_stream_blocks: Any

    # Fallback metrics
    provider_fallback_successes: Any

    # Run-first rollout metrics
    run_first_rollout: Any
    run_first_first_tool: Any
    run_first_fallback_after_run: Any
    run_first_completion_proxy: Any


class ChatMetricsCollector:
    """Collects and manages chat-specific metrics."""

    def __init__(self):
        """Initialize the chat metrics collector."""
        self.telemetry = get_telemetry_manager()
        self.meter = self.telemetry.get_meter("tldw_server.chat")
        self.tracer = self.telemetry.get_tracer("tldw_server.chat")
        # Metrics are recorded via OTel instruments; no separate JSON registry needed

        # Initialize metrics
        self.metrics = self._initialize_metrics()

        # Track active operations
        self.active_requests = 0
        self.active_streams = 0
        self.active_transactions = 0
        self._active_lock = threading.Lock()

        # Cost tracking
        self.token_costs = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},  # per 1K tokens
            "gpt-3.5-turbo": {"prompt": 0.001, "completion": 0.002},
            "claude-3": {"prompt": 0.015, "completion": 0.075},
            # Add more models as needed
        }

    def _initialize_metrics(self) -> ChatMetrics:
        """Initialize all chat metrics."""
        return ChatMetrics(
            # Request metrics
            requests_total=self.meter.create_counter(
                name="chat_requests_total",
                description="Total number of chat requests",
                unit="1"
            ),
            request_duration=self.meter.create_histogram(
                name="chat_request_duration_seconds",
                description="Chat request duration",
                unit="s"
            ),
            request_size_bytes=self.meter.create_histogram(
                name="chat_request_size_bytes",
                description="Size of chat requests",
                unit="By"
            ),
            response_size_bytes=self.meter.create_histogram(
                name="chat_response_size_bytes",
                description="Size of chat responses",
                unit="By"
            ),

            # Streaming metrics
            streaming_duration=self.meter.create_histogram(
                name="chat_streaming_duration_seconds",
                description="Duration of streaming responses",
                unit="s"
            ),
            streaming_chunks=self.meter.create_histogram(
                name="chat_streaming_chunks_total",
                description="Number of chunks in streaming response",
                unit="1"
            ),
            streaming_heartbeats=self.meter.create_counter(
                name="chat_streaming_heartbeats_total",
                description="Total heartbeat messages sent",
                unit="1"
            ),
            streaming_timeouts=self.meter.create_counter(
                name="chat_streaming_timeouts_total",
                description="Number of streaming timeouts",
                unit="1"
            ),

            # Token metrics
            tokens_prompt=self.meter.create_histogram(
                name="chat_tokens_prompt",
                description="Number of prompt tokens",
                unit="1"
            ),
            tokens_completion=self.meter.create_histogram(
                name="chat_tokens_completion",
                description="Number of completion tokens",
                unit="1"
            ),
            tokens_total=self.meter.create_histogram(
                name="chat_tokens_total",
                description="Total tokens used",
                unit="1"
            ),

            # Character metrics
            character_loads=self.meter.create_counter(
                name="chat_character_loads_total",
                description="Number of character loads",
                unit="1"
            ),
            character_cache_hits=self.meter.create_counter(
                name="chat_character_cache_hits_total",
                description="Character cache hits",
                unit="1"
            ),
            character_cache_misses=self.meter.create_counter(
                name="chat_character_cache_misses_total",
                description="Character cache misses",
                unit="1"
            ),

            # Conversation metrics
            conversations_created=self.meter.create_counter(
                name="chat_conversations_created_total",
                description="Number of conversations created",
                unit="1"
            ),
            conversations_resumed=self.meter.create_counter(
                name="chat_conversations_resumed_total",
                description="Number of conversations resumed",
                unit="1"
            ),
            messages_saved=self.meter.create_counter(
                name="chat_messages_saved_total",
                description="Number of messages saved to database",
                unit="1"
            ),

            # Validation metrics
            validation_failures=self.meter.create_counter(
                name="chat_validation_failures_total",
                description="Number of validation failures",
                unit="1"
            ),
            validation_duration=self.meter.create_histogram(
                name="chat_validation_duration_seconds",
                description="Time spent validating requests",
                unit="s"
            ),

            # Image metrics
            images_processed=self.meter.create_counter(
                name="chat_images_processed_total",
                description="Number of images processed",
                unit="1"
            ),
            image_validation_duration=self.meter.create_histogram(
                name="chat_image_validation_duration_seconds",
                description="Time spent validating images",
                unit="s"
            ),
            image_size_bytes=self.meter.create_histogram(
                name="chat_image_size_bytes",
                description="Size of processed images",
                unit="By"
            ),

            # Database metrics
            db_transactions=self.meter.create_counter(
                name="chat_db_transactions_total",
                description="Number of database transactions",
                unit="1"
            ),
            db_retries=self.meter.create_counter(
                name="chat_db_retries_total",
                description="Number of transaction retries",
                unit="1"
            ),
            db_rollbacks=self.meter.create_counter(
                name="chat_db_rollbacks_total",
                description="Number of transaction rollbacks",
                unit="1"
            ),
            db_operation_duration=self.meter.create_histogram(
                name="chat_db_operation_duration_seconds",
                description="Database operation duration",
                unit="s"
            ),

            # Error metrics
            errors_total=self.meter.create_counter(
                name="chat_errors_total",
                description="Total number of errors",
                unit="1"
            ),
            rate_limits=self.meter.create_counter(
                name="chat_rate_limits_total",
                description="Number of rate limit hits",
                unit="1"
            ),
            auth_failures=self.meter.create_counter(
                name="chat_auth_failures_total",
                description="Number of authentication failures",
                unit="1"
            ),

            # LLM metrics
            llm_requests=self.meter.create_counter(
                name="chat_llm_requests_total",
                description="Number of LLM API calls",
                unit="1"
            ),
            llm_errors=self.meter.create_counter(
                name="chat_llm_errors_total",
                description="Number of LLM API errors",
                unit="1"
            ),
            llm_latency=self.meter.create_histogram(
                name="chat_llm_latency_seconds",
                description="LLM API call latency",
                unit="s"
            ),
            llm_cost_estimate=self.meter.create_histogram(
                name="chat_llm_cost_estimate_usd",
                description="Estimated cost of LLM API calls",
                unit="USD"
            ),

            # Moderation metrics
            moderation_input_flags=self.meter.create_counter(
                name="chat_moderation_input_flag_total",
                description="Number of input messages flagged by moderation",
                unit="1",
            ),
            moderation_output_redacts=self.meter.create_counter(
                name="chat_moderation_output_redact_total",
                description="Number of outputs redacted by moderation",
                unit="1",
            ),
            moderation_output_blocks=self.meter.create_counter(
                name="chat_moderation_output_block_total",
                description="Number of outputs blocked by moderation",
                unit="1",
            ),
            moderation_stream_blocks=self.meter.create_counter(
                name="chat_moderation_stream_block_total",
                description="Number of streaming responses blocked by moderation",
                unit="1",
            )
            ,
            # Fallback metrics
            provider_fallback_successes=self.meter.create_counter(
                name="chat_provider_fallback_success_total",
                description="Count of successful provider fallback transitions",
                unit="1",
            ),
            # Run-first rollout metrics
            run_first_rollout=self.meter.create_counter(
                name="chat_run_first_rollout_total",
                description="Count of run-first rollout exposures",
                unit="1",
            ),
            run_first_first_tool=self.meter.create_counter(
                name="chat_run_first_first_tool_total",
                description="Count of first selected tools under run-first rollout",
                unit="1",
            ),
            run_first_fallback_after_run=self.meter.create_counter(
                name="chat_run_first_fallback_after_run_total",
                description="Count of typed-tool fallbacks after run under rollout",
                unit="1",
            ),
            run_first_completion_proxy=self.meter.create_counter(
                name="chat_run_first_completion_proxy_total",
                description="Count of completion proxy outcomes under rollout",
                unit="1",
            )
        )

    @asynccontextmanager
    async def track_request(
        self,
        provider: str,
        model: str,
        streaming: bool,
        client_id: str = "unknown"
    ):
        """
        Context manager to track a complete chat request.

        Args:
            provider: LLM provider name
            model: Model name
            streaming: Whether this is a streaming request
            client_id: Client identifier
        """
        start_time = time.time()
        labels = {
            ChatMetricLabels.PROVIDER.value: provider,
            ChatMetricLabels.MODEL.value: model,
            ChatMetricLabels.STREAMING.value: str(streaming),
            ChatMetricLabels.CLIENT_ID.value: client_id
        }

        # Create span for tracing
        with self.tracer.start_as_current_span(
            "chat_request",
            attributes=labels
        ) as span:
            with self._active_lock:
                self.active_requests += 1

            try:
                # Increment request counter
                self.metrics.requests_total.add(1, labels)
                _record_endpoint_metric("chat_requests_total", 1.0)

                yield span

                # Mark success
                labels[ChatMetricLabels.STATUS.value] = "success"
                span.set_attribute("status", "success")

            except Exception as e:
                # Track error
                labels[ChatMetricLabels.STATUS.value] = "error"
                labels[ChatMetricLabels.ERROR_TYPE.value] = type(e).__name__

                self.metrics.errors_total.add(1, labels)
                _record_endpoint_metric("chat_errors_total", 1.0)
                span.record_exception(e)
                span.set_attribute("status", "error")
                raise

            finally:
                # Record duration
                duration = time.time() - start_time
                self.metrics.request_duration.record(duration, labels)
                _record_endpoint_metric("chat_request_duration_seconds", duration)

                with self._active_lock:
                    self.active_requests = max(0, self.active_requests - 1)
                span.set_attribute("duration", duration)

    @asynccontextmanager
    async def track_streaming(self, conversation_id: str):
        """
        Track a streaming response.

        Args:
            conversation_id: Conversation identifier
        """
        start_time = time.time()
        chunk_count = 0
        heartbeat_count = 0
        span = None
        try:
            # Use a detached span here to avoid contextvars detach failures when
            # async streaming generators close under cancellation/GeneratorExit.
            span = self.tracer.start_span(
                "streaming_response",
                attributes={"conversation_id": conversation_id}
            )
        except Exception as e:
            logger.debug(f"Unable to start streaming span for {conversation_id}: {e}")

        with self._active_lock:
            self.active_streams += 1

        try:
            # Provide tracking methods
            class StreamTracker:
                def add_chunk(self):
                    nonlocal chunk_count
                    chunk_count += 1

                def add_heartbeat(self):
                    nonlocal heartbeat_count
                    heartbeat_count += 1
                    collector.metrics.streaming_heartbeats.add(
                        1,
                        {ChatMetricLabels.CONVERSATION_ID.value: conversation_id}
                    )

                def timeout(self):
                    collector.metrics.streaming_timeouts.add(
                        1,
                        {ChatMetricLabels.CONVERSATION_ID.value: conversation_id}
                    )

            collector = self
            yield StreamTracker()

        finally:
            duration = time.time() - start_time

            # Record metrics
            self.metrics.streaming_duration.record(
                duration,
                {ChatMetricLabels.CONVERSATION_ID.value: conversation_id}
            )
            _record_endpoint_metric("chat_streaming_duration_seconds", duration)
            self.metrics.streaming_chunks.record(
                chunk_count,
                {ChatMetricLabels.CONVERSATION_ID.value: conversation_id}
            )

            with self._active_lock:
                self.active_streams = max(0, self.active_streams - 1)

            if span is not None:
                try:
                    span.set_attributes({
                        "duration": duration,
                        "chunks": chunk_count,
                        "heartbeats": heartbeat_count
                    })
                except Exception as e:
                    logger.debug(f"Unable to set streaming span attributes for {conversation_id}: {e}")
                try:
                    span.end()
                except Exception as e:
                    logger.debug(f"Unable to end streaming span for {conversation_id}: {e}")

    @asynccontextmanager
    async def track_database_operation(self, operation_type: str):
        """
        Track a database operation.

        Args:
            operation_type: Type of operation (e.g., "save_message", "create_conversation")
        """
        start_time = time.time()
        # Mark a transaction active while this context is open
        with self._active_lock:
            self.active_transactions += 1

        with self.tracer.start_as_current_span(
            f"db_{operation_type}",
            attributes={"operation": operation_type}
        ) as span:
            try:
                yield span
            finally:
                duration = time.time() - start_time
                self.metrics.db_operation_duration.record(
                    duration,
                    {"operation": operation_type}
                )
                span.set_attribute("duration", duration)
                # Decrement active transaction count
                with self._active_lock:
                    self.active_transactions = max(0, self.active_transactions - 1)

    def track_validation_failure(self, validation_type: str, error_message: str):
        """
        Track a validation failure.

        Args:
            validation_type: Type of validation that failed
            error_message: Error message
        """
        self.metrics.validation_failures.add(
            1,
            {
                ChatMetricLabels.VALIDATION_TYPE.value: validation_type,
                "error": error_message[:100]  # Truncate long messages
            }
        )
        logger.warning(f"Validation failure: {validation_type} - {error_message}")

    def track_auth_failure(self, reason: str):
        """
        Track an authentication failure.

        Args:
            reason: Reason for failure
        """
        self.metrics.auth_failures.add(1, {"reason": reason})
        logger.warning(f"Authentication failure: {reason}")

    def track_rate_limit(self, client_id: str):
        """
        Track a rate limit hit.

        Args:
            client_id: Client that hit the rate limit
        """
        self.metrics.rate_limits.add(
            1,
            {ChatMetricLabels.CLIENT_ID.value: client_id}
        )
        logger.info(f"Rate limit hit for client: {client_id}")

    def track_tokens(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        provider: str
    ):
        """
        Track token usage and estimate costs.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
            provider: Provider name
        """
        labels = {
            ChatMetricLabels.MODEL.value: model,
            ChatMetricLabels.PROVIDER.value: provider
        }

        # Record token counts
        self.metrics.tokens_prompt.record(prompt_tokens, labels)
        self.metrics.tokens_completion.record(completion_tokens, labels)
        self.metrics.tokens_total.record(prompt_tokens + completion_tokens, labels)
        _record_endpoint_metric("chat_tokens_prompt", float(prompt_tokens))
        _record_endpoint_metric("chat_tokens_completion", float(completion_tokens))

        # Estimate cost
        if model in self.token_costs:
            costs = self.token_costs[model]
            prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
            completion_cost = (completion_tokens / 1000) * costs["completion"]
            total_cost = prompt_cost + completion_cost

            self.metrics.llm_cost_estimate.record(total_cost, labels)
            _record_endpoint_metric("chat_llm_cost_estimate_usd", total_cost)

            logger.debug(
                f"Token usage: {prompt_tokens} prompt, {completion_tokens} completion. "
                f"Estimated cost: ${total_cost:.4f}"
            )

    def track_character_access(self, character_id: str, cache_hit: bool):
        """
        Track character access.

        Args:
            character_id: Character identifier
            cache_hit: Whether this was a cache hit
        """
        labels = {ChatMetricLabels.CHARACTER.value: character_id}

        self.metrics.character_loads.add(1, labels)

        if cache_hit:
            self.metrics.character_cache_hits.add(1, labels)
        else:
            self.metrics.character_cache_misses.add(1, labels)

    def track_conversation(self, conversation_id: str, is_new: bool):
        """
        Track conversation creation/resumption.

        Args:
            conversation_id: Conversation identifier
            is_new: Whether this is a new conversation
        """
        labels = {ChatMetricLabels.CONVERSATION_ID.value: conversation_id}

        if is_new:
            self.metrics.conversations_created.add(1, labels)
            _record_endpoint_metric("chat_conversations_created_total", 1.0)
        else:
            self.metrics.conversations_resumed.add(1, labels)

    def track_message_saved(self, conversation_id: str, message_type: str):
        """
        Track a message being saved.

        Args:
            conversation_id: Conversation identifier
            message_type: Type of message (user/assistant)
        """
        self.metrics.messages_saved.add(
            1,
            {
                ChatMetricLabels.CONVERSATION_ID.value: conversation_id,
                ChatMetricLabels.MESSAGE_TYPE.value: message_type
            }
        )
        _record_endpoint_metric("chat_messages_saved_total", 1.0)

    def track_image_processing(self, size_bytes: int, validation_time: float):
        """
        Track image processing.

        Args:
            size_bytes: Size of the image in bytes
            validation_time: Time taken to validate the image
        """
        self.metrics.images_processed.add(1)
        self.metrics.image_size_bytes.record(size_bytes)
        self.metrics.image_validation_duration.record(validation_time)

    def track_transaction(self, success: bool, retries: int = 0):
        """
        Track a database transaction.

        Args:
            success: Whether the transaction succeeded
            retries: Number of retries needed
        """
        self.metrics.db_transactions.add(
            1,
            {ChatMetricLabels.STATUS.value: "success" if success else "failed"}
        )

        if retries > 0:
            self.metrics.db_retries.add(
                retries,
                {ChatMetricLabels.RETRY_COUNT.value: str(retries)}
            )

        if not success:
            self.metrics.db_rollbacks.add(1)

    def track_llm_call(
        self,
        provider: str,
        model: str,
        latency: float,
        success: bool,
        error_type: Optional[str] = None
    ):
        """
        Track an LLM API call.

        Args:
            provider: Provider name
            model: Model name
            latency: Call latency in seconds
            success: Whether the call succeeded
            error_type: Type of error if failed
        """
        labels = {
            ChatMetricLabels.PROVIDER.value: provider,
            ChatMetricLabels.MODEL.value: model,
            ChatMetricLabels.STATUS.value: "success" if success else "error"
        }

        self.metrics.llm_requests.add(1, labels)
        self.metrics.llm_latency.record(latency, labels)

        if not success and error_type:
            error_labels = {**labels, ChatMetricLabels.ERROR_TYPE.value: error_type}
            self.metrics.llm_errors.add(1, error_labels)

    def get_active_metrics(self) -> dict[str, int]:
        """Get counts of active operations."""
        with self._active_lock:
            return {
                "active_requests": self.active_requests,
                "active_streams": self.active_streams,
                "active_transactions": self.active_transactions,
            }

    def reset_active_metrics(self) -> None:
        """Reset active operation counters to zero."""
        with self._active_lock:
            self.active_requests = 0
            self.active_streams = 0
            self.active_transactions = 0

    def track_provider_fallback_success(
        self,
        requested_provider: str,
        selected_provider: str,
        *,
        streaming: bool,
        queued: bool,
    ) -> None:
        """Record a successful fallback from requested_provider to selected_provider.

        Args:
            requested_provider: The provider originally requested by the client
            selected_provider: The provider actually used (fallback)
            streaming: Whether this was a streaming call
            queued: Whether this call went through the request queue
        """
        try:
            labels = {
                ChatMetricLabels.PROVIDER.value: requested_provider,
                "selected_provider": selected_provider,
                ChatMetricLabels.STREAMING.value: str(bool(streaming)).lower(),
                "queued": str(bool(queued)).lower(),
            }
            self.metrics.provider_fallback_successes.add(1, labels)
            logger.info(
                f"Fallback success: requested={requested_provider}, selected={selected_provider}, "
                f"streaming={streaming}, queued={queued}"
            )
        except Exception as metrics_error:
            # Metrics must never break the flow
            logger.debug("Fallback metrics emission failed", exc_info=metrics_error)

    # --- Cardinality bounding for run-first metric labels ---
    _KNOWN_PROVIDERS: frozenset[str] = frozenset({
        "openai", "anthropic", "google", "groq", "mistral",
        "ollama", "kobold", "tabby", "vllm", "aphrodite",
        "deepseek", "cohere", "huggingface", "openrouter",
        "local", "custom_openai", "unknown",
    })
    _KNOWN_INELIGIBLE_REASONS: frozenset[str] = frozenset({"none", *RUN_FIRST_INELIGIBLE_REASONS})
    _KNOWN_TOOLS: frozenset[str] = frozenset({
        "run", "unknown",
    })
    _KNOWN_OUTCOMES: frozenset[str] = frozenset({
        "success", "error", "timeout", "blocked",
        "end_turn", "max_iterations", "cancelled",
    })

    @staticmethod
    def _bound_label(value: str, known: frozenset[str]) -> str:
        candidate = str(value or "").strip().lower() or "unknown"
        return candidate if candidate in known else "other"
    def _run_first_base_labels(
        self,
        *,
        presentation_variant: str,
        cohort: str,
        provider: str,
        model: str,
        streaming: bool,
        eligible: bool,
        ineligible_reason: str | None,
    ) -> dict[str, str]:
        provider_prefix = str(provider or "").strip().lower().split("/")[0] or "unknown"
        return {
            ChatMetricLabels.PRESENTATION_VARIANT.value: str(presentation_variant or "").strip() or "unknown",
            ChatMetricLabels.COHORT.value: str(cohort or "").strip() or "unknown",
            ChatMetricLabels.PROVIDER.value: self._bound_label(provider_prefix, self._KNOWN_PROVIDERS),
            ChatMetricLabels.MODEL.value: "set" if str(model or "").strip() else "unknown",
            ChatMetricLabels.STREAMING.value: str(bool(streaming)).lower(),
            ChatMetricLabels.ELIGIBLE.value: str(bool(eligible)).lower(),
            ChatMetricLabels.INELIGIBLE_REASON.value: self._bound_label(
                str(ineligible_reason or "").strip() or "none",
                self._KNOWN_INELIGIBLE_REASONS,
            ),
        }

    def _safe_counter_add(
        self,
        counter: Any,
        value: int,
        labels: dict[str, str],
        metric_name: str,
    ) -> None:
        """Emit a counter update without allowing telemetry failures to escape."""
        try:
            counter.add(value, labels)
        except Exception as metrics_error:
            logger.debug("{} emission failed", metric_name, exc_info=metrics_error)
    def track_run_first_rollout(
        self,
        *,
        presentation_variant: str,
        cohort: str,
        provider: str,
        model: str,
        streaming: bool,
        eligible: bool,
        ineligible_reason: str | None,
    ) -> None:
        labels = self._run_first_base_labels(
            presentation_variant=presentation_variant,
            cohort=cohort,
            provider=provider,
            model=model,
            streaming=streaming,
            eligible=eligible,
            ineligible_reason=ineligible_reason,
        )
        self._safe_counter_add(
            self.metrics.run_first_rollout,
            1,
            labels,
            "chat_run_first_rollout_total",
        )

    def track_run_first_first_tool(
        self,
        *,
        presentation_variant: str,
        cohort: str,
        provider: str,
        model: str,
        streaming: bool,
        eligible: bool,
        ineligible_reason: str | None = None,
        first_tool: str,
    ) -> None:
        labels = self._run_first_base_labels(
            presentation_variant=presentation_variant,
            cohort=cohort,
            provider=provider,
            model=model,
            streaming=streaming,
            eligible=eligible,
            ineligible_reason=ineligible_reason,
        )
        labels[ChatMetricLabels.FIRST_TOOL.value] = self._bound_label(first_tool, self._KNOWN_TOOLS)
        self._safe_counter_add(
            self.metrics.run_first_first_tool,
            1,
            labels,
            "chat_run_first_first_tool_total",
        )

    def track_run_first_fallback_after_run(
        self,
        *,
        presentation_variant: str,
        cohort: str,
        provider: str,
        model: str,
        streaming: bool,
        eligible: bool,
        ineligible_reason: str | None = None,
        fallback_tool: str,
    ) -> None:
        labels = self._run_first_base_labels(
            presentation_variant=presentation_variant,
            cohort=cohort,
            provider=provider,
            model=model,
            streaming=streaming,
            eligible=eligible,
            ineligible_reason=ineligible_reason,
        )
        labels[ChatMetricLabels.FALLBACK_TOOL.value] = self._bound_label(fallback_tool, self._KNOWN_TOOLS)
        self._safe_counter_add(
            self.metrics.run_first_fallback_after_run,
            1,
            labels,
            "chat_run_first_fallback_after_run_total",
        )

    def track_run_first_completion_proxy(
        self,
        *,
        presentation_variant: str,
        cohort: str,
        provider: str,
        model: str,
        streaming: bool,
        eligible: bool,
        ineligible_reason: str | None = None,
        outcome: str,
    ) -> None:
        labels = self._run_first_base_labels(
            presentation_variant=presentation_variant,
            cohort=cohort,
            provider=provider,
            model=model,
            streaming=streaming,
            eligible=eligible,
            ineligible_reason=ineligible_reason,
        )
        labels[ChatMetricLabels.OUTCOME.value] = self._bound_label(outcome, self._KNOWN_OUTCOMES)
        self._safe_counter_add(
            self.metrics.run_first_completion_proxy,
            1,
            labels,
            "chat_run_first_completion_proxy_total",
        )

    # ---------------- Moderation helpers ----------------
    def track_moderation_input(self, user_id: str, action: str, category: str = "default"):
        labels = {
            ChatMetricLabels.USER.value: str(user_id),
            ChatMetricLabels.ACTION.value: str(action),
            ChatMetricLabels.CATEGORY.value: str(category),
        }
        self.metrics.moderation_input_flags.add(1, labels)

    def track_moderation_output(self, user_id: str, action: str, streaming: bool = False, category: str = "default"):
        labels = {
            ChatMetricLabels.USER.value: str(user_id),
            ChatMetricLabels.ACTION.value: str(action),
            ChatMetricLabels.CATEGORY.value: str(category),
            ChatMetricLabels.STREAMING.value: str(bool(streaming)).lower(),
        }
        if action == "block":
            self.metrics.moderation_output_blocks.add(1, labels)
        elif action == "redact":
            self.metrics.moderation_output_redacts.add(1, labels)

    def track_moderation_stream_block(self, user_id: str, category: str = "default"):
        labels = {
            ChatMetricLabels.USER.value: str(user_id),
            ChatMetricLabels.CATEGORY.value: str(category),
        }
        self.metrics.moderation_stream_blocks.add(1, labels)


# Global instance
_chat_metrics_collector: Optional[ChatMetricsCollector] = None
_chat_metrics_collector_lock = threading.Lock()


def get_chat_metrics() -> ChatMetricsCollector:
    """Get the global chat metrics collector instance."""
    global _chat_metrics_collector
    if _chat_metrics_collector is None:
        with _chat_metrics_collector_lock:
            if _chat_metrics_collector is None:
                _chat_metrics_collector = ChatMetricsCollector()
    return _chat_metrics_collector

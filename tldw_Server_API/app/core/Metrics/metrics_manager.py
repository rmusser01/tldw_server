"""
Centralized metrics management for the tldw_server application.

This module provides a unified interface for all metric operations,
supporting both OpenTelemetry and fallback implementations.
"""

import os
import re
import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from loguru import logger

from .telemetry import OTEL_AVAILABLE, get_telemetry_manager
from .stt_metrics import iter_stt_metric_definitions

if OTEL_AVAILABLE:
    from opentelemetry.metrics import CallbackOptions, Observation


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    UP_DOWN_COUNTER = "up_down_counter"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    labels: list[str] = field(default_factory=list)
    buckets: Optional[list[float]] = None  # For histograms


@dataclass
class MetricValue:
    """A metric value with metadata."""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


class MetricsRegistry:
    """Registry for all application metrics."""

    _PROM_METRIC_NAME_RE = re.compile(r"[^a-zA-Z0-9_:]")
    _PROM_LABEL_KEY_RE = re.compile(r"[^a-zA-Z0-9_]")

    def __init__(self):
        """Initialize the metrics registry."""
        raw_maxlen = os.getenv("METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED")
        if raw_maxlen is None:
            raw_maxlen = os.getenv("METRICS_RING_BUFFER_MAXLEN", "10000")
            if "METRICS_RING_BUFFER_MAXLEN" in os.environ:
                logger.warning(
                    "METRICS_RING_BUFFER_MAXLEN is deprecated; use METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED"
                )
        try:
            buffer_maxlen = int(raw_maxlen)
        except ValueError:
            buffer_maxlen = 10000
        if buffer_maxlen <= 0:
            buffer_maxlen = None
        raw_series_cap = os.getenv("METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC", "10000")
        try:
            cumulative_series_cap = int(raw_series_cap)
        except ValueError:
            cumulative_series_cap = 10000
        if cumulative_series_cap <= 0:
            cumulative_series_cap = None

        self._lock = threading.RLock()
        self.metrics: dict[str, MetricDefinition] = {}
        self._persistent_metric_definitions: dict[str, MetricDefinition] = {}
        self.instruments: dict[str, Any] = {}
        # Rolling window of metric samples; size configurable via METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED.
        self.values: dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_maxlen))
        # Cumulative aggregates for Prometheus export (monotonic counters, full histograms).
        self._cumulative_counters: dict[str, dict[tuple[tuple[str, str], ...], float]] = defaultdict(dict)
        self._cumulative_histograms: dict[str, dict[tuple[tuple[str, str], ...], dict[str, Any]]] = defaultdict(dict)
        self._cumulative_series_cap: int | None = cumulative_series_cap
        self._cumulative_series_dropped: dict[str, int] = defaultdict(int)
        self._cumulative_series_warned: set[str] = set()
        self.callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._legacy_cb_alias_metrics = {
            "circuit_breaker_state",
            "circuit_breaker_failures_total",
            "circuit_breaker_successes_total",
            "circuit_breaker_rejections_total",
            "circuit_breaker_trips_total",
            "circuit_breaker_timeouts_total",
        }

        # Register standard metrics
        self._register_standard_metrics()

    @property
    def telemetry(self):
        """Always return the *current* global telemetry manager so that a
        shutdown/re-init cycle is picked up automatically."""
        return get_telemetry_manager()

    @property
    def meter(self):
        """Return a meter from the current telemetry manager."""
        return self.telemetry.get_meter("tldw_server.metrics")

    @classmethod
    def _normalize_metric_name(cls, name: str) -> str:
        """Normalize a metric name to Prometheus-safe characters."""
        if name is None:
            return "metric"
        name_str = str(name)
        if not name_str:
            return "metric"
        normalized = cls._PROM_METRIC_NAME_RE.sub("_", name_str)
        normalized = re.sub(r"_+", "_", normalized)
        if not normalized:
            return "metric"
        if not re.match(r"[a-zA-Z_:]", normalized[0]):
            normalized = f"metric_{normalized}"
        return normalized

    @classmethod
    def _normalize_label_name(cls, name: str) -> str:
        """Normalize a label key to Prometheus-safe characters."""
        if name is None:
            return "label"
        name_str = str(name)
        if not name_str:
            return "label"
        normalized = cls._PROM_LABEL_KEY_RE.sub("_", name_str)
        normalized = re.sub(r"_+", "_", normalized)
        if not normalized:
            return "label"
        if not re.match(r"[a-zA-Z_]", normalized[0]):
            normalized = f"label_{normalized}"
        return normalized

    @classmethod
    def _normalize_labels(
        cls,
        labels: Optional[dict[str, Any]],
        reject_collisions: bool = False,
    ) -> dict[str, str]:
        """Normalize label keys and coerce values to strings.

        If reject_collisions is True, raise ValueError when distinct original
        keys normalize to the same label name but carry different values.
        """
        if not labels:
            return {}
        normalized: dict[str, str] = {}
        for key, value in labels.items():
            normalized_key = cls._normalize_label_name(str(key))
            normalized_value = "" if value is None else str(value)
            if normalized_key in normalized and normalized[normalized_key] != normalized_value:
                if reject_collisions:
                    raise ValueError(
                        f"Conflicting labels normalize to the same key {normalized_key!r}"
                    )
                logger.debug(f"Label key collision after normalization: {key} -> {normalized_key}")
            normalized[normalized_key] = normalized_value
        return normalized

    # Public alias so external modules (e.g. metrics_logger) need not reach
    # into a private method.
    normalize_labels = _normalize_labels

    @classmethod
    def _normalize_label_key(cls, labels: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        """Return a stable, sortable label key from the label dict."""
        if not labels:
            return ()
        normalized = cls._normalize_labels(labels)
        return tuple(sorted(normalized.items()))

    @staticmethod
    def _escape_label_value(value: str) -> str:
        """Escape label values for Prometheus text exposition."""
        return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    @classmethod
    def _format_label_str(cls, labels: dict[str, Any]) -> str:
        """Format labels for Prometheus exposition with proper escaping."""
        if not labels:
            return ""
        normalized = cls._normalize_labels(labels)
        parts = []
        for key, value in sorted(normalized.items()):
            escaped = cls._escape_label_value(value)
            parts.append(f'{key}="{escaped}"')
        return ",".join(parts)

    @staticmethod
    def _clone_metric_definition(definition: MetricDefinition) -> MetricDefinition:
        """Create a stable copy of a metric definition for persistence."""
        return MetricDefinition(
            name=definition.name,
            type=definition.type,
            description=definition.description,
            unit=definition.unit,
            labels=list(definition.labels),
            buckets=list(definition.buckets) if definition.buckets is not None else None,
        )

    def _register_standard_metrics(self):
        """Register standard application metrics."""
        # HTTP metrics
        self.register_metric(
            MetricDefinition(
                name="http_requests_total",
                type=MetricType.COUNTER,
                description="Total number of HTTP requests",
                labels=["method", "endpoint", "status"]
            )
        )

        self.register_metric(
            MetricDefinition(
                name="http_request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="HTTP request duration in seconds",
                unit="s",
                labels=["method", "endpoint"],
                buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
            )
        )

        # Database metrics
        self.register_metric(
            MetricDefinition(
                name="db_connections_active",
                type=MetricType.GAUGE,
                description="Number of active database connections",
                labels=["database"]
            )
        )

        self.register_metric(
            MetricDefinition(
                name="db_queries_total",
                type=MetricType.COUNTER,
                description="Total number of database queries",
                labels=["database", "operation"]
            )
        )

        self.register_metric(
            MetricDefinition(
                name="db_query_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Database query duration in seconds",
                unit="s",
                labels=["database", "operation"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="privilege_snapshots_table_bytes",
                type=MetricType.GAUGE,
                description="On-disk size of the privilege_snapshots table",
                unit="bytes",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="privilege_snapshots_table_rows",
                type=MetricType.GAUGE,
                description="Number of rows stored in the privilege_snapshots table",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="privilege_cache_hits_total",
                type=MetricType.COUNTER,
                description="Total number of privilege cache hits",
                labels=["backend", "layer"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="privilege_cache_misses_total",
                type=MetricType.COUNTER,
                description="Total number of privilege cache misses",
                labels=["backend", "layer"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="privilege_cache_invalidations_total",
                type=MetricType.COUNTER,
                description="Number of privilege cache invalidations broadcast",
                labels=["backend"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="privilege_cache_generation",
                type=MetricType.GAUGE,
                description="Current privilege cache generation counter",
                labels=["backend"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="privilege_cache_entries",
                type=MetricType.GAUGE,
                description="Current number of entries tracked in the local privilege cache",
                labels=["backend"],
            )
        )

        # LLM metrics
        self.register_metric(
            MetricDefinition(
                name="llm_requests_total",
                type=MetricType.COUNTER,
                description="Total number of LLM API requests",
                labels=["provider", "model", "status"]
            )
        )

        self.register_metric(
            MetricDefinition(
                name="llm_tokens_used_total",
                type=MetricType.COUNTER,
                description="Total number of tokens used",
                labels=["provider", "model", "type"]  # label type: prompt/completion
            )
        )

        self.register_metric(
            MetricDefinition(
                name="llm_request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="LLM request duration in seconds",
                unit="s",
                labels=["provider", "model"],
                buckets=[0.1, 0.5, 1, 2.5, 5, 10, 30, 60]
            )
        )

        self.register_metric(
            MetricDefinition(
                name="llm_cost_dollars",
                type=MetricType.COUNTER,
                description="Cumulative LLM API cost in dollars",
                unit="$",
                labels=["provider", "model"]
            )
        )
        # Detailed LLM usage variants for dashboards
        self.register_metric(
            MetricDefinition(
                name="llm_cost_dollars_by_user",
                type=MetricType.COUNTER,
                description="Cumulative LLM API cost in dollars by user",
                unit="$",
                labels=["provider", "model", "user_id"],
            )
        )

        # Realtime voice latency metrics (STT/TTS/voice-to-voice)
        buckets_s = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
        self.register_metric(
            MetricDefinition(
                name="stt_final_latency_seconds",
                type=MetricType.HISTOGRAM,
                description="End-of-speech to final transcript latency (seconds)",
                unit="s",
                labels=["model", "variant", "endpoint"],
                buckets=buckets_s,
            )
        )
        self.register_metric(
            MetricDefinition(
                name="tts_ttfb_seconds",
                type=MetricType.HISTOGRAM,
                description="TTS time-to-first-byte (seconds)",
                unit="s",
                labels=["provider", "voice", "format"],
                buckets=buckets_s,
            )
        )
        self.register_metric(
            MetricDefinition(
                name="voice_to_voice_seconds",
                type=MetricType.HISTOGRAM,
                description="Voice end-of-speech to first audio byte (seconds)",
                unit="s",
                labels=["provider", "route"],
                buckets=buckets_s,
            )
        )
        self.register_metric(
            MetricDefinition(
                name="audio_chat_latency_seconds",
                type=MetricType.HISTOGRAM,
                description="End-to-end latency for non-streaming audio chat turns",
                unit="s",
                labels=["stt_provider", "llm_provider", "tts_provider"],
                buckets=buckets_s,
            )
        )

        # Audio streaming health counters
        self.register_metric(
            MetricDefinition(
                name="audio_stream_underruns_total",
                type=MetricType.COUNTER,
                description="Total audio stream underruns",
                labels=["provider"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="audio_stream_errors_total",
                type=MetricType.COUNTER,
                description="Total audio streaming errors",
                labels=["component", "provider"],
            )
        )
        for definition in iter_stt_metric_definitions(MetricDefinition, MetricType):
            self.register_metric(definition)
        self.register_metric(
            MetricDefinition(
                name="llm_cost_dollars_by_operation",
                type=MetricType.COUNTER,
                description="Cumulative LLM API cost in dollars by operation",
                unit="$",
                labels=["provider", "model", "operation"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="llm_tokens_used_total_by_user",
                type=MetricType.COUNTER,
                description="Total number of tokens used by user",
                labels=["provider", "model", "type", "user_id"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="llm_tokens_used_total_by_operation",
                type=MetricType.COUNTER,
                description="Total number of tokens used by operation",
                labels=["provider", "model", "type", "operation"],
            )
        )

        # Infrastructure / Redis metrics
        self.register_metric(
            MetricDefinition(
                name="infra_redis_connection_attempts_total",
                type=MetricType.COUNTER,
                description="Redis client connection attempts",
                labels=["mode", "context", "outcome"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="infra_redis_connection_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Redis client connection duration in seconds",
                unit="s",
                labels=["mode", "context", "outcome"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="infra_redis_connection_errors_total",
                type=MetricType.COUNTER,
                description="Redis client connection errors",
                labels=["mode", "context", "error"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="infra_redis_fallback_total",
                type=MetricType.COUNTER,
                description="Redis client fallbacks to in-memory stub",
                labels=["mode", "context", "reason"],
            )
        )

        # RAG metrics
        self.register_metric(
            MetricDefinition(
                name="rag_queries_total",
                type=MetricType.COUNTER,
                description="Total number of RAG queries",
                labels=["pipeline", "status"]
            )
        )

        # Slides metrics
        self.register_metric(
            MetricDefinition(
                name="slides_generation_latency_seconds",
                type=MetricType.HISTOGRAM,
                description="Slides generation latency in seconds",
                unit="s",
                labels=["source_type"],
                buckets=[0.5, 1, 2.5, 5, 10, 20, 30, 60]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="slides_generation_errors_total",
                type=MetricType.COUNTER,
                description="Slides generation errors",
                labels=["source_type", "error"]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="slides_export_latency_seconds",
                type=MetricType.HISTOGRAM,
                description="Slides export latency in seconds",
                unit="s",
                labels=["format"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="slides_export_errors_total",
                type=MetricType.COUNTER,
                description="Slides export errors",
                labels=["format", "error"]
            )
        )

        # Generic application observability counters
        self.register_metric(
            MetricDefinition(
                name="app_exception_events_total",
                type=MetricType.COUNTER,
                description="Application exception events",
                labels=["component", "event"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="app_warning_events_total",
                type=MetricType.COUNTER,
                description="Application warning events",
                labels=["component", "event"],
            )
        )

        # Sandbox (code interpreter) metrics
        self.register_metric(
            MetricDefinition(
                name="sandbox_sessions_created_total",
                type=MetricType.COUNTER,
                description="Total sandbox sessions created",
                labels=["runtime"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_runs_started_total",
                type=MetricType.COUNTER,
                description="Total sandbox runs started",
                labels=["runtime"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_runs_completed_total",
                type=MetricType.COUNTER,
                description="Total sandbox runs finished",
                labels=["runtime", "outcome", "reason"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_run_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Sandbox run duration in seconds",
                unit="s",
                labels=["runtime", "outcome", "reason"],
                buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 60, 120, 300]
            )
        )
        # WS & queue metrics used by sandbox endpoints/streams
        self.register_metric(
            MetricDefinition(
                name="sandbox_log_truncations_total",
                type=MetricType.COUNTER,
                description="Total number of sandbox log truncations",
                labels=["component", "reason"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_queue_full_total",
                type=MetricType.COUNTER,
                description="Total number of sandbox queue-full events",
                labels=["component", "runtime", "reason"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_queue_ttl_expired_total",
                type=MetricType.COUNTER,
                description="Total number of sandbox queue TTL expirations",
                labels=["component", "runtime", "reason"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_queue_wait_seconds",
                type=MetricType.HISTOGRAM,
                description="Queue wait time before sandbox run execution",
                unit="s",
                labels=["runtime"],
                buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_ws_connections_opened_total",
                type=MetricType.COUNTER,
                description="Total sandbox WS connections opened",
                labels=["component"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_ws_heartbeats_sent_total",
                type=MetricType.COUNTER,
                description="Total sandbox WS heartbeats sent",
                labels=["component"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_ws_disconnects_total",
                type=MetricType.COUNTER,
                description="Total sandbox WS disconnects",
                labels=["component"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_ws_queue_drops_total",
                type=MetricType.COUNTER,
                description="Total WS queue drops (oldest dropped)",
                labels=["component", "reason"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_upload_bytes_total",
                type=MetricType.COUNTER,
                description="Total bytes uploaded to sandbox workspaces",
                unit="bytes",
                labels=["kind"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_upload_files_total",
                type=MetricType.COUNTER,
                description="Total files uploaded to sandbox workspaces",
                labels=["kind"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_janitor_runs_total",
                type=MetricType.COUNTER,
                description="Total artifact janitor runs",
                labels=["mode"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_janitor_removed_runs_total",
                type=MetricType.COUNTER,
                description="Total terminal runs with artifacts removed by janitor",
                labels=["mode"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_janitor_removed_files_total",
                type=MetricType.COUNTER,
                description="Total artifact files removed by janitor",
                labels=["mode"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_janitor_removed_bytes_total",
                type=MetricType.COUNTER,
                description="Total artifact bytes removed by janitor",
                unit="bytes",
                labels=["mode"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_janitor_duration_ms",
                type=MetricType.HISTOGRAM,
                description="Artifact janitor run duration in milliseconds",
                unit="ms",
                labels=["mode"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_reconcile_runs_total",
                type=MetricType.COUNTER,
                description="Total artifact usage reconciliation runs",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_reconcile_corrected_users_total",
                type=MetricType.COUNTER,
                description="Total users whose artifact usage counters were corrected",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_reconcile_corrected_bytes_total",
                type=MetricType.COUNTER,
                description="Total absolute artifact-byte correction volume",
                unit="bytes",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_reconcile_duration_ms",
                type=MetricType.HISTOGRAM,
                description="Artifact usage reconciliation duration in milliseconds",
                unit="ms",
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_maintenance_cycles_total",
                type=MetricType.COUNTER,
                description="Total artifact maintenance cycles executed by service",
                labels=["trigger"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="sandbox_artifact_maintenance_cycle_duration_ms",
                type=MetricType.HISTOGRAM,
                description="Artifact maintenance cycle duration in milliseconds",
                unit="ms",
                labels=["trigger"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
            )
        )

        self.register_metric(
            MetricDefinition(
                name="rag_retrieval_latency_seconds",
                type=MetricType.HISTOGRAM,
                description="RAG retrieval latency in seconds",
                unit="s",
                labels=["source", "pipeline"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
            )
        )

        self.register_metric(
            MetricDefinition(
                name="rag_documents_retrieved",
                type=MetricType.HISTOGRAM,
                description="Number of documents retrieved",
                labels=["source", "pipeline"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500]
            )
        )

        self.register_metric(
            MetricDefinition(
                name="rag_cache_hits_total",
                type=MetricType.COUNTER,
                description="Total RAG cache hits",
                labels=["cache_type"]
            )
        )

        self.register_metric(
            MetricDefinition(
                name="rag_cache_misses_total",
                type=MetricType.COUNTER,
                description="Total RAG cache misses",
                labels=["cache_type"]
            )
        )

        # RAG retrieval quality histograms
        self.register_metric(
            MetricDefinition(
                name="rag_retrieval_precision",
                type=MetricType.HISTOGRAM,
                description="Precision@K of RAG retrieval results",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_retrieval_recall",
                type=MetricType.HISTOGRAM,
                description="Recall@K of RAG retrieval results",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_retrieval_mrr",
                type=MetricType.HISTOGRAM,
                description="MRR of RAG retrieval results",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_retrieval_ndcg",
                type=MetricType.HISTOGRAM,
                description="NDCG@K of RAG retrieval results",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_retrieval_f1",
                type=MetricType.HISTOGRAM,
                description="F1@K of RAG retrieval results",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
        )

        # RAG reranker (LLM scoring) guardrails and activity
        self.register_metric(
            MetricDefinition(
                name="rag_reranker_llm_timeouts_total",
                type=MetricType.COUNTER,
                description="Total LLM reranker timeouts",
                labels=["strategy"]  # e.g., llm_scoring
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_reranker_llm_exceptions_total",
                type=MetricType.COUNTER,
                description="Total LLM reranker exceptions",
                labels=["strategy"]
            )
        )

        # Post-generation verification (adaptive check) metrics
        self.register_metric(
            MetricDefinition(
                name="rag_adaptive_retries_total",
                type=MetricType.COUNTER,
                description="Total adaptive post-check repair retries",
            )
        )
        # RAG factuality gauges (observability)
        self.register_metric(
            MetricDefinition(
                name="rag_hard_citation_coverage",
                type=MetricType.GAUGE,
                description="Per-answer hard-citation coverage ratio (0.0-1.0)",
                labels=["strategy"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_nli_unsupported_ratio",
                type=MetricType.GAUGE,
                description="Per-answer NLI unsupported ratio (refuted+NEI)/total",
                labels=["strategy"],
            )
        )

        # -----------------------------
        # Chatbook/Templating/Commands metrics
        # -----------------------------
        # Template rendering
        self.register_metric(
            MetricDefinition(
                name="template_render_success_total",
                type=MetricType.COUNTER,
                description="Successful template renders",
                labels=["source"],  # source=dict|chatbook|unknown
            )
        )
        self.register_metric(
            MetricDefinition(
                name="template_render_failure_total",
                type=MetricType.COUNTER,
                description="Failed template renders",
                labels=["source", "reason"],  # reason=parse|exception
            )
        )
        self.register_metric(
            MetricDefinition(
                name="template_render_timeout_total",
                type=MetricType.COUNTER,
                description="Template renders exceeding timeout threshold",
                labels=["source"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="template_output_truncated_total",
                type=MetricType.COUNTER,
                description="Template outputs truncated due to cap",
                labels=["source"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="template_render_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Template render duration in seconds",
                unit="s",
                labels=["source"],
                buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
            )
        )

        # Command router
        self.register_metric(
            MetricDefinition(
                name="chat_command_invoked_total",
                type=MetricType.COUNTER,
                description="Slash commands invoked",
                labels=["command", "status"],  # status=success|error|rate_limited|denied
            )
        )
        self.register_metric(
            MetricDefinition(
                name="chat_command_errors_total",
                type=MetricType.COUNTER,
                description="Slash command errors",
                labels=["command", "reason"],  # reason=exception|permission_denied|rate_limited
            )
        )
        self.register_metric(
            MetricDefinition(
                name="chat_conversation_search_requests_total",
                type=MetricType.COUNTER,
                description="Conversation search/list requests by strategy and outcome",
                labels=["query_strategy", "order_by", "deleted_scope", "outcome"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="chat_conversation_search_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Conversation search/list request duration in seconds",
                unit="s",
                labels=["query_strategy", "order_by", "deleted_scope", "outcome"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
            )
        )

        # Dictionary validator
        self.register_metric(
            MetricDefinition(
                name="chat_dictionary_validate_requests_total",
                type=MetricType.COUNTER,
                description="Dictionary validation requests",
                labels=["strict"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="chat_dictionary_validate_errors_total",
                type=MetricType.COUNTER,
                description="Dictionary validation errors by code",
                labels=["code"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="chat_dictionary_validate_warnings_total",
                type=MetricType.COUNTER,
                description="Dictionary validation warnings by code",
                labels=["code"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="chat_dictionary_validate_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Dictionary validation duration in seconds",
                unit="s",
                labels=["strict"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_unsupported_claims_total",
                type=MetricType.COUNTER,
                description="Total unsupported claims (refuted + NEI) observed in post-check",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_adaptive_fix_success_total",
                type=MetricType.COUNTER,
                description="Total adaptive post-check repairs that succeeded",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_postcheck_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Duration of post-generation verification and repair",
                unit="s",
                labels=["outcome"],
                buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20]
            )
        )
        # Adaptive rerun metrics
        self.register_metric(
            MetricDefinition(
                name="rag_adaptive_rerun_performed_total",
                type=MetricType.COUNTER,
                description="Total number of adaptive RAG reruns performed",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_adaptive_rerun_adopted_total",
                type=MetricType.COUNTER,
                description="Total number of adaptive RAG reruns whose results were adopted",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_adaptive_rerun_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Duration of adaptive RAG reruns",
                unit="s",
                labels=["adopted"],
                buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_reranker_llm_budget_exhausted_total",
                type=MetricType.COUNTER,
                description="Total LLM reranker budget exhaustions",
                labels=["strategy"]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_reranker_llm_docs_scored_total",
                type=MetricType.COUNTER,
                description="Total documents scored by LLM reranker",
                labels=["strategy"]
            )
        )

        # Per-phase timers and budgets (observability/SLOs)
        self.register_metric(
            MetricDefinition(
                name="rag_phase_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Duration per RAG pipeline phase",
                unit="s",
                labels=["phase", "difficulty"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_reranking_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Reranking duration (overall) in seconds",
                unit="s",
                labels=["strategy"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
            )
        )
        # Rewrite cache metrics
        self.register_metric(
            MetricDefinition(
                name="rag_rewrite_cache_hits_total",
                type=MetricType.COUNTER,
                description="Total rewrite cache hits",
                labels=["corpus", "intent"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_rewrite_cache_misses_total",
                type=MetricType.COUNTER,
                description="Total rewrite cache misses",
                labels=["corpus", "intent", "reason"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_rewrite_cache_puts_total",
                type=MetricType.COUNTER,
                description="Total rewrite cache writes",
                labels=["corpus", "intent"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_batch_query_reuse_total",
                type=MetricType.COUNTER,
                description="Total reused results in batch planning",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_phase_budget_exhausted_total",
                type=MetricType.COUNTER,
                description="Budget exhaustion events per phase",
                labels=["phase"]
            )
        )

        # Faithfulness tracking for SLOs
        self.register_metric(
            MetricDefinition(
                name="rag_total_claims_checked_total",
                type=MetricType.COUNTER,
                description="Total claims evaluated during post-generation verification",
            )
        )

        # Quality evaluation (nightly eval set) - dashboard inputs
        self.register_metric(
            MetricDefinition(
                name="rag_eval_faithfulness_score",
                type=MetricType.GAUGE,
                description="Average faithfulness score (1 - unsupported_ratio) on eval set",
                labels=["dataset"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_eval_coverage_score",
                type=MetricType.GAUGE,
                description="Average citation coverage on eval set",
                labels=["dataset"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_eval_last_run_timestamp",
                type=MetricType.GAUGE,
                description="Last successful eval run UNIX timestamp",
                labels=["dataset"],
            )
        )

        # Generation gating due to low evidence after reranking calibration
        self.register_metric(
            MetricDefinition(
                name="rag_generation_gated_total",
                type=MetricType.COUNTER,
                description="Total number of times answer generation was gated due to low relevance probability",
                labels=["strategy"]
            )
        )

        # Content policy / sanitation metrics
        self.register_metric(
            MetricDefinition(
                name="rag_policy_filtered_chunks_total",
                type=MetricType.COUNTER,
                description="Documents affected by content policy (pii/phi)",
                labels=["mode"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_sanitized_docs_total",
                type=MetricType.COUNTER,
                description="Documents sanitized by HTML allow-list",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_ocr_dropped_docs_total",
                type=MetricType.COUNTER,
                description="Documents dropped due to low OCR confidence",
            )
        )

        # Generation guardrails
        self.register_metric(
            MetricDefinition(
                name="rag_injection_chunks_downweighted_total",
                type=MetricType.COUNTER,
                description="Total retrieved chunks downweighted due to instruction-injection risk",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_numeric_mismatches_total",
                type=MetricType.COUNTER,
                description="Total numeric tokens from answers not found in sources",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="rag_missing_hard_citations_total",
                type=MetricType.COUNTER,
                description="Total answers with missing supporting spans for one or more sentences",
            )
        )

        # Embedding metrics
        self.register_metric(
            MetricDefinition(
                name="embeddings_generated_total",
                type=MetricType.COUNTER,
                description="Total number of embeddings generated",
                labels=["provider", "model"]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="hyde_questions_generated_total",
                type=MetricType.COUNTER,
                description="Total HYDE/doc2query questions generated",
                labels=["provider", "model", "source"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="hyde_generation_failures_total",
                type=MetricType.COUNTER,
                description="Total HYDE/doc2query generation failures",
                labels=["provider", "model", "source", "reason"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="hyde_vectors_written_total",
                type=MetricType.COUNTER,
                description="Total HYDE/doc2query vectors written to storage",
                labels=["store"],
            )
        )

        # Upload metrics
        self.register_metric(
            MetricDefinition(
                name="uploads_total",
                type=MetricType.COUNTER,
                description="Total number of uploaded files",
                labels=["user_id", "media_type"]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="upload_bytes_total",
                type=MetricType.COUNTER,
                description="Total bytes uploaded",
                unit="bytes",
                labels=["user_id", "media_type"]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="ingestion_requests_total",
                type=MetricType.COUNTER,
                description="Total ingestion requests by media type and outcome",
                labels=["media_type", "outcome"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="ingestion_processing_seconds",
                type=MetricType.HISTOGRAM,
                description="Ingestion processing duration in seconds",
                unit="s",
                labels=["media_type", "processor"],
                buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="ingestion_validation_failures_total",
                type=MetricType.COUNTER,
                description="Validation failures observed during ingestion",
                labels=["reason", "path_kind"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="ingestion_chunks_total",
                type=MetricType.COUNTER,
                description="Total persisted chunks across ingestion flows",
                labels=["media_type", "chunk_method"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="ingestion_embeddings_enqueue_total",
                type=MetricType.COUNTER,
                description="Embeddings enqueue attempts/outcomes for ingestion flows",
                labels=["path_kind", "outcome"],
            )
        )

        self.register_metric(
            MetricDefinition(
                name="embedding_generation_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Embedding generation duration in seconds",
                unit="s",
                labels=["provider", "model"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            )
        )

        # System metrics
        self.register_metric(
            MetricDefinition(
                name="system_cpu_usage_percent",
                type=MetricType.GAUGE,
                description="System CPU usage percentage",
                unit="%"
            )
        )

        self.register_metric(
            MetricDefinition(
                name="system_memory_usage_bytes",
                type=MetricType.GAUGE,
                description="System memory usage in bytes",
                unit="bytes"
            )
        )

        self.register_metric(
            MetricDefinition(
                name="system_disk_usage_bytes",
                type=MetricType.GAUGE,
                description="System disk usage in bytes",
                unit="bytes",
                labels=["mount_point"]
            )
        )

        # Error metrics
        self.register_metric(
            MetricDefinition(
                name="errors_total",
                type=MetricType.COUNTER,
                description="Total number of errors",
                labels=["component", "error_type"]
            )
        )

        # Scraping metrics
        self.register_metric(
            MetricDefinition(
                name="scrape_fetch_total",
                type=MetricType.COUNTER,
                description="Total number of scrape fetch attempts",
                labels=["backend", "outcome"],  # outcome: success|no_extract|error
            )
        )
        self.register_metric(
            MetricDefinition(
                name="scrape_fetch_latency_seconds",
                type=MetricType.HISTOGRAM,
                description="Latency of scrape fetch path",
                unit="s",
                labels=["backend"],
                buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="scrape_playwright_fallback_total",
                type=MetricType.COUNTER,
                description="Total number of fallbacks to Playwright",
                labels=["reason"],  # reason: js_required|no_extract|error
            )
        )
        self.register_metric(
            MetricDefinition(
                name="scrape_blocked_by_robots_total",
                type=MetricType.COUNTER,
                description="Total scrapes blocked by robots policy",
                labels=["domain"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="scrape_content_length_bytes",
                type=MetricType.HISTOGRAM,
                description="Extracted scrape content length in bytes",
                unit="bytes",
                labels=["backend"],
                buckets=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="extraction_strategy_total",
                type=MetricType.COUNTER,
                description="Total extraction strategy attempts",
                labels=["strategy", "status"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="extraction_strategy_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Duration of article extraction strategy runs",
                unit="s",
                labels=["strategy", "status"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="extraction_content_length_bytes",
                type=MetricType.HISTOGRAM,
                description="Extracted article content length in bytes by strategy",
                unit="bytes",
                labels=["strategy"],
                buckets=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
            )
        )

        # Security metrics
        self.register_metric(
            MetricDefinition(
                name="security_ssrf_block_total",
                type=MetricType.COUNTER,
                description="Total number of outbound URL validations blocked (SSRF protection)",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="security_headers_responses_total",
                type=MetricType.COUNTER,
                description="Total number of responses where security headers were applied",
            )
        )

        # BYOK metrics
        self.register_metric(
            MetricDefinition(
                name="byok_resolution_total",
                type=MetricType.COUNTER,
                description="Total number of BYOK credential resolutions by source",
                labels=["provider", "source", "allowlisted", "byok_enabled"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="byok_missing_credentials_total",
                type=MetricType.COUNTER,
                description="Total number of missing provider credential errors",
                labels=["provider", "operation", "allowlisted", "byok_enabled"],
            )
        )

        # Storage quota metrics (per user)
        self.register_metric(
            MetricDefinition(
                name="user_storage_used_mb",
                type=MetricType.GAUGE,
                description="Per-user storage used in MB",
                unit="MB",
                labels=["user_id"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="user_storage_quota_mb",
                type=MetricType.GAUGE,
                description="Per-user storage quota in MB",
                unit="MB",
                labels=["user_id"],
            )
        )

        # Circuit breaker metrics
        self.register_metric(
            MetricDefinition(
                name="circuit_breaker_state",
                type=MetricType.GAUGE,
                description="Circuit breaker state (0=closed, 1=open, 2=half-open)",
                labels=["category", "service", "operation"]
            )
        )

        self.register_metric(
            MetricDefinition(
                name="circuit_breaker_failures_total",
                type=MetricType.COUNTER,
                description="Total circuit breaker failures",
                labels=["category", "service", "operation", "outcome"]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="circuit_breaker_successes_total",
                type=MetricType.COUNTER,
                description="Total circuit breaker successes",
                labels=["category", "service", "operation"]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="circuit_breaker_rejections_total",
                type=MetricType.COUNTER,
                description="Total circuit breaker rejections",
                labels=["category", "service", "operation"]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="circuit_breaker_trips_total",
                type=MetricType.COUNTER,
                description="Total circuit breaker trips",
                labels=["category", "service", "reason"]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="circuit_breaker_timeouts_total",
                type=MetricType.COUNTER,
                description="Total circuit breaker call timeouts",
                labels=["category", "service", "operation"]
            )
        )

        # Prompt Studio metrics
        self.register_metric(
            MetricDefinition(
                name="prompt_studio_queue_depth",
                type=MetricType.GAUGE,
                description="Number of queued Prompt Studio jobs",
                labels=["backend"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="prompt_studio_processing",
                type=MetricType.GAUGE,
                description="Number of processing Prompt Studio jobs",
                labels=["backend"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="prompt_studio_leases_active",
                type=MetricType.GAUGE,
                description="Active leases for Prompt Studio jobs",
                labels=["backend"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="prompt_studio_leases_expiring_soon",
                type=MetricType.GAUGE,
                description="Prompt Studio leases expiring soon",
                labels=["backend"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="prompt_studio_leases_stale_processing",
                type=MetricType.GAUGE,
                description="Processing jobs with missing/expired lease",
                labels=["backend"],
            )
        )

        # Agentic RAG metrics
        self.register_metric(
            MetricDefinition(
                name="agentic_tool_calls_total",
                type=MetricType.COUNTER,
                description="Total agentic tool calls",
                labels=["tool"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="agentic_tool_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Duration per agentic tool call",
                unit="s",
                labels=["tool"],
                buckets=[0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="agentic_cache_hits_total",
                type=MetricType.COUNTER,
                description="Total agentic cache hits",
                labels=["cache_type"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="agentic_span_length_chars",
                type=MetricType.HISTOGRAM,
                description="Histogram of agentic span lengths (chars)",
                unit="chars",
                labels=["phase"],
                buckets=[64, 128, 256, 512, 1024, 2048, 4096]
            )
        )
        self.register_metric(
            MetricDefinition(
                name="span_bytes_read_total",
                type=MetricType.COUNTER,
                description="Total bytes read while assembling spans",
                unit="bytes",
                labels=["tool"],
            )
        )

        # FVA (Falsification-Verification Alignment) metrics
        self.register_metric(
            MetricDefinition(
                name="fva_falsification_triggered_total",
                type=MetricType.COUNTER,
                description="Total claims where falsification was triggered",
                labels=["reason"],  # low_confidence, sparse_evidence, forced, uncertain
            )
        )
        self.register_metric(
            MetricDefinition(
                name="fva_status_changes_total",
                type=MetricType.COUNTER,
                description="Total claim status changes after FVA adjudication",
                labels=["from_status", "to_status"],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="fva_anti_context_docs",
                type=MetricType.HISTOGRAM,
                description="Number of anti-context documents retrieved per claim",
                buckets=[0, 1, 2, 3, 5, 10, 20],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="fva_processing_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="FVA pipeline processing duration",
                unit="s",
                labels=["phase"],  # total, retrieval, adjudication
                buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="fva_wasted_falsification_total",
                type=MetricType.COUNTER,
                description="Falsifications that found no anti-context (wasted effort)",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="fva_claims_processed_total",
                type=MetricType.COUNTER,
                description="Total claims processed through FVA pipeline",
                labels=["final_status"],  # verified, contested, refuted, unverified
            )
        )
        self.register_metric(
            MetricDefinition(
                name="fva_adjudication_scores",
                type=MetricType.HISTOGRAM,
                description="Adjudication score distribution",
                labels=["score_type"],  # support, contradict, contestation
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
        )
        self.register_metric(
            MetricDefinition(
                name="fva_timeout_total",
                type=MetricType.COUNTER,
                description="Total FVA falsification timeouts",
            )
        )
        self.register_metric(
            MetricDefinition(
                name="fva_budget_exhausted_total",
                type=MetricType.COUNTER,
                description="Total times FVA budget was exhausted",
            )
        )

    def register_metric(self, definition: MetricDefinition, persistent: bool = True) -> bool:
        """
        Register a new metric definition.

        Args:
            definition: MetricDefinition object
            persistent: Whether the definition should survive reset()

        Returns:
            True if registered successfully
        """
        with self._lock:
            normalized_name = self._normalize_metric_name(definition.name)
            if normalized_name != definition.name:
                logger.debug(f"Normalizing metric name: {definition.name} -> {normalized_name}")

            if normalized_name in self.metrics:
                logger.warning(f"Metric {normalized_name} already registered")
                return False

            if normalized_name != definition.name:
                definition = self._clone_metric_definition(
                    MetricDefinition(
                        name=normalized_name,
                        type=definition.type,
                        description=definition.description,
                        unit=definition.unit,
                        labels=definition.labels,
                        buckets=definition.buckets,
                    )
                )

            if persistent:
                self._persistent_metric_definitions[normalized_name] = self._clone_metric_definition(definition)

            self.metrics[normalized_name] = definition

            # Create OpenTelemetry instrument
            if OTEL_AVAILABLE and self.meter and getattr(self.telemetry, "config", None) and self.telemetry.config.enable_metrics:
                instrument = self._create_instrument(definition)
                if instrument:
                    self.instruments[definition.name] = instrument

            logger.debug(f"Registered metric: {definition.name}")
            return True

    def normalize_metric_name(self, name: str) -> str:
        """Public helper for normalizing metric names."""
        return self._normalize_metric_name(name)

    def _create_instrument(self, definition: MetricDefinition):
        """Create an OpenTelemetry instrument for a metric definition."""
        try:
            if definition.type == MetricType.COUNTER:
                return self.meter.create_counter(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )

            elif definition.type == MetricType.GAUGE:
                # Gauges are handled via callbacks in OpenTelemetry
                return self.meter.create_observable_gauge(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit,
                    callbacks=[lambda options: self._gauge_callback(definition.name, options)]
                )

            elif definition.type == MetricType.HISTOGRAM:
                # If custom buckets were defined, register a view before creating the instrument
                try:
                    if definition.buckets:
                        self.telemetry.register_histogram_view(definition.name, definition.buckets)
                except Exception as view_error:
                    logger.debug("Metrics manager failed to register histogram view", exc_info=view_error)
                return self.meter.create_histogram(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )

            elif definition.type == MetricType.UP_DOWN_COUNTER:
                return self.meter.create_up_down_counter(
                    name=definition.name,
                    description=definition.description,
                    unit=definition.unit
                )

        except Exception as e:
            logger.error(f"Failed to create instrument for {definition.name}: {e}")
            return None

    def _gauge_callback(self, metric_name: str, options: 'CallbackOptions'):
        """Callback for observable gauge metrics.

        Emits one Observation per distinct label set using the latest value
        recorded for that label set. This preserves label cardinality in OTel.
        """
        observations = []

        try:
            # Group latest value by label set
            latest_by_labels: dict[tuple[tuple[str, str], ...], MetricValue] = {}
            with self._lock:
                if metric_name in self.values:
                    for mv in self.values[metric_name]:
                        # Build a stable key from sorted labels
                        label_key = self._normalize_label_key(mv.labels)
                        latest_by_labels[label_key] = mv

            for mv in latest_by_labels.values():
                observations.append(Observation(value=mv.value, attributes=mv.labels))
        except Exception as e:
            # Gauges must never break scrapes; return empty to avoid exporter errors
            logger.debug(f"Gauge callback error for {metric_name}: {e}")
        return observations

    def record(
        self,
        metric_name: str,
        value: float,
        labels: Optional[dict[str, str]] = None,
        _emit_legacy_alias: bool = True,
        _normalized: bool = False,
    ):
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Value to record
            labels: Optional labels/dimensions
            _normalized: If True, skip label normalization (caller already normalized).
        """
        original_name = metric_name
        metric_name = self._normalize_metric_name(metric_name)
        if not _normalized:
            try:
                labels = self._normalize_labels(labels, reject_collisions=True)
            except ValueError as exc:
                logger.warning("Rejecting metric {} due to conflicting normalized labels: {}", original_name, exc)
                return
        instrument = None
        callbacks: list[Callable] = []

        with self._lock:
            if metric_name not in self.metrics:
                logger.warning(f"Metric {original_name} not registered")
                return

            definition = self.metrics[metric_name]
            label_key = self._normalize_label_key(labels)
            store_sample = True

            if definition.type in (MetricType.COUNTER, MetricType.UP_DOWN_COUNTER):
                series = self._cumulative_counters[metric_name]
                if (
                    label_key not in series
                    and self._cumulative_series_cap is not None
                    and len(series) >= self._cumulative_series_cap
                ):
                    self._cumulative_series_dropped[metric_name] += 1
                    if metric_name not in self._cumulative_series_warned:
                        logger.warning(
                            "Cumulative series cap reached for metric {} (cap={}); dropping new label sets",
                            metric_name,
                            self._cumulative_series_cap,
                        )
                        self._cumulative_series_warned.add(metric_name)
                    store_sample = False
            elif definition.type == MetricType.HISTOGRAM:
                series = self._cumulative_histograms[metric_name]
                if (
                    label_key not in series
                    and self._cumulative_series_cap is not None
                    and len(series) >= self._cumulative_series_cap
                ):
                    self._cumulative_series_dropped[metric_name] += 1
                    if metric_name not in self._cumulative_series_warned:
                        logger.warning(
                            "Cumulative series cap reached for metric {} (cap={}); dropping new label sets",
                            metric_name,
                            self._cumulative_series_cap,
                        )
                        self._cumulative_series_warned.add(metric_name)
                    store_sample = False

            if store_sample:
                # Store value for aggregation only when the series is admitted.
                metric_value = MetricValue(value=value, labels=labels)
                self.values[metric_name].append(metric_value)
                if original_name != metric_name:
                    # Keep a non-normalized alias for tests that read raw keys.
                    self.values[original_name].append(metric_value)

                if definition.type in (MetricType.COUNTER, MetricType.UP_DOWN_COUNTER):
                    series = self._cumulative_counters[metric_name]
                    current = series.get(label_key, 0.0)
                    series[label_key] = current + value
                elif definition.type == MetricType.HISTOGRAM:
                    series = self._cumulative_histograms[metric_name]
                    hist = series.get(label_key)
                    if hist is None:
                        hist = {"count": 0, "sum": 0.0, "buckets": defaultdict(int)}
                        series[label_key] = hist
                    hist["count"] += 1
                    hist["sum"] += value
                    if definition.buckets:
                        for bucket in definition.buckets:
                            if value <= bucket:
                                hist["buckets"][bucket] += 1

            instrument = self.instruments.get(metric_name)
            callbacks = list(self.callbacks.get(metric_name, []))

        # Record in OpenTelemetry
        if instrument:
            try:
                if definition.type == MetricType.COUNTER:
                    instrument.add(value, attributes=labels)

                elif definition.type == MetricType.HISTOGRAM:
                    instrument.record(value, attributes=labels)

                elif definition.type == MetricType.UP_DOWN_COUNTER:
                    instrument.add(value, attributes=labels)

                # Gauges are handled via callbacks

            except Exception as e:
                logger.error(f"Failed to record metric {metric_name}: {e}")

        # Execute callbacks
        for callback in callbacks:
            try:
                callback(metric_name, value, labels)
            except Exception as e:
                logger.error(f"Metric callback error: {e}")

        if _emit_legacy_alias:
            self._emit_circuit_breaker_alias(metric_name, value, labels)

    def _emit_circuit_breaker_alias(self, metric_name: str, value: float, labels: dict[str, str]):
        if metric_name not in self._legacy_cb_alias_metrics:
            return
        category = labels.get("category")
        service = labels.get("service")
        if not category or not service:
            return
        if service.startswith(f"{category}:"):
            return
        legacy_labels = labels.copy()
        legacy_labels["service"] = f"{category}:{service}"
        self.record(metric_name, value, legacy_labels, _emit_legacy_alias=False)

    def increment(self, metric_name: str, value: float = 1, labels: Optional[dict[str, str]] = None):
        """
        Increment a counter metric.

        Args:
            metric_name: Name of the counter metric
            value: Amount to increment (default 1)
            labels: Optional labels
        """
        self.record(metric_name, value, labels)

    def set_gauge(self, metric_name: str, value: float, labels: Optional[dict[str, str]] = None):
        """
        Set a gauge metric value.

        Args:
            metric_name: Name of the gauge metric
            value: Value to set
            labels: Optional labels
        """
        self.record(metric_name, value, labels)

    def observe(self, metric_name: str, value: float, labels: Optional[dict[str, str]] = None):
        """
        Observe a value for histogram metric.

        Args:
            metric_name: Name of the histogram metric
            value: Value to observe
            labels: Optional labels
        """
        self.record(metric_name, value, labels)

    @contextmanager
    def timer(self, metric_name: str, labels: Optional[dict[str, str]] = None):
        """
        Context manager to time an operation.

        Args:
            metric_name: Name of the histogram metric for timing
            labels: Optional labels

        Yields:
            Timer context
        """
        start_time = time.monotonic()
        try:
            yield
        finally:
            duration = time.monotonic() - start_time
            self.observe(metric_name, duration, labels)

    def add_callback(self, metric_name: str, callback: Callable):
        """
        Add a callback for metric events.

        Args:
            metric_name: Name of the metric
            callback: Callable(metric_name, value, labels)
        """
        metric_name = self._normalize_metric_name(metric_name)
        with self._lock:
            self.callbacks[metric_name].append(callback)

    def get_cumulative_counter(self, metric_name: str, labels: Optional[dict[str, Any]] = None) -> float:
        """Get the cumulative counter value for a metric/label set."""
        metric_name = self._normalize_metric_name(metric_name)
        label_key = self._normalize_label_key(labels or {})
        with self._lock:
            return self._cumulative_counters.get(metric_name, {}).get(label_key, 0.0)

    def get_cumulative_counter_total(self, metric_name: str) -> float:
        """Get the cumulative counter total across every label-set for a metric."""
        metric_name = self._normalize_metric_name(metric_name)
        with self._lock:
            series = self._cumulative_counters.get(metric_name, {})
            return float(sum(series.values()))

    def get_cumulative_counter_totals_by_label(self, metric_name: str, label_name: str) -> dict[str, float]:
        """Aggregate cumulative counter totals grouped by a specific label key."""
        metric_name = self._normalize_metric_name(metric_name)
        normalized_label = self._normalize_label_name(label_name)
        totals: dict[str, float] = defaultdict(float)
        with self._lock:
            series = self._cumulative_counters.get(metric_name, {})
            for label_key, value in series.items():
                label_dict = dict(label_key)
                label_value = label_dict.get(normalized_label)
                if label_value is None:
                    continue
                totals[str(label_value)] += float(value)
        return dict(totals)

    def get_metric_stats(self, metric_name: str, labels: Optional[dict[str, str]] = None) -> dict[str, Any]:
        """
        Get statistics for a metric.

        Args:
            metric_name: Name of the metric
            labels: Optional label filter

        Returns:
            Dictionary with metric statistics
        """
        metric_name = self._normalize_metric_name(metric_name)
        with self._lock:
            if metric_name not in self.values:
                return {}
            values = list(self.values[metric_name])
        if labels:
            # Filter by exact match on provided labels
            normalized_labels = self._normalize_labels(labels)
            values = [val for val in values if all(
                val.labels.get(key) == expected for key, expected in normalized_labels.items()
            )]

        if not values:
            return {}

        numeric_values = [v.value for v in values]

        return {
            "count": len(numeric_values),
            "sum": sum(numeric_values),
            "mean": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "stddev": statistics.stdev(numeric_values) if len(numeric_values) > 1 else 0,
            "latest": numeric_values[-1],
            "latest_timestamp": values[-1].timestamp
        }

    def get_metric_sample_counts_by_label(
        self,
        metric_name: str,
        label_name: str,
        *,
        missing_label: str,
    ) -> dict[str, int]:
        """Return sample counts grouped by a normalized label key."""
        metric_name = self._normalize_metric_name(metric_name)
        normalized_label = self._normalize_label_name(label_name)
        totals: dict[str, int] = defaultdict(int)
        with self._lock:
            histogram_series = dict(self._cumulative_histograms.get(metric_name, {}))
            if histogram_series:
                for label_key, histogram in histogram_series.items():
                    label_value = dict(label_key).get(normalized_label) or missing_label
                    totals[str(label_value)] += int(histogram.get("count", 0))
                return dict(sorted(totals.items()))
            values = list(self.values.get(metric_name, ()))

        for metric_value in values:
            label_value = metric_value.labels.get(normalized_label) or missing_label
            totals[str(label_value)] += 1
        return dict(sorted(totals.items()))

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """
        Get all current metric values and statistics.

        Returns:
            Dictionary of metric names to their statistics
        """
        result = {}

        with self._lock:
            metrics_items = list(self.metrics.items())

        for metric_name, definition in metrics_items:
            stats = self.get_metric_stats(metric_name)
            if stats:
                result[metric_name] = {
                    "type": definition.type.value,
                    "description": definition.description,
                    "unit": definition.unit,
                    "stats": stats
                }

        return result

    def export_prometheus_format(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        with self._lock:
            for metric_name, definition in self.metrics.items():
                metric_name = self._normalize_metric_name(metric_name)
                prom_type = (
                    "counter" if definition.type == MetricType.COUNTER else
                    "gauge" if definition.type in (MetricType.GAUGE, MetricType.UP_DOWN_COUNTER) else
                    "histogram" if definition.type == MetricType.HISTOGRAM else
                    definition.type.value
                )

                if definition.type in (MetricType.COUNTER, MetricType.UP_DOWN_COUNTER):
                    series = self._cumulative_counters.get(metric_name)
                    if not series:
                        continue

                    lines.append(f"# HELP {metric_name} {definition.description}")
                    lines.append(f"# TYPE {metric_name} {prom_type}")

                    for label_key, total in series.items():
                        label_dict = dict(label_key)
                        label_str = self._format_label_str(label_dict)
                        if label_str:
                            lines.append(f"{metric_name}{{{label_str}}} {total}")
                        else:
                            lines.append(f"{metric_name} {total}")

                    # Emit alias series for cache_* metrics to rag_cache_* for consistency
                    if metric_name in {"cache_hits_total", "cache_misses_total"}:
                        alias_name = (
                            "rag_cache_hits_total" if metric_name == "cache_hits_total" else "rag_cache_misses_total"
                        )
                        alias_series = self._cumulative_counters.get(alias_name)
                        if not alias_series:
                            lines.append(f"# HELP {alias_name} Aliased from {metric_name} for RAG cache consistency")
                            lines.append(f"# TYPE {alias_name} counter")
                            for label_key, total in series.items():
                                label_dict = dict(label_key)
                                if "cache" in label_dict:
                                    label_dict["cache_type"] = label_dict.pop("cache")
                                alias_labels = self._format_label_str(label_dict)
                                if alias_labels:
                                    lines.append(f"{alias_name}{{{alias_labels}}} {total}")
                                else:
                                    lines.append(f"{alias_name} {total}")

                elif definition.type == MetricType.HISTOGRAM:
                    series = self._cumulative_histograms.get(metric_name)
                    if not series:
                        continue

                    lines.append(f"# HELP {metric_name} {definition.description}")
                    lines.append(f"# TYPE {metric_name} {prom_type}")

                    for label_key, hist in series.items():
                        label_dict = dict(label_key)
                        if definition.buckets:
                            for bucket in definition.buckets:
                                count = hist["buckets"].get(bucket, 0)
                                bucket_labels = dict(label_dict)
                                bucket_labels["le"] = str(bucket)
                                label_str = self._format_label_str(bucket_labels)
                                lines.append(f"{metric_name}_bucket{{{label_str}}} {count}")

                        bucket_labels = dict(label_dict)
                        bucket_labels["le"] = "+Inf"
                        label_str = self._format_label_str(bucket_labels)
                        lines.append(f"{metric_name}_bucket{{{label_str}}} {hist['count']}")

                        label_str = self._format_label_str(label_dict)
                        if label_str:
                            lines.append(f"{metric_name}_sum{{{label_str}}} {hist['sum']}")
                            lines.append(f"{metric_name}_count{{{label_str}}} {hist['count']}")
                        else:
                            lines.append(f"{metric_name}_sum {hist['sum']}")
                            lines.append(f"{metric_name}_count {hist['count']}")

                else:
                    if metric_name not in self.values:
                        continue

                    lines.append(f"# HELP {metric_name} {definition.description}")
                    lines.append(f"# TYPE {metric_name} {prom_type}")

                    label_groups = defaultdict(list)
                    for value in self.values[metric_name]:
                        label_key = self._normalize_label_key(value.labels)
                        label_groups[label_key].append(value)

                    for label_key, values in label_groups.items():
                        latest = values[-1]
                        label_dict = dict(label_key)
                        label_str = self._format_label_str(label_dict)
                        if label_str:
                            lines.append(f"{metric_name}{{{label_str}}} {latest.value}")
                        else:
                            lines.append(f"{metric_name} {latest.value}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset runtime state and restore persistent registered definitions."""
        persistent_definitions = []
        with self._lock:
            persistent_definitions = [
                self._clone_metric_definition(definition)
                for definition in self._persistent_metric_definitions.values()
            ]
            self.metrics.clear()
            self.instruments.clear()
            self.callbacks.clear()
            self.values.clear()
            self._cumulative_counters.clear()
            self._cumulative_histograms.clear()
            self._cumulative_series_dropped.clear()
            self._cumulative_series_warned.clear()

            for definition in persistent_definitions:
                self.register_metric(definition, persistent=False)


# Global metrics registry instance
_metrics_registry: Optional[MetricsRegistry] = None
_metrics_registry_lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    """
    Get or create the global metrics registry.

    Returns:
        MetricsRegistry instance
    """
    global _metrics_registry
    if _metrics_registry is None:
        with _metrics_registry_lock:
            if _metrics_registry is None:
                _metrics_registry = MetricsRegistry()
    return _metrics_registry


# Convenience functions for common operations
def record_metric(metric_name: str, value: float, labels: Optional[dict[str, str]] = None):
    """Record a metric value."""
    get_metrics_registry().record(metric_name, value, labels)


def increment_counter(metric_name: str, value: float = 1, labels: Optional[dict[str, str]] = None):
    """Increment a counter metric."""
    get_metrics_registry().increment(metric_name, value, labels)


def set_gauge(metric_name: str, value: float, labels: Optional[dict[str, str]] = None):
    """Set a gauge metric value."""
    get_metrics_registry().set_gauge(metric_name, value, labels)


def observe_histogram(metric_name: str, value: float, labels: Optional[dict[str, str]] = None):
    """Observe a value for histogram metric."""
    get_metrics_registry().observe(metric_name, value, labels)


@contextmanager
def time_operation(metric_name: str, labels: Optional[dict[str, str]] = None):
    """Time an operation and record to histogram metric."""
    with get_metrics_registry().timer(metric_name, labels):
        yield

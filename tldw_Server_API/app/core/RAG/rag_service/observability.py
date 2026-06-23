# observability.py
"""
Observability module for RAG pipeline.

Provides comprehensive metrics, tracing, monitoring, and alerting capabilities
for the RAG service using OpenTelemetry and custom metrics.
"""

import asyncio
import statistics
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger

# Try to import OpenTelemetry components
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    metrics = trace = None
    OTLPMetricExporter = OTLPSpanExporter = None
    MeterProvider = PeriodicExportingMetricReader = Resource = None
    TracerProvider = BatchSpanProcessor = None
    Status = StatusCode = None
    OTEL_AVAILABLE = False
    logger.warning("OpenTelemetry not available. Using fallback metrics.")


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class TraceSpan:
    """A trace span for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def end(self):
        """End the span and calculate duration."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class Alert:
    """An alert triggered by metric thresholds."""
    id: str
    metric_name: str
    condition: str
    threshold: float
    current_value: float
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores metrics."""

    def __init__(self, max_history: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_history: Maximum metric points to keep in memory
        """
        self.max_history = max_history
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)

    def record_counter(self, name: str, value: float = 1, labels: Optional[dict[str, str]] = None):
        """Record counter metric."""
        key = self._get_metric_key(name, labels)
        self.counters[key] += value

        self.metrics[name].append(MetricPoint(
            name=name,
            value=self.counters[key],
            labels=labels or {},
            metric_type=MetricType.COUNTER
        ))

    def record_gauge(self, name: str, value: float, labels: Optional[dict[str, str]] = None):
        """Record gauge metric."""
        key = self._get_metric_key(name, labels)
        self.gauges[key] = value

        self.metrics[name].append(MetricPoint(
            name=name,
            value=value,
            labels=labels or {},
            metric_type=MetricType.GAUGE
        ))

    def record_histogram(self, name: str, value: float, labels: Optional[dict[str, str]] = None):
        """Record histogram metric."""
        key = self._get_metric_key(name, labels)

        if key not in self.histograms:
            self.histograms[key] = []

        self.histograms[key].append(value)

        # Keep only recent values
        if len(self.histograms[key]) > self.max_history:
            self.histograms[key] = self.histograms[key][-self.max_history:]

        self.metrics[name].append(MetricPoint(
            name=name,
            value=value,
            labels=labels or {},
            metric_type=MetricType.HISTOGRAM
        ))

    def _get_metric_key(self, name: str, labels: Optional[dict[str, str]]) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def get_metric_stats(self, name: str, window_seconds: int = 300) -> dict[str, Any]:
        """
        Get statistics for a metric over time window.

        Args:
            name: Metric name
            window_seconds: Time window in seconds

        Returns:
            Statistics dictionary
        """
        if name not in self.metrics:
            return {}

        now = time.time()
        cutoff = now - window_seconds

        # Filter points within window
        recent_points = [
            p for p in self.metrics[name]
            if p.timestamp >= cutoff
        ]

        if not recent_points:
            return {}

        values = [p.value for p in recent_points]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "p95": np.percentile(values, 95) if values else 0,
            "p99": np.percentile(values, 99) if values else 0
        }

    def get_histogram_percentiles(
        self,
        name: str,
        labels: Optional[dict[str, str]] = None,
        percentiles: list[float] | None = None
    ) -> dict[float, float]:
        """Get percentiles for histogram metric."""
        if percentiles is None:
            percentiles = [50, 75, 90, 95, 99]
        key = self._get_metric_key(name, labels)

        if key not in self.histograms or not self.histograms[key]:
            return {}

        values = self.histograms[key]
        return {
            p: float(np.percentile(values, p))
            for p in percentiles
        }


class Tracer:
    """Distributed tracing implementation."""

    def __init__(self):
        """Initialize tracer."""
        self.spans: dict[str, TraceSpan] = {}
        self.active_spans: list[TraceSpan] = []
        self.completed_spans: deque = deque(maxlen=1000)

        # OpenTelemetry tracer if available
        self.otel_tracer = None
        if OTEL_AVAILABLE:
            try:
                trace.set_tracer_provider(TracerProvider())
                self.otel_tracer = trace.get_tracer(__name__)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.warning("Failed to initialize OpenTelemetry tracer")

    @contextmanager
    def span(self, operation: str, attributes: Optional[dict[str, Any]] = None):
        """
        Create a trace span context manager.

        Args:
            operation: Operation name
            attributes: Span attributes

        Yields:
            TraceSpan instance
        """
        span = self.start_span(operation, attributes)

        try:
            yield span
            span.status = "ok"
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = "span_failed"
            span.attributes["error_type"] = type(e).__name__
            raise
        finally:
            self.end_span(span)

    def start_span(
        self,
        operation: str,
        attributes: Optional[dict[str, Any]] = None,
        parent: Optional[TraceSpan] = None
    ) -> TraceSpan:
        """Start a new trace span."""
        import uuid

        # Generate IDs
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:8]

        # Get parent from context if not provided
        if not parent and self.active_spans:
            parent = self.active_spans[-1]

        parent_span_id = parent.span_id if parent else None

        # Create span
        span = TraceSpan(
            trace_id=trace_id if not parent else parent.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
            attributes=attributes or {}
        )

        self.spans[span_id] = span
        self.active_spans.append(span)

        # OpenTelemetry span if available
        if self.otel_tracer:
            try:
                otel_span = self.otel_tracer.start_span(operation)
                if attributes:
                    for key, value in attributes.items():
                        otel_span.set_attribute(key, str(value))
                span.attributes["otel_span"] = otel_span
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.debug("Failed to create OpenTelemetry span")

        return span

    def end_span(self, span: TraceSpan):
        """End a trace span."""
        span.end()

        # Remove from active spans
        if span in self.active_spans:
            self.active_spans.remove(span)

        # Move to completed
        self.completed_spans.append(span)

        # End OpenTelemetry span if present
        if "otel_span" in span.attributes:
            try:
                otel_span = span.attributes.pop("otel_span")
                if span.status == "error":
                    otel_span.set_status(Status(StatusCode.ERROR))
                otel_span.end()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.debug("Failed to end OpenTelemetry span")

    def add_event(self, span: TraceSpan, name: str, attributes: Optional[dict[str, Any]] = None):
        """Add event to span."""
        event = {
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        span.events.append(event)

    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        """Get all spans for a trace."""
        return [
            span for span in self.completed_spans
            if span.trace_id == trace_id
        ]


class PerformanceMonitor:
    """Monitors and profiles performance."""

    def __init__(self):
        """Initialize performance monitor."""
        self.operation_times: dict[str, list[float]] = defaultdict(list)
        self.slow_operations: deque = deque(maxlen=100)
        self.memory_usage: deque = deque(maxlen=1000)

    @contextmanager
    def measure(self, operation: str, threshold_ms: float = 1000):
        """
        Measure operation performance.

        Args:
            operation: Operation name
            threshold_ms: Slow operation threshold in milliseconds

        Yields:
            Start time
        """
        start_time = time.time()

        try:
            yield start_time
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.operation_times[operation].append(duration_ms)

            # Track slow operations
            if duration_ms > threshold_ms:
                self.slow_operations.append({
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "timestamp": start_time,
                    "stack": traceback.format_stack()
                })

                logger.warning(
                    f"Slow operation detected: {operation} took {duration_ms:.2f}ms "
                    f"(threshold: {threshold_ms}ms)"
                )

    def get_operation_stats(self, operation: str) -> dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.operation_times:
            return {}

        times = self.operation_times[operation]

        return {
            "count": len(times),
            "min_ms": min(times),
            "max_ms": max(times),
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "p95_ms": float(np.percentile(times, 95)) if times else 0.0,
            "p99_ms": float(np.percentile(times, 99)) if times else 0.0
        }

    def get_memory_usage(self) -> dict[str, Any]:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()

            memory_info = process.memory_info()
            memory_percent = process.memory_percent()

            usage: dict[str, Any] = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": memory_percent
            }

            self.memory_usage.append({
                "timestamp": time.time(),
                **usage
            })

            return usage

        except ImportError:
            return {"error": "psutil not available"}
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, psutil.Error):
            return {"error": "memory_usage_unavailable"}


class AlertManager:
    """Manages alerts based on metric thresholds."""

    def __init__(self):
        """Initialize alert manager."""
        self.rules: list[dict[str, Any]] = []
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_handlers: list[Callable] = []

    def add_rule(
        self,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity,
        message_template: str,
        cooldown_seconds: int = 300
    ):
        """
        Add alert rule.

        Args:
            metric_name: Metric to monitor
            condition: Condition (>, <, ==, !=)
            threshold: Threshold value
            severity: Alert severity
            message_template: Message template
            cooldown_seconds: Cooldown between alerts
        """
        self.rules.append({
            "metric_name": metric_name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "message_template": message_template,
            "cooldown_seconds": cooldown_seconds,
            "last_triggered": 0
        })

    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)

    def check_rules(self, metrics: dict[str, float]):
        """
        Check alert rules against current metrics.

        Args:
            metrics: Current metric values
        """
        now = time.time()

        for rule in self.rules:
            metric_name = rule["metric_name"]

            if metric_name not in metrics:
                continue

            value = metrics[metric_name]

            # Check condition
            triggered = False
            if rule["condition"] == ">":
                triggered = value > rule["threshold"]
            elif rule["condition"] == "<":
                triggered = value < rule["threshold"]
            elif rule["condition"] == "==":
                triggered = value == rule["threshold"]
            elif rule["condition"] == "!=":
                triggered = value != rule["threshold"]

            if triggered:
                # Check cooldown
                if now - rule["last_triggered"] < rule["cooldown_seconds"]:
                    continue

                # Create alert
                alert = Alert(
                    id=f"{metric_name}_{int(now)}",
                    metric_name=metric_name,
                    condition=rule["condition"],
                    threshold=rule["threshold"],
                    current_value=value,
                    severity=rule["severity"],
                    message=rule["message_template"].format(
                        metric=metric_name,
                        value=value,
                        threshold=rule["threshold"]
                    )
                )

                # Store alert
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                rule["last_triggered"] = now

                # Call handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except (AttributeError, ConnectionError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                        logger.error("Alert handler failed")

                logger.warning("Alert triggered")

    def clear_alert(self, alert_id: str):
        """Clear an active alert."""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]


class ObservabilitySystem:
    """Main observability system coordinating all components."""

    def __init__(
        self,
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        enable_monitoring: bool = True,
        enable_alerting: bool = True,
        otlp_endpoint: Optional[str] = None
    ):
        """
        Initialize observability system.

        Args:
            enable_metrics: Enable metrics collection
            enable_tracing: Enable distributed tracing
            enable_monitoring: Enable performance monitoring
            enable_alerting: Enable alerting
            otlp_endpoint: OpenTelemetry endpoint
        """
        self.metrics = MetricsCollector() if enable_metrics else None
        self.tracer = Tracer() if enable_tracing else None
        self.monitor = PerformanceMonitor() if enable_monitoring else None
        self.alerts = AlertManager() if enable_alerting else None

        # Setup OpenTelemetry if available and endpoint provided
        if OTEL_AVAILABLE and otlp_endpoint:
            self._setup_opentelemetry(otlp_endpoint)

        # Setup default alert rules
        if self.alerts:
            self._setup_default_alerts()

        # Start monitoring loop
        self.monitoring_task = None
        if enable_monitoring:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    def _setup_opentelemetry(self, endpoint: str):
        """Setup OpenTelemetry exporters."""
        try:
            # Setup tracing
            span_exporter = OTLPSpanExporter(endpoint=endpoint)
            span_processor = BatchSpanProcessor(span_exporter)

            tracer_provider = TracerProvider(
                resource=Resource.create({"service.name": "rag-service"})
            )
            tracer_provider.add_span_processor(span_processor)
            trace.set_tracer_provider(tracer_provider)

            # Setup metrics
            metric_exporter = OTLPMetricExporter(endpoint=endpoint)
            metric_reader = PeriodicExportingMetricReader(
                exporter=metric_exporter,
                export_interval_millis=60000
            )

            meter_provider = MeterProvider(
                resource=Resource.create({"service.name": "rag-service"}),
                metric_readers=[metric_reader]
            )
            metrics.set_meter_provider(meter_provider)

            logger.info("OpenTelemetry configured")

        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.error("Failed to setup OpenTelemetry")

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        if self.alerts is None:
            return
        # High latency alert
        self.alerts.add_rule(
            metric_name="rag_pipeline_duration_ms",
            condition=">",
            threshold=5000,
            severity=AlertSeverity.WARNING,
            message_template="RAG pipeline latency high: {value:.2f}ms > {threshold}ms"
        )

        # Error rate alert
        self.alerts.add_rule(
            metric_name="rag_error_rate",
            condition=">",
            threshold=0.05,
            severity=AlertSeverity.ERROR,
            message_template="RAG error rate high: {value:.2%} > {threshold:.2%}"
        )

        # Memory usage alert
        self.alerts.add_rule(
            metric_name="memory_percent",
            condition=">",
            threshold=80,
            severity=AlertSeverity.WARNING,
            message_template="Memory usage high: {value:.1f}% > {threshold}%"
        )

    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                # Collect system metrics
                if self.monitor:
                    memory = self.monitor.get_memory_usage()

                    if self.metrics and "percent" in memory:
                        self.metrics.record_gauge("memory_percent", memory["percent"])
                        self.metrics.record_gauge("memory_rss_mb", memory.get("rss_mb", 0))

                # Check alert rules
                if self.alerts and self.metrics:
                    current_metrics = {
                        "memory_percent": memory.get("percent", 0),
                        "rag_error_rate": self._calculate_error_rate()
                    }
                    self.alerts.check_rules(current_metrics)

                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute

            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.error("Monitoring loop error")
                await asyncio.sleep(60)

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        if not self.metrics:
            return 0.0

        # Get recent success/failure counts
        stats = self.metrics.get_metric_stats("rag_pipeline_success", window_seconds=300)
        success = stats.get("count", 0)

        stats = self.metrics.get_metric_stats("rag_pipeline_failure", window_seconds=300)
        failure = stats.get("count", 0)

        total = success + failure
        return failure / total if total > 0 else 0.0

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """Get metrics for dashboard display."""
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "performance": {},
            "alerts": {}
        }

        # Collect metrics
        if self.metrics:
            dashboard["metrics"] = {
                "counters": dict(self.metrics.counters),
                "gauges": dict(self.metrics.gauges),
                "histograms": {
                    name: self.metrics.get_histogram_percentiles(name)
                    for name in ["rag_pipeline_duration_ms", "embedding_duration_ms"]
                    if name in self.metrics.histograms
                }
            }

        # Collect performance stats
        if self.monitor:
            dashboard["performance"] = {
                "memory": self.monitor.get_memory_usage(),
                "slow_operations": list(self.monitor.slow_operations)[-10:],
                "operation_stats": {
                    op: self.monitor.get_operation_stats(op)
                    for op in ["search", "embedding", "reranking"]
                    if op in self.monitor.operation_times
                }
            }

        # Collect alerts
        if self.alerts:
            dashboard["alerts"] = {
                "active": [
                    {
                        "id": alert.id,
                        "metric": alert.metric_name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp
                    }
                    for alert in self.alerts.active_alerts.values()
                ],
                "history_count": len(self.alerts.alert_history)
            }

        return dashboard


# Global instance
_observability: Optional[ObservabilitySystem] = None


def get_observability() -> ObservabilitySystem:
    """Get or create global observability instance."""
    global _observability
    if _observability is None:
        _observability = ObservabilitySystem()
    return _observability


# Pipeline integration functions

async def observe_pipeline(context: Any, **kwargs) -> Any:
    """Add observability to RAG pipeline."""
    if not context.config.get("observability", {}).get("enabled", True):
        return context

    obs = get_observability()

    # Start trace span
    span = None
    if obs.tracer:
        span = obs.tracer.start_span(
            "rag_pipeline",
            attributes={
                "query": context.query[:100],  # Truncate for safety
                "user_id": context.metadata.get("user_id", "anonymous")
            }
        )
        context.metadata["trace_id"] = span.trace_id
        context.metadata["span_id"] = span.span_id

    # Measure performance
    start_time = time.time()

    try:
        # Process pipeline
        result = context

        # Record success metrics
        if obs.metrics:
            duration_ms = (time.time() - start_time) * 1000
            obs.metrics.record_histogram("rag_pipeline_duration_ms", duration_ms)
            obs.metrics.record_counter("rag_pipeline_success")

        if span:
            span.status = "ok"

        return result

    except Exception as e:
        # Record failure metrics
        if obs.metrics:
            obs.metrics.record_counter("rag_pipeline_failure")
            obs.metrics.record_counter(f"rag_pipeline_error_{type(e).__name__}")

        if span:
            span.status = "error"
            span.attributes["error"] = "rag_pipeline_failed"
            span.attributes["error_type"] = type(e).__name__

        raise

    finally:
        if span and obs.tracer:
            obs.tracer.end_span(span)

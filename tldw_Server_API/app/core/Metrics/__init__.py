"""
Unified metrics and telemetry module for tldw_server.

This module provides comprehensive observability through:
- OpenTelemetry-based metrics and tracing
- Prometheus-compatible metrics export
- Distributed tracing support
- Easy-to-use decorators for automatic instrumentation
"""

# Telemetry initialization and management
# Decorators for easy instrumentation
from .decorators import (
    cache_metrics,
    count_calls,
    measure_latency,
    monitor_resource,
    track_errors,
    track_llm_usage,
    track_metrics,
)

# Legacy compatibility (will be deprecated)
from .metrics_logger import log_counter, log_histogram, log_resource_usage, timeit

# Metrics management
from .metrics_manager import (
    MetricDefinition,
    MetricsRegistry,
    MetricType,
    MetricValue,
    get_metrics_registry,
    increment_counter,
    observe_histogram,
    record_metric,
    set_gauge,
    time_operation,
)
from .telemetry import (
    OTEL_AVAILABLE,
    TelemetryConfig,
    TelemetryManager,
    get_telemetry_manager,
    initialize_telemetry,
    instrument_fastapi_app,
    shutdown_telemetry,
)

# Distributed tracing
from .traces import (
    TraceContext,
    TracingManager,
    add_span_event,
    get_tracing_manager,
    record_span_exception,
    set_span_attribute,
    set_span_attributes,
    set_span_status,
    start_async_span,
    start_span,
    trace_method,
    trace_operation,
)

__all__ = [
    # Telemetry
    "TelemetryConfig",
    "TelemetryManager",
    "get_telemetry_manager",
    "initialize_telemetry",
    "shutdown_telemetry",
    "instrument_fastapi_app",
    "OTEL_AVAILABLE",

    # Metrics
    "MetricType",
    "MetricDefinition",
    "MetricValue",
    "MetricsRegistry",
    "get_metrics_registry",
    "record_metric",
    "increment_counter",
    "set_gauge",
    "observe_histogram",
    "time_operation",

    # Tracing
    "TraceContext",
    "TracingManager",
    "get_tracing_manager",
    "trace_operation",
    "trace_method",
    "start_span",
    "start_async_span",
    "add_span_event",
    "set_span_attribute",
    "set_span_attributes",
    "record_span_exception",
    "set_span_status",

    # Decorators
    "track_metrics",
    "measure_latency",
    "count_calls",
    "track_errors",
    "monitor_resource",
    "track_llm_usage",
    "cache_metrics",

    # Legacy
    "log_counter",
    "log_histogram",
    "timeit",
    "log_resource_usage"
]

# Version information
__version__ = "2.0.0"

# core.http_client is foundation and cannot import this package, so its metrics and
# tracing are wired in from here (TASK-13950.12).
from .http_client_metrics import install_http_client_observability  # noqa: E402

install_http_client_observability()

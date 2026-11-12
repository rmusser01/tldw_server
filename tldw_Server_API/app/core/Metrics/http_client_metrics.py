"""Metrics and tracing for core.http_client, installed when core.Metrics is imported (TASK-13950.12).

core.http_client is foundation and cannot import core.Metrics, so core.Metrics injects its
registry and tracing-manager getters and registers the http_client metric families here.
"""

from __future__ import annotations

from contextlib import suppress

from tldw_Server_API.app.core import http_client

from .metrics_manager import MetricDefinition, MetricsRegistry, MetricType, get_metrics_registry
from .traces import get_tracing_manager

_HTTP_CLIENT_METRIC_REGISTRATION_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError)


def register_http_client_metrics(registry: MetricsRegistry) -> None:
    """Register the four http_client metric families; a failing registration is skipped."""
    with suppress(*_HTTP_CLIENT_METRIC_REGISTRATION_ERRORS):
        registry.register_metric(
            MetricDefinition(
                name="http_client_requests_total",
                type=MetricType.COUNTER,
                description="Total number of outbound HTTP client requests",
                labels=["method", "host", "status"],
            )
        )
    with suppress(*_HTTP_CLIENT_METRIC_REGISTRATION_ERRORS):
        registry.register_metric(
            MetricDefinition(
                name="http_client_request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="Outbound HTTP client request duration (seconds)",
                unit="s",
                labels=["method", "host"],
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
            )
        )
    with suppress(*_HTTP_CLIENT_METRIC_REGISTRATION_ERRORS):
        registry.register_metric(
            MetricDefinition(
                name="http_client_retries_total",
                type=MetricType.COUNTER,
                description="Total retries attempted by HTTP client",
                labels=["reason"],
            )
        )
    with suppress(*_HTTP_CLIENT_METRIC_REGISTRATION_ERRORS):
        registry.register_metric(
            MetricDefinition(
                name="http_client_egress_denials_total",
                type=MetricType.COUNTER,
                description="Total egress policy denials for outbound HTTP",
                labels=["reason"],
            )
        )


def install_http_client_observability() -> None:
    """Install the hooks first, then register the families; a broken registry never fails the import."""
    http_client.install_observability(metrics_registry=get_metrics_registry, tracing_manager=get_tracing_manager)
    registry = None
    with suppress(*_HTTP_CLIENT_METRIC_REGISTRATION_ERRORS):
        registry = get_metrics_registry()
    if registry is not None:
        register_http_client_metrics(registry)

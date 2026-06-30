"""
Watchlists telemetry metrics registration and helpers.

Importing this module is safe in runtime and tests; failures are non-fatal.
"""
from __future__ import annotations

from loguru import logger

from tldw_Server_API.app.core.Metrics import MetricDefinition, MetricType, get_metrics_registry

_REGISTERED = False


def register_watchlists_telemetry_metrics() -> None:
    """Register watchlists telemetry metrics once per process."""
    global _REGISTERED
    if _REGISTERED:
        return
    reg = get_metrics_registry()
    reg.register_metric(
        MetricDefinition(
            name="watchlists_onboarding_telemetry_ingest_total",
            type=MetricType.COUNTER,
            description="Watchlists onboarding telemetry ingest attempts",
            labels=["result"],
        )
    )
    reg.register_metric(
        MetricDefinition(
            name="watchlists_onboarding_telemetry_summary_requests_total",
            type=MetricType.COUNTER,
            description="Watchlists onboarding/RC telemetry summary requests",
            labels=["endpoint", "status"],
        )
    )
    reg.register_metric(
        MetricDefinition(
            name="watchlists_onboarding_telemetry_summary_duration_seconds",
            type=MetricType.HISTOGRAM,
            description="Watchlists onboarding/RC telemetry summary request duration",
            unit="s",
            labels=["endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
        )
    )
    _REGISTERED = True


def record_onboarding_ingest_result(result: str) -> None:
    """Increment onboarding ingest attempt counters grouped by result label."""
    try:
        register_watchlists_telemetry_metrics()
        get_metrics_registry().increment(
            "watchlists_onboarding_telemetry_ingest_total",
            1,
            labels={"result": str(result or "unknown")},
        )
    except Exception:  # noqa: BLE001
        logger.debug("watchlists telemetry metrics ingest emit skipped")


def record_summary_request(endpoint: str, status: str, duration_seconds: float) -> None:
    """Record summary request counters and duration histogram samples."""
    try:
        register_watchlists_telemetry_metrics()
        reg = get_metrics_registry()
        labels = {
            "endpoint": str(endpoint or "unknown"),
            "status": str(status or "unknown"),
        }
        reg.increment("watchlists_onboarding_telemetry_summary_requests_total", 1, labels=labels)
        reg.observe(
            "watchlists_onboarding_telemetry_summary_duration_seconds",
            max(0.0, float(duration_seconds)),
            labels={"endpoint": str(endpoint or "unknown")},
        )
    except Exception:  # noqa: BLE001
        logger.debug("watchlists telemetry metrics summary emit skipped")


try:
    register_watchlists_telemetry_metrics()
except Exception:  # noqa: BLE001
    logger.debug("watchlists telemetry metrics registration skipped")

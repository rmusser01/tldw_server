# metrics_logger.py
#
# Imports
import functools
import time
from datetime import datetime, timezone

import psutil

#
# Third-party Imports
#
# Local Imports
# Avoid importing Utils to prevent circular deps (Utils imports http_client in some paths).
from loguru import logger

from tldw_Server_API.app.core.Metrics.metrics_manager import (
    MetricDefinition,
    MetricType,
    get_metrics_registry,
)

#
############################################################################################################
#
# Functions:

def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp with a single Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _normalize_labels_once(
    metric_name: str, labels: dict[str, str] | None
) -> dict[str, str] | None:
    """Normalize labels once and return them, or ``None`` if they collide."""
    try:
        registry = get_metrics_registry()
        return registry.normalize_labels(labels, reject_collisions=True)
    except ValueError as exc:
        logger.warning(
            "metrics_logger: rejecting {} due to conflicting normalized labels: {}",
            metric_name,
            exc,
        )
        return None


def _bridge_to_registry(
    metric_name: str,
    metric_type: MetricType,
    value: float,
    normalized_labels: dict[str, str],
) -> None:
    """Best-effort bridge from log-based metrics to the in-process registry.

    Expects *already-normalized* labels so the recording hot path can skip
    redundant normalization.
    """
    try:
        registry = get_metrics_registry()
        normalized_name = registry.normalize_metric_name(metric_name)
        if normalized_name not in registry.metrics:
            registry.register_metric(
                MetricDefinition(
                    name=normalized_name,
                    type=metric_type,
                    description=f"Auto-bridged metric for {metric_name}",
                    labels=list(normalized_labels.keys()),
                ),
                persistent=False,
            )
        registry.record(metric_name, value, normalized_labels, _normalized=True)
    except Exception as exc:
        logger.debug("metrics_logger: registry bridge failed: {err}", err=exc)


def log_counter(metric_name: str, labels: dict[str, str] | None = None, value: float = 1) -> None:
    normalized = _normalize_labels_once(metric_name, labels)
    if normalized is None:
        return
    log_entry = {
        "event": metric_name,
        "type": "counter",
        "value": value,
        "labels": normalized,
        "timestamp": _utc_timestamp(),
    }
    logger.bind(**log_entry).info("metric")
    _bridge_to_registry(metric_name, MetricType.COUNTER, value, normalized)


def log_histogram(metric_name: str, value: float, labels: dict[str, str] | None = None) -> None:
    normalized = _normalize_labels_once(metric_name, labels)
    if normalized is None:
        return
    log_entry = {
        "event": metric_name,
        "type": "histogram",
        "value": value,
        "labels": normalized,
        "timestamp": _utc_timestamp(),
    }
    logger.bind(**log_entry).info("metric")
    _bridge_to_registry(metric_name, MetricType.HISTOGRAM, value, normalized)


def log_gauge(metric_name: str, value: float, labels: dict[str, str] | None = None) -> None:
    """Log an instantaneous measurement (gauge).

    The current metrics backend is log-based, so we simply emit a structured
    log with type 'gauge'. This is non-breaking and allows callers to express
    semantics more clearly. Downstream exporters can map these to Prometheus
    Gauges or equivalent.
    """
    normalized = _normalize_labels_once(metric_name, labels)
    if normalized is None:
        return
    log_entry = {
        "event": metric_name,
        "type": "gauge",
        "value": value,
        "labels": normalized,
        "timestamp": _utc_timestamp(),
    }
    logger.bind(**log_entry).info("metric")
    _bridge_to_registry(metric_name, MetricType.GAUGE, value, normalized)


def timeit(func):
    """
    Decorator that times the execution of the wrapped function
    and logs the result using log_histogram. Optionally, you could also
    log a counter each time the function is called.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.monotonic()
        result = func(*args, **kwargs)
        elapsed = time.monotonic() - start

        # Print to console (optional)
        print(f"{func.__name__} executed in {elapsed:.2f} seconds.")

        # Log how long the function took (histogram)
        log_histogram(
            metric_name=f"{func.__name__}_duration_seconds",
            value=elapsed,
            labels={"function": func.__name__}
        )

        # (Optional) log how many times the function has been called
        log_counter(
            metric_name=f"{func.__name__}_calls",
            labels={"function": func.__name__}
        )

        return result
    return wrapper
    # Add '@timeit' decorator to functions you want to time


def log_resource_usage():
    process = psutil.Process()
    memory = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    cpu = process.cpu_percent(interval=0.1)
    print(f"Memory: {memory:.2f} MB, CPU: {cpu:.2f}%")

#
# End of Functions
############################################################################################################

# # Prometheus
# # metrics_logger.py (Prometheus version)
# from prometheus_client import Counter, Histogram, start_http_server
# import logging
# from functools import wraps
# import time
#
# # Initialize Prometheus metrics
# VIDEOS_PROCESSED = Counter('videos_processed_total', 'Total number of videos processed', ['whisper_model', 'api_name'])
# VIDEOS_FAILED = Counter('videos_failed_total', 'Total number of videos failed to process', ['whisper_model', 'api_name'])
# TRANSCRIPTIONS_GENERATED = Counter('transcriptions_generated_total', 'Total number of transcriptions generated', ['whisper_model'])
# SUMMARIES_GENERATED = Counter('summaries_generated_total', 'Total number of summaries generated', ['whisper_model'])
# VIDEO_PROCESSING_TIME = Histogram('video_processing_time_seconds', 'Time spent processing videos', ['whisper_model', 'api_name'])
# TOTAL_PROCESSING_TIME = Histogram('total_processing_time_seconds', 'Total time spent processing all videos', ['whisper_model', 'api_name'])
#
# def init_metrics_server(port=8000):
#     start_http_server(port)
#
# def log_counter(metric_name, labels=None, value=1):
#     if metric_name == "videos_processed_total":
#         VIDEOS_PROCESSED.labels(**(labels or {})).inc(value)
#     elif metric_name == "videos_failed_total":
#         VIDEOS_FAILED.labels(**(labels or {})).inc(value)
#     elif metric_name == "transcriptions_generated_total":
#         TRANSCRIPTIONS_GENERATED.labels(**(labels or {})).inc(value)
#     elif metric_name == "summaries_generated_total":
#         SUMMARIES_GENERATED.labels(**(labels or {})).inc(value)
#
# def log_histogram(metric_name, value, labels=None):
#     if metric_name == "video_processing_time_seconds":
#         VIDEO_PROCESSING_TIME.labels(**(labels or {})).observe(value)
#     elif metric_name == "total_processing_time_seconds":
#         TOTAL_PROCESSING_TIME.labels(**(labels or {})).observe(value)


# # main.py or equivalent entry point
# from metrics_logger import init_metrics_server
#
#
# def main():
#     # Start Prometheus metrics server on port 8000
#     init_metrics_server(port=8000)
#
#     # Initialize and launch your Gradio app
#     create_video_transcription_tab()
#
#
# if __name__ == "__main__":
#     main()

# prometheus.yml
# scrape_configs:
#   - job_name: 'video_transcription_app'
#     static_configs:
#       - targets: ['localhost:8000']  # Replace with your application's host and port

#
# End of metrics_logger.py
############################################################################################################

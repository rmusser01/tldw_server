import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.Metrics import cache_metrics, measure_latency, monitor_resource
import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager

pytestmark = pytest.mark.monitoring


def test_measure_latency_exports_default_buckets(monkeypatch):


    monkeypatch.setenv("METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED", "50")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        @measure_latency(metric_name="test_latency_seconds")
        def sample():
            return "ok"

        sample()
        text = registry.export_prometheus_format()

        assert "test_latency_seconds_bucket" in text
        assert 'test_latency_seconds_bucket{le="0.1"}' in text
    finally:
        metrics_manager._metrics_registry = None


def test_cache_metrics_preserves_tuple_return(monkeypatch):
    monkeypatch.setenv("METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED", "20")
    metrics_manager._metrics_registry = None
    metrics_manager.get_metrics_registry()

    try:
        @cache_metrics("tuple_cache", track_ratio=False)
        def fetch_tuple():
            return ("payload", True)

        assert fetch_tuple() == ("payload", True)
    finally:
        metrics_manager._metrics_registry = None


def test_cache_hit_ratio_ignores_tuple_second_value_without_from_cache(monkeypatch):
    monkeypatch.setenv("METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED", "20")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        @cache_metrics("demo_cache", track_ratio=True)
        def fetch(cache_hit: bool):
            return ("payload", cache_hit)

        fetch(False)
        fetch(True)
        fetch(True)
        fetch(True)

        stats = registry.get_metric_stats("cache_hit_ratio", {"cache": "demo_cache"})
        assert stats["latest"] == pytest.approx(0.0)
    finally:
        metrics_manager._metrics_registry = None


def test_monitor_resource_thread_safe_active_gauge(monkeypatch):
    monkeypatch.setenv("METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED", "1000")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        resource_name = "threadsafe_resource"
        base_metric = "monitor_resource_threadsafe_test"
        count_metric = f"{base_metric}_active"
        workers = 16
        launch_barrier = threading.Barrier(workers)
        entered_barrier = threading.Barrier(workers)

        @monitor_resource(resource_name=resource_name, metric_name=base_metric, track_count=True, track_usage=False)
        def guarded_op():
            entered_barrier.wait(timeout=5)
            time.sleep(0.01)
            return "ok"

        def run_once():
            launch_barrier.wait(timeout=5)
            return guarded_op()

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = [future.result() for future in [pool.submit(run_once) for _ in range(workers)]]

        assert all(result == "ok" for result in results)
        stats = registry.get_metric_stats(count_metric, {"resource": resource_name})
        assert stats["latest"] == 0
        assert stats["min"] >= 0
        assert stats["max"] == workers
    finally:
        metrics_manager._metrics_registry = None

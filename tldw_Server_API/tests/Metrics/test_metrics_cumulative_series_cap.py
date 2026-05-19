import pytest

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager
from tldw_Server_API.app.core.Metrics.metrics_manager import MetricDefinition, MetricType


pytestmark = pytest.mark.unit


def test_cumulative_counter_series_cap_drops_new_label_sets(monkeypatch):
    monkeypatch.setenv("METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC", "1")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="cap_counter_total",
                type=MetricType.COUNTER,
                description="counter cap test",
                labels=["series"],
            )
        )

        registry.increment("cap_counter_total", labels={"series": "a"})
        registry.increment("cap_counter_total", labels={"series": "b"})  # dropped
        registry.increment("cap_counter_total", labels={"series": "a"})

        assert registry.get_cumulative_counter("cap_counter_total", {"series": "a"}) == 2
        assert registry.get_cumulative_counter("cap_counter_total", {"series": "b"}) == 0
        assert registry.get_metric_stats("cap_counter_total", labels={"series": "b"}) == {}

        text = registry.export_prometheus_format()
        assert 'cap_counter_total{series="a"}' in text
        assert 'cap_counter_total{series="b"}' not in text
        assert registry.get_all_metrics()["cap_counter_total"]["stats"]["count"] == 2
    finally:
        metrics_manager._metrics_registry = None


def test_cumulative_histogram_series_cap_drops_new_label_sets(monkeypatch):
    monkeypatch.setenv("METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC", "1")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="cap_hist_seconds",
                type=MetricType.HISTOGRAM,
                description="hist cap test",
                labels=["series"],
                buckets=[0.1, 1.0],
            )
        )

        registry.observe("cap_hist_seconds", 0.2, labels={"series": "a"})
        registry.observe("cap_hist_seconds", 0.3, labels={"series": "b"})  # dropped
        registry.observe("cap_hist_seconds", 0.4, labels={"series": "a"})

        text = registry.export_prometheus_format()
        assert registry.get_metric_stats("cap_hist_seconds", labels={"series": "b"}) == {}
        assert 'cap_hist_seconds_count{series="a"} 2' in text
        assert 'cap_hist_seconds_count{series="b"}' not in text
        assert registry.get_all_metrics()["cap_hist_seconds"]["stats"]["count"] == 2
    finally:
        metrics_manager._metrics_registry = None


def test_cumulative_series_cap_still_dispatches_callbacks_and_instruments(monkeypatch):
    monkeypatch.setenv("METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC", "1")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    class DummyInstrument:
        def __init__(self):
            self.calls = []

        def add(self, value, attributes=None):
            self.calls.append((value, dict(attributes or {})))

    callback_calls = []

    def callback(metric_name, value, labels):
        callback_calls.append((metric_name, value, dict(labels)))

    try:
        registry.register_metric(
            MetricDefinition(
                name="cap_dispatch_total",
                type=MetricType.COUNTER,
                description="counter cap dispatch test",
                labels=["series"],
            )
        )
        registry.instruments["cap_dispatch_total"] = DummyInstrument()
        registry.add_callback("cap_dispatch_total", callback)

        registry.increment("cap_dispatch_total", labels={"series": "a"})
        registry.increment("cap_dispatch_total", labels={"series": "b"})  # capped, but still dispatched

        assert registry.get_cumulative_counter_total("cap_dispatch_total") == 1
        assert registry.get_metric_stats("cap_dispatch_total", labels={"series": "b"}) == {}
        assert registry.get_all_metrics()["cap_dispatch_total"]["stats"]["count"] == 1
        assert callback_calls == [
            ("cap_dispatch_total", 1, {"series": "a"}),
            ("cap_dispatch_total", 1, {"series": "b"}),
        ]
        assert registry.instruments["cap_dispatch_total"].calls == [
            (1, {"series": "a"}),
            (1, {"series": "b"}),
        ]
    finally:
        metrics_manager._metrics_registry = None


def test_cumulative_counter_series_cap_does_not_leak_into_stats(monkeypatch):
    monkeypatch.setenv("METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC", "1")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="cap_counter_stats_total",
                type=MetricType.COUNTER,
                description="counter cap stats test",
                labels=["series"],
            )
        )

        registry.increment("cap_counter_stats_total", labels={"series": "a"})
        registry.increment("cap_counter_stats_total", labels={"series": "b"})  # dropped
        registry.increment("cap_counter_stats_total", labels={"series": "a"})

        stats = registry.get_metric_stats("cap_counter_stats_total")
        assert stats.get("count") == 2
        assert registry.get_cumulative_counter("cap_counter_stats_total", {"series": "a"}) == 2
        assert registry.get_cumulative_counter("cap_counter_stats_total", {"series": "b"}) == 0

        metrics = registry.get_all_metrics()
        assert metrics["cap_counter_stats_total"]["stats"]["count"] == 2
    finally:
        metrics_manager._metrics_registry = None


def test_cumulative_histogram_series_cap_does_not_leak_into_stats(monkeypatch):
    monkeypatch.setenv("METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC", "1")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="cap_hist_stats_seconds",
                type=MetricType.HISTOGRAM,
                description="hist cap stats test",
                labels=["series"],
                buckets=[0.1, 1.0],
            )
        )

        registry.observe("cap_hist_stats_seconds", 0.2, labels={"series": "a"})
        registry.observe("cap_hist_stats_seconds", 0.3, labels={"series": "b"})  # dropped
        registry.observe("cap_hist_stats_seconds", 0.4, labels={"series": "a"})

        stats = registry.get_metric_stats("cap_hist_stats_seconds")
        assert stats.get("count") == 2
        metrics = registry.get_all_metrics()
        assert metrics["cap_hist_stats_seconds"]["stats"]["count"] == 2

        text = registry.export_prometheus_format()
        assert 'cap_hist_stats_seconds_count{series="a"} 2' in text
        assert 'cap_hist_stats_seconds_count{series="b"}' not in text
    finally:
        metrics_manager._metrics_registry = None


def test_audio_stt_requests_bucket_unknown_models_into_single_series(monkeypatch):
    monkeypatch.setenv("METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC", "300")
    metrics_manager._metrics_registry = None

    from tldw_Server_API.app.core.Metrics.stt_metrics import emit_stt_request_total

    registry = metrics_manager.get_metrics_registry()

    try:
        for idx in range(400):
            emit_stt_request_total(
                endpoint="audio.transcriptions",
                provider="custom-provider",
                model=f"custom-model-{idx}",
                status="ok",
            )

        assert registry.get_cumulative_counter_total("audio_stt_requests_total") == 400
        assert registry.get_cumulative_counter_totals_by_label("audio_stt_requests_total", "model") == {"other": 400.0}
        assert registry.get_cumulative_counter_totals_by_label("audio_stt_requests_total", "provider") == {
            "other": 400.0
        }
        assert registry._cumulative_series_dropped.get("audio_stt_requests_total", 0) == 0  # nosec B101
    finally:
        metrics_manager._metrics_registry = None

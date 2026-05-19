import pytest

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager
from tldw_Server_API.app.core.Metrics import metrics_logger
from tldw_Server_API.app.core.Metrics.metrics_manager import (
    MetricDefinition,
    MetricType,
    get_metrics_registry,
)


@pytest.fixture(autouse=True)
def reset_global_registry():
    metrics_manager._metrics_registry = None
    try:
        yield
    finally:
        metrics_manager._metrics_registry = None


@pytest.mark.unit
def test_metrics_logger_bridge_records_counter():
    registry = get_metrics_registry()
    registry.reset()

    metric_name = "bridge_counter_test_total"
    metrics_logger.log_counter(metric_name, labels={"source": "test"}, value=2)

    assert registry.get_cumulative_counter(metric_name, {"source": "test"}) == 2


@pytest.mark.unit
def test_metrics_logger_bridge_records_histogram():
    registry = get_metrics_registry()
    registry.reset()

    metric_name = "bridge_histogram_test_seconds"
    metrics_logger.log_histogram(metric_name, 0.5, labels={"source": "test"})

    stats = registry.get_metric_stats(metric_name, labels={"source": "test"})
    assert stats.get("count") == 1
    assert stats.get("latest") == 0.5


@pytest.mark.unit
def test_metrics_registry_sample_counts_use_cumulative_histogram_series():
    registry = get_metrics_registry()
    registry.reset()

    metric_name = "bridge_histogram_sample_count_test_seconds"
    metrics_logger.log_histogram(metric_name, 0.5, labels={"assistant_kind": "persona"})
    metrics_logger.log_histogram(metric_name, 0.7, labels={"assistant_kind": "persona"})
    metrics_logger.log_histogram(metric_name, 0.3, labels={"assistant_kind": "character"})

    normalized_name = registry.normalize_metric_name(metric_name)
    registry.values[normalized_name].clear()
    registry.values[metric_name].clear()

    assert registry.get_metric_sample_counts_by_label(
        metric_name,
        "assistant_kind",
        missing_label="none",
    ) == {"character": 1, "persona": 2}


@pytest.mark.unit
def test_metrics_logger_bridge_records_gauge():
    registry = get_metrics_registry()
    registry.reset()

    metric_name = "bridge_gauge_test"
    metrics_logger.log_gauge(metric_name, 3.14, labels={"source": "test"})

    stats = registry.get_metric_stats(metric_name, labels={"source": "test"})
    assert stats.get("count") == 1
    assert stats.get("latest") == 3.14


@pytest.mark.unit
def test_metrics_logger_bridge_rejects_conflicting_labels_before_registration(monkeypatch):
    registry = get_metrics_registry()
    registry.reset()

    metric_name = "bridge collision total"
    normalized_name = registry.normalize_metric_name(metric_name)
    original_register_metric = registry.register_metric
    called = {"value": False}

    def spy_register_metric(definition, persistent=True):
        called["value"] = True
        return original_register_metric(definition, persistent=persistent)

    logged_entries = []

    class DummyBoundLogger:
        def info(self, message):
            logged_entries.append(message)

    def fake_bind(**kwargs):
        logged_entries.append(kwargs)
        return DummyBoundLogger()

    monkeypatch.setattr(metrics_logger.logger, "bind", fake_bind)

    try:
        registry.register_metric = spy_register_metric  # type: ignore[method-assign]
        metrics_logger.log_counter(metric_name, labels={"x-y": "left", "x_y": "right"}, value=1)

        assert called["value"] is False
        assert logged_entries == []
        assert normalized_name not in registry.metrics
        assert registry.get_all_metrics() == {}
        assert registry.get_cumulative_counter_total(normalized_name) == 0
    finally:
        registry.register_metric = original_register_metric  # type: ignore[method-assign]


@pytest.mark.unit
def test_metrics_logger_bridge_re_registers_metric_definition_after_reset():
    registry = get_metrics_registry()
    registry.reset()

    metric_name = "bridge reset metric total"
    normalized_name = registry.normalize_metric_name(metric_name)
    metrics_logger.log_histogram(metric_name, 0.5, labels={"source": "test"})

    assert normalized_name in registry.metrics
    assert registry.metrics[normalized_name].type == MetricType.HISTOGRAM

    registry.reset()

    assert normalized_name not in registry.metrics
    assert "http_requests_total" in registry.metrics

    metrics_logger.log_counter(metric_name, labels={"source": "test"}, value=2)

    assert normalized_name in registry.metrics
    assert registry.metrics[normalized_name].type == MetricType.COUNTER
    assert registry.get_cumulative_counter(normalized_name, {"source": "test"}) == 2


@pytest.mark.unit
def test_metrics_registry_reset_preserves_custom_metric_definitions_and_records():
    registry = get_metrics_registry()
    registry.reset()

    metric_name = "workflow custom metric total"
    normalized_name = registry.normalize_metric_name(metric_name)
    registry.register_metric(
        MetricDefinition(
            name=metric_name,
            type=MetricType.COUNTER,
            description="workflow custom test",
            labels=["source"],
        )
    )

    assert normalized_name in registry.metrics
    assert registry.metrics[normalized_name].type == MetricType.COUNTER

    registry.increment(metric_name, labels={"source": "test"})
    assert registry.get_cumulative_counter(normalized_name, {"source": "test"}) == 1

    registry.reset()

    assert normalized_name in registry.metrics
    assert registry.metrics[normalized_name].type == MetricType.COUNTER
    assert registry.get_metric_stats(metric_name) == {}
    registry.increment(metric_name, labels={"source": "test"})
    assert registry.get_cumulative_counter(normalized_name, {"source": "test"}) == 1


@pytest.mark.unit
def test_metrics_registry_reset_does_not_duplicate_histogram_views():
    registry = get_metrics_registry()
    registry.reset()

    registry.register_metric(
        MetricDefinition(
            name="view_reset_histogram_seconds",
            type=MetricType.HISTOGRAM,
            description="view replay test",
            buckets=[0.1, 1.0],
        )
    )

    pending_before = len(getattr(registry.telemetry, "_pending_views", []))
    registry.reset()
    pending_after_first_reset = len(getattr(registry.telemetry, "_pending_views", []))
    registry.reset()
    pending_after_second_reset = len(getattr(registry.telemetry, "_pending_views", []))

    assert pending_after_first_reset >= pending_before
    assert pending_after_second_reset == pending_after_first_reset


@pytest.mark.unit
def test_metrics_logger_bridge_preserves_raw_name_alias_in_registry_values():
    registry = get_metrics_registry()
    registry.reset()

    metric_name = "bridge alias metric total"
    normalized_name = registry.normalize_metric_name(metric_name)
    metrics_logger.log_counter(metric_name, labels={"source": "test"}, value=1)

    assert metric_name in registry.values
    assert normalized_name in registry.values

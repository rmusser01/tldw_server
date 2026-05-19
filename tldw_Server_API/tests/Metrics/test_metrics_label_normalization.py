import pytest

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager
from tldw_Server_API.app.core.Metrics.metrics_manager import MetricDefinition, MetricType


pytestmark = pytest.mark.unit


def test_record_rejects_conflicting_labels_after_normalization():
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        metric_name = "label_collision_total"
        registry.register_metric(
            MetricDefinition(
                name=metric_name,
                type=MetricType.COUNTER,
                description="label collision test",
                labels=["x_y"],
            )
        )

        registry.record(metric_name, 1, labels={"x-y": "left", "x_y": "right"})

        assert registry.get_metric_stats(metric_name) == {}
        assert registry.get_cumulative_counter_total(metric_name) == 0
        assert registry.get_all_metrics() == {}
        assert registry.export_prometheus_format().strip() == ""
    finally:
        metrics_manager._metrics_registry = None

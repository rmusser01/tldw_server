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


def test_user_id_labels_are_hashed_before_public_export() -> None:
    """Hash legacy user_id labels before exporting public metrics."""
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="privacy_upload_total",
                type=MetricType.COUNTER,
                description="privacy label test",
                labels=["user_hash", "media_type"],
            )
        )

        registry.increment(
            "privacy_upload_total",
            labels={"user_id": "alice@example.com", "media_type": "pdf"},
        )

        text = registry.export_prometheus_format()
        assert "alice@example.com" not in text
        assert "user_id" not in text
        assert 'user_hash="' in text
    finally:
        metrics_manager._metrics_registry = None


def test_standard_metric_definitions_do_not_use_raw_user_id_labels() -> None:
    """Keep standard metric definitions on the public user_hash label."""
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        offenders = {
            name: definition.labels
            for name, definition in registry.metrics.items()
            if "user_id" in definition.labels
        }

        assert offenders == {}
    finally:
        metrics_manager._metrics_registry = None


def test_counter_rejects_negative_increment() -> None:
    """Ignore negative increments for monotonic counter metrics."""
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="non_negative_counter_total",
                type=MetricType.COUNTER,
                description="counter non-negative test",
                labels=["source"],
            )
        )

        registry.increment("non_negative_counter_total", value=5, labels={"source": "test"})
        registry.increment("non_negative_counter_total", value=-2, labels={"source": "test"})

        assert registry.get_cumulative_counter("non_negative_counter_total", {"source": "test"}) == 5
    finally:
        metrics_manager._metrics_registry = None


def test_user_id_label_name_aliases_counter_aggregation_lookup() -> None:
    """Allow legacy user_id lookup names for hashed counter label aggregates."""
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="privacy_lookup_total",
                type=MetricType.COUNTER,
                description="privacy lookup test",
                labels=["user_hash"],
            )
        )

        registry.increment("privacy_lookup_total", labels={"user_id": "alice@example.com"})
        registry.increment("privacy_lookup_total", labels={"user_id": "alice@example.com"})

        by_user_id = registry.get_cumulative_counter_totals_by_label("privacy_lookup_total", "user_id")
        by_user_hash = registry.get_cumulative_counter_totals_by_label("privacy_lookup_total", "user_hash")

        assert by_user_id == by_user_hash
        assert list(by_user_id.values()) == [2.0]
        assert all("alice@example.com" not in label_value for label_value in by_user_id)
    finally:
        metrics_manager._metrics_registry = None


def test_user_id_label_name_aliases_histogram_sample_count_lookup() -> None:
    """Allow legacy user_id lookup names for hashed histogram label aggregates."""
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="privacy_lookup_seconds",
                type=MetricType.HISTOGRAM,
                description="privacy histogram lookup test",
                labels=["user_hash"],
                buckets=[1.0],
            )
        )

        registry.observe("privacy_lookup_seconds", 0.2, labels={"user_id": "alice@example.com"})
        registry.observe("privacy_lookup_seconds", 0.4, labels={"user_id": "alice@example.com"})

        by_user_id = registry.get_metric_sample_counts_by_label(
            "privacy_lookup_seconds",
            "user_id",
            missing_label="none",
        )
        by_user_hash = registry.get_metric_sample_counts_by_label(
            "privacy_lookup_seconds",
            "user_hash",
            missing_label="none",
        )

        assert by_user_id == by_user_hash
        assert list(by_user_id.values()) == [2]
        assert all("alice@example.com" not in label_value for label_value in by_user_id)
    finally:
        metrics_manager._metrics_registry = None

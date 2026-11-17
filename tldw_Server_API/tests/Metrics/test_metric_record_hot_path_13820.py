"""Recording a validated metric must not repeat label sanitization per frame."""

import pytest

from tldw_Server_API.app.core.Metrics.metrics_manager import MetricDefinition, MetricsRegistry, MetricType


@pytest.mark.unit
@pytest.mark.parametrize("pre_normalized", [False, True])
def test_record_normalizes_once_and_preserves_canonical_histogram_labels(monkeypatch, pre_normalized):
    registry = MetricsRegistry()
    name = "record_hot_path_ms"
    registry.register_metric(MetricDefinition(name=name, type=MetricType.HISTOGRAM, description="Hot-path regression"))
    raw = {"user_id": "private-user", "event-kind": "history", "size": 500}
    canonical = registry.normalize_labels(raw, reject_collisions=True)
    normalize = MetricsRegistry._normalize_labels
    calls = []

    def counted(cls, labels, reject_collisions=False):
        calls.append(labels)
        return normalize(labels, reject_collisions=reject_collisions)

    monkeypatch.setattr(MetricsRegistry, "_normalize_labels", classmethod(counted))
    registry.record(name, 2.5, canonical if pre_normalized else raw, _normalized=pre_normalized)

    assert len(calls) == (0 if pre_normalized else 1)
    assert registry.values[name][-1].labels == canonical
    assert tuple(sorted(canonical.items())) in registry._cumulative_histograms[name]

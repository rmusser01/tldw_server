import pytest

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager
from tldw_Server_API.app.core.Resource_Governance.metrics_rg import (
    ensure_rg_metrics_registered,
    record_shadow_mismatch,
)
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry


pytestmark = pytest.mark.rate_limit


def test_rg_metrics_registration_recovers_after_registry_replacement(monkeypatch):
    ensure_rg_metrics_registered()
    original_registry = get_metrics_registry()
    assert "rg_decisions_total" in original_registry.metrics

    monkeypatch.setattr(metrics_manager, "_metrics_registry", None)

    ensure_rg_metrics_registered()

    replacement_registry = get_metrics_registry()
    assert replacement_registry is not original_registry
    assert "rg_decisions_total" in replacement_registry.metrics
    assert "rg_denials_total" in replacement_registry.metrics


def test_record_shadow_mismatch_increments_counter():


    ensure_rg_metrics_registered()
    reg = get_metrics_registry()

    labels = {
        "module": "chat",
        "route": "/api/v1/chat/completions",
        "policy_id": "chat.default",
        "legacy": "allow",
        "rg": "deny",
    }

    before = reg.get_metric_stats("rg_shadow_decision_mismatch_total", labels=labels) or {}
    before_count = int(before.get("count", 0) or 0)

    record_shadow_mismatch(**labels)

    after = reg.get_metric_stats("rg_shadow_decision_mismatch_total", labels=labels) or {}
    after_count = int(after.get("count", 0) or 0)

    assert after_count == before_count + 1

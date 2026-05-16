"""Unit tests for persona telemetry metrics aggregation in evaluations surfaces."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as unified_service_module
from tldw_Server_API.app.core.Evaluations.persona_telemetry_metrics import (
    get_persona_telemetry_metrics_summary,
)
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.tests.Persona.persona_chat_quality_cases import case_by_id


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_metrics_registry() -> None:
    registry = get_metrics_registry()
    registry.reset()
    yield
    registry.reset()


def _base_labels() -> dict[str, str]:
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "user_id": "1",
        "character_id": "10",
    }


def test_persona_telemetry_metrics_summary_defaults_to_zeroes_when_empty():
    summary = get_persona_telemetry_metrics_summary()

    assert summary["samples"] == 0
    assert summary["ioo"]["count"] == 0
    assert summary["ioo"]["mean"] == 0.0
    assert summary["ior"]["count"] == 0
    assert summary["lcs"]["count"] == 0
    assert summary["alerts"]["ioo_threshold_exceeded_total"] == 0
    assert summary["alerts"]["ioo_sustained_alert_total"] == 0
    assert summary["alerts"]["ior_out_of_band_total"] == 0
    assert summary["alerts"]["safety_flag_total"] == 0
    assert summary["ior_out_of_band_by_band"] == {}
    assert summary["safety_flags"] == {}


def test_persona_telemetry_metrics_summary_aggregates_histograms_and_counters():
    labels = _base_labels()

    log_histogram("chat_persona_ioo_ratio", 0.12, labels=labels)
    log_histogram("chat_persona_ioo_ratio", 0.36, labels=labels)
    log_histogram("chat_persona_ior_ratio", 0.25, labels=labels)
    log_histogram("chat_persona_ior_ratio", 0.55, labels=labels)
    log_histogram("chat_persona_lcs_ratio", 0.10, labels=labels)
    log_histogram("chat_persona_lcs_ratio", 0.40, labels=labels)

    log_counter("chat_persona_ioo_threshold_exceeded_total", labels=labels, value=2)
    log_counter("chat_persona_ioo_sustained_alert_total", labels=labels, value=1)
    log_counter("chat_persona_ior_out_of_band_total", labels={**labels, "band": "low"}, value=1)
    log_counter("chat_persona_ior_out_of_band_total", labels={**labels, "band": "high"}, value=3)
    log_counter("chat_persona_safety_flag_total", labels={**labels, "flag": "ioo_high"}, value=2)
    log_counter("chat_persona_safety_flag_total", labels={**labels, "flag": "refusal_detected"}, value=1)

    summary = get_persona_telemetry_metrics_summary()

    assert summary["samples"] == 2
    assert summary["ioo"]["count"] == 2
    assert summary["ioo"]["mean"] == 0.24
    assert summary["ioo"]["min"] == 0.12
    assert summary["ioo"]["max"] == 0.36
    assert summary["ioo"]["latest"] == 0.36

    assert summary["alerts"]["ioo_threshold_exceeded_total"] == 2
    assert summary["alerts"]["ioo_sustained_alert_total"] == 1
    assert summary["alerts"]["ior_out_of_band_total"] == 4
    assert summary["alerts"]["safety_flag_total"] == 3
    assert summary["ior_out_of_band_by_band"] == {"high": 3, "low": 1}
    assert summary["safety_flags"] == {"ioo_high": 2, "refusal_detected": 1}


def test_persona_telemetry_metrics_summary_groups_persona_backed_labels():
    fixture_case = case_by_id("PC-CASE-019")
    assert "PC-TEL-001" in fixture_case["labels"]  # nosec B101

    labels = {
        **_base_labels(),
        "character_id": "none",
        "assistant_kind": fixture_case["assistant_kind"],
        "assistant_id": fixture_case["assistant_id"],
    }
    log_histogram("chat_persona_ioo_ratio", 0.18, labels=labels)

    summary = get_persona_telemetry_metrics_summary()

    assert summary["samples"] == 1
    assert summary["samples_by_assistant_kind"] == {"persona": 1}
    assert summary["samples_by_assistant_id"] == {fixture_case["assistant_id"]: 1}


@pytest.mark.asyncio
async def test_unified_evaluation_service_metrics_summary_includes_persona_telemetry(monkeypatch):
    labels = _base_labels()
    log_histogram("chat_persona_ioo_ratio", 0.2, labels=labels)
    log_counter("chat_persona_ioo_threshold_exceeded_total", labels=labels, value=1)

    class _AdvancedMetricsStub:
        enabled = False

    monkeypatch.setattr(unified_service_module, "advanced_metrics", _AdvancedMetricsStub(), raising=False)

    service = object.__new__(UnifiedEvaluationService)
    summary = await UnifiedEvaluationService.get_metrics_summary(service)

    assert summary["metrics_enabled"] is False
    assert "persona_telemetry" in summary
    assert summary["persona_telemetry"]["samples"] == 1
    assert summary["persona_telemetry"]["alerts"]["ioo_threshold_exceeded_total"] == 1

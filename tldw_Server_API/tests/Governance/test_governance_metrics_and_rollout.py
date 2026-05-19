from __future__ import annotations

import configparser

import pytest

from tldw_Server_API.app.core import config as app_config
from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import MetricsCollector
from tldw_Server_API.tests.run_first_constants import (
    PHASE2C_RUN_FIRST_COHORT,
    PHASE2C_RUN_FIRST_CSV,
)

pytestmark = pytest.mark.unit


def test_rollout_modes_resolve_off_shadow_enforce(monkeypatch):
    monkeypatch.delenv("GOVERNANCE_ROLLOUT_MODE", raising=False)

    assert app_config.resolve_governance_rollout_mode("off") == "off"
    assert app_config.resolve_governance_rollout_mode("shadow") == "shadow"
    assert app_config.resolve_governance_rollout_mode("enforce") == "enforce"
    assert app_config.resolve_governance_rollout_mode("invalid-value") == "off"

    monkeypatch.setenv("GOVERNANCE_ROLLOUT_MODE", "shadow")
    assert app_config.resolve_governance_rollout_mode() == "shadow"


def test_run_first_rollout_resolvers_default_off_without_config(monkeypatch):
    monkeypatch.delenv("ACP_RUN_FIRST_ROLLOUT_MODE", raising=False)
    monkeypatch.delenv("ACP_RUN_FIRST_PROVIDER_ALLOWLIST", raising=False)
    monkeypatch.delenv("ACP_RUN_FIRST_PRESENTATION_VARIANT", raising=False)
    monkeypatch.setattr(app_config, "load_comprehensive_config", lambda: configparser.ConfigParser())

    assert app_config.resolve_acp_run_first_rollout_mode() == "off"
    assert app_config.resolve_acp_run_first_provider_allowlist() == []
    assert app_config.resolve_acp_run_first_presentation_variant() == "acp_phase2b_v1"


def test_run_first_rollout_resolvers_accept_default_on(monkeypatch):
    monkeypatch.setenv("ACP_RUN_FIRST_ROLLOUT_MODE", "default_on")

    assert app_config.resolve_acp_run_first_rollout_mode() == "default_on"


def test_run_first_rollout_provider_allowlist_parses_csv(monkeypatch):
    monkeypatch.setenv(
        "ACP_RUN_FIRST_PROVIDER_ALLOWLIST",
        PHASE2C_RUN_FIRST_CSV,
    )

    assert app_config.resolve_acp_run_first_provider_allowlist() == PHASE2C_RUN_FIRST_COHORT


def test_run_first_rollout_acp_mode_uses_acp_config_section(monkeypatch):
    monkeypatch.delenv("ACP_RUN_FIRST_ROLLOUT_MODE", raising=False)
    monkeypatch.delenv("ACP_RUN_FIRST_PROVIDER_ALLOWLIST", raising=False)
    monkeypatch.delenv("ACP_RUN_FIRST_PRESENTATION_VARIANT", raising=False)

    parser = configparser.ConfigParser()
    parser.add_section("ACP")
    parser.set("ACP", "run_first_rollout_mode", "gated")
    parser.set(
        "ACP",
        "run_first_provider_allowlist",
        PHASE2C_RUN_FIRST_CSV,
    )
    parser.set("ACP", "run_first_presentation_variant", "acp_phase2a_v2")
    monkeypatch.setattr(app_config, "load_comprehensive_config", lambda: parser)

    assert app_config.resolve_acp_run_first_rollout_mode() == "gated"
    assert app_config.resolve_acp_run_first_provider_allowlist() == PHASE2C_RUN_FIRST_COHORT
    assert app_config.resolve_acp_run_first_presentation_variant() == "acp_phase2a_v2"


def test_metrics_use_low_cardinality_labels_only():
    collector = MetricsCollector(enable_prometheus=False)
    collector.record_governance_check(
        surface="mcp_tool:req-12345",
        category="workspace-very-specific-9912",
        status="dynamic-status",
        rollout_mode="experimental",
    )

    internal = collector.get_internal_metrics(period_seconds=300)
    assert "governance_check" in internal

    labels = internal["governance_check"]["labels"][0]["labels"]
    assert labels["surface"] == "other"
    assert labels["category"] == "other"
    assert labels["status"] == "unknown"
    assert labels["rollout_mode"] == "off"


def test_audit_trace_persists_policy_and_rule_revision_refs():
    from tldw_Server_API.app.core.Governance.metrics import GovernanceMetrics

    collector = MetricsCollector(enable_prometheus=False)
    governance_metrics = GovernanceMetrics(metrics_collector=collector)

    trace = governance_metrics.record_check(
        surface="mcp_tool",
        category="security",
        status="deny",
        rollout_mode="shadow",
        policy_revision_ref="policy:v2",
        rule_revision_ref="rule:17",
    )

    assert trace["policy_revision_ref"] == "policy:v2"
    assert trace["rule_revision_ref"] == "rule:17"
    assert trace["rollout_mode"] == "shadow"


def test_resolve_run_first_cohort_label_maps_override_off_when_rollout_off():
    assert app_config.resolve_run_first_cohort_label(
        "off",
        eligible=False,
        ineligible_reason="rollout_off",
    ) == "override_off"

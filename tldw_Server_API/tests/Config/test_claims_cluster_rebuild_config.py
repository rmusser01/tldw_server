"""Cluster execution controls are bounded independently of semantic settings."""

import configparser
from dataclasses import asdict

import pytest

from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.Claims_Extraction import claims_jobs

LIMITS = {
    "exact_claims": 50_000,
    "embedding_claims": 5_000,
    "text_bytes": 33_554_432,
    "vector_width": 4_096,
    "embedding_components": 2_000_000,
    "similarity_component_comparisons": 50_000_000,
    "cleanup_references": 50_000,
}
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("global_flag,feature", [(False, False), (False, True), (True, False), (True, True)])
def test_resource_flag_is_conjunctive(global_flag, feature):
    assert claims_jobs.claims_cluster_rebuild_jobs_enabled(
        {"CLAIMS_JOBS_ENABLED": global_flag, "CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED": feature}
    ) is (global_flag and feature)


def test_all_defaults():
    assert not claims_jobs.claims_cluster_rebuild_jobs_enabled({})
    assert asdict(claims_jobs.claims_cluster_rebuild_limits({})) == LIMITS


@pytest.mark.parametrize("field,maximum", LIMITS.items())
@pytest.mark.parametrize("value", [1, "1", " 1 ", "max"])
def test_each_limit_accepts_bounded_config_integers(field, maximum, value):
    value = maximum if value == "max" else value
    result = claims_jobs.claims_cluster_rebuild_limits({f"CLAIMS_CLUSTER_REBUILD_MAX_{field.upper()}": value})
    assert getattr(result, field) == int(value)


@pytest.mark.parametrize("field,maximum", LIMITS.items())
@pytest.mark.parametrize(
    "bad", [True, False, 0, -1, 1.0, float("nan"), float("inf"), "1.0", "NaN", "", "1_000", "\u0661", None, "over"]
)
def test_each_limit_rejects_invalid_or_widened_values(field, maximum, bad):
    value = maximum + 1 if bad == "over" else bad
    with pytest.raises(ValueError):
        claims_jobs.claims_cluster_rebuild_limits({f"CLAIMS_CLUSTER_REBUILD_MAX_{field.upper()}": value})


def test_injected_settings_win_but_environment_overrides_process_settings(monkeypatch):
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_MAX_EXACT_CLAIMS", "7")
    monkeypatch.setenv("CLAIMS_JOBS_ENABLED", "true")
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "true")
    assert claims_jobs.claims_cluster_rebuild_limits({}).exact_claims == 50_000
    assert claims_jobs.claims_cluster_rebuild_limits().exact_claims == 7
    assert not claims_jobs.claims_cluster_rebuild_jobs_enabled({})
    assert claims_jobs.claims_cluster_rebuild_jobs_enabled()


def test_loader_config_defaults_and_environment(monkeypatch):
    parser = configparser.ConfigParser()
    parser.add_section("ClaimsMonitoring")
    parser.set("ClaimsMonitoring", "CLAIMS_CLUSTER_REBUILD_MAX_VECTOR_WIDTH", "23")
    monkeypatch.setattr(config, "_load_env_files_early", lambda: None)
    monkeypatch.setattr(config, "load_and_log_configs", lambda: {})
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)
    for field in LIMITS:
        monkeypatch.delenv(f"CLAIMS_CLUSTER_REBUILD_MAX_{field.upper()}", raising=False)
    loaded = config.load_settings()
    assert loaded["CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED"] is False
    assert loaded["CLAIMS_JOBS_MAX_RETRIES_CLUSTER_REBUILD"] == "3"
    assert claims_jobs.claims_cluster_rebuild_limits(loaded).vector_width == 23
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_MAX_VECTOR_WIDTH", "11")
    assert claims_jobs.claims_cluster_rebuild_limits(config.load_settings()).vector_width == 11

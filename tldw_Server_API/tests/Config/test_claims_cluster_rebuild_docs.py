"""Executable operator-documentation contract, including unsafe toggle ordering."""

from pathlib import Path

import pytest

from tldw_Server_API.app.core.Claims_Extraction.claims_jobs import (
    claims_cluster_rebuild_jobs_enabled,
    claims_cluster_rebuild_limits,
)

ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = ROOT / "Docs/Operations/Claims_Cluster_Rebuild_Jobs.md"
pytestmark = pytest.mark.unit


def runbook():
    assert RUNBOOK.is_file(), "Missing operational rollout/rollback/restore runbook"
    return RUNBOOK.read_text(encoding="utf-8")


def test_flags_defaults_and_all_hard_limits_match_real_configuration():
    text = runbook()
    assert not claims_cluster_rebuild_jobs_enabled({})
    assert not claims_cluster_rebuild_jobs_enabled({"CLAIMS_JOBS_ENABLED": True})
    assert claims_cluster_rebuild_jobs_enabled(
        {"CLAIMS_JOBS_ENABLED": True, "CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED": True}
    )
    for key in ("CLAIMS_JOBS_ENABLED", "CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED", "CLAIMS_JOBS_WORKER_ENABLED"):
        assert key in text
    limits = claims_cluster_rebuild_limits({})
    for name, field in (
        ("EXACT_CLAIMS", "exact_claims"),
        ("EMBEDDING_CLAIMS", "embedding_claims"),
        ("TEXT_BYTES", "text_bytes"),
        ("VECTOR_WIDTH", "vector_width"),
        ("EMBEDDING_COMPONENTS", "embedding_components"),
        ("SIMILARITY_COMPONENT_COMPARISONS", "similarity_component_comparisons"),
        ("CLEANUP_REFERENCES", "cleanup_references"),
    ):
        assert f"| CLAIMS_CLUSTER_REBUILD_MAX_{name} | {getattr(limits, field)} |" in text
    assert "| CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED | false |" in text
    assert "| CLAIMS_JOBS_MAX_RETRIES_CLUSTER_REBUILD | 3 |" in text


def test_both_safe_urls_and_key_transport_limits_are_explicit():
    text = runbook()
    for required in (
        "/api/v1/jobs/owned/{job_uuid}",
        "/api/v1/jobs/{job_id}?domain=claims&job_uuid={job_uuid}",
        "Idempotency-Key",
        "86400",
        "128",
        "keyless",
        "404",
        "Location",
        "202",
        "409",
    ):
        assert required in text


def test_rollback_waits_for_physical_handlers_before_enabling_legacy_sync():
    text = runbook()
    rollback = text.split("## Rollback", 1)[1].split("## Restore", 1)[0]
    steps = [
        "Quiesce ingress",
        "keep async true",
        "Drain or cancel",
        "physical handlers/threads/processes",
        "uniformly flip",
        "false",
        "reopen synchronous ingress",
    ]
    positions = [rollback.index(step) for step in steps]
    assert positions == sorted(positions)
    assert "false starts LEGACY SYNC; it is NOT stop-admission" in rollback
    assert "terminal status alone" in rollback
    assert "never drop heads or receipts" in rollback


def test_restore_requires_independent_evidence_not_just_matching_uuid():
    restore = runbook().split("## Restore", 1)[1]
    for required in (
        "both async and legacy writers",
        "checkpoint provenance",
        "authority equality",
        "Jobs counter >= domain watermark",
        "equal-generation UUID",
        "receipt completeness",
        "matching UUID alone is insufficient",
        "stays disabled",
        "no automatic restore repair",
    ):
        assert required in restore


@pytest.mark.parametrize("name", ["Claims_API_and_Schema.md", "Claims_Extraction.md", "Jobs_Manager.md"])
def test_code_documentation_links_the_operator_runbook(name):
    text = (ROOT / "Docs/Code_Documentation" / name).read_text(encoding="utf-8")
    assert "../Operations/Claims_Cluster_Rebuild_Jobs.md" in text
    assert "claims_rebuild_clusters" in text

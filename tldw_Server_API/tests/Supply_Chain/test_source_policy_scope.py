"""Separate optional dependency-review approvals from the base source scan."""

import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.dependency_review import derive_allow_ghsas
from Helper_Scripts.Supply_Chain.exception_policy import evaluate_trivy_report, load_policy

pytestmark = pytest.mark.unit
POLICY = Path(".github/supply-chain/vulnerability-exceptions.json")
TODAY = date(2026, 9, 11)
OPTIONAL_IDS = {"TASK-13013.7.38-LIGHTNING-01"} | {f"TASK-13013.7.40-OPTIONAL-{index:02d}" for index in range(1, 6)}
BASE_IDENTITIES = (
    ("chromadb", "1.5.9", "CVE-2026-45829", "CRITICAL"),
    ("chromadb", "1.5.9", "CVE-2026-45833", "CRITICAL"),
    ("chromadb", "1.5.9", "CVE-2026-45830", "HIGH"),
    ("chromadb", "1.5.9", "CVE-2026-45831", "HIGH"),
    ("nltk", "3.10.3", "CVE-2026-81726", "HIGH"),
)


def base_report() -> dict:
    """Use the five literal High/Critical identities in the production export."""
    return {
        "Results": [
            {
                "Target": "source",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": cve,
                        "Severity": severity,
                        "InstalledVersion": version,
                        "PkgIdentifier": {"PURL": f"pkg:pypi/{name}@{version}"},
                    }
                    for name, version, cve, severity in BASE_IDENTITIES
                ],
            }
        ]
    }


def test_base_source_admission_does_not_require_absent_optional_findings() -> None:
    policy = load_policy(POLICY, today=TODAY)
    decision = evaluate_trivy_report(base_report(), component="source-python-root", policy=policy, today=TODAY)
    assert len(decision.excepted) == 5
    assert not decision.blocking and not decision.unmatched_exception_ids


def test_missing_base_finding_still_rejects_its_unmatched_approval() -> None:
    report = base_report()
    report["Results"][0]["Vulnerabilities"].pop()
    policy = load_policy(POLICY, today=TODAY)
    decision = evaluate_trivy_report(report, component="source-python-root", policy=policy, today=TODAY)
    assert len(decision.unmatched_exception_ids) == 1


@pytest.mark.parametrize("exception_id", sorted(OPTIONAL_IDS))
@pytest.mark.parametrize("component", ["source-python-root", "image-app"])
def test_optional_dependency_review_approval_cannot_waive_runtime_scans(exception_id: str, component: str) -> None:
    policy = load_policy(POLICY, today=TODAY)
    record = next(item for item in policy.exceptions if item.id == exception_id)
    finding = {
        "VulnerabilityID": record.vulnerability_id,
        "Severity": record.severity,
        "InstalledVersion": record.installed_version,
        "PkgIdentifier": {"PURL": record.purl},
    }
    decision = evaluate_trivy_report(
        {"Results": [{"Target": "source", "Vulnerabilities": [finding]}]},
        component=component,
        policy=policy,
        today=TODAY,
    )
    assert len(decision.blocking) == 1 and not decision.excepted


def test_duplicate_approvals_across_python_scopes_do_not_authorize_advisory() -> None:
    policy = load_policy(POLICY, today=TODAY)
    record = next(item for item in policy.exceptions if item.id == "TASK-13013.7.38-LIGHTNING-01")
    duplicate = replace(record, id="TEST-DUPLICATE", component="source-python-root")
    policy = replace(policy, exceptions=(*policy.exceptions, duplicate))
    change = {
        "change_type": "added",
        "manifest": "uv.lock",
        "ecosystem": "pip",
        "name": "lightning",
        "version": "2.6.5",
        "package_url": "pkg:pypi/lightning@2.6.5",
        "vulnerabilities": [{"advisory_ghsa_id": "GHSA-qqmf-gpg7-g8gw", "severity": "high"}],
    }
    assert derive_allow_ghsas([change], policy=policy, today=TODAY) == ()


def test_scope_correction_preserves_every_approval_and_other_record() -> None:
    """Undo only the six scope corrections to recover the exact TASK43 policy."""
    raw = json.loads(POLICY.read_text())
    moved = [record for record in raw["exceptions"] if record["id"] in OPTIONAL_IDS]
    assert len(moved) == 6
    assert {record["component"] for record in moved} == {"dependency-review-python-root"}
    for record in moved:
        record["component"] = "source-python-root"
    assert hashlib.sha256((json.dumps(raw, indent=2, sort_keys=True) + "\n").encode()).hexdigest() == (
        "6c2b09f66cbafe5bc93ca3f5a942125e72e4da62afd77c296197d0029910bf5a"
    )

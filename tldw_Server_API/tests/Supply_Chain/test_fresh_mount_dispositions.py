"""Keep fresh-candidate mount exclusions within their reviewed identities."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.exception_policy import evaluate_trivy_report, load_policy

pytestmark = pytest.mark.unit
POLICY = Path(".github/supply-chain/vulnerability-exceptions.json")
EVIDENCE = json.loads(Path("Docs/Evidence/TASK-13013.7.39-fresh-candidates.json").read_text())
TODAY = date(2026, 9, 10)
COMPONENTS = ("image-app", "image-worker", "image-audio-worker")
PACKAGES = ("mount", "libmount1")
CVES = ("CVE-2026-76642", "CVE-2026-78410")


def _finding(package: str, cve: str) -> dict:
    """Provide an exact report identity independently of policy record contents."""
    return {
        "VulnerabilityID": cve,
        "PkgIdentifier": {"PURL": f"pkg:deb/debian/{package}@2.41.5-0%2Bdeb13u1?arch=amd64&distro=debian-13.6"},
        "InstalledVersion": "2.41.5-0+deb13u1",
        "Severity": "HIGH",
    }


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("cve", CVES)
def test_exact_mount_identity_is_allowed_only_during_approval(component: str, package: str, cve: str) -> None:
    policy = load_policy(POLICY, today=TODAY)
    report = {"Results": [{"Target": "Debian", "Vulnerabilities": [_finding(package, cve)]}]}
    for day, expected in ((9, 0), (10, 1), (17, 1), (18, 0)):
        decision = evaluate_trivy_report(report, component=component, policy=policy, today=date(2026, 9, day))
        assert len(decision.excepted) == expected
        assert len(decision.blocking) == 1 - expected


@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("cve", CVES)
def test_changed_mount_identity_or_source_scope_is_not_covered(package: str, cve: str) -> None:
    finding = _finding(package, cve)
    purl = finding["PkgIdentifier"]["PURL"]
    changed = [
        {**finding, "VulnerabilityID": "CVE-2099-99999"},
        {**finding, "InstalledVersion": "2.41.6"},
        {**finding, "Severity": "CRITICAL"},
        *[
            {**finding, "PkgIdentifier": {"PURL": purl.replace(old, new)}}
            for old, new in (
                ("arch=amd64", "arch=arm64"),
                ("distro=debian-13.6", "distro=debian-13.7"),
                (f"/{package}@", "/different-package@"),
                ("@2.41.5-0%2Bdeb13u1", "@2.41.6"),
            )
        ],
    ]
    policy = load_policy(POLICY, today=TODAY)
    report = {"Results": [{"Target": "Debian", "Vulnerabilities": changed}]}
    for component in COMPONENTS:
        decision = evaluate_trivy_report(report, component=component, policy=policy, today=TODAY)
        assert len(decision.blocking) == len(changed)
        assert not decision.excepted
    report["Results"][0]["Vulnerabilities"] = [finding]
    decision = evaluate_trivy_report(report, component="source-python-root", policy=policy, today=TODAY)
    assert len(decision.blocking) == 1


def test_all_310_prior_records_are_preserved_and_only_12_mount_records_added() -> None:
    raw = json.loads(POLICY.read_text())
    added = [record for record in raw["exceptions"] if record["id"].startswith("TASK-13013.7.39-")]
    assert len(added) == 12
    assert added == EVIDENCE["supported_dispositions"]
    raw["exceptions"] = [record for record in raw["exceptions"] if record not in added]
    assert len(raw["exceptions"]) == 310
    assert (
        hashlib.sha256((json.dumps(raw, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
        == EVIDENCE["baseline_policy_sha256"]
    )


@pytest.mark.parametrize("component,remaining", [("image-app", 160), ("image-worker", 3), ("image-audio-worker", 160)])
def test_complete_fresh_report_removes_only_four_reviewed_mount_rows(component: str, remaining: int) -> None:
    binding = EVIDENCE["raw_reports"]
    archive = Path(binding["archive"])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == binding["sha256"]
    with zipfile.ZipFile(archive) as reports:
        name = f"trivy-{component}.json"
        raw = reports.read(name)
    assert hashlib.sha256(raw).hexdigest() == binding["members"][name]["sha256"]
    report = json.loads(raw)
    original = json.dumps(report, sort_keys=True)
    policy = load_policy(POLICY, today=TODAY)
    before = replace(policy, exceptions=tuple(r for r in policy.exceptions if not r.id.startswith("TASK-13013.7.39-")))
    prior = evaluate_trivy_report(report, component=component, policy=before, today=TODAY)
    actual = evaluate_trivy_report(report, component=component, policy=policy, today=TODAY)
    removed = set(prior.blocking) - set(actual.blocking)
    assert len(removed) == 4
    assert {r.vulnerability_id for r in removed} == set(CVES)
    assert len(actual.blocking) == remaining
    assert set(prior.excepted).issubset(actual.excepted)
    assert not actual.unmatched_exception_ids
    assert json.dumps(report, sort_keys=True) == original

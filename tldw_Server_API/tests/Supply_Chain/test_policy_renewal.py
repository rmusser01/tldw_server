"""Verify the approved renewal against immutable retained policy and scan evidence."""

import hashlib
import json
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.exception_policy import PolicyError, evaluate_trivy_report, load_policy

pytestmark = pytest.mark.unit
ARCHIVE = Path("Docs/Evidence/TASK-13013.7.45-ci-image-bypass-evidence.zip")
START = date(2026, 9, 28)
END = date(2026, 10, 2)
POLICIES = [
    (
        "vulnerability-exceptions.json",
        "canonical-policy.json",
        342,
        "835c71df894b511380083ad18117217479ac98491b04695ad7fa47164c3288aa",
    ),
    (
        "ci-image-risk-acceptance.json",
        "ci-policy.json",
        310,
        "9701672ff9aaa7042b8f40332564ec211a510399e91caaabbefe1fc08427aad9",
    ),
]


@pytest.mark.parametrize("filename,member,count,digest", POLICIES)
def test_renewal_preserves_every_prior_identity_and_disposition(
    filename: str, member: str, count: int, digest: str
) -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        prior_bytes = archive.read(member)
    assert hashlib.sha256(prior_bytes).hexdigest() == digest
    prior = json.loads(prior_bytes)["exceptions"]
    path = Path(".github/supply-chain") / filename
    current = json.loads(path.read_text())["exceptions"]
    assert len(current) == len(prior) == count
    for old, new in zip(prior, current, strict=True):
        assert new == {
            **old,
            "id": old["id"] + "-r20260928",
            "supersedes": old["id"],
            "created_on": START.isoformat(),
            "expires_on": END.isoformat(),
        }
    assert len(load_policy(path, today=START).exceptions) == count
    assert len(load_policy(path, today=END).exceptions) == count
    with pytest.raises(PolicyError, match="expires_on"):
        load_policy(path, today=date(2026, 10, 3))


@pytest.mark.parametrize("component", ["image-app", "image-audio-worker"])
def test_retained_reports_have_identical_coverage_and_ci_acceptance_stays_separate(component: str) -> None:
    policies = [load_policy(Path(".github/supply-chain") / row[0], today=START) for row in POLICIES]
    canonical, ci = policies
    assert {record.component for record in ci.exceptions} == {"image-app", "image-audio-worker"}
    combined = replace(canonical, exceptions=canonical.exceptions + ci.exceptions)
    with zipfile.ZipFile(ARCHIVE) as archive:
        report = json.loads(archive.read(f"{component.removeprefix('image-')}/evidence/trivy-{component}.json"))
    original = json.dumps(report, sort_keys=True)
    release = evaluate_trivy_report(report, component=component, policy=canonical, today=START)
    assert len(release.blocking) == 155 and len(release.excepted) == 80
    for today in (START, END):
        decision = evaluate_trivy_report(report, component=component, policy=combined, today=today)
        assert not decision.blocking and not decision.unmatched_exception_ids
        assert len(decision.excepted) == 235
    for today in (date(2026, 9, 27), date(2026, 10, 3)):
        assert not evaluate_trivy_report(report, component=component, policy=combined, today=today).excepted
    assert json.dumps(report, sort_keys=True) == original


@pytest.mark.parametrize("field", ["vulnerability_id", "purl", "installed_version", "severity", "component"])
def test_renewal_does_not_accept_changed_finding_identities(field: str) -> None:
    for filename, _, _, _ in POLICIES:
        policy = load_policy(Path(".github/supply-chain") / filename, today=START)
        for record in policy.exceptions:
            values = {
                "vulnerability_id": record.vulnerability_id,
                "purl": record.purl,
                "installed_version": record.installed_version,
                "severity": record.severity,
                "component": record.component,
            }
            values[field] = (
                ("CRITICAL" if record.severity == "HIGH" else "HIGH")
                if field == "severity"
                else ("source-admin-ui" if field == "component" else values[field] + "-changed")
            )
            report = {
                "Results": [
                    {
                        "Target": "renewal boundary",
                        "Vulnerabilities": [
                            {
                                "VulnerabilityID": values["vulnerability_id"],
                                "PkgIdentifier": {"PURL": values["purl"]},
                                "InstalledVersion": values["installed_version"],
                                "Severity": values["severity"],
                            }
                        ],
                    }
                ]
            }
            decision = evaluate_trivy_report(report, component=values["component"], policy=policy, today=START)
            assert len(decision.blocking) == 1 and not decision.excepted

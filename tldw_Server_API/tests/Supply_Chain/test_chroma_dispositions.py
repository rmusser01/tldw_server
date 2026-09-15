"""Verify the approved embedded-profile Chroma dispositions against scan findings."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.exception_policy import PolicyError, evaluate_trivy_report, load_policy

pytestmark = pytest.mark.unit

POLICY = Path(".github/supply-chain/vulnerability-exceptions.json")
TODAY = date(2026, 9, 10)
ADVISORIES = [
    ("CVE-2026-45829", "CRITICAL"),
    ("CVE-2026-45833", "CRITICAL"),
    ("CVE-2026-45830", "HIGH"),
    ("CVE-2026-45831", "HIGH"),
]
COMPONENTS = ["source-python-root", "image-app", "image-worker", "image-audio-worker"]


def _finding(vulnerability: str, severity: str, **overrides: object) -> dict:
    return {
        "VulnerabilityID": vulnerability,
        "PkgIdentifier": {"PURL": "pkg:pypi/chromadb@1.5.9"},
        "InstalledVersion": "1.5.9",
        "Severity": severity,
        **overrides,
    }


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize(("vulnerability", "severity"), ADVISORIES)
def test_embedded_chroma_finding_is_excepted_without_hiding_other_findings(
    component: str, vulnerability: str, severity: str
) -> None:
    report = {
        "Results": [
            {
                "Target": "Python",
                "Vulnerabilities": [
                    _finding(vulnerability, severity),
                    _finding("CVE-2099-12345", "HIGH"),
                ],
            }
        ]
    }
    original = json.dumps(report, sort_keys=True)
    decision = evaluate_trivy_report(report, component=component, policy=load_policy(POLICY, today=TODAY), today=TODAY)

    assert [item.vulnerability_id for item in decision.excepted] == [vulnerability]
    assert [item.vulnerability_id for item in decision.blocking] == ["CVE-2099-12345"]
    assert json.dumps(report, sort_keys=True) == original


@pytest.mark.parametrize(
    "overrides",
    [
        {"InstalledVersion": "1.5.8"},
        {"PkgIdentifier": {"PURL": "pkg:pypi/chromadb@1.5.8"}},
        {"PkgIdentifier": {"PURL": "pkg:pypi/different@1.5.9"}},
        {"Severity": "HIGH"},
    ],
)
def test_chroma_identity_changes_are_not_excepted(overrides: dict) -> None:
    report = {
        "Results": [{"Target": "Python", "Vulnerabilities": [_finding("CVE-2026-45829", "CRITICAL", **overrides)]}]
    }
    decision = evaluate_trivy_report(
        report, component="source-python-root", policy=load_policy(POLICY, today=TODAY), today=TODAY
    )
    assert not decision.excepted
    assert len(decision.blocking) == 1


def test_chroma_exception_does_not_apply_to_another_component() -> None:
    report = {"Results": [{"Target": "Python", "Vulnerabilities": [_finding("CVE-2026-45829", "CRITICAL")]}]}
    decision = evaluate_trivy_report(
        report, component="source-admin-ui", policy=load_policy(POLICY, today=TODAY), today=TODAY
    )
    assert not decision.excepted
    assert len(decision.blocking) == 1


def test_existing_os_approvals_are_preserved() -> None:
    packet = json.loads(Path("Docs/Evidence/TASK-13013.7.23-applicability-dispositions.json").read_text())
    expected = packet["proposed_policy"]["exceptions"]
    for record in expected:
        record["approval"] = "https://github.com/rmusser01/tldw_server/pull/2869"
    actual = json.loads(POLICY.read_text())["exceptions"]
    assert [r for r in actual if r["id"].startswith("TASK-13013.7.23-OS-")] == expected


def test_chroma_approval_expires_after_september_17() -> None:
    records = load_policy(POLICY, today=date(2026, 9, 17)).exceptions
    assert len([record for record in records if record.purl == "pkg:pypi/chromadb@1.5.9"]) == 16
    with pytest.raises(PolicyError, match="expires_on"):
        load_policy(POLICY, today=date(2026, 9, 18))

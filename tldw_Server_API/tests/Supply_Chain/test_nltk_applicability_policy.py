"""Verify scoped advisory policy without importing or exercising NLTK."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.dependency_review import derive_allow_ghsas, validate_allow_ghsas
from Helper_Scripts.Supply_Chain.exception_policy import PolicyError, evaluate_trivy_report, load_policy

pytestmark = pytest.mark.unit
POLICY = Path(".github/supply-chain/vulnerability-exceptions.json")
TODAY = date(2026, 9, 10)
GHSA = "GHSA-8mgp-746c-j5xp"
CVE = "CVE-2026-81726"
PURL = "pkg:pypi/nltk@3.10.3"
COMPONENTS = ("source-python-root", "image-app", "image-worker", "image-audio-worker")


def _change(**overrides: object) -> dict:
    """Use the dependency comparison wire format, independently of adapter aliases."""
    return {
        "change_type": "added",
        "manifest": "uv.lock",
        "ecosystem": "pip",
        "name": "nltk",
        "version": "3.10.3",
        "package_url": PURL,
        "scope": "runtime",
        "vulnerabilities": [{"severity": "high", "advisory_ghsa_id": GHSA}],
        **overrides,
    }


@pytest.mark.parametrize("component", COMPONENTS)
def test_exact_finding_is_excepted_with_identity_boundaries_and_raw_preservation(component: str) -> None:
    finding = {
        "VulnerabilityID": CVE,
        "PkgIdentifier": {"PURL": PURL},
        "InstalledVersion": "3.10.3",
        "Severity": "HIGH",
    }
    others = [
        {**finding, "VulnerabilityID": "CVE-2099-12345"},
        {**finding, "PkgIdentifier": {"PURL": PURL + "?extra=custom"}},
        {**finding, "InstalledVersion": "3.10.4"},
        {**finding, "Severity": "CRITICAL"},
    ]
    report = {"Results": [{"Target": "Python", "Vulnerabilities": [finding, *others]}]}
    original = json.dumps(report, sort_keys=True)
    policy = load_policy(POLICY, today=TODAY)
    decision = evaluate_trivy_report(report, component=component, policy=policy, today=TODAY)
    assert len(decision.excepted) == 1 and len(decision.blocking) == 4
    assert json.dumps(report, sort_keys=True) == original
    assert not evaluate_trivy_report(report, component="image-webui", policy=policy, today=TODAY).excepted


@pytest.mark.parametrize("manifest", ["pyproject.toml", "uv.lock"])
@pytest.mark.parametrize("today", [TODAY, date(2026, 9, 17)])
def test_exact_canonical_source_approval_projects_to_dependency_review(manifest: str, today: date) -> None:
    policy = load_policy(POLICY, today=TODAY)
    assert derive_allow_ghsas([_change(manifest=manifest)], policy=policy, today=today) == (GHSA,)


@pytest.mark.parametrize("today", [date(2026, 9, 9), date(2026, 9, 18)])
def test_advisory_alias_does_not_survive_approval_dates(today: date) -> None:
    policy = load_policy(POLICY, today=TODAY)
    assert derive_allow_ghsas([_change()], policy=policy, today=today) == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": "3.10.4"},
        {"version": ""},
        {"package_url": PURL + "?extra=custom"},
        {"name": "chromadb"},
        {"manifest": "nested/uv.lock"},
        {"ecosystem": "npm"},
        {"vulnerabilities": [{"severity": "critical", "advisory_ghsa_id": GHSA}]},
    ],
)
def test_uncovered_occurrence_prevents_advisory_wide_allowance(overrides: dict) -> None:
    policy = load_policy(POLICY, today=TODAY)
    changes = [_change(), _change(**overrides)]
    assert derive_allow_ghsas(changes, policy=policy, today=TODAY) == ()
    with pytest.raises(PolicyError, match="no longer fully covers"):
        validate_allow_ghsas(GHSA, changes, policy=policy, today=TODAY)


def test_image_approvals_cannot_authorize_source_dependency_review() -> None:
    policy = load_policy(POLICY, today=TODAY)
    images_only = replace(policy, exceptions=tuple(r for r in policy.exceptions if r.component != "source-python-root"))
    assert derive_allow_ghsas([_change()], policy=images_only, today=TODAY) == ()


def test_only_four_scoped_records_are_added_and_previous_records_are_preserved() -> None:
    evidence = json.loads(Path("Docs/Evidence/TASK-13013.7.32-nltk-applicability.json").read_text())
    policy = json.loads(POLICY.read_text())
    added = [r for r in policy["exceptions"] if r["id"].startswith("TASK-13013.7.32-")]
    assert {r["component"] for r in added} == set(COMPONENTS) and len(added) == 4
    assert all(r["created_on"] == "2026-09-10" and r["expires_on"] == "2026-09-17" for r in added)
    policy["exceptions"] = [r for r in policy["exceptions"] if r["id"] in evidence["baseline_record_ids"]]
    assert len(policy["exceptions"]) == 268
    assert (
        hashlib.sha256((json.dumps(policy, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
        == evidence["baseline_policy_sha256"]
    )

"""Bound TASK40 allowances without importing model or native runtime libraries."""

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
OPTIONAL = (
    ("hydra-core", "1.3.2", "CVE-2026-68508", "GHSA-2cp2-2r3c-7p7r"),
    ("nemo-toolkit", "2.0.0", "CVE-2025-33245", "GHSA-9379-mwvr-7wxx"),
    ("nemo-toolkit", "2.0.0", "CVE-2025-33253", "GHSA-hvjw-vp7g-39h5"),
    ("nemo-toolkit", "2.0.0", "CVE-2026-24157", "GHSA-m4jw-wgmf-889x"),
    ("nemo-toolkit", "2.0.0", "CVE-2026-24159", "GHSA-v7v2-m736-cf3c"),
)
ACL_PURL = "pkg:deb/debian/libacl1@2.3.2-2%2Bb1?arch=amd64&distro=debian-13.6"
XML_VERSION = "2.12.7+dfsg+really2.9.14-2.1+deb13u3"
XML_PURL = "pkg:deb/debian/libxml2@2.12.7%2Bdfsg%2Breally2.9.14-2.1%2Bdeb13u3?arch=amd64&distro=debian-13.6"
IDENTITIES = (
    tuple(
        ("dependency-review-python-root", f"pkg:pypi/{name}@{version}", version, cve)
        for name, version, cve, _ghsa in OPTIONAL
    )
    + tuple(
        (component, ACL_PURL, "2.3.2-2+b1", "CVE-2026-54369")
        for component in ("image-app", "image-worker", "image-audio-worker")
    )
    + tuple((component, XML_PURL, XML_VERSION, "CVE-2026-86140") for component in ("image-app", "image-audio-worker"))
    + tuple(
        (component, purl, version, cve)
        for component in ("image-app", "image-audio-worker")
        for purl, version, cve in (
            (
                "pkg:deb/debian/libsndfile1@1.2.2-2%2Bdeb13u1?arch=amd64&distro=debian-13.6",
                "1.2.2-2+deb13u1",
                "CVE-2026-37555",
            ),
            (
                "pkg:deb/debian/libtiff6@4.7.0-3%2Bdeb13u3?arch=amd64&distro=debian-13.6",
                "4.7.0-3+deb13u3",
                "CVE-2026-36849",
            ),
        )
    )
)


def _finding(purl: str, version: str, cve: str) -> dict:
    return {
        "VulnerabilityID": cve,
        "PkgIdentifier": {"PURL": purl},
        "InstalledVersion": version,
        "Severity": "HIGH",
    }


def _change(name: str, version: str, ghsa: str, /, **overrides: object) -> dict:
    return {
        "change_type": "added",
        "manifest": "uv.lock",
        "ecosystem": "pip",
        "name": name,
        "version": version,
        "package_url": f"pkg:pypi/{name}@{version}",
        "scope": "runtime",
        "vulnerabilities": [{"severity": "high", "advisory_ghsa_id": ghsa}],
        **overrides,
    }


@pytest.mark.parametrize("component,purl,version,cve", IDENTITIES)
@pytest.mark.parametrize("day,active", [(9, False), (10, True), (17, True), (18, False)])
def test_exact_reviewed_identity_is_allowed_only_during_approval(
    component: str, purl: str, version: str, cve: str, day: int, active: bool
) -> None:
    policy = load_policy(POLICY, today=TODAY)
    report = {"Results": [{"Target": "reviewed", "Vulnerabilities": [_finding(purl, version, cve)]}]}
    decision = evaluate_trivy_report(report, component=component, policy=policy, today=date(2026, 9, day))
    assert (len(decision.excepted), len(decision.blocking)) == ((1, 0) if active else (0, 1))


@pytest.mark.parametrize("component,purl,version,cve", IDENTITIES)
def test_reviewed_identity_does_not_hide_other_findings_or_components(
    component: str, purl: str, version: str, cve: str
) -> None:
    finding = _finding(purl, version, cve)
    changed = [
        {**finding, "VulnerabilityID": "CVE-2099-99999"},
        {**finding, "InstalledVersion": "different-version"},
        {**finding, "Severity": "CRITICAL"},
        {**finding, "PkgIdentifier": {"PURL": purl.replace("@", "-different-package@")}},
        {**finding, "PkgIdentifier": {"PURL": purl + "&extra=unreviewed"}},
    ]
    if purl.startswith("pkg:deb/"):
        changed.extend(
            {**finding, "PkgIdentifier": {"PURL": purl.replace(old, new)}}
            for old, new in (("arch=amd64", "arch=arm64"), ("distro=debian-13.6", "distro=debian-13.7"))
        )
    report = {"Results": [{"Target": "reviewed", "Vulnerabilities": [finding, *changed]}]}
    original = json.dumps(report, sort_keys=True)
    policy = load_policy(POLICY, today=TODAY)
    decision = evaluate_trivy_report(report, component=component, policy=policy, today=TODAY)
    assert len(decision.excepted) == 1 and len(decision.blocking) == len(changed)
    assert json.dumps(report, sort_keys=True) == original
    other = evaluate_trivy_report(report, component="image-webui", policy=policy, today=TODAY)
    assert not other.excepted


@pytest.mark.parametrize("name,version,cve,ghsa", OPTIONAL)
@pytest.mark.parametrize("day", [10, 17])
def test_every_exact_source_occurrence_projects_to_its_own_advisory(
    name: str, version: str, cve: str, ghsa: str, day: int
) -> None:
    policy = load_policy(POLICY, today=TODAY)
    changes = [_change(name, version, ghsa, manifest=manifest) for manifest in ("uv.lock", "pyproject.toml")]
    assert derive_allow_ghsas(changes, policy=policy, today=date(2026, 9, day)) == (ghsa,)
    without_approval = replace(policy, exceptions=tuple(r for r in policy.exceptions if r.vulnerability_id != cve))
    assert derive_allow_ghsas(changes, policy=without_approval, today=TODAY) == ()


@pytest.mark.parametrize("name,version,cve,ghsa", OPTIONAL)
@pytest.mark.parametrize("day", [9, 18])
def test_source_advisory_allowance_expires_and_cannot_start_early(
    name: str, version: str, cve: str, ghsa: str, day: int
) -> None:
    policy = load_policy(POLICY, today=TODAY)
    with pytest.raises(PolicyError, match="no longer fully covers"):
        validate_allow_ghsas(ghsa, [_change(name, version, ghsa)], policy=policy, today=date(2026, 9, day))


@pytest.mark.parametrize("name,version,cve,ghsa", OPTIONAL)
@pytest.mark.parametrize(
    "overrides",
    [
        {"version": "different-version"},
        {"version": ""},
        {"package_url": "pkg:pypi/different-package@1.0"},
        {"name": "different-package"},
        {"manifest": "nested/uv.lock"},
        {"ecosystem": "npm"},
        {"severity": "critical"},
    ],
)
def test_one_uncovered_source_occurrence_blocks_global_advisory_allowance(
    name: str, version: str, cve: str, ghsa: str, overrides: dict
) -> None:
    policy = load_policy(POLICY, today=TODAY)
    changes = [_change(name, version, ghsa), _change(name, version, ghsa, **overrides)]
    if "severity" in overrides:
        changes[1]["vulnerabilities"] = [{"severity": overrides["severity"], "advisory_ghsa_id": ghsa}]
    assert derive_allow_ghsas(changes, policy=policy, today=TODAY) == ()
    with pytest.raises(PolicyError, match="no longer fully covers"):
        validate_allow_ghsas(ghsa, changes, policy=policy, today=TODAY)


@pytest.mark.parametrize("component", ["image-app", "image-audio-worker"])
def test_nonvalidating_xml_scope_does_not_allow_entity_parser_advisory(component: str) -> None:
    policy = load_policy(POLICY, today=TODAY)
    finding = {**_finding(XML_PURL, XML_VERSION, "CVE-2026-6653"), "Severity": "CRITICAL"}
    report = {"Results": [{"Target": "Debian", "Vulnerabilities": [finding]}]}
    decision = evaluate_trivy_report(report, component=component, policy=policy, today=TODAY)
    assert len(decision.blocking) == 1 and not decision.excepted


@pytest.mark.parametrize("component", ["image-app", "image-worker", "image-audio-worker"])
@pytest.mark.parametrize("cve", ["CVE-2026-11822", "CVE-2026-11824"])
def test_unresolved_sqlite_advisories_remain_gated(component: str, cve: str) -> None:
    policy = load_policy(POLICY, today=TODAY)
    finding = _finding(
        "pkg:deb/debian/libsqlite3-0@3.46.1-7%2Bdeb13u1?arch=amd64&distro=debian-13.6",
        "3.46.1-7+deb13u1",
        cve,
    )
    report = {"Results": [{"Target": "Debian", "Vulnerabilities": [finding]}]}
    decision = evaluate_trivy_report(report, component=component, policy=policy, today=TODAY)
    assert len(decision.blocking) == 1 and not decision.excepted


def test_exact_new_dispositions_preserve_all_322_prior_records() -> None:
    raw = json.loads(POLICY.read_text())
    added = [r for r in raw["exceptions"] if r["id"].startswith("TASK-13013.7.40-")]
    assert len(added) == 14
    assert {(r["component"], r["purl"], r["installed_version"], r["vulnerability_id"]) for r in added} == set(
        IDENTITIES
    )
    # TASK42 separately verifies preservation of all 336 records present after TASK40.
    raw["exceptions"] = [r for r in raw["exceptions"] if r not in added and not r["id"].startswith("TASK-13013.7.42-")]
    assert len(raw["exceptions"]) == 322
    # Reconstruct the historical component before the TASK44 scope-only correction.
    for record in raw["exceptions"]:
        if record["id"] == "TASK-13013.7.38-LIGHTNING-01":
            record["component"] = "source-python-root"
    assert (
        hashlib.sha256((json.dumps(raw, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
        == "011ad751ad8e08db1140c9a7811eb7a0146ff96a4745abe840b8f30ab2bdfb90"
    )

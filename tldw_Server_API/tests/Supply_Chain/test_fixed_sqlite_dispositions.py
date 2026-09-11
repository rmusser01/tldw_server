"""Keep fixed-package scanner dispositions separate from affected SQLite."""

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.exception_policy import evaluate_trivy_report, load_policy

pytestmark = pytest.mark.unit
POLICY = Path(".github/supply-chain/vulnerability-exceptions.json")
COMPONENTS = ("image-app", "image-worker", "image-audio-worker")
CVES = ("CVE-2026-11822", "CVE-2026-11824")
PURL = "pkg:deb/debian/libsqlite3-0@3.53.4-2?arch=amd64&distro=debian-13.6"


def test_fixed_package_dispositions_preserve_all_336_prior_records() -> None:
    """Bind the exact six additions and every earlier serialized policy record."""
    raw = json.loads(POLICY.read_text())
    added = [r for r in raw["exceptions"] if r["id"].startswith("TASK-13013.7.42-")]
    assert len(added) == 6
    assert {
        (r["component"], r["vulnerability_id"], r["purl"], r["installed_version"], r["severity"]) for r in added
    } == {(component, cve, PURL, "3.53.4-2", "HIGH") for component in COMPONENTS for cve in CVES}
    raw["exceptions"] = [r for r in raw["exceptions"] if r not in added]
    assert len(raw["exceptions"]) == 336
    # TASK44 separately proves only these six component fields changed.
    optional_ids = {"TASK-13013.7.38-LIGHTNING-01"} | {f"TASK-13013.7.40-OPTIONAL-{index:02d}" for index in range(1, 6)}
    for record in raw["exceptions"]:
        if record["id"] in optional_ids:
            record["component"] = "source-python-root"
    assert hashlib.sha256((json.dumps(raw, indent=2, sort_keys=True) + "\n").encode()).hexdigest() == (
        "33c9c0eface2dd7b5c2f82be398197b95f3dcecba4c68effdfbcc21a6e0918a6"
    )


def finding(cve: str, **overrides: object) -> dict:
    """Make one scanner row without changing its security-relevant identity."""
    return {
        "VulnerabilityID": cve,
        "PkgIdentifier": {"PURL": PURL},
        "InstalledVersion": "3.53.4-2",
        "Severity": "HIGH",
        **overrides,
    }


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("cve", CVES)
@pytest.mark.parametrize("day,active", [(9, False), (10, True), (17, True), (18, False)])
def test_fixed_package_is_allowed_only_during_review_window(component: str, cve: str, day: int, active: bool) -> None:
    policy = load_policy(POLICY, today=date(2026, 9, 10))
    report = {"Results": [{"Target": "Debian", "Vulnerabilities": [finding(cve)]}]}
    decision = evaluate_trivy_report(report, component=component, policy=policy, today=date(2026, 9, day))
    assert (len(decision.excepted), len(decision.blocking)) == ((1, 0) if active else (0, 1))


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("cve", CVES)
def test_fixed_package_allowance_cannot_cover_unreviewed_rows(component: str, cve: str) -> None:
    unreviewed = [
        finding(
            cve,
            InstalledVersion="3.46.1-7+deb13u1",
            PkgIdentifier={"PURL": PURL.replace("3.53.4-2", "3.46.1-7%2Bdeb13u1")},
        ),
        finding(cve, InstalledVersion="3.53.4-3"),
        finding(cve, Severity="CRITICAL"),
        finding("CVE-2099-99999"),
        finding(cve, PkgIdentifier={"PURL": PURL.replace("amd64", "arm64")}),
        finding(cve, PkgIdentifier={"PURL": PURL.replace("13.6", "13.7")}),
        finding(cve, PkgIdentifier={"PURL": PURL.replace("libsqlite3-0", "different-package")}),
    ]
    policy = load_policy(POLICY, today=date(2026, 9, 10))
    report = {"Results": [{"Target": "Debian", "Vulnerabilities": [finding(cve), *unreviewed]}]}
    decision = evaluate_trivy_report(report, component=component, policy=policy, today=date(2026, 9, 10))
    assert (len(decision.excepted), len(decision.blocking)) == (1, 7)
    other = evaluate_trivy_report(report, component="image-webui", policy=policy, today=date(2026, 9, 10))
    assert len(other.blocking) == 8 and not other.excepted

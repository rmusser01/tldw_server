"""Keep verified native-package dispositions within their documented scope."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.exception_policy import PolicyError, evaluate_trivy_report, load_policy

pytestmark = pytest.mark.unit

POLICY = Path(".github/supply-chain/vulnerability-exceptions.json")
EVIDENCE = json.loads(Path("Docs/Evidence/TASK-13013.7.29-deployment-applicability.json").read_text())
RECORDS = EVIDENCE["supported_dispositions"]
NATIVE_EVIDENCE = json.loads(Path("Docs/Evidence/TASK-13013.7.30-native-applicability.json").read_text())
NATIVE_RECORDS = NATIVE_EVIDENCE["supported_dispositions"]
TODAY = date(2026, 9, 10)


@pytest.mark.parametrize("record", RECORDS + NATIVE_RECORDS, ids=lambda record: record["id"])
def test_verified_package_match_is_excepted_without_hiding_other_findings(record: dict) -> None:
    finding = {
        "VulnerabilityID": record["vulnerability_id"],
        "PkgIdentifier": {"PURL": record["purl"]},
        "InstalledVersion": record["installed_version"],
        "Severity": record["severity"],
    }
    # Independently vary every matching dimension, including architecture in the PURL.
    changed = [
        {**finding, "VulnerabilityID": "CVE-2099-12345"},
        {**finding, "InstalledVersion": "changed-version"},
        {**finding, "PkgIdentifier": {"PURL": record["purl"].replace("arch=", "arch=changed-")}},
        {**finding, "PkgIdentifier": {"PURL": record["purl"].replace("distro=", "distro=changed-")}},
        {**finding, "PkgIdentifier": {"PURL": record["purl"].replace("/debian/", "/debian/changed-")}},
        {**finding, "Severity": "HIGH" if record["severity"] == "CRITICAL" else "CRITICAL"},
    ]
    report = {"Results": [{"Target": "Debian", "Vulnerabilities": [finding, *changed]}]}
    original = json.dumps(report, sort_keys=True)
    policy = load_policy(POLICY, today=TODAY)
    decision = evaluate_trivy_report(report, component=record["component"], policy=policy, today=TODAY)

    assert len(decision.excepted) == 1
    assert len(decision.blocking) == len(changed)
    assert json.dumps(report, sort_keys=True) == original
    other_component = evaluate_trivy_report(report, component="source-python-root", policy=policy, today=TODAY)
    assert not other_component.excepted


def test_previous_os_and_chroma_dispositions_are_unchanged() -> None:
    policy = json.loads(POLICY.read_text())
    historical_ids = {f"TASK-13013.7.23-OS-{n:02d}" for n in range(1, 6)} | {
        f"TASK-13013.7.28-CHROMA-{n:02d}" for n in range(1, 17)
    }
    policy["exceptions"] = [r for r in policy["exceptions"] if r["id"] in historical_ids]
    serialized = (json.dumps(policy, indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(serialized).hexdigest() == EVIDENCE["baseline_policy_sha256"]


def test_only_the_72_verified_records_are_added_and_expire_on_schedule() -> None:
    records = load_policy(POLICY, today=date(2026, 9, 17)).exceptions
    actual = [r for r in records if r.id.startswith("TASK-13013.7.29-")]
    assert len(actual) == len(RECORDS) == 72
    assert all(r.created_on == TODAY and r.expires_on == date(2026, 9, 17) for r in actual)
    assert {(r.component, r.purl, r.installed_version, r.vulnerability_id, r.severity) for r in actual} == {
        (r["component"], r["purl"], r["installed_version"], r["vulnerability_id"], r["severity"]) for r in RECORDS
    }
    with pytest.raises(PolicyError, match="expires_on"):
        load_policy(POLICY, today=date(2026, 9, 18))


def test_previous_93_dispositions_are_unchanged() -> None:
    policy = json.loads(POLICY.read_text())
    policy["exceptions"] = [r for r in policy["exceptions"] if r["id"] in NATIVE_EVIDENCE["baseline_record_ids"]]
    assert len(policy["exceptions"]) == 93
    serialized = (json.dumps(policy, indent=2, sort_keys=True) + "\n").encode()
    assert hashlib.sha256(serialized).hexdigest() == NATIVE_EVIDENCE["baseline_policy_sha256"]


def test_only_the_175_native_records_are_added_and_every_record_expires_on_schedule() -> None:
    records = load_policy(POLICY, today=date(2026, 9, 17)).exceptions
    actual = [r for r in records if r.id.startswith("TASK-13013.7.30-")]
    assert len(actual) == len(NATIVE_RECORDS) == 175
    # Check every date directly: an older expired record must not mask a later expiry.
    assert all(r.created_on == TODAY and r.expires_on == date(2026, 9, 17) for r in actual)
    assert {(r.component, r.purl, r.installed_version, r.vulnerability_id, r.severity) for r in actual} == {
        (r["component"], r["purl"], r["installed_version"], r["vulnerability_id"], r["severity"])
        for r in NATIVE_RECORDS
    }
    with pytest.raises(PolicyError, match="expires_on"):
        load_policy(POLICY, today=date(2026, 9, 18))


@pytest.mark.parametrize("record", NATIVE_EVIDENCE["retained_unresolved"])
def test_related_native_findings_outside_the_supported_scope_remain_blocking(record: dict) -> None:
    report = {
        "Results": [
            {
                "Target": "Debian",
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": record["vulnerability_id"],
                        "PkgIdentifier": {"PURL": record["purl"]},
                        "InstalledVersion": record["installed_version"],
                        "Severity": record["severity"],
                    }
                ],
            }
        ]
    }
    decision = evaluate_trivy_report(
        report, component=record["component"], policy=load_policy(POLICY, today=TODAY), today=TODAY
    )
    assert len(decision.blocking) == 1
    assert not decision.excepted

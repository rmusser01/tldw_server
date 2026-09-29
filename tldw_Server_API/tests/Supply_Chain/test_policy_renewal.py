"""Verify the approved renewal against immutable retained policy and scan evidence."""

import hashlib
import json
import zipfile
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.exception_policy import PolicyError, evaluate_trivy_report, load_policy

pytestmark = pytest.mark.unit
ARCHIVE = Path("Docs/Evidence/TASK-13013.7.45-ci-image-bypass-evidence.zip")
RENEWAL = Path("Docs/Evidence/TASK-13013.7.47-sep28-admission.zip")
DISPOSITIONS = Path("Docs/Evidence/TASK-13013.7.47-sep28-dispositions.json")
PROPOSED_CI = Path("Docs/Evidence/TASK-13013.7.47-sep28-proposed-ci-additions.json")
START = date(2026, 9, 28)
ADMISSION = date(2026, 9, 29)
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
    path = zipfile.Path(RENEWAL, filename)
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
    policies = [load_policy(zipfile.Path(RENEWAL, row[0]), today=START) for row in POLICIES]
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
        policy = load_policy(Path(".github/supply-chain") / filename, today=ADMISSION)
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
            decision = evaluate_trivy_report(report, component=values["component"], policy=policy, today=ADMISSION)
            assert len(decision.blocking) == 1 and not decision.excepted


@pytest.mark.parametrize("filename", [row[0] for row in POLICIES])
def test_live_policy_contains_only_retained_records_and_reviewed_dispositions(filename: str) -> None:
    with zipfile.ZipFile(RENEWAL) as archive:
        renewed = json.loads(archive.read(filename))
        retired = set(json.loads(archive.read("retirement.json"))["retired_ids"])
    renewed["exceptions"] = [record for record in renewed["exceptions"] if record["id"] not in retired]
    if filename == "vulnerability-exceptions.json":
        renewed["exceptions"] += json.loads(DISPOSITIONS.read_text())["exceptions"]
    else:
        renewed["exceptions"] += approved_ci_records()
    assert json.loads((Path(".github/supply-chain") / filename).read_text()) == renewed


@pytest.mark.parametrize("component", ["app", "audio-worker", "worker", "webui", "admin-ui"])
def test_retirement_preserves_all_current_findings_and_removes_stale_matches(component: str) -> None:
    canonical, ci = [load_policy(Path(".github/supply-chain") / row[0], today=ADMISSION) for row in POLICIES]
    policy = (
        replace(canonical, exceptions=canonical.exceptions + ci.exceptions)
        if component in {"app", "audio-worker"}
        else canonical
    )
    with zipfile.ZipFile(RENEWAL) as archive:
        report = json.loads(archive.read(f"{component}/trivy-image-{component}.json"))
        original = json.loads(archive.read(f"{component}/scan-decision-image-{component}.json"))
    decision = evaluate_trivy_report(report, component=f"image-{component}", policy=policy, today=ADMISSION)
    reviewed = load_policy(DISPOSITIONS, today=START)
    reviewed = replace(
        reviewed,
        exceptions=reviewed.exceptions
        + tuple(r for r in ci.exceptions if r.id.startswith("TASK-13013.7.47-SEP28-CI-")),
    )
    allowed = evaluate_trivy_report(report, component=f"image-{component}", policy=reviewed, today=ADMISSION).excepted
    moved = [asdict(finding) for finding in allowed]
    assert len(moved) == (13 if component in {"app", "audio-worker"} else 0)
    assert [asdict(finding) for finding in decision.blocking] == [
        row for row in original["blocking"] if row not in moved
    ]

    def key(row: dict[str, str]) -> str:
        return json.dumps(row, sort_keys=True)

    assert sorted([asdict(finding) for finding in decision.excepted], key=key) == sorted(
        original["excepted"] + moved, key=key
    )
    assert not decision.unmatched_exception_ids


def approved_ci_records() -> list[dict]:
    """Bind approval metadata to the six immutable proposed finding identities."""
    return [
        {
            **record,
            "created_on": ADMISSION.isoformat(),
            "rationale": "Requester explicitly approved these six additional app/audio CI records on 2026-09-29 through October 2. Temporary CI-only risk acceptance: current media processing reaches this native library; application exploitation is not demonstrated and non-applicability is not established.",
            "mitigation": "Exact identity and expiry remain enforced. Full build, runtime and scanner evidence remains required. Canonical release admission continues blocking this finding. Evidence and approval scope: Docs/Evidence/TASK-13013.7.47-sep28-dispositions.md.",
        }
        for record in json.loads(PROPOSED_CI.read_text())["exceptions"]
    ]


@pytest.mark.parametrize("component", ["app", "audio-worker"])
def test_approved_ci_additions_leave_release_and_pcre2_blocked(component: str) -> None:
    canonical, ci = [load_policy(Path(".github/supply-chain") / row[0], today=ADMISSION) for row in POLICIES]
    approved = tuple(r for r in ci.exceptions if r.id.startswith("TASK-13013.7.47-SEP28-CI-"))
    assert len(approved) == 6
    assert {r.vulnerability_id for r in approved} == {"CVE-2026-96889", "CVE-2026-86138", "CVE-2026-86139"}
    assert not {r.id for r in approved} & {r.id for r in canonical.exceptions}
    with zipfile.ZipFile(RENEWAL) as archive:
        report = json.loads(archive.read(f"{component}/trivy-image-{component}.json"))
    previous = replace(
        canonical, exceptions=canonical.exceptions + tuple(r for r in ci.exceptions if r not in approved)
    )
    active = replace(canonical, exceptions=canonical.exceptions + ci.exceptions)
    before = evaluate_trivy_report(report, component=f"image-{component}", policy=previous, today=ADMISSION)
    after = evaluate_trivy_report(report, component=f"image-{component}", policy=active, today=ADMISSION)
    assert len(before.blocking) == 6 and len(after.blocking) == 3
    assert all("/libpcre2-8-0@" in finding.purl for finding in after.blocking)
    assert not after.unmatched_exception_ids
    release = evaluate_trivy_report(report, component=f"image-{component}", policy=canonical, today=ADMISSION)
    assert {r.vulnerability_id for r in approved} <= {f.vulnerability_id for f in release.blocking}
    for today in (ADMISSION, END):
        assert (
            len(evaluate_trivy_report(report, component=f"image-{component}", policy=active, today=today).blocking) == 3
        )
    with pytest.raises(PolicyError, match="expires_on"):
        load_policy(Path(".github/supply-chain/ci-image-risk-acceptance.json"), today=date(2026, 10, 3))

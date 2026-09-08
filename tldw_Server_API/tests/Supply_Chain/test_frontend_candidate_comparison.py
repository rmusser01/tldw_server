"""Failure controls for frontend scanner comparison and workflow evidence."""

import importlib.util
import json
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/frontend/compare.py"
WORKFLOW = ROOT / ".github/workflows/frontend-runtime-candidate.yml"
BASELINE = "sha256:" + "a" * 64
CANDIDATE = "sha256:" + "b" * 64


def comparator():
    """Load the real comparison module after asserting that it exists."""
    assert SCRIPT.is_file(), "frontend scanner comparator is not implemented"
    spec = importlib.util.spec_from_file_location("frontend_compare", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trivy(config, findings=()):
    """Create the smallest valid Trivy image report used by these controls."""
    return {
        "SchemaVersion": 2,
        "Metadata": {"ImageID": config},
        "Results": [
            {
                "Target": "image (debian 12)",
                "Class": "os-pkgs",
                "Type": "debian",
                "Vulnerabilities": list(findings),
            }
        ],
    }


def trivy_finding(cve, package, version, severity, *, path=None, fixed=None):
    """Create one Trivy match while retaining optional location/fix fields."""
    finding = {
        "VulnerabilityID": cve,
        "PkgName": package,
        "InstalledVersion": version,
        "Severity": severity,
        "Status": "affected",
    }
    if path is not None:
        finding["PkgPath"] = path
    if fixed is not None:
        finding.update({"FixedVersion": fixed, "Status": "fixed"})
    return finding


def grype(config, matches=(), ignored=()):
    """Create the smallest valid Grype image report used by these controls."""
    return {
        "descriptor": {
            "name": "grype",
            "version": "0.118.0",
            "db": {
                "status": {
                    "schemaVersion": "v6.1.9",
                    "built": "2026-09-08T06:30:10Z",
                    "valid": True,
                }
            },
        },
        "source": {"type": "image", "target": {"imageID": config}},
        "matches": list(matches),
        "ignoredMatches": list(ignored),
    }


def grype_match(cve, package, version, severity, *, package_type="deb", path="/var/lib/dpkg/status"):
    """Create one Grype match with a concrete package occurrence."""
    return {
        "vulnerability": {
            "id": cve,
            "namespace": "debian:distro:debian:12",
            "severity": severity,
            "fix": {"versions": [], "state": "unknown"},
        },
        "artifact": {
            "id": f"{package}-{version}-{path}",
            "name": package,
            "version": version,
            "type": package_type,
            "locations": [{"path": path}],
        },
        "matchDetails": [{"type": "exact-direct-match", "matcher": "dpkg-matcher"}],
    }


def write_reports(tmp_path, baseline_trivy, candidate_trivy, baseline_grype, candidate_grype):
    """Write a complete scanner pair and return paths in CLI order."""
    reports = {
        "baseline-trivy.json": baseline_trivy,
        "candidate-trivy.json": candidate_trivy,
        "baseline-grype.json": baseline_grype,
        "candidate-grype.json": candidate_grype,
    }
    paths = []
    for name, report in reports.items():
        path = tmp_path / name
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    return paths


def compare(tmp_path, baseline_trivy, candidate_trivy, baseline_grype=None, candidate_grype=None):
    """Run real comparison logic for a pair of scanner reports."""
    paths = write_reports(
        tmp_path,
        baseline_trivy,
        candidate_trivy,
        baseline_grype or grype(BASELINE),
        candidate_grype or grype(CANDIDATE),
    )
    return comparator().compare_reports(*paths, BASELINE, CANDIDATE)


def test_below_threshold_finding_persists_and_severity_change_is_explicit(tmp_path):
    baseline = trivy(BASELINE, [trivy_finding("CVE-low", "zlib1g", "1.2.13", "LOW")])
    candidate = trivy(CANDIDATE, [trivy_finding("CVE-low", "zlib1g", "1.3", "UNKNOWN")])

    comparison = compare(tmp_path, baseline, candidate)

    match = comparison["scanners"]["trivy"]["matches"][0]
    assert comparison["admitted"] is False
    assert match["classification"] == "still-reported-severity-changed"
    assert match["baseline"][0]["vulnerability"]["Severity"] == "LOW"
    assert match["candidate"][0]["vulnerability"]["Severity"] == "UNKNOWN"


def test_package_rename_is_ambiguous_and_never_claimed_fixed(tmp_path):
    baseline = trivy(BASELINE, [trivy_finding("CVE-z", "zlib1g", "1.2.13", "HIGH")])
    candidate = trivy(CANDIDATE, [trivy_finding("CVE-z", "zlib-ng", "2.2.5", "HIGH")])

    comparison = compare(tmp_path, baseline, candidate)

    classes = [item["classification"] for item in comparison["scanners"]["trivy"]["matches"]]
    assert classes == ["introduced", "baseline-only-unproven"]
    assert all("fixed" not in classification for classification in classes)


def test_introduced_grype_match_is_reported_separately(tmp_path):
    introduced = grype_match("CVE-new", "libc6", "2.39", "Medium")

    comparison = compare(
        tmp_path,
        trivy(BASELINE),
        trivy(CANDIDATE),
        grype(BASELINE),
        grype(CANDIDATE, [introduced]),
    )

    trivy_matches = comparison["scanners"]["trivy"]["matches"]
    grype_matches = comparison["scanners"]["grype"]["matches"]
    assert trivy_matches == []
    assert grype_matches[0]["classification"] == "introduced"


def test_duplicate_versions_paths_and_ignored_rows_are_retained(tmp_path):
    old_a = grype_match("CVE-dup", "openssl", "3.0.1", "Low", path="/usr/lib/a")
    old_b = grype_match("CVE-dup", "openssl", "3.0.2", "Low", path="/usr/lib/b")
    new = grype_match("CVE-dup", "openssl", "3.0.3", "Low", path="/usr/lib/c")
    ignored = grype_match("CVE-ignored", "curl", "8.0", "Negligible")

    comparison = compare(
        tmp_path,
        trivy(BASELINE),
        trivy(CANDIDATE),
        grype(BASELINE, [old_a, old_b], [ignored]),
        grype(CANDIDATE, [new], [ignored, ignored]),
    )

    scanner = comparison["scanners"]["grype"]
    assert [row["artifact"]["version"] for row in scanner["matches"][0]["baseline"]] == ["3.0.1", "3.0.2"]
    assert len(scanner["ignoredMatches"]["baseline"]) == 1
    assert len(scanner["ignoredMatches"]["candidate"]) == 2


@pytest.mark.parametrize(
    "scanner,broken",
    [
        ("trivy", {}),
        ("trivy", {"Metadata": {}, "Results": []}),
        ("grype", {"source": {"target": {}}, "matches": []}),
        ("grype", {"source": {"target": {"imageID": BASELINE}}, "matches": "wrong"}),
    ],
)
def test_malformed_report_is_rejected(tmp_path, scanner, broken):
    reports = [trivy(BASELINE), trivy(CANDIDATE), grype(BASELINE), grype(CANDIDATE)]
    reports[0 if scanner == "trivy" else 2] = broken
    paths = write_reports(tmp_path, *reports)
    module = comparator()

    with pytest.raises(module.ComparisonError, match=scanner):
        module.compare_reports(*paths, BASELINE, CANDIDATE)


def test_missing_report_is_rejected(tmp_path):
    paths = write_reports(tmp_path, trivy(BASELINE), trivy(CANDIDATE), grype(BASELINE), grype(CANDIDATE))
    paths[2].unlink()
    module = comparator()

    with pytest.raises(module.ComparisonError, match="read"):
        module.compare_reports(*paths, BASELINE, CANDIDATE)


@pytest.mark.parametrize("scanner", ["trivy", "grype"])
def test_source_identity_mismatch_is_rejected(tmp_path, scanner):
    wrong = "sha256:" + "c" * 64
    reports = [trivy(BASELINE), trivy(CANDIDATE), grype(BASELINE), grype(CANDIDATE)]
    reports[0 if scanner == "trivy" else 2] = trivy(wrong) if scanner == "trivy" else grype(wrong)
    paths = write_reports(tmp_path, *reports)
    module = comparator()

    with pytest.raises(module.ComparisonError, match="expected config digest"):
        module.compare_reports(*paths, BASELINE, CANDIDATE)


def test_scanner_fixed_metadata_cannot_produce_a_fixed_claim(tmp_path):
    baseline = trivy(
        BASELINE,
        [trivy_finding("CVE-feed-fixed", "libc6", "2.36", "HIGH", fixed="2.37")],
    )

    comparison = compare(tmp_path, baseline, trivy(CANDIDATE))

    match = comparison["scanners"]["trivy"]["matches"][0]
    assert match["classification"] == "baseline-only-unproven"
    assert match["baseline"][0]["vulnerability"]["Status"] == "fixed"
    assert comparison["admitted"] is False


def test_cli_writes_deterministic_json_and_fails_closed(tmp_path):
    paths = write_reports(tmp_path, trivy(BASELINE), trivy(CANDIDATE), grype(BASELINE), grype(CANDIDATE))
    output = tmp_path / "comparison.json"
    argv = [
        sys.executable,
        str(SCRIPT),
        "--baseline-trivy",
        str(paths[0]),
        "--candidate-trivy",
        str(paths[1]),
        "--baseline-grype",
        str(paths[2]),
        "--candidate-grype",
        str(paths[3]),
        "--baseline-config",
        BASELINE,
        "--candidate-config",
        CANDIDATE,
        "--output",
        str(output),
    ]

    result = subprocess.run(argv, check=False, capture_output=True, text=True)  # nosec B603

    assert result.returncode == 0
    assert output.read_text(encoding="utf-8").endswith("\n")
    assert json.loads(output.read_text(encoding="utf-8"))["admitted"] is False


def test_workflow_freezes_database_once_scans_validated_pair_and_always_uploads():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "ghcr.io/aquasecurity/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969"
        in workflow
    )
    assert "anchore/syft:v1.51.1@sha256:95fe0835e5bebc6f8b1f8acef68d47d63d594ef4c0f25c097ff853b23cbac74c" in workflow
    assert "anchore/grype:v0.118.0@sha256:8a93fc48da96bd6ec5981279d099b69de11541dc68fdf222fb9161f8ff284af7" in workflow
    assert workflow.count("--download-db-only") == 1
    assert workflow.count("db update") == 1
    assert "for role in baseline candidate" in workflow
    assert '"oci-dir:/evidence/$role-layout"' in workflow
    assert '--volume "$ARTIFACTS/$role-layout:/input:ro"' in workflow
    assert '--input "/input@$subject"' in workflow
    assert '--input "/evidence/$role.oci.tar"' not in workflow
    assert "baseline-config-digest.txt" in workflow
    assert "candidate-config-digest.txt" in workflow
    assert "trivy-db-sha256-before.txt" in workflow
    assert "trivy-db-sha256-after.txt" in workflow
    assert "grype-db-sha256-before.txt" in workflow
    assert "grype-db-sha256-after.txt" in workflow
    assert "vulnerability-comparison.json" in workflow
    assert "steps.load.outcome == 'success'" in workflow
    assert "if: ${{ always() }}" in workflow

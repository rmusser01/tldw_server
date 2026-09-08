#!/usr/bin/env python3
"""Compare exact frontend image scan pairs without making release claims."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class ComparisonError(ValueError):
    """Raised when scanner evidence is missing, malformed, or misbound."""


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"could not read {label} report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"{label} report must be a JSON object")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonError(f"{context} must be a non-empty string")
    return value


def _digest(value: str, context: str) -> str:
    if DIGEST.fullmatch(value) is None:
        raise ComparisonError(f"{context} must be a lowercase sha256 config digest")
    return value


def _check_identity(actual: Any, expected: str, context: str) -> None:
    if actual != expected:
        raise ComparisonError(f"{context} does not match expected config digest {expected}: {actual!r}")


def _trivy_rows(report: dict[str, Any], expected: str, role: str) -> list[dict[str, Any]]:
    label = f"{role} trivy"
    if report.get("SchemaVersion") != 2:
        raise ComparisonError(f"{label} SchemaVersion must be 2")
    metadata = report.get("Metadata")
    if not isinstance(metadata, dict):
        raise ComparisonError(f"{label} Metadata must be an object")
    _check_identity(metadata.get("ImageID"), expected, f"{label} Metadata.ImageID")
    results = report.get("Results")
    if not isinstance(results, list):
        raise ComparisonError(f"{label} Results must be an array")

    rows: list[dict[str, Any]] = []
    for result_index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ComparisonError(f"{label} Results[{result_index}] must be an object")
        vulnerabilities = result.get("Vulnerabilities", [])
        if vulnerabilities is None:
            vulnerabilities = []
        if not isinstance(vulnerabilities, list):
            raise ComparisonError(f"{label} Results[{result_index}].Vulnerabilities must be an array or null")
        for finding_index, finding in enumerate(vulnerabilities):
            if not isinstance(finding, dict):
                raise ComparisonError(
                    f"{label} Results[{result_index}].Vulnerabilities[{finding_index}] must be an object"
                )
            for field in ("VulnerabilityID", "PkgName", "InstalledVersion", "Severity"):
                _string(finding.get(field), f"{label} finding {field}")
            rows.append(
                {
                    "target": result.get("Target"),
                    "class": result.get("Class"),
                    "type": result.get("Type"),
                    "vulnerability": finding,
                }
            )
    return rows


def _grype_rows(
    report: dict[str, Any], expected: str, role: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    label = f"{role} grype"
    descriptor = report.get("descriptor")
    if not isinstance(descriptor, dict) or descriptor.get("name") != "grype":
        raise ComparisonError(f"{label} descriptor must identify grype")
    _string(descriptor.get("version"), f"{label} descriptor.version")
    database = descriptor.get("db")
    status = database.get("status") if isinstance(database, dict) else None
    if not isinstance(status, dict):
        raise ComparisonError(f"{label} descriptor.db.status must be an object")
    _string(status.get("schemaVersion"), f"{label} database schemaVersion")
    _string(status.get("built"), f"{label} database built timestamp")
    if status.get("valid") is not True:
        raise ComparisonError(f"{label} database status must be valid")

    source = report.get("source")
    target = source.get("target") if isinstance(source, dict) else None
    if not isinstance(target, dict):
        raise ComparisonError(f"{label} source.target must be an object")
    _check_identity(target.get("imageID"), expected, f"{label} source.target.imageID")

    matches = report.get("matches")
    ignored = report.get("ignoredMatches", [])
    if not isinstance(matches, list):
        raise ComparisonError(f"{label} matches must be an array")
    if ignored is None:
        ignored = []
    if not isinstance(ignored, list) or not all(isinstance(row, dict) for row in ignored):
        raise ComparisonError(f"{label} ignoredMatches must be an array of objects")

    for index, match in enumerate(matches):
        if not isinstance(match, dict):
            raise ComparisonError(f"{label} matches[{index}] must be an object")
        vulnerability = match.get("vulnerability")
        artifact = match.get("artifact")
        if not isinstance(vulnerability, dict) or not isinstance(artifact, dict):
            raise ComparisonError(f"{label} matches[{index}] must contain vulnerability and artifact objects")
        for field in ("id", "severity"):
            _string(vulnerability.get(field), f"{label} vulnerability {field}")
        for field in ("name", "version", "type"):
            _string(artifact.get(field), f"{label} artifact {field}")
    return matches, ignored, status


def _trivy_match_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    finding = row["vulnerability"]
    package_type = str(row.get("type") or "")
    if row.get("class") == "os-pkgs" and package_type in {"debian", "ubuntu"}:
        package_type = "deb"
    return (
        finding["VulnerabilityID"],
        finding["PkgName"],
        package_type,
        str(row.get("class") or ""),
    )


def _grype_match_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    package_type = row["artifact"]["type"]
    namespace = str(row["vulnerability"].get("namespace") or "")
    if package_type == "deb" and namespace.startswith(("debian:distro:debian:", "ubuntu:distro:ubuntu:")):
        namespace = "deb:distro"
    return (
        row["vulnerability"]["id"],
        row["artifact"]["name"],
        package_type,
        namespace,
    )


def _severity(row: dict[str, Any], scanner: str) -> str:
    if scanner == "trivy":
        return row["vulnerability"]["Severity"]
    return row["vulnerability"]["severity"]


def _version(row: dict[str, Any], scanner: str) -> str:
    if scanner == "trivy":
        return row["vulnerability"]["InstalledVersion"]
    return row["artifact"]["version"]


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))


def _group(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[str, ...]],
    scanner: str,
) -> list[dict[str, Any]]:
    baseline_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    candidate_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in baseline:
        baseline_groups[key(row)].append(row)
    for row in candidate:
        candidate_groups[key(row)].append(row)

    output = []
    for item_key in sorted(set(baseline_groups) | set(candidate_groups)):
        baseline_rows = _sort_rows(baseline_groups[item_key])
        candidate_rows = _sort_rows(candidate_groups[item_key])
        if baseline_rows and candidate_rows:
            if {_severity(row, scanner) for row in baseline_rows} != {
                _severity(row, scanner) for row in candidate_rows
            }:
                classification = "still-reported-severity-changed"
            elif {_version(row, scanner) for row in baseline_rows} != {
                _version(row, scanner) for row in candidate_rows
            }:
                classification = "still-reported-version-changed"
            else:
                classification = "still-reported"
        elif candidate_rows:
            classification = "introduced"
        else:
            classification = "baseline-only-unproven"
        output.append(
            {
                "key": list(item_key),
                "classification": classification,
                "baseline": baseline_rows,
                "candidate": candidate_rows,
            }
        )
    return output


def compare_reports(
    baseline_trivy: Path,
    candidate_trivy: Path,
    baseline_grype: Path,
    candidate_grype: Path,
    baseline_config: str,
    candidate_config: str,
) -> dict[str, Any]:
    """Validate exact identities and retain complete scanner-specific comparisons."""
    baseline_config = _digest(baseline_config, "baseline config")
    candidate_config = _digest(candidate_config, "candidate config")
    if baseline_config == candidate_config:
        raise ComparisonError("baseline and candidate config digests must differ")

    trivy_baseline = _trivy_rows(_load(baseline_trivy, "baseline trivy"), baseline_config, "baseline")
    trivy_candidate = _trivy_rows(_load(candidate_trivy, "candidate trivy"), candidate_config, "candidate")
    grype_baseline, ignored_baseline, status_baseline = _grype_rows(
        _load(baseline_grype, "baseline grype"), baseline_config, "baseline"
    )
    grype_candidate, ignored_candidate, status_candidate = _grype_rows(
        _load(candidate_grype, "candidate grype"), candidate_config, "candidate"
    )
    if status_baseline != status_candidate:
        raise ComparisonError("baseline and candidate grype reports do not use the same database status")

    return {
        "schemaVersion": 1,
        "scope": "frontend-candidate-vulnerability-comparison-not-release-admission",
        "admitted": False,
        "fixedClaims": [],
        "identities": {"baselineConfig": baseline_config, "candidateConfig": candidate_config},
        "scanners": {
            "trivy": {
                "matches": _group(trivy_baseline, trivy_candidate, _trivy_match_key, "trivy"),
            },
            "grype": {
                "databaseStatus": status_baseline,
                "matches": _group(grype_baseline, grype_candidate, _grype_match_key, "grype"),
                "ignoredMatches": {
                    "baseline": _sort_rows(ignored_baseline),
                    "candidate": _sort_rows(ignored_candidate),
                },
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trivy", required=True, type=Path)
    parser.add_argument("--candidate-trivy", required=True, type=Path)
    parser.add_argument("--baseline-grype", required=True, type=Path)
    parser.add_argument("--candidate-grype", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fail-closed comparison CLI."""
    args = _parser().parse_args(argv)
    try:
        comparison = compare_reports(
            args.baseline_trivy,
            args.candidate_trivy,
            args.baseline_grype,
            args.candidate_grype,
            args.baseline_config,
            args.candidate_config,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (ComparisonError, OSError) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

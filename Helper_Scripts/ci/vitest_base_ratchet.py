"""Compare Vitest failures against an exact-base report without static suppressions."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RatchetError(ValueError):
    """Raised when a Vitest report cannot be compared safely."""


@dataclass(frozen=True, order=True)
class FailedTest:
    """Identify one failed assertion within its owning package."""

    file: str
    full_name: str


@dataclass(frozen=True)
class RatchetResult:
    """Classify exact-base failures and blocking regressions."""

    inherited: tuple[FailedTest, ...]
    regressions: tuple[FailedTest, ...]

    @property
    def passes(self) -> bool:
        """Return whether the head introduced no blocking failures."""

        return not self.regressions


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object or raise a fail-closed report error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RatchetError(f"cannot read Vitest report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RatchetError(f"Vitest report {path} must contain a JSON object")
    return payload


def _relative_test_path(raw_name: object, package_root: Path) -> str:
    """Normalize a Vitest result path beneath its owning package root."""

    if not isinstance(raw_name, str) or not raw_name.strip():
        raise RatchetError("Vitest result is missing a test file name")
    if "\n" in raw_name or "\r" in raw_name:
        raise RatchetError("Vitest test file names cannot contain newlines")

    normalized_root = package_root.resolve()
    result_path = Path(raw_name)
    if not result_path.is_absolute():
        result_path = normalized_root / result_path
    try:
        relative = result_path.resolve().relative_to(normalized_root)
    except ValueError as exc:
        raise RatchetError(
            f"Vitest result path is outside package root: {raw_name}"
        ) from exc
    if relative == Path("."):
        raise RatchetError("Vitest result path must identify a test file")
    return relative.as_posix()


def _load_failures(report_path: Path, package_root: Path) -> tuple[FailedTest, ...]:
    """Read failed assertion identities from a Vitest JSON report."""

    payload = _load_json(report_path)
    if not isinstance(payload.get("success"), bool):
        raise RatchetError(f"Vitest report {report_path} is missing boolean success")
    test_results = payload.get("testResults")
    if not isinstance(test_results, list):
        raise RatchetError(f"Vitest report {report_path} is missing testResults")

    failures: set[FailedTest] = set()
    for test_result in test_results:
        if not isinstance(test_result, dict):
            raise RatchetError(f"Vitest report {report_path} has an invalid test result")
        relative_path = _relative_test_path(test_result.get("name"), package_root)
        assertion_results = test_result.get("assertionResults")
        if not isinstance(assertion_results, list):
            raise RatchetError(
                f"Vitest result {relative_path} is missing assertionResults"
            )

        failed_assertions = []
        for assertion in assertion_results:
            if not isinstance(assertion, dict):
                raise RatchetError(
                    f"Vitest result {relative_path} has an invalid assertion"
                )
            if assertion.get("status") == "failed":
                failed_assertions.append(assertion)

        if test_result.get("status") == "failed" and not failed_assertions:
            raise RatchetError(
                f"collection-level failure in {relative_path} cannot be ratcheted"
            )

        for assertion in failed_assertions:
            full_name = assertion.get("fullName")
            if not isinstance(full_name, str) or not full_name.strip():
                raise RatchetError(
                    f"failed assertion in {relative_path} is missing fullName"
                )
            failures.add(FailedTest(file=relative_path, full_name=full_name))

    if payload["success"] and failures:
        raise RatchetError(
            f"Vitest report {report_path} claims success but contains failed assertions"
        )
    if not payload["success"] and not failures:
        raise RatchetError(
            f"Vitest report {report_path} failed without comparable assertions"
        )
    return tuple(sorted(failures))


def failing_test_files(report_path: Path, package_root: Path) -> tuple[str, ...]:
    """Return sorted package-relative files containing failed assertions."""

    failures = _load_failures(report_path, package_root)
    return tuple(sorted({failure.file for failure in failures}))


def validate_success_report(report_path: Path, package_root: Path) -> None:
    """Require a valid successful Vitest JSON report."""

    if _load_failures(report_path, package_root):
        raise RatchetError(f"expected a successful Vitest report: {report_path}")


def _load_changed_files(path: Path) -> set[str]:
    """Read normalized repository-relative changed paths."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RatchetError(f"cannot read changed-file list {path}: {exc}") from exc
    return {line.strip().replace("\\", "/") for line in lines if line.strip()}


def _normalized_package_repo_path(path: Path) -> Path:
    """Validate a repository-relative package path."""

    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RatchetError(f"package repo path must be repository-relative: {path}")
    return path


def compare_reports(
    *,
    head_report: Path,
    base_report: Path,
    head_package_root: Path,
    base_package_root: Path,
    package_repo_path: Path,
    changed_files_path: Path,
) -> RatchetResult:
    """Classify head failures, blocking new or test-file-modified failures."""

    head_failures = _load_failures(head_report, head_package_root)
    if not head_failures:
        raise RatchetError("head report has no failures to compare")
    base_failures = set(_load_failures(base_report, base_package_root))
    changed_files = _load_changed_files(changed_files_path)
    package_path = _normalized_package_repo_path(package_repo_path)

    inherited: list[FailedTest] = []
    regressions: list[FailedTest] = []
    for failure in head_failures:
        repo_test_path = (package_path / failure.file).as_posix()
        if repo_test_path in changed_files or failure not in base_failures:
            regressions.append(failure)
        else:
            inherited.append(failure)

    return RatchetResult(
        inherited=tuple(inherited),
        regressions=tuple(regressions),
    )


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--report", type=Path, required=True)
    extract.add_argument("--package-root", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate-success")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--package-root", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--head-report", type=Path, required=True)
    compare.add_argument("--base-report", type=Path, required=True)
    compare.add_argument("--head-package-root", type=Path, required=True)
    compare.add_argument("--base-package-root", type=Path, required=True)
    compare.add_argument("--package-repo-path", type=Path, required=True)
    compare.add_argument("--changed-files", type=Path, required=True)
    return parser


def _run_extract(args: argparse.Namespace) -> int:
    """Write failed package-relative test files for exact-base replay."""

    files = failing_test_files(args.report, args.package_root)
    if not files:
        raise RatchetError("failed Vitest process produced no failed assertions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{path}\n" for path in files), encoding="utf-8")
    print(f"[vitest-ratchet] replay_files={len(files)}")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    """Compare reports and return nonzero for regressions."""

    result = compare_reports(
        head_report=args.head_report,
        base_report=args.base_report,
        head_package_root=args.head_package_root,
        base_package_root=args.base_package_root,
        package_repo_path=args.package_repo_path,
        changed_files_path=args.changed_files,
    )
    print(
        "[vitest-ratchet] "
        f"inherited={len(result.inherited)} regressions={len(result.regressions)}"
    )
    for failure in result.regressions:
        print(
            f"[vitest-ratchet] regression: {failure.file} > {failure.full_name}",
            file=sys.stderr,
        )
    return 0 if result.passes else 1


def _run_validate_success(args: argparse.Namespace) -> int:
    """Validate a successful Vitest report."""

    validate_success_report(args.report, args.package_root)
    print("[vitest-ratchet] successful report validated")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Vitest base-ratchet CLI."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "extract":
            return _run_extract(args)
        if args.command == "validate-success":
            return _run_validate_success(args)
        return _run_compare(args)
    except RatchetError as exc:
        print(f"[vitest-ratchet] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

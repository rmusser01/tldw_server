"""Compare Vitest failures against an exact-base report without static suppressions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ASSERTION_STATUSES = frozenset(
    {"passed", "failed", "skipped", "pending", "todo", "disabled"}
)
_TEST_RESULT_STATUSES = frozenset({"passed", "failed"})
_STRICT_COUNTERS = (
    "numTotalTestSuites",
    "numPassedTestSuites",
    "numFailedTestSuites",
    "numPendingTestSuites",
    "numTotalTests",
    "numPassedTests",
    "numFailedTests",
    "numPendingTests",
    "numTodoTests",
)
_SAFETY_ERROR_COUNTERS = (
    "unhandledErrorCount",
    "moduleErrorCount",
    "hookErrorCount",
)


class RatchetError(ValueError):
    """Raised when a Vitest report cannot be compared safely."""


@dataclass(frozen=True, order=True)
class FailedTest:
    """Identify one failed assertion within its owning package."""

    file: str
    full_name: str
    failure_messages: tuple[str, ...] = ()


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


def _nonnegative_int(payload: dict[str, Any], key: str, report_path: Path) -> int:
    """Read a non-negative integer field without accepting booleans."""

    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RatchetError(
            f"Vitest report {report_path} has invalid non-negative counter {key}"
        )
    return value


def _strict_counters(payload: dict[str, Any], report_path: Path) -> dict[str, int]:
    """Validate strict Vitest counters and reject an empty test run."""

    counters = {
        key: _nonnegative_int(payload, key, report_path) for key in _STRICT_COUNTERS
    }
    if counters["numTotalTests"] == 0:
        raise RatchetError(f"Vitest report {report_path} executed zero tests")
    if (
        counters["numPassedTests"]
        + counters["numFailedTests"]
        + counters["numPendingTests"]
        + counters["numTodoTests"]
        != counters["numTotalTests"]
    ):
        raise RatchetError(f"Vitest report {report_path} has inconsistent test counters")
    if (
        counters["numPassedTestSuites"]
        + counters["numFailedTestSuites"]
        + counters["numPendingTestSuites"]
        != counters["numTotalTestSuites"]
    ):
        raise RatchetError(f"Vitest report {report_path} has inconsistent suite counters")
    return counters


def _normalize_failure_message(raw_message: object, package_root: Path) -> str:
    """Normalize only the head/base package-root portion of a failure message."""

    if not isinstance(raw_message, str) or not raw_message.strip():
        raise RatchetError("failed assertion has an empty or invalid failure message")
    normalized_root = package_root.resolve()
    root_variants = {
        str(normalized_root),
        normalized_root.as_posix(),
        str(normalized_root).replace("/", "\\"),
    }
    normalized = raw_message
    for root_variant in sorted(root_variants, key=len, reverse=True):
        if root_variant:
            normalized = normalized.replace(root_variant, "<PACKAGE_ROOT>")
    return normalized


def _validate_safety_report(
    safety_report_path: Path,
    *,
    expected_reason: str,
    expected_module_count: int,
    expected_test_count: int,
) -> None:
    """Require a clean custom-reporter summary for a completed Vitest run."""

    payload = _load_json(safety_report_path)
    if payload.get("schemaVersion") != 1:
        raise RatchetError(
            f"Vitest safety report {safety_report_path} has an unsupported schemaVersion"
        )
    if payload.get("reason") != expected_reason:
        raise RatchetError(
            f"Vitest safety report {safety_report_path} has unexpected run reason"
        )

    counts = {
        key: _nonnegative_int(payload, key, safety_report_path)
        for key in ("moduleCount", "testCount", *_SAFETY_ERROR_COUNTERS)
    }
    if counts["moduleCount"] != expected_module_count:
        raise RatchetError(
            f"Vitest safety report {safety_report_path} moduleCount does not match JSON"
        )
    if counts["testCount"] != expected_test_count:
        raise RatchetError(
            f"Vitest safety report {safety_report_path} testCount does not match JSON"
        )
    for counter_name in _SAFETY_ERROR_COUNTERS:
        if counts[counter_name] != 0:
            raise RatchetError(
                f"Vitest safety report {safety_report_path} has nonzero {counter_name}"
            )


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


def _load_failures(
    report_path: Path,
    package_root: Path,
    *,
    strict: bool = False,
    safety_report_path: Path | None = None,
) -> tuple[FailedTest, ...]:
    """Read failed assertion identities from a Vitest JSON report."""

    payload = _load_json(report_path)
    if not isinstance(payload.get("success"), bool):
        raise RatchetError(f"Vitest report {report_path} is missing boolean success")
    test_results = payload.get("testResults")
    if not isinstance(test_results, list):
        raise RatchetError(f"Vitest report {report_path} is missing testResults")
    counters = _strict_counters(payload, report_path) if strict else None

    failures: list[FailedTest] = []
    assertion_status_counts: Counter[str] = Counter()
    for test_result in test_results:
        if not isinstance(test_result, dict):
            raise RatchetError(f"Vitest report {report_path} has an invalid test result")
        relative_path = _relative_test_path(test_result.get("name"), package_root)
        test_result_status = test_result.get("status")
        if test_result_status not in _TEST_RESULT_STATUSES:
            raise RatchetError(
                f"Vitest result {relative_path} has an invalid test result status: "
                f"{test_result_status!r}"
            )
        assertion_results = test_result.get("assertionResults")
        if not isinstance(assertion_results, list):
            raise RatchetError(
                f"Vitest result {relative_path} is missing assertionResults"
            )
        if strict:
            file_message = test_result.get("message", "")
            if not isinstance(file_message, str):
                raise RatchetError(
                    f"Vitest result {relative_path} has an invalid file-level message"
                )
            if file_message.strip():
                raise RatchetError(
                    f"Vitest result {relative_path} contains a file-level error"
                )

        failed_assertions = []
        for assertion in assertion_results:
            if not isinstance(assertion, dict):
                raise RatchetError(
                    f"Vitest result {relative_path} has an invalid assertion"
                )
            assertion_status = assertion.get("status")
            if assertion_status not in _ASSERTION_STATUSES:
                raise RatchetError(
                    f"Vitest result {relative_path} has an invalid assertion status: "
                    f"{assertion_status!r}"
                )
            assertion_status_counts[assertion_status] += 1
            if assertion_status == "failed":
                failed_assertions.append(assertion)

        if test_result_status == "failed" and not failed_assertions:
            raise RatchetError(
                f"collection-level failure in {relative_path} cannot be ratcheted"
            )
        if test_result_status == "passed" and failed_assertions:
            raise RatchetError(
                f"passed Vitest result {relative_path} contains failed assertions"
            )

        for assertion in failed_assertions:
            full_name = assertion.get("fullName")
            if not isinstance(full_name, str) or not full_name.strip():
                raise RatchetError(
                    f"failed assertion in {relative_path} is missing fullName"
                )
            failure_messages: tuple[str, ...] = ()
            if strict:
                raw_messages = assertion.get("failureMessages")
                if not isinstance(raw_messages, list) or not raw_messages:
                    raise RatchetError(
                        f"failed assertion in {relative_path} is missing failureMessages"
                    )
                failure_messages = tuple(
                    _normalize_failure_message(message, package_root)
                    for message in raw_messages
                )
            failures.append(
                FailedTest(
                    file=relative_path,
                    full_name=full_name,
                    failure_messages=failure_messages,
                )
            )

    if strict:
        assert counters is not None
        observed_counts = {
            "numTotalTests": sum(assertion_status_counts.values()),
            "numPassedTests": assertion_status_counts["passed"],
            "numFailedTests": assertion_status_counts["failed"],
            "numPendingTests": sum(
                assertion_status_counts[status]
                for status in ("skipped", "pending", "disabled")
            ),
            "numTodoTests": assertion_status_counts["todo"],
        }
        for counter_name, observed_count in observed_counts.items():
            if counters[counter_name] != observed_count:
                raise RatchetError(
                    f"Vitest report {report_path} counter {counter_name} does not "
                    f"match assertionResults"
                )
        if safety_report_path is None:
            raise RatchetError("strict Vitest validation requires a safety report")
        _validate_safety_report(
            safety_report_path,
            expected_reason="passed" if payload["success"] else "failed",
            expected_module_count=len(test_results),
            expected_test_count=counters["numTotalTests"],
        )

    if payload["success"] and failures:
        raise RatchetError(
            f"Vitest report {report_path} claims success but contains failed assertions"
        )
    if not payload["success"] and not failures:
        raise RatchetError(
            f"Vitest report {report_path} failed without comparable assertions"
        )
    if strict:
        return tuple(sorted(failures))
    return tuple(sorted(set(failures)))


def failing_test_files(
    report_path: Path,
    package_root: Path,
    *,
    strict: bool = False,
    safety_report_path: Path | None = None,
) -> tuple[str, ...]:
    """Return sorted package-relative files containing failed assertions."""

    failures = _load_failures(
        report_path,
        package_root,
        strict=strict,
        safety_report_path=safety_report_path,
    )
    return tuple(sorted({failure.file for failure in failures}))


def validate_success_report(
    report_path: Path,
    package_root: Path,
    *,
    strict: bool = False,
    safety_report_path: Path | None = None,
) -> None:
    """Require a valid successful Vitest JSON report."""

    if _load_failures(
        report_path,
        package_root,
        strict=strict,
        safety_report_path=safety_report_path,
    ):
        raise RatchetError(f"expected a successful Vitest report: {report_path}")


def _load_changed_files(path: Path, *, strict: bool = False) -> set[str]:
    """Read normalized repository-relative changed paths."""

    try:
        raw_data = path.read_bytes()
        text = raw_data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RatchetError(f"cannot read changed-file list {path}: {exc}") from exc
    if strict:
        if raw_data and not raw_data.endswith(b"\0"):
            raise RatchetError(
                f"strict changed-file list {path} must be NUL-delimited"
            )
        entries = text.split("\0")[:-1] if text else []
    else:
        entries = text.splitlines()

    changed_files: set[str] = set()
    for entry in entries:
        normalized = entry if strict else entry.strip().replace("\\", "/")
        if not normalized:
            continue
        candidate = Path(normalized)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RatchetError(f"changed-file path must be repository-relative: {entry}")
        changed_files.add(normalized)
    return changed_files


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
    strict: bool = False,
    head_safety_report_path: Path | None = None,
    base_safety_report_path: Path | None = None,
) -> RatchetResult:
    """Classify head failures, blocking new or test-file-modified failures."""

    head_failures = _load_failures(
        head_report,
        head_package_root,
        strict=strict,
        safety_report_path=head_safety_report_path,
    )
    if not head_failures:
        raise RatchetError("head report has no failures to compare")
    loaded_base_failures = _load_failures(
        base_report,
        base_package_root,
        strict=strict,
        safety_report_path=base_safety_report_path,
    )
    base_failures = Counter(loaded_base_failures) if strict else set(loaded_base_failures)
    changed_files = _load_changed_files(changed_files_path, strict=strict)
    package_path = _normalized_package_repo_path(package_repo_path)

    inherited: list[FailedTest] = []
    regressions: list[FailedTest] = []
    for failure in head_failures:
        repo_test_path = (package_path / failure.file).as_posix()
        present_in_base = (
            base_failures[failure] > 0 if strict else failure in base_failures
        )
        if repo_test_path in changed_files or not present_in_base:
            regressions.append(failure)
        else:
            inherited.append(failure)
            if strict:
                base_failures[failure] -= 1

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
    extract.add_argument("--strict", action="store_true")
    extract.add_argument("--safety-report", type=Path)

    validate = subparsers.add_parser("validate-success")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--package-root", type=Path, required=True)
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--safety-report", type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--head-report", type=Path, required=True)
    compare.add_argument("--base-report", type=Path, required=True)
    compare.add_argument("--head-package-root", type=Path, required=True)
    compare.add_argument("--base-package-root", type=Path, required=True)
    compare.add_argument("--package-repo-path", type=Path, required=True)
    compare.add_argument("--changed-files", type=Path, required=True)
    compare.add_argument("--strict", action="store_true")
    compare.add_argument("--head-safety-report", type=Path)
    compare.add_argument("--base-safety-report", type=Path)
    return parser


def _run_extract(args: argparse.Namespace) -> int:
    """Write failed package-relative test files for exact-base replay."""

    files = failing_test_files(
        args.report,
        args.package_root,
        strict=args.strict,
        safety_report_path=args.safety_report,
    )
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
        strict=args.strict,
        head_safety_report_path=args.head_safety_report,
        base_safety_report_path=args.base_safety_report,
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

    validate_success_report(
        args.report,
        args.package_root,
        strict=args.strict,
        safety_report_path=args.safety_report,
    )
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

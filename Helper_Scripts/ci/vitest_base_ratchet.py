"""Compare Vitest failures against an exact-base report without static suppressions."""

from __future__ import annotations

import argparse
import json
import re
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
_EXECUTION_SUITE_STATES = frozenset(
    {"passed", "failed", "skipped", "pending", "queued"}
)
_EXECUTION_SUITE_MODES = frozenset({"run", "only", "skip", "todo"})
_SUITE_STATUS_RANK = {"unknown": -1, "pending": 0, "passed": 1, "failed": 2}
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
_SAFETY_BLOCKING_COUNTERS = (
    "unhandledErrorCount",
    "moduleErrorCount",
    "hookErrorCount",
    "incompleteSuiteCount",
    "incompleteTestCount",
)
_VOLATILE_NODE_SCHEDULER_FRAME_PATTERNS = (
    re.compile(
        r"\s*at runNextTicks \(node:internal/process/task_queues:\d+:\d+\)\s*"
    ),
    re.compile(r"\s*at processTimers \(node:internal/timers:\d+:\d+\)\s*"),
)
_V8_STACK_FRAME_PATTERN = re.compile(r"\s+at .+")
_V8_SOURCE_FRAME_PATTERN = re.compile(
    r"\s+at (?:.+ \()?(?:file://)?"
    r"(?:<PACKAGE_ROOT>|<REPOSITORY_ROOT>|/).+:\d+:\d+\)?\s*"
)
_COLLECTION_FAILURE_FULL_NAME = "<collection failure>"


class RatchetError(ValueError):
    """Raised when a Vitest report cannot be compared safely."""


@dataclass(frozen=True, order=True)
class FailedTest:
    """Identify one comparable test or collection failure within its package."""

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


@dataclass(frozen=True)
class ReportContext:
    """Record canonical modules plus reconciled test and suite identities."""

    files: tuple[str, ...]
    tests: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    suites: tuple[tuple[tuple[str, ...], str], ...]
    counters: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ExecutionContext:
    """Record reporter-observed module order and exact runtime suite tree."""

    modules: tuple[str, ...]
    suites: tuple[tuple[str, tuple[int, ...], str, str, str | None], ...]


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
        raise RatchetError(
            f"Vitest report {report_path} counter numTotalTestSuites is "
            "inconsistent with suite status counters"
        )
    return counters


def _normalize_failure_message(
    raw_message: object,
    package_root: Path,
    repository_root: Path | None = None,
) -> str:
    """Normalize checkout-specific package and repository roots in a failure."""

    if not isinstance(raw_message, str) or not raw_message.strip():
        raise RatchetError("failure has an empty or invalid diagnostic message")
    normalized = raw_message
    replacements = [(package_root.resolve(), "<PACKAGE_ROOT>")]
    if repository_root is not None:
        replacements.append((repository_root.resolve(), "<REPOSITORY_ROOT>"))
    variants = {
        variant: replacement
        for root, replacement in replacements
        for variant in (str(root), root.as_posix(), str(root).replace("/", "\\"))
        if variant
    }
    for root_variant in sorted(variants, key=len, reverse=True):
        normalized = normalized.replace(root_variant, variants[root_variant])
    lines = normalized.splitlines()
    stack_start = len(lines)
    while stack_start and _V8_STACK_FRAME_PATTERN.fullmatch(lines[stack_start - 1]):
        stack_start -= 1

    filtered_lines = lines[:stack_start]
    saw_stable_frame = False
    for line in lines[stack_start:]:
        is_volatile = any(
            pattern.fullmatch(line)
            for pattern in _VOLATILE_NODE_SCHEDULER_FRAME_PATTERNS
        )
        if is_volatile and saw_stable_frame:
            continue
        filtered_lines.append(line)
        if _V8_SOURCE_FRAME_PATTERN.fullmatch(line):
            saw_stable_frame = True
    normalized = "\n".join(filtered_lines)
    if not normalized.strip():
        raise RatchetError("failure has an empty diagnostic after normalization")
    return normalized


def _validate_safety_report(
    safety_report_path: Path,
    *,
    expected_reason: str,
    expected_module_count: int,
    expected_suite_count: int,
    expected_suite_status_counts: dict[str, int],
    expected_test_count: int,
) -> None:
    """Require a clean custom-reporter summary for a completed Vitest run."""

    payload = _load_json(safety_report_path)
    if payload.get("schemaVersion") != 2:
        raise RatchetError(
            f"Vitest safety report {safety_report_path} has an unsupported schemaVersion"
        )
    if payload.get("reason") != expected_reason:
        raise RatchetError(
            f"Vitest safety report {safety_report_path} has unexpected run reason"
        )

    counts = {
        key: _nonnegative_int(payload, key, safety_report_path)
        for key in (
            "moduleCount",
            "suiteCount",
            "passedSuiteCount",
            "failedSuiteCount",
            "pendingSuiteCount",
            "testCount",
            *_SAFETY_BLOCKING_COUNTERS,
        )
    }
    if counts["moduleCount"] != expected_module_count:
        raise RatchetError(
            f"Vitest safety report {safety_report_path} moduleCount does not match JSON"
        )
    if counts["suiteCount"] != expected_suite_count:
        raise RatchetError(
            f"Vitest safety report {safety_report_path} suiteCount does not match JSON"
        )
    if (
        counts["passedSuiteCount"]
        + counts["failedSuiteCount"]
        + counts["pendingSuiteCount"]
        != counts["suiteCount"]
    ):
        raise RatchetError(
            f"Vitest safety report {safety_report_path} has inconsistent suite counters"
        )
    for safety_counter, vitest_counter in (
        ("pendingSuiteCount", "numPendingTestSuites"),
        ("failedSuiteCount", "numFailedTestSuites"),
        ("passedSuiteCount", "numPassedTestSuites"),
    ):
        if counts[safety_counter] != expected_suite_status_counts[vitest_counter]:
            raise RatchetError(
                f"Vitest safety report {safety_report_path} {safety_counter} "
                "does not match JSON"
            )
    if counts["testCount"] != expected_test_count:
        raise RatchetError(
            f"Vitest safety report {safety_report_path} testCount does not match JSON"
        )
    for counter_name in _SAFETY_BLOCKING_COUNTERS:
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


def _assertion_suite_status(assertion_status: str) -> str:
    """Map assertion evidence without guessing pending parent-suite modes."""

    if assertion_status == "failed":
        return "failed"
    if assertion_status in {"passed", "skipped", "disabled"}:
        return "passed"
    return "unknown"


def _record_suite_status(
    suite_statuses: dict[tuple[str, ...], str],
    suite_key: tuple[str, ...],
    status: str,
) -> None:
    """Merge suite evidence with unknown below Vitest's known status precedence."""

    current_status = suite_statuses.get(suite_key)
    if (
        current_status is None
        or _SUITE_STATUS_RANK[status] > _SUITE_STATUS_RANK[current_status]
    ):
        suite_statuses[suite_key] = status


def _validate_observable_suite_counts(
    report_path: Path,
    counters: dict[str, int],
    suite_statuses: dict[tuple[str, ...], str],
) -> None:
    """Require raw counters to cover every suite observable in Vitest JSON."""

    observed_counts = Counter(suite_statuses.values())
    if len(suite_statuses) > counters["numTotalTestSuites"]:
        raise RatchetError(
            f"Vitest report {report_path} exposes more suites than its total counter"
        )
    for status, counter_name in (
        ("passed", "numPassedTestSuites"),
        ("failed", "numFailedTestSuites"),
        ("pending", "numPendingTestSuites"),
    ):
        if observed_counts[status] > counters[counter_name]:
            raise RatchetError(
                f"Vitest report {report_path} counter {counter_name} cannot cover "
                "its observable suite hierarchy"
            )


def _load_report_context(report_path: Path, package_root: Path) -> ReportContext:
    """Read and reconcile one ordered Vitest execution context."""

    payload = _load_json(report_path)
    if not isinstance(payload.get("success"), bool):
        raise RatchetError(f"Vitest report {report_path} is missing boolean success")
    test_results = payload.get("testResults")
    if not isinstance(test_results, list):
        raise RatchetError(f"Vitest report {report_path} is missing testResults")
    counters = {
        key: _nonnegative_int(payload, key, report_path) for key in _STRICT_COUNTERS
    }

    files: list[str] = []
    tests_by_file: dict[str, list[tuple[str, str]]] = {}
    assertion_status_counts: Counter[str] = Counter()
    suite_statuses: dict[tuple[str, ...], str] = {}
    for test_result in test_results:
        if not isinstance(test_result, dict):
            raise RatchetError(f"Vitest report {report_path} has an invalid test result")
        relative_path = _relative_test_path(test_result.get("name"), package_root)
        test_result_status = test_result.get("status")
        if test_result_status not in _TEST_RESULT_STATUSES:
            raise RatchetError(
                f"Vitest result {relative_path} has an invalid test result status"
            )
        assertion_results = test_result.get("assertionResults")
        if not isinstance(assertion_results, list):
            raise RatchetError(
                f"Vitest result {relative_path} is missing assertionResults"
            )
        files.append(relative_path)
        tests_by_file[relative_path] = []
        _record_suite_status(
            suite_statuses,
            (relative_path,),
            test_result_status,
        )
        for assertion in assertion_results:
            if not isinstance(assertion, dict):
                raise RatchetError(
                    f"Vitest result {relative_path} has an invalid assertion"
                )
            assertion_status = assertion.get("status")
            if assertion_status == "pending":
                raise RatchetError(
                    f"Vitest result {relative_path} contains an unfinished assertion"
                )
            if assertion_status not in _ASSERTION_STATUSES:
                raise RatchetError(
                    f"Vitest result {relative_path} has an invalid assertion status"
                )
            full_name = assertion.get("fullName")
            if not isinstance(full_name, str) or not full_name.strip():
                raise RatchetError(
                    f"Vitest result {relative_path} has an invalid test identity"
                )
            ancestor_titles = assertion.get("ancestorTitles")
            if (
                not isinstance(ancestor_titles, list)
                or any(
                    not isinstance(title, str) or not title.strip()
                    for title in ancestor_titles
                )
            ):
                raise RatchetError(
                    f"Vitest result {relative_path} has invalid ancestorTitles"
                )

            assertion_status_counts[assertion_status] += 1
            tests_by_file[relative_path].append((full_name, assertion_status))
            suite_status = _assertion_suite_status(assertion_status)
            suite_keys = [(relative_path,)]
            suite_keys.extend(
                (relative_path, *ancestor_titles[:depth])
                for depth in range(1, len(ancestor_titles) + 1)
            )
            for suite_key in suite_keys:
                _record_suite_status(suite_statuses, suite_key, suite_status)

    if len(files) != len(set(files)):
        raise RatchetError(f"Vitest report {report_path} repeats a module path")
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
                "match report structure"
            )
    if (
        counters["numPassedTestSuites"]
        + counters["numFailedTestSuites"]
        + counters["numPendingTestSuites"]
        != counters["numTotalTestSuites"]
    ):
        raise RatchetError(
            f"Vitest report {report_path} counter numTotalTestSuites is "
            "inconsistent with suite status counters"
        )
    _validate_observable_suite_counts(report_path, counters, suite_statuses)
    return ReportContext(
        files=tuple(sorted(files)),
        tests=tuple(
            (relative_path, tuple(tests_by_file[relative_path]))
            for relative_path in sorted(tests_by_file)
        ),
        suites=tuple(sorted(suite_statuses.items())),
        counters=tuple((key, counters[key]) for key in _STRICT_COUNTERS),
    )


def _load_execution_context(
    order_report_path: Path,
    package_root: Path,
    report_context: ReportContext,
) -> ExecutionContext:
    """Validate reporter-observed modules and runtime suites against Vitest JSON."""

    payload = _load_json(order_report_path)
    schema_version = payload.get("schemaVersion")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in {1, 2}
    ):
        raise RatchetError(
            f"Vitest order report {order_report_path} has an unsupported schemaVersion"
        )
    module_count = _nonnegative_int(payload, "moduleCount", order_report_path)
    raw_modules = payload.get("modules")
    if not isinstance(raw_modules, list):
        raise RatchetError(
            f"Vitest order report {order_report_path} is missing modules"
        )
    modules = tuple(
        _relative_test_path(raw_module, package_root) for raw_module in raw_modules
    )
    if module_count != len(modules):
        raise RatchetError(
            f"Vitest order report {order_report_path} moduleCount does not match modules"
        )
    if len(modules) != len(set(modules)):
        raise RatchetError(f"Vitest order report {order_report_path} repeats a module")
    if set(modules) != set(report_context.files):
        raise RatchetError(
            f"Vitest order report {order_report_path} order manifest does not match JSON"
        )
    counters = dict(report_context.counters)
    if schema_version == 1:
        if counters["numTotalTestSuites"] != len(report_context.suites):
            raise RatchetError(
                f"Vitest order report {order_report_path} cannot prove counter "
                "numTotalTestSuites with schemaVersion 1"
            )
        return ExecutionContext(modules=modules, suites=())

    suite_count = _nonnegative_int(payload, "suiteCount", order_report_path)
    raw_suites = payload.get("suites")
    if not isinstance(raw_suites, list):
        raise RatchetError(
            f"Vitest order report {order_report_path} is missing runtime suites"
        )
    runtime_suites: dict[
        tuple[str, tuple[int, ...]], tuple[str, str, str | None]
    ] = {}
    for raw_suite in raw_suites:
        if not isinstance(raw_suite, dict):
            raise RatchetError(
                f"Vitest order report {order_report_path} has an invalid runtime suite"
            )
        module = _relative_test_path(raw_suite.get("module"), package_root)
        if module not in report_context.files:
            raise RatchetError(
                f"Vitest order report {order_report_path} suite references an unknown module"
            )
        raw_path = raw_suite.get("path")
        if (
            not isinstance(raw_path, list)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in raw_path
            )
        ):
            raise RatchetError(
                f"Vitest order report {order_report_path} has an invalid suite path"
            )
        path = tuple(raw_path)
        name = raw_suite.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RatchetError(
                f"Vitest order report {order_report_path} has an invalid suite name"
            )
        state = raw_suite.get("state")
        if state not in _EXECUTION_SUITE_STATES:
            raise RatchetError(
                f"Vitest order report {order_report_path} has an invalid suite state"
            )
        if state in {"pending", "queued"}:
            raise RatchetError(
                f"Vitest order report {order_report_path} contains an unfinished suite"
            )
        mode = raw_suite.get("mode")
        if "mode" not in raw_suite or (
            (not path and mode is not None)
            or (path and mode not in _EXECUTION_SUITE_MODES)
        ):
            raise RatchetError(
                f"Vitest order report {order_report_path} has an invalid suite mode"
            )
        key = (module, path)
        if key in runtime_suites:
            raise RatchetError(
                f"Vitest order report {order_report_path} repeats a suite path"
            )
        runtime_suites[key] = (name, state, mode)

    if suite_count != len(runtime_suites):
        raise RatchetError(
            f"Vitest order report {order_report_path} suiteCount does not match suites"
        )
    if suite_count != counters["numTotalTestSuites"]:
        raise RatchetError(
            f"Vitest order report {order_report_path} suiteCount does not match JSON"
        )

    runtime_observable: dict[tuple[str, ...], str] = {}
    runtime_status_counts: Counter[str] = Counter()
    for (module, path), (name, state, mode) in runtime_suites.items():
        if not path:
            if name != module:
                raise RatchetError(
                    f"Vitest order report {order_report_path} has an invalid module root suite"
                )
            observable_key = (module,)
        else:
            parent_key = (module, path[:-1])
            if parent_key not in runtime_suites:
                raise RatchetError(
                    f"Vitest order report {order_report_path} has an orphaned suite"
                )
            ancestor_names = []
            current_path = path
            while current_path:
                ancestor_names.append(runtime_suites[(module, current_path)][0])
                current_path = current_path[:-1]
            observable_key = (module, *reversed(ancestor_names))
        aggregate_status = (
            "failed"
            if state == "failed"
            else "pending"
            if state in {"pending", "queued"} or mode == "todo"
            else "passed"
        )
        runtime_status_counts[aggregate_status] += 1
        _record_suite_status(runtime_observable, observable_key, aggregate_status)

    if {(module, ()) for module in report_context.files} - runtime_suites.keys():
        raise RatchetError(
            f"Vitest order report {order_report_path} omits a module root suite"
        )
    for status, counter_name in (
        ("passed", "numPassedTestSuites"),
        ("failed", "numFailedTestSuites"),
        ("pending", "numPendingTestSuites"),
    ):
        if runtime_status_counts[status] != counters[counter_name]:
            raise RatchetError(
                f"Vitest order report {order_report_path} {status} suite count "
                "does not match JSON"
            )
    for observable_key, status in report_context.suites:
        runtime_status = runtime_observable.get(observable_key)
        if runtime_status is None or (
            status != "unknown" and runtime_status != status
        ):
            raise RatchetError(
                f"Vitest order report {order_report_path} does not match JSON suite hierarchy"
            )

    return ExecutionContext(
        modules=modules,
        suites=tuple(
            sorted(
                (module, path, name, state, mode)
                for (module, path), (name, state, mode) in runtime_suites.items()
            )
        ),
    )


def _load_failures(
    report_path: Path,
    package_root: Path,
    *,
    strict: bool = False,
    safety_report_path: Path | None = None,
    repository_root: Path | None = None,
) -> tuple[FailedTest, ...]:
    """Read comparable failure identities from a Vitest JSON report."""

    payload = _load_json(report_path)
    if not isinstance(payload.get("success"), bool):
        raise RatchetError(f"Vitest report {report_path} is missing boolean success")
    test_results = payload.get("testResults")
    if not isinstance(test_results, list):
        raise RatchetError(f"Vitest report {report_path} is missing testResults")
    counters = _strict_counters(payload, report_path) if strict else None

    failures: list[FailedTest] = []
    assertion_status_counts: Counter[str] = Counter()
    suite_statuses: dict[tuple[str, ...], str] = {}
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
            _record_suite_status(
                suite_statuses,
                (relative_path,),
                test_result_status,
            )
        failed_assertions = []
        for assertion in assertion_results:
            if not isinstance(assertion, dict):
                raise RatchetError(
                    f"Vitest result {relative_path} has an invalid assertion"
                )
            assertion_status = assertion.get("status")
            if assertion_status == "pending":
                raise RatchetError(
                    f"Vitest result {relative_path} contains an unfinished assertion"
                )
            if assertion_status not in _ASSERTION_STATUSES:
                raise RatchetError(
                    f"Vitest result {relative_path} has an invalid assertion status: "
                    f"{assertion_status!r}"
                )
            assertion_status_counts[assertion_status] += 1
            if strict:
                ancestor_titles = assertion.get("ancestorTitles")
                if (
                    not isinstance(ancestor_titles, list)
                    or any(
                        not isinstance(title, str) or not title.strip()
                        for title in ancestor_titles
                    )
                ):
                    raise RatchetError(
                        f"Vitest result {relative_path} has invalid ancestorTitles"
                    )
                suite_status = _assertion_suite_status(assertion_status)
                suite_keys = [(relative_path,)]
                suite_keys.extend(
                    (relative_path, *ancestor_titles[:depth])
                    for depth in range(1, len(ancestor_titles) + 1)
                )
                for suite_key in suite_keys:
                    _record_suite_status(suite_statuses, suite_key, suite_status)
            if assertion_status == "failed":
                failed_assertions.append(assertion)

        if test_result_status == "failed" and not failed_assertions and assertion_results:
            raise RatchetError(
                f"file-level failure in {relative_path} after assertion "
                "collection cannot be ratcheted"
            )
        if test_result_status == "passed" and failed_assertions:
            raise RatchetError(
                f"passed Vitest result {relative_path} contains failed assertions"
            )
        if "message" not in test_result:
            raise RatchetError(
                f"Vitest result {relative_path} is missing file-level message"
            )
        file_message = test_result["message"]
        if not isinstance(file_message, str):
            raise RatchetError(
                f"Vitest result {relative_path} has an invalid file-level message"
            )
        allows_collection_diagnostic = (
            not strict
            and test_result_status == "failed"
            and not assertion_results
        )
        if file_message.strip() and not allows_collection_diagnostic:
            raise RatchetError(
                f"Vitest result {relative_path} contains a file-level error"
            )
        if test_result_status == "failed" and not failed_assertions:
            if not file_message.strip():
                raise RatchetError(
                    f"collection-level failure in {relative_path} is missing "
                    "an error message"
                )
            failures.append(
                FailedTest(
                    file=relative_path,
                    full_name=_COLLECTION_FAILURE_FULL_NAME,
                    failure_messages=(
                        _normalize_failure_message(
                            file_message,
                            package_root,
                            repository_root,
                        ),
                    ),
                )
            )

        for assertion in failed_assertions:
            full_name = assertion.get("fullName")
            if not isinstance(full_name, str) or not full_name.strip():
                raise RatchetError(
                    f"failed assertion in {relative_path} is missing fullName"
                )
            raw_messages = assertion.get("failureMessages")
            if not isinstance(raw_messages, list) or not raw_messages:
                raise RatchetError(
                    f"failed assertion in {relative_path} is missing failureMessages"
                )
            failure_messages = tuple(
                _normalize_failure_message(
                    message,
                    package_root,
                    repository_root,
                )
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
        if counters is None:
            raise RatchetError("strict Vitest validation is missing counters")
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
        _validate_observable_suite_counts(report_path, counters, suite_statuses)
        if safety_report_path is None:
            raise RatchetError("strict Vitest validation requires a safety report")
        _validate_safety_report(
            safety_report_path,
            expected_reason="passed" if payload["success"] else "failed",
            expected_module_count=len(test_results),
            expected_suite_count=counters["numTotalTestSuites"],
            expected_suite_status_counts=counters,
            expected_test_count=counters["numTotalTests"],
        )

        if payload["success"] and (
            counters["numFailedTestSuites"] or counters["numFailedTests"]
        ):
            raise RatchetError(
                f"Vitest report {report_path} claims success but has failed counters"
            )

    if payload["success"] and failures:
        raise RatchetError(
            f"Vitest report {report_path} claims success but contains failures"
        )
    if not payload["success"] and not failures:
        raise RatchetError(
            f"Vitest report {report_path} failed without comparable failures"
        )
    return tuple(sorted(failures))


def failing_test_files(
    report_path: Path,
    package_root: Path,
    *,
    strict: bool = False,
    safety_report_path: Path | None = None,
) -> tuple[str, ...]:
    """Return sorted package-relative files containing comparable failures."""

    failures = _load_failures(
        report_path,
        package_root,
        strict=strict,
        safety_report_path=safety_report_path,
    )
    return tuple(sorted({failure.file for failure in failures}))


def test_result_files(
    report_path: Path,
    package_root: Path,
    order_report_path: Path,
) -> tuple[str, ...]:
    """Return the reporter-observed normalized module execution order."""

    context = _load_report_context(report_path, package_root)
    return _load_execution_context(order_report_path, package_root, context).modules


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


def _repository_root(package_root: Path, package_repo_path: Path) -> Path:
    """Derive and verify a checkout root from its package path suffix."""

    resolved_package_root = package_root.resolve()
    repository_root = resolved_package_root
    for _ in package_repo_path.parts:
        repository_root = repository_root.parent
    if (repository_root / package_repo_path).resolve() != resolved_package_root:
        raise RatchetError(
            f"package root {package_root} does not end with {package_repo_path}"
        )
    return repository_root


def _validate_equivalent_context(
    *,
    head_report: Path,
    base_report: Path,
    head_package_root: Path,
    base_package_root: Path,
    head_order_report_path: Path,
    base_order_report_path: Path,
) -> None:
    """Require equal runtime order, identities, and counters for context replay."""

    head_context = _load_report_context(head_report, head_package_root)
    base_context = _load_report_context(base_report, base_package_root)
    if head_context.files != base_context.files:
        raise RatchetError("head and base Vitest module manifests differ")
    head_execution = _load_execution_context(
        head_order_report_path,
        head_package_root,
        head_context,
    )
    base_execution = _load_execution_context(
        base_order_report_path,
        base_package_root,
        base_context,
    )
    if head_execution.modules != base_execution.modules:
        raise RatchetError("head and base Vitest module execution order differs")
    if head_context.tests != base_context.tests:
        raise RatchetError("head and base Vitest test identities or file-local order differ")
    if head_context.suites != base_context.suites:
        raise RatchetError("head and base Vitest suite identities differ")
    if head_execution.suites != base_execution.suites:
        raise RatchetError("head and base Vitest runtime suite trees differ")
    if head_context.counters != base_context.counters:
        raise RatchetError("head and base Vitest test or suite counts differ")


def compare_reports(
    *,
    head_report: Path,
    base_report: Path,
    head_package_root: Path,
    base_package_root: Path,
    package_repo_path: Path,
    changed_files_path: Path,
    strict: bool = False,
    require_equivalent_context: bool = False,
    head_safety_report_path: Path | None = None,
    base_safety_report_path: Path | None = None,
    head_order_report_path: Path | None = None,
    base_order_report_path: Path | None = None,
) -> RatchetResult:
    """Classify head failures, blocking new or test-file-modified failures.

    Args:
        head_report: Vitest JSON report produced from the pull-request head.
        base_report: Vitest JSON report produced from the base revision.
        head_package_root: Package root used to normalize head test paths.
        base_package_root: Package root used to normalize base test paths.
        package_repo_path: Repository-relative path to the tested package.
        changed_files_path: File containing repository-relative changed paths.
        strict: Require complete reports, exact failure fingerprints, and
            NUL-delimited changed paths.
        require_equivalent_context: Require equal module manifests, runtime
            order, test identities, suite identities, and reconciled counters.
        head_safety_report_path: Reporter lifecycle evidence for the head run.
        base_safety_report_path: Reporter lifecycle evidence for the base run.
        head_order_report_path: Reporter-observed head module execution order.
        base_order_report_path: Reporter-observed base module execution order.

    Returns:
        The inherited failures and blocking regressions from the head report.

    Raises:
        RatchetError: If a report, path list, or strict safety contract is
            missing, malformed, incomplete, or inconsistent.
    """

    package_path = _normalized_package_repo_path(package_repo_path)
    head_repository_root = _repository_root(head_package_root, package_path)
    base_repository_root = _repository_root(base_package_root, package_path)
    if require_equivalent_context:
        if head_order_report_path is None or base_order_report_path is None:
            raise RatchetError(
                "equivalent context requires head and base execution-order reports"
            )
        _validate_equivalent_context(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_package_root,
            base_package_root=base_package_root,
            head_order_report_path=head_order_report_path,
            base_order_report_path=base_order_report_path,
        )

    head_failures = _load_failures(
        head_report,
        head_package_root,
        strict=strict,
        safety_report_path=head_safety_report_path,
        repository_root=head_repository_root,
    )
    if not head_failures:
        raise RatchetError("head report has no failures to compare")
    loaded_base_failures = _load_failures(
        base_report,
        base_package_root,
        strict=strict,
        safety_report_path=base_safety_report_path,
        repository_root=base_repository_root,
    )
    base_failures = Counter(loaded_base_failures)
    changed_files = _load_changed_files(changed_files_path, strict=strict)
    inherited: list[FailedTest] = []
    regressions: list[FailedTest] = []
    for failure in head_failures:
        repo_test_path = (package_path / failure.file).as_posix()
        present_in_base = base_failures[failure] > 0
        if repo_test_path in changed_files or not present_in_base:
            regressions.append(failure)
        else:
            inherited.append(failure)
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

    extract_context = subparsers.add_parser("extract-context")
    extract_context.add_argument("--report", type=Path, required=True)
    extract_context.add_argument("--order-report", type=Path, required=True)
    extract_context.add_argument("--package-root", type=Path, required=True)
    extract_context.add_argument("--output", type=Path, required=True)

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
    compare.add_argument("--require-equivalent-context", action="store_true")
    compare.add_argument("--head-safety-report", type=Path)
    compare.add_argument("--base-safety-report", type=Path)
    compare.add_argument("--head-order-report", type=Path)
    compare.add_argument("--base-order-report", type=Path)
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
        raise RatchetError("failed Vitest process produced no comparable failures")
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
        require_equivalent_context=args.require_equivalent_context,
        head_safety_report_path=args.head_safety_report,
        base_safety_report_path=args.base_safety_report,
        head_order_report_path=args.head_order_report,
        base_order_report_path=args.base_order_report,
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


def _run_extract_context(args: argparse.Namespace) -> int:
    """Write the exact executed module manifest for context replay."""

    files = test_result_files(args.report, args.package_root, args.order_report)
    if not files:
        raise RatchetError("Vitest report produced no executed module context")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{path}\n" for path in files), encoding="utf-8")
    # frontend-unit-tests runs this helper before project Python dependencies exist.
    # Keep this status line on the standard-library stream instead of importing Loguru.
    sys.stdout.write(f"[vitest-ratchet] context_files={len(files)}\n")
    return 0


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
        if args.command == "extract-context":
            return _run_extract_context(args)
        if args.command == "validate-success":
            return _run_validate_success(args)
        return _run_compare(args)
    except RatchetError as exc:
        print(f"[vitest-ratchet] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

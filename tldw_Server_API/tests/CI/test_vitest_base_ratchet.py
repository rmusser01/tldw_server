import json
from pathlib import Path

import pytest
from Helper_Scripts.ci import vitest_base_ratchet
from Helper_Scripts.ci.vitest_base_ratchet import (
    RatchetError,
    compare_reports,
    failing_test_files,
    validate_success_report,
)


def _write_report(
    path: Path,
    package_root: Path,
    failures: dict[str, list[str]],
    *,
    collection_failures: tuple[str, ...] = (),
    failure_messages: dict[str, list[str]] | None = None,
) -> None:
    test_results = []
    failed_test_count = 0
    for relative_path, full_names in failures.items():
        failed_test_count += len(full_names)
        messages = (failure_messages or {}).get(relative_path)
        if messages is not None and len(messages) != len(full_names):
            raise ValueError("failure message count must match failed assertion count")
        test_results.append(
            {
                "name": str(package_root / relative_path),
                "status": "failed",
                "assertionResults": [
                    {
                        "title": full_name,
                        "fullName": full_name,
                        "status": "failed",
                        "ancestorTitles": [],
                        "failureMessages": [
                            messages[index]
                            if messages is not None
                            else "expected true to be false"
                        ],
                    }
                    for index, full_name in enumerate(full_names)
                ],
                "message": "",
            }
        )
    for relative_path in collection_failures:
        test_results.append(
            {
                "name": str(package_root / relative_path),
                "status": "failed",
                "assertionResults": [],
                "message": "failed to import test module",
            }
        )

    payload = {
        "success": not test_results,
        "numTotalTestSuites": len(test_results),
        "numPassedTestSuites": 0,
        "numFailedTestSuites": len(test_results),
        "numPendingTestSuites": 0,
        "numTotalTests": failed_test_count,
        "numPassedTests": 0,
        "numFailedTests": failed_test_count,
        "numPendingTests": 0,
        "numTodoTests": 0,
        "testResults": test_results,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_success_report(path: Path, package_root: Path) -> None:
    payload = {
        "success": True,
        "numTotalTestSuites": 1,
        "numPassedTestSuites": 1,
        "numFailedTestSuites": 0,
        "numPendingTestSuites": 0,
        "numTotalTests": 1,
        "numPassedTests": 1,
        "numFailedTests": 0,
        "numPendingTests": 0,
        "numTodoTests": 0,
        "testResults": [
            {
                "name": str(package_root / "src/passed.test.ts"),
                "status": "passed",
                "assertionResults": [
                    {
                        "fullName": "suite passes",
                        "status": "passed",
                        "ancestorTitles": [],
                        "failureMessages": [],
                    }
                ],
                "message": "",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _append_passed_result(
    path: Path,
    package_root: Path,
    relative_path: str,
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["numTotalTestSuites"] += 1
    payload["numPassedTestSuites"] += 1
    payload["numTotalTests"] += 1
    payload["numPassedTests"] += 1
    payload["testResults"].append(
        {
            "name": str(package_root / relative_path),
            "status": "passed",
            "assertionResults": [
                {
                    "fullName": f"{relative_path} passes",
                    "status": "passed",
                    "ancestorTitles": [],
                    "failureMessages": [],
                }
            ],
            "message": "",
        }
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_safety_report(
    path: Path,
    *,
    reason: str,
    test_count: int,
    module_count: int = 1,
    suite_count: int | None = None,
    passed_suite_count: int | None = None,
    failed_suite_count: int | None = None,
    pending_suite_count: int = 0,
    incomplete_suite_count: int = 0,
    incomplete_test_count: int = 0,
    unhandled_error_count: int = 0,
    module_error_count: int = 0,
    hook_error_count: int = 0,
) -> None:
    resolved_suite_count = module_count if suite_count is None else suite_count
    resolved_failed_suite_count = (
        1 if reason == "failed" else 0
    ) if failed_suite_count is None else failed_suite_count
    resolved_passed_suite_count = (
        resolved_suite_count - resolved_failed_suite_count - pending_suite_count
        if passed_suite_count is None
        else passed_suite_count
    )
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "reason": reason,
                "moduleCount": module_count,
                "suiteCount": resolved_suite_count,
                "passedSuiteCount": resolved_passed_suite_count,
                "failedSuiteCount": resolved_failed_suite_count,
                "pendingSuiteCount": pending_suite_count,
                "incompleteSuiteCount": incomplete_suite_count,
                "incompleteTestCount": incomplete_test_count,
                "testCount": test_count,
                "unhandledErrorCount": unhandled_error_count,
                "moduleErrorCount": module_error_count,
                "hookErrorCount": hook_error_count,
            }
        ),
        encoding="utf-8",
    )


def _write_order_report(path: Path, modules: tuple[str, ...]) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "moduleCount": len(modules),
                "modules": list(modules),
            }
        ),
        encoding="utf-8",
    )


def _write_structured_order_report(
    path: Path,
    modules: tuple[str, ...],
    failures: list[dict[str, object]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 3,
                "moduleCount": len(modules),
                "modules": list(modules),
                "suiteCount": len(modules),
                "suites": [
                    {
                        "module": module,
                        "path": [],
                        "name": module,
                        "state": "failed",
                        "mode": None,
                    }
                    for module in modules
                ],
                "failureCount": len(failures),
                "failures": failures,
            }
        ),
        encoding="utf-8",
    )


def _structured_failure(
    module: str,
    full_name: str,
    message: str,
    stacks: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "module": module,
        "ancestorTitles": [],
        "title": full_name,
        "fullName": full_name,
        "errors": [
            {
                "name": "AssertionError",
                "message": message,
                "stacks": stacks,
            }
        ],
    }


def test_failing_test_files_normalizes_and_sorts_package_paths(tmp_path: Path) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    _write_report(
        report_path,
        package_root,
        {
            "src/z.test.ts": ["suite z"],
            "src/a.test.ts": ["suite a", "suite a second"],
        },
    )

    assert failing_test_files(report_path, package_root) == (
        "src/a.test.ts",
        "src/z.test.ts",
    )


@pytest.mark.unit
def test_failing_test_files_extracts_collection_failure_path(tmp_path: Path) -> None:
    """Replay a collection failure only through its owning test file."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    _write_report(
        report_path,
        package_root,
        {},
        collection_failures=("src/import-error.test.ts",),
    )

    assert failing_test_files(report_path, package_root) == (
        "src/import-error.test.ts",
    )


@pytest.mark.unit
def test_test_result_files_uses_runtime_order_not_json_report_order(
    tmp_path: Path,
) -> None:
    """Extract every module in reporter-observed runtime order."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    order_report_path = tmp_path / "head-order.json"
    _write_report(
        report_path,
        package_root,
        {"src/z-failed.test.ts": ["suite fails"]},
    )
    _append_passed_result(report_path, package_root, "src/a-passed.test.ts")
    _write_order_report(
        order_report_path,
        ("src/a-passed.test.ts", "src/z-failed.test.ts"),
    )

    assert vitest_base_ratchet.test_result_files(
        report_path,
        package_root,
        order_report_path,
    ) == (
        "src/a-passed.test.ts",
        "src/z-failed.test.ts",
    )


@pytest.mark.unit
def test_test_result_files_rejects_order_manifest_mismatch(tmp_path: Path) -> None:
    """Reject an order sidecar that omits a JSON-reported module."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    order_report_path = tmp_path / "head-order.json"
    _write_report(
        report_path,
        package_root,
        {"src/failed.test.ts": ["suite fails"]},
    )
    _append_passed_result(report_path, package_root, "src/passed.test.ts")
    _write_order_report(order_report_path, ("src/failed.test.ts",))

    with pytest.raises(RatchetError, match="order.*manifest"):
        vitest_base_ratchet.test_result_files(
            report_path,
            package_root,
            order_report_path,
        )


@pytest.mark.unit
def test_test_result_files_rejects_float_order_schema_version(tmp_path: Path) -> None:
    """Require the order sidecar schema version to be an integer."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    order_report_path = tmp_path / "head-order.json"
    _write_report(
        report_path,
        package_root,
        {"src/failed.test.ts": ["suite fails"]},
    )
    _write_order_report(order_report_path, ("src/failed.test.ts",))
    order_payload = json.loads(order_report_path.read_text(encoding="utf-8"))
    order_payload["schemaVersion"] = 1.0
    order_report_path.write_text(json.dumps(order_payload), encoding="utf-8")

    with pytest.raises(RatchetError, match="unsupported schemaVersion"):
        vitest_base_ratchet.test_result_files(
            report_path,
            package_root,
            order_report_path,
        )


@pytest.mark.unit
def test_test_result_files_accepts_real_all_skipped_vitest_fixture(
    tmp_path: Path,
) -> None:
    """Match Vitest 4 suite counters for a completed describe.skip file."""

    package_root = tmp_path / "head" / "apps" / "tldw-frontend"
    relative_path = "extension/__tests__/writing-utils.test.ts"
    fixture_path = (
        Path(__file__).parent / "fixtures" / "vitest-all-skipped.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    payload["testResults"][0]["name"] = str(package_root / relative_path)
    report_path = tmp_path / "head.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    order_report_path = tmp_path / "head-order.json"
    _write_order_report(order_report_path, (relative_path,))

    assert vitest_base_ratchet.test_result_files(
        report_path,
        package_root,
        order_report_path,
    ) == (relative_path,)

@pytest.mark.unit
def test_test_result_files_accepts_reporter_proven_hidden_suites(
    tmp_path: Path,
) -> None:
    """Use runtime suite evidence for suites omitted from Vitest JSON assertions."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    relative_path = "src/empty-suite.test.ts"
    report_path = tmp_path / "head.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["success"] = False
    payload["numTotalTestSuites"] = 4
    payload["numPassedTestSuites"] = 2
    payload["numFailedTestSuites"] = 2
    payload["testResults"][0]["name"] = str(package_root / relative_path)
    payload["testResults"][0]["status"] = "failed"
    payload["testResults"][0]["assertionResults"][0]["ancestorTitles"] = [
        "visible suite"
    ]
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    order_report_path = tmp_path / "head-order.json"
    order_report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "moduleCount": 1,
                "modules": [relative_path],
                "suiteCount": 4,
                "suites": [
                    {
                        "module": relative_path,
                        "path": [],
                        "name": relative_path,
                        "state": "failed",
                        "mode": None,
                    },
                    {
                        "module": relative_path,
                        "path": [0],
                        "name": "visible suite",
                        "state": "passed",
                        "mode": "run",
                    },
                    {
                        "module": relative_path,
                        "path": [1],
                        "name": "empty suite",
                        "state": "failed",
                        "mode": "run",
                    },
                    {
                        "module": relative_path,
                        "path": [2],
                        "name": "empty skipped suite",
                        "state": "skipped",
                        "mode": "skip",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert vitest_base_ratchet.test_result_files(
        report_path,
        package_root,
        order_report_path,
    ) == (relative_path,)


@pytest.mark.unit
def test_test_result_files_classifies_todo_suite_as_pending(tmp_path: Path) -> None:
    """Use suite mode to distinguish todo from ordinary skipped suites."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    relative_path = "src/todo-suite.test.ts"
    report_path = tmp_path / "head.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assertion = payload["testResults"][0]["assertionResults"][0]
    payload["testResults"][0]["name"] = str(package_root / relative_path)
    assertion["fullName"] = "todo suite pending case"
    assertion["status"] = "todo"
    assertion["ancestorTitles"] = ["todo suite"]
    payload["numTotalTestSuites"] = 2
    payload["numPassedTestSuites"] = 1
    payload["numPendingTestSuites"] = 1
    payload["numPassedTests"] = 0
    payload["numTodoTests"] = 1
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    order_report_path = tmp_path / "head-order.json"
    order_report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "moduleCount": 1,
                "modules": [relative_path],
                "suiteCount": 2,
                "suites": [
                    {
                        "module": relative_path,
                        "path": [],
                        "name": relative_path,
                        "state": "passed",
                        "mode": None,
                    },
                    {
                        "module": relative_path,
                        "path": [0],
                        "name": "todo suite",
                        "state": "skipped",
                        "mode": "todo",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert vitest_base_ratchet.test_result_files(
        report_path,
        package_root,
        order_report_path,
    ) == (relative_path,)

    order_payload = json.loads(order_report_path.read_text(encoding="utf-8"))
    order_payload["suites"][1]["mode"] = "skip"
    order_report_path.write_text(json.dumps(order_payload), encoding="utf-8")
    with pytest.raises(RatchetError, match="suite count does not match JSON"):
        vitest_base_ratchet.test_result_files(
            report_path,
            package_root,
            order_report_path,
        )

    order_payload["suites"][1].update({"state": "pending", "mode": "run"})
    order_report_path.write_text(json.dumps(order_payload), encoding="utf-8")
    with pytest.raises(RatchetError, match="unfinished suite"):
        vitest_base_ratchet.test_result_files(
            report_path,
            package_root,
            order_report_path,
        )

    order_payload["suites"][1].update({"state": "skipped", "mode": "todo"})
    order_report_path.write_text(json.dumps(order_payload), encoding="utf-8")
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["testResults"][0]["assertionResults"][0]["status"] = "pending"
    report_payload["numPendingTests"] = 1
    report_payload["numTodoTests"] = 0
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")
    with pytest.raises(RatchetError, match="unfinished assertion"):
        vitest_base_ratchet.test_result_files(
            report_path,
            package_root,
            order_report_path,
        )


@pytest.mark.unit
def test_vitest_todo_suite_shapes_match_raw_status_counters(tmp_path: Path) -> None:
    """Accept Vitest's distinct todo-only parent and describe.todo semantics."""

    package_root = tmp_path / "head" / "admin-ui"
    relative_path = "src/todo-shapes.test.ts"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    order_report_path = tmp_path / "head-order.json"
    report_path.write_text(
        json.dumps(
            {
                "success": True,
                "numTotalTestSuites": 4,
                "numPassedTestSuites": 3,
                "numFailedTestSuites": 0,
                "numPendingTestSuites": 1,
                "numTotalTests": 4,
                "numPassedTests": 1,
                "numFailedTests": 0,
                "numPendingTests": 0,
                "numTodoTests": 3,
                "testResults": [
                    {
                        "name": str(package_root / relative_path),
                        "status": "passed",
                        "assertionResults": [
                            {
                                "fullName": "todo-only parent future behavior",
                                "status": "todo",
                                "ancestorTitles": ["todo-only parent"],
                                "failureMessages": [],
                            },
                            {
                                "fullName": "mixed parent current behavior",
                                "status": "passed",
                                "ancestorTitles": ["mixed parent"],
                                "failureMessages": [],
                            },
                            {
                                "fullName": "mixed parent future behavior",
                                "status": "todo",
                                "ancestorTitles": ["mixed parent"],
                                "failureMessages": [],
                            },
                            {
                                "fullName": "todo suite never collected",
                                "status": "todo",
                                "ancestorTitles": ["todo suite"],
                                "failureMessages": [],
                            },
                        ],
                        "message": "",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_safety_report(
        safety_path,
        reason="passed",
        test_count=4,
        suite_count=4,
        passed_suite_count=3,
        pending_suite_count=1,
    )
    order_report_path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "moduleCount": 1,
                "modules": [relative_path],
                "suiteCount": 4,
                "suites": [
                    {
                        "module": relative_path,
                        "path": [],
                        "name": relative_path,
                        "state": "passed",
                        "mode": None,
                    },
                    {
                        "module": relative_path,
                        "path": [0],
                        "name": "todo-only parent",
                        "state": "skipped",
                        "mode": "skip",
                    },
                    {
                        "module": relative_path,
                        "path": [1],
                        "name": "mixed parent",
                        "state": "passed",
                        "mode": "run",
                    },
                    {
                        "module": relative_path,
                        "path": [2],
                        "name": "todo suite",
                        "state": "skipped",
                        "mode": "todo",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert vitest_base_ratchet.test_result_files(
        report_path,
        package_root,
        order_report_path,
    ) == (relative_path,)
    validate_success_report(
        report_path,
        package_root,
        strict=True,
        safety_report_path=safety_path,
    )


@pytest.mark.unit
def test_test_result_files_rejects_newline_module_path(tmp_path: Path) -> None:
    """Reject module names that cannot be transported one path per line."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    order_report_path = tmp_path / "head-order.json"
    _write_report(
        report_path,
        package_root,
        {"src/unsafe\nmodule.test.ts": ["suite fails"]},
    )
    _write_order_report(order_report_path, ("src/unsafe\nmodule.test.ts",))

    with pytest.raises(RatchetError, match="cannot contain newlines"):
        vitest_base_ratchet.test_result_files(
            report_path,
            package_root,
            order_report_path,
        )


@pytest.mark.unit
def test_failing_test_files_rejects_collection_failure_without_message(
    tmp_path: Path,
) -> None:
    """Require a diagnostic fingerprint before ratcheting a collection error."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    _write_report(
        report_path,
        package_root,
        {},
        collection_failures=("src/import-error.test.ts",),
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["testResults"][0]["message"] = ""
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RatchetError, match="missing an error message"):
        failing_test_files(report_path, package_root)


@pytest.mark.unit
def test_compare_reports_accepts_exact_unchanged_collection_failure(
    tmp_path: Path,
) -> None:
    """Accept an unchanged collection error only when its base fingerprint matches."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    collection_failure = ("src/import-error.test.ts",)
    _write_report(
        head_report,
        head_root,
        {},
        collection_failures=collection_failure,
    )
    _write_report(
        base_report,
        base_root,
        {},
        collection_failures=collection_failure,
    )
    head_payload = json.loads(head_report.read_text(encoding="utf-8"))
    head_payload["testResults"][0]["message"] = (
        f"failed to import {head_root}/src/dependency.ts"
    )
    head_report.write_text(json.dumps(head_payload), encoding="utf-8")
    base_payload = json.loads(base_report.read_text(encoding="utf-8"))
    base_payload["testResults"][0]["message"] = (
        f"failed to import {base_root}/src/dependency.ts"
    )
    base_report.write_text(json.dumps(base_payload), encoding="utf-8")
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert result.passes
    assert len(result.inherited) == 1
    assert result.regressions == ()


@pytest.mark.unit
def test_compare_reports_normalizes_repository_root_in_collection_message(
    tmp_path: Path,
) -> None:
    """Normalize sibling-package paths beneath different checkout roots."""

    head_repo = tmp_path / "head"
    base_repo = tmp_path / "base"
    head_root = head_repo / "apps" / "packages" / "ui"
    base_root = base_repo / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    collection_failure = ("src/import-error.test.ts",)
    _write_report(
        head_report,
        head_root,
        {},
        collection_failures=collection_failure,
    )
    _write_report(
        base_report,
        base_root,
        {},
        collection_failures=collection_failure,
    )
    head_payload = json.loads(head_report.read_text(encoding="utf-8"))
    head_payload["testResults"][0]["message"] = (
        f"failed through {head_repo}/apps/tldw-frontend/src/runtime.ts"
    )
    head_report.write_text(json.dumps(head_payload), encoding="utf-8")
    base_payload = json.loads(base_report.read_text(encoding="utf-8"))
    base_payload["testResults"][0]["message"] = (
        f"failed through {base_repo}/apps/tldw-frontend/src/runtime.ts"
    )
    base_report.write_text(json.dumps(base_payload), encoding="utf-8")
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert result.passes
    assert len(result.inherited) == 1


@pytest.mark.unit
def test_compare_reports_rejects_unequal_execution_context(tmp_path: Path) -> None:
    """Reject matching failures when the base omitted a passing head module."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/failed.test.ts": ["same inherited failure"]}
    _write_report(head_report, head_root, failures)
    _append_passed_result(head_report, head_root, "src/order-context.test.ts")
    _write_report(base_report, base_root, failures)
    _write_order_report(
        head_order_report,
        ("src/failed.test.ts", "src/order-context.test.ts"),
    )
    _write_order_report(base_order_report, ("src/failed.test.ts",))
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    with pytest.raises(RatchetError, match="module manifests"):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("apps/packages/ui"),
            changed_files_path=changed_files,
            require_equivalent_context=True,
            head_order_report_path=head_order_report,
            base_order_report_path=base_order_report,
        )


@pytest.mark.unit
def test_compare_reports_ignores_json_order_when_runtime_order_matches(
    tmp_path: Path,
) -> None:
    """Do not confuse Vitest JSON serialization order with runtime order."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/failed.test.ts": ["same inherited failure"]}
    _write_report(head_report, head_root, failures)
    _append_passed_result(head_report, head_root, "src/order-context.test.ts")
    _write_report(base_report, base_root, failures)
    _append_passed_result(base_report, base_root, "src/order-context.test.ts")
    base_payload = json.loads(base_report.read_text(encoding="utf-8"))
    base_payload["testResults"].reverse()
    base_report.write_text(json.dumps(base_payload), encoding="utf-8")
    runtime_order = ("src/failed.test.ts", "src/order-context.test.ts")
    _write_order_report(head_order_report, runtime_order)
    _write_order_report(base_order_report, runtime_order)
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
        require_equivalent_context=True,
        head_order_report_path=head_order_report,
        base_order_report_path=base_order_report,
    )

    assert result.passes


@pytest.mark.unit
def test_compare_reports_rejects_reordered_runtime_context(tmp_path: Path) -> None:
    """Reject the same modules when reporter-observed runtime order differs."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/failed.test.ts": ["same inherited failure"]}
    _write_report(head_report, head_root, failures)
    _append_passed_result(head_report, head_root, "src/order-context.test.ts")
    _write_report(base_report, base_root, failures)
    _append_passed_result(base_report, base_root, "src/order-context.test.ts")
    _write_order_report(
        head_order_report,
        ("src/failed.test.ts", "src/order-context.test.ts"),
    )
    _write_order_report(
        base_order_report,
        ("src/order-context.test.ts", "src/failed.test.ts"),
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    with pytest.raises(RatchetError, match="execution order"):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("apps/packages/ui"),
            changed_files_path=changed_files,
            require_equivalent_context=True,
            head_order_report_path=head_order_report,
            base_order_report_path=base_order_report,
        )


@pytest.mark.unit
def test_compare_reports_rejects_changed_passing_test_identity(
    tmp_path: Path,
) -> None:
    """Reject equal aggregate counts when a passing test identity differs."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/failed.test.ts": ["same inherited failure"]}
    _write_report(head_report, head_root, failures)
    _append_passed_result(head_report, head_root, "src/order-context.test.ts")
    _write_report(base_report, base_root, failures)
    _append_passed_result(base_report, base_root, "src/order-context.test.ts")
    base_payload = json.loads(base_report.read_text(encoding="utf-8"))
    base_payload["testResults"][1]["assertionResults"][0]["fullName"] = (
        "different passing identity"
    )
    base_report.write_text(json.dumps(base_payload), encoding="utf-8")
    runtime_order = ("src/failed.test.ts", "src/order-context.test.ts")
    _write_order_report(head_order_report, runtime_order)
    _write_order_report(base_order_report, runtime_order)
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    with pytest.raises(RatchetError, match="test identities"):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("apps/packages/ui"),
            changed_files_path=changed_files,
            require_equivalent_context=True,
            head_order_report_path=head_order_report,
            base_order_report_path=base_order_report,
        )


@pytest.mark.unit
def test_compare_reports_rejects_changed_suite_identity(tmp_path: Path) -> None:
    """Reject equal suite counts when an ancestor suite identity differs."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/failed.test.ts": ["same inherited failure"]}
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    for report_path, ancestor in (
        (head_report, "head suite"),
        (base_report, "base suite"),
    ):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload["testResults"][0]["assertionResults"][0]["ancestorTitles"] = [
            ancestor
        ]
        payload["numTotalTestSuites"] = 2
        payload["numFailedTestSuites"] = 2
        report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_order_report(head_order_report, ("src/failed.test.ts",))
    _write_order_report(base_order_report, ("src/failed.test.ts",))
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    with pytest.raises(RatchetError, match="suite identities"):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("apps/packages/ui"),
            changed_files_path=changed_files,
            require_equivalent_context=True,
            head_order_report_path=head_order_report,
            base_order_report_path=base_order_report,
        )


@pytest.mark.unit
@pytest.mark.parametrize("counter_name", ("numTotalTests", "numTotalTestSuites"))
def test_compare_reports_rejects_unequal_execution_counts(
    tmp_path: Path,
    counter_name: str,
) -> None:
    """Reject matching module manifests with different test totals."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/failed.test.ts": ["same inherited failure"]}
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    base_payload = json.loads(base_report.read_text(encoding="utf-8"))
    base_payload[counter_name] += 1
    base_report.write_text(json.dumps(base_payload), encoding="utf-8")
    _write_order_report(head_order_report, ("src/failed.test.ts",))
    _write_order_report(base_order_report, ("src/failed.test.ts",))
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    with pytest.raises(RatchetError, match=counter_name):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("apps/packages/ui"),
            changed_files_path=changed_files,
            require_equivalent_context=True,
            head_order_report_path=head_order_report,
            base_order_report_path=base_order_report,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("total_counter", "status_counter"),
    (
        ("numTotalTests", "numFailedTests"),
        ("numTotalTestSuites", "numFailedTestSuites"),
    ),
)
def test_compare_reports_rejects_equal_malformed_execution_counts(
    tmp_path: Path,
    total_counter: str,
    status_counter: str,
) -> None:
    """Reject equal head/base counters that disagree with report structure."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/failed.test.ts": ["same inherited failure"]}
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    for report_path in (head_report, base_report):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        payload[total_counter] += 1
        payload[status_counter] += 1
        report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_order_report(head_order_report, ("src/failed.test.ts",))
    _write_order_report(base_order_report, ("src/failed.test.ts",))
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    with pytest.raises(RatchetError, match=total_counter):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("apps/packages/ui"),
            changed_files_path=changed_files,
            require_equivalent_context=True,
            head_order_report_path=head_order_report,
            base_order_report_path=base_order_report,
        )


@pytest.mark.unit
def test_compare_reports_rejects_changed_collection_failure_fingerprint(
    tmp_path: Path,
) -> None:
    """Reject a collection error when its exact base error message differs."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    collection_failure = ("src/import-error.test.ts",)
    _write_report(
        head_report,
        head_root,
        {},
        collection_failures=collection_failure,
    )
    _write_report(
        base_report,
        base_root,
        {},
        collection_failures=collection_failure,
    )
    head_payload = json.loads(head_report.read_text(encoding="utf-8"))
    head_payload["testResults"][0]["message"] = "new import failure"
    head_report.write_text(json.dumps(head_payload), encoding="utf-8")
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert not result.passes
    assert len(result.regressions) == 1


@pytest.mark.unit
def test_failing_test_files_rejects_file_failure_after_assertion_collection(
    tmp_path: Path,
) -> None:
    """Do not classify hook-style file failures as import failures."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["success"] = False
    payload["numPassedTestSuites"] = 0
    payload["numFailedTestSuites"] = 1
    payload["testResults"][0]["status"] = "failed"
    payload["testResults"][0]["message"] = "afterAll hook failed"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RatchetError, match="after assertion collection"):
        failing_test_files(report_path, package_root)


@pytest.mark.unit
def test_package_compare_rejects_file_error_hidden_beside_assertion(
    tmp_path: Path,
) -> None:
    """Reject a package file error even when one assertion also failed."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/setup.test.ts": ["ordinary assertion"]}
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    payload = json.loads(head_report.read_text(encoding="utf-8"))
    payload["testResults"][0]["message"] = "afterAll cleanup failed"
    head_report.write_text(json.dumps(payload), encoding="utf-8")
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    with pytest.raises(RatchetError, match="file-level error"):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("apps/packages/ui"),
            changed_files_path=changed_files,
        )


@pytest.mark.unit
def test_compare_reports_rejects_collection_failure_in_changed_test_file(
    tmp_path: Path,
) -> None:
    """Reject a matching collection error when the owning test file changed."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    collection_failure = ("src/import-error.test.ts",)
    _write_report(
        head_report,
        head_root,
        {},
        collection_failures=collection_failure,
    )
    _write_report(
        base_report,
        base_root,
        {},
        collection_failures=collection_failure,
    )
    changed_files.write_text(
        "apps/packages/ui/src/import-error.test.ts\n",
        encoding="utf-8",
    )

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert not result.passes
    assert len(result.regressions) == 1


def test_validate_success_report_accepts_clean_vitest_report(tmp_path: Path) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    _write_report(report_path, package_root, {})

    validate_success_report(report_path, package_root)


def test_validate_success_report_rejects_failed_vitest_report(tmp_path: Path) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    _write_report(
        report_path,
        package_root,
        {"src/failed.test.ts": ["suite still fails"]},
    )

    with pytest.raises(RatchetError, match="expected a successful"):
        validate_success_report(report_path, package_root)


@pytest.mark.unit
def test_validate_success_report_rejects_zero_assertion_file_error(
    tmp_path: Path,
) -> None:
    """Do not treat a passed empty module with a diagnostic as clean."""

    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    report_path.write_text(
        json.dumps(
            {
                "success": True,
                "testResults": [
                    {
                        "name": str(package_root / "src/empty.test.ts"),
                        "status": "passed",
                        "assertionResults": [],
                        "message": "afterAll cleanup failed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RatchetError, match="file-level error"):
        validate_success_report(report_path, package_root)


def test_validate_success_report_rejects_unknown_test_result_status(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    report_path.write_text(
        json.dumps(
            {
                "success": True,
                "testResults": [
                    {
                        "name": str(package_root / "src/passed.test.ts"),
                        "status": "flaky",
                        "assertionResults": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RatchetError, match="test result status"):
        validate_success_report(report_path, package_root)


def test_validate_success_report_rejects_unknown_assertion_status(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    report_path.write_text(
        json.dumps(
            {
                "success": True,
                "testResults": [
                    {
                        "name": str(package_root / "src/passed.test.ts"),
                        "status": "passed",
                        "assertionResults": [
                            {
                                "fullName": "suite reports a known status",
                                "status": "flaky",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RatchetError, match="assertion status"):
        validate_success_report(report_path, package_root)


def test_strict_validate_success_report_rejects_zero_tests(tmp_path: Path) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_report(report_path, package_root, {})
    _write_safety_report(
        safety_path,
        reason="passed",
        test_count=0,
        module_count=0,
    )

    with pytest.raises(RatchetError, match="zero tests"):
        validate_success_report(
            report_path,
            package_root,
            strict=True,
            safety_report_path=safety_path,
        )


def test_strict_validate_success_report_rejects_counter_mismatch(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["numTotalTests"] = 2
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(safety_path, reason="passed", test_count=1)

    with pytest.raises(RatchetError, match="counter"):
        validate_success_report(
            report_path,
            package_root,
            strict=True,
            safety_report_path=safety_path,
        )


@pytest.mark.unit
def test_strict_validate_success_report_rejects_unproven_suite_count(
    tmp_path: Path,
) -> None:
    """Reject hidden suite counters not proven by the lifecycle reporter."""

    package_root = tmp_path / "head" / "admin-ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["numTotalTestSuites"] = 2
    payload["numPassedTestSuites"] = 2
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(safety_path, reason="passed", test_count=1)

    with pytest.raises(RatchetError, match="suiteCount does not match JSON"):
        validate_success_report(
            report_path,
            package_root,
            strict=True,
            safety_report_path=safety_path,
        )


@pytest.mark.unit
def test_strict_validate_success_report_accepts_nested_suite_hierarchy(
    tmp_path: Path,
) -> None:
    """Match Vitest JSON semantics for a file root with nested describes."""

    package_root = tmp_path / "head" / "admin-ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["testResults"][0]["assertionResults"][0]["ancestorTitles"] = [
        "outer suite",
        "inner suite",
    ]
    payload["numTotalTestSuites"] = 3
    payload["numPassedTestSuites"] = 3
    assert len(payload["testResults"]) == 1
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(
        safety_path,
        reason="passed",
        test_count=1,
        suite_count=3,
    )

    validate_success_report(
        report_path,
        package_root,
        strict=True,
        safety_report_path=safety_path,
    )


@pytest.mark.unit
def test_strict_validate_success_report_accepts_reporter_proven_hidden_suites(
    tmp_path: Path,
) -> None:
    """Accept empty or skipped suites proven by the lifecycle reporter."""

    package_root = tmp_path / "head" / "admin-ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["numTotalTestSuites"] = 2
    payload["numPassedTestSuites"] = 2
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(
        safety_path,
        reason="passed",
        test_count=1,
        suite_count=2,
    )

    validate_success_report(
        report_path,
        package_root,
        strict=True,
        safety_report_path=safety_path,
    )


@pytest.mark.unit
def test_strict_validate_success_report_rejects_hidden_suite_status_mismatch(
    tmp_path: Path,
) -> None:
    """Require reporter status categories to match every raw suite counter."""

    package_root = tmp_path / "head" / "admin-ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["numTotalTestSuites"] = 2
    payload["numPassedTestSuites"] = 2
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(
        safety_path,
        reason="passed",
        test_count=1,
        suite_count=2,
        passed_suite_count=1,
        pending_suite_count=1,
    )

    with pytest.raises(RatchetError, match="pendingSuiteCount does not match JSON"):
        validate_success_report(
            report_path,
            package_root,
            strict=True,
            safety_report_path=safety_path,
        )


@pytest.mark.unit
def test_strict_validate_success_report_rejects_failed_suite_counter(
    tmp_path: Path,
) -> None:
    """Reject a success flag that contradicts a failed-suite counter."""

    package_root = tmp_path / "head" / "admin-ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["numTotalTestSuites"] = 2
    payload["numPassedTestSuites"] = 1
    payload["numFailedTestSuites"] = 1
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(
        safety_path,
        reason="passed",
        test_count=1,
        suite_count=2,
        failed_suite_count=1,
    )

    with pytest.raises(RatchetError, match="claims success"):
        validate_success_report(
            report_path,
            package_root,
            strict=True,
            safety_report_path=safety_path,
        )


@pytest.mark.unit
def test_strict_validate_success_report_rejects_unfinished_test(
    tmp_path: Path,
) -> None:
    """Reject a completed success report that still contains a pending test."""

    package_root = tmp_path / "head" / "admin-ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assertion = payload["testResults"][0]["assertionResults"][0]
    assertion["status"] = "pending"
    payload["numPassedTests"] = 0
    payload["numPendingTests"] = 1
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(
        safety_path,
        reason="passed",
        test_count=1,
        incomplete_test_count=1,
    )

    with pytest.raises(RatchetError, match="unfinished assertion"):
        validate_success_report(
            report_path,
            package_root,
            strict=True,
            safety_report_path=safety_path,
        )


@pytest.mark.unit
def test_strict_validate_success_report_rejects_hidden_incomplete_suite(
    tmp_path: Path,
) -> None:
    """Reject reporter-observed runtime work left pending at run completion."""

    package_root = tmp_path / "head" / "admin-ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["numTotalTestSuites"] = 2
    payload["numPendingTestSuites"] = 1
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(
        safety_path,
        reason="passed",
        test_count=1,
        suite_count=2,
        passed_suite_count=1,
        pending_suite_count=1,
        incomplete_suite_count=1,
    )

    with pytest.raises(RatchetError, match="incompleteSuiteCount"):
        validate_success_report(
            report_path,
            package_root,
            strict=True,
            safety_report_path=safety_path,
        )


@pytest.mark.unit
def test_strict_validate_success_report_requires_file_message(
    tmp_path: Path,
) -> None:
    """Require file-level collection diagnostics in strict reports."""

    package_root = tmp_path / "head" / "admin-ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    del payload["testResults"][0]["message"]
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(safety_path, reason="passed", test_count=1)

    with pytest.raises(RatchetError, match="missing file-level message"):
        validate_success_report(
            report_path,
            package_root,
            strict=True,
            safety_report_path=safety_path,
        )


@pytest.mark.parametrize(
    "safety_field",
    ("unhandledErrorCount", "moduleErrorCount", "hookErrorCount"),
)
def test_strict_validate_success_report_rejects_safety_errors(
    tmp_path: Path,
    safety_field: str,
) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    safety_counts = {
        "unhandled_error_count": 0,
        "module_error_count": 0,
        "hook_error_count": 0,
    }
    argument_by_field = {
        "unhandledErrorCount": "unhandled_error_count",
        "moduleErrorCount": "module_error_count",
        "hookErrorCount": "hook_error_count",
    }
    safety_counts[argument_by_field[safety_field]] = 1
    _write_safety_report(
        safety_path,
        reason="passed",
        test_count=1,
        **safety_counts,
    )

    with pytest.raises(RatchetError, match=safety_field):
        validate_success_report(
            report_path,
            package_root,
            strict=True,
            safety_report_path=safety_path,
        )


def test_failing_test_files_rejects_passed_result_with_failed_assertion(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    report_path.write_text(
        json.dumps(
            {
                "success": False,
                "testResults": [
                    {
                        "name": str(package_root / "src/contradictory.test.ts"),
                        "status": "passed",
                        "assertionResults": [
                            {
                                "fullName": "suite contradictory status",
                                "status": "failed",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RatchetError, match="passed.*failed assertion"):
        failing_test_files(report_path, package_root)


def test_compare_reports_accepts_only_exact_unchanged_base_failures(tmp_path: Path) -> None:
    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    failures = {
        "src/routes/route.test.ts": [
            "route governance covers every route",
            "route governance requires metadata",
        ]
    }
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert result.passes
    assert len(result.inherited) == 2
    assert result.regressions == ()


def test_compare_reports_rejects_failure_absent_from_base(tmp_path: Path) -> None:
    head_root = tmp_path / "head" / "apps" / "tldw-frontend"
    base_root = tmp_path / "base" / "apps" / "tldw-frontend"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    _write_report(
        head_report,
        head_root,
        {"__tests__/new.test.ts": ["feature keeps working"]},
    )
    _write_report(base_report, base_root, {})
    changed_files.write_text("apps/tldw-frontend/lib/feature.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/tldw-frontend"),
        changed_files_path=changed_files,
    )

    assert not result.passes
    assert [failure.full_name for failure in result.regressions] == [
        "feature keeps working"
    ]


def test_compare_reports_rejects_inherited_failure_in_modified_test_file(
    tmp_path: Path,
) -> None:
    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/changed.test.ts": ["same inherited test name"]}
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    changed_files.write_text(
        "apps/packages/ui/src/changed.test.ts\n",
        encoding="utf-8",
    )

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert not result.passes
    assert [failure.full_name for failure in result.regressions] == [
        "same inherited test name"
    ]
    assert result.inherited == ()


@pytest.mark.unit
def test_package_compare_rejects_changed_failure_cause(tmp_path: Path) -> None:
    """Do not inherit the same test identity with a different diagnostic."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/route.test.ts": ["route stays protected"]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            "src/route.test.ts": [f"expected 403 at {head_root}, received 500"]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={
            "src/route.test.ts": [f"expected 403 at {base_root}, received 401"]
        },
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert not result.passes
    assert result.regressions[0].failure_messages == (
        "expected 403 at <PACKAGE_ROOT>, received 500",
    )


@pytest.mark.unit
def test_package_compare_preserves_raw_node_scheduler_frames_without_provenance(
    tmp_path: Path,
) -> None:
    """Fail closed when raw diagnostic text only looks like a stack trace."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/route.test.ts": ["route stays protected"]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            "src/route.test.ts": [
                f"route failed at {head_root}/src/route.test.ts:42:7\n"
                f"    at routeHandler ({head_root}/src/route.test.ts:42:7)\n"
                "    at runNextTicks (node:internal/process/task_queues:65:5)\n"
                "    at processTimers (node:internal/timers:538:9)"
            ]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={
            "src/route.test.ts": [
                f"route failed at {base_root}/src/route.test.ts:42:7\n"
                f"    at routeHandler ({base_root}/src/route.test.ts:42:7)"
            ]
        },
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert not result.passes
    assert result.inherited == ()
    assert result.regressions[0].failure_messages == (
        "route failed at <PACKAGE_ROOT>/src/route.test.ts:42:7\n"
        "    at routeHandler (<PACKAGE_ROOT>/src/route.test.ts:42:7)\n"
        "    at runNextTicks (node:internal/process/task_queues:65:5)\n"
        "    at processTimers (node:internal/timers:538:9)",
    )


@pytest.mark.unit
def test_package_compare_uses_structured_failure_provenance(
    tmp_path: Path,
) -> None:
    """Normalize scheduler frames only when Vitest identifies structured frames."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    relative_path = "src/route.test.ts"
    full_name = "route stays protected"
    failures = {relative_path: [full_name]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            relative_path: [
                f"route failed at {head_root}/src/route.test.ts:42:7\n"
                f"    at routeHandler ({head_root}/src/route.test.ts:42:7)\n"
                "    at runNextTicks (node:internal/process/task_queues:65:5)\n"
                "    at processTimers (node:internal/timers:538:9)\n"
                f"    at VitestRunner.run ({head_root}/node_modules/vitest/runner.js:10:2)"
            ]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={
            relative_path: [
                f"route failed at {base_root}/src/route.test.ts:42:7\n"
                f"    at routeHandler ({base_root}/src/route.test.ts:42:7)\n"
                f"    at VitestRunner.run ({base_root}/node_modules/vitest/runner.js:99:8)"
            ]
        },
    )
    _write_structured_order_report(
        head_order_report,
        (relative_path,),
        [
            _structured_failure(
                relative_path,
                full_name,
                f"route failed at {head_root}/src/route.test.ts:42:7",
                [
                    {
                        "method": "routeHandler",
                        "file": f"{head_root}/src/route.test.ts",
                        "line": 42,
                        "column": 7,
                    },
                    {
                        "method": "runNextTicks",
                        "file": "node:internal/process/task_queues",
                        "line": 65,
                        "column": 5,
                    },
                    {
                        "method": "processTimers",
                        "file": "node:internal/timers",
                        "line": 538,
                        "column": 9,
                    },
                ],
            )
        ],
    )
    _write_structured_order_report(
        base_order_report,
        (relative_path,),
        [
            _structured_failure(
                relative_path,
                full_name,
                f"route failed at {base_root}/src/route.test.ts:42:7",
                [
                    {
                        "method": "routeHandler",
                        "file": f"{base_root}/src/route.test.ts",
                        "line": 42,
                        "column": 7,
                    }
                ],
            )
        ],
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
        head_order_report_path=head_order_report,
        base_order_report_path=base_order_report,
    )

    assert result.passes
    assert len(result.inherited) == 1
    assert result.regressions == ()


@pytest.mark.unit
def test_package_compare_preserves_scheduler_text_in_structured_messages(
    tmp_path: Path,
) -> None:
    """Never classify scheduler-looking error.message content as a parsed frame."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    relative_path = "src/route.test.ts"
    full_name = "route stays protected"
    failures = {relative_path: [full_name]}
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    base_message = "route failed\n    at rendered (/diagnostics/view.ts:12:3)"
    head_message = (
        f"{base_message}\n"
        "    at runNextTicks (node:internal/process/task_queues:65:5)"
    )
    _write_structured_order_report(
        head_order_report,
        (relative_path,),
        [_structured_failure(relative_path, full_name, head_message, [])],
    )
    _write_structured_order_report(
        base_order_report,
        (relative_path,),
        [_structured_failure(relative_path, full_name, base_message, [])],
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
        head_order_report_path=head_order_report,
        base_order_report_path=base_order_report,
    )

    assert not result.passes
    fingerprint = json.loads(result.regressions[0].failure_messages[0])
    assert fingerprint["message"] == head_message
    assert fingerprint["stacks"] == []


@pytest.mark.unit
def test_package_compare_rejects_one_sided_failure_provenance(tmp_path: Path) -> None:
    """Do not silently fall back when only one checkout has provenance."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    changed_files = tmp_path / "changed.txt"
    relative_path = "src/route.test.ts"
    full_name = "route stays protected"
    failures = {relative_path: [full_name]}
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    _write_structured_order_report(
        head_order_report,
        (relative_path,),
        [
            _structured_failure(
                relative_path,
                full_name,
                "expected true to be false",
                [],
            )
        ],
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    with pytest.raises(RatchetError, match="head and base execution-order reports"):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("apps/packages/ui"),
            changed_files_path=changed_files,
            head_order_report_path=head_order_report,
        )


@pytest.mark.unit
def test_package_compare_rejects_malformed_failure_provenance(tmp_path: Path) -> None:
    """Reject a schema-3 sidecar that omits its structured failures."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    relative_path = "src/route.test.ts"
    full_name = "route stays protected"
    failures = {relative_path: [full_name]}
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    structured_failures = [
        _structured_failure(
            relative_path,
            full_name,
            "expected true to be false",
            [],
        )
    ]
    _write_structured_order_report(
        head_order_report,
        (relative_path,),
        structured_failures,
    )
    _write_structured_order_report(
        base_order_report,
        (relative_path,),
        structured_failures,
    )
    base_payload = json.loads(base_order_report.read_text(encoding="utf-8"))
    del base_payload["failures"]
    base_order_report.write_text(json.dumps(base_payload), encoding="utf-8")
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    with pytest.raises(RatchetError, match="missing structured failures"):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("apps/packages/ui"),
            changed_files_path=changed_files,
            head_order_report_path=head_order_report,
            base_order_report_path=base_order_report,
        )


@pytest.mark.unit
def test_package_compare_consumes_duplicate_structured_failure_identities(
    tmp_path: Path,
) -> None:
    """Preserve duplicate failures instead of overwriting them by identity."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_order_report = tmp_path / "head-order.json"
    base_order_report = tmp_path / "base-order.json"
    changed_files = tmp_path / "changed.txt"
    relative_path = "src/route.test.ts"
    full_name = "duplicate route check"
    failures = {relative_path: [full_name, full_name]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            relative_path: ["raw head error one", "raw head error two"]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={
            relative_path: ["raw base error one", "raw base error two"]
        },
    )
    structured_failures = [
        _structured_failure(relative_path, full_name, "first error", []),
        _structured_failure(relative_path, full_name, "second error", []),
    ]
    _write_structured_order_report(
        head_order_report,
        (relative_path,),
        structured_failures,
    )
    _write_structured_order_report(
        base_order_report,
        (relative_path,),
        structured_failures,
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
        head_order_report_path=head_order_report,
        base_order_report_path=base_order_report,
    )

    assert result.passes
    assert len(result.inherited) == 2
    assert result.regressions == ()


@pytest.mark.unit
def test_package_compare_preserves_exact_scheduler_looking_assertion_content(
    tmp_path: Path,
) -> None:
    """Do not treat assertion content as a scheduler stack frame."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/route.test.ts": ["route stays protected"]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            "src/route.test.ts": [
                "route failed\n"
                "    at runNextTicks (node:internal/process/task_queues:65:5)"
            ]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={"src/route.test.ts": ["route failed"]},
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert not result.passes
    assert result.regressions[0].failure_messages == (
        "route failed\n"
        "    at runNextTicks (node:internal/process/task_queues:65:5)",
    )


@pytest.mark.unit
def test_package_compare_preserves_rendered_at_text_in_raw_fallback(
    tmp_path: Path,
) -> None:
    """Preserve raw diagnostic text that happens to use indented at-lines."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/route.test.ts": ["route stays protected"]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            "src/route.test.ts": [
                "route failed\n"
                "    at rendered content\n"
                "    at runNextTicks (node:internal/process/task_queues:65:5)"
            ]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={
            "src/route.test.ts": ["route failed\n    at rendered content"]
        },
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert not result.passes
    assert result.regressions[0].failure_messages == (
        "route failed\n"
        "    at rendered content\n"
        "    at runNextTicks (node:internal/process/task_queues:65:5)",
    )


@pytest.mark.unit
def test_package_compare_preserves_scheduler_only_failure_messages(
    tmp_path: Path,
) -> None:
    """Treat scheduler-looking raw text as diagnostic content."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/route.test.ts": ["route stays protected"]}
    scheduler_only = (
        "    at runNextTicks (node:internal/process/task_queues:65:5)\n"
        "    at processTimers (node:internal/timers:538:9)"
    )
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={"src/route.test.ts": [scheduler_only]},
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={"src/route.test.ts": [scheduler_only]},
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert result.passes
    assert result.inherited[0].failure_messages == (scheduler_only,)


@pytest.mark.unit
def test_package_compare_preserves_scheduler_markers_in_custom_diagnostics(
    tmp_path: Path,
) -> None:
    """Do not erase scheduler markers from raw custom diagnostics."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    failures = {"src/route.test.ts": ["route stays protected"]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            "src/route.test.ts": [
                "route failed\nat rendered text node: node:internal/timers:new"
            ]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={
            "src/route.test.ts": [
                "route failed\nat rendered text node: node:internal/timers:old"
            ]
        },
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert not result.passes
    assert result.regressions[0].failure_messages == (
        "route failed\nat rendered text node: node:internal/timers:new",
    )


@pytest.mark.unit
def test_package_compare_preserves_duplicate_failure_multiplicity(
    tmp_path: Path,
) -> None:
    """Require one matching base occurrence for each head failure."""

    head_root = tmp_path / "head" / "apps" / "packages" / "ui"
    base_root = tmp_path / "base" / "apps" / "packages" / "ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    changed_files = tmp_path / "changed.txt"
    _write_report(
        head_report,
        head_root,
        {"src/duplicate.test.ts": ["same name", "same name"]},
    )
    _write_report(
        base_report,
        base_root,
        {"src/duplicate.test.ts": ["same name"]},
    )
    changed_files.write_text("apps/packages/ui/src/runtime.ts\n", encoding="utf-8")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("apps/packages/ui"),
        changed_files_path=changed_files,
    )

    assert len(result.inherited) == 1
    assert len(result.regressions) == 1


def test_strict_compare_rejects_changed_failure_cause(tmp_path: Path) -> None:
    head_root = tmp_path / "head" / "admin-ui"
    base_root = tmp_path / "base" / "admin-ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_safety = tmp_path / "head-safety.json"
    base_safety = tmp_path / "base-safety.json"
    changed_files = tmp_path / "changed.bin"
    failures = {"src/route.test.ts": ["route stays protected"]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            "src/route.test.ts": [f"expected 403 at {head_root}, received 500"]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={
            "src/route.test.ts": [f"expected 403 at {base_root}, received 401"]
        },
    )
    _write_safety_report(head_safety, reason="failed", test_count=1)
    _write_safety_report(base_safety, reason="failed", test_count=1)
    changed_files.write_bytes(b"admin-ui/src/runtime.ts\0")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("admin-ui"),
        changed_files_path=changed_files,
        strict=True,
        head_safety_report_path=head_safety,
        base_safety_report_path=base_safety,
    )

    assert not result.passes
    assert len(result.regressions) == 1
    assert result.regressions[0].failure_messages == (
        "expected 403 at <PACKAGE_ROOT>, received 500",
    )


@pytest.mark.unit
def test_strict_compare_preserves_scheduler_markers_in_package_frames(
    tmp_path: Path,
) -> None:
    """Keep marker-like function names when the frame source is user code."""

    head_root = tmp_path / "head" / "admin-ui"
    base_root = tmp_path / "base" / "admin-ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_safety = tmp_path / "head-safety.json"
    base_safety = tmp_path / "base-safety.json"
    changed_files = tmp_path / "changed.bin"
    failures = {"src/route.test.ts": ["route stays protected"]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            "src/route.test.ts": [
                "route failed\n"
                f"at node:internal/process/task_queues:new "
                f"({head_root}/src/route.ts:42:7)"
            ]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={
            "src/route.test.ts": [
                "route failed\n"
                f"at node:internal/process/task_queues:old "
                f"({base_root}/src/route.ts:42:7)"
            ]
        },
    )
    _write_safety_report(head_safety, reason="failed", test_count=1)
    _write_safety_report(base_safety, reason="failed", test_count=1)
    changed_files.write_bytes(b"admin-ui/src/runtime.ts\0")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("admin-ui"),
        changed_files_path=changed_files,
        strict=True,
        head_safety_report_path=head_safety,
        base_safety_report_path=base_safety,
    )

    assert not result.passes
    assert result.regressions[0].failure_messages == (
        "route failed\n"
        "at node:internal/process/task_queues:new "
        "(<PACKAGE_ROOT>/src/route.ts:42:7)",
    )


@pytest.mark.unit
def test_strict_compare_does_not_treat_rendered_at_text_as_stack_context(
    tmp_path: Path,
) -> None:
    """Apply the positive source-frame requirement in strict mode."""

    head_root = tmp_path / "head" / "admin-ui"
    base_root = tmp_path / "base" / "admin-ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_safety = tmp_path / "head-safety.json"
    base_safety = tmp_path / "base-safety.json"
    changed_files = tmp_path / "changed.bin"
    failures = {"src/route.test.ts": ["route stays protected"]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={
            "src/route.test.ts": [
                "route failed\n"
                "    at rendered content\n"
                "    at processTimers (node:internal/timers:538:9)"
            ]
        },
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={
            "src/route.test.ts": ["route failed\n    at rendered content"]
        },
    )
    _write_safety_report(head_safety, reason="failed", test_count=1)
    _write_safety_report(base_safety, reason="failed", test_count=1)
    changed_files.write_bytes(b"admin-ui/src/runtime.ts\0")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("admin-ui"),
        changed_files_path=changed_files,
        strict=True,
        head_safety_report_path=head_safety,
        base_safety_report_path=base_safety,
    )

    assert not result.passes
    assert result.regressions[0].failure_messages == (
        "route failed\n"
        "    at rendered content\n"
        "    at processTimers (node:internal/timers:538:9)",
    )


def test_strict_compare_accepts_same_cause_after_package_root_normalization(
    tmp_path: Path,
) -> None:
    head_root = tmp_path / "head" / "admin-ui"
    base_root = tmp_path / "base" / "admin-ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_safety = tmp_path / "head-safety.json"
    base_safety = tmp_path / "base-safety.json"
    changed_files = tmp_path / "changed.bin"
    failures = {"src/route.test.ts": ["route stays protected"]}
    _write_report(
        head_report,
        head_root,
        failures,
        failure_messages={"src/route.test.ts": [f"failure at {head_root}/src"]},
    )
    _write_report(
        base_report,
        base_root,
        failures,
        failure_messages={"src/route.test.ts": [f"failure at {base_root}/src"]},
    )
    _write_safety_report(head_safety, reason="failed", test_count=1)
    _write_safety_report(base_safety, reason="failed", test_count=1)
    changed_files.write_bytes(b"admin-ui/src/runtime.ts\0")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("admin-ui"),
        changed_files_path=changed_files,
        strict=True,
        head_safety_report_path=head_safety,
        base_safety_report_path=base_safety,
    )

    assert result.passes
    assert len(result.inherited) == 1


def test_strict_compare_preserves_duplicate_failure_multiplicity(
    tmp_path: Path,
) -> None:
    head_root = tmp_path / "head" / "admin-ui"
    base_root = tmp_path / "base" / "admin-ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_safety = tmp_path / "head-safety.json"
    base_safety = tmp_path / "base-safety.json"
    changed_files = tmp_path / "changed.bin"
    _write_report(
        head_report,
        head_root,
        {"src/duplicate.test.ts": ["same name", "same name"]},
    )
    _write_report(
        base_report,
        base_root,
        {"src/duplicate.test.ts": ["same name"]},
    )
    _write_safety_report(head_safety, reason="failed", test_count=2)
    _write_safety_report(base_safety, reason="failed", test_count=1)
    changed_files.write_bytes(b"admin-ui/src/runtime.ts\0")

    result = compare_reports(
        head_report=head_report,
        base_report=base_report,
        head_package_root=head_root,
        base_package_root=base_root,
        package_repo_path=Path("admin-ui"),
        changed_files_path=changed_files,
        strict=True,
        head_safety_report_path=head_safety,
        base_safety_report_path=base_safety,
    )

    assert len(result.inherited) == 1
    assert len(result.regressions) == 1


def test_strict_compare_rejects_file_error_hidden_beside_assertion(
    tmp_path: Path,
) -> None:
    head_root = tmp_path / "head" / "admin-ui"
    base_root = tmp_path / "base" / "admin-ui"
    head_report = tmp_path / "head.json"
    base_report = tmp_path / "base.json"
    head_safety = tmp_path / "head-safety.json"
    base_safety = tmp_path / "base-safety.json"
    changed_files = tmp_path / "changed.bin"
    failures = {"src/setup.test.ts": ["ordinary assertion"]}
    _write_report(head_report, head_root, failures)
    _write_report(base_report, base_root, failures)
    payload = json.loads(head_report.read_text(encoding="utf-8"))
    payload["testResults"][0]["message"] = "afterAll cleanup failed"
    head_report.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(head_safety, reason="failed", test_count=1)
    _write_safety_report(base_safety, reason="failed", test_count=1)
    changed_files.write_bytes(b"admin-ui/src/runtime.ts\0")

    with pytest.raises(RatchetError, match="file-level error"):
        compare_reports(
            head_report=head_report,
            base_report=base_report,
            head_package_root=head_root,
            base_package_root=base_root,
            package_repo_path=Path("admin-ui"),
            changed_files_path=changed_files,
            strict=True,
            head_safety_report_path=head_safety,
            base_safety_report_path=base_safety,
        )

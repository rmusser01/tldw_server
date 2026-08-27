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
    unhandled_error_count: int = 0,
    module_error_count: int = 0,
    hook_error_count: int = 0,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "reason": reason,
                "moduleCount": module_count,
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
def test_strict_validate_success_report_rejects_suite_hierarchy_mismatch(
    tmp_path: Path,
) -> None:
    """Reject suite counters that disagree with the assertion hierarchy."""

    package_root = tmp_path / "head" / "admin-ui"
    report_path = tmp_path / "head.json"
    safety_path = tmp_path / "head-safety.json"
    _write_success_report(report_path, package_root)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["numTotalTestSuites"] = 2
    payload["numPassedTestSuites"] = 2
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_safety_report(safety_path, reason="passed", test_count=1)

    with pytest.raises(RatchetError, match="suite.*assertion hierarchy"):
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
    _write_safety_report(safety_path, reason="passed", test_count=1)

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

import json
from pathlib import Path

import pytest
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
) -> None:
    test_results = []
    failed_test_count = 0
    for relative_path, full_names in failures.items():
        failed_test_count += len(full_names)
        test_results.append(
            {
                "name": str(package_root / relative_path),
                "status": "failed",
                "assertionResults": [
                    {
                        "fullName": full_name,
                        "status": "failed",
                        "failureMessages": ["expected true to be false"],
                    }
                    for full_name in full_names
                ],
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
        "numFailedTests": failed_test_count,
        "testResults": test_results,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_failing_test_files_rejects_collection_failures(tmp_path: Path) -> None:
    package_root = tmp_path / "head" / "apps" / "packages" / "ui"
    report_path = tmp_path / "head.json"
    _write_report(
        report_path,
        package_root,
        {},
        collection_failures=("src/import-error.test.ts",),
    )

    with pytest.raises(RatchetError, match="collection-level"):
        failing_test_files(report_path, package_root)


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

"""Run the required outbound privacy scenarios and emit only validated evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

from .outbound_backend_smoke import COVERAGE as BACKEND_COVERAGE
from .outbound_browser_smoke import COVERAGE as BROWSER_COVERAGE
from .outbound_browser_smoke import VIOLATION_CODES as BROWSER_VIOLATION_CODES
from .outbound_isolation import (
    CONTAINER_PREFLIGHT_FAILURE,
    CONTAINER_REPORT,
    IsolationError,
    assert_isolated_runtime,
    load_prepared_image,
    run_isolated,
)

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = {
    "backend": ("inventory", "backend-default", "backend-opt-in"),
    "browser": ("shipped-setup", "production-webui", "extension-worker", "extension-content-script"),
}
_PYTEST_GROUPS = ("inventory", "backend-opt-in")
_PYTEST_CASE_REASONS = {
    "failure": "pytest_case_failed",
    "error": "pytest_case_failed",
    "skipped": "pytest_case_skipped",
    "missing": "pytest_case_missing",
}
_PYTEST_RUN_REASONS = {
    "pytest_timed_out", "pytest_launch_failed", "pytest_evidence_missing",
    "pytest_evidence_invalid", "pytest_nonzero_exit",
}
_MAX_VIOLATIONS = 16
COVERAGE = {
    "inventory": ("reviewed-declarations", "direct-http-static-guard"),
    "backend-default": tuple(BACKEND_COVERAGE),
    "backend-opt-in": ("synthetic-hf-sdk-redirect", "synthetic-openai-adapter", "package-and-git-argv-only"),
    **{scenario: tuple(labels) for scenario, labels in BROWSER_COVERAGE.items()},
}
_FAILURES = {
    "runner": {
        "isolation_failed", "nonzero_container_exit", "invalid_scenario_evidence",
        "prepared_image_missing", "prepared_image_invalid", "prepared_image_stale",
        "container_timeout", "container_report_missing", "container_preflight_failed",
    },
    "backend": {
        "backend_child_failed", "backend_missing_shutdown_evidence", "backend_unexpected_network",
        "backend_import_outside_fixture", "backend_lifespan_failed", "backend_readiness_contract_failed",
        "backend_not_ready", "backend_stop_timeout", "backend_setup_redirect_failed", "backend_setup_status_failed",
        "backend_setup_defaults_failed", "backend_config_outside_fixture", "backend_setup_write_failed",
        "backend_setup_read_failed", "backend_setup_state_failed", "backend_setup_persistence_failed",
        "backend_note_write_failed", "backend_note_read_failed", "backend_note_persistence_failed",
        "backend_execution_failed",
    },
    **dict.fromkeys(_PYTEST_GROUPS, _PYTEST_RUN_REASONS.union(_PYTEST_CASE_REASONS.values())),
    **dict.fromkeys(SCENARIOS["browser"], BROWSER_VIOLATION_CODES),
}
_TESTS = {
    "CI/test_outbound_inventory.py": "inventory",
    "lint/test_http_usage_guard.py": "inventory",
    "CI/test_outbound_backend_smoke.py": "inventory",
    "CI/test_outbound_check.py": "inventory",
    "CI/test_outbound_opt_in.py": "backend-opt-in",
    "Setup/test_install_manager_dependencies.py": "backend-opt-in",
}
_TEST_MODULE_GROUPS = {
    f"tldw_Server_API.tests.{path.removesuffix('.py').replace('/', '.')}": group for path, group in _TESTS.items()
}
_OPT_IN_CASES = (
    "download-default", "download", "download-origin", "download-redirect", "download-file",
    "download-repository", "download-caught-dns", "provider", "provider-default", "provider-origin",
    "provider-caught-dns",
)
_REQUIRED_CASES = {
    (f"tldw_Server_API.tests.{module}", name)
    for module, name in {
        ("CI.test_outbound_inventory", "test_shipped_inventory_anchors_and_kokoro_identity"),
        ("lint.test_http_usage_guard", "test_outbound_http_goes_through_core_http_client"),
        ("CI.test_outbound_backend_smoke", "test_exact_numeric_and_static_names_never_hit_dns"),
        ("CI.test_outbound_backend_smoke", "test_app_configures_onnx_telemetry_before_native_imports[None-1]"),
        ("CI.test_outbound_check", "test_failed_scenario_is_valid_terminal_evidence"),
        ("Setup.test_install_manager_dependencies", "test_kokoro_package_install_argv_matches_reviewed_inventory[cpu-python-pip]"),
        ("Setup.test_install_manager_dependencies", "test_kokoro_package_install_argv_matches_reviewed_inventory[cuda-uv-fallback]"),
        ("Setup.test_install_manager_dependencies", "test_installer_vcs_argv_matches_reviewed_source_when_enabled[higgs-higgs-audio]"),
        ("Setup.test_install_manager_dependencies", "test_installer_vcs_argv_matches_reviewed_source_when_enabled[vibevoice-vibevoice]"),
        ("Setup.test_install_manager_dependencies", "test_omnivoice_sidecar_argv_matches_reviewed_inventory"),
    } | {
        ("CI.test_outbound_opt_in", f"test_real_download_and_provider_outbound_evidence[{case}]")
        for case in _OPT_IN_CASES
    }
}
_PYTEST_CHECKS = {
    group: {module.rsplit(".", 1)[1] for module, owner in _TEST_MODULE_GROUPS.items() if owner == group}
    | {name for module, name in _REQUIRED_CASES if _TEST_MODULE_GROUPS[module] == group}
    for group in _PYTEST_GROUPS
}
_BACKEND_MODULE = "Helper_Scripts.ci.outbound_check"
_BROWSER_MODULE = "Helper_Scripts.ci.outbound_browser_smoke"


class EvidenceError(ValueError):
    """Container evidence cannot be accepted or included in the public report."""


def _valid_violation(value: object, scenario: str) -> bool:
    if not isinstance(value, dict) or value.keys() not in ({"component", "reason"}, {"component", "reason", "check"}):
        return False
    component, reason = value["component"], value["reason"]
    owner = "backend" if scenario == "backend-default" else scenario
    if component not in ("runner", owner) or not isinstance(reason, str) or reason not in _FAILURES[component]:
        return False
    if reason in _PYTEST_CASE_REASONS.values():
        check = value.get("check")
        return isinstance(check, str) and check in _PYTEST_CHECKS[component]
    return "check" not in value


def validate_result(payload: object, expected: tuple[str, ...]) -> list[dict]:
    """Validate exact scenario coverage, status and known-safe diagnostics."""
    if not isinstance(payload, dict) or payload.keys() != {"scenarios"}:
        raise EvidenceError("invalid_scenario_evidence")
    records = payload["scenarios"]
    if not isinstance(records, list) or len(records) != len(expected):
        raise EvidenceError("invalid_scenario_evidence")
    by_scenario = {}
    for item in records:
        if not isinstance(item, dict) or item.keys() != {"scenario", "status", "coverage", "violations"}:
            raise EvidenceError("invalid_scenario_evidence")
        scenario, status, coverage, violations = (item[key] for key in ("scenario", "status", "coverage", "violations"))
        if not isinstance(scenario, str) or scenario not in expected or scenario in by_scenario:
            raise EvidenceError("invalid_scenario_evidence")
        if status == "pass" and coverage == list(COVERAGE[scenario]) and violations == []:
            by_scenario[scenario] = item
        elif status == "fail" and coverage == [] and isinstance(violations, list) and 0 < len(violations) <= _MAX_VIOLATIONS:
            if not all(_valid_violation(value, scenario) for value in violations):
                raise EvidenceError("invalid_scenario_evidence")
            by_scenario[scenario] = item
        else:
            raise EvidenceError("invalid_scenario_evidence")
    return [by_scenario[scenario] for scenario in expected]


def _failure(scenario: str, component: str, reason: str) -> dict:
    return {"scenario": scenario, "status": "fail", "coverage": [],
            "violations": [{"component": component, "reason": reason}]}


def _success(scenario: str) -> dict:
    return {"scenario": scenario, "status": "pass", "coverage": list(COVERAGE[scenario]), "violations": []}


def _write_report(path: Path, scenarios: list[dict]) -> None:
    path.write_text(json.dumps({"scenarios": scenarios}, indent=2) + "\n", encoding="utf-8")


def _read_pytest_cases(path: Path) -> dict[tuple[str, str], str]:
    from defusedxml import ElementTree
    from defusedxml.common import DefusedXmlException

    try:
        if path.stat().st_size > 1024 * 1024:
            raise EvidenceError("pytest_evidence_invalid")
        root = ElementTree.parse(path).getroot()
    except FileNotFoundError:
        raise EvidenceError("pytest_evidence_missing") from None
    except (OSError, ElementTree.ParseError, DefusedXmlException):
        raise EvidenceError("pytest_evidence_invalid") from None
    if root.tag != "testsuites":
        raise EvidenceError("pytest_evidence_invalid")
    cases = {}
    for case in root.findall("./testsuite/testcase"):
        name, class_name = case.get("name"), case.get("classname")
        outcome = case[0].tag if len(case) else "passed"
        if (
            not name or class_name not in _TEST_MODULE_GROUPS or len(case) > 1
            or outcome not in {"passed", "failure", "error", "skipped"}
        ):
            raise EvidenceError("pytest_evidence_invalid")
        entry = (class_name, name)
        if entry in cases:
            raise EvidenceError("pytest_evidence_invalid")
        cases[entry] = outcome
    if not cases:
        raise EvidenceError("pytest_evidence_invalid")
    return cases


def _pytest_run_failure(reason: str) -> dict[str, dict]:
    return {group: _failure(group, group, reason) for group in _PYTEST_GROUPS}


def _pytest_scenarios(path: Path, exit_code: int) -> dict[str, dict]:
    """Require all selected tests and report only allowlisted failure identifiers."""
    try:
        cases = _read_pytest_cases(path)
    except EvidenceError as exc:
        return _pytest_run_failure(str(exc))
    problems = {group: set() for group in _PYTEST_GROUPS}
    for entry in cases.keys() | _REQUIRED_CASES:
        outcome = cases.get(entry, "missing")
        if outcome != "passed":
            module, name = entry
            hint = name if entry in _REQUIRED_CASES else module.rsplit(".", 1)[1]
            problems[_TEST_MODULE_GROUPS[module]].add((_PYTEST_CASE_REASONS[outcome], hint))
    if exit_code and not any(problems.values()):
        return _pytest_run_failure("pytest_nonzero_exit")
    return {
        group: {
            "scenario": group, "status": "fail", "coverage": [],
            "violations": [
                {"component": group, "reason": reason, "check": hint}
                for reason, hint in sorted(rows)[:_MAX_VIOLATIONS]
            ],
        } if rows else _success(group)
        for group, rows in problems.items()
    }


def _run_backend_inside() -> int:
    from .outbound_backend_smoke import run_backend_smoke

    with tempfile.TemporaryDirectory(prefix="outbound-tests-") as directory:
        work = Path(directory)
        unit_report = work / "pytest.xml"
        argv = [
            sys.executable, "-m", "pytest", "--noconftest", "-o", "addopts=", "-p", "no:cacheprovider",
            "--junitxml", str(unit_report), "-q", *(f"tldw_Server_API/tests/{test}" for test in _TESTS),
        ]
        env = {
            "PATH": os.defpath + ":/usr/local/bin",
            "HOME": str(work),
            "TMPDIR": str(work),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "TLDW_OUTBOUND_TEST_ISOLATED": "1",
        }
        try:
            tests = subprocess.run(  # nosec B603
                argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=120, check=False,
            )
        except subprocess.TimeoutExpired:
            cases = _pytest_run_failure("pytest_timed_out")
        except OSError:
            cases = _pytest_run_failure("pytest_launch_failed")
        else:
            cases = _pytest_scenarios(unit_report, tests.returncode)
        backend = work / "backend.json"
        run_backend_smoke(ROOT, backend)
        cases["backend-default"] = validate_result(json.loads(backend.read_text()), ("backend-default",))[0]
        result = [cases[scenario] for scenario in SCENARIOS["backend"]]
        _write_report(Path(CONTAINER_REPORT), result)
        return int(any(item["status"] != "pass" for item in result))


def _run_slice(target: str, report_dir: Path) -> list[dict]:
    expected = SCENARIOS[target]
    if target == "backend":
        command, timeout = ["-m", _BACKEND_MODULE, "--backend-inside"], 280
    else:
        command, timeout = ["-m", _BROWSER_MODULE, "--inside-container"], 420
    try:
        image_id = load_prepared_image(ROOT, target, report_dir)
        with tempfile.TemporaryDirectory(prefix="outbound-private-") as directory:
            work = Path(directory)
            exit_code = run_isolated(image_id, command, work, timeout=timeout)
            result_path = work / "result.json"
            if result_path.stat().st_size > 256 * 1024:
                raise EvidenceError("invalid_scenario_evidence")
            payload = json.loads(result_path.read_text())
            if payload == CONTAINER_PREFLIGHT_FAILURE and exit_code == 1:
                raise IsolationError("outbound container preflight failed", code="container_preflight_failed")
            result = validate_result(payload, expected)
        if exit_code == 0 or any(item["status"] == "fail" for item in result):
            return result
        reason = "nonzero_container_exit"
    except IsolationError as exc:
        reason = exc.code if exc.code in _FAILURES["runner"] else "isolation_failed"
    except ValueError:
        reason = "invalid_scenario_evidence"
    # Raw child and Docker exceptions are never public evidence.
    except Exception:  # noqa: BLE001
        reason = "isolation_failed"
    return [_failure(scenario, "runner", reason) for scenario in expected]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--backend-inside"]:
        try:
            assert_isolated_runtime()
        except IsolationError:
            print("Backend fixture requires the isolated container preflight.", file=sys.stderr)
            return 1
        return _run_backend_inside()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=(*SCENARIOS, "all"))
    parser.add_argument("--report-dir", required=True, type=Path)
    config = parser.parse_args(args)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="outbound-run-", dir=config.report_dir))
    targets = tuple(SCENARIOS) if config.target == "all" else (config.target,)
    records = [record for target in targets for record in _run_slice(target, config.report_dir)]
    path = output / "report.json"
    _write_report(path, records)
    for record in records:
        reasons = ", ".join(
            f"{item['component']}/{item['reason']}" + (f" [{item['check']}]" if "check" in item else "")
            for item in record["violations"]
        )
        print(f"{record['scenario']} {record['status']}" + (f" ({reasons})" if reasons else ""))
    print(f"Outbound privacy report {path}")
    return int(any(record["status"] != "pass" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())

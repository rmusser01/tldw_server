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
COVERAGE = {
    "inventory": ("reviewed-declarations", "direct-http-static-guard"),
    "backend-default": tuple(BACKEND_COVERAGE),
    "backend-opt-in": ("synthetic-hf-sdk-redirect", "synthetic-openai-adapter", "package-and-git-argv-only"),
    **{scenario: tuple(labels) for scenario, labels in BROWSER_COVERAGE.items()},
}
_FAILURES = {
    "runner": {"isolation_failed", "nonzero_container_exit", "invalid_scenario_evidence"},
    "inventory": {"required_pytest_failed"},
    "backend": {
        "backend_child_failed", "backend_missing_shutdown_evidence", "backend_unexpected_network",
        "backend_import_outside_fixture", "backend_lifespan_failed", "backend_readiness_contract_failed",
        "backend_not_ready", "backend_stop_timeout", "backend_setup_redirect_failed", "backend_setup_status_failed",
        "backend_setup_defaults_failed", "backend_config_outside_fixture", "backend_setup_write_failed",
        "backend_setup_read_failed", "backend_setup_state_failed", "backend_setup_persistence_failed",
        "backend_note_write_failed", "backend_note_read_failed", "backend_note_persistence_failed",
        "backend_execution_failed",
    },
    "backend-opt-in": {"required_pytest_failed"},
    **dict.fromkeys(SCENARIOS["browser"], BROWSER_VIOLATION_CODES),
}
_TESTS = (
    "CI/test_outbound_inventory.py", "lint/test_http_usage_guard.py", "CI/test_outbound_backend_smoke.py",
    "CI/test_outbound_opt_in.py", "Setup/test_install_manager_dependencies.py",
    "CI/test_outbound_check.py",
)
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
_BACKEND_MODULE = "Helper_Scripts.ci.outbound_check"
_BROWSER_MODULE = "Helper_Scripts.ci.outbound_browser_smoke"


class EvidenceError(ValueError):
    """Container evidence cannot be accepted or included in the public report."""


def _valid_violation(value: object) -> bool:
    if not isinstance(value, dict) or value.keys() != {"component", "reason"}:
        return False
    component, reason = value["component"], value["reason"]
    return (
        isinstance(component, str) and isinstance(reason, str)
        and reason in _FAILURES.get(component, ())
    )


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
        elif status == "fail" and coverage == [] and isinstance(violations, list) and 0 < len(violations) <= 16:
            if not all(_valid_violation(value) for value in violations):
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


def pytest_evidence(path: Path, required: set[tuple[str, str]]) -> bool:
    """Require every expected test and a clean exit for all reported test cases."""
    from defusedxml import ElementTree
    from defusedxml.common import DefusedXmlException

    try:
        if path.stat().st_size > 1024 * 1024:
            return False
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError, DefusedXmlException):
        return False
    if root.tag != "testsuites":
        return False
    seen = set()
    for case in root.findall("./testsuite/testcase"):
        name, class_name = case.get("name"), case.get("classname")
        if not name or not class_name or len(case):
            return False
        entry = (class_name, name)
        if entry in seen:
            return False
        seen.add(entry)
    return bool(seen) and required <= seen


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
            passed = tests.returncode == 0 and pytest_evidence(unit_report, _REQUIRED_CASES)
        except (OSError, subprocess.TimeoutExpired):
            passed = False
        cases = {
            scenario: _success(scenario) if passed else _failure(scenario, scenario, "required_pytest_failed")
            for scenario in ("inventory", "backend-opt-in")
        }
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
            result = validate_result(json.loads(result_path.read_text()), expected)
        if exit_code == 0 or any(item["status"] == "fail" for item in result):
            return result
        reason = "nonzero_container_exit"
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
        reasons = ", ".join(item["reason"] for item in record["violations"])
        print(f"{record['scenario']} {record['status']}" + (f" ({reasons})" if reasons else ""))
    print(f"Outbound privacy report {path}")
    return int(any(record["status"] != "pass" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())

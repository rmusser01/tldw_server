"""Contract for selecting outbound scenarios and accepting only terminal evidence."""

import json

import pytest
from Helper_Scripts.ci import outbound_check as check
from Helper_Scripts.ci import outbound_isolation as isolation


def record(scenario):
    return {"scenario": scenario, "status": "pass", "coverage": list(check.COVERAGE[scenario]), "violations": []}


@pytest.mark.parametrize("selection,expected", [
    ("backend", ("inventory", "backend-default", "backend-opt-in")),
    ("browser", ("shipped-setup", "production-webui", "extension-worker", "extension-content-script")),
    ("all", ("inventory", "backend-default", "backend-opt-in", "shipped-setup",
             "production-webui", "extension-worker", "extension-content-script")),
])
def test_selected_scenarios_run_and_fresh_reports_are_separate(monkeypatch, tmp_path, selection, expected):
    def run_container(image, argv, report, *, timeout):
        assert timeout > 0
        cases = check.SCENARIOS[image]
        (report / "result.json").write_text(json.dumps({"scenarios": [record(case) for case in cases]}))
        return 0

    monkeypatch.setattr(check, "load_prepared_image", lambda root, target, path: target)
    monkeypatch.setattr(check, "run_isolated", run_container)
    for _ in range(2):
        assert check.main([selection, "--report-dir", str(tmp_path)]) == 0
    reports = sorted(tmp_path.glob("outbound-run-*/report.json"))
    assert len(reports) == 2
    for report in reports:
        assert [entry["scenario"] for entry in json.loads(report.read_text())["scenarios"]] == list(expected)


@pytest.mark.parametrize("bad_records", [
    lambda result: [],
    lambda result: [result, result],
    lambda result: [result | {"status": "skipped"}],
    lambda result: [result | {"coverage": []}],
    lambda result: [result | {"coverage": ["possibly good"]}],
    lambda result: [result | {"detail": "private path"}],
    lambda result: [result | {"scenario": "another-scenario"}],
    lambda result: [result | {"violations": [{"reason": "backend_child_failed", "component": "backend"}]}],
])
def test_missing_duplicate_skipped_and_malformed_evidence_fails(bad_records):
    result = record("backend-default")
    with pytest.raises(check.EvidenceError, match="invalid_scenario_evidence"):
        check.validate_result({"scenarios": bad_records(result)}, ("backend-default",))


def test_failed_scenario_is_valid_terminal_evidence():
    result = {"scenario": "backend-default", "status": "fail", "coverage": [],
              "violations": [{"reason": "backend_child_failed", "component": "backend"}]}
    assert check.validate_result({"scenarios": [result]}, ("backend-default",)) == [result]


def test_internal_backend_runner_requires_container_preflight(monkeypatch):
    monkeypatch.delenv("TLDW_OUTBOUND_PREFLIGHT", raising=False)
    monkeypatch.setattr(check, "_run_backend_inside", lambda: pytest.fail("launched the backend tests"))
    assert check.main(["--backend-inside"]) == 1


@pytest.mark.parametrize("failure", ["missing", "exit", "crash", "raw-report"])
def test_fail_closed_and_keep_raw_credentials_out_of_every_report(monkeypatch, tmp_path, failure, capsys):
    fixture_url = "https://private.invalid/customer/path?token=OUTBOUND_SECRET"

    def run_container(image, argv, report, *, timeout):
        if failure == "crash":
            raise RuntimeError(fixture_url)
        cases = check.SCENARIOS[image]
        results = [record(case) for case in cases]
        if failure == "raw-report":
            results[0] |= {"status": "fail", "coverage": [], "violations": [
                {"component": "backend", "reason": fixture_url},
            ]}
        if failure != "missing":
            (report / "result.json").write_text(json.dumps({"scenarios": results}))
        return 1 if failure == "exit" else 0

    monkeypatch.setattr(check, "load_prepared_image", lambda root, target, path: target)
    monkeypatch.setattr(check, "run_isolated", run_container)
    assert check.main(["backend", "--report-dir", str(tmp_path)]) != 0
    (report,) = tmp_path.glob("**/*.json")
    results = json.loads(report.read_text())["scenarios"]
    assert any(entry["status"] == "fail" for entry in results)
    assert "OUTBOUND_SECRET" not in report.read_text()
    output = capsys.readouterr()
    assert "private.invalid" not in output.out + output.err


def test_cli_reports_preflight_failure_separately_from_bad_result(monkeypatch, tmp_path, capsys):
    def refused(_):
        raise isolation.IsolationError("private fixture path")

    def run_container(image, argv, report, *, timeout):
        monkeypatch.setattr(isolation, "_preflight", refused)
        monkeypatch.setattr(isolation, "CONTAINER_REPORT", str(report / "result.json"))
        return isolation.main(["_inside", "/unused", "--", "-c"])

    monkeypatch.setattr(check, "load_prepared_image", lambda root, target, path: target)
    monkeypatch.setattr(check, "run_isolated", run_container)
    assert check.main(["backend", "--report-dir", str(tmp_path)]) == 1
    stdout = capsys.readouterr().out
    assert "runner/container_preflight_failed" in stdout
    assert "private fixture path" not in stdout


@pytest.mark.parametrize("code", ["prepared_image_stale", "container_timeout"])
def test_cli_preserves_safe_isolation_reason(monkeypatch, tmp_path, capsys, code):
    def unavailable(*args, **kwargs):
        raise isolation.IsolationError("https://private.invalid/OUTBOUND_SECRET", code=code)

    monkeypatch.setattr(check, "load_prepared_image", unavailable)
    assert check.main(["backend", "--report-dir", str(tmp_path)]) == 1
    stdout = capsys.readouterr().out
    assert f"runner/{code}" in stdout
    assert "OUTBOUND_SECRET" not in stdout


_OPT_IN_CASE = (
    "tldw_Server_API.tests.CI.test_outbound_opt_in",
    "test_real_download_and_provider_outbound_evidence[download-default]",
)


def _junit_report(path, overrides, extra=()):
    cases = []
    for module, name in sorted(check._REQUIRED_CASES):
        child = overrides.get((module, name), "")
        if child is not None:
            cases.append(f'<testcase classname="{module}" name="{name}">{child}</testcase>')
    path.write_text("<testsuites><testsuite>" + "".join(cases + list(extra)) + "</testsuite></testsuites>")


@pytest.mark.parametrize(("child", "status", "reason"), [
    ("<failure message='OUTBOUND_SECRET'/>", 1, "pytest_case_failed"),
    ("<skipped/>", 0, "pytest_case_skipped"),
    (None, 0, "pytest_case_missing"),
])
def test_pytest_evidence_identifies_the_affected_scenario_and_case(tmp_path, child, status, reason):
    report = tmp_path / "pytest.xml"
    _junit_report(report, {_OPT_IN_CASE: child})
    results = check._pytest_scenarios(report, status)
    assert results["inventory"] == record("inventory")
    expected = {"component": "backend-opt-in", "reason": reason, "check": _OPT_IN_CASE[1]}
    assert results["backend-opt-in"]["violations"] == [expected]
    assert check.validate_result(
        {"scenarios": list(results.values())}, ("inventory", "backend-opt-in")
    ) == list(results.values())
    assert "OUTBOUND_SECRET" not in json.dumps(results)


def test_pytest_evidence_uses_only_allowlisted_case_or_file_ids(tmp_path):
    report = tmp_path / "pytest.xml"
    _junit_report(report, {}, extra=[
        f'<testcase classname="{_OPT_IN_CASE[0]}" name="OUTBOUND_SECRET"><failure/></testcase>',
    ])
    result = check._pytest_scenarios(report, 1)["backend-opt-in"]
    assert result["violations"] == [
        {"component": "backend-opt-in", "reason": "pytest_case_failed", "check": "test_outbound_opt_in"}
    ]
    result["violations"][0]["check"] = "OUTBOUND_SECRET"
    with pytest.raises(check.EvidenceError, match="invalid_scenario_evidence"):
        check.validate_result({"scenarios": [result]}, ("backend-opt-in",))


def test_pytest_evidence_rejects_duplicate_results(tmp_path):
    report = tmp_path / "pytest.xml"
    module, name = _OPT_IN_CASE
    _junit_report(report, {}, extra=[f'<testcase classname="{module}" name="{name}"/>'])
    results = check._pytest_scenarios(report, 0)
    assert all(record["status"] == "fail" for record in results.values())

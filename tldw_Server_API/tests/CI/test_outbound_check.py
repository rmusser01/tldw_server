"""Contract for selecting outbound scenarios and accepting only terminal evidence."""

import json

import pytest
from Helper_Scripts.ci import outbound_check as check


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
    assert "private.invalid" not in capsys.readouterr().err


def test_pytest_evidence_requires_every_required_case_and_no_skip(tmp_path):
    xml = tmp_path / "pytest.xml"
    class_name = "tldw_Server_API.tests.CI.test_outbound_opt_in"
    case_id = "test_real_download_and_provider_outbound_evidence[download-default]"
    parent = '<testsuites><testsuite>{}</testsuite></testsuites>'
    good = f'<testcase classname="{class_name}" name="{case_id}"/>'
    skipped = f'<testcase classname="{class_name}" name="{case_id}"><skipped/></testcase>'
    for cases in (skipped, "", good + good):
        xml.write_text(parent.format(cases))
        assert not check.pytest_evidence(xml, {(class_name, case_id)})
    xml.write_text(parent.format(good))
    assert check.pytest_evidence(xml, {(class_name, case_id)})

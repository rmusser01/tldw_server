from pathlib import Path

import yaml


def test_ci_runs_governed_preflight_final_gate_on_python310() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["preflight-python-310"]
    steps = job["steps"]

    setup = next(step for step in steps if step.get("name") == "Setup Python 3.10 and dependencies")
    assert setup["uses"] == "./.github/actions/setup-python-deps"
    assert setup["with"]["python-version"] == "3.10"

    run = next(step for step in steps if step.get("name") == "Run governed preflight final gate")["run"]
    assert "test_phase3_preflight_architecture.py" in run
    assert "test_phase3_preflight_browser.py" in run
    assert "test_phase3_preflight_browser_analyzers.py" in run

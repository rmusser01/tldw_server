from pathlib import Path

import pytest
import yaml


def test_frontend_required_runs_family_guardrails_e2e_for_targeted_changes() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = data["jobs"]["frontend-required"]["steps"]

    matching_steps = [step for step in steps if step.get("name") == "Run family guardrails e2e"]

    assert len(matching_steps) == 1
    step = matching_steps[0]
    assert "family_guardrails_changed" in step["if"]
    assert step["working-directory"] == "apps/tldw-frontend"
    assert step["run"] == "bun run e2e:family-guardrails"


def test_frontend_required_does_not_publish_or_enforce_license_policy() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = data["jobs"]["frontend-required"]["steps"]

    checkout = next(step for step in steps if step.get("name") == "Checkout")
    step_names = {step.get("name") for step in steps}
    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert checkout["if"] == "needs.changes.outputs.frontend_changed == 'true'"
    assert "fetch-depth" not in checkout.get("with", {})
    assert "Enforce temporary frontend licensing contribution freeze" not in step_names
    assert "check_frontend_license_gate.py" not in workflow_text
    assert "frontend-license-policy/trusted/" not in workflow_text


@pytest.mark.unit
def test_frontend_coverage_report_cannot_starve_required_gates() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = data["jobs"]["frontend-required"]["steps"]

    coverage_step = next(
        step for step in steps if step.get("name") == "Frontend coverage summary (report-only)"
    )

    assert coverage_step["continue-on-error"] is True
    assert coverage_step["timeout-minutes"] == 17
    assert "timeout --kill-after=30s 15m bun run test:coverage" in coverage_step["run"]

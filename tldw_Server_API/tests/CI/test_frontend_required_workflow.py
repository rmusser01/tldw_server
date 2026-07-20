from pathlib import Path

import yaml


def test_frontend_required_runs_family_guardrails_e2e_for_targeted_changes() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = data["jobs"]["frontend-required"]["steps"]

    matching_steps = [
        step
        for step in steps
        if step.get("name") == "Run family guardrails e2e"
    ]

    assert len(matching_steps) == 1
    step = matching_steps[0]
    assert "family_guardrails_changed" in step["if"]
    assert step["working-directory"] == "apps/tldw-frontend"
    assert step["run"] == "bun run e2e:family-guardrails"


def test_frontend_required_uses_base_branch_licensing_gate() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = data["jobs"]["frontend-required"]["steps"]

    checkout = next(step for step in steps if step.get("name") == "Checkout")
    gate = next(
        step
        for step in steps
        if step.get("name")
        == "Enforce temporary frontend licensing contribution freeze"
    )

    assert checkout["with"]["fetch-depth"] == 0
    assert "github.event.pull_request.user.login" in str(gate["env"])
    assert "git show" in gate["run"]
    assert ":Helper_Scripts/ci/check_frontend_license_gate.py" in gate["run"]
    assert "git diff --name-only" in gate["run"]

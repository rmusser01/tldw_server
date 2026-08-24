from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
def test_frontend_required_budget_covers_broad_changed_suite() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert data["jobs"]["frontend-required"]["timeout-minutes"] >= 120


@pytest.mark.unit
def test_frontend_required_bounds_pathological_impact_expansion() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = data["jobs"]["frontend-unit-tests"]["steps"]
    unit_step = next(
        step
        for step in steps
        if step.get("name") == "Run package-owned frontend unit tests"
    )
    script = unit_step["run"]

    assert 'IMPACTED_TEST_LIMIT="500"' in script
    assert 'bunx vitest list --changed="${BASE_SHA}" --filesOnly' in script
    assert 'git diff --name-only --diff-filter=ACMR "$BASE_SHA" "$HEAD_SHA"' in script
    assert "USE_DIRECT_TESTS=1" in script
    assert (
        'package_vitest_args=("${direct_test_files[@]}" "${common_vitest_args[@]}")'
        in script
    )
    assert '"${head_command[@]}"' in script
    assert "No directly changed frontend tests were found" in script


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
    assert checkout["with"]["fetch-depth"] == 0
    assert "Enforce temporary frontend licensing contribution freeze" not in step_names
    assert "check_frontend_license_gate.py" not in workflow_text
    assert "frontend-license-policy/trusted/" not in workflow_text


@pytest.mark.unit
def test_frontend_required_fails_closed_on_unit_shard_outcomes() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    jobs = data["jobs"]
    unit_job = jobs["frontend-unit-tests"]
    final_job = jobs["frontend-required"]

    assert "needs.admission.outputs.should_run == 'true'" in unit_job["if"]
    assert "needs.changes.result == 'success'" in unit_job["if"]
    assert "needs.changes.outputs.tldw_frontend_changed == 'true'" in unit_job["if"]
    assert final_job["needs"] == ["changes", "admission", "frontend-unit-tests"]

    guard = next(
        step
        for step in final_job["steps"]
        if step.get("name") == "Require frontend unit shard success"
    )
    assert guard["env"] == {
        "TLDW_FRONTEND_CHANGED": "${{ needs.changes.outputs.tldw_frontend_changed }}",
        "UNIT_SHARDS_RESULT": "${{ needs.frontend-unit-tests.result }}",
    }
    assert '"$TLDW_FRONTEND_CHANGED" == "false"' in guard["run"]
    assert '"$TLDW_FRONTEND_CHANGED" != "true"' not in guard["run"]
    assert '"$UNIT_SHARDS_RESULT" == "success"' in guard["run"]
    assert '"$UNIT_SHARDS_RESULT" == "skipped"' in guard["run"]
    assert "exit 1" in guard["run"]


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

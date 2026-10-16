import json
import os
import shutil
import subprocess  # nosec B404 - exercises the repository-owned CI guard in a scratch directory
from pathlib import Path

import pytest
import yaml


def test_frontend_required_enforces_strict_shared_security_boundaries() -> None:
    """The incremental strict project must execute as a failing merge gate."""
    data = yaml.safe_load(Path(".github/workflows/frontend-required.yml").read_text())
    step = next(
        step
        for step in data["jobs"]["frontend-required"]["steps"]
        if step.get("name") == "Run strict shared-boundary typecheck"
    )
    assert step["if"] == "needs.changes.outputs.tldw_frontend_changed == 'true'"
    assert step["working-directory"] == "apps/tldw-frontend"
    assert step["run"] == "bunx tsc --project tsconfig.strict.json"
    assert not step.get("continue-on-error", False)
    strict_project = json.loads(Path("apps/tldw-frontend/tsconfig.strict.json").read_text())
    assert strict_project["compilerOptions"]["strict"] is True
    assert "../packages/ui/src/utils/absolute-url-guard.ts" in strict_project["include"]
    assert "../packages/ui/src/utils/api-key.ts" in strict_project["include"]


def test_frontend_required_gates_webui_typecheck() -> None:
    data = yaml.safe_load(Path(".github/workflows/frontend-required.yml").read_text())
    step = next(
        step for step in data["jobs"]["frontend-required"]["steps"] if step.get("name") == "Run WebUI typecheck"
    )
    assert step["if"] == "needs.changes.outputs.tldw_frontend_changed == 'true'"
    assert step["working-directory"] == "apps/tldw-frontend"
    assert step["run"] == "bun run typecheck --incremental false"
    assert not step.get("continue-on-error", False)


@pytest.mark.unit
def test_frontend_required_budget_covers_broad_changed_suite() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))

    assert data["jobs"]["frontend-required"]["timeout-minutes"] >= 120


@pytest.mark.unit
def test_frontend_required_handles_indirect_pathological_expansion() -> None:
    workflow_path = Path(".github/workflows/frontend-required.yml")
    data = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = data["jobs"]["frontend-unit-tests"]["steps"]
    unit_step = next(step for step in steps if step.get("name") == "Run package-owned frontend unit tests")
    script = unit_step["run"]

    assert 'IMPACTED_TEST_LIMIT="500"' in script
    assert 'bunx vitest list --changed="${BASE_SHA}" --filesOnly' in script
    assert 'git diff --name-only --diff-filter=ACMR "$BASE_SHA" "$HEAD_SHA"' in script
    assert "USE_DIRECT_TESTS=1" in script
    assert 'package_vitest_args=("${direct_test_files[@]}" "${common_vitest_args[@]}")' in script
    assert '"${head_command[@]}"' in script
    assert "No directly changed frontend tests were found" not in script
    assert "retaining the sharded dependency-impact set" in script
    assert "USE_DIRECT_TESTS=0" in script


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
    assert "needs.await_license.outputs.license_passed == 'true'" in unit_job["if"]
    assert final_job["needs"] == ["changes", "admission", "frontend-unit-tests", "await_license"]

    guard = next(step for step in final_job["steps"] if step.get("name") == "Require frontend unit shard success")
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

    coverage_step = next(step for step in steps if step.get("name") == "Run all WebUI-owned tests with coverage")

    assert not coverage_step.get("continue-on-error", False)
    assert coverage_step["timeout-minutes"] == 17
    assert "--maxWorkers=2" in coverage_step["run"]
    assert "timeout --kill-after=30s 15m bun run test:coverage" in coverage_step["run"]


@pytest.mark.unit
def test_frontend_coverage_reports_both_package_owned_suites() -> None:
    """Full shared coverage is bounded independently of the required gate."""
    data = yaml.safe_load(Path(".github/workflows/frontend-required.yml").read_text())
    jobs = data["jobs"]
    webui = next(
        step for step in jobs["frontend-required"]["steps"]
        if step.get("name") == "Run all WebUI-owned tests with coverage"
    )
    assert webui["working-directory"] == "apps/tldw-frontend"
    assert not webui.get("continue-on-error", False)
    coverage = jobs["shared-ui-coverage"]
    shards = coverage["strategy"]["matrix"]["shard"]
    assert shards == list(range(1, len(shards) + 1))
    run = next(step for step in coverage["steps"] if step.get("name") == "Run shared UI coverage shard")
    assert run["working-directory"] == "apps/packages/ui"
    assert "--shard=" in run["run"]
    assert "--exclude" not in run["run"]
    assert "timeout --kill-after=30s 15m" in run["run"]
    assert "shared-ui-coverage-report" not in jobs["frontend-required"]["needs"]
    report = jobs["shared-ui-coverage-report"]
    assert "shared-ui-coverage" in report["needs"]
    assert "needs.shared-ui-coverage.result == 'success'" in report["if"]
    guard = next(step for step in report["steps"] if step.get("name") == "Validate coverage shards")
    assert int(guard["env"]["SHARD_COUNT"]) == len(shards)


@pytest.mark.unit
@pytest.mark.parametrize("missing", [None, 1, 8, 16])
def test_shared_coverage_merge_rejects_missing_shards(tmp_path: Path, missing: int | None) -> None:
    """Native Vitest accepts partial blob sets; our boundary must reject them."""
    data = yaml.safe_load(Path(".github/workflows/frontend-required.yml").read_text())
    report = data["jobs"]["shared-ui-coverage-report"]
    guard = next(step for step in report["steps"] if step.get("name") == "Validate coverage shards")
    count = int(guard["env"]["SHARD_COUNT"])
    for shard in range(1, count + 1):
        if shard != missing:
            (tmp_path / f"blob-{shard}-{count}.json").write_text("{}")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Workflow guard requires bash")
    result = subprocess.run(  # nosec B603 - trusted workflow guard with only temporary report files
        [bash, "-c", guard["run"]],
        cwd=tmp_path,
        env={**os.environ, **guard["env"], "REPORTS_DIR": str(tmp_path)},
        timeout=10,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == (missing is None)


@pytest.mark.unit
def test_ux_gates_use_explicit_playwright_browser_instead_of_puppeteer_download() -> None:
    data = yaml.safe_load(Path(".github/workflows/frontend-ux-gates.yml").read_text())
    for name in ("onboarding-gate", "smoke-gate"):
        steps = data["jobs"][name]["steps"]
        install = next(step for step in steps if step.get("name") == "Install frontend dependencies")
        assert install["env"]["PUPPETEER_SKIP_DOWNLOAD"] == "true"
        assert install["run"] == "bun install --frozen-lockfile"
        browser = next(step for step in steps if step.get("name") == "Install Playwright Chromium")
        assert "playwright install --with-deps chromium" in browser["run"]
        assert not browser.get("continue-on-error", False)
        assert steps.index(install) < steps.index(browser)

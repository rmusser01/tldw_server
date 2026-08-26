"""Contracts for the admin UI exact-base Vitest failure ratchet."""

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
def test_vitest_digest_contract_documents_event_dependent_trust() -> None:
    """Do not present PR-controlled digest checks as a trusted-code boundary."""

    workflow_source = Path(".github/workflows/frontend-required.yml").read_text(
        encoding="utf-8"
    )

    assert "DIGEST_TRUST_BOUNDARY:" in workflow_source
    assert "workflow_run uses the default-branch workflow definition" in workflow_source
    assert "pull_request is a same-workflow consistency check" in workflow_source
    assert "workflow_dispatch follows the selected workflow ref" in workflow_source
    assert "trusted workflow digest" not in workflow_source


@pytest.mark.unit
def test_admin_ui_unit_gate_uses_fail_closed_exact_base_ratchet() -> None:
    """Keep inherited failures bounded without weakening later admin UI gates."""

    workflow = yaml.safe_load(
        Path(".github/workflows/frontend-required.yml").read_text(encoding="utf-8")
    )
    triggers = workflow.get("on", workflow.get(True))
    dispatch_input = triggers["workflow_dispatch"]["inputs"]["base_sha"]
    assert dispatch_input["required"] is True
    assert dispatch_input["type"] == "string"

    steps = workflow["jobs"]["frontend-required"]["steps"]
    named_steps = {step.get("name"): step for step in steps}

    install_step = named_steps["Install admin-ui dependencies"]
    assert install_step["run"] == "bun install --frozen-lockfile"

    unit_step = named_steps["Run admin-ui unit tests"]
    assert unit_step["shell"] == "bash"
    assert unit_step.get("continue-on-error") is not True
    assert "working-directory" not in unit_step

    run_script = str(unit_step["run"])
    required_contracts = (
        "set -uo pipefail",
        'BASE_SHA="${{ needs.admission.outputs.base_sha }}"',
        "GITHUB_EVENT_PATH",
        '["pull_request"]["base"]["sha"]',
        'BASE_SHA="${{ inputs.base_sha }}"',
        '[[ ! "$BASE_SHA" =~ ^[0-9a-fA-F]{40}$',
        'git diff --name-only -z --diff-filter=ACMRT "$BASE_SHA" "$HEAD_SHA"',
        'git worktree add --detach "$BASE_WORKTREE" "$BASE_SHA"',
        "bun install --frozen-lockfile",
        'RATCHET_SCRIPT="${GITHUB_WORKSPACE}/Helper_Scripts/ci/vitest_base_ratchet.py"',
        'SAFETY_REPORTER="${GITHUB_WORKSPACE}/admin-ui/scripts/ci/vitest-safety-reporter.mjs"',
        'EXPECTED_RATCHET_SHA256=',
        'EXPECTED_SAFETY_REPORTER_SHA256=',
        'sha256sum --check --status',
        '"--passWithNoTests=false"',
        '"--reporter=default"',
        '"--reporter=json"',
        '"--reporter=${SAFETY_REPORTER}"',
        '"--outputFile.json=${HEAD_REPORT}"',
        'TLDW_VITEST_SAFETY_REPORT="$HEAD_SAFETY_REPORT"',
        "if (( HEAD_STATUS != 1 )); then",
        'python3 "$RATCHET_SCRIPT" validate-success',
        "--strict",
        '--safety-report "$HEAD_SAFETY_REPORT"',
        'python3 "$RATCHET_SCRIPT" extract',
        'FAILED_FILES+=("./${test_file}")',
        'git diff --quiet --no-ext-diff "$BASE_SHA" "$HEAD_SHA" -- "admin-ui/${test_file}"',
        'bunx vitest run "${FAILED_FILES[@]}"',
        'TLDW_VITEST_SAFETY_REPORT="$BASE_SAFETY_REPORT"',
        "if (( BASE_STATUS != 1 )); then",
        'python3 "$RATCHET_SCRIPT" compare',
        '--head-safety-report "$HEAD_SAFETY_REPORT"',
        '--base-safety-report "$BASE_SAFETY_REPORT"',
        '--package-repo-path admin-ui',
        '--changed-files "$CHANGED_FILES_PATH"',
    )
    missing = [contract for contract in required_contracts if contract not in run_script]
    assert not missing, f"admin UI Vitest ratchet contracts missing: {missing}"
    assert "bun run test" not in run_script
    assert "HEAD^" not in run_script

    hash_contracts = dict(
        re.findall(
            r'^(EXPECTED_(?:RATCHET|SAFETY_REPORTER)_SHA256)="([0-9a-f]{64})"$',
            run_script,
            flags=re.MULTILINE,
        )
    )
    artifact_paths = {
        "EXPECTED_RATCHET_SHA256": Path(
            "Helper_Scripts/ci/vitest_base_ratchet.py"
        ),
        "EXPECTED_SAFETY_REPORTER_SHA256": Path(
            "admin-ui/scripts/ci/vitest-safety-reporter.mjs"
        ),
    }
    assert hash_contracts.keys() == artifact_paths.keys()
    for contract_name, artifact_path in artifact_paths.items():
        actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert hash_contracts[contract_name] == actual_hash

    unit_index = steps.index(unit_step)
    assert unit_index < steps.index(named_steps["Run admin-ui real-backend e2e"])
    assert unit_index < steps.index(named_steps["Run admin-ui build"])


@pytest.mark.unit
def test_admin_ui_real_backend_projects_run_in_separate_next_processes() -> None:
    """Prevent real-backend projects from sharing one concurrent Next workspace."""

    playwright_config = Path("admin-ui/playwright.config.ts").read_text(
        encoding="utf-8"
    )
    next_config = Path("admin-ui/next.config.mjs").read_text(encoding="utf-8")
    global_setup = Path(
        "admin-ui/tests/e2e/real-backend/helpers/global-setup.ts"
    ).read_text(encoding="utf-8")
    global_teardown = Path(
        "admin-ui/tests/e2e/real-backend/helpers/global-teardown.ts"
    ).read_text(encoding="utf-8")
    package = json.loads(Path("admin-ui/package.json").read_text(encoding="utf-8"))
    workflow = yaml.safe_load(
        Path(".github/workflows/frontend-required.yml").read_text(encoding="utf-8")
    )
    workflow_steps = workflow["jobs"]["frontend-required"]["steps"]
    real_backend_step = next(
        step
        for step in workflow_steps
        if step.get("name") == "Run admin-ui real-backend e2e"
    )

    assert "shouldAutoStart && !shouldStartRealBackendUiServers" in playwright_config
    assert "getRequestedRealBackendProjects(process.argv)" in playwright_config
    assert "requestedRealBackendProjects.length !== 1" in playwright_config
    assert "requestedRealBackendProject === 'chromium-real-jwt'" in playwright_config
    assert (
        "requestedRealBackendProject === 'chromium-real-single-user'"
        in playwright_config
    )
    assert "bunx next start -p ${realJwtProject.uiPort}" in playwright_config
    assert "bunx next start -p ${realSingleUserProject.uiPort}" in playwright_config
    assert "NEXT_DIST_DIR" not in playwright_config
    assert "const isRealBackendE2eBuild =" in next_config
    assert "process.env.TLDW_ADMIN_E2E_REAL_BACKEND === 'true'" in next_config
    assert "output: isRealBackendE2eBuild ? undefined : 'standalone'" in next_config
    assert "distDir:" not in next_config

    assert "getRequestedRealBackendProjects(process.argv)" in global_setup
    assert "getRequestedRealBackendProjects(process.argv)" in global_teardown
    assert "for (const projectName of requestedRealBackendProjects)" in global_setup
    assert "for (const projectName of requestedRealBackendProjects)" in global_teardown

    scripts = package["scripts"]
    assert scripts["test:real-backend"] == (
        "bun run build:real-backend && "
        "bun run test:real-backend:jwt:run && "
        "bun run test:real-backend:single-user:run"
    )
    assert scripts["test:real-backend:jwt"] == (
        "bun run build:real-backend && bun run test:real-backend:jwt:run"
    )
    assert scripts["test:real-backend:single-user"] == (
        "bun run build:real-backend && "
        "bun run test:real-backend:single-user:run"
    )
    assert "TLDW_ADMIN_E2E_REAL_BACKEND=true" in scripts["build:real-backend"]
    assert "next build" in scripts["build:real-backend"]
    assert "--project=chromium-real-jwt" in scripts["test:real-backend:jwt:run"]
    assert (
        "--project=chromium-real-single-user"
        in scripts["test:real-backend:single-user:run"]
    )
    assert real_backend_step["run"] == "bun run test:real-backend"


@pytest.mark.unit
def test_package_ratchet_uses_required_base_and_trusted_helper() -> None:
    """Keep the shared helper immutable in the package-owned ratchet too."""

    workflow = yaml.safe_load(
        Path(".github/workflows/frontend-required.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["frontend-unit-tests"]["steps"]
    unit_step = next(
        step for step in steps if step.get("name") == "Run package-owned frontend unit tests"
    )
    run_script = str(unit_step["run"])

    assert 'BASE_SHA="${{ inputs.base_sha }}"' in run_script
    assert "HEAD^" not in run_script
    assert '[[ ! "$BASE_SHA" =~ ^[0-9a-fA-F]{40}$' in run_script
    assert 'EXPECTED_RATCHET_SHA256="' in run_script
    assert "sha256sum --check --status" in run_script

    expected_hash = re.search(
        r'^EXPECTED_RATCHET_SHA256="([0-9a-f]{64})"$',
        run_script,
        flags=re.MULTILINE,
    )
    assert expected_hash is not None
    actual_hash = hashlib.sha256(
        Path("Helper_Scripts/ci/vitest_base_ratchet.py").read_bytes()
    ).hexdigest()
    assert expected_hash.group(1) == actual_hash


@pytest.mark.unit
def test_canonical_admin_webhooks_use_a_dedicated_full_suite_shard() -> None:
    """Keep canonical webhook tests owned without widening the legacy shard."""

    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    matrix_jobs = (
        "full-suite-linux-312-shards",
        "full-suite-linux-313-shards",
        "full-suite-macos-312-shards",
        "full-suite-windows-312-shards",
        "full-suite-os-313-release-shards",
    )

    for job_name in matrix_jobs:
        shards = workflow["jobs"][job_name]["strategy"]["matrix"]["shard"]
        paths_by_name = {
            shard["name"]: set(str(shard["paths"]).split()) for shard in shards
        }
        assert paths_by_name["admin-webhooks-canonical"] == {
            "tldw_Server_API/tests/Admin_Webhooks"
        }
        assert (
            "tldw_Server_API/tests/Admin_Webhooks"
            not in paths_by_name["admin-watchlists-webhooks"]
        )

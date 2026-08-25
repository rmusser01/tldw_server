"""Contracts for the admin UI exact-base Vitest failure ratchet."""

from pathlib import Path

import pytest
import yaml


@pytest.mark.unit
def test_admin_ui_unit_gate_uses_fail_closed_exact_base_ratchet() -> None:
    """Keep inherited failures bounded without weakening later admin UI gates."""

    workflow = yaml.safe_load(
        Path(".github/workflows/frontend-required.yml").read_text(encoding="utf-8")
    )
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
        'git diff --name-only --diff-filter=ACMR "$BASE_SHA" "$HEAD_SHA"',
        'git worktree add --detach "$BASE_WORKTREE" "$BASE_SHA"',
        "bun install --frozen-lockfile",
        'RATCHET_SCRIPT="${GITHUB_WORKSPACE}/Helper_Scripts/ci/vitest_base_ratchet.py"',
        '"--reporter=default"',
        '"--reporter=json"',
        '"--outputFile.json=${HEAD_REPORT}"',
        "if (( HEAD_STATUS != 1 )); then",
        'python3 "$RATCHET_SCRIPT" validate-success',
        'python3 "$RATCHET_SCRIPT" extract',
        'FAILED_FILES+=("./${test_file}")',
        'bunx vitest run "${FAILED_FILES[@]}"',
        "if (( BASE_STATUS != 1 )); then",
        'python3 "$RATCHET_SCRIPT" compare',
        '--package-repo-path admin-ui',
        '--changed-files "$CHANGED_FILES_PATH"',
    )
    missing = [contract for contract in required_contracts if contract not in run_script]
    assert not missing, f"admin UI Vitest ratchet contracts missing: {missing}"
    assert "bun run test" not in run_script

    unit_index = steps.index(unit_step)
    assert unit_index < steps.index(named_steps["Run admin-ui real-backend e2e"])
    assert unit_index < steps.index(named_steps["Run admin-ui build"])


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

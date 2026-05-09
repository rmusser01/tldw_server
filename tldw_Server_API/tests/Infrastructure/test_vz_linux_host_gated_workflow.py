"""Contract tests for the Apple Silicon host-gated vz_linux workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "vz-linux-host-gated.yml"
POLICY_PATH = REPO_ROOT / "Docs" / "Sandbox" / "vz-linux-host-gated-ci-acceptance-policy.md"
SMOKE_SCRIPT_PATH = REPO_ROOT / "tools" / "vz-linux-image" / "scripts" / "run-host-e2e-smoke.sh"


def _load_workflow() -> dict[str, Any]:
    """Load the workflow YAML while preserving the parsed structure for assertions."""
    with WORKFLOW_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _workflow_triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the workflow trigger map despite PyYAML's YAML 1.1 boolean key parsing."""
    # PyYAML's YAML 1.1 resolver parses an unquoted "on" key as True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)  # nosec B101
    return triggers


def test_vz_linux_host_gated_workflow_is_manual_and_nightly() -> None:
    """The real VZ workflow should remain manual/nightly instead of normal CI."""
    workflow = _load_workflow()
    triggers = _workflow_triggers(workflow)

    assert set(triggers) == {"workflow_dispatch", "schedule"}  # nosec B101
    assert "workflow_dispatch" in triggers  # nosec B101
    assert "schedule" in triggers  # nosec B101
    workflow_dispatch = triggers["workflow_dispatch"]
    assert isinstance(workflow_dispatch, dict)  # nosec B101
    inputs = workflow_dispatch.get("inputs")
    assert isinstance(inputs, dict)  # nosec B101
    assert inputs["bundle_path"]["required"] is False  # nosec B101
    assert inputs["include_failure_drills"]["required"] is False  # nosec B101
    assert inputs["include_failure_drills"]["default"] is False  # nosec B101
    assert inputs["include_failure_drills"]["type"] == "boolean"  # nosec B101


def test_vz_linux_host_gated_workflow_targets_prepared_apple_silicon_runner() -> None:
    """The job must target only prepared self-hosted Apple Silicon VZ runners."""
    workflow = _load_workflow()
    job = workflow["jobs"]["vz-linux-host-gated-smoke"]

    assert job["runs-on"] == ["self-hosted", "macOS", "ARM64", "vz-linux"]  # nosec B101
    assert job["timeout-minutes"] <= 120  # nosec B101
    assert "TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH" in job["env"]  # nosec B101


def test_vz_linux_host_gated_workflow_preserves_explicit_skip_sign_false() -> None:
    """Manual skip_sign=false should override a true repository variable."""
    workflow = _load_workflow()
    job = workflow["jobs"]["vz-linux-host-gated-smoke"]
    skip_sign_expr = job["env"]["TLDW_SANDBOX_VZ_HELPER_SKIP_SIGN"]
    truthy_fallback = "inputs.skip_sign || vars.TLDW_SANDBOX_VZ_HELPER_SKIP_SIGN"

    assert "inputs.skip_sign == null" in skip_sign_expr  # nosec B101
    assert "vars.TLDW_SANDBOX_VZ_HELPER_SKIP_SIGN || 'false'" in skip_sign_expr  # nosec B101
    assert truthy_fallback not in skip_sign_expr  # nosec B101


def test_vz_linux_host_gated_workflow_rejects_untrusted_refs() -> None:
    """Manual self-hosted runs must not execute arbitrary feature-branch code."""
    workflow = _load_workflow()
    job = workflow["jobs"]["vz-linux-host-gated-smoke"]

    assert "refs/heads/main" in job["if"]  # nosec B101
    assert "refs/heads/dev" in job["if"]  # nosec B101
    assert "github.ref" in job["if"]  # nosec B101


def test_vz_linux_host_gated_workflow_schedule_requires_repo_opt_in() -> None:
    """Nightly host-gated runs should require an explicit repository variable."""
    workflow = _load_workflow()
    job = workflow["jobs"]["vz-linux-host-gated-smoke"]

    assert "github.event_name == 'workflow_dispatch'" in job["if"]  # nosec B101
    assert "vars.TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY == '1'" in job["if"]  # nosec B101


def test_vz_linux_host_gated_workflow_uses_operator_smoke_script() -> None:
    """The workflow should delegate real VM work to the repo's operator smoke script."""
    workflow = _load_workflow()
    steps = workflow["jobs"]["vz-linux-host-gated-smoke"]["steps"]
    run_blocks = "\n".join(str(step.get("run", "")) for step in steps)

    assert "tools/vz-linux-image/scripts/run-host-e2e-smoke.sh" in run_blocks  # nosec B101
    assert "--bundle" in run_blocks  # nosec B101
    assert "--python" in run_blocks  # nosec B101
    assert 'chmod 700 "${runtime_dir}/serial"' in run_blocks  # nosec B101
    assert "TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH" in run_blocks  # nosec B101


def test_vz_linux_host_gated_workflow_failure_drills_are_manual_opt_in() -> None:
    """Failure drills should only run when manual dispatch opts in."""
    workflow = _load_workflow()
    job = workflow["jobs"]["vz-linux-host-gated-smoke"]
    steps = job["steps"]
    run_blocks = "\n".join(str(step.get("run", "")) for step in steps)

    assert "include_failure_drills" in _workflow_triggers(workflow)["workflow_dispatch"]["inputs"]  # nosec B101
    assert "TLDW_SANDBOX_VZ_INCLUDE_FAILURE_DRILLS" not in job["env"]  # nosec B101
    assert "inputs.include_failure_drills" in run_blocks  # nosec B101
    assert "--include-failure-drills" in run_blocks  # nosec B101
    assert "vars.TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY" not in run_blocks  # nosec B101


def test_vz_linux_host_gated_operator_smoke_runs_recovery_slice() -> None:
    """The delegated operator smoke should include the non-destructive recovery slice."""
    script = SMOKE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "run_real_vz_linux_pytest" in script  # nosec B101
    assert "run_real_vz_linux_host_smoke" in script  # nosec B101
    assert "-m vz_linux_host_smoke" in script  # nosec B101


def test_vz_linux_host_gated_operator_smoke_has_manual_failure_drill_slice() -> None:
    """The operator smoke should expose failure drills without enabling them by default."""
    script = SMOKE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "--include-failure-drills" in script  # nosec B101
    assert "run_real_vz_linux_failure_drills" in script  # nosec B101
    assert "-m vz_linux_host_failure_drill" in script  # nosec B101


def test_vz_linux_host_gated_acceptance_policy_requires_recovery_smoke() -> None:
    """The host-gated policy should cover diagnostics plus dry-run repair."""
    policy = POLICY_PATH.read_text(encoding="utf-8").lower()

    assert "recovery diagnostics" in policy  # nosec B101
    assert "dry-run reconciliation repair" in policy  # nosec B101
    assert "does not terminate vms" in policy  # nosec B101


def test_vz_linux_host_gated_workflow_pins_external_actions() -> None:
    """The self-hosted workflow should use immutable SHAs for third-party actions."""
    workflow = _load_workflow()
    steps = workflow["jobs"]["vz-linux-host-gated-smoke"]["steps"]
    uses_entries = [str(step.get("uses", "")) for step in steps if step.get("uses")]

    assert "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd" in uses_entries  # nosec B101
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in uses_entries  # nosec B101


def test_vz_linux_host_gated_workflow_uses_minimal_permissions_and_uploads_logs() -> None:
    """Self-hosted VZ runs should use minimal repo permissions and preserve helper logs."""
    workflow = _load_workflow()
    steps = workflow["jobs"]["vz-linux-host-gated-smoke"]["steps"]
    upload_steps = [
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]

    assert workflow["permissions"] == {"contents": "read"}  # nosec B101
    assert len(upload_steps) == 1  # nosec B101
    assert upload_steps[0]["if"] == "always()"  # nosec B101
    assert upload_steps[0]["with"]["if-no-files-found"] == "ignore"  # nosec B101
    assert "${{ runner.temp }}/tldw-vz-helper-ci/**" in upload_steps[0]["with"]["path"]  # nosec B101


def test_vz_linux_host_gated_acceptance_policy_doc_exists_and_references_workflow() -> None:
    """The host-gated workflow should have an operator-facing acceptance policy."""
    policy = POLICY_PATH.read_text(encoding="utf-8")

    assert ".github/workflows/vz-linux-host-gated.yml" in policy  # nosec B101
    assert "TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY" in policy  # nosec B101
    assert "blocking regression" in policy.lower()  # nosec B101

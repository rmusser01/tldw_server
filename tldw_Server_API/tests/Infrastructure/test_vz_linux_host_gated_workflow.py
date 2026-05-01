"""Contract tests for the Apple Silicon host-gated vz_linux workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "vz-linux-host-gated.yml"


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

    assert "workflow_dispatch" in triggers  # nosec B101
    assert "schedule" in triggers  # nosec B101
    assert triggers["workflow_dispatch"]["inputs"]["bundle_path"]["required"] is False  # nosec B101


def test_vz_linux_host_gated_workflow_targets_prepared_apple_silicon_runner() -> None:
    """The job must target only prepared self-hosted Apple Silicon VZ runners."""
    workflow = _load_workflow()
    job = workflow["jobs"]["vz-linux-host-gated-smoke"]

    assert job["runs-on"] == ["self-hosted", "macOS", "ARM64", "vz-linux"]  # nosec B101
    assert job["timeout-minutes"] <= 120  # nosec B101
    assert "TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH" in job["env"]  # nosec B101


def test_vz_linux_host_gated_workflow_rejects_untrusted_refs() -> None:
    """Manual self-hosted runs must not execute arbitrary feature-branch code."""
    workflow = _load_workflow()
    job = workflow["jobs"]["vz-linux-host-gated-smoke"]

    assert "refs/heads/main" in job["if"]  # nosec B101
    assert "refs/heads/dev" in job["if"]  # nosec B101
    assert "github.ref" in job["if"]  # nosec B101


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


def test_vz_linux_host_gated_workflow_pins_external_actions() -> None:
    """The self-hosted workflow should use immutable SHAs for third-party actions."""
    workflow = _load_workflow()
    steps = workflow["jobs"]["vz-linux-host-gated-smoke"]["steps"]
    uses_entries = [str(step.get("uses", "")) for step in steps if step.get("uses")]

    assert "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd" in uses_entries  # nosec B101
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in uses_entries  # nosec B101

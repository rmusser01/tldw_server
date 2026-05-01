from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "vz-linux-host-gated.yml"


def _load_workflow() -> dict:
    with WORKFLOW_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _workflow_triggers(workflow: dict) -> dict:
    # PyYAML's YAML 1.1 resolver parses an unquoted "on" key as True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)  # nosec B101
    return triggers


def test_vz_linux_host_gated_workflow_is_manual_and_nightly() -> None:
    workflow = _load_workflow()
    triggers = _workflow_triggers(workflow)

    assert "workflow_dispatch" in triggers  # nosec B101
    assert "schedule" in triggers  # nosec B101
    assert triggers["workflow_dispatch"]["inputs"]["bundle_path"]["required"] is False  # nosec B101


def test_vz_linux_host_gated_workflow_targets_prepared_apple_silicon_runner() -> None:
    workflow = _load_workflow()
    job = workflow["jobs"]["vz-linux-host-gated-smoke"]

    assert job["runs-on"] == ["self-hosted", "macOS", "ARM64", "vz-linux"]  # nosec B101
    assert job["timeout-minutes"] <= 120  # nosec B101
    assert "TLDW_VZ_HOST_E2E_BUNDLE_PATH" in job["env"]  # nosec B101


def test_vz_linux_host_gated_workflow_uses_operator_smoke_script() -> None:
    workflow = _load_workflow()
    steps = workflow["jobs"]["vz-linux-host-gated-smoke"]["steps"]
    run_blocks = "\n".join(str(step.get("run", "")) for step in steps)

    assert "tools/vz-linux-image/scripts/run-host-e2e-smoke.sh" in run_blocks  # nosec B101
    assert "--bundle" in run_blocks  # nosec B101
    assert "--python" in run_blocks  # nosec B101
    assert "TLDW_VZ_HOST_E2E_BUNDLE_PATH" in run_blocks  # nosec B101

from pathlib import Path

import pytest
import yaml


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _workflow_on(workflow: dict) -> dict:
    return workflow[True]


def _install_step_run(workflow: dict, job_name: str = "build-and-check") -> str:
    steps = workflow["jobs"][job_name]["steps"]
    install_steps = [step for step in steps if step.get("name") == "Install packaging tools"]
    assert install_steps, "Install packaging tools step missing"
    return install_steps[0]["run"]


def _detect_version_step_run(workflow: dict) -> str:
    steps = workflow["jobs"]["detect-version"]["steps"]
    detect_steps = [step for step in steps if step.get("id") == "detect"]
    assert detect_steps, "Detect version step missing"
    return detect_steps[0]["run"]


def _job_run_scripts(workflow: dict, job_name: str) -> list[str]:
    """Return the shell scripts configured for runnable workflow steps.

    Args:
        workflow: Parsed GitHub Actions workflow document.
        job_name: Workflow job key to inspect.

    Returns:
        The ordered ``run`` script values for steps in the selected job.
    """
    return [step["run"] for step in workflow["jobs"][job_name]["steps"] if "run" in step]


def test_pypi_package_workflow_installs_setuptools_backend() -> None:
    workflow = _load(".github/workflows/pypi-package.yml")
    run_script = _install_step_run(workflow)
    assert "setuptools" in run_script
    assert "wheel" in run_script


def test_publish_pypi_workflow_installs_setuptools_backend() -> None:
    workflow = _load(".github/workflows/publish-pypi.yml")
    run_script = _install_step_run(workflow, job_name="build")
    assert "setuptools" in run_script
    assert "wheel" in run_script


def test_publish_pypi_workflow_preserves_manual_dispatch_and_gates_push() -> None:
    workflow = _load(".github/workflows/publish-pypi.yml")
    on = _workflow_on(workflow)
    target = on["workflow_dispatch"]["inputs"]["target"]
    push = on["push"]

    assert set(on) == {"workflow_dispatch", "push"}
    assert "release" not in on
    assert push["branches"] == ["main"]
    assert push["paths"] == ["pyproject.toml", ".github/workflows/publish-pypi.yml"]
    assert target["options"] == ["testpypi", "pypi"]
    assert target["default"] == "testpypi"

    detect_version = workflow["jobs"]["detect-version"]
    assert detect_version["outputs"]["should_publish"] == "${{ steps.detect.outputs.should_publish }}"

    release_gate = workflow["jobs"]["release-gate"]
    assert release_gate["if"] == (  # nosec B101
        "${{ github.event_name == 'workflow_dispatch' || needs.detect-version.outputs.should_publish == 'true' }}"
    )

    build = workflow["jobs"]["build"]
    assert build["needs"] == ["detect-version", "release-gate"]  # nosec B101

    publish_testpypi = workflow["jobs"]["publish-testpypi"]
    assert publish_testpypi["if"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi' }}"
    )

    publish_pypi = workflow["jobs"]["publish-pypi"]
    assert publish_pypi["if"] == (
        "${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'pypi') || "
        "(github.event_name == 'push' && needs.detect-version.outputs.should_publish == 'true') }}"
    )


def test_publish_pypi_detect_version_handles_decode_and_timeout_failures() -> None:
    workflow = _load(".github/workflows/publish-pypi.yml")
    run_script = _detect_version_step_run(workflow)

    assert "json.JSONDecodeError" in run_script
    assert "TimeoutError" in run_script


@pytest.mark.unit
def test_publish_pypi_workflow_uses_targeted_release_gate_not_full_pytest_suite() -> None:
    """Verify the PyPI release gate is bounded but still tests app startup.

    Returns:
        None. The pytest assertions encode the workflow release-gate contract.
    """
    workflow = _load(".github/workflows/publish-pypi.yml")
    run_scripts = "\n".join(
        script for job_name in workflow["jobs"] for script in _job_run_scripts(workflow, job_name)
    )

    assert all(  # nosec B101
        line.strip() != "python -m pytest -q" for line in run_scripts.splitlines()
    )

    release_gate_runs = "\n".join(_job_run_scripts(workflow, "release-gate"))
    assert "tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py" in release_gate_runs  # nosec B101
    assert "Helper_Scripts/ci/minimal_env_smoke.py --timeout 150" in release_gate_runs  # nosec B101

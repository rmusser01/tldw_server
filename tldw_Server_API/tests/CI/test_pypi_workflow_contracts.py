from pathlib import Path

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
    assert push["paths"] == ["pyproject.toml"]
    assert target["options"] == ["testpypi", "pypi"]
    assert target["default"] == "testpypi"

    detect_version = workflow["jobs"]["detect-version"]
    assert detect_version["outputs"]["should_publish"] == "${{ steps.detect.outputs.should_publish }}"

    test_suite = workflow["jobs"]["test-suite"]
    assert test_suite["if"] == (
        "${{ github.event_name == 'workflow_dispatch' || needs.detect-version.outputs.should_publish == 'true' }}"
    )

    build = workflow["jobs"]["build"]
    assert build["needs"] == ["detect-version", "test-suite"]

    publish_testpypi = workflow["jobs"]["publish-testpypi"]
    assert publish_testpypi["if"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi' }}"
    )

    publish_pypi = workflow["jobs"]["publish-pypi"]
    assert publish_pypi["if"] == (
        "${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'pypi') || "
        "(github.event_name == 'push' && needs.detect-version.outputs.should_publish == 'true') }}"
    )

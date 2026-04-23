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


def test_publish_pypi_workflow_is_manual_dispatch_only() -> None:
    workflow = _load(".github/workflows/publish-pypi.yml")
    on = _workflow_on(workflow)
    target = on["workflow_dispatch"]["inputs"]["target"]

    assert set(on) == {"workflow_dispatch"}
    assert "release" not in on
    assert target["options"] == ["testpypi", "pypi"]
    assert target["default"] == "testpypi"

    publish_testpypi = workflow["jobs"]["publish-testpypi"]
    assert publish_testpypi["if"] == (
        "${{ github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi' }}"
    )

    publish_pypi = workflow["jobs"]["publish-pypi"]
    assert publish_pypi["if"] == "${{ github.event_name == 'workflow_dispatch' && inputs.target == 'pypi' }}"

from pathlib import Path

import pytest
import yaml


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _workflow_on(workflow: dict) -> dict:
    return workflow[True]


def _get_step(steps: list[dict], name: str) -> dict:
    """Return a named workflow step or fail the contract test clearly."""
    matching = [step for step in steps if step.get("name") == name]
    assert matching, f"{name} step missing"
    return matching[0]


def test_publish_docker_workflow_is_release_driven() -> None:
    workflow = _load(".github/workflows/publish-docker.yml")
    on = _workflow_on(workflow)

    assert "release" in on
    assert on["release"] == {"types": ["published"]}
    assert "workflow_dispatch" in on


def test_publish_docker_matrix_remains_app_worker_audio_worker() -> None:
    workflow = _load(".github/workflows/publish-docker.yml")
    matrix = workflow["jobs"]["push_to_registries"]["strategy"]["matrix"]["include"]

    assert [entry["name"] for entry in matrix] == ["app", "worker", "audio-worker"]


@pytest.mark.unit
def test_publish_docker_release_workflow_targets_ghcr_only() -> None:
    """Docker release publishing must target GHCR and avoid Docker Hub credentials."""
    workflow = _load(".github/workflows/publish-docker.yml")
    job = workflow["jobs"]["push_to_registries"]
    steps = job["steps"]
    meta_step = _get_step(steps, "Extract metadata (tags, labels) for GHCR")

    assert "DOCKERHUB_IMAGE" not in workflow.get("env", {})
    assert not any(step.get("name") == "Log in to Docker Hub" for step in steps)
    assert "DOCKERHUB" not in meta_step["with"]["images"]
    assert "matrix.dockerhub_suffix" not in meta_step["with"]["images"]
    assert "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}${{ matrix.ghcr_suffix }}" in meta_step["with"]["images"]
    assert all("dockerhub_suffix" not in entry for entry in job["strategy"]["matrix"]["include"])


@pytest.mark.unit
def test_publish_pypi_workflow_installs_portaudio_before_dev_dependencies() -> None:
    """PyPI release tests must install PortAudio before installing dev extras."""
    workflow = _load(".github/workflows/publish-pypi.yml")
    on = _workflow_on(workflow)
    steps = workflow["jobs"]["test-suite"]["steps"]
    install_step = _get_step(steps, "Install FFmpeg and PortAudio (Linux)")
    setup_step = _get_step(steps, "Setup Python")
    deps_step = _get_step(steps, "Install test dependencies")

    assert ".github/workflows/publish-pypi.yml" in on["push"]["paths"]
    assert install_step["uses"] == "./.github/actions/setup-ffmpeg"
    assert install_step["with"]["install-ffmpeg"] == "false"
    assert install_step["with"]["install-portaudio"] == "true"
    assert steps.index(install_step) < steps.index(setup_step) < steps.index(deps_step)
    assert 'python -m pip install -e ".[dev]"' in deps_step["run"]


def test_publish_ghcr_main_workflow_remains_push_to_main_driven() -> None:
    workflow = _load(".github/workflows/publish-ghcr-main.yml")
    on = _workflow_on(workflow)

    assert "push" in on
    assert on["push"] == {"branches": ["main"]}
    assert "release" not in on
    assert "workflow_dispatch" not in on


def test_publish_ghcr_main_matrix_remains_app_webui_admin_ui() -> None:
    workflow = _load(".github/workflows/publish-ghcr-main.yml")
    matrix = workflow["jobs"]["publish-ghcr-main"]["strategy"]["matrix"]["include"]

    assert [entry["name"] for entry in matrix] == ["app", "webui", "admin-ui"]

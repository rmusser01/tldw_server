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
def test_publish_pypi_workflow_runs_targeted_release_contract_gate() -> None:
    """PyPI release gating must avoid running the full backend test suite.

    Returns:
        None. The pytest assertions encode the release workflow contract.
    """
    workflow = _load(".github/workflows/publish-pypi.yml")
    on = _workflow_on(workflow)
    steps = workflow["jobs"]["release-gate"]["steps"]
    install_step = _get_step(steps, "Install FFmpeg and PortAudio (Linux)")
    setup_step = _get_step(steps, "Setup Python")
    deps_step = _get_step(steps, "Install test dependencies")
    tests_step = _get_step(steps, "Run release contract tests")
    smoke_step = _get_step(steps, "Run minimal startup smoke")

    assert ".github/workflows/publish-pypi.yml" in on["push"]["paths"]  # nosec B101
    assert install_step["uses"] == "./.github/actions/setup-ffmpeg"  # nosec B101
    assert install_step["with"]["install-ffmpeg"] == "false"  # nosec B101
    assert install_step["with"]["install-portaudio"] == "true"  # nosec B101
    assert steps.index(install_step) < steps.index(setup_step) < steps.index(deps_step)  # nosec B101
    assert 'python -m pip install -e ".[dev]"' in deps_step["run"]  # nosec B101
    assert "tldw_Server_API/tests/CI/test_pypi_workflow_contracts.py" in tests_step["run"]  # nosec B101
    assert all(  # nosec B101
        line.strip() != "python -m pytest -q" for line in tests_step["run"].splitlines()
    )
    assert smoke_step["run"] == "python Helper_Scripts/ci/minimal_env_smoke.py --timeout 150"  # nosec B101


def test_publish_ghcr_main_workflow_remains_push_to_main_driven() -> None:
    workflow = _load(".github/workflows/publish-ghcr-main.yml")
    on = _workflow_on(workflow)

    assert "push" in on
    assert on["push"] == {"branches": ["main"]}
    assert "release" not in on
    assert "workflow_dispatch" not in on


def test_publish_ghcr_main_matrix_is_backend_only_during_frontend_freeze() -> None:
    workflow = _load(".github/workflows/publish-ghcr-main.yml")
    matrix = workflow["jobs"]["publish-ghcr-main"]["strategy"]["matrix"]["include"]

    assert matrix == [
        {
            "name": "app",
            "dockerfile": "Dockerfiles/Dockerfile.prod",
            "image_suffix": "",
            "build_args": "",
        }
    ]


def test_publish_ghcr_main_preserves_backend_publish_controls() -> None:
    workflow = _load(".github/workflows/publish-ghcr-main.yml")
    steps = workflow["jobs"]["publish-ghcr-main"]["steps"]
    metadata = _get_step(steps, "Extract metadata (tags, labels) for GHCR")
    publish = _get_step(steps, "Build and push Docker images")
    attestation = _get_step(steps, "Generate artifact attestation (GHCR)")

    assert metadata["with"]["tags"].splitlines() == [
        "type=raw,value=main",
        "type=sha,format=short",
    ]
    assert publish["with"] == {
        "context": ".",
        "file": "${{ matrix.dockerfile }}",
        "push": True,
        "build-args": "${{ matrix.build_args }}",
        "tags": "${{ steps.meta.outputs.tags }}",
        "labels": "${{ steps.meta.outputs.labels }}",
        "cache-from": "type=gha",
        "cache-to": "type=gha,mode=max",
    }
    assert attestation["with"] == {
        "subject-name": "${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}${{ matrix.image_suffix }}",
        "subject-digest": "${{ steps.push.outputs.digest }}",
        "push-to-registry": True,
    }


def test_container_build_check_covers_workers_without_publishing_images() -> None:
    workflow = _load(".github/workflows/container-build-check.yml")
    job = workflow["jobs"]["build"]
    matrix = job["strategy"]["matrix"]["include"]
    build = _get_step(job["steps"], "Build container images")

    assert [entry["name"] for entry in matrix] == ["app", "worker", "audio-worker", "webui", "admin-ui"]
    assert [entry["dockerfile"] for entry in matrix] == [
        "Dockerfiles/Dockerfile.prod",
        "Dockerfiles/Dockerfile.worker",
        "Dockerfiles/Dockerfile.audio_gpu_worker",
        "Dockerfiles/Dockerfile.webui",
        "Dockerfiles/Dockerfile.admin-ui",
    ]
    assert build["with"]["push"] is False
    assert workflow["permissions"] == {"contents": "read"}
    assert [entry["backend"] for entry in matrix] == [True, True, True, False, False]
    assert build["with"]["load"] == "${{ matrix.backend }}"


def test_container_backend_smoke_uses_the_built_image_and_isolated_imports() -> None:
    """Backend packaging omissions must fail before the matrix result is green."""
    workflow = _load(".github/workflows/container-build-check.yml")
    steps = workflow["jobs"]["build"]["steps"]
    build = _get_step(steps, "Build container images")
    smoke = _get_step(steps, "Verify backend local package imports")

    assert steps.index(smoke) > steps.index(build)
    assert smoke["if"] == "matrix.backend"
    assert smoke["env"]["IMAGE_REF"] == build["with"]["tags"]
    assert "docker image inspect --format" in smoke["run"]
    assert '"$image_id" -I -c' in smoke["run"]
    assert "--entrypoint python" in smoke["run"]
    assert "import mcp_unified; import tldw_profile_core" in smoke["run"]
    assert "--network none" in smoke["run"]
    assert "--read-only" in smoke["run"]
    assert not smoke.get("continue-on-error", False)


def test_frontend_required_enforces_shared_hooks_and_preserves_full_lint() -> None:
    """Shared UI must reach the hook gate even though frontend lint runs locally."""
    workflow = _load(".github/workflows/frontend-required.yml")
    steps = workflow["jobs"]["frontend-required"]["steps"]
    lint = _get_step(steps, "Run frontend lint")
    hooks = _get_step(steps, "Run shared UI hook enforcement")

    assert lint["run"] == "bun run lint"
    assert lint["working-directory"] == "apps/tldw-frontend"
    assert hooks["if"] == lint["if"]
    assert hooks["working-directory"] == "apps/tldw-frontend"
    assert hooks["run"] == "bun scripts/check-shared-hooks.mjs"
    assert not hooks.get("continue-on-error", False)

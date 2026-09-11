import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
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


def test_backend_scan_is_pinned_offline_and_does_not_publish_or_filter_findings() -> None:
    """A scanner downgrade, network fallback or partial upload must fail the contract."""
    job = _load(".github/workflows/container-build-check.yml")["jobs"]["build"]
    steps = job["steps"]
    prepare = _get_step(steps, "Prepare backend scanner evidence")
    scan = _get_step(steps, "Collect backend SBOM and vulnerabilities")
    finalize = _get_step(steps, "Verify and hash backend evidence")
    upload = _get_step(steps, "Upload backend evidence")
    assert steps.index(prepare) > steps.index(_get_step(steps, "Verify backend local package imports"))
    assert job["env"]["TRIVY_IMAGE"] == (
        "ghcr.io/aquasecurity/trivy:0.74.0@sha256:" "62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969"
    )
    assert prepare["if"] == scan["if"] == "matrix.backend"
    assert "docker.sock" not in prepare["run"]
    assert "--download-db-only" in prepare["run"]
    for required in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges",
        "--image-src docker",
        "--skip-db-update",
        "--scanners vuln",
        "--list-all-pkgs",
        "--ignore-unfixed=false",
        '"$IMAGE_ID"',
    ):
        assert required in scan["run"]
    assert scan["env"]["IMAGE_ID"] == "${{ steps.backend_image.outputs.image_id }}"
    assert "--severity" not in scan["run"]
    assert "--input" not in scan["run"]
    assert "docker save" not in scan["run"]
    assert 1 <= scan["timeout-minutes"] <= 20
    for step in (prepare, scan, finalize, upload):
        assert not step.get("continue-on-error", False)
    assert finalize["if"] == upload["if"] == "${{ always() && matrix.backend }}"
    assert upload["uses"] == "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
    assert upload["with"]["if-no-files-found"] == "error"


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "stale",
        "future",
        "wrong_image",
        "empty_packages",
        "missing_sbom",
        "wrong_scanner",
        "invalid_source",
        "empty_sbom",
        "naive_time",
    ],
)
def test_backend_evidence_verifier_rejects_invalid_artifacts(tmp_path: Path, defect: str | None) -> None:
    """Execute the actual workflow verifier against independent artifact fixtures."""
    steps = _load(".github/workflows/container-build-check.yml")["jobs"]["build"]["steps"]
    step = _get_step(steps, "Verify and hash backend evidence")
    script = step["run"].split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    image_id = "sha256:" + "a" * 64
    now = datetime.now(timezone.utc)
    age = timedelta(hours=25) if defect == "stale" else timedelta(minutes=-10 if defect == "future" else 1)
    identity = {
        "image_id": image_id,
        "source_sha": "bad" if defect == "invalid_source" else "b" * 40,
        "os": "linux",
        "architecture": "amd64",
    }
    scanner_pin = "scanner@sha256:" + "c" * 64
    timestamp = (now - age).replace(tzinfo=None) if defect == "naive_time" else now - age
    scanner = {
        "image": "wrong" if defect == "wrong_scanner" else scanner_pin,
        "databases": {name: {"UpdatedAt": timestamp.isoformat(), "sha256": "d" * 64} for name in ("db", "java-db")},
    }
    report = {
        "Metadata": {"ImageID": "wrong" if defect == "wrong_image" else image_id},
        "Results": [
            {
                "Packages": [] if defect == "empty_packages" else [{"Name": "example", "Version": "1"}],
                "Vulnerabilities": [{"VulnerabilityID": "CVE-example", "Severity": "CRITICAL"}],
            }
        ],
    }
    files = {
        "image.json": identity,
        "scanner.json": scanner,
        "vulnerabilities.json": report,
        "sbom.cdx.json": {
            "bomFormat": "CycloneDX",
            "components": [] if defect == "empty_sbom" else [{"name": "example", "version": "1"}],
        },
    }
    for name, payload in files.items():
        if defect != "missing_sbom" or name != "sbom.cdx.json":
            (tmp_path / name).write_text(json.dumps(payload))
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "EVIDENCE_DIR": str(tmp_path), "TRIVY_IMAGE": scanner_pin},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is (defect is None), result.stderr
    if defect is None:
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["image"]["image_id"] == image_id
        assert (
            manifest["reports"]["vulnerabilities.json"]
            == hashlib.sha256((tmp_path / "vulnerabilities.json").read_bytes()).hexdigest()
        )
        assert (
            json.loads((tmp_path / "vulnerabilities.json").read_text())["Results"][0]["Vulnerabilities"]
            == report["Results"][0]["Vulnerabilities"]
        )

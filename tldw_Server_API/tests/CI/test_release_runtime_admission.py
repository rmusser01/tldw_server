"""Exercise release runtime identity, admission and artifact-integrity boundaries."""

from __future__ import annotations

import hashlib
import json
import os

# Tests execute fixed local scripts against controlled external-tool responses.
import subprocess  # nosec B404
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


def _backend_step(name: str) -> dict:
    workflow = yaml.safe_load(Path(".github/workflows/publish-docker.yml").read_text())
    steps = workflow["jobs"]["build-backend-candidates"]["steps"]
    matches = [step for step in steps if step.get("name") == name]
    assert len(matches) == 1, f"missing or duplicate release step: {name}"
    return matches[0]


@pytest.mark.parametrize(
    "failure", [None, "manifest", "config", "repository", "platform", "inspect", "probe", "output"]
)
def test_release_runtime_evidence_rejects_identity_drift_and_probe_failure(
    tmp_path: Path,
    failure: str | None,
) -> None:
    """Run the release script; replace only the Docker/registry boundary."""
    step = _backend_step("Verify embedded backend runtime")
    script = step["run"]
    script = script.replace("${{ matrix.name }}", "app")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    config = "sha256:" + "c" * 64
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "config": {"digest": "invalid" if failure == "config" else config},
            "layers": [],
        }
    ).encode()
    platform = "sha256:" + hashlib.sha256(manifest).hexdigest()
    subject = "sha256:" + "a" * 64
    repository = "ghcr.io/example/app"
    expected_ref = f"{repository}@{platform}"
    (tmp_path / "manifest.json").write_bytes(b"{}" if failure == "manifest" else manifest)
    (tmp_path / "inspect.json").write_text(
        json.dumps(
            [
                {
                    "Id": config,
                    "Os": "linux",
                    "Architecture": "arm64" if failure == "platform" else "amd64",
                    "RepoDigests": [f"{repository}@{subject}" if failure == "repository" else expected_ref],
                }
            ]
        )
    )
    if failure == "inspect":
        (tmp_path / "inspect.json").write_text("[]")
    observation = {
        "versions": {"chromadb": "1.5.9", "pyjwt": "2.10.1", "cryptography": "45.0.0"},
        "lock_sha256": "b" * 64,
        "chroma_backend": "chromadb.api.rust",
        "jwt_ec_backend": "cryptography.hazmat.bindings._rust.openssl.ec",
        "checks": {
            "per_user_collection_and_query_isolation": True,
            "foreign_collection_uuid_isolation": True,
            "no_tcp_listeners": True,
        },
        "os_facts": {"perl": {"status": "absent"}, "systemd_homed": {"paths": {}}},
        "scope": "isolated embedded-client probe; not application startup or a vulnerability waiver",
    }
    (tmp_path / "observation.json").write_text("invalid" if failure == "output" else json.dumps(observation))
    docker = r"""
docker() {
  case "$1 $2 $3" in
    "buildx imagetools inspect")
      test "$4" = "$EXPECTED_REF" && test "$5" = "--raw" || return 91
      cat manifest.json ;;
    "pull --platform linux/amd64")
      test "$4" = "$EXPECTED_REF" || return 92 ;;
    "image inspect "*)
      test "$3" = "$EXPECTED_REF" || return 93
      cat inspect.json ;;
    "run --rm --pull")
      local required
      for required in '--pull never' '--platform linux/amd64' '--network none' \
        '--cap-drop ALL' '--security-opt no-new-privileges:true' '--read-only' \
        '--user ' '--tmpfs /tmp:rw,nosuid,nodev,size=512m' \
        '/runtime_probe.py:/probe/runtime_probe.py:ro' '/uv.lock:/probe/uv.lock:ro'; do
        [[ " $* " == *"$required"* ]] || return 94
      done
      [[ "$*" == *"--entrypoint python $EXPECTED_REF /probe/runtime_probe.py probe --lock /probe/uv.lock" ]] || return 95
      test "$FAILURE" != probe || return 96
      cat observation.json ;;
    *) return 97 ;;
  esac
}
"""
    # Resolve the actual workflow environment using controlled build/identity outputs.
    values = {
        "${{ steps.build.outputs.digest }}": subject,
        "${{ steps.identity.outputs.platform_digest }}": platform,
        "${{ env.REGISTRY }}": "ghcr.io",
        "${{ env.IMAGE_NAME }}": "example/app",
        "${{ matrix.image_suffix }}": "",
    }
    step_env = dict(step["env"])
    for key, value in step_env.items():
        for expression, replacement in values.items():
            value = value.replace(expression, replacement)
        step_env[key] = value
    # Only Docker is replaced; the workflow's shell and validation Python execute.
    result = subprocess.run(  # nosec B607
        ["bash", "-c", docker + script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            **step_env,
            "EXPECTED_REF": expected_ref,
            "FAILURE": failure or "",
        },
        shell=False,  # nosec B603
        check=False,
    )
    output = evidence / "runtime-image-app.json"
    assert (result.returncode == 0) is (failure is None), result.stdout + result.stderr
    if failure is None:
        assert json.loads(output.read_text()) == {
            **observation,
            "subject_digest": subject,
            "platform_manifest_digest": platform,
            "config_digest": config,
        }
    else:
        assert not output.exists()


@pytest.mark.parametrize("policy", ["success", "failure", "skipped"])
@pytest.mark.parametrize("runtime", ["success", "failure", "skipped"])
def test_release_backend_admission_requires_successful_runtime_and_policy(
    policy: str,
    runtime: str,
) -> None:
    script = _backend_step("Require backend admission")["run"]
    runtime_id = _backend_step("Verify embedded backend runtime")["id"]
    script = script.replace("${{ steps.policy.outcome }}", policy)
    script = script.replace("${{ steps." + runtime_id + ".outcome }}", runtime)
    # Execute the actual gate with controlled step outcomes.
    result = subprocess.run(  # nosec B607
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        shell=False,  # nosec B603
        check=False,
    )
    assert (result.returncode == 0) is (policy == runtime == "success")


@pytest.mark.parametrize("filename", ["runtime-image-app.json", "platform-manifest-app.json"])
def test_release_backend_checksums_detect_runtime_evidence_tampering(tmp_path: Path, filename: str) -> None:
    script = _backend_step("Hash backend evidence")["run"]
    script = script.replace("${{ matrix.name }}", "app")
    script = script.replace("${{ steps.attest.outputs.bundle-path }}", str(tmp_path / "bundle.jsonl"))
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    for name in (
        "image-app.json",
        "sbom-image-app.cdx.json",
        "trivy-image-app.json",
        "scan-decision-image-app.json",
        "subject-app.json",
        "runtime-image-app.json",
        "platform-manifest-app.json",
    ):
        (evidence / name).write_text("{}\n")
    (tmp_path / "bundle.jsonl").write_text("{}\n")
    (tmp_path / "scanner").mkdir()
    (tmp_path / "scanner/scanner-metadata.json").write_text("{}\n")
    # Hash only task-owned fixtures using the real workflow command.
    result = subprocess.run(  # nosec B607
        ["bash", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        shell=False,  # nosec B603
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (evidence / filename).write_text("tampered\n")
    # Verify the generated checksum manifest with the standard local tool.
    checked = subprocess.run(  # nosec B607
        ["sha256sum", "-c", "SHA256SUMS-image-app"],
        cwd=evidence,
        capture_output=True,
        text=True,
        shell=False,  # nosec B603
        check=False,
    )
    assert checked.returncode != 0

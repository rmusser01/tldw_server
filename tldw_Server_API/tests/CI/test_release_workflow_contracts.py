"""Security contracts for image candidate admission and release promotion."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

RELEASE_WORKFLOW = Path(".github/workflows/publish-docker.yml")
MAIN_WORKFLOW = Path(".github/workflows/publish-ghcr-main.yml")
CONTAINER_WORKFLOW = Path(".github/workflows/container-build-check.yml")
PINNED_TRIVY = (
    "ghcr.io/aquasecurity/trivy:0.74.0@sha256:"
    "62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workflow_on(workflow: dict) -> dict:
    return workflow[True]


def _get_step(steps: list[dict], name: str) -> dict:
    matching = [step for step in steps if step.get("name") == name]
    assert matching, f"{name} step missing"
    return matching[0]


def _run(step: dict) -> str:
    script = step.get("run")
    assert isinstance(script, str), step
    return script


def _assert_changed_actions_are_pinned(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:") and not stripped.endswith(".yml"):
            target = stripped.removeprefix("uses:").strip()
            if target.startswith("./"):
                continue
            assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}(?:\s+#.*)?", target), line


@pytest.mark.parametrize(
    ("existing", "lookup_error", "post_digest", "succeeds", "creates"),
    [
        ("sha256:" + "a" * 64, "", "sha256:" + "a" * 64, True, False),
        ("sha256:" + "b" * 64, "", "sha256:" + "a" * 64, False, False),
        ("", "not found", "sha256:" + "a" * 64, True, True),
        ("", "unauthorized", "sha256:" + "a" * 64, False, False),
        ("", "network unavailable", "sha256:" + "a" * 64, False, False),
        ("", "not found", "sha256:" + "b" * 64, False, True),
    ],
)
def test_version_promotion_never_replaces_existing_content(
    tmp_path: Path, existing: str, lookup_error: str, post_digest: str,
    succeeds: bool, creates: bool,
) -> None:
    """Run the real promotion script with a controlled external registry boundary."""
    workflow = _load(RELEASE_WORKFLOW)
    script = _run(_get_step(workflow["jobs"]["promote-version"]["steps"], "Promote verified version tag"))
    script = script.replace("${{ matrix.name }}", "app").replace("${{ matrix.image_suffix }}", "")
    evidence = tmp_path / "admitted/release-evidence"
    evidence.mkdir(parents=True)
    candidate = "ghcr.io/example/app:candidate@sha256:" + "a" * 64
    (evidence / "image-app.json").write_text(json.dumps({
        "subject_digest": "sha256:" + "a" * 64, "reference": candidate,
    }))
    # Only Docker's registry calls are replaced; jq and all promotion logic run.
    registry = r'''
docker() {
  test "$1 $2" = "buildx imagetools" || return 90
  case "$3" in
    create)
      test "$4" = "--tag" && test "$5" = "ghcr.io/example/app:1.2.3" || return 91
      test "$6" = "$CANDIDATE" || return 92
      touch created
      ;;
    inspect)
      test "$4" = "ghcr.io/example/app:1.2.3" || return 93
      if test -e created; then printf '"%s"\n' "$POST_DIGEST"
      elif test -n "$LOOKUP_ERROR"; then printf 'ERROR: %s: %s\n' "$4" "$LOOKUP_ERROR" >&2; return 1
      else printf '"%s"\n' "$EXISTING_DIGEST"; fi
      ;;
    *) return 94 ;;
  esac
}
'''
    result = subprocess.run(  # nosec B603,B607 - controlled local workflow with a fake registry
        ["bash", "-c", registry + script], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "REGISTRY": "ghcr.io", "IMAGE_NAME": "example/app",
             "RELEASE_VERSION": "1.2.3", "EXISTING_DIGEST": existing,
             "LOOKUP_ERROR": lookup_error, "POST_DIGEST": post_digest, "CANDIDATE": candidate},
        check=False,
    )
    assert (result.returncode == 0) is succeeds, result.stdout + result.stderr
    assert (tmp_path / "created").exists() is creates


def test_release_workflow_is_manual_draft_admission_only() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    on = _workflow_on(workflow)

    assert set(on) == {"workflow_dispatch"}
    inputs = on["workflow_dispatch"]["inputs"]
    assert inputs["release_tag"]["required"] is True
    assert inputs["confirmation"]["required"] is True
    assert "release" not in on

    validate = workflow["jobs"]["validate-release"]
    script = _run(_get_step(validate["steps"], "Verify trusted draft release"))
    assert "isDraft" in script
    assert "refs/tags/" in script
    assert "^v?[0-9]+\\.[0-9]+\\.[0-9]+$" in script
    assert _get_step(validate["steps"], "Verify trusted draft release")["env"][
        "EXPECTED_SHA"
    ] == "${{ github.sha }}"
    assert "RELEASE_CONFIRMATION" in script


def test_release_workflow_requires_reusable_source_admission() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    source = workflow["jobs"]["source-admission"]

    assert source["uses"] == "./.github/workflows/sbom.yml"
    for job_name in ("build-backend-candidates", "build-frontend-candidates", "scan-reference-images"):
        assert "source-admission" in workflow["jobs"][job_name]["needs"]


def test_release_workflow_has_exact_owned_and_reference_matrices() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    jobs = workflow["jobs"]
    backend = jobs["build-backend-candidates"]["strategy"]["matrix"]["include"]
    frontend = jobs["build-frontend-candidates"]["strategy"]["matrix"]["include"]
    references = jobs["scan-reference-images"]["strategy"]["matrix"]["include"]

    assert [entry["name"] for entry in backend] == ["app", "worker", "audio-worker"]
    assert [entry["name"] for entry in frontend] == ["webui", "admin-ui"]
    assert [entry["name"] for entry in references] == [
        "caddy",
        "postgres",
        "redis",
        "prometheus",
        "alertmanager",
        "grafana",
    ]


def test_release_backend_candidates_are_scanned_attested_and_gated_before_promotion() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    job = workflow["jobs"]["build-backend-candidates"]
    steps = job["steps"]
    build = _get_step(steps, "Build and push immutable candidate")

    assert build["with"]["push"] is True
    assert build["with"]["platforms"] == "linux/amd64"
    assert build["with"]["provenance"] == "mode=max"
    assert build["with"]["sbom"] is True
    tags = build["with"]["tags"]
    assert "${{ github.run_id }}-${{ github.run_attempt }}" in tags
    assert ":latest" not in tags
    assert "release_version" not in tags

    names = [step.get("name") for step in steps]
    assert names.index("Build and push immutable candidate") < names.index(
        "Scan exact backend subject"
    )
    assert names.index("Scan exact backend subject") < names.index(
        "Attest exact backend subject"
    )
    assert names.index("Attest exact backend subject") < names.index(
        "Upload backend evidence"
    )
    assert names.index("Upload backend evidence") < names.index(
        "Require backend admission"
    )
    attest = _get_step(steps, "Attest exact backend subject")
    assert attest["id"] == "attest"
    record = _run(_get_step(steps, "Write backend evidence record"))
    hashes = _run(_get_step(steps, "Hash backend evidence"))
    assert "steps.attest.outputs.attestation-url" in str(
        _get_step(steps, "Write backend evidence record")["env"]
    )
    assert '"provenance_ref": os.environ["PROVENANCE_REF"]' in record
    assert "steps.attest.outputs.bundle-path" in hashes
    assert "provenance-image-${{ matrix.name }}.jsonl" in hashes


def test_frontend_candidates_are_local_build_and_scan_only() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    job = workflow["jobs"]["build-frontend-candidates"]
    build = _get_step(job["steps"], "Build local frontend OCI candidate")

    assert job["permissions"] == {
        "contents": "read",
        "attestations": "write",
        "id-token": "write",
    }
    assert "packages" not in job["permissions"]
    assert build["with"]["push"] is False
    assert build["with"]["platforms"] == "linux/amd64"
    assert build["with"]["provenance"] == "mode=max"
    assert build["with"]["sbom"] is True
    assert "type=oci" in build["with"]["outputs"]
    assert all("docker/login-action" not in str(step.get("uses", "")) for step in job["steps"])
    attest = _get_step(job["steps"], "Attest exact frontend subject")
    assert attest["id"] == "attest"
    assert attest["with"]["push-to-registry"] is False
    record_step = _get_step(job["steps"], "Write frontend evidence record")
    record = _run(record_step)
    hashes = _run(_get_step(job["steps"], "Hash frontend evidence"))
    assert "steps.attest.outputs.attestation-url" in str(record_step["env"])
    assert '"provenance_ref": os.environ["PROVENANCE_REF"]' in record
    assert "steps.attest.outputs.bundle-path" in hashes
    assert "provenance-image-${{ matrix.name }}.jsonl" in hashes


def test_frontend_evidence_retains_exact_attested_subject_bytes(tmp_path: Path) -> None:
    """The offline signature verifier needs the OCI bytes, not an identity summary."""
    layout = tmp_path / "input"
    layout.mkdir()
    child = "sha256:" + "b" * 64
    raw = json.dumps({"schemaVersion": 2, "manifests": [{
        "digest": child, "platform": {"os": "linux", "architecture": "amd64"},
    }]}).encode()
    (layout / "index.json").write_bytes(raw)
    with tarfile.open(tmp_path / "webui.oci.tar", "w") as archive:
        archive.add(layout / "index.json", arcname="index.json")
    job = _load(RELEASE_WORKFLOW)["jobs"]["build-frontend-candidates"]
    script = _run(_get_step(job["steps"], "Extract exact frontend OCI subject"))
    script = script.replace("${{ matrix.name }}", "webui")
    subprocess.run(["bash", "-c", script], cwd=tmp_path, check=True, env={
        **os.environ, "SUBJECT_DIGEST": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "RUNNER_TEMP": str(tmp_path), "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    })

    assert (tmp_path / "evidence/subject-webui.json").read_bytes() == raw


def test_release_scanner_database_is_fresh_shared_and_offline() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    assert workflow["env"]["TRIVY_IMAGE"] == PINNED_TRIVY

    scanner = workflow["jobs"]["prepare-scanner"]
    prepare = _run(_get_step(scanner["steps"], "Prepare fresh pinned Trivy database"))
    assert "--download-db-only" in prepare
    assert "timedelta(hours=24)" in prepare

    for job_name, step_name in (
        ("build-backend-candidates", "Scan exact backend subject"),
        ("build-frontend-candidates", "Scan exact frontend archive"),
        ("scan-reference-images", "Scan exact reference subject"),
    ):
        job = workflow["jobs"][job_name]
        assert "prepare-scanner" in job["needs"]
        scan = _run(_get_step(job["steps"], step_name))
        assert "--platform linux/amd64" in scan
        assert "--skip-db-update" in scan
        assert "--ignore-unfixed=false" in scan

    backend_scan = _run(
        _get_step(
            workflow["jobs"]["build-backend-candidates"]["steps"],
            "Scan exact backend subject",
        )
    )
    frontend_scan = _run(
        _get_step(
            workflow["jobs"]["build-frontend-candidates"]["steps"],
            "Scan exact frontend archive",
        )
    )
    reference_scan = _run(
        _get_step(
            workflow["jobs"]["scan-reference-images"]["steps"],
            "Scan exact reference subject",
        )
    )
    assert "--network none" not in backend_scan
    assert "--network none" in frontend_scan
    assert "--network none" not in reference_scan


@pytest.mark.parametrize(
    ("path", "job_name", "step_name", "offline"),
    [
        (CONTAINER_WORKFLOW, "build-and-scan", "Scan local OCI candidate", True),
        (RELEASE_WORKFLOW, "build-backend-candidates", "Scan exact backend subject", False),
        (RELEASE_WORKFLOW, "build-frontend-candidates", "Scan exact frontend archive", True),
        (RELEASE_WORKFLOW, "scan-reference-images", "Scan exact reference subject", False),
    ],
)
@pytest.mark.parametrize("scanner_status", [0, 42])
def test_image_scans_use_private_disk_scratch_and_preserve_guards(
    tmp_path: Path, path: Path, job_name: str, step_name: str,
    offline: bool, scanner_status: int,
) -> None:
    """Execute each scan path; catch capped scratch, reuse, leaks, or lost guards."""
    workflow = _load(path)
    script = _run(_get_step(workflow["jobs"][job_name]["steps"], step_name))
    script = script.replace("${{ matrix.name }}", "app")
    runner_temp = tmp_path / "runner temp"
    runner_temp.mkdir()
    digest = "sha256:" + "a" * 64
    reference = "ghcr.io/example/app@" + digest
    # Replace only Docker: observe the real helper's arguments and live scratch.
    docker = r'''
docker() {
  python - "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
scratch = [args[i + 1].removesuffix(":/tmp:rw")
           for i, arg in enumerate(args[:-1])
           if arg == "--volume" and args[i + 1].endswith(":/tmp:rw")]
mode = None
if scratch:
    directory = Path(scratch[0])
    mode = directory.stat().st_mode & 0o777
    (directory / "scanner-output").write_text("temporary scanner data")
with Path("docker-calls.jsonl").open("a") as output:
    print(json.dumps({"args": args, "scratch": scratch, "mode": mode}), file=output)
sys.exit(int(os.environ["SCANNER_STATUS"]))
PY
}
'''
    result = subprocess.run(
        ["bash", "-c", docker + script],  # nosec B603,B607
        cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "RUNNER_TEMP": str(runner_temp),
             "TRIVY_IMAGE": workflow["env"]["TRIVY_IMAGE"], "SUBJECT_DIGEST": digest,
             "CANDIDATE_REF": reference, "REFERENCE": reference,
             "SCANNER_STATUS": str(scanner_status)},
        shell=False,  # nosec B603
        check=False,
    )
    assert result.returncode == scanner_status, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / "docker-calls.jsonl").read_text().splitlines()]
    assert len(calls) == (2 if scanner_status == 0 else 1)
    scratch_paths = []
    for call, scan_format in zip(calls, ("json", "cyclonedx")):
        assert len(call["scratch"]) == 1, "image scan needs disk-backed /tmp"
        scratch = Path(call["scratch"][0])
        scratch_paths.append(scratch)
        assert scratch.parent == runner_temp
        assert call["mode"] == 0o700
        assert not scratch.exists(), "scratch must be removed on success and failure"
        args = call["args"]
        assert args[:4] == ["run", "--rm", "--platform", "linux/amd64"]
        assert args[args.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
        assert args[args.index("--cap-drop") + 1] == "ALL"
        assert args[args.index("--security-opt") + 1] == "no-new-privileges:true"
        assert "--read-only" in args
        assert "--tmpfs" not in args
        assert ("--network" in args) is offline
        if offline:
            assert args[args.index("--network") + 1] == "none"
            assert f"{runner_temp}/app-layout:/input:ro" in args
            assert args[args.index("--input") + 1] == "/input@" + digest
        else:
            assert args[-1] == reference
        scanner_args = args[args.index(PINNED_TRIVY) + 1:]
        assert scanner_args[:3] == ["image", "--platform", "linux/amd64"]
        assert "--skip-db-update" in scanner_args
        assert "--ignore-unfixed=false" in scanner_args
        assert scanner_args[scanner_args.index("--scanners") + 1] == "vuln"
        assert scanner_args[scanner_args.index("--format") + 1] == scan_format
        assert not any(arg.startswith(("--skip-files", "--skip-dirs")) for arg in scanner_args)
    assert len(set(scratch_paths)) == len(calls), "each scanner invocation needs private scratch"


def test_release_component_evidence_is_checksummed_before_upload() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    jobs = workflow["jobs"]

    for job_name, step_name in (
        ("build-backend-candidates", "Hash backend evidence"),
        ("build-frontend-candidates", "Hash frontend evidence"),
        ("scan-reference-images", "Hash reference evidence"),
    ):
        script = _run(_get_step(jobs[job_name]["steps"], step_name))
        assert "SHA256SUMS-image-${{ matrix.name }}" in script
        assert "sha256sum -c" in script

    admission = _run(
        _get_step(
            jobs["admit-release-evidence"]["steps"],
            "Assemble and verify release evidence",
        )
    )
    assert 'test "$checksum_count" = "11"' in admission
    assert "sha256sum -c SHA256SUMS.source" in admission
    assert "sha256sum -c SHA256SUMS.scan" in admission


def test_release_admits_complete_evidence_before_any_promotion() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    jobs = workflow["jobs"]
    admission = jobs["admit-release-evidence"]

    assert set(admission["needs"]) == {
        "validate-release",
        "source-admission",
        "prepare-scanner",
        "build-backend-candidates",
        "build-frontend-candidates",
        "scan-reference-images",
    }
    script = _run(_get_step(admission["steps"], "Assemble and verify release evidence"))
    assert "release_evidence.py assemble" in script
    assert "release_evidence.py verify" in script
    assert "admit-release-evidence" in jobs["promote-version"]["needs"]
    assert "promote-version" in jobs["promote-floating"]["needs"]
    assert "promote-floating" in jobs["publish-release"]["needs"]


def test_release_promotions_copy_verified_digests_in_two_stages() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    jobs = workflow["jobs"]
    version = _run(_get_step(jobs["promote-version"]["steps"], "Promote verified version tag"))
    floating = _run(_get_step(jobs["promote-floating"]["steps"], "Promote verified floating tags"))

    assert "docker buildx imagetools create" in version
    assert "expected_digest" in version
    assert "release_version" in version
    assert "docker buildx imagetools create" in floating
    assert "expected_digest" in floating
    assert ":latest" in floating
    assert "major_tag" in floating
    assert "minor_tag" in floating


def test_release_assets_are_verified_before_draft_is_published() -> None:
    workflow = _load(RELEASE_WORKFLOW)
    steps = workflow["jobs"]["publish-release"]["steps"]
    names = [step.get("name") for step in steps]

    assert names.index("Verify admitted release manifest") < names.index(
        "Upload and verify release assets"
    )
    assert names.index("Upload and verify release assets") < names.index(
        "Publish verified draft release"
    )
    assert "release_evidence.py verify" in _run(
        _get_step(steps, "Verify admitted release manifest")
    )
    assert "gh release upload" in _run(_get_step(steps, "Upload and verify release assets"))
    assert "--draft=false" in _run(_get_step(steps, "Publish verified draft release"))


def test_release_workflow_only_publishes_to_ghcr() -> None:
    text = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "ghcr.io" in text
    assert "docker.io" not in text
    assert "quay.io" not in text


def test_main_publish_scans_and_gates_unique_candidate_before_aliases() -> None:
    workflow = _load(MAIN_WORKFLOW)
    jobs = workflow["jobs"]
    assert jobs["source-admission"]["uses"] == "./.github/workflows/sbom.yml"

    job = jobs["build-scan-promote"]
    assert "source-admission" in job["needs"]
    assert job["env"]["TRIVY_IMAGE"] == PINNED_TRIVY
    steps = job["steps"]
    names = [step.get("name") for step in steps]
    assert names.index("Build and push immutable candidate") < names.index(
        "Scan exact candidate subject"
    )
    assert names.index("Scan exact candidate subject") < names.index(
        "Require candidate admission"
    )
    assert names.index("Require candidate admission") < names.index(
        "Promote verified main aliases"
    )
    assert "sha256sum -c" in _run(_get_step(steps, "Hash candidate evidence"))
    attest = _get_step(steps, "Attest exact candidate subject")
    assert attest["id"] == "attest"
    hashes = _run(_get_step(steps, "Hash candidate evidence"))
    assert "steps.attest.outputs.bundle-path" in hashes
    assert "provenance-image-app.jsonl" in hashes

    build = _get_step(steps, "Build and push immutable candidate")
    tags = build["with"]["tags"]
    assert "${{ github.run_id }}-${{ github.run_attempt }}" in tags
    assert ":main" not in tags
    assert "sha-" not in tags


def test_container_build_check_covers_all_images_without_registry_writes() -> None:
    workflow = _load(CONTAINER_WORKFLOW)
    job = workflow["jobs"]["build-and-scan"]
    matrix = job["strategy"]["matrix"]["include"]

    assert [entry["name"] for entry in matrix] == [
        "app",
        "worker",
        "audio-worker",
        "webui",
        "admin-ui",
    ]
    assert workflow["permissions"] == {"contents": "read"}
    build = _get_step(job["steps"], "Build local OCI candidate")
    assert build["with"]["push"] is False
    assert build["with"]["platforms"] == "linux/amd64"
    assert build["with"]["provenance"] == "mode=max"
    assert build["with"]["sbom"] is True
    assert "type=oci" in build["with"]["outputs"]
    scan = _run(_get_step(job["steps"], "Scan local OCI candidate"))
    assert "--input" in scan
    assert "--platform linux/amd64" in scan
    assert "--network none" in scan
    assert "sha256sum -c" in _run(
        _get_step(job["steps"], "Hash container admission evidence")
    )
    assert "packages: write" not in CONTAINER_WORKFLOW.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "path", [RELEASE_WORKFLOW, MAIN_WORKFLOW, CONTAINER_WORKFLOW]
)
def test_changed_container_workflow_actions_are_commit_pinned(path: Path) -> None:
    _assert_changed_actions_are_pinned(path)

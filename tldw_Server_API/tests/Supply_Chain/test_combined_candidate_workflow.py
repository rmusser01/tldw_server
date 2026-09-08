"""Behavioral regression tests for the combined Expat candidate workflow."""

import os
import subprocess  # nosec B404
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/combined-expat-candidate.yml"


def _identity_step() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["assemble"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == "Bind execution image to retained OCI")


def _run_identity_step(
    tmp_path: Path, inspect_status: int, *, wrong_target: bool = False
) -> subprocess.CompletedProcess[str]:
    evidence = tmp_path / "combined-evidence"
    evidence.mkdir()
    config = "sha256:" + "a" * 64
    identity_step = _identity_step()
    if wrong_target:
        identity_step = identity_step.replace('"tldw-combined-expat:$GITHUB_SHA"', '"wrong-target:$GITHUB_SHA"')
    script = f"""
docker() {{
  [[ "$#" == 5 && "$1" == image && "$2" == inspect &&
     "$3" == "tldw-combined-expat:$GITHUB_SHA" && "$4" == --format &&
     "$5" == '{{{{.Id}}}}' ]] || return 91
  if (( INSPECT_STATUS != 0 )); then return "$INSPECT_STATUS"; fi
  printf '%s\n' "$EXPECTED_CONFIG"
}}
python3() {{
  test "$#" = 6
  test "$1" = Dockerfiles/candidates/combined-expat/image-identity.py
  test "$2" = "$RUNNER_TEMP/combined-evidence/candidate.oci.tar"
  test "$3" = --config
  test "$4" = "$EXPECTED_CONFIG"
  test "$5" = --commit
  test "$6" = "$GITHUB_SHA"
  printf 'verifier-invoked\n' >> "$COMMAND_LOG"
  printf '{{"config_digest":"%s"}}\n' "$4"
}}
{identity_step}
"""
    return subprocess.run(  # nosec B603
        ["/bin/bash", "-c", script],
        cwd=ROOT,
        env={
            **os.environ,
            "COMMAND_LOG": str(tmp_path / "commands.log"),
            "EXPECTED_CONFIG": config,
            "GITHUB_SHA": "b" * 40,
            "INSPECT_STATUS": str(inspect_status),
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_identity_step_passes_inspected_config_to_verifier(tmp_path):
    result = _run_identity_step(tmp_path, inspect_status=0)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "commands.log").read_text() == "verifier-invoked\n"
    assert (tmp_path / "combined-evidence/image-identity.json").read_text() == (
        '{"config_digest":"sha256:' + "a" * 64 + '"}\n'
    )


def test_identity_step_stops_when_image_inspection_fails(tmp_path):
    result = _run_identity_step(tmp_path, inspect_status=37)

    assert result.returncode == 37, result.stdout + result.stderr
    assert not (tmp_path / "commands.log").exists()
    assert not (tmp_path / "combined-evidence/image-identity.json").exists()


def test_identity_step_rejects_wrong_image_target(tmp_path):
    result = _run_identity_step(tmp_path, inspect_status=0, wrong_target=True)

    assert result.returncode == 91, result.stdout + result.stderr
    assert not (tmp_path / "commands.log").exists()
    assert not (tmp_path / "combined-evidence/image-identity.json").exists()

"""Run the pinned workflow generator against package roots without local locks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import pytest
import yaml

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("TLDW_TEST_SBOM_DOCKER") != "1",
        reason="Set TLDW_TEST_SBOM_DOCKER=1 to run the pinned Docker generator",
    ),
]
ROOT = Path(__file__).resolve().parents[3]


def test_workflow_emits_package_metadata_without_child_locks(tmp_path: Path) -> None:
    """A clean workspace produces locked dependencies and all three root identities."""
    git, bash, docker = (shutil.which(command) for command in ("git", "bash", "docker"))
    assert git and bash and docker, "The integration test requires Git, Bash, and Docker"
    workflow = yaml.safe_load((ROOT / ".github/workflows/sbom.yml").read_text())
    step = next(step for step in workflow["jobs"]["generate-apps-workspace"]["steps"]
                if step.get("name") == "Generate required-only Bun workspace CycloneDX")
    # Execute fixed repository-owned commands with resolved tools and no shell interpolation.
    manifests = subprocess.check_output([  # nosec B603
        git, "ls-files", "--", "apps/**/package.json", "apps/package.json", "apps/bun.lock",
    ], cwd=ROOT, text=True).splitlines()
    for filename in manifests:
        destination = tmp_path / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / filename, destination)
    env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
    env.update(workflow["env"], GITHUB_WORKSPACE=str(tmp_path))
    subprocess.run([  # nosec B603
        bash, "-c", step["run"],
    ], cwd=tmp_path, env=env, check=True, capture_output=True, text=True, timeout=180)
    payload = json.loads((tmp_path / "artifacts/source-sbom/sbom-apps-workspace.cdx.json").read_text())
    identities = {(item["group"], item["name"], item["version"], item["purl"])
                  for item in payload["metadata"]["component"]["components"]}
    assert identities == {
        ("", "tldw-frontend", "0.1.0", "pkg:npm/tldw-frontend@0.1.0"),
        ("", "tldw-assistant", "0.1.0", "pkg:npm/tldw-assistant@0.1.0"),
        ("@tldw", "ui", "0.1.0", "pkg:npm/%40tldw/ui@0.1.0"),
    }
    assert payload["components"]
    assert (tmp_path / "apps/bun.lock").read_bytes() == (ROOT / "apps/bun.lock").read_bytes()
    # This path is a bounded container-only tmpfs, not a host temporary directory.
    validator_tmpfs = "/tmp:rw,nosuid,nodev,size=128m"  # nosec B108
    subprocess.run([  # nosec B603
        docker, "run", "--rm", "--platform", "linux/amd64", "--network", "none",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--read-only",
        "--tmpfs", validator_tmpfs,
        "--volume", f"{tmp_path / 'artifacts/source-sbom'}:/evidence:ro",
        workflow["env"]["CYCLONEDX_IMAGE"], "validate", "--input-file",
        "/evidence/sbom-apps-workspace.cdx.json",
    ], check=True, capture_output=True, text=True, timeout=60)

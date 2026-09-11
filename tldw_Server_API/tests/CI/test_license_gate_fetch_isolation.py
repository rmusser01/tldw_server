"""Execute the trusted evaluator against hostile, locally hosted pull request refs.

The subprocess import is intentional: this integration test exercises Git and
the trusted workflow shell rather than mocking their security behavior.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER = Path("Helper_Scripts/ci/check_frontend_license_gate.py")
GIT = shutil.which("git")
BASH = shutil.which("bash")


def git(directory: Path, *arguments: str) -> str:
    assert GIT is not None
    # Fixed executable; arguments are isolated fixture data.
    return subprocess.check_output(  # nosec B603
        [GIT, "-C", str(directory), *arguments], text=True, stderr=subprocess.PIPE
    ).strip()


@pytest.mark.parametrize("scenario", ["allowed", "hostile_classifier", "wrong_head", "wrong_base"])
def test_fetch_of_pr_objects_never_executes_or_checks_out_pr_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    assert BASH is not None
    # Literal shell version probe.
    bash_major = subprocess.check_output(  # nosec B603
        [BASH, "-c", "printf '%s' \"${BASH_VERSINFO[0]}\""], text=True
    )
    if int(bash_major) < 4:
        pytest.skip("Requires the runner's Bash >=4; Bash 3 ignores errexit for [[ guards ]]")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init", "-b", "main")
    git(remote, "config", "user.email", "fixture@example.invalid")
    git(remote, "config", "user.name", "Security fixture")
    (remote / CLASSIFIER).parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / CLASSIFIER, remote / CLASSIFIER)
    git(remote, "add", ".")
    git(remote, "commit", "-m", "trusted classifier")
    base_sha = git(remote, "rev-parse", "HEAD")
    git(remote, "checkout", "-b", "attacker")
    marker = tmp_path / "attacker-executed"
    if scenario == "hostile_classifier":
        (remote / CLASSIFIER).write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n", encoding="utf-8")
    else:
        (remote / "README.md").write_text("PR content\n", encoding="utf-8")
    git(remote, "add", ".")
    git(remote, "commit", "-m", "untrusted PR")
    head_sha = git(remote, "rev-parse", "HEAD")
    git(remote, "update-ref", "refs/pull/1/head", head_sha)
    trusted = tmp_path / "trusted"
    git(tmp_path, "clone", "--branch", "main", "--single-branch", str(remote), str(trusted))
    # Only redirect the fixed public remote to an isolated local fixture. The
    # evaluator itself still fetches the same validated refs and immutable SHAs.
    git(trusted, "config", f"url.{remote}.insteadOf", "https://github.com/fixture/repository.git")
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/frontend-license-gate.yml").read_text())
    steps = workflow["jobs"]["frontend-license-gate-audit"]["steps"]
    script = next(step["run"] for step in steps if step.get("id") == "evaluate")
    output = tmp_path / "github-output"
    environment = {
        **os.environ,
        "STATUS_REPOSITORY": "fixture/repository",
        "PR_NUMBER": "1",
        "BASE_REF": "main",
        "BASE_SHA": head_sha if scenario == "wrong_base" else base_sha,
        "HEAD_SHA": base_sha if scenario == "wrong_head" else head_sha,
        "PR_AUTHOR": "external-contributor",
        "REPOSITORY_OWNER": "owner",
        "GITHUB_OUTPUT": str(output),
    }
    # Trusted base workflow, never the hostile PR script.
    result = subprocess.run(  # nosec B603
        [BASH, "-c", script], cwd=trusted, env=environment, capture_output=True, text=True, timeout=30
    )
    assert not marker.exists(), result.stderr
    assert git(trusted, "rev-parse", "HEAD") == base_sha
    assert (trusted / CLASSIFIER).read_bytes() == (REPO_ROOT / CLASSIFIER).read_bytes()
    assert git(trusted, "status", "--porcelain") == ""
    if scenario == "allowed":
        assert result.returncode == 0, result.stderr
        assert output.read_text() == "verdict=success\n"
    else:
        assert result.returncode != 0
        assert output.read_text() == "verdict=failure\n"

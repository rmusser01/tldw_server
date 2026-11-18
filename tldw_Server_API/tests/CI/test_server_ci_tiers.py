"""Server CI keeps uncertain changes broad and publication behind exact-run results."""

import os
import shutil
import subprocess  # nosec B404 -- Execute the fixed workflow shell against literal test outcomes.
import sys
from pathlib import Path

import pytest
import yaml
from Helper_Scripts.ci.server_ci_tier import select_tier

from tldw_Server_API.tests.CI.test_merge_queue_gate_workflows import _evaluate

BASH_REQUIRED = pytest.mark.skipif(shutil.which("bash") is None, reason="requires Bash")
WORD_CHUNKER = b"tldw_Server_API/app/core/Chunking/strategies/words.py"


def _workflow(name: str) -> dict:
    return yaml.safe_load(Path(f".github/workflows/{name}.yml").read_text())


@pytest.mark.parametrize(
    ("diff", "tier"),
    [
        (b"M\0" + WORD_CHUNKER + b"\0", "light"),
        (b"M\0" + WORD_CHUNKER + b"\0A\0Docs/Development/Notes.md\0", "light"),
        (b"A\0Docs/Development/Notes.md\0", "unchanged"),
        (b"M\0backlog/tasks/task-15572 - Current task.md\0", "unchanged"),
        (b"M\0" + WORD_CHUNKER + b"\0M\0pyproject.toml\0", "full"),
        (b"M\0tldw_Server_API/app/core/AuthNZ/settings.py\0", "full"),
        (b"M\0unknown/file.py\0", "full"),
        (b"M\0.github/workflows/ci.yml\0", "full"),
        (b"D\0" + WORD_CHUNKER + b"\0", "full"),
        (b"A\0" + WORD_CHUNKER + b"\0", "full"),
        (b"T\0" + WORD_CHUNKER + b"\0", "full"),
        (b"R100\0a.py\0" + WORD_CHUNKER + b"\0", "full"),
        (b"M\0" + WORD_CHUNKER, "full"),
        (b"", "full"),
    ],
)
def test_pr_path_routing_defaults_to_full(diff, tier):
    assert select_tier(diff) == tier  # nosec B101 -- Test assertion.


def test_main_pr_has_an_unfiltered_distinct_release_check():
    workflow = _workflow("ci")
    assert workflow[True]["pull_request"] == {"branches": ["dev", "main"]}  # nosec B101
    verdict = workflow["jobs"]["release-qualified"]
    assert _evaluate(  # nosec B101
        verdict["if"],
        {"github.event_name": "pull_request", "github.event.pull_request.base.ref": "main", "inputs.nightly": ""},
        cancelled=True,
    )
    assert workflow["jobs"]["server-ci-verdict"]["if"] == "always()"  # nosec B101
    assert "github.event.pull_request.base.ref == 'main' && 'Main PR release qualification'" in verdict["name"]  # nosec B101


@pytest.mark.parametrize("name", ("publish-pypi", "publish-docker"))
def test_publishers_call_read_only_qualification(name):
    jobs = _workflow(name)["jobs"]
    qualification = jobs["qualify"]
    assert qualification["uses"] == "./.github/workflows/ci.yml"  # nosec B101
    assert all(access == "read" for access in qualification["permissions"].values())  # nosec B101
    downstream = (
        ("publish-testpypi", "publish-pypi", "rolling-image") if name == "publish-pypi" else ("push_to_registries",)
    )
    for job in downstream:
        assert "qualify" in jobs[job]["needs"]  # nosec B101
        assert "needs.qualify.outputs.qualified == 'true'" in jobs[job]["if"]  # nosec B101


@BASH_REQUIRED
@pytest.mark.parametrize("result", ("failure", "cancelled", "skipped", ""))
def test_release_verdict_rejects_every_missing_or_unsuccessful_dependency(result):
    job = _workflow("ci")["jobs"]["release-qualified"]
    step = job["steps"][0]
    for missing in step["env"]:
        values = dict.fromkeys(step["env"], "success")
        values[missing] = result
        proc = subprocess.run(  # nosec B603, B607 -- Executes checked-in Bash with literal outcomes.
            ["bash", "-e", "-c", step["run"]], env={**os.environ, **values}, capture_output=True
        )
        assert proc.returncode != 0, missing  # nosec B101


@BASH_REQUIRED
def test_release_verdict_accepts_all_expected_jobs_succeeding(tmp_path):
    step = _workflow("ci")["jobs"]["release-qualified"]["steps"][0]
    proc = subprocess.run(  # nosec B603, B607 -- Executes checked-in Bash with literal outcomes.
        ["bash", "-e", "-c", step["run"]],
        env={**os.environ, **dict.fromkeys(step["env"], "success"), "GITHUB_OUTPUT": str(tmp_path / "output")},
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr  # nosec B101
    assert (tmp_path / "output").read_text() == "qualified=true\n"  # nosec B101


@pytest.mark.parametrize("tier", ("full", "light", "unchanged", "nightly", "release"))
def test_light_and_release_select_their_own_matrices(tier):
    jobs = _workflow("ci")["jobs"]
    context = {
        "needs.changes.result": "success",
        "needs.changes.outputs.tier": tier,
        "needs.lint.result": "success",
        "needs.syntax-check.result": "success",
    }
    assert _evaluate(jobs["light-backend"]["if"], context) == (tier == "light")  # nosec B101
    assert _evaluate(jobs["full-suite-linux-312-shards"]["if"], context) == (  # nosec B101
        tier in {"full", "nightly", "release"}
    )
    assert _evaluate(jobs["full-suite-os-313-release-shards"]["if"], context) == (  # nosec B101
        tier == "release"
    )


@BASH_REQUIRED
@pytest.mark.parametrize(
    ("event", "ref", "nightly", "enabled", "same_repo", "expected"),
    [
        ("pull_request", "dev", "false", "true", "true", "light"),
        ("pull_request", "dev", "false", "true", "false", "full"),
        ("pull_request", "dev", "false", "", "true", "full"),
        ("pull_request", "main", "false", "true", "true", "release"),
        ("workflow_dispatch", "dev", "true", "", "false", "nightly"),
        ("workflow_dispatch", "main", "true", "", "false", None),
        ("workflow_dispatch", "main", "false", "", "false", "release"),
        ("push", "main", "false", "true", "false", "release"),
    ],
)
def test_select_tier_from_event(tmp_path, event, ref, nightly, enabled, same_repo, expected):
    changes = _workflow("ci")["jobs"]["changes"]
    step = next(step for step in changes["steps"] if step.get("id") == "tier")
    output = tmp_path / "output"
    diff = tmp_path / "diff"
    diff.write_bytes(b"M\0" + WORD_CHUNKER + b"\0")
    proc = subprocess.run(  # nosec B603, B607 -- Exercise the checked-in shell with a fixture diff.
        [
            "bash",
            "-e",
            "-c",
            'git() { cat "$FIXTURE_DIFF"; }; python() { "$FIXTURE_PYTHON" "$@"; };\n' + step["run"],
        ],
        env={
            **os.environ,
            "FIXTURE_DIFF": str(diff),
            "FIXTURE_PYTHON": sys.executable,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_SHA": "f" * 40,
            "GITHUB_REF": f"refs/heads/{ref}",
            "GITHUB_EVENT_NAME": event,
            "DEV_PR": str(event == "pull_request" and ref == "dev").lower(),
            "SAME_REPO": same_repo,
            "BASE_SHA": "a" * 40,
            "NIGHTLY": nightly,
            "OPT_IN": enabled,
            "BACKEND_CHANGED": "true",
        },
        capture_output=True,
    )
    if expected is None:
        assert proc.returncode != 0 and not output.exists()  # nosec B101
    else:
        assert proc.returncode == 0, proc.stderr  # nosec B101
        assert output.read_text() == f"tier={expected}\n"  # nosec B101


@BASH_REQUIRED
@pytest.mark.parametrize(
    ("tier", "result", "passed"),
    [
        ("light", {}, True),
        ("light", {"LIGHT": "skipped"}, False),
        ("nightly", {}, True),
        ("nightly", {"LINUX": "cancelled"}, False),
        ("nightly", {"RATE_LIMITS": "failure"}, False),
        ("full", {"LICENSE": "false"}, False),
        ("", {}, False),
    ],
)
def test_server_ci_verdict_requires_the_selected_jobs(tier, result, passed):
    step = _workflow("ci")["jobs"]["server-ci-verdict"]["steps"][0]
    values = {**dict.fromkeys(step["env"], "success"), "LICENSE": "true", "TIER": tier}
    if tier == "light":
        values.update(LINUX="skipped", RATE_LIMITS="skipped")
    else:
        values["LIGHT"] = "skipped"
    proc = subprocess.run(  # nosec B603, B607 -- Verify the selected verdict against literal outcomes.
        ["bash", "-e", "-c", step["run"]], env={**os.environ, **values, **result}, capture_output=True
    )
    assert (proc.returncode == 0) == passed, proc.stdout  # nosec B101

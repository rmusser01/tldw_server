"""Server CI keeps dev focused and main and publication behind full qualification."""

import os
import shutil
import subprocess  # nosec B404 -- Execute the fixed workflow shell against literal test outcomes.
from pathlib import Path

import pytest
import yaml

from tldw_Server_API.tests.CI.test_merge_queue_gate_workflows import _evaluate

BASH_REQUIRED = pytest.mark.skipif(shutil.which("bash") is None, reason="requires Bash")


def _workflow(name: str) -> dict:
    return yaml.safe_load(Path(f".github/workflows/{name}.yml").read_text())


def test_main_pr_has_an_unfiltered_distinct_release_check():
    workflow = _workflow("ci")
    assert workflow[True]["pull_request"] == {"branches": ["main"]}  # nosec B101
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


@pytest.mark.parametrize(
    ("workflow", "job", "event", "target"),
    [
        ("publish-docker", "push_to_registries", "release", ""),
        ("publish-pypi", "rolling-image", "push", ""),
        ("publish-pypi", "publish-testpypi", "workflow_dispatch", "testpypi"),
        ("publish-pypi", "publish-pypi", "workflow_dispatch", "pypi"),
        ("publish-pypi", "publish-pypi", "push", ""),
    ],
)
def test_publication_requires_successful_prerequisites(workflow, job, event, target):
    job_config = _workflow(workflow)["jobs"][job]
    prerequisites = job_config["needs"]
    if isinstance(prerequisites, str):
        prerequisites = [prerequisites]
    successes = {f"needs.{dependency}.result": "success" for dependency in prerequisites}
    context = {
        **successes,
        "github.event_name": event,
        "inputs.target": target,
        "needs.detect-version.outputs.should_publish": "true",
        "needs.qualify.outputs.qualified": "true",
    }
    condition = job_config["if"].removeprefix("${{").removesuffix("}}").strip()
    assert _evaluate(condition, context)  # nosec B101
    assert not _evaluate(condition, context, cancelled=True)  # nosec B101
    for status in successes:
        assert not _evaluate(condition, {**context, status: "failure"}), status  # nosec B101


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


@pytest.mark.parametrize("tier", ("light", "unchanged", "nightly", "release"))
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
        tier in {"nightly", "release"}
    )
    for job in (
        "full-suite-linux-313-shards",
        "full-suite-macos-312-shards",
        "full-suite-windows-312-shards",
        "full-suite-os-313-release-shards",
        "full-suite-windows-313-release-shards",
    ):
        assert _evaluate(jobs[job]["if"], context) == (tier == "release"), job  # nosec B101
    package_condition = jobs["release-package"]["if"]
    assert _evaluate(package_condition, context) == (tier == "release")  # nosec B101
    assert not _evaluate(package_condition, context, cancelled=True)  # nosec B101
    context["needs.changes.result"] = "failure"
    assert not _evaluate(package_condition, context)  # nosec B101


@BASH_REQUIRED
@pytest.mark.parametrize(
    ("event", "ref", "nightly", "backend_changed", "expected"),
    [
        ("pull_request", "dev", "false", "true", "light"),
        ("pull_request", "dev", "false", "false", "unchanged"),
        ("pull_request", "dev", "false", "", None),
        ("pull_request", "main", "false", "false", "release"),
        ("workflow_run", "dev", "false", "true", "light"),
        ("workflow_dispatch", "dev", "true", "false", "nightly"),
        ("workflow_dispatch", "main", "true", "false", None),
        ("workflow_dispatch", "dev", "false", "false", "release"),
        ("push", "main", "false", "false", "release"),
        ("release", "v0.1.48", "false", "false", "release"),
    ],
)
def test_select_tier_from_event(tmp_path, event, ref, nightly, backend_changed, expected):
    changes = _workflow("ci")["jobs"]["changes"]
    step = next(step for step in changes["steps"] if step.get("id") == "tier")
    output = tmp_path / "output"
    proc = subprocess.run(  # nosec B603, B607 -- Exercise the checked-in routing step.
        ["bash", "-e", "-c", step["run"]],
        env={
            **os.environ,
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REF": "refs/pull/1/merge" if event == "pull_request" else f"refs/heads/{ref}",
            "GITHUB_EVENT_NAME": event,
            "DEV_PR": str(event == "pull_request" and ref == "dev").lower(),
            "NIGHTLY": nightly,
            "BACKEND_CHANGED": backend_changed,
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
        ("light", {"LICENSE": "false"}, False),
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

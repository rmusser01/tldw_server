"""Contract tests for the merge queue's dispatch path through the trusted license gate.

The queue rebases the front pull request with GITHUB_TOKEN, which starts no
`pull_request_target` run, so the rebased head would never get the required
`frontend-license-policy/trusted/dev` status. The queue therefore dispatches
`frontend-license-gate.yml` on `dev` with `pr=<number>` (merge queue design, section 4.4).

These tests pin the shape of that path, guard the two jobs' shared scripts against drift,
and execute the dispatch job's resolve step against a stubbed `gh`.

The subprocess import is intentional: the resolve step's validation is shell, and these
tests run that shell rather than re-implementing it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/frontend-license-gate.yml"
AUDIT_JOB = "frontend-license-gate-audit"
DISPATCH_JOB = "frontend-license-gate-dispatch"
RESOLVE_STEP = "Resolve pull request metadata"
TRUSTED_STEPS = (
    "Mark trusted policy pending",
    "Checkout trusted policy",
    "Evaluate immutable pull request metadata",
    "Publish trusted policy result",
)
# The audit job's env and steps as they were before the dispatch path existed. The
# pull_request_target path gates every pull request, so adding the dispatch path was not
# allowed to change it; neither is anything later, without re-pinning this on purpose.
AUDIT_ENV_AND_STEPS_SHA256 = "060bdcbac7837a75e000153d895152aa751523d9198cec6ee7ccb53fa22f0aca"
# What the shared scripts read that only the pull request can say. The audit job takes
# these from the event payload; the dispatch job's resolve step exports them.
RESOLVED_NAMES = {"PR_NUMBER", "PR_AUTHOR", "HEAD_SHA", "BASE_REF", "BASE_SHA", "STATUS_CONTEXT"}
EXPRESSION = re.compile(r"\$\{\{.*?\}\}")

REPOSITORY = "fixture/repository"
HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40
BASH = shutil.which("bash")
JQ = shutil.which("jq")
# Records every call, serves `gh api <path>` from a fixture file, and refuses anything
# else, so a test that sees only GET paths in the log has also seen that nothing was posted.
GH_STUB = """#!/bin/sh
printf '%s\\n' "$*" >> "${GH_CALLS}"
[ "$#" -eq 2 ] && [ "$1" = api ] || exit 64
fixture="${GH_FIXTURES}/$(printf '%s' "$2" | tr / _).json"
[ -f "${fixture}" ] || exit 1
cat "${fixture}"
"""


def load_workflow() -> dict[str, Any]:
    data = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def triggers(data: dict[str, Any]) -> dict[str, Any]:
    # `on` is a YAML 1.1 boolean, so PyYAML parses the key as True.
    return data.get("on", data.get(True))


def step_named(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def pull_request(**overrides: Any) -> dict[str, Any]:
    pull = {
        "number": 7,
        "state": "open",
        "base": {"ref": "dev"},
        "head": {"sha": HEAD_SHA},
        "user": {"login": "external-contributor"},
    }
    return {**pull, **overrides}


def branch_tip(branch: str = "dev", sha: Any = BASE_SHA) -> dict[str, Any]:
    return {"ref": f"refs/heads/{branch}", "object": {"type": "commit", "sha": sha}}


def run_resolve(
    tmp_path: Path,
    pr_input: str,
    pull: dict[str, Any] | None = None,
    tips: dict[str, dict[str, Any]] | None = None,
) -> tuple[subprocess.CompletedProcess[str], str, list[str]]:
    """Run the dispatch job's resolve step with `gh` replaced by fixture files.

    Args:
        tmp_path: Scratch directory for the stub, its fixtures and the GITHUB_ENV file.
        pr_input: The `pr` dispatch input, exactly as typed.
        pull: What `gh api repos/<repo>/pulls/7` returns; None makes that read fail.
        tips: What `gh api repos/<repo>/git/ref/heads/<branch>` returns, by branch.

    Returns:
        The finished process, what the step wrote to GITHUB_ENV, and the `gh` calls it made.
    """
    if BASH is None or JQ is None:
        pytest.skip("Requires bash and jq, which the runner image provides")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    api_root = f"repos_{REPOSITORY.replace('/', '_')}"
    if pull is not None:
        (fixtures / f"{api_root}_pulls_7.json").write_text(json.dumps(pull), encoding="utf-8")
    for branch, tip in (tips or {}).items():
        (fixtures / f"{api_root}_git_ref_heads_{branch}.json").write_text(json.dumps(tip), encoding="utf-8")
    stub_directory = tmp_path / "bin"
    stub_directory.mkdir()
    stub = stub_directory / "gh"
    stub.write_text(GH_STUB, encoding="utf-8")
    stub.chmod(0o700)
    github_env = tmp_path / "github-env"
    github_env.touch()
    calls = tmp_path / "gh-calls"
    calls.touch()
    environment = {
        "PATH": f"{stub_directory}{os.pathsep}{os.environ['PATH']}",
        "STATUS_REPOSITORY": REPOSITORY,
        "PR_INPUT": pr_input,
        "GITHUB_ENV": str(github_env),
        "GH_CALLS": str(calls),
        "GH_FIXTURES": str(fixtures),
    }
    script = step_named(load_workflow()["jobs"][DISPATCH_JOB], RESOLVE_STEP)["run"]
    # Fixed executable running the trusted workflow's own step; inputs are fixture data.
    result = subprocess.run(  # nosec B603
        [BASH, "-c", script], env=environment, capture_output=True, text=True, timeout=30
    )
    return result, github_env.read_text(encoding="utf-8"), calls.read_text(encoding="utf-8").splitlines()


def test_dispatch_trigger_takes_one_required_pr_string_beside_the_untouched_pr_trigger() -> None:
    declared = triggers(load_workflow())

    assert set(declared) == {"pull_request_target", "workflow_dispatch"}
    assert declared["pull_request_target"] == {
        "branches": ["main", "dev"],
        "types": ["opened", "reopened", "synchronize", "ready_for_review", "edited"],
    }
    assert set(declared["workflow_dispatch"]) == {"inputs"}
    assert set(declared["workflow_dispatch"]["inputs"]) == {"pr"}
    pr_input = dict(declared["workflow_dispatch"]["inputs"]["pr"])
    assert pr_input.pop("description")
    assert pr_input == {"required": True, "type": "string"}


def test_pull_request_target_job_gained_an_event_guard_and_nothing_else() -> None:
    job = load_workflow()["jobs"][AUDIT_JOB]
    encoded = json.dumps({"env": job["env"], "steps": job["steps"]}, sort_keys=True, separators=(",", ":")).encode()

    assert job["if"] == "github.event_name == 'pull_request_target'"
    assert set(job) == {"if", "runs-on", "timeout-minutes", "env", "steps"}
    assert [step["name"] for step in job["steps"]] == list(TRUSTED_STEPS)
    assert hashlib.sha256(encoded).hexdigest() == AUDIT_ENV_AND_STEPS_SHA256


def test_dispatch_job_runs_only_for_a_dispatch_on_dev_with_least_privilege() -> None:
    data = load_workflow()
    audit, dispatch = data["jobs"][AUDIT_JOB], data["jobs"][DISPATCH_JOB]

    assert set(data["jobs"]) == {AUDIT_JOB, DISPATCH_JOB}
    assert dispatch["if"] == "github.event_name == 'workflow_dispatch' && github.ref == 'refs/heads/dev'"
    assert set(dispatch) == {"if", "runs-on", "timeout-minutes", "permissions", "env", "steps"}
    assert dispatch["runs-on"] == audit["runs-on"]
    assert dispatch["timeout-minutes"] == audit["timeout-minutes"]
    assert dispatch["permissions"] == {"contents": "read", "pull-requests": "read", "statuses": "write"}
    # The audit job declares none of its own, so it keeps exactly the workflow's.
    assert data["permissions"] == {"contents": "read", "statuses": "write"}


def test_pr_input_reaches_the_shell_only_through_env() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    data = load_workflow()
    resolve = step_named(data["jobs"][DISPATCH_JOB], RESOLVE_STEP)
    script = resolve["run"]

    assert [expression for expression in EXPRESSION.findall(text) if "inputs" in expression] == [
        "${{ github.event.pull_request.number || inputs.pr }}",
        "${{ inputs.pr }}",
    ]
    assert resolve["env"] == {"PR_INPUT": "${{ inputs.pr }}"}
    for job_name, job in data["jobs"].items():
        for step in job["steps"]:
            assert "${{" not in step.get("run", ""), (job_name, step["name"])
    # Validated before it is sent anywhere. The behavioural tests below show the rest: a
    # rejected input makes no API call and is never echoed back.
    assert "readonly number_pattern='^[1-9][0-9]*$'" in script
    assert script.index('[[ "${PR_INPUT}" =~ ${number_pattern} ]] || fail') < script.index("gh api")


def test_dispatch_job_repeats_the_trusted_steps_verbatim() -> None:
    jobs = load_workflow()["jobs"]
    audit, dispatch = jobs[AUDIT_JOB], jobs[DISPATCH_JOB]

    assert [step["name"] for step in dispatch["steps"]] == [RESOLVE_STEP, *TRUSTED_STEPS]
    for name in ("Evaluate immutable pull request metadata", "Publish trusted policy result"):
        assert step_named(dispatch, name)["run"] == step_named(audit, name)["run"], name
    # Not only the two scripts: the pending step, the pinned shallow checkout of
    # github.sha, the evaluate step's id and the publisher's `if` and env are shared too.
    assert dispatch["steps"][1:] == audit["steps"]


def test_dispatch_job_supplies_every_variable_the_shared_scripts_read() -> None:
    jobs = load_workflow()["jobs"]
    audit, dispatch = jobs[AUDIT_JOB], jobs[DISPATCH_JOB]
    resolve = step_named(dispatch, RESOLVE_STEP)
    exported = re.findall(r"printf '([A-Z_]+)=", resolve["run"])

    assert set(resolve) == {"name", "shell", "env", "run"}
    assert resolve["shell"] == "bash"
    assert sorted(exported) == sorted(RESOLVED_NAMES)
    # Static values come from the same expressions as the audit job's. Nothing the resolve
    # step exports is also set on the job, so there is no precedence to reason about.
    assert dispatch["env"] == {name: value for name, value in audit["env"].items() if name not in RESOLVED_NAMES}
    assert set(dispatch["env"]) | RESOLVED_NAMES == set(audit["env"])
    # One write, as the script's last command: nothing is exported before every check passed.
    assert resolve["run"].count('"${GITHUB_ENV}"') == 1
    assert resolve["run"].rstrip().endswith('} >> "${GITHUB_ENV}"')
    assert "GITHUB_OUTPUT" not in resolve["run"]


def test_shared_evaluate_script_runs_on_what_the_dispatch_job_provides(tmp_path: Path) -> None:
    """The evaluate script is the audit job's text; here it has only the dispatch job's inputs.

    It runs under `set -u` with the job's static env plus what the resolve step exported and
    nothing else, so a variable the script reads that this job does not supply fails here.
    The owner path is used because it needs no network: it stops before any fetch.
    """
    if BASH is None:
        pytest.skip("Requires bash")
    # Literal shell version probe.
    bash_major = subprocess.check_output(  # nosec B603
        [BASH, "-c", "printf '%s' \"${BASH_VERSINFO[0]}\""], text=True
    )
    if int(bash_major) < 4:
        pytest.skip("Requires the runner's Bash >=4; Bash 3 has no ${VAR,,} and ignores errexit for [[ guards ]]")
    dispatch = load_workflow()["jobs"][DISPATCH_JOB]
    # Everything on the job except the API token, which the evaluate script never reads.
    static = {"STATUS_REPOSITORY": REPOSITORY, "REPOSITORY_OWNER": "repository-owner"}
    assert set(static) | {"GH_TOKEN"} == set(dispatch["env"])
    pull = pull_request(user={"login": "Repository-Owner"})

    resolved, exported, _ = run_resolve(tmp_path, "7", pull, {"dev": branch_tip()})
    assert resolved.returncode == 0, resolved.stdout + resolved.stderr
    output = tmp_path / "github-output"
    environment = {
        "PATH": os.environ["PATH"],
        **static,
        **dict(line.split("=", 1) for line in exported.splitlines()),
        "GITHUB_OUTPUT": str(output),
    }
    # Trusted workflow script and the repository's own classifier; inputs are fixture data.
    evaluated = subprocess.run(  # nosec B603
        [BASH, "-c", step_named(dispatch, "Evaluate immutable pull request metadata")["run"]],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert evaluated.returncode == 0, evaluated.stderr
    assert output.read_text(encoding="utf-8") == "verdict=success\n"


def test_concurrency_group_keys_a_dispatch_on_its_pull_request() -> None:
    assert load_workflow()["concurrency"] == {
        "group": "frontend-license-gate-${{ github.event.pull_request.number || inputs.pr }}",
        "cancel-in-progress": True,
    }


@pytest.mark.parametrize(
    ("base_ref", "author"),
    [
        ("dev", "external-contributor"),
        ("main", "RMusser01"),
        ("dev", "dependabot[bot]"),
        ("dev", "a"),
        ("dev", "x" * 39),
    ],
)
def test_resolve_exports_validated_metadata_for_an_open_pull_request(
    tmp_path: Path, base_ref: str, author: str
) -> None:
    pull = pull_request(base={"ref": base_ref}, user={"login": author})

    result, exported, calls = run_resolve(tmp_path, "7", pull, {base_ref: branch_tip(base_ref)})

    assert result.returncode == 0, result.stdout + result.stderr
    assert exported == (
        "PR_NUMBER=7\n"
        f"PR_AUTHOR={author}\n"
        f"HEAD_SHA={HEAD_SHA}\n"
        f"BASE_REF={base_ref}\n"
        f"BASE_SHA={BASE_SHA}\n"
        f"STATUS_CONTEXT=frontend-license-policy/trusted/{base_ref}\n"
    )
    assert calls == [f"api repos/{REPOSITORY}/pulls/7", f"api repos/{REPOSITORY}/git/ref/heads/{base_ref}"]


def test_resolve_takes_the_base_sha_from_the_branch_tip_not_the_pull_request(tmp_path: Path) -> None:
    pull = pull_request(base={"ref": "dev", "sha": "c" * 40})

    result, exported, _ = run_resolve(tmp_path, "7", pull, {"dev": branch_tip()})

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"BASE_SHA={BASE_SHA}\n" in exported
    assert "c" * 40 not in exported


@pytest.mark.parametrize(
    "pr_input",
    ["", "0", "07", "7 ", " 7", "-7", "+7", "7.0", "1e3", "seven", "#7", "7;id", "$(id)", "7/../../issues/7"]
    + ["7\n8", "7\n"],
)
def test_resolve_rejects_a_non_numeric_pr_before_any_api_call(tmp_path: Path, pr_input: str) -> None:
    result, exported, calls = run_resolve(tmp_path, pr_input, pull_request(), {"dev": branch_tip()})

    assert result.returncode != 0
    assert exported == ""
    assert calls == []
    assert result.stdout == "::error::The pr input is not a pull request number.\n"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"number": 8}, "GitHub returned a different pull request."),
        ({"number": None}, "GitHub returned a different pull request."),
        ({"state": "closed"}, "The pull request is not open."),
        ({"state": None}, "The pull request is not open."),
        ({"base": {"ref": "feature"}}, "The pull request base is not dev or main."),
        ({"base": {"ref": "refs/heads/dev"}}, "The pull request base is not dev or main."),
        ({"base": {"ref": "dev\nBASH_ENV=/tmp/payload"}}, "The pull request base is not dev or main."),
        ({"base": None}, "The pull request base is not dev or main."),
        ({"head": {"sha": "A" * 40}}, "The pull request head is not a commit SHA."),
        ({"head": {"sha": "a" * 39}}, "The pull request head is not a commit SHA."),
        ({"head": {"sha": "a" * 41}}, "The pull request head is not a commit SHA."),
        ({"head": {"sha": "a" * 40 + "\nBASH_ENV=/tmp/payload"}}, "The pull request head is not a commit SHA."),
        ({"head": {"sha": None}}, "The pull request head is not a commit SHA."),
        ({"user": None}, "The pull request author is not a well-formed login."),
        ({"user": {"login": ""}}, "The pull request author is not a well-formed login."),
        ({"user": {"login": "-rf"}}, "The pull request author is not a well-formed login."),
        ({"user": {"login": "two words"}}, "The pull request author is not a well-formed login."),
        ({"user": {"login": "x" * 40}}, "The pull request author is not a well-formed login."),
        ({"user": {"login": "app[bot]x"}}, "The pull request author is not a well-formed login."),
        ({"user": {"login": "[bot]"}}, "The pull request author is not a well-formed login."),
        ({"user": {"login": "eve\nBASH_ENV=/tmp/payload"}}, "The pull request author is not a well-formed login."),
        ({"user": {"login": "eve<<EOF"}}, "The pull request author is not a well-formed login."),
    ],
)
def test_resolve_rejects_an_unacceptable_pull_request_and_exports_nothing(
    tmp_path: Path, overrides: dict[str, Any], message: str
) -> None:
    result, exported, calls = run_resolve(tmp_path, "7", pull_request(**overrides), {"dev": branch_tip()})

    assert result.returncode != 0
    assert exported == ""
    assert calls == [f"api repos/{REPOSITORY}/pulls/7"]
    assert result.stdout == f"::error::{message}\n"


@pytest.mark.parametrize(
    ("tip", "message"),
    [
        (branch_tip(sha="B" * 40), "The base branch tip is not a commit SHA."),
        (branch_tip(sha="b" * 40 + "\nBASH_ENV=/tmp/payload"), "The base branch tip is not a commit SHA."),
        (branch_tip(sha=None), "The base branch tip is not a commit SHA."),
        (branch_tip("dev-old"), "GitHub returned a different base branch."),
        ([branch_tip("dev"), branch_tip("dev-old")], None),
    ],
)
def test_resolve_rejects_a_malformed_base_branch_tip(tmp_path: Path, tip: Any, message: str | None) -> None:
    result, exported, calls = run_resolve(tmp_path, "7", pull_request(), {"dev": tip})

    assert result.returncode != 0
    assert exported == ""
    assert calls == [f"api repos/{REPOSITORY}/pulls/7", f"api repos/{REPOSITORY}/git/ref/heads/dev"]
    if message is not None:
        assert result.stdout == f"::error::{message}\n"


@pytest.mark.parametrize("missing", ["pull", "tip"])
def test_resolve_fails_closed_when_an_api_read_fails(tmp_path: Path, missing: str) -> None:
    pull = None if missing == "pull" else pull_request()

    result, exported, calls = run_resolve(tmp_path, "7", pull, {})

    assert result.returncode != 0
    assert exported == ""
    assert calls[0] == f"api repos/{REPOSITORY}/pulls/7"
    assert len(calls) == (1 if missing == "pull" else 2)

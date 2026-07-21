from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/frontend-license-gate.yml"
ACTIONLINT_PATH = REPO_ROOT / ".github/workflows/actionlint.yml"
CHECKOUT_ACTION = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
JOB_ID = "frontend-license-gate-audit"
STATUS_CONTEXT = "frontend-license-policy/trusted"
CLASSIFIER = "Helper_Scripts/ci/check_frontend_license_gate.py"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_workflow_uses_the_base_controlled_trigger_and_minimum_permissions() -> None:
    data = load_yaml(WORKFLOW_PATH)
    triggers = data.get("on", data.get(True))

    assert set(triggers) == {"pull_request_target"}
    assert triggers["pull_request_target"]["branches"] == ["main", "dev"]
    assert data["permissions"] == {"contents": "read", "statuses": "write"}
    assert set(data["jobs"]) == {JOB_ID}


def test_workflow_checks_out_only_the_trusted_base_revision() -> None:
    job = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]
    steps = job["steps"]
    action_steps = [step for step in steps if "uses" in step]

    assert action_steps == [
        {
            "name": "Checkout trusted policy",
            "uses": CHECKOUT_ACTION,
            "with": {
                "ref": "${{ github.sha }}",
                "fetch-depth": 0,
                "persist-credentials": False,
            },
        }
    ]

    run_scripts = "\n".join(step.get("run", "") for step in steps)
    forbidden_commands = (
        "git checkout",
        "git switch",
        "git reset",
        "git restore",
        "git worktree",
        "git show",
        "pip install",
        "python -m pip",
        "npm install",
        "pnpm install",
        "yarn install",
        "bun install",
        "uv sync",
        "poetry install",
        "gh run download",
    )
    assert not any(command in run_scripts for command in forbidden_commands)
    assert "artifact" not in run_scripts.casefold()
    assert not any(str(step.get("uses", "")).startswith("./") for step in steps)
    assert not any("cache" in str(step.get("uses", "")).casefold() for step in steps)
    assert not any("artifact" in str(step.get("uses", "")).casefold() for step in steps)


def test_workflow_posts_pending_before_a_fail_closed_evaluation() -> None:
    job = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]
    steps = job["steps"]
    pending_index = next(index for index, step in enumerate(steps) if step.get("name") == "Mark trusted policy pending")
    checkout_index = next(index for index, step in enumerate(steps) if step.get("name") == "Checkout trusted policy")
    evaluate_index = next(index for index, step in enumerate(steps) if step.get("id") == "evaluate")
    pending_script = steps[pending_index]["run"]

    assert pending_index < checkout_index < evaluate_index
    assert '"repos/${STATUS_REPOSITORY}/statuses/${HEAD_SHA}"' in pending_script
    assert "-f state=pending" in pending_script
    assert '-f context="${STATUS_CONTEXT}"' in pending_script
    assert job["env"]["STATUS_CONTEXT"] == STATUS_CONTEXT
    assert JOB_ID != STATUS_CONTEXT

    evaluate_script = steps[evaluate_index]["run"]
    normalized = " ".join(evaluate_script.replace("\\\n", " ").split())
    assert "verdict=failure" in evaluate_script
    assert "trap emit_verdict EXIT" in evaluate_script
    assert '[[ "${PR_NUMBER}" =~ ^[1-9][0-9]*$ ]]' in evaluate_script
    assert '[[ "${BASE_SHA}" =~ ^[0-9a-f]{40}$ ]]' in evaluate_script
    assert '[[ "${HEAD_SHA}" =~ ^[0-9a-f]{40}$ ]]' in evaluate_script
    assert '[[ "${BASE_REF}" == main || "${BASE_REF}" == dev ]]' in evaluate_script
    assert (
        'git fetch --no-tags --depth=1 "${public_remote}" ' '"+refs/heads/${BASE_REF}:refs/remotes/license-gate/base"'
    ) in normalized
    assert (
        'git fetch --no-tags --depth=1 "${public_remote}" '
        '"+refs/pull/${PR_NUMBER}/head:refs/remotes/license-gate/pr-head"'
    ) in normalized
    assert '[[ "${fetched_base}" == "${BASE_SHA}" ]]' in evaluate_script
    assert '[[ "${fetched_head}" == "${HEAD_SHA}" ]]' in evaluate_script

    diff_match = re.search(r"git --no-pager diff (?P<arguments>.*?) \| python3", normalized)
    assert diff_match is not None
    diff_arguments = diff_match.group("arguments")
    for argument in ("--name-only", "-z", "--no-renames", "--no-ext-diff", "--no-textconv"):
        assert argument in diff_arguments.split()
    assert diff_arguments.endswith('"${BASE_SHA}" "${HEAD_SHA}" --')
    assert normalized.count(f"python3 {CLASSIFIER}") == 2
    assert normalized.count("--null") == 2


def test_workflow_publishes_success_only_for_an_explicit_success_verdict() -> None:
    steps = load_yaml(WORKFLOW_PATH)["jobs"][JOB_ID]["steps"]
    publisher = next(step for step in steps if step.get("name") == "Publish trusted policy result")
    script = publisher["run"]

    assert publisher["if"] == "always()"
    assert publisher["env"] == {"VERDICT": "${{ steps.evaluate.outputs.verdict }}"}
    assert "state=failure" in script
    assert script.count("state=success") == 1
    assert re.search(r'if \[\[ "\$\{VERDICT\}" == success \]\]; then\s+state=success', script)
    assert '-f context="${STATUS_CONTEXT}"' in script
    assert '[[ "${state}" == success ]]' in script


def test_actionlint_targets_the_trusted_workflow() -> None:
    steps = load_yaml(ACTIONLINT_PATH)["jobs"]["actionlint"]["steps"]
    invocation = next(step for step in steps if step.get("name") == "Run actionlint on targeted workflows")["run"]

    assert ".github/workflows/frontend-license-gate.yml" in invocation.split()

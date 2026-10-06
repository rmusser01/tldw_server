"""Merge-queue contract of frontend-required.yml (spec sections 4.6, 4.8 and 4.9).

Spec: Docs/superpowers/specs/2026-10-04-merge-queue-design.md
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from Helper_Scripts.ci import merge_queue as mq

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "frontend-required.yml"
CHECKOUT_ACTION = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
GATE = "frontend-required"
TICK = "queue-tick"
CHANGES_GUARD = "Require change detection success"
LICENSE_GUARD = "Require the license audit to have passed"
SHARD_GUARD = "Require frontend unit shard success"


def _wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _jobs() -> dict:
    return _wf()["jobs"]


def _normalized(expression: object) -> str:
    return re.sub(r"\s+", "", str(expression))


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _run_step(step: dict, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 B607 - runs the workflow's own step script
        ["bash", "-c", step["run"]],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


# --- 4.8: the manual-dispatch name guard ----------------------------------------------------


def test_name_guard_exempts_only_the_queue_actor():
    """A dispatch publishes the protected name only when the queue's own token made it.

    Three rows: pull_request and workflow_run publish `frontend-required`; a dispatch by
    `github-actions[bot]` (GITHUB_TOKEN, how the queue starts the gate) publishes it too; every
    other dispatch publishes `frontend-required-diagnostic`. The rows are evaluated in
    test_admin_ui_vitest_ratchet_workflow.py; this pins the expression to the queue's actor.
    """
    assert mq.QUEUE_ACTOR == "github-actions[bot]"
    assert _jobs()[GATE]["name"] == (
        "${{ github.event_name == 'workflow_dispatch' && "
        f"github.actor != '{mq.QUEUE_ACTOR}' && "
        "'frontend-required-diagnostic' || 'frontend-required' }}"
    )


def test_nothing_else_in_the_workflow_depends_on_who_dispatched_it():
    """Apart from the job name, a queue dispatch takes the same path as any other dispatch,
    which for the same head and `base_sha` tests what the pull_request run tests."""
    rendered = json.dumps(_wf())  # parsed values only: comments are not behaviour
    assert rendered.count("github.actor") == 1
    assert rendered.count("frontend-required-diagnostic") == 1
    assert "triggering_actor" not in rendered


# --- 4.9: a gate whose change detection failed must be red -----------------------------------


def test_gate_job_is_not_skipped_when_change_detection_did_not_succeed():
    """GitHub counts a skipped required job as satisfied, so `changes` must stay out of the `if`.

    The gate runs whenever `changes` was started, so it always has a result to judge, and also
    when the license wait did not pass, so that case goes red too (TASK-13502).
    """
    jobs = _jobs()
    gate_if = _normalized(jobs[GATE]["if"])
    changes_if = _normalized(jobs["changes"]["if"])
    assert "needs.changes" not in gate_if
    license_arm = _normalized(
        "(github.event_name != 'workflow_run' && needs.await_license.result != 'skipped' && "
        "needs.await_license.outputs.license_passed != 'true')"
    )
    assert changes_if.endswith("))")
    assert gate_if == changes_if[:-1] + "||" + license_arm + ")"
    assert gate_if.startswith("always()&&!cancelled()&&")
    assert "changes" in _needs(jobs[GATE])
    assert "continue-on-error" not in jobs[GATE]


def _changes_guard() -> dict:
    steps = _jobs()[GATE]["steps"]
    names = [step.get("name") for step in steps]
    # Only the license refusal comes before it: when that fails, the guard is skipped with the rest.
    assert names.index(CHANGES_GUARD) == 1 and names[0] == LICENSE_GUARD
    return steps[1]


def test_gate_first_step_refuses_a_license_wait_that_did_not_pass():
    refusal = _jobs()[GATE]["steps"][0]
    assert refusal["name"] == LICENSE_GUARD
    assert _normalized(refusal["if"]) == _normalized(
        "github.event_name != 'workflow_run' && needs.await_license.result != 'skipped' && "
        "needs.await_license.outputs.license_passed != 'true'"
    )
    assert "exit 1" in refusal["run"] and "::error::" in refusal["run"]


def test_gate_change_guard_judges_the_changes_result():
    step = _changes_guard()
    assert step["name"] == CHANGES_GUARD
    assert step["env"] == {"CHANGES_RESULT": "${{ needs.changes.result }}"}
    assert step["shell"] == "bash"
    assert "if" not in step, "the judgement happens in the script, for every run of the job"
    assert "continue-on-error" not in step


@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped", ""])
def test_gate_change_guard_fails_when_change_detection_did_not_succeed(result: str):
    done = _run_step(_changes_guard(), CHANGES_RESULT=result)
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stdout.startswith("::error::"), done.stdout
    assert f"changes job result: {result})" in done.stdout


def test_gate_change_guard_passes_when_change_detection_succeeded():
    done = _run_step(_changes_guard(), CHANGES_RESULT="success")
    assert (done.returncode, done.stdout, done.stderr) == (0, "", "")


def test_gate_ends_red_when_failed_change_detection_skipped_the_unit_shards():
    """`frontend-unit-tests` needs a successful `changes`, so a failed detection skips it; the
    gate must not read "skipped" as "nothing to test"."""
    jobs = _jobs()
    assert "needs.changes.result == 'success'" in jobs["frontend-unit-tests"]["if"]
    steps = jobs[GATE]["steps"]
    names = [step.get("name") for step in steps]
    assert names.index(CHANGES_GUARD) == 1 < names.index(SHARD_GUARD)
    # Second line of defence: the shard guard also refuses the empty outputs of a failed `changes`.
    done = _run_step(steps[names.index(SHARD_GUARD)], TLDW_FRONTEND_CHANGED="", UNIT_SHARDS_RESULT="skipped")
    assert done.returncode == 1, done.stdout + done.stderr
    # Nothing later in the job runs once a guard failed, unless it is itself gated on an output
    # that a failed `changes` leaves empty.
    for step in steps[2:]:
        condition = str(step.get("if", ""))
        if "always()" in condition or "failure()" in condition:
            assert "needs.changes.outputs." in condition, step.get("name")


# --- 4.6: the failure-only queue-tick ---------------------------------------------------------


def test_queue_tick_runs_only_after_a_failed_gate_with_the_queue_enabled():
    """With MERGE_QUEUE unset every conjunct after it is moot: the job is skipped on every run."""
    tick = _jobs()[TICK]
    assert tick["needs"] == [GATE]
    assert _normalized(tick["if"]) == _normalized(
        """
        always() && !cancelled() &&
        needs.frontend-required.result == 'failure' &&
        (vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on') &&
        (github.event_name == 'workflow_dispatch' ||
         (github.event_name == 'pull_request' &&
          github.event.pull_request.head.repo.full_name == github.repository))
        """
    )


def test_queue_tick_is_the_queue_job_with_a_shorter_budget():
    """Same permissions and steps as merge-queue.yml's `queue` job: dev's script, never the PR's."""
    tick = _jobs()[TICK]
    queue = yaml.safe_load((WORKFLOWS / "merge-queue.yml").read_text(encoding="utf-8"))["jobs"]["queue"]
    assert set(tick) == {"name", "needs", "if", "runs-on", "timeout-minutes", "permissions", "steps"}
    assert tick["name"] == "Merge queue tick"
    assert tick["runs-on"] == "ubuntu-latest"
    assert tick["timeout-minutes"] == 10
    assert tick["permissions"] == queue["permissions"] == {
        "contents": "write", "pull-requests": "write", "actions": "write",
        "checks": "read", "statuses": "read",
    }
    assert tick["steps"] == queue["steps"]
    checkout, advance = tick["steps"]
    assert checkout["uses"] == CHECKOUT_ACTION
    assert checkout["with"] == {"ref": "dev", "persist-credentials": False}
    assert advance["run"] == "python3 -m Helper_Scripts.ci.merge_queue"
    assert advance["env"] == {"GH_TOKEN": "${{ github.token }}", "MERGE_QUEUE": "${{ vars.MERGE_QUEUE }}"}


def test_queue_tick_checkout_is_pinned_like_every_other_checkout_in_the_workflow():
    pins = {
        step["uses"]
        for job in _jobs().values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("actions/checkout@")
    }
    assert pins == {CHECKOUT_ACTION}


def test_queue_tick_is_outside_every_needs_and_holds_the_only_write_scopes():
    """Outside every `needs` the tick can never delay the required check or turn it red."""
    wf = _wf()
    assert wf["permissions"] == {"contents": "read"}
    for name, job in wf["jobs"].items():
        assert TICK not in _needs(job), name
        if name != TICK:
            assert "write" not in job.get("permissions", {}).values(), name

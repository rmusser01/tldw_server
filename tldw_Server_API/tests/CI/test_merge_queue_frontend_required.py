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


def test_every_dispatch_publishes_the_diagnostic_name():
    """D2: no dispatch publishes the protected name, whoever started it; a pull_request run does."""
    name = _jobs()[GATE]["name"]
    assert _normalized(name) == _normalized(
        "${{ github.event_name == 'workflow_dispatch' && 'frontend-required-diagnostic' || 'frontend-required' }}"
    )
    assert "github.actor" not in WORKFLOW.read_text(encoding="utf-8")


def test_nothing_else_in_the_workflow_depends_on_who_dispatched_it():
    """Nothing in the workflow depends on who dispatched it: every dispatch publishes the diagnostic name."""
    rendered = json.dumps(_wf())  # parsed values only: comments are not behaviour
    assert rendered.count("github.actor") == 0
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


def test_queue_tick_holds_only_actions_write_checks_nothing_out_and_nothing_waits_on_it():
    jobs = _jobs()
    assert jobs[TICK]["permissions"] == {"actions": "write"}
    assert not any("uses" in step for step in jobs[TICK]["steps"])
    assert all(TICK not in _needs(job) for job in jobs.values())

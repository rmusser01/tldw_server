"""What the merge queue needs from the required gate workflows (spec sections 4.3, 4.6 and 4.9).

Spec: Docs/superpowers/specs/2026-10-04-merge-queue-design.md. `frontend-required.yml` and
`frontend-license-gate.yml` have their own tests.
"""

from __future__ import annotations

import os
import re
import subprocess
from itertools import product
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Workflow file -> id of the job whose `name` is the required context.
GATES = {
    "backend-required.yml": "backend-required",
    "security-required.yml": "security-required",
    "coverage-required.yml": "coverage-required",
    "e2e-required.yml": "e2e-required",
    "container-build-check.yml": "container-build-check",
}
# The gates with a `changes` job. container-build-check builds every image on every run.
CHANGE_GATED = tuple(name for name in GATES if name != "container-build-check.yml")

LICENSE_STEP = "Require the license audit to have passed"
CHANGES_STEP = "Require change detection to have succeeded"


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def _wf(name: str) -> dict:
    return yaml.safe_load(_text(name))


def _triggers(name: str) -> dict:
    wf = _wf(name)
    return wf.get("on", wf.get(True))  # `on` is a YAML 1.1 boolean


def _gate(name: str) -> dict:
    return _wf(name)["jobs"][GATES[name]]


def _step(job: dict, step_name: str) -> dict:
    (step,) = [s for s in job["steps"] if s.get("name") == step_name]
    return step


_TOKEN = re.compile(r"'[^']*'|&&|\|\||[=!]=|[!()]|[A-Za-z_][\w.-]*(?:\(\))?")


def _evaluate(expression: str, values: dict[str, str], cancelled: bool = False) -> bool:
    """Evaluate the part of the Actions expression language these `if:` conditions use.

    Covers `!`, `==`, `!=`, `&&`, `||`, parentheses, string literals, `always()`, `cancelled()`
    and context names, with GitHub's precedence. A name missing from `values` is a KeyError,
    so a condition cannot quietly read something the test did not set.
    """
    tokens = _TOKEN.findall(expression)
    assert "".join(tokens) == "".join(expression.split()), f"unsupported syntax in: {expression}"
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> str:
        nonlocal position
        position += 1
        return tokens[position - 1]

    def operand() -> object:
        token = take()
        if token == "(":
            value = either()
            assert take() == ")"
            return value
        if token == "!":
            return not operand()
        if token == "always()":
            return True
        if token == "cancelled()":
            return cancelled
        return token[1:-1] if token.startswith("'") else values[token]

    def comparison() -> object:
        left = operand()
        if peek() in ("==", "!="):
            return (take() == "==") == (left == operand())
        return left

    def both() -> object:
        value = comparison()
        while peek() == "&&":
            take()
            right = comparison()
            value = value and right
        return value

    def either() -> object:
        value = both()
        while peek() == "||":
            take()
            right = both()
            value = value or right
        return value

    result = either()
    assert position == len(tokens), f"trailing tokens in: {expression}"
    return bool(result)


# --- 4.3: the comparison base on a dispatched run ---------------------------------------------


@pytest.mark.parametrize("name", CHANGE_GATED)
def test_gate_declares_an_optional_base_sha_dispatch_input(name):
    """Without the input a dispatch compares against HEAD^ and can no-op with nothing tested."""
    inputs = _triggers(name)["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"base_sha"}
    assert inputs["base_sha"]["type"] == "string"
    assert inputs["base_sha"]["required"] is False


def test_container_build_check_declares_no_dispatch_input():
    """The queue dispatches it with no inputs: it has no change detection to feed."""
    assert not (_triggers("container-build-check.yml")["workflow_dispatch"] or {}).get("inputs")


@pytest.mark.parametrize("name", GATES)
def test_no_dispatch_input_is_interpolated_into_a_shell_script(name):
    """An input reaches a script through `env:`, never as text spliced into the script."""
    for job_id, job in _wf(name)["jobs"].items():
        for step in job.get("steps", []):
            assert "inputs." not in step.get("run", ""), (name, job_id, step.get("name"))


@pytest.mark.parametrize(
    ("dispatch_base", "event_base", "expected"),
    [
        ("a" * 40, "", "a" * 40),  # queue dispatch: dev's tip
        ("", "b" * 40, "b" * 40),  # pull_request and workflow_run resolve what they always did
        ("", "", ""),  # hand dispatch without a base: falls through to the HEAD^ fallback
    ],
)
def test_backend_type_check_prefers_the_dispatch_base(dispatch_base, event_base, expected):
    """Run the step's own base resolution, up to its HEAD^ fallback."""
    step = _step(_gate("backend-required.yml"), "Type check changed backend modules")
    assert step["env"] == {"DISPATCH_BASE_SHA": "${{ inputs.base_sha }}"}
    resolution, fallback = step["run"].split('if [[ -z "${BASE_SHA:-}"', 1)
    assert "git rev-parse HEAD^" in fallback
    script = re.sub(r"\$\{\{[^}]*\}\}", event_base, resolution) + '\nprintf %s "$BASE_SHA"'
    result = subprocess.run(  # the workflow's own script, in a scratch shell
        ["bash", "-c", script],
        env={**os.environ, "DISPATCH_BASE_SHA": dispatch_base},
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert result.stdout == expected


def test_backlog_ratchet_base_prefers_the_dispatch_base():
    step = _step(_gate("backend-required.yml"), "Enforce CI contracts and code ratchets")
    assert step["env"]["BACKLOG_TASK_FORMAT_BASE"] == (
        "${{ inputs.base_sha || needs.admission.outputs.base_sha || "
        "github.event.pull_request.base.sha || github.event.before }}"
    )


def test_dependency_review_runs_on_a_queue_dispatch():
    """It used to admit only pull_request and workflow_run, so a dispatched gate passed without it."""
    step = _step(_gate("security-required.yml"), "Dependency review (high/critical)")
    for event, base, runs in [
        ("pull_request", "", True),
        ("workflow_run", "", True),
        ("workflow_dispatch", "a" * 40, True),
        ("workflow_dispatch", "", False),  # no base to compare against
        ("push", "", False),
    ]:
        assert _evaluate(step["if"], {"github.event_name": event, "inputs.base_sha": base}) is runs, (event, base)
    assert step["with"] == {
        "base-ref": (
            "${{ inputs.base_sha || needs.admission.outputs.base_sha || github.event.pull_request.base.sha }}"
        ),
        # On a dispatch the first two are empty and github.sha is the dispatched branch's head.
        "head-ref": (
            "${{ needs.admission.outputs.head_sha || github.event.pull_request.head.sha || github.sha }}"
        ),
        "fail-on-severity": "high",
    }
    assert "continue-on-error" not in step


@pytest.mark.parametrize("name", ["coverage-required.yml", "e2e-required.yml", "container-build-check.yml"])
def test_gate_resolves_no_comparison_base_of_its_own(name):
    """These reach a base only through detect-required-gate-changes, which reads the input.

    A step added here that resolves its own base must take `inputs.base_sha` first (spec 4.3).
    """
    text = _text(name)
    for marker in ("github.event.pull_request.base", "github.event.before", "HEAD^"):
        assert marker not in text, marker


# --- 4.9: a gate whose change detection failed is red -----------------------------------------


class Row(NamedTuple):
    """One combination of what a gate job can see when GitHub evaluates its `if`."""

    event: str
    admission: str
    should_run: str
    awaited: str
    license_passed: str
    changes: str
    cancelled: bool

    @property
    def admitted(self) -> bool:
        """Whether `changes` was let through, by admission or by the license verdict."""
        if self.event == "workflow_run":
            return self.admission == "success" and self.should_run == "true"
        return self.awaited == "skipped" or self.license_passed == "true"

    @property
    def license_negative(self) -> bool:
        return self.event != "workflow_run" and self.awaited != "skipped" and self.license_passed != "true"

    @property
    def values(self) -> dict[str, str]:
        return {
            "github.event_name": self.event,
            "needs.admission.result": self.admission,
            "needs.admission.outputs.should_run": self.should_run,
            "needs.await_license.result": self.awaited,
            "needs.await_license.outputs.license_passed": self.license_passed,
            "needs.changes.result": self.changes,
        }


def _rows(reachable_only: bool = True) -> list[Row]:
    """Every (event, admission, license verdict, changes result, cancelled) combination.

    `admission` runs only on workflow_run and `await_license` only on pull_request. `changes`
    carries the same admission condition as the gates, so it is skipped exactly when it was not
    admitted; `reachable_only=False` adds the other combinations too.
    """
    ran = [("success", "true"), ("success", "false"), ("failure", "")]
    skipped = [("skipped", "")]
    rows = []
    for event in ("pull_request", "workflow_dispatch", "workflow_run"):
        admissions = ran + skipped if event == "workflow_run" else skipped
        verdicts = ran if event == "pull_request" else skipped
        for (admission, should_run), (awaited, passed), changes, cancelled in product(
            admissions, verdicts, ("success", "failure", "cancelled", "skipped"), (False, True)
        ):
            row = Row(event, admission, should_run, awaited, passed, changes, cancelled)
            if not reachable_only or (changes == "skipped") != row.admitted:
                rows.append(row)
    return rows


def _outcome(job: dict, row: Row) -> str:
    """What the gate job does for a row: `skipped`, red by `license`, red by `changes`, or `proceeds`."""
    if not _evaluate(job["if"], row.values, row.cancelled):
        return "skipped"
    for step in job["steps"]:  # a step without a status function runs only while none has failed
        if step.get("name") == LICENSE_STEP and _evaluate(step["if"], row.values):
            return "license"
        if step.get("name") == CHANGES_STEP and _evaluate(step["if"], row.values):
            return "changes"
    return "proceeds"


def _expected_outcome(name: str, row: Row) -> str:
    if row.cancelled:
        return "skipped"  # a cancelled run stays cancelled
    if name == "backend-required.yml" and row.license_negative:
        return "license"  # the one gate that turns a negative license verdict red
    if not row.admitted:
        return "skipped"  # e.g. workflow_run with admission declined: skipped by design
    return "proceeds" if row.changes == "success" else "changes"


@pytest.mark.parametrize("name", CHANGE_GATED)
def test_gate_is_red_when_change_detection_did_not_succeed(name):
    """A skipped required job counts as satisfied, so the gate must run and fail instead.

    Checked over the whole truth table: the gate does what it did before on every row except
    those where `changes` ran and did not succeed, which are now red.
    """
    gate = _gate(name)
    newly_red = 0
    for row in _rows():
        assert _outcome(gate, row) == _expected_outcome(name, row), row
        newly_red += _outcome(gate, row) == "changes"
    assert newly_red, "no row exercised a failed change detection"


@pytest.mark.parametrize("name", ["backend-required.yml", "security-required.yml"])
def test_gate_that_sees_admission_is_red_if_changes_was_skipped_when_it_should_have_run(name):
    """These two gates need admission and await_license themselves, so they can tell "skipped by
    design" from "skipped although admitted" (which the `changes` condition should make impossible)."""
    gate = _gate(name)
    assert {"changes", "admission", "await_license"} <= set(gate["needs"])
    rows = [r for r in _rows(reachable_only=False) if r.admitted and r.changes == "skipped" and not r.cancelled]
    assert rows
    for row in rows:
        assert _outcome(gate, row) == "changes", row


@pytest.mark.parametrize("name", CHANGE_GATED)
def test_change_detection_guard_fails_before_any_other_work(name):
    gate = _gate(name)
    steps = gate["steps"]
    names = [step.get("name") for step in steps]
    index = names.index(CHANGES_STEP)
    # Only the license refusal may come first: when it fails, the guard is skipped with the rest.
    assert names[:index] == ([LICENSE_STEP] if name == "backend-required.yml" else [])

    guard = steps[index]
    assert guard["if"] == "needs.changes.result != 'success'"
    assert guard["env"] == {"CHANGES_RESULT": "${{ needs.changes.result }}"}
    result = subprocess.run(  # the workflow's own script, in a scratch shell
        ["bash", "-e", "-c", guard["run"]],
        env={**os.environ, "CHANGES_RESULT": "failure"},
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 1
    assert result.stdout.startswith("::error::") and "'failure'" in result.stdout

    # Nothing can turn that failure back into a pass, and nothing after it runs.
    assert "continue-on-error" not in guard and "continue-on-error" not in gate
    for step in steps[index + 1:]:
        condition = str(step.get("if", ""))
        assert not re.search(r"\b(always|failure|cancelled)\(\)", condition), step.get("name")


def test_backend_required_still_turns_a_negative_license_verdict_red():
    refusal = _gate("backend-required.yml")["steps"][0]
    assert refusal["name"] == LICENSE_STEP
    assert refusal["run"] == (
        'echo "::error::License audit did not pass for this PR head, so the gates were not run. '
        'Fix the licence policy failure and re-run."\nexit 1\n'
    )


# --- 4.6: waking the queue after a red gate ---------------------------------------------------


@pytest.mark.parametrize("name", GATES)
def test_queue_tick_runs_the_queue_from_dev_with_the_queue_permissions(name):
    jobs = _wf(name)["jobs"]
    gate = GATES[name]
    assert jobs[gate]["name"] == gate, "the gate job's name is the required context"
    tick = dict(jobs["queue-tick"])
    tick.pop("if")  # covered by test_queue_tick_runs_only_after_a_failed_gate_while_the_queue_is_enabled
    queue = _wf("merge-queue.yml")["jobs"]["queue"]  # pinned by test_merge_queue_workflow.py
    assert tick == {
        "name": "Merge queue tick",
        "needs": [gate],
        "runs-on": "ubuntu-latest",
        "timeout-minutes": 10,
        "permissions": queue["permissions"],
        "steps": queue["steps"],
    }
    assert queue["steps"][0]["with"] == {"ref": "dev", "persist-credentials": False}
    checkouts = set(re.findall(r"actions/checkout@\S+", _text(name)))
    assert checkouts == {queue["steps"][0]["uses"]}, "one pinned checkout action per workflow"


@pytest.mark.parametrize("name", GATES)
def test_queue_tick_runs_only_after_a_failed_gate_while_the_queue_is_enabled(name):
    condition = _wf(name)["jobs"]["queue-tick"]["if"]
    events = [  # (event, PR head repository)
        ("workflow_dispatch", ""),
        ("pull_request", "owner/repo"),
        ("pull_request", "fork/repo"),
        ("workflow_run", ""),
        ("push", ""),
    ]
    ran = 0
    for (event, head_repo), result, mode, cancelled in product(
        events, ("failure", "success", "skipped", "cancelled"), ("", "off", "dry", "on"), (False, True)
    ):
        values = {
            f"needs.{GATES[name]}.result": result,
            "vars.MERGE_QUEUE": mode,
            "github.event_name": event,
            "github.event.pull_request.head.repo.full_name": head_repo,
            "github.repository": "owner/repo",
        }
        expected = (
            not cancelled
            and result == "failure"
            and mode in ("dry", "on")
            and (event, head_repo) in (("workflow_dispatch", ""), ("pull_request", "owner/repo"))
        )
        assert _evaluate(condition, values, cancelled) is expected, (event, head_repo, result, mode, cancelled)
        ran += expected
    assert ran == 4  # two events x two modes: with MERGE_QUEUE unset it never runs


@pytest.mark.parametrize("name", GATES)
def test_queue_tick_cannot_turn_a_required_check_red(name):
    """Nothing waits on it, so the required job has reported before it starts."""
    for job_id, job in _wf(name)["jobs"].items():
        needs = job.get("needs", [])
        assert "queue-tick" not in ([needs] if isinstance(needs, str) else needs), job_id


@pytest.mark.parametrize("name", GATES)
def test_conditions_read_only_the_jobs_they_need(name):
    """`needs.<job>` is empty for a job that is not listed in `needs`, which reads as "not success"."""
    jobs = _wf(name)["jobs"]
    for job_id in (GATES[name], "queue-tick"):
        job = jobs[job_id]
        conditions = [job["if"], *(str(step.get("if", "")) for step in job["steps"])]
        read = {match for condition in conditions for match in re.findall(r"needs\.([\w-]+)\.", condition)}
        assert read <= set(job["needs"]), (job_id, read)

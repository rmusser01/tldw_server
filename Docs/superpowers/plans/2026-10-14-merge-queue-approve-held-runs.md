# Merge queue: approve held gate runs instead of dispatching them — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the tldw_server merge queue able to land a PR: after rebasing the front PR it approves the six held
`pull_request` gate runs GitHub creates, re-runs a failed gate instead of dispatching it, and wakes itself through a
`security-required.yml` "queue kick" instead of running inside a gate run.

**Architecture:** `Helper_Scripts/ci/merge_queue.py` keeps its seven-context decision table (`decide_front`, pure)
and gains the approve / re-run / strike / kick layer ported from the tldw_chatbook queue (`scripts/merge_queue.py` at
tldw_chatbook `dev` `2716b844e1`), adapted from one required check to six gate contexts plus the license status.
`security-required.yml` hosts the kick; the six `queue-tick` jobs shrink to one `gh api` dispatch of that kick.

**Tech Stack:** Python 3.11 (stdlib only in the script), `gh` CLI, GitHub Actions YAML, pytest + PyYAML for tests,
backlog-py for the task record.

**Spec:** `Docs/superpowers/specs/2026-10-11-merge-queue-approve-held-runs-design.md` (approved 2026-10-14, D1-D3 as
recommended). It amends `Docs/superpowers/specs/2026-10-04-merge-queue-design.md`; read both.

## Global Constraints

- All work targets `dev`. Nothing may depend on a file on `main`: no `schedule`, `workflow_run`, `pull_request_target`,
  `check_suite` or `check_run` triggers in queue code.
- No personal access token, GitHub App key or deploy key anywhere. Only the built-in `GITHUB_TOKEN` (`github.token`).
- The queue never arms auto-merge, never merges, never pushes (existing guard tests stay green).
- **The queue never dispatches a check context.** The only dispatches allowed: `frontend-license-gate.yml` on `dev`
  with `pr`, and `security-required.yml` on `dev` with `queue_kick=true` (optionally `wait_run`).
- **What may be approved:** only a run that is held (`action_required`), has event `pull_request`, was caused by
  `github-actions[bot]`, has this repository as its head repository, and belongs to one of the six gate workflows
  (`backend-required.yml`, `security-required.yml`, `coverage-required.yml`, `frontend-required.yml`,
  `e2e-required.yml`, `container-build-check.yml`). Only for the front PR.
- D1: the non-gate held runs on a queue-rebased head are deleted, never approved.
- D2: `frontend-required.yml` publishes `frontend-required-diagnostic` on every `workflow_dispatch`, whoever started it.
- D3: the kick host is `security-required.yml`, dispatched on `refs/heads/dev` with `queue_kick=true`.
- Errors: a transient error (5xx, 429, rate limit, abuse detection, network, GraphQL "Something went wrong") fails the
  queue run and never disarms a PR. A refusal is re-read first; otherwise a strike: one warning comment, eviction only
  once that comment is at least 10 minutes old (`STRIKE_GAP`).
- Evictions post the comment first and disarm second, and name their cause (`evict-no-run`, `evict-rerun`,
  `evict-approve` join `evict-conflict`, `evict-failed-twice`, `evict-stuck`, `evict-workflows`, `evict-rebase`,
  `evict-rebase-unmoved`).
- `MERGE_QUEUE` stays unset through the merge of this work. Nothing changes for any PR until the owner switches it.
- Python 3.11: no backslash inside an f-string's `{...}` expression; build such pieces in a variable first.
- Dispatch inputs are strings: `queue_kick` is `type: string` and compared as `github.event.inputs.queue_kick == 'true'`.
- A cancelled gate run is restarted at most once per gate per head; any other completed gate run without its check is
  one failure. Nothing is re-run on every wake.
- Task records only through backlog-py (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task ...`).
- Every pytest call runs from the implementation worktree root with the server venv:
  `PATH=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin:$PATH python -m pytest ...` (contract tests shell
  out and need that `bin` on `PATH`).

## Review Focus

1. A held run of a non-gate workflow (`ci.yml`, 200+ jobs) on the rebased head must never be approved, including on
   the start and `_decide` approval paths, and must be deleted. Tests (Task 1): Part C
   `test_rebase_dispatches_the_license_then_approves_exactly_the_six_gates_then_cleans_up_then_cancels`, Part D
   `test_decide_never_approves_a_non_gate_held_run`.
2. A run held for another actor (an agent's push awaiting a person) or with a fork as head repository must never be
   approved. Test (Task 1, Part B): `test_only_the_queues_own_held_gate_runs_are_approvable`.
3. A transient GitHub error during approve, re-run or the compare check must fail the run and disarm nobody. Tests
   (Task 1): Part B `test_a_transient_approve_error_raises_and_disarms_nobody`, Part D
   `test_a_transient_rerun_error_fails_the_run_and_disarms_nobody` and `test_a_failed_behind_check_never_disarms`,
   Part E `test_a_github_side_error_on_approve_or_rerun_fails_the_run_and_disarms_nobody`.
4. A kick must not cancel a running kick, and a `queue_kick=true` dispatch on any ref other than `dev` must run the
   gate as today. Tests (Task 3): `test_security_required_kick_skips_every_gate_job_only_on_dev` and
   `test_a_kick_never_cancels_a_running_kick`.
5. Nothing may be re-run on every wake: a gate run that finished without its check, or a run that keeps being
   cancelled, must end in an eviction, not hold the line (review findings M2, m1). A gate run someone dispatched by
   hand must neither stall the line nor satisfy or fail a context. Tests (Task 1): Part A
   `test_a_dispatched_gate_run_is_ignored_whoever_started_it` and
   `test_a_completed_gate_run_without_its_check_is_one_failure_unless_cancelled_or_held`, Part D
   `test_a_green_gate_run_without_its_check_is_retried_once_then_evicted`,
   `test_start_restarts_a_cancelled_run_of_a_missing_gate_once_then_evicts` and
   `test_a_retry_attempt_that_ends_cancelled_is_restarted_once_then_evicted`.

Revised 2026-10-14 after an independent review that applied the first draft to a copy of the repository: tasks 1-4 of
that draft are folded into Task 1 (they could not each end green), the restart and no-check caps, the green-row compare,
license-first retry order and the post-rebase wake were added, and every existing test the change breaks is listed.

---

## File map

| File | Responsibility | Task |
|---|---|---|
| `Helper_Scripts/ci/merge_queue.py` | the queue: read, decide, act | 1 |
| `tldw_Server_API/tests/CI/test_merge_queue_rules.py` | pure decision table | 1 |
| `tldw_Server_API/tests/CI/test_merge_queue_actions.py` | action layer and run loop against a fake `gh` | 1, 2 |
| `.github/workflows/security-required.yml` | kick inputs, `queue-kick` job, gates skip on a kick, kick-safe concurrency | 3 |
| `.github/workflows/{backend,coverage,e2e,frontend}-required.yml`, `container-build-check.yml` | `queue-tick` sends a kick | 3 |
| `.github/workflows/frontend-required.yml` | name guard without the bot exemption (D2) | 3 |
| `tldw_Server_API/tests/CI/test_merge_queue_gate_workflows.py` | tick and kick shapes | 3 |
| `tldw_Server_API/tests/CI/test_merge_queue_frontend_required.py` | name guard | 3 |
| `tldw_Server_API/tests/CI/test_admin_ui_vitest_ratchet_workflow.py` | the protected-name guard's truth table | 3 |
| `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py` | trigger digest, job list, tick/kick exemptions, concurrency, gate `if` pins | 3 |
| `Docs/Development/CI_REQUIRED_GATES.md`, `AGENTS.md`, `CLAUDE.md`, `Docs/ADR/063-in-repo-merge-queue.md` (+ `Docs/Published` copy), `Docs/superpowers/specs/2026-10-04-merge-queue-design.md` | docs | 4 |
| `backlog/tasks/task-<N> - ...md` | task record | 0, 4 |

`.github/workflows/merge-queue.yml` does not change.

---

### Task 0: Workspace and task record

**Files:**
- Create: worktree `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/mq-approve-held`
- Create: `backlog/tasks/task-<N> - Merge-queue-approve-held-gate-runs-instead-of-dispatching-them.md` (via backlog-py)

- [ ] **Step 1: Create the worktree from the spec branch**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2
git fetch -q origin
git worktree add -b ci/merge-queue-approve-held-runs .worktrees/mq-approve-held origin/docs/merge-queue-approve-held-runs-spec
cd .worktrees/mq-approve-held
git rebase -q origin/dev
git log --oneline -4   # the spec and plan commits sit on top of origin/dev
```

- [ ] **Step 2: Baseline: the queue and contract tests pass before any change**

```bash
PATH=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin:$PATH python -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/CI/test_merge_queue_rules.py tldw_Server_API/tests/CI/test_merge_queue_actions.py \
  tldw_Server_API/tests/CI/test_merge_queue_workflow.py tldw_Server_API/tests/CI/test_merge_queue_gate_workflows.py \
  tldw_Server_API/tests/CI/test_merge_queue_frontend_required.py tldw_Server_API/tests/CI/test_merge_queue_license_dispatch.py \
  tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py
```
Expected: all pass. Record the count in the task notes later. If anything fails here, stop: it is not this work's
failure, and report it.

- [ ] **Step 3: Pick the task id and create the task**

```bash
git fetch origin dev --quiet
{ git ls-tree -r --name-only origin/dev backlog/; gh pr list --state open --limit 500 --json files --jq '.[].files[].path'; } \
  | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1
# N = that number + 1
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task create \
  "Merge queue: approve held gate runs instead of dispatching them" --id TASK-<N> -s "In Progress" -l ci -l merge-queue \
  -d "The queue as merged in #3188 cannot land a PR: it starts the six gates with workflow_dispatch, and a dispatched run's check never enters the PR's status rollup, so the PR stays BLOCKED with every gate green (probe #3220). Probe #3221 showed what works with the built-in token alone: a token rebase creates held pull_request runs, and approving a gate's held run makes its check count. Spec: Docs/superpowers/specs/2026-10-11-merge-queue-approve-held-runs-design.md (owner-approved 2026-10-14, D1-D3 as recommended)." \
  --ac "After a rebase the queue dispatches the license gate on dev, approves exactly the six held gate runs, deletes the other held runs and cancels the old head's live runs" \
  --ac "A missing gate is started by approving its held run or re-running its cancelled run, never by a dispatch; with neither, a no-run strike evicts after 10 minutes" \
  --ac "A gate that failed once is re-run in place (failed jobs) and its held re-run approved; a second failure evicts" \
  --ac "No code path dispatches a check context; a guard test fails if one does" \
  --ac "A failed gate's queue-tick only dispatches security-required.yml on dev with queue_kick=true and its run id; the kick waits for that run, then runs the queue, and never cancels another kick" \
  --ac "frontend-required.yml publishes frontend-required-diagnostic on every dispatch" \
  --ac "Transient GitHub errors fail the queue run and disarm nobody; other refusals strike before evicting" \
  --ac "A green front PR is rebased, not evicted as stuck, when dev has commits its head lacks" \
  --ac "Workflow contract tests, CI_REQUIRED_GATES.md, AGENTS.md, CLAUDE.md and ADR-063 match the new behaviour" \
  --plan "Docs/superpowers/plans/2026-10-14-merge-queue-approve-held-runs.md"
git add backlog/ && git commit -q -m "chore(backlog): TASK-<N> merge queue approves held gate runs"
```

---

### Task 1: The script — approve, re-run and kick instead of dispatch

One task with one green gate. The read layer, the primitives, the rebase path and the decision loop are tied together
through `apply` and `run`, so the existing tests cannot be green part-way (review finding M5). Parts A-D each add code
and their own new tests, and each ends by running **only those new tests, by exact name**; they pass on their own.
Part E fixes every existing test the change breaks, from a complete list, deletes the dead code, and runs the whole
queue test set: that is the gate. Commit once, at the end of Part E. The task review covers the whole task.

#### Part A: Read layer — only `pull_request` runs count

Removes everything that existed to make dispatches work (`base_sha`, `queue_only_dispatch`, `silent_green`, `baseRef`)
and stops dispatched runs from counting, either as a check or as a stand-in. A completed gate `pull_request` run that
reported no check now counts as one failure unless it was cancelled or is held: a renamed gate job, a startup failure or
a timeout then gets one retry and is evicted on the second, instead of being re-run on every wake (review finding M2).

**Files:**
- Modify: `Helper_Scripts/ci/merge_queue.py` (`Context`, `CONTEXTS`, `PrState`, `PR_FIELDS`, `_parse_pr`,
  `required_run_stand_ins`, `read_contexts`; new `counted`, `BY_NAME`, `GATE_WORKFLOWS`, `LICENSE`)
- Test: `tldw_Server_API/tests/CI/test_merge_queue_rules.py`, `tldw_Server_API/tests/CI/test_merge_queue_actions.py`

**Interfaces:**
- Produces: `GATE_WORKFLOWS: frozenset[str]`, `BY_NAME: dict[str, Context]`, `LICENSE: Context`,
  `counted(checks: tuple[CheckRun, ...], runs: list[dict]) -> tuple[CheckRun, ...]`,
  `required_run_stand_ins(runs: list[dict], checks: tuple[CheckRun, ...]) -> tuple[CheckRun, ...]` (pull_request
  runs only). `Context` has exactly `name`, `workflow`, `kind`. `PrState` no longer has `base_sha`.

- [ ] **Step A1: Write the failing tests**

In `test_merge_queue_actions.py`, change the `_run_of` default event to `pull_request` (dispatched runs no longer
count, so every existing test that means "a run of this gate" must say `pull_request`; tests that mean a dispatch pass
`event="workflow_dispatch"` explicitly; Part E lists them):

```python
def _run_of(workflow, rid, status="in_progress", conclusion=None, event="pull_request", **extra):
    return {"id": rid, "path": f".github/workflows/{workflow}", "event": event, "status": status,
            "conclusion": conclusion, "html_url": f"https://run/{rid}", **extra}
```

Drop `base` from `_node` (and the `baseRef` key) and `rebased_base` from `FakeGh`. Add these tests (they call the read
layer directly, so they do not depend on Parts B-D):

```python
@pytest.mark.parametrize("actor", ["github-actions[bot]", "someone"])
@pytest.mark.parametrize(("status", "conclusion"), [("in_progress", None), ("completed", "success"),
                                                     ("completed", "failure"), ("completed", "startup_failure")])
def test_a_dispatched_gate_run_is_ignored_whoever_started_it(actor, status, conclusion):
    """Spec section 7, stand-ins: a dispatched gate run never counts (S5), so it can neither hold the
    line as in flight nor fail a context."""
    run = _run_of("e2e-required.yml", 77, status, conclusion, event="workflow_dispatch",
                  actor={"login": actor}, triggering_actor={"login": actor}, check_suite_id=770)
    assert mq.required_run_stand_ins([run], ()) == ()


def test_a_dispatched_runs_check_does_not_count():
    """A dispatched run reports its check on the head commit, but the check is not in the PR's rollup (S5):
    a green one must not read as passed, a red one must not read as failed. A pull_request run's check counts."""
    dispatched = _run_of("e2e-required.yml", 77, "completed", "success", event="workflow_dispatch", check_suite_id=770)
    pr_run = _run_of("backend-required.yml", 78, "completed", "success", check_suite_id=780)
    from_dispatch = mq.CheckRun(E2E, "completed", "success", NOW, "u", 770)
    from_pr = mq.CheckRun(BACKEND, "completed", "failure", NOW, "u", 780)
    unlisted = mq.CheckRun(SECURITY, "completed", "success", NOW, "u", 999)
    assert mq.counted((from_dispatch, from_pr, unlisted), [dispatched, pr_run]) == (from_pr, unlisted)


@pytest.mark.parametrize(("conclusion", "stands_in"), [
    ("success", True), ("neutral", True), ("skipped", True), ("failure", True), ("startup_failure", True),
    ("timed_out", True), ("cancelled", False), ("action_required", False),
])
def test_a_completed_gate_run_without_its_check_is_one_failure_unless_cancelled_or_held(conclusion, stands_in):
    """Review finding M2: a run that finished without reporting its check (a renamed gate job, a startup
    failure) must count as a failure, so it is retried once and then evicted, not re-run on every wake.
    A cancelled run is restarted instead (Part D), and a held run is approved."""
    run = _run_of("e2e-required.yml", 70, "completed", conclusion, check_suite_id=700, updated_at="2026-10-04T11:00:00Z")
    stand_ins = mq.required_run_stand_ins([run], ())
    assert [(c.context, c.conclusion) for c in stand_ins] == ([(E2E, "failure")] if stands_in else [])
    reported = (mq.CheckRun(E2E, "completed", "success", NOW, "u", 700),)
    assert mq.required_run_stand_ins([run], reported) == (), "a run whose suite reported the check speaks through it"
```

In `test_merge_queue_rules.py`, replace `test_the_seven_required_contexts_and_how_each_is_started` and drop
`"base_sha": "d" * 40` from `_pr`:

```python
def test_the_seven_required_contexts() -> None:
    """Six gate check runs started by approving their held pull_request runs; one license status
    dispatched on dev (spec 4.1)."""
    assert [(c.name, c.workflow, c.kind) for c in mq.CONTEXTS] == [
        ("backend-required", "backend-required.yml", "check"),
        ("security-required", "security-required.yml", "check"),
        ("coverage-required", "coverage-required.yml", "check"),
        ("frontend-required", "frontend-required.yml", "check"),
        ("e2e-required", "e2e-required.yml", "check"),
        ("container-build-check", "container-build-check.yml", "check"),
        ("frontend-license-policy/trusted/dev", "frontend-license-gate.yml", "status"),
    ]
    assert mq.GATE_WORKFLOWS == {c.workflow for c in mq.CONTEXTS[:6]}
    assert mq.LICENSE.name == "frontend-license-policy/trusted/dev"
    assert mq.REPO.count("/") == 1 and mq.BASE == "dev"
```

- [ ] **Step A2: Run them to see them fail**

Run: `python -m pytest -q -p no:cacheprovider "tldw_Server_API/tests/CI/test_merge_queue_rules.py::test_the_seven_required_contexts" tldw_Server_API/tests/CI/test_merge_queue_actions.py -k "test_a_dispatched_gate_run_is_ignored_whoever_started_it or test_a_dispatched_runs_check_does_not_count or test_a_completed_gate_run_without_its_check_is_one_failure_unless_cancelled_or_held"`
Expected: FAIL (`GATE_WORKFLOWS` and `counted` missing; a dispatched or green-without-check run still stands in wrongly).

- [ ] **Step A3: Implement**

Replace the `Context` dataclass and the lines after `CONTEXTS`:

```python
@dataclass(frozen=True)
class Context:
    """One required status on `dev` and the workflow that reports it (spec 4.1).

    Attributes:
        name: The context, as branch protection names it.
        workflow: The workflow file that reports it.
        kind: `check` for a gate's check run, which counts only from a `pull_request` run on the
            PR head; `status` for the license commit status, posted by the license gate the
            queue dispatches on `dev`.
    """

    name: str
    workflow: str
    kind: str = "check"


CONTEXTS = (
    Context("backend-required", "backend-required.yml"),
    Context("security-required", "security-required.yml"),
    Context("coverage-required", "coverage-required.yml"),
    Context("frontend-required", "frontend-required.yml"),
    Context("e2e-required", "e2e-required.yml"),
    Context("container-build-check", "container-build-check.yml"),
    Context("frontend-license-policy/trusted/dev", "frontend-license-gate.yml", kind="status"),
)
ALL_CONTEXTS = tuple(c.name for c in CONTEXTS)
BY_NAME = {c.name: c for c in CONTEXTS}
GATE_WORKFLOWS = frozenset(c.workflow for c in CONTEXTS if c.kind == "check")
LICENSE = next(c for c in CONTEXTS if c.kind == "status")
```

In `PrState`, delete the `base_sha` field and its docstring line, and change `head_ref`'s docstring to "The head
branch name.". In `PR_FIELDS`, delete the `baseRef { target { oid } }` line and the comment above `PR_FIELDS`. In
`_parse_pr`, delete the `base_sha=` argument. In `read_pr`'s docstring, delete ", and dev's tip with it".
`BROKEN_RUN_CONCLUSIONS` is no longer read by `required_run_stand_ins`; Part D's `_retry_gate` stops reading it too,
so delete the constant in Part E if nothing else uses it.

Replace `required_run_stand_ins` and `read_contexts`, and add `counted` above them:

```python
def counted(checks: tuple[CheckRun, ...], runs: list[dict]) -> tuple[CheckRun, ...]:
    """Drop gate check runs that came from a run other than `pull_request` (a `workflow_dispatch`).

    Such a check is attached to the head commit but absent from the PR's status rollup, so it
    never satisfies the context (spec S5). A check whose run is not listed is kept.

    Args:
        checks: The gate contexts' check runs on the head, from `read_checks`.
        runs: Every workflow run on the head, from `runs_on`.

    Returns:
        The checks that count.
    """
    not_counted = {r.get("check_suite_id") for r in runs
                   if _workflow_name(r) in GATE_WORKFLOWS and r.get("event") != "pull_request"}
    return tuple(c for c in checks if c.suite_id is None or c.suite_id not in not_counted)


# A completed run with one of these conclusions is no result: a cancelled run is restarted, a
# held one approved. Any other completed run that reported no check is a failure of its gate.
NO_RESULT_CONCLUSIONS = frozenset({"cancelled", "action_required"})


def required_run_stand_ins(runs: list[dict], checks: tuple[CheckRun, ...]) -> tuple[CheckRun, ...]:
    """Gate `pull_request` runs on a head that have not reported their check, as stand-ins.

    - A LIVE run stands in as an in-flight check: each gate is a needs-gated job with no check run
      until the jobs before it finish, so an in-flight front PR would otherwise look like it has
      no run.
    - A COMPLETED run with no check run for its context in its check suite stands in as one
      failure, whatever it concluded, unless it was cancelled or is held (`NO_RESULT_CONCLUSIONS`):
      a startup failure, a timeout before the gate job, or a gate job the branch renamed or removed
      would otherwise read as "no run" and be started again on every wake. As a failure it is
      retried once and then evicted. A run whose suite DID report the check is judged by that check
      (a failed run whose queue-tick failed is not a gate failure).

    Only `pull_request` runs stand in: a dispatched gate run can never satisfy a context (spec
    S5), whoever started it, so waiting on one would only stall the line. A workflow-run
    conclusion is only ever read as a failure here, never as a merge signal. The license gate runs
    on `dev`, not on this head, so it has no stand-in.

    Args:
        runs: The workflow runs on the head, from `runs_on`.
        checks: The gate contexts' counted check runs on that head.

    Returns:
        One stand-in per live run and per completed run that reported no check.
    """
    by_workflow = {c.workflow: c for c in CONTEXTS if c.kind == "check"}
    reported = {(c.context, c.suite_id) for c in checks if c.suite_id is not None}
    stand_ins = []
    for run in runs:
        ctx = by_workflow.get(_workflow_name(run))
        if ctx is None or run.get("event") != "pull_request":
            continue
        url = run.get("html_url", "")
        if run.get("status") in LIVE_RUN_STATUSES:
            stand_ins.append(CheckRun(ctx.name, run["status"], None, None, url))
        elif (run.get("status") == "completed" and run.get("conclusion") not in NO_RESULT_CONCLUSIONS
              and (ctx.name, run.get("check_suite_id")) not in reported):
            stand_ins.append(CheckRun(ctx.name, "completed", "failure", _ts(run.get("updated_at")), url))
    return tuple(stand_ins)


def read_contexts(gh: GhApi, sha: str) -> tuple[CheckRun, ...]:
    """Read all seven required contexts' runs on a commit (spec 4.1).

    Workflow runs are read before check runs: a run seen as completed already has its check
    runs, so "completed without reporting its check" is never an artefact of the read order.

    Args:
        gh: The GitHub client.
        sha: The head commit.

    Returns:
        Counted check runs, workflow-run stand-ins and commit statuses, each tagged with its context.

    Raises:
        GhError: A call failed.
    """
    runs = runs_on(gh, sha)
    checks = counted(read_checks(gh, sha), runs)
    return checks + required_run_stand_ins(runs, checks) + read_statuses(gh, sha)
```

The queue no longer runs inside a gate run (Part D's ticks only send a kick), so `required_run_stand_ins` drops the
`run.get("id") == _own_run_id()` exclusion; `_own_run_id` stays for the cancel loop in Part C.

- [ ] **Step A4: Run the Part A tests**

Run the command of Step A2. Expected: PASS.

#### Part B: GitHub primitives — approve, re-run, strike, kick (ported)

Each function here is ported from tldw_chatbook `scripts/merge_queue.py` at `2716b844e1`; the code below is the port,
already adapted. Where the chatbook text says "the required check", read "a gate".

**Files:**
- Modify: `Helper_Scripts/ci/merge_queue.py`
- Test: `tldw_Server_API/tests/CI/test_merge_queue_actions.py`

**Interfaces:**
- Consumes: `GATE_WORKFLOWS`, `LICENSE` (Part A).
- Produces (exact signatures later tasks call):
  - `class CommentRefused(GhError)`
  - `_transient(exc: GhError) -> bool`
  - `_is_held(run: dict) -> bool`, `_queue_held(run: dict) -> bool` (gate workflows only)
  - `approve_held_runs(gh: GhApi, sha: str, log) -> set[str]` (workflow names approved; best-effort)
  - `_approve_strictly(gh, pr: PrState, run: dict, log, now: Callable[[], datetime]) -> bool` (True = evicted)
  - `has_comment(gh, number: int, kind: str, sha: str) -> bool`
  - `marker_age(gh, number: int, kind: str, sha: str, now: datetime) -> timedelta | None`
  - `comment_once(gh, number: int, kind: str, sha: str, body: str, log=print) -> bool`
  - `_strike(gh, pr, kind: str, warning: str, eviction: Action, wake_first: bool, log, now: Callable) -> bool`
  - `_evict(gh, pr, action: Action, log) -> bool` (comment, then `_disarm`); `_disarm(gh, pr, log) -> None`
  - `dispatch_license(gh, pr: PrState, log) -> None`
  - `wake(gh, wait_run: int | None = None) -> None`; `_wake_best_effort(gh, log) -> None`
  - `wait_for_run(gh, run_id: int, sleep, log) -> None`
  - `_going_again(gh, run_id: int) -> bool`
  - `_rerun(gh, pr, run: dict, failed_only: bool, log, sleep, now: Callable) -> str` (`started`/`pending`/`refused`)

- [ ] **Step B1: Extend the fake and write the failing tests**

Change `FakeGh`'s default `rebase_error_text="rebase refused"` to `rebase_error_text="gh: rebase refused (HTTP 422)"`.
The ported `_transient` reads an error with no `gh: ` line as a network failure (that is how gh reports one), so the old
default would now take the transient path in every rebase test; a real refusal always carries `gh: ... (HTTP 4xx)`.
Never change `_transient` to make a test pass: "an outage never disarms" depends on it (review finding M3). Keep
`dispatch_refused`, `dispatch_status` and `DISPATCH_ERRORS`: the license-refusal test still uses them.

In `FakeGh.__init__` add keyword arguments `approve_error=None, rerun_error=None, run_status=None, behind_by=None`
and store them (`self.run_status = run_status or {}`, `self.run_reads = {}`, `self.behind_by = behind_by or {}`). Add a
helper to mutate a listed run:

```python
    def _set_run(self, rid, **fields):
        for runs in self.runs.values():
            for run in runs:
                if run.get("id") == rid:
                    run.update(fields)
```

Replace the comments read in `graphql` (an entry is a body, or `(body, createdAt)`; a bare body has no timestamp):

```python
        if "comments(last" in query:
            nodes = []
            for index, entry in enumerate(self.comments.get(v["number"], []), start=1):
                body, created = entry if isinstance(entry, tuple) else (entry, None)
                nodes.append({"databaseId": index, "body": body, "createdAt": created})
            return {"data": {"repository": {"pullRequest": {"comments": {"nodes": nodes}}}}}
```

Add to `rest`, before the final `raise`:

```python
        if method == "POST" and path.endswith("/approve"):
            rid = path.split("/runs/")[1].split("/")[0]
            self._record(("approve", rid))
            if self.approve_error:
                raise mq.GhError(self.approve_error)
            self._set_run(int(rid), status="queued", conclusion=None)  # GitHub starts it
            return None
        if method == "POST" and path.endswith(("/rerun", "/rerun-failed-jobs")):
            rid = path.split("/runs/")[1].split("/")[0]
            self._record(("rerun", rid, "failed" if path.endswith("-failed-jobs") else "all"))
            if self.rerun_error:
                raise mq.GhError(self.rerun_error)
            self._set_run(int(rid), status="queued", conclusion=None)
            return None
        if method == "GET" and re.search(r"/actions/runs/\d+$", path):
            rid = int(path.rsplit("/", 1)[1])
            seen = self.run_reads.get(rid, 0)
            self.run_reads[rid] = seen + 1
            if rid in self.run_status:
                statuses = self.run_status[rid]
                entry = statuses[min(seen, len(statuses) - 1)]
                return {"id": rid, **entry} if isinstance(entry, dict) else {"id": rid, "status": entry}
            run = next((r for runs in self.runs.values() for r in runs if r.get("id") == rid), {"status": "completed"})
            return dict(run)
        if method == "GET" and "/compare/" in path:
            self.reads += 1
            behind = self.behind_by.get(path.split("...")[1].split("?")[0], 0)
            if isinstance(behind, Exception):
                raise behind
            return {"behind_by": behind}
```

Add helpers next to `_run_of`:

```python
GATE_FILES = tuple(c.workflow for c in mq.CONTEXTS if c.kind == "check")


def _held(rid, workflow="backend-required.yml", actor="github-actions[bot]", event="pull_request", repo=None):
    """A run GitHub holds for approval because the queue's token caused its event (spec S2)."""
    return {"id": rid, "path": f".github/workflows/{workflow}", "event": event, "status": "completed",
            "conclusion": "action_required", "triggering_actor": {"login": actor},
            "head_repository": {"full_name": repo or mq.REPO}, "html_url": f"https://run/{rid}"}


def _six_held(first=31):
    """The six gate workflows' held runs on a queue-rebased head."""
    return [_held(first + i, workflow) for i, workflow in enumerate(GATE_FILES)]


def _pr_state(**kw):
    """A front PrState for calling one action function directly."""
    base = {"number": 1, "node_id": "PR_1", "head_sha": OLD, "head_ref": "feat/1", "same_repo": True,
            "armed_at": NOW, "is_draft": False, "merge_state": "BLOCKED", "head_committed_at": NOW}
    base.update(kw)
    return mq.PrState(**base)
```

Tests:

```python
@pytest.mark.parametrize(("run", "approvable"), [
    (_held(1, "backend-required.yml"), True),
    (_held(1, "container-build-check.yml"), True),
    (_held(1, "ci.yml"), False),                                   # D1: never a non-gate workflow
    (_held(1, "pre-commit.yml"), False),
    (_held(1, actor="some-agent[bot]"), False),                    # held for a person to approve
    (_held(1, repo="someone/fork"), False),                        # GitHub's outside-contributor gate
    (_held(1, event="workflow_dispatch"), False),
    (dict(_held(1), conclusion="success"), False),                 # not held at all
])
def test_only_the_queues_own_held_gate_runs_are_approvable(run, approvable):
    assert mq._queue_held(run) is approvable


def test_approve_held_runs_approves_only_the_queues_gate_runs():
    runs = [_held(31, "backend-required.yml"), _held(32, "ci.yml"), _held(33, "e2e-required.yml"),
            _held(34, actor="person")]
    gh = FakeGh([_node(1)], runs={OLD: runs})
    assert mq.approve_held_runs(gh, OLD, lambda m: None) == {"backend-required.yml", "e2e-required.yml"}
    assert [c for c in gh.calls if c[0] == "approve"] == [("approve", "31"), ("approve", "33")]


@pytest.mark.parametrize("error", [
    "gh api -X POST failed: gh: Server Error (HTTP 502)",
    "gh api graphql failed: gh: Something went wrong while executing your query. (HTTP 200)",
    'gh api -X POST failed: Post "https://api.github.com/repos/x/y/actions/runs/1/approve": EOF',
    "gh api -X POST failed: gh: API rate limit exceeded (HTTP 403)",
])
def test_a_transient_approve_error_raises_and_disarms_nobody(error):
    gh = FakeGh([_node(1)], runs={OLD: [_held(31)]}, approve_error=error)
    with pytest.raises(mq.GhError):
        mq._approve_strictly(gh, _pr_state(), _held(31), lambda m: None, lambda: NOW)
    assert [c[0] for c in gh.calls] == ["approve"]


def test_a_refused_approval_of_a_run_no_longer_held_stands_down():
    gh = FakeGh([_node(1)], approve_error="gh: Not Found (HTTP 404)", run_status={31: [{"status": "queued"}]})
    assert mq._approve_strictly(gh, _pr_state(), _held(31), lambda m: None, lambda: NOW) is False
    assert [c[0] for c in gh.calls] == ["approve"]


def test_a_refused_approval_strikes_then_evicts_after_the_gap():
    refused = "gh: Resource not accessible by integration (HTTP 403)"
    first = FakeGh([_node(1)], runs={OLD: [_held(31)]}, approve_error=refused)
    assert mq._approve_strictly(first, _pr_state(), _held(31), lambda m: None, lambda: NOW) is False
    assert first.calls[-1][0] == "comment" and f"merge-queue:approve-error:{OLD}" in first.calls[-1][2]
    warned = (f"<!-- merge-queue:approve-error:{OLD} -->\nx", "2026-10-04T11:49:00Z")  # 11 minutes old
    later = FakeGh([_node(1)], runs={OLD: [_held(31)]}, approve_error=refused, comments={1: [warned]})
    assert mq._approve_strictly(later, _pr_state(), _held(31), lambda m: None, lambda: NOW) is True
    assert [c[0] for c in later.calls] == ["approve", "comment", "disarm"]
    assert f"merge-queue:evict-approve:{OLD}" in later.calls[1][2]


def test_a_strike_inside_the_gap_neither_warns_again_nor_evicts():
    warned = (f"<!-- merge-queue:no-run:{OLD} -->\nx", "2026-10-04T11:55:00Z")  # 5 minutes old
    gh = FakeGh([_node(1)], comments={1: [warned]})
    evicted = mq._strike(gh, _pr_state(), "no-run", "w", mq.Action("evict", "e", slug="no-run"), False,
                         lambda m: None, lambda: NOW)
    assert evicted is False and gh.calls == []


def test_evict_comments_first_then_disarms():
    gh = FakeGh([_node(1)])
    assert mq._evict(gh, _pr_state(), mq.Action("evict", "why", slug="conflict"), lambda m: None) is True
    assert [c[0] for c in gh.calls] == ["comment", "disarm"]


def test_a_failed_disarm_of_a_pr_still_armed_raises_but_an_already_disarmed_pr_is_fine():
    still_armed = FakeGh([_node(1)], disarm_error=True)
    with pytest.raises(mq.GhError):
        mq._evict(still_armed, _pr_state(), mq.Action("evict", "why", slug="x"), lambda m: None)
    already = FakeGh([_node(1)], disarm_error=True, reread={1: _node(1, armed=None)})
    assert mq._evict(already, _pr_state(), mq.Action("evict", "why", slug="x"), lambda m: None) is True


def test_wake_dispatches_the_kick_on_dev_and_nothing_else():
    gh = FakeGh([_node(1)])
    mq.wake(gh)
    mq.wake(gh, 99)
    assert gh.dispatched() == [
        ("dispatch", "security-required.yml", {"ref": "dev", "inputs[queue_kick]": "true"}),
        ("dispatch", "security-required.yml", {"ref": "dev", "inputs[queue_kick]": "true", "inputs[wait_run]": "99"}),
    ]


def test_wait_for_run_returns_once_the_run_completes_and_gives_up_after_the_bound():
    done = FakeGh([_node(1)], run_status={5: ["in_progress", "in_progress", "completed"]})
    mq.wait_for_run(done, 5, lambda s: None, lambda m: None)
    assert done.run_reads[5] == 3
    lines = []
    stuck = FakeGh([_node(1)], run_status={5: ["in_progress"]})
    mq.wait_for_run(stuck, 5, lambda s: None, lines.append)
    assert stuck.run_reads[5] == mq.WAIT_RUN_TRIES and "deciding anyway" in lines[-1]


@pytest.mark.parametrize(("error", "outcome"), [
    ("gh: The workflow run cannot be rerun (HTTP 422)", "refused"),
    ("gh: Conflict (HTTP 409)", "refused"),
    ("gh: Unable to re-run this workflow run because it was created over a month ago (HTTP 403)", "refused"),
])
def test_rerun_refusals(error, outcome):
    run = _run_of("e2e-required.yml", 70, "completed", "failure")
    gh = FakeGh([_node(1)], runs={OLD: [run]}, rerun_error=error)
    assert mq._rerun(gh, _pr_state(), run, True, lambda m: None, lambda s: None, lambda: NOW) == outcome


def test_a_rerun_answered_with_an_error_while_a_racing_run_re_ran_it_stands_down():
    run = _run_of("e2e-required.yml", 70, "completed", "failure")
    gh = FakeGh([_node(1)], runs={OLD: [run]}, rerun_error="gh: Conflict (HTTP 409)",
                run_status={70: ["in_progress"]})
    assert mq._rerun(gh, _pr_state(), run, True, lambda m: None, lambda s: None, lambda: NOW) == "started"


def test_a_held_rerun_is_approved():
    run = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    gh = FakeGh([_node(1)], runs={OLD: [run, _held(71, "e2e-required.yml")]},
                run_status={70: ["completed", "in_progress"]})
    assert mq._rerun(gh, _pr_state(), run, True, lambda m: None, lambda s: None, lambda: NOW) == "started"
    assert ("rerun", "70", "failed") in gh.calls and ("approve", "71") in gh.calls


def test_comment_once_removes_its_own_copy_when_a_racing_run_posted_first():
    posted = []

    class Racing(FakeGh):
        def rest(self, method, path, fields=None):
            if method == "POST" and path.endswith("/comments"):
                posted.append(fields["body"])
                return {"id": 50}
            if method == "DELETE" and "/issues/comments/" in path:
                self._record(("delete-comment", path.rsplit("/", 1)[1]))
                return None
            return super().rest(method, path, fields)

        def graphql(self, query, **v):
            if "comments(last" in query and posted:  # the second read sees the racer's copy, id 7
                nodes = [{"databaseId": 7, "body": posted[0], "createdAt": None},
                         {"databaseId": 50, "body": posted[0], "createdAt": None}]
                return {"data": {"repository": {"pullRequest": {"comments": {"nodes": nodes}}}}}
            return super().graphql(query, **v)

    gh = Racing([_node(1)])
    assert mq.comment_once(gh, 1, "rebased", NEW, "body", lambda m: None) is False
    assert ("delete-comment", "50") in gh.calls
```

- [ ] **Step B2: Run them to see them fail**

Run: `python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/CI/test_merge_queue_actions.py -k "test_only_the_queues_own_held_gate_runs_are_approvable or test_approve_held_runs_approves_only_the_queues_gate_runs or test_a_transient_approve_error_raises_and_disarms_nobody or test_a_refused_approval_of_a_run_no_longer_held_stands_down or test_a_refused_approval_strikes_then_evicts_after_the_gap or test_a_strike_inside_the_gap_neither_warns_again_nor_evicts or test_evict_comments_first_then_disarms or test_a_failed_disarm_of_a_pr_still_armed_raises_but_an_already_disarmed_pr_is_fine or test_wake_dispatches_the_kick_on_dev_and_nothing_else or test_wait_for_run_returns_once_the_run_completes_and_gives_up_after_the_bound or test_rerun_refusals or test_a_rerun_answered_with_an_error_while_a_racing_run_re_ran_it_stands_down or test_a_held_rerun_is_approved or test_comment_once_removes_its_own_copy_when_a_racing_run_posted_first"`
Expected: FAIL with `AttributeError` (`_queue_held`, `_strike`, `wake`, ... do not exist).

- [ ] **Step B3: Implement**

Replace the constants block from `# Who a run started with GITHUB_TOKEN belongs to` down to `WORKFLOW_REFUSAL = ...`:

```python
# A queue rebase (GITHUB_TOKEN) makes GitHub create the PR's pull_request runs held for approval
# (spec S2). Approving a gate's held run is what makes its check count (S4); a dispatched run's
# check never reaches the PR's rollup (S5). So the queue approves and never dispatches a gate.
QUEUE_ACTOR = "github-actions[bot]"
HELD_RUN_POLLS = 10
HELD_RUN_POLL_S = 3
# The workflow the queue dispatches to wake itself (spec 4.6, D3): GitHub starts a dispatch only
# for a workflow file that exists on main, and merge-queue.yml does not yet.
KICK_WORKFLOW = "security-required.yml"
# How a re-run GitHub answered with an error is read (`_rerun_error`): refused for good (409/422,
# or a 403 saying the run is over a month old), transient (`_transient`, re-raised so the next
# event retries; never disarms), or unknown (re-raised with a `rerun-error` warning, and counted
# as refused once that warning is STRIKE_GAP old).
RERUN_REFUSALS = ("HTTP 409", "HTTP 422")
RERUN_REFUSAL_403_TEXT = "month ago"
# How long a kick waits for the run that sent it to complete before deciding. The kick's job
# timeout is 15 minutes (as merge-queue.yml's); the wait plus one front's worst case (an UNKNOWN
# settle, a rebase's polls, re-runs of up to six gates) must fit.
# ponytail: a run still live after this is a GitHub fault; the kick decides anyway and the next event recovers.
WAIT_RUN_TRIES = 30
WAIT_RUN_DELAY_S = 6.0
# A queue run takes on no further front, and starts no young-head wait that would end, past this;
# it wakes a fresh run instead.
RUN_BUDGET = timedelta(minutes=4)
# A lasting failure to get a head going (a rebase that did not take effect, no run to start, an
# unclassified re-run error, a refused approval) is warned about once per head and evicts only
# once the warning is at least this old (spec section 7, errors).
STRIKE_GAP = timedelta(minutes=10)
# GitHub refuses to let an Actions token create or update a file under .github/workflows
# ("refusing to allow a GitHub App to create or update workflow ... without `workflows`
# permission"), and GITHUB_TOKEN can never hold that permission.
WORKFLOW_REFUSAL = ("workflow", "permission")
```

`BRANCH_REFUSALS` goes with that block; its only reader, `_dispatch_contexts`, is deleted in Part C.

Change `COMMENTS_QUERY`'s node selection to `nodes { databaseId body createdAt }`.

Add `CommentRefused` right after `GhError`:

```python
class CommentRefused(GhError):
    """GitHub will never take the queue's comment on this PR (its conversation is locked)."""
```

Replace `comment_once`, `_evict` and `dispatch` with the following block (keep `_best_effort` as it is; leave
`dispatch_request` and `_dispatch_contexts` for Part C to delete). This block is the port; keep its comments:

```python
def _transient(exc: GhError) -> bool:
    """Whether an error is GitHub's or the network's, saying nothing about the PR.

    Matches what gh prints: a REST error `gh: <message> (HTTP 502)`, a non-JSON one `gh: HTTP 502`,
    a GraphQL timeout `gh: Something went wrong while executing your query...`, rate limits and
    abuse detection, and a network failure, where gh prints no `gh: ` server message at all
    (e.g. `Post "https://api.github.com/graphql": EOF`).

    Args:
        exc: The error a call raised.

    Returns:
        True for an outage or throttling error; False for GitHub's answer about this PR.
    """
    message = str(exc).split(" failed: ", 1)[-1]
    lowered = message.lower()
    # ponytail: "no `gh: ` line" = network. Should gh ever drop that prefix, every error reads as
    # transient: queue runs fail visibly and nothing is disarmed; match the new shape then.
    server_answered = any(line.startswith("gh: ") for line in message.splitlines())
    return ("HTTP 5" in message or "HTTP 429" in message or "rate limit" in lowered or "abuse detection" in lowered
            or "something went wrong while executing your query" in lowered or not server_answered)


def has_comment(gh: GhApi, number: int, kind: str, sha: str) -> bool:
    """Whether the queue already posted a comment of this kind for this head.

    Raises:
        GhError: The read failed.
    """
    return marker_age(gh, number, kind, sha, datetime.now(timezone.utc)) is not None


def marker_age(gh: GhApi, number: int, kind: str, sha: str, now: datetime) -> timedelta | None:
    """How long ago the queue posted a comment of this kind for this head.

    Returns:
        Its age, `timedelta.max` if it has no timestamp, or None if there is no such comment.

    Raises:
        GhError: The read failed.
    """
    marker = f"<!-- merge-queue:{kind}:{sha} -->"
    owner, name = _owner_name()
    data = gh.graphql(COMMENTS_QUERY, owner=owner, name=name, number=number)
    for node in data["data"]["repository"]["pullRequest"]["comments"]["nodes"]:
        if marker in (node.get("body") or ""):
            posted = _ts(node.get("createdAt"))
            return timedelta.max if posted is None else now - posted
    return None


def comment_once(gh: GhApi, number: int, kind: str, sha: str, body: str,
                 log: Callable[[str], None] = print) -> bool:
    """Post a comment unless one of this kind already exists for this head.

    Returns:
        True if it posted, False if the marker was already there (or a racing queue run posted
        the same comment first, in which case this run's copy is removed).

    Raises:
        CommentRefused: The PR's conversation is locked.
        GhError: A call failed.
    """
    marker = f"<!-- merge-queue:{kind}:{sha} -->"
    owner, name = _owner_name()
    data = gh.graphql(COMMENTS_QUERY, owner=owner, name=name, number=number)
    bodies = [n.get("body") or "" for n in data["data"]["repository"]["pullRequest"]["comments"]["nodes"]]
    if any(marker in b for b in bodies):
        return False
    try:
        posted = gh.rest("POST", f"repos/{REPO}/issues/{number}/comments", {"body": f"{marker}\n{body}"})
    except GhError as exc:
        if "is locked" in str(exc).lower() and not _transient(exc):
            raise CommentRefused(str(exc)) from exc
        raise
    # Two racing queue runs can both pass the check above and both post. Each then looks again and
    # the later one removes its own copy, so one comment remains.
    own_id = posted.get("id") if isinstance(posted, dict) else None
    if own_id is None:
        return True
    try:
        again = gh.graphql(COMMENTS_QUERY, owner=owner, name=name, number=number)
        ids = [n["databaseId"] for n in again["data"]["repository"]["pullRequest"]["comments"]["nodes"]
               if marker in (n.get("body") or "") and n.get("databaseId") is not None]
        if ids and min(ids) < own_id:
            gh.rest("DELETE", f"repos/{REPO}/issues/comments/{own_id}")
            return False
    except GhError as exc:
        # The comment is posted; tidying up must never stop what the caller does next. At worst two copies stay.
        log(f"  best-effort comment de-duplication on #{number} failed: {exc}")
    return True


def _comment_if_allowed(gh: GhApi, number: int, kind: str, sha: str, log: Callable[[str], None], body: str) -> None:
    try:
        comment_once(gh, number, kind, sha, body, log)
    except CommentRefused as exc:
        log(f"  cannot comment on #{number} ({exc}); skipped")


def _disarm(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    try:
        gh.graphql(DISARM_MUTATION, id=pr.node_id)
    except GhError as exc:
        # Two racing runs can evict the same PR (a merge fires both push:dev and
        # pull_request:closed), and the second disarm hits an already-disarmed PR. Any other
        # failure must not leave the PR armed under a "removed" comment: fail the run, and the
        # next one evicts it again.
        if read_pr(gh, pr.number).armed_at is not None:
            raise
        log(f"  #{pr.number} was already disarmed ({exc})")


def _evict(gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None]) -> bool:
    """Tell the PR why, then disarm it; always returns True (it left the line)."""
    # Comment first: if the job dies between the two calls the PR stays armed and the next run
    # evicts it again (the comment is deduplicated), instead of being disarmed with no reason given.
    links = "".join(f"\n- {u}" for u in action.links)
    comment_once(
        gh, pr.number, f"evict-{action.slug}", pr.head_sha,
        f"Merge queue: removed from the line ({action.reason}). Auto-merge is now off. Fix the cause, then "
        f"re-arm with `gh pr merge {pr.number} --auto --merge` to rejoin at the back.{links}",
        log,
    )
    _disarm(gh, pr, log)
    log(f"  evicted #{pr.number}: {action.reason}")
    return True


def _strike(gh: GhApi, pr: PrState, kind: str, warning: str, eviction: Action, wake_first: bool,
            log: Callable[[str], None], now: Callable[[], datetime]) -> bool:
    """Count one failure to get this head going (see STRIKE_GAP).

    Args:
        gh: The GitHub client.
        pr: The front PR.
        kind: The warning comment's kind (`rebase-failed`, `rebase-unmoved`, `no-run`, `approve-error`).
        warning: The warning's text.
        eviction: The eviction for a strike at least STRIKE_GAP after the warning.
        wake_first: Wake a queue kick after the warning, to look again soon.
        log: Receives one line per notable step.
        now: The clock.

    Returns:
        True if the PR was evicted.

    Raises:
        GhError: A call failed.
    """
    age = marker_age(gh, pr.number, kind, pr.head_sha, now())
    if age is None:
        comment_once(gh, pr.number, kind, pr.head_sha, warning, log)
        if wake_first:
            _wake_best_effort(gh, log)
        return False
    if age < STRIKE_GAP:
        log(f"  #{pr.number}: {kind} again within {STRIKE_GAP}; a later event decides")
        return False
    return _evict(gh, pr, eviction, log)


def dispatch(gh: GhApi, workflow: str, ref: str, inputs: dict[str, str] | None = None) -> None:
    """Start a workflow_dispatch run. Only `dispatch_license` and `wake` call this (guard-tested).

    Raises:
        GhError: GitHub refused the dispatch.
    """
    fields = {"ref": ref}
    for key, value in (inputs or {}).items():
        fields[f"inputs[{key}]"] = value
    gh.rest("POST", f"repos/{REPO}/actions/workflows/{workflow}/dispatches", fields)


def dispatch_license(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    """Start dev's copy of the license gate for the PR (spec 4.4): the one context the queue dispatches.

    Any error re-raises: the dispatch runs on dev, so a refusal is never the PR's fault.
    """
    dispatch(gh, LICENSE.workflow, BASE, {"pr": str(pr.number)})
    log(f"  dispatched {LICENSE.workflow} on {BASE} for #{pr.number}")


def wake(gh: GhApi, wait_run: int | None = None) -> None:
    """Start a queue kick: `security-required.yml` on dev with `queue_kick=true` (spec 4.6, D3).

    Its gate jobs skip and only its `queue-kick` job runs, which waits for `wait_run` (if given)
    and then runs the queue. GitHub keeps one kick running and one pending; a newer pending kick
    replaces an older pending one, which is enough because every kick decides from fresh state.

    Raises:
        GhError: The dispatch failed.
    """
    inputs = {"queue_kick": "true"}
    if wait_run is not None:
        inputs["wait_run"] = str(wait_run)
    dispatch(gh, KICK_WORKFLOW, BASE, inputs)


def _wake_best_effort(gh: GhApi, log: Callable[[str], None]) -> None:
    _best_effort(log, "wake a queue kick", lambda: wake(gh))


def wait_for_run(gh: GhApi, run_id: int, sleep: Callable[[float], None], log: Callable[[str], None]) -> None:
    """Wait, bounded, for a workflow run to complete; a run still live after the bound is left."""
    error: GhError | None = None
    for _ in range(WAIT_RUN_TRIES):
        try:
            if gh.rest("GET", f"repos/{REPO}/actions/runs/{run_id}").get("status") not in LIVE_RUN_STATUSES:
                return
            error = None
        except GhError as exc:  # a failed read is retried within the bound, never fatal
            error = exc
        sleep(WAIT_RUN_DELAY_S)
    seen = f"last read failed: {error}" if error else "still live"
    log(f"run {run_id} not seen complete after {WAIT_RUN_TRIES * WAIT_RUN_DELAY_S:.0f}s ({seen}); deciding anyway")


def _is_held(run: dict) -> bool:
    """Whether GitHub holds this run for approval (it lists as completed, `action_required`)."""
    return "action_required" in (run.get("status"), run.get("conclusion"))


def _queue_held(run: dict) -> bool:
    """Whether this is a held run the queue may approve (spec section 7, "What may be approved").

    A gate workflow's pull_request run that the queue's own token caused, on this repository's
    branch. Never a non-gate workflow (D1: `ci.yml` alone is 200+ jobs), never a run from another
    repository (GitHub's approval gate for outside contributors), never one another actor caused,
    such as an agent's push waiting for a person.
    """
    return (_is_held(run) and run.get("event") == "pull_request"
            and _workflow_name(run) in GATE_WORKFLOWS
            and (run.get("triggering_actor") or {}).get("login") == QUEUE_ACTOR
            and (run.get("head_repository") or {}).get("full_name") == REPO)


def approve_held_runs(gh: GhApi, sha: str, log: Callable[[str], None]) -> set[str]:
    """Approve the gate runs GitHub holds on this head because the queue's token caused them.

    Best-effort per run: one refused approval must not stop the others.

    Returns:
        The workflow names whose held runs were approved.
    """
    try:
        held = runs_on(gh, sha, status="action_required")
    except GhError as exc:
        log(f"  best-effort list held runs failed: {exc}")
        return set()
    approved: set[str] = set()
    for run in held:
        if not _queue_held(run):
            continue
        try:
            gh.rest("POST", f"repos/{REPO}/actions/runs/{run['id']}/approve")
        except GhError as exc:
            log(f"  best-effort approve run {run['id']} failed: {exc}")
            continue
        approved.add(_workflow_name(run))
        log(f"  approved held run {run['id']} ({_workflow_name(run)})")
    return approved


def _approve_strictly(gh: GhApi, pr: PrState, run: dict, log: Callable[[str], None],
                      now: Callable[[], datetime]) -> bool:
    """Approve one held gate run a start or retry depends on.

    Not best-effort: a transient error re-raises (the run fails; nobody is disarmed). Any other
    refusal is re-read first: a run no longer held was approved by a racing queue run. One still
    held is an `approve-error` strike.

    Returns:
        True if the PR was evicted.

    Raises:
        GhError: A transient error, or a failed re-read.
    """
    try:
        gh.rest("POST", f"repos/{REPO}/actions/runs/{run['id']}/approve")
    except GhError as exc:
        if _transient(exc):
            raise
        if not _is_held(gh.rest("GET", f"repos/{REPO}/actions/runs/{run['id']}")):
            log(f"  run {run['id']} is no longer held; a racing queue run approved it")
            return False
        error = str(exc)[:200]
        return _strike(
            gh, pr, "approve-error",
            f"Merge queue: GitHub refused to approve one of this PR's gate runs ({error}). It will try again, and "
            "remove it from the line if GitHub still refuses 10 minutes from now.",
            Action("evict", f"GitHub keeps refusing to approve its gate run: {error}", slug="approve"), False, log, now,
        )
    log(f"  approved held run {run['id']} ({_workflow_name(run)})")
    return False


def _going_again(gh: GhApi, run_id: int) -> bool:
    """Whether a run is live or held again, i.e. a racing queue run has re-run it since we looked."""
    fresh = gh.rest("GET", f"repos/{REPO}/actions/runs/{run_id}")
    return fresh.get("status") in LIVE_RUN_STATUSES or _is_held(fresh)


def _rerun_error(exc: GhError) -> str:
    """Classify a re-run error as `refused`, `transient` or `unknown` (see RERUN_REFUSALS)."""
    text = str(exc)
    if any(code in text for code in RERUN_REFUSALS) or ("HTTP 403" in text and RERUN_REFUSAL_403_TEXT in text.lower()):
        return "refused"
    if _transient(exc):
        return "transient"
    return "unknown"


def _rerun(gh: GhApi, pr: PrState, run: dict, failed_only: bool, log: Callable[[str], None],
           sleep: Callable[[float], None], now: Callable[[], datetime]) -> str:
    """Re-run a gate run in place, then approve the re-run if GitHub holds it.

    A re-run adds an attempt to the same check suite, so it replaces the old result for branch
    protection while `read_checks` (filter=all) still lists the old one.

    Returns:
        `started` (by this run, or by a racing queue run), `pending` (re-run, not live after the
        approval polls: the caller wakes a kick once it has recorded the attempt), or `refused`.

    Raises:
        GhError: A transient error, or an unknown one before its warning is STRIKE_GAP old.
    """
    path = f"repos/{REPO}/actions/runs/{run['id']}/{'rerun-failed-jobs' if failed_only else 'rerun'}"
    try:
        gh.rest("POST", path)
    except GhError as exc:
        kind = _rerun_error(exc)
        if kind == "transient":
            raise
        # Whatever code GitHub refused with, a racing queue run may have re-run it first.
        if _going_again(gh, run["id"]):
            log(f"  #{pr.number}: run {run['id']} was re-run by a racing queue run; standing down")
            return "started"
        if kind == "unknown":
            age = marker_age(gh, pr.number, "rerun-error", pr.head_sha, now())
            if age is None:
                comment_once(
                    gh, pr.number, "rerun-error", pr.head_sha,
                    f"Merge queue: re-running a required gate failed ({str(exc)[:200]}); will try again, and "
                    "remove it from the line if it still fails 10 minutes from now.",
                    log,
                )
                raise
            if age < STRIKE_GAP:
                raise
        log(f"  #{pr.number}: re-run of run {run['id']} refused: {exc}")
        return "refused"
    # The re-run's actor is the queue's token, so GitHub may hold it for approval again.
    for _ in range(HELD_RUN_POLLS):
        sleep(HELD_RUN_POLL_S)
        approve_held_runs(gh, pr.head_sha, log)
        if gh.rest("GET", f"repos/{REPO}/actions/runs/{run['id']}").get("status") in LIVE_RUN_STATUSES:
            log(f"  #{pr.number}: re-ran run {run['id']}")
            return "started"
    log(f"  #{pr.number}: re-ran run {run['id']}; not live yet")
    return "pending"
```

Note `has_comment` above reads with the wall clock only to reuse `marker_age`; it never compares the age.

- [ ] **Step B4: Run the Part B tests**

Run the command of Step B2. Expected: PASS.

#### Part C: The rebase path

Spec section 6, `BEHIND` row: rebase, wait for the new head, dispatch the license gate, approve the six held gate runs,
delete the other held runs, cancel the old head's live runs, comment once.

**Files:**
- Modify: `Helper_Scripts/ci/merge_queue.py` (`_rebase`, `cleanup_approval_runs`; new `_wake_if_moved`,
  `_approve_gates`; delete `_dispatch_contexts` and `dispatch_request`)
- Test: `tldw_Server_API/tests/CI/test_merge_queue_actions.py`

**Interfaces:**
- Consumes: Part B primitives.
- Produces: `_rebase(gh, pr, log, sleep, now: Callable[[], datetime]) -> bool`;
  `cleanup_approval_runs(gh, pr, log) -> None` (never deletes a gate's held run).

- [ ] **Step C1: Write the failing tests**

```python
def test_rebase_dispatches_the_license_then_approves_exactly_the_six_gates_then_cleans_up_then_cancels():
    held, ci = _six_held(), _held(40, "ci.yml")
    old_live, queue_run = _run_of("backend-required.yml", 11), _run_of("merge-queue.yml", 12)
    gh = FakeGh([_node(1)], runs={OLD: [old_live, queue_run], NEW: held + [ci]})
    assert _run(gh)[0][1].kind == "rebase"
    calls = gh.calls
    assert calls[0] == ("rebase", "PR_1", OLD)
    license_at = calls.index(("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "1"}))
    approvals = [i for i, c in enumerate(calls) if c[0] == "approve"]
    assert sorted(calls[i][1] for i in approvals) == sorted(str(r["id"]) for r in held)
    assert license_at < min(approvals), "spec 4.4: the license is dispatched before any gate is approved"
    assert ("approve", "40") not in calls and ("delete", "40") in calls, "D1: ci.yml's held run is deleted"
    assert not any(c[0] == "delete" and c[1] in {str(r["id"]) for r in held} for c in calls)
    assert calls.index(("cancel", "11")) > max(approvals)
    assert ("cancel", "12") not in calls, "a merge-queue.yml run on the old head is never cancelled"
    assert calls[-1][0] == "comment" and f"merge-queue:rebased:{NEW}" in calls[-1][2]
    assert len(gh.dispatched()) == 1, "nothing else is dispatched: all six gates were approved"


def test_a_rebase_whose_gates_are_not_all_held_in_time_wakes_a_kick():
    gh = FakeGh([_node(1)], runs={NEW: _six_held()[:4]})
    _run(gh)
    assert ("dispatch", "security-required.yml", {"ref": "dev", "inputs[queue_kick]": "true"}) in gh.calls
    assert "approved 4 of its 6 required gates" in gh.calls[-1][2]


def test_rebase_approves_only_after_the_new_head_appears():
    gh = FakeGh([_node(1)], runs={NEW: _six_held()}, rebase_lag=2)
    _run(gh)
    first_approve = next(i for i, e in enumerate(gh.events) if e[0] == "approve")
    assert ("read_pr", NEW) in gh.events[:first_approve]


def test_a_rebase_whose_head_never_moves_strikes_and_wakes():
    gh = FakeGh([_node(1)], rebase_lag=None)
    _run(gh)
    assert [c[0] for c in gh.calls] == ["rebase", "comment", "dispatch"]
    assert f"merge-queue:rebase-unmoved:{OLD}" in gh.calls[1][2]
    assert gh.calls[2][1] == "security-required.yml"
    warned = (f"<!-- merge-queue:rebase-unmoved:{OLD} -->\nx", "2026-10-04T11:40:00Z")
    later = FakeGh([_node(1)], rebase_lag=None, comments={1: [warned]})
    _run(later)
    assert [c[0] for c in later.calls][-2:] == ["comment", "disarm"]


def test_a_transient_rebase_error_with_the_head_moved_stands_down_and_with_it_unmoved_fails_the_run():
    error = "gh api graphql failed: gh: Something went wrong while executing your query. (HTTP 200)"
    moved = FakeGh([_node(1)], rebase_error=True, rebase_error_text=error, reread={1: _node(1, head=NEW)})
    _run(moved)
    assert not any(c[0] in ("disarm", "comment") for c in moved.calls)
    unmoved = FakeGh([_node(1)], rebase_error=True, rebase_error_text=error)
    with pytest.raises(mq.GhError):
        _run(unmoved)
    assert not any(c[0] in ("disarm", "comment") for c in unmoved.calls)


def test_repeated_rebase_failure_evicts_only_after_the_strike_gap():
    refused = "gh: Unprocessable (HTTP 422)"
    first = FakeGh([_node(1)], rebase_error=True, rebase_error_text=refused)
    _run(first)
    assert [c[0] for c in first.calls] == ["rebase", "comment"]
    soon = FakeGh([_node(1)], rebase_error=True, rebase_error_text=refused,
                  comments={1: [(f"<!-- merge-queue:rebase-failed:{OLD} -->\nx", "2026-10-04T11:58:00Z")]})
    _run(soon)
    assert [c[0] for c in soon.calls] == ["rebase"]
    late = FakeGh([_node(1)], rebase_error=True, rebase_error_text=refused,
                  comments={1: [(f"<!-- merge-queue:rebase-failed:{OLD} -->\nx", "2026-10-04T11:45:00Z")]})
    _run(late)
    assert [c[0] for c in late.calls] == ["rebase", "comment", "disarm"]


def test_cleanup_deletes_only_the_queues_non_gate_held_runs():
    runs = [_held(31, "backend-required.yml"), _held(40, "ci.yml"), _held(41, "pre-commit.yml", actor="person")]
    gh = FakeGh([_node(1)], runs={OLD: runs})
    mq.cleanup_approval_runs(gh, _pr_state(), lambda m: None)
    assert gh.calls == [("delete", "40")]


def test_a_failure_after_the_rebase_took_effect_wakes_a_kick_and_fails_the_run():
    """Review improvement I5: the new head's gate runs are held, and nothing else would wake the queue
    to approve them if this run dies after the branch moved."""
    gh = FakeGh([_node(1)], dispatch_refused={LICENSE_GATE}, dispatch_status=502)
    with pytest.raises(mq.GhError):
        _run(gh)
    assert [c[1] for c in gh.dispatched()] == [LICENSE_GATE, "security-required.yml"]
    assert not any(c[0] in ("disarm", "comment") for c in gh.calls)
```

Existing tests this part breaks are fixed in Part E, from its list.

- [ ] **Step C2: Run them to see them fail**

Run: `python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/CI/test_merge_queue_actions.py -k "test_rebase_dispatches_the_license_then_approves_exactly_the_six_gates_then_cleans_up_then_cancels or test_a_rebase_whose_gates_are_not_all_held_in_time_wakes_a_kick or test_rebase_approves_only_after_the_new_head_appears or test_a_rebase_whose_head_never_moves_strikes_and_wakes or test_a_transient_rebase_error_with_the_head_moved_stands_down_and_with_it_unmoved_fails_the_run or test_repeated_rebase_failure_evicts_only_after_the_strike_gap or test_cleanup_deletes_only_the_queues_non_gate_held_runs or test_a_failure_after_the_rebase_took_effect_wakes_a_kick_and_fails_the_run"`
Expected: FAIL (the old `_rebase` dispatches; `cleanup_approval_runs` deletes gate runs).

- [ ] **Step C3: Implement**

Delete `_dispatch_contexts`, `dispatch_request` and `_host_workflow` (it existed only for the "host workflow last" rule,
which goes with the dispatches; review finding m5). `apply`'s dispatch/retry branch still calls `_dispatch_contexts`
until Part D replaces `apply`. Replace `_rebase` and `cleanup_approval_runs`; add the two helpers:

```python
def _wake_if_moved(gh: GhApi, pr: PrState, sleep: Callable[[float], None]) -> bool:
    """Wake a kick if the PR's head moved; return whether it did.

    Raises:
        GhError: The re-read or the wake failed. The caller then treats the head as unmoved, so a
            wake that never went out is not mistaken for a stand-down.
    """
    sleep(REBASE_POLL_S)  # an accepted rebase moves the branch about 1 s after the mutation returns
    if read_pr(gh, pr.number).head_sha == pr.head_sha:
        return False
    wake(gh)
    return True


def _approve_gates(gh: GhApi, sha: str, log: Callable[[str], None], sleep: Callable[[float], None]) -> set[str]:
    """Approve the six held gate runs on a fresh head, waiting (bounded) for them to appear."""
    approved: set[str] = set()
    for attempt in range(HELD_RUN_POLLS):
        if attempt:
            sleep(HELD_RUN_POLL_S)
        approved |= approve_held_runs(gh, sha, log)
        if GATE_WORKFLOWS <= approved:
            break
    return approved


def _rebase(gh: GhApi, pr: PrState, log: Callable[[str], None], sleep: Callable[[float], None],
            now: Callable[[], datetime]) -> bool:
    """Rebase the front PR onto dev with its head pinned, then get its seven contexts going (spec section 6).

    Returns:
        True if the PR was evicted, False otherwise.

    Raises:
        GhError: A transient error with the head unmoved, or the license dispatch failed.
    """
    try:
        gh.graphql(REBASE_MUTATION, id=pr.node_id, oid=pr.head_sha)
    except GhError as exc:
        if _transient(exc):
            # Each merge wakes two queue runs, and the one that loses the rebase race is answered
            # with a generic error; a lost response looks the same. With the head moved, nothing
            # failed: stand down. With it unmoved it is GitHub's error: fail the run so the next
            # event retries; it never counts toward an eviction.
            try:
                moved = _wake_if_moved(gh, pr, sleep)
            except GhError as read_exc:
                log(f"  re-read or wake after a failed rebase failed: {read_exc}")
                moved = False
            if moved:
                log(f"  rebase of #{pr.number} was answered with an error, but its head moved; standing down")
                return False
            raise
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha == pr.head_sha and fresh.merge_state != "DIRTY":
            # A racing run's rebase may have been accepted with its ref update still landing.
            sleep(REBASE_POLL_S)
            fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            log(f"  rebase skipped: head moved to {fresh.head_sha[:10]}")
            return False
        if fresh.merge_state == "DIRTY":
            return _evict(gh, fresh, Action("evict", "conflicts with dev (rebase refused)", slug="conflict"), log)
        if all(word in str(exc).lower() for word in WORKFLOW_REFUSAL):
            reason = (
                "dev changed workflow files since this branch was cut, and the queue's token is not allowed to "
                "rebase across them; rebase onto dev by hand (`git fetch origin dev && git rebase origin/dev`, "
                "then force-push with lease)"
            )
            return _evict(gh, fresh, Action("evict", reason, slug="workflows"), log)
        error = str(exc)[:200]
        log(f"  rebase failed: {exc}")
        return _strike(
            gh, fresh, "rebase-failed",
            f"Merge queue: rebasing onto dev failed ({error}); will retry, and remove it from the line if "
            "rebasing still fails 10 minutes from now.",
            Action("evict", f"rebase onto dev keeps failing: {error}", slug="rebase"), False, log, now,
        )
    # The mutation returns the PRE-rebase headRefOid and the branch moves about 1 s later.
    rebased = None
    for _ in range(REBASE_POLLS):
        sleep(REBASE_POLL_S)
        fresh = read_pr(gh, pr.number)
        if fresh.head_sha != pr.head_sha:
            rebased = fresh
            break
    if rebased is None:
        log(f"  rebase of #{pr.number} accepted but the head never moved")
        # If the branch moves later its held runs need a queue run to approve them: the warning
        # wakes a kick. A later unmoved rebase evicts, or every event would rebase again.
        return _strike(
            gh, pr, "rebase-unmoved",
            "Merge queue: the rebase onto dev was accepted, but the branch has not moved yet; a queue run will "
            "look again shortly, and remove it from the line if the branch still does not move 10 minutes from now.",
            Action("evict", "rebase onto dev was accepted but the branch never moved, twice at least 10 minutes apart",
                   slug="rebase-unmoved"), True, log, now,
        )
    try:
        old_runs = runs_on(gh, pr.head_sha)
        # Spec 4.4: the license status first, so each gate's await_license finds it (when
        # LICENSE_FIRST_CI_ENABLED is true). Then the gates: approving their held runs is what makes
        # their checks count (S4). Then the other held runs go (D1), then the old head's live runs.
        dispatch_license(gh, rebased, log)
        approved = _approve_gates(gh, rebased.head_sha, log, sleep)
    except GhError:
        # The branch has moved and its gate runs are held: nothing else would wake the queue to
        # approve them. The kick finds an up-to-date head and approves, so this cannot chain.
        _wake_best_effort(gh, log)
        raise
    cleanup_approval_runs(gh, rebased, log)
    for run in old_runs:
        if run.get("status") not in LIVE_RUN_STATUSES:
            continue
        if run.get("id") == _own_run_id() or _workflow_name(run) == QUEUE_WORKFLOW:
            continue
        _best_effort(log, f"cancel run {run['id']}",
                     lambda rid=run["id"]: gh.rest("POST", f"repos/{REPO}/actions/runs/{rid}/cancel"))
    total = len(GATE_WORKFLOWS)
    if len(approved) < total:
        # The rest are still held, and a held run never runs its own queue-tick: wake a kick.
        _wake_best_effort(gh, log)
        started = f"approved {len(approved)} of its {total} required gates; a woken queue run approves the rest"
    else:
        started = f"approved its {total} required gates"
    _comment_if_allowed(
        gh, pr.number, "rebased", rebased.head_sha, log,
        f"Merge queue: this PR is next. Rebased onto `{BASE}` (head `{rebased.head_sha[:10]}`); dispatched the "
        f"license gate and {started}.",
    )
    log(f"  rebased #{pr.number} {pr.head_sha[:10]} -> {rebased.head_sha[:10]}")
    return False


def cleanup_approval_runs(gh: GhApi, pr: PrState, log: Callable[[str], None]) -> None:
    """Delete the non-gate runs GitHub holds on the front head because the queue's token caused them (D1).

    A queue rebase makes GitHub hold a run for every pull_request workflow; only the six gates are
    approved. Deleting the others removes the "Approve and run" button that would start `ci.yml`'s
    200+ jobs on a click. A gate's held run is never deleted here. Best-effort end to end.
    """
    try:
        runs = runs_on(gh, pr.head_sha, status="action_required")
    except GhError as exc:
        log(f"  best-effort list approval-pending runs failed: {exc}")
        return
    for run in runs:
        if (run.get("triggering_actor") or {}).get("login") != QUEUE_ACTOR or _workflow_name(run) in GATE_WORKFLOWS:
            continue
        _best_effort(log, f"delete approval-pending run {run['id']}",
                     lambda rid=run["id"]: gh.rest("DELETE", f"repos/{REPO}/actions/runs/{rid}"))
```

In `apply`, change the rebase branch to `return _rebase(gh, pr, log, sleep, now)` (Part D changes `apply`'s `now` to a
callable; until then pass `lambda: now`).

- [ ] **Step C4: Run the Part C tests**

Run the command of Step C2. Expected: PASS.

#### Part D: Start, retry, the decision step and the run loop

**Files:**
- Modify: `Helper_Scripts/ci/merge_queue.py` (`Action`, `decide_front`, `apply`, `run`, `main`; new `_gate_runs`,
  `_start`, `_retry`, `_retry_gate`, `_decide`; `UNQUEUED_NOTES`)
- Test: `tldw_Server_API/tests/CI/test_merge_queue_rules.py`, `tldw_Server_API/tests/CI/test_merge_queue_actions.py`

**Interfaces:**
- Consumes: Parts A-C.
- Produces: action kinds `wait`, `rebase`, `start`, `retry`, `evict` (slug `young` on the young-head wait);
  `apply(gh, pr, action, log, sleep, now: Callable[[], datetime]) -> bool`;
  `run(gh, mode, now=..., sleep=..., log=..., wait_run: int | None = None) -> list[tuple[int, Action]]`;
  `main()` reads `WAIT_RUN` (digits only).

- [ ] **Step D1: Write the failing tests**

In `test_merge_queue_rules.py`, in `test_decide_front_table` change every expected `"dispatch"` to `"start"` and add
three rows (with ids `future-dated-head-starts`, `slightly-ahead-head-waits` and the existing young row keeps `wait`):

```python
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E,)), head_committed_at=NOW + timedelta(minutes=5)),
         "start", (E2E,)),
        (_pr(merge_state="BLOCKED", checks=_green(without=(E2E,)), head_committed_at=NOW + timedelta(seconds=30)),
         "wait", (E2E,)),
```

and add:

```python
def test_a_young_head_wait_is_marked_young() -> None:
    pr = _pr(merge_state="BLOCKED", checks=_green(without=(E2E,)), head_committed_at=NOW - timedelta(minutes=1))
    assert mq.decide_front(pr, NOW).slug == "young"
```

In `test_merge_queue_actions.py`:

```python
def test_decide_approves_held_gate_runs_first_so_they_read_as_running():
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_held(35, "e2e-required.yml")]}, **_green({E2E: None}))
    decision = _run(gh)[0][1]
    assert ("approve", "35") in gh.calls
    assert (decision.kind, decision.contexts) == ("wait", (E2E,))


def test_decide_never_approves_a_non_gate_held_run():
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_held(40, "ci.yml")]}, **_green())
    _run(gh)
    assert ("approve", "40") not in gh.calls


def test_start_restarts_a_cancelled_run_of_a_missing_gate_once_then_evicts():
    """Review finding m1: a cancelled run says nothing about the PR, so it is restarted in full, once per
    gate per head; a second cancellation on the same head evicts instead of restarting forever."""
    def cancelled():
        return _run_of("e2e-required.yml", 70, "completed", "cancelled", created_at="2026-10-04T11:00:00Z")
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [cancelled()]}, run_status={70: ["in_progress"]},
                **_green({E2E: None}))
    assert _run(gh)[0][1].kind == "start"
    assert ("rerun", "70", "all") in gh.calls
    assert gh.calls[-1][0] == "comment" and f"merge-queue:restart-{E2E}:{OLD}" in gh.calls[-1][2]
    assert not gh.dispatched(), "a gate is never dispatched"
    again = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [cancelled()]},
                   comments={1: [f"<!-- merge-queue:restart-{E2E}:{OLD} -->\nx"]}, **_green({E2E: None}))
    _run(again)
    assert [c[0] for c in again.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-rerun:{OLD}" in again.calls[0][2]


def test_start_restarts_a_run_whose_gate_job_was_cancelled_whatever_the_run_concluded():
    """Review point n1: a run can conclude `failure` (another job failed) while its gate job was
    cancelled; the context is missing, and the run is restarted once like a cancelled run."""
    run = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "cancelled", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [run]}, checks={OLD: checks},
                statuses={OLD: [_status()]}, run_status={70: ["in_progress"]})
    assert _run(gh)[0][1].kind == "start"
    assert ("rerun", "70", "all") in gh.calls


def test_start_approves_a_held_gate_run_the_decide_step_could_not_approve():
    pr = _pr_state()
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [_held(35, "e2e-required.yml")]})
    assert mq._start(gh, pr, (E2E,), lambda m: None, lambda s: None, lambda: NOW) is False
    assert gh.calls == [("approve", "35")]


def test_start_dispatches_a_missing_license_on_dev_and_nothing_for_the_gates_that_have_runs():
    gh = FakeGh([_node(1, state="BLOCKED")], **_green(statuses=[]))
    assert _run(gh)[0][1].contexts == (LICENSE,)
    assert gh.dispatched() == [("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "1"})]


def test_start_with_no_run_at_all_warns_then_evicts_after_the_gap():
    first = FakeGh([_node(1, state="BLOCKED")], **_green({E2E: None}))
    _run(first)
    assert [c[0] for c in first.calls] == ["comment"] and f"merge-queue:no-run:{OLD}" in first.calls[0][2]
    warned = (f"<!-- merge-queue:no-run:{OLD} -->\nx", "2026-10-04T11:45:00Z")
    later = FakeGh([_node(1, state="BLOCKED")], comments={1: [warned]}, **_green({E2E: None}))
    _run(later)
    assert [c[0] for c in later.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-no-run:{OLD}" in later.calls[0][2]


def test_first_failure_re_runs_the_failed_jobs_of_that_gates_run_and_comments_once():
    failed = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed]}, checks={OLD: checks},
                statuses={OLD: [_status()]}, run_status={70: ["in_progress"]})
    assert _run(gh)[0][1].kind == "retry"
    assert ("rerun", "70", "failed") in gh.calls
    assert gh.calls[-1][0] == "comment" and f"merge-queue:retry-{E2E}:{OLD}" in gh.calls[-1][2]
    assert not gh.dispatched()


def test_a_broken_runs_second_retry_on_the_same_head_evicts():
    broken = _run_of("security-required.yml", 70, "completed", "startup_failure", check_suite_id=700)
    marker = f"<!-- merge-queue:retry-{SECURITY}:{OLD} -->\nx"
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [broken]}, comments={1: [marker]},
                **_green({SECURITY: None}))
    _run(gh)
    assert [c[0] for c in gh.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-failed-twice:{OLD}" in gh.calls[0][2]


def test_a_green_gate_run_without_its_check_is_retried_once_then_evicted():
    """Review finding M2: a PR whose own workflow copy renamed its gate job finishes green with no check.
    That is one failure: re-run once in full, then evicted, never re-run on every wake."""
    def silent():
        return _run_of("e2e-required.yml", 70, "completed", "success", check_suite_id=700)
    first = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [silent()]}, run_status={70: ["in_progress"]},
                   **_green({E2E: None}))
    assert _run(first)[0][1].kind == "retry"
    assert ("rerun", "70", "all") in first.calls
    marker = f"<!-- merge-queue:retry-{E2E}:{OLD} -->\nx"
    second = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [silent()]}, comments={1: [marker]},
                    **_green({E2E: None}))
    _run(second)
    assert [c[0] for c in second.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-failed-twice:{OLD}" in second.calls[0][2]


def test_a_retry_attempt_that_ends_cancelled_is_restarted_once_then_evicted():
    """Review finding m1: a retry that keeps ending cancelled (a gate hitting its timeout) must not be
    re-run on every wake."""
    def state(comments):
        run = _run_of("e2e-required.yml", 70, "completed", "cancelled", check_suite_id=700)
        checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
        return FakeGh([_node(1, state="BLOCKED")], runs={OLD: [run]}, checks={OLD: checks},
                      statuses={OLD: [_status()]}, comments={1: comments}, run_status={70: ["in_progress"]})
    retried = f"<!-- merge-queue:retry-{E2E}:{OLD} -->\nx"
    first = state([retried])
    _run(first)
    assert ("rerun", "70", "all") in first.calls
    assert f"merge-queue:restart-{E2E}:{OLD}" in first.calls[-1][2]
    second = state([retried, f"<!-- merge-queue:restart-{E2E}:{OLD} -->\ny"])
    second.run_status = {}
    _run(second)
    assert [c[0] for c in second.calls] == ["comment", "disarm"]
    assert f"merge-queue:evict-rerun:{OLD}" in second.calls[0][2]


def test_a_failed_license_is_dispatched_again_before_any_gate_is_re_run():
    """Review finding m2: with LICENSE_FIRST_CI_ENABLED a gate waits for the license status, so the
    license goes first on a retry as it does on a start."""
    failed = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed]}, checks={OLD: checks},
                statuses={OLD: [_status("failure")]}, run_status={70: ["in_progress"]})
    assert _run(gh)[0][1].contexts == (E2E, LICENSE)
    assert [c[0] for c in gh.calls if c[0] in ("dispatch", "rerun")] == ["dispatch", "rerun"]


def test_a_failed_gate_whose_run_is_still_live_is_left_for_its_kick():
    """The failed run's own queue-tick is still running when a racing queue run decides."""
    live = _run_of("e2e-required.yml", 70, "in_progress", None, check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [live]}, checks={OLD: checks}, statuses={OLD: [_status()]})
    _run(gh)
    assert gh.calls == []


def test_a_refused_rerun_evicts():
    failed = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed]}, checks={OLD: checks},
                statuses={OLD: [_status()]}, rerun_error="gh: The workflow run cannot be rerun (HTTP 422)")
    _run(gh)
    assert [c[0] for c in gh.calls][-2:] == ["comment", "disarm"]
    assert f"merge-queue:evict-rerun:{OLD}" in gh.calls[-2][2]


def test_a_transient_rerun_error_fails_the_run_and_disarms_nobody():
    failed = _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)
    checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed]}, checks={OLD: checks},
                statuses={OLD: [_status()]}, rerun_error="gh: Server Error (HTTP 502)")
    with pytest.raises(mq.GhError):
        _run(gh)
    assert not any(c[0] in ("disarm", "comment") for c in gh.calls)


HOURS_OLD = "2026-10-04T04:00:00Z"


def _old_green():
    return {"checks": {OLD: [_check(n, "success", completed=HOURS_OLD) for n in CHECK_NAMES]},
            "statuses": {OLD: [_status(created=HOURS_OLD)]}}


def test_an_old_green_head_that_dev_has_moved_past_is_rebased_not_evicted_as_stuck():
    """Spec section 6, the stuck row (TASK-34948 in tldw_chatbook): right after a merge the next PR's
    merge state can still read up to date while its green runs are hours old."""
    gh = FakeGh([_node(1, state="CLEAN")], behind_by={OLD: 2}, **_old_green())
    decision = _run(gh)[0][1]
    assert decision.kind == "rebase" and gh.calls[0] == ("rebase", "PR_1", OLD)
    assert not any(c[0] == "disarm" for c in gh.calls)


def test_an_old_green_head_up_to_date_with_dev_is_still_evicted_as_stuck():
    gh = FakeGh([_node(1, state="CLEAN")], **_old_green())
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.slug) == ("evict", "stuck")


def test_a_fresh_green_head_that_dev_has_moved_past_is_rebased_at_once():
    """Review finding m3: the stale merge state also hides a head whose green runs are minutes old;
    auto-merge cannot fire on a head that is behind, so waiting would hold the line."""
    gh = FakeGh([_node(1, state="CLEAN")], behind_by={OLD: 3}, **_green())
    decision = _run(gh)[0][1]
    assert decision.kind == "rebase" and gh.calls[0] == ("rebase", "PR_1", OLD)


def test_a_failed_behind_check_never_disarms():
    gh = FakeGh([_node(1, state="CLEAN")], behind_by={OLD: mq.GhError("gh: Server Error (HTTP 502)")}, **_old_green())
    with pytest.raises(mq.GhError):
        _run(gh)
    assert gh.calls == []


def test_a_kick_waits_for_its_run_before_reading_the_line():
    gh = FakeGh([_node(1, state="DIRTY")], run_status={99: ["in_progress", "completed"]})
    mq.run(gh, "on", now=lambda: NOW, sleep=lambda s: None, log=lambda m: None, wait_run=99)
    assert gh.run_reads[99] == 2 and gh.line_reads >= 1


def test_a_young_head_is_waited_out_once_then_started():
    young = _node(1, state="BLOCKED", committed="2026-10-04T11:59:00Z")
    gh = FakeGh([young], **_green({E2E: None}))
    slept = []
    decisions = mq.run(gh, "on", now=lambda: NOW, sleep=slept.append, log=lambda m: None)
    assert decisions[0][1].kind == "start" and slept and max(slept) <= mq.YOUNG_HEAD.total_seconds() + 1


def test_main_passes_only_a_numeric_wait_run(monkeypatch):
    seen = {}

    def fake_run(gh, mode, wait_run=None):
        seen["wait"] = wait_run
        return []

    monkeypatch.setattr(mq, "run", fake_run)
    monkeypatch.setenv("MERGE_QUEUE", "on")
    for value, expected in (("123", 123), ("", None), ("12; rm -rf /", None)):
        seen.clear()
        monkeypatch.setenv("WAIT_RUN", value)
        mq.main()
        assert seen.get("wait") == expected, value
```

Existing tests this part breaks are fixed in Part E, from its list.

- [ ] **Step D2: Run them to see them fail**

Run: `python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/CI/test_merge_queue_rules.py tldw_Server_API/tests/CI/test_merge_queue_actions.py -k "test_a_young_head_wait_is_marked_young or test_decide_front_table or test_decide_approves_held_gate_runs_first_so_they_read_as_running or test_decide_never_approves_a_non_gate_held_run or test_start_restarts_a_cancelled_run_of_a_missing_gate_once_then_evicts or test_start_approves_a_held_gate_run_the_decide_step_could_not_approve or test_start_restarts_a_run_whose_gate_job_was_cancelled_whatever_the_run_concluded or test_start_dispatches_a_missing_license_on_dev_and_nothing_for_the_gates_that_have_runs or test_start_with_no_run_at_all_warns_then_evicts_after_the_gap or test_first_failure_re_runs_the_failed_jobs_of_that_gates_run_and_comments_once or test_a_broken_runs_second_retry_on_the_same_head_evicts or test_a_green_gate_run_without_its_check_is_retried_once_then_evicted or test_a_retry_attempt_that_ends_cancelled_is_restarted_once_then_evicted or test_a_failed_license_is_dispatched_again_before_any_gate_is_re_run or test_a_failed_gate_whose_run_is_still_live_is_left_for_its_kick or test_a_refused_rerun_evicts or test_a_transient_rerun_error_fails_the_run_and_disarms_nobody or test_an_old_green_head_that_dev_has_moved_past_is_rebased_not_evicted_as_stuck or test_an_old_green_head_up_to_date_with_dev_is_still_evicted_as_stuck or test_a_fresh_green_head_that_dev_has_moved_past_is_rebased_at_once or test_a_failed_behind_check_never_disarms or test_a_kick_waits_for_its_run_before_reading_the_line or test_a_young_head_is_waited_out_once_then_started or test_main_passes_only_a_numeric_wait_run"`
Expected: FAIL (`start` kind missing, `_start` missing, `wait_run` unknown).

- [ ] **Step D3: Implement**

`Action`: docstring kinds become `wait`, `rebase`, `start`, `retry` or `evict`; `slug` doc adds "`young` on the
young-head wait".

`decide_front`: replace the missing branch:

```python
    if missing:
        # The commit date comes from the author's clock: one a little ahead counts as young too, one
        # further ahead does not, or the line would wait for that clock.
        if abs(now - pr.head_committed_at) <= YOUNG_HEAD:
            return Action("wait", "head is under 3 minutes old; its own runs may not be visible yet",
                          slug="young", contexts=missing)
        return Action("start", f"no counted run on the up-to-date head: {', '.join(missing)}", contexts=missing)
```

and give the green wait row a slug, so `_decide` can check it against `dev` (review finding m3):

```python
        return Action("wait", "green; auto-merge should fire", slug="green")
```

Add after `cleanup_approval_runs`:

```python
def _gate_runs(gh: GhApi, sha: str) -> list[dict]:
    """The gate workflows' pull_request runs on this head, newest first."""
    runs = [r for r in runs_on(gh, sha) if _workflow_name(r) in GATE_WORKFLOWS and r.get("event") == "pull_request"]
    return sorted(runs, key=lambda r: r.get("created_at") or "", reverse=True)


def _restart_cancelled(gh: GhApi, pr: PrState, ctx: Context, run: dict, links: tuple[str, ...],
                       log: Callable[[str], None], sleep: Callable[[float], None], now: Callable[[], datetime]) -> bool:
    """Re-run a cancelled gate run in full, once per gate per head (the `restart-<context>` marker).

    A cancelled run says nothing about the PR (a timeout, a person, a superseded run), so it is
    restarted rather than counted as a failure. A second cancellation on the same head evicts:
    a gate that keeps being cancelled, such as one that hits its job timeout, would otherwise be
    restarted on every wake and hold the line forever (review finding m1).

    Returns:
        True if the PR was evicted.
    """
    kind = f"restart-{ctx.name}"
    if has_comment(gh, pr.number, kind, pr.head_sha):
        if _going_again(gh, run["id"]):
            log(f"  #{pr.number}: run {run['id']} was restarted by a racing queue run; standing down")
            return False
        return _evict(gh, pr, Action("evict", f"{ctx.name} was cancelled again after the queue restarted it",
                                     links, "rerun"), log)
    outcome = _rerun(gh, pr, run, False, log, sleep, now)
    if outcome == "refused":
        return _evict(gh, pr, Action("evict", f"GitHub refused to restart the cancelled {ctx.name} run; push a "
                                     "commit, or close and reopen the PR", links, "no-run"), log)
    comment_once(gh, pr.number, kind, pr.head_sha,
                 f"Merge queue: {ctx.name}'s run on `{pr.head_sha[:10]}` was cancelled; restarting it once.", log)
    if outcome == "pending":
        _wake_best_effort(gh, log)
    return False


def _start(gh: GhApi, pr: PrState, names: tuple[str, ...], log: Callable[[str], None],
           sleep: Callable[[float], None], now: Callable[[], datetime]) -> bool:
    """Get the named missing contexts going on an up-to-date head (spec 4.1, start).

    The license is dispatched on dev. A gate's held run is approved; with none, its newest completed
    pull_request run (cancelled, or whose gate job was cancelled) is restarted once (`_restart_cancelled`). (A run that finished
    without reporting its check is a failure stand-in, so its context is failed, not missing, and
    goes through `_retry_gate`.) A dispatch would not count (S5), so a gate with neither gets a
    `no-run` strike. It looks
    again for up to HELD_RUN_POLLS x HELD_RUN_POLL_S first: a head's age is its commit date, not
    its push, so its runs may not be listed yet.

    Returns:
        True if the PR was evicted.

    Raises:
        GhError: A call failed for a reason other than a refusal.
    """
    if any(BY_NAME[n].kind == "status" for n in names):
        dispatch_license(gh, pr, log)
    pending = {BY_NAME[n].workflow: n for n in names if BY_NAME[n].kind == "check"}
    for attempt in range(HELD_RUN_POLLS):
        if not pending:
            return False
        if attempt:
            sleep(HELD_RUN_POLL_S)
        for run in runs_on(gh, pr.head_sha, status="action_required"):
            if _workflow_name(run) in pending and _queue_held(run):
                if _approve_strictly(gh, pr, run, log, now):
                    return True
                pending.pop(_workflow_name(run), None)
        runs = _gate_runs(gh, pr.head_sha)
        for workflow, name in list(pending.items()):
            mine = [r for r in runs if _workflow_name(r) == workflow]
            if any(r.get("status") in LIVE_RUN_STATUSES for r in mine):
                log(f"  #{pr.number}: a {name} run appeared on {pr.head_sha[:10]}; standing down for it")
                del pending[workflow]
                continue
            # A missing context's completed runs ended with no counted result for it: the run, or its
            # gate job, was cancelled (any other completed run without the check is a failure stand-in,
            # Part A; review point n1). A run held for someone else is theirs, never ours to touch.
            redo = next((r for r in mine if r.get("status") == "completed" and not _is_held(r)), None)
            if redo is None:
                continue
            if _restart_cancelled(gh, pr, BY_NAME[name], redo, (), log, sleep, now):
                return True
            del pending[workflow]
    if not pending:
        return False
    which = ", ".join(sorted(pending.values()))
    return _strike(
        gh, pr, "no-run",
        f"Merge queue: found no run of {which} on this head that the queue can start. It will look again, and "
        "remove it from the line if there is still none 10 minutes from now. A push, or closing and reopening the "
        "PR, starts its CI.",
        Action("evict", f"no run of {which} on this head that the queue can start; push a commit, or close and "
               "reopen the PR", slug="no-run"), False, log, now,
    )


def _retry_gate(gh: GhApi, pr: PrState, ctx: Context, links: tuple[str, ...], log: Callable[[str], None],
                sleep: Callable[[float], None], now: Callable[[], datetime]) -> bool:
    """Re-run the run whose gate failed once, in its own check suite.

    The failed run's own queue-tick sends a kick that waits for the run to complete, because
    GitHub only re-runs a completed run; a run still live is left for that kick. A run held again
    (its re-run's approval failed) is approved. One retry per gate per head: a run that failed
    without reporting its check keeps its run id when re-run, so its failures never add up to two;
    the `retry-<context>` marker caps it instead. A retry attempt that ended cancelled is restarted
    once (`_restart_cancelled`), not counted.

    Returns:
        True if the PR was evicted.
    """
    failed = sorted((c for c in pr.checks if c.context == ctx.name and c.status == "completed"
                     and c.conclusion not in PASSING and c.conclusion != "cancelled"),
                    key=lambda c: c.completed_at or datetime.min.replace(tzinfo=timezone.utc))
    suite = failed[-1].suite_id if failed else None
    runs = [r for r in _gate_runs(gh, pr.head_sha) if _workflow_name(r) == ctx.workflow]
    run = next((r for r in runs if suite is not None and r.get("check_suite_id") == suite), None)
    if run is None:  # a stand-in failure: a completed run that reported no check (Part A)
        reported = {c.suite_id for c in pr.checks if c.context == ctx.name and c.suite_id is not None}
        run = next((r for r in runs if r.get("status") == "completed"
                    and r.get("conclusion") not in NO_RESULT_CONCLUSIONS
                    and r.get("check_suite_id") not in reported), None)
    if run is None:
        return _evict(gh, pr, Action("evict", f"{ctx.name} failed and its run cannot be found to re-run",
                                     links, "rerun"), log)
    if run.get("status") in LIVE_RUN_STATUSES:
        log(f"  #{pr.number}: {ctx.name} run {run['id']} is still live; its kick re-runs it")
        return False
    if _queue_held(run):
        return _approve_strictly(gh, pr, run, log, now)
    if run.get("conclusion") == "cancelled":
        return _restart_cancelled(gh, pr, ctx, run, links, log, sleep, now)
    kind = f"retry-{ctx.name}"
    if has_comment(gh, pr.number, kind, pr.head_sha):
        if _going_again(gh, run["id"]):
            log(f"  #{pr.number}: run {run['id']} was re-run by a racing queue run; standing down")
            return False
        why = f"{ctx.name} failed again after its retry"
        if suite is None:  # review point n4: the run reported no check at all
            why += " (its run ended without reporting the check; was the gate job renamed or removed?)"
        return _evict(gh, pr, Action("evict", why, links, "failed-twice"), log)
    outcome = _rerun(gh, pr, run, suite is not None, log, sleep, now)
    if outcome == "refused":
        return _evict(gh, pr, Action("evict", f"{ctx.name} failed once and GitHub refused to re-run it",
                                     links, "rerun"), log)
    listed = "".join(f"\n- {u}" for u in links)
    comment_once(gh, pr.number, kind, pr.head_sha,
                 f"Merge queue: {ctx.name} failed once on `{pr.head_sha[:10]}`; re-running it.{listed}", log)
    if outcome == "pending":
        _wake_best_effort(gh, log)
    return False


def _retry(gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None], sleep: Callable[[float], None],
           now: Callable[[], datetime]) -> bool:
    """Retry each context that failed once: the license by a new dispatch on dev, a gate by a re-run.

    Returns:
        True if the PR was evicted.
    """
    listed = "".join(f"\n- {u}" for u in action.links)
    # The license first, as on a start (spec 4.4): with LICENSE_FIRST_CI_ENABLED a gate's
    # await_license waits for it. (Re-enabling license-first also needs a gate the license refused
    # to be re-run in full: a failed-jobs re-run keeps the old await_license verdict.)
    for name in sorted(action.contexts, key=lambda n: BY_NAME[n].kind != "status"):
        ctx = BY_NAME[name]
        if ctx.kind == "status":
            dispatch_license(gh, pr, log)
            comment_once(gh, pr.number, f"retry-{name}", pr.head_sha,
                         f"Merge queue: {name} failed once on `{pr.head_sha[:10]}`; dispatched the license gate "
                         f"on `{BASE}` again.{listed}", log)
            continue
        if _retry_gate(gh, pr, ctx, action.links, log, sleep, now):
            return True
    return False
```

Replace `apply`:

```python
def apply(
    gh: GhApi, pr: PrState, action: Action, log: Callable[[str], None], sleep: Callable[[float], None],
    now: Callable[[], datetime],
) -> bool:
    """Perform one decided action (spec section 7).

    Returns:
        True if the PR left the line (evicted), so the caller moves on to the next front.

    Raises:
        GhError: A read or a non-best-effort call failed.
    """
    try:
        if action.kind == "rebase":
            return _rebase(gh, pr, log, sleep, now)
        if action.kind in ("start", "retry"):
            # Two queue runs can decide the same action for the same head. Once the first one's run
            # is live (or the license status pending), the other sees it here and stands down for
            # that context.
            appeared = _appeared(gh, pr, action.contexts, now())
            if appeared:
                log(f"  #{pr.number}: a run appeared since the decision for {', '.join(appeared)}; leaving those")
            targets = tuple(n for n in action.contexts if n not in appeared)
            if not targets:
                return False
            if action.kind == "start":
                return _start(gh, pr, targets, log, sleep, now)
            return _retry(gh, pr, replace(action, contexts=targets), log, sleep, now)
        if action.kind == "evict":
            return _evict(gh, pr, action, log)
        return False
    except CommentRefused as exc:
        # The queue keeps its retry and strike counts in comments; on a locked PR it can neither
        # count nor explain, so it takes the PR out of the line rather than stall the line on it.
        log(f"  cannot comment on #{pr.number} ({exc}); removing it from the line without a comment")
        _disarm(gh, pr, log)
        return True
```

Add `_decide` and replace `run` and `main`:

```python
def _decide(gh: GhApi, pr: PrState, mode: str, now: Callable[[], datetime],
            log: Callable[[str], None]) -> tuple[PrState, Action]:
    """Read the front PR's contexts and decide (spec section 6).

    Raises:
        GhError: A read failed.
    """
    if mode == "on" and pr.merge_state not in ("BEHIND", "DIRTY"):
        # Held gate runs left by an earlier rebase or re-run: approving them first lets the
        # decision below see them as the live runs they become. A BEHIND front is about to be
        # rebased and a DIRTY one evicted.
        approve_held_runs(gh, pr.head_sha, log)
    pr = replace(pr, checks=read_contexts(gh, pr.head_sha))
    action = decide_front(pr, now())
    if action.slug in ("stuck", "green"):
        # Right after a merge the next PR's merge state can still read up to date (tldw_chatbook
        # #3070, TASK-34948). Auto-merge cannot fire on a head that is behind, so a green head is
        # checked against dev before the queue waits on it or evicts it as stuck. One GET.
        compare = gh.rest("GET", f"repos/{REPO}/compare/{BASE}...{pr.head_sha}?per_page=1") or {}
        if compare.get("behind_by", 0) > 0:
            action = Action("rebase", "behind dev (its merge state had not caught up yet)", contexts=ALL_CONTEXTS)
    return pr, action


def run(
    gh: GhApi,
    mode: str | None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    wait_run: int | None = None,
) -> list[tuple[int, Action]]:
    """One queue pass: decide for the front PR and, in `on` mode, act; repeat after evictions.

    Args:
        gh: The GitHub client.
        mode: The MERGE_QUEUE value; only `dry` and `on` (any case or padding) do anything.
        now: The clock (injected by tests).
        sleep: Waits between rereads (injected by tests).
        log: Receives the run's log lines.
        wait_run: A run to wait for (bounded) before deciding: set by a kick that a failed gate's
            queue-tick sent, so the failed run has completed and can be re-run.

    Returns:
        The (PR number, action) decisions taken or, in dry mode, proposed.

    Raises:
        GhError: A read or a non-best-effort call failed; the next event retries.
    """
    mode = (mode or "").strip().lower()
    if mode not in ("dry", "on"):
        log("merge queue is off (MERGE_QUEUE is not 'dry' or 'on')")
        return []
    started = now()
    if wait_run:
        wait_for_run(gh, wait_run, sleep, log)
    prs = read_prs(gh)
    comment_unqueued(gh, prs, mode, log)
    decisions: list[tuple[int, Action]] = []
    left: set[int] = set()
    for index, pr in enumerate(line_of(prs)[:MAX_FRONTS_PER_RUN]):
        if index and now() - started > RUN_BUDGET:
            log(f"run budget spent before #{pr.number}; a fresh queue run continues")
            if mode == "on":
                _wake_best_effort(gh, log)
            break
        pr = settle_unknown(gh, pr, sleep)
        if pr.armed_at is None:
            continue
        pr, action = _decide(gh, pr, mode, now, log)
        if mode == "on" and action.slug == "young":
            # A kick woken right after a rebase whose held runs had not appeared can land before
            # GitHub lists the head's runs, and nothing may wake the queue again for this head.
            # Wait out the window once and decide again (capped at one window).
            remaining = (pr.head_committed_at + YOUNG_HEAD - now()).total_seconds()
            pause = min(max(0.0, remaining), YOUNG_HEAD.total_seconds()) + 1
            if now() - started + timedelta(seconds=pause) > RUN_BUDGET:
                log(f"#{pr.number}: no budget left to wait out its young head; a fresh queue run will")
                _wake_best_effort(gh, log)
            else:
                sleep(pause)
                fresh = settle_unknown(gh, read_pr(gh, pr.number), sleep)
                if fresh.armed_at != pr.armed_at:
                    break  # disarmed or re-armed meanwhile; that event wakes a fresh run
                same_head = fresh.head_sha == pr.head_sha
                pr, action = _decide(gh, fresh, mode, now, log)
                if action.slug == "young" and same_head:
                    # Waited a full window and it still looks young: its commit clock runs ahead.
                    action = Action("start", "no run after waiting out the young-head window", contexts=action.contexts)
        decisions.append((pr.number, action))
        log(f"#{pr.number}: {action.kind} - {action.reason}")
        left_line = action.kind == "evict"
        if mode == "on":
            if action.kind != "wait" and not _still_front(gh, pr, left):
                log(f"#{pr.number}: no longer the armed front PR on this head; nothing done, the next wake decides")
                break
            cleanup_approval_runs(gh, pr, log)
            left_line = apply(gh, pr, action, log, sleep, now)
        if not left_line:
            break
        left.add(pr.number)
    return decisions


def main() -> int:
    """Run one queue pass with the real gh CLI and append the decisions to the job summary.

    `WAIT_RUN` (a kick's `wait_run` input) is used only when it is all digits.

    Returns:
        The process exit code (0; a failed call raises instead).
    """
    wait = os.environ.get("WAIT_RUN", "").strip()
    decisions = run(Gh(), os.environ.get("MERGE_QUEUE"), wait_run=int(wait) if wait.isdigit() else None)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Merge queue\n\n| PR | Action | Reason |\n|---|---|---|\n")
            for number, action in decisions:
                fh.write(f"| #{number} | {action.kind} | {action.reason} |\n")
    return 0
```

Replace `UNQUEUED_NOTES`:

```python
UNQUEUED_NOTES = {
    "fork": "Merge queue: fork PRs are not queued, because approving a fork's held runs would bypass GitHub's "
            "approval gate for outside contributors' workflows. A maintainer merges this one by hand.",
    "bot": "Merge queue: PRs opened by a bot or app are not queued, because a queue-approved run would run this "
           "branch's workflows without the token limits and approval gates GitHub applies to bot-authored "
           "runs. A maintainer merges this one by hand.",
}
```

Update the module docstring: "Runs inside GitHub Actions with the built-in GITHUB_TOKEN (merge-queue.yml, and the
queue-kick job of security-required.yml that a failed gate's queue-tick dispatches)." Point "Spec:" at both specs.

- [ ] **Step D4: Run the Part D tests**

Run the command of Step D2. Expected: PASS.

#### Part E: Every existing test the change breaks, the dead code, and the gate

The list below is complete for `test_merge_queue_rules.py` and `test_merge_queue_actions.py`: it is the set of tests
that failed when the review applied Parts A-D to a copy of the repository (2026-10-14), plus the dispatch-era tests
whose behaviour is gone. Nothing outside it may be edited to make the gate pass; if another test fails, stop and
report it.

- [ ] **Step E1: Delete the tests whose behaviour is gone**

From `test_merge_queue_actions.py` delete: `test_dev_tip_is_read_together_with_the_pr_state`,
`test_dispatch_requests_per_context`, `test_no_run_at_all_dispatches_all_seven_with_their_inputs`,
`test_first_failure_retries_only_the_failed_contexts`, `test_an_unknown_dev_tip_dispatches_nothing_and_fails_the_run`,
`test_rebase_dispatches_with_the_dev_tip_read_alongside_the_new_head`, `test_repeated_rebase_failure_evicts`,
`test_rebase_dispatches_only_after_the_new_head_appears`, `test_rebase_whose_head_never_moves_dispatches_nothing`,
`test_rebase_dispatches_before_cancelling_and_spares_the_queue`, `_as_tick_of`,
`test_a_tick_dispatches_its_own_workflow_last_after_everything_else`,
`test_a_tick_retrying_its_own_gate_comments_before_the_dispatch_that_cancels_it`,
`test_outside_a_gate_run_the_comment_follows_the_dispatches`, `test_rebase_deletes_the_new_heads_empty_approval_runs`,
`test_a_second_retry_on_the_same_head_dispatches_without_a_second_comment`, `test_disarm_failure_still_comments`,
`test_a_hand_started_frontend_diagnostic_is_not_a_run_of_the_required_context` (covered for every gate by
`test_a_dispatched_gate_run_is_ignored_whoever_started_it`), `test_own_run_is_not_counted_as_live` (the queue no
longer runs inside a gate run), `_queue_run` together with BOTH of its users,
`test_a_queue_dispatched_run_that_finished_green_without_its_check_is_a_failure_not_a_reason_to_redispatch` and the
parametrized `test_other_runs_without_the_check_are_not_failures` (its decorator calls `_queue_run` at import, so
deleting the helper alone breaks collection of the whole file; it is replaced below),
`test_a_queue_dispatched_run_that_reported_its_check_is_judged_by_the_check`,
`test_a_dispatch_refused_by_the_pr_branch_evicts_and_the_line_moves_on`,
`test_a_dispatch_refused_after_a_rebase_evicts_on_the_new_head`, the helpers `_dispatch_call` and `ALL_SEVEN`, and
the constant `DEV` (nothing uses it afterwards). Keep `DISPATCH_ERRORS` and FakeGh's `dispatch_refused` / `dispatch_status`.

- [ ] **Step E2: Change these tests, exactly as given**

1. `test_the_license_context_is_read_from_the_commit_status_history`: in its parametrize list, the three rows that
   expect `"dispatch"` (ids `no-status-is-missing`, `a-pending-older-than-15-minutes-is-a-lost-run`,
   `other-contexts-are-ignored`) expect `"start"`.
2. `test_a_live_run_of_another_workflow_is_not_a_required_context`: `("dispatch", (E2E,))` becomes `("start", (E2E,))`.
3. `test_a_pr_disarmed_since_the_decision_gets_no_checks`: `["dispatch"]` becomes `["start"]`.
4. `test_nothing_is_dispatched_or_said_when_every_targeted_context_already_started`: ids `["start", "retry"]`, and
   `in ("dispatch", "retry")` becomes `in ("start", "retry")`.
5. `test_run_that_failed_without_reporting_its_check_counts_as_a_failure_of_that_context`: replace
   `assert once.dispatched() == [_dispatch_call("security-required")]` with
   `assert ("rerun", "31", "all") in once.calls and not once.dispatched()`; in its docstring, "which would dispatch
   forever" becomes "which would be started forever".
6. `test_two_failed_hand_diagnostics_do_not_evict_a_green_pr`: add `event="workflow_dispatch"` to its `_run_of(...)`.
7. `test_a_failed_license_status_retries_only_the_license_gate_on_dev`: its last assertion becomes

   ```python
    assert comment[2] == (f"<!-- merge-queue:retry-{LICENSE}:{OLD} -->\nMerge queue: {LICENSE} failed once on "
                          f"`{OLD[:10]}`; dispatched the license gate on `dev` again.")
   ```
8. `test_evicted_front_hands_over_in_the_same_run`: `assert gh.calls[0] == ("disarm", "PR_1")` becomes
   `assert [c[0] for c in gh.calls[:2]] == ["comment", "disarm"]`.
9. `test_a_rebase_across_workflow_changes_evicts_at_once_with_the_hand_rebase_steps`:
   `assert gh.calls[1] == ("disarm", "PR_1")` becomes `assert gh.calls[1][0] == "comment" and gh.calls[2] == ("disarm", "PR_1")`.
10. `test_first_rebase_failure_warns_without_evicting`: its last assertion becomes
    `assert "rebasing onto dev failed (gh: rebase refused (HTTP 422)); will retry" in body`.
11. `test_any_other_rebase_refusal_still_gets_one_retry`: `"will retry once"` becomes `"will retry"`.
12. `test_a_refused_license_dispatch_fails_the_run_and_disarms_nobody`: its last assertion becomes
    `assert [c[1] for c in gh.dispatched()] == ([LICENSE_GATE, "security-required.yml"] if state == "BEHIND" else [LICENSE_GATE])`
    (after a rebase took effect, a failed run wakes a kick: Part C).
13. The autouse fixture `_no_real_run_id`: its docstring becomes "Isolate GITHUB_RUN_ID and GITHUB_WORKFLOW_REF from
    the CI run executing these tests; a rebase never cancels the run executing the queue."
14. Replace these five tests with the code below: `test_missing_contexts_only_are_dispatched`,
    `test_a_failure_is_retried_while_other_contexts_are_still_running`,
    `test_on_mode_rebases_front_only_and_starts_exactly_the_seven_contexts`,
    `test_a_run_that_appeared_since_the_decision_stops_only_that_contexts_dispatch`,
    `test_a_github_side_dispatch_error_fails_the_run_and_disarms_nobody`; and add the replacement for
    `test_other_runs_without_the_check_are_not_failures`:

```python
def test_missing_contexts_are_started_without_any_gate_dispatch():
    cancelled = _run_of("container-build-check.yml", 71, "completed", "cancelled")
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [cancelled]}, run_status={71: ["in_progress"]},
                **_green({CONTAINER: "cancelled"}, statuses=[]))
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("start", (CONTAINER, LICENSE))
    assert gh.dispatched() == [("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "1"})]
    assert ("rerun", "71", "all") in gh.calls


def test_a_failure_is_retried_while_other_contexts_are_still_running():
    """Only a failed gate wakes the queue: waiting for the others would lose the retry."""
    failed = _run_of("backend-required.yml", 60, "completed", "failure", check_suite_id=600)
    checks = _checks({BACKEND: None, E2E: None}) + [_check(BACKEND, "failure", suite=600)]
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [failed, _run_of("e2e-required.yml", 70)]},
                checks={OLD: checks}, statuses={OLD: [_status()]}, run_status={60: ["in_progress"]})
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (BACKEND,))
    assert ("rerun", "60", "failed") in gh.calls and not gh.dispatched()


def test_on_mode_rebases_the_front_only():
    runs = {OLD: [
        _run_of("backend-required.yml", 11),
        _run_of("ci.yml", 12),
        _run_of("pre-commit.yml", 13, "completed", "success"),
        _run_of("merge-queue.yml", 14, "completed", "success"),
    ], NEW: _six_held()}
    gh = FakeGh([_node(1), _node(2, armed="2026-10-04T11:00:00Z")], runs=runs)
    _run(gh)
    assert [c for c in gh.calls if c[0] == "rebase"] == [("rebase", "PR_1", OLD)]
    assert gh.dispatched() == [("dispatch", LICENSE_GATE, {"ref": "dev", "inputs[pr]": "1"})]
    assert [c for c in gh.calls if c[0] == "cancel"] == [("cancel", "11"), ("cancel", "12")]
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert comment[1] == 1 and f"<!-- merge-queue:rebased:{NEW} -->" in comment[2]
    assert "approved its 6 required gates" in comment[2]
    assert all(c[1] != "PR_2" for c in gh.calls if c[0] in ("rebase", "disarm"))


def test_a_run_that_appeared_since_the_decision_stops_only_that_contexts_retry():
    """merge-queue.yml and a kick can decide the same retry for one head. The second re-reads the live
    runs first and leaves the contexts already started."""
    checks = _checks({BACKEND: None, E2E: None}) + [_check(BACKEND, "failure", suite=600),
                                                    _check(E2E, "failure", suite=700)]
    runs = [_run_of("backend-required.yml", 60, "completed", "failure", check_suite_id=600),
            _run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)]
    gh = FakeGh([_node(1, state="BLOCKED")], checks={OLD: checks}, statuses={OLD: [_status()]}, runs={OLD: runs},
                late_runs={OLD: [_run_of("backend-required.yml", 88)]}, run_status={70: ["in_progress"]})
    decision = _run(gh)[0][1]
    assert (decision.kind, decision.contexts) == ("retry", (BACKEND, E2E))
    assert [c for c in gh.calls if c[0] == "rerun"] == [("rerun", "70", "failed")]
    comment = next(c for c in gh.calls if c[0] == "comment")
    assert f"merge-queue:retry-{E2E}:{OLD}" in comment[2]


@pytest.mark.parametrize("path", ["approve", "rerun"])
@pytest.mark.parametrize("error", ["gh: Server Error (HTTP 502)", "gh: API rate limit exceeded (HTTP 403)"])
def test_a_github_side_error_on_approve_or_rerun_fails_the_run_and_disarms_nobody(path, error):
    """A 5xx or rate limit is GitHub's error, not the PR's: the run fails before any disarm, so the next
    event retries, and nothing behind the front is touched."""
    nodes = [_node(1, state="BLOCKED")] + [_node(i, armed=f"2026-10-04T1{i}:00:00Z", state="BLOCKED") for i in range(2, 6)]
    if path == "approve":
        gh = FakeGh(nodes, runs={OLD: [_held(35, "e2e-required.yml")]}, approve_error=error, **_green({E2E: None}))
    else:
        checks = _checks({E2E: None}) + [_check(E2E, "failure", suite=700)]
        gh = FakeGh(nodes, runs={OLD: [_run_of("e2e-required.yml", 70, "completed", "failure", check_suite_id=700)]},
                    checks={OLD: checks}, statuses={OLD: [_status()]}, rerun_error=error)
    with pytest.raises(mq.GhError):
        _run(gh)
    assert not any(c[0] in ("disarm", "comment") for c in gh.calls)
    assert not any(c[0] == "rebase" for c in gh.calls), "nothing behind the front was touched"


@pytest.mark.parametrize(("run", "kind"), [
    (_run_of("frontend-required.yml", 31, "completed", "success", event="workflow_dispatch", check_suite_id=731,
             actor={"login": "someone"}), "start"),
    (_run_of("frontend-required.yml", 31, "completed", "cancelled", check_suite_id=731), "start"),
    (_run_of("frontend-required.yml", 31, "completed", "success", check_suite_id=731), "retry"),
    (_run_of("frontend-required.yml", 31, "completed", "skipped", check_suite_id=731), "retry"),
], ids=["a-dispatch-is-ignored", "cancelled-is-no-run", "green-without-its-check-fails", "skipped-without-its-check-fails"])
def test_what_a_gate_run_without_its_check_means(run, kind):
    gh = FakeGh([_node(1, state="BLOCKED")], runs={OLD: [dict(run)]}, **_green({"frontend-required": None}))
    decision = _run(gh, "dry")[0][1]
    assert (decision.kind, decision.contexts) == (kind, ("frontend-required",))
```

15. The two `gh` boundary tests:

Rewrite `test_gh_argument_typing`:

```python
def test_gh_argument_typing():
    seen = []
    gh = mq.Gh(runner=lambda args: (seen.append(args), "{}")[1])
    gh.graphql("query { x }", number=5, id="X")
    mq.wake(gh, 42)
    mq.dispatch_license(gh, _pr_state(number=7), lambda m: None)
    gh.rest("POST", f"repos/{mq.REPO}/actions/runs/31/approve")
    gh.rest("POST", f"repos/{mq.REPO}/actions/runs/31/rerun-failed-jobs")
    graphql_args, kick_args, license_args, approve_args, rerun_args = seen
    assert graphql_args[graphql_args.index("-F") + 1] == "number=5" and "id=X" in graphql_args
    assert kick_args == ["api", "-X", "POST", f"repos/{mq.REPO}/actions/workflows/security-required.yml/dispatches",
                         "-f", "ref=dev", "-f", "inputs[queue_kick]=true", "-f", "inputs[wait_run]=42"]
    assert license_args == ["api", "-X", "POST", f"repos/{mq.REPO}/actions/workflows/frontend-license-gate.yml/dispatches",
                            "-f", "ref=dev", "-f", "inputs[pr]=7"]
    assert approve_args == ["api", "-X", "POST", f"repos/{mq.REPO}/actions/runs/31/approve"]
    assert rerun_args == ["api", "-X", "POST", f"repos/{mq.REPO}/actions/runs/31/rerun-failed-jobs"]
```

Rewrite `test_main_drives_the_real_gh_argv_end_to_end` for a BEHIND front PR whose new head has the six held gate runs
and one held `ci.yml` run. Keep the existing `fake_subprocess_run` shape, add `monkeypatch.setattr(mq,
"HELD_RUN_POLL_S", 0)`, and add these two branches to it, before its generic `elif args[2] == "GET":` branch:

```python
    held = [_held(31 + i, workflow) for i, workflow in enumerate(GATE_FILES)] + [_held(40, "ci.yml")]
    ...
        elif args[2] == "GET" and f"head_sha={NEW}&status=action_required" in args[3]:
            out = {"total_count": len(held), "workflow_runs": list(held)}
        elif args[2] == "POST" and args[3].endswith("/approve"):
            rid = int(args[3].split("/runs/")[1].split("/")[0])
            held[:] = [r for r in held if r["id"] != rid]  # GitHub starts it; it is no longer held
            out = None
```

(`args` is `argv[1:]`, so `args[2]` is the HTTP method and `args[3]` the path.) Its docstring becomes "main() through
the real Gh and _run_gh down to the subprocess argv, with canned JSON for a BEHIND front PR: line query, context reads,
pinned-head rebase, poll reread, the license dispatch, the six approvals, the non-gate cleanup, comment." Delete
everything after `assert mq.main() == 0` except the `repo = f"repos/{owner}/{name}"` line (the old `seen[:-1] == [...]`
list, its helpers and the two assertions after it all go), and put these assertions there instead; in `argv` the
method is index 3 and a GraphQL query index 4:

```python
    mutating = [a for a in seen if a[3:4] in (["POST"], ["DELETE"]) or a[4:5] == [f"query={mq.REBASE_MUTATION}"]]
    assert mutating[0] == ["gh", "api", "graphql", "-f", f"query={mq.REBASE_MUTATION}", "-f", "id=PR_7", "-f", f"oid={OLD}"]
    assert mutating[1] == ["gh", "api", "-X", "POST", f"{repo}/actions/workflows/frontend-license-gate.yml/dispatches",
                           "-f", "ref=dev", "-f", "inputs[pr]=7"]
    assert [a[4] for a in mutating[2:8]] == [f"{repo}/actions/runs/{31 + i}/approve" for i in range(6)]
    assert mutating[8] == ["gh", "api", "-X", "DELETE", f"{repo}/actions/runs/40"]
    assert mutating[9][:6] == ["gh", "api", "-X", "POST", f"{repo}/issues/7/comments", "-f"]
    assert mutating[9][6].startswith(f"body=<!-- merge-queue:rebased:{NEW} -->\nMerge queue: this PR is next.")
    assert len(mutating) == 10
    assert "| #7 | rebase | behind dev |" in summary.read_text(encoding="utf-8")
```

- [ ] **Step E3: Delete the dead code in the script**

Delete `BROKEN_RUN_CONCLUSIONS` (no reader is left after Parts A and D) and confirm with
`grep -n "BROKEN_RUN_CONCLUSIONS\|_host_workflow\|dispatch_request\|_dispatch_contexts\|BRANCH_REFUSALS\|base_sha\|queue_only_dispatch\|silent_green" Helper_Scripts/ci/merge_queue.py`
printing nothing. Update the module docstring: "Runs inside GitHub Actions with the built-in GITHUB_TOKEN
(merge-queue.yml, and the queue-kick job of security-required.yml that a failed gate's queue-tick dispatches)." and
point "Spec:" at both specs.

- [ ] **Step E4: The gate**

Run: `python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/CI/test_merge_queue_rules.py tldw_Server_API/tests/CI/test_merge_queue_actions.py tldw_Server_API/tests/CI/test_merge_queue_workflow.py`
Expected: every test passes. A failure not explained by this task's list is a defect in the code, not a test to edit:
fix the code, or stop and report.

- [ ] **Step E5: Commit**

```bash
git add Helper_Scripts/ci/merge_queue.py tldw_Server_API/tests/CI/test_merge_queue_rules.py tldw_Server_API/tests/CI/test_merge_queue_actions.py
git commit -q -m "ci(merge-queue): approve and re-run gates instead of dispatching them; kicks, restart caps, stale-state guard (TASK-<N>)"
```


---

### Task 2: Guards — never dispatch a check context

**Files:**
- Test: `tldw_Server_API/tests/CI/test_merge_queue_actions.py`

- [ ] **Step 1: Write the guard tests**

```python
def test_the_queue_never_dispatches_a_check_context():
    """Spec section 7: the only dispatches are the license gate and the kick, both on dev. Checked on
    the syntax tree (only `dispatch_license` and `wake` call `dispatch`) and on behaviour (every
    dispatch any scenario in this file records is one of the two)."""
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    callers = {
        fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dispatch"
    }
    assert callers == {"dispatch_license", "wake"}
    assert SCRIPT.read_text(encoding="utf-8").count("/dispatches") == 1, "one place builds a dispatch"


def _assert_only_allowed_dispatches(gh):
    for _, workflow, fields in gh.dispatched():
        assert fields["ref"] == "dev", fields
        license_call = workflow == LICENSE_GATE and set(fields) == {"ref", "inputs[pr]"}
        kick = (workflow == "security-required.yml" and fields.get("inputs[queue_kick]") == "true"
                and set(fields) <= {"ref", "inputs[queue_kick]", "inputs[wait_run]"})
        assert license_call or kick, (workflow, fields)


@pytest.mark.parametrize("scenario", [
    {"nodes": [_node(1)], "runs": {NEW: []}},                                             # rebase, nothing held
    {"nodes": [_node(1, state="BLOCKED")], "checks": {OLD: []}, "statuses": {OLD: []}},  # nothing at all
    {"nodes": [_node(1, state="BLOCKED")], **_green({E2E: "failure"})},                    # failed once
    {"nodes": [_node(1, state="BLOCKED")], **_green(statuses=[_status("failure")])},       # license failed once
], ids=["rebase", "nothing", "gate-failed", "license-failed"])
def test_no_scenario_dispatches_anything_but_the_license_and_the_kick(scenario):
    scenario = dict(scenario)
    nodes = scenario.pop("nodes")
    gh = FakeGh(nodes, **scenario)
    try:
        _run(gh)
    except mq.GhError:
        pass
    _assert_only_allowed_dispatches(gh)
```

- [ ] **Step 2: Run, then mutation-check each guard**

Run: `python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/CI/test_merge_queue_rules.py tldw_Server_API/tests/CI/test_merge_queue_actions.py`
Expected: PASS (the whole file, including the two new guard tests).

Mutation check (do it, then revert with `git checkout -- Helper_Scripts/ci/merge_queue.py`):
1. In `_start`, add `dispatch(gh, "backend-required.yml", pr.head_ref)` as the first line. Expected:
   `test_the_queue_never_dispatches_a_check_context` FAILS.
2. In `_queue_held`, delete the `and _workflow_name(run) in GATE_WORKFLOWS` line. Expected:
   `test_only_the_queues_own_held_gate_runs_are_approvable` and the rebase test FAIL.
3. In `cleanup_approval_runs`, delete `or _workflow_name(run) in GATE_WORKFLOWS`. Expected:
   `test_cleanup_deletes_only_the_queues_non_gate_held_runs` FAILS.
4. In `_approve_strictly`, change `if _transient(exc): raise` to `if False: raise`. Expected:
   `test_a_transient_approve_error_raises_and_disarms_nobody` FAILS.
5. In `_decide`, delete the `compare` block. Expected: `test_an_old_green_head_that_dev_has_moved_past_is_rebased_not_evicted_as_stuck`
   and `test_a_fresh_green_head_that_dev_has_moved_past_is_rebased_at_once` FAIL.
6. In `_restart_cancelled`, replace `if has_comment(gh, pr.number, kind, pr.head_sha):` with `if False:`. Expected:
   `test_start_restarts_a_cancelled_run_of_a_missing_gate_once_then_evicts` and
   `test_a_retry_attempt_that_ends_cancelled_is_restarted_once_then_evicted` FAIL.
7. In `required_run_stand_ins`, change `not in NO_RESULT_CONCLUSIONS` to `in ("failure", "startup_failure", "timed_out")`.
   Expected: `test_a_green_gate_run_without_its_check_is_retried_once_then_evicted` and
   `test_a_completed_gate_run_without_its_check_is_one_failure_unless_cancelled_or_held` FAIL.
Record each mutation and its failing test in the task notes.

- [ ] **Step 3: Commit**

```bash
git add tldw_Server_API/tests/CI/test_merge_queue_actions.py
git commit -q -m "test(merge-queue): guard that no check context is ever dispatched (TASK-<N>)"
```

---

### Task 3: Workflows — the kick, the ticks, the frontend name guard

**Files:**
- Modify: `.github/workflows/security-required.yml`, `.github/workflows/backend-required.yml`,
  `.github/workflows/coverage-required.yml`, `.github/workflows/e2e-required.yml`,
  `.github/workflows/frontend-required.yml`, `.github/workflows/container-build-check.yml`
- Test: `tldw_Server_API/tests/CI/test_merge_queue_gate_workflows.py`,
  `tldw_Server_API/tests/CI/test_merge_queue_frontend_required.py`,
  `tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`

**Interfaces:**
- Consumes: `mq.KICK_WORKFLOW == "security-required.yml"`, input names `queue_kick` and `wait_run`, env `WAIT_RUN`
  read by `main()` (Task 1, Part D).

- [ ] **Step 1: Write the failing workflow tests**

In `test_merge_queue_gate_workflows.py`, replace `test_queue_tick_runs_the_queue_from_dev_with_the_queue_permissions`
and `test_queue_tick_runs_only_after_a_failed_gate_while_the_queue_is_enabled`, and add the kick tests:

```python
KICK_RUN = (
    'gh api -X POST "repos/${REPOSITORY}/actions/workflows/security-required.yml/dispatches" '
    '-f ref=dev -f "inputs[queue_kick]=true" -f "inputs[wait_run]=${RUN_ID}"'
)


# All six gate workflows. GATES above leaves out frontend-required.yml, which its own test file covers.
TICK_GATES = {**GATES, "frontend-required.yml": "frontend-required"}


@pytest.mark.parametrize("name", TICK_GATES)
def test_queue_tick_only_sends_a_kick(name):
    """Spec 4.6: the tick cannot re-run its own still-running run, so it asks a kick on dev to wait
    for it and run the queue. It needs nothing but `actions: write` and no checkout."""
    jobs = _wf(name)["jobs"]
    gate = TICK_GATES[name]
    tick = dict(jobs["queue-tick"])
    tick.pop("if")
    assert tick == {
        "name": "Merge queue tick",
        "needs": [gate],
        "runs-on": "ubuntu-latest",
        "timeout-minutes": 5,
        "permissions": {"actions": "write"},
        "steps": [{
            "name": "Wake the merge queue",
            "env": {"GH_TOKEN": "${{ github.token }}", "REPOSITORY": "${{ github.repository }}",
                    "RUN_ID": "${{ github.run_id }}"},
            "run": KICK_RUN,
        }],
    }


@pytest.mark.parametrize("name", TICK_GATES)
def test_queue_tick_runs_only_after_a_failed_gate_on_a_same_repo_pull_request(name):
    condition = _wf(name)["jobs"]["queue-tick"]["if"]
    events = [("workflow_dispatch", ""), ("pull_request", "owner/repo"), ("pull_request", "fork/repo"),
              ("workflow_run", ""), ("push", "")]
    ran = 0
    for (event, head_repo), result, mode, cancelled in product(
        events, ("failure", "success", "skipped", "cancelled"), ("", "off", "dry", "on"), (False, True)
    ):
        values = {
            f"needs.{TICK_GATES[name]}.result": result,
            "vars.MERGE_QUEUE": mode,
            "github.event_name": event,
            "github.event.pull_request.head.repo.full_name": head_repo,
            "github.repository": "owner/repo",
        }
        expected = (not cancelled and result == "failure" and mode in ("dry", "on")
                    and (event, head_repo) == ("pull_request", "owner/repo"))
        assert _evaluate(condition, values, cancelled) is expected, (event, head_repo, result, mode, cancelled)
        ran += expected
    assert ran == 2  # one event x two modes: never on a dispatch, never with MERGE_QUEUE unset


def _kick_values(event, flag, ref, mode="on"):
    return {"github.event_name": event, "github.event.inputs.queue_kick": flag, "github.ref": ref,
            "vars.MERGE_QUEUE": mode}


def test_the_kick_job_runs_only_for_a_flagged_dispatch_on_dev_with_the_queue_enabled():
    job = _wf("security-required.yml")["jobs"]["queue-kick"]
    for event, flag, ref, mode in product(("workflow_dispatch", "pull_request"), ("true", "false", ""),
                                          ("refs/heads/dev", "refs/heads/feat/x"), ("", "off", "dry", "on")):
        expected = event == "workflow_dispatch" and flag == "true" and ref == "refs/heads/dev" and mode in ("dry", "on")
        assert _evaluate(job["if"], _kick_values(event, flag, ref, mode)) is expected, (event, flag, ref, mode)


def test_the_kick_job_is_the_queue_job_reading_wait_run():
    job = dict(_wf("security-required.yml")["jobs"]["queue-kick"])
    job.pop("if")
    queue = _wf("merge-queue.yml")["jobs"]["queue"]
    steps = [dict(step) for step in queue["steps"]]
    steps[1] = dict(steps[1], env={**steps[1]["env"], "WAIT_RUN": "${{ github.event.inputs.wait_run }}"})
    assert job == {"name": "Merge queue kick", "runs-on": "ubuntu-latest", "timeout-minutes": queue["timeout-minutes"],
                   "permissions": queue["permissions"], "steps": steps}
    assert queue["timeout-minutes"] == 15, "a kick may wait 3 minutes and then re-run up to six gates"


def test_security_required_kick_skips_every_gate_job_only_on_dev():
    """Spec 4.6: a flagged dispatch on dev runs only the kick; on any other ref, or without the flag,
    the gate jobs run as today."""
    jobs = _wf("security-required.yml")["jobs"]
    for job_id in ("changes", "security-required"):
        condition = jobs[job_id]["if"]
        assert ("always() && !cancelled() && (github.event.inputs.queue_kick != 'true' || "
                "github.ref != 'refs/heads/dev') &&") in " ".join(condition.split())
    kick = {"github.event.inputs.queue_kick": "true", "github.ref": "refs/heads/dev"}
    other_ref = {"github.event.inputs.queue_kick": "true", "github.ref": "refs/heads/feat/x"}
    unflagged = {"github.event.inputs.queue_kick": "", "github.ref": "refs/heads/dev"}
    base = {"github.event_name": "workflow_dispatch", "needs.await_license.result": "skipped",
            "needs.await_license.outputs.license_passed": "", "needs.admission.result": "skipped",
            "needs.admission.outputs.should_run": ""}
    assert not _evaluate(jobs["changes"]["if"], {**base, **kick})
    assert _evaluate(jobs["changes"]["if"], {**base, **other_ref})
    assert _evaluate(jobs["changes"]["if"], {**base, **unflagged})


def test_a_kick_never_cancels_a_running_kick():
    concurrency = _wf("security-required.yml")["concurrency"]
    assert concurrency["cancel-in-progress"] == "${{ github.event.inputs.queue_kick != 'true' }}"
```

The evaluator raises `KeyError` on any name a condition reads that the test did not set, by design, so every existing
test in this file that evaluates `security-required.yml`'s `changes` or gate `if` must set the two new names. Make these
changes, exactly:

- In the `_rows` / `_outcome` table, add `"github.event.inputs.queue_kick": ""` and `"github.ref": "refs/pull/1/merge"`
  to the values every row starts from (the other gates ignore extra names).
- In `test_every_gate_turns_a_license_wait_that_did_not_pass_red`, add the same two entries to the `values` dict
  inside the loop and to the `passing` dict (it builds its own dicts, not the table's rows).
- In `test_gate_declares_an_optional_base_sha_dispatch_input`, replace `assert set(inputs) == {"base_sha"}` with
  `assert set(inputs) == ({"base_sha", "queue_kick", "wait_run"} if name == "security-required.yml" else {"base_sha"})`.

In `test_merge_queue_frontend_required.py`, replace `test_name_guard_exempts_only_the_queue_actor`:

```python
def test_every_dispatch_publishes_the_diagnostic_name():
    """D2: no dispatch publishes the protected name, whoever started it; a pull_request run does."""
    name = _jobs()[GATE]["name"]
    assert _normalized(name) == _normalized(
        "${{ github.event_name == 'workflow_dispatch' && 'frontend-required-diagnostic' || 'frontend-required' }}"
    )
    assert "github.actor" not in WORKFLOW.read_text(encoding="utf-8")
```

In the same file, `test_nothing_else_in_the_workflow_depends_on_who_dispatched_it` changes
`assert rendered.count("github.actor") == 1` to `== 0`, and its docstring becomes "Nothing in the workflow depends on
who dispatched it: every dispatch publishes the diagnostic name." Replace that file's tick tests
`test_queue_tick_runs_only_after_a_failed_gate_with_the_queue_enabled`,
`test_queue_tick_is_the_queue_job_with_a_shorter_budget`,
`test_queue_tick_checkout_is_pinned_like_every_other_checkout_in_the_workflow` and
`test_queue_tick_is_outside_every_needs_and_holds_the_only_write_scopes` with this one (the tick's full shape is pinned
by `test_queue_tick_only_sends_a_kick`, which now covers all six gates, and by `_assert_exact_queue_tick` in the
license contract):

```python
def test_queue_tick_holds_only_actions_write_checks_nothing_out_and_nothing_waits_on_it():
    jobs = _jobs()
    assert jobs[TICK]["permissions"] == {"actions": "write"}
    assert not any("uses" in step for step in jobs[TICK]["steps"])
    assert all(TICK not in _needs(job) for job in jobs.values())
```

In `tldw_Server_API/tests/CI/test_admin_ui_vitest_ratchet_workflow.py` (not in backend-required's contract step, so
only the whole-directory run in Task 5 would otherwise catch it), replace the docstring and the actor loop of
`test_manual_dispatch_cannot_publish_the_required_check_name`:

```python
    """Keep diagnostic manual ratchets from satisfying branch protection: no dispatch, by anyone,
    publishes the protected name (D2 of the 2026-10-11 merge-queue redesign)."""
    ...
    for actor in ("octocat", "github-actions", "github-actions[bot]", "dependabot[bot]", ""):
        assert published("workflow_dispatch", actor) == "frontend-required-diagnostic"
    # The events that always published the required name still do, whoever the actor is.
    for event_name in ("pull_request", "workflow_run"):
        for actor in ("octocat", "github-actions[bot]"):
            assert published(event_name, actor) == "frontend-required"
```

(the `workflow = ...`, `name = ...` and `published` lines between the docstring and the loops stay as they are. Deleted:
the line asserting that the `github-actions[bot]` dispatch publishes `frontend-required`, and the two old comments
inside the replaced lines, "A dispatch started by anyone but the queue stays diagnostic. ..." and "The queue's own
dispatch publishes the required name.", which D2 makes false).

In `test_license_first_workflow_contracts.py`:
- `ORIGINAL_JOB_NAMES["security-required.yml"]` becomes `("changes", "security-required", "queue-tick", "queue-kick")`.
- Replace `_assert_exact_queue_tick` and add `_assert_exact_queue_kick` next to it:

```python
QUEUE_KICK_RUN = (
    'gh api -X POST "repos/${REPOSITORY}/actions/workflows/security-required.yml/dispatches" '
    '-f ref=dev -f "inputs[queue_kick]=true" -f "inputs[wait_run]=${RUN_ID}"'
)


def _assert_exact_queue_tick(name: str, data: dict[str, Any]) -> None:
    """Pin the whole queue-tick job, so its exemption cannot be widened by editing the job."""
    gate = QUEUE_TICK_GATES[name]
    job = dict(data["jobs"]["queue-tick"])
    assert _normalized(job.pop("if")) == _normalized(
        f"""
        always() && !cancelled() &&
        needs.{gate}.result == 'failure' &&
        (vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on') &&
        github.event_name == 'pull_request' &&
        github.event.pull_request.head.repo.full_name == github.repository
        """
    ), name
    assert job == {
        "name": "Merge queue tick",
        "needs": [gate],
        "runs-on": "ubuntu-latest",
        "timeout-minutes": 5,
        "permissions": {"actions": "write"},
        "steps": [
            {
                "name": "Wake the merge queue",
                "env": {
                    "GH_TOKEN": "${{ github.token }}",
                    "REPOSITORY": "${{ github.repository }}",
                    "RUN_ID": "${{ github.run_id }}",
                },
                "run": QUEUE_KICK_RUN,
            },
        ],
    }, name
    # Nothing may wait on it: it can then never delay a required check or turn it red.
    for other_name, other in data["jobs"].items():
        assert "queue-tick" not in _needs(other), (name, other_name)


def _assert_exact_queue_kick(name: str, data: dict[str, Any]) -> None:
    """Pin the whole queue-kick job: it exists only in security-required.yml and only runs dev's queue code."""
    assert name == "security-required.yml", name
    job = dict(data["jobs"]["queue-kick"])
    assert _normalized(job.pop("if")) == _normalized(
        """
        github.event_name == 'workflow_dispatch' &&
        github.event.inputs.queue_kick == 'true' &&
        github.ref == 'refs/heads/dev' &&
        (vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on')
        """
    ), name
    assert job == {
        "name": "Merge queue kick",
        "runs-on": "ubuntu-latest",
        "timeout-minutes": 15,
        "permissions": {
            "contents": "write",
            "pull-requests": "write",
            "actions": "write",
            "checks": "read",
            "statuses": "read",
        },
        "steps": [
            {
                "name": "Checkout dev",
                "uses": CHECKOUT_ACTION,
                "with": {"ref": "dev", "persist-credentials": False},
            },
            {
                "name": "Advance the queue",
                "env": {
                    "GH_TOKEN": "${{ github.token }}",
                    "MERGE_QUEUE": "${{ vars.MERGE_QUEUE }}",
                    "WAIT_RUN": "${{ github.event.inputs.wait_run }}",
                },
                "run": "python3 -m Helper_Scripts.ci.merge_queue",
            },
        ],
    }, name
    for other_name, other in data["jobs"].items():
        assert "queue-kick" not in _needs(other), (name, other_name)
```
- At both places that special-case `queue-tick` (`test_runner_roots_cannot_bypass_admission_and_checkouts_are_immutable`
  and `test_admitted_jobs_have_no_secrets_or_write_scoped_credentials`), add the same branch for
  `job_name == "queue-kick"` calling `_assert_exact_queue_kick`. A `queue-kick` job in any other workflow must fail
  (assert `name == "security-required.yml"` inside the helper).
- `test_runner_roots_cannot_bypass_admission_and_checkouts_are_immutable` pins the `if` of `security-required.yml`'s
  `changes` job (through `admission_clause`) and of its `security-required` job (the `LICENSE_RED_GATES` literal)
  exactly. Insert this immediately before its `assert _normalized(job.get("if")) == _normalized(expected_condition),
  (name, job_name)` line, and change nothing else in that test:

```python
                if name == "security-required.yml" and job_name in ("changes", "security-required"):
                    # A queue kick runs no gate job (spec 4.6 of the 2026-10-11 merge-queue redesign);
                    # every other event takes the condition unchanged.
                    expected_condition = expected_condition.replace(
                        "always() && !cancelled() &&",
                        "always() && !cancelled() && "
                        "(github.event.inputs.queue_kick != 'true' || github.ref != 'refs/heads/dev') &&",
                        1,
                    )
```
- In `test_pr_context_and_base_diff_logic_are_workflow_run_safe`, add the exception
  `if name == "security-required.yml": expected_cancel = "${{ github.event.inputs.queue_kick != 'true' }}"`, with a
  comment: a kick must never cancel a running kick (spec 4.6).
- Re-pin `DIRECT_TRIGGER_DIGESTS` for every workflow whose `on:` changed (Step 3 changes `security-required.yml`, and
  the `base_sha` description in `backend-required.yml`, `coverage-required.yml`, `e2e-required.yml`): compute with the
  test's own formula after Step 3 and paste the new values:

```bash
python - <<'PY'
import hashlib, json, yaml, pathlib
for n in ("backend-required.yml", "coverage-required.yml", "e2e-required.yml", "security-required.yml"):
    d = yaml.safe_load(pathlib.Path(".github/workflows", n).read_text())
    t = dict(d["on"] if "on" in d else d[True]); t.pop("workflow_run", None)
    print(n, hashlib.sha256(json.dumps(t, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/CI/test_merge_queue_gate_workflows.py tldw_Server_API/tests/CI/test_merge_queue_frontend_required.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`
Expected: FAIL (no `queue-kick` job, ticks still run the queue, frontend guard still exempts the bot).

- [ ] **Step 3: Edit the workflows**

`security-required.yml`, `on.workflow_dispatch.inputs` becomes:

```yaml
    inputs:
      base_sha:
        description: Exact base commit for change detection (manual diagnostic dispatches)
        required: false
        type: string
      queue_kick:
        description: Set to 'true' only by the merge queue, on dev; runs the queue instead of the gate
        required: false
        type: string
      wait_run:
        description: With queue_kick, the id of a run the queue waits for before deciding
        required: false
        type: string
```

Its `concurrency.cancel-in-progress` becomes `${{ github.event.inputs.queue_kick != 'true' }}` (group unchanged), with
a comment above: "A queue kick (spec 4.6 of the 2026-10-11 merge-queue redesign) must never cancel a running kick;
every other run keeps cancel-in-progress."

In both the `changes` job's and the `security-required` job's `if`, insert
`(github.event.inputs.queue_kick != 'true' || github.ref != 'refs/heads/dev') &&` right after the leading
`always() && !cancelled() &&`, leaving the rest of each condition as it is. (Written without a leading `!`, and with the
status functions still first; review improvement I2.) `queue_kick` is a string input: the dispatch API sends every
input as a string, and comparing a boolean input to `'true'` is a known trap (`inputs.queue_kick` would be a boolean).

Delete the old `queue-tick` job's comment and job in all six gate workflows and put this in its place (with
`<gate>` = the workflow's gate job id: `backend-required`, `security-required`, `coverage-required`,
`frontend-required`, `e2e-required`, `container-build-check`):

```yaml
  # Wakes the merge queue when the gate above fails (spec 4.6 of
  # Docs/superpowers/specs/2026-10-11-merge-queue-approve-held-runs-design.md). A red gate causes
  # none of the events merge-queue.yml listens for, and GitHub only re-runs a completed run, so
  # this job cannot retry its own run: it dispatches a queue kick on dev that waits for this run
  # to complete, then runs the queue. Not required and in no job's `needs`, so it cannot turn the
  # required check red. Skipped while MERGE_QUEUE is unset or off, for forks, and on dispatches.
  queue-tick:
    name: Merge queue tick
    needs: [<gate>]
    if: >-
      always() && !cancelled() &&
      needs.<gate>.result == 'failure' &&
      (vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on') &&
      github.event_name == 'pull_request' &&
      github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      actions: write
    steps:
      - name: Wake the merge queue
        env:
          GH_TOKEN: ${{ github.token }}
          REPOSITORY: ${{ github.repository }}
          RUN_ID: ${{ github.run_id }}
        run: gh api -X POST "repos/${REPOSITORY}/actions/workflows/security-required.yml/dispatches" -f ref=dev -f "inputs[queue_kick]=true" -f "inputs[wait_run]=${RUN_ID}"
```

Append to `security-required.yml`, after its `queue-tick`:

```yaml
  # The merge queue's wake-up (spec 4.6, D3). A queue-tick, or the queue itself, dispatches this
  # workflow on dev with queue_kick=true: GitHub starts a dispatch only for a workflow file that
  # exists on main, and merge-queue.yml does not yet. Every gate job above skips; this job waits
  # for the run named by wait_run (if any), then runs the queue. It checks out dev, never a pull
  # request, so the code holding these write permissions is always dev's.
  queue-kick:
    name: Merge queue kick
    if: >-
      github.event_name == 'workflow_dispatch' &&
      github.event.inputs.queue_kick == 'true' &&
      github.ref == 'refs/heads/dev' &&
      (vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on')
    runs-on: ubuntu-latest
    # Same budget as merge-queue.yml's job: a kick may wait 3 minutes for its run, then re-run up to six gates.
    timeout-minutes: 15
    permissions:
      # write: rebase the front PR's branch, comment and disarm, approve/re-run/cancel/delete runs.
      contents: write
      pull-requests: write
      actions: write
      # read: the six required check runs and the license commit status on the front PR's head.
      checks: read
      statuses: read
    steps:
      - name: Checkout dev
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd
        with:
          ref: dev
          persist-credentials: false
      - name: Advance the queue
        env:
          GH_TOKEN: ${{ github.token }}
          MERGE_QUEUE: ${{ vars.MERGE_QUEUE }}
          WAIT_RUN: ${{ github.event.inputs.wait_run }}
        run: python3 -m Helper_Scripts.ci.merge_queue
```

In `backend-required.yml`, `coverage-required.yml` and `e2e-required.yml`, change the `base_sha` input description to
`Exact base commit for change detection (manual diagnostic dispatches)`. In `frontend-required.yml`, replace the gate
job's `name:` line and its comment with:

```yaml
    # A dispatch publishes a diagnostic name, so it can never satisfy the protected check (D2 of the
    # 2026-10-11 merge-queue redesign: the queue no longer dispatches this gate, so no actor is exempt).
    name: ${{ github.event_name == 'workflow_dispatch' && 'frontend-required-diagnostic' || 'frontend-required' }}
```

Re-pin `DIRECT_TRIGGER_DIGESTS` with the command in Step 1.

- [ ] **Step 4: Run the workflow tests and actionlint**

Run: `python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/CI/test_merge_queue_gate_workflows.py tldw_Server_API/tests/CI/test_merge_queue_frontend_required.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py tldw_Server_API/tests/CI/test_merge_queue_workflow.py tldw_Server_API/tests/CI/test_admin_ui_vitest_ratchet_workflow.py tldw_Server_API/tests/CI/test_merge_queue_license_dispatch.py`
Expected: PASS.

Run actionlint the way `actionlint.yml` does (download it as that workflow does, then
`./actionlint -color -config-file .github/actionlint.yaml`). Expected: no findings in the six edited files.

Mutation check (then revert): (1) drop `github.ref == 'refs/heads/dev' &&` from the kick job: the kick-`if` test
FAILS; (2) set security-required's `cancel-in-progress: true`: `test_a_kick_never_cancels_a_running_kick` FAILS;
(5) drop the kick exclusion from the `changes` job: `test_security_required_kick_skips_every_gate_job_only_on_dev` and
the contract pin FAIL;
(3) restore `github.actor != 'github-actions[bot]' &&` in the frontend name: `test_every_dispatch_publishes_the_diagnostic_name`
FAILS; (4) add `contents: write` to one tick: the tick shape test and `_assert_exact_queue_tick` FAIL.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows tldw_Server_API/tests/CI
git commit -q -m "ci(merge-queue): queue-tick sends a kick to security-required on dev; frontend name guard has no exemption (TASK-<N>)"
```

---

### Task 4: Docs and task record

**Files:**
- Modify: `Docs/Development/CI_REQUIRED_GATES.md` ("Merge Queue" section), `AGENTS.md` and `CLAUDE.md` (the queue
  `on` bullet), `Docs/ADR/063-in-repo-merge-queue.md`, `Docs/superpowers/specs/2026-10-04-merge-queue-design.md`
  (one pointer line), `Docs/Published/` (regenerated), the task file (backlog-py)

- [ ] **Step 1: `CI_REQUIRED_GATES.md`**

- After the "designed in" line add: "and amended by `Docs/superpowers/specs/2026-10-11-merge-queue-approve-held-runs-design.md`
  (TASK-<N>)".
- Rule 4 becomes: "4. The queue approves a rebased PR's six required gate runs itself; approving one by hand is
  harmless. Never dispatch a required gate by hand to unblock a PR: a dispatched run's check does not count toward the
  PR's required statuses, so the PR stays blocked."
- Rule 5's list of causes adds "a gate run GitHub would not re-run or approve".
- The "Behind `dev`" bullet becomes: "**Behind `dev`:** rebases it with a pinned head. GitHub then holds the new head's
  `pull_request` runs for approval. The queue dispatches `frontend-license-gate.yml` on `dev` with the PR number,
  approves the held runs of the six required gates, deletes the other held runs (so `ci.yml` and the rest are not
  re-run), cancels the old head's runs and comments once."
- The "A status failed" bullet becomes: "**A status failed:** re-runs that gate's failed jobs once (the license gate is
  dispatched on `dev` again), and evicts the PR if it fails a second time on the same head."
- Replace the "Non-required workflows" sentence with: "Non-required workflows are not re-run on the rebased head: their
  held runs are deleted."
- The `frontend-required.yml` bullet becomes: "`frontend-required.yml` publishes `frontend-required-diagnostic`, never
  the protected name, on any dispatch."
- The wake bullet becomes: "The queue wakes on arm, disarm, close, pushes to `dev`, and a failed required gate: that
  run's `queue-tick` job dispatches `security-required.yml` on `dev` with `queue_kick=true`, whose `queue-kick` job
  waits for the failed run to finish and then runs the queue. Nothing wakes it on a timer. If the line looks stuck,
  arming or disarming a PR moves it."
- "Switching it on" step 3 adds: "one throwaway PR with a deliberately failing gate shows the tick, the kick and the
  retry, and that the retry's own tick fires when the retry fails too. Check on the first PR that the queue's rebase
  commits (committed by `github-actions[bot]`) do not make the `dev` ruleset ask for an approving review: its
  `require_extra_approval_for_unattributed_changes` is on, and GitHub does not say which commits it counts. If they
  do, switch that option off for `dev`; nothing in the queue can change who commits a rebase."

- [ ] **Step 2: `AGENTS.md` and `CLAUDE.md`**

In the queue `on` bullet replace `Never click "Approve and run" on a queue-rebased PR.` with `The queue approves a
rebased PR's gate runs itself; never dispatch a required gate by hand to unblock a PR (a dispatched check does not
count).`

- [ ] **Step 3: ADR-063 and the old spec**

In ADR-063 Consequences, replace the bullet "The seven required statuses are produced by `workflow_dispatch` ..." with
"After a rebase the queue approves the six held `pull_request` gate runs GitHub creates for the new head and dispatches
only the license gate on `dev`; a dispatched gate's check would not count (amendment below)." and replace the
`frontend-required.yml` bullet with "`frontend-required.yml` keeps its guard: no dispatch publishes the protected check
name." Append:

```markdown
## Amendment (2026-10-14): approve held runs, never dispatch a gate

The queue as first merged (#3188) could not land a PR: a `workflow_dispatch` run's check never enters the PR's status
rollup (probe #3220). Probe #3221 showed that a `GITHUB_TOKEN` rebase makes GitHub hold a `pull_request` run for every
workflow, and that approving a gate's held run makes its check count. The queue now approves the six gate runs,
re-runs a failed gate in place, deletes the other held runs, and wakes itself through a `security-required.yml` kick
on `dev`. Spec: `Docs/superpowers/specs/2026-10-11-merge-queue-approve-held-runs-design.md` (TASK-<N>). Owner decisions
D1-D3 (2026-10-14): delete the non-gate held runs; remove the `frontend-required` bot exemption; host the kick in
`security-required.yml`.
```

At the top of `Docs/superpowers/specs/2026-10-04-merge-queue-design.md`, under its status line, add:
`- **Amended by:** \`Docs/superpowers/specs/2026-10-11-merge-queue-approve-held-runs-design.md\` (sections 4.1-4.8, 6, 7 and 9 changed there).`

Run `Helper_Scripts/refresh_docs_published.sh` (the onboarding docs gate checks `Docs/Published`).

- [ ] **Step 4: Task notes**

```bash
# The notes say: the approach; that the action layer is a port of tldw_chatbook scripts/merge_queue.py at 2716b844e1;
# the twelve mutation checks of Task 2 Step 2 and Task 3 Step 4, each with the test that failed; the CI test counts
# from Task 0 Step 2 and Task 5 Step 1; the review findings fixed (M1-M5, m1-m5, m7, I2, I4, I5); the files changed.
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-<N> --notes "$NOTES"
for i in 1 2 3 4 5 6 7 8 9; do PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-<N> --check-ac $i; done
```

Leave the status `In Progress`: the task is done only after the owner's one-PR trial (spec section 7, step 4).

- [ ] **Step 5: Commit**

```bash
git add Docs AGENTS.md CLAUDE.md backlog
git commit -q -m "docs(ci): merge queue approves held gate runs; ADR-063 amendment (TASK-<N>)"
```

---

### Task 5: Full verification and the PR

- [ ] **Step 1: The whole CI test directory and backend-required's contract step**

```bash
PATH=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin:$PATH python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/CI
```
Expected: PASS. Compare the count with Task 0 Step 2: fewer dispatch-era tests, more approval tests; list the
removed tests by name in the PR body.

- [ ] **Step 2: Lint and the repository's own gates**

```bash
PATH=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin:$PATH ruff check Helper_Scripts/ci/merge_queue.py tldw_Server_API/tests/CI/test_merge_queue_*.py
PATH=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin:$PATH bandit -q Helper_Scripts/ci/merge_queue.py
PATH=/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin:$PATH pre-commit run --files $(git diff --name-only origin/dev...HEAD)
```
Expected: no new findings against `origin/dev` (compare counts for any that pre-exist).

- [ ] **Step 3: Push and open the PR (do not arm, do not merge)**

```bash
git push -u origin ci/merge-queue-approve-held-runs
gh pr create --base dev --title "ci(merge-queue): approve held gate runs instead of dispatching them (TASK-<N>)" --body-file <body>
```
The body states: why (#3220, #3221), what changed by file, the five mutation checks, test counts, that `MERGE_QUEUE`
stays unset, and the owner's rollout steps (spec section 7: Allow auto-merge, `dry` for a day, one throwaway PR under
`on` with the switch restored in the same command on a deadline, a second throwaway PR with a failing gate). End with
the Claude Code attribution line. Then follow the owner's procedure: wait for Qodo, address every comment, and merge
only on the owner's word.

---

## Rollout (owner, after the PR merges; from spec section 7)

1. Switch on **Allow auto-merge** (repository settings).
2. `gh variable set MERGE_QUEUE --body dry` for about a day; compare the logged decisions with what armed PRs do.
3. One throwaway PR under `on`, the switch set and restored by the same watching command on a deadline. It must show
   the rebase, the license status, six approved gates in the rollup, auto-merge firing, and the four open points of
   spec section 2. Also: the `queue_kick` string input is accepted, the kick's skipped `security-required` check on
   `dev`'s tip shows on no PR, and the `dev` ruleset does not ask for an approving review on the queue's rebased commits
   (`require_extra_approval_for_unattributed_changes` is on; if it does ask, switch it off for `dev`). A second
   throwaway PR with a deliberately failing gate shows the tick, the kick and the retry, and that the retry's own tick
   fires when the retry fails (`rerun-failed-jobs` must re-run the dependent `queue-tick`).
4. `on` for all traffic. Rollback at any step: `gh variable set MERGE_QUEUE --body off`.

## Accepted residuals (from the second review, 2026-10-14)

- n2: `_restart_cancelled` and `_retry_gate` re-run first and comment second, as the chatbook queue does. If the comment
  fails transiently the cap marker is lost and one extra restart or retry is possible; posting the marker first would be
  worse (a failed re-run would then evict on the next wake). Bounded at one extra run.
- n3: a cancelled gate run wakes nothing (the tick fires on `failure` only), so a cancellation is acted on at the next
  queue event. What a gate job reports when it hits its timeout is an open point of the trial; if it is `cancelled`,
  let the tick also fire on `needs.<gate>.result == 'cancelled'` (its `!cancelled()` already skips superseded runs).

## Known gap, not in this plan

A failed **license** status wakes nothing: the license gate runs on `dev` and has no queue-tick. With every gate green
and the license red, the front PR waits for the next arm, disarm, close or push to `dev`. This was true before the
redesign too; the spec does not change it. Raise it with the owner if the trial shows it.

# In-repo merge queue for `dev` — design

- **Date:** 2026-10-04
- **Status:** Owner-approved direction (2026-10-04: "PRs shouldn't have to race each other"); ships switched off
- **Task:** TASK-13452. **ADR:** 063.
- **Origin:** port of the tldw_chatbook queue (tldw_chatbook PR #2996, ADR-218 there). The queue rules and safety
  invariants are the same; section 4 is what this repository's CI forces to be different.
- **Related:** `Docs/Development/CI_REQUIRED_GATES.md`, TASK-13355.

## 1. Problem

`dev` requires seven statuses on a head that is up to date with `dev` (strict). Every merge therefore puts every other
ready PR behind, each author or agent rebases, and all of them restart the required gates to race for the next merge.
Only one can win; the rest repeat. Measured on PR #3155 (2026-10-04): the required gates took 65 to 120 minutes per
cycle, `dev` moved twice during those cycles, and four other ready PRs were behind at the same time.

## 2. Goals and non-goals

Goals:
1. PRs merge into `dev` one at a time, in the order they were armed for auto-merge. Only the PR at the front is rebased
   and tested. Every other armed PR is left untouched until it reaches the front.
2. All seven required statuses are produced on the head that actually merges.
3. No scripts for the owner or an agent to run, and no personal access token.

Non-goals:
- Raising the one-merge-per-gate-cycle ceiling (strict rules impose it).
- Fork PRs and PRs opened by bots or apps: they are never queued and stay manual.
- Re-running non-required workflows on the rebased head (section 4.5).

## 3. Constraints

- Only the built-in `GITHUB_TOKEN`. No PAT, App key or deploy key.
- Nothing may depend on a file or change on `main`. `schedule`, `workflow_run`, `check_suite` and `check_run` triggers
  read the default branch, so the queue uses none of them.
- PRs are rebased onto `dev`, never merged with `dev`.
- The queue never enables auto-merge, never merges and never pushes. A merge made with `GITHUB_TOKEN` creates no
  workflow runs, which would silently stop the queue and `dev`'s post-merge checks.
- Runner minutes are the scarce resource (TASK-13355 history, shard cap in #3150). The queue must not add jobs to the
  common green path.

## 4. What differs from the chatbook queue

### 4.1 Seven required contexts instead of one

| Context | Kind | Produced by | Queue dispatch |
|---|---|---|---|
| `backend-required` | check run | `backend-required.yml` | on the PR branch, input `base_sha` |
| `security-required` | check run | `security-required.yml` | on the PR branch, input `base_sha` |
| `coverage-required` | check run | `coverage-required.yml` | on the PR branch, input `base_sha` |
| `frontend-required` | check run | `frontend-required.yml` | on the PR branch, input `base_sha` (already required there) |
| `e2e-required` | check run | `e2e-required.yml` | on the PR branch, input `base_sha` |
| `container-build-check` | check run | `container-build-check.yml` | on the PR branch, no input |
| `frontend-license-policy/trusted/dev` | commit status | `frontend-license-gate.yml` | on `dev`, input `pr` |

The queue reads each context's runs on the head and reduces them to one of `missing`, `running`, `passed`, `failed`
(once) or `failed twice`. The front PR's verdict is the combination, in this order:

- any context `failed twice`: evict;
- else any context `failed` once: dispatch those contexts again (retry) and say so in one comment;
- else any context `running`: wait;
- else any context `missing`: dispatch those contexts, if the head is older than 3 minutes;
- else all `passed`: the green rows of section 6.

Failures are acted on while other contexts are still running. Only a failed gate wakes the queue (4.6), so a gate that
failed early while a slower one later passed would otherwise never be retried or evicted. A retry dispatches only the
contexts that failed, not all seven.

The license context is a commit status with no run behind it. A `pending` status older than 15 minutes is treated as
absent, because a license run cancelled after posting `pending` would otherwise read as running forever.

### 4.2 A token rebase starts nothing

Events caused by `GITHUB_TOKEN` create no workflow runs except `workflow_dispatch`. After the queue rebases the front
PR, no `pull_request` run and no `pull_request_target` run starts on the new head, so none of the seven contexts would
ever be reported. The queue dispatches all seven itself (table above).

### 4.3 Comparison base on a dispatched run

`.github/actions/detect-required-gate-changes` already honours `github.event.inputs.base_sha`, but only
`frontend-required.yml` declares that input. On any other dispatch the action falls back to `HEAD^`, so a gate would
compare only the last commit and could pass as a no-op with nothing tested. Each of `backend-required.yml`,
`security-required.yml`, `coverage-required.yml` and `e2e-required.yml` gains an optional `base_sha` dispatch input, and
every step that resolves a comparison base uses `inputs.base_sha` before any `pull_request` field or `HEAD^`. The queue
always passes `dev`'s tip.

`security-required.yml` runs dependency review only for `pull_request` and `workflow_run` events. It gains a dispatch
arm (`workflow_dispatch` with a non-empty `base_sha`), so a queue-dispatched run cannot pass without it.

If `dev` moves between the queue's read and the run, `base_sha` is no longer an ancestor of the head. The diff then
lists the PR's files plus `dev`'s newer ones, a superset, so a gate can only run more, never less.

### 4.4 The license status

`frontend-license-gate.yml` runs on `pull_request_target` and posts the commit status. It gains a `workflow_dispatch`
trigger with a `pr` input, served by a separate job so the existing `pull_request_target` job stays as it is apart from
an event guard. The dispatch job:

- runs only from `refs/heads/dev` (`if:` on `github.ref`), so the policy code is always `dev`'s;
- resolves the PR through the API (number must be numeric, PR open, base `dev` or `main`), taking the head SHA and
  author from the PR and the base SHA from the base branch's current tip;
- then runs the audit job's own steps, copied verbatim. A contract test fails if the two jobs' steps drift apart.

The workflow's concurrency group becomes `frontend-license-gate-<PR number or dispatch input>`, with the input read
through `github.event.inputs`, which exists on every event. A `pull_request_target` run keeps the group it had, and a
dispatch for PR N shares N's group.

A PR branch could add this trigger to its own copy and dispatch it there with `statuses: write`. That is no new
exposure: any same-repo workflow run can already declare that permission, which GitHub documents for write access.

### 4.5 Non-required workflows are not re-run

The chatbook queue re-dispatches every workflow that ran on the old head. Here the queue dispatches only the seven
required contexts. `ci.yml` alone is over 200 jobs on a PR and its dispatch form also starts the macOS and Windows
matrices. Those workflows ran on the author's last pushed head, and `dev`'s own push workflows run after the merge.

### 4.6 Waking the queue without spending runners

`merge-queue.yml` wakes on a PR being armed, disarmed or closed, and on pushes to `dev`. When the front PR goes green,
auto-merge fires, `dev` is pushed and the queue wakes: no extra job is needed on the green path.

A red gate produces no such event, so each of the six required workflows gains a `queue-tick` job that runs **only when
its gate job failed** and the queue mode is `dry` or `on`, for a dispatch or a same-repo `pull_request` run. It is not
required and not in any gate's `needs`, so it can never turn a required check red. Its job-level permissions are the
queue's: `contents`, `pull-requests` and `actions` write, `checks` and `statuses` read.

A tick runs inside its gate's own run. When the queue then dispatches that same workflow on the same branch (a retry
of that gate, or a rebase), the new run joins the same concurrency group (workflow, event, ref, cancel-in-progress) and
cancels the run the tick is executing in. The script therefore sends the host workflow's dispatch last, after every
other dispatch, the cancellation of superseded runs and the comment, so a cancelled tick loses nothing.

Known gaps, accepted:
- A failed license status has no tick (the trusted workflow stays minimal), and "green but auto-merge did not fire" is
  noticed only at the next wake. Both wait for the next arm, disarm, close, push or tick.
- A tick fires on a gate result of `failure` only. If a gate job that hits its timeout reports `cancelled`, no tick
  fires and the script reads a cancelled run as no run; the next wake dispatches it again.
- With the queue on, a failed gate on a Dependabot PR starts a tick whose token is read-only. If the front PR needs an
  action at that moment the tick fails, as a red non-required check on the Dependabot PR; nothing is evicted.
- Every PR shows six skipped `Merge queue tick` rows and one skipped `frontend-license-gate-dispatch` row.

### 4.7 Repository settings the owner must change before `on`

- **Allow auto-merge** is off. Arming auto-merge is how a PR joins the line and how it merges.
- **Always suggest updating pull request branches** is off. Whether `updatePullRequestBranch` needs it is not known;
  the trial answers that (section 8).

### 4.8 `frontend-required` and its manual-dispatch guard

`frontend-required.yml` names its gate job `frontend-required-diagnostic` on a `workflow_dispatch`, so a person
starting the workflow by hand cannot satisfy the protected check name
(`test_manual_dispatch_cannot_publish_the_required_check_name`). The guard stays. It is narrowed to dispatches whose
`github.actor` is not `github-actions[bot]`: a dispatch made with `GITHUB_TOKEN`, which is how the queue starts it,
publishes `frontend-required`; every other dispatch still publishes the diagnostic name.

A workflow in a PR branch could dispatch with its own `GITHUB_TOKEN` and so reach the real name. That needs write
access and an edited workflow, the same trust level as editing `frontend-required.yml` in the branch.

### 4.9 A gate whose change detection failed must be red

In the five gates with a `changes` job, the gate job required `needs.changes.result == 'success'`. When `changes`
failed, the gate job was skipped, and GitHub counts a skipped required job as satisfied. This predates the queue (a
runner flake in `changes` is enough) and a dispatch adds one more way to reach it (a `base_sha` that is not a commit in
the clone). The gate job now also runs when `changes` did not succeed, and fails in a guard step before anything
else runs. In `backend-required.yml` the existing arm and step that turn a negative license verdict red are unchanged
and stay first. The other four gates never had that arm: on a negative verdict they are still skipped, and
`backend-required` and the license status are what block the merge.

## 5. Architecture

1. **`Helper_Scripts/ci/merge_queue.py`.** Standard library plus `gh`. A pure `line_of` and `decide_front`, a read
   layer, and an action layer. Run as `python3 -m Helper_Scripts.ci.merge_queue`.
2. **`.github/workflows/merge-queue.yml`.** Triggers: `pull_request` types `auto_merge_enabled`, `auto_merge_disabled`,
   `closed` for base `dev`, and `push` to `dev`. One job, guarded by the mode and by same-repo, with job-level
   permissions. It checks out `dev` and runs the script.
3. **The six required workflows.** 4.3, 4.6, 4.8 and 4.9.
4. **`frontend-license-gate.yml`.** The dispatch job (4.4).
5. **Docs.** `CI_REQUIRED_GATES.md`, mode-dependent merge rules in `AGENTS.md` and `CLAUDE.md`, ADR-063.

## 6. Queue rules

**The line.** Open PRs with base `dev`, auto-merge armed, not a draft, head in this repository, opened by a user.
Ordered by `autoMergeRequest.enabledAt`, oldest first. An armed fork PR or bot-authored PR gets one comment saying it
is not queued.

**Front PR.** `UNKNOWN` merge state is re-read up to 12 times, 10 seconds apart, then left alone.

| Front PR state | Action |
|---|---|
| `BEHIND` | Rebase (section 7), then dispatch all seven contexts, cancel the old head's live runs, comment once |
| `DIRTY` | Evict: conflicts with `dev` |
| Up to date, verdict per 4.1 is evict, retry, wait or dispatch | That action |
| All passed, `BLOCKED` with unresolved review threads | Evict: blocked by unresolved conversations |
| All passed, last context completed 15 minutes ago or less | Wait: auto-merge is about to fire |
| All passed, last context completed more than 15 minutes ago | Evict: auto-merge did not fire, re-arm to retry |

After an eviction the queue evaluates the new front in the same run, at most 10 times. PRs behind the front are never
rebased, dispatched or commented on.

**Modes**, from the repository variable `MERGE_QUEUE`: unset or `off` does nothing; `dry` writes decisions to the job
summary with no side effects; `on` acts.

## 7. Actions and safety invariants

- **Rebase:** `updatePullRequestBranch(updateMethod: REBASE, expectedHeadOid)`. The mutation returns the old head and
  the branch moves about a second later, so the queue polls for the new head (10 times, 3 seconds apart) before
  dispatching. A refused rebase is re-read, and re-read once more after 3 seconds, before it counts as this run's own
  failure: head moved means someone else acted; `DIRTY` evicts; otherwise one `rebase-failed` comment, then eviction on
  the second failure for the same head.
- **Dispatch:** immediately before dispatching a context, re-read its live runs on the head and skip it if one
  appeared. A branch refusing a dispatch (HTTP 422 or 404) evicts the PR; any other error fails the queue run, so an
  outage never disarms a PR. A check context is never dispatched without `dev`'s tip as `base_sha`.
- **Evict:** `disablePullRequestAutoMerge` plus one comment. Re-arming puts the PR at the back.
- **Comments** carry `<!-- merge-queue:<kind>:<head-sha> -->` and are never repeated for the same kind and head.
- **List reads** follow pagination to the end (up to 10 pages of 100) and fail the run beyond that. A head here carries
  over 250 check runs, so the check-run read always spans pages.
- **Forbidden,** enforced by a guard test: enabling auto-merge, merging, any git push.
- Every action is safe to repeat; two racing queue runs produce at most one rebase.

## 8. Rollout and rollback

1. This PR merges the normal way with `MERGE_QUEUE` unset. The queue does nothing; the one change every PR sees is 4.9.
2. Owner enables auto-merge in repository settings.
3. Owner sets `MERGE_QUEUE=dry` for about a day and compares the logged decisions with what armed PRs actually do.
4. Trial with one low-risk PR under `on`. It answers what cannot be checked without a real run on this repository:
   - the rebase mutation with branch updates switched off (4.7);
   - a `GITHUB_TOKEN` dispatch reports `github.actor == 'github-actions[bot]'`, so `frontend-required` is published
     under its real name (4.8);
   - the license dispatch on `dev` posts the status on the PR head (4.4);
   - dependency review accepts explicit refs on a dispatch (4.3);
   - the job-level write permissions of `queue-tick` are granted;
   - a tick that dispatches its own workflow completes everything else first (4.6);
   - what result a gate job reports when it hits its timeout (4.6).
5. `on` for all traffic.

Rollback: `gh variable set MERGE_QUEUE --body off`, effective immediately.

## 9. Agent rules (AGENTS.md and CLAUDE.md)

Checked with `gh variable get MERGE_QUEUE`.

- **`on`:** arm auto-merge once reviews are settled, then leave the PR alone. Never rebase or update an armed PR, never
  merge an armed PR by hand, and disarm (`gh pr merge <n> --disable-auto`) before pushing more work. Never click
  "Approve and run" on a queue-rebased PR.
- **`off` or `dry`:** today's procedure (rebase the one PR about to merge, wait for the required gates, merge).

## 10. Testing

- Table tests for `decide_front`, one per row of 4.1 and section 6, plus line ordering and eligibility.
- Action-layer tests against a fake `gh`: pinned-head rebase, dispatch only after a successful rebase, per-context
  dispatch inputs (branch and `base_sha` for five gates, branch alone for `container-build-check`, `dev` and `pr` for
  the license gate), retry of failed contexts only, comment dedup, `dry` makes no mutating call, refused dispatch
  evicts, pagination.
- Guard test: the script cannot arm, merge or push.
- Workflow-shape tests: `merge-queue.yml` triggers and permissions; the five gates with a `changes` job declare
  `base_sha` and fail when `changes` did not succeed; all six have a failure-only `queue-tick` outside every `needs`;
  the `frontend-required` name guard exempts only `github-actions[bot]`; the license dispatch job is `dev`-only and its
  evaluate and publish scripts equal the `pull_request_target` job's.
- Existing contract tests under `tldw_Server_API/tests/CI` and `tests/Infrastructure` stay green. The license-first
  contract excludes `merge-queue.yml` by name and exempts `queue-tick` from its no-write-credentials rule, each behind a
  test that pins the whole exempted shape.
- `backend-required` runs the six `test_merge_queue_*` files in its contract step, so they are enforced by a required
  check.

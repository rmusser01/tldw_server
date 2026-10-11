# Merge queue for `dev`: approve the held gate runs instead of dispatching them

- **Date:** 2026-10-11
- **Status:** Draft for owner review. Nothing here is implemented.
- **Amends:** `Docs/superpowers/specs/2026-10-04-merge-queue-design.md` (TASK-13462, ADR-063), merged switched off in
  #3188. Sections of that spec are cited as "old 4.2" and so on. Anything this document does not mention stays as written
  there.
- **Reference implementation:** the tldw_chatbook queue after its own redesign, `scripts/merge_queue.py` and
  `Docs/superpowers/specs/2026-10-03-merge-queue-design.md` at tldw_chatbook `dev` `897a320538`. It made this same change
  on 2026-10-10 (PR #3039, ten review rounds) and landed a PR end to end (#3060).

## 1. Why

The queue as built cannot land a PR. After it rebases the front PR it starts the six gate workflows with
`workflow_dispatch`. A dispatched run's check never enters the PR's status rollup, so the required check stays missing and
the PR stays `BLOCKED` with every gate green. tldw_chatbook hit this live on 2026-10-06; two probes on this repository on
2026-10-11 confirm it here and show what does work.

The queue has never been switched on in this repository, so nothing is broken in use today.

## 2. Verified on this repository (2026-10-11)

Two throwaway PRs, both closed unmerged with their branches deleted. Only the built-in `GITHUB_TOKEN` acted; no repository
setting was changed.

| # | Fact | Evidence |
|---|------|----------|
| S1 | `updatePullRequestBranch(updateMethod: REBASE, expectedHeadOid)` works with "Always suggest updating pull request branches" off. The mutation returns the old head and the branch moves within seconds | #3221: `fc50771461` -> `bee4ea42ae`, probe run 38152463265 |
| S2 | That rebase creates a **held** `pull_request` run (`completed` / `action_required`, actor `github-actions[bot]`) for every `pull_request` workflow: 12 runs, the six gate workflows among them, plus `ci.yml`, `pre-commit`, `codeql`, `e2e-smoke`, `sbom` and `frontend-ux-gates` | #3221, same run |
| S3 | It creates **no** `pull_request_target` run, so nothing posts the license status on the new head. The license run seen there came from a bot's `edited` event 15 seconds later | #3221: run 38152491613, triggering actor `cubic-dev-ai[bot]` |
| S4 | `POST /actions/runs/{id}/approve` with the token starts a held run. It runs as attempt 2, and its check **is** in the PR's rollup as the required context | #3221: run 38152479149; `security-required` SUCCESS, `isRequired=true`, event `pull_request` |
| S5 | A gate started by `workflow_dispatch` on the PR branch passes and is attached to the head commit, but is **absent** from the PR's rollup | #3220: run 38106344344 |
| S6 | The license context is a commit status and counts whoever posts it | #3220 and #3221: `frontend-license-policy/trusted/dev` SUCCESS, `isRequired=true` |

Old 4.2 ("a token rebase starts nothing") is wrong (S2), and old 4.7's open question about the update-branch setting is
answered (S1).

Not yet verified here; the one-PR trial (section 7) answers them:

- `frontend-required` reports under its real name in an approved `pull_request` run. Its manual-dispatch guard keys on
  `workflow_dispatch`, so it should not apply.
- A re-run of a failed gate run adds an attempt to the same check suite and is read as the newer result (tldw_chatbook V3).
- The `queue-tick` job's write permissions are granted inside an approved run.
- What a gate job reports when it hits its timeout.

## 3. The change in one paragraph

After rebasing the front PR, the queue posts the license status the way it does today (a dispatch on `dev`), then
**approves the six held gate runs** that the rebase created. It never dispatches a check context again. A gate that fails
is retried by **re-running that run**, not by starting a new one. The held runs of the other six workflows are not
approved, so a queue rebase never re-runs the full suite; that keeps old 4.5's decision, by a different route.

## 4. What changes, by section of the old spec

**4.1, how each context is started.**

| Context | Today | After |
|---|---|---|
| `backend-required`, `security-required`, `coverage-required`, `frontend-required`, `e2e-required`, `container-build-check` | dispatch on the PR branch | approve the held `pull_request` run on the head; if there is none, re-run the head's existing run for that workflow |
| `frontend-license-policy/trusted/dev` | dispatch on `dev` with `pr` | unchanged |

The verdict order (evict, retry, wait, start, green) is unchanged. Only "start" and "retry" act differently:

- **Start** (a context is missing on a head older than 3 minutes): approve its held run. With no held run, re-run the
  workflow's completed run on that head (a cancelled run, or one that never reported its check). With no run at all, evict
  (`evict-no-run`): the queue has nothing it may act on, and dispatching would not count (S5).
- **Retry** (a context failed once): re-run the failed jobs of that context's run. GitHub may hold the re-run because the
  token caused it; the queue approves it.
- The license status keeps its dispatch for both.

**4.2 is replaced** by S2 and S3.

**4.3, `base_sha`.** The queue no longer passes it: an approved run is a `pull_request` run with its own base. The inputs
and the dependency-review dispatch arm stay in the workflows as a manual diagnostic. They cannot satisfy a required check
(S5). `dispatch_request` keeps only the license arm.

**4.4, the license status.** Unchanged, and now ordered: the queue dispatches it **before** approving the gates. Each
gate's `await_license` job runs in a `pull_request` run and, when `LICENSE_FIRST_CI_ENABLED` is `true`, waits up to 15
minutes for that status on the head. With the variable unset, as it is today, the wait passes at once.

**4.5, non-required workflows.** Still not run on the rebased head. Their six held runs are deleted, as the queue already
does for every held run today (`cleanup_approval_runs`); the function now skips the six gate workflows. Deleting removes
the "Approve and run" button that would otherwise start `ci.yml`'s 200 jobs on a click. See decision D1.

**4.6, waking the queue.** A `queue-tick` runs inside the failed gate's own run, and GitHub only re-runs a run that has
completed, so a tick can no longer retry its own gate directly. Instead:

- the tick starts a **queue kick**: one `workflow_dispatch` on `dev` carrying `wait_run=<its own run id>`;
- the kick waits for that run to complete (30 tries, 6 seconds apart, then decides anyway) and runs the queue, which can
  now re-run it.

GitHub only starts a `workflow_dispatch` run for a workflow file that exists on `main`, and `merge-queue.yml` does not
yet. The kick is therefore hosted by `security-required.yml`, which does: a dispatch on `refs/heads/dev` with
`queue_kick=true` skips every gate job and runs only a `queue-kick` job with the queue's permissions. On any other ref,
or without the flag, the workflow behaves as today. The kick job has its own concurrency group with
`cancel-in-progress: false`, so one kick never cancels another mid-run; GitHub keeps one running and one pending.

The queue also sends a kick when it rebases a PR but cannot see the new head in time, or loses a race to another queue
run, exactly where the chatbook queue does.

The "host workflow dispatched last" rule of old 4.6 goes away with the dispatches.

**4.7, repository settings.** Only **Allow auto-merge** still has to be switched on by the owner. The update-branch
setting can stay off (S1).

**4.8, the `frontend-required` name guard.** The exemption for dispatches by `github-actions[bot]` was added for the queue
and nothing uses it now. It is removed, restoring the guard as it was before #3188: no dispatch by anyone publishes the
required name. See decision D2.

**4.9** is unchanged.

**Section 6, the `BEHIND` row** becomes: rebase, wait for the new head, dispatch the license status, approve the six held
gate runs, delete the other held runs, cancel the old head's live runs, comment once.

**Section 7, actions.** The rules of the chatbook spec's sections 7 and 8 apply as written there, with "the required
check" read as "each of the six gate contexts". The ones that matter most:

- **What may be approved.** Only a run that is held, has event `pull_request`, was caused by `github-actions[bot]`, has
  this repository as its head repository, and belongs to one of the six gate workflows. Only for the front PR, which is
  same-repo, user-authored and armed. A run held for any other reason (an outside contributor's PR, an agent's push
  awaiting a person) is never approved.
- **Never dispatch a check context.** Enforced by a guard test, next to the existing "never arm, merge or push" guard.
- **Stand-ins.** A live gate run counts as in flight only if its event is `pull_request`. A dispatched gate run is
  ignored, whoever started it.
- **Errors.** A transient error (5xx, 429, rate limit, network, GraphQL "Something went wrong") fails the queue run and
  never disarms a PR. A refusal GitHub actually gave is re-read first: a run that is live or held again was handled by a
  racing queue run, so this one stands down. Otherwise it is a strike: one warning comment, and an eviction only once
  that comment is 10 minutes old.
- **Evictions** name their cause (`evict-no-run`, `evict-rerun`, `evict-approve` join the existing kinds), post the
  comment first and disarm second.

**Section 9, agent rules.** "Never click Approve and run on a queue-rebased PR" becomes: the queue approves the gate runs
itself; approving a gate by hand is harmless; never dispatch a required gate by hand to unblock a PR, because a dispatched
check does not count (S5).

## 5. Files

| File | Change |
|---|---|
| `Helper_Scripts/ci/merge_queue.py` | Approve, re-run and kick in place of check dispatches; stand-ins ignore dispatched runs; `cleanup_approval_runs` skips the gates. Ported from the chatbook script, which carries the race and error handling |
| `.github/workflows/security-required.yml` | `queue_kick` and `wait_run` dispatch inputs; the `queue-kick` job; gate jobs skip on a kick |
| The six gate workflows | `queue-tick` sends a kick instead of running the queue in place |
| `.github/workflows/frontend-required.yml` | Remove the `github-actions[bot]` exemption from the name guard (D2) |
| `tldw_Server_API/tests/CI/test_merge_queue_*.py` | Follow the rules above; new guard test for "never dispatch a check context" |
| `test_license_first_workflow_contracts.py` and the other workflow contract tests | Re-pin the changed jobs and inputs; the `queue-tick` exemption shrinks to what a kick needs |
| `Docs/Development/CI_REQUIRED_GATES.md`, `AGENTS.md`, `CLAUDE.md`, ADR-063, the 2026-10-04 spec | Updated to match; ADR-063 gains a dated amendment |

`merge-queue.yml` itself does not change.

## 6. Testing

- Table tests for the changed verdict rows: start approves a held run; start re-runs a cancelled run; start with no run
  evicts; retry re-runs failed jobs and approves the held re-run; failed twice still evicts.
- Action tests against the fake `gh`: license dispatched before any approval; only the six gate workflows approved;
  `ci.yml`'s held run never approved and is deleted; a run held for another actor or from a fork never approved; no call
  ever dispatches a gate workflow except the kick; a transient error disarms nobody.
- One boundary test against the real `Gh` class's argument building, for the approve and re-run calls.
- Workflow-shape tests: the kick runs only on `refs/heads/dev` with the flag and skips every gate job; the kick
  job's concurrency never cancels; each `queue-tick` only sends a kick; the `frontend-required` guard has no
  exemption.
- Each new guard is mutation-checked: break the rule, see the test fail.
- The six `test_merge_queue_*` files stay in `backend-required`'s contract step.

## 7. Rollout

1. The implementation PR merges the normal way with `MERGE_QUEUE` unset. Nothing changes for any PR.
2. Owner switches on **Allow auto-merge**.
3. `MERGE_QUEUE=dry` for about a day; compare the logged decisions with what armed PRs do.
4. One throwaway PR under `on`, with the switch set and restored by the same watching command, on a deadline. It must
   show: the rebase, the license status, six approved gates in the rollup, auto-merge firing, and the four open points of
   section 2. A deliberately failing gate on a second throwaway PR shows the tick, the kick and the retry.
5. `on` for all traffic.

Rollback at any step: `gh variable set MERGE_QUEUE --body off`.

## 8. What this does not do

- It does not make the gates less flaky. The chatbook queue's first day showed that a flaky required check plus the
  fail-twice rule evicts good PRs; the same will be true here and needs its own work if it shows.
- It does not change which checks are required, or when the full suite runs on an author's own push.
- It does not let the queue rebase a PR when both the PR and `dev` changed the same workflow file. The token may not
  rewrite workflow files; such a PR is evicted with the hand-rebase steps (old section 7).

## 9. Owner decisions

| # | Decision | Recommendation |
|---|----------|----------------|
| D1 | The six non-gate held runs on a rebased head: delete them, or leave them held | **Delete.** It is what the queue does today, and it removes a button that starts the full suite |
| D2 | The `frontend-required` guard's exemption for `github-actions[bot]` dispatches: remove it, or leave it unused | **Remove.** Nothing needs it, and it is a relaxation of a guard that a contract test protects |
| D3 | Host for the queue kick: `security-required.yml`, or another workflow that already exists on `main` | **`security-required.yml`.** It is the lightest gate and already takes dispatch inputs. Moving the kick to `merge-queue.yml` becomes possible only after that file reaches `main` |

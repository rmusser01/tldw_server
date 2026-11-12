---
id: TASK-13361
title: A cancelled license-gate job publishes a policy-violation status
status: Done
assignee: []
created_date: '2026-09-23 17:13'
updated_date: '2026-09-27 23:59'
labels:
  - ci
  - security
  - tech-debt
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`.github/workflows/frontend-license-gate.yml` reports infrastructure cancellation as a security-policy failure, so a timed-out checkout is indistinguishable from a genuine license violation.

Verified on PR #3000, 2026-09-23 (a Python-only change that touches no frontend file):

Job `frontend-license-gate-audit`, run 35881979095, steps:
```
1 success   Set up job
2 success   Mark trusted policy pending
3 cancelled Checkout trusted policy      <- killed by the job timeout under queue pressure
4 skipped   Evaluate immutable pull request metadata
5 failure   Publish trusted policy result
```

The publish step (`:98-116`) is `if: always()`, which in GitHub Actions **includes
cancellation**. Its default is `state=failure`, flipped to success only when
`steps.evaluate.outputs.verdict == success`. With step 4 skipped, VERDICT is empty, so it
POSTs a **commit status** of `frontend-license-policy/trusted/dev = failure` with
'Trusted frontend license policy failed closed', then exits 1.

So a cancelled checkout stamps a red policy status on the commit. Same root cause as the
`pre-commit` cancellation on PR #2996: under the queue pressure described in TASK-13359,
jobs finally start and then hit `timeout-minutes` while still inside `actions/checkout`.
Expect recurrence on every PR while the queue is saturated.

**Recommended fix, one line:** change the publish step's condition from `if: always()` to
`if: ${{ !cancelled() }}`.

This preserves every fail-closed path -- I checked each:
- genuine policy violation -> Evaluate runs, verdict != success -> not cancelled -> publishes `failure`. Preserved.
- Evaluate crashes -> not cancelled -> VERDICT empty -> publishes `failure`. Preserved.
- job cancelled -> publish skipped -> the status stays `pending`, set by step 2 at `:31-39` before the checkout. Still blocks the merge, because pending is not success, but no longer asserts a violation, and a re-run resolves it cleanly.
- cancelled before step 2 -> no status at all -> a required context that is absent also blocks. Safe.

So the security property is 'never publish success without evaluating', and `!cancelled()`
does not weaken it. What changes is only that a cancellation stops being reported as a
violation.

Flagging rather than fixing because this is a security gate and its failure semantics are an
owner call. The workaround meanwhile is `gh run rerun <id> --failed`, which republishes the
status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A cancelled license-gate job leaves the trusted-policy status pending, not failure
- [x] #2 A genuine policy violation still publishes failure
- [x] #3 A crashed evaluation step still publishes failure
- [x] #4 Verified by cancelling a run deliberately and observing the resulting commit status
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
BETTER PRIMARY FIX FOUND -- remove the cause, not just the mislabelling.

The audit job checks out at fetch-depth: 0 (:45), a FULL-history clone. Measured locally:
834 MB of history across 17,899 commits. The job's timeout is 5 minutes (:19), and the
failing run took 5m03s, dying inside actions/checkout. So the clone alone does not reliably
fit the budget under contention.

And the job does not use that history in either path:

- Owner-authored PRs (PR_AUTHOR == REPOSITORY_OWNER, :65-71): the evaluate step runs
  check_frontend_license_gate.py with --null </dev/null and exits success immediately. It
  touches no git object at all.
- Other PRs (:73-76): it does its OWN fetches, explicitly shallow --
  `git fetch --no-tags --depth=1` for the base ref and for refs/pull/N/head -- then diffs
  the two fetched SHAs. It deliberately does not rely on the checkout's history.

Nothing else in the job reads history: 'Mark trusted policy pending' and 'Publish trusted
policy result' are both gh api calls. The checkout is needed only for the working tree, to
have Helper_Scripts/ci/check_frontend_license_gate.py on disk.

So `fetch-depth: 1` is sufficient and provably behaviour-preserving, and it removes the
timeout exposure rather than softening how the timeout is reported. `git diff A B` still
works after the two --depth=1 fetches, since both commits' trees are present.

Every PR currently in flight is owner-authored, so all of them take the path that reads no
history while still paying for the full clone.

RECOMMENDED, in order:
1. `fetch-depth: 0` -> `1` on the audit checkout (:45). Removes the cause.
2. `if: always()` -> `if: ${{ !cancelled() }}` on the publish step (:99). Stops a
   cancellation from being reported as a policy violation, for whatever still cancels.

(1) is the one that matters; (2) is defence in depth. Both are one-line changes.

THIRD OCCURRENCE of this cancellation pattern today, so it is not a one-off: pre-commit on
PR #2996 (10-minute timeout, cancelled in checkout), and on PR #3000 both the license audit
(5-minute) and pre-commit. All three died inside actions/checkout. Under the duplicate-run
load in TASK-13359 this will keep recurring.

Fixed on chore/close-fixed-review-tasks (#3029): the publish step is if: "!cancelled()". A cancelled run leaves the pending status from the job's first step, which stays fail-closed. A policy violation or a crashed evaluate step is a failure, not a cancellation, so it still publishes failure (ACs 2 and 3). Pinned in test_frontend_license_gate_workflow.py. AC #4 (cancel a live run and observe the status) can only be done after merge, so the task stays In Progress until then.

CORRECTION 2026-09-27: the !cancelled() fix from #3029 does NOT work, and AC #1 was checked wrongly. Verified live on PR #3031: run 36342617339 was cancelled during Checkout after the pending status posted, and the publisher still ran and posted failure. Run 36342606942, cancelled by this workflow's own cancel-in-progress, did the same. The cancelled step left cancelled() false for the following steps. Replaced on fix/license-gate-publish-on-verdict: the publisher is now gated on always() && steps.evaluate.outcome in {success, failure}, so it publishes only when a verdict exists. always() is needed so a genuine violation (evaluate failed) is not skipped by the implicit success(). A checkout failure now also leaves pending, which stays fail-closed. AC #1 and AC #4 are to be re-verified live after merge.

Follow-up from Qodo on #3032, declined there with reasons: a run superseded by cancel-in-progress but cancelled only after evaluate completes can still publish its verdict over the newer run's status. Pre-existing (always() had it too) and low impact, since superseding runs evaluate the same head SHA under a base-qualified context. A real guard means a latest-run check inside the hash-pinned publish body, which is a deliberate trusted-gate change for its own PR.

CORRECTION 2 (2026-09-27): both live tests above were invalid. pull_request_target runs the workflow file from the DEFAULT branch (main), not the PR base (dev). A runner debug log of rerun 36345134991 attempt 2 shows the Publish step evaluated plain always(), which is main's version. So neither #3029 (!cancelled()) nor #3032 (verdict-gated condition) has run, and the earlier claim that cancelled() stayed false after a cancelled step is unproven: the runs never evaluated it. #3032's condition stays as the fix. It also covers a cancel before evaluation, whatever cancelled() does. It takes effect only when dev is released to main. AC #1 and AC #4 must be verified live after that release. The same applies to any change to a pull_request_target workflow: it is inert until it reaches main.

VERIFIED LIVE 2026-09-27, after v0.1.45 put the fix on main: license-gate run 36353925069 on PR #3035 was cancelled during Checkout after posting pending. Evaluate was skipped, Publish trusted policy result was skipped, and frontend-license-policy/trusted/dev stayed pending. Before v0.1.45 the same cancellation posted failure (runs 36342617339, 36345134991). All ACs are met.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

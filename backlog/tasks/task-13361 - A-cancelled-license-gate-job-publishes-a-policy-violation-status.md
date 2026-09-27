---
id: TASK-13361
title: A cancelled license-gate job publishes a policy-violation status
status: Done
assignee: []
created_date: '2026-09-23 17:13'
updated_date: '2026-09-27 17:24'
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
- [ ] #1 A cancelled license-gate job leaves the trusted-policy status pending, not failure
- [ ] #2 A genuine policy violation still publishes failure
- [ ] #3 A crashed evaluation step still publishes failure
- [ ] #4 Verified by cancelling a run deliberately and observing the resulting commit status
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cause fixed by fetch-depth: 1 (#3004). Mislabel fixed here: the publish step is now if: "!cancelled()", so a cancelled run leaves the pending status instead of posting failure. Stays fail-closed: pending never satisfies the required check. Pinned in test_frontend_license_gate_workflow.py.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

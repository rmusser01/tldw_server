# PR3437 Review and CI Merge Preparation (TASK-15477)

ADR assessment: accepted ADR-097 and ADR-021 govern existing behavior. Correcting
implementation, tests and integration gates requires no new durable decision.
No merge or auto-merge is authorized by this request.

## Stage 1: Triage and Independent Review
**Goal**: Inventory every PR comment, failed CI job and independent code finding.
**Success Criteria**: Complete paginated comments/threads and exact job logs;
an independent requesting-code-review assessment of the current full PR.
**Tests**: Inspect base/head SHAs, actual failed steps and existing task history.
**Status**: Complete

## Stage 2: Correct and Verify
**Goal**: Address verified findings and CI root causes without weakening gates.
**Success Criteria**: Regression-first fixes where applicable, focused green
checks, touched compilation/lint/Bandit and independent fix review.
**Tests**: Exact failed commands, relevant regression suites and security scans;
preserve all budgets, ownership, privacy and fail-closed caller contracts.
**Status**: Complete

Native AuthNZ reproduction found that sequential app lifespans retain closed
Moderation admission. Add restart and retained-ownership regressions before
correcting trusted startup; never reset PID counters or discard live owners.

## Stage 3: Publish and Prepare for Merge
**Goal**: Push corrections, respond to comments and report real merge gates.
**Success Criteria**: PR updated, every finding disposition recorded, required
CI genuinely passing or exact external blockers stated; preserve the recorded
requester Change summary waiver for this PR. Preserve worktree; never forge statuses.
**Tests**: Latest remote PR head, reviews, checks, conflicts and clean worktree.
**Status**: In Progress

Reviewed source52efef9d1b was published after conflict-free devcb70f0c5d1
rebase; all33 range-diff patches unchanged. Final stable-head global suite1076
passed/18.09 percent; native AuthNZ421passed/45.22 percent; focused lifecycle84
passed; ratchets43passed; production/test lifecycle Bandit clean. All actionable
independent findings addressed. Required CI awaits queued trusted license audit
and requester-owned Change summary remains pending. No merge or auto-merge.

## November 8 Follow-Up

Requester explicitly waived the Change summary for this PR. This is a recorded
requester exception, not an AI-authored human summary or merge authorization.
Current CI has completed: backend OpenAPI drift and full-suite shard failures
remain. Recheck all comments, group failures by root cause, fix with regression
evidence, independently review, and publish once verified. Keep production
shutdown admission and actual native ownership intact; test lifecycle isolation
must use trusted preparation rather than resetting counters or dropping owners.

Final local evidence is recorded in
Docs/superpowers/reviews/moderation-backend/2026-11-08-pr3437-ci-follow-up.md.
The six uv504 setup failures have no established application cause and the exact
download endpoint now responds200; publication naturally retries CI. Native
Chat1737, Audio149, global1082/18.09 percent, MCP849, VN73 including PostgreSQL,
and minimal DB cleanup5 plus API-key3 pass. Whole-project collection88874 pass,
not execution. Final independent review accepts every corrected scope with no
actionable findings. Publication and current-head CI remain before closing the
work item. No merge or auto-merge.

Dev subsequently advanced to476527719a. Rebase preserved all35 patches without
conflicts; validated integration source1e2a4e801f. Post-rebase manifest271passed,
canonical OpenAPI/path check passes,77 PR Python paths compile. Independent
integration review accepted with no actionable findings. Final evidence update
and exact-lease publication follow; required CI remains pending.

## November 13 Follow-Up

The requester now authorizes merging once current-head CI is green and review
is clear. Rebase onto dev, verify the browser and Notes/Persona changes already
landed there, repair the Chat audit override leak and preserve the authenticated
Rules audit introduced by dev. Refresh the canonical API fingerprint and run
the moderation regression suites and static checks before publishing. Stage 3
remains in progress until GitHub has evaluated the new head.

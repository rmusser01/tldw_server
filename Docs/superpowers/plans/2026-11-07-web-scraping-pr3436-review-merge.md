# PR 3436 Review And Merge

**Task:** TASK-15466. User requests rebase onto latest `dev`, address all issues
and comments, use `superpowers:requesting-code-review` instead of Qodo, then merge.
The requester explicitly answered "waived" to the human Change-summary gate;
record that exception for this PR only, without inventing human-authored prose.
ADR assessment: existing ADR-096 governs; no new architecture decision is planned.
Preserve all functionality, analyzer behavior and existing architecture.

## Stage 1: Rebase
**Goal:** Preserve the reviewed implementation on latest fetched dev `cb70f0c5`.
**Success Criteria:** Reversible backup, no lost upstream or user work, clean rebase.
**Tests:** Range-diff and changed-path review; current task/ADR namespace remains unique.
**Status:** In Progress.

## Stage 2: Address Review And CI
**Goal:** Triage all discussion, reviews and failed CI; run an independent reviewer.
**Success Criteria:** No unresolved substantive findings; no Qodo request or bypass.
**Tests:** Regression-first fixes, focused CI reproductions and scoped fix review.
**Status:** Not Started.

## Stage 3: Verify And Publish
**Goal:** Verify the final source, record limits and update PR3436 safely.
**Success Criteria:** Relevant regressions, native/security/consumer gates and normal
hooks pass; existing unrelated failures are proven and reported, not suppressed.
**Tests:** Appropriate pytest, Ruff/compile/format, Bandit, inventory/task/ID checks.
**Status:** Not Started.

## Stage 4: Merge And Verify
**Goal:** Merge only the reviewed final head once repository gates permit.
**Success Criteria:** Actual GitHub merge receipt and resulting dev commit verified;
PR-specific summary waiver recorded; no unapproved check bypass or unrelated cleanup.
**Tests:** Latest base/head/check/review audit; verify merged state and merge commit.
**Status:** Not Started.

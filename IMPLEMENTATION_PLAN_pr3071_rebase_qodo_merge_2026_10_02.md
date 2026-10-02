# PR3071 Latest Dev Rebase, Qodo Review, and Merge

Task: TASK-13421.1. PR: https://github.com/rmusser01/tldw_server/pull/3071.

Goal: Rebase the existing PR, resolve verified review/CI findings, and merge only
the qualified final head. Preserve the requester-authored Change summary.

ADR check: No new ADR required; integration and corrective review work do not
change a durable architecture rule. ADR-002, ADR-004, and ADR-006 govern tracking,
human ownership, and security verification.

## Stage 1: Preserve and Rebase
**Goal**: Rebase on fetched dev38b09af8e92b3d9a2a00aced22d6b992dbae9435,
including its metadata-only advancement from the initial df17 baseline.
**Success Criteria**: Backup ref retained; complete result equals expected clean
integration tree cbf4388b66855d5d255b784b87ae3dae28c04061, except this task/plan.
**Tests**: Git ancestry/tree comparison; original stash inventory unchanged.
**Status**: Complete

## Stage 2: Verify and Publish
**Goal**: Verify the rebased implementation and resolve published-ADR drift.
**Success Criteria**: Docs refresh and affected owning regressions pass; normal
hooks pass; publish using a lease bound to original head 0c9d592231dfee52642ce970f9f8cf159ebdfd49.
**Tests**: Docs refresh suite, changed CI/test-isolation regressions, owning Chat
Workspace regressions, touched-scope Bandit, and exact production-source binding
to prior real Chrome desktop/mobile acceptance. Unit doubles are not UAT.
**Status**: In Progress

Verified: docs33; combined owning273; incoming fixtures52; Chat180passed/24skipped;
CI helper/workflow56 and formatted helper5. Independent review's child-PATH
lookup defect is fixed with a red/green regression; no runtime changes. Dev
advanced to38b09af8e92b3d9a2a00aced22d6b992dbae9435 with only relay task metadata;
include that final delta before publishing. Retain initial path-guard test
invocation failures and raw Bandit test-assert diagnostics in the evidence.
The final rebase completes cleanly and matches expected treea082f5f9. Every
applicable configured pre-commit check passes; publication remains pending.

## Stage 3: Qodo and CI
**Goal**: Mark ready, evaluate every PR finding, and qualify the final head.
**Success Criteria**: Each actionable finding fixed with regression coverage or
answered with verified evidence; no unresolved blocking finding; exact-head CI
successful. Do not interpret missing Qodo review as approval.
**Tests**: Per-fix red/green tests, security checks, final GitHub checks/threads.
**Status**: Not Started

## Stage 4: Merge and Read Back
**Goal**: Merge the verified PR into dev without bypassing checks.
**Success Criteria**: Human summary byte-identical, latest head/base rechecked,
GitHub reports MERGED and its merge commit is verified on dev; task finalized.
**Tests**: Fresh PR/review/check readback and GitHub merge outcome. Preserve live
services, profiles, tabs, drafts, databases, unrelated files, and backup refs.
**Status**: Not Started

# PR 908 Merge Conflict Resolution Design

## Context

PR 908 is stacked on `feat/production-readiness-gaps`, and GitHub now reports it as `CONFLICTING`. The local worktree is clean, which means the conflict is not an in-progress local merge. It comes from divergence between:

- the PR 908 boundary-redesign commits on `codex/pr898-boundary-redesign`
- the later PR 898 fixes that landed on `feat/production-readiness-gaps`

The shared merge base is `1b35062ee4a2fef3c4a27e04e1aa0e418a41f96a`.

## Goals

- Make PR 908 mergeable again against the current head of `feat/production-readiness-gaps`.
- Preserve the Jobs and metering boundary redesign as the authoritative version in overlapping files.
- Keep the non-overlapping PR 898 fixes from the base branch intact.
- Re-run the scoped Jobs and metering verification after conflict resolution.

## Non-Goals

- Reopening the design of PR 908 or pulling new scope into it.
- Reworking unrelated files that do not participate in the branch divergence.
- Re-litigating the behavioral choices already accepted in PR 908.

## Root Cause

`codex/pr898-boundary-redesign` was built from `1b35062ee...`, while `feat/production-readiness-gaps` advanced to `8f7a96469` with additional edits in several of the same files:

- `tldw_Server_API/app/core/Jobs/manager.py`
- `tldw_Server_API/app/services/stripe_metering_service.py`
- `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`
- `tldw_Server_API/tests/test_stripe_metering.py`

Because both branches changed those files after the shared base, GitHub cannot auto-merge PR 908.

## Approved Resolution Strategy

### 1. Rebase PR 908 onto the current base branch

Replay `codex/pr898-boundary-redesign` on top of `feat/production-readiness-gaps` rather than merging the base branch in. That keeps the stacked PR history clean and mirrors the conflict set GitHub is reporting.

### 2. Treat PR 908 as authoritative in overlapping files

For files touched by both branches, keep the PR 908 side when behavior differs. This matches the user’s chosen policy that the architecture/redesign branch should win in overlaps. During rebase conflict resolution, that means selecting the rebased commit side for the overlapping files and only reintroducing base-branch content if it is strictly non-conflicting and still needed.

### 3. Keep base-only files untouched

Files changed only on `feat/production-readiness-gaps` stay as provided by the new base. PR 908 should only add its repository-boundary and associated test/doc changes on top.

### 4. Verify the scope that actually overlaps

After the rebase, rerun the focused Jobs and metering suite:

- `tldw_Server_API/tests/Billing/test_authnz_metering_repository.py`
- `tldw_Server_API/tests/Jobs/test_fair_share_integration.py`
- `tldw_Server_API/tests/Jobs/test_jobs_repository.py`
- `tldw_Server_API/tests/test_stripe_metering.py`

Then confirm the branch is clean and GitHub reports PR 908 as mergeable.

## Risks

- Using the PR 908 side blindly could drop a base-branch fix if one of the redesign commits predates it and did not already absorb it.
- Rebase conflict labels (`ours` vs `theirs`) are easy to misuse; for this workflow, the replayed PR 908 commit side is the one that should win in overlapping files.
- The branch must be force-pushed after rebase, so verification needs to happen before publishing.

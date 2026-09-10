# PR2939 rebase, review, and merge

Tracked in TASK-13241. Preserve the original issue repairs and live UAT record.

## Stage 1: Rebase on current dev
**Goal**: Reapply the PR to the latest fetched development branch.
**Success Criteria**: Clean rebase, preserved patch intent, recorded old/new base and head.
**Tests**: Inspect conflicts and range-diff; run affected regressions after review repairs.
**Status**: In Progress

## Stage 2: Address review and CI findings
**Goal**: Verify and resolve each Qodo comment and failed required check.
**Success Criteria**: Queued macros preserve request-time model defaults; configuration errors are diagnosable; test contracts and documentation are clear; every architecture comment has a supported disposition.
**Tests**: Red-green macro/default and logging regressions, public array-query outcomes, session migration regressions, affected CI guards, lint and Bandit.
**Status**: Not Started

## Stage 3: Verify, respond, and merge
**Goal**: Publish reviewed fixes, receive review on the latest changes, and integrate into dev.
**Success Criteria**: Fresh verification and required checks pass, all review findings are addressed, and the repository's human-written Change summary merge requirement is satisfied before merging.
**Tests**: Final review, current-head CI and remote/base checks, merge result verification.
**Status**: Not Started

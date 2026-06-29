## Stage 1: Rebase And Inventory
**Goal**: Rebase PR #2512 onto latest `origin/dev` and confirm the active review-thread scope.
**Success Criteria**: Branch rebases cleanly, review comments are mapped to touched files, and Backlog task `TASK-12056` tracks the work.
**Tests**: Git rebase status and `gh` review-thread inventory.
**Status**: Complete

## Stage 2: Review Fixes
**Goal**: Address reviewer-reported correctness, async-safety, metrics, typing, docstring, and test hygiene issues.
**Success Criteria**: Behavior changes have focused tests and endpoint/core changes follow existing embeddings patterns.
**Tests**: Targeted embeddings unit/endpoint tests for fallback behavior, provider resolution, cache metrics, BYOK endpoint handling, and adapter execution safety.
**Status**: Complete

## Stage 3: Verification
**Goal**: Validate the touched scope before committing.
**Success Criteria**: Targeted pytest runs, compile checks, Bandit on touched production code, shard coverage check, and diff hygiene pass or any skip is documented.
**Tests**: `pytest`, `compileall`, `bandit`, `check_shard_coverage.py`, and `git diff --check`.
**Status**: Complete

## Stage 4: PR Update
**Goal**: Commit, force-push with lease, and close the GitHub review loop.
**Success Criteria**: PR branch is updated, review threads have concrete replies, unresolved threads are resolved, and PR checks are inspected.
**Tests**: `gh pr checks` / review-thread GraphQL verification.
**Status**: Complete

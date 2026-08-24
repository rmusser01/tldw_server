# PR 2764 Final Rebase, Qodo Review, and Merge Plan

## Stage 1: Refresh and rebase
**Goal**: Rebase the PR branch onto the latest `origin/dev` in the existing isolated worktree.
**Success Criteria**: The worktree is clean before the operation, the rebase completes without unresolved conflicts, and the branch is force-pushed with lease.
**Tests**: Confirm `origin/dev...HEAD` reports zero commits behind and PR metadata points to the rebased head.
**Status**: Complete

## Stage 2: Qodo review validation
**Goal**: Wait for Qodo to review the rebased head and collect every posted finding and inline comment.
**Success Criteria**: Each item is mapped to current code and classified as valid, invalid, duplicate, or already addressed with supporting evidence.
**Tests**: Inspect current review submissions, issue comments, and unresolved review threads for the final head SHA.
**Status**: In Progress

## Stage 3: Remediation and verification
**Goal**: Address every validated issue without broadening the MCP execution-hardening scope.
**Success Criteria**: Regression tests cover behavioral fixes, focused MCP and package gates pass, and Bandit reports no findings in touched production code.
**Tests**: Run affected tests after each fix, then the focused MCP matrix, standalone RC gate, compile/whitespace checks, Ruff delta review, and Bandit.
**Status**: Not Started

## Stage 4: Review closure and merge
**Goal**: Resolve addressed threads and merge PR #2764 into `dev` after required gates are clean.
**Success Criteria**: No unresolved actionable comments remain, required checks are successful or validly skipped, the PR is mergeable, and GitHub reports it merged.
**Tests**: Read back review-thread state, required check rollup, merge status, and merge commit SHA.
**Status**: Not Started

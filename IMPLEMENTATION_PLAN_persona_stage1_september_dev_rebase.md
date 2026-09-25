# Persona Stage 1 September dev rebase

## Stage 1: Review the current PR and integration surface
**Goal**: Identify the exact PR commit range, current `origin/dev`, Qodo findings, and overlapping Persona files.
**Success Criteria**: The original branch tip is recorded, every Qodo finding is classified, and a recovery ref exists before rewriting history.
**Tests**: `git status --porcelain=v1`; `git merge-base HEAD origin/dev`; inspect PR review and comments.
**Status**: Complete

## Stage 2: Rebase the scoped Persona series
**Goal**: Replay the Persona Stage 1 net change on current `origin/dev`, preserving newer Persona behavior. The original commit-by-commit replay was aborted when its historic migration conflicted with the current schema; a recovery branch retains the original series.
**Success Criteria**: The integration branch has no unrelated commits or unresolved conflicts; current migration numbering and APIs remain coherent.
**Tests**: `git diff --check`; compare changed-file list with the original PR; migration tests.
**Status**: Complete

## Stage 3: Repair and verify integration regressions
**Goal**: Resolve confirmed behavior or test failures caused by the rebase using failing tests before code fixes.
**Success Criteria**: Affected backend and frontend tests, type/lint checks, and Bandit pass or have documented, scoped failures.
**Tests**: Persona API/DB and migration pytest suites; Persona Buddy frontend tests; frontend type/lint; Bandit on touched Python files.
**Status**: Complete

## Stage 4: Update PR and review closure
**Goal**: Push the rebased branch safely and inspect fresh GitHub/Qodo status.
**Success Criteria**: Remote head matches local head, PR is no longer conflicted, and every actionable Qodo comment is resolved or answered in-thread.
**Tests**: GitHub PR metadata, review comments, and checks after push.
**Status**: In Progress

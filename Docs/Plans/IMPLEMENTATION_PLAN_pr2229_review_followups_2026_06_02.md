## Stage 1: Rebase And Review Triage
**Goal**: Rebase PR 2229 onto latest `origin/dev` and verify each unresolved review thread against current code.
**Success Criteria**: Branch rebased without conflicts; actionable comments are grouped by affected behavior.
**Tests**: `git rebase origin/dev`; GitHub review-thread inspection.
**Status**: Complete

## Stage 2: Regression Coverage
**Goal**: Add focused tests for OpenUI fallback source fidelity, sensitive key detection, action forwarding, request override merging, and metadata persistence.
**Success Criteria**: Tests fail against current buggy behavior before production fixes.
**Tests**: Focused Vitest files under `apps/packages/ui/src`.
**Status**: Complete

## Stage 3: Review Remediation
**Goal**: Patch the validated review issues with minimal changes matching existing UI/chat patterns.
**Success Criteria**: OpenUI actions are defensive, request overrides merge consistently, metadata survives persistence paths, and documentation/package nits are fixed.
**Tests**: Focused Vitest files pass after implementation.
**Status**: Complete

## Stage 4: Verification And Push
**Goal**: Run relevant TypeScript/test/security gates, update Backlog task, commit, and push the rebased branch.
**Success Criteria**: Verification results are recorded; PR branch is force-pushed with lease.
**Tests**: `bunx vitest run ...`; `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`; Bandit scope decision recorded.
**Status**: Complete

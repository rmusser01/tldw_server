# PR 2714 Review Remediation Plan

Backlog task: TASK-12098
Pull request: https://github.com/rmusser01/tldw_server/pull/2714

## Stage 1: Review Triage and Regression Coverage
**Goal**: Verify every inline and summary-level finding against the current branch and encode valid behavioral defects as focused regression tests.
**Success Criteria**: Every reviewer comment is classified as fix, already resolved, or rejected with repository-specific evidence; new tests fail for each valid behavior defect.
**Tests**: Focused Vitest/Jest and pytest cases for archive scanning, restore certification, auth isolation, notification lifecycle, API credential invalidation, config parsing, and backend failure handling.
**Status**: Complete

## Stage 2: Full-Account UAT and Backend Corrections
**Goal**: Make browser UAT certify a clean, exact account round trip while keeping archive privacy checks bounded and backend work non-blocking.
**Success Criteria**: The UAT validates all exported account categories, media bytes, and embedding values against a clean destination; backend review defects are fixed without weakening the full-export contract.
**Tests**: Chatbooks UAT helper/fixture tests, Media DB helper tests, Chatbooks endpoint tests, AuthNZ repair tests, config tests, profile update tests, and frontend fixture tests.
**Status**: Complete

## Stage 3: WebUI and Extension Corrections
**Goal**: Resolve reviewed UX recovery, rendering, notification lifecycle, polling, SSE, timeout, and cross-origin credential defects.
**Success Criteria**: Recovery preserves source context, server errors are sanitized, scope changes never expose stale counts, polls cannot overlap, terminal states do not retry, and server changes clear credentials.
**Tests**: Component/hook/service/API/runtime bootstrap tests plus static E2E timeout and label assertions.
**Status**: Complete

## Stage 4: Repository Metadata and Quality Gates
**Goal**: Normalize affected Backlog records and complete changed-scope verification.
**Success Criteria**: Structured task markers are valid, pending human-summary work remains non-terminal, focused suites pass, lint/type checks pass, and Bandit reports no new findings.
**Tests**: Backlog validation, pytest, frontend and extension test suites, formatter/linter/type checks, compile checks, and Bandit on touched Python paths.
**Status**: Complete

## Stage 5: Rebase, Publish, and Review Closure
**Goal**: Rebase on the latest development branch if it advances, publish the fixes, and close every PR conversation with evidence.
**Success Criteria**: Branch is current with `origin/dev`, commits are pushed, each review thread has a specific resolution or technical rationale, and all available PR checks are green or a documented external blocker remains.
**Tests**: `git diff --check`, PR check inspection, unresolved-thread query, and final clean-worktree review excluding known unrelated files.
**Status**: In Progress

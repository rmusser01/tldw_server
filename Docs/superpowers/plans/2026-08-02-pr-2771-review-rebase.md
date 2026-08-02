# PR #2771 Review, Rebase, and Merge Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebase PR #2771 onto the latest `dev`, resolve every actionable review and PR-specific CI issue, verify the complete change, and merge once repository policy and required checks are satisfied.

**Architecture:** Preserve the existing prompt-improvement design and make only evidence-backed review corrections. Treat failures introduced by stale branch ancestry separately from feature regressions, use focused TDD for behavior changes, and retain the repository's existing endpoint, exception, test-marker, and extension-E2E conventions.

**Tech Stack:** Git/GitHub CLI, FastAPI/Pydantic, pytest, Next.js/React/Vitest/Playwright, Bun, Ruff, mypy, Bandit, Backlog.md.

## Stage 1: Rebase and Rebaseline
**Goal**: Rebase the clean PR branch onto the latest `origin/dev` and establish the post-rebase review/CI baseline.
**Success Criteria**: Rebase completes without losing PR changes or staging the two unrelated watchlist templates; changed-file and conflict audits are clean.
**Tests**: `git status --short`, `git diff --check`, merge-base/ancestry checks, current PR check inventory.
**Status**: Complete

## Stage 2: Review Findings with TDD
**Goal**: Adjudicate all seven Qodo findings against current code and fix every valid issue with minimal changes.
**Success Criteria**: The limiter uses true inactivity semantics; capabilities follow the project rate-limit convention; Python docstrings/types/exceptions meet policy; tests use accepted markers and deterministic assertions; extension E2E cannot silently skip required CI coverage.
**Tests**: Focused RED/GREEN tests for each behavioral change plus static checks for documentation, typing, markers, and skip policy.
**Status**: Complete

## Stage 3: Repair Post-Rebase CI Regressions
**Goal**: Reproduce and fix only failures that remain attributable to PR #2771 after rebasing.
**Success Criteria**: Shard coverage, wizard coverage, backend-required, frontend-required, and extension E2E root failures either pass or are proven unrelated/current-base failures with evidence.
**Tests**: Local equivalents of failing GitHub Actions steps and focused regression suites.
**Status**: In Progress

## Stage 4: Full Verification and Review Closure
**Goal**: Run proportional backend/frontend/security verification, commit, push with force-with-lease, and close review feedback with evidence.
**Success Criteria**: Scoped tests/build/lint/type checks/Bandit/diff checks pass; the inline thread is replied to and resolved; all top-level findings receive an evidence-backed disposition; fresh required checks pass.
**Tests**: Focused pytest/Vitest/Playwright suites, WebUI and extension builds, Ruff/ESLint/Prettier/mypy as applicable, Bandit on touched production Python, `git diff --check`, GitHub required checks.
**Status**: Not Started

## Stage 5: Policy Gate and Merge
**Goal**: Confirm the requester-authored Change summary and merge PR #2771 into `dev`.
**Success Criteria**: Human Change summary is present, PR is mergeable, required checks and review threads are clear, GitHub reports the PR merged, and `origin/dev` contains the merge commit.
**Tests**: PR body policy check, `gh pr checks`, review-thread query, post-merge PR state and ancestry verification.
**Status**: Not Started

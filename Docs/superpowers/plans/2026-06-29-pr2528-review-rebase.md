# PR 2528 Review Rebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #2528 onto latest `dev`, address active moderation policy compiler review comments, and push the verified branch.

**Architecture:** Keep fixes scoped to the moderation policy compiler design changes and adjacent tests/docs. Verify each review comment against current code before changing behavior, and prefer small regression tests for valid behavioral issues.

**Tech Stack:** Python, pytest, FastAPI project conventions, Backlog.md, GitHub CLI.

---

## Stage 1: Rebase And Inventory
**Goal**: Rebase PR #2528 onto latest `origin/dev` and identify all active review threads/comments.
**Success Criteria**: Branch rebases cleanly or conflicts are resolved, review comments are mapped to touched files, and Backlog task `TASK-12022` tracks the work.
**Tests**: Git rebase status and GitHub review-thread query.
**Status**: Complete

## Stage 2: Review Fixes
**Goal**: Verify each review comment against current code and implement only technically valid fixes.
**Success Criteria**: Each active comment has a code/test/doc change or a documented technical rationale.
**Tests**: Failing regression tests first where behavior changes are needed, then focused pytest runs.
**Status**: Complete

## Stage 3: Verification
**Goal**: Run focused tests and required quality gates for touched scope.
**Success Criteria**: Focused moderation compiler tests/docs checks, compile checks, Bandit for touched production files, and relevant CI guard checks pass or are reported with evidence.
**Tests**: Commands recorded in `TASK-12015`.
**Status**: Complete

## Stage 4: Push And Resolve
**Goal**: Commit, push the rebased branch, reply to/resolved addressed review threads, and report remaining CI status.
**Success Criteria**: PR branch is updated on GitHub, review threads are resolved or documented as non-actionable, and final status is reported.
**Tests**: `gh pr view`, `gh pr checks`, and review-thread query.
**Status**: Complete

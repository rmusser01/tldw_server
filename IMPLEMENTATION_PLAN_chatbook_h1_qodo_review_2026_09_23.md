# Chatbook H1 Qodo Review Implementation Plan

**Goal:** Resolve verified Qodo findings on PR #2968 without weakening history ownership, and keep stacked PR #3002 aligned for review and merge.

**Baseline:** Server `dev` `158287db30a28450e79fdbc38253168a468be88f`; H1 PR #2968 at `1f5a2016f3758014f88c20d6144d8fe3bcfb73c6` before review fixes; H2 PR #3002 stacked at `454635c33cf298ff10dde7e558231209c951e255`. Use only the isolated H1/H2 worktrees. Do not touch the main/UAT worktree or H2's untracked production evidence.

## Stage 1: Classify and reproduce Qodo findings
**Goal:** Read all inline comments, confirm behavior against source and existing patterns, and record a fix or reasoned rejection for each finding.
**Success Criteria:** Every H1 Qodo comment is mapped to a root cause and test path; H2 comments are collected when posted.
**Tests:** Focused existing selected-history, ordinary-chat, character-message, and route tests to establish baseline; new behavioral regressions must fail before fixes.
**Status:** Complete

## Stage 2: Repair confirmed behavior and security regressions
**Goal:** Fix idle-controller send/edit/delete, source-switch review state, server deletion display reconciliation, unresolved skill-directory admission, and any other confirmed high-impact H1 defects.
**Success Criteria:** Selected history keeps owner fences while ordinary chat paths remain usable; selected-history admission fails closed when skill visibility cannot be established.
**Tests:** Targeted Vitest for both chat hooks and review component, plus focused Python endpoint/security tests, each with red/green evidence.
**Status:** In Progress

## Stage 3: Resolve remaining review items and stacked H2
**Goal:** Address validated async database, typing, documentation, equality, fixture, and cross-repository-contract findings with the smallest safe changes; process H2 Qodo feedback against its internal-storage scope.
**Success Criteria:** Every Qodo thread is fixed or answered technically; no P1/P2 review issue remains. H2 descends the final H1 head and stays scoped to groundwork.
**Tests:** Affected Python/frontend suites, SQLite/PostgreSQL fixtures when needed, Ruff/format and Bandit for touched Python.
**Status:** Not Started

## Stage 4: Verify and integrate
**Goal:** Recheck latest `origin/dev`, required CI and PR review status, then merge H1 followed by H2 only when both are qualified.
**Success Criteria:** PR heads and ancestry are verified, required checks pass, human-written Change summaries remain intact, and GitHub confirms both merge commits.
**Tests:** Fresh focused suites on final heads, `git diff --check`, touched-source Bandit, and GitHub required-check results.
**Status:** Not Started

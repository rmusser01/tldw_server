# PR2979 latest-dev integration — 2026-09-28

Task: TASK-13260.278.18. Resume PR-first repairs; full UAT remains paused.
ADR required: no new ADR for integration. Preserve ADR020 storage and ADR049
Chat ownership; reassess if a resolution changes a durable architecture rule.

## Stage 1: Preserve and rebase
**Goal:** Retain b71c04cde8 and replay UAT work onto fetched dev38c1455d46.
**Success Criteria:** Recovery ref exists; conflicts retain both intended behaviors.
**Tests:** Clean worktree, ancestry, range-diff and overlap review.
**Status:** In Progress

## Stage 2: Verify overlapping behavior
**Goal:** Check changed Chat, AuthNZ, Notes, providers and CI boundaries.
**Success Criteria:** Focused controls pass, including required official PostgreSQL.
**Tests:** Existing relevant suites, frontend types/lint, touched Python Bandit.
**Status:** Not Started

## Stage 3: Publish and handle review
**Goal:** Update PR2979 and resolve current CI/Qodo findings before full UAT.
**Success Criteria:** Current-base reviewable head; honest tracker and PR gate status.
**Tests:** Hosted checks and exact PR head/base; generated-artifact exclusion.
**Status:** Not Started

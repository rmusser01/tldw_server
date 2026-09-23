# Chatbook H2 Qodo Review Implementation Plan

**Goal:** Resolve verified Qodo findings on PR #3002 while preserving its internal native-fork storage scope and its dependency on H1.

## Stage 1: Reproduce substantive findings
**Goal:** Confirm snapshot-bound assistant projection and caller-owned PostgreSQL transaction behavior against real storage paths.
**Success Criteria:** Failing focused tests demonstrate both reported defects before fixes.
**Tests:** Native fork projection and workspace lifecycle tests on the supported backends.
**Status:** Complete

## Stage 2: Repair native fork correctness and contracts
**Goal:** Fix snapshot binding and transaction ownership; apply appropriate typing, documentation, test classification, and domain-error improvements.
**Success Criteria:** The original chat stays unchanged, fork projection recognizes its saved assistant, and workspace deletion cannot commit unrelated writes.
**Tests:** Focused projection, migration, transaction, and workspace lifecycle suites.
**Status:** Complete

## Stage 3: Verify, review, and integrate
**Goal:** Answer every Qodo thread with evidence, run relevant lint/Bandit/tests, restack on the final H1 head, and merge only after required PR gates pass.
**Success Criteria:** No actionable P1/P2 issue remains and GitHub confirms final checks and merge ancestry.
**Tests:** `git diff --check`, Ruff, Bandit, targeted SQLite/PostgreSQL tests, required GitHub checks.
**Status:** In Progress

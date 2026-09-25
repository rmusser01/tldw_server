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

All new functions in the three H2 migration, transaction, and workspace-lifecycle
test modules now have parameter and return annotations. The focused suite passed
64 SQLite cases; 68 PostgreSQL parametrizations skipped because PostgreSQL was
unavailable locally. Ruff and `git diff --check` passed. Final PostgreSQL and
required CI evidence is still pending.

On 2026-09-24, H2 was restacked onto H1 `a7a89a80db`, which descends server
`dev` `91c32e3126`. The focused post-restack run passed 172 tests, with 68
PostgreSQL-dependent parametrizations skipped locally; `git diff --check`
passed. The current PR head is `0f308f49b3` before this documentation update.
H2 was subsequently restacked onto H1's workflow-only shard fix
`aabca0478f`; `range-diff` shows all five H2 commits unchanged. Required
GitHub checks and final H1 merge ancestry remain pending.
After server `dev` advanced to `0db48866a5`, H2 was restacked onto H1
`f3bbacf367`. All six existing H2 commits are unchanged by range-diff,
and the stacked diff check passes. Required CI is pending on the new head.
Server `dev` then advanced to `a2f5e1b816` and H2 was restacked onto H1
`fa3a36997c`. All seven prior H2 commits are unchanged by range-diff;
the stacked diff check passes. Required CI must rerun on the new head.
H1 later corrected an `unavailable` history-owner TypeScript narrowing error
and moved to `63e95039db`. H2 restacked cleanly; all eight existing H2
commits are unchanged by range-diff, and the stacked diff check passes.

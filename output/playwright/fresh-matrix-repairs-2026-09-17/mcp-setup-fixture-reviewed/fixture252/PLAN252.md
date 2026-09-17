# PLAN252 — TASK13260.194 / UAT252

## Stage 1: Confirm fixture cause
**Goal:** Reproduce the original missing-schema test failing before its assertion.
**Success Criteria:** Managed DROP TABLE is rejected by the unchanged profile write guard.
**Tests:** Original test on current frozen production, approved required-PG runner.
**Status:** Complete

## Stage 2: Correct only test-owned schema setup
**Goal:** Build the malformed disposable SQLite fixture through a dedicated schema connection.
**Success Criteria:** Preserve the exact original missing-table assertion and managed repository under test; close setup connection explicitly. No production changes.
**Tests:** Focused fixture GREEN; existing repository/setup suite and251 actual PG/SQLite controls; existing guard negative cases.
**Status:** Complete

## Stage 3: Freeze evidence
**Goal:** Retain attributable patch and hash-bound original/final results.
**Success Criteria:** Ruff/Bandit comparison, compile/diff, unchanged assertion AST, source hashes and safe evidence manifest.
**Tests:** Scoped static checks and final source verification.
**Status:** Complete

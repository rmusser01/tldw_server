# UAT296 — Literal exemplar search and transaction recovery

Task: TASK13260.233. Treat source/user prose as text and isolate failures in optional exemplar retrieval.

## Stage 1: causal regression
**Goal**: Reproduce punctuation failures and the poisoned PostgreSQL transaction.
**Success Criteria**: Actual SQLite/PostgreSQL controls expose the failure while preserving search ownership and pending caller work.
**Tests**: Punctuation, empty/stopword text, matching/nonmatching exemplars, failed read followed by a successful read, outer rollback.
**Status**: Complete

Evidence: final causal run has five PostgreSQL failures and thirteen passing controls. Initial SQLite autocommit fixture mistake was corrected before the final causal run.

## Stage 2: bounded repair
**Goal**: Use PostgreSQL plain-text parsing and a transaction/savepoint around exemplar lookup.
**Success Criteria**: Focused cross-backend and selector controls pass; no new lint/security findings; independent review clear.
**Tests**: Official PostgreSQL fixture plus SQLite repository, ownership, selector and transaction controls; Ruff/Bandit.
**Status**: Complete

Evidence: 118 focused tests pass; the sole skipped opt-in performance test passes when rerun with PERF=1. Four additional managed-operation transaction controls pass on SQLite/actual PostgreSQL. No PostgreSQL skips. Bandit zero findings; Ruff retains only two unchanged baseline TRY203 findings outside the repaired method. Independent review found no production issue and requested the now-passing managed-transaction controls.

## Stage 3: targeted acceptance
**Goal**: Repeat the source-prose exemplar path on committed code with real PostgreSQL.
**Success Criteria**: Ordinary punctuation no longer causes query/aborted-transaction errors; evidence and limits recorded.
**Tests**: Native Media/Chat flow and backend/cluster audit with immutable source and isolated fixtures.
**Status**: In Progress

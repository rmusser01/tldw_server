# Scoped PostgreSQL Email Benchmark Plan

**Goal:** Reuse the synthetic email-search benchmark against isolated PostgreSQL with real per-user RLS and record bounded scale evidence.

**Scope:** TASK-13369. Keep Gmail/worker/model egress disabled; use disposable local PostgreSQL databases and synthetic data only. Do not claim the 1M target from a smaller run.

## Stage 1: Define backend and scope contract
**Goal:** Specify how the existing benchmark selects SQLite or PostgreSQL and establishes numeric user scope without storing credentials.
**Success Criteria:** CLI rejects PostgreSQL without scope or with backend mismatch; reports the actual backend.
**Tests:** Focused CLI/unit regressions written red first.
**Status:** Complete

## Stage 2: Run staged synthetic PostgreSQL workload
**Goal:** Populate an isolated non-superuser PostgreSQL database in bounded steps and run the existing query mix with forced RLS.
**Success Criteria:** Fixture count and positive query matches verified, representative latency captured, no personal mail/Gmail/model calls.
**Tests:** 1k smoke, then 10k or the largest safe bounded fixture; compare exact query mix to SQLite report.
**Status:** Complete

## Stage 3: Record evidence and commit
**Goal:** Publish backend-specific JSON/report with limitations, update Backlog and checklist, review and commit.
**Success Criteria:** Dataset, hardware, backend, timing, skips and remaining gates are explicit; clean worktree after commit.
**Tests:** Focused pytest, Ruff, Bandit, git diff --check, final report inspection.
**Status:** In Progress

**Blocker:** Successful 10k reports and RLS checks are preserved, but final
disposable PostgreSQL database/role cleanup is unconfirmed after recurring host
disk exhaustion and Docker I/O failure. The private manifest and cleanup script
remain in `/tmp`; restore stable storage before another large fixture.

# Prototype Workspace Risk Gate 2 Persistence Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for implementation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden prototype workspace persistence so multi-step backend operations, cleanup rules, and query/index assumptions are explicit, tested, and documented for Risk Gate 2.

**Architecture:** Keep the existing AuthNZ repository/service split. Add transaction-bound repository execution for multi-step prototype operations instead of introducing a new DB_Management subsystem. Keep runtime job durability out of scope for Risk Gate 3.

**Tech Stack:** FastAPI backend, AuthNZ `DatabasePool`, SQLite migrations, async repository/service tests with pytest.

---

## Stage 1: Transaction-Bound Repository Execution

**Goal:** Make prototype repository methods usable inside one AuthNZ transaction for both SQLite and PostgreSQL-backed pools.

**Success Criteria:** Service code can create a transaction-bound `PrototypeWorkspacesRepo` and run existing repo methods without per-statement commits. Placeholder conversion remains compatible with the current `DatabasePool` behavior.

**Tests:** Add repo unit tests proving transaction-bound execution does not call top-level pool `execute`/`fetchone`/`fetchall`, and that PostgreSQL-style transaction adapters receive converted placeholders.

**Status:** Complete

- [x] Add failing tests in `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py`.
- [x] Add a narrow transaction helper/adapter in `tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py`.
- [x] Run the new focused repo tests and keep the existing prototype repo tests green.

## Stage 2: Multi-Step Service Compensation And Atomicity

**Goal:** Remove partial-write windows from workspace creation and session snapshot saving, while preserving existing compensation behavior where an external preview broker operation is involved.

**Success Criteria:** Failed seed snapshot creation rolls back the workspace row. Failed session-state persistence rolls back the candidate snapshot row. Promotion failure after preview grant still revokes the preview and restores workspace state.

**Tests:** Add service tests in `tldw_Server_API/tests/PrototypeWorkspaces/test_promotion_service.py` or a new focused persistence test file for failed intermediate writes.

**Status:** Complete

- [x] Write failing tests for create-workspace seed failure and session snapshot state failure.
- [x] Update `PrototypeWorkspaceService.create_workspace` and `save_session_snapshot` to use the transaction-bound repo.
- [x] Add or extend promotion compensation tests around persisted preview handle rollback.

## Stage 3: Cleanup And Retention Rules

**Goal:** Define and implement backend cleanup semantics for expired/revoked prototype collaboration state without changing runtime job durability.

**Success Criteria:** Cleanup behavior is explicit for archived workspaces, revoked actors, expired sessions, stale pending promotion requests, and inactive preview handles.

**Tests:** Add repository tests for cleanup cutoffs and non-destructive behavior for active records.

**Status:** Complete

- [x] Add failing cleanup/retention tests against the in-memory SQLite fixture.
- [x] Add repository cleanup helpers that only touch prototype tables and return count summaries.
- [x] Document the retention rules in `Docs/API-related/Prototype_Workspaces_API.md`.

## Stage 4: Migration, Index, And Query-Plan Evidence

**Goal:** Record and test the index/query-plan coverage needed by workspace detail, session lookup, active actor lookup, promotion listing, and preview-handle lookup paths.

**Success Criteria:** Migration tests assert required prototype indexes exist. Query-plan evidence is recorded in docs. Any missing index is added to migration 086 with idempotent `CREATE INDEX IF NOT EXISTS` statements.

**Tests:** Add SQLite migration/index tests and fake PostgreSQL/table-discovery compatibility tests where practical.

**Status:** Complete

- [x] Add failing migration tests for the required prototype indexes.
- [x] Add missing idempotent indexes in `tldw_Server_API/app/core/AuthNZ/migrations.py`.
- [x] Document SQLite/PostgreSQL behavior, placeholder conversion, and query-plan review in `Docs/API-related/Prototype_Workspaces_API.md`.

## Stage 5: Verification And PR Closeout

**Goal:** Finish Risk Gate 2 with focused automated evidence and the security gate required by the repo.

**Success Criteria:** Focused tests pass, Bandit has been run on touched backend files, TASK-363 has verification notes and a final summary, and GitHub issue #1454 can be linked from the PR.

**Tests:** `python -m pytest tldw_Server_API/tests/PrototypeWorkspaces -q`; focused migration/repo tests as added; Bandit on touched backend files.

**Status:** Complete

- [x] Run the focused prototype workspace suite.
- [x] Run Bandit on touched backend paths.
- [x] Update TASK-363 acceptance criteria and final summary.
- [x] Prepare a PR summary that explains what changed and why, leaving the required human `Change summary` for merge readiness.

# DB Parity & Structural Debt Implementation Plan (Credit Batch 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the SQLite/Postgres schema parity gap in ChaChaNotes, implement the no-op `migrate_schema()` stubs, resolve the Evaluations Postgres adapter TODO, and settle the ChromaDB "Point 2 / Point 6" TODOs.

**Architecture:** Audit → decide (port vs descope) → execute per store, with parity tests mirroring the repo's existing migration-test pattern. Each backend store is an independent slice.

**Tech Stack:** SQLite + PostgreSQL (psycopg/asyncpg per existing patterns), pytest (Postgres fixture via `tldw_Server_API/tests/AuthNZ/conftest.py` isolated environment), ChromaDB.

**Spec:** This plan; verified code state 2026-10-01. Stage tasks: TASK-13403 (Stages 1-2, ChaChaNotes parity), TASK-13404 (Stage 3, Scheduler stubs), TASK-13405 (Stage 4, Evaluations adapter), TASK-13406 (Stage 5, ChromaDB TODOs).

## Global Constraints

- `source .venv/bin/activate` first; Postgres tests use the AuthNZ conftest fixture (Dockerized Postgres unless `TLDW_TEST_NO_DOCKER=1`) — never roll your own DB setup.
- Rule from AGENTS.md: all DB operations through `/app/core/DB_Management/` abstractions, no raw SQL outside them.
- `ChaChaNotes_DB.py` is ~45k lines — never bulk-refactor it in these PRs; change only the migration ladder and version constants. (Monolith decomposition is explicitly out of scope.)
- Implement-or-descope decisions must be recorded in the task file with rationale — no silent deferral.
- Bandit per stage on touched paths.

---

## Stage 1: ChaChaNotes parity audit (SQLite v68 vs Postgres v72)

**Goal:** Produce the authoritative divergence inventory before touching code.
**Success Criteria:** `Docs/Design/2026-10-XX-chacha-schema-parity-audit.md` lists every migration step between SQLite `_CURRENT_SCHEMA_VERSION = 68` and `_POSTGRES_SCHEMA_VERSION = 72` (`app/core/DB_Management/ChaChaNotes_DB.py:755-756`), classifies each as PORT / DESCOPE / ALREADY-EQUIVALENT, and notes the Postgres-only path gated `>= 71` at `:24902`.
**Tests:** N/A (audit); output feeds Stage 2 tests.
**Status:** Not Started

Backlog task: TASK-13403.

- [ ] Diff both migration ladders in `ChaChaNotes_DB.py` (SQLite ladder vs Postgres ladder, versions 63-72).
- [ ] For each PG-only migration: identify the feature it enables, its endpoint/feature-flag consumers (`grep -rn` the new columns/tables), and whether SQLite users currently hit silent feature loss or errors.
- [ ] Write the audit doc with the PORT/DESCOPE table and a recommendation.

## Stage 2: Execute parity decision

**Goal:** Bring SQLite to the decided state with parity tests.
**Success Criteria:** If PORT: SQLite ladder reaches the same version semantics; a parity test mirrors the existing `test_chacha_postgres_migration_*` pattern (`grep -rln "chacha_postgres_migration" tldw_Server_API/tests`) for each ported step, and both backends pass. If DESCOPE: version constants annotated, divergence documented in the audit doc and `Docs/Database_Migrations.md`, and PG-only paths fail fast on SQLite with a clear error instead of silently missing features.
**Tests:** Per-migration parity tests on both backends (Postgres via AuthNZ fixture).
**Status:** Not Started

- [ ] Write failing parity tests for the first ported migration; run to confirm FAIL on SQLite.
- [ ] Implement the SQLite migration step; PASS; repeat per step.
- [ ] Update `Docs/Database_Migrations.md`; commit `fix(db): port ChaChaNotes schema vNN to SQLite (TASK-<id>)`.

## Stage 3: Scheduler `migrate_schema()` stubs

**Goal:** Replace the two `pass`-only `migrate_schema()` implementations and the placeholder list method.
**Success Criteria:** `app/core/Scheduler/backends/postgresql_backend.py:236` and `sqlite_backend.py:961` either perform real migrations or raise an explicit `NotImplementedError` with a tracking-task reference; `sqlite_backend.py:686` (returns `[]  # TODO: Implement if needed`) resolved the same way; each path has a test.
**Tests:** Unit tests per backend: successful no-op migration when current, migration when stale, explicit error if unimplemented.
**Status:** Not Started

- [ ] Determine what schema drift actually occurs in Scheduler tables (read both backends' DDL paths).
- [ ] TDD per backend; Bandit; commit.

## Stage 4: Evaluations PostgreSQL adapter

**Goal:** Resolve `app/core/Evaluations/db_adapter.py:358` ("TODO: Implement PostgreSQL connection using psycopg2 or asyncpg").
**Success Criteria:** Either a working Postgres adapter following existing DB_Management patterns (with tests via the Postgres fixture), or fail-fast descope: the TODO replaced by an explicit unsupported-backend error plus a task recording the decision.
**Tests:** Adapter round-trip tests on Postgres; clean error test on unsupported path.
**Status:** Not Started

- [ ] Check how the AuthNZ Postgres support connects (`app/core/AuthNZ/`) and reuse its pattern/tooling.
- [ ] TDD; Bandit; commit.

## Stage 5: ChromaDB "Point 2 / Point 6" TODOs

**Goal:** Settle the two long-standing embedding-store TODOs: per-model collection management (`app/core/Embeddings/ChromaDB_Library.py:552` "FIXME - Implement this") and chunk-reference sync back to Media DB/FTS (`:816`, `:1242` "TODO: Point 6").
**Success Criteria:** Each TODO is either implemented with tests or removed with a decision note in the task; no `FIXME/TODO: Point N` markers remain without a tracking task ID.
**Tests:** Collection-creation test per embedding model config; chunk-ref round-trip test (insert → SQL/FTS rows exist) if implemented.
**Status:** Not Started

- [ ] Investigate current behavior: what happens today when a second embedding model is configured (silent sharing? error?) and whether FTS searches miss vector-only chunks.
- [ ] Write the failing test that documents today's broken/absent behavior first; then implement or descope per evidence.
- [ ] Bandit; commit.

# PR3084 Sync SQLite bootstrap transaction repair — 2026-10-04

Associated task: TASK-13260.278.18.83.45. Source baseline: published d667bb748dae7dc13918cb16cd8b68c0179f63b2. Separate branch excludes unpublished closure and exhausted ingestion tracking. API replacement approval remains pending. All native/UAT/installer gates and STOP limits remain open.

ADR required: no. Restore the existing transaction and serialization contract without changing APIs, schemas, backend policy or PostgreSQL locking. ADR-059 governs task editing.

## Stage 1: Establish the transaction failure
**Goal**: Trace the new hosted concurrent initializer error and verify the candidate implicit-commit boundary.
**Success Criteria**: Record one unchanged original target result, then observe a deterministic regression fail on the unchanged source because bootstrap DDL survives a later exception.
**Tests**: Original concurrent SQLite initializer case; new real-backend failed-bootstrap rollback case. Each operation600s outer/300s case; max3 diagnostic/fix attempts, no unchanged retries.
**Status**: Complete

## Stage 2: Retain the SQLite bootstrap write transaction
**Goal**: Preserve the write transaction across all seven fixed Sync schema callsites through one private helper; PostgreSQL continues delegating to its backend.
**Success Criteria**: New rollback regression and the complete original store file pass with all original assertions intact; SQLite holds its current transaction throughout base bootstrap. PostgreSQL branch, shared backend and schemas unchanged. Compile, touched Ruff and baseline/current Bandit assessed.
**Tests**: Full test_sync_v2_store.py including applicable official isolated PostgreSQL fixtures. No stopped ingestion/native/main reload controls or full project replay. Existing proper Python3.12 environment only.
**Status**: Complete

## Stage 3: Review and publish the qualified repair
**Goal**: Independently review the actual source/tests/tracking and publish normally only after exact source/base/remote guards.
**Success Criteria**: Independent review clear; normal reviewed-patch commit and exact-lease publication; authored body and human summary preserved/read back. Fresh final-head hosted gates remain required and no merge acceptance claimed. Remove only this owned plan after all unit stages complete.
**Tests**: Actual patch review, diff check, official owned task format check, immediate dev/remote freshness. No proof artifacts, tracking-only publication or direct-dev push.
**Status**: In Progress

Stage1 results: one unchanged concurrent initializer target passed (1passed/30warnings/2.76s pytest/5.715s outer/natural0), without reproducing the hosted schema-change exception. The new real-backend bootstrap rollback regression failed as expected on unchanged production:28 tables survived the injected later exception (1failed/30warnings/2.65s/5.417s outer/natural1). This proves implicit commit/lock loss, not a complete reproduction of the hosted interleaving. Fixed schema inspected:78 table/index statements, no triggers, embedded transaction statements or comments. No retries or stopped-domain actions.

First partial attempt (superseded) results: full original store file plus the new regression passed204/30warnings/10.65s pytest/13.560s outer/natural0/no skips reported, with applicable official isolatedPG fixtures. Only ensure_schema changes in production; all schema constants, PostgreSQL/shared backend code and original test functions/assertions identical by AST. Two files compile. Ruff old/current lint0; both inherited formatter flags retained, only the new test formatted without AST changes. Bandit production0/0errors; initial test scan540baseline/542current shows exactly two new LOW B101 pytest assertions. Added narrowly justified test-only nosec B101 annotations following existing DB test convention, retaining both assertions; final changed-test scan540identical signatures/0errors/no new, with independent review pending. Initial Ruff cache write was blocked, corrected only invocation to no-cache; no environment/source workaround. This repairs the confirmed implicit-commit boundary; exact hosted interleaving was not locally reproduced. All final-head hosted/native/UAT gates remain open.

Review correction: base-only repair is incomplete because six downstream helpers also use executescript. The CI exact create_tables callsite was not retained; no unchanged log redownload or base-only hosted-fix claim. Second bounded repair attempt strengthens regression at both early and final bootstrap failure points, verifies the late case fails on the partial repair, and preserves transactions for all seven fixed-schema calls. Existing204pass is historical partial-patch evidence. First review also noted final changed-test scan already completed540identical/0errors/no new; that pending wording is superseded here. Three-fix-attempt ceiling remains; no hidden retries or domain STOP replay.

Second causal result on partial source: strengthened early/final regression1passed/1failed/30warnings/3.92s pytest/6.390s outer/natural1. Final helper failure still left28 committed tables. The corrected source now routes all seven fixed Sync schema calls through _create_schema_tables, preserving SQLite transaction and retaining backend delegation for PostgreSQL/unknown backend; full affected qualification next. No source constants/schema, original tests or shared backend changes.

Final corrected-source qualification: all205 store cases passed/30warnings/9.20s pytest/11.553s outer/natural0/no skips reported, including applicable official isolatedPG fixtures. Seven original production functions are identical after only normalizing schema call targets; _create_schema_tables is the sole new private helper. All original test functions/assertions and schemas unchanged. Two Python files compile; final Ruff old/currentlint0, inherited format flags retained; production Bandit0/0errors and test540identical signatures/0errors/no new. Workflow pull_request/attempt1/d667/.github/workflows/ci.yml binding confirmed. Independent corrected actual4path and final staged review, normal publication and fresh final-head hosted qualification remain; no claim that the exact hosted interleaving was reproduced.

Independent corrected4path actual-source/test/prose review CLEAR/no findings; both prior Important transaction-coverage and Minor scan-wording findings addressed. Reviewer confirms current fixed schema split safety, complete SQLite lock coverage, unchanged PostgreSQL/unknown-backend and separate migration behavior, real early/final rollback coverage, original assertions and narrow test-only B101 annotations. Final staged review/normal guarded publication next; hosted causation and all fresh hosted/native/UAT gates remain open.

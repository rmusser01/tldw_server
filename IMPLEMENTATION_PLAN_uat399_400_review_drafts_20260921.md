# Restore owned Content Review ingestion (UAT399/UAT400)

## Design
Reuse the existing verified Quick Ingest authority and request-scope checks. Add optional owner metadata to the three existing draft records; unknown legacy owners remain stored but hidden. Require a captured operation for draft storage and each network continuation. Remount Content Review when the verified authority generation changes. Clear only that owner's rows. Reuse the legacy processed-result decoder for the shipped wizard, retaining exact content and original files; do not instantiate the legacy UI hook with dummy dependencies.

## Stage 1: Account boundaries
**Goal**: Close UAT400 storage, mounted state and late-operation gaps.
**Success Criteria**: Foreign IDs and legacy rows cannot be read, overwritten or cleared; stale operations cannot continue.
**Tests**: Causal storage and mounted component regressions, scope-path controls.
**Status**: In Progress

## Stage 2: Shipped wizard handoff
**Goal**: Close UAT399 using shared draft creation.
**Success Criteria**: One owned draft per successful processed result, retained files, correct batch navigation, recoverable failures and no duplicate completion writes.
**Tests**: Actual wizard completion, ordinary process-only controls, mixed success and repeated/reloaded completion.
**Status**: Complete

## Stage 3: Production diagnostic follow-up
**Goal**: Validate a newly identified application/harness candidate.
**Success Criteria**: Owned SQLite and official PostgreSQL review edit, AI and commit/reload checks; account-switch evidence recorded separately from full matrix.
**Tests**: Scoped regression/types/lint, builds, exact native-case reports with source/artifact provenance.
**Status**: In Progress

## Verification checkpoint
214 affected UI tests across 8 files and 18 biology oracle tests pass, with no failures or skips. Ten real Chromium IndexedDB checks pass. Source repair includes follow-on UAT401 manual-save failure cleanup. Frontend TypeScript passes; native archived application follow-up remains pending. See the running tracker for retained failed invocations and precise scope.

Diagnostic3 aa45:3passed/3failed per SQLite/official PostgreSQL, zero skips/retries. Native handoff and edit/reset pass; Study source/scheduling journey passes. Commit/readback is blocked by a repaired response-field oracle, AI Fix by UAT403 literal default model (causal7controls/1failure then8passes). UAT402 webpack namespace retention fixed with named Sentry import; both production bundlers pass unchanged budgets. Full native account-switch/local-save-failure and new production follow-up remain outstanding.

Diagnostic4 closes UAT389/395/399/403 through all4 required production Content Review cases on SQLite and official PostgreSQL. Native controlled local-save recovery closes401 separately on immutableaa45. UAT400 remains open for actual multi-user account-switch/recovery acceptance. Journey harness fixes390/391/404/405 are tracked separately; full qualification is not complete.

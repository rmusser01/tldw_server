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
**Status**: In Progress

## Stage 3: Production diagnostic follow-up
**Goal**: Validate a newly identified application/harness candidate.
**Success Criteria**: Owned SQLite and official PostgreSQL review edit, AI and commit/reload checks; account-switch evidence recorded separately from full matrix.
**Tests**: Scoped regression/types/lint, builds, exact native-case reports with source/artifact provenance.
**Status**: Not Started

## Verification checkpoint
214 affected UI tests across 8 files and 18 biology oracle tests pass, with no failures or skips. Ten real Chromium IndexedDB checks pass. Source repair includes follow-on UAT401 manual-save failure cleanup. Frontend TypeScript passes; native archived application follow-up remains pending. See the running tracker for retained failed invocations and precise scope.

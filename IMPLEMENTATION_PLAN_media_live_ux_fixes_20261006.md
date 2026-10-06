# Remaining live Media UX fixes — TASK-13504–13507

Approved scope: user continuation of the solutions in Docs/Reviews/2026-10-05-media-live-validation.md. Baseline: latest dev 1fc353c3f67c93ba05102e7b0136ac4acac8f510. ADR required: no; fixes reuse existing detail, queue, session, ownership and localization contracts. ADR-059 governs task editing. No source/provider persistence changes.

## Stage 1: Real source identity — TASK-13504
**Goal**: Normalize nested detail identity at the shared multi-review fetch boundary.
**Success Criteria**: Nested titles/types survive absent current-page rows in reading/comparison/export; flat DTOs retain behavior.
**Tests**: Actual nested payload regression in existing review tests; legacy fixtures stay green.
**Status**: Complete

## Stage 2: Clean reading content — TASK-13505
**Goal**: Remove valid ingestion metadata envelopes from default single/multi reading while preserving raw content for other uses.
**Success Criteria**: Actual web article reads cleanly; malformed envelopes and ordinary metadata mentions remain intact; raw exports/analysis unchanged.
**Tests**: Valid/malformed/body text parser checks plus reader integration against actual stored payload.
**Status**: In Progress

## Stage 3: Retry confirmation and count copy — TASK-13506/13507
**Goal**: Configure/Review reflects actual retry scope and prior saved states; saved-item CTA wording respects item count.
**Success Criteria**: Resumed mixed batch shows and submits only the failed item; previous successes remain accurately labeled; one/many labels agree visually and accessibly.
**Tests**: Existing wizard session integration and result/history localization checks.
**Status**: Not Started

## Stage 4: Verification and review
**Goal**: Verify the changed flows on real isolated API content and current client builds; create a reviewable result.
**Success Criteria**: Scoped tests/builds pass, screenshots prove desktop/mobile behavior, source review complete, tracking and evidence updated; original checkout unchanged and test services stopped.
**Tests**: Red/green regressions, affected suites, static/type/build checks, real browser checks, canonical task checks and applicable security validation.
**Status**: Not Started

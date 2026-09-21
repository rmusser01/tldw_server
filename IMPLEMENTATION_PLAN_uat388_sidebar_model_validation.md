# UAT388: actionable sidebar model validation

Backlog: TASK13260.277.38. Bounded repair to the existing sidebar composer.

## Stage 1: Establish the cause
**Goal**: Reproduce disappearing missing/unavailable-model errors and hidden Casual recovery.
**Success Criteria**: Actual submit/effect/render code fails behavioral checks with the real form state; valid send controls pass.
**Tests**: Retained draft and alert, unrelated rerender, empty/image-only input, unavailable selection, valid submit/queue.
**Status**: Complete

Native candidate12 SQLite013–015 retains the draft, sends no API request and shows no settled alert. Pro016/017 exposes the missing selection and existing picker; configured019 sends successfully (200, saved, reply8). Error-clear effects depend on the newly raised error itself. Both effects already claim they should clear only on interaction. Existing useSimpleForm provides a stable clearFieldError; existing ModelSelect owns selection and provider discovery. Keep those implementations.

The initial behavioral regression reproduced 8 failures with 5 controls. Independent review then confirmed that the existing availability catalog was disabled without optional audio. A real QueryClient regression executing the actual catalog and availability initializers reproduced 5 failures with 10 controls after correcting two harness setup mistakes. These results are preserved in `/tmp/uat388-red.log` and `/tmp/uat388-query-red3.log`.

## Stage 2: Repair and review
**Goal**: Clear validation on actual draft/model changes and expose the existing picker when Casual mode needs a model.
**Success Criteria**: No new form abstraction, no changed send/queue transport or account ownership, passing regression and matched static checks, independent review.
**Tests**: Stage1 regressions plus adjacent composer, queue, owner, and model-selection controls. Bandit is inapplicable if scope remains TypeScript only.
**Status**: Complete

Production keeps the existing model fetch and shares its query key with ModelSelect, enabled by connection readiness. Both voice options and validation use that catalog. The error effect depends only on draft/model changes; Casual recovery uses the existing picker when connected. All 15 direct regressions and 166 tests across 9 surrounding suites pass. Independent rereview has no remaining actionable findings. Final lint has 0 errors and 34 warnings versus 36 baseline warnings, with no additions. Final matched TypeScript remains 426→426, with no added or touched diagnostics (`/tmp/uat388-type-final.log`). Bandit is inapplicable to this TypeScript-only change. Stage3 is outstanding, so this plan and Backlog task remain open and no completion or merge is claimed.

## Stage 3: Native acceptance
**Goal**: Verify missing-model recovery on fresh SQLite and official-fixture PostgreSQL.
**Success Criteria**: Visible settled error, retained draft, usable model picker, actual successful send and canonical readback. Record all evidence in the running tracker; captures stay ignored/local.
**Tests**: Packaged native Casual-mode controls on both databases.
**Status**: Not Started

Automatic approval review could not complete the PostgreSQL browser action because the weekly Codex usage limit is exhausted. The action did not execute; this was not a safety rejection. Native escalation-dependent checks remain pending until review is available. Local source/tests can proceed.

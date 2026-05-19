---
id: TASK-349.1.4
title: Stage 4D Watchlist Items tab Updates triage refresh
status: Done
assignee: []
created_date: '2026-05-15 18:19'
updated_date: '2026-05-15 19:12'
labels:
  - watchlists
  - stage4
  - frontend
  - ux
dependencies:
  - TASK-349.1.1
  - TASK-349.1.2
  - TASK-349.1.3
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md
documentation:
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-349.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Refresh the selected Watchlist Items tab into an alert-aware Updates triage surface using the Stage 4 backend/client contracts. Scope includes visible triage UX, alert context, backend batch actions, per-Watchlist saved views, copy, and focused frontend tests, while leaving Stage 5 report-builder work out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Visible selected-Watchlist review copy moves toward Updates/review-queue language while preserving route and API compatibility.
- [x] #2 ItemsTab sends server-backed sort/filter/alert-match parameters and no longer relies on current-page client sorting as primary ordering.
- [x] #3 Item rows and reader show compact content-alert context when alert summaries are present, with clear handoff to the Alerts tab.
- [x] #4 Selected/page/all-filtered batch review actions use the backend batch endpoint and preserve clear success/partial/failure feedback.
- [x] #5 Saved views load/save/update/delete through the selected Watchlist API, with a recoverable import path for legacy localStorage views.
- [x] #6 Focused component, accessibility, keyboard, batch, and copy-contract tests cover the refreshed triage workflow.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after Stage 4C commit b5df2e63b. Scope: visible Items/Updates triage refresh using existing Stage 4 backend/client contracts; preserve route compatibility and defer report-builder work to Stage 5.

Implemented Updates-oriented selected-Watchlist triage copy, alert-match smart filtering, server-backed sort/filter params, row/reader alert context, backend batch review actions, and selected-Watchlist saved views with legacy localStorage import.

Verification: focused Vitest slice passed: 7 files / 81 tests; locale JSON parse check passed; git diff --check passed; debug/any scan found no matches in touched frontend files.

Known skip: Bandit is not applicable to this Stage 4D frontend TypeScript/locale-only slice. API docs and real-server CDP smoke remain scoped to Stage 4E.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4D refreshed the selected Watchlist Items tab into an Updates triage surface backed by the Stage 4 API/client contracts. It now uses server-backed ordering/filtering, exposes alert-match context, routes batch review through the batch endpoint, persists saved views per Watchlist with legacy import, and updates focused frontend tests and copy contracts.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

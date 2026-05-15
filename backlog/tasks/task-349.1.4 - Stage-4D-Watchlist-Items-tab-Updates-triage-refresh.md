---
id: TASK-349.1.4
title: Stage 4D Watchlist Items tab Updates triage refresh
status: To Do
assignee: []
created_date: '2026-05-15 18:19'
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
- [ ] #1 Visible selected-Watchlist review copy moves toward Updates/review-queue language while preserving route and API compatibility.
- [ ] #2 ItemsTab sends server-backed sort/filter/alert-match parameters and no longer relies on current-page client sorting as primary ordering.
- [ ] #3 Item rows and reader show compact content-alert context when alert summaries are present, with clear handoff to the Alerts tab.
- [ ] #4 Selected/page/all-filtered batch review actions use the backend batch endpoint and preserve clear success/partial/failure feedback.
- [ ] #5 Saved views load/save/update/delete through the selected Watchlist API, with a recoverable import path for legacy localStorage views.
- [ ] #6 Focused component, accessibility, keyboard, batch, and copy-contract tests cover the refreshed triage workflow.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

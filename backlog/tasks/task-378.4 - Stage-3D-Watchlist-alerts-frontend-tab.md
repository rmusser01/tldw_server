---
id: TASK-378.4
title: Stage 3D Watchlist alerts frontend tab
status: Done
assignee: []
created_date: '2026-05-15 14:53'
updated_date: '2026-05-15 15:33'
labels:
  - watchlists
  - stage3
  - frontend
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-378
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add frontend types, service methods, copy, and an Alerts tab for selected Watchlist content alert rules and inbox triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Alerts tab supports content alert rule management and alert inbox triage for the selected Watchlist.
- [x] #2 Alert evidence, source context, severity, created time, and review actions are visible.
- [x] #3 Copy keeps content alerts separate from pipeline health issues and works in constrained viewports.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Watchlist-scoped content alert frontend types and service methods for rule CRUD plus alert list/detail/update. Added AlertsTab with rule create/edit/enable-delete, inbox filters, evidence display, read/unread/dismiss actions, and explicit content-alert vs health-issue boundary copy. Integrated Alerts into full/progressive/reduced Watchlists navigation, orientation copy, quick action copy, help docs, and mirrored locale keys.

Verification: focused Stage 3D Vitest passed: src/services/__tests__/watchlists-content-alerts.test.ts, src/components/Option/Watchlists/AlertsTab/__tests__/AlertsTab.test.tsx, src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx, src/components/Option/Watchlists/__tests__/watchlists-stage3-copy-contract.test.ts: 4 files, 9 tests passed. Nearby Watchlists regressions passed: experimental IA, orientation guidance, help links, run notifications, static guard: 5 files, 26 tests passed. git diff --check passed. Full frontend tsc still fails on existing repo-wide baseline; filtered tsc output for Watchlists/AlertsTab/services/watchlists/types/watchlists returned no matches after the local copy-contract cast fix. Bandit not applicable to this frontend-only slice; backend Bandit was run for prior Stage 3 backend changes.

Scope note: backend Stage 3B list-alerts API currently exposes status, severity, rule_id, source_id, page, and size. Stage 3D implemented API-backed filters for those fields and client-side search over loaded alerts; API-backed date-window and full text alert search remain future API enhancements rather than frontend-only query parameters.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the first Watchlists Alerts tab surface: frontend contracts for content alert rules and alert inbox items, Watchlist-scoped service methods, local rule/inbox management UI, review actions, health-boundary copy, tab navigation integration, and locale/help text. Verification passed for focused Stage 3D tests and nearby Watchlists shell regressions; full frontend typecheck remains blocked by existing unrelated baseline errors, with no filtered Watchlists/AlertsTab errors.
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

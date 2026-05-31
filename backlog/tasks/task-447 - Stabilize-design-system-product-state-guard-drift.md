---
id: TASK-447
title: Stabilize design-system product-state guard drift
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-20 01:50'
labels:
  - design-system
  - product-state
  - ui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1873'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the current blocked product-state guard findings on dev to canonical design-system primitives and registry-backed labels. Scope: SpeechPlaygroundPage, SttPlaygroundPage, WatchlistsPlaygroundPage, and AudioReadinessStrip. Also remove stale baseline entries produced by the migration and verify the design-system guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Product-state guard reports no blocked findings for the migrated Speech, STT, Watchlists, and Audio readiness scope.
- [x] #2 Focused Vitest coverage for the migrated Audio, STT, Speech, and Watchlists surfaces passes.
- [x] #3 PR #1873 review feedback is addressed for Speech alert structure/i18n and Audio readiness registry-backed labels.
- [x] #4 Verification results and known skips are recorded in the final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the current Speech/STT/Watchlists guard drift to design-system Alert, registry-backed Audio readiness canonical labels, and pruned stale baseline entries. Addressed PR #1873 review feedback by moving Speech model-load details into the alert body, translating TTS error titles, and replacing Audio readiness canonical strings with registry-backed exported labels. Verification: focused Vitest suite passed; verify:design-system-state passed with 335 allowed legacy exceptions and no blocked findings; git diff --check passed; full TypeScript still fails on existing repo-wide debt with no diagnostics in touched files; Bandit skipped because touched code is TypeScript/JSON/Backlog only.
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

---
id: TASK-447
title: Stabilize design-system product-state guard drift
status: Done
labels:
- design-system
- product-state
- ui
modified_files:
- apps/packages/ui/src/components/Option/Audio/AudioReadinessStrip.tsx
- apps/packages/ui/src/components/Option/Audio/__tests__/AudioReadinessStrip.test.tsx
- apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx
- apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
references:
- https://github.com/rmusser01/tldw_server/pull/1873
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the current blocked product-state guard findings on dev to canonical design-system primitives and registry-backed labels. Scope: SpeechPlaygroundPage, SttPlaygroundPage, WatchlistsPlaygroundPage, and AudioReadinessStrip. Also remove stale baseline entries produced by the migration and verify the design-system guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the current Speech/STT/Watchlists guard drift to design-system Alert, registry-backed Audio readiness canonical labels, and pruned stale baseline entries. Addressed PR #1873 review feedback by moving Speech model-load details into the alert body, translating TTS error titles, and replacing Audio readiness canonical strings with registry-backed exported labels. Verification: focused Vitest suite passed; verify:design-system-state passed with 335 allowed legacy exceptions and no blocked findings; git diff --check passed; full TypeScript still fails on existing repo-wide debt with no diagnostics in touched files; Bandit skipped because touched code is TypeScript/JSON/Backlog only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
